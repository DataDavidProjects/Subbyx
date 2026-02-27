from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.config import config as fraud_config
from routes.fraud.schemas import CheckoutRequest, CheckoutResponse
from services.fraud.features.base import Entity
from services.fraud.features.registry import registry
from services.fraud.inference import predict as get_score

_CUSTOMER_FEATURE_KEYS = {f.name for f in registry.get_by_entity(Entity.CUSTOMER_ID)}

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fraud-checkout"])


def load_blacklist() -> set:
    blacklist_path = Path(shared_config.get("paths", {}).get("blacklist", "data/blacklist.json"))
    if not blacklist_path.is_absolute():
        # checkout.py: routes/fraud/checkout.py -> parent: fraud, routes, backend, src, Subbyx (project root)
        blacklist_path = Path(__file__).parent.parent.parent.parent.parent / blacklist_path

    logger.debug("loading blacklist from %s", blacklist_path)

    if blacklist_path.exists():
        with open(blacklist_path) as f:
            data = json.load(f)
            emails = set(data.get("emails", []))
            logger.debug("blacklist loaded, %d entries", len(emails))
            return emails

    logger.warning("blacklist file not found at %s", blacklist_path)
    return set()


def determine_segment(email: str, timestamp: str | None = None) -> tuple[str, str]:
    """Determine segment based on historical data from Feast features."""
    from services.fraud.features import get_features

    HISTORICAL_FEATURE_KEYS = ["charge_count", "intent_count", "prior_checkout_count"]

    features = get_features(email=email)

    has_history = False
    history_details = []

    for check_feature in HISTORICAL_FEATURE_KEYS:
        value = features.get(check_feature)
        if value is not None and value > 0:
            has_history = True
            history_details.append(f"{check_feature}={value}")

    if has_history:
        logger.debug("email %s has historical data: %s", email, ", ".join(history_details))
        return (
            fraud_config["segment_keys"]["returning"],
            f"Has historical data: {', '.join(history_details)}",
        )

    logger.debug("email %s has no historical data", email)
    return (
        fraud_config["segment_keys"]["new_customer"],
        "No historical data available",
    )


def build_request_features(checkout_data: dict, segment: str) -> dict:
    """Build request-level features from checkout data (no Feast call)."""
    request_feature_keys = fraud_config.get("features", {}).get("request", [])
    features = {
        "segment": segment,
        **{key: checkout_data.get(key) for key in request_feature_keys},
    }
    logger.debug("request features assembled: %d keys", len(features))
    return features


def build_customer_data(request) -> dict:
    """Build customer data dict from request for computing request-time features."""
    return {
        "email": request.email,
        "customer_name": request.customer_name,
        "document_name": request.document_name,
        "account_name": request.account_name,
        "card_owner_name": request.checkout_data.get("card_owner_name"),
        "card_gender": request.checkout_data.get("card_gender"),
        "card_country": request.checkout_data.get("card_country"),
        "card_province": request.checkout_data.get("card_province"),
        "email_on_file": request.checkout_data.get("email_on_file"),
        "card_name": request.checkout_data.get("card_name"),
        "has_high_end_device": request.has_high_end_device,
        "gender": request.gender,
        "birth_date": request.birth_date,
        "birth_province": request.birth_province,
        "birth_country": request.birth_country,
    }


def get_decision(score: float, segment: str) -> tuple[str, str]:
    segments = fraud_config.get("segments", {})

    if segment in segments:
        threshold = segments[segment].get("threshold")
    else:
        if segment == fraud_config["segment_keys"]["returning"]:
            threshold = fraud_config["segments"]["RETURNING"]["default_threshold"]
        else:
            threshold = fraud_config["segments"]["NEW_CUSTOMER"]["default_threshold"]

    block_decision = shared_config["decisions"]["block"]
    approve_decision = shared_config["decisions"]["approve"]

    if score > threshold:
        logger.debug(
            "score %.4f exceeds threshold %s -> %s",
            score,
            threshold,
            block_decision,
        )
        return block_decision, f"Score {score:.2f} exceeds threshold {threshold}"

    logger.debug(
        "score %.4f within threshold %s -> %s",
        score,
        threshold,
        approve_decision,
    )
    return approve_decision, f"Score {score:.2f} within threshold {threshold}"


@router.post("/v1/checkout", response_model=CheckoutResponse)
def fraud_checkout(request: CheckoutRequest) -> CheckoutResponse:
    logger.info("[BACKEND] ========================================")
    logger.info(
        "[BACKEND] checkout request: customer_id=%s, email=%s",
        request.customer_id,
        request.email,
    )
    logger.info(
        "[BACKEND] checkout_data=%s",
        request.checkout_data,
    )

    # 1. Segment determination (always first — rules can be segment-aware)
    logger.info("[BACKEND] Step 1: Determining segment for email=%s", request.email)
    segment, segment_reason = determine_segment(request.email, request.timestamp)
    logger.info(
        "[BACKEND] Step 1 complete: segment=%s, reason=%s",
        segment,
        segment_reason,
    )

    # 2. Rules engine (pre-model checks)
    logger.info("[BACKEND] Step 2: Checking rules engine")
    blacklist = load_blacklist()
    if request.email in blacklist:
        logger.info(
            "[BACKEND] Step 2: BLACKLIST triggered for email=%s (segment=%s)",
            request.email,
            segment,
        )
        return CheckoutResponse(
            decision="BLOCK",
            reason="Email %s in blacklist" % request.email,
            rule_triggered="blacklist",
            score=None,
            segment=segment,
            segment_reason=segment_reason,
        )

    # 2a. Stripe risk check
    from routes.fraud.rules.stripe_risk import load_charges_with_highest_risk

    high_risk_emails = load_charges_with_highest_risk()
    if request.email in high_risk_emails:
        logger.info(
            "[BACKEND] Step 2a: STRIPE_RISK triggered for email=%s (segment=%s)",
            request.email,
            segment,
        )
        return CheckoutResponse(
            decision="BLOCK",
            reason="Email has highest risk level from Stripe",
            rule_triggered="stripe_risk",
            score=None,
            segment=segment,
            segment_reason=segment_reason,
        )

    logger.info("[BACKEND] Step 2: No rules triggered")

    # 3. Build request features (checkout data + segment); Feast fetch happens per-model
    logger.info("[BACKEND] Step 3: Building request features")
    request_features = build_request_features(request.checkout_data, segment)
    customer_data = build_customer_data(request)
    logger.info(
        "[BACKEND] Step 3 complete: request_features keys=%s, customer_data keys=%s",
        list(request_features.keys()),
        [k for k, v in customer_data.items() if v is not None],
    )

    # 4. Model scoring — each model fetches its own Feast features independently
    logger.info(
        "[BACKEND] Step 4: Calling model scoring for customer_id=%s, email=%s",
        request.customer_id,
        request.email,
    )
    result = get_score(
        customer_id=request.customer_id,
        email=request.email,
        request_features=request_features,
    )
    logger.info(
        "[BACKEND] Step 4 complete: model_score=%.4f, production=%.4f, shadow=%.4f, canary=%s, scored_by=%s",
        result.score,
        result.production_score,
        result.shadow_score,
        result.canary_score,
        result.scored_by,
    )

    # 5. Decision engine
    logger.info("[BACKEND] Step 5: Applying decision threshold for segment=%s", segment)
    decision, reason = get_decision(result.score, segment)
    logger.info(
        "[BACKEND] Step 5 complete: decision=%s, reason=%s",
        decision,
        reason,
    )

    logger.info(
        "[BACKEND] FINAL: email=%s, decision=%s, score=%.4f, segment=%s, scored_by=%s",
        request.email,
        decision,
        result.score,
        segment,
        result.scored_by,
    )
    logger.info("[BACKEND] ========================================")

    # Fetch all feature groups for the response display
    from services.fraud.features import get_features

    logger.info("[BACKEND] Fetching features for UI display")
    email_features = get_features(request.customer_id, request.email)

    customer_features = {k: v for k, v in email_features.items() if k in _CUSTOMER_FEATURE_KEYS}
    charge_features = {
        k: v
        for k, v in email_features.items()
        if k
        in {
            "charge_count",
            "charge_failure_rate",
            "recurring_charge_rate",
            "distinct_cards",
            "prepaid_card_rate",
            "blocked_rate",
            "avg_risk_score",
            "max_risk_score",
        }
    }
    payment_intent_features = {
        k: v
        for k, v in email_features.items()
        if k in {"intent_count", "total_failures", "payment_failure_rate"}
    }
    checkout_history_features = {
        k: v
        for k, v in email_features.items()
        if k in {"prior_checkout_count", "avg_subscription_value", "distinct_categories"}
    }
    logger.info(
        "[BACKEND] Features fetched: customer=%d, charges=%d, payment_intents=%d, checkout_history=%d",
        len(customer_features),
        len(charge_features),
        len(payment_intent_features),
        len(checkout_history_features),
    )

    return CheckoutResponse(
        decision=decision,
        reason=reason,
        rule_triggered=None,
        score=result.score,
        segment=segment,
        segment_reason=segment_reason,
        features={
            "customer": customer_features,
            "charges": charge_features,
            "payment_intents": payment_intent_features,
            "checkout_history": checkout_history_features,
        },
        production_score=result.production_score,
        shadow_score=result.shadow_score,
        canary_score=result.canary_score,
        scored_by=result.scored_by,
    )
