from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.config import config as fraud_config
from routes.fraud.schemas import CheckoutRequest, CheckoutResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fraud-checkout"])

DATA_DIR = Path("/Users/davidelupis/Desktop/Subbyx/data")


@lru_cache(maxsize=1)
def load_blacklist() -> frozenset:
    blacklist_path = Path(shared_config.get("paths", {}).get("blacklist", "data/blacklist.json"))
    if not blacklist_path.is_absolute():
        # checkout.py: routes/fraud/checkout.py -> parent: fraud, routes, backend, src, Subbyx (project root)
        blacklist_path = Path(__file__).parent.parent.parent.parent.parent / blacklist_path

    logger.debug("loading blacklist from %s", blacklist_path)

    if blacklist_path.exists():
        with open(blacklist_path) as f:
            data = json.load(f)
            emails = frozenset(data.get("emails", []))
            logger.debug("blacklist loaded, %d entries", len(emails))
            return emails

    logger.warning("blacklist file not found at %s", blacklist_path)
    return frozenset()


@lru_cache(maxsize=1)
def load_checkouts() -> pd.DataFrame:
    """Load checkouts CSV with caching."""
    checkouts_path = DATA_DIR / "01-clean" / "checkouts.csv"
    logger.debug("loading checkouts from %s", checkouts_path)
    return pd.read_csv(checkouts_path)


def has_completed_checkout(customer_id: str, cutoff_time: str | None = None) -> bool:
    """Check if customer has any completed checkouts before cutoff_time (PIT correct).

    Completed checkout = status='complete' AND mode IN ('payment', 'subscription')
    """
    if not customer_id:
        return False

    df = load_checkouts()

    # Filter to completed real subscriptions
    completed = df[(df["status"] == "complete") & (df["mode"].isin(["payment", "subscription"]))]

    # PIT filtering
    if cutoff_time:
        completed = completed[pd.to_datetime(completed["created"]) < pd.to_datetime(cutoff_time)]

    return customer_id in completed["customer"].values


def determine_segment(
    customer_id: str,
    email: str | None = None,
    fiscal_code: str | None = None,
    card_fingerprint: str | None = None,
    timestamp: str | None = None,
) -> tuple[str, str]:
    """Determine segment via customer_id check for completed checkouts.

    If customer has prior completed checkout (status='complete', mode IN payment/subscription)
    before timestamp → RETURNING.
    Otherwise → NEW_CUSTOMER.
    """
    if not customer_id:
        return (
            fraud_config["segment_keys"]["new_customer"],
            "No customer_id provided",
        )

    # Check for completed checkout with PIT
    if has_completed_checkout(customer_id, timestamp):
        reason = f"Prior completed checkout found (cutoff: {timestamp or 'none'})"
        logger.debug("customer_id %s -> RETURNING (%s)", customer_id, reason)
        return fraud_config["segment_keys"]["returning"], reason

    logger.debug("customer_id %s -> NEW_CUSTOMER (no prior checkouts)", customer_id)
    return (
        fraud_config["segment_keys"]["new_customer"],
        "No prior completed checkouts",
    )


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
    logger.info("[BACKEND] checkout request: checkout_id=%s", request.checkout_id)

    # 0. Resolve context from checkout_id
    from services.fraud.context import resolve_checkout
    from services.fraud.features.request_features import extract_request_features

    ctx = resolve_checkout(request.checkout_id)
    logger.info(
        "[BACKEND] Step 0 complete: email=%s, customer_id=%s, store_id=%s",
        ctx.email,
        ctx.customer_id,
        ctx.store_id,
    )

    # 1. Segment determination
    logger.info("[BACKEND] Step 1: Determining segment for customer_id=%s", ctx.customer_id)
    segment, segment_reason = determine_segment(
        ctx.customer_id,
        ctx.email,
        ctx.fiscal_code,
        ctx.card_fingerprint,
        ctx.timestamp,
    )
    logger.info(
        "[BACKEND] Step 1 complete: segment=%s, reason=%s",
        segment,
        segment_reason,
    )

    # 2. Rules engine (pre-model checks)
    logger.info("[BACKEND] Step 2: Checking rules engine")
    blacklist = load_blacklist()
    if ctx.email in blacklist:
        logger.info(
            "[BACKEND] Step 2: BLACKLIST triggered for email=%s (segment=%s)",
            ctx.email,
            segment,
        )
        return CheckoutResponse(
            decision="BLOCK",
            reason="Email %s in blacklist" % ctx.email,
            rule_triggered="blacklist",
            score=None,
            segment=segment,
            segment_reason=segment_reason,
        )

    # 2a. Stripe risk check
    from routes.fraud.rules.stripe_risk import load_charges_with_highest_risk

    high_risk_emails = load_charges_with_highest_risk(ctx.timestamp)
    if ctx.email in high_risk_emails:
        logger.info(
            "[BACKEND] Step 2a: STRIPE_RISK triggered for email=%s (segment=%s)",
            ctx.email,
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

    # 2b. Fiscal code duplicate check
    from routes.fraud.rules.fiscal_code import (
        is_duplicate_fiscal_code,
        load_fiscal_code_to_emails,
    )

    if ctx.fiscal_code:
        fiscal_code_mapping = load_fiscal_code_to_emails(ctx.timestamp)
        if is_duplicate_fiscal_code(
            ctx.fiscal_code,
            ctx.email,
            fiscal_code_mapping,
        ):
            emails = fiscal_code_mapping.get(ctx.fiscal_code, set())
            logger.info(
                "[BACKEND] Step 2b: FISCAL_CODE_DUPLICATE triggered for fiscal_code=%s (segment=%s)",
                ctx.fiscal_code,
                segment,
            )
            return CheckoutResponse(
                decision="BLOCK",
                reason="Fiscal code %s used with different emails: %s"
                % (ctx.fiscal_code, ", ".join(sorted(emails))),
                rule_triggered="fiscal_code_duplicate",
                score=None,
                segment=segment,
                segment_reason=segment_reason,
            )

    # 3. Fetch batch features with all entity keys
    logger.info(
        "[BACKEND] Step 3: Fetching features for email=%s, customer_id=%s, store_id=%s",
        ctx.email,
        ctx.customer_id,
        ctx.store_id,
    )

    from services.fraud.features import get_features as get_feast_features

    feast_features = get_feast_features(
        email=ctx.email,
        customer_id=ctx.customer_id,
        store_id=ctx.store_id,
        card_fingerprint=ctx.card_fingerprint,
        fiscal_code=ctx.fiscal_code,
    )

    # 3a. Payment failure rate check (pre-model rule for RETURNING customers)
    # Uses the features fetched in Step 3, but runs BEFORE model scoring
    if segment == "RETURNING":
        from routes.fraud.rules.payment_failure import check_payment_failure

        pf_triggered, pf_reason = check_payment_failure(feast_features, segment)
        if pf_triggered:
            logger.info(
                "[BACKEND] Step 3a: PAYMENT_FAILURE rule triggered (segment=%s)",
                segment,
            )
            return CheckoutResponse(
                decision="BLOCK",
                reason=pf_reason,
                rule_triggered="payment_failure",
                score=None,
                segment=segment,
                segment_reason=segment_reason,
            )

    # 4. Merge request features and align with model prefixes
    request_features = extract_request_features(ctx)
    all_features = {**feast_features}

    # Map request features to their prefixed counterparts if needed by model
    # Note: checkout_features removed from Feast - these are pure request-time values
    # Card features are now only request-time (via CheckoutContext), not from Feast
    mapping = {
        "subscription_value": "checkout_features__subscription_value",
        "grade": "checkout_features__grade",
        "category": "checkout_features__category",
    }
    for req_key, feat_key in mapping.items():
        if req_key in request_features:
            all_features[feat_key] = request_features[req_key]

    # Fill null Feast features with request-time values when available.
    # payment_intent_features__subscription_value is often null in the
    # online store (PI not yet created), but we know the value from the
    # checkout request — use it to avoid imputing -1.
    _feast_fallbacks = {
        "payment_intent_features__subscription_value": "subscription_value",
    }
    for feast_key, req_key in _feast_fallbacks.items():
        if all_features.get(feast_key) is None and req_key in request_features:
            all_features[feast_key] = request_features[req_key]

    # Keep original request features for fallback/selection
    all_features.update(request_features)

    # Add missing-indicator flags (train/serve parity)
    from services.fraud.features.selection.transformers import AddMissingIndicators

    AddMissingIndicators.enrich_dict(all_features)

    from services.fraud.inference.model import score_models

    result = score_models(all_features)

    # Build response features from MLflow columns
    from services.fraud.inference.model import production_model

    mlflow_feature_cols = production_model.feature_columns
    response_features: dict = {col: all_features.get(col) for col in mlflow_feature_cols}

    # Attach feature metadata (label + description) from Feast tags
    try:
        from services.fraud.features.metadata import get_feature_metadata

        response_features["__meta"] = get_feature_metadata()
    except Exception as meta_exc:
        logger.warning("Failed to attach feature metadata: %s", meta_exc)

    logger.info(
        "[BACKEND] Step 4 complete: model_score=%.4f, production=%.4f, shadow=%.4f, scored_by=%s",
        result.score,
        result.production_score,
        result.shadow_score,
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
        ctx.email,
        decision,
        result.score,
        segment,
        result.scored_by,
    )
    logger.info("[BACKEND] ========================================")

    return CheckoutResponse(
        decision=decision,
        reason=reason,
        rule_triggered=None,
        score=result.score,
        segment=segment,
        segment_reason=segment_reason,
        features=response_features,
        production_score=result.production_score,
        shadow_score=result.shadow_score,
        scored_by=result.scored_by,
    )
