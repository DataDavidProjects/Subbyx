from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.config import config as fraud_config
from routes.fraud.schemas import CheckoutRequest, CheckoutResponse
from services.fraud.inference import predict as get_score

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fraud-checkout"])


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


def determine_segment(customer_id: str, email: str | None = None) -> tuple[str, str]:
    """Determine segment based on available features from Feast."""
    from services.fraud.features import get_features

    features = get_features(customer_id=customer_id)

    has_features = any(v is not None for v in features.values())

    if has_features:
        logger.debug("customer_id %s has feature data", customer_id)
        return (
            fraud_config["segment_keys"]["returning"],
            "Has historical feature data",
        )

    logger.debug("customer_id %s has no feature data", customer_id)
    return (
        fraud_config["segment_keys"]["new_customer"],
        "No historical data available",
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
    logger.info("[BACKEND] Step 1: Determining segment for customer_id=%s", request.customer_id)
    segment, segment_reason = determine_segment(request.customer_id, request.email)
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

    # 3. Model scoring — features come from FeatureService
    logger.info(
        "[BACKEND] Step 3: Calling model scoring for customer_id=%s, email=%s",
        request.customer_id,
        request.email,
    )
    result = get_score(
        customer_id=request.customer_id,
        email=request.email,
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

    return CheckoutResponse(
        decision=decision,
        reason=reason,
        rule_triggered=None,
        score=result.score,
        segment=segment,
        segment_reason=segment_reason,
        features=result.features or {},
        production_score=result.production_score,
        shadow_score=result.shadow_score,
        canary_score=result.canary_score,
        scored_by=result.scored_by,
    )
