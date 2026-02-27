from __future__ import annotations

import logging

from fastapi import APIRouter

from routes.fraud.config import config as fraud_config
from routes.fraud.schemas import SegmentDetermineRequest, SegmentDetermineResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["segment"])

# Feature views and a representative feature to check for historical data
HISTORICAL_FEATURES = {
    "charge_features": "charge_count",
    "payment_intent_features": "intent_count",
    "checkout_history_features": "prior_checkout_count",
}


def _check_historical_data_in_feast(email: str) -> tuple[bool, str]:
    """Check if email has any historical data in Feast feature store."""
    logger.info("[SEGMENT] Checking historical data in Feast for email=%s", email)

    try:
        from services.fraud.features.store import store

        has_history = False
        history_details = []

        for feature_view, check_feature in HISTORICAL_FEATURES.items():
            try:
                full_feature = f"{feature_view}:{check_feature}"
                logger.debug(
                    "[SEGMENT] Checking feature_view=%s, feature=%s",
                    feature_view,
                    check_feature,
                )
                response = store.get_online_features(
                    features=[full_feature],
                    entity_rows=[{"email": email}],
                )
                row = response.to_dict()
                values = row.get(check_feature, [])
                if values and values[0] is not None:
                    has_history = True
                    history_details.append(f"{feature_view}.{check_feature}={values[0]}")
                    logger.debug(
                        "[SEGMENT] Found %s.%s=%s for email=%s",
                        feature_view,
                        check_feature,
                        values[0],
                        email,
                    )
            except Exception as exc:
                logger.debug(
                    "[SEGMENT] Error checking %s for email %s: %s",
                    feature_view,
                    email,
                    exc,
                )

        if has_history:
            result = True, f"Has historical data: {', '.join(history_details)}"
            logger.info("[SEGMENT] Result: %s", result[1])
            return result
        result = False, "No historical data available"
        logger.info("[SEGMENT] Result: %s", result[1])
        return result
    except Exception as exc:
        logger.warning("[SEGMENT] Failed to check Feast for historical data: %s", exc)
        return False, "Unable to check historical data"


@router.post("/v1/segment/determine", response_model=SegmentDetermineResponse)
def determine_segment(
    request: SegmentDetermineRequest,
) -> SegmentDetermineResponse:
    logger.debug("determining segment for email %s", request.email)

    # Check Feast for historical features
    has_history, reason = _check_historical_data_in_feast(request.email)

    if has_history:
        logger.debug("email %s has historical data: %s", request.email, reason)
        return SegmentDetermineResponse(
            segment=fraud_config["segment_keys"]["returning"],
            reason=reason,
        )
    else:
        logger.debug("email %s has no historical data: %s", request.email, reason)
        return SegmentDetermineResponse(
            segment=fraud_config["segment_keys"]["new_customer"],
            reason=reason,
        )
