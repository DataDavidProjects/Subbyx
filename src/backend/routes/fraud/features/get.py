from __future__ import annotations

import logging

from fastapi import APIRouter

from routes.fraud.config import config as fraud_config
from routes.fraud.schemas import FeaturesGetRequest, FeaturesGetResponse
from services.fraud.features import get_features

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fraud-features"])


REQUEST_FEATURES = fraud_config.get("features", {}).get("request", [])


@router.post("/v1/features/get", response_model=FeaturesGetResponse)
def get_features_endpoint(request: FeaturesGetRequest) -> FeaturesGetResponse:
    logger.debug(
        "fetching features for customer_id=%s, email=%s, segment=%s",
        request.customer_id,
        request.email,
        request.segment,
    )

    customer_features = get_features(request.customer_id, request.email)
    logger.debug(
        "customer features for customer_id=%s: %d keys",
        request.customer_id,
        len(customer_features),
    )

    features = {
        "segment": request.segment,
        **{key: request.checkout_data.get(key) for key in REQUEST_FEATURES},
        **customer_features,
    }

    logger.debug("assembled %d total features", len(features))
    return FeaturesGetResponse(features=features)
