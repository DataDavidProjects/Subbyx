from __future__ import annotations

from fastapi import APIRouter

from routes.features.schema import FeaturesGetResponse
from services.fraud.features import get_features


router = APIRouter(prefix="/features", tags=["features"])


@router.get("/email/{email}", response_model=FeaturesGetResponse)
def get_email_features(email: str) -> FeaturesGetResponse:
    features = get_features(email=email)
    return FeaturesGetResponse(features=features)
