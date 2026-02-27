from __future__ import annotations

from services.fraud.inference.model import ScoringResult, score_all


def predict(
    features: dict | None = None,
    customer_id: str | None = None,
    email: str | None = None,
    request_features: dict | None = None,
) -> ScoringResult:
    return score_all(
        features=features,
        customer_id=customer_id,
        email=email,
        request_features=request_features,
    )
