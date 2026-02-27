from __future__ import annotations

from services.fraud.inference.model import ScoringResult, score_models


def predict(**entities: str) -> ScoringResult:
    """Predict using features from FeatureService.

    Args:
        **entities: Entity key-value pairs (e.g., customer_id="...", email="...")

    Returns:
        ScoringResult with score and model comparisons
    """
    if not entities:
        return ScoringResult(
            score=0.5,
            scored_by="none",
            features=None,
        )

    from services.fraud.features import get_features

    features = get_features(**entities)
    return score_models(features)
