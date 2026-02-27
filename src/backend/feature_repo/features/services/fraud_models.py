from __future__ import annotations

from feast import FeatureService, FeatureView

from features.views.customers import customer_features


def get_all_feature_views() -> list[FeatureView]:
    """Return all FeatureViews used by fraud detection."""
    return [customer_features]


fraud_model_production = FeatureService(
    name="fraud_model_production",
    features=get_all_feature_views(),
    description="Feature service for fraud detection production model",
)

fraud_model_shadow = FeatureService(
    name="fraud_model_shadow",
    features=get_all_feature_views(),
    description="Feature service for fraud detection shadow model",
)
