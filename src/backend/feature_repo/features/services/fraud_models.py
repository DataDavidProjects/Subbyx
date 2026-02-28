from __future__ import annotations

from feast import FeatureService, FeatureView

from features.views.charges import charge_features
from features.views.customers import customer_features

ALL_VIEWS = [customer_features, charge_features]

# Paste feature selection output here — auto-routed to views below
PRODUCTION_FEATURES = [
    "email_emails_match_score",
    "doc_name_email_match_score",
    "high_end_count",
    "high_end_rate",
    "distinct_cards",
    "prepaid_card_rate",
    "blocked_rate",
    "avg_risk_score",
    "max_risk_score",
]


SHADOW_FEATURES = [
    "email_emails_match_score",
    "doc_name_email_match_score",
    "high_end_count",
]


def _select_from_views(
    feature_names: list[str],
    views: list[FeatureView],
) -> list[FeatureView]:
    """Route a flat feature list to per-view projections."""
    view_lookup = {field.name: fv for fv in views for field in fv.schema}
    per_view: dict[str, list[str]] = {}
    for name in feature_names:
        fv = view_lookup[name]  # KeyError = unknown feature
        per_view.setdefault(fv.name, []).append(name)

    return [fv[per_view[fv.name]] for fv in views if fv.name in per_view]


fraud_model_production = FeatureService(
    name="fraud_model_production",
    features=_select_from_views(PRODUCTION_FEATURES, ALL_VIEWS),
    description="Feature service for fraud detection production model",
)

fraud_model_shadow = FeatureService(
    name="fraud_model_shadow",
    features=_select_from_views(SHADOW_FEATURES, ALL_VIEWS),
    description="Feature service for fraud detection shadow model",
)
