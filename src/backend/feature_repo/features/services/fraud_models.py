"""
Fraud model feature services.

Defines three Feast FeatureServices:
  - train_model_service  : ALL features (used by create_training_data.py for PIT joins)
  - fraud_model_production : selected features (loaded from selected_features.yaml)
  - fraud_model_shadow     : identity-focused challenger (fixed feature set)

Data flow:
  feature_selection.py  -->  selected_features.yaml  -->  PRODUCTION_FEATURES
                                                      -->  train_model.py
                                                      -->  fraud_model_production (Feast serving)
"""

from __future__ import annotations

from pathlib import Path

import yaml
from feast import FeatureService, FeatureView

from features.views.charges import charge_features, charge_stats_features
from features.views.checkouts import checkout_features
from features.views.customers import customer_features, customer_profile_features
from features.views.payment_intents import payment_intent_features
from features.views.addresses import address_features
from features.views.stores import store_features, store_stats_features
from features.views.geo_time import geo_time_features


# =============================================================================
# Feature Views
# =============================================================================

ALL_VIEWS: list[FeatureView] = [
    charge_features,          # per-charge event: risk_score, card_brand, card_issuer
    charge_stats_features,    # expanding-window per-email: n_charges, failures, rate
    checkout_features,        # checkout context: grade, sku, subscription_value
    payment_intent_features,  # Stripe PI: amount, status, n_failures
    customer_features,        # identity signals: name/email match scores, fiscal_code
    customer_profile_features,  # derived: n_emails_per_fiscal_code, address_mismatch
    address_features,         # raw geo: locality, city, state, postal_code
    store_features,           # store metadata: partner, province, area
    store_stats_features,     # expanding-window per-store: success_rate, avg_value
    geo_time_features,        # rolling geo-velocity + temporal (PIT-correct)
]


# =============================================================================
# Production Features
# =============================================================================
# Source of truth: selected_features.yaml (written by `make feature-select`).
# The selection pipeline (variance -> MI -> VIF) decides which features
# go into production. This file just reads the result.
#
# Fallback: _BASELINE_PRODUCTION_FEATURES is used only before the first
# selection run (e.g. fresh clone, no YAML yet).
# =============================================================================

_SELECTED_FEATURES_YAML = Path(__file__).resolve().parents[2] / "selected_features.yaml"

# Manually curated baseline — used ONLY when selected_features.yaml is absent.
# Rationale for each feature group:
_BASELINE_PRODUCTION_FEATURES = [
    # -- Payment behaviour (strongest fraud signal after identity) --
    # Fraudsters often have zero prior charges; returning users with
    # high failure rates are suspicious (card testing pattern).
    "charge_stats_features__n_charges",
    "charge_stats_features__n_failures",
    "charge_stats_features__failure_rate",

    # -- Stripe risk score (external signal) --
    # Radar's ML score; strong alone but needs local context to avoid
    # false positives on legitimate high-risk regions.
    "charge_features__outcome_risk_score",

    # -- Transaction context --
    # Higher subscription values attract more fraud; amount is the
    # single most predictive non-identity feature (MI ~0.068).
    "payment_intent_features__amount",

    # -- Store reputation --
    # Stores with historically low success rates correlate with fraud
    # clusters (compromised partner or lax KYC).
    "store_stats_features__store_success_rate",

    # -- Identity integrity --
    # Multiple emails per fiscal code = synthetic identity.
    # Address mismatch between billing and residential = classic flag.
    "customer_profile_features__n_emails_per_fiscal_code",
    "customer_profile_features__is_address_mismatch",

    # -- Temporal patterns --
    # Late-night checkouts (22-06) show ~3x fraud rate vs business hours.
    "geo_time_features__checkout_hour",
    "geo_time_features__is_late_night",

    # -- Geo-velocity (replaces static province encoding) --
    # Rolling 30d request count captures regional attack surges without
    # permanently penalizing high-traffic provinces.
    "geo_time_features__province_n_requests_30d",
]


def _load_selected_features() -> list[str]:
    """Load features from the YAML written by feature_selection.py."""
    if _SELECTED_FEATURES_YAML.exists():
        with open(_SELECTED_FEATURES_YAML) as f:
            cfg = yaml.safe_load(f)
        feats = cfg.get("selected_features", [])
        if feats:
            return feats
    return _BASELINE_PRODUCTION_FEATURES


PRODUCTION_FEATURES: list[str] = _load_selected_features()


# =============================================================================
# Shadow Features (challenger model)
# =============================================================================
# Identity-focused subset — tests whether name/email match scores alone
# can outperform the broader production set. Fixed (not driven by selection).

SHADOW_FEATURES = [
    # Identity match scores (doc vs name, email vs emails on file)
    "customer_features__doc_name_email_match_score",
    "customer_features__email_emails_match_score",
    # Minimal context to avoid pure-identity overfitting
    "payment_intent_features__amount",
    "store_stats_features__store_success_rate",
    "charge_features__outcome_risk_score",
    "charge_stats_features__n_charges",
    # Identity integrity (same as production)
    "customer_profile_features__is_address_mismatch",
    "customer_profile_features__n_emails_per_fiscal_code",
]


# =============================================================================
# View Projection Helper
# =============================================================================

def _select_from_views(
    feature_names: list[str],
    views: list[FeatureView],
) -> list[FeatureView]:
    """Route a flat feature list (view__field format) to per-view Feast projections."""
    view_map = {fv.name: fv for fv in views}
    field_to_view = {field.name: fv for fv in views for field in fv.schema}

    per_view: dict[str, list[str]] = {}
    for name in feature_names:
        if "__" in name:
            view_name, field_name = name.split("__", 1)
            if view_name in view_map:
                per_view.setdefault(view_name, []).append(field_name)
                continue

        if name in field_to_view:
            fv = field_to_view[name]
            per_view.setdefault(fv.name, []).append(name)
        else:
            raise KeyError(
                f"Feature '{name}' not found in any FeatureView. "
                f"Available views: {list(view_map.keys())}"
            )

    return [view_map[v_name][fields] for v_name, fields in per_view.items()]


# =============================================================================
# Feature Services (registered in Feast)
# =============================================================================

# Training: all views, no projection — feature_selection.py decides later
train_model_service = FeatureService(
    name="train_model_service",
    features=ALL_VIEWS,
    description="All features for training data generation (PIT join via Feast).",
)

# Production serving: only the selected features
fraud_model_production = FeatureService(
    name="fraud_model_production",
    features=_select_from_views(PRODUCTION_FEATURES, ALL_VIEWS),
    description="Production model serving — features from selected_features.yaml.",
)

# Shadow serving: identity-focused challenger
fraud_model_shadow = FeatureService(
    name="fraud_model_shadow",
    features=_select_from_views(SHADOW_FEATURES, ALL_VIEWS),
    description="Shadow challenger model — identity-focused feature subset.",
)
