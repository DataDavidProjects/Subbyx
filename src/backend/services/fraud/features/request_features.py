from __future__ import annotations

from datetime import datetime
from typing import Any

from services.fraud.context import CheckoutContext

REQUEST_FEATURE_SCHEMA: dict[str, dict[str, Any]] = {
    "subscription_value": {"field": "subscription_value", "default": 0.0, "dtype": "float"},
    "grade": {"field": "grade", "default": "", "dtype": "str"},
    "category": {"field": "category", "default": "", "dtype": "str"},
    "gender": {"field": "gender", "default": "", "dtype": "str"},
    "birth_date": {"field": "birth_date", "default": "", "dtype": "str"},
    "birth_province": {"field": "birth_province", "default": "", "dtype": "str"},
    "birth_country": {"field": "birth_country", "default": "", "dtype": "str"},
    "has_high_end_device": {"field": "has_high_end_device", "default": False, "dtype": "bool"},
    "is_night_time": {"field": "is_night_time", "default": False, "dtype": "bool"},
    "is_high_value": {"field": "is_high_value", "default": False, "dtype": "bool"},
    "email_domain": {"field": "email_domain", "default": "", "dtype": "str"},
    # Category risk features (derived)
    "is_high_risk_category": {"field": None, "default": False, "dtype": "bool"},
    "is_storage_variant": {"field": None, "default": False, "dtype": "bool"},
    "is_smartphone_or_watch": {"field": None, "default": False, "dtype": "bool"},
    "category_risk_tier": {"field": None, "default": "low", "dtype": "str"},
    # Card risk features for cold start (derived from checkout context)
    "card_brand": {"field": "card_brand", "default": "", "dtype": "str"},
    "card_funding": {"field": "card_funding", "default": "", "dtype": "str"},
    "card_cvc_check": {"field": "card_cvc_check", "default": "", "dtype": "str"},
    "is_debit_card": {"field": None, "default": False, "dtype": "bool"},
    "is_prepaid_card": {"field": None, "default": False, "dtype": "bool"},
    "is_high_risk_card": {"field": None, "default": False, "dtype": "bool"},
    "card_cvc_fail": {"field": None, "default": False, "dtype": "bool"},
    "card_cvc_unavailable": {"field": None, "default": False, "dtype": "bool"},
}


def extract_request_features(ctx: CheckoutContext) -> dict[str, Any]:
    """Extract model-ready request features from a CheckoutContext.

    Returns a dict with consistent keys regardless of None fields in context.
    """
    features: dict[str, Any] = {}

    # 1. Base features from context fields (skip derived features with field=None)
    for feature_name, spec in REQUEST_FEATURE_SCHEMA.items():
        if spec["field"] is None:
            continue  # Derived features handled separately below
        if hasattr(ctx, spec["field"]):
            value = getattr(ctx, spec["field"], None)
            if value is None or value == "":
                value = spec["default"]
            features[feature_name] = value

    # 2. Derived: is_night_time (22:00 - 06:00)
    try:
        # Example timestamp: 2024-08-11 07:29:43
        dt = datetime.strptime(ctx.timestamp, "%Y-%m-%d %H:%M:%S")
        features["is_night_time"] = dt.hour >= 22 or dt.hour < 6
    except (ValueError, TypeError):
        features["is_night_time"] = REQUEST_FEATURE_SCHEMA["is_night_time"]["default"]

    # 3. Derived: is_high_value (> 100)
    features["is_high_value"] = ctx.subscription_value > 100.0

    # 4. Derived: email_domain
    if ctx.email and "@" in ctx.email:
        features["email_domain"] = ctx.email.split("@")[-1].lower()
    else:
        features["email_domain"] = REQUEST_FEATURE_SCHEMA["email_domain"]["default"]

    # 5. Derived: category risk features (initialize with defaults)
    features["is_storage_variant"] = REQUEST_FEATURE_SCHEMA["is_storage_variant"]["default"]
    features["is_smartphone_or_watch"] = REQUEST_FEATURE_SCHEMA["is_smartphone_or_watch"]["default"]
    features["is_high_risk_category"] = REQUEST_FEATURE_SCHEMA["is_high_risk_category"]["default"]
    features["category_risk_tier"] = REQUEST_FEATURE_SCHEMA["category_risk_tier"]["default"]

    category = str(ctx.category).strip() if ctx.category else ""
    cat_lower = category.lower()

    # Check for storage variants (e.g., "256Gb", "128Gb", "512Gb")
    features["is_storage_variant"] = any(x in cat_lower for x in ["gb", "tb", "cpu"])

    # Check for smartphone or smartwatch categories
    features["is_smartphone_or_watch"] = any(x in cat_lower for x in ["smartphone", "smartwatch"])

    # High-risk categories based on historical fraud analysis (>10% fraud rate)
    # Includes: storage variants (14.1%), smartwatches (12.6%), smartphones (11.4%)
    high_risk_keywords = ["smartphone", "smartwatch"]
    features["is_high_risk_category"] = (
        features["is_storage_variant"] or
        any(x in cat_lower for x in high_risk_keywords)
    )

    # Category risk tier: low, medium, high
    if features["is_storage_variant"]:
        features["category_risk_tier"] = "high"
    elif features["is_smartphone_or_watch"]:
        features["category_risk_tier"] = "medium"
    else:
        features["category_risk_tier"] = "low"

    # 6. Derived: card risk features for cold start (initialize with defaults)
    features["is_debit_card"] = REQUEST_FEATURE_SCHEMA["is_debit_card"]["default"]
    features["is_prepaid_card"] = REQUEST_FEATURE_SCHEMA["is_prepaid_card"]["default"]
    features["is_high_risk_card"] = REQUEST_FEATURE_SCHEMA["is_high_risk_card"]["default"]
    features["card_cvc_fail"] = REQUEST_FEATURE_SCHEMA["card_cvc_fail"]["default"]
    features["card_cvc_unavailable"] = REQUEST_FEATURE_SCHEMA["card_cvc_unavailable"]["default"]

    card_funding = str(ctx.card_funding).lower().strip() if ctx.card_funding else ""
    card_cvc = str(ctx.card_cvc_check).lower().strip() if ctx.card_cvc_check else ""

    # Card funding type risk (based on historical analysis)
    # Debit cards: 9.0% fraud rate (highest)
    # Prepaid cards: 4.5% fraud rate
    # Credit cards: 2.7% fraud rate (lowest)
    features["is_debit_card"] = card_funding == "debit"
    features["is_prepaid_card"] = card_funding == "prepaid"

    # CVC check risk indicators
    # fail: highest risk (but rare)
    # unavailable: slightly elevated risk
    features["card_cvc_fail"] = card_cvc == "fail"
    features["card_cvc_unavailable"] = card_cvc == "unavailable"

    # Composite high-risk card indicator
    # Debit cards are highest risk (9% vs 2.7% for credit)
    # CVC check failures are also high risk
    # CVC unavailable is elevated risk regardless of card type
    features["is_high_risk_card"] = (
        features["is_debit_card"] or
        features["card_cvc_fail"] or
        features["card_cvc_unavailable"]
    )

    return features
