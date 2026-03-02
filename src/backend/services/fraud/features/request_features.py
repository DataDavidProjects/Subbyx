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

    return features
