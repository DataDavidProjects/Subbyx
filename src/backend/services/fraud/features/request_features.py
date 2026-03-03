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


def _schema_default(feature_name: str) -> Any:
    return REQUEST_FEATURE_SCHEMA[feature_name]["default"]


def _normalize_str(value: Any) -> str:
    return str(value).strip().lower() if value else ""


def _extract_base_features(ctx: CheckoutContext) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for feature_name, spec in REQUEST_FEATURE_SCHEMA.items():
        field_name = spec["field"]
        if field_name is None or not hasattr(ctx, field_name):
            continue

        value = getattr(ctx, field_name, None)
        if value is None or value == "":
            value = spec["default"]
        features[feature_name] = value
    return features


def _derive_is_night_time(timestamp: str | None) -> bool:
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.hour >= 22 or dt.hour < 6
    except (ValueError, TypeError):
        return _schema_default("is_night_time")


def _derive_email_domain(email: str | None) -> str:
    if email and "@" in email:
        return email.split("@")[-1].lower()
    return _schema_default("email_domain")


def _derive_category_risk_features(category: str | None) -> dict[str, Any]:
    cat_lower = _normalize_str(category)
    is_storage_variant = any(keyword in cat_lower for keyword in ("gb", "tb", "cpu"))
    is_smartphone_or_watch = any(
        keyword in cat_lower for keyword in ("smartphone", "smartwatch")
    )
    is_high_risk_category = is_storage_variant or is_smartphone_or_watch

    if is_storage_variant:
        category_risk_tier = "high"
    elif is_smartphone_or_watch:
        category_risk_tier = "medium"
    else:
        category_risk_tier = "low"

    return {
        "is_storage_variant": is_storage_variant,
        "is_smartphone_or_watch": is_smartphone_or_watch,
        "is_high_risk_category": is_high_risk_category,
        "category_risk_tier": category_risk_tier,
    }


def _derive_card_risk_features(card_funding: str | None, card_cvc_check: str | None) -> dict[str, bool]:
    funding = _normalize_str(card_funding)
    cvc = _normalize_str(card_cvc_check)

    is_debit_card = funding == "debit"
    is_prepaid_card = funding == "prepaid"
    card_cvc_fail = cvc == "fail"
    card_cvc_unavailable = cvc == "unavailable"

    return {
        "is_debit_card": is_debit_card,
        "is_prepaid_card": is_prepaid_card,
        "card_cvc_fail": card_cvc_fail,
        "card_cvc_unavailable": card_cvc_unavailable,
        "is_high_risk_card": is_debit_card or card_cvc_fail or card_cvc_unavailable,
    }


def extract_request_features(ctx: CheckoutContext) -> dict[str, Any]:
    """Extract model-ready request features from a CheckoutContext.

    Returns a dict with consistent keys regardless of None fields in context.
    """
    features = _extract_base_features(ctx)

    # Derived from checkout context and normalized in one place.
    features["is_night_time"] = _derive_is_night_time(ctx.timestamp)
    features["is_high_value"] = ctx.subscription_value > 100.0
    features["email_domain"] = _derive_email_domain(ctx.email)

    features.update(_derive_category_risk_features(ctx.category))
    features.update(_derive_card_risk_features(ctx.card_funding, ctx.card_cvc_check))

    return features
