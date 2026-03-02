"""
Feast FeatureView for rolling geo-velocity and temporal features.

Auto-generates Field entries for all window/stat combinations
to stay in sync with the compute module's PROVINCE_WINDOWS / POSTAL_WINDOWS.
"""

from pathlib import Path
from feast import FeatureView, FileSource, Field
from feast.types import Float64
from features.entities.email import email

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

# Must match compute/geo_time_features.py
PROVINCE_WINDOWS = [5, 10, 30, 60]
POSTAL_WINDOWS = [5, 10, 30]

_PROVINCE_STAT_META = {
    "n_requests": (
        "Province Requests ({w}d)",
        "Number of Subbyx checkouts from the customer's home province in the prior {w} days. "
        "A surge signals an active fraud attack cluster in the region.",
    ),
    "n_frauds": (
        "Province Confirmed Frauds ({w}d)",
        "Number of confirmed dunning cases originating from this province in the prior {w} days.",
    ),
    "fraud_rate": (
        "Province Fraud Rate ({w}d)",
        "Rolling fraud rate for this province over the prior {w} days (0.0–1.0). "
        "Decays naturally when an attack dissipates — avoids static geo bias.",
    ),
}

_POSTAL_STAT_META = {
    "n_requests": (
        "Postal Code Requests ({w}d)",
        "Checkout volume from this postal code in the prior {w} days. "
        "Detects hyper-local attack clusters faster than province-level signal.",
    ),
    "n_frauds": (
        "Postal Code Confirmed Frauds ({w}d)",
        "Confirmed fraud cases from this exact postal code in the prior {w} days.",
    ),
    "fraud_rate": (
        "Postal Code Fraud Rate ({w}d)",
        "Rolling fraud rate for this postal code over the prior {w} days (0.0–1.0). "
        "Tighter scope and shorter window means faster attack detection.",
    ),
}

# Temporal fields (static metadata)
_TEMPORAL_FIELDS = [
    Field(
        name="checkout_hour",
        dtype=Float64,
        tags={
            "label": "Checkout Hour (0–23)",
            "description": (
                "Hour of day the checkout was submitted. "
                "Hour 23 shows ~67% historical fraud rate vs 10–15% during business hours."
            ),
        },
    ),
    Field(
        name="checkout_dow",
        dtype=Float64,
        tags={
            "label": "Checkout Day of Week (0=Mon)",
            "description": "Day of week (0=Monday, 6=Sunday). Saturday shows elevated fraud; Sunday is lowest risk.",
        },
    ),
    Field(
        name="is_weekend",
        dtype=Float64,
        tags={
            "label": "Is Weekend",
            "description": "Flag (1) if the checkout was submitted on Saturday or Sunday.",
        },
    ),
    Field(
        name="is_late_night",
        dtype=Float64,
        tags={
            "label": "Is Late Night (22–06)",
            "description": "Flag (1) if the checkout was submitted between 22:00 and 06:00.",
        },
    ),
]

# Build Province Fields
_province_fields = []
for w in PROVINCE_WINDOWS:
    for stat, (label_tpl, desc_tpl) in _PROVINCE_STAT_META.items():
        _province_fields.append(
            Field(
                name=f"province_{stat}_{w}d",
                dtype=Float64,
                tags={
                    "label": label_tpl.format(w=w),
                    "description": desc_tpl.format(w=w),
                },
            )
        )

# Build Postal Fields
_postal_fields = []
for w in POSTAL_WINDOWS:
    for stat, (label_tpl, desc_tpl) in _POSTAL_STAT_META.items():
        _postal_fields.append(
            Field(
                name=f"postal_{stat}_{w}d",
                dtype=Float64,
                tags={
                    "label": label_tpl.format(w=w),
                    "description": desc_tpl.format(w=w),
                },
            )
        )

geo_time_features_source = FileSource(
    path=str(_SOURCES_DIR / "geo_time_features.parquet"),
    timestamp_field="created",
)

geo_time_features = FeatureView(
    name="geo_time_features",
    entities=[email],
    description=(
        "Rolling geo-velocity and temporal features. "
        "All rolling features are point-in-time correct (strict look-back, current event excluded). "
        f"Province windows: {PROVINCE_WINDOWS}d. Postal windows: {POSTAL_WINDOWS}d."
    ),
    schema=_TEMPORAL_FIELDS + _province_fields + _postal_fields,
    source=geo_time_features_source,
)
