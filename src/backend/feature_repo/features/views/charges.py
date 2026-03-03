from datetime import timedelta
from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64

from features.entities.email import email
from features.entities.card_fingerprint import card_fingerprint

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

charge_features_source = FileSource(
    path=str(_SOURCES_DIR / "charges.parquet"),
    timestamp_field="created",
)

charge_features = FeatureView(
    name="charge_features",
    entities=[email],
    ttl=timedelta(0),
    description="Latest charge attributes for the customer's email.",
    schema=[
        Field(
            name="outcome_risk_score",
            dtype=Float64,
            tags={
                "label": "Stripe Risk Score",
                "description": "Stripe's machine-learning risk score for the most recent charge (0–100). Higher values indicate higher fraud risk.",
            },
        ),
    ],
    source=charge_features_source,
)

charge_stats_source = FileSource(
    path=str(_SOURCES_DIR / "charge_stats.parquet"),
    timestamp_field="created",
)

charge_stats_features = FeatureView(
    name="charge_stats_features",
    entities=[email],
    ttl=timedelta(0),
    description="Aggregated charge history statistics per customer email.",
    schema=[
        Field(
            name="n_charges",
            dtype=Float64,
            tags={
                "label": "Total Charges",
                "description": "Total number of payment charges attempted by this customer.",
            },
        ),
        Field(
            name="n_failures",
            dtype=Float64,
            tags={
                "label": "Failed Charges",
                "description": "Number of payment charges that failed for this customer.",
            },
        ),
        Field(
            name="failure_rate",
            dtype=Float64,
            tags={
                "label": "Charge Failure Rate",
                "description": "Ratio of failed charges to total charges (0.0–1.0). High values indicate payment issues.",
            },
        ),
    ],
    source=charge_stats_source,
)

card_features_source = FileSource(
    path=str(_SOURCES_DIR / "card_features.parquet"),
    timestamp_field="created",
)

card_features = FeatureView(
    name="card_features",
    entities=[card_fingerprint],
    ttl=timedelta(0),
    description="Latest card attributes per card fingerprint.",
    schema=[
        Field(
            name="card_brand",
            dtype=String,
            tags={
                "label": "Card Brand",
                "description": "Payment card brand (e.g. Visa, Mastercard).",
            },
        ),
        Field(
            name="card_funding",
            dtype=String,
            tags={
                "label": "Card Funding",
                "description": "Card funding type (e.g. credit, debit).",
            },
        ),
        Field(
            name="card_cvc_check",
            dtype=String,
            tags={
                "label": "CVC Check Result",
                "description": "Result of CVC verification check.",
            },
        ),
    ],
    source=card_features_source,
)
