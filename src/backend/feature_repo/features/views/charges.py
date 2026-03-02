from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64

from features.entities.email import email

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

charge_features_source = FileSource(
    path=str(_SOURCES_DIR / "charges.parquet"),
    timestamp_field="created",
)

charge_features = FeatureView(
    name="charge_features",
    entities=[email],
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
        Field(
            name="card_brand",
            dtype=String,
            tags={
                "label": "Card Brand",
                "description": "Payment card brand (e.g. Visa, Mastercard) used on the most recent charge.",
            },
        ),
        Field(
            name="card_issuer",
            dtype=String,
            tags={
                "label": "Card Issuer",
                "description": "Name of the bank or institution that issued the payment card.",
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
