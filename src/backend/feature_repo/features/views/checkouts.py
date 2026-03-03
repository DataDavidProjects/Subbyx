from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64

from features.entities.email import email

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

checkout_features_source = FileSource(
    path=str(_SOURCES_DIR / "checkouts.parquet"),
    timestamp_field="created",
)

checkout_features = FeatureView(
    name="checkout_features",
    entities=[email],
    description="Product and checkout attributes from the most recent session for this customer.",
    schema=[
        Field(
            name="grade",
            dtype=String,
            tags={
                "label": "Device Grade",
                "description": "Cosmetic condition grade of the device (A=Excellent, B=Good, C=Fair, new=New).",
            },
        ),
        Field(
            name="subscription_value",
            dtype=Float64,
            tags={
                "label": "Subscription Value (€)",
                "description": "Monthly subscription fee in Euros for the device being checked out.",
            },
        ),
        Field(
            name="sku",
            dtype=String,
            tags={
                "label": "Product SKU",
                "description": "Stock-keeping unit identifier for the specific product variant.",
            },
        ),
        Field(
            name="category",
            dtype=String,
            tags={
                "label": "Product Category",
                "description": "Device category (e.g. smartphones, laptops, tablets, tv, gaming).",
            },
        ),
        Field(
            name="condition",
            dtype=String,
            tags={
                "label": "Device Condition",
                "description": "Physical condition description of the device being subscribed to.",
            },
        ),
    ],
    source=checkout_features_source,
)

# ---------------------------------------------------------------------------
# Checkout velocity (rolling-window behavioral signals)
# ---------------------------------------------------------------------------

checkout_velocity_source = FileSource(
    path=str(_SOURCES_DIR / "checkout_velocity.parquet"),
    timestamp_field="created",
)

checkout_velocity_features = FeatureView(
    name="checkout_velocity_features",
    entities=[email],
    description="Rolling-window checkout velocity signals: frequency, failure patterns, category switching, value downtrade.",
    schema=[
        Field(
            name="n_checkouts_7d",
            dtype=Float64,
            tags={
                "label": "Checkouts (7d)",
                "description": "Total checkouts by this email in the prior 7 days.",
            },
        ),
        Field(
            name="n_expired_7d",
            dtype=Float64,
            tags={
                "label": "Expired Checkouts (7d)",
                "description": "Expired (failed) checkouts by this email in the prior 7 days.",
            },
        ),
        Field(
            name="expired_ratio_7d",
            dtype=Float64,
            tags={
                "label": "Expired Ratio (7d)",
                "description": "Fraction of checkouts that expired in the prior 7 days (0 if none).",
            },
        ),
        Field(
            name="n_distinct_categories_30d",
            dtype=Float64,
            tags={
                "label": "Distinct Categories (30d)",
                "description": "Number of distinct product categories checked out in the prior 30 days.",
            },
        ),
        Field(
            name="max_value_30d",
            dtype=Float64,
            tags={
                "label": "Max Subscription Value (30d)",
                "description": "Maximum subscription value (€) across checkouts in the prior 30 days.",
            },
        ),
    ],
    source=checkout_velocity_source,
)
