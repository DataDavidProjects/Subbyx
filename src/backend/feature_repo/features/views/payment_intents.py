from datetime import timedelta
from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64, Int64

from features.entities.email import email

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

payment_intent_features_source = FileSource(
    path=str(_SOURCES_DIR / "payment_intents.parquet"),
    timestamp_field="created",
)

payment_intent_features = FeatureView(
    name="payment_intent_features",
    entities=[email],
    ttl=timedelta(0),
    description="Payment intent attributes for the most recent transaction attempt.",
    schema=[
        Field(
            name="amount",
            dtype=Float64,
            tags={
                "label": "Payment Amount (€)",
                "description": "Total transaction amount in Euros as recorded on the payment intent.",
            },
        ),
        Field(
            name="subscription_value",
            dtype=Float64,
            tags={
                "label": "Subscription Value (€)",
                "description": "Declared subscription value in Euros associated with the payment intent.",
            },
        ),
        Field(
            name="n_failures",
            dtype=Int64,
            tags={
                "label": "Payment Intent Failures",
                "description": "Number of failed payment attempts on this specific payment intent.",
            },
        ),
        Field(
            name="status",
            dtype=String,
            tags={
                "label": "Payment Status",
                "description": "Current status of the payment intent (e.g. succeeded, requires_payment_method, canceled).",
            },
        ),
    ],
    source=payment_intent_features_source,
)

payment_intent_stats_source = FileSource(
    path=str(_SOURCES_DIR / "payment_intent_stats.parquet"),
    timestamp_field="created",
)

payment_intent_stats_features = FeatureView(
    name="payment_intent_stats_features",
    entities=[email],
    ttl=timedelta(0),
    description="Aggregated payment intent history statistics per customer email.",
    schema=[
        Field(
            name="n_payment_intents",
            dtype=Float64,
            tags={
                "label": "Total Payment Intents",
                "description": "Total number of payment intents attempted by this customer.",
            },
        ),
        Field(
            name="n_failures",
            dtype=Float64,
            tags={
                "label": "Failed Payment Intents",
                "description": "Number of payment intents that failed or require payment method.",
            },
        ),
        Field(
            name="failure_rate",
            dtype=Float64,
            tags={
                "label": "Payment Intent Failure Rate",
                "description": "Ratio of failed payment intents to total (0.0–1.0). High values indicate payment issues.",
            },
        ),
        Field(
            name="n_succeeded",
            dtype=Float64,
            tags={
                "label": "Succeeded Payments",
                "description": "Number of payment intents that succeeded.",
            },
        ),
        Field(
            name="success_rate",
            dtype=Float64,
            tags={
                "label": "Payment Success Rate",
                "description": "Ratio of succeeded payments to total (0.0–1.0). Higher is better.",
            },
        ),
    ],
    source=payment_intent_stats_source,
)
