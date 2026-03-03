from datetime import timedelta
from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64, Int64

from features.entities.customer_id import customer_id

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

customer_features_source = FileSource(
    path=str(_SOURCES_DIR / "customers.parquet"),
    timestamp_field="created",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer_id],
    ttl=timedelta(0),
    description="Base identity attributes for the customer.",
    schema=[
        Field(
            name="fiscal_code",
            dtype=String,
            tags={
                "label": "Fiscal Code",
                "description": "National tax identifier (codice fiscale) of the customer.",
            },
        ),
        Field(
            name="card_owner_names_card_owner_names_match_score",
            dtype=Float64,
            tags={
                "label": "Cardholder Name Match",
                "description": "Similarity score between the name on the payment card and the account name (0.0–1.0). Low scores indicate a potential mismatch.",
            },
        ),
        Field(
            name="doc_name_email_match_score",
            dtype=Float64,
            tags={
                "label": "Document vs Email Name Match",
                "description": "Similarity score between the name on the identity document and the name derived from the email address (0.0–1.0).",
            },
        ),
        Field(
            name="email_emails_match_score",
            dtype=Float64,
            tags={
                "label": "Email Consistency Score",
                "description": "Similarity score between the email used at checkout and previously known emails for this fiscal code (0.0–1.0). Low scores suggest a new or suspicious email.",
            },
        ),
        Field(
            name="account_card_names_match_score",
            dtype=Float64,
            tags={
                "label": "Account vs Card Name Match",
                "description": "Similarity score between the Subbyx account name and the cardholder name (0.0–1.0).",
            },
        ),
        Field(
            name="high_end_count",
            dtype=Float64,
            tags={
                "label": "High-End Device Count",
                "description": "Number of high-end devices on file for this customer.",
            },
        ),
        Field(
            name="high_end_rate",
            dtype=Float64,
            tags={
                "label": "High-End Device Rate",
                "description": "Fraction of transactions associated with high-end devices (0.0–1.0).",
            },
        ),
    ],
    source=customer_features_source,
)

customer_profile_source = FileSource(
    path=str(_SOURCES_DIR / "customer_profile.parquet"),
    timestamp_field="created",
)

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer_id],
    ttl=timedelta(0),
    description="Derived behavioral and anomaly profile features per customer.",
    schema=[
        Field(
            name="n_emails_per_fiscal_code",
            dtype=Int64,
            tags={
                "label": "Emails per Fiscal Code",
                "description": "Number of distinct emails associated with this customer's fiscal code. High values may indicate account sharing or identity abuse.",
            },
        ),
        Field(
            name="is_address_mismatch",
            dtype=Int64,
            tags={
                "label": "Address Mismatch",
                "description": "Flag (0 or 1) indicating whether the residential address and the shipping address differ. A mismatch can be a fraud signal.",
            },
        ),
    ],
    source=customer_profile_source,
)
