from __future__ import annotations

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64

from features.entities.email import email

charge_features_source = FileSource(
    name="charge_features_source",
    path="/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources/charge_features.parquet",
    timestamp_field="created",
)

charge_features = FeatureView(
    name="charge_features",
    entities=[email],
    ttl=timedelta(days=0),
    schema=[
        Field(name="distinct_cards", dtype=Int64),
        Field(name="prepaid_card_rate", dtype=Float64),
        Field(name="blocked_rate", dtype=Float64),
        Field(name="avg_risk_score", dtype=Float64),
        Field(name="max_risk_score", dtype=Float64),
    ],
    online=True,
    source=charge_features_source,
    description="Charge-level features aggregated by email",
)
