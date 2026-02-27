from __future__ import annotations

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64

from features.entities.customer_id import customer_id

customer_source = FileSource(
    name="customer_source",
    path="/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources/customer_features.parquet",
    timestamp_field="created",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer_id],
    ttl=timedelta(days=0),
    schema=[
        Field(name="email_emails_match_score", dtype=Float64),
        Field(name="doc_name_email_match_score", dtype=Float64),
        Field(name="high_end_count", dtype=Int64),
        Field(name="high_end_rate", dtype=Float64),
    ],
    online=True,
    source=customer_source,
    description="Customer features by customer_id",
)
