from __future__ import annotations

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Int64

from features.entities.email import email

email_history_source = FileSource(
    name="email_history_source",
    path="/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources/email_history.parquet",
    timestamp_field="created",
)

email_history = FeatureView(
    name="email_history",
    entities=[email],
    ttl=timedelta(days=0),
    schema=[
        Field(name="prior_charge_count", dtype=Int64),
    ],
    online=True,
    source=email_history_source,
    description="Email charge history for segment determination",
)
