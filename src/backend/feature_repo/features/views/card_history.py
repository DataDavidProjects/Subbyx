from __future__ import annotations

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Int64

from features.entities.card_fingerprint import card_fingerprint

card_history_source = FileSource(
    name="card_history_source",
    path="/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources/card_history.parquet",
    timestamp_field="created",
)

card_history = FeatureView(
    name="card_history",
    entities=[card_fingerprint],
    ttl=timedelta(days=0),
    schema=[
        Field(name="prior_card_charge_count", dtype=Int64),
    ],
    online=True,
    source=card_history_source,
    description="Card fingerprint charge history for segment determination",
)
