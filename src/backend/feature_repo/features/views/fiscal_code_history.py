from __future__ import annotations

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Int64

from features.entities.fiscal_code import fiscal_code

fiscal_code_history_source = FileSource(
    name="fiscal_code_history_source",
    path="/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources/fiscal_code_history.parquet",
    timestamp_field="created",
)

fiscal_code_history = FeatureView(
    name="fiscal_code_history",
    entities=[fiscal_code],
    ttl=timedelta(days=0),
    schema=[
        Field(name="prior_customer_count", dtype=Int64),
    ],
    online=True,
    source=fiscal_code_history_source,
    description="Fiscal code customer history for segment determination",
)
