from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String

from features.entities.customer_id import customer_id

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

address_features_source = FileSource(
    path=str(_SOURCES_DIR / "addresses.parquet"),
    timestamp_field="created",
)

address_features = FeatureView(
    name="address_features",
    entities=[customer_id],
    schema=[
        Field(name="locality", dtype=String),
        Field(name="city", dtype=String),
        Field(name="state", dtype=String),
        Field(name="country", dtype=String),
        Field(name="postal_code", dtype=String),
    ],
    source=address_features_source,
)
