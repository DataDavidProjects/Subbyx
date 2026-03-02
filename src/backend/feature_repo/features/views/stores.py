from pathlib import Path

from feast import FeatureView, FileSource, Field
from feast.types import String, Float64

from features.entities.store_id import store_id

_SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"

store_features_source = FileSource(
    path=str(_SOURCES_DIR / "stores.parquet"),
    timestamp_field="created",
)

store_features = FeatureView(
    name="store_features",
    entities=[store_id],
    description="Partner store identity and geographic attributes.",
    schema=[
        Field(
            name="partner_name",
            dtype=String,
            tags={
                "label": "Store Name",
                "description": "Name of the partner store or retailer where the checkout originated.",
            },
        ),
        Field(
            name="state",
            dtype=String,
            tags={
                "label": "Store Region",
                "description": "Regional division (regione) of Italy where the store is located.",
            },
        ),
        Field(
            name="province",
            dtype=String,
            tags={
                "label": "Store Province",
                "description": "Province (provincia) of Italy where the store is located.",
            },
        ),
        Field(
            name="area",
            dtype=String,
            tags={
                "label": "Store Area",
                "description": "Broader geographic area classification for the store (e.g. Nord, Sud, Centro).",
            },
        ),
    ],
    source=store_features_source,
)

store_stats_source = FileSource(
    path=str(_SOURCES_DIR / "store_stats.parquet"),
    timestamp_field="created",
)

store_stats_features = FeatureView(
    name="store_stats_features",
    entities=[store_id],
    description="Aggregated performance statistics for the partner store.",
    schema=[
        Field(
            name="store_success_rate",
            dtype=Float64,
            tags={
                "label": "Store Success Rate",
                "description": "Historical ratio of completed (non-dunning) subscriptions at this store. Low values indicate the store is associated with more problematic customers.",
            },
        ),
        Field(
            name="store_avg_value",
            dtype=Float64,
            tags={
                "label": "Store Avg. Subscription Value (€)",
                "description": "Average subscription value (in Euros) across all checkouts at this store.",
            },
        ),
    ],
    source=store_stats_source,
)
