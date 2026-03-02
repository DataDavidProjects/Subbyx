"""
Feature metadata utilities.

Reads label/description tags from Feast FeatureView Field definitions
and exposes a mapping used to enrich API responses.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_feature_metadata() -> dict[str, dict[str, str]]:
    """
    Return a mapping of full feature name -> {label, description}.

    Full feature names use the 'view_name__field_name' convention
    (same as full_feature_names=True returned by Feast).

    Result is cached so the registry is only read once per process.
    """
    try:
        from services.fraud.features.store import store

        if store is None:
            logger.warning("Feast store not available – no metadata returned")
            return {}

        metadata: dict[str, dict[str, str]] = {}

        for fv in store.list_feature_views():
            for field in fv.schema:
                full_name = f"{fv.name}__{field.name}"
                tags = field.tags or {}
                metadata[full_name] = {
                    "label": tags.get("label", field.name.replace("_", " ").title()),
                    "description": tags.get("description", ""),
                }

        logger.info("Loaded metadata for %d features from Feast registry", len(metadata))
        return metadata

    except Exception as exc:
        logger.warning("Failed to load feature metadata from Feast: %s", exc)
        return {}
