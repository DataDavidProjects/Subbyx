"""
Feature metadata utilities.

Reads label/description tags from Feast FeatureView Field definitions
and exposes a mapping used to enrich API responses.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


# Metadata for features that are NOT Feast view fields (request features,
# missing indicators).  These are computed at inference/training time and
# have no Feast Field tags, so we define them here.
_NON_FEAST_METADATA: dict[str, dict[str, str]] = {
    "subscription_value": {
        "label": "Subscription Value (EUR)",
        "description": "Monthly subscription fee from the checkout request.",
    },
    "charge_features__missing": {
        "label": "No Charge History",
        "description": (
            "1 if no charge/payment card data exists for this customer "
            "(new customer or no prior transactions)."
        ),
    },
    "pi_features__missing": {
        "label": "No Payment Intent History",
        "description": (
            "1 if no payment intent data exists for this customer "
            "(no prior payment attempts recorded)."
        ),
    },
    "address_features__missing": {
        "label": "No Address Data",
        "description": "1 if no residential address is on file for this customer.",
    },
}


@lru_cache(maxsize=1)
def get_feature_metadata() -> dict[str, dict[str, str]]:
    """
    Return a mapping of full feature name -> {label, description}.

    Full feature names use the 'view_name__field_name' convention
    (same as full_feature_names=True returned by Feast).

    Also includes metadata for non-Feast features (request features,
    missing indicators) defined in ``_NON_FEAST_METADATA``.

    Result is cached so the registry is only read once per process.
    """
    try:
        from services.fraud.features.store import store

        if store is None:
            logger.warning("Feast store not available – no metadata returned")
            return dict(_NON_FEAST_METADATA)

        metadata: dict[str, dict[str, str]] = {}

        for fv in store.list_feature_views():
            for field in fv.schema:
                full_name = f"{fv.name}__{field.name}"
                tags = field.tags or {}
                metadata[full_name] = {
                    "label": tags.get("label", field.name.replace("_", " ").title()),
                    "description": tags.get("description", ""),
                }

        # Add non-Feast feature metadata
        metadata.update(_NON_FEAST_METADATA)

        logger.info("Loaded metadata for %d features from Feast registry", len(metadata))
        return metadata

    except Exception as exc:
        logger.warning("Failed to load feature metadata from Feast: %s", exc)
        return dict(_NON_FEAST_METADATA)
