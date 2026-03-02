from __future__ import annotations

import logging
import time

from services.fraud.features.store import store

logger = logging.getLogger(__name__)

FEATURE_SERVICE_NAME = "fraud_model_production"


def _get_feature_service():
    """Get the FeatureService object."""
    if store is None:
        return None
    try:
        return store.get_feature_service(FEATURE_SERVICE_NAME)
    except Exception as exc:
        logger.warning("Failed to get feature service %s: %s", FEATURE_SERVICE_NAME, exc)
        return None


def _get_required_entities() -> set[str]:
    """Get entity names required by the FeatureService."""
    feature_service = _get_feature_service()
    if feature_service is None:
        return {"customer_id"}  # fallback

    entities = set()
    for proj in feature_service.feature_view_projections:
        for join_key in proj.join_key_map.values():
            entities.add(join_key)
        if not proj.join_key_map and proj.features:
            fv = store.get_feature_view(proj.name)
            if fv and fv.entities:
                for ent in fv.entities:
                    entities.add(ent if isinstance(ent, str) else ent.name)
    return entities if entities else {"customer_id"}


def _from_feast(entity: str, key: str) -> dict:
    """Fetch features from Feast online store using FeatureService."""
    if store is None:
        logger.warning("Feast store not available")
        return {}

    feature_service = _get_feature_service()
    if feature_service is None:
        return {}

    entity_row = {entity: key}
    t0 = time.perf_counter()

    try:
        response = store.get_online_features(
            features=feature_service,
            entity_rows=[entity_row],
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        row = response.to_dict()
        results = {}
        for feat_name, values in row.items():
            results[feat_name] = values[0] if values else None

        logger.info(
            "Feast lookup service=%s entity_row=%s: %d features (%.1fms)",
            FEATURE_SERVICE_NAME,
            entity_row,
            len(results),
            elapsed_ms,
        )

        return results
    except Exception as exc:
        logger.warning(
            "Failed to fetch features for service=%s entity_row=%s: %s",
            FEATURE_SERVICE_NAME,
            entity_row,
            exc,
        )
        return {}


def get_features(**entities: str) -> dict:
    """Fetch all features from Feast online store using a single lookup row.

    Args:
        **entities: Entity key-value pairs (email, customer_id, store_id, etc.)

    Returns:
        dict: Features with view__feature prefixed names.
    """
    if store is None:
        logger.warning("Feast store not available")
        return {}

    feature_service = _get_feature_service()
    if feature_service is None:
        return {}

    # 1. Build a single entity row with all provided keys
    # Clean up empty values and non-required entities
    # If a required entity is missing, use a placeholder so Feast lookup doesn't fail
    required_entities = _get_required_entities()
    entity_row = {}
    for name in required_entities:
        val = entities.get(name)
        if val is None or val == "" or val == "nan":
            val = "__UNKNOWN__"
        entity_row[name] = val

    if not entity_row:
        logger.warning("No valid entity keys provided for Feast lookup. Entities=%s", entities)
        return {}

    t0 = time.perf_counter()
    try:
        # Fetch features for the whole service in one go
        response = store.get_online_features(
            features=feature_service,
            entity_rows=[entity_row],
            full_feature_names=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        data = response.to_dict()
        results = {}
        for feat_name, values in data.items():
            if feat_name in required_entities:
                continue
            val = values[0] if values else None
            # Convert NaN to None for JSON serialization
            import math

            if isinstance(val, float) and math.isnan(val):
                val = None
            results[feat_name] = val

        logger.info(
            "Feast lookup service=%s entities=%s: %d features (%.1fms)",
            FEATURE_SERVICE_NAME,
            list(entity_row.keys()),
            len(results),
            elapsed_ms,
        )
        return results

    except Exception as exc:
        logger.error(
            "Failed Feast lookup for service=%s entities=%s: %s",
            FEATURE_SERVICE_NAME,
            entity_row,
            exc,
        )
        return {}
