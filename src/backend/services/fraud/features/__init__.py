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
    """Fetch features from Feast online store.

    Args:
        **entities: Entity key-value pairs to lookup (e.g., customer_id="...", email="...")

    Returns:
        dict: Merged feature values from all entity lookups.
    """
    required_entities = _get_required_entities()
    all_features: dict = {}

    for entity_name, key in entities.items():
        if not key:
            continue
        if entity_name not in required_entities:
            logger.debug("Skipping entity %s - not in FeatureService", entity_name)
            continue
        features = _from_feast(entity_name, key)
        for k, v in features.items():
            if all_features.get(k) is None:
                all_features[k] = v

    if not all_features:
        logger.warning("no feature values found for entities=%s", entities)

    return all_features
