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
    """Fetch all features from Feast online store by performing lookups
    for each provided entity key and merging results.

    This avoids the 'Missing join key values' KeyError that occurs when
    a FeatureService combines views with different entities but the
    lookup row doesn't contain all of them.

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

    results = {}
    t0 = time.perf_counter()

    # 1. Map entities to their relevant feature views in the service
    # so we only perform necessary lookups.
    entity_to_views: dict[str, list] = {}
    for projection in feature_service.feature_view_projections:
        fv = store.get_feature_view(projection.name)
        for ent in fv.entities:
            ent_name = ent if isinstance(ent, str) else ent.name
            if ent_name in entities and entities[ent_name]:
                entity_to_views.setdefault(ent_name, []).append(projection)

    # 2. Perform one lookup per entity that we actually have a value for
    for ent_name, projections in entity_to_views.items():
        val = entities[ent_name]
        if not val or val == "nan":
            continue

        try:
            response = store.get_online_features(
                features=projections,
                entity_rows=[{ent_name: val}],
                full_feature_names=True,
            )
            data = response.to_dict()
            for feat_name, values in data.items():
                if feat_name == ent_name:
                    continue
                val_out = values[0] if values else None
                import math
                if isinstance(val_out, float) and math.isnan(val_out):
                    val_out = None
                
                # Only overwrite if the new value is NOT null, 
                # or if we don't have a value yet.
                if val_out is not None or feat_name not in results:
                    results[feat_name] = val_out

        except Exception as exc:
            logger.error("Feast lookup failed for entity %s=%s: %s", ent_name, val, exc)

    # 3. Add extra features needed by Rules Engine if missing
    # (Similar to step 2 in previous version, but using individual lookups)
    extra_rule_fields = {
        "email": [
            "charge_stats_features:n_charges",
            "charge_stats_features:n_failures",
            "charge_stats_features:failure_rate",
            "payment_intent_stats_features:n_payment_intents",
            "payment_intent_stats_features:n_failures",
            "payment_intent_stats_features:failure_rate",
        ]
    }
    
    for ent_name, fields in extra_rule_fields.items():
        if ent_name in entities and entities[ent_name]:
            try:
                response = store.get_online_features(
                    features=fields,
                    entity_rows=[{ent_name: entities[ent_name]}],
                    full_feature_names=True,
                )
                data = response.to_dict()
                for feat_name, values in data.items():
                    if feat_name == ent_name: continue
                    val_out = values[0] if values else None
                    if val_out is not None or feat_name not in results:
                        results[feat_name] = val_out
            except Exception:
                pass

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "Feast multi-lookup service=%s entities=%s: %d features (%.1fms)",
        FEATURE_SERVICE_NAME,
        list(entities.keys()),
        len(results),
        elapsed_ms,
    )
    return results
