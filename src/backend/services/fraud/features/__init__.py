from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

from services.fraud.features.base import Entity, Source
from services.fraud.features.registry import registry
from services.fraud.features.store import store

from services.fraud.features import groups  # noqa: F401

logger = logging.getLogger(__name__)

FEATURE_SERVICE_NAME = "fraud_model_production"


def _get_feature_service() -> str | None:
    """Get the active FeatureService name."""
    return FEATURE_SERVICE_NAME


def _from_feast(entity: str, key: str) -> dict:
    """Fetch features from Feast online store using FeatureService."""
    if store is None:
        logger.warning("Feast store not available")
        return {}

    feature_service = _get_feature_service()
    if not feature_service:
        return {}

    entity_enum = Entity(entity)
    features = registry.get_by_entity(entity_enum)
    if not features:
        return {}

    view_to_features: dict[str, list] = defaultdict(list)
    for feat in features:
        if feat.feature_view is not None:
            view_to_features[feat.feature_view].append(feat)

    if not view_to_features:
        return {}

    results: dict = {}
    entity_row = {entity: key}

    for view_name, feats in view_to_features.items():
        feature_refs = [f"{view_name}:{f.column or f.name}" for f in feats]

        t0 = time.perf_counter()
        try:
            response = store.get_online_features(
                features=feature_refs,
                entity_rows=[entity_row],
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            row = response.to_dict()
            for feat in feats:
                value = row.get(feat.column or feat.name, [None])[0]
                results[feat.name] = value

            logger.debug(
                "Feast lookup service=%s entity_row=%s, view=%s: %d features (%.1fms)",
                feature_service,
                entity_row,
                view_name,
                len(feats),
                elapsed_ms,
            )
        except Exception as exc:
            logger.warning(
                "Failed to fetch features for service=%s entity_row=%s, view=%s: %s",
                feature_service,
                entity_row,
                view_name,
                exc,
            )

    return results


def get_features(
    customer_id: str | None = None,
    email: str | None = None,
) -> dict:
    """Fetch features from Feast online store.

    Looks up features by customer_id and/or email from the Feast online store.
    Uses FeatureService for feature retrieval.
    """
    all_features: dict = {}
    lookups = [("customer_id", customer_id), ("email", email)]

    for entity, key in lookups:
        if not key:
            continue
        features = _from_feast(entity, key)
        for k, v in features.items():
            if all_features.get(k) is None:
                all_features[k] = v

    # Derived/computed features
    for feat in registry.get_by_source(Source.COMPUTED):
        if feat.compute_fn is not None:
            try:
                all_features[feat.name] = feat.compute_fn(all_features)
            except Exception as exc:
                logger.warning("Failed to compute feature %s: %s", feat.name, exc)

    if not all_features:
        logger.warning("no feature values found for customer_id=%s, email=%s", customer_id, email)

    return all_features
