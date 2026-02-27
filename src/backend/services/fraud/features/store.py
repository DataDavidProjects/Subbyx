from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "feature_repo"

# Ensure FEAST_REDIS_HOST is set so feature_store.yaml $FEAST_REDIS_HOST expands correctly
os.environ.setdefault("FEAST_REDIS_HOST", "localhost:6379")

store = None
try:
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(REPO_PATH))
    logger.info("Feast FeatureStore initialized (repo=%s)", REPO_PATH)
except Exception as exc:
    logger.warning("Feast unavailable, running without online store: %s", exc)
