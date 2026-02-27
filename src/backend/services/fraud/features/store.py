from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "feature_repo"

store = None
try:
    from feast import FeatureStore

    store = FeatureStore(repo_path=str(REPO_PATH))
    logger.info("Feast FeatureStore initialized (repo=%s)", REPO_PATH)
except Exception as exc:
    logger.warning("Feast unavailable, running without online store: %s", exc)
