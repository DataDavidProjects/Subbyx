from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from services.fraud.inference.model_loader import (
    LoadedModel,
    model_loader,
    _config as mlflow_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCTION_MODEL_URI = "models:/fraud-detector@production"
SHADOW_MODEL_URI = "models:/fraud-detector@shadow"
DEFAULT_CANARY_TRAFFIC_PCT = 10


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------


def score_models(features: dict) -> ScoringResult:
    """Run features through production and shadow models.

    Shadow model always runs silently for comparison.
    When canary is enabled, a % of traffic is routed to shadow for A/B testing.
    """
    production_score = production_model.predict(features)

    # Shadow: score silently for offline comparison + canary traffic
    shadow_score: float | None = None
    scored_by = "production"

    if _shadow_model is not None:
        try:
            shadow_score = _shadow_model.predict(features)
            logger.debug("shadow score: %.4f", shadow_score)

            # Canary: route a % of traffic to shadow for A/B test
            if _canary_traffic_pct > 0 and random.randint(1, 100) <= _canary_traffic_pct:
                scored_by = "shadow"
                logger.debug("canary: routed to shadow")
        except Exception as exc:
            logger.warning("Shadow model failed: %s", exc)

    effective_score = shadow_score if scored_by == "shadow" else production_score

    return ScoringResult(
        score=effective_score,
        scored_by=scored_by,
        features=features,
        production_score=production_score,
        shadow_score=shadow_score,
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    score: float
    scored_by: str
    features: dict | None = None
    production_score: float | None = None
    shadow_score: float | None = None


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


class Model:
    """Lazy-loading wrapper around an MLflow model."""

    def __init__(self, name: str, uri: str | None = None) -> None:
        self._name = name
        self._uri = uri
        self._loaded: LoadedModel | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded is not None:
            return
        self._loaded = model_loader.get_model(self._name)

    @property
    def feature_columns(self) -> list[str]:
        self._ensure_loaded()
        return self._loaded.feature_columns if self._loaded else []

    def predict(self, features: dict) -> float:
        self._ensure_loaded()

        if self._loaded is None or self._loaded.model is None:
            raise RuntimeError(f"Model {self._name} is not loaded")

        try:
            cols = self.feature_columns
            row = {
                col: float(features[col]) if features.get(col) is not None else np.nan
                for col in cols
            }
            df = pd.DataFrame([row], columns=cols)
            proba = self._loaded.model.predict_proba(df)[0]
            score = float(proba[1]) if len(proba) > 1 else float(proba[0])
            logger.info("[MODEL] %s SCORE=%.4f", self._name, score)
            return score
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[MODEL] %s predict failed: %s", self._name, exc)
            raise RuntimeError(f"Model {self._name} prediction failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Module-level instances (initialised once at import time)
# ---------------------------------------------------------------------------


production_model = Model("production", PRODUCTION_MODEL_URI)

# Shadow model: runs alongside production for offline comparison + canary traffic
_shadow_model: Model | None = None
_shadow_config = mlflow_config.get("shadow", {})
if _shadow_config.get("enabled"):
    _shadow_model = Model("shadow", _shadow_config.get("model_uri", SHADOW_MODEL_URI))

# Canary: route a % of live traffic to shadow for A/B testing
_canary_traffic_pct: int = 0
_canary_config = _shadow_config.get("canary", {})
if _canary_config.get("enabled"):
    _canary_traffic_pct = _canary_config.get("traffic_percentage", DEFAULT_CANARY_TRAFFIC_PCT)
