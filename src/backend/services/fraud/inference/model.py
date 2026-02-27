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
CANARY_MODEL_URI = "models:/fraud-detector@canary"
DEFAULT_CANARY_TRAFFIC_PCT = 10


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------


def score_models(features: dict) -> ScoringResult:
    """Run features through production, shadow, and canary models.

    Shadow model always runs silently for comparison.
    Canary model replaces production for a % of traffic when enabled.
    """
    production_score = production_model.predict(features)

    # Shadow: score silently for offline comparison
    shadow_score: float | None = None
    if _shadow_model is not None:
        try:
            shadow_score = _shadow_model.predict(features)
            logger.debug("shadow score: %.4f", shadow_score)
        except Exception as exc:
            logger.warning("Shadow model failed: %s", exc)

    # Canary: randomly route a slice of traffic to the new model
    canary_score: float | None = None
    scored_by = "production"

    if _canary_model is not None and random.randint(1, 100) <= _canary_traffic_pct:
        try:
            canary_score = _canary_model.predict(features)
            scored_by = "canary"
            logger.debug("canary score: %.4f (used)", canary_score)
        except Exception as exc:
            logger.warning("Canary model failed: %s", exc)

    effective_score = canary_score if scored_by == "canary" else production_score

    return ScoringResult(
        score=effective_score,
        scored_by=scored_by,
        features=features,
        production_score=production_score,
        shadow_score=shadow_score,
        canary_score=canary_score,
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
    canary_score: float | None = None


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

        if self._loaded and self._loaded.model is not None:
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
            except Exception as exc:
                logger.warning("[MODEL] %s predict failed: %s", self._name, exc)

        # Fallback: random score when the model is unavailable
        score = random.random()
        logger.warning("[MODEL] %s unavailable, RANDOM=%.4f", self._name, score)
        return score


# ---------------------------------------------------------------------------
# Module-level instances (initialised once at import time)
# ---------------------------------------------------------------------------


production_model = Model("production", PRODUCTION_MODEL_URI)

# Shadow model: runs alongside production for offline A/B comparison
_shadow_model: Model | None = None
_shadow_config = mlflow_config.get("shadow", {})
if _shadow_config.get("enabled"):
    _shadow_model = Model(
        "shadow", _shadow_config.get("model_uri", SHADOW_MODEL_URI)
    )

# Canary model: gradually rolls out a new model to a % of live traffic
_canary_model: Model | None = None
_canary_traffic_pct: int = 0
_canary_config = mlflow_config.get("canary", {})
if _canary_config.get("enabled"):
    _canary_model = Model(
        "canary", _canary_config.get("model_uri", CANARY_MODEL_URI)
    )
    _canary_traffic_pct = _canary_config.get("traffic_percentage", DEFAULT_CANARY_TRAFFIC_PCT)
