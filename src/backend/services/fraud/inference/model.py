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


@dataclass
class ScoringResult:
    score: float
    production_score: float
    shadow_score: float | None = None
    canary_score: float | None = None
    scored_by: str = "production"


class ModelService:
    def __init__(
        self,
        model_name: str,
        model_uri: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_uri = model_uri
        self._loaded_model: LoadedModel | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded_model is not None:
            return
        self._loaded_model = model_loader.get_model(self._model_name)

    @property
    def is_model_loaded(self) -> bool:
        self._ensure_loaded()
        return self._loaded_model is not None and self._loaded_model.model is not None

    @property
    def feature_columns(self) -> list[str]:
        self._ensure_loaded()
        if self._loaded_model:
            return self._loaded_model.feature_columns
        return []

    def _features_to_dataframe(self, features: dict) -> pd.DataFrame:
        self._ensure_loaded()
        feature_columns = self.feature_columns
        row = {}
        for col in feature_columns:
            val = features.get(col)
            row[col] = float(val) if val is not None else np.nan
        df = pd.DataFrame([row], columns=feature_columns)
        return df

    def fetch_features(self, customer_id: str, email: str) -> dict:
        logger.info(
            "[MODEL] fetch_features() called for model=%s, customer_id=%s, email=%s",
            self._model_name,
            customer_id,
            email,
        )

        self._ensure_loaded()

        from services.fraud.features import get_features

        return get_features(customer_id=customer_id, email=email)

    def score(
        self,
        customer_id: str | None = None,
        email: str | None = None,
        request_features: dict | None = None,
    ) -> float:
        self._ensure_loaded()
        features: dict = {}
        if customer_id and email:
            features = self.fetch_features(customer_id, email)
        if request_features:
            features = {**features, **request_features}
        return self.predict(features)

    def predict(self, features: dict) -> float:
        logger.info("[MODEL] predict() called for model=%s", self._model_uri)

        self._ensure_loaded()
        if self._loaded_model and self._loaded_model.model is not None:
            try:
                df = self._features_to_dataframe(features)
                logger.debug(
                    "[MODEL] model=%s: features sent to model: %s",
                    self._model_uri,
                    features,
                )
                logger.debug(
                    "[MODEL] model=%s: model feature columns: %s",
                    self._model_uri,
                    self.feature_columns,
                )
                proba = self._loaded_model.model.predict_proba(df)[0]
                score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                logger.info(
                    "[MODEL] model=%s: SCORE=%.4f (using ML model)",
                    self._model_uri,
                    score,
                )
                return score
            except Exception as exc:
                logger.warning(
                    "[MODEL] model=%s predict failed: %s, falling back to random",
                    self._model_uri,
                    exc,
                )

        score = random.random()
        logger.warning(
            "[MODEL] model=%s not available, fallback RANDOM score=%.4f",
            self._model_uri,
            score,
        )
        return score


model_service = ModelService("production", "models:/fraud-detector@production")

_shadow_service: ModelService | None = None
_shadow_config = mlflow_config.get("shadow", {})
if _shadow_config.get("enabled"):
    _shadow_service = ModelService(
        "shadow", _shadow_config.get("model_uri", "models:/fraud-detector@shadow")
    )

_canary_service: ModelService | None = None
_canary_traffic_pct: int = 0
_canary_config = mlflow_config.get("canary", {})
if _canary_config.get("enabled"):
    _canary_service = ModelService(
        "canary", _canary_config.get("model_uri", "models:/fraud-detector@canary")
    )
    _canary_traffic_pct = _canary_config.get("traffic_percentage", 10)


def score_all(
    features: dict | None = None,
    customer_id: str | None = None,
    email: str | None = None,
    request_features: dict | None = None,
) -> ScoringResult:
    use_per_model = email is not None and customer_id is not None

    if use_per_model:
        production_score = model_service.score(customer_id, email, request_features)
    else:
        production_score = model_service.predict(features or {})

    shadow_score: float | None = None
    if _shadow_service is not None:
        try:
            if use_per_model:
                shadow_score = _shadow_service.score(customer_id, email, request_features)
            else:
                shadow_score = _shadow_service.predict(features or {})
            logger.debug("shadow score: %.4f", shadow_score)
        except Exception as exc:
            logger.warning("Shadow model scoring failed: %s", exc)

    canary_score: float | None = None
    scored_by = "production"
    if _canary_service is not None and random.randint(1, 100) <= _canary_traffic_pct:
        try:
            if use_per_model:
                canary_score = _canary_service.score(customer_id, email, request_features)
            else:
                canary_score = _canary_service.predict(features or {})
            scored_by = "canary"
            logger.debug("canary score: %.4f (used for decision)", canary_score)
        except Exception as exc:
            logger.warning("Canary model scoring failed, falling back to production: %s", exc)

    effective_score = canary_score if scored_by == "canary" else production_score

    return ScoringResult(
        score=effective_score,
        production_score=production_score,
        shadow_score=shadow_score,
        canary_score=canary_score,
        scored_by=scored_by,
    )
