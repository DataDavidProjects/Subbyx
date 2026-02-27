from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH) as _f:
    _config = yaml.safe_load(_f)

_MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", "http://localhost:5002")


@dataclass
class LoadedModel:
    model: object
    feature_columns: list[str]
    model_uri: str
    feature_service_name: str | None = None


class ModelLoader:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Load all models (production, shadow, canary) at startup."""
        if self._loaded:
            return

        mlflow_config = _config.get("mlflow", {})
        shadow_config = _config.get("shadow", {})
        canary_config = _config.get("canary", {})

        tracking_uri = mlflow_config.get("tracking_uri", _MLFLOW_SERVER)

        production_uri = mlflow_config.get("model_uri")
        if production_uri:
            self._load_model("production", production_uri, tracking_uri)

        if shadow_config.get("enabled"):
            shadow_uri = shadow_config.get("model_uri")
            if shadow_uri:
                self._load_model("shadow", shadow_uri, tracking_uri)

        if canary_config.get("enabled"):
            canary_uri = canary_config.get("model_uri")
            if canary_uri:
                self._load_model("canary", canary_uri, tracking_uri)

        self._loaded = True

    def _load_model(self, name: str, model_uri: str, tracking_uri: str) -> None:
        """Load a single model and its feature columns."""
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)

            client = mlflow.MlflowClient()

            if "@" in model_uri:
                name_with_prefix, alias = model_uri.split("@", 1)
                model_name = name_with_prefix.replace("models:/", "")
                versions = client.get_model_version_by_alias(model_name, alias)
                model_version = versions.version
            else:
                model_name = model_uri.replace("models:/", "")
                versions = client.search_model_versions(f"name='{model_name}'", max_results=1)
                if not versions:
                    logger.warning("model=%s no versions found, skipping", model_uri)
                    return
                model_version = versions[0].version

            model_version_details = client.get_model_version(model_name, model_version)
            run_id = model_version_details.run_id

            run = client.get_run(run_id)
            params = run.data.params if run.data else {}
            feature_columns_param = params.get("feature_columns", "")
            feature_columns = feature_columns_param.split(",") if feature_columns_param else []
            feature_service_name = params.get("feature_service_name", None)

            logger.info("model=%s loading model from MLflow...", model_uri)
            try:
                model = mlflow.sklearn.load_model(model_uri)
            except Exception:
                model = mlflow.lightgbm.load_model(model_uri)

            self._models[name] = LoadedModel(
                model=model,
                feature_columns=feature_columns,
                model_uri=model_uri,
                feature_service_name=feature_service_name,
            )
            logger.info(
                "model=%s loaded successfully, feature_columns=%s",
                model_uri,
                feature_columns,
            )

        except Exception as exc:
            logger.warning("model=%s failed to load: %s", model_uri, exc)

    def get_model(self, name: str) -> LoadedModel | None:
        """Get a loaded model by name (production, shadow, canary)."""
        if not self._loaded:
            self.load_all()
        return self._models.get(name)

    def is_loaded(self) -> bool:
        return self._loaded


model_loader = ModelLoader()
