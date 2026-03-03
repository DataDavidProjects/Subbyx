from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import mlflow
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
        tracking_uri = mlflow_config.get("tracking_uri", _MLFLOW_SERVER)

        for name, model_uri in self._collect_model_targets():
            self._load_model(name, model_uri, tracking_uri)

        self._loaded = True

    def _collect_model_targets(self) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []

        production_uri = _config.get("mlflow", {}).get("model_uri")
        if production_uri:
            targets.append(("production", production_uri))

        for name in ("shadow", "canary"):
            cfg = _config.get(name, {})
            model_uri = cfg.get("model_uri")
            if cfg.get("enabled") and model_uri:
                targets.append((name, model_uri))

        return targets

    def _resolve_model_version(
        self,
        client: object,
        model_uri: str,
    ) -> tuple[str, str] | None:
        if "@" in model_uri:
            name_with_prefix, alias = model_uri.split("@", 1)
            model_name = name_with_prefix.replace("models:/", "")
            model_version = client.get_model_version_by_alias(model_name, alias).version
            return model_name, model_version

        model_name = model_uri.replace("models:/", "")
        versions = client.search_model_versions(f"name='{model_name}'", max_results=1)
        if not versions:
            logger.warning("model=%s no versions found, skipping", model_uri)
            return None
        return model_name, versions[0].version

    @staticmethod
    def _extract_run_metadata(
        client: object, model_name: str, model_version: str
    ) -> tuple[list[str], str | None]:
        model_version_details = client.get_model_version(model_name, model_version)
        run = client.get_run(model_version_details.run_id)
        params = run.data.params if run.data else {}

        feature_columns_param = params.get("feature_columns", "")
        feature_columns = feature_columns_param.split(",") if feature_columns_param else []
        feature_service_name = params.get("feature_service_name")
        return feature_columns, feature_service_name

    @staticmethod
    def _load_mlflow_model(mlflow: object, model_uri: str) -> object:
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception:
            return mlflow.lightgbm.load_model(model_uri)

    def _load_model(self, name: str, model_uri: str, tracking_uri: str) -> None:
        """Load a single model and its feature columns."""
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.MlflowClient()
            resolved = self._resolve_model_version(client, model_uri)
            if not resolved:
                return
            model_name, model_version = resolved
            feature_columns, feature_service_name = self._extract_run_metadata(
                client,
                model_name,
                model_version,
            )

            logger.info("model=%s loading model from MLflow...", model_uri)
            model = self._load_mlflow_model(mlflow, model_uri)

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
