from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import yaml

_TRAINING_BASE = Path("/Users/davidelupis/Desktop/Subbyx/data/04-modeling")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SELECTED_FEATURES_YAML = (
    _PROJECT_ROOT / "src" / "backend" / "feature_repo" / "selected_features.yaml"
)


def _load_selected_features() -> list[str]:
    """Load the feature list written by feature_selection.py."""
    if _SELECTED_FEATURES_YAML.exists():
        with open(_SELECTED_FEATURES_YAML) as f:
            cfg = yaml.safe_load(f)
        feats = cfg.get("selected_features", [])
        if feats:
            return feats
    # Fallback: read directly from fraud_models.py if YAML not generated yet
    sys.path.insert(0, str(_PROJECT_ROOT / "src" / "backend" / "feature_repo"))
    from features.services.fraud_models import PRODUCTION_FEATURES  # noqa

    return PRODUCTION_FEATURES


# Shadow remains the identity-focused challenger (not changed by selection)
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "backend" / "feature_repo"))
from features.services.fraud_models import SHADOW_FEATURES  # noqa


def _latest_training_dir() -> Path:
    subdirs = sorted(
        [d for d in _TRAINING_BASE.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return subdirs[-1] if subdirs else _TRAINING_BASE


TRAINING_DIR = _latest_training_dir()

MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", "http://localhost:5002")
ARTIFACT_ROOT = "/Users/davidelupis/Desktop/Subbyx/data/mlflow/artifacts"
REGISTERED_MODEL_NAME = "fraud-detector"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": 10,
    "min_child_samples": 30,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 3,
    "verbose": -1,
}

# ---- Training profiles ----
# production: driven by selected_features.yaml (output of feature_selection.py)
# shadow: identity-focused challenger fixed feature set
PROFILES: dict[str, dict] = {
    "production": {
        "features": _load_selected_features(),
        "alias": "production",
        "feature_service": "fraud_model_production",
    },
    "shadow": {
        "features": SHADOW_FEATURES,
        "alias": "shadow",
        "feature_service": "fraud_model_shadow",
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="production",
        help="Training profile: selects features and MLflow alias (default: production)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    alias = profile["alias"]
    feature_service = profile["feature_service"]

    mlflow.set_tracking_uri(MLFLOW_SERVER)
    try:
        exp = mlflow.get_experiment_by_name("fraud-detector")
        if exp is None:
            mlflow.create_experiment("fraud-detector", artifact_location=ARTIFACT_ROOT)
    except Exception:
        mlflow.create_experiment("fraud-detector", artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment("fraud-detector")

    print(f"Profile: {args.profile} -> alias={alias}")

    print("Loading datasets...")
    train = pd.read_parquet(TRAINING_DIR / "train.parquet")
    val = pd.read_parquet(TRAINING_DIR / "validation.parquet")
    test = pd.read_parquet(TRAINING_DIR / "test.parquet")

    feature_columns = profile["features"]

    # Validate that all requested features exist in the data
    missing = set(feature_columns) - set(train.columns)
    if missing:
        raise ValueError(f"Features not found in training data: {missing}")

    print(f"Features ({len(feature_columns)}): {feature_columns}")

    X_train = train[feature_columns]
    y_train = train["label"]

    X_val = val[feature_columns]
    y_val = val["label"]

    X_test = test[feature_columns]
    y_test = test["label"]

    print(f"Train: {len(X_train)} ({y_train.mean() * 100:.1f}% fraud)")
    print(f"Val: {len(X_val)} ({y_val.mean() * 100:.1f}% fraud)")
    print(f"Test: {len(X_test)} ({y_test.mean() * 100:.1f}% fraud)")

    print("\nTraining Pipeline (Imputer + LightGBM with early stopping)...")
    from lightgbm import early_stopping, log_evaluation

    imputer = SimpleImputer(strategy="constant", fill_value=-1)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_columns)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=feature_columns)

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train_imp,
        y_train,
        eval_set=[(X_val_imp, y_val)],
        callbacks=[early_stopping(50, verbose=True), log_evaluation(50)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    # Wrap into Pipeline for MLflow serialization (imputer already fitted)
    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("classifier", model),
        ]
    )

    y_val_pred = pipeline.predict(X_val)
    y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    val_metrics = {
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
        "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
        "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
    }
    # ROC-AUC / AUC-PR are undefined when only one class is present
    if y_val.nunique() > 1:
        val_metrics["val_roc_auc"] = roc_auc_score(y_val, y_val_pred_proba)
        val_metrics["val_auc_pr"] = average_precision_score(y_val, y_val_pred_proba)
    else:
        print("\n  Warning: validation set has only one class — skipping ROC-AUC / AUC-PR")

    print("\nValidation Metrics:")
    print(f"  AUC-PR: {val_metrics.get('val_auc_pr', 'N/A')}")
    print(f"  ROC-AUC: {val_metrics.get('val_roc_auc', 'N/A')}")
    print(f"  F1: {val_metrics['val_f1']:.4f}")

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "auc_pr": average_precision_score(y_test, y_pred_proba),
    }

    print("\nTest Metrics:")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")

    input_schema = Schema([ColSpec("double", name=col) for col in feature_columns])
    output_schema = Schema([ColSpec("double", name="probability")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    with mlflow.start_run(run_name=f"{alias}-lightgbm-pipeline"):
        mlflow.log_param("profile", args.profile)
        mlflow.log_param("feature_columns", ",".join(feature_columns))
        mlflow.log_param("feature_service_name", feature_service)
        mlflow.log_params(LGBM_PARAMS)
        for name, value in {**val_metrics, **metrics}.items():
            if not (math.isnan(value) or math.isinf(value)):
                mlflow.log_metric(name, value)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=signature,
            input_example=X_train.iloc[:1].to_dict(orient="records")[0],
        )

    client = mlflow.MlflowClient()
    latest = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    if latest:
        version = latest[0].version
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)
        print(f"\nModel registered: {REGISTERED_MODEL_NAME} v{version} -> @{alias}")


if __name__ == "__main__":
    main()
