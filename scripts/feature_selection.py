"""
Feature selection pipeline.

Reads the latest training data, runs a multi-stage selection pipeline
(variance → mutual information → correlation → VIF), then writes the result to
`selected_features.yaml` — the single source of truth consumed by
`train_model.py` and `fraud_models.py`.

Pipeline flow:
  make features → make feature-select → make train → make threshold-optimize
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn import set_config
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

# Ensure scikit-learn output is always a pandas DataFrame
set_config(transform_output="pandas")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "backend"))

from services.fraud.features.selection import (
    CorrelationGroupPruner,
    RemoveHighVIFFeatures,
    SelectKBestMutualInfo,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_TRAINING_BASE = PROJECT_ROOT / "data" / "04-modeling"
_SELECTED_FEATURES_PATH = (
    PROJECT_ROOT / "src" / "backend" / "feature_repo" / "selected_features.yaml"
)

# Constants
VIF_THRESHOLD = 10.0
MI_MIN_SCORE = 0.02  # keep features with any predictive signal
CORR_THRESHOLD = 0.5


def _latest_training_dir() -> Path:
    subdirs = sorted(
        [d for d in _TRAINING_BASE.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return subdirs[-1] if subdirs else _TRAINING_BASE


def main() -> None:
    training_dir = _latest_training_dir()
    logger.info("Loading training data from: %s", training_dir)

    train = pd.read_parquet(training_dir / "train.parquet")

    exclude = {
        "checkout_id",
        "created",
        "email",
        "customer_id",
        "store_id",
        "event_timestamp",
        "label",
    }

    # Only numeric columns
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude]
    X = train[feature_cols].fillna(0)
    y = train["label"]

    logger.info("=" * 60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)
    logger.info("Input: %d features across %d training samples", len(feature_cols), len(X))

    # -------------------------------------------------------------------
    # Define and Run Pipeline
    # -------------------------------------------------------------------
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=0)),
            ("mi", SelectKBestMutualInfo(min_score=MI_MIN_SCORE)),
            ("correlation", CorrelationGroupPruner(threshold=CORR_THRESHOLD)),
            ("vif", RemoveHighVIFFeatures(threshold=VIF_THRESHOLD)),
        ]
    )

    logger.info("\nExecuting selection stages...")
    X_selected = pipeline.fit_transform(X, y)

    selected_features = list(X_selected.columns)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("RESULT: %d features selected", len(selected_features))
    for f in selected_features:
        logger.info("  • %s", f)
    logger.info("=" * 60)

    # -------------------------------------------------------------------
    # Write YAML — consumed by train_model.py and fraud_models.py
    # -------------------------------------------------------------------
    output = {
        "selected_features": selected_features,
        "training_dir": str(training_dir),
        "vif_threshold": VIF_THRESHOLD,
        "mi_min_score": MI_MIN_SCORE,
        "corr_threshold": CORR_THRESHOLD,
        "n_input_features": len(feature_cols),
        "n_selected": len(selected_features),
    }
    _SELECTED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SELECTED_FEATURES_PATH, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    logger.info("\nSaved selected features to: %s", _SELECTED_FEATURES_PATH)


if __name__ == "__main__":
    main()
