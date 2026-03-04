from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Constants
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_BASE = _PROJECT_ROOT / "data" / "04-modeling"
MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", "http://localhost:5002")
MODEL_NAME = "fraud-detector"
BETA = 0.5  # Weight precision 2x more than recall


def _latest_training_dir() -> Path:
    """Resolve the latest timestamped subdirectory under the training base."""
    subdirs = sorted(
        [d for d in _TRAINING_BASE.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    if subdirs:
        return subdirs[-1]
    return _TRAINING_BASE


def calculate_segment_metrics(y_true, y_prob, threshold, beta=0.5):
    """Calculate metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f_beta": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "tp": np.sum((y_pred == 1) & (y_true == 1)),
        "fp": np.sum((y_pred == 1) & (y_true == 0)),
        "fn": np.sum((y_pred == 0) & (y_true == 1)),
        "tn": np.sum((y_pred == 0) & (y_true == 0)),
    }


def find_optimal_threshold(y_true, y_prob, beta=0.5):
    """Search for threshold that maximizes F-beta score."""
    thresholds = np.linspace(0.01, 0.95, 200)
    best_score = -1
    best_metrics = None

    for t in thresholds:
        m = calculate_segment_metrics(y_true, y_prob, t, beta)
        if m["f_beta"] > best_score:
            best_score = m["f_beta"]
            best_metrics = m

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Optimize fraud detection thresholds")
    parser.add_argument(
        "--alias",
        default="production",
        help="Model alias to optimize (default: production)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_SERVER)

    # 1. Load model
    model_uri = f"models:/{MODEL_NAME}@{args.alias}"
    logger.info("Loading model from %s...", model_uri)
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.info(
            "Hint: Make sure MLflow is running and the model is registered with alias @%s",
            args.alias,
        )
        sys.exit(1)

    # 2. Load latest test/val data
    data_dir = _latest_training_dir()
    logger.info("Loading data from %s...", data_dir)
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # 3. Extract feature list from the fitted pipeline (guaranteed match with training)
    feature_cols = list(model.named_steps["imputer"].feature_names_in_)
    logger.info("Using features: %s", feature_cols)

    X = test_df[feature_cols]
    y = test_df["label"]

    # 3. Generate probabilities
    logger.info("Generating predictions...")
    y_prob = model.predict_proba(X)[:, 1]

    # 4. Segment data (mimicking determined logic in backend)
    # NEW_CUSTOMER if n_charges is 0 or null
    is_new = (test_df["charge_stats_features__n_charges"] == 0) | (
        test_df["charge_stats_features__n_charges"].isna()
    )

    segments = {
        "NEW_CUSTOMER": {"y_true": y[is_new], "y_prob": y_prob[is_new]},
        "RETURNING": {"y_true": y[~is_new], "y_prob": y_prob[~is_new]},
    }

    logger.info("\n" + "=" * 60)
    logger.info(f"THRESHOLD OPTIMIZATION (BETA={BETA})")
    logger.info("=" * 60)
    logger.info(
        f"{'SEGMENT':<15} | {'COUNT':<6} | {'FRAUD%':<7} | {'BEST_T':<6} | {'F0.5':<6} | {'PREC':<6} | {'RECALL':<6}"
    )
    logger.info("-" * 60)

    recommended_config = {}

    for name, data in segments.items():
        count = len(data["y_true"])
        if count == 0:
            logger.warning(f"No data for segment {name}")
            continue

        fraud_pct = data["y_true"].mean() * 100
        best = find_optimal_threshold(data["y_true"], data["y_prob"], beta=BETA)

        if best:
            logger.info(
                f"{name:<15} | {count:<6} | {fraud_pct:>6.1f}% | {best['threshold']:>6.3f} | {best['f_beta']:>6.3f} | {best['precision']:>6.3f} | {best['recall']:>6.3f}"
            )
            recommended_config[name] = best["threshold"]
        else:
            logger.info(
                f"{name:<15} | {count:<6} | {fraud_pct:>6.1f}% | {'N/A':>6} | {'N/A':>6} | {'N/A':>6} | {'N/A':>6}"
            )

    logger.info("=" * 60)
    logger.info("\nRecommended YAML configuration for src/backend/routes/fraud/config.yaml:")
    logger.info("segments:")
    for name, t in recommended_config.items():
        logger.info(f"  {name}:")
        logger.info(f"    threshold: {round(float(t), 2)}")


if __name__ == "__main__":
    main()
