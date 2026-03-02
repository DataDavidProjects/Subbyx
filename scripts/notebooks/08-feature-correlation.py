from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path("/Users/davidelupis/Desktop/Subbyx/data/01-clean/")
MODEL_PATH = Path("/Users/davidelupis/Desktop/Subbyx/data/04-modeling/")
LOG_PATH = Path("/Users/davidelupis/Desktop/Subbyx/scripts/notebooks/logs/")

logger = setup_logger("feature-correlation", LOG_PATH / "feature-correlation.log")


def load_training_data() -> pd.DataFrame:
    """Load the latest training data."""
    subdirs = sorted([d for d in MODEL_PATH.iterdir() if d.is_dir()], key=lambda p: p.name)
    latest = subdirs[-1]
    logger.info("Loading from: %s", latest)
    return pd.read_parquet(latest / "train.parquet")


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation of each feature with the fraud label."""
    exclude = {
        "checkout_id",
        "created",
        "email",
        "customer_id",
        "store_id",
        "event_timestamp",
        "label",
    }

    results = []

    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype not in [np.float64, np.int64, np.float32, np.int32]:
            continue

        # Calculate correlation
        corr = df[col].corr(df["label"])

        # Calculate AUC-ROC (more robust for binary classification)
        try:
            valid = df[[col, "label"]].dropna()
            if len(valid) > 10 and valid["label"].nunique() > 1:
                auc = roc_auc_score(valid["label"], valid[col])
                # Convert AUC to signed value (AUC < 0.5 means negative correlation)
                if auc < 0.5:
                    auc = -(0.5 - auc)
            else:
                auc = np.nan
        except:
            auc = np.nan

        # Calculate means for fraud vs clean
        fraud_mean = df[df["label"] == 1][col].mean() if len(df[df["label"] == 1]) > 0 else np.nan
        clean_mean = df[df["label"] == 0][col].mean() if len(df[df["label"] == 0]) > 0 else np.nan

        # Count nulls
        null_pct = df[col].isna().mean() * 100

        results.append(
            {
                "feature": col,
                "correlation": corr,
                "auc_roc": auc,
                "fraud_mean": fraud_mean,
                "clean_mean": clean_mean,
                "diff": fraud_mean - clean_mean,
                "null_pct": null_pct,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("auc_roc", ascending=False, key=abs)

    return results_df


def print_top_features(results: pd.DataFrame, n: int = 20) -> None:
    """Print top features by absolute correlation."""
    logger.info("=" * 80)
    logger.info("TOP %d FEATURES BY CORRELATION WITH FRAUD LABEL", n)
    logger.info("=" * 80)
    logger.info("%-60s %8s %8s %10s %10s %10s", "Feature", "Corr", "AUC", "Fraud", "Clean", "Diff")
    logger.info("-" * 80)

    for _, row in results.head(n).iterrows():
        logger.info(
            "%-60s %8.3f %8.3f %10.3f %10.3f %10.3f",
            row["feature"],
            row["correlation"],
            row["auc_roc"],
            row["fraud_mean"],
            row["clean_mean"],
            row["diff"],
        )


def analyze_feature_patterns(df: pd.DataFrame) -> None:
    """Analyze specific feature patterns."""
    logger.info("=" * 80)
    logger.info("FEATURE PATTERNS ANALYSIS")
    logger.info("=" * 80)

    identity_features = [
        c for c in df.columns if "email" in c.lower() or "name" in c.lower() or "match" in c.lower()
    ]
    logger.info("--- Identity Features ---")
    for f in identity_features:
        if f in df.columns and df[f].dtype in [np.float64, np.int64, np.float32, np.int32]:
            fraud = df[df["label"] == 1][f].mean()
            clean = df[df["label"] == 0][f].mean()
            logger.info("%s: fraud=%.3f, clean=%.3f, diff=%.3f", f, fraud, clean, fraud - clean)

    payment_features = [
        c
        for c in df.columns
        if "charge" in c.lower() or "payment" in c.lower() or "success" in c.lower()
    ]
    logger.info("--- Payment/Charge Features ---")
    for f in payment_features:
        if f in df.columns and df[f].dtype in [np.float64, np.int64, np.float32, np.int32]:
            fraud = df[df["label"] == 1][f].mean()
            clean = df[df["label"] == 0][f].mean()
            logger.info("%s: fraud=%.3f, clean=%.3f, diff=%.3f", f, fraud, clean, fraud - clean)

    geo_features = [
        c
        for c in df.columns
        if "province" in c.lower() or "postal" in c.lower() or "time" in c.lower()
    ]
    logger.info("--- Geo/Time Features ---")
    for f in geo_features[:10]:
        if f in df.columns and df[f].dtype in [np.float64, np.int64, np.float32, np.int32]:
            fraud = df[df["label"] == 1][f].mean()
            clean = df[df["label"] == 0][f].mean()
            logger.info("%s: fraud=%.3f, clean=%.3f, diff=%.3f", f, fraud, clean, fraud - clean)


def check_label_definition(df: pd.DataFrame) -> None:
    """Check the fraud label definition."""
    logger.info("=" * 80)
    logger.info("FRAUD LABEL ANALYSIS")
    logger.info("=" * 80)

    customers = pd.read_csv(DATA_PATH / "customers.csv")
    checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
    checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]

    merged = checkouts.merge(
        customers[["id", "email", "dunning_days"]], left_on="customer", right_on="id", how="left"
    )
    merged = merged.dropna(subset=["email", "dunning_days"])

    logger.info("Total checkouts: %d", len(merged))
    logger.info(
        "Fraud (dunning > 15): %d (%.1f%%)",
        (merged["dunning_days"] > 15).sum(),
        100 * (merged["dunning_days"] > 15).mean(),
    )
    logger.info(
        "Clean (dunning <= 15): %d (%.1f%%)",
        (merged["dunning_days"] <= 15).sum(),
        100 * (merged["dunning_days"] <= 15).mean(),
    )

    charges = pd.read_csv(DATA_PATH / "charges.csv")
    returning_emails = set(charges["email"].unique())
    merged["is_returning"] = merged["email"].isin(returning_emails)

    logger.info("--- By Segment ---")
    for segment, group in merged.groupby("is_returning"):
        seg_name = "Returning" if segment else "New"
        fraud_pct = 100 * (group["dunning_days"] > 15).mean()
        logger.info("%s: %d checkouts, %.1f%% fraud rate", seg_name, len(group), fraud_pct)


def main() -> None:
    logger.info("Starting feature correlation analysis")

    df = load_training_data()
    logger.info("Loaded %d samples with %d features", len(df), len(df.columns))
    logger.info(
        "Dataset: %d samples, Fraud rate: %.1f%%, Features: %d",
        len(df),
        df["label"].mean() * 100,
        len(df.columns),
    )

    results = analyze_correlations(df)
    print_top_features(results, n=30)
    analyze_feature_patterns(df)
    check_label_definition(df)

    # Save results
    output_path = LOG_PATH / "feature_correlations.csv"
    results.to_csv(output_path, index=False)
    logger.info("Saved correlations to: %s", output_path)


if __name__ == "__main__":
    main()
