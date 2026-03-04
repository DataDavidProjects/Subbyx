import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating charge features...")

    # Define paths
    repo_root = Path(__file__).resolve().parents[5]
    charges_csv = repo_root / "data" / "01-clean" / "charges.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"

    base_parquet = output_dir / "charges.parquet"
    stats_parquet = output_dir / "charge_stats.parquet"

    if not charges_csv.exists():
        logger.error("Source file not found: %s", charges_csv)
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Reading %s", charges_csv)
    df = pd.read_csv(charges_csv)

    # CRITICAL: Convert 'created' to datetime and ensure UTC
    logger.info("Converting timestamps...")
    df["created"] = pd.to_datetime(df["created"])
    if df["created"].dt.tz is None:
        df["created"] = df["created"].dt.tz_localize("UTC")
    else:
        df["created"] = df["created"].dt.tz_convert("UTC")

    # CRITICAL: Drop rows with missing entity key (email)
    n_missing = df["email"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing email", n_missing)
        df = df.dropna(subset=["email"])

    # Force string types for Arrow compatibility
    string_cols = ["email", "status", "payment_intent", "card_brand", "card_issuer"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "")

    # 1. Save Base Charges Parquet
    logger.info("Saving base charges to %s", base_parquet)
    # NO prefixes - use clean columns from source
    df.to_parquet(base_parquet, index=False)

    # 2. Compute Derived Stats Features
    logger.info("Computing derived charge stats...")
    df = df.sort_values(by=["email", "created"]).reset_index(drop=True)

    df["is_failure"] = (df["status"] == "failed").astype(int)

    grouped = df.groupby("email")

    # n_charges (cumulative count)
    df["n_charges"] = grouped.cumcount() + 1
    # n_failures (cumulative sum)
    df["n_failures"] = grouped["is_failure"].cumsum()
    # failure_rate (PIT correct)
    df["failure_rate"] = df["n_failures"] / df["n_charges"]

    # Select columns for Stats Parquet
    stats_df = df[["email", "created", "n_charges", "n_failures", "failure_rate"]]

    # Save Stats Parquet
    logger.info("Saving charge stats to %s", stats_parquet)
    stats_df.to_parquet(stats_parquet, index=False)

    logger.info("Charge features generated successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
