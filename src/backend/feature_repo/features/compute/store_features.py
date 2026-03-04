import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating store features...")

    # Define paths
    repo_root = Path(__file__).resolve().parents[5]
    checkouts_csv = repo_root / "data" / "01-clean" / "checkouts.csv"
    stores_csv = repo_root / "data" / "01-clean" / "stores.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"

    base_parquet = output_dir / "stores.parquet"
    stats_parquet = output_dir / "store_stats.parquet"

    if not checkouts_csv.exists() or not stores_csv.exists():
        logger.error("Source files not found.")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Base Stores Parquet
    logger.info("Reading base stores from %s", stores_csv)
    stores_df = pd.read_csv(stores_csv)

    # CRITICAL: Drop rows with missing entity key (store_id)
    n_missing = stores_df["store_id"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing store_id from base stores", n_missing)
        stores_df = stores_df.dropna(subset=["store_id"])

    # Add dummy 'created' timestamp for Feast
    logger.info("Adding dummy 'created' column for static store data...")
    stores_df["created"] = pd.Timestamp("2020-01-01", tz="UTC")

    # Force string types for Arrow compatibility
    string_cols = [
        "name",
        "partner_name",
        "store_name",
        "address",
        "zip",
        "state",
        "province",
        "area",
    ]
    for col in string_cols:
        if col in stores_df.columns:
            stores_df[col] = stores_df[col].astype(str).replace("nan", "")

    logger.info("Saving base stores to %s", base_parquet)
    # NO prefixes - use clean columns from source
    stores_df.to_parquet(base_parquet, index=False)

    # 2. Generate Store Stats (Expanding Windows)
    logger.info("Reading checkouts for store stats from %s", checkouts_csv)
    df = pd.read_csv(checkouts_csv)

    # Convert created to datetime-like objects and sort
    df["created"] = pd.to_datetime(df["created"])
    df = df.sort_values(by=["store_id", "created"]).reset_index(drop=True)

    # Success indicator
    df["is_success"] = (df["status"] == "complete").astype(int)

    # Subscription value (handle NaNs)
    df["subscription_value"] = pd.to_numeric(df["subscription_value"], errors="coerce").fillna(0.0)

    # Group by store_id and compute expanding features
    grouped = df.groupby("store_id")

    logger.info("Computing expanding statistics for stores...")
    # Calculate store_success_rate (expanding mean of success indicator)
    df["store_success_rate"] = (
        grouped["is_success"].expanding().mean().reset_index(level=0, drop=True)
    )

    # Calculate store_avg_value (expanding mean of subscription_value)
    df["store_avg_value"] = (
        grouped["subscription_value"].expanding().mean().reset_index(level=0, drop=True)
    )

    # CRITICAL: Drop rows with missing entity key (store_id)
    n_missing = df["store_id"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing store_id", n_missing)
        df = df.dropna(subset=["store_id"])

    # Select columns for Feast
    output_df = df[["store_id", "created", "store_success_rate", "store_avg_value"]]

    # Ensure timestamps are in UTC for Feast
    if output_df["created"].dt.tz is None:
        output_df["created"] = output_df["created"].dt.tz_localize("UTC")
    else:
        output_df["created"] = output_df["created"].dt.tz_convert("UTC")

    # Save to Parquet
    logger.info("Saving store stats to %s", stats_parquet)
    output_df.to_parquet(stats_parquet, index=False)

    logger.info("Store features generated successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
