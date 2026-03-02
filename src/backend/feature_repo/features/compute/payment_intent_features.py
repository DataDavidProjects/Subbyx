import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating base payment intent features...")

    # Define paths
    repo_root = Path("/Users/davidelupis/Desktop/Subbyx")
    pi_csv = repo_root / "data" / "01-clean" / "payment_intents.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"
    output_parquet = output_dir / "payment_intents.parquet"

    if not pi_csv.exists():
        logger.error("Source file not found: %s", pi_csv)
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Reading %s", pi_csv)
    df = pd.read_csv(pi_csv)

    # Load customers to get email (required entity)
    customers_csv = repo_root / "data" / "01-clean" / "customers.csv"
    logger.info("Reading %s for join...", customers_csv)
    customers_df = pd.read_csv(customers_csv)

    # Merge to get email
    # Payment Intents 'customer' maps to Customer 'id'
    logger.info("Merging payment intents with customers to resolve emails...")
    df = df.merge(
        customers_df[["id", "email"]],
        left_on="customer",
        right_on="id",
        how="left",
        suffixes=("", "_cust"),
    )

    # CRITICAL: Clean up missing emails
    n_missing = df["email"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing email", n_missing)
        df = df.dropna(subset=["email"])

    # CRITICAL: Convert 'created' to datetime and ensure UTC
    logger.info("Converting timestamps...")
    df["created"] = pd.to_datetime(df["created"])
    if df["created"].dt.tz is None:
        df["created"] = df["created"].dt.tz_localize("UTC")
    else:
        df["created"] = df["created"].dt.tz_convert("UTC")

    # Force string types for Arrow compatibility
    string_cols = ["id", "status", "payment_intent", "customer", "email", "latest_charge"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "")

    # Save to Parquet
    logger.info("Saving to %s", output_parquet)
    # NO prefixes - use clean columns from source
    df.to_parquet(output_parquet, index=False)

    logger.info("Base payment intent features generated successfully: %d rows", len(df))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
