import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating customer features...")

    # Define paths
    repo_root = Path("/Users/davidelupis/Desktop/Subbyx")
    customers_csv = repo_root / "data" / "01-clean" / "customers.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"

    base_parquet = output_dir / "customers.parquet"
    profile_parquet = output_dir / "customer_profile.parquet"

    if not customers_csv.exists():
        logger.error("Source file not found: %s", customers_csv)
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Reading %s", customers_csv)
    df = pd.read_csv(customers_csv)

    # CRITICAL: Convert 'created' to datetime and ensure UTC
    logger.info("Converting timestamps...")
    df["created"] = pd.to_datetime(df["created"])
    if df["created"].dt.tz is None:
        df["created"] = df["created"].dt.tz_localize("UTC")
    else:
        df["created"] = df["created"].dt.tz_convert("UTC")

    # 1. Save Base Customers Parquet
    logger.info("Renaming 'id' to 'customer_id' for Feast entity...")
    df = df.rename(columns={"id": "customer_id"})

    # CRITICAL: Drop rows with missing entity key (customer_id)
    n_missing = df["customer_id"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing customer_id", n_missing)
        df = df.dropna(subset=["customer_id"])

    logger.info("Saving base customers to %s", base_parquet)
    df.to_parquet(base_parquet, index=False)

    # 2. Compute Derived Profile Features
    logger.info("Computing derived profile features...")
    df = df.sort_values(by=["fiscal_code", "created"]).reset_index(drop=True)

    # n_emails_per_fiscal_code (PIT correct)
    df["is_unique_email"] = ~df.duplicated(subset=["fiscal_code", "email"])
    df["n_emails_per_fiscal_code"] = df.groupby("fiscal_code")["is_unique_email"].cumsum()

    # is_address_mismatch
    res = df["residential_address_id"].fillna("MISSING_RES")
    ship = df["shipping_address_id"].fillna("MISSING_SHIP")
    df["is_address_mismatch"] = (res != ship).astype(int)

    # Select columns for Profile Parquet
    profile_df = df[["customer_id", "created", "n_emails_per_fiscal_code", "is_address_mismatch"]]

    # Save Profile Parquet
    logger.info("Saving customer profile to %s", profile_parquet)
    profile_df.to_parquet(profile_parquet, index=False)

    logger.info("Customer features generated successfully.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
