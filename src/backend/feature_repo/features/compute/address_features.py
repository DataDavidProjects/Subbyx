import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating base address features...")

    # Define paths
    repo_root = Path("/Users/davidelupis/Desktop/Subbyx")
    address_csv = repo_root / "data" / "01-clean" / "addresses.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"
    output_parquet = output_dir / "addresses.parquet"

    if not address_csv.exists():
        logger.error("Source file not found: %s", address_csv)
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Reading %s", address_csv)
    df = pd.read_csv(address_csv)

    # CRITICAL: Drop rows with missing entity key (id)
    n_missing = df["id"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing id from base addresses", n_missing)
        df = df.dropna(subset=["id"])

    # Load customers to map address_id -> customer_id
    customers_csv = repo_root / "data" / "01-clean" / "customers.csv"
    logger.info("Reading %s to map addresses to customers...", customers_csv)
    customers_df = pd.read_csv(customers_csv)

    # We need to map address ID back to a customer.
    # A customer has residential and shipping addresses.
    # We'll create a mapping of address_id -> customer_id
    res_map = customers_df[["id", "residential_address_id"]].rename(
        columns={"id": "customer_id", "residential_address_id": "addr_id"}
    )
    ship_map = customers_df[["id", "shipping_address_id"]].rename(
        columns={"id": "customer_id", "shipping_address_id": "addr_id"}
    )
    addr_mapping = pd.concat([res_map, ship_map]).dropna().drop_duplicates(subset=["addr_id"])

    logger.info("Merging addresses with customer mapping...")
    df = df.merge(addr_mapping, left_on="id", right_on="addr_id", how="left")

    # CRITICAL: Drop addresses that don't belong to any customer (Feast entity check)
    n_unmapped = df["customer_id"].isnull().sum()
    if n_unmapped > 0:
        logger.warning("Dropping %d addresses not linked to any customer", n_unmapped)
        df = df.dropna(subset=["customer_id"])

    # Add dummy 'created' timestamp for Feast (static table)
    # Using a timestamp far in the past
    logger.info("Adding dummy 'created' column for static address data...")
    df["created"] = pd.Timestamp("2020-01-01", tz="UTC")

    # Force string types for Arrow compatibility
    string_cols = ["id", "locality", "city", "state", "country", "postal_code", "customer_id"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "")

    # Save to Parquet
    logger.info("Saving to %s", output_parquet)
    # NO prefixes - use clean columns from source
    df.to_parquet(output_parquet, index=False)

    logger.info("Base address features generated successfully: %d rows", len(df))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
