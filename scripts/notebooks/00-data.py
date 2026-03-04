from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("data-summary", LOG_PATH / "data-summary.log")


def main() -> None:
    logger.info("Starting data summary analysis")

    customers = pd.read_csv(DATA_PATH / "customers.csv")
    addresses = pd.read_csv(DATA_PATH / "addresses.csv")
    charges = pd.read_csv(DATA_PATH / "charges.csv")
    checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
    payment_intents = pd.read_csv(DATA_PATH / "payment_intents.csv")
    stores = pd.read_csv(DATA_PATH / "stores.csv")

    logger.info("=== CUSTOMERS HEAD ===\n%s", customers.head())
    logger.info("=== ADDRESSES HEAD ===\n%s", addresses.head())
    logger.info("=== CHARGES HEAD ===\n%s", charges.head())
    logger.info("=== CHECKOUTS HEAD ===\n%s", checkouts.head())
    logger.info("=== PAYMENT_INTENTS HEAD ===\n%s", payment_intents.head())
    logger.info("=== STORES HEAD ===\n%s", stores.head())

    logger.info("=== SHAPES ===")
    logger.info("Customers: %s", customers.shape)
    logger.info("Addresses: %s", addresses.shape)
    logger.info("Charges: %s", charges.shape)
    logger.info("Checkouts: %s", checkouts.shape)
    logger.info("Payment Intents: %s", payment_intents.shape)
    logger.info("Stores: %s", stores.shape)

    logger.info("=== NULL COUNTS ===")
    logger.info("Customers:\n%s", customers.isnull().sum())
    logger.info("Addresses:\n%s", addresses.isnull().sum())
    logger.info("Charges:\n%s", charges.isnull().sum())
    logger.info("Checkouts:\n%s", checkouts.isnull().sum())
    logger.info("Payment Intents:\n%s", payment_intents.isnull().sum())
    logger.info("Stores:\n%s", stores.isnull().sum())

    logger.info("=== FISCAL CODE DUPLICATES ===")
    codice_fiscale_email = (
        customers.groupby("fiscal_code").agg({"id": "nunique"}).sort_values(by="id", ascending=False)
    )
    logger.info("\n%s", codice_fiscale_email.loc[codice_fiscale_email["id"] > 1])

    logger.info("Data summary complete")


if __name__ == "__main__":
    main()
