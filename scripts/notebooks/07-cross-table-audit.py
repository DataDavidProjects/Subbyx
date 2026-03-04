from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("cross-table-audit", LOG_PATH / "cross-table-audit.log")


class CrossTableAnalyzer:
    def __init__(self, logger):
        self.logger = logger
        self.customers = pd.read_csv(DATA_PATH / "customers.csv")
        self.checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
        self.charges = pd.read_csv(DATA_PATH / "charges.csv")
        self.payment_intents = pd.read_csv(DATA_PATH / "payment_intents.csv")
        self.addresses = pd.read_csv(DATA_PATH / "addresses.csv")
        self.stores = pd.read_csv(DATA_PATH / "stores.csv")

    def run_full_analysis(self) -> None:
        self.logger.info("=== CROSS-TABLE ANALYSIS ===")
        self.validate_all_foreign_keys()
        self.analyze_customer_relationships()
        self.analyze_checkout_relationships()
        self.analyze_payment_relationships()
        self.summarize_data_quality()

    def validate_all_foreign_keys(self) -> None:
        self.logger.info("=== Foreign Key Validation ===")

        customer_ids = set(self.customers["id"])
        store_ids = set(self.stores["store_id"].dropna())
        address_ids = set(self.addresses["id"])
        payment_intent_ids = set(self.payment_intents["id"])

        self.validate_fk("checkouts", "customers", "customer", customer_ids)
        self.validate_fk("charges", "customers", "customer", customer_ids)
        self.validate_fk("payment_intents", "customers", "customer", customer_ids)
        self.validate_fk("checkouts", "stores", "store_id", store_ids)
        self.validate_fk("charges", "payment_intents", "payment_intent", payment_intent_ids)

        self.validate_address_fk("customers", "residential_address_id", address_ids)
        self.validate_address_fk("customers", "shipping_address_id", address_ids)

    def validate_fk(self, from_table: str, to_table: str, fk_column: str, valid_ids: set) -> None:
        df = getattr(self, from_table)
        total = len(df)
        null_count = df[fk_column].isna().sum()
        non_null = df[df[fk_column].notna()]
        invalid_count = (~non_null[fk_column].isin(valid_ids)).sum()

        self.logger.info(
            "%s.%s -> %s.id: null=%d, invalid=%d, valid_rate=%.2f%%",
            from_table,
            fk_column,
            to_table,
            null_count,
            invalid_count,
            ((total - null_count - invalid_count) / (total - null_count) * 100)
            if (total - null_count) > 0
            else 0,
        )

    def validate_address_fk(self, table: str, fk_column: str, valid_ids: set) -> None:
        df = getattr(self, table)
        total = len(df)
        null_count = df[fk_column].isna().sum()
        non_null = df[df[fk_column].notna()]
        invalid_count = (~non_null[fk_column].isin(valid_ids)).sum()

        self.logger.info(
            "%s.%s -> addresses.id: null=%d, invalid=%d, valid_rate=%.2f%%",
            table,
            fk_column,
            null_count,
            invalid_count,
            ((total - null_count - invalid_count) / (total - null_count) * 100)
            if (total - null_count) > 0
            else 0,
        )

    def analyze_customer_relationships(self) -> None:
        self.logger.info("=== Customer Relationship Analysis ===")

        customer_ids = set(self.customers["id"])

        checkouts_with_customer = self.checkouts[self.checkouts["customer"].isin(customer_ids)]
        charges_with_customer = self.charges[self.charges["customer"].isin(customer_ids)]
        pi_with_customer = self.payment_intents[self.payment_intents["customer"].isin(customer_ids)]

        self.logger.info(
            "Checkouts with valid customer: %d/%d (%.2f%%)",
            len(checkouts_with_customer),
            len(self.checkouts),
            len(checkouts_with_customer) / len(self.checkouts) * 100,
        )
        self.logger.info(
            "Charges with valid customer: %d/%d (%.2f%%)",
            len(charges_with_customer),
            len(self.charges),
            len(charges_with_customer) / len(self.charges) * 100,
        )
        self.logger.info(
            "Payment Intents with valid customer: %d/%d (%.2f%%)",
            len(pi_with_customer),
            len(self.payment_intents),
            len(pi_with_customer) / len(self.payment_intents) * 100,
        )

        customers_with_checkouts = self.checkouts[self.checkouts["customer"].isin(customer_ids)][
            "customer"
        ].nunique()
        customers_with_charges = self.charges[self.charges["customer"].isin(customer_ids)][
            "customer"
        ].nunique()
        customers_with_pi = self.payment_intents[
            self.payment_intents["customer"].isin(customer_ids)
        ]["customer"].nunique()

        self.logger.info(
            "Customers with checkouts: %d/%d (%.2f%%)",
            customers_with_checkouts,
            len(self.customers),
            customers_with_checkouts / len(self.customers) * 100,
        )
        self.logger.info(
            "Customers with charges: %d/%d (%.2f%%)",
            customers_with_charges,
            len(self.customers),
            customers_with_charges / len(self.customers) * 100,
        )
        self.logger.info(
            "Customers with payment intents: %d/%d (%.2f%%)",
            customers_with_pi,
            len(self.customers),
            customers_with_pi / len(self.customers) * 100,
        )

    def analyze_checkout_relationships(self) -> None:
        self.logger.info("=== Checkout Relationship Analysis ===")

        checkout_modes = self.checkouts["mode"].value_counts()
        self.logger.info("Checkout modes:\n%s", checkout_modes)

        payment_checkouts = self.checkouts[self.checkouts["mode"].isin(["payment", "subscription"])]
        self.logger.info("Payment/Subscription checkouts: %d", len(payment_checkouts))

        store_ids = set(self.stores["store_id"].dropna())
        payment_with_store = payment_checkouts[payment_checkouts["store_id"].isin(store_ids)]
        self.logger.info(
            "Payment checkouts with valid store: %d/%d (%.2f%%)",
            len(payment_with_store),
            len(payment_checkouts),
            len(payment_with_store) / len(payment_checkouts) * 100
            if len(payment_checkouts) > 0
            else 0,
        )

    def analyze_payment_relationships(self) -> None:
        self.logger.info("=== Payment Relationship Analysis ===")

        payment_intent_ids = set(self.payment_intents["id"])

        charges_with_pi = self.charges[self.charges["payment_intent"].isin(payment_intent_ids)]
        self.logger.info(
            "Charges with valid payment_intent: %d/%d (%.2f%%)",
            len(charges_with_pi),
            len(self.charges),
            len(charges_with_pi) / len(self.charges) * 100,
        )

        pi_with_charges = self.payment_intents[
            self.payment_intents["id"].isin(self.charges["payment_intent"])
        ]
        self.logger.info(
            "Payment intents with at least one charge: %d/%d (%.2f%%)",
            len(pi_with_charges),
            len(self.payment_intents),
            len(pi_with_charges) / len(self.payment_intents) * 100,
        )

    def summarize_data_quality(self) -> None:
        self.logger.info("=== Data Quality Summary ===")

        issues = []

        if len(self.addresses[self.addresses.duplicated(subset=["id"], keep=False)]) > 0:
            issues.append("addresses: duplicate IDs found")

        invalid_store_checkouts = self.checkouts[
            ~self.checkouts["store_id"].isin(set(self.stores["store_id"].dropna()))
            & self.checkouts["store_id"].notna()
        ]
        if len(invalid_store_checkouts) > 0:
            issues.append(f"checkouts: {len(invalid_store_checkouts)} invalid store_ids")

        invalid_latest_charge = self.payment_intents[
            ~self.payment_intents["latest_charge"].isin(set(self.charges["id"]))
            & self.payment_intents["latest_charge"].notna()
        ]
        if len(invalid_latest_charge) > 0:
            issues.append(
                f"payment_intents: {len(invalid_latest_charge)} invalid latest_charge references"
            )

        if issues:
            self.logger.warning("ISSUES FOUND:")
            for issue in issues:
                self.logger.warning("  - %s", issue)
        else:
            self.logger.info("No cross-table data quality issues found")


def main():
    logger.info("Starting cross-table audit analysis")
    analyzer = CrossTableAnalyzer(logger)
    analyzer.run_full_analysis()
    logger.info("Cross-table audit complete")


if __name__ == "__main__":
    main()
