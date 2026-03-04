from __future__ import annotations

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("stores-audit", LOG_PATH / "stores-audit.log")


class PlotSaver:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.plots_dir = LOG_PATH / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def save(self, column: str) -> None:
        filename = f"{self.prefix}_{column}.png"
        filepath = self.plots_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close()
        logger.info("Saved plot: %s", filepath)


class DataLoader:
    @staticmethod
    def load_stores() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "stores.csv")


class DataQualityChecker:
    def __init__(self, df: pd.DataFrame, table_name: str, logger):
        self.df = df
        self.table_name = table_name
        self.logger = logger

    def get_shape(self) -> tuple:
        return self.df.shape

    def get_columns(self) -> list:
        return self.df.columns.tolist()

    def get_null_counts(self) -> pd.Series:
        return self.df.isnull().sum()

    def get_duplicates(self) -> int:
        return self.df.duplicated().sum()

    def generate_report(self) -> None:
        self.logger.info("=== %s Data Quality Report ===", self.table_name)
        self.logger.info("Shape: %s", self.get_shape())
        self.logger.info("Columns: %s", self.get_columns())
        self.logger.info("Duplicates: %s", self.get_duplicates())
        self.logger.info("Null counts:\n%s", self.get_null_counts())


class CategoricalAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, plot_saver: PlotSaver):
        self.df = df
        self.logger = logger
        self.plot_saver = plot_saver

    def get_value_counts(self, column: str) -> pd.Series:
        return self.df[column].value_counts()

    def plot_value_counts(self, column: str, top_n: int = 20) -> None:
        plt.figure(figsize=(12, 6))
        self.get_value_counts(column).head(top_n).plot(kind="bar")
        plt.title(f"Value counts for {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        self.plot_saver.save(column)


class StoresAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, customers_ref=None, checkouts_ref=None):
        self.df = df
        self.logger = logger
        self.customers_ref = customers_ref
        self.checkouts_ref = checkouts_ref
        self.plot_saver = PlotSaver("stores")
        self.quality = DataQualityChecker(df, "stores", logger)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()
        self.logger.info(
            "=== Partner Distribution ===\n%s", self.categorical.get_value_counts("partner_name")
        )
        self.logger.info("=== Area Distribution ===\n%s", self.categorical.get_value_counts("area"))
        self.analyze_store_id_uniqueness()
        self.analyze_required_fields()
        self.analyze_address_completeness()

        # Deep stores analysis
        self.analyze_partner_coverage()
        self.analyze_geographic_coverage()
        self.analyze_store_types()
        self.analyze_duplicates()
        self.analyze_fraud_rate_by_store()

    def analyze_partners(self) -> None:
        self.categorical.plot_value_counts("partner_name")

    def analyze_stores(self) -> None:
        self.logger.info(
            "Store Names (top 20):\n%s", self.categorical.get_value_counts("store_name").head(20)
        )

    def analyze_geography(self) -> None:
        self.categorical.plot_value_counts("area")
        self.logger.info(
            "Province Distribution:\n%s", self.categorical.get_value_counts("province").head(20)
        )

    def analyze_store_id_uniqueness(self) -> None:
        self.logger.info("=== Store ID Uniqueness ===")
        total = len(self.df)
        unique_store_ids = self.df["store_id"].nunique()
        self.logger.info("Total stores: %d", total)
        self.logger.info("Unique store_ids: %d", unique_store_ids)

        duplicate_store_ids = self.df[self.df.duplicated(subset=["store_id"], keep=False)]
        self.logger.info("Duplicate store_ids: %d", len(duplicate_store_ids))

    def analyze_required_fields(self) -> None:
        self.logger.info("=== Required Fields Completeness ===")
        total = len(self.df)

        required_fields = ["name", "store_id", "partner_name", "store_name"]
        for field in required_fields:
            null_count = self.df[field].isna().sum()
            self.logger.info("Null %s: %d (%.2f%%)", field, null_count, (null_count / total) * 100)

    def analyze_address_completeness(self) -> None:
        self.logger.info("=== Address Completeness ===")
        total = len(self.df)

        null_address = self.df["address"].isna().sum()
        null_zip = self.df["zip"].isna().sum()
        null_state = self.df["state"].isna().sum()
        null_province = self.df["province"].isna().sum()

        self.logger.info("Null address: %d (%.2f%%)", null_address, (null_address / total) * 100)
        self.logger.info("Null zip: %d (%.2f%%)", null_zip, (null_zip / total) * 100)
        self.logger.info("Null state: %d (%.2f%%)", null_state, (null_state / total) * 100)
        self.logger.info("Null province: %d (%.2f%%)", null_province, (null_province / total) * 100)

    def analyze_partner_coverage(self) -> None:
        self.logger.info("=== PARTNER COVERAGE ANALYSIS ===")

        partner_dist = self.df["partner_name"].value_counts()

        self.logger.info("Partner store counts:")
        for partner, count in partner_dist.items():
            self.logger.info("  %s: %d stores", partner, count)

        # Major partners
        major_partners = partner_dist[partner_dist >= 5]
        self.logger.info("Major partners (5+ stores): %d", len(major_partners))

        # Single store partners
        single = partner_dist[partner_dist == 1]
        self.logger.info("Single-store partners: %d", len(single))

    def analyze_geographic_coverage(self) -> None:
        self.logger.info("=== GEOGRAPHIC COVERAGE ANALYSIS ===")

        # Area distribution
        area_dist = self.df["area"].value_counts()
        self.logger.info("Stores by area:")
        for area, count in area_dist.items():
            self.logger.info("  %s: %d stores", area, count)

        # State coverage
        self.logger.info("State coverage:")
        state_dist = self.df["state"].value_counts()
        for state, count in state_dist.items():
            if pd.notna(state):
                self.logger.info("  %s: %d stores", state, count)

        # Coverage gap analysis
        italian_regions = [
            "Abruzzo",
            "Basilicata",
            "Calabria",
            "Campania",
            "Emilia-Romagna",
            "Friuli-Venezia Giulia",
            "Lazio",
            "Liguria",
            "Lombardia",
            "Marche",
            "Molise",
            "Piemonte",
            "Puglia",
            "Sardegna",
            "Sicilia",
            "Toscana",
            "Trentino-Alto Adige",
            "Umbria",
            "Valle d'Aosta",
            "Veneto",
        ]
        covered = set(state_dist.index) & set(italian_regions)
        uncovered = set(italian_regions) - covered

        self.logger.info("Italian regions covered: %d/%d", len(covered), len(italian_regions))
        if uncovered:
            self.logger.info("Uncovered regions: %s", list(uncovered))

    def analyze_store_types(self) -> None:
        self.logger.info("=== STORE TYPE ANALYSIS ===")

        # Online vs physical
        self.df["is_online"] = self.df["store_id"].str.contains(
            "b2c|online|web", case=False, na=False
        )

        online = self.df["is_online"].sum()
        physical = len(self.df) - online

        self.logger.info("Store types:")
        self.logger.info("  Online/digital: %d (%.2f%%)", online, online / len(self.df) * 100)
        self.logger.info("  Physical: %d (%.2f%%)", physical, physical / len(self.df) * 100)

        # Partner type analysis
        self.logger.info("Partner type breakdown:")
        partner_types = {}
        for partner in self.df["partner_name"].unique():
            if pd.notna(partner):
                if "garmin" in partner.lower():
                    partner_types.setdefault("Garmin", 0)
                    partner_types["Garmin"] += 1
                elif "euronics" in partner.lower():
                    partner_types.setdefault("Euronics", 0)
                    partner_types["Euronics"] += 1
                elif "onedistribution" in partner.lower():
                    partner_types.setdefault("OneDistribution", 0)
                    partner_types["OneDistribution"] += 1
                elif "subbyx" in partner.lower():
                    partner_types.setdefault("Subbyx", 0)
                    partner_types["Subbyx"] += 1
                else:
                    partner_types.setdefault("Other", 0)
                    partner_types["Other"] += 1

        for ptype, count in partner_types.items():
            self.logger.info("  %s: %d stores", ptype, count)

    def analyze_fraud_rate_by_store(self) -> None:
        self.logger.info("=== FRAUD RATE BY STORE ===")

        if self.customers_ref is None or self.checkouts_ref is None:
            self.logger.info("Skipping: customers or checkouts reference not provided")
            return

        # Join checkouts with customers to get store->customer->fraud
        merged = self.checkouts_ref.merge(
            self.customers_ref[["id", "dunning_days"]],
            left_on="customer",
            right_on="id",
            how="left",
            suffixes=("_checkout", "_customer"),
        )

        # Filter to checkouts with valid store
        merged = merged[merged["store_id"].notna()]

        # Calculate fraud rate per store
        merged["is_fraud"] = merged["dunning_days"] > 15

        store_fraud = merged.groupby("store_id").agg(
            {"id_checkout": "count", "is_fraud": ["sum", "mean"]}
        )
        store_fraud.columns = ["total_checkouts", "fraud_count", "fraud_rate"]
        store_fraud = store_fraud.reset_index()

        # Filter to stores with 10+ checkouts for meaningful stats
        significant_stores = store_fraud[store_fraud["total_checkouts"] >= 10].copy()
        significant_stores = significant_stores.sort_values("fraud_rate", ascending=False)

        self.logger.info("Stores with 10+ checkouts: %d", len(significant_stores))

        if len(significant_stores) > 0:
            avg_fraud = significant_stores["fraud_rate"].mean() * 100
            self.logger.info("Average fraud rate: %.2f%%", avg_fraud)

            # High risk stores (>15% fraud)
            high_risk = significant_stores[significant_stores["fraud_rate"] > 0.15]
            self.logger.info("High-risk stores (>15%% fraud): %d", len(high_risk))

            # Low risk stores (<5% fraud)
            low_risk = significant_stores[significant_stores["fraud_rate"] < 0.05]
            self.logger.info("Low-risk stores (<5%% fraud): %d", len(low_risk))

            # Top 10 highest fraud
            self.logger.info("Top 10 highest fraud rate stores:")
            for _, row in significant_stores.head(10).iterrows():
                self.logger.info(
                    "  %s: %d checkouts, %d fraud (%.2f%%)",
                    row["store_id"],
                    int(row["total_checkouts"]),
                    int(row["fraud_count"]),
                    row["fraud_rate"] * 100,
                )

            # Top 10 lowest fraud
            self.logger.info("Top 10 lowest fraud rate stores:")
            for _, row in significant_stores.tail(10).iloc[::-1].iterrows():
                self.logger.info(
                    "  %s: %d checkouts, %d fraud (%.2f%%)",
                    row["store_id"],
                    int(row["total_checkouts"]),
                    int(row["fraud_count"]),
                    row["fraud_rate"] * 100,
                )

    def analyze_duplicates(self) -> None:
        self.logger.info("=== DUPLICATE ANALYSIS ===")

        # Duplicate store names
        name_counts = self.df["store_name"].value_counts()
        dup_names = name_counts[name_counts > 1]
        self.logger.info("Duplicate store names: %d", len(dup_names))
        if len(dup_names) > 0:
            for name, count in dup_names.items():
                self.logger.info("  %s: %d occurrences", name, count)

        # Check address duplicates
        addr_key = self.df["address"].astype(str) + "_" + self.df["zip"].astype(str)
        addr_dups = addr_key.value_counts()
        addr_dups = addr_dups[addr_dups > 1]
        self.logger.info("Duplicate addresses: %d", len(addr_dups))


def main():
    logger.info("Starting stores audit analysis")
    stores = pd.read_csv(DATA_PATH / "stores.csv")
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
    analyzer = StoresAnalyzer(stores, logger, customers_ref=customers, checkouts_ref=checkouts)
    analyzer.run_full_analysis()
    logger.info("Stores audit complete")


if __name__ == "__main__":
    main()
