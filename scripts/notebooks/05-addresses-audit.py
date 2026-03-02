from __future__ import annotations

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path("/Users/davidelupis/Desktop/Subbyx/data/01-clean/")
LOG_PATH = Path("/Users/davidelupis/Desktop/Subbyx/scripts/notebooks/logs/")

logger = setup_logger("addresses-audit", LOG_PATH / "addresses-audit.log")


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
    def load_addresses() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "addresses.csv")


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


class AddressesAnalyzer:
    def __init__(self, df: pd.DataFrame, logger):
        self.df = df
        self.logger = logger
        self.plot_saver = PlotSaver("addresses")
        self.quality = DataQualityChecker(df, "addresses", logger)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()
        self.logger.info(
            "=== Country Distribution ===\n%s", self.categorical.get_value_counts("country")
        )
        self.analyze_duplicate_ids()
        self.analyze_postal_code_format()
        self.analyze_geographic_completeness()

        # Deep addresses analysis
        self.analyze_duplicate_patterns()
        self.analyze_geographic_distribution()
        self.analyze_address_quality_issues()
        self.analyze_locality()

    def analyze_geography(self) -> None:
        self.categorical.plot_value_counts("country")
        self.logger.info(
            "City Distribution (top 20):\n%s", self.categorical.get_value_counts("city").head(20)
        )
        self.logger.info(
            "State Distribution (top 20):\n%s", self.categorical.get_value_counts("state").head(20)
        )

    def analyze_duplicate_ids(self) -> None:
        self.logger.info("=== Duplicate ID Analysis ===")
        total = len(self.df)
        duplicate_ids = self.df[self.df.duplicated(subset=["id"], keep=False)]
        unique_ids = self.df["id"].nunique()

        self.logger.info("Total addresses: %d", total)
        self.logger.info("Unique IDs: %d", unique_ids)
        self.logger.info(
            "Addresses with duplicate IDs: %d (%.2f%%)",
            len(duplicate_ids),
            (len(duplicate_ids) / total) * 100,
        )

        id_counts = self.df.groupby("id").size()
        duplicate_id_counts = id_counts[id_counts > 1]
        self.logger.info("IDs appearing more than once: %d", len(duplicate_id_counts))
        if len(duplicate_id_counts) > 0:
            self.logger.info("Max occurrences of a single ID: %d", duplicate_id_counts.max())
            self.logger.info(
                "Sample duplicate IDs and their counts:\n%s", duplicate_id_counts.head(10)
            )

    def analyze_postal_code_format(self) -> None:
        self.logger.info("=== Postal Code Format Analysis ===")
        null_postal = self.df["postal_code"].isna().sum()
        self.logger.info(
            "Null postal_code: %d (%.2f%%)", null_postal, (null_postal / len(self.df)) * 100
        )

        non_null_postal = self.df[~self.df["postal_code"].isna()]
        if len(non_null_postal) > 0:
            non_null_postal = non_null_postal.copy()
            non_null_postal["postal_str"] = non_null_postal["postal_code"].astype(int).astype(str)
            non_null_postal["postal_length"] = non_null_postal["postal_str"].str.len()

            self.logger.info(
                "Postal code length distribution:\n%s",
                non_null_postal["postal_length"].value_counts(),
            )

            valid_italian = non_null_postal[non_null_postal["postal_length"] == 5]
            invalid_length = len(non_null_postal) - len(valid_italian)
            self.logger.info("Valid Italian format (5 digits): %d", len(valid_italian))
            self.logger.info("Invalid length: %d", invalid_length)

    def analyze_geographic_completeness(self) -> None:
        self.logger.info("=== Geographic Completeness Analysis ===")
        total = len(self.df)

        null_city = self.df["city"].isna().sum()
        null_state = self.df["state"].isna().sum()
        null_admin = self.df["administrative_area_level_1"].isna().sum()
        null_country = self.df["country"].isna().sum()

        self.logger.info("Null city: %d (%.2f%%)", null_city, (null_city / total) * 100)
        self.logger.info("Null state: %d (%.2f%%)", null_state, (null_state / total) * 100)
        self.logger.info(
            "Null administrative_area_level_1: %d (%.2f%%)", null_admin, (null_admin / total) * 100
        )
        self.logger.info("Null country: %d (%.2f%%)", null_country, (null_country / total) * 100)

        complete_address = self.df[
            (~self.df["city"].isna()) & (~self.df["state"].isna()) & (~self.df["country"].isna())
        ]
        self.logger.info(
            "Complete address (city, state, country): %d (%.2f%%)",
            len(complete_address),
            (len(complete_address) / total) * 100,
        )

    def analyze_duplicate_patterns(self) -> None:
        self.logger.info("=== DUPLICATE PATTERN ANALYSIS ===")

        # Analyze duplicate ID patterns
        id_counts = self.df.groupby("id").size()
        duplicate_id_counts = id_counts[id_counts > 1]

        self.logger.info("ID occurrence distribution:")
        occ_dist = id_counts.value_counts().sort_index()
        for occ, count in occ_dist.items():
            self.logger.info("  Appears %d times: %d IDs", occ, count)

        # Check if duplicates have same data
        self.logger.info("Duplicate content analysis:")
        for occ in [2, 3, 4, 5]:
            sample_ids = duplicate_id_counts[duplicate_id_counts == occ].head(5).index
            for addr_id in sample_ids:
                subset = self.df[self.df["id"] == addr_id]
                cities = subset["city"].unique()
                self.logger.info(
                    "  ID %s: %d records, cities: %s", str(addr_id)[:8], len(subset), cities
                )

    def analyze_geographic_distribution(self) -> None:
        self.logger.info("=== GEOGRAPHIC DISTRIBUTION ===")

        # Top cities
        self.logger.info("Top 15 cities:")
        for city, count in self.df["city"].value_counts().head(15).items():
            if pd.notna(city):
                self.logger.info("  %s: %d addresses", city, count)

        # Top states
        self.logger.info("Top 15 states:")
        for state, count in self.df["state"].value_counts().head(15).items():
            if pd.notna(state):
                self.logger.info("  %s: %d addresses", state, count)

        # Administrative areas
        self.logger.info("Top 10 administrative areas:")
        for area, count in self.df["administrative_area_level_1"].value_counts().head(10).items():
            if pd.notna(area):
                self.logger.info("  %s: %d addresses", area, count)

    def analyze_address_quality_issues(self) -> None:
        self.logger.info("=== ADDRESS QUALITY ISSUES ===")

        # Same address used multiple times
        self.logger.info("Potential address quality issues:")

        # Group by full address
        address_key = (
            self.df["city"].astype(str)
            + "_"
            + self.df["state"].astype(str)
            + "_"
            + self.df["postal_code"].astype(str)
        )
        address_counts = address_key.value_counts()

        reused = address_counts[address_counts > 1]
        self.logger.info("Reused address combinations: %d", len(reused))

        if len(reused) > 0:
            self.logger.info("Most reused addresses:")
            for addr, count in reused.head(5).items():
                self.logger.info("  %s: %d times", addr, count)

    def analyze_locality(self) -> None:
        self.logger.info("=== LOCALITY ANALYSIS ===")

        # Locality distribution
        self.logger.info("Top 15 localities:")
        locality_dist = self.df["locality"].value_counts().head(15)
        for loc, count in locality_dist.items():
            if pd.notna(loc):
                self.logger.info("  %s: %d addresses", loc, count)


def main():
    logger.info("Starting addresses audit analysis")
    df = DataLoader.load_addresses()
    analyzer = AddressesAnalyzer(df, logger)
    analyzer.run_full_analysis()
    logger.info("Addresses audit complete")


if __name__ == "__main__":
    main()
