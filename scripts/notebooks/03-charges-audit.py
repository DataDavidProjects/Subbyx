from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger
from plotly_utils import (
    PURPLE_THEME,
    create_histogram,
    create_bar_chart,
    create_horizontal_bar,
    create_pie_chart,
    create_line_chart,
    create_gauge,
    save_html,
)


DATA_PATH = Path("/Users/davidelupis/Desktop/Subbyx/data/01-clean/")
LOG_PATH = Path("/Users/davidelupis/Desktop/Subbyx/scripts/notebooks/logs/")

logger = setup_logger("charges-audit", LOG_PATH / "charges-audit.log")


class PlotSaver:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.plots_dir = LOG_PATH / "plots"
        self.plots_dir.mkdir(exist_ok=True)

    def save(self, fig, name: str) -> None:
        filename = f"{self.prefix}_{name}"
        save_html(fig, filename, self.plots_dir)
        logger.info("Saved plot: %s", filename)


class DataLoader:
    @staticmethod
    def load_charges() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "charges.csv")


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


class NumericAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, plot_saver: PlotSaver):
        self.df = df
        self.logger = logger
        self.plot_saver = plot_saver

    def describe_numeric(self) -> pd.DataFrame:
        return self.df.describe()

    def get_numeric_columns(self) -> list:
        return self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    def plot_distribution(self, column: str, bins: int = 30) -> None:
        fig = create_histogram(self.df, column, bins=bins)
        self.plot_saver.save(fig, column)


class CategoricalAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, plot_saver: PlotSaver):
        self.df = df
        self.logger = logger
        self.plot_saver = plot_saver

    def get_categorical_columns(self) -> list:
        return self.df.select_dtypes(include=["object", "category"]).columns.tolist()

    def get_value_counts(self, column: str) -> pd.Series:
        return self.df[column].value_counts()

    def plot_value_counts(self, column: str, top_n: int = 20) -> None:
        vc = self.get_value_counts(column).head(top_n).reset_index()
        vc.columns = [column, "count"]
        fig = create_bar_chart(
            vc, column, "count", f"Value Counts: {column}", color=PURPLE_THEME["secondary"]
        )
        self.plot_saver.save(fig, column)


class ChargesAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, customers_ref=None, payment_intents_ref=None):
        self.df = df
        self.logger = logger
        self.customers_ref = customers_ref
        self.payment_intents_ref = payment_intents_ref
        self.plot_saver = PlotSaver("charges")
        self.quality = DataQualityChecker(df, "charges", logger)
        self.numeric = NumericAnalyzer(df, logger, self.plot_saver)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()

        self.logger.info("=== Paid vs Unpaid ===\n%s", self.categorical.get_value_counts("paid"))
        self.logger.info(
            "=== Status Distribution ===\n%s", self.categorical.get_value_counts("status")
        )
        self.logger.info(
            "=== Outcome Risk ===\n%s", self.categorical.get_value_counts("outcome_risk")
        )
        self.analyze_customer_fk()
        self.analyze_payment_intent_fk()
        self.analyze_card_fingerprint()
        self.analyze_recurrent_vs_paid()
        self.analyze_outcome_details()
        self.analyze_temporal_patterns()

        # Deep payment behavior analysis
        self.analyze_payment_success_rates()
        self.analyze_failure_patterns()
        self.analyze_card_brand_analysis()
        self.analyze_customer_payment_behavior()
        self.analyze_outcome_risk_correlation()

        self.generate_plots()

    def analyze_payment_status(self) -> None:
        self.categorical.plot_value_counts("paid")

    def analyze_outcomes(self) -> None:
        self.categorical.plot_value_counts("outcome_status", top_n=15)
        self.categorical.plot_value_counts("outcome_risk")
        self.logger.info(
            "Outcome Risk Score:\n%s", self.numeric.describe_numeric()["outcome_risk_score"]
        )

    def analyze_failure_codes(self) -> None:
        self.logger.info("Failure Codes:\n%s", self.categorical.get_value_counts("failure_code"))
        self.categorical.plot_value_counts("failure_code", top_n=15)

    def analyze_cards(self) -> None:
        self.logger.info("Card Brands:\n%s", self.categorical.get_value_counts("card_brand"))
        self.logger.info(
            "Card Issuers (top 20):\n%s", self.categorical.get_value_counts("card_issuer").head(20)
        )

    def analyze_recurrent(self) -> None:
        self.logger.info("Is Recurrent:\n%s", self.categorical.get_value_counts("is_recurrent"))

    def analyze_customer_fk(self) -> None:
        self.logger.info("=== Customer FK Validation ===")
        if self.customers_ref is not None:
            valid_customers = set(self.customers_ref["id"])
            null_cust = self.df["customer"].isna().sum()
            invalid_cust = self.df[
                ~self.df["customer"].isin(valid_customers) & self.df["customer"].notna()
            ]
            self.logger.info("Null customer: %d", null_cust)
            self.logger.info("Invalid customer (not in customers table): %d", len(invalid_cust))
        else:
            self.logger.info("Customer reference not provided - skipping FK validation")

    def analyze_payment_intent_fk(self) -> None:
        self.logger.info("=== Payment Intent FK Validation ===")
        if self.payment_intents_ref is not None:
            valid_pi = set(self.payment_intents_ref["id"])
            null_pi = self.df["payment_intent"].isna().sum()
            invalid_pi = self.df[~self.df["payment_intent"].isin(valid_pi)]
            self.logger.info("Null payment_intent: %d", null_pi)
            self.logger.info(
                "Invalid payment_intent (not in payment_intents table): %d", len(invalid_pi)
            )
        else:
            self.logger.info("Payment intent reference not provided - skipping FK validation")

    def analyze_card_fingerprint(self) -> None:
        self.logger.info("=== Card Fingerprint Analysis ===")
        unique_cards = self.df["card_fingerprint"].nunique()
        total_charges = len(self.df)
        self.logger.info("Unique card fingerprints: %d", unique_cards)
        self.logger.info("Total charges: %d", total_charges)
        self.logger.info("Avg charges per card: %.2f", total_charges / unique_cards)

        card_counts = self.df.groupby("card_fingerprint").size()
        multi_charge_cards = card_counts[card_counts > 5]
        self.logger.info("Cards with >5 charges: %d", len(multi_charge_cards))

        self.logger.info(
            "Card brand distribution:\n%s", self.categorical.get_value_counts("card_brand")
        )
        self.logger.info(
            "Card funding distribution:\n%s", self.categorical.get_value_counts("card_funding")
        )

    def analyze_recurrent_vs_paid(self) -> None:
        self.logger.info("=== Is Recurrent vs Paid Analysis ===")
        cross_tab = pd.crosstab(self.df["is_recurrent"], self.df["paid"], margins=True)
        self.logger.info("\n%s", cross_tab)

        recurrent_paid_rate = self.df[self.df["is_recurrent"] == True]["paid"].mean()
        first_paid_rate = self.df[self.df["is_recurrent"] == False]["paid"].mean()
        self.logger.info("Paid rate (recurrent): %.2f%%", recurrent_paid_rate * 100)
        self.logger.info("Paid rate (first payment): %.2f%%", first_paid_rate * 100)

    def analyze_outcome_details(self) -> None:
        self.logger.info("=== Outcome Details ===")
        self.logger.info(
            "Outcome risk level:\n%s", self.categorical.get_value_counts("outcome_risk_level")
        )
        self.logger.info("Outcome type:\n%s", self.categorical.get_value_counts("outcome_type"))
        self.logger.info(
            "Outcome risk score:\n%s", self.numeric.describe_numeric()["outcome_risk_score"]
        )

    def analyze_temporal_patterns(self) -> None:
        self.logger.info("=== Temporal Patterns ===")
        self.df["created"] = pd.to_datetime(self.df["created"])
        self.df["created_date"] = self.df["created"].dt.date
        self.df["created_hour"] = self.df["created"].dt.hour

        daily_counts = self.df.groupby("created_date").size()
        self.logger.info(
            "Daily charge counts - min: %d, max: %d, mean: %.2f",
            daily_counts.min(),
            daily_counts.max(),
            daily_counts.mean(),
        )

        hour_dist = self.df["created_hour"].value_counts().sort_index()
        self.logger.info("Hour distribution:\n%s", hour_dist)

    def analyze_payment_success_rates(self) -> None:
        self.logger.info("=== PAYMENT SUCCESS RATE ANALYSIS ===")

        overall_rate = self.df["paid"].mean() * 100
        self.logger.info("Overall payment success rate: %.2f%%", overall_rate)

        # By recurrent
        self.logger.info("Success rate by payment type:")
        for is_recurrent in [False, True]:
            subset = self.df[self.df["is_recurrent"] == is_recurrent]
            rate = subset["paid"].mean() * 100
            self.logger.info(
                "  %s: %.2f%% (%d/%d)",
                "First payment" if not is_recurrent else "Recurrent",
                rate,
                subset["paid"].sum(),
                len(subset),
            )

        # By card brand
        self.logger.info("Success rate by card brand:")
        for brand in self.df["card_brand"].unique():
            if pd.notna(brand):
                subset = self.df[self.df["card_brand"] == brand]
                rate = subset["paid"].mean() * 100
                self.logger.info(
                    "  %s: %.2f%% (%d/%d)", brand, rate, subset["paid"].sum(), len(subset)
                )

    def analyze_failure_patterns(self) -> None:
        self.logger.info("=== FAILURE PATTERN ANALYSIS ===")

        failed = self.df[self.df["paid"] == False]

        self.logger.info(
            "Total failed charges: %d (%.2f%%)", len(failed), len(failed) / len(self.df) * 100
        )

        # Failure codes
        self.logger.info("Failure code distribution:")
        failure_codes = failed["failure_code"].value_counts()
        for code, count in failure_codes.items():
            self.logger.info("  %s: %d (%.2f%%)", code, count, count / len(failed) * 100)

        # Outcome status for failed
        self.logger.info("Outcome status for failed charges:")
        outcome_dist = failed["outcome_status"].value_counts()
        for status, count in outcome_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", status, count, count / len(failed) * 100)

    def analyze_card_brand_analysis(self) -> None:
        self.logger.info("=== CARD BRAND ANALYSIS ===")

        brand_dist = self.df["card_brand"].value_counts()
        self.logger.info("Card brand distribution:")
        for brand, count in brand_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", brand, count, count / len(self.df) * 100)

        # Card funding
        self.logger.info("Card funding distribution:")
        funding_dist = self.df["card_funding"].value_counts()
        for funding, count in funding_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", funding, count, count / len(self.df) * 100)

        # Top issuers
        self.logger.info("Top 10 card issuers:")
        issuer_dist = self.df["card_issuer"].value_counts().head(10)
        for issuer, count in issuer_dist.items():
            self.logger.info("  %s: %d", issuer, count)

        # Card CVC check
        self.logger.info("Card CVC check distribution:")
        cvc_dist = self.df["card_cvc_check"].value_counts()
        for cvc, count in cvc_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", cvc, count, count / len(self.df) * 100)

    def analyze_customer_payment_behavior(self) -> None:
        self.logger.info("=== CUSTOMER PAYMENT BEHAVIOR ===")

        # Charges per customer
        customer_charge_counts = self.df["customer"].value_counts()

        self.logger.info("Charge counts per customer:")
        self.logger.info("  1 charge: %d customers", (customer_charge_counts == 1).sum())
        self.logger.info(
            "  2-5 charges: %d customers",
            ((customer_charge_counts >= 2) & (customer_charge_counts <= 5)).sum(),
        )
        self.logger.info(
            "  6-20 charges: %d customers",
            ((customer_charge_counts >= 6) & (customer_charge_counts <= 20)).sum(),
        )
        self.logger.info("  20+ charges: %d customers", (customer_charge_counts > 20).sum())

        # Payment behavior patterns
        self.logger.info("Customer payment behavior patterns:")

        customer_stats = (
            self.df.groupby("customer")
            .agg({"paid": ["count", "sum", "mean"], "is_recurrent": "mean"})
            .round(4)
        )
        customer_stats.columns = [
            "total_charges",
            "successful_payments",
            "success_rate",
            "recurrent_ratio",
        ]

        # High failure customers
        high_fail = customer_stats[customer_stats["success_rate"] < 0.3]
        self.logger.info("  High-risk customers (<30 percent success): %d", len(high_fail))

        # Perfect payment customers
        perfect = customer_stats[customer_stats["success_rate"] == 1.0]
        self.logger.info("  Perfect payment customers (100 percent success): %d", len(perfect))

    def analyze_outcome_risk_correlation(self) -> None:
        self.logger.info("=== OUTCOME RISK CORRELATION ===")

        # Risk vs actual payment
        self.logger.info("Payment success by risk level:")
        for risk in self.df["outcome_risk"].unique():
            if pd.notna(risk):
                subset = self.df[self.df["outcome_risk"] == risk]
                rate = subset["paid"].mean() * 100
                self.logger.info("  %s: %.2f%% success", risk, rate)

    def generate_plots(self) -> None:

        self.logger.info("=== Generating Plotly Visualizations ===")

        self.plot_success_gauge()
        self.plot_first_vs_recurrent()
        self.plot_card_brand()
        self.plot_failure_codes()
        self.plot_outcome_risk()
        self.plot_daily_volume()
        self.plot_customer_segments()

    def plot_success_gauge(self) -> None:
        success_rate = self.df["paid"].mean() * 100
        fig = create_gauge(success_rate, "Payment Success Rate", threshold=80)
        self.plot_saver.save(fig, "success_gauge")

    def plot_first_vs_recurrent(self) -> None:
        first_recurrent = self.df.groupby("is_recurrent")["paid"].mean().reset_index()
        first_recurrent.columns = ["is_recurrent", "success_rate"]
        first_recurrent["success_rate"] = first_recurrent["success_rate"] * 100

        fig = create_bar_chart(
            first_recurrent,
            "is_recurrent",
            "success_rate",
            "First vs Recurrent Payment Success",
            color=PURPLE_THEME["secondary"],
        )
        self.plot_saver.save(fig, "first_vs_recurrent")

    def plot_card_brand(self) -> None:
        brand_counts = self.df["card_brand"].value_counts().reset_index()
        brand_counts.columns = ["brand", "count"]

        fig = create_pie_chart(brand_counts, "brand", "count", "Card Brand Distribution", hole=True)
        self.plot_saver.save(fig, "card_brand")

    def plot_failure_codes(self) -> None:
        failed = self.df[~self.df["paid"]]
        failure_codes = failed["failure_code"].value_counts().head(10).reset_index()
        failure_codes.columns = ["failure_code", "count"]

        fig = create_horizontal_bar(
            failure_codes, "count", "failure_code", "Failure Codes", color=PURPLE_THEME["error"]
        )
        self.plot_saver.save(fig, "failure_codes")

    def plot_outcome_risk(self) -> None:
        risk_paid = self.df.groupby("outcome_risk")["paid"].mean().reset_index()
        risk_paid.columns = ["outcome_risk", "success_rate"]
        risk_paid = risk_paid.dropna().sort_values("success_rate")
        risk_paid["success_rate"] = risk_paid["success_rate"] * 100

        fig = create_bar_chart(
            risk_paid,
            "outcome_risk",
            "success_rate",
            "Success Rate by Risk Level",
            color=PURPLE_THEME["primary_dark"],
        )
        self.plot_saver.save(fig, "outcome_risk")

    def plot_daily_volume(self) -> None:
        self.df["created"] = pd.to_datetime(self.df["created"])
        daily = self.df.set_index("created").resample("D").agg({"id": "count"}).reset_index()
        daily.columns = ["date", "count"]

        fig = create_line_chart(daily, "date", "count", "Daily Charge Volume")
        self.plot_saver.save(fig, "daily_volume")

    def plot_customer_segments(self) -> None:
        if self.customers_ref is not None:
            merged = self.df.merge(
                self.customers_ref[["id", "gender"]], left_on="customer", right_on="id", how="left"
            )
            gender_counts = merged["gender"].value_counts().reset_index()
            gender_counts.columns = ["gender", "count"]

            fig = create_pie_chart(
                gender_counts, "gender", "count", "Customer Segments by Gender", hole=True
            )
            self.plot_saver.save(fig, "customer_segments")


def main():
    logger.info("Starting charges audit analysis")
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    payment_intents = pd.read_csv(DATA_PATH / "payment_intents.csv")
    df = DataLoader.load_charges()
    analyzer = ChargesAnalyzer(
        df, logger, customers_ref=customers, payment_intents_ref=payment_intents
    )
    analyzer.run_full_analysis()
    logger.info("Charges audit complete")


if __name__ == "__main__":
    main()
