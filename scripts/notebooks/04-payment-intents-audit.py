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
    create_funnel,
    create_scatter,
    save_html,
)


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("payment-intents-audit", LOG_PATH / "payment-intents-audit.log")


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
    def load_payment_intents() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "payment_intents.csv")


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


class PaymentIntentsAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, customers_ref=None, charges_ref=None):
        self.df = df
        self.logger = logger
        self.customers_ref = customers_ref
        self.charges_ref = charges_ref
        self.plot_saver = PlotSaver("payment_intents")
        self.quality = DataQualityChecker(df, "payment_intents", logger)
        self.numeric = NumericAnalyzer(df, logger, self.plot_saver)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()

        self.logger.info(
            "=== Status Distribution ===\n%s", self.categorical.get_value_counts("status")
        )
        self.logger.info(
            "=== Amount vs Received ===\n%s",
            self.numeric.describe_numeric()[["amount", "amount_received"]],
        )
        self.logger.info("=== n_failures ===\n%s", self.numeric.describe_numeric()["n_failures"])
        self.analyze_customer_fk()
        self.analyze_amount_consistency()
        self.analyze_subscription_value_consistency()
        self.analyze_latest_charge_link()
        self.analyze_canceled_vs_failed()
        self.analyze_temporal_patterns()

        # Deep payment intent analysis
        self.analyze_intent_status_funnel()
        self.analyze_failure_analysis()
        self.analyze_payment_error_codes()
        self.analyze_retry_patterns()
        self.analyze_customer_intent_behavior()
        self.analyze_outliers()

        self.generate_plots()

    def analyze_status(self) -> None:
        self.categorical.plot_value_counts("status")

    def analyze_amounts(self) -> None:
        self.numeric.plot_distribution("amount")
        self.numeric.plot_distribution("amount_received")

    def analyze_failures(self) -> None:
        self.categorical.plot_value_counts("n_failures")

    def analyze_errors(self) -> None:
        self.logger.info(
            "Payment Error Codes:\n%s",
            self.categorical.get_value_counts("payment_error_code").head(20),
        )
        self.categorical.plot_value_counts("payment_error_code", top_n=15)

    def analyze_cancellations(self) -> None:
        self.logger.info(
            "Cancellation Reasons:\n%s", self.categorical.get_value_counts("cancellation_reason")
        )

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

    def analyze_amount_consistency(self) -> None:
        self.logger.info("=== Amount vs Amount Received Consistency ===")
        has_received = self.df[self.df["amount_received"] > 0]
        self.logger.info("Payment intents with amount_received > 0: %d", len(has_received))

        fully_paid = self.df[self.df["amount"] == self.df["amount_received"]]
        partially_paid = self.df[
            (self.df["amount_received"] > 0) & (self.df["amount_received"] < self.df["amount"])
        ]
        no_payment = self.df[
            (self.df["amount_received"] == 0) | (self.df["amount_received"].isna())
        ]

        self.logger.info("Fully paid (amount == amount_received): %d", len(fully_paid))
        self.logger.info("Partially paid: %d", len(partially_paid))
        self.logger.info("No payment received: %d", len(no_payment))

    def analyze_subscription_value_consistency(self) -> None:
        self.logger.info("=== Subscription Value Analysis ===")
        self.logger.info("Null subscription_value: %d", self.df["subscription_value"].isna().sum())

        non_null = self.df[~self.df["subscription_value"].isna()]
        if len(non_null) > 0:
            self.logger.info(
                "Unique subscription values: %d", non_null["subscription_value"].nunique()
            )
            self.logger.info(
                "Subscription value distribution:\n%s",
                non_null["subscription_value"].value_counts().head(10),
            )

    def analyze_latest_charge_link(self) -> None:
        self.logger.info("=== Latest Charge Linkage ===")
        if self.charges_ref is not None:
            valid_charges = set(self.charges_ref["id"])
            null_lc = self.df["latest_charge"].isna().sum()
            invalid_lc = self.df[
                ~self.df["latest_charge"].isin(valid_charges) & self.df["latest_charge"].notna()
            ]
            self.logger.info("Null latest_charge: %d", null_lc)
            self.logger.info("Invalid latest_charge (not in charges table): %d", len(invalid_lc))
            valid_lc = self.df[self.df["latest_charge"].isin(valid_charges)]
            self.logger.info("Valid latest_charge: %d", len(valid_lc))
        else:
            self.logger.info("Charge reference not provided - skipping latest_charge validation")

    def analyze_canceled_vs_failed(self) -> None:
        self.logger.info("=== Canceled vs Failed Analysis ===")
        canceled = self.df[self.df["status"] == "canceled"]
        failed = self.df[self.df["status"] == "requires_payment_method"]

        self.logger.info("Canceled payment intents: %d", len(canceled))
        self.logger.info("Requires payment method (likely failed): %d", len(failed))

        canceled_with_reason = canceled[~canceled["cancellation_reason"].isna()]
        self.logger.info("Canceled with reason: %d", len(canceled_with_reason))
        self.logger.info(
            "Cancellation reason distribution:\n%s",
            canceled["cancellation_reason"].value_counts(dropna=False),
        )

    def analyze_temporal_patterns(self) -> None:
        self.logger.info("=== Temporal Patterns ===")
        self.df["created"] = pd.to_datetime(self.df["created"])
        self.df["created_date"] = self.df["created"].dt.date
        self.df["created_hour"] = self.df["created"].dt.hour

        daily_counts = self.df.groupby("created_date").size()
        self.logger.info(
            "Daily payment intent counts - min: %d, max: %d, mean: %.2f",
            daily_counts.min(),
            daily_counts.max(),
            daily_counts.mean(),
        )

    def analyze_intent_status_funnel(self) -> None:
        self.logger.info("=== PAYMENT INTENT STATUS FUNNEL ===")

        total = len(self.df)
        status_dist = self.df["status"].value_counts()

        self.logger.info("Status funnel:")
        for status, count in status_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", status, count, count / total * 100)

        # Success rate
        succeeded = (self.df["status"] == "succeeded").sum()
        self.logger.info("Overall success rate: %.2f%%", succeeded / total * 100)

        # Failed intents
        failed = self.df[self.df["status"] == "requires_payment_method"]
        self.logger.info(
            "Failed/pending intents: %d (%.2f%%)", len(failed), len(failed) / total * 100
        )

    def analyze_failure_analysis(self) -> None:
        self.logger.info("=== FAILURE ANALYSIS ===")

        # Intents with failures
        with_failures = self.df[self.df["n_failures"] > 0]
        no_failures = self.df[self.df["n_failures"] == 0]

        self.logger.info(
            "Intents with payment failures: %d (%.2f%%)",
            len(with_failures),
            len(with_failures) / len(self.df) * 100,
        )
        self.logger.info(
            "Intents with no failures: %d (%.2f%%)",
            len(no_failures),
            len(no_failures) / len(self.df) * 100,
        )

        # n_failures distribution
        self.logger.info("n_failures distribution:")
        for threshold in [0, 1, 5, 10, 20, 50]:
            count = (self.df["n_failures"] > threshold).sum()
            self.logger.info("  >%d failures: %d intents", threshold, count)

        # Success by failure count
        self.logger.info("Success rate by failure count:")
        for n in [0, 1, 5, 10]:
            subset = self.df[self.df["n_failures"] == n]
            if len(subset) > 0:
                success = (subset["status"] == "succeeded").sum()
                self.logger.info(
                    "  n_failures=%d: %.2f%% success (%d/%d)",
                    n,
                    success / len(subset) * 100,
                    success,
                    len(subset),
                )

    def analyze_payment_error_codes(self) -> None:
        self.logger.info("=== PAYMENT ERROR CODES ===")

        error_codes = self.df[self.df["payment_error_code"].notna()]["payment_error_code"]
        self.logger.info("Payment intents with error codes: %d", len(error_codes))

        if len(error_codes) > 0:
            self.logger.info("Error code distribution:")
            for code, count in error_codes.value_counts().head(10).items():
                self.logger.info("  %s: %d", code, count)

    def analyze_retry_patterns(self) -> None:
        self.logger.info("=== RETRY PATTERN ANALYSIS ===")

        # Multiple intents per customer
        customer_intent_counts = self.df["customer"].value_counts()

        self.logger.info("Payment intents per customer:")
        self.logger.info("  1 intent: %d customers", (customer_intent_counts == 1).sum())
        self.logger.info(
            "  2-5 intents: %d customers",
            ((customer_intent_counts >= 2) & (customer_intent_counts <= 5)).sum(),
        )
        self.logger.info("  6+ intents: %d customers", (customer_intent_counts > 5).sum())

        # Customer retry behavior
        retry_customers = customer_intent_counts[customer_intent_counts > 1].index
        retry_df = self.df[self.df["customer"].isin(retry_customers)]

        self.logger.info("Retry customer behavior:")
        retry_success = (retry_df["status"] == "succeeded").sum()
        self.logger.info("  Total intents: %d", len(retry_df))
        self.logger.info(
            "  Succeeded: %d (%.2f%%)", retry_success, retry_success / len(retry_df) * 100
        )

    def analyze_customer_intent_behavior(self) -> None:
        self.logger.info("=== CUSTOMER PAYMENT INTENT BEHAVIOR ===")

        # Group by customer
        customer_stats = (
            self.df.groupby("customer")
            .agg(
                {"id": "count", "status": lambda x: (x == "succeeded").sum(), "n_failures": "mean"}
            )
            .round(2)
        )
        customer_stats.columns = ["total_intents", "successful_intents", "avg_failures"]
        customer_stats["success_rate"] = (
            customer_stats["successful_intents"] / customer_stats["total_intents"]
        ).round(4)

        # High failure customers
        high_fail = customer_stats[customer_stats["avg_failures"] > 10]
        self.logger.info("Customers with high avg failures (>10): %d", len(high_fail))

        # Successful customers
        perfect = customer_stats[customer_stats["success_rate"] == 1.0]
        self.logger.info("Perfect payment customers (100 percent success): %d", len(perfect))

        # Never succeeded
        never_success = customer_stats[customer_stats["successful_intents"] == 0]
        self.logger.info("Never succeeded customers: %d", len(never_success))

    def analyze_outliers(self) -> None:
        self.logger.info("=== OUTLIER DETECTION ===")

        # Extreme amounts
        valid_amounts = self.df["amount"].dropna()
        if len(valid_amounts) > 0:
            q1, q3 = valid_amounts.quantile(0.25), valid_amounts.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            extreme_amounts = (valid_amounts > upper).sum()
            self.logger.info("Extreme amounts (>Q3+3*IQR): %d", extreme_amounts)

        # Extreme failure counts
        valid_failures = self.df["n_failures"].dropna()
        if len(valid_failures) > 0:
            extreme_failures = (valid_failures > 50).sum()
            self.logger.info("Extreme failure counts (>50): %d", extreme_failures)

    def generate_plots(self) -> None:

        self.logger.info("=== Generating Plotly Visualizations ===")

        self.plot_intent_funnel()
        self.plot_status_distribution()
        self.plot_amount_vs_received()
        self.plot_n_failures()
        self.plot_cancellation()
        self.plot_daily_volume()
        self.plot_success_by_failures()

    def plot_intent_funnel(self) -> None:
        status_order = [
            "requires_payment_method",
            "requires_confirmation",
            "processing",
            "succeeded",
            "canceled",
            "requires_action",
        ]
        status_counts = (
            self.df["status"].value_counts().reindex(status_order).dropna().reset_index()
        )
        status_counts.columns = ["status", "count"]

        fig = create_funnel(status_counts, "count", "status", "Payment Intent Funnel")
        self.plot_saver.save(fig, "intent_funnel")

    def plot_status_distribution(self) -> None:
        status_counts = self.df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]

        fig = create_pie_chart(status_counts, "status", "count", "Status Distribution", hole=True)
        self.plot_saver.save(fig, "status_distribution")

    def plot_amount_vs_received(self) -> None:
        valid_data = self.df[["amount", "amount_received"]].dropna()
        valid_data = valid_data[valid_data["amount"] > 0]

        fig = create_scatter(valid_data, "amount", "amount_received", "Amount vs Amount Received")
        self.plot_saver.save(fig, "amount_vs_received")

    def plot_n_failures(self) -> None:
        fig = create_histogram(
            self.df, "n_failures", title="Number of Failures Distribution", bins=20
        )
        self.plot_saver.save(fig, "n_failures")

    def plot_cancellation(self) -> None:
        canceled_df = self.df[self.df["canceled_at"].notna()]
        cancel_reasons = canceled_df.groupby("cancellation_reason").size().head(10).reset_index()
        cancel_reasons.columns = ["reason", "count"]

        fig = create_horizontal_bar(
            cancel_reasons, "count", "reason", "Cancellation Reasons", color=PURPLE_THEME["error"]
        )
        self.plot_saver.save(fig, "cancellation")

    def plot_daily_volume(self) -> None:
        self.df["created"] = pd.to_datetime(self.df["created"])
        daily = self.df.set_index("created").resample("D").agg({"id": "count"}).reset_index()
        daily.columns = ["date", "count"]

        fig = create_line_chart(daily, "date", "count", "Daily Payment Intent Volume")
        self.plot_saver.save(fig, "daily_volume")

    def plot_success_by_failures(self) -> None:
        self.df["failure_bin"] = pd.cut(
            self.df["n_failures"],
            bins=[-1, 0, 2, 5, 10, 100],
            labels=["0", "1-2", "3-5", "6-10", "10+"],
        )
        failure_stats = (
            self.df.groupby("failure_bin", observed=True)["status"]
            .apply(lambda x: (x == "succeeded").mean())
            .reset_index()
        )
        failure_stats.columns = ["n_failures", "success_rate"]
        failure_stats["success_rate"] = failure_stats["success_rate"] * 100
        failure_stats = failure_stats.dropna()

        fig = create_line_chart(
            failure_stats, "n_failures", "success_rate", "Success Rate by Failure Count"
        )
        self.plot_saver.save(fig, "success_by_failures")


def main():
    logger.info("Starting payment_intents audit analysis")
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    charges = pd.read_csv(DATA_PATH / "charges.csv")
    df = DataLoader.load_payment_intents()
    analyzer = PaymentIntentsAnalyzer(df, logger, customers_ref=customers, charges_ref=charges)
    analyzer.run_full_analysis()
    logger.info("Payment intents audit complete")


if __name__ == "__main__":
    main()
