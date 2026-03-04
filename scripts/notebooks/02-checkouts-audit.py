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
    save_html,
)


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("checkouts-audit", LOG_PATH / "checkouts-audit.log")


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
    def load_checkouts() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "checkouts.csv")


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


class CheckoutsAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, customers_ref=None, stores_ref=None):
        self.df = df
        self.logger = logger
        self.customers_ref = customers_ref
        self.stores_ref = stores_ref
        self.plot_saver = PlotSaver("checkouts")
        self.quality = DataQualityChecker(df, "checkouts", logger)
        self.numeric = NumericAnalyzer(df, logger, self.plot_saver)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()

        self.logger.info(
            "=== Status Distribution ===\n%s", self.categorical.get_value_counts("status")
        )
        self.logger.info("=== Mode Distribution ===\n%s", self.categorical.get_value_counts("mode"))
        self.logger.info(
            "=== Subscription Value ===\n%s", self.numeric.describe_numeric()["subscription_value"]
        )
        self.analyze_dates()
        self.analyze_customer_fk()
        self.analyze_store_fk()
        self.analyze_nulls_by_mode()
        self.analyze_sku()
        self.analyze_condition()
        self.analyze_temporal_patterns()

        # Deep funnel analysis
        self.analyze_checkout_funnel()
        self.analyze_status_by_mode()
        self.analyze_subscription_value_distribution()
        self.analyze_category_grade_analysis()
        self.analyze_addon_impact()
        self.analyze_customer_checkout_patterns()
        self.analyze_store_performance()
        self.analyze_payment_intent_linkage()
        self.analyze_outliers()

        self.generate_plots()

    def analyze_dates(self) -> None:
        self.df["created"] = pd.to_datetime(self.df["created"])
        min_date = self.df["created"].min()
        max_date = self.df["created"].max()
        total_days = (max_date - min_date).days

        self.logger.info("=== Date Analysis ===")
        self.logger.info(
            "Date range: %s to %s", min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")
        )
        self.logger.info("Total days: %s", total_days)

        self.df["created_month"] = self.df["created"].dt.to_period("M")
        monthly_counts = self.df.groupby("created_month").size()
        self.logger.info("Monthly distribution:\n%s", monthly_counts)

    def analyze_status(self) -> None:
        self.categorical.plot_value_counts("status")

    def analyze_mode(self) -> None:
        self.categorical.plot_value_counts("mode")

    def analyze_subscription_value(self) -> None:
        self.numeric.plot_distribution("subscription_value")

    def analyze_grades(self) -> None:
        self.logger.info("Grade Distribution:\n%s", self.categorical.get_value_counts("grade"))
        self.categorical.plot_value_counts("grade")

    def analyze_addons(self) -> None:
        addon_cols = [
            "has_linked_products",
            "has_vetrino",
            "has_protezione_totale",
            "has_protezione_furto",
        ]
        self.logger.info("=== Add-on Features ===")
        for col in addon_cols:
            self.logger.info("%s: %s", col, self.df[col].value_counts().to_dict())

    def analyze_categories(self) -> None:
        self.logger.info("Product Categories:\n%s", self.categorical.get_value_counts("category"))
        self.categorical.plot_value_counts("category", top_n=10)

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

    def analyze_store_fk(self) -> None:
        self.logger.info("=== Store FK Validation ===")
        if self.stores_ref is not None:
            valid_stores = set(self.stores_ref["store_id"].dropna())
            null_store = self.df["store_id"].isna().sum()
            invalid_store = self.df[
                ~self.df["store_id"].isin(valid_stores) & self.df["store_id"].notna()
            ]
            self.logger.info("Null store_id: %d", null_store)
            self.logger.info("Invalid store_id (not in stores table): %d", len(invalid_store))
            if len(invalid_store) > 0:
                self.logger.info(
                    "Invalid store_ids: %s", invalid_store["store_id"].unique().tolist()[:10]
                )
        else:
            self.logger.info("Store reference not provided - skipping FK validation")

    def analyze_nulls_by_mode(self) -> None:
        self.logger.info("=== Null Analysis by Mode ===")
        for mode in self.df["mode"].unique():
            mode_df = self.df[self.df["mode"] == mode]
            self.logger.info("Mode: %s (count: %d)", mode, len(mode_df))
            null_cols = mode_df.isnull().sum()
            for col, null_count in null_cols.items():
                if null_count > 0:
                    self.logger.info(
                        "  %s: %d null (%.2f%%)", col, null_count, (null_count / len(mode_df)) * 100
                    )

    def analyze_sku(self) -> None:
        self.logger.info("=== SKU Analysis ===")
        total = len(self.df)
        null_sku = self.df["sku"].isna().sum()
        self.logger.info("Null SKU: %d (%.2f%%)", null_sku, (null_sku / total) * 100)
        non_null_sku = self.df[~self.df["sku"].isna()]
        self.logger.info("Unique SKUs: %d", non_null_sku["sku"].nunique())
        self.logger.info("Top 10 SKUs:\n%s", non_null_sku["sku"].value_counts().head(10))

    def analyze_condition(self) -> None:
        self.logger.info("=== Product Condition Distribution ===")
        self.logger.info("\n%s", self.categorical.get_value_counts("condition"))
        null_condition = self.df["condition"].isna().sum()
        self.logger.info(
            "Null condition: %d (%.2f%%)", null_condition, (null_condition / len(self.df)) * 100
        )

    def analyze_temporal_patterns(self) -> None:
        self.logger.info("=== Temporal Patterns ===")
        self.df["created"] = pd.to_datetime(self.df["created"])
        self.df["created_hour"] = self.df["created"].dt.hour
        self.df["created_dayofweek"] = self.df["created"].dt.dayofweek

        hour_dist = self.df["created_hour"].value_counts().sort_index()
        self.logger.info("Hour distribution:\n%s", hour_dist)

        dow_dist = self.df["created_dayofweek"].value_counts().sort_index()
        dow_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }
        self.logger.info(
            "Day of week distribution: %s",
            {dow_names.get(k, k): v for k, v in dow_dist.to_dict().items()},
        )

    def analyze_checkout_funnel(self) -> None:
        self.logger.info("=== CHECKOUT FUNNEL ANALYSIS ===")

        total = len(self.df)

        # Mode funnel
        setup = (self.df["mode"] == "setup").sum()
        payment = (self.df["mode"] == "payment").sum()
        subscription = (self.df["mode"] == "subscription").sum()

        self.logger.info("Mode funnel:")
        self.logger.info("  Setup (registration): %d (%.2f%%)", setup, setup / total * 100)
        self.logger.info(
            "  Payment (subscription request): %d (%.2f%%)", payment, payment / total * 100
        )
        self.logger.info("  Subscription: %d (%.2f%%)", subscription, subscription / total * 100)

        # Status funnel
        complete = (self.df["status"] == "complete").sum()
        expired = (self.df["status"] == "expired").sum()

        self.logger.info("Status funnel:")
        self.logger.info("  Complete: %d (%.2f%%)", complete, complete / total * 100)
        self.logger.info("  Expired: %d (%.2f%%)", expired, expired / total * 100)

        # Conversion funnel: setup -> payment -> complete
        setup_count = self.df[self.df["mode"] == "setup"]
        payment_count = self.df[self.df["mode"].isin(["payment", "subscription"])]
        complete_count = self.df[self.df["status"] == "complete"]

        self.logger.info("Conversion funnel:")
        self.logger.info("  Total checkouts: %d", total)
        self.logger.info(
            "  Payment/Subscription requests: %d (%.2f%% of total)",
            len(payment_count),
            len(payment_count) / total * 100,
        )
        self.logger.info(
            "  Complete: %d (%.2f%% of total, %.2f%% of payment)",
            len(complete_count),
            len(complete_count) / total * 100,
            len(complete_count) / len(payment_count) * 100 if len(payment_count) > 0 else 0,
        )

    def analyze_status_by_mode(self) -> None:
        self.logger.info("=== STATUS BY MODE ANALYSIS ===")

        crosstab = pd.crosstab(self.df["mode"], self.df["status"], normalize="index") * 100
        self.logger.info("Completion rate by mode:")
        for mode in self.df["mode"].unique():
            mode_df = self.df[self.df["mode"] == mode]
            complete = (mode_df["status"] == "complete").sum()
            total = len(mode_df)
            rate = complete / total * 100 if total > 0 else 0
            self.logger.info("  %s: %.2f%% complete (%d/%d)", mode, rate, complete, total)

    def analyze_subscription_value_distribution(self) -> None:
        self.logger.info("=== SUBSCRIPTION VALUE ANALYSIS ===")

        # Only payment/subscription modes have subscription values
        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]
        valid_sv = payment_df[payment_df["subscription_value"].notna()]

        self.logger.info(
            "Payment/Subscription checkouts with value: %d/%d", len(valid_sv), len(payment_df)
        )

        if len(valid_sv) > 0:
            self.logger.info("Subscription value statistics:")
            self.logger.info("\n%s", valid_sv["subscription_value"].describe())

            # Value buckets
            self.logger.info("Value distribution buckets:")
            buckets = [0, 20, 40, 60, 80, 100, 200, 300]
            for i in range(len(buckets) - 1):
                count = (
                    (valid_sv["subscription_value"] >= buckets[i])
                    & (valid_sv["subscription_value"] < buckets[i + 1])
                ).sum()
                self.logger.info(
                    "  %d-%d: %d (%.2f%%)",
                    buckets[i],
                    buckets[i + 1],
                    count,
                    count / len(valid_sv) * 100,
                )

    def analyze_category_grade_analysis(self) -> None:
        self.logger.info("=== CATEGORY AND GRADE ANALYSIS ===")

        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]

        # Top categories
        self.logger.info("Top 10 categories:")
        top_cats = payment_df["category"].value_counts().head(10)
        for cat, count in top_cats.items():
            self.logger.info("  %s: %d", cat, count)

        # Grade distribution
        self.logger.info("Grade distribution:")
        grade_dist = payment_df["grade"].value_counts()
        for grade, count in grade_dist.items():
            self.logger.info("  %s: %d (%.2f%%)", grade, count, count / len(payment_df) * 100)

        # Category by status
        self.logger.info("Completion rate by category (top 5):")
        for cat in top_cats.head(5).index:
            cat_df = payment_df[payment_df["category"] == cat]
            complete = (cat_df["status"] == "complete").sum()
            self.logger.info(
                "  %s: %.2f%% complete (%d/%d)",
                cat,
                complete / len(cat_df) * 100 if len(cat_df) > 0 else 0,
                complete,
                len(cat_df),
            )

    def analyze_addon_impact(self) -> None:
        self.logger.info("=== ADD-ON IMPACT ANALYSIS ===")

        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]

        addon_cols = [
            "has_linked_products",
            "has_vetrino",
            "has_protezione_totale",
            "has_protezione_furto",
        ]

        for addon in addon_cols:
            with_addon = payment_df[payment_df[addon] == True]
            without_addon = payment_df[payment_df[addon] == False]

            with_complete = (with_addon["status"] == "complete").sum() if len(with_addon) > 0 else 0
            without_complete = (
                (without_addon["status"] == "complete").sum() if len(without_addon) > 0 else 0
            )

            with_rate = with_complete / len(with_addon) * 100 if len(with_addon) > 0 else 0
            without_rate = (
                without_complete / len(without_addon) * 100 if len(without_addon) > 0 else 0
            )

            self.logger.info("%s:", addon)
            self.logger.info(
                "  With addon: %.2f%% complete (%d/%d)", with_rate, with_complete, len(with_addon)
            )
            self.logger.info(
                "  Without addon: %.2f%% complete (%d/%d)",
                without_rate,
                without_complete,
                len(without_addon),
            )

    def analyze_customer_checkout_patterns(self) -> None:
        self.logger.info("=== CUSTOMER CHECKOUT PATTERNS ===")

        customer_counts = self.df["customer"].value_counts()

        self.logger.info("Checkouts per customer:")
        self.logger.info("  1 checkout: %d customers", (customer_counts == 1).sum())
        self.logger.info(
            "  2-5 checkouts: %d customers", ((customer_counts >= 2) & (customer_counts <= 5)).sum()
        )
        self.logger.info("  6+ checkouts: %d customers", (customer_counts > 5).sum())
        self.logger.info("  Max checkouts: %d", customer_counts.max())

        # Repeat customers analysis
        repeat_customers = customer_counts[customer_counts > 1].index
        repeat_df = self.df[self.df["customer"].isin(repeat_customers)]

        self.logger.info("Repeat customer behavior:")
        repeat_complete = (repeat_df["status"] == "complete").sum()
        self.logger.info("  Total checkouts: %d", len(repeat_df))
        self.logger.info(
            "  Complete: %d (%.2f%%)", repeat_complete, repeat_complete / len(repeat_df) * 100
        )

    def analyze_store_performance(self) -> None:
        self.logger.info("=== STORE PERFORMANCE ANALYSIS ===")

        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]

        # Store checkout volume
        store_counts = payment_df["store_id"].value_counts()
        self.logger.info("Top 10 stores by volume:")
        for store, count in store_counts.head(10).items():
            if pd.notna(store):
                store_df = payment_df[payment_df["store_id"] == store]
                complete = (store_df["status"] == "complete").sum()
                rate = complete / count * 100 if count > 0 else 0
                self.logger.info("  %s: %d checkouts, %.2f%% complete", store, count, rate)

    def analyze_payment_intent_linkage(self) -> None:
        self.logger.info("=== PAYMENT INTENT LINKAGE ===")

        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]

        # Payment intent null by mode
        self.logger.info("Payment intent null by mode:")
        for mode in payment_df["mode"].unique():
            mode_df = payment_df[payment_df["mode"] == mode]
            null_pi = mode_df["payment_intent"].isna().sum()
            self.logger.info("  %s: %d null (%.2f%%)", mode, null_pi, null_pi / len(mode_df) * 100)

    def analyze_outliers(self) -> None:
        self.logger.info("=== OUTLIER DETECTION ===")

        # Extreme subscription values
        payment_df = self.df[self.df["mode"].isin(["payment", "subscription"])]
        valid_sv = payment_df["subscription_value"].dropna()

        if len(valid_sv) > 0:
            q1, q3 = valid_sv.quantile(0.25), valid_sv.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + 3 * iqr
            outliers = (valid_sv > upper).sum()
            self.logger.info("Extreme subscription values (>Q3+3*IQR): %d", outliers)

        # Check for unusual product combinations
        self.logger.info("Product combination analysis:")
        prod_combos = self.df.groupby(["category", "grade"]).size().head(10)
        self.logger.info("Top category-grade combinations:\n%s", prod_combos)

    def generate_plots(self) -> None:

        self.logger.info("=== Generating Plotly Visualizations ===")

        self.plot_checkout_funnel()
        self.plot_mode_distribution()
        self.plot_status_by_mode()
        self.plot_subscription_value()
        self.plot_category_distribution()
        self.plot_store_performance()
        self.plot_monthly_trend()

    def plot_checkout_funnel(self) -> None:
        status_order = ["created", "pending", "complete", "failed"]
        status_counts = (
            self.df["status"].value_counts().reindex(status_order).dropna().reset_index()
        )
        status_counts.columns = ["status", "count"]

        fig = create_funnel(status_counts, "count", "status", "Checkout Funnel")
        self.plot_saver.save(fig, "checkout_funnel")

    def plot_mode_distribution(self) -> None:
        mode_counts = self.df["mode"].value_counts().reset_index()
        mode_counts.columns = ["mode", "count"]

        fig = create_pie_chart(mode_counts, "mode", "count", "Mode Distribution", hole=True)
        self.plot_saver.save(fig, "mode_distribution")

    def plot_status_by_mode(self) -> None:
        status_mode = self.df.groupby(["mode", "status"]).size().reset_index(name="count")

        import plotly.express as px

        fig = px.bar(
            status_mode,
            x="mode",
            y="count",
            color="status",
            title="Status by Mode",
            barmode="stack",
        )
        fig.update_layout(
            template="plotly_white", title_font=dict(size=16, color=PURPLE_THEME["primary_dark"])
        )
        self.plot_saver.save(fig, "status_by_mode")

    def plot_subscription_value(self) -> None:
        valid_sv = self.df[self.df["subscription_value"].notna()]["subscription_value"]
        fig = create_histogram(
            valid_sv.to_frame(),
            "subscription_value",
            title="Subscription Value Distribution",
            bins=30,
        )
        self.plot_saver.save(fig, "subscription_value")

    def plot_category_distribution(self) -> None:
        cat_counts = self.df["category"].value_counts().head(15).reset_index()
        cat_counts.columns = ["category", "count"]

        fig = create_bar_chart(
            cat_counts,
            "category",
            "count",
            "Category Distribution",
            color=PURPLE_THEME["primary_dark"],
        )
        self.plot_saver.save(fig, "category_distribution")

    def plot_store_performance(self) -> None:
        if self.stores_ref is not None:
            store_stats = (
                self.df.groupby("store_id")
                .agg({"id": "count", "status": lambda x: (x == "complete").mean()})
                .reset_index()
            )
            store_stats.columns = ["store_id", "total_checkouts", "completion_rate"]
            store_stats = store_stats[store_stats["total_checkouts"] >= 10].sort_values(
                "completion_rate"
            )

            fig = create_horizontal_bar(
                store_stats.tail(15),
                "completion_rate",
                "store_id",
                "Store Completion Rate",
                color=PURPLE_THEME["secondary"],
            )
            self.plot_saver.save(fig, "store_performance")

    def plot_monthly_trend(self) -> None:
        self.df["created"] = pd.to_datetime(self.df["created"])
        monthly = self.df.set_index("created").resample("ME").agg({"id": "count"}).reset_index()
        monthly.columns = ["month", "total_checkouts"]

        fig = create_line_chart(monthly, "month", "total_checkouts", "Monthly Checkout Trend")
        self.plot_saver.save(fig, "monthly_trend")


def main():
    logger.info("Starting checkouts audit analysis")
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    stores = pd.read_csv(DATA_PATH / "stores.csv")
    df = DataLoader.load_checkouts()
    analyzer = CheckoutsAnalyzer(df, logger, customers_ref=customers, stores_ref=stores)
    analyzer.run_full_analysis()
    logger.info("Checkouts audit complete")


if __name__ == "__main__":
    main()
