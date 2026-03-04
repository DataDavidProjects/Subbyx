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
    create_line_chart,
    create_treemap,
    save_html,
)


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "01-clean"
LOG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "notebooks" / "logs"

logger = setup_logger("customers-audit", LOG_PATH / "customers-audit.log")


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
    def load_customers() -> pd.DataFrame:
        return pd.read_csv(DATA_PATH / "customers.csv")


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

    def get_dtypes(self) -> pd.Series:
        return self.df.dtypes

    def generate_report(self) -> None:
        self.logger.info("=== %s Data Quality Report ===", self.table_name)
        self.logger.info("Shape: %s", self.get_shape())
        self.logger.info("Columns: %s", self.get_columns())
        self.logger.info("Duplicates: %s", self.get_duplicates())
        self.logger.info("Null counts:\n%s", self.get_null_counts())
        self.logger.info("Data types:\n%s", self.get_dtypes())


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
        fig = create_histogram(
            self.df, column, bins=bins, add_fraud_line=(column == "dunning_days")
        )
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


class CustomersAnalyzer:
    def __init__(self, df: pd.DataFrame, logger, customers_ref=None):
        self.df = df
        self.logger = logger
        self.customers_ref = customers_ref
        self.plot_saver = PlotSaver("customers")
        self.quality = DataQualityChecker(df, "customers", logger)
        self.numeric = NumericAnalyzer(df, logger, self.plot_saver)
        self.categorical = CategoricalAnalyzer(df, logger, self.plot_saver)

    def run_full_analysis(self) -> None:
        self.quality.generate_report()

        self.logger.info("=== Numeric Analysis ===")
        self.logger.info("\n%s", self.numeric.describe_numeric())

        self.logger.info("=== Key Feature Analysis ===")
        self.analyze_dunning_days()
        self.analyze_match_scores()
        self.analyze_high_end()
        self.analyze_email_duplicates()
        self.analyze_gender_distribution()
        self.analyze_birth_data_quality()
        self.analyze_address_linkage()
        self.analyze_temporal_patterns()

        # Deep fraud analysis
        self.analyze_fraud_thresholds()
        self.analyze_fraud_correlations()
        self.analyze_email_domain_fraud()
        self.analyze_birth_country_fraud()
        self.analyze_address_fraud_patterns()
        self.analyze_identity_fields()
        self.analyze_outliers()

    def analyze_dunning_days(self) -> None:
        self.logger.info("--- dunning_days (Target Variable) ---")
        self.logger.info("\n%s", self.numeric.describe_numeric()["dunning_days"])
        self.numeric.plot_distribution("dunning_days", bins=50)

    def analyze_match_scores(self) -> None:
        score_cols = [c for c in self.df.columns if "match_score" in c]
        self.logger.info("--- Match Scores ---")
        for col in score_cols:
            self.logger.info("%s:\n%s", col, self.df[col].describe())

    def analyze_high_end(self) -> None:
        self.logger.info("--- High End Features ---")
        self.logger.info("high_end_count:\n%s", self.df["high_end_count"].describe())
        self.logger.info("high_end_rate:\n%s", self.df["high_end_rate"].describe())

    def analyze_email_duplicates(self) -> None:
        self.logger.info("=== Email Duplicate Analysis ===")
        email_dupes = self.df[self.df.duplicated(subset=["email"], keep=False)]
        unique_emails = self.df["email"].nunique()
        total_customers = len(self.df)
        self.logger.info(
            "Total customers: %d, Unique emails: %d, Customers with duplicate emails: %d",
            total_customers,
            unique_emails,
            len(email_dupes),
        )
        self.logger.info(
            "Duplicate email rate: %.2f%%",
            (len(email_dupes) / total_customers) * 100,
        )
        email_counts = self.df.groupby("email").size()
        multi_customer_emails = email_counts[email_counts > 1]
        self.logger.info("Emails with multiple customers: %d", len(multi_customer_emails))
        if len(multi_customer_emails) > 0:
            self.logger.info("Max customers per email: %d", multi_customer_emails.max())

    def analyze_gender_distribution(self) -> None:
        self.logger.info("=== Gender Distribution ===")
        gender_counts = self.df["gender"].value_counts(dropna=False)
        self.logger.info("\n%s", gender_counts)
        null_count = self.df["gender"].isna().sum()
        self.logger.info(
            "Missing gender: %d (%.2f%%)", null_count, (null_count / len(self.df)) * 100
        )

    def analyze_birth_data_quality(self) -> None:
        self.logger.info("=== Birth Data Quality ===")
        null_birth_date = self.df["birth_date"].isna().sum()
        null_birth_province = self.df["birth_province"].isna().sum()
        null_birth_country = self.df["birth_country"].isna().sum()
        null_birth_continent = self.df["birth_continent"].isna().sum()
        self.logger.info(
            "Missing birth_date: %d (%.2f%%)",
            null_birth_date,
            (null_birth_date / len(self.df)) * 100,
        )
        self.logger.info(
            "Missing birth_province: %d (%.2f%%)",
            null_birth_province,
            (null_birth_province / len(self.df)) * 100,
        )
        self.logger.info(
            "Missing birth_country: %d (%.2f%%)",
            null_birth_country,
            (null_birth_country / len(self.df)) * 100,
        )
        self.logger.info(
            "Missing birth_continent: %d (%.2f%%)",
            null_birth_continent,
            (null_birth_continent / len(self.df)) * 100,
        )
        self.logger.info(
            "Birth country distribution:\n%s",
            self.df["birth_country"].value_counts(dropna=False).head(10),
        )
        self.logger.info(
            "Birth continent distribution:\n%s",
            self.df["birth_continent"].value_counts(dropna=False),
        )

    def analyze_address_linkage(self) -> None:
        self.logger.info("=== Address Linkage Analysis ===")
        total = len(self.df)
        residential_null = self.df["residential_address_id"].isna().sum()
        shipping_null = self.df["shipping_address_id"].isna().sum()
        self.logger.info(
            "Residential address linked: %d/%d (%.2f%%)",
            total - residential_null,
            total,
            ((total - residential_null) / total) * 100,
        )
        self.logger.info(
            "Shipping address linked: %d/%d (%.2f%%)",
            total - shipping_null,
            total,
            ((total - shipping_null) / total) * 100,
        )
        both_linked = (
            (~self.df["residential_address_id"].isna()) & (~self.df["shipping_address_id"].isna())
        ).sum()
        self.logger.info(
            "Both addresses linked: %d (%.2f%%)", both_linked, (both_linked / total) * 100
        )
        self.logger.info(
            "Only residential: %d",
            (
                ~self.df["residential_address_id"].isna() & self.df["shipping_address_id"].isna()
            ).sum(),
        )
        self.logger.info(
            "Only shipping: %d",
            (
                self.df["residential_address_id"].isna() & ~self.df["shipping_address_id"].isna()
            ).sum(),
        )

    def analyze_temporal_patterns(self) -> None:
        self.logger.info("=== Temporal Patterns ===")
        self.df["created"] = pd.to_datetime(self.df["created"])
        self.df["created_hour"] = self.df["created"].dt.hour
        self.df["created_dayofweek"] = self.df["created"].dt.dayofweek
        self.df["created_month"] = self.df["created"].dt.month
        self.df["created_year_month"] = self.df["created"].dt.to_period("M")

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

        self.logger.info(
            "Monthly creation distribution:\n%s",
            self.df["created_year_month"].value_counts().sort_index(),
        )

    def analyze_fraud_thresholds(self) -> None:
        self.logger.info("=== FRAUD THRESHOLD ANALYSIS ===")

        # Test different thresholds
        thresholds = [5, 10, 15, 20, 30, 45, 60]
        self.logger.info("Fraud rate at different dunning thresholds:")
        for t in thresholds:
            fraud_count = (self.df["dunning_days"] > t).sum()
            fraud_rate = fraud_count / len(self.df) * 100
            self.logger.info("  Threshold >%d days: %d (%.2f%%)", t, fraud_count, fraud_rate)

        # Recommended threshold analysis (>15 days is standard)
        self.df["is_fraud_15"] = self.df["dunning_days"] > 15
        self.df["is_fraud_30"] = self.df["dunning_days"] > 30

        fraud_15 = self.df["is_fraud_15"].sum()
        fraud_30 = self.df["is_fraud_30"].sum()

        self.logger.info("RECOMMENDED: Using >15 days threshold")
        self.logger.info(
            "  Fraudulent customers: %d (%.2f%%)", fraud_15, fraud_15 / len(self.df) * 100
        )
        self.logger.info(
            "  Non-fraudulent customers: %d (%.2f%%)",
            len(self.df) - fraud_15,
            (len(self.df) - fraud_15) / len(self.df) * 100,
        )

        # High-risk segment (>30 days)
        self.logger.info("HIGH-RISK SEGMENT (>30 days):")
        self.logger.info(
            "  High-risk customers: %d (%.2f%%)", fraud_30, fraud_30 / len(self.df) * 100
        )

    def analyze_fraud_correlations(self) -> None:
        self.logger.info("=== FEATURE CORRELATIONS WITH FRAUD ===")

        # Create fraud flag
        self.df["is_fraud"] = self.df["dunning_days"] > 15

        # Correlation analysis
        score_cols = [c for c in self.df.columns if "match_score" in c]

        self.logger.info("Match score correlations with fraud (>15 days):")
        for col in score_cols:
            valid = self.df[[col, "is_fraud"]].dropna()
            if len(valid) > 0 and valid[col].std() > 0:
                corr = valid[col].corr(valid["is_fraud"])
                self.logger.info("  %s: %.4f", col, corr)

        # High-end features correlation
        valid_he = self.df[["high_end_count", "high_end_rate", "is_fraud"]].dropna()
        if len(valid_he) > 0:
            self.logger.info("High-end feature correlations with fraud:")
            self.logger.info(
                "  high_end_count: %.4f", valid_he["high_end_count"].corr(valid_he["is_fraud"])
            )
            self.logger.info(
                "  high_end_rate: %.4f", valid_he["high_end_rate"].corr(valid_he["is_fraud"])
            )

        # Gender vs fraud
        self.logger.info("Fraud rate by gender:")
        gender_fraud = self.df.groupby("gender")["is_fraud"].mean() * 100
        self.logger.info("\n%s", gender_fraud)

        # Email duplicates vs fraud
        self.df["has_duplicate_email"] = self.df.duplicated(subset=["email"], keep=False)
        dup_fraud = self.df.groupby("has_duplicate_email")["is_fraud"].mean() * 100
        self.logger.info("Fraud rate by email duplication:")
        self.logger.info("  Unique email: %.2f%%", dup_fraud.get(False, 0))
        self.logger.info("  Duplicate email: %.2f%%", dup_fraud.get(True, 0))

    def analyze_email_domain_fraud(self) -> None:
        self.logger.info("=== EMAIL DOMAIN FRAUD ANALYSIS ===")

        self.df["is_fraud"] = self.df["dunning_days"] > 15
        self.df["email_domain"] = self.df["email"].str.split("@").str[1]

        domain_stats = (
            self.df.groupby("email_domain")
            .agg({"id": "count", "is_fraud": ["sum", "mean"]})
            .round(4)
        )
        domain_stats.columns = ["total", "fraud_count", "fraud_rate"]
        domain_stats = domain_stats.sort_values("total", ascending=False)

        self.logger.info("Top 15 domains by volume with fraud rates:")
        for idx, row in domain_stats.head(15).iterrows():
            self.logger.info(
                "  %s: %d customers, %d fraud (%.2f%%)",
                idx,
                int(row["total"]),
                int(row["fraud_count"]),
                row["fraud_rate"] * 100,
            )

        # High-risk domains (min 10 customers)
        high_risk = domain_stats[
            (domain_stats["total"] >= 10) & (domain_stats["fraud_rate"] > 0.15)
        ]
        if len(high_risk) > 0:
            self.logger.info("HIGH-RISK DOMAINS (>15% fraud, min 10 customers):")
            for idx, row in high_risk.iterrows():
                self.logger.info("  %s: %.2f%% fraud rate", idx, row["fraud_rate"] * 100)

    def analyze_birth_country_fraud(self) -> None:
        self.logger.info("=== BIRTH COUNTRY FRAUD ANALYSIS ===")

        self.df["is_fraud"] = self.df["dunning_days"] > 15

        country_stats = (
            self.df.groupby("birth_country")
            .agg({"id": "count", "is_fraud": ["sum", "mean"]})
            .round(4)
        )
        country_stats.columns = ["total", "fraud_count", "fraud_rate"]
        country_stats = country_stats.sort_values("total", ascending=False)

        self.logger.info("Top 10 birth countries with fraud rates:")
        for idx, row in country_stats.head(10).iterrows():
            if pd.notna(idx):
                self.logger.info(
                    "  %s: %d customers, %d fraud (%.2f%%)",
                    idx,
                    int(row["total"]),
                    int(row["fraud_count"]),
                    row["fraud_rate"] * 100,
                )

    def analyze_address_fraud_patterns(self) -> None:
        self.logger.info("=== ADDRESS LINKAGE FRAUD PATTERNS ===")

        self.df["is_fraud"] = self.df["dunning_days"] > 15

        # Residential address vs fraud
        has_res = self.df["residential_address_id"].notna()
        fraud_by_res = self.df.groupby(has_res)["is_fraud"].mean() * 100
        self.logger.info("Fraud rate by residential address:")
        self.logger.info("  Has residential: %.2f%%", fraud_by_res.get(True, 0))
        self.logger.info("  No residential: %.2f%%", fraud_by_res.get(False, 0))

        # Shipping address vs fraud
        has_ship = self.df["shipping_address_id"].notna()
        fraud_by_ship = self.df.groupby(has_ship)["is_fraud"].mean() * 100
        self.logger.info("Fraud rate by shipping address:")
        self.logger.info("  Has shipping: %.2f%%", fraud_by_ship.get(True, 0))
        self.logger.info("  No shipping: %.2f%%", fraud_by_ship.get(False, 0))

    def analyze_identity_fields(self) -> None:
        self.logger.info("=== IDENTITY FIELD ANALYSIS ===")

        email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        valid_emails = self.df["email"].str.match(email_pattern, na=False)
        self.logger.info(
            "Valid email format: %d/%d (%.2f%%)",
            valid_emails.sum(),
            len(self.df),
            valid_emails.sum() / len(self.df) * 100,
        )

        self.logger.info("Fiscal code analysis:")
        fc_lengths = self.df["fiscal_code"].str.len().value_counts()
        self.logger.info("Fiscal code length distribution:\n%s", fc_lengths)

        zero_dunning = (self.df["dunning_days"] == 0).sum()
        self.logger.info(
            "Customers with zero dunning days: %d (%.2f%%)",
            zero_dunning,
            zero_dunning / len(self.df) * 100,
        )

    def analyze_outliers(self) -> None:
        self.logger.info("=== OUTLIER DETECTION ===")

        extreme_dunning = (self.df["dunning_days"] > 60).sum()
        self.logger.info("Extreme dunning (>60 days): %d", extreme_dunning)

        for col in ["doc_name_email_match_score", "email_emails_match_score"]:
            if col in self.df.columns:
                valid = self.df[col].dropna()
                if len(valid) > 0:
                    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
                    iqr = q3 - q1
                    upper = q3 + 3 * iqr
                    outliers = (valid > upper).sum()
                    self.logger.info("%s outliers (>Q3+3*IQR): %d", col, outliers)

        self.generate_plots()

    def generate_plots(self) -> None:
        self.logger.info("=== Generating Plotly Visualizations ===")

        self.plot_dunning_histogram()
        self.plot_match_scores()
        self.plot_fraud_by_gender()
        self.plot_fraud_by_country()
        self.plot_email_domain_fraud()
        self.plot_monthly_trend()

    def plot_dunning_histogram(self) -> None:
        fig = create_histogram(
            self.df,
            "dunning_days",
            title="Dunning Days Distribution (Fraud Threshold: 15 days)",
            bins=50,
            add_fraud_line=True,
        )
        self.plot_saver.save(fig, "dunning_histogram")

    def plot_match_scores(self) -> None:
        score_cols = [c for c in self.df.columns if "match_score" in c]
        score_data = self.df[score_cols].melt(var_name="Score Type", value_name="Score")

        import plotly.express as px

        fig = px.box(score_data, x="Score Type", y="Score", title="Match Score Distributions")
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=12, color=PURPLE_THEME["text"]),
            title_font=dict(size=16, color=PURPLE_THEME["primary_dark"]),
        )
        self.plot_saver.save(fig, "match_scores_box")

    def plot_fraud_by_gender(self) -> None:
        self.df["is_fraud"] = self.df["dunning_days"] > 15
        gender_fraud = self.df.groupby("gender")["is_fraud"].mean().reset_index()
        gender_fraud.columns = ["gender", "fraud_rate"]
        gender_fraud["fraud_rate"] = gender_fraud["fraud_rate"] * 100

        fig = create_bar_chart(
            gender_fraud,
            "gender",
            "fraud_rate",
            "Fraud Rate by Gender",
            color=PURPLE_THEME["secondary"],
        )
        fig.update_yaxes(title="Fraud Rate (%)")
        self.plot_saver.save(fig, "fraud_by_gender")

    def plot_fraud_by_country(self) -> None:
        country_stats = (
            self.df.groupby("birth_country").agg({"id": "count", "is_fraud": "mean"}).reset_index()
        )
        country_stats.columns = ["country", "total", "fraud_rate"]
        country_stats = country_stats[country_stats["total"] >= 10].sort_values(
            "fraud_rate", ascending=True
        )
        country_stats["fraud_rate"] = country_stats["fraud_rate"] * 100

        fig = create_horizontal_bar(
            country_stats.tail(15),
            "fraud_rate",
            "country",
            "Fraud Rate by Birth Country (min 10 customers)",
            color=PURPLE_THEME["primary_dark"],
        )
        fig.update_xaxes(title="Fraud Rate (%)")
        self.plot_saver.save(fig, "fraud_by_country")

    def plot_email_domain_fraud(self) -> None:
        self.df["email_domain"] = self.df["email"].str.split("@").str[1]

        domain_stats = (
            self.df.groupby("email_domain").agg({"id": "count", "is_fraud": "mean"}).reset_index()
        )
        domain_stats.columns = ["domain", "total", "fraud_rate"]
        domain_stats = domain_stats.sort_values("total", ascending=False)

        fig = create_treemap(
            domain_stats.head(50),
            path=["domain"],
            values="total",
            title="Email Domain Volume (Treemap)",
        )
        self.plot_saver.save(fig, "email_domain_treemap")

    def plot_monthly_trend(self) -> None:
        self.df["created"] = pd.to_datetime(self.df["created"])
        monthly = (
            self.df.set_index("created")
            .resample("M")
            .agg({"id": "count", "is_fraud": "mean"})
            .reset_index()
        )
        monthly.columns = ["month", "total_customers", "fraud_rate"]
        monthly["fraud_rate"] = monthly["fraud_rate"] * 100

        fig = create_line_chart(
            monthly,
            "month",
            ["total_customers", "fraud_rate"],
            "Monthly Customer Trends",
            color=PURPLE_THEME["primary"],
        )
        fig.update_yaxes(title="Count / Rate (%)")
        self.plot_saver.save(fig, "monthly_trend")


def main():
    logger.info("Starting customers audit analysis")
    df = DataLoader.load_customers()
    analyzer = CustomersAnalyzer(df, logger)
    analyzer.run_full_analysis()
    logger.info("Customers audit complete")


if __name__ == "__main__":
    main()
