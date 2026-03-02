import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating base payment intent features...")

    repo_root = Path("/Users/davidelupis/Desktop/Subbyx")
    pi_csv = repo_root / "data" / "01-clean" / "payment_intents.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"
    output_parquet = output_dir / "payment_intents.parquet"

    if not pi_csv.exists():
        logger.error("Source file not found: %s", pi_csv)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", pi_csv)
    df = pd.read_csv(pi_csv)

    customers_csv = repo_root / "data" / "01-clean" / "customers.csv"
    logger.info("Reading %s for join...", customers_csv)
    customers_df = pd.read_csv(customers_csv)

    logger.info("Merging payment intents with customers to resolve emails...")
    df = df.merge(
        customers_df[["id", "email"]],
        left_on="customer",
        right_on="id",
        how="left",
        suffixes=("", "_cust"),
    )

    n_missing = df["email"].isnull().sum()
    if n_missing > 0:
        logger.warning("Dropping %d rows with missing email", n_missing)
        df = df.dropna(subset=["email"])

    logger.info("Converting timestamps...")
    df["created"] = pd.to_datetime(df["created"])
    if df["created"].dt.tz is None:
        df["created"] = df["created"].dt.tz_localize("UTC")
    else:
        df["created"] = df["created"].dt.tz_convert("UTC")

    string_cols = ["id", "status", "payment_intent", "customer", "email", "latest_charge"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "")

    logger.info("Saving to %s", output_parquet)
    df.to_parquet(output_parquet, index=False)

    logger.info("Base payment intent features generated: %d rows", len(df))

    generate_stats(df)


def generate_stats(df: pd.DataFrame) -> None:
    logger.info("Generating payment intent stats features...")

    output_dir = Path("/Users/davidelupis/Desktop/Subbyx/src/backend/feature_repo/data/sources")
    stats_parquet = output_dir / "payment_intent_stats.parquet"

    df = df.sort_values(by=["email", "created"]).reset_index(drop=True)

    df["is_failure"] = df["status"].isin(["requires_payment_method", "canceled"]).astype(int)
    df["is_success"] = (df["status"] == "succeeded").astype(int)

    grouped = df.groupby("email")

    df["n_payment_intents"] = grouped.cumcount() + 1
    df["n_failures"] = grouped["is_failure"].cumsum()
    df["failure_rate"] = df["n_failures"] / df["n_payment_intents"]
    df["n_succeeded"] = grouped["is_success"].cumsum()
    df["success_rate"] = df["n_succeeded"] / df["n_payment_intents"]

    stats_df = df[
        [
            "email",
            "created",
            "n_payment_intents",
            "n_failures",
            "failure_rate",
            "n_succeeded",
            "success_rate",
        ]
    ]

    logger.info("Saving payment intent stats to %s", stats_parquet)
    stats_df.to_parquet(stats_parquet, index=False)

    logger.info("Payment intent stats generated successfully.")
    logger.info("  n_payment_intents: cumulative count")
    logger.info("  n_failures: cumulative failed attempts")
    logger.info("  failure_rate: n_failures / n_payment_intents")
    logger.info("  n_succeeded: cumulative succeeded payments")
    logger.info("  success_rate: n_succeeded / n_payment_intents")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
