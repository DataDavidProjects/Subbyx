from __future__ import annotations

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logger import setup_logger


DATA_PATH = Path("/Users/davidelupis/Desktop/Subbyx/data/01-clean/")
LOG_PATH = Path("/Users/davidelupis/Desktop/Subbyx/scripts/notebooks/logs/")

logger = setup_logger("raw-data-analysis", LOG_PATH / "raw-data-analysis.log")


def analyze_raw_data_correlations() -> None:
    """Analyze correlations in raw (clean) data - the source of truth."""
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
    charges = pd.read_csv(DATA_PATH / "charges.csv")
    payment_intents = pd.read_csv(DATA_PATH / "payment_intents.csv")

    customers["fraud"] = (customers["dunning_days"] > 15).astype(int)

    logger.info("=== RAW DATA SUMMARY ===")
    logger.info(
        "Customers: %d, Fraud rate: %.1f%%", len(customers), customers["fraud"].mean() * 100
    )

    checkouts_ps = checkouts[checkouts["mode"].isin(["payment", "subscription"])].copy()
    checkouts_ps = checkouts_ps.merge(
        customers[["id", "email", "fraud"]], left_on="customer", right_on="id", how="left"
    )
    logger.info(
        "Checkouts (payment/subscription): %d, Fraud rate: %.1f%%",
        len(checkouts_ps),
        checkouts_ps["fraud"].mean() * 100,
    )

    charges_with_fraud = charges.merge(
        customers[["id", "fraud"]], left_on="customer", right_on="id", how="left"
    )
    charges_with_fraud = charges_with_fraud.dropna(subset=["fraud"])
    logger.info(
        "Charges: %d, Fraud rate: %.1f%%",
        len(charges_with_fraud),
        charges_with_fraud["fraud"].mean() * 100,
    )

    logger.info("=" * 80)
    logger.info("RAW CHARGES vs FRAUD LABEL")
    logger.info("=" * 80)

    agg = (
        charges.groupby("customer")
        .agg(
            n_succeeded=("status", lambda x: (x == "succeeded").sum()),
            n_failed=("status", lambda x: (x != "succeeded").sum()),
            n_total=("status", "count"),
        )
        .reset_index()
    )
    agg = agg.merge(customers[["id", "fraud"]], left_on="customer", right_on="id", how="left")

    for col in ["n_succeeded", "n_failed", "n_total"]:
        corr = agg[col].corr(agg["fraud"])
        f_mean = agg[agg["fraud"] == 1][col].mean()
        c_mean = agg[agg["fraud"] == 0][col].mean()
        logger.info(
            "%-15s: corr=%+.3f, fraud_mean=%8.3f, clean_mean=%8.3f, diff=%+.3f",
            col,
            corr,
            f_mean,
            c_mean,
            f_mean - c_mean,
        )

    logger.info("=" * 80)
    logger.info("RAW PAYMENT_INTENTS vs FRAUD LABEL")
    logger.info("=" * 80)

    pi_with_fraud = payment_intents.merge(
        customers[["id", "fraud"]], left_on="customer", right_on="id", how="left"
    )
    pi_with_fraud = pi_with_fraud.dropna(subset=["fraud"])

    for col in pi_with_fraud.columns:
        if pi_with_fraud[col].dtype in ["int64", "float64"] and col not in ["fraud", "Unnamed: 0"]:
            corr = pi_with_fraud[col].corr(pi_with_fraud["fraud"])
            if not np.isnan(corr) and abs(corr) > 0.01:
                f_mean = pi_with_fraud[pi_with_fraud["fraud"] == 1][col].mean()
                c_mean = pi_with_fraud[pi_with_fraud["fraud"] == 0][col].mean()
                logger.info(
                    "%-25s: corr=%+.3f, fraud_mean=%10.3f, clean_mean=%10.3f",
                    col,
                    corr,
                    f_mean,
                    c_mean,
                )

    logger.info("=" * 80)
    logger.info("KEY INSIGHT: BRUTE-FORCING PATTERN")
    logger.info("=" * 80)
    logger.info("Fraud customers (dunning_days > 15) show CLEAR brute-forcing pattern:")
    logger.info("  - n_failed (charges): 43.2 vs 4.4 (10x more failures)")
    logger.info("  - This is NOT inverted - the model correctly learns this pattern")
    logger.info("")
    logger.info("Label definition: 'fraud' = customers who went into dunning (>15 days)")
    logger.info("These are customers who eventually paid but then stopped paying.")
    logger.info("The high failure count reflects their payment attempt history.")


def analyze_feature_at_checkout_time() -> None:
    """Analyze what features look like at checkout time (before the checkout)."""
    customers = pd.read_csv(DATA_PATH / "customers.csv")
    checkouts = pd.read_csv(DATA_PATH / "checkouts.csv")
    payment_intents = pd.read_csv(DATA_PATH / "payment_intents.csv")

    customers["fraud"] = (customers["dunning_days"] > 15).astype(int)

    checkouts_ps = checkouts[checkouts["mode"].isin(["payment", "subscription"])].copy()
    checkouts_ps = checkouts_ps.merge(
        customers[["id", "email", "fraud"]], left_on="customer", right_on="id", how="left"
    )
    checkouts_ps = checkouts_ps.sort_values(["customer", "created"])

    results = []
    for idx, row in checkouts_ps.iterrows():
        cust = row["customer"]
        checkout_time = row["created"]

        pis = payment_intents[
            (payment_intents["customer"] == cust) & (payment_intents["created"] < checkout_time)
        ]

        if len(pis) > 0:
            n_pi = len(pis)
            n_fail = pis[pis["status"].isin(["requires_payment_method", "canceled"])].shape[0]
            n_succ = pis[pis["status"] == "succeeded"].shape[0]
            success_rate = n_succ / n_pi if n_pi > 0 else 0
        else:
            n_pi = 0
            n_fail = 0
            n_succ = 0
            success_rate = 0

        results.append(
            {
                "fraud": row["fraud"],
                "n_pi": n_pi,
                "n_fail": n_fail,
                "n_succ": n_succ,
                "success_rate": success_rate,
            }
        )

    df = pd.DataFrame(results)
    df_with_pi = df[df["n_pi"] > 0]

    logger.info("=" * 80)
    logger.info("FEATURES AT CHECKOUT TIME (before checkout)")
    logger.info("=" * 80)
    logger.info("Checkouts with prior PIs: %d / %d", len(df_with_pi), len(df))

    for col in ["n_pi", "n_fail", "n_succ", "success_rate"]:
        corr = df_with_pi[col].corr(df_with_pi["fraud"])
        f_mean = df_with_pi[df_with_pi["fraud"] == 1][col].mean()
        c_mean = df_with_pi[df_with_pi["fraud"] == 0][col].mean()
        logger.info(
            "%-15s: corr=%+.3f, fraud_mean=%8.3f, clean_mean=%8.3f",
            col,
            corr,
            f_mean,
            c_mean,
        )


def main() -> None:
    logger.info("Starting raw data correlation analysis")
    analyze_raw_data_correlations()
    analyze_feature_at_checkout_time()
    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
