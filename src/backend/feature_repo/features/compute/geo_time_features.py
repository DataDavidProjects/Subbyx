"""
Geo-velocity and temporal feature computation.

Computes multi-window rolling geo-velocity features (PIT-correct)
and checkout temporal features. All windows are strictly exclusive
(look-back only, current event excluded).

Hierarchical Bayesian smoothing (empirical Bayes):
  Sparse local fraud rates are shrunk toward the parent geographic level
  to avoid zero-variance features for regions with few observations.

    smoothed_rate = (n_local + prior_weight * parent_rate)
                    / (n_local_requests + prior_weight)

  Hierarchy:  postal → province → national

Feature windows:
  Province-level:  5d, 10d, 30d, 60d
  Postal code:     5d, 10d, 30d

Temporal features (derived directly from checkout timestamp, no look-back):
  checkout_hour, checkout_dow, is_weekend, is_late_night
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path("/Users/davidelupis/Desktop/Subbyx")
_DATA_CLEAN = _REPO_ROOT / "data" / "01-clean"
_OUTPUT_DIR = _REPO_ROOT / "src" / "backend" / "feature_repo" / "data" / "sources"

# ---------------------------------------------------------------------------
# Window configuration
# ---------------------------------------------------------------------------
PROVINCE_WINDOWS = [5, 10, 30, 60]
POSTAL_WINDOWS = [5, 10, 30]

# Bayesian shrinkage weight — roughly how many local observations are needed
# before the local rate dominates the parent prior.  10 is a standard choice
# (equivalent to a Beta(prior_weight*p, prior_weight*(1-p)) prior).
PRIOR_WEIGHT = 10


def _compute_national_rolling(
    df: pd.DataFrame,
    label_col: str,
    window_days: int,
) -> np.ndarray:
    """PIT-correct national (global) fraud rate for each row.

    Returns an array of length ``len(df)`` aligned to the **sorted**
    DataFrame index (caller must sort by ``created`` first or pass an
    already-sorted frame).
    """
    times = df["created"].values
    labels = np.nan_to_num(df[label_col].values.astype(float), nan=0.0)
    cum_frauds = np.concatenate([[0.0], np.cumsum(labels)])
    window_ns = np.timedelta64(window_days, "D")

    n = len(df)
    national_rate = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lo = np.searchsorted(times[:i], times[i] - window_ns, side="left")
        n_req = i - lo
        if n_req > 0:
            national_rate[i] = (cum_frauds[i] - cum_frauds[lo]) / n_req

    return national_rate


def _compute_rolling_geo(
    df: pd.DataFrame,
    geo_col: str,
    label_col: str,
    window_days: int,
    prefix: str,
    parent_rates: np.ndarray | None = None,
    prior_weight: float = PRIOR_WEIGHT,
) -> pd.DataFrame:
    """
    For every row, count requests and fraud events from the same
    geographic region within the PRIOR `window_days` days.

    Uses a binary-search approach (O(n log n)) rather than O(n²) to
    scale to larger datasets.

    When *parent_rates* is provided (an array aligned to the sorted df),
    the raw fraud rate is replaced with a Bayesian-smoothed estimate::

        smoothed = (n_frauds + prior_weight * parent_rate)
                   / (n_requests + prior_weight)

    This shrinks sparse local estimates toward the parent geographic level,
    eliminating the zero-variance problem for regions with few observations.

    Returns columns:
      {prefix}_n_requests_{window_days}d
      {prefix}_n_frauds_{window_days}d
      {prefix}_fraud_rate_{window_days}d
    """
    req_col = f"{prefix}_n_requests_{window_days}d"
    fraud_col = f"{prefix}_n_frauds_{window_days}d"
    rate_col = f"{prefix}_fraud_rate_{window_days}d"

    # Preserve original index so parent_rates (aligned to caller's frame)
    # stay in sync after sorting.
    df = df.copy()
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values("created").reset_index(drop=True)
    sort_order = df["_orig_pos"].values  # maps sorted pos → original pos

    window_ns = np.timedelta64(window_days, "D")

    n_requests = np.zeros(len(df), dtype=np.float32)
    n_frauds = np.zeros(len(df), dtype=np.float32)

    for region, group in df.groupby(geo_col, sort=False):
        if not region or region in ("nan", ""):
            continue
        times = group["created"].values
        labels = np.nan_to_num(group[label_col].values.astype(float), nan=0.0)
        idxs = group.index.values

        # Cumulative fraud count for fast prefix-sum look-ups
        cum_frauds = np.concatenate([[0.0], np.cumsum(labels)])

        for i, (ts, idx) in enumerate(zip(times, idxs)):
            lo = np.searchsorted(times[:i], ts - window_ns, side="left")
            n_req = i - lo
            n_frd = cum_frauds[i] - cum_frauds[lo]
            n_requests[idx] = n_req
            n_frauds[idx] = n_frd

    if parent_rates is not None:
        # Re-order parent_rates (aligned to caller's index) to match
        # the sorted order used inside this function.
        parent_sorted = np.asarray(parent_rates, dtype=np.float64)[sort_order]

        # Bayesian shrinkage: blend local rate with parent-level prior.
        # When n_requests is large the local rate dominates; when small
        # (or zero) we fall back to the parent rate.
        fraud_rate = (n_frauds + prior_weight * parent_sorted) / (
            n_requests + prior_weight
        )
    else:
        fraud_rate = np.where(n_requests > 0, n_frauds / n_requests, 0.0)

    return pd.DataFrame(
        {req_col: n_requests, fraud_col: n_frauds, rate_col: fraud_rate},
        index=df.index,
    )


def generate() -> None:
    logger.info("Generating geo-velocity and temporal features...")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkouts_csv = _DATA_CLEAN / "checkouts.csv"
    customers_csv = _DATA_CLEAN / "customers.csv"
    addresses_csv = _DATA_CLEAN / "addresses.csv"

    for p in (checkouts_csv, customers_csv, addresses_csv):
        if not p.exists():
            logger.error("Source file not found: %s", p)
            return

    logger.info("Loading source files...")
    checkouts = pd.read_csv(checkouts_csv)
    customers = pd.read_csv(customers_csv)
    addresses = pd.read_csv(addresses_csv)

    checkouts["created"] = pd.to_datetime(checkouts["created"], utc=True)
    customers["created"] = pd.to_datetime(customers["created"], utc=True)

    # ------------------------------------------------------------------
    # 1. Fraud label
    # ------------------------------------------------------------------
    threshold_days = 15
    customers["is_fraud"] = (customers["dunning_days"].fillna(0) > threshold_days).astype(int)

    checkouts = checkouts.rename(columns={"customer": "customer_id"})
    checkouts = checkouts.merge(
        customers[["id", "email", "is_fraud", "residential_address_id"]].rename(
            columns={"id": "customer_id"}
        ),
        on="customer_id",
        how="left",
    )

    addresses["province"] = addresses["state"].fillna("").astype(str).str.strip()
    addresses["postal_code"] = addresses["postal_code"].astype(str).str.strip()
    checkouts = checkouts.merge(
        addresses[["id", "province", "postal_code"]].rename(
            columns={"id": "residential_address_id"}
        ),
        on="residential_address_id",
        how="left",
    )
    checkouts["province_clean"] = checkouts["province"].fillna("").str.strip()
    checkouts["postal_clean"] = checkouts["postal_code"].fillna("").str.strip()

    # ------------------------------------------------------------------
    # 2. Temporal features (no look-back—just derived from the timestamp)
    # ------------------------------------------------------------------
    logger.info("Computing temporal features...")
    checkouts["checkout_hour"] = checkouts["created"].dt.hour
    checkouts["checkout_dow"] = checkouts["created"].dt.dayofweek
    checkouts["is_weekend"] = (checkouts["checkout_dow"] >= 5).astype(int)
    checkouts["is_late_night"] = (
        (checkouts["checkout_hour"] >= 22) | (checkouts["checkout_hour"] < 6)
    ).astype(int)

    # ------------------------------------------------------------------
    # 3. National-level rolling prior (used as parent for province)
    # ------------------------------------------------------------------
    # Sort once — national and province computations share this order.
    checkouts = checkouts.sort_values("created").reset_index(drop=True)

    all_windows = sorted(set(PROVINCE_WINDOWS) | set(POSTAL_WINDOWS))
    national_rates: dict[int, np.ndarray] = {}
    for w in all_windows:
        logger.info("Computing national rolling prior (%dd)...", w)
        national_rates[w] = _compute_national_rolling(
            checkouts, label_col="is_fraud", window_days=w,
        )

    # ------------------------------------------------------------------
    # 4. Province-level rolling windows (smoothed with national prior)
    # ------------------------------------------------------------------
    for w in PROVINCE_WINDOWS:
        logger.info("Computing province rolling features (%dd, smoothed)...", w)
        cols = _compute_rolling_geo(
            df=checkouts[["created", "province_clean", "is_fraud"]].rename(
                columns={"province_clean": "geo"}
            ),
            geo_col="geo",
            label_col="is_fraud",
            window_days=w,
            prefix="province",
            parent_rates=national_rates[w],
        )
        checkouts = checkouts.join(cols)

    # ------------------------------------------------------------------
    # 5. Postal-code rolling windows (smoothed with province prior)
    # ------------------------------------------------------------------
    for w in POSTAL_WINDOWS:
        logger.info("Computing postal rolling features (%dd, smoothed)...", w)
        # Use the (already-smoothed) province fraud rate as parent prior
        province_rate_col = f"province_fraud_rate_{w}d"
        if province_rate_col in checkouts.columns:
            parent = checkouts[province_rate_col].values
        else:
            # Fallback to national if province window not available
            parent = national_rates[w]

        cols = _compute_rolling_geo(
            df=checkouts[["created", "postal_clean", "is_fraud"]].rename(
                columns={"postal_clean": "geo"}
            ),
            geo_col="geo",
            label_col="is_fraud",
            window_days=w,
            prefix="postal",
            parent_rates=parent,
        )
        checkouts = checkouts.join(cols)

    # ------------------------------------------------------------------
    # 6. Feast output (entity = email, timestamp = created)
    # ------------------------------------------------------------------
    temporal_cols = ["checkout_hour", "checkout_dow", "is_weekend", "is_late_night"]
    province_cols = [
        f"province_{stat}_{w}d"
        for w in PROVINCE_WINDOWS
        for stat in ["n_requests", "n_frauds", "fraud_rate"]
    ]
    postal_cols = [
        f"postal_{stat}_{w}d"
        for w in POSTAL_WINDOWS
        for stat in ["n_requests", "n_frauds", "fraud_rate"]
    ]

    output_cols = ["email", "created"] + temporal_cols + province_cols + postal_cols
    output = checkouts[output_cols].copy()
    output = output.dropna(subset=["email"])
    output["email"] = output["email"].astype(str)
    for col in output.columns:
        if col not in ("email", "created"):
            output[col] = output[col].astype(float)

    out_path = _OUTPUT_DIR / "geo_time_features.parquet"
    output.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(output), out_path)

    # Summary
    windows_summary = {
        w: checkouts[f"province_fraud_rate_{w}d"].describe() for w in PROVINCE_WINDOWS
    }
    for w, s in windows_summary.items():
        logger.info("province_fraud_rate_%dd — mean=%.3f, max=%.3f", w, s["mean"], s["max"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
