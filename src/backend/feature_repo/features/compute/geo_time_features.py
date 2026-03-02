"""
Geo-velocity and temporal feature computation.

Computes multi-window rolling geo-velocity features (PIT-correct)
and checkout temporal features. All windows are strictly exclusive
(look-back only, current event excluded).

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


def _compute_rolling_geo(
    df: pd.DataFrame,
    geo_col: str,
    label_col: str,
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    """
    For every row, count requests and fraud events from the same
    geographic region within the PRIOR `window_days` days.

    Uses a binary-search approach (O(n log n)) rather than O(n²) to
    scale to larger datasets.

    Returns columns:
      {prefix}_n_requests_{window_days}d
      {prefix}_n_frauds_{window_days}d
      {prefix}_fraud_rate_{window_days}d
    """
    req_col = f"{prefix}_n_requests_{window_days}d"
    fraud_col = f"{prefix}_n_frauds_{window_days}d"
    rate_col = f"{prefix}_fraud_rate_{window_days}d"

    df = df.sort_values("created").reset_index(drop=True)
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
    # 3. Province-level rolling windows
    # ------------------------------------------------------------------
    for w in PROVINCE_WINDOWS:
        logger.info("Computing province rolling features (%dd)...", w)
        cols = _compute_rolling_geo(
            df=checkouts[["created", "province_clean", "is_fraud"]].rename(
                columns={"province_clean": "geo"}
            ),
            geo_col="geo",
            label_col="is_fraud",
            window_days=w,
            prefix="province",
        )
        checkouts = checkouts.join(cols)

    # ------------------------------------------------------------------
    # 4. Postal-code rolling windows
    # ------------------------------------------------------------------
    for w in POSTAL_WINDOWS:
        logger.info("Computing postal rolling features (%dd)...", w)
        cols = _compute_rolling_geo(
            df=checkouts[["created", "postal_clean", "is_fraud"]].rename(
                columns={"postal_clean": "geo"}
            ),
            geo_col="geo",
            label_col="is_fraud",
            window_days=w,
            prefix="postal",
        )
        checkouts = checkouts.join(cols)

    # ------------------------------------------------------------------
    # 5. Feast output (entity = email, timestamp = created)
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
