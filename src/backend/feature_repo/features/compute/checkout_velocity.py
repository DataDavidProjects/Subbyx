from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path("/Users/davidelupis/Desktop/Subbyx")
_SOURCES_DIR = _REPO_ROOT / "src" / "backend" / "feature_repo" / "data" / "sources"


def generate() -> None:
    logger.info("Generating checkout velocity features...")
    _SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    src_path = _SOURCES_DIR / "checkouts.parquet"
    if not src_path.exists():
        logger.error("Source file not found: %s", src_path)
        return

    df = pd.read_parquet(src_path)
    df["created"] = pd.to_datetime(df["created"], utc=True)
    df = df.sort_values(["email", "created"]).reset_index(drop=True)

    n = len(df)
    n_checkouts_7d = np.zeros(n, dtype=np.float64)
    n_expired_7d = np.zeros(n, dtype=np.float64)
    expired_ratio_7d = np.zeros(n, dtype=np.float64)
    n_distinct_categories_30d = np.zeros(n, dtype=np.float64)
    max_value_30d = np.zeros(n, dtype=np.float64)

    window_7d = np.timedelta64(7, "D")
    window_30d = np.timedelta64(30, "D")

    for _email, group in df.groupby("email", sort=False):
        idxs = group.index.values
        times = group["created"].values
        statuses = group["status"].values.astype(str)
        categories = group["category"].values.astype(str)
        values = group["subscription_value"].values.astype(np.float64)

        is_expired = (statuses == "expired").astype(np.float64)
        cum_expired = np.concatenate([[0.0], np.cumsum(is_expired)])

        for i in range(len(group)):
            idx = idxs[i]
            ts = times[i]

            # --- 7-day window (current event excluded: times[:i]) ---
            lo_7 = np.searchsorted(times[:i], ts - window_7d, side="left")
            n_ck = i - lo_7
            n_checkouts_7d[idx] = n_ck
            n_exp = cum_expired[i] - cum_expired[lo_7]
            n_expired_7d[idx] = n_exp
            expired_ratio_7d[idx] = n_exp / n_ck if n_ck > 0 else 0.0

            # --- 30-day window (current event excluded) ---
            lo_30 = np.searchsorted(times[:i], ts - window_30d, side="left")
            if i > lo_30:
                n_distinct_categories_30d[idx] = len(set(categories[lo_30:i]))
                max_value_30d[idx] = np.nanmax(values[lo_30:i])

    output = pd.DataFrame(
        {
            "email": df["email"],
            "created": df["created"],
            "n_checkouts_7d": n_checkouts_7d,
            "n_expired_7d": n_expired_7d,
            "expired_ratio_7d": expired_ratio_7d,
            "n_distinct_categories_30d": n_distinct_categories_30d,
            "max_value_30d": max_value_30d,
        }
    )
    output = output.dropna(subset=["email"])
    output["email"] = output["email"].astype(str)

    out_path = _SOURCES_DIR / "checkout_velocity.parquet"
    output.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(output), out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
