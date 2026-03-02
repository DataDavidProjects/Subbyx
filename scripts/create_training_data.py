from __future__ import annotations

from datetime import datetime
import logging
import sys
import yaml
import pandas as pd
from pathlib import Path
from feast import FeatureStore

# Add backend + feature_repo to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "backend" / "feature_repo"))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column configuration — update these when features change
# ---------------------------------------------------------------------------
LABEL_COL = "label"
LABEL_SOURCE_COL = "dunning_days"
CHECKOUT_ID_COL = "checkout_id"
TIMESTAMP_COL = "created"
CHECKOUT_JOIN_KEY = "customer"
CUSTOMER_JOIN_KEY = "id"


# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "backend" / "services" / "fraud" / "training" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "01-clean"
FEATURE_REPO_PATH = PROJECT_ROOT / "src" / "backend" / "feature_repo"

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

OUTPUT_DIR = Path(_config["data"]["training_dir"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_dates = _config["dates"]
_fraud_threshold = _config["modeling"]["fraud_threshold_days"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # -- Load raw data --
    logger.info("Loading data from %s", DATA_DIR)
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    checkouts = pd.read_csv(DATA_DIR / "checkouts.csv")

    # -- Filter to real subscriptions only --
    checkouts = checkouts[checkouts["mode"].isin(["payment", "subscription"])]
    logger.info("Filtered to real subscriptions: %d rows", len(checkouts))

    # -- Create labels --
    logger.info("Creating labels (dunning_days > %d)", _fraud_threshold)
    customers[LABEL_COL] = (customers[LABEL_SOURCE_COL] > _fraud_threshold).astype(int)

    # -- Build entity DataFrame (the spine) --
    logger.info("Building entity DataFrame (spine)")

    # Import request feature names to guarantee train/serve parity
    from services.fraud.features.request_features import REQUEST_FEATURE_SCHEMA

    # Derive has_high_end_device from high_end_count before merge
    customers["has_high_end_device"] = (
        pd.to_numeric(customers["high_end_count"], errors="coerce").fillna(0) > 0
    )

    # Merge checkouts with customers — carry request feature columns through
    customer_merge_cols = [
        "id",
        "email",
        LABEL_COL,
        "gender",
        "birth_date",
        "birth_province",
        "birth_country",
        "has_high_end_device",
    ]
    entity_df = checkouts.merge(
        customers[customer_merge_cols],
        left_on=CHECKOUT_JOIN_KEY,
        right_on=CUSTOMER_JOIN_KEY,
        how="left",
    )
    entity_df = entity_df.rename(
        columns={
            "id_x": CHECKOUT_ID_COL,
            "customer": "customer_id",
            TIMESTAMP_COL: "event_timestamp",
        }
    )
    entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)
    entity_df = entity_df.dropna(subset=[LABEL_COL])

    # -- Diagnostic: entity key coverage after merge --
    n_null_email = entity_df["email"].isna().sum()
    n_null_cust = entity_df["customer_id"].isna().sum()
    n_null_store = entity_df["store_id"].isna().sum()
    logger.info(
        "Entity key coverage: email=%d/%d null, customer_id=%d null, store_id=%d null",
        n_null_email,
        len(entity_df),
        n_null_cust,
        n_null_store,
    )

    # Drop rows with null email — no customer record means no identity/charge features;
    # training on all-zero features teaches the model that "zeros = legit", which is wrong.
    pre_len = len(entity_df)
    entity_df = entity_df.dropna(subset=["email"])
    logger.info("Dropped %d rows with null email (no customer match)", pre_len - len(entity_df))

    # -- Derive request features (must match extracts_request_features logic) --
    logger.info("Deriving real-time request features from historical context...")

    # 1. is_night_time (22:00 - 06:00)
    entity_df["is_night_time"] = (
        (entity_df["event_timestamp"].dt.hour >= 22) | (entity_df["event_timestamp"].dt.hour < 6)
    ).astype(int)

    # 2. is_high_value (> 100)
    entity_df["subscription_value"] = pd.to_numeric(
        entity_df["subscription_value"], errors="coerce"
    ).fillna(0.0)
    entity_df["is_high_value"] = (entity_df["subscription_value"] > 100.0).astype(int)

    # 3. email_domain
    entity_df["email_domain"] = entity_df["email"].str.split("@").str[-1].str.lower().fillna("")

    # Keep entity keys + request feature columns needed for train/serve parity
    request_feature_cols = list(REQUEST_FEATURE_SCHEMA.keys())
    entity_cols = [
        CHECKOUT_ID_COL,
        "email",
        "customer_id",
        "store_id",
        "event_timestamp",
        LABEL_COL,
    ] + request_feature_cols

    # Filter columns and enforce types
    entity_df = entity_df[entity_cols].copy()
    logger.info(
        "Entity DataFrame: %d rows, request features: %s", len(entity_df), request_feature_cols
    )

    # -- Fetch features from Feast per view (avoid INNER JOIN across views) --
    # When using a single FeatureService, Feast INNER-joins all views — if ANY
    # view has no data for a row (e.g. no charge history), the entire row is
    # dropped.  Fetching per view + LEFT JOIN preserves all rows: only the
    # specific features without data become NULL (later filled with defaults).
    feast_df = entity_df.copy()
    feast_df["store_id"] = feast_df["store_id"].fillna("__UNKNOWN__")

    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))

    # Import all views to iterate over them individually
    from features.services.fraud_models import ALL_VIEWS

    merge_keys = [CHECKOUT_ID_COL, "email", "event_timestamp"]
    entity_col_set = set(entity_cols)
    training_df = entity_df.copy()
    all_feature_cols: list[str] = []

    for fv in ALL_VIEWS:
        # Build "view_name:field_name" references that Feast expects
        feature_refs = [f"{fv.name}:{field.name}" for field in fv.schema]
        logger.info("Fetching view: %s (%d features)...", fv.name, len(feature_refs))
        view_df = store.get_historical_features(
            entity_df=feast_df,
            features=feature_refs,
            full_feature_names=True,
        ).to_df()

        feat_cols = [c for c in view_df.columns if c not in entity_col_set]
        logger.info("  %s returned %d rows, %d feature cols", fv.name, len(view_df), len(feat_cols))

        if not feat_cols:
            continue

        all_feature_cols.extend(feat_cols)

        # LEFT JOIN this view's features back to the spine
        training_df = training_df.merge(
            view_df[merge_keys + feat_cols],
            on=merge_keys,
            how="left",
        )

    logger.info(
        "After per-view LEFT JOINs: %d rows (entity_df had %d)", len(training_df), len(entity_df)
    )
    feature_cols = all_feature_cols

    # -- Fill NULLs with defaults (no prior history) --
    for col in feature_cols:
        null_count = training_df[col].isna().sum()
        if null_count > 0:
            # Type-safe fill: use 0 for numeric, "" for objects/strings
            if pd.api.types.is_numeric_dtype(training_df[col]):
                logger.info("Filling %d NULLs in %s with 0", null_count, col)
                training_df[col] = training_df[col].fillna(0)
            else:
                logger.info('Filling %d NULLs in %s with ""', null_count, col)
                training_df[col] = training_df[col].fillna("")

    # -- Rename back --
    training_df = training_df.rename(columns={"event_timestamp": TIMESTAMP_COL})

    total = len(training_df)
    fraud_count = int(training_df[LABEL_COL].sum())
    fraud_pct = training_df[LABEL_COL].mean() * 100
    logger.info("Total samples: %d  |  fraud: %d (%.1f%%)", total, fraud_count, fraud_pct)

    # -- Split --
    train_start = _dates.get("train_start", "2024-01-01")
    train_end = _dates["train_end"]
    val_start = _dates.get("val_start", train_end)
    val_end = _dates["val_end"]
    test_start = _dates.get("test_start", val_end)
    test_end = _dates["test_end"]

    train = training_df[
        (training_df[TIMESTAMP_COL] >= train_start) & (training_df[TIMESTAMP_COL] < train_end)
    ]
    val = training_df[
        (training_df[TIMESTAMP_COL] >= val_start) & (training_df[TIMESTAMP_COL] < val_end)
    ]
    test = training_df[
        (training_df[TIMESTAMP_COL] >= test_start) & (training_df[TIMESTAMP_COL] < test_end)
    ]

    logger.info(
        "Splits (%s to %s, %s to %s, %s to %s):",
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
    )
    logger.info("  train: %d samples (%.1f%% fraud)", len(train), train[LABEL_COL].mean() * 100)
    logger.info("  val:   %d samples (%.1f%% fraud)", len(val), val[LABEL_COL].mean() * 100)
    logger.info("  test:  %d samples (%.1f%% fraud)", len(test), test[LABEL_COL].mean() * 100)

    # -- Save --
    TIMESTAMP_SNAPSHOT = datetime.now().strftime("%Y%m%d-%H%M%S").replace(" ", "-")
    logger.info("Saving to %s", OUTPUT_DIR / TIMESTAMP_SNAPSHOT)
    (OUTPUT_DIR / TIMESTAMP_SNAPSHOT).mkdir(parents=True, exist_ok=True)

    train.to_parquet((OUTPUT_DIR / TIMESTAMP_SNAPSHOT / "train.parquet"), index=False)
    val.to_parquet((OUTPUT_DIR / TIMESTAMP_SNAPSHOT / "validation.parquet"), index=False)
    test.to_parquet((OUTPUT_DIR / TIMESTAMP_SNAPSHOT / "test.parquet"), index=False)

    logger.info("Saved to %s", OUTPUT_DIR / TIMESTAMP_SNAPSHOT)


if __name__ == "__main__":
    main()
