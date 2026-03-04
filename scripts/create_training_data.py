from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from feast import FeatureStore
import pandas as pd
import yaml

# Add backend + feature_repo to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_PATH = PROJECT_ROOT / "src" / "backend"
FEATURE_REPO_PATH = BACKEND_PATH / "feature_repo"
sys.path.insert(0, str(BACKEND_PATH))
sys.path.insert(0, str(FEATURE_REPO_PATH))

from features.services.fraud_models import ALL_VIEWS
from services.fraud.features.request_features import REQUEST_FEATURE_SCHEMA
from services.fraud.features.selection import AddMissingIndicators

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
CONFIG_PATH = PROJECT_ROOT / "src" / "backend" / "services" / "fraud" / "training" / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "01-clean"

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

OUTPUT_DIR = Path(_config["data"]["training_dir"])
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_dates = _config["dates"]
_fraud_threshold = _config["modeling"]["fraud_threshold_days"]


def build_entity_df(
    customers: pd.DataFrame,
    checkouts: pd.DataFrame,
) -> pd.DataFrame:
    customers["has_high_end_device"] = (
        pd.to_numeric(customers["high_end_count"], errors="coerce").fillna(0) > 0
    )

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

    pre_len = len(entity_df)
    entity_df = entity_df.dropna(subset=["email"])
    logger.info("Dropped %d rows with null email (no customer match)", pre_len - len(entity_df))

    entity_df["is_night_time"] = (
        (entity_df["event_timestamp"].dt.hour >= 22) | (entity_df["event_timestamp"].dt.hour < 6)
    ).astype(int)

    entity_df["subscription_value"] = pd.to_numeric(
        entity_df["subscription_value"], errors="coerce"
    ).fillna(0.0)
    entity_df["is_high_value"] = (entity_df["subscription_value"] > 100.0).astype(int)

    entity_df["email_domain"] = entity_df["email"].str.split("@").str[-1].str.lower().fillna("")

    cat_lower = entity_df["category"].fillna("").str.strip().str.lower()
    entity_df["is_storage_variant"] = cat_lower.str.contains("gb|tb|cpu", regex=True).astype(int)
    entity_df["is_smartphone_or_watch"] = cat_lower.str.contains(
        "smartphone|smartwatch", regex=True
    ).astype(int)
    entity_df["is_high_risk_category"] = (
        (entity_df["is_storage_variant"] == 1)
        | cat_lower.str.contains("smartphone|smartwatch", regex=True)
    ).astype(int)
    entity_df["category_risk_tier"] = "low"
    entity_df.loc[entity_df["is_smartphone_or_watch"] == 1, "category_risk_tier"] = "medium"
    entity_df.loc[entity_df["is_storage_variant"] == 1, "category_risk_tier"] = "high"

    return entity_df


def derive_card_features(
    entity_df: pd.DataFrame,
    checkouts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    charges = pd.read_csv(DATA_DIR / "charges.csv")

    checkout_payment = checkouts[["id", "payment_intent"]].copy()
    checkout_payment = checkout_payment.rename(columns={"id": CHECKOUT_ID_COL})
    checkout_payment = checkout_payment.dropna(subset=["payment_intent"])

    payment_to_card = charges[
        ["payment_intent", "card_fingerprint", "card_brand", "card_funding", "card_cvc_check"]
    ].drop_duplicates(subset=["payment_intent"])

    card_df = checkout_payment.merge(payment_to_card, on="payment_intent", how="left")

    entity_df = entity_df.merge(
        card_df[[CHECKOUT_ID_COL, "card_brand", "card_funding", "card_cvc_check"]],
        on=CHECKOUT_ID_COL,
        how="left",
    )

    entity_df["card_brand"] = entity_df["card_brand"].fillna("")
    entity_df["card_funding"] = entity_df["card_funding"].fillna("")
    entity_df["card_cvc_check"] = entity_df["card_cvc_check"].fillna("")

    card_funding_lower = entity_df["card_funding"].fillna("").str.lower().str.strip()
    card_cvc_lower = entity_df["card_cvc_check"].fillna("").str.lower().str.strip()

    entity_df["is_debit_card"] = (card_funding_lower == "debit").astype(int)
    entity_df["is_prepaid_card"] = (card_funding_lower == "prepaid").astype(int)
    entity_df["card_cvc_fail"] = (card_cvc_lower == "fail").astype(int)
    entity_df["card_cvc_unavailable"] = (card_cvc_lower == "unavailable").astype(int)
    entity_df["is_high_risk_card"] = (
        (entity_df["is_debit_card"] == 1)
        | (entity_df["card_cvc_fail"] == 1)
        | (entity_df["card_cvc_unavailable"] == 1)
    ).astype(int)
    return entity_df, payment_to_card


def split_by_dates(training_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return train, val, test


def build_feast_entity_df(
    entity_df: pd.DataFrame,
    checkouts: pd.DataFrame,
    payment_to_card: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare entity rows for Feast historical retrieval."""
    feast_df = entity_df.copy()
    feast_df["store_id"] = feast_df["store_id"].fillna("__UNKNOWN__")

    checkout_to_card = payment_to_card[["payment_intent", "card_fingerprint"]].copy()
    checkout_to_card = checkout_to_card.rename(columns={"id": CHECKOUT_ID_COL})

    checkout_to_fingerprint = checkouts[["id", "payment_intent"]].copy()
    checkout_to_fingerprint = checkout_to_fingerprint.rename(columns={"id": CHECKOUT_ID_COL})
    checkout_to_fingerprint = checkout_to_fingerprint.dropna(subset=["payment_intent"])
    checkout_to_fingerprint = checkout_to_fingerprint.merge(
        checkout_to_card, on="payment_intent", how="left"
    )
    checkout_to_fingerprint = checkout_to_fingerprint[
        [CHECKOUT_ID_COL, "card_fingerprint"]
    ].drop_duplicates()

    feast_df = feast_df.merge(checkout_to_fingerprint, on=CHECKOUT_ID_COL, how="left")
    feast_df["card_fingerprint"] = feast_df["card_fingerprint"].fillna("__UNKNOWN__")
    return feast_df


def fetch_features_per_view(
    store: FeatureStore,
    feast_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    entity_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch historical features view-by-view and left-join them to the spine."""
    merge_keys = [CHECKOUT_ID_COL, "email", "event_timestamp"]
    entity_col_set = set(entity_cols)
    training_df = entity_df.copy()
    all_feature_cols: list[str] = []

    for fv in ALL_VIEWS:
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

        feat_cols = [c for c in feat_cols if c != "card_fingerprint"]
        if not feat_cols:
            continue

        all_feature_cols.extend(feat_cols)
        training_df = training_df.merge(view_df[merge_keys + feat_cols], on=merge_keys, how="left")

    logger.info(
        "After per-view LEFT JOINs: %d rows (entity_df had %d)", len(training_df), len(entity_df)
    )
    return training_df, all_feature_cols


def add_missing_indicators(
    training_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_ind = AddMissingIndicators()
    training_df = missing_ind.fit_transform(training_df)
    indicator_cols = list(missing_ind.get_indicators().keys())

    if indicator_cols:
        feature_cols.extend(indicator_cols)
        logger.info("Added missing indicators: %s", indicator_cols)

    return training_df, feature_cols, indicator_cols


def remove_duplicate_card_batch_features(
    training_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    duplicate_card_cols = [
        c
        for c in training_df.columns
        if "card_brand" in c or "card_funding" in c or "card_cvc_check" in c
    ]
    cols_to_drop = [
        c for c in duplicate_card_cols if c.startswith(("charge_features__", "card_features__"))
    ]
    if cols_to_drop:
        logger.info("Dropping duplicate card batch features: %s", cols_to_drop)
        training_df = training_df.drop(columns=cols_to_drop)
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    return training_df, feature_cols


def apply_missing_value_policy(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    indicator_cols: list[str],
) -> pd.DataFrame:
    """Keep numeric NaN values; fill only non-numeric nulls with empty string."""
    for col in feature_cols:
        if col in indicator_cols:
            continue

        null_count = training_df[col].isna().sum()
        if null_count == 0:
            continue

        if pd.api.types.is_numeric_dtype(training_df[col]):
            logger.info("Keeping %d NaNs in %s (LightGBM native handling)", null_count, col)
            continue

        logger.info('Filling %d NULLs in %s with ""', null_count, col)
        training_df[col] = training_df[col].fillna("")

    return training_df


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
    request_feature_cols = list(REQUEST_FEATURE_SCHEMA.keys())
    entity_df = build_entity_df(customers, checkouts)

    # -- Derive card features from charges table via payment_intent --
    logger.info("Deriving card features from charges table...")
    entity_df, payment_to_card = derive_card_features(entity_df, checkouts)

    entity_cols = [
        CHECKOUT_ID_COL,
        "email",
        "customer_id",
        "store_id",
        "event_timestamp",
        LABEL_COL,
    ]
    entity_cols = entity_cols + request_feature_cols
    entity_df = entity_df[entity_cols].copy()
    logger.info(
        "Entity DataFrame: %d rows, request features: %s", len(entity_df), request_feature_cols
    )

    # Fetch per view with LEFT JOIN to keep full training spine.
    feast_df = build_feast_entity_df(entity_df, checkouts, payment_to_card)
    store = FeatureStore(repo_path=str(FEATURE_REPO_PATH))
    training_df, feature_cols = fetch_features_per_view(
        store=store,
        feast_df=feast_df,
        entity_df=entity_df,
        entity_cols=entity_cols,
    )

    training_df, feature_cols, indicator_cols = add_missing_indicators(training_df, feature_cols)
    training_df, feature_cols = remove_duplicate_card_batch_features(training_df, feature_cols)
    training_df = apply_missing_value_policy(training_df, feature_cols, indicator_cols)

    # -- Rename back --
    training_df = training_df.rename(columns={"event_timestamp": TIMESTAMP_COL})

    total = len(training_df)
    fraud_count = int(training_df[LABEL_COL].sum())
    fraud_pct = training_df[LABEL_COL].mean() * 100
    logger.info("Total samples: %d  |  fraud: %d (%.1f%%)", total, fraud_count, fraud_pct)

    # -- Split --
    train, val, test = split_by_dates(training_df)

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
