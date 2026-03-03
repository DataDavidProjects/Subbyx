import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate() -> None:
    logger.info("Generating card features...")

    repo_root = Path("/Users/davidelupis/Desktop/Subbyx")
    charges_csv = repo_root / "data" / "01-clean" / "charges.csv"
    output_dir = repo_root / "src" / "backend" / "feature_repo" / "data" / "sources"

    output_parquet = output_dir / "card_features.parquet"

    if not charges_csv.exists():
        logger.error("Source file not found: %s", charges_csv)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading %s", charges_csv)
    df = pd.read_csv(charges_csv)

    logger.info("Converting timestamps...")
    df["created"] = pd.to_datetime(df["created"])
    if df["created"].dt.tz is None:
        df["created"] = df["created"].dt.tz_localize("UTC")
    else:
        df["created"] = df["created"].dt.tz_convert("UTC")

    df = df.dropna(subset=["card_fingerprint"])
    df = df.sort_values(by=["card_fingerprint", "created"])

    latest_cards = df.groupby("card_fingerprint").first().reset_index()

    string_cols = ["card_brand", "card_funding", "card_cvc_check"]
    for col in string_cols:
        if col in latest_cards.columns:
            latest_cards[col] = latest_cards[col].astype(str).replace("nan", "")

    card_df = latest_cards[
        ["card_fingerprint", "created", "card_brand", "card_funding", "card_cvc_check"]
    ]

    logger.info("Saving card features to %s", output_parquet)
    card_df.to_parquet(output_parquet, index=False)

    logger.info("Card features generated successfully: %d cards", len(card_df))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    generate()
