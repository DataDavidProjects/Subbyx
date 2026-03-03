from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache
import logging

from feast import FeatureStore

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CHARGES_CSV = _DATA_DIR / "charges.csv"

_FEATURES_REPO = _BASE / "src" / "backend" / "feature_repo"

_card_store: FeatureStore | None = None


def _get_card_store() -> FeatureStore | None:
    global _card_store
    if _card_store is None:
        try:
            _card_store = FeatureStore(repo_path=str(_FEATURES_REPO))
        except Exception as e:
            logger.warning("Failed to initialize Feast store for cards: %s", e)
    return _card_store


class CardDetails:
    """Card details for cold start risk assessment."""

    __slots__ = ["fingerprint", "brand", "funding", "cvc_check"]

    def __init__(
        self,
        fingerprint: str,
        brand: str,
        funding: str,
        cvc_check: str,
    ) -> None:
        self.fingerprint = fingerprint
        self.brand = brand
        self.funding = funding
        self.cvc_check = cvc_check


@lru_cache(maxsize=4096)
def get_card_for_payment_intent(payment_intent_id: str | None) -> str | None:
    """
    Resolve a card fingerprint from a payment intent ID using the CSV data source.

    Args:
        payment_intent_id: The payment intent ID to look up

    Returns:
        The matched card fingerprint string, or None if not found or ID is None
    """
    card = get_card_details_for_payment_intent(payment_intent_id)
    return card.fingerprint if card else None


def get_card_details_for_payment_intent(payment_intent_id: str | None) -> CardDetails | None:
    """
    Resolve full card details from a payment intent ID.

    First resolves card_fingerprint from payment_intent via CSV, then fetches
    card attributes from Feast.

    Args:
        payment_intent_id: The payment intent ID to look up

    Returns:
        CardDetails object with fingerprint, brand, funding, cvc_check, or None if not found
    """
    if payment_intent_id is None:
        return None

    fingerprint = _get_card_fingerprint_from_csv(payment_intent_id)
    if fingerprint is None:
        return None

    brand, funding, cvc_check = _get_card_attributes_from_feast(fingerprint)

    return CardDetails(
        fingerprint=fingerprint,
        brand=brand or "",
        funding=funding or "",
        cvc_check=cvc_check or "",
    )


def _get_card_fingerprint_from_csv(payment_intent_id: str) -> str | None:
    """Resolve card_fingerprint from payment_intent using CSV."""
    logger.debug("Looking up card fingerprint for payment_intent: %s", payment_intent_id)

    if not _CHARGES_CSV.exists():
        logger.error("Charges CSV file not found: %s", _CHARGES_CSV)
        return None

    try:
        with open(_CHARGES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("payment_intent") == payment_intent_id:
                    fingerprint = row.get("card_fingerprint", "")
                    if fingerprint:
                        return fingerprint
                    return None

        logger.debug("Card fingerprint not found for payment_intent: %s", payment_intent_id)
        return None

    except Exception as e:
        logger.error(
            "Failed to fetch card fingerprint for payment_intent %s: %s", payment_intent_id, e
        )
        return None


def _get_card_attributes_from_feast(
    card_fingerprint: str,
) -> tuple[str | None, str | None, str | None]:
    """Fetch card attributes from Feast using card_fingerprint entity."""
    store = _get_card_store()
    if store is None:
        logger.warning("Feast store not available for card attributes")
        return None, None, None

    try:
        response = store.get_online_features(
            features=[
                "card_features:card_brand",
                "card_features:card_funding",
                "card_features:card_cvc_check",
            ],
            entity_rows=[{"card_fingerprint": card_fingerprint}],
        )
        data = response.to_dict()

        brand = data.get("card_features__card_brand", [None])[0]
        funding = data.get("card_features__card_funding", [None])[0]
        cvc_check = data.get("card_features__card_cvc_check", [None])[0]

        if brand or funding or cvc_check:
            logger.debug("Fetched card attributes from Feast for %s", card_fingerprint)

        return brand, funding, cvc_check

    except Exception as e:
        logger.warning("Failed to fetch card attributes from Feast for %s: %s", card_fingerprint, e)
        return None, None, None


__all__ = ["get_card_for_payment_intent", "get_card_details_for_payment_intent", "CardDetails"]
