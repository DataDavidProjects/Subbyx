from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CHARGES_CSV = _DATA_DIR / "charges.csv"


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


@lru_cache(maxsize=4096)
def get_card_details_for_payment_intent(payment_intent_id: str | None) -> CardDetails | None:
    """
    Resolve full card details from a payment intent ID using the CSV data source.

    Args:
        payment_intent_id: The payment intent ID to look up

    Returns:
        CardDetails object with fingerprint, brand, funding, cvc_check, or None if not found
    """
    if payment_intent_id is None:
        return None

    logger.debug("Looking up card details for payment_intent: %s", payment_intent_id)

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
                        return CardDetails(
                            fingerprint=fingerprint,
                            brand=row.get("card_brand", ""),
                            funding=row.get("card_funding", ""),
                            cvc_check=row.get("card_cvc_check", ""),
                        )
                    return None

        logger.debug("Card details not found for payment_intent: %s", payment_intent_id)
        return None

    except Exception as e:
        logger.error("Failed to fetch card details for payment_intent %s: %s", payment_intent_id, e)
        return None


__all__ = ["get_card_for_payment_intent", "get_card_details_for_payment_intent", "CardDetails"]
