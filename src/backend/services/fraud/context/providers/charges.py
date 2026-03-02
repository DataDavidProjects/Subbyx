from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CHARGES_CSV = _DATA_DIR / "charges.csv"


@lru_cache(maxsize=4096)
def get_card_for_payment_intent(payment_intent_id: str | None) -> str | None:
    """
    Resolve a card fingerprint from a payment intent ID using the CSV data source.

    Args:
        payment_intent_id: The payment intent ID to look up

    Returns:
        The matched card fingerprint string, or None if not found or ID is None
    """
    if payment_intent_id is None:
        return None

    logger.debug("Looking up card fingerprint for payment_intent: %s", payment_intent_id)

    if not _CHARGES_CSV.exists():
        logger.error("Charges CSV file not found: %s", _CHARGES_CSV)
        # We don't raise FileNotFoundError here to keep it graceful as per requirements,
        # but the error should be logged.
        return None

    try:
        with open(_CHARGES_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("payment_intent") == payment_intent_id:
                    fingerprint = row.get("card_fingerprint")
                    if fingerprint:
                        return fingerprint
                    return None

        logger.debug("Fingerprint not found for payment_intent: %s", payment_intent_id)
        return None

    except Exception as e:
        logger.error("Failed to fetch fingerprint for payment_intent %s: %s", payment_intent_id, e)
        return None


__all__ = ["get_card_for_payment_intent"]
