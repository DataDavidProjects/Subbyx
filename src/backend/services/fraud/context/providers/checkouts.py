from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache

import logging

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CHECKOUTS_CSV = _DATA_DIR / "checkouts.csv"


class CheckoutRecord:
    """A single checkout record from the data source."""

    __slots__ = [
        "id",
        "customer_id",
        "store_id",
        "payment_intent",
        "created",
        "subscription_value",
        "grade",
        "category",
        "status",
        "mode",
    ]

    def __init__(
        self,
        id: str,
        customer_id: str,
        store_id: str,
        payment_intent: str,
        created: str,
        subscription_value: float,
        grade: str,
        category: str,
        status: str,
        mode: str,
    ) -> None:
        self.id = id
        self.customer_id = customer_id
        self.store_id = store_id
        self.payment_intent = payment_intent
        self.created = created
        self.subscription_value = subscription_value
        self.grade = grade
        self.category = category
        self.status = status
        self.mode = mode


@lru_cache(maxsize=1024)
def get_by_id(checkout_id: str) -> CheckoutRecord:
    """
    Fetch a checkout record by ID from the CSV data source.

    Args:
        checkout_id: The checkout ID to look up

    Returns:
        A populated CheckoutRecord

    Raises:
        ValueError: If the checkout_id is not found
    """
    logger.info("Looking up checkout_id: %s", checkout_id)

    if not _CHECKOUTS_CSV.exists():
        logger.error("Checkouts CSV file not found: %s", _CHECKOUTS_CSV)
        raise FileNotFoundError(f"Checkouts data file not found: {_CHECKOUTS_CSV}")

    try:
        with open(_CHECKOUTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == checkout_id:
                    try:
                        return CheckoutRecord(
                            id=row["id"],
                            customer_id=row.get("customer", ""),
                            store_id=row.get("store_id", ""),
                            payment_intent=row.get("payment_intent", ""),
                            created=row["created"],
                            subscription_value=float(row["subscription_value"])
                            if row.get("subscription_value")
                            else 0.0,
                            grade=row.get("grade", ""),
                            category=row.get("category", ""),
                            status=row["status"],
                            mode=row["mode"],
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning("Malformed checkout row for id %s: %s", checkout_id, e)
                        raise ValueError(f"Malformed checkout data for id {checkout_id}") from e

        logger.warning("Checkout ID not found: %s", checkout_id)
        raise ValueError(f"Checkout ID not found: {checkout_id}")

    except Exception as e:
        logger.error("Failed to fetch checkout_id %s: %s", checkout_id, e)
        raise


__all__ = ["CheckoutRecord", "get_by_id"]
