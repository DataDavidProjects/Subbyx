from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CUSTOMERS_CSV = _DATA_DIR / "customers.csv"


class CustomerRecord:
    """A single customer record from the data source."""

    __slots__ = [
        "id",
        "email",
        "fiscal_code",
        "gender",
        "birth_date",
        "birth_province",
        "birth_country",
        "has_high_end_device",
    ]

    def __init__(
        self,
        id: str,
        email: str,
        fiscal_code: str,
        gender: str,
        birth_date: str,
        birth_province: str,
        birth_country: str,
        has_high_end_device: bool,
    ) -> None:
        self.id = id
        self.email = email
        self.fiscal_code = fiscal_code
        self.gender = gender
        self.birth_date = birth_date
        self.birth_province = birth_province
        self.birth_country = birth_country
        self.has_high_end_device = has_high_end_device


@lru_cache(maxsize=2048)
def get_by_id(customer_id: str) -> CustomerRecord:
    """
    Fetch a customer record by ID from the CSV data source.

    Args:
        customer_id: The customer ID to look up

    Returns:
        A populated CustomerRecord

    Raises:
        ValueError: If the customer_id is not found
    """
    logger.info("Looking up customer_id: %s", customer_id)

    if not _CUSTOMERS_CSV.exists():
        logger.error("Customers CSV file not found: %s", _CUSTOMERS_CSV)
        raise FileNotFoundError(f"Customers data file not found: {_CUSTOMERS_CSV}")

    try:
        with open(_CUSTOMERS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == customer_id:
                    try:
                        # Map high_end_count to has_high_end_device
                        high_end_count = row.get("high_end_count", "0")
                        has_high_end_device = float(high_end_count) > 0 if high_end_count else False

                        return CustomerRecord(
                            id=row["id"],
                            email=row.get("email", ""),
                            fiscal_code=row.get("fiscal_code", ""),
                            gender=row.get("gender", ""),
                            birth_date=row.get("birth_date", ""),
                            birth_province=row.get("birth_province", ""),
                            birth_country=row.get("birth_country", ""),
                            has_high_end_device=has_high_end_device,
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning("Malformed customer row for id %s: %s", customer_id, e)
                        raise ValueError(f"Malformed customer data for id {customer_id}") from e

        logger.warning("Customer ID not found: %s", customer_id)
        raise ValueError(f"Customer ID not found: {customer_id}")

    except Exception as e:
        logger.error("Failed to fetch customer_id %s: %s", customer_id, e)
        raise


__all__ = ["CustomerRecord", "get_by_id"]
