from __future__ import annotations

import csv
from pathlib import Path
from functools import lru_cache
import logging

from feast import FeatureStore

logger = logging.getLogger(__name__)

_BASE = Path(__file__).resolve().parents[6]
_DATA_DIR = _BASE / "data" / "01-clean"
_CUSTOMERS_CSV = _DATA_DIR / "customers.csv"

_FEATURES_REPO = _BASE / "src" / "backend" / "feature_repo"

_customer_store: FeatureStore | None = None


def _get_customer_store() -> FeatureStore | None:
    global _customer_store
    if _customer_store is None:
        try:
            _customer_store = FeatureStore(repo_path=str(_FEATURES_REPO))
        except Exception as e:
            logger.warning("Failed to initialize Feast store for customers: %s", e)
    return _customer_store


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
    Fetch a customer record by ID using Feast (with CSV fallback).

    Args:
        customer_id: The customer ID to look up

    Returns:
        A populated CustomerRecord

    Raises:
        ValueError: If the customer_id is not found
    """
    logger.info("Looking up customer_id: %s", customer_id)

    record = _get_customer_from_feast(customer_id)
    if record:
        return record

    record = _get_customer_from_csv(customer_id)
    if record:
        return record

    logger.warning("Customer ID not found: %s", customer_id)
    raise ValueError(f"Customer ID not found: {customer_id}")


def _get_customer_from_feast(customer_id: str) -> CustomerRecord | None:
    """Fetch customer from Feast online store."""
    store = _get_customer_store()
    if store is None:
        return None

    try:
        response = store.get_online_features(
            features=[
                "customer_features:email",
                "customer_features:fiscal_code",
                "customer_features:gender",
                "customer_features:birth_date",
                "customer_features:birth_province",
                "customer_features:birth_country",
                "customer_features:high_end_count",
            ],
            entity_rows=[{"customer_id": customer_id}],
        )
        data = response.to_dict()

        email = data.get("customer_features__email", [None])[0]
        if email is None:
            return None

        fiscal_code = data.get("customer_features__fiscal_code", [None])[0] or ""
        gender = data.get("customer_features__gender", [None])[0] or ""
        birth_date = data.get("customer_features__birth_date", [None])[0] or ""
        birth_province = data.get("customer_features__birth_province", [None])[0] or ""
        birth_country = data.get("customer_features__birth_country", [None])[0] or ""
        high_end_count = data.get("customer_features__high_end_count", [None])[0]

        has_high_end_device = (
            bool(high_end_count and high_end_count > 0) if high_end_count else False
        )

        logger.debug("Fetched customer from Feast: %s", customer_id)

        return CustomerRecord(
            id=customer_id,
            email=email,
            fiscal_code=fiscal_code,
            gender=gender,
            birth_date=birth_date,
            birth_province=birth_province,
            birth_country=birth_country,
            has_high_end_device=has_high_end_device,
        )

    except Exception as e:
        logger.warning("Failed to fetch customer from Feast for %s: %s", customer_id, e)
        return None


def _get_customer_from_csv(customer_id: str) -> CustomerRecord | None:
    """Fetch customer from CSV (fallback)."""
    if not _CUSTOMERS_CSV.exists():
        logger.error("Customers CSV file not found: %s", _CUSTOMERS_CSV)
        return None

    try:
        with open(_CUSTOMERS_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id") == customer_id:
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

        return None

    except Exception as e:
        logger.error("Failed to fetch customer_id %s from CSV: %s", customer_id, e)
        return None


__all__ = ["CustomerRecord", "get_by_id"]
