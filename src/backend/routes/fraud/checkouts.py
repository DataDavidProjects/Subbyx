"""
Checkouts API endpoint.

Note on local vs Docker Compose development:
- In Docker Compose: uses 'backend' as hostname (set via NEXT_PUBLIC_API_URL in frontend)
- Locally: uses 'localhost:8001' for backend
- The backend always reads from the same data path regardless of deployment method
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from routes.config import shared_config
from services.fraud.training.config import config as training_config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fraud-checkouts"])


def get_project_root() -> Path:
    """Get project root path."""
    return Path(__file__).parent.parent.parent.parent.parent


def get_data_path() -> Path:
    data_path = Path(shared_config.get("paths", {}).get("data_dir", "data/01-clean"))
    if not data_path.is_absolute():
        data_path = get_project_root() / data_path

    logger.debug("data path: %s (exists=%s)", data_path, data_path.exists())
    return data_path


def load_customers_map() -> dict[str, dict]:
    """
    Load customer_id -> {email, is_fraud} mapping from data directory.
    """
    customers_map: dict[str, dict] = {}
    threshold = training_config.get("modeling", {}).get("fraud_threshold_days", 15)

    data_path = get_data_path()
    customers_file = data_path / "customers.csv"
    if customers_file.exists():
        df = pd.read_csv(customers_file, index_col=0)
        for _, row in df.iterrows():
            cid = str(row.get("id", ""))
            email = str(row.get("email", ""))
            dunning_days = row.get("dunning_days", 0)
            is_fraud = bool(dunning_days > threshold)

            if cid:
                customers_map[cid] = {
                    "email": email if email != "nan" else None,
                    "is_fraud": is_fraud,
                }
        logger.debug("loaded %d customers details from data", len(customers_map))

    return customers_map


@router.get("/v1/checkouts")
async def get_checkouts(
    mode: Optional[str] = Query(None, description="Filter by mode (payment, setup)"),
    status: Optional[str] = Query(None, description="Filter by status (complete, expired)"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by email, customer ID or SKU"),
    category: Optional[str] = Query(None, description="Filter by category"),
    grade: Optional[str] = Query(None, description="Filter by grade"),
    sort_order: Optional[str] = Query("desc", description="Sort by date: 'asc' or 'desc'"),
    is_fraud: Optional[bool] = Query(None, description="Filter by ground truth status"),
) -> dict:
    """
    Get checkouts from future data (test set) with pagination and filtering.
    """
    logger.debug(
        "get_checkouts called: page=%d, page_size=%d, mode=%s, status=%s, "
        "search=%s, category=%s, grade=%s, is_fraud=%s, limit=%s",
        page,
        page_size,
        mode,
        status,
        search,
        category,
        grade,
        is_fraud,
        limit,
    )

    data_path = get_data_path()
    checkouts_path = data_path / "checkouts.csv"

    if not checkouts_path.exists():
        logger.error("checkouts data not found at %s", checkouts_path)
        raise HTTPException(status_code=404, detail="Checkouts data not found")

    df = pd.read_csv(checkouts_path, index_col=0)
    df = df.rename(columns={"customer": "customer_id"})
    df["created"] = pd.to_datetime(df["created"], utc=True)
    df = df.infer_objects()

    dates_config = training_config.get("dates", {})
    test_start = pd.Timestamp(dates_config.get("test_start", "2024-09-01"), tz="UTC")
    test_end = pd.Timestamp(dates_config.get("test_end", "2024-10-31"), tz="UTC")
    df = df[(df["created"] >= test_start) & (df["created"] < test_end)]

    if mode:
        df = df[df["mode"] == mode]
    if status:
        df = df[df["status"] == status]
    if category:
        df = df[df["category"] == category]
    if grade:
        df = df[df["grade"] == grade]

    customers_data = load_customers_map()
    df["customer_id"] = df["customer_id"].astype(str)

    # Map email and label from customer data
    df["email"] = df["customer_id"].apply(lambda x: customers_data.get(x, {}).get("email"))
    df["is_fraud"] = df["customer_id"].apply(
        lambda x: customers_data.get(x, {}).get("is_fraud", False)
    )

    # Join card_fingerprint from charges via payment_intent
    charges_path = data_path / "charges.csv"
    if charges_path.exists():
        charges_df = pd.read_csv(charges_path, index_col=0)
        pi_to_card = dict(
            zip(
                charges_df["payment_intent"].astype(str),
                charges_df["card_fingerprint"].astype(str),
            )
        )
        df["card_fingerprint"] = df["payment_intent"].astype(str).map(pi_to_card)
    else:
        df["card_fingerprint"] = None

    search_lower = search.lower().strip() if search else None

    if search_lower:
        df = df[
            df["email"].astype(str).str.lower().str.contains(search_lower, na=False)
            | df["customer_id"].str.lower().str.contains(search_lower, na=False)
            | df["sku"].astype(str).str.lower().str.contains(search_lower, na=False)
        ]

    if is_fraud is not None:
        df = df[df["is_fraud"] == is_fraud]

    df = df.sort_values("created", ascending=(sort_order == "asc"))

    total = len(df)
    logger.debug("found %d checkouts matching filters", total)

    if not limit:
        start = (page - 1) * page_size
        end = start + page_size
        paginated = df.iloc[start:end].to_dict(orient="records")
    else:
        paginated = df.to_dict(orient="records")

    total_pages = max(1, -(-total // page_size))

    logger.debug("returning page %d/%d (%d items)", page, total_pages, len(paginated))

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "checkouts": paginated,
    }
