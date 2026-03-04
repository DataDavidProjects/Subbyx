from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.schemas import RuleCheckRequest, RuleCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rules"])

DATA_DIR = Path(__file__).resolve().parents[6] / "data"
CUSTOMERS_PATH = DATA_DIR / "01-clean" / "customers.csv"


def load_fiscal_code_to_emails(cutoff_time: str | None = None) -> dict[str, set[str]]:
    """Load fiscal_code → email mappings from customers created before cutoff_time.

    Args:
        cutoff_time: ISO format timestamp to filter customers (PIT correct).
                    If None, uses all historical data.
    """
    if not CUSTOMERS_PATH.exists():
        logger.warning("customers.csv not found at %s", CUSTOMERS_PATH)
        return {}

    df = pd.read_csv(CUSTOMERS_PATH)

    if cutoff_time:
        df = df[pd.to_datetime(df["created"]) < pd.to_datetime(cutoff_time)]

    mapping: dict[str, set[str]] = defaultdict(set)

    for _, row in df.iterrows():
        fiscal_code = row.get("fiscal_code")
        email = row.get("email")
        if fiscal_code and email:
            mapping[str(fiscal_code)].add(str(email))

    logger.debug("loaded %d fiscal_code → email mappings (cutoff: %s)", len(mapping), cutoff_time)
    return dict(mapping)


def is_duplicate_fiscal_code(
    fiscal_code: str, current_email: str, mapping: dict[str, set[str]]
) -> bool:
    emails = mapping.get(fiscal_code)
    if not emails:
        return False
    return current_email not in emails and len(emails) > 0


@router.post("/v1/rules/fiscal-code/check", response_model=RuleCheckResponse)
def check_fiscal_code(request: RuleCheckRequest) -> RuleCheckResponse:
    fiscal_code = request.fiscal_code
    email = request.email

    if not fiscal_code or not email:
        return RuleCheckResponse(
            triggered=False,
            rule="fiscal_code_duplicate",
            reason="No fiscal_code or email provided",
        )

    mapping = load_fiscal_code_to_emails(request.timestamp)

    if is_duplicate_fiscal_code(fiscal_code, email, mapping):
        emails = mapping[fiscal_code]
        logger.info(
            "fiscal_code %s has multiple emails: %s (current: %s)",
            fiscal_code,
            emails,
            email,
        )
        return RuleCheckResponse(
            triggered=True,
            rule="fiscal_code_duplicate",
            reason="Fiscal code %s used with different emails: %s"
            % (fiscal_code, ", ".join(sorted(emails))),
        )

    return RuleCheckResponse(
        triggered=False,
        rule="fiscal_code_duplicate",
        reason="Fiscal code has no duplicate emails",
    )
