from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.schemas import RuleCheckRequest, RuleCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rules"])

HIGHEST_RISK_LEVEL = "highest"


def load_charges_with_highest_risk(cutoff_time: str | None = None) -> frozenset:
    """Load emails that have charges with highest risk level before cutoff_time.

    Args:
        cutoff_time: ISO format timestamp to filter charges (PIT correct).
                    If None, uses all historical data.
    """
    data_dir = Path(shared_config.get("paths", {}).get("data_dir", "data/01-clean"))
    if not data_dir.is_absolute():
        data_dir = Path(__file__).parent.parent.parent.parent.parent.parent.parent / data_dir

    charges_path = data_dir / "charges.csv"
    if not charges_path.exists():
        logger.warning("charges.csv not found at %s", charges_path)
        return frozenset()

    logger.debug("loading charges to find highest risk emails from %s", charges_path)
    df = pd.read_csv(charges_path)

    if cutoff_time:
        df = df[pd.to_datetime(df["created"]) < pd.to_datetime(cutoff_time)]

    df_highest = df[df["outcome_risk_level"] == HIGHEST_RISK_LEVEL]
    emails = frozenset(df_highest["email"].dropna().unique())
    logger.debug("found %d emails with highest risk level (cutoff: %s)", len(emails), cutoff_time)
    return emails


@router.post("/v1/rules/stripe_risk/check", response_model=RuleCheckResponse)
def check_stripe_risk(request: RuleCheckRequest) -> RuleCheckResponse:
    logger.debug("checking stripe_risk for email %s", request.email)

    high_risk_emails = load_charges_with_highest_risk(request.timestamp)

    if request.email in high_risk_emails:
        logger.info("email %s has highest risk level from Stripe", request.email)
        return RuleCheckResponse(
            triggered=True,
            rule="stripe_risk",
            reason="Email has highest risk level from Stripe",
        )

    logger.debug("email %s does not have highest risk level", request.email)
    return RuleCheckResponse(
        triggered=False,
        rule="stripe_risk",
        reason="Email does not have highest risk level",
    )
