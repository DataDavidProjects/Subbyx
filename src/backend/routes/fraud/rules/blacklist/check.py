from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from routes.config import shared_config
from routes.fraud.schemas import BlacklistCheckRequest, BlacklistCheckResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rules"])


def load_blacklist() -> set:
    blacklist_path = Path(shared_config.get("paths", {}).get("blacklist", "data/blacklist.json"))
    if not blacklist_path.is_absolute():
        # check.py: routes/fraud/rules/blacklist/check.py
        # parent: blacklist, rules, fraud, routes, backend, src, Subbyx (project root)
        blacklist_path = (
            Path(__file__).parent.parent.parent.parent.parent.parent.parent / blacklist_path
        )

    logger.debug("loading blacklist from %s", blacklist_path)

    if blacklist_path.exists():
        with open(blacklist_path) as f:
            data = json.load(f)
            emails = set(data.get("emails", []))
            logger.debug("blacklist loaded, %d entries", len(emails))
            return emails

    logger.warning("blacklist file not found at %s", blacklist_path)
    return set()


@router.post("/v1/rules/blacklist/check", response_model=BlacklistCheckResponse)
def check_blacklist(request: BlacklistCheckRequest) -> BlacklistCheckResponse:
    logger.debug("checking blacklist for email %s", request.email)
    blacklist = load_blacklist()

    if request.email in blacklist:
        logger.info("email %s is in blocklist", request.email)
        return BlacklistCheckResponse(
            triggered=True,
            rule="blacklist",
            reason="Email %s is in blocklist" % request.email,
        )

    logger.debug("email %s not in blocklist", request.email)
    return BlacklistCheckResponse(
        triggered=False, rule="blacklist", reason="Email not in blocklist"
    )
