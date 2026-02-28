from __future__ import annotations

import logging

from fastapi import APIRouter

from routes.fraud.checkout import determine_segment as _determine_segment
from routes.fraud.schemas import SegmentDetermineRequest, SegmentDetermineResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["segment"])


@router.post("/v1/segment/determine", response_model=SegmentDetermineResponse)
def determine_segment(
    request: SegmentDetermineRequest,
) -> SegmentDetermineResponse:
    logger.debug(
        "determining segment for email=%s, fiscal_code=%s, card_fingerprint=%s",
        request.email,
        request.fiscal_code,
        request.card_fingerprint,
    )

    segment, reason = _determine_segment(
        customer_id="",
        email=request.email,
        fiscal_code=request.fiscal_code,
        card_fingerprint=request.card_fingerprint,
    )

    logger.debug("segment=%s, reason=%s", segment, reason)
    return SegmentDetermineResponse(segment=segment, reason=reason)
