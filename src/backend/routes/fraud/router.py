from fastapi import APIRouter

from routes.fraud import segment, features
from routes.fraud import checkout, checkouts
from routes.fraud.rules import blacklist, fiscal_code, stripe_risk


router = APIRouter(prefix="/fraud", tags=["fraud"])

router.include_router(segment.router)
router.include_router(features.router)
router.include_router(blacklist.router)
router.include_router(fiscal_code.router)
router.include_router(stripe_risk.router)
router.include_router(checkout.router)
router.include_router(checkouts.router)
