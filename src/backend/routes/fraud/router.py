from fastapi import APIRouter

from routes.fraud import segment, features
from routes.fraud import checkout, checkouts
from routes.fraud.rules.blacklist import router as blacklist_router
from routes.fraud.rules.stripe_risk import router as stripe_risk_router


router = APIRouter(prefix="/fraud", tags=["fraud"])

router.include_router(segment.router)
router.include_router(features.router)
router.include_router(blacklist_router)
router.include_router(stripe_risk_router)
router.include_router(checkout.router)
router.include_router(checkouts.router)
