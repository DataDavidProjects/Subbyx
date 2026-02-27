from fastapi import APIRouter

from routes.fraud import segment, features, rules
from routes.fraud import checkout, checkouts


router = APIRouter(prefix="/fraud", tags=["fraud"])

router.include_router(segment.router)
router.include_router(features.router)
router.include_router(rules.router)
router.include_router(checkout.router)
router.include_router(checkouts.router)
