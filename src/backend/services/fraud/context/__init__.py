from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

from .providers import checkouts, customers, charges
from .providers.charges import get_card_details_for_payment_intent

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckoutContext:
    """Stable contract carrying all entity keys and request-time fields."""

    # Entity keys
    checkout_id: str
    customer_id: str
    email: str
    store_id: str
    card_fingerprint: Optional[str]
    fiscal_code: str

    # Request-time fields
    timestamp: str
    gender: str
    birth_date: str
    birth_province: str
    birth_country: str
    has_high_end_device: bool
    subscription_value: float
    grade: str
    category: str

    # Card fields (for cold start risk assessment)
    card_brand: str
    card_funding: str
    card_cvc_check: str


def resolve_checkout(checkout_id: str) -> CheckoutContext:
    """Orchestrate providers to build a fully-populated CheckoutContext.

    Args:
        checkout_id: The checkout ID to resolve.

    Returns:
        A CheckoutContext with all entity keys and request-time fields populated.

    Raises:
        ValueError: If the checkout or customer record cannot be found.
        FileNotFoundError: If underlying data files are missing.
    """
    logger.info("Resolving checkout context for checkout_id=%s", checkout_id)

    checkout = checkouts.get_by_id(checkout_id)
    customer = customers.get_by_id(checkout.customer_id)
    card_fingerprint = charges.get_card_for_payment_intent(checkout.payment_intent)

    # Get full card details for cold start risk assessment
    card_details = get_card_details_for_payment_intent(checkout.payment_intent)
    if card_details:
        card_brand = card_details.brand
        card_funding = card_details.funding
        card_cvc_check = card_details.cvc_check
    else:
        card_brand = ""
        card_funding = ""
        card_cvc_check = ""

    return CheckoutContext(
        checkout_id=checkout.id,
        customer_id=checkout.customer_id,
        email=customer.email,
        store_id=checkout.store_id,
        card_fingerprint=card_fingerprint,
        fiscal_code=customer.fiscal_code,
        timestamp=checkout.created,
        gender=customer.gender,
        birth_date=customer.birth_date,
        birth_province=customer.birth_province,
        birth_country=customer.birth_country,
        has_high_end_device=customer.has_high_end_device,
        subscription_value=checkout.subscription_value,
        grade=checkout.grade,
        category=checkout.category,
        card_brand=card_brand,
        card_funding=card_funding,
        card_cvc_check=card_cvc_check,
    )


__all__ = ["CheckoutContext", "resolve_checkout"]
