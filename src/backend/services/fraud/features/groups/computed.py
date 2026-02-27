from services.fraud.features.base import Entity
from services.fraud.features.registry import register_computed


def total_failure_rate(features: dict) -> float:
    """Combined failure rate from charges and payment intents."""
    charge_fr = features.get("charge_failure_rate") or 0.0
    payment_fr = features.get("payment_failure_rate") or 0.0
    if charge_fr + payment_fr > 0:
        return (charge_fr + payment_fr) / 2
    return 0.0


def is_new_customer(features: dict) -> float:
    """Derived: 1 if customer has no prior checkouts."""
    prior_count = features.get("prior_checkout_count") or 0
    return 1.0 if prior_count == 0 else 0.0


def combined_risk_score(features: dict) -> float:
    """Weighted combination of risk signals."""
    charge_fr = features.get("charge_failure_rate") or 0.0
    payment_fr = features.get("payment_failure_rate") or 0.0
    prior_count = features.get("prior_checkout_count") or 0

    # Weight: payment failures more important, more checkouts = lower risk
    score = (charge_fr * 0.3) + (payment_fr * 0.5) + (0.2 * min(prior_count / 10, 1.0))
    return score


register_computed(
    name="total_failure_rate",
    entity=Entity.EMAIL,
    compute_fn=total_failure_rate,
    dependencies=["charge_failure_rate", "payment_failure_rate"],
    group="computed",
)

register_computed(
    name="is_new_customer",
    entity=Entity.EMAIL,
    compute_fn=is_new_customer,
    dependencies=["prior_checkout_count"],
    group="computed",
)

register_computed(
    name="combined_risk_score",
    entity=Entity.EMAIL,
    compute_fn=combined_risk_score,
    dependencies=["charge_failure_rate", "payment_failure_rate", "prior_checkout_count"],
    group="computed",
)
