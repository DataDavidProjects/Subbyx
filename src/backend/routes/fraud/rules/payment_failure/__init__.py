"""Payment failure rate rule.

Catches returning customers with extreme payment failure rates that the
PIT-trained model misses due to train/serve feature skew.  The online
Feast store returns *latest cumulative* stats while the model was trained
on point-in-time values at checkout time — so very high failure rates
that built up post-checkout are invisible to the model.

This rule acts as a safety net: if a returning customer has a failure
rate above the threshold AND enough attempts to make the rate reliable,
we flag it for review.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Thresholds — conservative to avoid false positives on legitimate
# customers with a few early failures.
FAILURE_RATE_THRESHOLD = 0.80
MIN_ATTEMPTS = 15


def check_payment_failure(
    features: dict,
    segment: str,
) -> tuple[bool, str]:
    """Return (triggered, reason) if the payment failure rule fires.

    Checks both charge and payment-intent failure rates across all segments.

    """
    checks: list[tuple[str, str, str]] = [
        (
            "payment_intent_stats_features__failure_rate",
            "payment_intent_stats_features__n_payment_intents",
            "payment intent",
        ),
        (
            "charge_stats_features__failure_rate",
            "charge_stats_features__n_charges",
            "charge",
        ),
    ]

    for rate_key, count_key, label in checks:
        # Check both the prefixed key (from get_features) and the
        # base key if the prefix is missing.
        rate = features.get(rate_key)
        if rate is None:
            rate = features.get(rate_key.split("__")[-1])

        count = features.get(count_key)
        if count is None:
            count = features.get(count_key.split("__")[-1])

        if rate is None or count is None:
            continue
        if isinstance(rate, float) and math.isnan(rate):
            continue

        rate = float(rate)
        count = float(count)

        if count >= MIN_ATTEMPTS and rate >= FAILURE_RATE_THRESHOLD:
            reason = (
                f"High {label} failure rate: {rate:.0%} "
                f"across {count:.0f} attempts "
                f"(threshold: {FAILURE_RATE_THRESHOLD:.0%} with >={MIN_ATTEMPTS} attempts)"
            )
            logger.info("payment_failure rule triggered: %s", reason)
            return True, reason

    return False, ""
