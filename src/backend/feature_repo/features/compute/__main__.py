"""Batch compute orchestrator — runs all feature generation modules sequentially."""

import logging
import time

from features.compute import (
    customer_features,
    charge_features,
    payment_intent_features,
    checkout_features,
    address_features,
    store_features,
    geo_time_features,
    card_features,
    checkout_velocity,
)

logger = logging.getLogger(__name__)

MODULES = [
    ("customer_features", customer_features),
    ("charge_features", charge_features),
    ("payment_intent_features", payment_intent_features),
    ("checkout_features", checkout_features),
    ("address_features", address_features),
    ("store_features", store_features),
    ("geo_time_features", geo_time_features),
    ("card_features", card_features),
    ("checkout_velocity", checkout_velocity),
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("Starting batch feature compute (%d modules)", len(MODULES))

    for name, module in MODULES:
        logger.info("Running %s ...", name)
        t0 = time.perf_counter()
        module.generate()
        elapsed = time.perf_counter() - t0
        logger.info("Finished %s in %.2fs", name, elapsed)

    logger.info("All feature modules completed.")


if __name__ == "__main__":
    main()
