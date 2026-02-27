from __future__ import annotations

from feast import Entity
from feast.value_type import ValueType

customer_id = Entity(
    name="customer_id",
    join_keys=["customer_id"],
    description="Customer ID entity for feature lookup",
    value_type=ValueType.STRING,
)
