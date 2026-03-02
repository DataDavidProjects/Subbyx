from feast import Entity
from feast.value_type import ValueType

store_id = Entity(
    name="store_id",
    join_keys=["store_id"],
    description="Store ID entity for feature lookup",
    value_type=ValueType.STRING,
)
