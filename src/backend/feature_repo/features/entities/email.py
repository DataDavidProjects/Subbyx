from feast import Entity
from feast.value_type import ValueType

email = Entity(
    name="email",
    join_keys=["email"],
    description="Email entity for feature lookup",
    value_type=ValueType.STRING,
)
