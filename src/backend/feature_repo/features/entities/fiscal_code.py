from feast import Entity
from feast.value_type import ValueType

fiscal_code = Entity(
    name="fiscal_code",
    join_keys=["fiscal_code"],
    description="Fiscal code entity for feature lookup",
    value_type=ValueType.STRING,
)
