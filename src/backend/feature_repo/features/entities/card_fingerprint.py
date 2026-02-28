from feast import Entity
from feast.value_type import ValueType

card_fingerprint = Entity(
    name="card_fingerprint",
    join_keys=["card_fingerprint"],
    description="Card fingerprint entity for feature lookup",
    value_type=ValueType.STRING,
)
