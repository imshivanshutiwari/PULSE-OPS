from feast import Entity
from feast.value_type import ValueType

customer_entity = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Customer unique identifier",
)

bike_entity = Entity(
    name="record_id",
    value_type=ValueType.INT64,
    description="Bike sharing record identifier",
)
