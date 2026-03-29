from feast import Entity

customer_entity = Entity(
    name="customer_id",
    description="Customer unique identifier",
)

bike_entity = Entity(
    name="record_id",
    description="Bike sharing record identifier",
)
