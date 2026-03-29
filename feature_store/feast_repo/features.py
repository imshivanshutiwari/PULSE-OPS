from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64
from feature_store.feast_repo.entities import customer_entity, bike_entity
from feature_store.feast_repo.data_sources import (
    adult_source,
    wine_source,
    bike_source,
)

income_features = FeatureView(
    name="income_features",
    entities=[customer_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="hours_per_week", dtype=Int64),
        Field(name="capital_gain", dtype=Float32),
        Field(name="capital_loss", dtype=Float32),
        Field(name="education_num", dtype=Int64),
    ],
    source=adult_source,
)

wine_quality_features = FeatureView(
    name="wine_quality_features",
    entities=[customer_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="fixed_acidity", dtype=Float32),
        Field(name="volatile_acidity", dtype=Float32),
        Field(name="citric_acid", dtype=Float32),
        Field(name="residual_sugar", dtype=Float32),
        Field(name="density", dtype=Float32),
        Field(name="pH", dtype=Float32),
        Field(name="sulphates", dtype=Float32),
        Field(name="alcohol", dtype=Float32),
    ],
    source=wine_source,
)

bike_features = FeatureView(
    name="bike_features",
    entities=[bike_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="season", dtype=Int64),
        Field(name="hr", dtype=Int64),
        Field(name="temp", dtype=Float32),
        Field(name="atemp", dtype=Float32),
        Field(name="hum", dtype=Float32),
        Field(name="windspeed", dtype=Float32),
    ],
    source=bike_source,
)
