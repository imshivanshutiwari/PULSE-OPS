from pathlib import Path
from feast import FileSource

_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

adult_source = FileSource(
    path=str(_RAW_DIR / "adult_features.parquet"),
    timestamp_field="event_timestamp",
)

wine_source = FileSource(
    path=str(_RAW_DIR / "wine_features.parquet"),
    timestamp_field="event_timestamp",
)

bike_source = FileSource(
    path=str(_RAW_DIR / "bike_features.parquet"),
    timestamp_field="event_timestamp",
)

credit_source = FileSource(
    path=str(_RAW_DIR / "credit_features.parquet"),
    timestamp_field="event_timestamp",
)
