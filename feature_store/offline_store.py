from datetime import datetime
from typing import List
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


class OfflineStore:
    """Historical feature retrieval from parquet files."""

    def get_historical_features(
        self,
        dataset_name: str,
        feature_cols: List[str],
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> pd.DataFrame:
        parquet_map = {
            "adult": RAW_DIR / "adult_features.parquet",
            "wine_quality": RAW_DIR / "wine_features.parquet",
            "bike_sharing": RAW_DIR / "bike_features.parquet",
            "german_credit": RAW_DIR / "credit_features.parquet",
        }
        path = parquet_map.get(dataset_name)
        if path is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}. Run 'make fetch' first.")
        df = pd.read_parquet(path)
        if feature_cols:
            available = [c for c in feature_cols if c in df.columns]
            if "event_timestamp" in df.columns:
                available.append("event_timestamp")
            df = df[available]
        if start_time and "event_timestamp" in df.columns:
            df = df[pd.to_datetime(df["event_timestamp"]) >= start_time]
        if end_time and "event_timestamp" in df.columns:
            df = df[pd.to_datetime(df["event_timestamp"]) <= end_time]
        logger.info(f"Historical features for {dataset_name}: {df.shape}")
        return df

    def get_feature_stats(self, dataset_name: str) -> dict:
        df = self.get_historical_features(dataset_name, [])
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
        }
