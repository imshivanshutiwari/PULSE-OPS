from pathlib import Path
from typing import Optional
import pandas as pd
from feature_store.offline_store import OfflineStore
from feature_store.online_store import OnlineStore
from utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    def __init__(self):
        self.offline = OfflineStore()
        self.online = OnlineStore()

    def compute_features(self, dataset_name: str, feature_cols: list = None) -> pd.DataFrame:
        return self.offline.get_historical_features(dataset_name, feature_cols or [])

    def materialize_to_online(self, dataset_name: str) -> int:
        df = self.offline.get_historical_features(dataset_name, [])
        self.online._load_dataset(dataset_name)
        count = len(self.online._cache.get(dataset_name, {}))
        logger.info(f"Materialized {count} records for {dataset_name}")
        return count

    def get_training_dataset(self, dataset_name: str, target_col: str, feature_cols: list = None) -> tuple:
        df = self.compute_features(dataset_name, feature_cols)
        if target_col in df.columns:
            X = df.drop(columns=[target_col, "event_timestamp"], errors="ignore")
            y = df[target_col]
            return X, y
        return df.drop(columns=["event_timestamp"], errors="ignore"), None
