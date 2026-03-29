import time
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


class OnlineStore:
    """Real-time feature serving with in-memory cache."""

    def __init__(self):
        self._cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._request_count = 0
        self._total_latency = 0.0

    def _load_dataset(self, dataset_name: str) -> None:
        if dataset_name in self._cache:
            return
        parquet_map = {
            "adult": RAW_DIR / "adult_features.parquet",
            "wine_quality": RAW_DIR / "wine_features.parquet",
            "bike_sharing": RAW_DIR / "bike_features.parquet",
            "german_credit": RAW_DIR / "credit_features.parquet",
        }
        path = parquet_map.get(dataset_name)
        if path and path.exists():
            df = pd.read_parquet(path)
            self._cache[dataset_name] = {i: row.to_dict() for i, row in df.iterrows()}
            logger.info(
                f"Loaded {dataset_name} into online store: {len(self._cache[dataset_name])} records"
            )

    def get_online_features(self, dataset_name: str, entity_id: int) -> Dict[str, Any]:
        t0 = time.time()
        self._load_dataset(dataset_name)
        features = self._cache.get(dataset_name, {}).get(entity_id, {})
        latency = time.time() - t0
        self._request_count += 1
        self._total_latency += latency
        return features

    def get_mean_latency_ms(self) -> float:
        if self._request_count == 0:
            return 0.0
        return (self._total_latency / self._request_count) * 1000

    def get_stats(self) -> dict:
        return {
            "request_count": self._request_count,
            "mean_latency_ms": self.get_mean_latency_ms(),
            "cached_datasets": list(self._cache.keys()),
        }
