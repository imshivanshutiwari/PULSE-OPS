import io
import os
from pathlib import Path

import pandas as pd
import requests
from sklearn.datasets import load_iris
from utils.logger import get_logger

logger = get_logger(__name__)

ADULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

RAW_DIR = Path(__file__).parent.parent / "raw"


class UCIFetcher:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_adult_dataset(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "adult.csv"
        if cache_path.exists():
            logger.info("Loading Adult dataset from cache")
            return pd.read_csv(cache_path)
        logger.info("Downloading Adult dataset from UCI...")
        resp = requests.get(ADULT_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=ADULT_COLUMNS,
            na_values=" ?",
            skipinitialspace=True,
        )
        df.dropna(inplace=True)
        df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1})
        df.to_csv(cache_path, index=False)
        logger.info(f"Adult dataset saved: {df.shape}")
        return df

    def fetch_wine_quality(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "wine_quality.csv"
        if cache_path.exists():
            logger.info("Loading Wine Quality from cache")
            return pd.read_csv(cache_path)
        logger.info("Downloading Wine Quality from UCI...")
        resp = requests.get(WINE_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep=";")
        df.columns = [c.replace(" ", "_") for c in df.columns]
        df.to_csv(cache_path, index=False)
        logger.info(f"Wine Quality saved: {df.shape}")
        return df

    def fetch_iris(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "iris.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        iris = load_iris(as_frame=True)
        df = iris.frame
        df.to_csv(cache_path, index=False)
        logger.info(f"Iris saved: {df.shape}")
        return df

    def cache_all(self, path: str = None) -> dict:
        paths = {}
        paths["adult"] = str(self.cache_dir / "adult.csv")
        paths["wine_quality"] = str(self.cache_dir / "wine_quality.csv")
        paths["iris"] = str(self.cache_dir / "iris.csv")
        self.fetch_adult_dataset()
        self.fetch_wine_quality()
        self.fetch_iris()
        return paths
