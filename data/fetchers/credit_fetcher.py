import io
from pathlib import Path

import pandas as pd
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

CREDIT_URL = (
    "https://archive.uci.edu/static/public/144/statlog+german+credit+data.zip"
)
RAW_DIR = Path(__file__).parent.parent / "raw"

GERMAN_COLUMNS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status", "guarantors",
    "residence_since", "property", "age", "installment_plans", "housing",
    "existing_credits", "job", "liable_people", "telephone", "foreign_worker",
    "creditworthiness",
]


class GermanCreditFetcher:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _download_raw(self) -> pd.DataFrame:
        import zipfile
        zip_path = self.cache_dir / "german_credit.zip"
        extract_dir = self.cache_dir / "german_credit_raw"
        if not zip_path.exists():
            logger.info("Downloading German Credit dataset...")
            resp = requests.get(CREDIT_URL, timeout=60)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(resp.content)
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        german_file = None
        for f in extract_dir.rglob("german.data"):
            german_file = f
            break
        if german_file is None:
            # fallback: direct URL
            direct_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            resp = requests.get(direct_url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(
                io.StringIO(resp.text),
                sep=" ",
                header=None,
                names=GERMAN_COLUMNS,
            )
            return df
        df = pd.read_csv(str(german_file), sep=" ", header=None, names=GERMAN_COLUMNS)
        return df

    def download_and_parse(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "german_credit.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        df = self._download_raw()
        df["creditworthiness"] = df["creditworthiness"].map({1: 0, 2: 1})
        df.to_csv(cache_path, index=False)
        logger.info(f"German Credit saved: {df.shape}")
        return df

    def encode_categorical(self) -> pd.DataFrame:
        df = self.download_and_parse()
        cat_cols = df.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            df[col] = df[col].astype("category").cat.codes
        return df
