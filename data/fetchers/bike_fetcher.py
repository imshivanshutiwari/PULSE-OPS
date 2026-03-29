import zipfile
from pathlib import Path

import pandas as pd
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

BIKE_URL = "https://archive.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW_DIR = Path(__file__).parent.parent / "raw"


class BikeSharingFetcher:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_and_extract(self) -> str:
        zip_path = self.cache_dir / "bike_sharing.zip"
        extract_dir = self.cache_dir / "bike_sharing"
        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info("Bike Sharing already extracted")
            return str(extract_dir)
        logger.info("Downloading Bike Sharing dataset...")
        resp = requests.get(BIKE_URL, timeout=120)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        logger.info(f"Extracted to {extract_dir}")
        return str(extract_dir)

    def load_hourly(self) -> pd.DataFrame:
        cache_path = self.cache_dir / "bike_sharing_hourly.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path)
        extract_dir = self.download_and_extract()
        hour_file = None
        for f in Path(extract_dir).rglob("hour.csv"):
            hour_file = f
            break
        if hour_file is None:
            raise FileNotFoundError("hour.csv not found in extracted files")
        df = pd.read_csv(hour_file)
        df.to_csv(cache_path, index=False)
        logger.info(f"Bike Sharing hourly: {df.shape}")
        return df
