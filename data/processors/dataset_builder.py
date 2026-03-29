import argparse
import sys
from pathlib import Path
import pandas as pd
from data.fetchers.uci_fetcher import UCIFetcher
from data.fetchers.bike_fetcher import BikeSharingFetcher
from data.fetchers.credit_fetcher import GermanCreditFetcher
from data.processors.feature_engineer import FeatureEngineer
from data.processors.data_validator import DataValidator
from utils.logger import get_logger

logger = get_logger(__name__)
RAW_DIR = Path(__file__).parent.parent / "raw"


class DatasetBuilder:
    def __init__(self):
        self.uci = UCIFetcher()
        self.bike = BikeSharingFetcher()
        self.credit = GermanCreditFetcher()
        self.engineer = FeatureEngineer()
        self.validator = DataValidator()

    def build_adult(self) -> pd.DataFrame:
        df = self.uci.fetch_adult_dataset()
        df = self.engineer.handle_missing(df)
        df = self.engineer.encode_categoricals(df)
        df = self.engineer.add_event_timestamp(df)
        parquet_path = RAW_DIR / "adult_features.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Adult features saved: {parquet_path}")
        return df

    def build_wine(self) -> pd.DataFrame:
        df = self.uci.fetch_wine_quality()
        df = self.engineer.handle_missing(df)
        df = self.engineer.add_event_timestamp(df)
        parquet_path = RAW_DIR / "wine_features.parquet"
        df.to_parquet(parquet_path, index=False)
        return df

    def build_bike(self) -> pd.DataFrame:
        df = self.bike.load_hourly()
        df = self.engineer.handle_missing(df)
        df = self.engineer.add_event_timestamp(df)
        parquet_path = RAW_DIR / "bike_features.parquet"
        df.to_parquet(parquet_path, index=False)
        return df

    def build_credit(self) -> pd.DataFrame:
        df = self.credit.encode_categorical()
        df = self.engineer.add_event_timestamp(df)
        parquet_path = RAW_DIR / "credit_features.parquet"
        df.to_parquet(parquet_path, index=False)
        return df

    def build_all(self) -> dict:
        return {
            "adult": self.build_adult(),
            "wine_quality": self.build_wine(),
            "bike_sharing": self.build_bike(),
            "german_credit": self.build_credit(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Fetch and build all datasets")
    args = parser.parse_args()
    if args.fetch:
        builder = DatasetBuilder()
        datasets = builder.build_all()
        for name, df in datasets.items():
            logger.info(f"{name}: {df.shape}")
