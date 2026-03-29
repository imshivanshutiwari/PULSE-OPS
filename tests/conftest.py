"""Pytest fixtures for PULSE-OPS tests."""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def adult_df():
    """Real UCI Adult dataset (cached)."""
    from data.fetchers.uci_fetcher import UCIFetcher

    fetcher = UCIFetcher()
    return fetcher.fetch_adult_dataset()


@pytest.fixture(scope="session")
def wine_df():
    """Real UCI Wine Quality dataset (cached)."""
    from data.fetchers.uci_fetcher import UCIFetcher

    fetcher = UCIFetcher()
    return fetcher.fetch_wine_quality()


@pytest.fixture(scope="session")
def bike_df():
    """Real Bike Sharing dataset (cached)."""
    from data.fetchers.bike_fetcher import BikeSharingFetcher

    fetcher = BikeSharingFetcher()
    return fetcher.load_hourly()


@pytest.fixture(scope="session")
def credit_df():
    """Real German Credit dataset (cached)."""
    from data.fetchers.credit_fetcher import GermanCreditFetcher

    fetcher = GermanCreditFetcher()
    return fetcher.download_and_parse()


@pytest.fixture(scope="session")
def trained_model(adult_df):
    """Trained XGBoost model on adult dataset."""
    from data.processors.feature_engineer import FeatureEngineer
    from data.processors.data_splitter import DataSplitter
    from models.classification.gradient_boosting import XGBoostClassifier

    engineer = FeatureEngineer()
    df = engineer.handle_missing(adult_df.copy())
    df = engineer.encode_categoricals(df)
    target_col = "income"
    splitter = DataSplitter()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df, target_col)

    model = XGBoostClassifier()
    model.build({"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1})
    model.fit(X_train, y_train, X_val, y_val)
    return model, X_test, y_test


@pytest.fixture(scope="session")
def mlflow_client():
    """MLflow client (connects to local or default URI)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        client = MlflowClient(tracking_uri=uri)
        client.search_experiments()
        return client
    except Exception:
        # Fall back to file-based tracking
        import tempfile

        tmp = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{tmp}")
        return MlflowClient(tracking_uri=f"file://{tmp}")


@pytest.fixture(scope="session")
def small_adult_df(adult_df):
    """Small 1000-row subset for fast tests."""
    return adult_df.sample(1000, random_state=42).reset_index(drop=True)


@pytest.fixture
def drift_suite(small_adult_df):
    """A pre-computed drift suite for retraining tests."""
    from data.processors.feature_engineer import FeatureEngineer
    from drift.evidently_detector import EvidentlyDriftDetector
    from data.fetchers.synthetic_drifter import PhysicsBasedDriftSimulator

    engineer = FeatureEngineer()
    df = engineer.handle_missing(small_adult_df.copy())
    df = engineer.encode_categoricals(df)
    num_df = df.select_dtypes(include=["number"]).fillna(0)
    reference = num_df.iloc[:500]
    drifter = PhysicsBasedDriftSimulator()
    current = drifter.simulate_sudden_drift(num_df.iloc[500:], shift_magnitude=3.0)
    detector = EvidentlyDriftDetector()
    common = list(set(reference.columns) & set(current.columns))
    return detector.run_full_suite(reference[common], current[common])
