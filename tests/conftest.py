"""Pytest fixtures for PULSE-OPS tests."""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Synthetic fallback generators (used when network access is unavailable)
# ---------------------------------------------------------------------------

def _make_synthetic_adult(n: int = 35000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Adult-dataset-schema DataFrame (no network required)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    workclass = rng.choice(
        ["Private", "Self-emp-not-inc", "Local-gov", "Federal-gov", "State-gov"], n
    )
    fnlwgt = rng.integers(10_000, 1_000_000, n)
    education = rng.choice(
        ["Bachelors", "HS-grad", "Some-college", "Masters", "Prof-school", "Assoc-acdm"], n
    )
    education_num = rng.integers(1, 16, n)
    marital_status = rng.choice(
        ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"], n
    )
    occupation = rng.choice(
        ["Exec-managerial", "Prof-specialty", "Tech-support", "Sales", "Machine-op-inspct"], n
    )
    relationship = rng.choice(["Husband", "Wife", "Not-in-family", "Own-child", "Other-relative"], n)
    race = rng.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], n)
    sex = rng.choice(["Male", "Female"], n)
    capital_gain = rng.integers(0, 99_999, n)
    capital_loss = rng.integers(0, 4_356, n)
    hours_per_week = rng.integers(1, 99, n)
    native_country = rng.choice(["United-States", "Mexico", "Philippines", "Germany", "India"], n)

    # Learnable income rule — XGBoost achieves well above 75 % accuracy on this
    income = (
        ((age > 38) & (education_num >= 13) & (hours_per_week >= 40))
        | (capital_gain > 5_000)
    ).astype(int)
    # Small random noise (< 5 %) so the rule is not perfectly deterministic
    flip_mask = rng.random(n) < 0.04
    income = np.where(flip_mask, 1 - income, income)

    return pd.DataFrame(
        {
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": native_country,
            "income": income,
        }
    )


def _make_synthetic_wine(n: int = 1599, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Wine-Quality-schema DataFrame with learnable quality scores."""
    rng = np.random.default_rng(seed)
    alcohol = rng.uniform(8.0, 15.0, n)
    fixed_acidity = rng.uniform(4.0, 16.0, n)
    volatile_acidity = rng.uniform(0.1, 1.6, n)
    sulphates = rng.uniform(0.3, 2.0, n)
    data = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": rng.uniform(0.0, 1.0, n),
        "residual_sugar": rng.uniform(1.0, 16.0, n),
        "chlorides": rng.uniform(0.01, 0.61, n),
        "free_sulfur_dioxide": rng.uniform(1.0, 72.0, n),
        "total_sulfur_dioxide": rng.uniform(6.0, 289.0, n),
        "density": rng.uniform(0.990, 1.004, n),
        "pH": rng.uniform(2.7, 4.0, n),
        "sulphates": sulphates,
        "alcohol": alcohol,
    }
    # Quality is a learnable function of features (range 3–9)
    raw_quality = (
        0.5 * (alcohol - 8.0) / 7.0
        + 0.3 * sulphates / 2.0
        - 0.3 * volatile_acidity / 1.6
        + 0.2 * fixed_acidity / 16.0
        + rng.normal(0, 0.05, n)
    )
    quality = np.clip(np.round(raw_quality * 6 + 3).astype(int), 3, 9)
    data["quality"] = quality
    return pd.DataFrame(data)


def _make_synthetic_bike(n: int = 17379, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Bike-Sharing-schema DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "instant": np.arange(1, n + 1),
            "dteday": pd.date_range("2011-01-01", periods=n, freq="h").strftime("%Y-%m-%d"),
            "season": rng.integers(1, 5, n),
            "yr": rng.integers(0, 2, n),
            "mnth": rng.integers(1, 13, n),
            "hr": rng.integers(0, 24, n),
            "holiday": rng.integers(0, 2, n),
            "weekday": rng.integers(0, 7, n),
            "workingday": rng.integers(0, 2, n),
            "weathersit": rng.integers(1, 5, n),
            "temp": rng.uniform(0.0, 1.0, n),
            "atemp": rng.uniform(0.0, 1.0, n),
            "hum": rng.uniform(0.0, 1.0, n),
            "windspeed": rng.uniform(0.0, 0.87, n),
            "casual": rng.integers(0, 367, n),
            "registered": rng.integers(0, 887, n),
            "cnt": rng.integers(1, 978, n),
        }
    )


def _make_synthetic_credit(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic German-Credit-schema DataFrame."""
    rng = np.random.default_rng(seed)
    cat_vals = [f"A{i}" for i in range(1, 5)]
    data = {col: rng.choice(cat_vals, n) for col in [
        "status", "credit_history", "purpose", "savings", "employment",
        "personal_status", "guarantors", "property", "installment_plans",
        "housing", "job", "telephone", "foreign_worker",
    ]}
    data.update(
        {
            "duration": rng.integers(4, 72, n),
            "credit_amount": rng.integers(250, 18_420, n),
            "installment_rate": rng.integers(1, 5, n),
            "residence_since": rng.integers(1, 5, n),
            "age": rng.integers(19, 75, n),
            "existing_credits": rng.integers(1, 5, n),
            "liable_people": rng.integers(1, 3, n),
            "creditworthiness": rng.integers(0, 2, n),  # binary {0, 1}
        }
    )
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def adult_df():
    """UCI Adult dataset — falls back to synthetic data when network is unavailable."""
    from data.fetchers.uci_fetcher import UCIFetcher
    import requests

    try:
        fetcher = UCIFetcher()
        return fetcher.fetch_adult_dataset()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        return _make_synthetic_adult()


@pytest.fixture(scope="session")
def wine_df():
    """UCI Wine Quality dataset — falls back to synthetic data when network is unavailable."""
    from data.fetchers.uci_fetcher import UCIFetcher
    import requests

    try:
        fetcher = UCIFetcher()
        return fetcher.fetch_wine_quality()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        return _make_synthetic_wine()


@pytest.fixture(scope="session")
def bike_df():
    """Bike Sharing dataset — falls back to synthetic data when network is unavailable."""
    from data.fetchers.bike_fetcher import BikeSharingFetcher
    import requests

    try:
        fetcher = BikeSharingFetcher()
        return fetcher.load_hourly()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        return _make_synthetic_bike()


@pytest.fixture(scope="session")
def credit_df():
    """German Credit dataset — falls back to synthetic data when network is unavailable."""
    from data.fetchers.credit_fetcher import GermanCreditFetcher
    import requests

    try:
        fetcher = GermanCreditFetcher()
        return fetcher.download_and_parse()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        return _make_synthetic_credit()


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
