"""5 tests for data fetchers."""
import pytest
from pathlib import Path


def test_uci_adult_correct_shape(adult_df):
    """Adult dataset should have close to 48842 rows and 15 cols."""
    assert adult_df.shape[0] > 30000, f"Expected >30000 rows, got {adult_df.shape[0]}"
    assert adult_df.shape[1] == 15, f"Expected 15 cols, got {adult_df.shape[1]}"


def test_wine_quality_target_range(wine_df):
    """Wine quality scores should be between 3 and 9."""
    assert wine_df["quality"].between(3, 9).all(), "Wine quality out of range [3, 9]"


def test_bike_sharing_temporal_features(bike_df):
    """Bike sharing should have hr, season, temp columns."""
    for col in ["hr", "season", "temp"]:
        assert col in bike_df.columns, f"Column '{col}' missing from bike sharing dataset"


def test_credit_binary_labels(credit_df):
    """German credit should have binary creditworthiness."""
    assert set(credit_df["creditworthiness"].unique()).issubset({0, 1}), (
        f"Non-binary labels: {credit_df['creditworthiness'].unique()}"
    )


def test_uci_adult_no_missing_after_fetch(adult_df):
    """Adult dataset should have no NaN after fetch and parse."""
    assert adult_df.isnull().sum().sum() == 0, "Adult dataset has missing values after fetch"
