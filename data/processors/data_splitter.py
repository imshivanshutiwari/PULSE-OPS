from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    def __init__(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        val_fraction = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=self.random_state
        )
        logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def temporal_split(
        self, df: pd.DataFrame, target_col: str, time_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        df = df.sort_values(time_col)
        split_idx = int(len(df) * (1 - self.test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return (
            train.drop(columns=[target_col]),
            test.drop(columns=[target_col]),
            train[target_col],
            test[target_col],
        )
