import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le
        return df

    def scale_numerics(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in num_cols:
            num_cols.remove(target_col)
        df[num_cols] = self.scaler.fit_transform(df[num_cols])
        return df

    def add_interaction_features(self, df: pd.DataFrame, col_pairs: list) -> pd.DataFrame:
        df = df.copy()
        for c1, c2 in col_pairs:
            if c1 in df.columns and c2 in df.columns:
                df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
        return df

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    def add_event_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        import datetime
        df = df.copy()
        base = datetime.datetime(2023, 1, 1)
        df["event_timestamp"] = pd.date_range(base, periods=len(df), freq="1min")
        return df
