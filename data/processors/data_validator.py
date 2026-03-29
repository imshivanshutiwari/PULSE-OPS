import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    def __init__(self, missing_threshold: float = 0.2, duplicate_threshold: float = 0.05):
        self.missing_threshold = missing_threshold
        self.duplicate_threshold = duplicate_threshold

    def check_missing(self, df: pd.DataFrame) -> dict:
        missing = df.isnull().mean()
        violating = missing[missing > self.missing_threshold].to_dict()
        passed = len(violating) == 0
        return {
            "passed": passed,
            "missing_rate": missing.to_dict(),
            "violating_columns": violating,
        }

    def check_duplicates(self, df: pd.DataFrame) -> dict:
        dup_rate = df.duplicated().mean()
        passed = dup_rate <= self.duplicate_threshold
        return {"passed": passed, "duplicate_rate": float(dup_rate)}

    def check_schema(self, df: pd.DataFrame, expected_columns: list) -> dict:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        passed = len(missing_cols) == 0
        return {
            "passed": passed,
            "missing_columns": list(missing_cols),
            "extra_columns": list(extra_cols),
        }

    def validate(self, df: pd.DataFrame, expected_columns: list = None) -> dict:
        results = {
            "missing_check": self.check_missing(df),
            "duplicate_check": self.check_duplicates(df),
        }
        if expected_columns:
            results["schema_check"] = self.check_schema(df, expected_columns)
        results["overall_passed"] = all(v["passed"] for v in results.values())
        return results
