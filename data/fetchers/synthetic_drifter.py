from typing import List
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsBasedDriftSimulator:
    """Simulate realistic data drift based on REAL dataset statistics."""

    def simulate_covariate_shift(
        self,
        df: pd.DataFrame,
        feature: str,
        shift_magnitude: float = 1.0,
    ) -> pd.DataFrame:
        drifted = df.copy()
        mu = df[feature].mean()
        sigma = df[feature].std()
        drifted[feature] = np.random.normal(
            loc=mu + shift_magnitude * sigma,
            scale=sigma,
            size=len(df),
        )
        logger.info(
            f"Covariate shift on '{feature}': mean {mu:.3f} → {drifted[feature].mean():.3f}"
        )
        return drifted

    def simulate_concept_drift(
        self,
        df: pd.DataFrame,
        target_col: str,
        noise_scale: float = 0.1,
    ) -> pd.DataFrame:
        drifted = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)
        for col in num_cols[:3]:
            corr = abs(df[col].corr(df[target_col])) if target_col in df.columns else 0.1
            noise = np.random.normal(0, noise_scale * corr, size=len(df))
            drifted[target_col] = (drifted[target_col] + noise).clip(
                df[target_col].min(), df[target_col].max()
            )
        return drifted

    def simulate_gradual_drift(
        self,
        df: pd.DataFrame,
        feature: str,
        n_steps: int = 10,
        shift_magnitude: float = 2.0,
    ) -> List[pd.DataFrame]:
        steps = []
        mu = df[feature].mean()
        sigma = df[feature].std()
        for i in range(n_steps):
            alpha = i / (n_steps - 1)
            drifted = df.copy()
            new_mean = mu + alpha * shift_magnitude * sigma
            drifted[feature] = np.random.normal(new_mean, sigma, size=len(df))
            steps.append(drifted)
        logger.info(f"Gradual drift simulated: {n_steps} steps on '{feature}'")
        return steps

    def simulate_sudden_drift(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        shift_magnitude: float = 3.0,
    ) -> pd.DataFrame:
        drifted = df.copy()
        cols = features or df.select_dtypes(include=[np.number]).columns.tolist()[:5]
        for col in cols:
            mu = df[col].mean()
            sigma = df[col].std()
            drifted[col] = np.random.normal(mu + shift_magnitude * sigma, sigma, size=len(df))
        logger.info(f"Sudden drift applied to {len(cols)} features")
        return drifted
