"""
detector.py – anomaly‑detection utilities.

Usage example:
    >>> from data_loader import iter_batches
    >>> from preprocess import preprocess
    >>> from detector    import ZScoreDetector
    >>> det = ZScoreDetector(threshold=3.5)
    >>> raw = next(iter_batches(batch_size=1000))
    >>> clean = preprocess(raw)
    >>> preds = det.predict(clean)
    >>> clean.loc[preds, :].head()   # rows flagged as anomalies
"""
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

NUMERIC_COLS = [
    "accel_x","accel_y","accel_z",
    "gyro_x","gyro_y","gyro_z",
    "lidar_range"
]

# ------------------------------------------------------------------------
class BaseDetector(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return a boolean array the same length as df: True = anomaly"""
        ...

# ------------------------------------------------------------------------
class ZScoreDetector(BaseDetector):
    """
    Flags a row as anomalous if *any* numeric column exceeds ±threshold σ.
    No training required.
    """
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
        # rolling means/std could also be used; here we assume df is already scaled.
    def fit(self, df: pd.DataFrame) -> None:
        pass    # nothing to fit
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        z = np.abs((df[NUMERIC_COLS] - df[NUMERIC_COLS].mean())
                   / df[NUMERIC_COLS].std(ddof=0))
        return (z > self.threshold).any(axis=1).values

# ------------------------------------------------------------------------
class IsoForestDetector(BaseDetector):
    """
    Unsupervised Isolation Forest.
      • fit() trains the model on reference (mostly‑normal) data.
      • predict() returns anomalies as True.
    """
    def __init__(self, contamination: float = 0.02, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
    def fit(self, df: pd.DataFrame) -> None:
        self.model.fit(df[NUMERIC_COLS])
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # sklearn returns -1 for anomaly, +1 for normal
        preds = self.model.predict(df[NUMERIC_COLS])
        return preds == -1

# ------------------- persistence helpers --------------------------------
def save_model(detector: IsoForestDetector, path: str | Path) -> None:
    joblib.dump(detector.model, path)

def load_model(path: str | Path) -> IsoForestDetector:
    model = joblib.load(path)
    det   = IsoForestDetector()
    det.model = model
    return det
