"""
preprocess.py – functions that clean and featurize sensor DataFrames.

Expected input:
    index: Timedelta or Datetime (from data_loader)
    columns: accel_x, accel_y, accel_z,
             gyro_x,  gyro_y,  gyro_z,
             lidar_range, is_anomaly
"""
from typing import Iterable, Generator, List
import pandas as pd
import numpy as np
from scipy import signal

# ---------- configurable parameters ----------
ROLL_WINDOW_S  = 0.5    # rolling window length in seconds
ROLL_STEP_S    = 0.02   # original sample step (50 Hz) for reference
DETREND_ORDER  = 1      # 0 = remove mean, 1 = remove linear trend
STAT_COLUMNS: List[str] = [
    "accel_x","accel_y","accel_z",
    "gyro_x","gyro_y","gyro_z","lidar_range"
]
# ---------------------------------------------

def detrend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with each numeric column detrended (SciPy)."""
    out = df.copy()
    for col in STAT_COLUMNS:
        out[col] = signal.detrend(out[col].values, type='linear' if DETREND_ORDER==1 else 'constant')
    return out

def add_rolling_features(df: pd.DataFrame,
                         window_s: float = ROLL_WINDOW_S) -> pd.DataFrame:
    """Add rolling mean & std columns for each signal."""
    win = int(round(window_s / ROLL_STEP_S))
    rolled = df[STAT_COLUMNS].rolling(window=win, center=False)
    means  = rolled.mean().add_suffix("_roll_mean")
    stds   = rolled.std().add_suffix("_roll_std")
    return pd.concat([df, means, stds], axis=1)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: detrend -> rolling features -> drop NaNs from rolling window.
    Returns a new DataFrame; does not mutate the input.
    """
    clean = detrend_signals(df)
    enriched = add_rolling_features(clean)
    enriched.dropna(inplace=True)
    return enriched

# Convenience wrapper for streaming use:
def preprocess_batches(
        batches: Iterable[pd.DataFrame]               # accept any iterable
) -> Generator[pd.DataFrame, None, None]:             
    """Apply preprocess() to each batch in an iterable."""
    for b in batches:
        yield preprocess(b)
