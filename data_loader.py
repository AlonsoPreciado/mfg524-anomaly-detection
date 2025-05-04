"""
data_loader.py – utilities to fetch sensor data from MongoDB into Pandas DataFrames.
"""
from typing import Generator
from datetime import timedelta
import pandas as pd
from pymongo import MongoClient

DEFAULT_URI  = "mongodb://localhost:27017"
DEFAULT_DB   = "robot_sensors"
DEFAULT_COLL = "imu_lidar"

def _make_df(raw_docs: list[dict]) -> pd.DataFrame:
    """Convert list of dicts into a clean DataFrame."""
    if not raw_docs:
        return pd.DataFrame()
    df = pd.DataFrame(raw_docs)
    df.drop(columns=["_id"], inplace=True, errors="ignore")
    # optional: convert timestamp (seconds) → pandas Timedelta index
    df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    return df

def load_all(
    uri: str = DEFAULT_URI,
    db: str  = DEFAULT_DB,
    coll: str = DEFAULT_COLL
) -> pd.DataFrame:
    """Load entire collection into a DataFrame (use only for small-ish datasets)."""
    client = MongoClient(uri)
    cursor = client[db][coll].find({})
    docs   = list(cursor)
    return _make_df(docs)

def iter_batches(
    batch_size: int = 500,
    uri: str = DEFAULT_URI,
    db: str  = DEFAULT_DB,
    coll: str = DEFAULT_COLL
) -> Generator[pd.DataFrame, None, None]:
    """Yield DataFrames of `batch_size` rows indefinitely."""
    client = MongoClient(uri)
    cursor = client[db][coll].find({}).batch_size(batch_size)
    buffer = []
    for doc in cursor:
        buffer.append(doc)
        if len(buffer) >= batch_size:
            yield _make_df(buffer)
            buffer.clear()
    if buffer:
        yield _make_df(buffer)
