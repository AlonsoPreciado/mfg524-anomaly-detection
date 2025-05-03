import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

def make_dataset(n:int, anomaly_rate:float, seed:int|None=None)->pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts  = np.arange(n)*0.02
    accel = 0.5*np.sin(0.2*ts)[:,None] + rng.normal(0,0.05,(n,3))
    gyro  = 0.3*np.cos(0.15*ts)[:,None] + rng.normal(0,0.03,(n,3))
    lidar = 1.0 + rng.normal(0,0.01,(n,1))
    df = pd.DataFrame(
        np.hstack([ts[:,None],accel,gyro,lidar]),
        columns=["timestamp","accel_x","accel_y","accel_z",
                 "gyro_x","gyro_y","gyro_z","lidar_range"],
    )
    anomalies = rng.choice(n,int(n*anomaly_rate),replace=False)
    df["is_anomaly"] = 0
    df.loc[anomalies,"accel_x"] += rng.normal(3,0.5,anomalies.size)
    df.loc[anomalies,"is_anomaly"] = 1
    return df

def save_files(df, out_dir=Path("data")):
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(out_dir/f"sensor_{stamp}.csv", index=False)
    df.to_json(out_dir/f"sensor_{stamp}.json", orient="records", lines=True)
    print("✔  Files saved to", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--anomaly-rate", type=float, default=0.01)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    save_files(make_dataset(args.samples, args.anomaly_rate, args.seed))
