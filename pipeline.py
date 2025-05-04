"""
pipeline.py – stream data  ➜  preprocess ➜  detect ➜  (optional) write anomalies.

Quick start (default Z‑score):
    python pipeline.py --batch 800

Isolation Forest + sink collection:
    python pipeline.py --detector iso --sink --batch 800 --model models/iso.joblib
"""
from pathlib import Path
import argparse, time
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from data_loader import iter_batches
from preprocess  import preprocess
from detector    import ZScoreDetector, IsoForestDetector, load_model, save_model

# ───────────────────────────────────────── Detector helpers ────────────
def build_detector(args):
    if args.detector == "z":
        return ZScoreDetector(threshold=args.threshold)

    if args.model and Path(args.model).exists():
        print(f"🔹 Loading Isolation Forest from {args.model}")
        return load_model(args.model)

    print("🔹 Training new Isolation Forest …")
    det = IsoForestDetector(contamination=args.contamination, random_state=42)
    batches = []
    for _ in range(args.train_batches):
        try:
            batches.append(preprocess(next(iter_batches(args.batch))))
        except StopIteration:
            break
    train_df = pd.concat(batches, ignore_index=True)
    det.fit(train_df)
    if args.model:
        save_model(det, args.model)
        print(f"   ✔ Model saved to {args.model}")
    return det

# ───────────────────────────────────────── Anomaly sink ────────────────
def write_anomalies(df: pd.DataFrame, flags, coll=None):
    if coll is None or not flags.any():
        return
    docs = df.loc[flags].reset_index()
    # convert Timedelta to float seconds for MongoDB
    docs["timestamp"] = docs["timestamp"].dt.total_seconds()
    coll.insert_many(docs.to_dict("records"))

# ───────────────────────────────────────── Main loop ───────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--detector", choices=["z", "iso"], default="z")
    ap.add_argument("--threshold", type=float, default=3.5)
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--model", help="path to save / load Isolation Forest")
    ap.add_argument("--train-batches", type=int, default=3)
    ap.add_argument("--sink", action="store_true",
                    help="write anomalies to robot_sensors.anomalies")
    args = ap.parse_args()

    sink_coll = None
    if args.sink:
        sink_coll = MongoClient("mongodb://localhost:27017") \
                    ["robot_sensors"]["anomalies"]
        sink_coll.delete_many({})  # clear previous

    detector = build_detector(args)
    print("📡  Streaming…  (Ctrl‑C to stop)")

    try:
        for batch in tqdm(iter_batches(args.batch)):
            clean = preprocess(batch)
            flags = detector.predict(clean)
            if flags.any():
                ts_list = clean.index[flags][:3]
                print(f"⚠ {flags.sum():>3} anomalies "
                      f"@ {list(ts_list)}{' …' if flags.sum()>3 else ''}")
            write_anomalies(clean, flags, sink_coll)
            time.sleep(0.02)  # pacing for demo
    except KeyboardInterrupt:
        print("\n⏹  Stopped.")

if __name__ == "__main__":
    main()
