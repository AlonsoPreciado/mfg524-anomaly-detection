"""
pipeline.py – end‑to‑end streaming pipeline.

Example 1  (quick, all defaults – pull from Mongo, Z‑score):
    python pipeline.py

Example 2  (Isolation Forest with saved model, 200‑row batches):
    python pipeline.py --detector iso --model models/iso_forest.joblib --batch 200
"""
from pathlib import Path
import argparse, sys, time
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from data_loader import iter_batches
from preprocess   import preprocess
from detector     import ZScoreDetector, IsoForestDetector, load_model, save_model

# ----------------------------- helpers ----------------------------------
def get_detector(args):
    if args.detector == "z":
        return ZScoreDetector(threshold=args.threshold)
    elif args.detector == "iso":
        if args.model and Path(args.model).exists():
            print(f"🔹 Loading model from {args.model}")
            return load_model(args.model)
        else:
            print("🔹 Training fresh Isolation Forest …")
            det = IsoForestDetector(contamination=args.contamination)
            # need training data – grab first N rows
            df_train = pd.concat([preprocess(b) for _, b in zip(range(args.train_batches),
                                                                iter_batches(args.batch))])
            det.fit(df_train)
            if args.model:
                save_model(det, args.model)
                print(f"   ✔ Model saved to {args.model}")
            return det
    else:
        raise ValueError("Unknown detector")

def output_anomalies(df: pd.DataFrame, flags, out_coll=None):
    """Print to stdout and optionally write docs to Mongo."""
    anomalies = df.loc[flags]
    if anomalies.empty:
        return
    print(f"⚠ Anomalies detected: {len(anomalies)} rows (timestamps: "
          f"{list(anomalies.index[:3])}{' …' if len(anomalies)>3 else ''})")
    if out_coll is not None:
        out_docs = anomalies.reset_index().to_dict("records")
        out_coll.insert_many(out_docs)

# ------------------------------- main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    # streaming params
    ap.add_argument("--batch", type=int, default=500, help="batch size")
    # detector params
    ap.add_argument("--detector", choices=["z", "iso"], default="z")
    ap.add_argument("--threshold", type=float, default=3.5,
                    help="z‑score sigma threshold (detector=z)")
    ap.add_argument("--contamination", type=float, default=0.02,
                    help="expected anomaly fraction (detector=iso)")
    ap.add_argument("--model", help="path to save / load Isolation Forest")
    ap.add_argument("--train-batches", type=int, default=3,
                    help="batches used to train Isolation Forest when no model file exists")
    # mongo output
    ap.add_argument("--sink", action="store_true",
                    help="write anomalies to MongoDB collection robot_sensors.anomalies")
    args = ap.parse_args()

    # optional sink collection
    sink_coll = None
    if args.sink:
        client = MongoClient("mongodb://localhost:27017")
        sink_coll = client["robot_sensors"]["anomalies"]
        sink_coll.delete_many({})          # clear previous run

    det = get_detector(args)

    print("📡  Streaming…  (Ctrl‑C to stop)")
    try:
        for batch in tqdm(iter_batches(args.batch)):
            clean = preprocess(batch)
            flags = det.predict(clean)
            output_anomalies(clean, flags, sink_coll)
            # simulate real‑time pacing (optional)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n⏹  Stopped by user.")

if __name__ == "__main__":
    main()
