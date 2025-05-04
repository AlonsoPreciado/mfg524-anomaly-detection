"""
mongo_loader.py — bulk‑insert JSON sensor files into MongoDB.

Usage example:
    python mongo_loader.py --uri mongodb://localhost:27017 ^
                           --files data\*.json
"""
import argparse, glob, json
from pathlib import Path
from pymongo import MongoClient

def load_files(client, db_name, coll_name, files):
    coll = client[db_name][coll_name]
    for fname in files:
        print(f"⏳  Loading {fname} ...")
        with open(fname, "r", encoding="utf‑8") as f:
            docs = [json.loads(line) for line in f]
        if docs:
            result = coll.insert_many(docs)
            print(f"   ✔  Inserted {len(result.inserted_ids)} docs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri",  default="mongodb://localhost:27017",
                    help="MongoDB connection string")
    ap.add_argument("--db",   default="robot_sensors")
    ap.add_argument("--coll", default="imu_lidar")
    ap.add_argument("--files", nargs="+", required=True,
                    help="JSON file(s) or wildcard pattern (e.g., data\\*.json)")
    args = ap.parse_args()

    paths = [p for pattern in args.files for p in glob.glob(pattern)]
    if not paths:
        raise SystemExit("No matching files. Check the --files pattern.")
    client = MongoClient(args.uri)
    load_files(client, args.db, args.coll, paths)
    print("Done.")
