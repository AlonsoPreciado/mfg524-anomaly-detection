"""
app.py – Streamlit dashboard for live sensor monitoring & anomaly overlay.
Run with:
    streamlit run app.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from streamlit_autorefresh import st_autorefresh

# ── Mongo connection settings ──────────────────────────────────────────
MONGO_URI   = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME     = "robot_sensors"
RAW_COLL    = "imu_lidar"
ANOM_COLL   = "anomalies"

REFRESH_MS  = 3000   # dashboard auto‑refresh interval (ms)
WINDOW_SEC  = 20     # live window length (seconds)

# ── Connect to MongoDB ─────────────────────────────────────────────────
cli   = MongoClient(MONGO_URI)
raw_c = cli[DB_NAME][RAW_COLL]
ano_c = cli[DB_NAME][ANOM_COLL]

# ── Streamlit page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sensor Anomaly Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Real‑time Sensor Streams")
st_autorefresh(interval=REFRESH_MS, key="autorefresh")

# ── Fetch the most‑recent WINDOW_SEC seconds of data ───────────────────
latest = raw_c.find_one(sort=[("timestamp", -1)])
if not latest:
    st.info("Waiting for data …")
    st.stop()

max_t  = latest["timestamp"]
min_t  = max_t - WINDOW_SEC
docs   = list(raw_c.find({"timestamp": {"$gte": min_t}}))
df     = pd.DataFrame(docs)

# Convert timestamp (float seconds) → index
df["time_s"] = df["timestamp"]          # keep a seconds column
df.set_index("time_s", inplace=True)
df.index.name = "seconds"

# ── Join anomaly flags from sink collection ────────────────────────────
anom = pd.DataFrame(ano_c.find({"timestamp": {"$gte": min_t}}))
if not anom.empty:
    df["is_anomaly"] = df.index.isin(anom["timestamp"])
else:
    df["is_anomaly"] = False

# ── Signal selector UI ─────────────────────────────────────────────────
numeric_cols = [
    "accel_x", "accel_y", "accel_z",
    "gyro_x",  "gyro_y",  "gyro_z",
    "lidar_range",
]
default_show = ["accel_x", "accel_y", "accel_z", "lidar_range"]

signals = st.multiselect(
    "Signals to display",
    options=numeric_cols,
    default=default_show,
)

if not signals:
    st.warning("Select at least one signal to plot.")
    st.stop()

# ── Plot selected signals (two columns) ────────────────────────────────
cols = st.columns(2)

for idx, sig in enumerate(signals):
    fig = px.line(
        df,
        x=df.index,
        y=sig,
        title=sig,
        labels={"x": "seconds", sig: sig},
    )
    # overlay anomalies
    anom_idx = df.index[df["is_anomaly"]]
    fig.add_scatter(
        x=anom_idx,
        y=df.loc[anom_idx, sig],
        mode="markers",
        marker=dict(size=6, color="red"),
        name="anomaly",
    )
    cols[idx % 2].plotly_chart(fig, use_container_width=True)

# ── Recent anomalies table ─────────────────────────────────────────────
st.markdown("---")
st.subheader("⚠ Recent anomalies")

if df["is_anomaly"].any():
    st.dataframe(
        df[df["is_anomaly"]][signals].tail(10),
        height=275,
    )
else:
    st.write("None in the current window.")
