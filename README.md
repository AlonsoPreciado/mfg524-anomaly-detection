# mfg524-anomaly-detection

This repo implements a **real‑time anomaly‑detection pipeline** for simulated
robotic‑sensor streams (IMU + LiDAR). 


1. **Data generator** – creates noisy IMU & LiDAR runs with injected spikes.  
2. **Mongo loader** – fast bulk‐insert of JSON sensor docs.  
3. **Pipeline** – streams mini‑batches, pre‑processes, and applies  
   *Z‑score* or *Isolation Forest* detectors.  
4. **Dashboard** – auto‑refreshing Plotly charts with anomaly markers.

---

## Setup 
```powershell
git clone https://github.com/AlonsoPreciado/mfg524-anomaly-detection.git
cd mfg524-anomaly-detection

# 1. Python env
python -m venv .venv
.\.venv\Scripts\Activate       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Start MongoDB via Docker
docker run -d --name mongodb -p 27017:27017 mongo:latest    # one‑time

# 3. Generate & load sample data
python synthetic_data_generator.py --samples 5000
python mongo_loader.py --files data\*.json
