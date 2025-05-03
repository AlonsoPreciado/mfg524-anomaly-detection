# mfg524-anomaly-detection

Simulation‑based pipeline for real‑time anomaly detection in robotic sensor streams.

## Quick start
```powershell
git clone https://github.com/AlonsoPreciado/mfg524-anomaly-detection.git
cd mfg524-anomaly-detection
python -m venv .venv
.\.venv\Scripts\Activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python synthetic_data_generator.py --samples 1000
pytest