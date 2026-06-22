# test_toolbox

# Thermal Vital Signs Toolbox

Contactless estimation of heart rate (HR) and respiration rate (RR)
from thermal video using facial region analysis.

Developed as part of a university project (Projektseminar, 4th semester).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place YOLO model
#    Copy YOLOv11_TFL_252.pt into models/

# 3. Configure dataset path
#    Edit configs/bp4d.yaml or configs/npz.yaml

# 4. Run
python main.py

## Project Structure

```
thermal_vital_signs/
│
├── main.py                      ← Entry point
├── yolo.py                      ← YOLO wrapper (supervisor, do not modify)
│
├── configs/
│   ├── run_config.yaml          ← Pipeline settings
│   ├── bp4d.yaml                ← BP4D+ dataset paths
│   └── npz.yaml                 ← NPZ dataset paths
│
├── data/
│   ├── bp4d_loader.py           ← BP4D+ loader (.wmv + .txt)
│   └── npz_loader.py            ← NPZ loader (.npz files)
│
├── models/
│   └── YOLOv11_TFL_252.pt      ← Trained YOLO model (not in repo)
│
├── preprocessing/
│   ├── roi_extraction.py        ← Keypoints → ROI boxes
│   ├── yolo_keypoints.py        ← 54 landmark definitions
│   ├── signal_extraction.py     ← Frames + ROIs → temperature signal
│   └── peak_extraction.py       ← Signal → BPM (filter + FFT)
│
├── methods/
│   ├── thermal_mean.py          ← Baseline: mean ROI temperature
│   ├── ica.py                   ← ICA-based source separation
│   └── garbey.py                ← Vessel-line FFT (Garbey 2007)
│
├── evaluation/
│   ├── metrics.py               ← MAE, RMSE, Pearson
│   ├── bland_altman.py          ← Bland-Altman & scatter plots
│   └── results_table.py         ← Comparison table (CSV + PDF)
│
├── utils/
│   └── yolo_processing.py       ← YOLO batch processing
│
└── results/                     ← Output (auto-created)
```
