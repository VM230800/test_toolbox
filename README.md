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
