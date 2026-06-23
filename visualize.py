"""
visualize.py – Keypoints + ROIs auf Frames anzeigen
Speichert: Einzelbild + Video
"""

import numpy as np
import yaml
import cv2
import os
from data.bp4d_loader import BP4DDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois

# ── Config laden ──
with open("configs/run_config.yaml") as f:
    config = yaml.safe_load(f)
with open("configs/bp4d.yaml") as f:
    ds_config = yaml.safe_load(f)

# ── Daten laden ──
dataset = BP4DDataset(
    root_dir=ds_config["root_dir"],
    subjects=["F001"],
    tasks=["T1"],
    fps=25,
)
sample = dataset[0]
frames = sample["frames"][:100]

# ── YOLO ──
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames,
    model_path=det["model_path"],
    target_size=tuple(det["target_size"]),
    padding=det.get("padding", 50),
)

# ── ROI Farben ──
ROI_COLORS = {
    "forehead":    (0, 255, 0),    # Grün
    "left_cheek":  (255, 0, 0),    # Blau
    "right_cheek": (255, 0, 0),    # Blau
    "nose":        (0, 255, 255),  # Gelb
    "philtrum":    (0, 165, 255),  # Orange
}


def draw_frame(frame, kp, frame_idx):
    """Zeichnet Keypoints + ROIs auf einen Frame."""
    vis = frame.copy()

    # ── Keypoints zeichnen ──
    for i in range(54):
        x, y = int(kp[i, 0]), int(kp[i, 1])
        if np.isnan(kp[i]).any():
            continue
        # Punkt
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        # Index-Nummer
        cv2.putText(vis, str(i), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                    (255, 255, 255), 1)

    # ── ROIs zeichnen ──
    if not np.isnan(kp).all():
        rois = compute_rois(kp)
        for roi_name, (cx, cy, r) in rois.items():
            color = ROI_COLORS.get(roi_name, (255, 255, 255))
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(vis, roi_name, (cx - 20, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        color, 1)

    # ── Frame-Nummer ──
    cv2.putText(vis, f"Frame {frame_idx}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)

    return vis


# ═══════════════════════════════════════════════
# 1. Einzelbild speichern
# ═══════════════════════════════════════════════
print("Einzelbild speichern...")
vis_frame = draw_frame(cropped[0], keypoints[0], 0)
os.makedirs("results", exist_ok=True)
cv2.imwrite("results/keypoints_frame0.png", vis_frame)
print("Gespeichert: results/keypoints_frame0.png")

# ═══════════════════════════════════════════════
# 2. Video speichern
# ═══════════════════════════════════════════════
print("\nVideo erstellen...")
h, w = cropped[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("results/keypoints_video.mp4", fourcc, 25, (w, h))

for i in range(len(cropped)):
    vis = draw_frame(cropped[i], keypoints[i], i)
    out.write(vis)
    if (i + 1) % 25 == 0:
        print(f"  Frame {i+1}/{len(cropped)}")

out.release()
print("Gespeichert: results/keypoints_video.mp4")
print("\nFERTIG! Öffne die Dateien:")
print("  open results/keypoints_frame0.png")
print("  open results/keypoints_video.mp4")
