"""Quick test with only 100 frames – all 3 methods."""
import numpy as np
import yaml
from data.bp4d_loader import BP4DDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod

# ── Load config ──
with open("configs/run_config.yaml") as f:
    config = yaml.safe_load(f)

with open("configs/bp4d.yaml") as f:
    ds_config = yaml.safe_load(f)

# ── Load dataset ──
dataset = BP4DDataset(
    root_dir=ds_config["root_dir"],
    subjects=["F001"],
    tasks=["T1"],
    fps=25,
)

sample = dataset[0]

# ── NUR 100 Frames ──
frames = sample["frames"][:100]
fps = sample["fps"]
print(f"Frames: {frames.shape}, {frames.dtype}")
print(f"RAM: ~{frames.nbytes / 1e6:.0f} MB")
print(f"Dauer: {len(frames)/fps:.1f}s bei {fps} FPS")

# ── YOLO ──
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames,
    model_path=det["model_path"],
    target_size=tuple(det["target_size"]),
    padding=det.get("padding", 50),
)

# ── ROIs ──
rois_per_frame = []
for i in range(len(keypoints)):
    kp = keypoints[i]
    if np.isnan(kp).all():
        rois_per_frame.append(None)
    else:
        rois_per_frame.append(compute_rois(kp))

n_ok = sum(1 for r in rois_per_frame if r is not None)
print(f"YOLO: {n_ok}/{len(keypoints)} frames with keypoints")

# ── Ground Truth ──
gt_hr = sample["hr_bpm"]
gt_rr = sample["rr_bpm"]
print(f"\nGround Truth: HR={gt_hr:.1f} BPM, RR={gt_rr:.1f} BPM")
print(f"{'─' * 50}")

# ── Methode 1: Thermal Mean ──
print("\n1. Thermal Mean:")
tm = ThermalMeanMethod(
    config["methods"]["thermal_mean"],
    config["signal"],
)
r1 = tm.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r1['hr_bpm']:.1f} BPM (GT: {gt_hr:.1f}, "
      f"Error: {abs(r1['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r1['rr_bpm']:.1f} BPM (GT: {gt_rr:.1f})")

# ── Methode 2: ICA ──
print("\n2. ICA:")
ica = ICAMethod(
    config["methods"]["ica"],
    config["signal"],
)
r2 = ica.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r2['hr_bpm']:.1f} BPM (GT: {gt_hr:.1f}, "
      f"Error: {abs(r2['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r2['rr_bpm']:.1f} BPM (GT: {gt_rr:.1f})")

# ── Methode 3: Garbey ──
print("\n3. Garbey:")
garbey = GarbeyMethod(
    config["methods"]["garbey"],
    config["signal"],
)
r3 = garbey.estimate(cropped, keypoints, fps)
print(f"   HR: {r3['hr_bpm']:.1f} BPM (GT: {gt_hr:.1f}, "
      f"Error: {abs(r3['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r3['rr_bpm']:.1f} BPM (GT: {gt_rr:.1f})")

# ── Zusammenfassung ──
print(f"\n{'═' * 50}")
print(f"  ZUSAMMENFASSUNG")
print(f"{'═' * 50}")
print(f"  {'Methode':<15} {'HR est':>8} {'HR GT':>8} {'Error':>8} {'RR est':>8} {'RR GT':>8}")
print(f"  {'─' * 58}")
print(f"  {'Thermal Mean':<15} {r1['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r1['hr_bpm']-gt_hr):>7.1f} {r1['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'ICA':<15} {r2['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r2['hr_bpm']-gt_hr):>7.1f} {r2['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'Garbey':<15} {r3['hr_bpm']:>7.1f} {gt_hr:>8.1f} {abs(r3['hr_bpm']-gt_hr):>7.1f} {r3['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"\nFERTIG!")








(toolbox) MacBook-Air-4:test_toolbox valeriamoltschanov$ python test_quick.py
BP4DDataset: 1 subjects, 1 samples
Frames: (100, 480, 726, 3), float32
RAM: ~418 MB
Dauer: 4.0s bei 25.0 FPS
[W NNPACK.cpp:64] Could not initialize NNPACK! Reason: Unsupported hardware.
YOLO: 100/100 frames with keypoints

Ground Truth: HR=95.9 BPM, RR=16.5 BPM
──────────────────────────────────────────────────

1. Thermal Mean:
   HR: 75.0 BPM (GT: 95.9, Error: 20.9)
   RR: nan BPM (GT: 16.5)

2. ICA:
/usr/local/Caskroom/miniforge/base/envs/toolbox/lib/python3.12/site-packages/sklearn/decomposition/_fastica.py:132: ConvergenceWarning: FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.
  warnings.warn(
   HR: 58.6 BPM (GT: 95.9, Error: 37.3)
   RR: 11.7 BPM (GT: 16.5)

3. Garbey:
   HR: 82.5 BPM (GT: 95.9, Error: 13.4)
   RR: 15.0 BPM (GT: 16.5)

══════════════════════════════════════════════════
  ZUSAMMENFASSUNG
══════════════════════════════════════════════════
  Methode           HR est    HR GT    Error   RR est    RR GT
  ──────────────────────────────────────────────────────────
  Thermal Mean       75.0     95.9    20.9      nan     16.5
  ICA                58.6     95.9    37.3     11.7     16.5
  Garbey             82.5     95.9    13.4     15.0     16.5

FERTIG!
(toolbox) MacBook-Air-4:test_toolbox valeriamoltschanov$ 
