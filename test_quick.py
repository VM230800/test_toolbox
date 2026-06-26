"""Quick test with BP4D+ dataset – 100 frames + signal comparison."""
import numpy as np
import yaml
from data.bp4d_loader import BP4DDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois
from preprocessing.signal_extraction import extract_all_roi_signals, interpolate_nan
from preprocessing.peak_extraction import bandpass_filter
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod
from utils.visualization import (
    save_roi_overlay,
    save_signal_comparison,
)

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

# ── Test get_metadata (neues Feature) ──
print("\n── Test get_metadata() ──")
meta = dataset.get_metadata(0)
print(f"Recording: {meta['recording_id']}")
print(f"FPS: {meta['fps']:.1f}")
print(f"Frames: {meta['total_frames']}")
print(f"HR GT: {meta['hr_bpm']:.1f} BPM")
print(f"RR GT: {meta['rr_bpm']:.1f} BPM")

# ── Load sample ──
sample = dataset[0]

frames = sample["frames"][:100]
fps = sample["fps"]
recording_id = sample.get("recording_id", "F001_T1")

print(f"\nFrames: {frames.shape}, {frames.dtype}")
print(f"RAM: ~{frames.nbytes / 1e6:.0f} MB")
print(f"Dauer: {len(frames)/fps:.1f}s bei {fps} FPS")

# ── Ground Truth ──
gt_hr = sample["hr_bpm"]
gt_rr = sample["rr_bpm"]
gt_pulse = sample.get("pulse_rate", None)
gt_resp = sample.get("resp_rate", None)

print(f"\nGround Truth: HR={gt_hr:.1f} BPM, RR={gt_rr:.1f} BPM")
if gt_pulse is not None:
    print(f"  Pulse Rate Signal: {len(gt_pulse)} Werte")
if gt_resp is not None:
    print(f"  Resp Rate Signal:  {len(gt_resp)} Werte")

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

# ── ROI Overlay speichern ──
save_roi_overlay(cropped[0], keypoints[0],
                 recording_id, "results/")

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

# ── Signal Comparison Plots ──
print(f"\n{'─' * 50}")
print("Signal Comparison Plots...")

methods_results = {
    "thermal_mean": r1,
    "ica": r2,
    "garbey": r3,
}

roi_names = config["methods"]["thermal_mean"]["rois"]
roi_signals = extract_all_roi_signals(
    cropped, rois_per_frame, roi_names)

for roi_name, raw_signal in roi_signals.items():
    clean = interpolate_nan(raw_signal)
    if np.isnan(clean).all():
        continue

    for method_name, result in methods_results.items():

        # ── HR comparison ──
        if gt_pulse is not None and len(gt_pulse) > 10:
            bp = config["signal"]["hr_bandpass"]
            try:
                filtered = bandpass_filter(
                    clean, fps,
                    low=bp["low"], high=bp["high"],
                    order=bp["order"],
                )
                # GT signal auf gleiche Länge schneiden
                gt_hr_signal = gt_pulse[:len(filtered)]
                if len(gt_hr_signal) < len(filtered):
                    # GT kürzer → predicted kürzen
                    filtered = filtered[:len(gt_hr_signal)]

                save_signal_comparison(
                    predicted_signal=filtered,
                    gt_signal=gt_hr_signal,
                    fps=fps,
                    predicted_bpm=result["hr_bpm"],
                    gt_bpm=gt_hr,
                    signal_type="hr",
                    method_name=method_name,
                    recording_id=recording_id,
                    save_dir="results/",
                    bandpass=(bp["low"], bp["high"]),
                )
            except Exception as e:
                print(f"    HR plot ({method_name}) failed: {e}")

        # ── RR comparison ──
        if (gt_resp is not None and len(gt_resp) > 10
                and not np.isnan(
                    result.get("rr_bpm", float("nan")))):
            bp_rr = config["signal"]["rr_bandpass"]
            try:
                filtered_rr = bandpass_filter(
                    clean, fps,
                    low=bp_rr["low"], high=bp_rr["high"],
                    order=bp_rr["order"],
                )
                gt_rr_signal = gt_resp[:len(filtered_rr)]
                if len(gt_rr_signal) < len(filtered_rr):
                    filtered_rr = filtered_rr[:len(gt_rr_signal)]

                save_signal_comparison(
                    predicted_signal=filtered_rr,
                    gt_signal=gt_rr_signal,
                    fps=fps,
                    predicted_bpm=result["rr_bpm"],
                    gt_bpm=gt_rr,
                    signal_type="rr",
                    method_name=method_name,
                    recording_id=recording_id,
                    save_dir="results/",
                    bandpass=(bp_rr["low"], bp_rr["high"]),
                )
            except Exception as e:
                print(f"    RR plot ({method_name}) failed: {e}")

    break  # Only first valid ROI

# ── Zusammenfassung ──
print(f"\n{'═' * 50}")
print(f"  ZUSAMMENFASSUNG – {recording_id}")
print(f"{'═' * 50}")
print(f"  {'Methode':<15} {'HR est':>8} {'HR GT':>8} "
      f"{'Error':>8} {'RR est':>8} {'RR GT':>8}")
print(f"  {'─' * 58}")
print(f"  {'Thermal Mean':<15} {r1['hr_bpm']:>7.1f} "
      f"{gt_hr:>8.1f} {abs(r1['hr_bpm']-gt_hr):>7.1f} "
      f"{r1['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'ICA':<15} {r2['hr_bpm']:>7.1f} "
      f"{gt_hr:>8.1f} {abs(r2['hr_bpm']-gt_hr):>7.1f} "
      f"{r2['rr_bpm']:>8.1f} {gt_rr:>8.1f}")
print(f"  {'Garbey':<15} {r3['hr_bpm']:>7.1f} "
      f"{gt_hr:>8.1f} {abs(r3['hr_bpm']-gt_hr):>7.1f} "
      f"{r3['rr_bpm']:>8.1f} {gt_rr:>8.1f}")

print(f"\nPlots gespeichert in: results/{recording_id}/")
print("FERTIG!")
