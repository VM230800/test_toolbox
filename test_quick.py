"""Quick test with BP4D+ dataset – 100 frames."""
import os
import numpy as np
import yaml
from scipy.signal import resample

from data.bp4d_loader import BP4DDataset
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois
from preprocessing.signal_extraction import (
    extract_all_roi_signals, interpolate_nan,
)
from preprocessing.peak_extraction import bandpass_filter
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod
from utils.visualization import (
    save_roi_overlay,
    save_method_roi_overlay,
    save_signal_comparison,
    save_gt_physiology_plot,
)

# ══════════════════════════════════════════════════
#  HIER ÄNDERN FÜR ANDEREN SUBJECT:
# ══════════════════════════════════════════════════
SUBJECT = "F004"
TASK = "T1"
# ══════════════════════════════════════════════════

# ── Load config ──
with open("configs/run_config.yaml") as f:
    config = yaml.safe_load(f)

with open("configs/bp4d.yaml") as f:
    ds_config = yaml.safe_load(f)

# ── Load dataset ──
dataset = BP4DDataset(
    root_dir=ds_config["root_dir"],
    subjects=[SUBJECT],
    tasks=[TASK],
    fps=25,
)

# ── Metadata ──
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
recording_id = sample.get(
    "recording_id", f"{SUBJECT}_{TASK}")

print(f"\nFrames: {frames.shape}, {frames.dtype}")
print(f"Dauer: {len(frames)/fps:.1f}s bei {fps} FPS")

# ── Ground Truth ──
gt_hr = sample["hr_bpm"]
gt_rr = sample["rr_bpm"]
print(f"Ground Truth: HR={gt_hr:.1f}, RR={gt_rr:.1f} BPM")

# ── Load raw physiology waveforms ──
physio_dir = os.path.join(
    ds_config["root_dir"], "Physiology", SUBJECT, TASK)

bp_wave = np.loadtxt(
    os.path.join(physio_dir, "BP_mmHg.txt"))
resp_wave = np.loadtxt(
    os.path.join(physio_dir, "Resp_Volts.txt"))
pulse_rate = np.loadtxt(
    os.path.join(physio_dir, "Pulse Rate_BPM.txt"))
resp_rate = np.loadtxt(
    os.path.join(physio_dir, "Respiration Rate_BPM.txt"))

fps_physio = len(bp_wave) / (meta["total_frames"] / fps)
print(f"Physiology: {len(bp_wave)} Werte, "
      f"~{fps_physio:.0f} Hz")

# ── YOLO ──
det = config["detection"]
cropped, keypoints = process_with_yolo(
    frames,
    model_path=det["model_path"],
    target_size=tuple(det["target_size"]),
    padding=det.get("padding", 100),
    confidence=0.15,
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

# ── ROI Overlay (allgemein) ──
save_roi_overlay(cropped[0], keypoints[0],
                 recording_id, "results/")

# ── Method-specific ROI Overlays ──
method_configs = {
    "thermal_mean": config["methods"]["thermal_mean"],
    "ica":          config["methods"]["ica"],
    "garbey":       config["methods"]["garbey"],
}

for m_name, m_cfg in method_configs.items():
    try:
        save_method_roi_overlay(
            cropped[0], keypoints[0],
            m_name, m_cfg,
            recording_id, "results/",
        )
    except Exception as e:
        print(f"  Method overlay failed ({m_name}): {e}")

# ── Methode 1: Thermal Mean ──
print("\n1. Thermal Mean:")
tm = ThermalMeanMethod(
    config["methods"]["thermal_mean"],
    config["signal"],
)
r1 = tm.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r1['hr_bpm']:.1f} (GT: {gt_hr:.1f}, "
      f"Error: {abs(r1['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r1['rr_bpm']:.1f} (GT: {gt_rr:.1f})")

# ── Methode 2: ICA ──
print("\n2. ICA:")
ica = ICAMethod(
    config["methods"]["ica"],
    config["signal"],
)
r2 = ica.estimate(cropped, rois_per_frame, fps)
print(f"   HR: {r2['hr_bpm']:.1f} (GT: {gt_hr:.1f}, "
      f"Error: {abs(r2['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r2['rr_bpm']:.1f} (GT: {gt_rr:.1f})")

# ── Methode 3: Garbey ──
print("\n3. Garbey:")
garbey = GarbeyMethod(
    config["methods"]["garbey"],
    config["signal"],
)
r3 = garbey.estimate(cropped, keypoints, fps)
print(f"   HR: {r3['hr_bpm']:.1f} (GT: {gt_hr:.1f}, "
      f"Error: {abs(r3['hr_bpm'] - gt_hr):.1f})")
print(f"   RR: {r3['rr_bpm']:.1f} (GT: {gt_rr:.1f})")

# ── Signal Plots ──
print(f"\n{'─' * 50}")
print("Generating plots...")

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

        # ── HR Comparison ──
        bp = config["signal"]["hr_bandpass"]
        try:
            filtered = bandpass_filter(
                clean, fps,
                low=bp["low"], high=bp["high"],
                order=bp["order"],
            )

            save_signal_comparison(
                predicted_signal=filtered,
                gt_signal=bp_wave,
                fps=fps,
                predicted_bpm=result["hr_bpm"],
                gt_bpm=gt_hr,
                signal_type="hr",
                method_name=method_name,
                recording_id=recording_id,
                save_dir="results/",
                bandpass=(bp["low"], bp["high"]),
                gt_fps=fps_physio,
            )
        except Exception as e:
            print(f"    HR ({method_name}): {e}")

        # ── RR Comparison ──
        bp_rr = config["signal"]["rr_bandpass"]
        try:
            filtered_rr = bandpass_filter(
                clean, fps,
                low=bp_rr["low"], high=bp_rr["high"],
                order=bp_rr["order"],
            )

            save_signal_comparison(
                predicted_signal=filtered_rr,
                gt_signal=resp_wave,
                fps=fps,
                predicted_bpm=result["rr_bpm"],
                gt_bpm=gt_rr,
                signal_type="rr",
                method_name=method_name,
                recording_id=recording_id,
                save_dir="results/",
                bandpass=(bp_rr["low"], bp_rr["high"]),
                gt_fps=fps_physio,
            )
        except Exception as e:
            print(f"    RR ({method_name}): {e}")

        # ── HR Physiology ──
        try:
            save_gt_physiology_plot(
                physio_signals={
                    "waveform": bp_wave,
                    "rate_bpm": pulse_rate,
                },
                predicted_signal=filtered,
                fps_video=fps,
                fps_physio=fps_physio,
                predicted_bpm=result["hr_bpm"],
                gt_bpm=gt_hr,
                signal_type="hr",
                method_name=method_name,
                recording_id=recording_id,
                save_dir="results/",
            )
        except Exception as e:
            print(f"    HR physio ({method_name}): {e}")

        # ── RR Physiology ──
        try:
            save_gt_physiology_plot(
                physio_signals={
                    "waveform": resp_wave,
                    "rate_bpm": resp_rate,
                },
                predicted_signal=filtered_rr,
                fps_video=fps,
                fps_physio=fps_physio,
                predicted_bpm=result["rr_bpm"],
                gt_bpm=gt_rr,
                signal_type="rr",
                method_name=method_name,
                recording_id=recording_id,
                save_dir="results/",
            )
        except Exception as e:
            print(f"    RR physio ({method_name}): {e}")

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

print(f"\nPlots: results/{recording_id}/")
print("FERTIG!")
