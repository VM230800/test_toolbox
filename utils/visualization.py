"""
utils/visualization.py
======================
Diagnostic plots and optional video clips.
Saved automatically when output.save_plots = true.
Video clips only when output.save_video = true.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from preprocessing.roi_extraction import compute_rois


ROI_COLORS = {
    "forehead":    (0, 255, 0),
    "left_cheek":  (255, 0, 0),
    "right_cheek": (255, 0, 0),
    "nose":        (0, 255, 255),
    "philtrum":    (0, 165, 255),
}


def _draw_overlays(frame, keypoints, frame_idx=None):
    """Draw keypoints + ROI circles on a frame."""
    vis = frame.copy()

    for i in range(54):
        if np.isnan(keypoints[i]).any():
            continue
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                    (255, 255, 255), 1)

    if not np.isnan(keypoints).all():
        rois = compute_rois(keypoints)
        for name, (cx, cy, r) in rois.items():
            color = ROI_COLORS.get(name, (255, 255, 255))
            cv2.circle(vis, (cx, cy), r, color, 2)
            cv2.putText(vis, name, (cx - 20, cy - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1)

    if frame_idx is not None:
        cv2.putText(vis, f"Frame {frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    return vis


# ══════════════════════════════════════════════════════════
# 1. ROI Overlay Image (~100 KB)
# ══════════════════════════════════════════════════════════

def save_roi_overlay(frame, keypoints, recording_id, save_dir):
    """Save one annotated frame as PNG."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)
    vis = _draw_overlays(frame, keypoints, frame_idx=0)
    path = os.path.join(rec_dir, f"{recording_id}_roi_overlay.png")
    cv2.imwrite(path, vis)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 2. Signal Analysis Plot (~150 KB)
# ══════════════════════════════════════════════════════════

def save_signal_plot(raw_signal, filtered_signal, fps,
                     estimated_bpm, ground_truth_bpm,
                     target, method_name,
                     recording_id, save_dir):
    """3-panel plot: raw signal, filtered signal, FFT with peaks."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    n = len(raw_signal)
    time = np.arange(n) / fps

    fig, axes = plt.subplots(3, 1, figsize=(10, 7))

    error = abs(estimated_bpm - ground_truth_bpm)
    gt_str = f"{ground_truth_bpm:.1f}" if not np.isnan(ground_truth_bpm) else "N/A"
    fig.suptitle(
        f"{recording_id} – {method_name} – {target.upper()}\n"
        f"Estimated: {estimated_bpm:.1f} BPM | "
        f"Ground Truth: {gt_str} BPM | "
        f"Error: {error:.1f} BPM",
        fontsize=11,
    )

    # Panel 1: Raw
    axes[0].plot(time, raw_signal[:n], color="steelblue",
                 linewidth=0.8)
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].set_title("Raw ROI Signal")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Filtered
    time_filt = np.arange(len(filtered_signal)) / fps
    axes[1].plot(time_filt, filtered_signal, color="darkorange",
                 linewidth=0.8)
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Bandpass Filtered")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: FFT
    n_fft = len(filtered_signal)
    window = np.hanning(n_fft)
    fft_vals = np.abs(np.fft.rfft(filtered_signal * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

    axes[2].plot(freqs, fft_vals, color="green", linewidth=0.8)

    # Estimated peak
    peak_freq = estimated_bpm / 60.0
    axes[2].axvline(peak_freq, color="red", linestyle="--",
                    linewidth=1.2,
                    label=f"Estimated: {estimated_bpm:.1f} BPM")

    # Ground truth
    if not np.isnan(ground_truth_bpm):
        gt_freq = ground_truth_bpm / 60.0
        axes[2].axvline(gt_freq, color="blue", linestyle=":",
                        linewidth=1.2,
                        label=f"GT: {ground_truth_bpm:.1f} BPM")

    if target == "hr":
        axes[2].set_xlim(0.5, 4.5)
    else:
        axes[2].set_xlim(0.0, 1.0)

    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Power")
    axes[2].set_title("FFT Power Spectrum")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(
        rec_dir,
        f"{recording_id}_{method_name}_{target}_signal.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 3. Signal Comparison Plot – Predicted vs Ground Truth
# ══════════════════════════════════════════════════════════

def save_signal_comparison(predicted_signal, gt_signal, fps,
                           predicted_bpm, gt_bpm,
                           signal_type, method_name,
                           recording_id, save_dir,
                           bandpass=(0.7, 3.5)):
    """
    Plot predicted signal vs ground truth signal.

    Shows two subplots:
        1. Time domain: both signals normalised and overlaid
        2. Frequency domain: FFT of both with peaks marked

    Args:
        predicted_signal: np.ndarray (N,), our extracted signal
        gt_signal:        np.ndarray (M,), ground truth signal
        fps:              float, sampling rate
        predicted_bpm:    float, our estimated BPM
        gt_bpm:           float, ground truth BPM
        signal_type:      str, "hr" or "rr"
        method_name:      str, e.g. "thermal_mean"
        recording_id:     str, e.g. "006_rec_0"
        save_dir:         str, output folder
        bandpass:         tuple (low_hz, high_hz) for shading
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    type_label = "Heart Rate" if signal_type == "hr" \
        else "Respiration"
    error = abs(predicted_bpm - gt_bpm)

    fig.suptitle(
        f"{type_label} Signal Comparison – {method_name}\n"
        f"{recording_id}    |    "
        f"Predicted: {predicted_bpm:.1f} BPM    |    "
        f"GT: {gt_bpm:.1f} BPM    |    "
        f"Error: {error:.1f} BPM",
        fontsize=13, fontweight="bold",
    )

    # ── Normalise both signals to [-1, 1] ──
    def normalise(sig):
        sig = sig - np.nanmean(sig)
        mx = np.nanmax(np.abs(sig))
        if mx > 0:
            sig = sig / mx
        return sig

    pred_clean = np.nan_to_num(predicted_signal.copy(), nan=0.0)
    gt_clean = np.nan_to_num(gt_signal.copy(), nan=0.0)

    pred_norm = normalise(pred_clean)
    gt_norm = normalise(gt_clean)

    # Match lengths (use shorter)
    n = min(len(pred_norm), len(gt_norm))
    pred_norm = pred_norm[:n]
    gt_norm = gt_norm[:n]
    time_axis = np.arange(n) / fps

    # ────────────────────────────────────
    # Subplot 1: Time Domain
    # ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(time_axis, gt_norm, color="red", alpha=0.7,
             linewidth=1.0,
             label=f"Ground Truth ({gt_bpm:.1f} BPM)")
    ax1.plot(time_axis, pred_norm, color="blue", alpha=0.7,
             linewidth=1.0,
             label=f"Predicted ({predicted_bpm:.1f} BPM)")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Normalised Amplitude")
    ax1.set_title("Time Domain (normalised)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Error badge
    ax1.text(
        0.02, 0.95,
        f"Error: {error:.1f} BPM",
        transform=ax1.transAxes,
        fontsize=12, fontweight="bold",
        color=("green" if error < 5
               else "orange" if error < 10
               else "red"),
        verticalalignment="top",
        bbox=dict(boxstyle="round",
                  facecolor="white", alpha=0.8),
    )

    # ────────────────────────────────────
    # Subplot 2: Frequency Domain (FFT)
    # ────────────────────────────────────
    ax2 = axes[1]

    window = np.hanning(n)
    fft_pred = np.abs(np.fft.rfft(pred_norm * window))
    fft_gt = np.abs(np.fft.rfft(gt_norm * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)

    # Convert to BPM axis
    bpm_axis = freqs * 60.0

    # Show relevant range only
    if signal_type == "hr":
        bpm_range = (40, 180)
    else:
        bpm_range = (5, 35)

    mask = (bpm_axis >= bpm_range[0]) & (bpm_axis <= bpm_range[1])

    if mask.any():
        bpm_plot = bpm_axis[mask]
        fft_pred_plot = fft_pred[mask]
        fft_gt_plot = fft_gt[mask]

        # Normalise FFT magnitudes for comparison
        mx_pred = fft_pred_plot.max()
        mx_gt = fft_gt_plot.max()
        if mx_pred > 0:
            fft_pred_plot = fft_pred_plot / mx_pred
        if mx_gt > 0:
            fft_gt_plot = fft_gt_plot / mx_gt

        ax2.plot(bpm_plot, fft_gt_plot, color="red",
                 alpha=0.7, linewidth=1.5,
                 label=f"GT Peak: {gt_bpm:.1f} BPM")
        ax2.plot(bpm_plot, fft_pred_plot, color="blue",
                 alpha=0.7, linewidth=1.5,
                 label=f"Predicted Peak: {predicted_bpm:.1f} BPM")

        # Mark peaks with vertical lines
        ax2.axvline(x=gt_bpm, color="red",
                     linestyle="--", alpha=0.5, linewidth=1)
        ax2.axvline(x=predicted_bpm, color="blue",
                     linestyle="--", alpha=0.5, linewidth=1)

        # Shade bandpass region
        bp_bpm_low = bandpass[0] * 60
        bp_bpm_high = bandpass[1] * 60
        ax2.axvspan(bp_bpm_low, bp_bpm_high,
                     alpha=0.1, color="green",
                     label="Bandpass Range")

    ax2.set_xlabel("Frequency [BPM]")
    ax2.set_ylabel("Normalised Magnitude")
    ax2.set_title("Frequency Domain (FFT)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Save ──
    plt.tight_layout()

    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    filename = f"{recording_id}_{method_name}_{signal_type}_comparison.png"
    filepath = os.path.join(rec_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved: {filepath}")


# ══════════════════════════════════════════════════════════
# 4. Video Clip – optional (~2-5 MB)
# ══════════════════════════════════════════════════════════

def save_roi_video(frames, keypoints, fps, recording_id,
                   save_dir, max_seconds=4):
    """Short video clip with keypoints + ROIs."""
    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    max_frames = int(max_seconds * fps)
    n = min(len(frames), max_frames)

    h, w = frames[0].shape[:2]
    path = os.path.join(rec_dir, f"{recording_id}_roi_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for i in range(n):
        vis = _draw_overlays(frames[i], keypoints[i], frame_idx=i)
        out.write(vis)

    out.release()
    print(f"  Saved: {path} ({n} frames, {n/fps:.1f}s)")


# ══════════════════════════════════════════════════════════
# 5. Ground Truth Physiology Plot
# ══════════════════════════════════════════════════════════

def save_gt_physiology_plot(physio_signals, predicted_signal,
                            fps_video, fps_physio,
                            predicted_bpm, gt_bpm,
                            signal_type, method_name,
                            recording_id, save_dir):
    """
    Plot raw ground truth physiology signal alongside
    our predicted thermal signal.

    Shows 3 subplots:
        1. Raw physiology waveform (BP_mmHg or Resp_Volts)
        2. Our predicted thermal signal (bandpass filtered)
        3. Both GT rate + predicted rate over time

    Args:
        physio_signals: dict with keys like:
            - "waveform":  np.ndarray, raw signal (BP or Resp_Volts)
            - "rate_bpm":  np.ndarray, BPM over time
        predicted_signal: np.ndarray, our bandpass-filtered signal
        fps_video:      float, video sampling rate (e.g. 25)
        fps_physio:     float, physiology sampling rate (e.g. 1000)
        predicted_bpm:  float, our single BPM estimate
        gt_bpm:         float, mean ground truth BPM
        signal_type:    str, "hr" or "rr"
        method_name:    str, e.g. "thermal_mean"
        recording_id:   str, e.g. "F001_T1"
        save_dir:       str, output folder
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    type_label = "Heart Rate" if signal_type == "hr" \
        else "Respiration"
    wave_label = "Blood Pressure [mmHg]" if signal_type == "hr" \
        else "Respiration [Volts]"
    error = abs(predicted_bpm - gt_bpm)

    fig.suptitle(
        f"{type_label} – Ground Truth vs Predicted\n"
        f"{recording_id}  |  {method_name}  |  "
        f"Predicted: {predicted_bpm:.1f} BPM  |  "
        f"GT: {gt_bpm:.1f} BPM  |  "
        f"Error: {error:.1f} BPM",
        fontsize=13, fontweight="bold",
    )

    waveform = physio_signals["waveform"]
    rate_bpm = physio_signals["rate_bpm"]

    # ── Match time range to video duration ──
    video_duration = len(predicted_signal) / fps_video
    n_physio = int(video_duration * fps_physio)
    n_physio = min(n_physio, len(waveform))

    waveform_clip = waveform[:n_physio]
    rate_clip = rate_bpm[:min(n_physio, len(rate_bpm))]

    time_physio = np.arange(len(waveform_clip)) / fps_physio
    time_rate = np.arange(len(rate_clip)) / fps_physio
    time_video = np.arange(len(predicted_signal)) / fps_video

    # ────────────────────────────────────
    # Subplot 1: Raw Physiology Waveform
    # ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(time_physio, waveform_clip, color="red",
             linewidth=0.4, alpha=0.8)
    ax1.set_ylabel(wave_label)
    ax1.set_title(
        f"Ground Truth Waveform "
        f"({fps_physio:.0f} Hz sampling)")
    ax1.grid(True, alpha=0.3)

    # Show zoomed view in first 5 seconds
    if video_duration > 5:
        ax1.set_xlim(0, min(5, video_duration))
        ax1.text(
            0.98, 0.95,
            f"Showing first 5s of {video_duration:.1f}s",
            transform=ax1.transAxes,
            fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round",
                      facecolor="wheat", alpha=0.8),
        )

    # ────────────────────────────────────
    # Subplot 2: Our Predicted Signal
    # ────────────────────────────────────
    ax2 = axes[1]

    # Normalise for display
    pred_norm = predicted_signal - np.mean(predicted_signal)
    mx = np.max(np.abs(pred_norm))
    if mx > 0:
        pred_norm = pred_norm / mx

    ax2.plot(time_video, pred_norm, color="blue",
             linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Normalised Amplitude")
    ax2.set_title(
        f"Predicted Signal – {method_name} "
        f"({fps_video:.0f} Hz sampling)")
    ax2.grid(True, alpha=0.3)

    if video_duration > 5:
        ax2.set_xlim(0, min(5, video_duration))

    # ────────────────────────────────────
    # Subplot 3: BPM over Time
    # ────────────────────────────────────
    ax3 = axes[2]

    ax3.plot(time_rate, rate_clip, color="red",
             linewidth=0.8, alpha=0.7,
             label=f"GT Rate (mean: {gt_bpm:.1f} BPM)")

    # Our prediction as horizontal line
    ax3.axhline(y=predicted_bpm, color="blue",
                linestyle="--", linewidth=1.5,
                label=f"Predicted: {predicted_bpm:.1f} BPM")

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Rate [BPM]")
    ax3.set_title(f"{type_label} Rate over Time")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Error badge
    ax3.text(
        0.02, 0.95,
        f"Error: {error:.1f} BPM",
        transform=ax3.transAxes,
        fontsize=12, fontweight="bold",
        color=("green" if error < 5
               else "orange" if error < 10
               else "red"),
        verticalalignment="top",
        bbox=dict(boxstyle="round",
                  facecolor="white", alpha=0.8),
    )

    # ── Save ──
    plt.tight_layout()

    rec_dir = os.path.join(save_dir, recording_id)
    os.makedirs(rec_dir, exist_ok=True)

    filename = (f"{recording_id}_{method_name}_"
                f"{signal_type}_physiology.png")
    filepath = os.path.join(rec_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved: {filepath}")
