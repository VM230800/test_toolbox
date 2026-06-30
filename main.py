"""
Main pipeline for the Thermal Vital Signs Toolbox.

Orchestrates the complete workflow:
    1. Load configuration
    2. Load dataset
    3. YOLO face detection + keypoint extraction
    4. ROI computation from keypoints
    5. Run enabled methods (thermal_mean, ica, garbey)
    6. Visualisation (ROI overlay, signal plots, optional video)
    7. Evaluate against ground truth
    8. Save results (CSV, plots, PDF table)

Supports two processing modes:
    - streaming: true  → RAM-friendly, one frame at a time
    - streaming: false → all frames in RAM (for small tests)
"""

import argparse
import gc
import os
import time
import warnings

import numpy as np
import pandas as pd
import yaml

# ── Data loading ──
from data.bp4d_loader import BP4DDataset
from data.npz_loader import NPZDataset

# ── Preprocessing ──
from utils.yolo_processing import process_with_yolo
from utils.yolo_processing import process_with_yolo_streaming
from preprocessing.roi_extraction import compute_rois

# ── Methods ──
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod

# ── Evaluation ──
from evaluation.metrics import evaluate_algorithm
from evaluation.results_table import ResultsTable

# ── Visualisation ──
from utils.visualization import (
    save_roi_overlay,
    save_method_roi_overlay,
    save_signal_plot,
    save_signal_comparison,
    save_gt_physiology_plot,
    save_roi_video,
)
from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)
from preprocessing.peak_extraction import bandpass_filter


# ─────────────────────────────────────────────────────────────────
# 1. Load configuration
# ─────────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load main config and dataset config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["dataset_config"]) as f:
        dataset_config = yaml.safe_load(f)

    return config, dataset_config


# ─────────────────────────────────────────────────────────────────
# 2. Load dataset
# ─────────────────────────────────────────────────────────────────

def load_dataset(config, dataset_config):
    """Load the dataset specified in config."""
    dataset_type = config["dataset"]

    if dataset_type == "bp4d":
        dataset = BP4DDataset(
            root_dir=dataset_config["root_dir"],
            subjects=dataset_config.get("subjects"),
            tasks=dataset_config.get("tasks"),
            warmup_seconds=dataset_config.get(
                "warmup_seconds", 0),
            fps=dataset_config.get("fps", 25),
        )
    elif dataset_type == "npz":
        dataset = NPZDataset(
            root_dir=dataset_config["root_dir"],
            subjects=dataset_config.get("subjects"),
            recordings=dataset_config.get("recordings"),
            warmup_seconds=dataset_config.get(
                "warmup_seconds", 10),
            fps=dataset_config.get("fps", 30),
        )
    else:
        raise ValueError(f"Unknown dataset: '{dataset_type}'")

    print(f"Dataset: {dataset_type}, {len(dataset)} samples")
    return dataset


# ─────────────────────────────────────────────────────────────────
# 3. YOLO + ROI computation
# ─────────────────────────────────────────────────────────────────

def run_yolo_and_rois(frames, config):
    """
    Face detection → crop → keypoints → ROIs.
    All frames in RAM (use for small tests).
    """
    det = config["detection"]

    cropped_frames, keypoints = process_with_yolo(
        frames=frames,
        model_path=det["model_path"],
        target_size=tuple(det["target_size"]),
        padding=det.get("padding", 50),
    )

    rois_per_frame = _keypoints_to_rois(keypoints)

    n_detected = sum(
        1 for r in rois_per_frame if r is not None)
    print(f"  YOLO: {n_detected}/{len(keypoints)} "
          f"frames with keypoints")

    return cropped_frames, keypoints, rois_per_frame


def run_yolo_and_rois_streaming(dataset, idx, config,
                                max_frames=None):
    """
    RAM-friendly: streams frames one-by-one through YOLO.
    """
    det = config["detection"]
    processing = config.get("processing", {})
    frame_step = processing.get("frame_step", 1)

    frame_iter = dataset.iter_frames(
        idx,
        max_frames=max_frames,
        frame_step=frame_step,
    )

    cropped_frames, keypoints = process_with_yolo_streaming(
        frame_iterator=frame_iter,
        model_path=det["model_path"],
        target_size=tuple(det["target_size"]),
        padding=det.get("padding", 50),
    )

    rois_per_frame = _keypoints_to_rois(keypoints)

    n_detected = sum(
        1 for r in rois_per_frame if r is not None)

    effective_fps = dataset.get_metadata(idx)["fps"]
    if frame_step > 1:
        effective_fps /= frame_step
        print(f"  Frame step: {frame_step} "
              f"(effective FPS: {effective_fps:.1f})")

    print(f"  YOLO: {n_detected}/{len(keypoints)} "
          f"frames with keypoints")

    return cropped_frames, keypoints, rois_per_frame


def _keypoints_to_rois(keypoints):
    """Convert keypoint array to list of ROI dicts."""
    rois_per_frame = []
    for i in range(len(keypoints)):
        kp = keypoints[i]
        if np.isnan(kp).all():
            rois_per_frame.append(None)
        else:
            rois_per_frame.append(compute_rois(kp))
    return rois_per_frame


# ─────────────────────────────────────────────────────────────────
# 4. Run methods
# ─────────────────────────────────────────────────────────────────

def init_methods(config):
    """Initialise all enabled methods."""
    methods_config = config["methods"]
    signal_config = config["signal"]
    methods = {}

    if methods_config.get("thermal_mean", {}).get(
            "enabled", False):
        methods["thermal_mean"] = ThermalMeanMethod(
            methods_config["thermal_mean"], signal_config)

    if methods_config.get("ica", {}).get("enabled", False):
        methods["ica"] = ICAMethod(
            methods_config["ica"], signal_config)

    if methods_config.get("garbey", {}).get("enabled", False):
        methods["garbey"] = GarbeyMethod(
            methods_config["garbey"], signal_config)

    print(f"Methods enabled: {list(methods.keys())}")
    return methods


def run_methods(methods, cropped_frames, keypoints,
                rois_per_frame, fps):
    """Run all enabled methods on one sample."""
    results = {}

    for name, method in methods.items():
        try:
            if name == "garbey":
                result = method.estimate(
                    cropped_frames, keypoints, fps)
            else:
                result = method.estimate(
                    cropped_frames, rois_per_frame, fps)

            results[name] = result
            print(f"  {name}: HR={result['hr_bpm']:.1f}, "
                  f"RR={result['rr_bpm']:.1f} BPM")

        except Exception as e:
            warnings.warn(f"  {name} failed: {e}")
            results[name] = {
                "hr_bpm": float("nan"),
                "rr_bpm": float("nan"),
                "method": name,
            }

    return results


# ─────────────────────────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────────────────────────

def save_visualisations(config, sample, cropped, keypoints,
                        rois, sample_results):
    """
    Save diagnostic plots for one recording.

    Handles both BP4D+ (with raw waveforms) and NPZ datasets.
    Uses raw BP/Resp waveforms when available for proper
    oscillating ground truth comparison.
    """
    output = config.get("output", {})
    save_dir = output.get("save_dir", "results/")
    recording_id = sample["recording_id"]
    fps = sample.get("fps", 25.0)

    if not output.get("save_plots", False):
        return

    # ── 1. ROI Overlay ──
    save_roi_overlay(cropped[0], keypoints[0],
                     recording_id, save_dir)

    # ── 1b. Method-specific ROI Overlays ──       ← HIER NEU
    for method_name in sample_results.keys():
        method_cfg = config["methods"].get(
            method_name, {})
        try:
            save_method_roi_overlay(
                cropped[0], keypoints[0],
                method_name, method_cfg,
                recording_id, save_dir,
            )
        except Exception as e:
            warnings.warn(
                f"    Method overlay failed "
                f"({method_name}): {e}")                        

    # ── 2. Optional Video ──
    if output.get("save_video", False):
        max_sec = output.get("video_seconds", 4)
        save_roi_video(cropped, keypoints, fps,
                       recording_id, save_dir,
                       max_seconds=max_sec)

    # ── 3. Determine GT signal source ──
    # BP4D+: use raw waveforms (BP_mmHg, Resp_Volts)
    # NPZ:   use pulse_rate / resp_rate signals
    bp_waveform = sample.get("bp_waveform", None)
    resp_waveform = sample.get("resp_waveform", None)
    physio_fps = sample.get("physio_fps", None)

    gt_pulse = sample.get("pulse_rate", None)
    gt_resp = sample.get("resp_rate", None)

    # Choose best HR ground truth signal
    if bp_waveform is not None:
        hr_gt_signal = bp_waveform
        hr_gt_fps = physio_fps
        hr_gt_source = "waveform"
    elif gt_pulse is not None:
        hr_gt_signal = gt_pulse
        hr_gt_fps = fps  # same as video if no physio_fps
        hr_gt_source = "rate"
    else:
        hr_gt_signal = None
        hr_gt_fps = None
        hr_gt_source = None

    # Choose best RR ground truth signal
    if resp_waveform is not None:
        rr_gt_signal = resp_waveform
        rr_gt_fps = physio_fps
        rr_gt_source = "waveform"
    elif gt_resp is not None:
        rr_gt_signal = gt_resp
        rr_gt_fps = fps
        rr_gt_source = "rate"
    else:
        rr_gt_signal = None
        rr_gt_fps = None
        rr_gt_source = None

    # ── 4. Signal Comparison Plots ──
    for method_name, result in sample_results.items():
        if np.isnan(result.get("hr_bpm", float("nan"))):
            continue

        method_cfg = config["methods"].get(method_name, {})
        roi_names = method_cfg.get("rois", [])

        if not roi_names:
            continue

        roi_signals = extract_all_roi_signals(
            cropped, rois, roi_names)

        for roi_name, raw_signal in roi_signals.items():
            clean = interpolate_nan(raw_signal)
            if np.isnan(clean).all():
                continue

            # ── HR Signal Comparison ──
            if hr_gt_signal is not None and len(hr_gt_signal) > 10:
                bp = config["signal"]["hr_bandpass"]
                try:
                    filtered = bandpass_filter(
                        clean, fps,
                        low=bp["low"], high=bp["high"],
                        order=bp["order"],
                    )

                    save_signal_comparison(
                        predicted_signal=filtered,
                        gt_signal=hr_gt_signal,
                        fps=fps,
                        predicted_bpm=result["hr_bpm"],
                        gt_bpm=sample.get(
                            "hr_bpm", float("nan")),
                        signal_type="hr",
                        method_name=method_name,
                        recording_id=recording_id,
                        save_dir=save_dir,
                        bandpass=(bp["low"], bp["high"]),
                        gt_fps=hr_gt_fps,
                    )
                except Exception as e:
                    warnings.warn(
                        f"    HR comparison failed: {e}")

            # ── RR Signal Comparison ──
            if (rr_gt_signal is not None
                    and len(rr_gt_signal) > 10
                    and not np.isnan(
                        result.get("rr_bpm", float("nan")))):
                bp_rr = config["signal"]["rr_bandpass"]
                try:
                    filtered_rr = bandpass_filter(
                        clean, fps,
                        low=bp_rr["low"],
                        high=bp_rr["high"],
                        order=bp_rr["order"],
                    )

                    save_signal_comparison(
                        predicted_signal=filtered_rr,
                        gt_signal=rr_gt_signal,
                        fps=fps,
                        predicted_bpm=result["rr_bpm"],
                        gt_bpm=sample.get(
                            "rr_bpm", float("nan")),
                        signal_type="rr",
                        method_name=method_name,
                        recording_id=recording_id,
                        save_dir=save_dir,
                        bandpass=(bp_rr["low"],
                                  bp_rr["high"]),
                        gt_fps=rr_gt_fps,
                    )
                except Exception as e:
                    warnings.warn(
                        f"    RR comparison failed: {e}")

            # ── Physiology Plots (BP4D+ only) ──
            if bp_waveform is not None and physio_fps:
                try:
                    filtered = bandpass_filter(
                        clean, fps,
                        low=config["signal"][
                            "hr_bandpass"]["low"],
                        high=config["signal"][
                            "hr_bandpass"]["high"],
                        order=config["signal"][
                            "hr_bandpass"]["order"],
                    )
                    save_gt_physiology_plot(
                        physio_signals={
                            "waveform": bp_waveform,
                            "rate_bpm": gt_pulse
                            if gt_pulse is not None
                            else np.array([]),
                        },
                        predicted_signal=filtered,
                        fps_video=fps,
                        fps_physio=physio_fps,
                        predicted_bpm=result["hr_bpm"],
                        gt_bpm=sample.get(
                            "hr_bpm", float("nan")),
                        signal_type="hr",
                        method_name=method_name,
                        recording_id=recording_id,
                        save_dir=save_dir,
                    )
                except Exception as e:
                    warnings.warn(
                        f"    HR physiology failed: {e}")

            if resp_waveform is not None and physio_fps:
                try:
                    filtered_rr = bandpass_filter(
                        clean, fps,
                        low=config["signal"][
                            "rr_bandpass"]["low"],
                        high=config["signal"][
                            "rr_bandpass"]["high"],
                        order=config["signal"][
                            "rr_bandpass"]["order"],
                    )
                    save_gt_physiology_plot(
                        physio_signals={
                            "waveform": resp_waveform,
                            "rate_bpm": gt_resp
                            if gt_resp is not None
                            else np.array([]),
                        },
                        predicted_signal=filtered_rr,
                        fps_video=fps,
                        fps_physio=physio_fps,
                        predicted_bpm=result["rr_bpm"],
                        gt_bpm=sample.get(
                            "rr_bpm", float("nan")),
                        signal_type="rr",
                        method_name=method_name,
                        recording_id=recording_id,
                        save_dir=save_dir,
                    )
                except Exception as e:
                    warnings.warn(
                        f"    RR physiology failed: {e}")

            break  # Only first valid ROI


# ─────────────────────────────────────────────────────────────────
# 6. Collect results for evaluation
# ─────────────────────────────────────────────────────────────────

def collect_results(method_results, sample, method_name):
    """Convert method output + GT into evaluation format."""
    return {
        "hr_estimated":    method_results.get(
            "hr_bpm", float("nan")),
        "hr_ground_truth": sample.get(
            "hr_bpm", float("nan")),
        "rr_estimated":    method_results.get(
            "rr_bpm", float("nan")),
        "rr_ground_truth": sample.get(
            "rr_bpm", float("nan")),
        "subject":         sample.get("subject", "?"),
        "task":            sample.get("task", "?"),
        "recording_id":    sample.get(
            "recording_id", "unknown"),
        "method":          method_name,
    }


# ─────────────────────────────────────────────────────────────────
# 7. Main pipeline
# ─────────────────────────────────────────────────────────────────

def run_pipeline(config_path="configs/run_config.yaml"):
    """Run the full pipeline."""

    print("=" * 60)
    print("  Thermal Vital Signs Toolbox")
    print("=" * 60)

    # ── Load config ──
    config, dataset_config = load_config(config_path)
    verbose = config.get("output", {}).get("verbose", True)

    # ── Load dataset ──
    dataset = load_dataset(config, dataset_config)

    # ── Init methods ──
    methods = init_methods(config)

    if not methods:
        print("No methods enabled. Check run_config.yaml.")
        return

    # ── Init results table ──
    save_dir = config.get("output", {}).get(
        "save_dir", "results/")
    os.makedirs(save_dir, exist_ok=True)
    table = ResultsTable(save_path=save_dir)

    all_results = {name: [] for name in methods}

    # ── Processing settings ──
    processing = config.get("processing", {})
    max_frames = processing.get("max_frames", None)
    use_streaming = processing.get("streaming", True)

    # ── Process each sample ──
    total_start = time.time()

    for idx in range(len(dataset)):

        # ── Load metadata (WITHOUT frames) ──
        meta = dataset.get_metadata(idx)
        recording_id = meta["recording_id"]
        fps = meta["fps"]

        print(f"\n{'─' * 50}")
        print(f"  Sample {idx + 1}/{len(dataset)}: "
              f"{recording_id}")
        print(f"  Total frames: {meta['total_frames']}, "
              f"FPS: {fps:.1f}")
        print(f"{'─' * 50}")

        # ── YOLO + ROIs ──
        try:
            if use_streaming:
                cropped, keypoints, rois = \
                    run_yolo_and_rois_streaming(
                        dataset, idx, config,
                        max_frames=max_frames,
                    )
            else:
                sample = dataset[idx]
                cropped, keypoints, rois = \
                    run_yolo_and_rois(
                        sample["frames"], config)
        except Exception as e:
            warnings.warn(
                f"  YOLO failed for {recording_id}: {e}")
            continue

        # ── Run methods ──
        sample_results = run_methods(
            methods, cropped, keypoints, rois, fps)

        # ── Save visualisations ──
        try:
            save_visualisations(
                config, meta, cropped, keypoints,
                rois, sample_results,
            )
        except Exception as e:
            warnings.warn(f"  Visualisation failed: {e}")

        # ── Collect for evaluation ──
        for method_name, result in sample_results.items():
            entry = collect_results(
                result, meta, method_name)
            all_results[method_name].append(entry)

            if verbose:
                gt_hr = entry["hr_ground_truth"]
                gt_rr = entry["rr_ground_truth"]
                est_hr = entry["hr_estimated"]
                est_rr = entry["rr_estimated"]
                print(f"    {method_name}: "
                      f"HR {est_hr:.1f} vs {gt_hr:.1f}, "
                      f"RR {est_rr:.1f} vs {gt_rr:.1f}")

        # ── Free RAM ──
        del cropped, keypoints, rois
        gc.collect()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Processing complete: {len(dataset)} samples "
          f"in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # ── Per-Sample CSV ──
    all_rows = []
    for method_name, results in all_results.items():
        for r in results:
            all_rows.append(r)

    if all_rows:
        df = pd.DataFrame(all_rows)
        summary_dir = os.path.join(save_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        sample_csv = os.path.join(
            summary_dir, "per_sample_results.csv")
        df.to_csv(sample_csv, index=False)
        print(f"Per-sample results: {sample_csv}")

    # ── Evaluate ──
    dataset_name = config["dataset"].upper()

    for method_name, results in all_results.items():
        if not results:
            continue

        print(f"\nEvaluating: {method_name}")

        eval_result = evaluate_algorithm(
            results,
            algo_name=method_name,
            save_dir=os.path.join(save_dir, "summary"),
        )

        table.add(dataset_name, method_name, "HR",
                   eval_result["hr"])
        table.add(dataset_name, method_name, "RR",
                   eval_result["rr"])

    # ── Save results ──
    table.save_path = os.path.join(save_dir, "summary")
    table.print()

    if config.get("output", {}).get("save_csv", True):
        table.save_csv()

    if config.get("output", {}).get("save_plots", True):
        table.save_pdf()

    print(f"\nResults saved to: {save_dir}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thermal Vital Signs Toolbox"
    )
    parser.add_argument(
        "--config",
        default="configs/run_config.yaml",
        help="Path to run configuration file",
    )
    args = parser.parse_args()

    run_pipeline(args.config)
