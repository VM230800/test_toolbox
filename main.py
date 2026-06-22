"""
main.py
=======
Main pipeline for the Thermal Vital Signs Toolbox.

Orchestrates the complete workflow:
    1. Load configuration
    2. Load dataset
    3. YOLO face detection + keypoint extraction
    4. ROI computation from keypoints
    5. Run enabled methods (thermal_mean, ica, garbey)
    6. Evaluate against ground truth
    7. Save results (CSV, plots, PDF table)

Usage:
    python main.py
    python main.py --config configs/run_config.yaml
"""

import argparse
import os
import time
import warnings

import numpy as np
import yaml

# ── Data loading ──
from data.bp4d_loader import BP4DDataset
from data.npz_loader import NPZDataset

# ── Preprocessing ──
from utils.yolo_processing import process_with_yolo
from preprocessing.roi_extraction import compute_rois

# ── Methods ──
from methods.thermal_mean import ThermalMeanMethod
from methods.ica import ICAMethod
from methods.garbey import GarbeyMethod

# ── Evaluation ──
from evaluation.metrics import evaluate_algorithm
from evaluation.results_table import ResultsTable


# ─────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load main config and dataset config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    with open(config["dataset_config"]) as f:
        dataset_config = yaml.safe_load(f)

    return config, dataset_config


# ─────────────────────────────────────────────────────────────────
# 2. Dataset loading
# ─────────────────────────────────────────────────────────────────

def load_dataset(config, dataset_config):
    """Load the dataset specified in config."""
    dataset_type = config["dataset"]

    if dataset_type == "bp4d":
        dataset = BP4DDataset(
            root_dir=dataset_config["root_dir"],
            subjects=dataset_config.get("subjects"),
            tasks=dataset_config.get("tasks"),
            warmup_seconds=dataset_config.get("warmup_seconds", 0),
            fps=dataset_config.get("fps", 25),
        )
    elif dataset_type == "npz":
        dataset = NPZDataset(
            root_dir=dataset_config["root_dir"],
            warmup_seconds=dataset_config.get("warmup_seconds", 10),
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

    Returns:
        cropped_frames: np.ndarray (N, H, W, 3) uint8
        keypoints:      np.ndarray (N, 54, 2) float32
        rois_per_frame: list of dict or None
    """
    det = config["detection"]

    cropped_frames, keypoints = process_with_yolo(
        frames=frames,
        model_path=det["model_path"],
        target_size=tuple(det["target_size"]),
        padding=det.get("padding", 50),
    )

    # Compute ROIs from keypoints for each frame
    rois_per_frame = []
    for i in range(len(keypoints)):
        kp = keypoints[i]
        if np.isnan(kp).all():
            rois_per_frame.append(None)
        else:
            rois = compute_rois(kp)
            rois_per_frame.append(rois)

    n_detected = sum(1 for r in rois_per_frame if r is not None)
    print(f"  YOLO: {n_detected}/{len(keypoints)} frames with keypoints")

    return cropped_frames, keypoints, rois_per_frame


# ─────────────────────────────────────────────────────────────────
# 4. Run methods
# ─────────────────────────────────────────────────────────────────

def init_methods(config):
    """Initialise all enabled methods."""
    methods_config = config["methods"]
    signal_config = config["signal"]
    methods = {}

    if methods_config.get("thermal_mean", {}).get("enabled", False):
        methods["thermal_mean"] = ThermalMeanMethod(
            methods_config["thermal_mean"], signal_config
        )

    if methods_config.get("ica", {}).get("enabled", False):
        methods["ica"] = ICAMethod(
            methods_config["ica"], signal_config
        )

    if methods_config.get("garbey", {}).get("enabled", False):
        methods["garbey"] = GarbeyMethod(
            methods_config["garbey"], signal_config
        )

    print(f"Methods enabled: {list(methods.keys())}")
    return methods


def run_methods(methods, cropped_frames, keypoints, rois_per_frame, fps):
    """
    Run all enabled methods on one sample.

    Returns:
        dict: method_name → result dict
    """
    results = {}

    for name, method in methods.items():
        try:
            if name == "garbey":
                # Garbey needs raw keypoints, not ROI boxes
                result = method.estimate(cropped_frames, keypoints, fps)
            else:
                # ICA and thermal_mean need ROI boxes
                result = method.estimate(cropped_frames, rois_per_frame, fps)

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
# 5. Collect results for evaluation
# ─────────────────────────────────────────────────────────────────

def collect_results(method_results, sample, method_name):
    """
    Convert method output + ground truth into the format
    that evaluate_algorithm() expects.

    Args:
        method_results: dict from method.estimate()
        sample:         dict from dataset loader
        method_name:    str

    Returns:
        dict with hr_estimated, hr_ground_truth, etc.
    """
    return {
        "hr_estimated":    method_results.get("hr_bpm", float("nan")),
        "hr_ground_truth": sample.get("hr_bpm", float("nan")),
        "rr_estimated":    method_results.get("rr_bpm", float("nan")),
        "rr_ground_truth": sample.get("rr_bpm", float("nan")),
        "subject":         sample.get("subject", "?"),
        "task":            sample.get("task", "?"),
        "method":          method_name,
    }


# ─────────────────────────────────────────────────────────────────
# 6. Main pipeline
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
    save_dir = config.get("output", {}).get("save_dir", "results/")
    os.makedirs(save_dir, exist_ok=True)
    table = ResultsTable(save_path=save_dir)

    # Per-method result collectors for evaluate_algorithm()
    all_results = {name: [] for name in methods}

    # ── Process each sample ──
    total_start = time.time()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        subject = sample.get("subject", f"sample_{idx}")
        task = sample.get("task", "")
        fps = sample.get("fps", 25.0)

        print(f"\n{'─' * 50}")
        print(f"  Sample {idx + 1}/{len(dataset)}: {subject}/{task}")
        print(f"{'─' * 50}")

        # ── YOLO + ROIs ──
        try:
            cropped, keypoints, rois = run_yolo_and_rois(
                sample["frames"], config
            )
        except Exception as e:
            warnings.warn(f"  YOLO failed for {subject}/{task}: {e}")
            continue

        # ── Run methods ──
        sample_results = run_methods(
            methods, cropped, keypoints, rois, fps
        )

        # ── Collect for evaluation ──
        for method_name, result in sample_results.items():
            entry = collect_results(result, sample, method_name)
            all_results[method_name].append(entry)

            if verbose:
                gt_hr = entry["hr_ground_truth"]
                gt_rr = entry["rr_ground_truth"]
                est_hr = entry["hr_estimated"]
                est_rr = entry["rr_estimated"]
                print(f"    {method_name}: "
                      f"HR {est_hr:.1f} vs {gt_hr:.1f}, "
                      f"RR {est_rr:.1f} vs {gt_rr:.1f}")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  Processing complete: {len(dataset)} samples in "
          f"{elapsed:.1f}s")
    print(f"{'=' * 60}")

     # ── Per-Sample CSV ──────────────────────────────────────  NEU
    import pandas as pd                                       # NEU
                                                              # NEU
    all_rows = []                                             # NEU
    for method_name, results in all_results.items():          # NEU
        for r in results:                                     # NEU
            all_rows.append(r)                                # NEU
                                                              # NEU
    if all_rows:                                              # NEU
        df = pd.DataFrame(all_rows)                           # NEU
        sample_csv = os.path.join(save_dir,                   # NEU
                                  "per_sample_results.csv")   # NEU
        df.to_csv(sample_csv, index=False)                    # NEU
        print(f"Per-sample results: {sample_csv}")            # NEU

    # ── Evaluate ──
    dataset_name = config["dataset"].upper()

    for method_name, results in all_results.items():
        if not results:
            continue

        print(f"\nEvaluating: {method_name}")

        eval_result = evaluate_algorithm(
            results,
            algo_name=method_name,
            save_dir=save_dir,
        )

        # Add to results table
        table.add(dataset_name, method_name, "HR", eval_result["hr"])
        table.add(dataset_name, method_name, "RR", eval_result["rr"])

    # ── Save results ──
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
