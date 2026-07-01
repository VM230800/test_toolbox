"""
evaluation/metrics.py
=====================
Evaluation metrics for HR and RR.

No signal processing. This module only receives final BPM values → outputs metrics and plots.
"""

import numpy as np
from evaluation.bland_altman import BlandAltman


# ═══════════════════════════════════════════════
# Compute evaluation metrics
# ═══════════════════════════════════════════════

def compute_metrics(estimated, ground_truth):
    """Compute all evaluation metrics for a list of estimates.

    Args:
        estimated:    List/array of estimated BPM values
        ground_truth: List/array of ground-truth BPM values

    Returns:
        Dictionary containing MAE, MAE_SE, RMSE, MAPE, Pearson, and n
    """
    est = np.asarray(estimated, dtype=float)
    gt  = np.asarray(ground_truth, dtype=float)

    valid = ~(np.isnan(est) | np.isnan(gt))
    est, gt = est[valid], gt[valid]
    n = len(est)

    if n == 0:
        return {k: float("nan") for k in
                ("MAE", "MAE_SE", "RMSE", "MAPE", "Pearson", "n")}

    abs_err = np.abs(est - gt)

    return {
        "MAE":     float(np.mean(abs_err)),
        "MAE_SE":  float(np.std(abs_err) / np.sqrt(n)),
        "RMSE":    float(np.sqrt(np.mean((est - gt) ** 2))),
        "MAPE":    float(np.mean(np.abs((est - gt) / (gt + 1e-10))) * 100),
        "Pearson": float(np.corrcoef(est, gt)[0, 1]) if n > 1 else float("nan"),
        "n":       n,
    }


def print_metrics(metrics, label=""):
    """Print evaluation metrics in a formatted way."""
    print(f"  {label}")
    print(f"    MAE     : {metrics['MAE']:.2f} ± {metrics['MAE_SE']:.2f} BPM")
    print(f"    RMSE    : {metrics['RMSE']:.2f} BPM")
    print(f"    MAPE    : {metrics['MAPE']:.1f}%")
    print(f"    Pearson : {metrics['Pearson']:.3f}")
    print(f"    n       : {metrics['n']}")


# ═══════════════════════════════════════════════
# Evaluate a single algorithm
# ═══════════════════════════════════════════════

def evaluate_algorithm(results, algo_name, save_dir="results/"):
    """Evaluate a complete algorithm.

    Args:
        results:   List of result dictionaries returned by method.estimate().
                   Each dictionary contains: hr_estimated, hr_ground_truth,
                   rr_estimated, rr_ground_truth
        algo_name: "ICA", "Garbey", "NoseROI"
        save_dir:  Directory where plots are saved

    Returns:
        Dictionary containing HR and RR metrics
    """
    hr_est = np.array([r["hr_estimated"] for r in results])
    hr_gt  = np.array([r["hr_ground_truth"] for r in results])
    rr_est = np.array([r["rr_estimated"] for r in results])
    rr_gt  = np.array([r["rr_ground_truth"] for r in results])

    # HR metrics
    hr_metrics = compute_metrics(hr_est, hr_gt)
    print(f"\n{'='*50}")
    print(f"  {algo_name} – Heart Rate")
    print(f"{'='*50}")
    print_metrics(hr_metrics, label="HR")

    # RR metrics
    rr_metrics = compute_metrics(rr_est, rr_gt)
    print(f"\n  {algo_name} – Respiration Rate")
    print(f"{'='*50}")
    print_metrics(rr_metrics, label="RR")

    # Bland-Altman Plots
    if hr_metrics["n"] >= 2:
        ba_hr = BlandAltman(
            gold_std    = hr_gt[~np.isnan(hr_gt) & ~np.isnan(hr_est)],
            new_measure = hr_est[~np.isnan(hr_gt) & ~np.isnan(hr_est)],
            save_path   = save_dir,
        )
        ba_hr.difference_plot(
            the_title=f"Heart Rate – {algo_name}",
            file_name=f"HR_{algo_name}_BlandAltman.pdf",
        )
        ba_hr.scatter_plot(
            the_title=f"Heart Rate – {algo_name}",
            file_name=f"HR_{algo_name}_Scatter.pdf",
            x_label="Ground Truth HR [BPM]",
            y_label="Estimated HR [BPM]",
        )

    if rr_metrics["n"] >= 2:
        ba_rr = BlandAltman(
            gold_std    = rr_gt[~np.isnan(rr_gt) & ~np.isnan(rr_est)],
            new_measure = rr_est[~np.isnan(rr_gt) & ~np.isnan(rr_est)],
            save_path   = save_dir,
        )
        ba_rr.difference_plot(
            the_title=f"Respiration Rate – {algo_name}",
            file_name=f"RR_{algo_name}_BlandAltman.pdf",
        )
        ba_rr.scatter_plot(
            the_title=f"Respiration Rate – {algo_name}",
            file_name=f"RR_{algo_name}_Scatter.pdf",
            x_label="Ground Truth RR [BPM]",
            y_label="Estimated RR [BPM]",
        )

    return {
        "algo": algo_name,
        "hr": hr_metrics,
        "rr": rr_metrics,
    }
