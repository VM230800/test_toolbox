"""
bland_altman.py
===============
Bland-Altman statistics and plots for the Thermal Vital Signs Toolbox.
 
Computes agreement metrics between a ground-truth reference signal
(e.g. contact ECG or respiratory belt) and a contactless estimate
derived from thermal video.
 
Two plot types are provided:
    scatter_plot     -- estimated vs. ground-truth values
    difference_plot  -- classic Bland-Altman: mean vs. difference
 
Both plots are saved as PDF files (vector format, publication-ready).
 
Usage
-----
    from evaluation.bland_altman import BlandAltman
 
    ba = BlandAltman(
        gold_std    = [72.1, 68.4, 75.0, ...],
        new_measure = [69.3, 70.1, 74.2, ...],
        save_path   = "results/bland_altman",
    )
    ba.print_stats()
    ba.difference_plot(the_title="Heart Rate -- ICA method on BP4D+")
    ba.scatter_plot(the_title="Heart Rate -- ICA method on BP4D+")
"""
 
from __future__ import annotations
 
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
 
 
class BlandAltman:
    """
    Computes Bland-Altman agreement statistics and generates plots.
 
    Parameters
    ----------
    gold_std    : array-like -- ground-truth values (contact sensor)
    new_measure : array-like -- toolbox estimates (thermal video)
    save_path   : str        -- directory where plots are saved
                               (created automatically if it does not exist)
    averaged    : bool       -- set True when each data point is already
                               a subject-level mean; adjusts the CI95
                               formula accordingly
    """
 
    def __init__(
        self,
        gold_std,
        new_measure,
        save_path: str = "results/bland_altman",
        averaged: bool = False,
    ):
        self.gold_std    = self._to_series(gold_std,    "gold_std")
        self.new_measure = self._to_series(new_measure, "new_measure")
 
        if len(self.gold_std) != len(self.new_measure):
            raise ValueError(
                f"gold_std and new_measure must have the same length "
                f"({len(self.gold_std)} vs {len(self.new_measure)})."
            )
 
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
 
        # Core statistics
        diffs = self.gold_std - self.new_measure
 
        self.mean_error              = float(diffs.mean())
        self.std_error               = float(diffs.std())
        self.mean_absolute_error     = float(diffs.abs().mean())
        self.mean_squared_error      = float((diffs ** 2).mean())
        self.root_mean_squared_error = float(np.sqrt(self.mean_squared_error))
        self.correlation             = float(
            np.corrcoef(self.gold_std, self.new_measure)[0, 1]
        )
 
        # 95% limits of agreement
        # When averaged=True the two measurements share error variance,
        # so the effective SD is scaled by sqrt(2).
        if averaged:
            effective_std = np.sqrt(2.0) * self.std_error
        else:
            effective_std = self.std_error
 
        self.CI95 = [
            self.mean_error + 1.96 * effective_std,
            self.mean_error - 1.96 * effective_std,
        ]
 
    # ------------------------------------------------------------------
    # Statistics output
    # ------------------------------------------------------------------
 
    def print_stats(self, round_amount: int = 4) -> None:
        """Print all computed metrics to stdout."""
        r = round_amount
        print(f"Mean error               = {round(self.mean_error,              r)}")
        print(f"Mean absolute error      = {round(self.mean_absolute_error,     r)}")
        print(f"Mean squared error       = {round(self.mean_squared_error,      r)}")
        print(f"Root mean squared error  = {round(self.root_mean_squared_error, r)}")
        print(f"Standard deviation error = {round(self.std_error,               r)}")
        print(f"Correlation              = {round(self.correlation,             r)}")
        print(f"+95% Limit of Agreement  = {round(self.CI95[0],                 r)}")
        print(f"-95% Limit of Agreement  = {round(self.CI95[1],                 r)}")
 
    def return_stats(self) -> dict:
        """Return all metrics as a plain dictionary (for CSV export etc.)."""
        return {
            "mean_error":              self.mean_error,
            "mean_absolute_error":     self.mean_absolute_error,
            "mean_squared_error":      self.mean_squared_error,
            "root_mean_squared_error": self.root_mean_squared_error,
            "std_error":               self.std_error,
            "correlation":             self.correlation,
            "CI_95_upper":             self.CI95[0],
            "CI_95_lower":             self.CI95[1],
        }
 
    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
 
    def scatter_plot(
        self,
        x_label: str       = "Ground Truth",
        y_label: str       = "Estimate",
        figure_size: tuple = (5, 5),
        show_legend: bool  = True,
        the_title: str     = "",
        file_name: str     = "scatter_plot.pdf",
        is_journal: bool   = False,
    ) -> None:
        """
        Scatter plot of estimate vs. ground-truth values.
 
        Points are density-coloured and slightly jittered so
        overlapping points remain visible. The ideal line of equality
        (slope = 1) is drawn as a dashed reference.
        """
        if is_journal:
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"]  = 42
 
        gold_j = self._jitter(self.gold_std.copy())
        new_j  = self._jitter(self.new_measure.copy())
 
        fig, ax = plt.subplots(figsize=figure_size)
 
        z  = gaussian_kde(np.vstack([gold_j, new_j]))(np.vstack([gold_j, new_j]))
        sc = ax.scatter(gold_j, new_j, c=z, s=40, cmap="plasma")
        plt.colorbar(sc, ax=ax, label="Point density")
 
        # Line of equality -- computed from the actual data range
        lim_min = min(gold_j.min(), new_j.min())
        lim_max = max(gold_j.max(), new_j.max())
        margin  = (lim_max - lim_min) * 0.05
        eq_vals = np.array([lim_min - margin, lim_max + margin])
        ax.plot(eq_vals, eq_vals, "--", color="black", linewidth=1,
                label="Line of equality")
 
        ax.set_xlim(eq_vals)
        ax.set_ylim(eq_vals)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        if show_legend:
            ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4)
 
        save_file = os.path.join(self.save_path, file_name)
        plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {save_file}")
 
    def difference_plot(
        self,
        x_label: str       = "Mean of Ground Truth and Estimate",
        y_label: str       = "Difference (Ground Truth - Estimate)",
        figure_size: tuple = (5, 5),
        show_legend: bool  = True,
        the_title: str     = "",
        file_name: str     = "bland_altman_difference_plot.pdf",
        is_journal: bool   = False,
    ) -> None:
        """
        Classic Bland-Altman difference plot.
 
        X-axis: mean of the two measurements.
        Y-axis: difference (ground truth minus estimate).
 
        Three horizontal reference lines are drawn:
            solid black  : mean error (bias)
            dashed black : +/- 95% limits of agreement
        """
        if is_journal:
            matplotlib.rcParams["pdf.fonttype"] = 42
            matplotlib.rcParams["ps.fonttype"]  = 42
 
        diffs = self.gold_std - self.new_measure
        avgs  = (self.gold_std + self.new_measure) / 2.0
 
        fig, ax = plt.subplots(figsize=figure_size)
 
        z = gaussian_kde(np.vstack([avgs, diffs]))(np.vstack([avgs, diffs]))
        ax.scatter(avgs, diffs, c=z, s=40, cmap="plasma", label="Observations")
 
        ax.axhline(self.mean_error, color="black", linewidth=1.2,
                   label=f"Mean error = {self.mean_error:.2f}")
        ax.axhline(self.CI95[0], color="black", linestyle="--", linewidth=0.9,
                   label=f"+95% LoA = {self.CI95[0]:.2f}")
        ax.axhline(self.CI95[1], color="black", linestyle="--", linewidth=0.9,
                   label=f"-95% LoA = {self.CI95[1]:.2f}")
 
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        ax.grid(True, linewidth=0.4)
        if show_legend:
            ax.legend(fontsize=8)
 
        save_file = os.path.join(self.save_path, file_name)
        plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {save_file}")
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _to_series(data, name: str) -> pd.Series:
        """Convert list, numpy array, or pandas Series to pandas Series."""
        if isinstance(data, pd.Series):
            return data.reset_index(drop=True)
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data, name=name, dtype=float)
        raise TypeError(
            f"{name} must be a list, numpy array, or pandas Series, "
            f"got {type(data)}."
        )
 
    @staticmethod
    def _jitter(arr: pd.Series) -> pd.Series:
        """
        Add tiny random noise so overlapping points remain visible.
        The noise magnitude is 1% of the data range.
        """
        data_range = arr.max() - arr.min()
        if data_range == 0:
            return arr
        return arr + np.random.randn(len(arr)) * 0.01 * data_range
 
 
# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    gt  = rng.normal(75, 10, 50)
    est = gt + rng.normal(2, 5, 50)
 
    ba = BlandAltman(
        gold_std    = gt,
        new_measure = est,
        save_path   = "/Users/valeriamoltschanov/Desktop/bland_altman_test",
    )
    ba.print_stats()
    ba.difference_plot(
        the_title = "Heart Rate -- ICA method on BP4D+",
        file_name = "hr_ica_difference.pdf",
    )
    ba.scatter_plot(
        the_title = "Heart Rate -- ICA method on BP4D+",
        file_name = "hr_ica_scatter.pdf",
        x_label   = "Ground Truth HR [BPM]",
        y_label   = "Estimated HR [BPM]",
    )
 
