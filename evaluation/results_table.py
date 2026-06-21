"""
results_table.py
================
Collects evaluation results from metrics_hr.py and metrics_rr.py and
produces a formatted comparison table across all methods and datasets.

The table is saved as both a CSV file (for further processing) and a
PDF file (publication-ready, for the report).

Expected usage in main.py
--------------------------
    from evaluation.results_table import ResultsTable

    table = ResultsTable(save_path="results/")

    # After running each method:
    table.add(
        dataset = "BP4D+",
        method  = "ICA",
        target  = "HR",
        metrics = evaluate_hr(signals, ground_truths, fs=25.0, method="FFT"),
    )
    table.add(
        dataset = "BP4D+",
        method  = "ICA",
        target  = "RR",
        metrics = evaluate_rr(signals, ground_truths, fs=25.0, method="FFT"),
    )

    # After all methods are done:
    table.print()
    table.save_csv()
    table.save_pdf()

Output CSV columns
------------------
    Dataset | Method | Target | MAE | MAE_SE | RMSE | MAPE | Pearson | SNR | n
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# Columns shown in the final table (in this order)
_DISPLAY_COLS = ["Dataset", "Method", "Target", "MAE", "MAE_SE",
                 "RMSE", "MAPE", "Pearson", "SNR_mean", "n"]

# How to rename columns for the PDF/print output
_COL_LABELS = {
    "Dataset":  "Dataset",
    "Method":   "Method",
    "Target":   "Target",
    "MAE":      "MAE [BPM]",
    "MAE_SE":   "± SE",
    "RMSE":     "RMSE",
    "MAPE":     "MAPE [%]",
    "Pearson":  "Pearson r",
    "SNR_mean": "SNR [dB]",
    "n":        "n",
}


class ResultsTable:
    """
    Accumulates per-method evaluation results and renders a comparison table.

    Parameters
    ----------
    save_path : str -- directory where CSV and PDF are saved
    """

    def __init__(self, save_path: str = "results"):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self._rows: list[dict] = []

    def add(
        self,
        dataset: str,
        method: str,
        target: str,
        metrics: dict,
    ) -> None:
        """
        Add one result row to the table.

        Parameters
        ----------
        dataset : str  -- e.g. "BP4D+"
        method  : str  -- e.g. "ICA", "Garbey", "ROI-Nase"
        target  : str  -- "HR" or "RR"
        metrics : dict -- output of evaluate_hr() or evaluate_rr()
        """
        row = {
            "Dataset":  dataset,
            "Method":   method,
            "Target":   target,
            "MAE":      round(metrics.get("MAE",      float("nan")), 3),
            "MAE_SE":   round(metrics.get("MAE_SE",   float("nan")), 3),
            "RMSE":     round(metrics.get("RMSE",     float("nan")), 3),
            "MAPE":     round(metrics.get("MAPE",     float("nan")), 2),
            "Pearson":  round(metrics.get("Pearson",  float("nan")), 3),
            "SNR_mean": round(metrics.get("SNR_mean", float("nan")), 2),
            "n":        int(metrics.get("n", 0)),
        }
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the accumulated results as a pandas DataFrame."""
        if not self._rows:
            return pd.DataFrame(columns=_DISPLAY_COLS)
        df = pd.DataFrame(self._rows)
        # Ensure consistent column order
        for col in _DISPLAY_COLS:
            if col not in df.columns:
                df[col] = float("nan")
        return df[_DISPLAY_COLS].sort_values(["Target", "Dataset", "Method"])

    def print(self) -> None:
        """Print the results table to stdout."""
        df = self.to_dataframe()
        if df.empty:
            print("No results to display.")
            return
        df_display = df.rename(columns=_COL_LABELS)
        print("\n" + "=" * 80)
        print("  Evaluation Results")
        print("=" * 80)
        print(df_display.to_string(index=False))
        print("=" * 80 + "\n")

    def save_csv(self, filename: str = "results.csv") -> str:
        """
        Save results as a CSV file.

        Returns the full path to the saved file.
        """
        df = self.to_dataframe()
        path = os.path.join(self.save_path, filename)
        df.to_csv(path, index=False)
        print(f"CSV saved: {path}")
        return path

    def save_pdf(self, filename: str = "results_table.pdf") -> str:
        """
        Save results as a formatted PDF table.

        The table is styled to be publication-ready:
        - HR and RR sections are visually separated
        - Best MAE per target/dataset group is highlighted in green
        - Standard errors are shown next to MAE values

        Returns the full path to the saved file.
        """
        df = self.to_dataframe()
        if df.empty:
            print("No results to save.")
            return ""

        # Build display version: merge MAE and MAE_SE into one column
        df_pdf = df.copy()
        df_pdf["MAE ± SE"] = df_pdf.apply(
            lambda r: f"{r['MAE']:.3f} ± {r['MAE_SE']:.3f}", axis=1
        )
        pdf_cols   = ["Dataset", "Method", "Target", "MAE ± SE",
                      "RMSE", "MAPE", "Pearson", "SNR_mean", "n"]
        col_labels = ["Dataset", "Method", "Target", "MAE ± SE [BPM]",
                      "RMSE", "MAPE [%]", "Pearson r", "SNR [dB]", "n"]

        df_pdf = df_pdf[pdf_cols]
        n_rows = len(df_pdf)

        fig_h = max(2.5, 0.5 * n_rows + 2.0)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText  = df_pdf.values,
            colLabels = col_labels,
            cellLoc   = "center",
            loc       = "center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(pdf_cols))))

        # Style header row
        for j in range(len(pdf_cols)):
            cell = tbl[0, j]
            cell.set_facecolor("#2C2C2A")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#444441")

        # Style data rows: alternate shading + highlight best MAE per group
        _best_mae = {}
        for _, grp in df.groupby(["Target", "Dataset"]):
            idx = grp["MAE"].idxmin()
            _best_mae[idx] = True

        for i, (orig_idx, row) in enumerate(df.iterrows()):
            row_num = i + 1
            bg = "#F9F8F5" if i % 2 == 0 else "white"
            for j in range(len(pdf_cols)):
                cell = tbl[row_num, j]
                cell.set_edgecolor("#D3D1C7")
                if orig_idx in _best_mae:
                    cell.set_facecolor("#EAF3DE")   # green highlight
                else:
                    cell.set_facecolor(bg)

        # Title and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        ax.set_title(
            f"Thermal Vital Signs Toolbox — Evaluation Results\n"
            f"Generated: {timestamp}",
            fontsize=10, pad=16, loc="left",
        )

        plt.tight_layout()
        path = os.path.join(self.save_path, filename)
        plt.savefig(path, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"PDF saved: {path}")
        return path


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Simulate what main.py would do after running all methods
    fake_hr_results = {
        "MAE": 8.32, "MAE_SE": 1.14, "RMSE": 11.5,
        "MAPE": 9.2, "Pearson": 0.61, "SNR_mean": 3.2, "n": 10,
    }
    fake_rr_results = {
        "MAE": 3.11, "MAE_SE": 0.52, "RMSE": 4.2,
        "MAPE": 18.4, "Pearson": 0.74, "SNR_mean": 5.1, "n": 10,
    }
    fake_hr_better = {
        "MAE": 5.10, "MAE_SE": 0.88, "RMSE": 7.3,
        "MAPE": 6.1, "Pearson": 0.82, "SNR_mean": 6.4, "n": 10,
    }
    fake_rr_better = {
        "MAE": 2.05, "MAE_SE": 0.31, "RMSE": 2.9,
        "MAPE": 12.1, "Pearson": 0.88, "SNR_mean": 7.8, "n": 10,
    }

    table = ResultsTable(
        save_path="/Users/valeriamoltschanov/Desktop/results_table_test"
    )

    table.add("BP4D+", "ICA",      "HR", fake_hr_results)
    table.add("BP4D+", "ICA",      "RR", fake_rr_results)
    table.add("BP4D+", "Garbey",   "HR", fake_hr_better)
    table.add("BP4D+", "Garbey",   "RR", fake_rr_better)
    table.add("BP4D+", "ROI-Nase", "HR", fake_hr_results)
    table.add("BP4D+", "ROI-Nase", "RR", fake_rr_better)

    table.print()
    table.save_csv()
    table.save_pdf()
