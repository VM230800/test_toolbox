"""
data/npz_loader.py
==================
Dataloader for the NPZ thermal dataset (Betreuer).
Handles thermal frames (real temperature in °C) and raw physiological
signals (pulse and respiration) from .npz files.

Dataset structure expected:
    IRData/
    ├── 005/
    │   ├── synchronized_data_0.npz
    │   ├── synchronized_data_1.npz
    │   └── ...
    ├── 006/
    │   └── ...
    └── 027/
        └── ...

Each .npz file contains:
    array1: (N, 768, 1024) float32  – thermal frames in °C
    array2: (N,) float64            – timestamps (for FPS calculation)
    array4: (N,) float64            – raw pulse signal
    array5: (N,) float64            – raw respiration signal
"""

import os
import glob
import numpy as np


class NPZDataset:

    def __init__(self, root_dir, subjects=None, recordings=None,
                 warmup_seconds=0, fps=30.0):
        """
        Initialize the NPZ dataset loader.

        Args:
            root_dir:        Path to the IRData root folder.
            subjects:        List of subject IDs, e.g. ["005", "007"].
                             If None, all available subjects are loaded.
            recordings:      List of recording IDs, e.g. [0, 1, 5].
                             If None, all available recordings are loaded.
            warmup_seconds:  Number of seconds to skip at the beginning
                             of each recording (camera warmup period).
            fps:             Fallback framerate if timestamps cannot
                             be used to compute the actual FPS.
        """
        self.root_dir = root_dir
        self.warmup_seconds = warmup_seconds
        self.default_fps = fps

        # ── Discover subjects ──
        if subjects is None:
            self.subjects = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ])
        else:
            self.subjects = subjects

        # ── Collect all valid (subject, recording_id, path) tuples ──
        self.samples = []
        for subj in self.subjects:
            subj_dir = os.path.join(root_dir, subj)
            npz_files = sorted(
                glob.glob(os.path.join(subj_dir, "synchronized_data_*.npz")),
                key=self._sort_key,
            )
            for npz_path in npz_files:
                # Extract recording ID from filename
                # "synchronized_data_5.npz" → 5
                fname = os.path.splitext(os.path.basename(npz_path))[0]
                rec_id = int(fname.split("_")[-1])

                # Skip recordings not in the requested list
                if recordings is not None and rec_id not in recordings:
                    continue

                self.samples.append((subj, rec_id, npz_path))

        print(f"NPZDataset: {len(self.subjects)} subjects, "
              f"{len(self.samples)} recordings")

    def __len__(self):
        """Return the total number of recordings."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return a single recording by index.

        Returns:
            dict with keys:
                - frames:       np.ndarray (N, H, W), float32, temperature in °C
                - fps:          float, computed from timestamps
                - subject:      str, subject ID (e.g. "005")
                - task:         str, recording name (e.g. "rec_0")
                - hr_bpm:       float, NaN (must be computed by pipeline)
                - rr_bpm:       float, NaN (must be computed by pipeline)
                - pulse_rate:   np.ndarray (N,), raw pulse signal
                - resp_rate:    np.ndarray (N,), raw respiration signal
                - signal_type:  str, "raw" (signals need peak detection)
        """
        subj, rec_id, npz_path = self.samples[idx]

        data = np.load(npz_path, allow_pickle=True)

        # ── Frames ──
        frames = data["array1"].astype(np.float32)

        # ── FPS from timestamps ──
        fps = self._compute_fps(data["array2"])

        # ── Raw physiology signals ──
        pulse_signal = data["array4"].astype(np.float64)
        resp_signal = data["array5"].astype(np.float64)

        # ── Remove warmup ──
        warmup_frames = int(self.warmup_seconds * fps)
        if warmup_frames > 0 and warmup_frames < len(frames):
            frames = frames[warmup_frames:]
            pulse_signal = pulse_signal[warmup_frames:]
            resp_signal = resp_signal[warmup_frames:]

        return {
            "frames":       frames,
            "fps":          fps,
            "subject":      subj,
            "task":         f"rec_{rec_id}",
            "recording_id": f"{subj}_rec_{rec_id}",
            "hr_bpm":       hr_bpm,                             
            "rr_bpm":       rr_bpm,                             
            "pulse_rate":   pulse_signal,
            "resp_rate":    resp_signal,
            "signal_type":  "raw",
        }

    def _compute_fps(self, timestamps):
        """
        Compute FPS from timestamp array.

        Takes the median of consecutive timestamp differences
        to be robust against occasional timing jitter.

        Args:
            timestamps: np.ndarray (N,) of timestamps

        Returns:
            float: computed FPS, or self.default_fps as fallback
        """
        if len(timestamps) < 2:
            return self.default_fps

        diffs = np.diff(timestamps)
        median_diff = np.median(diffs)

        if median_diff <= 0:
            return self.default_fps

        return 1.0 / median_diff

    @staticmethod
    def _sort_key(path):
        """
        Sort key for npz filenames by numeric ID.
        "synchronized_data_5.npz" → 5
        """
        fname = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(fname.split("_")[-1])
        except ValueError:
            return 0
