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
                fname = os.path.splitext(os.path.basename(npz_path))[0]
                rec_id = int(fname.split("_")[-1])

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

        # ── Compute BPM from raw signals ──
        hr_bpm = self._compute_bpm_from_peaks(
            pulse_signal, fps, freq_range=(0.7, 3.5))
        rr_bpm = self._compute_bpm_from_peaks(
            resp_signal, fps, freq_range=(0.1, 0.7))

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
        """
        if len(timestamps) < 2:
            return self.default_fps

        diffs = np.diff(timestamps)
        median_diff = np.median(diffs)

        if median_diff <= 0:
            return self.default_fps

        return 1.0 / median_diff

    @staticmethod
    def _compute_bpm_from_peaks(signal, fps, freq_range=(0.7, 3.5)):
        """
        Compute BPM from a raw physiological signal using FFT.

        Args:
            signal:     np.ndarray (N,), raw pulse or resp signal
            fps:        float, sampling rate
            freq_range: tuple (low, high) in Hz

        Returns:
            float: estimated BPM, or NaN if computation fails
        """
        try:
            if len(signal) < 10:
                return float("nan")

            # Remove DC offset
            signal = signal - np.mean(signal)

            # Bandpass via FFT
            n = len(signal)
            window = np.hanning(n)
            fft_vals = np.abs(np.fft.rfft(signal * window))
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)

            # Only look in the valid frequency range
            mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

            if not mask.any():
                return float("nan")

            fft_masked = fft_vals.copy()
            fft_masked[~mask] = 0

            # Find dominant frequency
            peak_idx = np.argmax(fft_masked)
            peak_freq = freqs[peak_idx]

            # Convert to BPM
            bpm = peak_freq * 60.0

            return float(bpm)

        except Exception:
            return float("nan")

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
