"""
data/npz_loader.py
==================
NPZ thermal dataset loader. Inherits from BaseLoader.
Only implements NPZ-specific logic.
"""

import os
import glob
import numpy as np
from data.base_loader import BaseLoader


class NPZDataset(BaseLoader):

    def __init__(self, root_dir, subjects=None, recordings=None,
                 warmup_seconds=0, fps=30.0,
                 cache_dir="cache", force_preprocess=False):
        self.recordings_filter = recordings

        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            warmup_seconds=warmup_seconds,
            fps=fps,
            cache_dir=cache_dir,
            force_preprocess=force_preprocess,
        )

    # ─────────────────────────────────────────────────────
    # NPZ-specific implementations
    # ─────────────────────────────────────────────────────

    def _discover_samples(self, subjects):
        """Find all (subject, rec_id, path) combinations."""
        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])

        samples = []
        for subj in subjects:
            subj_dir = os.path.join(self.root_dir, subj)
            npz_files = sorted(
                glob.glob(os.path.join(
                    subj_dir, "synchronized_data_*.npz")),
                key=self._sort_key,
            )
            for npz_path in npz_files:
                fname = os.path.splitext(
                    os.path.basename(npz_path))[0]
                rec_id = int(fname.split("_")[-1])

                if (self.recordings_filter is not None
                        and rec_id not in self.recordings_filter):
                    continue

                samples.append((subj, rec_id, npz_path))

        return samples

    def _load_frames(self, sample_info):
        """Load all frames from NPZ file."""
        subj, rec_id, npz_path = sample_info

        data = np.load(npz_path, allow_pickle=True)
        return data["array1"].astype(np.float16)

    def _load_single_frame(self, sample_info, frame_idx):
        """Load one frame from NPZ file (for streaming)."""
        subj, rec_id, npz_path = sample_info

        data = np.load(npz_path, allow_pickle=True)
        return data["array1"][frame_idx].astype(np.float32)

    def _get_total_frames(self, sample_info):
        """Get frame count from NPZ array."""
        subj, rec_id, npz_path = sample_info

        data = np.load(npz_path, allow_pickle=True)
        return len(data["array1"])

    def _get_fps(self, sample_info):
        """Compute FPS from timestamps."""
        subj, rec_id, npz_path = sample_info

        data = np.load(npz_path, allow_pickle=True)
        timestamps = data["array2"]

        if len(timestamps) < 2:
            return self.default_fps

        median_diff = np.median(np.diff(timestamps))

        if median_diff <= 0:
            return self.default_fps

        # Timestamps in milliseconds → convert
        if median_diff > 1.0:
            return 1000.0 / median_diff
        else:
            return 1.0 / median_diff

    def _load_ground_truth(self, sample_info, fps):
        """Compute HR/RR from raw physiological signals."""
        subj, rec_id, npz_path = sample_info

        data = np.load(npz_path, allow_pickle=True)

        # Apply warmup removal to signals too
        warmup_frames = int(self.warmup_seconds * fps)
        pulse_signal = data["array4"][warmup_frames:].astype(
            np.float64)
        resp_signal = data["array5"][warmup_frames:].astype(
            np.float64)

        hr_bpm = self._compute_bpm_from_fft(
            pulse_signal, fps, freq_range=(0.7, 3.5))
        rr_bpm = self._compute_bpm_from_fft(
            resp_signal, fps, freq_range=(0.1, 0.7))

        return {
            "hr_bpm":      hr_bpm,
            "rr_bpm":      rr_bpm,
            "pulse_rate":  pulse_signal,
            "resp_rate":   resp_signal,
            "signal_type": "raw",
        }

    def _get_subject(self, sample_info):
        return sample_info[0]

    def _get_task(self, sample_info):
        return f"rec_{sample_info[1]}"

    # ─────────────────────────────────────────────────────
    # NPZ-specific helper methods
    # ─────────────────────────────────────────────────────

    @staticmethod
    def _compute_bpm_from_fft(signal, fps,
                               freq_range=(0.7, 3.5)):
        """Compute BPM from raw signal using FFT."""
        try:
            if len(signal) < 10:
                return float("nan")

            signal = signal - np.mean(signal)

            n = len(signal)
            window = np.hanning(n)
            fft_vals = np.abs(np.fft.rfft(signal * window))
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)

            mask = ((freqs >= freq_range[0])
                    & (freqs <= freq_range[1]))
            if not mask.any():
                return float("nan")

            fft_masked = fft_vals.copy()
            fft_masked[~mask] = 0

            peak_idx = np.argmax(fft_masked)
            peak_freq = freqs[peak_idx]

            return float(peak_freq * 60.0)

        except Exception:
            return float("nan")

    @staticmethod
    def _sort_key(path):
        """Sort npz filenames by numeric ID."""
        fname = os.path.splitext(
            os.path.basename(path))[0]
        try:
            return int(fname.split("_")[-1])
        except ValueError:
            return 0
