"""
data/npz_loader.py
==================
NPZ thermal dataset loader. Inherits from BaseLoader.
Only implements NPZ-specific logic.

Metadata (timestamps, physio) loaded separately
from frames. Frames array loaded once on first access,
then cached for instant per-frame streaming.
"""

import os
import glob
import numpy as np
from data.base_loader import BaseLoader


class NPZDataset(BaseLoader):

    def __init__(self, root_dir, subjects=None,
                 recordings=None,
                 warmup_seconds=0, fps=30.0,
                 cache_dir="cache",
                 force_preprocess=False):
        self.recordings_filter = recordings
        self._npz_cache = {}

        super().__init__(
            root_dir=root_dir,
            subjects=subjects,
            warmup_seconds=warmup_seconds,
            fps=fps,
            cache_dir=cache_dir,
            force_preprocess=force_preprocess,
        )

    # ─────────────────────────────────────────────────────
    # NPZ cache – metadata and frames separate
    # ─────────────────────────────────────────────────────

    def _get_npz_data(self, npz_path):
        """
        Load NPZ metadata (timestamps, physio) only.
        Does not load frames – use _get_frames_array().
        """
        if npz_path not in self._npz_cache:
            raw = np.load(npz_path, allow_pickle=True)
            self._npz_cache[npz_path] = {
                "array2": raw["array2"],
                "array4": raw["array4"],
                "array5": raw["array5"],
                "_shape": raw["array1"].shape,
            }
            raw.close()
        return self._npz_cache[npz_path]

    def _get_frames_array(self, npz_path):
        """
        Load frames array once and cache it.
        First call: ~30 sec (reads from disk).
        All subsequent calls: instant (from RAM).
        """
        cache_key = npz_path + "_frames"
        if cache_key not in self._npz_cache:
            print(f"    Loading frames into RAM: "
                  f"{os.path.basename(npz_path)}...")
            raw = np.load(npz_path, allow_pickle=True)
            self._npz_cache[cache_key] = raw["array1"]
            raw.close()
            shape = self._npz_cache[cache_key].shape
            print(f"    Done: {shape}")
        return self._npz_cache[cache_key]

    def _clear_cache(self, npz_path=None):
        """Free cached NPZ data to release RAM."""
        if npz_path is None:
            self._npz_cache.clear()
        else:
            keys = [k for k in self._npz_cache
                    if k.startswith(npz_path)]
            for k in keys:
                del self._npz_cache[k]

    # ─────────────────────────────────────────────────────
    # NPZ-specific implementations
    # ─────────────────────────────────────────────────────

    def _discover_samples(self, subjects):
        """Find all (subject, rec_id, path) combos."""
        if subjects is None:
            subjects = sorted([
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(
                    os.path.join(self.root_dir, d))
            ])

        samples = []
        for subj in subjects:
            subj_dir = os.path.join(
                self.root_dir, subj)
            npz_files = sorted(
                glob.glob(os.path.join(
                    subj_dir,
                    "synchronized_data_*.npz")),
                key=self._sort_key,
            )
            for npz_path in npz_files:
                fname = os.path.splitext(
                    os.path.basename(npz_path))[0]
                rec_id = int(fname.split("_")[-1])

                if (self.recordings_filter is not None
                        and rec_id
                        not in self.recordings_filter):
                    continue

                samples.append(
                    (subj, rec_id, npz_path))

        return samples

    def _load_frames(self, sample_info):
        """Load all frames from NPZ file."""
        subj, rec_id, npz_path = sample_info
        frames = self._get_frames_array(npz_path)
        result = frames.astype(np.float16)
        self._clear_cache(npz_path)
        return result

    def _load_single_frame(self, sample_info, frame_idx):
        """
        Load one frame. Frames array loaded once
        on first call, then cached for instant access.
        """
        subj, rec_id, npz_path = sample_info
        frames = self._get_frames_array(npz_path)
        return frames[frame_idx].astype(np.float16)

    def _get_total_frames(self, sample_info):
        """Get frame count without loading frames."""
        subj, rec_id, npz_path = sample_info
        data = self._get_npz_data(npz_path)
        return data["_shape"][0]

    def _get_fps(self, sample_info):
        """Compute FPS from timestamps."""
        subj, rec_id, npz_path = sample_info
        data = self._get_npz_data(npz_path)
        timestamps = data["array2"]

        if len(timestamps) < 2:
            return self.default_fps

        median_diff = np.median(np.diff(timestamps))

        if median_diff <= 0:
            return self.default_fps

        if median_diff > 1.0:
            return 1000.0 / median_diff
        else:
            return 1.0 / median_diff

    def _load_ground_truth(self, sample_info, fps):
        """Compute HR/RR from raw physiological signals."""
        subj, rec_id, npz_path = sample_info
        data = self._get_npz_data(npz_path)

        warmup_frames = int(self.warmup_seconds * fps)
        pulse_signal = data["array4"][
            warmup_frames:].astype(np.float64)
        resp_signal = data["array5"][
            warmup_frames:].astype(np.float64)

        hr_bpm = self._compute_bpm_from_fft(
            pulse_signal, fps,
            freq_range=(0.7, 3.5))
        rr_bpm = self._compute_bpm_from_fft(
            resp_signal, fps,
            freq_range=(0.1, 0.7))

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
            fft_vals = np.abs(
                np.fft.rfft(signal * window))
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
