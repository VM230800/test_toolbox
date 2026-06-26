"""
methods/thermal_mean.py
=======================
Vital sign estimation via mean ROI temperature fluctuations.

Method:
    1. Extract mean temperature per ROI per frame (→ time series)
    2. Bandpass filter to isolate HR or RR frequency range
    3. FFT to find dominant frequency
    4. Convert to BPM

This is the simplest of the three methods and serves as a
baseline. It uses the shared signal extraction and peak
extraction modules without any method-specific processing.

The underlying principle:
    Blood pulsation causes periodic temperature fluctuations
    on the skin surface (~0.01-0.1°C per heartbeat). These
    fluctuations are captured by averaging the temperature
    over a well-placed ROI (e.g. forehead, cheeks) and
    analysing the resulting time series in the frequency domain.

References:
    - Garbey et al. (2007): foundational thermal pulse work
    - Cho et al. (2017): ROI selection for thermal vital signs
    - Tarmizi et al. (2022): "A Review of Facial Thermography
      Assessment for Vital Signs Estimation"
"""

import numpy as np

from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)
from preprocessing.peak_extraction import (
    bandpass_filter,
    estimate_frequency_fft,
    estimate_frequency_peaks,
)


class ThermalMeanMethod:
    """
    Baseline method: mean ROI temperature → FFT → BPM.

    Usage:
        method = ThermalMeanMethod(
            config["methods"]["thermal_mean"],
            config["signal"],
        )
        result = method.estimate(frames, rois_per_frame, fps)
    """

    def __init__(self, method_config, signal_config):
        """
        Args:
            method_config: dict from run_config.yaml, e.g.
                {"enabled": True, "target": "both",
                 "rois": ["forehead", "left_cheek", "right_cheek"]}
            signal_config: dict from run_config.yaml "signal" section
        """
        self.rois = method_config["rois"]
        self.target = method_config.get("target", "both")
        self.signal_config = signal_config

    def estimate(self, frames, rois_per_frame, fps):
        """
        Run thermal mean estimation on one recording.

        Args:
            frames:         np.ndarray (N, H, W) or (N, H, W, 3)
            rois_per_frame: list of dict (from roi_extraction)
            fps:            float

        Returns:
            dict with keys:
                - hr_bpm:      float (or NaN)
                - rr_bpm:      float (or NaN)
                - roi_results: dict, per-ROI BPM estimates
                - method:      str, "thermal_mean"
        """
        # ── 1. Extract temperature signals per ROI ──
        roi_signals = extract_all_roi_signals(
            frames, rois_per_frame, self.rois
        )

        method = self.signal_config.get("peak_method", "fft")
        fft_window = self.signal_config.get("fft_window", 512)

        # ── 2. Estimate HR ──
        hr_bpm = float("nan")
        hr_roi_results = {}

        if self.target in ("hr", "both"):
            bp = self.signal_config["hr_bandpass"]
            valid_bpms = []

            for roi_name, signal in roi_signals.items():
                bpm = self._estimate_single_roi(
                    signal, fps, bp, method, fft_window)
                hr_roi_results[roi_name] = bpm
                if not np.isnan(bpm):
                    valid_bpms.append(bpm)

            if valid_bpms:
                hr_bpm = float(np.median(valid_bpms))

        # ── 3. Estimate RR ──
        rr_bpm = float("nan")
        rr_roi_results = {}

        if self.target in ("rr", "both"):
            bp = self.signal_config["rr_bandpass"]
            valid_bpms = []

            for roi_name, signal in roi_signals.items():
                bpm = self._estimate_single_roi(
                    signal, fps, bp, method, fft_window)
                rr_roi_results[roi_name] = bpm
                if not np.isnan(bpm):
                    valid_bpms.append(bpm)

            if valid_bpms:
                rr_bpm = float(np.median(valid_bpms))

        # ── 4. Build result ──
        result = {
            "hr_bpm":      hr_bpm,
            "rr_bpm":      rr_bpm,
            "roi_results": {
                "hr": hr_roi_results,
                "rr": rr_roi_results,
            },
            "method":      "thermal_mean",
        }

        return result

    def _estimate_single_roi(self, signal, fps, bp_config,
                             method, fft_window):
        """
        Estimate BPM from a single ROI signal.

        Args:
            signal:     np.ndarray (N,), may contain NaN
            fps:        float
            bp_config:  dict with "low", "high", "order"
            method:     "fft" or "peak_detection"
            fft_window: int

        Returns:
            float, BPM or NaN
        """
        signal_clean = interpolate_nan(signal)

        if np.isnan(signal_clean).all():
            return float("nan")

        min_length = bp_config["order"] * 3 + 1
        if len(signal_clean) < min_length:
            return float("nan")

        try:
            filtered = bandpass_filter(
                signal_clean, fps,
                low=bp_config["low"],
                high=bp_config["high"],
                order=bp_config["order"],
            )
        except ValueError:
            return float("nan")

        if method == "fft":
            freq_hz = estimate_frequency_fft(
                filtered, fps, fft_window)
        elif method == "peak_detection":
            freq_hz = estimate_frequency_peaks(
                filtered, fps)
        else:
            freq_hz = estimate_frequency_fft(
                filtered, fps, fft_window)

        return freq_hz * 60.0
