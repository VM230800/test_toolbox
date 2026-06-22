"""
methods/garbey.py
=================
Vital sign estimation based on Garbey et al. (2007).

Method:
    1. Define a line along a blood vessel between two keypoints
    2. Sample temperature values along this line for each frame
    3. Average across the line width (perpendicular samples)
    4. Per line point: normalise, mirror, compute FFT
    5. Average power spectra across all line points
    6. Find dominant frequency in the physiological band
    7. Subharmonic correction (if detected peak is half the true rate)

Key difference to other methods:
    This method does NOT use area-based ROIs. Instead it samples
    along a 1D line that follows a superficial blood vessel.
    This is closer to the original thermal pulse measurement idea:
    the blood pulse travels along the vessel, creating a spatial-
    temporal pattern that can be captured along the vessel path.

References:
    Garbey, M., Sun, N., Merla, A., & Pavlidis, I. (2007).
    "Contact-Free Measurement of Cardiac Pulse Based on the
    Analysis of Thermal Imagery."
    IEEE Transactions on Biomedical Engineering, 54(8), 1418-1426.
"""

import warnings
import numpy as np


# ─────────────────────────────────────────────────────────────────
# Line definitions
# ─────────────────────────────────────────────────────────────────

# Default keypoint pairs for vessel lines.
# These indices refer to the YOLO 54-keypoint model
# (see preprocessing/yolo_keypoints.py).
#
# HR: Line along the left temple (superficial temporal artery)
#     Keypoint 0 (Schläfe oben) → Keypoint 2 (Wange oben)
#
# RR: Line from mouth corner to lower lip (airflow region)
#     Keypoint 48 (Mundwinkel links) → Keypoint 53 (Unterlippe)

DEFAULT_LINES = {
    "hr": {"p1": 0, "p2": 2, "name": "temporal_artery_left"},
    "rr": {"p1": 48, "p2": 53, "name": "nasal_airflow"},
}


# ─────────────────────────────────────────────────────────────────
# Line-based signal extraction (Garbey-specific)
# ─────────────────────────────────────────────────────────────────

def _extract_line_values(frames, keypoints, line_def,
                         line_points=10, line_width=3):
    """
    Sample temperature values along a vessel line for all frames.

    For each frame, a line is drawn between two keypoints.
    The line is divided into equally spaced sample points.
    At each point, values are averaged perpendicular to the line
    over a width of (2 * line_width + 1) pixels.

    Args:
        frames:      np.ndarray (N, H, W), temperature frames
        keypoints:   np.ndarray (N, K, 2), keypoint coordinates
        line_def:    dict with "p1" and "p2" (keypoint indices)
        line_points: int, number of sample points along the line
        line_width:  int, half-width for perpendicular averaging

    Returns:
        np.ndarray (N, line_points), temperature along the line
    """
    n_frames, H, W = frames.shape
    result = np.zeros((n_frames, line_points), dtype=np.float32)

    for i in range(n_frames):
        kp = keypoints[i]

        # Skip frames with NaN keypoints
        if np.isnan(kp[line_def["p1"]]).any() or \
           np.isnan(kp[line_def["p2"]]).any():
            if i > 0:
                result[i] = result[i - 1]  # repeat last valid
            continue

        p1 = kp[line_def["p1"]]
        p2 = kp[line_def["p2"]]

        # Line direction and perpendicular
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-3:
            if i > 0:
                result[i] = result[i - 1]
            continue

        direction = direction / length
        perpendicular = np.array([-direction[1], direction[0]])

        # Sample along the line
        for j in range(line_points):
            fraction = j / (line_points - 1)
            center = p1 + fraction * (p2 - p1)

            # Average perpendicular to the line
            values = []
            for offset in range(-line_width, line_width + 1):
                pixel = center + offset * perpendicular
                x = int(round(pixel[0]))
                y = int(round(pixel[1]))
                if 0 <= x < W and 0 <= y < H:
                    values.append(frames[i, y, x])

            if values:
                result[i, j] = np.mean(values)
            elif i > 0:
                result[i, j] = result[i - 1, j]

    return result


# ─────────────────────────────────────────────────────────────────
# Spectral analysis (Garbey-specific)
# ─────────────────────────────────────────────────────────────────

def _averaged_power_spectrum(line_values, fps):
    """
    Compute averaged power spectrum across all line points.

    Per line point:
        1. Remove mean (eliminate DC component)
        2. Mirror the signal (reduces Gibbs phenomenon at edges)
        3. Compute FFT power spectrum

    Then average all spectra across line points.

    Args:
        line_values: np.ndarray (N, n_points), temperature along line
        fps:         float, sampling rate

    Returns:
        freqs:    np.ndarray, frequency axis in Hz
        spectrum: np.ndarray, averaged power spectrum
    """
    n_frames, n_points = line_values.shape
    spectra = []

    for j in range(n_points):
        signal = line_values[:, j]

        # Remove mean
        signal = signal - np.mean(signal)

        # Mirror to reduce edge effects
        mirrored = np.concatenate([signal, signal[::-1]])

        # Power spectrum
        spectrum = np.abs(np.fft.rfft(mirrored)) ** 2
        spectra.append(spectrum)

    avg_spectrum = np.mean(spectra, axis=0)
    freqs = np.fft.rfftfreq(2 * n_frames, d=1.0 / fps)

    return freqs, avg_spectrum


def _find_dominant_frequency(freqs, spectrum, low_hz, high_hz):
    """
    Find the frequency with the highest power in the given band.

    Args:
        freqs:    np.ndarray, frequency axis
        spectrum: np.ndarray, power spectrum
        low_hz:   float, lower bound
        high_hz:  float, upper bound

    Returns:
        float, dominant frequency in Hz
    """
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not mask.any():
        warnings.warn(
            f"No FFT bins in [{low_hz:.2f}, {high_hz:.2f}] Hz. "
            f"Returning band center."
        )
        return (low_hz + high_hz) / 2.0

    band_freqs = freqs[mask]
    band_power = spectrum[mask]

    return float(band_freqs[np.argmax(band_power)])


def _subharmonic_correction(freq_hz, freqs, spectrum, low_hz, high_hz):
    """
    Correct subharmonic detection error.

    The FFT sometimes picks up half the true frequency as the
    dominant peak. If the double frequency also has a strong
    spectral line (>= 50% of the original peak) and falls within
    the valid band, use the double frequency instead.

    This correction is not in the original Garbey paper but was
    found necessary during development.

    Args:
        freq_hz: float, detected dominant frequency
        freqs:   np.ndarray, frequency axis
        spectrum: np.ndarray, power spectrum
        low_hz:  float, lower band limit
        high_hz: float, upper band limit

    Returns:
        float, corrected frequency in Hz
    """
    double_freq = freq_hz * 2.0

    if double_freq < low_hz or double_freq > high_hz:
        return freq_hz

    idx_original = np.argmin(np.abs(freqs - freq_hz))
    idx_double = np.argmin(np.abs(freqs - double_freq))

    power_original = spectrum[idx_original]
    power_double = spectrum[idx_double]

    if power_double >= 0.5 * power_original:
        return double_freq

    return freq_hz


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

class GarbeyMethod:
    """
    Garbey (2007) vessel-line based vital sign estimator.

    Usage:
        method = GarbeyMethod(config["methods"]["garbey"], config["signal"])
        result = method.estimate(frames, keypoints, fps)

    Note:
        Unlike ICA and thermal_mean, this method receives raw
        keypoints (N, 54, 2) instead of rois_per_frame, because
        it defines its own line-based ROIs from keypoint pairs.
    """

    def __init__(self, method_config, signal_config):
        """
        Args:
            method_config: dict from run_config.yaml, e.g.
                {"enabled": True, "target": "both",
                 "line_points": 10, "line_width": 3,
                 "lines": {"hr": {"p1": 0, "p2": 2}, ...}}
            signal_config: dict from run_config.yaml "signal" section
        """
        self.target = method_config.get("target", "both")
        self.line_points = method_config.get("line_points", 10)
        self.line_width = method_config.get("line_width", 3)
        self.signal_config = signal_config

        # Line definitions: from config or defaults
        self.lines = method_config.get("lines", DEFAULT_LINES)

    def estimate(self, frames, keypoints, fps):
        """
        Run Garbey estimation on one recording.

        Args:
            frames:    np.ndarray (N, H, W), temperature frames
            keypoints: np.ndarray (N, 54, 2), per-frame keypoints
            fps:       float

        Returns:
            dict with keys:
                - hr_bpm: float (or NaN)
                - rr_bpm: float (or NaN)
                - method: str, "garbey"
        """
        # Handle 3-channel frames → use first channel
        if frames.ndim == 4:
            frames = frames[:, :, :, 0].astype(np.float32)

        hr_bpm = float("nan")
        rr_bpm = float("nan")

        if self.target in ("hr", "both") and "hr" in self.lines:
            hr_bpm = self._estimate_single(
                frames, keypoints, fps, "hr"
            )

        if self.target in ("rr", "both") and "rr" in self.lines:
            rr_bpm = self._estimate_single(
                frames, keypoints, fps, "rr"
            )

        return {
            "hr_bpm": hr_bpm,
            "rr_bpm": rr_bpm,
            "method": "garbey",
        }

    def _estimate_single(self, frames, keypoints, fps, target):
        """
        Estimate one vital sign (HR or RR).

        Args:
            frames:    np.ndarray (N, H, W)
            keypoints: np.ndarray (N, 54, 2)
            fps:       float
            target:    "hr" or "rr"

        Returns:
            float, estimated rate in BPM
        """
        line_def = self.lines[target]
        bp = self.signal_config[f"{target}_bandpass"]
        low_hz = bp["low"]
        high_hz = bp["high"]

        # 1. Extract line values
        line_values = _extract_line_values(
            frames, keypoints, line_def,
            line_points=self.line_points,
            line_width=self.line_width,
        )

        # Check for valid data
        if np.isnan(line_values).all():
            return float("nan")

        # 2. Averaged power spectrum
        freqs, spectrum = _averaged_power_spectrum(line_values, fps)

        # 3. Dominant frequency
        freq_hz = _find_dominant_frequency(freqs, spectrum, low_hz, high_hz)

        # 4. Subharmonic correction
        freq_hz = _subharmonic_correction(
            freq_hz, freqs, spectrum, low_hz, high_hz
        )

        return freq_hz * 60.0
