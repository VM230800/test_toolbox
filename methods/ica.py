"""
methods/ica.py
==============
ICA-based vital sign estimation from thermal video.

Uses FastICA to separate mixed thermal signals into
independent components, then selects the component
with the strongest periodic signal in the HR/RR band.

References:
    - Poh et al. (2010): "Non-contact, automated cardiac
      pulse measurements using video imaging and blind
      source separation"
    - Adapted from rPPG (visible light) to thermal imaging
"""

import warnings
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.decomposition import FastICA

from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)


# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────

def _detrend(signal, fps):
    """
    Remove slow drift using Savitzky-Golay filter.
    Window adapts to signal length.
    """
    window = int(4.0 * fps)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)

    # Window must be smaller than signal
    if len(signal) <= window:
        return signal - np.mean(signal)

    trend = savgol_filter(signal, window, polyorder=2)
    return signal - trend


def _preprocess_for_ica(signals, fps, low_hz, high_hz):
    """
    Prepare multi-ROI signals for ICA decomposition.

    Pipeline:
        1. Interpolate NaN gaps
        2. Remove slow drift
        3. Bandpass filter
        4. Z-score normalisation
    """
    nyq = fps / 2.0
    low_norm = np.clip(low_hz / nyq, 0.001, 0.999)
    high_norm = np.clip(high_hz / nyq, 0.001, 0.999)

    if low_norm >= high_norm:
        return None, []

    roi_names = []
    channels = []

    for name, raw_signal in signals.items():
        # 1. Interpolate NaN
        s = interpolate_nan(raw_signal)
        if np.isnan(s).all():
            continue

        # 2. Detrend
        s = _detrend(s, fps)

        # 3. Bandpass
        if low_norm < high_norm:
            try:
                b, a = butter(
                    4, [low_norm, high_norm],
                    btype="band")
                s = filtfilt(b, a, s)
            except Exception:
                continue

        # 4. Normalise
        std = s.std()
        if std > 1e-10:
            s = (s - s.mean()) / std
        else:
            continue

        roi_names.append(name)
        channels.append(s)

    if len(channels) < 2:
        return None, []

    matrix = np.column_stack(channels)
    return matrix, roi_names


# ─────────────────────────────────────────────────────────
# ICA decomposition
# ─────────────────────────────────────────────────────────

def _run_ica(signals, n_components=None):
    """
    Apply FastICA with built-in whitening.
    Simple and robust – no PCA pre-whitening needed.
    """
    if n_components is None:
        n_components = signals.shape[1]

    n_components = min(n_components, signals.shape[1])

    ica = FastICA(
        n_components=n_components,
        max_iter=1000,
        tol=1e-4,
        random_state=42,
        whiten="unit-variance",
    )

    try:
        result = ica.fit_transform(signals)
        return result
    except Exception as e:
        warnings.warn(f"ICA failed: {e}")
        return signals


# ─────────────────────────────────────────────────────────
# Component selection
# ─────────────────────────────────────────────────────────

def _select_best_component(components, fps,
                            low_hz, high_hz):
    """
    Select ICA component with strongest FFT peak
    in the target frequency band.

    Simple and reliable: just find which component
    has the most concentrated energy in the band.
    """
    N, K = components.shape

    best_score = -1.0
    best_freq = 0.0
    best_idx = 0

    # Zero-pad for better frequency resolution
    nfft = max(512, N * 2)

    for k in range(K):
        comp = components[:, k]

        # FFT with zero-padding
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
        window = np.hanning(N)
        fft_vals = np.abs(
            np.fft.rfft(comp * window, n=nfft))

        # Only look in target band
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not mask.any():
            continue

        band_fft = fft_vals[mask]
        band_freqs = freqs[mask]

        # Score = peak power / total band power
        total = band_fft.sum()
        if total < 1e-10:
            continue

        peak_idx = np.argmax(band_fft)
        peak_power = band_fft[peak_idx]
        peak_freq = band_freqs[peak_idx]

        # Concentration: how much energy is at the peak
        # vs spread across the band
        score = float(peak_power / total)

        if score > best_score:
            best_score = score
            best_idx = k
            best_freq = peak_freq

    return best_freq, best_idx


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

class ICAMethod:
    """
    ICA-based vital sign estimator.

    Extracts temperature signals from multiple face ROIs,
    applies ICA to separate mixed sources, then selects
    the component with the strongest periodic signal.
    """

    def __init__(self, method_config, signal_config):
        self.rois = method_config["rois"]
        self.target = method_config.get("target", "both")
        self.n_components = method_config.get(
            "n_components", None)
        self.signal_config = signal_config

    def estimate(self, frames, rois_per_frame, fps):
        """
        Run ICA estimation on one recording.

        Args:
            frames:         (N, H, W, 3) uint8
            rois_per_frame: list of ROI dicts
            fps:            float

        Returns:
            dict with hr_bpm, rr_bpm, method
        """
        # ── 1. Extract ROI signals ──
        roi_signals = extract_all_roi_signals(
            frames, rois_per_frame, self.rois)

        valid_count = sum(
            1 for s in roi_signals.values()
            if not np.isnan(s).all()
        )
        if valid_count < 2:
            warnings.warn(
                "ICA needs at least 2 valid ROI signals.")
            return self._empty_result()

        # ── 2. Estimate HR ──
        hr_bpm = float("nan")

        if self.target in ("hr", "both"):
            hr_bp = self.signal_config["hr_bandpass"]
            matrix_hr, names_hr = _preprocess_for_ica(
                roi_signals, fps,
                hr_bp["low"], hr_bp["high"],
            )

            if (matrix_hr is not None
                    and matrix_hr.shape[1] >= 2):
                components_hr = _run_ica(
                    matrix_hr, self.n_components)
                hr_freq, hr_idx = \
                    _select_best_component(
                        components_hr, fps,
                        hr_bp["low"], hr_bp["high"],
                    )
                if hr_freq > 0:
                    hr_bpm = hr_freq * 60.0

        # ── 3. Estimate RR ──
        rr_bpm = float("nan")

        if self.target in ("rr", "both"):
            rr_bp = self.signal_config["rr_bandpass"]
            matrix_rr, names_rr = _preprocess_for_ica(
                roi_signals, fps,
                rr_bp["low"], rr_bp["high"],
            )

            if (matrix_rr is not None
                    and matrix_rr.shape[1] >= 2):
                components_rr = _run_ica(
                    matrix_rr, self.n_components)
                rr_freq, rr_idx = \
                    _select_best_component(
                        components_rr, fps,
                        rr_bp["low"], rr_bp["high"],
                    )
                if rr_freq > 0:
                    rr_bpm = rr_freq * 60.0

        return {
            "hr_bpm": hr_bpm,
            "rr_bpm": rr_bpm,
            "method": "ica",
        }

    @staticmethod
    def _empty_result():
        return {
            "hr_bpm": float("nan"),
            "rr_bpm": float("nan"),
            "method": "ica",
        }
