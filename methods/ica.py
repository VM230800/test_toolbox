"""
methods/ica.py
==============
ICA-based vital sign estimation from thermal video.

Improvements over basic ICA:
    - PCA whitening before ICA (reduces noise dimensions)
    - Multiple random restarts for robust decomposition
    - Combined component selection: FFT peak + autocorrelation
    - Better detrending (4s window)
    - Zero-padding for improved frequency resolution

References:
    - Poh et al. (2010): "Non-contact, automated cardiac pulse
      measurements using video imaging and blind source separation"
    - Adapted from rPPG (visible light) to thermal imaging
"""

import warnings
import numpy as np
from scipy.signal import (
    butter, filtfilt, savgol_filter, welch,
)
from sklearn.decomposition import FastICA, PCA

from preprocessing.signal_extraction import (
    extract_all_roi_signals,
    interpolate_nan,
)


# ─────────────────────────────────────────────────────────
# Improved preprocessing
# ─────────────────────────────────────────────────────────

def _detrend(signal, fps):
    """
    Remove slow drift using Savitzky-Golay filter.
    Uses 4-second window for better baseline removal.
    """
    window = int(4.0 * fps)
    if window % 2 == 0:
        window += 1
    window = max(window, 5)
    if len(signal) <= window:
        return signal - np.mean(signal)

    trend = savgol_filter(signal, window, polyorder=2)
    return signal - trend


def _preprocess_for_ica(signals, fps, low_hz, high_hz):
    """
    Prepare multi-ROI signals for ICA decomposition.

    Improved pipeline:
        1. Interpolate NaN gaps
        2. Remove slow drift (4s Savitzky-Golay)
        3. Bandpass filter (order 5 for steeper rolloff)
        4. Z-score normalisation
        5. PCA whitening (reduces noise dimensions)
    """
    nyq = fps / 2.0
    low_norm = np.clip(low_hz / nyq, 0.001, 0.999)
    high_norm = np.clip(high_hz / nyq, 0.001, 0.999)

    roi_names = []
    channels = []

    for name, raw_signal in signals.items():
        # 1. Interpolate NaN
        s = interpolate_nan(raw_signal)
        if np.isnan(s).all():
            continue

        # 2. Detrend (4s window)
        s = _detrend(s, fps)

        # 3. Bandpass (steeper filter, order 5)
        if low_norm < high_norm:
            try:
                b, a = butter(
                    5, [low_norm, high_norm], btype="band")
                s = filtfilt(b, a, s)
            except Exception:
                b, a = butter(
                    3, [low_norm, high_norm], btype="band")
                s = filtfilt(b, a, s)

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

    # 5. PCA whitening
    try:
        n_keep = min(matrix.shape[1], matrix.shape[0] // 10)
        n_keep = max(n_keep, 2)
        n_keep = min(n_keep, matrix.shape[1])

        pca = PCA(n_components=n_keep, whiten=True)
        matrix = pca.fit_transform(matrix)

        # Update roi names
        roi_names = [f"PC{i}" for i in range(n_keep)]
    except Exception:
        pass  # Fall back to unwhitened

    return matrix, roi_names


# ─────────────────────────────────────────────────────────
# Improved ICA decomposition
# ─────────────────────────────────────────────────────────

def _run_ica(signals, n_components=None, n_restarts=5):
    """
    Apply FastICA with multiple random restarts.

    Runs ICA multiple times with different seeds and
    keeps the result with best convergence.
    """
    if n_components is None:
        n_components = signals.shape[1]

    n_components = min(n_components, signals.shape[1])

    best_result = None
    best_score = -np.inf

    for seed in range(n_restarts):
        ica = FastICA(
            n_components=n_components,
            max_iter=2000,
            tol=1e-4,
            random_state=seed,
            whiten=False,  # Already whitened by PCA
        )

        try:
            result = ica.fit_transform(signals)

            # Score: sum of kurtosis (higher = more
            # non-Gaussian = better separation)
            kurtosis = np.mean(result ** 4, axis=0) - 3.0
            score = np.sum(np.abs(kurtosis))

            if score > best_score:
                best_score = score
                best_result = result

        except Exception:
            continue

    if best_result is None:
        warnings.warn("All ICA restarts failed.")
        return signals

    return best_result


# ─────────────────────────────────────────────────────────
# Improved component selection
# ─────────────────────────────────────────────────────────

def _autocorrelation_score(signal, fps, low_hz, high_hz):
    """
    Compute periodicity score using autocorrelation.

    A periodic signal (like a pulse) has strong peaks
    in its autocorrelation at the expected lag.
    """
    n = len(signal)
    if n < 10:
        return 0.0

    # Expected lag range
    min_lag = int(fps / high_hz)
    max_lag = int(fps / low_hz)
    max_lag = min(max_lag, n // 2)

    if min_lag >= max_lag or max_lag < 2:
        return 0.0

    # Normalised autocorrelation
    sig = signal - np.mean(signal)
    norm = np.sum(sig ** 2)
    if norm < 1e-10:
        return 0.0

    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[n - 1:]  # Only positive lags
    autocorr = autocorr / norm

    # Find strongest peak in expected range
    search = autocorr[min_lag:max_lag + 1]
    if len(search) == 0:
        return 0.0

    return float(np.max(search))


def _spectral_score(signal, fps, low_hz, high_hz):
    """
    Compute spectral concentration score.

    Uses Welch's method for smoother spectrum.
    Score = peak power / total band power.
    Higher = more concentrated = more likely a vital sign.
    """
    n = len(signal)

    # Zero-pad for better frequency resolution
    nfft = max(256, 1)
    while nfft < n * 2:
        nfft <<= 1

    # Welch's method (smoother than raw FFT)
    try:
        nperseg = min(n, nfft // 2)
        freqs, psd = welch(
            signal, fs=fps,
            nperseg=nperseg,
            nfft=nfft,
            noverlap=nperseg // 2,
        )
    except Exception:
        # Fallback to simple FFT
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)
        psd = np.abs(np.fft.rfft(signal, n=nfft)) ** 2

    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not mask.any():
        return 0.0, 0.0

    band_psd = psd[mask]
    band_freqs = freqs[mask]

    total = band_psd.sum()
    if total < 1e-10:
        return 0.0, 0.0

    peak_idx = np.argmax(band_psd)
    peak_freq = float(band_freqs[peak_idx])
    concentration = float(band_psd[peak_idx] / total)

    return concentration, peak_freq


def _select_best_component(components, fps, low_hz, high_hz):
    """
    Select best ICA component using combined scoring.

    Combines:
        - Spectral concentration (FFT peak / band energy)
        - Autocorrelation periodicity
        - Signal kurtosis (non-Gaussianity)

    Weights:
        score = 0.5 * spectral + 0.3 * autocorr + 0.2 * kurtosis
    """
    N, K = components.shape

    best_score = -1.0
    best_freq = 0.0
    best_idx = 0
    all_scores = []

    for k in range(K):
        comp = components[:, k]

        # 1. Spectral concentration (0-1)
        spec_score, peak_freq = _spectral_score(
            comp, fps, low_hz, high_hz)

        # 2. Autocorrelation periodicity (0-1)
        auto_score = _autocorrelation_score(
            comp, fps, low_hz, high_hz)

        # 3. Kurtosis (non-Gaussianity, normalised)
        kurt = abs(
            float(np.mean(comp ** 4) / (
                np.mean(comp ** 2) ** 2 + 1e-10)
            ) - 3.0
        )
        kurt_norm = min(kurt / 10.0, 1.0)

        # Combined score
        score = (0.5 * spec_score
                 + 0.3 * auto_score
                 + 0.2 * kurt_norm)

        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_idx = k
            best_freq = peak_freq

    return best_freq, best_idx, all_scores


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

class ICAMethod:
    """
    Improved ICA-based vital sign estimator.

    Key improvements:
        - PCA whitening before ICA
        - Multiple ICA restarts
        - Combined component selection
        - Better detrending and filtering
    """

    def __init__(self, method_config, signal_config):
        self.rois = method_config["rois"]
        self.target = method_config.get("target", "both")
        self.n_components = method_config.get(
            "n_components", None)
        self.n_restarts = method_config.get(
            "n_restarts", 5)
        self.signal_config = signal_config

    def estimate(self, frames, rois_per_frame, fps):
        """
        Run improved ICA estimation on one recording.
        """
        # ── 1. Extract temperature signals per ROI ──
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
        hr_idx = -1

        if self.target in ("hr", "both"):
            hr_bp = self.signal_config["hr_bandpass"]
            matrix_hr, names_hr = _preprocess_for_ica(
                roi_signals, fps,
                hr_bp["low"], hr_bp["high"],
            )

            if (matrix_hr is not None
                    and matrix_hr.shape[1] >= 2):
                components_hr = _run_ica(
                    matrix_hr, self.n_components,
                    self.n_restarts,
                )
                hr_freq, hr_idx, _ = \
                    _select_best_component(
                        components_hr, fps,
                        hr_bp["low"], hr_bp["high"],
                    )
                hr_bpm = hr_freq * 60.0

        # ── 3. Estimate RR ──
        rr_bpm = float("nan")
        rr_idx = -1

        if self.target in ("rr", "both"):
            rr_bp = self.signal_config["rr_bandpass"]
            matrix_rr, names_rr = _preprocess_for_ica(
                roi_signals, fps,
                rr_bp["low"], rr_bp["high"],
            )

            if (matrix_rr is not None
                    and matrix_rr.shape[1] >= 2):
                components_rr = _run_ica(
                    matrix_rr, self.n_components,
                    self.n_restarts,
                )
                rr_freq, rr_idx, _ = \
                    _select_best_component(
                        components_rr, fps,
                        rr_bp["low"], rr_bp["high"],
                    )
                rr_bpm = rr_freq * 60.0

        return {
            "hr_bpm":           hr_bpm,
            "rr_bpm":           rr_bpm,
            "hr_component_idx": hr_idx,
            "rr_component_idx": rr_idx,
            "method":           "ica",
        }

    @staticmethod
    def _empty_result():
        return {
            "hr_bpm":           float("nan"),
            "rr_bpm":           float("nan"),
            "hr_component_idx": -1,
            "rr_component_idx": -1,
            "method":           "ica",
        }
