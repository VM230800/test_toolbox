"""
preprocessing/signal_extraction.py
==================================
Extracts temperature time series from ROIs across all frames.

Input:  Thermal frames + ROI definitions per frame
Output: 1D temperature signal per ROI (mean temperature over time)

This module does no filtering or frequency estimation.
It only converts spatial ROI data into temporal signals.
"""

import numpy as np


def extract_roi_signal(frames, rois_per_frame, roi_name):
    """
    Compute mean temperature of a single ROI across all frames.

    For each frame, the ROI is a circular patch defined by
    (center_x, center_y, radius) from roi_extraction.compute_rois().
    The mean pixel value inside this patch is taken as the
    temperature for that frame.

    Args:
        frames:         np.ndarray (N, H, W) or (N, H, W, 3)
                        Thermal frames (cropped).
        rois_per_frame: list of dict or None, length N.
                        Each dict maps ROI names to (cx, cy, radius).
                        None entries = no detection for that frame.
        roi_name:       str, e.g. "forehead", "nose", "philtrum"

    Returns:
        np.ndarray (N,) float64, mean temperature per frame.
        NaN for frames where the ROI was not available.
    """
    n_frames = len(frames)
    signal = np.full(n_frames, np.nan, dtype=np.float64)

    for i in range(n_frames):
        rois = rois_per_frame[i]

        # No detection for this frame
        if rois is None or roi_name not in rois:
            continue

        cx, cy, r = rois[roi_name]
        frame = frames[i]

        # 3-channel → use first channel only
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        h, w = frame.shape

        # Clamp ROI box to frame boundaries
        y_min = max(0, cy - r)
        y_max = min(h, cy + r)
        x_min = max(0, cx - r)
        x_max = min(w, cx + r)

        # Skip if ROI is outside the frame
        if y_min >= y_max or x_min >= x_max:
            continue

        roi_patch = frame[y_min:y_max, x_min:x_max]
        signal[i] = float(np.nanmean(roi_patch))

    return signal


def extract_all_roi_signals(frames, rois_per_frame, roi_names):
    """
    Extract temperature signals for multiple ROIs at once.

    Args:
        frames:         np.ndarray (N, H, W) or (N, H, W, 3)
        rois_per_frame: list of dict or None, length N
        roi_names:      list of str, e.g. ["forehead", "nose"]

    Returns:
        dict: roi_name → np.ndarray (N,) float64
    """
    signals = {}
    for name in roi_names:
        signals[name] = extract_roi_signal(frames, rois_per_frame, name)
    return signals


def interpolate_nan(signal):
    """
    Fill NaN gaps in a signal using linear interpolation.

    Frames where YOLO found no keypoints produce NaN values.
    This function fills those gaps so that downstream filtering
    (which cannot handle NaN) works correctly.

    Args:
        signal: np.ndarray (N,), may contain NaN

    Returns:
        np.ndarray (N,), NaN-free (unless the entire signal is NaN)
    """
    nans = np.isnan(signal)

    # All NaN → nothing to interpolate
    if nans.all():
        return signal.copy()

    # No NaN → nothing to do
    if not nans.any():
        return signal.copy()

    x = np.arange(len(signal))
    signal_clean = signal.copy()
    signal_clean[nans] = np.interp(x[nans], x[~nans], signal[~nans])

    return signal_clean
