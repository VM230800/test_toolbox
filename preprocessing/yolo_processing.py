"""
utils/yolo_processing.py
========================
Universal YOLO-based face cropping and keypoint extraction.

This module is intentionally format-agnostic: it operates on numpy
arrays of frames, not on file paths. Frames can be passed in any
format produced by the dataset loaders:

  - (N, H, W, 3) uint8    –  e.g. from BP4D+ video via OpenCV
  - (N, H, W, 3) float32  –  e.g. from BP4D+ with float conversion
  - (N, H, W)    float32  –  e.g. from NPZ thermal data (°C values)

The conversion to the format YOLO expects (uint8, 3 channels) happens
automatically inside process_with_yolo().

...rest of existing module docstring...
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
from ultralytics import YOLO


NUM_KEYPOINTS = 54


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_uint8_3ch(frames: np.ndarray) -> np.ndarray:
    """Convert any loader output to (N, H, W, 3) uint8 for YOLO."""

    if frames.ndim == 4 and frames.dtype == np.uint8:
        return frames

    if frames.ndim == 3:
        f_min = frames.min(axis=(1, 2), keepdims=True)
        f_max = frames.max(axis=(1, 2), keepdims=True)
        rng = f_max - f_min
        rng[rng == 0] = 1.0
        normed = ((frames - f_min) / rng * 255).astype(np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    if frames.ndim == 4 and frames.dtype == np.float32:
        if frames.max() <= 1.0:
            return (frames * 255).astype(np.uint8)
        return np.clip(frames, 0, 255).astype(np.uint8)

    raise ValueError(
        f"Unsupported frame format: shape={frames.shape}, "
        f"dtype={frames.dtype}"
    )


def _white_hot(frame: np.ndarray) -> np.ndarray:
    """...existing code unchanged..."""


def _detect_face_box(...):
    """...existing code unchanged..."""


def _extract_keypoints(...):
    """...existing code unchanged..."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_with_yolo(
    frames: np.ndarray,
    model_path: str,
    target_size: tuple[int, int] = (400, 400),
    padding: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a video to a stable face region and extract keypoints.

    Accepts frames in ANY format from the dataset loaders.
    Conversion to uint8 3-channel happens automatically.

    ...rest of existing docstring...
    """
    # ---- Auto-convert to YOLO-compatible format ----------------------
    frames = _ensure_uint8_3ch(frames)

    # ---- Everything below is UNCHANGED --------------------------------
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (N, H, W, 3), got {frames.shape}"
        )

    n_frames = frames.shape[0]
    model = YOLO(model_path)

    # Step 1: stable face box
    xmin, ymin, xmax, ymax = _detect_face_box(model, frames[0], padding=padding)

    # Step 2: crop + resize
    cropped_frames = np.empty(
        (n_frames, target_size[1], target_size[0], 3), dtype=np.uint8
    )
    for i in range(n_frames):
        crop = frames[i, ymin:ymax, xmin:xmax]
        cropped_frames[i] = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    # Step 3: keypoints
    keypoints = np.empty((n_frames, NUM_KEYPOINTS, 2), dtype=np.float32)
    n_missing = 0
    for i in range(n_frames):
        kp = _extract_keypoints(model, cropped_frames[i])
        keypoints[i] = kp
        if np.isnan(kp).all():
            n_missing += 1

    if n_missing > 0:
        warnings.warn(
            f"YOLO found no keypoints in {n_missing}/{n_frames} frames."
        )

    return cropped_frames, keypoints
