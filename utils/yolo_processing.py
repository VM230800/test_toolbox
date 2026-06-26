"""
utils/yolo_processing.py
========================
Universal YOLO-based face cropping and keypoint extraction.

Accepts frames in any format from the dataset loaders:
  - (N, H, W, 3) uint8    from BP4D+ video via OpenCV
  - (N, H, W, 3) float32  from BP4D+ with float conversion
  - (N, H, W)    float32  from NPZ thermal data

Provides two modes:
  - process_with_yolo()            → all frames in RAM (original)
  - process_with_yolo_streaming()  → one frame at a time (RAM-friendly)
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

def _ensure_uint8_3ch(frames):
    """Convert any loader output to (N, H, W, 3) uint8 for YOLO."""

    if frames.ndim == 4 and frames.dtype == np.uint8:
        return frames

    # Single channel float (e.g. NPZ thermal data)
    if frames.ndim == 3:
        f_min = frames.min(axis=(1, 2), keepdims=True)
        f_max = frames.max(axis=(1, 2), keepdims=True)
        rng = f_max - f_min
        rng[rng == 0] = 1.0
        normed = ((frames - f_min) / rng * 255).astype(np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    # 3-channel float
    if frames.ndim == 4 and frames.dtype == np.float32:
        if frames.max() <= 1.0:
            return (frames * 255).astype(np.uint8)
        return np.clip(frames, 0, 255).astype(np.uint8)

    raise ValueError(
        f"Unsupported frame format: shape={frames.shape}, "
        f"dtype={frames.dtype}"
    )


def _ensure_single_uint8_3ch(frame):
    """Convert a SINGLE frame to (H, W, 3) uint8 for YOLO."""

    # Already correct format
    if frame.ndim == 3 and frame.dtype == np.uint8:
        return frame

    # Single channel float (NPZ thermal: temperature in °C)
    if frame.ndim == 2:
        vmin = np.percentile(frame, 1)
        vmax = np.percentile(frame, 99)
        if vmax > vmin:
            norm = (frame - vmin) / (vmax - vmin) * 255
        else:
            norm = np.zeros_like(frame)
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return np.stack([norm, norm, norm], axis=-1)

    # 3-channel float
    if frame.ndim == 3 and frame.dtype == np.float32:
        if frame.max() <= 1.0:
            return (frame * 255).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)

    raise ValueError(
        f"Unsupported frame format: shape={frame.shape}, "
        f"dtype={frame.dtype}"
    )


def _detect_face_box(model, frame, padding=50, confidence=0.5):
    """
    Detect the largest face bounding box in a single frame.

    Args:
        model:      YOLO model instance
        frame:      np.ndarray (H, W, 3) uint8
        padding:    int, pixels to add around detection
        confidence: float, minimum confidence

    Returns:
        tuple (x1, y1, x2, y2), clamped to frame boundaries
    """
    h, w = frame.shape[:2]

    results = model(frame, verbose=False, conf=confidence)

    if len(results) == 0 or results[0].boxes is None:
        warnings.warn("No face detected, using full frame.")
        return 0, 0, w, h

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        warnings.warn("No face detected, using full frame.")
        return 0, 0, w, h

    # Take largest box
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = np.argmax(areas)
    x1, y1, x2, y2 = boxes[best].astype(int)

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    return x1, y1, x2, y2


def _extract_keypoints(model, frame, confidence=0.5):
    """
    Extract facial keypoints from a single cropped frame.

    Args:
        model:      YOLO model instance
        frame:      np.ndarray (H, W, 3) uint8
        confidence: float, minimum confidence

    Returns:
        np.ndarray (54, 2) float32, NaN where no keypoint detected
    """
    keypoints = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)

    results = model(frame, verbose=False, conf=confidence)

    if len(results) == 0 or results[0].keypoints is None:
        return keypoints

    kps = results[0].keypoints.xy
    if kps is None or len(kps) == 0:
        return keypoints

    kp_array = kps[0].cpu().numpy()

    n_detected = min(kp_array.shape[0], NUM_KEYPOINTS)
    keypoints[:n_detected] = kp_array[:n_detected]

    return keypoints


# ---------------------------------------------------------------------------
# Public API – Original (all frames in RAM)
# ---------------------------------------------------------------------------

def process_with_yolo(frames, model_path, target_size=(400, 400),
                      padding=50):
    """
    Crop a video to a stable face region and extract keypoints.

    Loads ALL frames into RAM. For large videos, use
    process_with_yolo_streaming() instead.

    Args:
        frames:      np.ndarray, raw frames from any loader
        model_path:  str, path to YOLO .pt model
        target_size: tuple (W, H), resize target after cropping
        padding:     int, pixels to add around face box

    Returns:
        cropped_frames: np.ndarray (N, H, W, 3) uint8
        keypoints:      np.ndarray (N, 54, 2) float32
    """
    # Auto-convert to YOLO-compatible format
    frames = _ensure_uint8_3ch(frames)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must have shape (N, H, W, 3), got {frames.shape}"
        )

    n_frames = frames.shape[0]
    model = YOLO(model_path)

    # Step 1: stable face box from first frame
    x1, y1, x2, y2 = _detect_face_box(model, frames[0], padding=padding)

    # Step 2: crop + resize all frames
    cropped_frames = np.empty(
        (n_frames, target_size[1], target_size[0], 3), dtype=np.uint8
    )
    for i in range(n_frames):
        crop = frames[i, y1:y2, x1:x2]
        cropped_frames[i] = cv2.resize(
            crop, target_size, interpolation=cv2.INTER_LINEAR
        )

    # Step 3: extract keypoints from each cropped frame
    keypoints = np.empty(
        (n_frames, NUM_KEYPOINTS, 2), dtype=np.float32
    )
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


# ---------------------------------------------------------------------------
# Public API – Streaming (RAM-friendly, one frame at a time)
# ---------------------------------------------------------------------------

def process_with_yolo_streaming(frame_iterator, model_path,
                                target_size=(400, 400), padding=50):
    """
    Process frames one-by-one through YOLO.

    Only ONE original frame is in RAM at a time.
    Only the small cropped frames are kept.

    The face bounding box is detected on the first frame
    and reused for all subsequent frames (stable crop).

    Args:
        frame_iterator: Generator/iterator that yields frames
                        one-by-one (from dataset.iter_frames())
        model_path:     str, path to YOLO .pt model
        target_size:    tuple (W, H), resize target after cropping
        padding:        int, pixels to add around face box

    Returns:
        cropped_frames: np.ndarray (N, H, W, 3) uint8
        keypoints:      np.ndarray (N, 54, 2) float32
    """
    model = YOLO(model_path)

    cropped_list = []
    keypoints_list = []
    face_box = None

    for i, frame_raw in enumerate(frame_iterator):

        # Convert single frame to uint8 3-channel
        frame = _ensure_single_uint8_3ch(frame_raw)

        # Detect face box on first frame only
        if face_box is None:
            face_box = _detect_face_box(
                model, frame, padding=padding)

        # Crop using stable box
        x1, y1, x2, y2 = face_box
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(
            crop, target_size, interpolation=cv2.INTER_LINEAR)

        # Extract keypoints from cropped frame
        kp = _extract_keypoints(model, crop_resized)

        cropped_list.append(crop_resized)
        keypoints_list.append(kp)

        # Original frame is NOT stored → RAM stays low
        del frame_raw, frame

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1} frames...")

    if not cropped_list:
        raise ValueError("No frames received from iterator")

    cropped_frames = np.array(cropped_list, dtype=np.uint8)
    all_keypoints = np.array(keypoints_list, dtype=np.float32)

    n_missing = sum(1 for k in keypoints_list
                    if np.isnan(k).all())

    if n_missing > 0:
        warnings.warn(
            f"YOLO found no keypoints in "
            f"{n_missing}/{len(cropped_list)} frames.")

    print(f"    Streaming complete: {len(cropped_list)} frames, "
          f"{len(cropped_list) - n_missing} with keypoints")

    return cropped_frames, all_keypoints
