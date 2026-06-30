"""
utils/yolo_processing.py
========================
Universal YOLO-based face cropping and keypoint extraction.

Accepts frames in any format from the dataset loaders:
  - (N, H, W, 3) uint8    from BP4D+ video via OpenCV
  - (N, H, W, 3) float16  from BP4D+ with float16 conversion
  - (N, H, W, 3) float32  from BP4D+ with float32 conversion
  - (N, H, W)    float32  from NPZ thermal data
  - (N, H, W)    float16  from NPZ thermal data

Provides two modes:
  - process_with_yolo()            → all frames in RAM
  - process_with_yolo_streaming()  → one frame at a time
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
from ultralytics import YOLO


NUM_KEYPOINTS = 54


# -----------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------

def _ensure_uint8_3ch(frames):
    """
    Convert any batch of frames to (N, H, W, 3) uint8.
    """
    arr = np.array(frames)

    # Already uint8 3-channel → done
    if (arr.ndim == 4 and arr.dtype == np.uint8
            and arr.shape[3] == 3):
        return arr

    # Any float → float32 for safe math
    if arr.dtype in (np.float16, np.float32, np.float64):
        arr = arr.astype(np.float32)

    # (N, H, W, 3) float in 0-255 range → just clip
    # (BP4D+ frames loaded by cv2 then cast to float)
    if arr.ndim == 4 and arr.shape[3] == 3:
        if arr.max() > 1.0 and arr.max() <= 255.5:
            return np.clip(arr, 0, 255).astype(np.uint8)
        # Thermal float → normalise channel 0
        gray = arr[:, :, :, 0]
        f_min = gray.min(axis=(1, 2), keepdims=True)
        f_max = gray.max(axis=(1, 2), keepdims=True)
        rng = f_max - f_min
        rng[rng == 0] = 1.0
        normed = ((gray - f_min) / rng * 255).astype(
            np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    # (N, H, W) grayscale → normalise → 3ch
    if arr.ndim == 3:
        f_min = arr.min(axis=(1, 2), keepdims=True)
        f_max = arr.max(axis=(1, 2), keepdims=True)
        rng = f_max - f_min
        rng[rng == 0] = 1.0
        normed = ((arr - f_min) / rng * 255).astype(
            np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    # (N, H, W, 1) → squeeze
    if arr.ndim == 4 and arr.shape[3] == 1:
        return _ensure_uint8_3ch(arr[:, :, :, 0])

    raise ValueError(
        f"Unsupported: shape={arr.shape}, "
        f"dtype={arr.dtype}")


def _ensure_single_uint8_3ch(frame):
    """
    Convert a SINGLE frame to (H, W, 3) uint8.
    Used by streaming mode.
    """
    arr = np.array(frame)

    # Already uint8 3-channel
    if (arr.ndim == 3 and arr.dtype == np.uint8
            and arr.shape[2] == 3):
        return arr

    # Any float → float32
    if arr.dtype in (np.float16, np.float32, np.float64):
        arr = arr.astype(np.float32)

    # (H, W, 3) float in 0-255 → just clip
    if arr.ndim == 3 and arr.shape[2] == 3:
        if arr.max() > 1.0 and arr.max() <= 255.5:
            return np.clip(arr, 0, 255).astype(np.uint8)
        gray = arr[:, :, 0]
        mn, mx = gray.min(), gray.max()
        rng = mx - mn if mx > mn else 1.0
        normed = ((gray - mn) / rng * 255).astype(
            np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    # (H, W) grayscale
    if arr.ndim == 2:
        mn, mx = arr.min(), arr.max()
        rng = mx - mn if mx > mn else 1.0
        normed = ((arr - mn) / rng * 255).astype(
            np.uint8)
        return np.stack([normed, normed, normed], axis=-1)

    # (H, W, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return _ensure_single_uint8_3ch(arr[:, :, 0])

    raise ValueError(
        f"Unsupported: shape={arr.shape}, "
        f"dtype={arr.dtype}")


def _detect_face_box(model, frame, padding=50,
                     confidence=0.5):
    """
    Detect the largest face bounding box in a frame.

    Returns:
        tuple (x1, y1, x2, y2)
    """
    h, w = frame.shape[:2]

    results = model(frame, verbose=False, conf=confidence)

    if len(results) == 0 or results[0].boxes is None:
        warnings.warn(
            "No face detected, using full frame.")
        return 0, 0, w, h

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        warnings.warn(
            "No face detected, using full frame.")
        return 0, 0, w, h

    # Take largest box
    areas = ((boxes[:, 2] - boxes[:, 0])
             * (boxes[:, 3] - boxes[:, 1]))
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

    Returns:
        np.ndarray (54, 2) float32, NaN where missing
    """
    keypoints = np.full(
        (NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)

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


# -----------------------------------------------------------
# Public API – All frames in RAM
# -----------------------------------------------------------

def process_with_yolo(frames, model_path,
                      target_size=(400, 400),
                      padding=50, confidence=0.5):
    """
    Crop video to stable face region + extract keypoints.

    Args:
        frames:      np.ndarray from any loader
        model_path:  path to YOLO .pt model
        target_size: (W, H) resize target
        padding:     pixels around face box
        confidence:  YOLO detection confidence

    Returns:
        cropped_frames: (N, H, W, 3) uint8
        keypoints:      (N, 54, 2) float32
    """
    frames = _ensure_uint8_3ch(frames)

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"frames must be (N,H,W,3), "
            f"got {frames.shape}")

    n_frames = frames.shape[0]
    model = YOLO(model_path)

    # Step 1: stable face box from first frame
    x1, y1, x2, y2 = _detect_face_box(
        model, frames[0],
        padding=padding,
        confidence=confidence,
    )

    # Step 2: crop + resize all frames
    cropped_frames = np.empty(
        (n_frames, target_size[1], target_size[0], 3),
        dtype=np.uint8)

    for i in range(n_frames):
        crop = frames[i, y1:y2, x1:x2]
        cropped_frames[i] = cv2.resize(
            crop, target_size,
            interpolation=cv2.INTER_LINEAR)

    # Step 3: extract keypoints
    keypoints = np.empty(
        (n_frames, NUM_KEYPOINTS, 2),
        dtype=np.float32)

    n_missing = 0
    for i in range(n_frames):
        kp = _extract_keypoints(
            model, cropped_frames[i],
            confidence=confidence,
        )
        keypoints[i] = kp
        if np.isnan(kp).all():
            n_missing += 1

    if n_missing > 0:
        warnings.warn(
            f"YOLO found no keypoints in "
            f"{n_missing}/{n_frames} frames.")

    return cropped_frames, keypoints


# -----------------------------------------------------------
# Public API – Streaming (RAM-friendly)
# -----------------------------------------------------------

def process_with_yolo_streaming(frame_iterator, model_path,
                                target_size=(400, 400),
                                padding=50, confidence=0.5):
    """
    Process frames one-by-one through YOLO.

    Only ONE original frame in RAM at a time.
    Face box detected on first frame, reused for all.

    Args:
        frame_iterator: yields frames one-by-one
        model_path:     path to YOLO .pt model
        target_size:    (W, H) resize target
        padding:        pixels around face box
        confidence:     YOLO detection confidence

    Returns:
        cropped_frames: (N, H, W, 3) uint8
        keypoints:      (N, 54, 2) float32
    """
    model = YOLO(model_path)

    cropped_list = []
    keypoints_list = []
    face_box = None

    for i, frame_raw in enumerate(frame_iterator):

        frame = _ensure_single_uint8_3ch(frame_raw)

        if face_box is None:
            face_box = _detect_face_box(
                model, frame,
                padding=padding,
                confidence=confidence,
            )

        x1, y1, x2, y2 = face_box
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(
            crop, target_size,
            interpolation=cv2.INTER_LINEAR)

        kp = _extract_keypoints(
            model, crop_resized,
            confidence=confidence,
        )

        cropped_list.append(crop_resized)
        keypoints_list.append(kp)

        del frame_raw, frame

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1} frames...")

    if not cropped_list:
        raise ValueError(
            "No frames received from iterator")

    cropped_frames = np.array(
        cropped_list, dtype=np.uint8)
    all_keypoints = np.array(
        keypoints_list, dtype=np.float32)

    n_missing = sum(
        1 for k in keypoints_list
        if np.isnan(k).all())

    if n_missing > 0:
        warnings.warn(
            f"YOLO found no keypoints in "
            f"{n_missing}/{len(cropped_list)} frames.")

    print(f"    Streaming complete: "
          f"{len(cropped_list)} frames, "
          f"{len(cropped_list) - n_missing} "
          f"with keypoints")

    return cropped_frames, all_keypoints
