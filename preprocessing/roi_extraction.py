"""
preprocessing/roi_extraction.py
================================
Computes 5 ROI boxes from YOLO keypoints (54-point model).
Used by ICAMethod and ThermalMeanMethod (Garbey works directly with the keypoints and defines its own line geometry, see methods/garbey.py).
"""

import numpy as np

from utils.yolo_keypoints import get_keypoint_index

ROI_CONFIG = {
    "forehead_inner_left":  get_keypoint_index("forehead_left_22"),       # Index 21
    "forehead_inner_right": get_keypoint_index("forehead_right_23"),      # Index 22
    "forehead_outer_left":  get_keypoint_index("forehead_left_18"),       # Index 17
    "forehead_outer_right": get_keypoint_index("forehead_right_27"),      # Index 26
    "left_eye_outer":       get_keypoint_index("left_eye_37"),            # Index 36
    "right_eye_outer":      get_keypoint_index("right_eye_46"),           # Index 45
    "left_mouth_corner":    get_keypoint_index("mouth_corner_left_49"),   # Index 48
    "right_mouth_corner":   get_keypoint_index("mouth_corner_right_51"),  # Index 50
    "left_nostril":         get_keypoint_index("nostril_left_33"),        # Index 32
    "right_nostril":        get_keypoint_index("nostril_right_35"),       # Index 34
    "upper_lip_top":        get_keypoint_index("upper_lip_50"),           # Index 49
}


def compute_rois(keypoints, dataset_name=None):
    """Computes 5 ROI boxes for a frame.

    Args:
        keypoints: (54, 2) Array – from YOLO-Modell
        
    Returns:
        dict: ROI-Name -> (center_x, center_y, radius)
    """
    idx = ROI_CONFIG
    kp = keypoints

    # Eye distance as scale reference
    eye_dist = np.linalg.norm(
        kp[idx["left_eye_outer"]] - kp[idx["right_eye_outer"]]
    )
    if eye_dist < 15:
        eye_dist = 50.0

    r_large = max(4, int(0.12 * eye_dist))
    r_small = max(3, int(0.08 * eye_dist))

    return {
        "forehead": (
            int((kp[idx["forehead_inner_left"], 0] + kp[idx["forehead_inner_right"], 0]) / 2),
            int((kp[idx["forehead_inner_left"], 1] + kp[idx["forehead_inner_right"], 1]) / 2 - 0.45 * eye_dist),
            r_large,
        ),
        "left_cheek": (
            int((kp[idx["left_eye_outer"], 0] + kp[idx["left_mouth_corner"], 0]) / 2),
            int((kp[idx["left_eye_outer"], 1] + kp[idx["left_mouth_corner"], 1]) / 2),
            r_large,
        ),
        "right_cheek": (
            int((kp[idx["right_eye_outer"], 0] + kp[idx["right_mouth_corner"], 0]) / 2),
            int((kp[idx["right_eye_outer"], 1] + kp[idx["right_mouth_corner"], 1]) / 2),
            r_large,
        ),
        "nose": (
            int((kp[idx["left_nostril"], 0] + kp[idx["right_nostril"], 0]) / 2),
            int((kp[idx["left_nostril"], 1] + kp[idx["right_nostril"], 1]) / 2),
            r_small,
        ),
        "philtrum": (
            int(kp[idx["upper_lip_top"], 0]),
            int(kp[idx["upper_lip_top"], 1]),
            r_small,
        ),
    }
