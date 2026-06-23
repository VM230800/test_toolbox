"""
Berechnet 5 ROI-Boxen aus YOLO-Keypoints (54 Punkte).
Wird von ALLEN Algorithmen genutzt.
"""

import numpy as np

# Indices basierend auf yolo_keypoints.py (54-Punkt-Modell)
ROI_CONFIGS = {
    "bp4d": {
        "left_brow_inner":     21,   # Stirnmitte links
        "right_brow_inner":    22,   # Stirnmitte rechts
        "left_eye_outer":      36,   # Äußerer Augenwinkel links
        "right_eye_outer":     45,   # Äußerer Augenwinkel rechts
        "left_mouth_corner":   48,   # Mundwinkel links
        "right_mouth_corner":  50,   # Mundwinkel rechts
        "left_nose_wing":      32,   # Nasenflügel links
        "right_nose_wing":     34,   # Nasenflügel rechts
        "upper_lip_center":    49,   # Oberlippe oben
        "left_brow_outer":     17,   # Stirn äußerst links
        "right_brow_outer":    26,   # Stirn äußerst rechts
    },
    "betreuer": {
        "left_brow_inner":     21,
        "right_brow_inner":    22,
        "left_eye_outer":      36,
        "right_eye_outer":     45,
        "left_mouth_corner":   48,
        "right_mouth_corner":  50,
        "left_nose_wing":      32,
        "right_nose_wing":     34,
        "upper_lip_center":    49,
        "left_brow_outer":     17,
        "right_brow_outer":    26,
    },
}


def compute_rois(keypoints, dataset_name="bp4d"):
    """Berechnet 5 ROI-Boxen für einen Frame.
    
    Args:
        keypoints: (54, 2) Array – vom YOLO-Modell
        dataset_name: "bp4d" oder "betreuer"
    
    Returns:
        dict: ROI-Name → (center_x, center_y, radius)
    """
    idx = ROI_CONFIGS[dataset_name]
    kp = keypoints

    # Augenabstand als Skalierung
    eye_dist = np.linalg.norm(
        kp[idx["left_eye_outer"]] - kp[idx["right_eye_outer"]]
    )
    if eye_dist < 15:
        eye_dist = 50.0

    r_large = max(4, int(0.12 * eye_dist))
    r_small = max(3, int(0.08 * eye_dist))

    return {
        "forehead": (
            int((kp[idx["left_brow_inner"], 0] + kp[idx["right_brow_inner"], 0]) / 2),
            int((kp[idx["left_brow_inner"], 1] + kp[idx["right_brow_inner"], 1]) / 2 - 0.45 * eye_dist),
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
            int((kp[idx["left_nose_wing"], 0] + kp[idx["right_nose_wing"], 0]) / 2),
            int((kp[idx["left_nose_wing"], 1] + kp[idx["right_nose_wing"], 1]) / 2),
            r_small,
        ),
        "philtrum": (
            int(kp[idx["upper_lip_center"], 0]),
            int(kp[idx["upper_lip_center"], 1]),
            r_small,
        ),
    }
