""" 
utils/yolo_keypoints.py
=======================
Facial keypoint and region definitions for YOLO-based landmark detection.

Contains:
- Mapping from keypoint indices to anatomical labels
- Region groupings for higher-level facial analysis
- Utility functions for name/index/region conversion 
"""

yolo_keypoint_names = {
    # --- Left facial contour (outer edge, top to bottom) ---
    0:  "contour_left_1",           # Image No. 1 – upper temple
    1:  "contour_left_2",           # Image No. 2 – lower temple
    2:  "contour_left_3",           # Image No. 3 – upper cheek
    3:  "contour_left_4",           # Image No. 4 – mid cheek
    4:  "contour_left_5",           # Image No. 5 – lower cheek
    5:  "contour_left_6",           # Image No. 6 – upper jaw
    6:  "contour_left_7",           # Image No. 7 – lower jaw
    7:  "contour_left_8",           # Image No. 8 – left chin

    # --- Chin center ---
    8:  "chin_9",                   # Image No. 9 – chin center

    # --- Right facial contour (outer edge, bottom to top) ---
    9:  "contour_right_10",        # Image No. 10 – right chin
    10: "contour_right_11",        # Image No. 11 – lower jaw
    11: "contour_right_12",        # Image No. 12 – upper jaw
    12: "contour_right_13",        # Image No. 13 – lower cheek
    13: "contour_right_14",        # Image No. 14 – mid cheek
    14: "contour_right_15",        # Image No. 15 – upper cheek
    15: "contour_right_16",        # Image No. 16 – lower temple
    16: "contour_right_17",        # Image No. 17 – upper temple

    # --- Forehead / hairline (upper row, left to right) ---
    17: "forehead_left_18",        # Image No. 18 – far left forehead
    18: "forehead_left_19",        # Image No. 19
    19: "forehead_left_20",        # Image No. 20
    20: "forehead_left_21",        # Image No. 21
    21: "forehead_left_22",        # Image No. 22 – forehead center
    22: "forehead_right_23",       # Image No. 23 – forehead center
    23: "forehead_right_24",       # Image No. 24
    24: "forehead_right_25",       # Image No. 25
    25: "forehead_right_26",       # Image No. 26
    26: "forehead_right_27",       # Image No. 27 – far right forehead

    # --- Nose ---
    27: "nasal_root_28",           # Image No. 28 – nasal root
    28: "nasal_bridge_29",         # Image No. 29 – upper nasal bridge
    29: "nasal_bridge_30",         # Image No. 30 – lower nasal bridge
    30: "nose_tip_31",             # Image No. 31 – nose tip
    
    31: "nostril_left_32",         # Image No. 32 – left nostril upper
    32: "nostril_left_33",         # Image No. 33 – left nostril middle
    33: "apex_34",                 # Image No. 34 – apex
    34: "nostril_right_35",        # Image No. 35 – right nostril middle
    35: "nostril_right_36",        # Image No. 36 – right nostril upper
    
    # --- Left eye region (from outer corner clockwise) ---
    36: "left_eye_37",             # Image No. 37 – outer eye corner
    37: "left_eye_38",             
    38: "left_eye_39",             
    39: "left_eye_40",             # Image No. 40 – inner eye corner
    40: "left_eye_41",             
    41: "left_eye_42",             

    # --- Right eye region (from inner corner clockwise) ---
    42: "right_eye_43",            # Image No. 43 – inner eye corner
    43: "right_eye_44",            
    44: "right_eye_45",            
    45: "right_eye_46",            # Image No. 46 – outer eye corner
    46: "right_eye_47",            
    47: "right_eye_48",            

    # --- Mouth ---
    48: "mouth_corner_left_49",    # Image No. 49 – left mouth corner
    49: "upper_lip_50",            # Image No. 50 – upper lip top
    50: "mouth_corner_right_51",   # Image No. 51 – right mouth corner
    51: "lower_lip_52",            # Image No. 52 – lower lip bottom
    52: "upper_lip_53",            # Image No. 53 – upper lip bottom
    53: "lower_lip_54",            # Image No. 54 – lower lip top
}

# Region mapping
yolo_regions = {
    # Temporal vessels
    "temple_left":  [0, 1, 2],
    "temple_right": [14, 15, 16],

    # Forehead
    "forehead": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],

    # Cheeks
    "cheek_left":  [3, 4, 5],
    "cheek_right": [11, 12, 13],

    # Nose region
    "nostrils": [32, 34],
    "nose_tip": [30],
    "nose": [28, 29, 30, 3, 32, 33, 34, 35],

    # Philtrum
    "philtrum": [33, 49],

    # Full face
    "face_full": list(range(54)),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_keypoint_index(name: str) -> int:
    """
    Returns the 0-based index of a keypoint given its name.
    """
    for idx, label in yolo_keypoint_names.items():
        if label == name:
            return idx
    raise ValueError(
        f"Keypoint '{name}' not found. "
        f"Available names: {list(yolo_keypoint_names.values())}"
    )


def get_keypoints_by_region(region: str) -> list[int]:
    """
    Returns all 0-based keypoint indices for an anatomical region.
    """
    if region not in yolo_regions:
        raise ValueError(
            f"Region '{region}' not found. "
            f"Available regions: {list(yolo_regions.keys())}"
        )
    return yolo_regions[region]


def get_keypoint_name(index: int) -> str:
    """
    Returns the anatomical name of a keypoint given its index.
    """
    if index not in yolo_keypoint_names:
        raise ValueError(f"Index {index} is out of valid range (0–53).")
    return yolo_keypoint_names[index]
