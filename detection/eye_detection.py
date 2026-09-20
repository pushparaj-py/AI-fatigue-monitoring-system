"""
Eye landmark extraction and region isolation module for RASD.
Defines standard MediaPipe Face Mesh landmark indices for both left and right eyes.
"""

# MediaPipe Face Mesh landmark indices for left and right eyes
# Left eye full contour
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Left eye 6 primary landmark points (for standard EAR calculation)
LEFT_EYE_KEY_POINTS = [362, 385, 387, 263, 373, 380]

# Right eye full contour
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Right eye 6 primary landmark points (for standard EAR calculation)
RIGHT_EYE_KEY_POINTS = [33, 160, 158, 133, 153, 144]


def extract_eye_landmarks(all_landmarks, image_shape=None):
    """
    Extracts and isolates left eye and right eye coordinates from full facial landmarks.

    Args:
        all_landmarks (list): List of (x, y, z) or landmark objects.
        image_shape (tuple, optional): (height, width) to convert normalized coords to pixels.

    Returns:
        dict: Isolated left_eye and right_eye landmark data structures.
    """
    if not all_landmarks:
        return {"left_eye": None, "right_eye": None}

    h, w = (image_shape[0], image_shape[1]) if image_shape else (1, 1)

    def get_points(indices):
        points = []
        for idx in indices:
            if idx < len(all_landmarks):
                lm = all_landmarks[idx]
                if hasattr(lm, "x"):
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append({"idx": idx, "x": lm.x, "y": lm.y, "z": lm.z, "px": px, "py": py})
                elif isinstance(lm, (tuple, list)):
                    px, py = int(lm[0] * w), int(lm[1] * h)
                    points.append({"idx": idx, "x": lm[0], "y": lm[1], "z": lm[2] if len(lm) > 2 else 0, "px": px, "py": py})
        return points

    return {
        "left_eye": {
            "contour": get_points(LEFT_EYE_CONTOUR),
            "key_points": get_points(LEFT_EYE_KEY_POINTS),
            "indices": LEFT_EYE_CONTOUR
        },
        "right_eye": {
            "contour": get_points(RIGHT_EYE_CONTOUR),
            "key_points": get_points(RIGHT_EYE_KEY_POINTS),
            "indices": RIGHT_EYE_CONTOUR
        }
    }
