"""
feature_extraction.py

Combines everything built so far into a single per-frame feature extractor:
- MediaPipe Face Landmarker (Tasks API) for face landmark detection
- EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) geometric calculation
- Trained CNN eye-state model inference (models/eye_model/eye_model.keras)
- Trained CNN yawn model inference (models/yawn_model/yawn_model.keras)

Note: newer MediaPipe versions (>=0.10.10-ish, and all 1.x releases) removed
the older `mp.solutions.face_mesh` API. This module uses the current
"Tasks API" (mp.tasks.vision.FaceLandmarker) instead, which is what
MediaPipe now expects for all new code. The underlying landmark model and
indexing (468/478 points) is the same, so the EAR/MAR index constants below
are unchanged from the original solutions-based version.

For each frame, returns a fixed-length feature vector:
    [left_EAR, right_EAR, avg_EAR, MAR, eye_cnn_score, yawn_cnn_score]
"""

import os
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import tensorflow as tf

# ---------------------------------------------------------------------------
# Load trained CNN models once, at import time (not per-frame — too slow)
# ---------------------------------------------------------------------------
EYE_MODEL_PATH = os.path.join("models", "eye_model", "eye_model.keras")
YAWN_MODEL_PATH = os.path.join("models", "yawn_model", "yawn_model.keras")

eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)

CNN_IMG_SIZE = 64  # must match the size used during training for both models

# ---------------------------------------------------------------------------
# MediaPipe Face Landmarker setup (Tasks API)
# ---------------------------------------------------------------------------
FACE_LANDMARKER_MODEL_PATH = os.path.join("models", "face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_model():
    """Downloads the Face Landmarker model file once, if not already present."""
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        os.makedirs(os.path.dirname(FACE_LANDMARKER_MODEL_PATH), exist_ok=True)
        print(f"Downloading Face Landmarker model to {FACE_LANDMARKER_MODEL_PATH} ...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
        print("Download complete.")


_ensure_face_landmarker_model()

_base_options = mp_python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
_landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=_base_options,
    num_faces=1,
    running_mode=mp_vision.RunningMode.IMAGE,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(_landmarker_options)

# Standard 6-point EAR landmark sets (same indices as the legacy solutions API —
# the underlying 468/478-point face mesh model and indexing is unchanged)
LEFT_EYE_KEY_POINTS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_KEY_POINTS = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for MAR
MOUTH_KEY_POINTS = [61, 291, 39, 181, 0, 17, 269, 405]

# Bounding-box landmarks used to crop the eye/mouth regions for CNN input
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def _landmark_to_px(landmark, frame_width, frame_height):
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))


def _calculate_ear(landmarks, key_points, frame_width, frame_height):
    pts = [_landmark_to_px(landmarks[i], frame_width, frame_height) for i in key_points]
    vertical_1 = _euclidean(pts[1], pts[5])
    vertical_2 = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _calculate_mar(landmarks, key_points, frame_width, frame_height):
    pts = [_landmark_to_px(landmarks[i], frame_width, frame_height) for i in key_points]
    vertical_1 = _euclidean(pts[1], pts[5])
    vertical_2 = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _crop_region(frame, landmarks, contour_indices, frame_width, frame_height, padding=5):
    pts = [_landmark_to_px(landmarks[i], frame_width, frame_height) for i in contour_indices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = max(min(xs) - padding, 0), min(max(xs) + padding, frame_width)
    y_min, y_max = max(min(ys) - padding, 0), min(max(ys) + padding, frame_height)

    if x_max <= x_min or y_max <= y_min:
        return None

    return frame[y_min:y_max, x_min:x_max]


def _prepare_for_cnn(cropped_region):
    if cropped_region is None or cropped_region.size == 0:
        return None
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY) if len(cropped_region.shape) == 3 else cropped_region
    resized = cv2.resize(gray, (CNN_IMG_SIZE, CNN_IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)


def extract_features(frame):
    """
    Runs the full per-frame feature extraction pipeline using MediaPipe's
    current Face Landmarker (Tasks API).

    Args:
        frame: a single BGR frame (as read from OpenCV/webcam or video file)

    Returns:
        A 6-element feature vector as a numpy array:
            [left_EAR, right_EAR, avg_EAR, MAR, eye_cnn_score, yawn_cnn_score]
        Returns None if no face was detected in the frame.
    """
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]  # list of NormalizedLandmark (x, y, z), same indexing as before

    left_ear = _calculate_ear(landmarks, LEFT_EYE_KEY_POINTS, frame_width, frame_height)
    right_ear = _calculate_ear(landmarks, RIGHT_EYE_KEY_POINTS, frame_width, frame_height)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = _calculate_mar(landmarks, MOUTH_KEY_POINTS, frame_width, frame_height)

    left_eye_crop = _crop_region(frame, landmarks, LEFT_EYE_CONTOUR, frame_width, frame_height)
    right_eye_crop = _crop_region(frame, landmarks, RIGHT_EYE_CONTOUR, frame_width, frame_height)
    # Note: no mouth crop needed here — the yawn CNN was trained on full frames
    # (see Phase I / extract_yawn_frames.py), so we feed it the full frame below
    # instead of a mouth-region crop, to match its training input format.

    eye_scores = []
    for crop in (left_eye_crop, right_eye_crop):
        prepared = _prepare_for_cnn(crop)
        if prepared is not None:
            score = eye_model.predict(prepared, verbose=0)[0][0]
            eye_scores.append(score)
    eye_cnn_score = float(np.mean(eye_scores)) if eye_scores else 0.5

    mouth_prepared = _prepare_for_cnn(frame)  # yawn CNN was trained on FULL frames
    # (Phase I saved whole video frames into Yawning/NotYawning folders, not
    # mouth-only crops — so inference must match that same full-frame format.)
    if mouth_prepared is not None:
        yawn_cnn_score = float(yawn_model.predict(mouth_prepared, verbose=0)[0][0])
    else:
        yawn_cnn_score = 0.0

    return np.array([left_ear, right_ear, avg_ear, mar, eye_cnn_score, yawn_cnn_score], dtype="float32")