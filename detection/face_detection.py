"""
face_detection.py

Integrates MediaPipe Face Mesh with the webcam feed.
- Detects facial landmarks on each frame and draws them (mesh overlay).
- Isolates and returns landmark coordinates for the left eye, right eye,
  and mouth regions, for use by eye-state and yawn detection in later phases.

No EAR/MAR calculation or classification happens here — this phase only
gets detection + region isolation working and visible.
"""

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# MediaPipe Face Mesh landmark index groups for each region.
# These indices are standard for the 468-point Face Mesh model.
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
MOUTH_LANDMARKS = [61, 291, 39, 181, 0, 17, 269, 405]

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def _extract_region_points(landmarks, indices, frame_width, frame_height):
    """Convert normalized MediaPipe landmarks to pixel coordinates for given indices."""
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x_px = int(lm.x * frame_width)
        y_px = int(lm.y * frame_height)
        points.append((x_px, y_px))
    return points


def process_frame(frame):
    """
    Runs Face Mesh detection on a single BGR frame.

    Returns:
        annotated_frame: frame with mesh/landmarks drawn on it
        regions: dict with 'left_eye', 'right_eye', 'mouth' pixel coordinate
                 lists, or None if no face was detected
    """
    frame_height, frame_width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    annotated_frame = frame.copy()
    regions = None

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Draw the full mesh so detection is visually confirmable.
        mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

        landmarks = face_landmarks.landmark

        left_eye = _extract_region_points(landmarks, LEFT_EYE_LANDMARKS, frame_width, frame_height)
        right_eye = _extract_region_points(landmarks, RIGHT_EYE_LANDMARKS, frame_width, frame_height)
        mouth = _extract_region_points(landmarks, MOUTH_LANDMARKS, frame_width, frame_height)

        # Highlight the isolated regions distinctly so they're visually confirmable too.
        for point in left_eye:
            cv2.circle(annotated_frame, point, 2, (0, 255, 0), -1)   # green = left eye
        for point in right_eye:
            cv2.circle(annotated_frame, point, 2, (255, 0, 0), -1)   # blue = right eye
        for point in mouth:
            cv2.circle(annotated_frame, point, 2, (0, 0, 255), -1)   # red = mouth

        regions = {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "mouth": mouth,
        }

    return annotated_frame, regions
