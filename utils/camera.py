"""
Camera capture and video streaming manager for RASD.
Handles OpenCV VideoCapture lifecycle, session-gated real-time drowsiness inference
via DrowsinessEngine, visual HUD overlay rendering, and thread-safe status reporting.
"""

import atexit
import threading
import time
import cv2
import numpy as np
from detection.drowsiness_engine import DrowsinessEngine
from detection.event_logger import get_event_logger

# ---------------------------------------------------------------------------
# Global Drowsiness Engine & Shared Detection State (Thread-Safe)
# ---------------------------------------------------------------------------
_engine = None
_engine_lock = threading.Lock()

_latest_status = {
    "session_active": False,
    "session_id": None,
    "face_detected": False,
    "eye_status": "N/A",
    "yawn_status": "No",
    "alertness_level": "STANDBY",
    "drowsiness_level": "STANDBY",
    "alertness_percent": 100,
    "lstm_drowsy_score": 0.0,
    "recent_yawn_count": 0,
    "last_updated": 0,
}
_status_lock = threading.Lock()


def get_drowsiness_engine():
    """Access or lazily initialize the singleton DrowsinessEngine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = DrowsinessEngine()
        return _engine


def get_latest_status():
    """Retrieve a copy of the latest detection status dictionary."""
    with _status_lock:
        status = dict(_latest_status)
    
    # Dynamically sync real-time session state from event_logger
    logger = get_event_logger()
    is_active = logger.is_session_active()
    session_id = logger.get_active_session_id()
    
    status["session_active"] = is_active
    status["session_id"] = session_id
    if not is_active:
        status["alertness_level"] = "STANDBY"
        status["drowsiness_level"] = "STANDBY"
    return status


def update_latest_status(status_dict):
    """Update the latest detection status dictionary thread-safely."""
    with _status_lock:
        _latest_status.update(status_dict)
        _latest_status["last_updated"] = time.time()


# ---------------------------------------------------------------------------
# HUD Overlay Renderer
# ---------------------------------------------------------------------------
def draw_detection_overlay(frame, status):
    """
    Overlays detection telemetry on the OpenCV video frame:
      - Eye Status (Open / Closed)
      - Yawning (Yes / No)
      - Alertness Level (ALERT / EARLY_DROWSINESS / DROWSY / CRITICAL / STANDBY)
    Using distinct color coding (Green / Yellow / Orange / Red / Gray).
    """
    # Colors in BGR format
    COLOR_GREEN = (50, 205, 50)      # Safe / Alert
    COLOR_YELLOW = (0, 220, 255)     # Early Drowsiness / Caution
    COLOR_ORANGE = (0, 140, 255)     # Drowsy / Warning
    COLOR_RED = (40, 40, 240)        # Critical / Emergency
    COLOR_GRAY = (180, 180, 180)     # No Face / Inactive
    COLOR_CYAN = (255, 200, 0)       # Standby info

    level = status.get("alertness_level", "NO_FACE")
    eye = status.get("eye_status", "N/A")
    yawn = status.get("yawn_status", "No")
    face_detected = status.get("face_detected", False)
    session_active = status.get("session_active", False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # 1. Standby overlay when session is not active
    if not session_active or level == "STANDBY":
        overlay = frame.copy()
        box_x1, box_y1 = 12, 12
        box_x2, box_y2 = 380, 95
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (70, 70, 75), 1, cv2.LINE_AA)

        cv2.putText(frame, "STATUS: STANDBY (Paused)", (box_x1 + 14, box_y1 + 32),
                    font, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Click 'Start Detection' to begin session", (box_x1 + 14, box_y1 + 65),
                    font, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        return

    # 2. Active Session HUD
    level_color_map = {
        "ALERT": COLOR_GREEN,
        "EARLY_DROWSINESS": COLOR_YELLOW,
        "DROWSY": COLOR_ORANGE,
        "CRITICAL": COLOR_RED,
        "NO_FACE": COLOR_GRAY,
        "STANDBY": COLOR_CYAN,
    }
    level_color = level_color_map.get(level, COLOR_GRAY)

    if face_detected:
        eye_color = COLOR_GREEN if eye == "Open" else COLOR_RED
        yawn_color = COLOR_ORANGE if yawn == "Yes" else COLOR_GREEN
    else:
        eye_color = COLOR_GRAY
        yawn_color = COLOR_GRAY

    # Render a semi-transparent HUD backdrop card
    overlay = frame.copy()
    box_x1, box_y1 = 12, 12
    box_x2, box_y2 = 360, 130
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 65), 1, cv2.LINE_AA)

    font_scale = 0.62

    # Line 1: Eye Status
    cv2.putText(frame, "Eye Status:", (box_x1 + 12, box_y1 + 30),
                font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f" {eye}", (box_x1 + 130, box_y1 + 30),
                font, font_scale, eye_color, thickness, cv2.LINE_AA)

    # Line 2: Yawning (Yes / No)
    cv2.putText(frame, "Yawning:", (box_x1 + 12, box_y1 + 65),
                font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f" {yawn}", (box_x1 + 130, box_y1 + 65),
                font, font_scale, yawn_color, thickness, cv2.LINE_AA)

    # Line 3: Alertness Level
    cv2.putText(frame, "Alertness Level:", (box_x1 + 12, box_y1 + 100),
                font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f" {level}", (box_x1 + 165, box_y1 + 100),
                font, font_scale, level_color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# VideoCamera Class
# ---------------------------------------------------------------------------
class VideoCamera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoCamera, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, camera_index=0):
        if self._initialized:
            return

        self.camera_index = camera_index
        self.cap = None
        self.active_clients = 0
        self.frame_lock = threading.Lock()
        self._initialized = True
        
        # Pre-initialize the DrowsinessEngine once during camera startup
        try:
            get_drowsiness_engine()
        except Exception as e:
            print(f"[WARN] DrowsinessEngine warm-up warning: {e}")

        # Register cleanup hook at exit
        atexit.register(self.release)

    def _open_camera(self):
        """Open VideoCapture with Windows DirectShow support or standard backend."""
        if self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return True
        return False

    def get_frame(self, process_drowsiness=True):
        """
        Capture a frame from the camera.
        Only runs DrowsinessEngine inference if an active detection session is running.
        """
        with self.frame_lock:
            logger = get_event_logger()
            is_session_active = logger.is_session_active()
            active_session_id = logger.get_active_session_id()

            if not self._open_camera():
                update_latest_status({
                    "session_active": is_session_active,
                    "session_id": active_session_id,
                    "face_detected": False,
                    "eye_status": "N/A",
                    "yawn_status": "No",
                    "alertness_level": "NO_FACE" if is_session_active else "STANDBY",
                    "drowsiness_level": "NO_FACE" if is_session_active else "STANDBY",
                    "alertness_percent": 0 if is_session_active else 100,
                })
                return self._create_placeholder_frame("Camera not accessible")

            success, frame = self.cap.read()
            if not success or frame is None:
                return self._create_placeholder_frame("Waiting for video feed...")

            if not is_session_active:
                # -------------------------------------------------------------
                # Standby Mode: Session is NOT active — do not run neural models
                # -------------------------------------------------------------
                standby_payload = {
                    "session_active": False,
                    "session_id": None,
                    "face_detected": False,
                    "eye_status": "N/A",
                    "yawn_status": "No",
                    "alertness_level": "STANDBY",
                    "drowsiness_level": "STANDBY",
                    "alertness_percent": 100,
                    "lstm_drowsy_score": 0.0,
                    "recent_yawn_count": 0,
                }
                update_latest_status(standby_payload)
                draw_detection_overlay(frame, standby_payload)

            elif process_drowsiness:
                # -------------------------------------------------------------
                # Active Session Mode: Run real-time DrowsinessEngine inference
                # -------------------------------------------------------------
                try:
                    engine = get_drowsiness_engine()
                    result = engine.process_frame(frame)
                    
                    face_detected = result.get("face_detected", False)
                    eye_raw = result.get("eye_status")
                    yawn_raw = result.get("yawn_status")
                    alertness_level = result.get("alertness_level", "ALERT")
                    alertness_percent = result.get("alertness_percent", 100 if face_detected else 0)
                    lstm_score = result.get("lstm_drowsy_score", 0.0)
                    yawn_count = result.get("recent_yawn_count", 0)

                    eye_status = eye_raw if (face_detected and eye_raw) else "N/A"
                    yawn_status = ("Yes" if yawn_raw == "Yawning" else "No") if face_detected else "N/A"

                    status_payload = {
                        "session_active": True,
                        "session_id": active_session_id,
                        "face_detected": face_detected,
                        "eye_status": eye_status,
                        "yawn_status": yawn_status,
                        "alertness_level": alertness_level,
                        "drowsiness_level": alertness_level,
                        "alertness_percent": alertness_percent,
                        "lstm_drowsy_score": round(float(lstm_score), 4),
                        "recent_yawn_count": int(yawn_count),
                    }
                    update_latest_status(status_payload)

                    # Overlay detection HUD on frame
                    draw_detection_overlay(frame, status_payload)

                    # Stateful event logging into MySQL (asynchronous)
                    try:
                        logger.record_frame_state(
                            alertness_level=alertness_level,
                            eye_status=eye_status,
                            yawn_status=yawn_status,
                            lstm_score=float(lstm_score),
                            recent_yawn_count=int(yawn_count),
                            alertness_percent=int(alertness_percent),
                            user_id=logger.session_user_id or 1
                        )
                    except Exception as err:
                        print(f"[WARN] Event logger record error: {err}")

                except Exception as e:
                    print(f"[ERROR] Frame processing error: {e}")

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                return self._create_placeholder_frame("Frame encoding error")

            return jpeg.tobytes()

    def _create_placeholder_frame(self, message="No Video"):
        """Generate a dark placeholder image with text message if camera is unavailable."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "RASD Camera Stream", (180, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(img, message, (190, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes()

    def add_client(self):
        """Increment active client count."""
        with self._lock:
            self.active_clients += 1

    def remove_client(self):
        """Decrement active client count and release hardware if no clients remain."""
        with self._lock:
            self.active_clients = max(0, self.active_clients - 1)
            if self.active_clients == 0:
                self.release()

    def release(self):
        """Safely release the OpenCV VideoCapture resource."""
        with self.frame_lock:
            try:
                get_event_logger().close_active_event()
            except Exception as e:
                print(f"[WARN] Error closing active event on camera release: {e}")

            if self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                except Exception as e:
                    print(f"[WARN] Error releasing camera: {e}")
                finally:
                    self.cap = None


camera_instance = VideoCamera()


def get_camera():
    """Access the singleton camera instance."""
    return camera_instance


def generate_video_stream():
    """
    Generator function for MJPEG multipart streaming.
    Ensures safe client connection and disconnection tracking.
    """
    camera = get_camera()
    camera.add_client()
    try:
        while True:
            frame_bytes = camera.get_frame(process_drowsiness=True)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    except (GeneratorExit, Exception):
        pass
    finally:
        camera.remove_client()
