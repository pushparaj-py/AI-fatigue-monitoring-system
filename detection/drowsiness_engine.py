"""
drowsiness_engine.py

The real-time Drowsiness Decision Engine (Section 7.9 / 7.8 of the RASD
synopsis). Combines:
    - Per-frame eye-state CNN score (via feature_extraction.py)
    - Per-frame yawn CNN score (via feature_extraction.py)
    - LSTM sustained-closure prediction, computed over a rolling window
      of the last SEQUENCE_LENGTH frames

...into a single alertness classification per frame:
    ALERT | EARLY_DROWSINESS | DROWSY | CRITICAL

Usage:
    engine = DrowsinessEngine()
    result = engine.process_frame(frame)   # frame = one BGR OpenCV frame
    # result is a dict — see process_frame()'s docstring for its fields
"""

import os
from collections import deque
import numpy as np
import tensorflow as tf

from detection.feature_extraction import extract_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LSTM_MODEL_PATH = os.path.join("models", "lstm_model_v3", "lstm_model.keras")
SEQUENCE_LENGTH = 20  # must match the window length used during LSTM training

EYE_CNN_SCORE_INDEX = 4
YAWN_CNN_SCORE_INDEX = 5

EYE_OPEN_THRESHOLD = 0.5
YAWN_DETECTED_THRESHOLD = 0.5
YAWN_CONSECUTIVE_FRAMES_REQUIRED = 3  # require this many frames in a row above
                                        # threshold before confirming a yawn —
                                        # filters out momentary single-frame
                                        # misfires (e.g. from talking/mouth shape)

# How many recent yawns (within the last YAWN_HISTORY_SIZE frames) count as
# "repeated yawning" for escalating the alert level
YAWN_HISTORY_SIZE = 60  # roughly the last ~1-2 minutes of sampled frames
REPEATED_YAWN_COUNT = 2

# Drowsiness level thresholds, based on the LSTM's sustained-closure probability
CRITICAL_LSTM_THRESHOLD = 0.7
DROWSY_LSTM_THRESHOLD = 0.5
EARLY_DROWSINESS_LSTM_THRESHOLD = 0.3


class DrowsinessEngine:
    def __init__(self):
        self.lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        self.feature_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.yawn_history = deque(maxlen=YAWN_HISTORY_SIZE)
        self.last_lstm_score = 0.0
        self.consecutive_yawn_frames = 0

    def process_frame(self, frame):
        """
        Processes a single frame and returns the current drowsiness assessment.

        Returns a dict with:
            face_detected (bool)
            eye_status ("Open" | "Closed" | None)
            yawn_status ("Yawning" | "No" | None)
            lstm_drowsy_score (float 0-1, sustained-closure probability;
                                0.0 until the rolling buffer fills up)
            recent_yawn_count (int, yawns in the recent history window)
            alertness_level ("ALERT" | "EARLY_DROWSINESS" | "DROWSY" | "CRITICAL" | "NO_FACE")
            alertness_percent (int 0-100, for display — 100 = fully alert)
        """
        features = extract_features(frame)

        if features is None:
            return {
                "face_detected": False,
                "eye_status": None,
                "yawn_status": None,
                "lstm_drowsy_score": self.last_lstm_score,
                "recent_yawn_count": sum(self.yawn_history),
                "alertness_level": "NO_FACE",
                "alertness_percent": 0,
            }

        eye_cnn_score = features[EYE_CNN_SCORE_INDEX]
        yawn_cnn_score = features[YAWN_CNN_SCORE_INDEX]

        eye_status = "Open" if eye_cnn_score >= EYE_OPEN_THRESHOLD else "Closed"

        # Track consecutive above-threshold frames before confirming a yawn,
        # to filter out momentary single-frame misfires.
        if yawn_cnn_score >= YAWN_DETECTED_THRESHOLD:
            self.consecutive_yawn_frames += 1
        else:
            self.consecutive_yawn_frames = 0

        is_yawning = self.consecutive_yawn_frames >= YAWN_CONSECUTIVE_FRAMES_REQUIRED
        yawn_status = "Yawning" if is_yawning else "No"
        self.yawn_history.append(1 if is_yawning else 0)

        self.feature_buffer.append(features)

        if len(self.feature_buffer) == SEQUENCE_LENGTH:
            sequence = np.array(self.feature_buffer, dtype="float32")
            sequence = np.expand_dims(sequence, axis=0)  # shape (1, 20, 6)
            lstm_drowsy_score = float(self.lstm_model.predict(sequence, verbose=0)[0][0])
            self.last_lstm_score = lstm_drowsy_score
        else:
            # Buffer not full yet — not enough history for a reliable LSTM
            # prediction, so report the last known score (starts at 0.0).
            lstm_drowsy_score = self.last_lstm_score

        recent_yawn_count = sum(self.yawn_history)

        alertness_level = self._classify_level(lstm_drowsy_score, recent_yawn_count)
        alertness_percent = int(round((1.0 - lstm_drowsy_score) * 100))

        return {
            "face_detected": True,
            "eye_status": eye_status,
            "yawn_status": yawn_status,
            "lstm_drowsy_score": lstm_drowsy_score,
            "recent_yawn_count": recent_yawn_count,
            "alertness_level": alertness_level,
            "alertness_percent": alertness_percent,
        }

    def _classify_level(self, lstm_score, recent_yawn_count):
        """Combines the LSTM sustained-closure score with recent yawn frequency
        into the four-level classification from the synopsis."""
        repeated_yawning = recent_yawn_count >= REPEATED_YAWN_COUNT

        if lstm_score >= CRITICAL_LSTM_THRESHOLD and repeated_yawning:
            return "CRITICAL"
        elif lstm_score >= CRITICAL_LSTM_THRESHOLD:
            return "DROWSY"  # sustained closure alone, without repeated yawning yet
        elif lstm_score >= DROWSY_LSTM_THRESHOLD:
            return "DROWSY"
        elif lstm_score >= EARLY_DROWSINESS_LSTM_THRESHOLD or repeated_yawning:
            return "EARLY_DROWSINESS"
        else:
            return "ALERT"