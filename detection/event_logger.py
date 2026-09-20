"""
detection/event_logger.py

Asynchronous database event logger and session lifecycle manager for RASD.
Handles starting and stopping detection sessions, tracking real-time event
transitions into drowsiness_events, and writing session aggregates to detection_statistics.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from database.db import get_db_connection


class DrowsinessEventLogger:
    """
    Manages stateful tracking and persistence of detection sessions,
    drowsiness events, and session statistics in MySQL.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="drowsiness_db_worker")
        
        # Session state
        self.active_session_id = None
        self.session_user_id = None
        self.session_start_timestamp = None
        self.alertness_history = []
        self.yawn_detected_frames_count = 0

        # Event state
        self.current_event_id = None
        self.current_severity = None
        self.current_event_type = None
        self.event_start_time = None

    def is_session_active(self):
        """Returns True if a detection session is currently active."""
        with self._lock:
            return self.active_session_id is not None

    def get_active_session_id(self):
        """Returns the ID of the currently active session, or None."""
        with self._lock:
            return self.active_session_id

    def start_session(self, user_id=1):
        """
        Starts a new detection session:
          - Creates a row in detection_sessions (end_time and overall_status left NULL)
          - Stores active_session_id for subsequent event logging
          - Resets metrics and telemetry accumulators
        """
        # If an existing session is open, finalize it first
        if self.is_session_active():
            self.stop_session()

        conn = get_db_connection()
        if not conn:
            print("[ERROR] Database connection failed while starting session.")
            return None

        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO detection_sessions (user_id, start_time, end_time, duration, overall_status)
                VALUES (%s, NOW(), NULL, 0, NULL)
            """
            cursor.execute(query, (user_id,))
            session_id = cursor.lastrowid
            
            with self._lock:
                self.active_session_id = session_id
                self.session_user_id = user_id
                self.session_start_timestamp = time.time()
                self.alertness_history = []
                self.yawn_detected_frames_count = 0
                self.current_event_id = None
                self.current_severity = None
                self.current_event_type = None
                self.event_start_time = None

            print(f"[SESSION START] Started new detection_session ID={session_id} for user_id={user_id}")
            return session_id
        except Exception as e:
            print(f"[ERROR] Failed to start detection session: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def stop_session(self):
        """
        Stops the currently active detection session:
          - Finalizes any ongoing drowsiness event
          - Computes duration (end_time - start_time)
          - Derives overall_status from logged events
          - Calculates aggregate metrics and inserts into detection_statistics
          - Updates detection_sessions row
        """
        with self._lock:
            session_id = self.active_session_id
            start_ts = self.session_start_timestamp
            user_id = self.session_user_id
            alertness_vals = list(self.alertness_history)
            yawn_frames = self.yawn_detected_frames_count

        if session_id is None:
            print("[WARN] stop_session called but no active session found.")
            return None

        # 1. Close any pending in-flight event immediately
        self.close_active_event()
        time.sleep(0.1)  # Brief pause to let async event completion finish

        # 2. Compute elapsed session duration in seconds
        now_ts = time.time()
        duration_sec = int(round(max(1, now_ts - start_ts))) if start_ts else 0

        # 3. Query event statistics from database for this session
        conn = get_db_connection()
        if not conn:
            print("[ERROR] Database connection failed while stopping session.")
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT event_type, severity FROM drowsiness_events WHERE session_id = %s",
                (session_id,)
            )
            events = cursor.fetchall()

            eye_closure_count = sum(1 for e in events if e["event_type"] in ("sustained_closure", "combined"))
            yawn_events = sum(1 for e in events if e["event_type"] in ("yawning", "combined"))
            warning_count = sum(1 for e in events if e["severity"] == "DROWSY")
            critical_count = sum(1 for e in events if e["severity"] == "CRITICAL")
            
            total_yawn_count = max(yawn_events, yawn_frames)

            # 4. Derive overall_status based on severity occurrences
            if critical_count > 0:
                overall_status = "Needs Attention"
            elif warning_count > 0:
                overall_status = "Moderate"
            else:
                overall_status = "Good"

            # 5. Compute average alertness percentage
            if alertness_vals:
                average_alertness = round(float(sum(alertness_vals)) / len(alertness_vals), 1)
            else:
                average_alertness = 100.0

            # 6. Update detection_sessions row
            update_session_query = """
                UPDATE detection_sessions
                SET end_time = NOW(), duration = %s, overall_status = %s
                WHERE id = %s
            """
            cursor.execute(update_session_query, (duration_sec, overall_status, session_id))

            # 7. Insert into detection_statistics
            insert_stats_query = """
                INSERT INTO detection_statistics
                (session_id, eye_closure_count, yawn_count, warning_count, critical_count, average_alertness)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    eye_closure_count = VALUES(eye_closure_count),
                    yawn_count = VALUES(yawn_count),
                    warning_count = VALUES(warning_count),
                    critical_count = VALUES(critical_count),
                    average_alertness = VALUES(average_alertness)
            """
            cursor.execute(insert_stats_query, (
                session_id,
                eye_closure_count,
                total_yawn_count,
                warning_count,
                critical_count,
                average_alertness
            ))

            summary = {
                "session_id": session_id,
                "user_id": user_id,
                "duration": duration_sec,
                "overall_status": overall_status,
                "eye_closure_count": eye_closure_count,
                "yawn_count": total_yawn_count,
                "warning_count": warning_count,
                "critical_count": critical_count,
                "average_alertness": average_alertness,
            }

            with self._lock:
                self.active_session_id = None
                self.session_user_id = None
                self.session_start_timestamp = None
                self.alertness_history = []
                self.yawn_detected_frames_count = 0
                self.current_event_id = None
                self.current_severity = None
                self.current_event_type = None
                self.event_start_time = None

            print(f"[SESSION STOP] Finalized session ID={session_id}: {summary}")
            return summary

        except Exception as e:
            print(f"[ERROR] Failed to stop detection session: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def record_frame_state(self, alertness_level, eye_status, yawn_status, lstm_score, recent_yawn_count, alertness_percent=100, user_id=1):
        """
        Evaluates per-frame alertness state transitions during an active session.
        Only logs events to MySQL when a session is active.
        """
        with self._lock:
            session_id = self.active_session_id
            if session_id is None:
                return  # No active session — do not log events

            # Track session metrics
            if alertness_percent is not None:
                self.alertness_history.append(alertness_percent)
            if yawn_status == "Yes":
                self.yawn_detected_frames_count += 1

            is_drowsy = alertness_level in ("DROWSY", "CRITICAL")
            currently_in_event = self.current_event_id is not None or self.event_start_time is not None

            if is_drowsy and not currently_in_event:
                # -------------------------------------------------------------
                # 1. NEW EVENT STARTS
                # -------------------------------------------------------------
                if (lstm_score >= 0.5) and (yawn_status == "Yes" or recent_yawn_count >= 2):
                    event_type = "combined"
                elif yawn_status == "Yes" or recent_yawn_count >= 2:
                    event_type = "yawning"
                else:
                    event_type = "sustained_closure"

                severity = alertness_level
                start_time = time.time()
                self.event_start_time = start_time
                self.current_severity = severity
                self.current_event_type = event_type
                
                # Asynchronously insert into database with active session_id
                self.executor.submit(self._async_insert_event, session_id, event_type, severity)

            elif is_drowsy and currently_in_event:
                # -------------------------------------------------------------
                # 2. EVENT CONTINUES (Check for severity escalation)
                # -------------------------------------------------------------
                if alertness_level == "CRITICAL" and self.current_severity == "DROWSY":
                    self.current_severity = "CRITICAL"
                    event_id = self.current_event_id
                    if event_id:
                        self.executor.submit(self._async_update_severity, event_id, "CRITICAL")

            elif not is_drowsy and currently_in_event:
                # -------------------------------------------------------------
                # 3. EVENT ENDED (Returned to ALERT / EARLY_DROWSINESS / NO_FACE)
                # -------------------------------------------------------------
                event_id = self.current_event_id
                start_time = self.event_start_time
                duration = round(max(0.0, time.time() - start_time), 2) if start_time else 0.0

                # Reset state
                self.current_event_id = None
                self.event_start_time = None
                self.current_severity = None
                self.current_event_type = None

                if event_id:
                    self.executor.submit(self._async_finalize_duration, event_id, duration)

    def _async_insert_event(self, session_id, event_type, severity):
        """Worker task: Inserts new event row into drowsiness_events table."""
        conn = get_db_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO drowsiness_events (session_id, event_type, severity, duration, timestamp)
                VALUES (%s, %s, %s, 0.0, NOW())
            """
            cursor.execute(query, (session_id, event_type, severity))
            event_id = cursor.lastrowid
            
            with self._lock:
                if self.event_start_time is not None:
                    self.current_event_id = event_id

            print(f"[DB EVENT] Logged new drowsiness event: ID={event_id}, session_id={session_id}, type={event_type}, severity={severity}")
        except Exception as e:
            print(f"[ERROR] Failed to insert drowsiness event: {e}")
        finally:
            cursor.close()
            conn.close()

    def _async_update_severity(self, event_id, new_severity):
        """Worker task: Updates severity level for ongoing event."""
        conn = get_db_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE drowsiness_events SET severity = %s WHERE id = %s",
                (new_severity, event_id)
            )
        except Exception as e:
            print(f"[ERROR] Failed to update event severity: {e}")
        finally:
            cursor.close()
            conn.close()

    def _async_finalize_duration(self, event_id, duration):
        """Worker task: Updates finalized duration upon event conclusion."""
        conn = get_db_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE drowsiness_events SET duration = %s WHERE id = %s",
                (duration, event_id)
            )
            print(f"[DB EVENT] Finalized drowsiness event ID={event_id} with duration={duration}s")
        except Exception as e:
            print(f"[ERROR] Failed to finalize event duration: {e}")
        finally:
            cursor.close()
            conn.close()

    def close_active_event(self):
        """Closes any pending event when session stops or stream concludes."""
        with self._lock:
            if self.current_event_id is not None and self.event_start_time is not None:
                duration = round(max(0.0, time.time() - self.event_start_time), 2)
                event_id = self.current_event_id
                self.current_event_id = None
                self.event_start_time = None
                self.current_severity = None
                self.current_event_type = None
                self.executor.submit(self._async_finalize_duration, event_id, duration)


# Global singleton instance
event_logger = DrowsinessEventLogger()


def get_event_logger():
    """Access the global DrowsinessEventLogger instance."""
    return event_logger
