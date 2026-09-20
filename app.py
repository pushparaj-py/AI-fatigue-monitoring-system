"""
Main Flask application entry point for Real-time Alertness & Somnolence Detection (RASD).
Handles user authentication, session-protected dashboard, and OpenCV live webcam streaming.
"""

from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from database.db import get_db_connection
from utils.camera import generate_video_stream, get_latest_status

app = Flask(__name__)
app.config.from_object(Config)


def login_required(f):
    """Decorator to require login for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/")
def index():
    """Root route redirects to dashboard if logged in, otherwise to login."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration route."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        conn = get_db_connection()
        if not conn:
            flash("Database connection failed. Please try again later.", "danger")
            return render_template("register.html")

        try:
            cursor = conn.cursor(dictionary=True)
            # Check if email is already registered
            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash("Email is already registered. Please log in or use another email.", "danger")
                return render_template("register.html")

            # Hash the password and insert the new user
            hashed_password = generate_password_hash(password)
            insert_query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (name, email, hashed_password))
            
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

        except Exception as err:
            flash(f"An error occurred during registration: {err}", "danger")
            return render_template("register.html")
        finally:
            cursor.close()
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login route."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html", role="employee")

        conn = get_db_connection()
        if not conn:
            flash("Database connection failed. Please try again later.", "danger")
            return render_template("login.html", role="employee")

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, name, email, password FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]

                flash(f"Welcome back, {user['name']}!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid email or password.", "danger")
                return render_template("login.html", role="employee")

        except Exception as err:
            flash(f"An error occurred during login: {err}", "danger")
            return render_template("login.html", role="employee")
        finally:
            cursor.close()
            conn.close()

    return render_template("login.html", role="employee")
@app.route("/dashboard")
@login_required
def dashboard():
    """User dashboard route with embedded live analytics charts and KPIs."""
    user_id = session.get("user_id")
    user_name = session.get("user_name", "User")

    conn = get_db_connection()
    kpis = {
        "total_sessions": 0,
        "total_duration_sec": 0,
        "total_duration_str": "0s",
        "avg_alertness": 100.0,
        "total_warnings": 0,
        "total_critical": 0,
    }
    recent_sessions = []
    chart_data = {
        "labels": [],
        "alertness_scores": [],
        "warning_counts": [],
        "critical_counts": [],
    }

    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Fetch user's finished sessions with statistics
            query = """
                SELECT 
                    s.id,
                    s.start_time,
                    s.end_time,
                    s.duration,
                    s.overall_status,
                    COALESCE(st.eye_closure_count, 0) AS eye_closure_count,
                    COALESCE(st.yawn_count, 0) AS yawn_count,
                    COALESCE(st.warning_count, 0) AS warning_count,
                    COALESCE(st.critical_count, 0) AS critical_count,
                    COALESCE(st.average_alertness, 100.0) AS average_alertness
                FROM detection_sessions s
                LEFT JOIN detection_statistics st ON s.id = st.session_id
                WHERE s.user_id = %s
                ORDER BY s.start_time DESC
            """
            cursor.execute(query, (user_id,))
            all_sessions = cursor.fetchall()

            if all_sessions:
                kpis["total_sessions"] = len(all_sessions)
                total_sec = sum(s["duration"] or 0 for s in all_sessions)
                kpis["total_duration_sec"] = total_sec
                if total_sec < 60:
                    kpis["total_duration_str"] = f"{total_sec}s"
                elif total_sec < 3600:
                    kpis["total_duration_str"] = f"{total_sec // 60}m {total_sec % 60}s"
                else:
                    kpis["total_duration_str"] = f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"

                avg_scores = [float(s["average_alertness"]) for s in all_sessions if s["average_alertness"] is not None]
                kpis["avg_alertness"] = round(sum(avg_scores) / len(avg_scores), 1) if avg_scores else 100.0
                kpis["total_warnings"] = sum(s["warning_count"] or 0 for s in all_sessions)
                kpis["total_critical"] = sum(s["critical_count"] or 0 for s in all_sessions)

                # Recent 5 for dashboard table
                recent_sessions = all_sessions[:5]

                # Recent 10 in chronological order for Chart.js
                chart_sessions = list(reversed(all_sessions[:10]))
                for s in chart_sessions:
                    dt_str = s["start_time"].strftime("%b %d %H:%M") if s.get("start_time") else f"#{s['id']}"
                    chart_data["labels"].append(f"#{s['id']} ({dt_str})")
                    chart_data["alertness_scores"].append(float(s["average_alertness"] or 100.0))
                    chart_data["warning_counts"].append(int(s["warning_count"] or 0))
                    chart_data["critical_counts"].append(int(s["critical_count"] or 0))

        except Exception as err:
            print(f"[ERROR] Failed to fetch dashboard data: {err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "dashboard.html",
        name=user_name,
        kpis=kpis,
        recent_sessions=recent_sessions,
        chart_data=chart_data
    )


@app.route("/history")
@login_required
def history():
    """History page showing table of all detection sessions and detailed events."""
    user_id = session.get("user_id")
    user_name = session.get("user_name", "User")

    sessions_list = []
    events_by_session = {}

    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            query = """
                SELECT 
                    s.id,
                    s.start_time,
                    s.end_time,
                    s.duration,
                    s.overall_status,
                    COALESCE(st.eye_closure_count, 0) AS eye_closure_count,
                    COALESCE(st.yawn_count, 0) AS yawn_count,
                    COALESCE(st.warning_count, 0) AS warning_count,
                    COALESCE(st.critical_count, 0) AS critical_count,
                    COALESCE(st.average_alertness, 100.0) AS average_alertness
                FROM detection_sessions s
                LEFT JOIN detection_statistics st ON s.id = st.session_id
                WHERE s.user_id = %s
                ORDER BY s.start_time DESC
            """
            cursor.execute(query, (user_id,))
            sessions_list = cursor.fetchall()

            if sessions_list:
                session_ids = [s["id"] for s in sessions_list]
                format_strings = ','.join(['%s'] * len(session_ids))
                events_query = f"""
                    SELECT id, session_id, event_type, severity, duration, timestamp
                    FROM drowsiness_events
                    WHERE session_id IN ({format_strings})
                    ORDER BY timestamp ASC
                """
                cursor.execute(events_query, tuple(session_ids))
                all_events = cursor.fetchall()

                for ev in all_events:
                    sid = ev["session_id"]
                    if sid not in events_by_session:
                        events_by_session[sid] = []
                    events_by_session[sid].append(ev)

        except Exception as err:
            print(f"[ERROR] Failed to fetch history data: {err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "history.html",
        name=user_name,
        sessions=sessions_list,
        events_by_session=events_by_session
    )


@app.route("/session_events/<int:session_id>")
@login_required
def session_events(session_id):
    """JSON endpoint returning event records for a specific detection session."""
    user_id = session.get("user_id")
    conn = get_db_connection()
    if not conn:
        return jsonify({"success": False, "error": "Database unavailable"}), 500

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT id FROM detection_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id)
        )
        sess_row = cursor.fetchone()
        if not sess_row:
            return jsonify({"success": False, "error": "Session not found"}), 404

        cursor.execute(
            "SELECT id, session_id, event_type, severity, duration, timestamp FROM drowsiness_events WHERE session_id = %s ORDER BY timestamp ASC",
            (session_id,)
        )
        events = cursor.fetchall()
        for ev in events:
            if ev.get("timestamp"):
                ev["timestamp"] = ev["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify({"success": True, "events": events})
    except Exception as err:
        return jsonify({"success": False, "error": str(err)}), 500
    finally:
        cursor.close()
        conn.close()


@app.route("/detection")
@login_required
def detection():
    """Dedicated real-time detection page."""
    user_name = session.get("user_name", "User")
    return render_template("detection.html", name=user_name)


@app.route("/video_feed")
@login_required
def video_feed():
    """
    Video streaming route. Returns multipart/x-mixed-replace MJPEG stream
    from OpenCV VideoCapture with DrowsinessEngine real-time analysis overlay.
    """
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detection_status")
@login_required
def detection_status():
    """
    JSON endpoint exposing the latest frame inference results from DrowsinessEngine.
    Polled periodically by the live detection web interface.
    """
    return jsonify(get_latest_status())


@app.route("/start_session", methods=["POST"])
@login_required
def start_session():
    """
    Starts a new live detection session for the authenticated user.
    Enables real-time DrowsinessEngine processing.
    """
    from detection.event_logger import get_event_logger
    user_id = session.get("user_id", 1)
    logger = get_event_logger()
    session_id = logger.start_session(user_id=user_id)
    if session_id:
        return jsonify({"success": True, "session_id": session_id})
    return jsonify({"success": False, "error": "Unable to initialize detection session"}), 500


@app.route("/stop_session", methods=["POST"])
@login_required
def stop_session():
    """
    Stops the active detection session, writes final duration and overall status
    to detection_sessions, and records aggregate metrics in detection_statistics.
    """
    from detection.event_logger import get_event_logger
    logger = get_event_logger()
    summary = logger.stop_session()
    if summary:
        return jsonify({"success": True, "summary": summary})
    return jsonify({"success": False, "error": "No active detection session to stop"}), 400


@app.route("/logout")
def logout():
    """Clear session and log out the user."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))

"""
Additions for app.py — paste these into your existing app.py, ABOVE the
`if __name__ == "__main__":` line at the bottom.

Also add this import near your other imports at the top of app.py:
    from datetime import datetime
"""

from datetime import datetime  # add this to your existing imports at the top of app.py


def admin_required(f):
    """Decorator to require a valid admin session for protected admin routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "admin_id" not in session:
            flash("Please log in as an administrator to access this page.", "warning")
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Admin login route — checks credentials against the admins table, NOT users.
    There is intentionally no admin registration route or signup page."""
    if "admin_id" in session:
        return redirect(url_for("admin_dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html", role="admin")

        conn = get_db_connection()
        if not conn:
            flash("Database connection failed. Please try again later.", "danger")
            return render_template("login.html", role="admin")

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, name, email, password FROM admins WHERE email = %s", (email,))
            admin = cursor.fetchone()

            if admin and check_password_hash(admin["password"], password):
                session["admin_id"] = admin["id"]
                session["admin_name"] = admin["name"]
                flash(f"Welcome back, {admin['name']}.", "success")
                return redirect(url_for("admin_dashboard"))
            else:
                flash("Invalid admin email or password.", "danger")
                return render_template("login.html", role="admin")

        except Exception as err:
            flash(f"An error occurred during login: {err}", "danger")
            return render_template("login.html", role="admin")
        finally:
            cursor.close()
            conn.close()

    return render_template("login.html", role="admin")


@app.route("/admin/logout")
def admin_logout():
    """Clear admin session only (does not affect a separate employee session)."""
    session.pop("admin_id", None)
    session.pop("admin_name", None)
    flash("Admin logged out successfully.", "info")
    return redirect(url_for("admin_login"))


def _classify_risk(critical_count, warning_count, yawn_count, microsleep_count):
    """Combines today's event counts into a single risk level. Thresholds are
    a reasonable starting point, not derived from validated research — tune
    based on real usage patterns."""
    if critical_count > 0:
        return "CRITICAL"
    elif warning_count > 0 or microsleep_count >= 3 or yawn_count >= 5:
        return "MODERATE"
    else:
        return "NORMAL"


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Management/HR dashboard — organization-wide view across all employees."""
    conn = get_db_connection()
    stats = {"total_employees": 0, "critical_count": 0, "yawns_today": 0, "microsleep_today": 0}
    department_labels, department_counts = [], []
    hourly_labels, hourly_counts = [], []
    employees = []

    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if conn:
        try:
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT COUNT(*) AS total FROM users")
            stats["total_employees"] = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT COUNT(DISTINCT ds.user_id) AS total
                FROM drowsiness_events de
                JOIN detection_sessions ds ON de.session_id = ds.id
                WHERE de.severity = 'CRITICAL' AND de.timestamp >= %s
            """, (today_start,))
            stats["critical_count"] = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT COALESCE(SUM(dstat.yawn_count), 0) AS total
                FROM detection_statistics dstat
                JOIN detection_sessions ds ON dstat.session_id = ds.id
                WHERE ds.start_time >= %s
            """, (today_start,))
            stats["yawns_today"] = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT COALESCE(SUM(dstat.eye_closure_count), 0) AS total
                FROM detection_statistics dstat
                JOIN detection_sessions ds ON dstat.session_id = ds.id
                WHERE ds.start_time >= %s
            """, (today_start,))
            stats["microsleep_today"] = cursor.fetchone()["total"]

            cursor.execute("""
                SELECT u.department AS department,
                       COALESCE(SUM(dstat.warning_count + dstat.critical_count), 0) AS fatigue_events
                FROM users u
                LEFT JOIN detection_sessions ds ON ds.user_id = u.id AND ds.start_time >= %s
                LEFT JOIN detection_statistics dstat ON dstat.session_id = ds.id
                GROUP BY u.department
                ORDER BY fatigue_events DESC
            """, (today_start,))
            dept_rows = cursor.fetchall()
            department_labels = [row["department"] for row in dept_rows]
            department_counts = [int(row["fatigue_events"]) for row in dept_rows]

            cursor.execute("""
                SELECT HOUR(timestamp) AS hour, COUNT(*) AS event_count
                FROM drowsiness_events
                WHERE timestamp >= %s
                GROUP BY HOUR(timestamp)
                ORDER BY hour
            """, (today_start,))
            hour_rows = cursor.fetchall()
            hourly_labels = [f"{row['hour']}:00" for row in hour_rows]
            hourly_counts = [row["event_count"] for row in hour_rows]

            cursor.execute("""
                SELECT
                    u.id AS employee_id, u.name AS name, u.department AS department,
                    COALESCE(SUM(dstat.yawn_count), 0) AS yawn_count,
                    COALESCE(SUM(dstat.eye_closure_count), 0) AS microsleep_count,
                    COALESCE(SUM(dstat.critical_count), 0) AS critical_count,
                    COALESCE(SUM(dstat.warning_count), 0) AS warning_count
                FROM users u
                LEFT JOIN detection_sessions ds ON ds.user_id = u.id AND ds.start_time >= %s
                LEFT JOIN detection_statistics dstat ON dstat.session_id = ds.id
                GROUP BY u.id, u.name, u.department
            """, (today_start,))
            for row in cursor.fetchall():
                employees.append({
                    "employee_id": row["employee_id"],
                    "name": row["name"],
                    "department": row["department"],
                    "yawn_count": row["yawn_count"],
                    "microsleep_count": row["microsleep_count"],
                    "risk_level": _classify_risk(
                        row["critical_count"], row["warning_count"],
                        row["yawn_count"], row["microsleep_count"]
                    ),
                })

            risk_order = {"CRITICAL": 0, "MODERATE": 1, "NORMAL": 2}
            employees.sort(key=lambda e: risk_order[e["risk_level"]])

        except Exception as err:
            print(f"[ERROR] Failed to fetch admin dashboard data: {err}")
        finally:
            cursor.close()
            conn.close()

    return render_template(
        "admin_dashboard.html",
        admin_name=session.get("admin_name", "Admin"),
        stats=stats,
        department_labels=department_labels,
        department_counts=department_counts,
        hourly_labels=hourly_labels,
        hourly_counts=hourly_counts,
        employees=employees,
    )


@app.route("/admin/employee/<int:employee_id>")
@admin_required
def admin_employee_detail(employee_id):
    """Admin view of one specific employee's full session history."""
    conn = get_db_connection()
    employee = None
    sessions_list = []

    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, name, email, department FROM users WHERE id = %s", (employee_id,))
            employee = cursor.fetchone()

            if employee:
                cursor.execute("""
                    SELECT
                        s.id, s.start_time, s.end_time, s.duration, s.overall_status,
                        COALESCE(st.eye_closure_count, 0) AS eye_closure_count,
                        COALESCE(st.yawn_count, 0) AS yawn_count,
                        COALESCE(st.warning_count, 0) AS warning_count,
                        COALESCE(st.critical_count, 0) AS critical_count,
                        COALESCE(st.average_alertness, 100.0) AS average_alertness
                    FROM detection_sessions s
                    LEFT JOIN detection_statistics st ON s.id = st.session_id
                    WHERE s.user_id = %s
                    ORDER BY s.start_time DESC
                """, (employee_id,))
                sessions_list = cursor.fetchall()

        except Exception as err:
            print(f"[ERROR] Failed to fetch employee detail: {err}")
        finally:
            cursor.close()
            conn.close()

    if not employee:
        flash("Employee not found.", "danger")
        return redirect(url_for("admin_dashboard"))

    return render_template("admin_employee_detail.html", employee=employee, sessions=sessions_list)
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
