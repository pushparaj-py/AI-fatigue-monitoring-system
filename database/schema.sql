-- ==========================================================
-- Database Schema for Real-time Alertness & Somnolence Detection (RASD)
-- Database Name: rasd_db
-- ==========================================================

CREATE DATABASE IF NOT EXISTS rasd_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE rasd_db;

-- ----------------------------------------------------------
-- 1. Table: users
-- Stores registered driver/user accounts
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------
-- 2. Table: admins
-- Stores administrator credentials and details.
-- NOTE: This table is intended to hold a single seeded admin account only.
-- There is no public admin registration flow in this application.
-- Admin accounts are created via a one-time seed script (database/seed_admin.py),
-- not through the web UI.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS admins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_admin_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------
-- 3. Table: detection_sessions
-- Records each monitoring session for a user
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS detection_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    start_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME NULL,
    duration INT DEFAULT 0 COMMENT 'Duration in seconds',
    overall_status VARCHAR(50) DEFAULT 'Normal' COMMENT 'e.g., Normal, Warning, Drowsy, Critical',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sessions_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    INDEX idx_session_user (user_id),
    INDEX idx_session_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------
-- 4. Table: drowsiness_events
-- Logs specific drowsiness/fatigue triggers during a session
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS drowsiness_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    event_type VARCHAR(50) NOT NULL COMMENT 'e.g., Eye Closure, Yawn, Microsleep, Head Nod',
    severity VARCHAR(20) NOT NULL DEFAULT 'Low' COMMENT 'e.g., Low, Medium, High, Critical',
    duration FLOAT DEFAULT 0.0 COMMENT 'Event duration in seconds',
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_events_session
        FOREIGN KEY (session_id) REFERENCES detection_sessions(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    INDEX idx_event_session (session_id),
    INDEX idx_event_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------
-- 5. Table: detection_statistics
-- Aggregated statistical metrics per detection session
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS detection_statistics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL UNIQUE,
    eye_closure_count INT DEFAULT 0,
    yawn_count INT DEFAULT 0,
    warning_count INT DEFAULT 0,
    critical_count INT DEFAULT 0,
    average_alertness FLOAT DEFAULT 100.0 COMMENT 'Percentage score (0 - 100)',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_statistics_session
        FOREIGN KEY (session_id) REFERENCES detection_sessions(id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    INDEX idx_stats_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------
-- 6. Table: activity_logs
-- System and user action audit trail
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS activity_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NULL,
    activity_type VARCHAR(50) NOT NULL COMMENT 'e.g., LOGIN, LOGOUT, SESSION_START, SESSION_END',
    description TEXT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45) NULL COMMENT 'Supports IPv4 and IPv6',
    CONSTRAINT fk_logs_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    INDEX idx_logs_user (user_id),
    INDEX idx_logs_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
