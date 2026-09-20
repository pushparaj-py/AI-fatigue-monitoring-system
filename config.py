"""
Application configuration settings for RASD.
"""

import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "rasd-secret-key-development-2026")
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "pushpa@123")
    DB_NAME = os.environ.get("DB_NAME", "rasd_db")
    DB_PORT = int(os.environ.get("DB_PORT", 3306))
