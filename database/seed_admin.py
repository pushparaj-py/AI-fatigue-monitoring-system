"""
One-time database seed script for the admin account.
This script creates the initial administrator record in the `admins` table.

Run manually from terminal:
    python database/seed_admin.py
"""

import sys
import os

# Ensure the project root is on sys.path so we can import database.db
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from werkzeug.security import generate_password_hash
from database.db import get_db_connection

# ==============================================================================
# ADMIN CONFIGURATION PLACEHOLDERS
# Edit these values before running the script:
# ==============================================================================
ADMIN_NAME = "Admin"
ADMIN_EMAIL = "admin@rasd.com"
ADMIN_PASSWORD = "admin123"
# ==============================================================================


def seed_admin(name=ADMIN_NAME, email=ADMIN_EMAIL, password=ADMIN_PASSWORD):
    """
    Seeds a single administrator account into the `admins` table.
    Ensures that duplicates are prevented if an admin already exists.
    """
    print("=" * 60)
    print("RASD Admin Seeding Utility")
    print("=" * 60)

    conn = get_db_connection()
    if not conn:
        print("[ERROR] Could not connect to the database. Ensure MySQL is running and configured.")
        return False

    try:
        cursor = conn.cursor(dictionary=True)

        # 1. Check if any admin account already exists
        cursor.execute("SELECT id, name, email FROM admins LIMIT 1")
        existing_admin = cursor.fetchone()

        if existing_admin:
            print(f"[INFO] An admin account already exists in the database:")
            print(f"       ID:    {existing_admin['id']}")
            print(f"       Name:  {existing_admin['name']}")
            print(f"       Email: {existing_admin['email']}")
            print("[INFO] No duplicate admin was inserted.")
            return True

        # 2. Check if the specified email already exists
        cursor.execute("SELECT id FROM admins WHERE email = %s", (email,))
        if cursor.fetchone():
            print(f"[INFO] An admin with email '{email}' already exists. No duplicate inserted.")
            return True

        # 3. Hash the password using werkzeug.security
        hashed_password = generate_password_hash(password)

        # 4. Insert the single admin record
        insert_query = """
            INSERT INTO admins (name, email, password)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (name, email, hashed_password))
        admin_id = cursor.lastrowid

        print(f"[SUCCESS] Admin account created successfully!")
        print(f"          ID:    {admin_id}")
        print(f"          Name:  {name}")
        print(f"          Email: {email}")
        print("-" * 60)
        print("[IMPORTANT] Remember to keep your admin credentials safe and update placeholder passwords.")
        return True

    except Exception as err:
        print(f"[ERROR] Failed to seed admin account: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    seed_admin()
