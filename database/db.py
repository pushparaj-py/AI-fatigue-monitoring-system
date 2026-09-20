"""
Database connection and helper module for RASD.
Uses mysql-connector-python to establish connections and interact with MySQL database.
"""

import os
import mysql.connector
from mysql.connector import Error, pooling

# Default MySQL connection configurations
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "pushpa@123"),
    "database": os.environ.get("DB_NAME", "rasd_db"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "autocommit": True,
    "charset": "utf8mb4",
    "collation": "utf8mb4_unicode_ci"
}


def get_db_connection(config_override=None, include_database=True):
    """
    Establish and return a MySQL database connection.

    Args:
        config_override (dict, optional): Custom connection dictionary parameters.
        include_database (bool, optional): Whether to specify the database name upon connecting.
                                           Set to False when creating the database itself.

    Returns:
        mysql.connector.connection_cext.CMySQLConnection | None: Active MySQL connection or None on failure.
    """
    config = DB_CONFIG.copy()
    if not include_database:
        config.pop("database", None)
    if config_override:
        config.update(config_override)

    try:
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            return connection
    except Error as err:
        print(f"[ERROR] MySQL connection error: {err}")
        return None


def init_db(schema_file=None):
    """
    Initialize the database using the SQL schema file.
    Creates the database and tables if they don't exist.

    Args:
        schema_file (str, optional): Path to the schema.sql file.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    if schema_file is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        schema_file = os.path.join(base_dir, "schema.sql")

    if not os.path.exists(schema_file):
        print(f"[ERROR] Schema file not found: {schema_file}")
        return False

    with open(schema_file, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    # First connect without specifying database to create database if not exists
    conn = get_db_connection(include_database=False)
    if not conn:
        print("[ERROR] Failed to connect to MySQL server to initialize database.")
        return False

    try:
        cursor = conn.cursor()
        # Parse statements separated by ';' ignoring comments
        statements = []
        current_stmt = []
        for line in schema_sql.splitlines():
            stripped = line.strip()
            if stripped.startswith("--") or not stripped:
                continue
            current_stmt.append(line)
            if stripped.endswith(";"):
                stmt_text = "\n".join(current_stmt).strip()
                if stmt_text:
                    statements.append(stmt_text)
                current_stmt = []

        for statement in statements:
            if statement:
                cursor.execute(statement)

        print("[SUCCESS] Database schema initialized successfully.")
        return True
    except Error as err:
        print(f"[ERROR] Failed to initialize schema: {err}")
        return False
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    print("Testing MySQL connection configuration...")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"Database: {DB_CONFIG['database']}")
    
    test_conn = get_db_connection()
    if test_conn and test_conn.is_connected():
        print("[SUCCESS] Successfully connected to MySQL database!")
        test_conn.close()
    else:
        print("[INFO] Could not connect with default settings. Check if MySQL server is running.")
