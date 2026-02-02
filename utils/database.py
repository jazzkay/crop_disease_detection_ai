import sqlite3
from datetime import datetime

DB_NAME = "database/history.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        crop TEXT,
        disease TEXT,
        confidence REAL,
        severity TEXT,
        reliability REAL
    )
    """)

    conn.commit()
    conn.close()


def insert_record(crop, disease, confidence, severity, reliability):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history 
    (timestamp, crop, disease, confidence, severity, reliability)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        crop, disease, confidence, severity, reliability
    ))

    conn.commit()
    conn.close()


def fetch_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows
