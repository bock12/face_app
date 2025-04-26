import sqlite3
from datetime import datetime

DB_PATH = "face_records.db"

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    conn = connect_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            contact TEXT,
            age TEXT,
            gender TEXT,
            address TEXT,
            occupation TEXT,
            image_path TEXT,
            encoding BLOB NOT NULL,
            date_added TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS unknown_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            encoding BLOB NOT NULL,
            date_detected TEXT NOT NULL
        )
    ''')

    conn.commit()
    conn.close()
