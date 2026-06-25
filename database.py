import sqlite3
from datetime import date

DB_PATH = "attendance.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            user_name  TEXT NOT NULL,
            date       TEXT NOT NULL,
            time       TEXT NOT NULL,
            confidence TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

def add_user(name, image_path):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO users (name, image_path) VALUES (?, ?)", (name, image_path))
    user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id

def get_all_users():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users ORDER BY name")
    users = c.fetchall()
    conn.close()
    return users

def get_user_by_name(name):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE name = ?", (name,))
    user = c.fetchone()
    conn.close()
    return user

def mark_attendance(user_id, user_name, confidence):
    conn = get_conn()
    c = conn.cursor()
    today = date.today().strftime("%Y-%m-%d")

    # Prevent duplicate attendance for same person on same day
    c.execute(
        "SELECT id FROM attendance WHERE user_id = ? AND date = ?",
        (user_id, today)
    )
    existing = c.fetchone()

    if not existing:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        c.execute(
            "INSERT INTO attendance (user_id, user_name, date, time, confidence) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_name, today, now, confidence)
        )
        conn.commit()
        conn.close()
        return True   # newly marked
    conn.close()
    return False      # already marked today

def get_attendance(filter_date=None):
    conn = get_conn()
    c = conn.cursor()
    if filter_date:
        c.execute(
            "SELECT * FROM attendance WHERE date = ? ORDER BY time DESC",
            (filter_date,)
        )
    else:
        c.execute("SELECT * FROM attendance ORDER BY date DESC, time DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def delete_user(user_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()