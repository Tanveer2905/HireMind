import sqlite3
import os
import uuid
from typing import Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def create_user(email: str, password_hash: str) -> str:
    user_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
        (user_id, email, password_hash)
    )
    conn.commit()
    conn.close()
    
    # Pre-create the user's isolated directories
    user_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users", user_id)
    os.makedirs(os.path.join(user_dir, "resumes"), exist_ok=True)
    os.makedirs(os.path.join(user_dir, "jd"), exist_ok=True)
    os.makedirs(os.path.join(user_dir, "faiss_index"), exist_ok=True)
    
    return user_id
