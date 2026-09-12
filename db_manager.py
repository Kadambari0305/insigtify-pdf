import os
import hashlib
import json
import uuid
import logging
from datetime import datetime
import sqlite3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# DB Path for SQLite Fallback
DB_DIR = os.path.join(os.path.dirname(__file__), "data")
SQLITE_DB_PATH = os.path.join(DB_DIR, "app_database.db")

def _hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

class DatabaseManager:
    def __init__(self):
        self.use_mongodb = False
        self.mongo_db = None
        self._init_connection()

    def _init_connection(self):
        """Try connecting to MongoDB, fallback to SQLite on DNS/connection failure."""
        mongo_uri = os.getenv("MONGO_URI")
        if mongo_uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
                client.admin.command('ping')
                self.mongo_db = client["pdf_chatbot"]
                self.use_mongodb = True
                logger.info("Connected to MongoDB Cloud successfully.")
                return
            except Exception as e:
                logger.warning(f"MongoDB connection failed ({e}). Falling back to SQLite local database.")
        
        # Fallback SQLite Initialization
        os.makedirs(DB_DIR, exist_ok=True)
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    email TEXT,
                    password_hash TEXT,
                    created_at TEXT
                )
            ''')
            # Upgrade table schema if old table exists with column 'password'
            cursor.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'password' in columns and 'password_hash' not in columns:
                cursor.execute("ALTER TABLE users RENAME COLUMN password TO password_hash")

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    title TEXT,
                    files TEXT,
                    messages TEXT,
                    updated_at TEXT
                )
            ''')
            conn.commit()

    def register_user(self, username: str, email: str, password: str) -> tuple[bool, str]:
        """Register a new user."""
        username = username.strip().lower()
        if not username or not password:
            return False, "Username and Password cannot be empty."

        pwd_hash = _hash_password(password)
        created_at = datetime.now().isoformat()

        if self.use_mongodb:
            try:
                users_col = self.mongo_db["users"]
                if users_col.find_one({"username": username}):
                    return False, "Username already exists."
                users_col.insert_one({
                    "username": username,
                    "email": email,
                    "password_hash": pwd_hash,
                    "created_at": created_at
                })
                return True, "User registered successfully!"
            except Exception as e:
                return False, f"Database error: {str(e)}"
        else:
            try:
                with sqlite3.connect(SQLITE_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
                    if cursor.fetchone():
                        return False, "Username already exists."
                    cursor.execute(
                        "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                        (username, email, pwd_hash, created_at)
                    )
                    conn.commit()
                    return True, "User registered successfully!"
            except Exception as e:
                return False, f"Database error: {str(e)}"

    def authenticate_user(self, username: str, password: str) -> tuple[bool, str]:
        """Authenticate user credentials."""
        username = username.strip().lower()
        pwd_hash = _hash_password(password)

        if self.use_mongodb:
            try:
                user = self.mongo_db["users"].find_one({"username": username})
                if user and user.get("password_hash") == pwd_hash:
                    return True, "Authentication successful."
                return False, "Invalid username or password."
            except Exception as e:
                return False, f"Database error: {str(e)}"
        else:
            try:
                with sqlite3.connect(SQLITE_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
                    row = cursor.fetchone()
                    if row and row[0] == pwd_hash:
                        return True, "Authentication successful."
                    return False, "Invalid username or password."
            except Exception as e:
                return False, f"Database error: {str(e)}"

    def save_chat_turn(self, username: str, session_id: str, session_title: str, files: list, user_msg: str, assistant_msg: str):
        """Save a Q&A chat turn to conversation history."""
        if not username:
            return
        username = str(username).strip().lower()
        now = datetime.now().isoformat()

        if self.use_mongodb:
            try:
                convs_col = self.mongo_db["conversations"]
                existing = convs_col.find_one({"session_id": session_id})
                
                new_turns = [
                    {"role": "user", "content": user_msg, "timestamp": now},
                    {"role": "assistant", "content": assistant_msg, "timestamp": now}
                ]

                if existing:
                    convs_col.update_one(
                        {"session_id": session_id},
                        {
                            "$push": {"messages": {"$each": new_turns}},
                            "$set": {"updated_at": now, "files": files, "title": session_title}
                        }
                    )
                else:
                    convs_col.insert_one({
                        "session_id": session_id,
                        "username": username,
                        "title": session_title,
                        "files": files,
                        "messages": new_turns,
                        "updated_at": now
                    })
            except Exception as e:
                logger.error(f"Failed to save chat turn to MongoDB: {e}")
        else:
            try:
                with sqlite3.connect(SQLITE_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT messages FROM conversations WHERE session_id = ?", (session_id,))
                    row = cursor.fetchone()

                    new_turns = [
                        {"role": "user", "content": user_msg, "timestamp": now},
                        {"role": "assistant", "content": assistant_msg, "timestamp": now}
                    ]

                    if row:
                        messages = json.loads(row[0])
                        messages.extend(new_turns)
                        cursor.execute(
                            "UPDATE conversations SET title = ?, files = ?, messages = ?, updated_at = ? WHERE session_id = ?",
                            (session_title, json.dumps(files), json.dumps(messages), now, session_id)
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO conversations (session_id, username, title, files, messages, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (session_id, username, session_title, json.dumps(files), json.dumps(new_turns), now)
                        )
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save chat turn to SQLite: {e}")

    def get_user_sessions(self, username: str) -> list:
        """Get all past chat sessions for a user."""
        if not username:
            return []
        username = str(username).strip().lower()
        sessions = []

        if self.use_mongodb:
            try:
                cursor = self.mongo_db["conversations"].find({"username": username}).sort("updated_at", -1)
                for doc in cursor:
                    sessions.append({
                        "session_id": doc["session_id"],
                        "title": doc.get("title", "Chat Session"),
                        "files": doc.get("files", []),
                        "messages": doc.get("messages", []),
                        "updated_at": doc.get("updated_at", "")
                    })
            except Exception as e:
                logger.error(f"Error fetching MongoDB sessions: {e}")
        else:
            try:
                with sqlite3.connect(SQLITE_DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT session_id, title, files, messages, updated_at FROM conversations WHERE username = ? ORDER BY updated_at DESC",
                        (username,)
                    )
                    rows = cursor.fetchall()
                    for r in rows:
                        sessions.append({
                            "session_id": r[0],
                            "title": r[1],
                            "files": json.loads(r[2]) if r[2] else [],
                            "messages": json.loads(r[3]) if r[3] else [],
                            "updated_at": r[4]
                        })
            except Exception as e:
                logger.error(f"Error fetching SQLite sessions: {e}")

        return sessions

# Global instance
db_manager = DatabaseManager()
