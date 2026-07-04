import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SQLiteStore:
    def __init__(self, db_path: str = "data/rag_store.db"):
        self.db_path = db_path
        self._local = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        return self._local.conn

    def _init_schema(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks(user_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)
        """)
        conn.commit()

    def add_chunks(self, chunks: list[str], user_id: str, start_index: int = 0) -> list[int]:
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        ids: list[int] = []
        for i, chunk in enumerate(chunks):
            cursor = conn.execute(
                "INSERT INTO chunks (chunk_index, text, user_id, created_at) VALUES (?, ?, ?, ?)",
                (start_index + i, chunk, user_id, now),
            )
            ids.append(cursor.lastrowid)
        conn.commit()
        logger.info(f"Stored {len(chunks)} chunks for user '{user_id}' (start_index={start_index})")
        return ids

    def get_chunks_by_user(self, user_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT chunk_index, text, created_at FROM chunks WHERE user_id = ? ORDER BY chunk_index",
            (user_id,),
        ).fetchall()
        return [{"index": r[0], "text": r[1], "created_at": r[2]} for r in rows]

    def get_chunk_indices_by_user(self, user_id: str) -> list[int]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT chunk_index FROM chunks WHERE user_id = ? ORDER BY chunk_index",
            (user_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_all_chunks(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT chunk_index, text, user_id, created_at FROM chunks ORDER BY chunk_index"
        ).fetchall()
        return [
            {"index": r[0], "text": r[1], "user_id": r[2], "created_at": r[3]}
            for r in rows
        ]

    def get_chunk_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def delete_user_chunks(self, user_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM chunks WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount

    def get_max_chunk_index(self) -> int:
        conn = self._get_conn()
        row = conn.execute("SELECT MAX(chunk_index) FROM chunks").fetchone()
        return row[0] if row[0] is not None else -1
