"""Notes rapides SQLite."""

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

_lock = threading.Lock()
DB_PATH = Path("./data/notes.db")


class NotesManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with _lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def add_note(self, content: str) -> int:
        with _lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "INSERT INTO notes (content, created_at) VALUES (?, ?)",
                    (content, datetime.now().isoformat()),
                )
                conn.commit()
                note_id = cursor.lastrowid
                logger.info(f"Note #{note_id} enregistrée")
                return note_id or 0
            except Exception:
                logger.exception("Erreur ajout note")
                raise
            finally:
                conn.close()

    def get_all_notes(self, limit: int = 50) -> list[dict[str, Any]]:
        with _lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT id, content, created_at FROM notes ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "created_at": row["created_at"],
                    }
                    for row in rows
                ]
            except Exception:
                logger.exception("Erreur lecture notes")
                return []
            finally:
                conn.close()

    def get_notes_text(self, limit: int = 20) -> str:
        notes = self.get_all_notes(limit)
        if not notes:
            return "Aucune note enregistrée."
        lines = []
        for note in notes:
            lines.append(f"[#{note['id']}] {note['content']} ({note['created_at'][:10]})")
        return "\n".join(lines)


_notes_instance: NotesManager | None = None


def get_notes_manager() -> NotesManager:
    global _notes_instance
    if _notes_instance is None:
        _notes_instance = NotesManager()
    return _notes_instance
