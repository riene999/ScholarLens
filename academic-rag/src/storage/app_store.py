"""Persistent local application state backed by SQLite."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteAppStore:
    """Store conversation history and local background-job state."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_session_id
                ON conversation_messages(session_id, id DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    chunks_added INTEGER,
                    created_at TEXT NOT NULL,
                    enqueued_at TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_index_jobs_created_at
                ON index_jobs(created_at DESC)
                """
            )

    def add_conversation_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        created_at = _utc_now()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, "user", user_content, created_at),
                    (session_id, "assistant", assistant_content, created_at),
                ],
            )

    def get_conversation_messages(
        self,
        session_id: str,
        limit: int,
    ) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM conversation_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, max(1, int(limit))),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear_conversation(self, session_id: str | None = None) -> None:
        with self._connect() as connection:
            if session_id is None:
                connection.execute("DELETE FROM conversation_messages")
            else:
                connection.execute(
                    "DELETE FROM conversation_messages WHERE session_id = ?",
                    (session_id,),
                )

    def conversation_session_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(DISTINCT session_id) AS count FROM conversation_messages"
            ).fetchone()
        return int(row["count"]) if row else 0

    def create_index_job(self, job_id: str, filename: str) -> dict:
        created_at = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO index_jobs(
                    job_id, status, filename, created_at, enqueued_at
                ) VALUES (?, 'queued', ?, ?, ?)
                """,
                (job_id, filename, created_at, created_at),
            )
        return self.get_index_job(job_id)

    def mark_index_job_started(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status = 'started', started_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (_utc_now(), job_id),
            )

    def mark_index_job_finished(self, job_id: str, chunks_added: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status = 'finished', chunks_added = ?, ended_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (int(chunks_added), _utc_now(), job_id),
            )

    def mark_index_job_failed(self, job_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status = 'failed', ended_at = ?, error = ?
                WHERE job_id = ?
                """,
                (_utc_now(), error, job_id),
            )

    def recover_interrupted_jobs(self) -> int:
        """Mark jobs left active by a previous process as failed."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET status = 'failed', ended_at = ?,
                    error = 'Service restarted before indexing completed'
                WHERE status IN ('queued', 'started')
                """,
                (_utc_now(),),
            )
        return int(cursor.rowcount)

    def get_index_job(self, job_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, filename, chunks_added, created_at,
                       enqueued_at, started_at, ended_at, error
                FROM index_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return dict(row) if row else None
