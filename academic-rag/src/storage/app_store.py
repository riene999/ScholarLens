"""Persistent local application state backed by SQLite."""

import hashlib
import json
import sqlite3
import uuid
import zlib
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
                CREATE TABLE IF NOT EXISTS conversation_rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_uid TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL,
                    tool_summary TEXT,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_rounds_session_id
                ON conversation_rounds(session_id, id ASC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS context_artifacts (
                    sha256 TEXT PRIMARY KEY,
                    artifact_type TEXT NOT NULL,
                    payload_blob BLOB NOT NULL,
                    codec TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS round_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    artifact_type TEXT NOT NULL,
                    tool_name TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY(round_id) REFERENCES conversation_rounds(id),
                    FOREIGN KEY(sha256) REFERENCES context_artifacts(sha256),
                    UNIQUE(round_id, sequence_index)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_round_artifacts_round_id
                ON round_artifacts(round_id, sequence_index)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    covers_through_round_id INTEGER NOT NULL,
                    previous_summary_id INTEGER,
                    artifact_refs_json TEXT NOT NULL DEFAULT '[]',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(previous_summary_id) REFERENCES memory_summaries(id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_summary_session
                ON memory_summaries(session_id, is_active, id DESC)
                """
            )
            self._migrate_legacy_messages(connection)
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

    @staticmethod
    def _migrate_legacy_messages(connection: sqlite3.Connection) -> None:
        """One-time upgrade of pre-round histories without altering their raw rows."""
        existing_rounds = connection.execute(
            "SELECT COUNT(*) AS count FROM conversation_rounds"
        ).fetchone()["count"]
        if existing_rounds:
            return
        rows = connection.execute(
            """
            SELECT session_id, role, content, created_at
            FROM conversation_messages
            ORDER BY session_id ASC, id ASC
            """
        ).fetchall()
        pending: dict[str, tuple[str, str]] = {}
        for row in rows:
            session_id = row["session_id"]
            if row["role"] == "user":
                previous = pending.get(session_id)
                if previous:
                    connection.execute(
                        """
                        INSERT INTO conversation_rounds(
                            round_uid, session_id, user_content, assistant_content,
                            token_count, created_at
                        ) VALUES (?, ?, ?, '', 0, ?)
                        """,
                        (uuid.uuid4().hex, session_id, previous[0], previous[1]),
                    )
                pending[session_id] = (row["content"], row["created_at"])
            elif row["role"] == "assistant" and session_id in pending:
                user_content, created_at = pending.pop(session_id)
                connection.execute(
                    """
                    INSERT INTO conversation_rounds(
                        round_uid, session_id, user_content, assistant_content,
                        token_count, created_at
                    ) VALUES (?, ?, ?, ?, 0, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        session_id,
                        user_content,
                        row["content"],
                        created_at,
                    ),
                )
        for session_id, (user_content, created_at) in pending.items():
            connection.execute(
                """
                INSERT INTO conversation_rounds(
                    round_uid, session_id, user_content, assistant_content,
                    token_count, created_at
                ) VALUES (?, ?, ?, '', 0, ?)
                """,
                (uuid.uuid4().hex, session_id, user_content, created_at),
            )

    def add_conversation_turn(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *,
        tool_summary: str | None = None,
        token_count: int = 0,
        artifacts: list[dict] | None = None,
    ) -> int:
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
            cursor = connection.execute(
                """
                INSERT INTO conversation_rounds(
                    round_uid, session_id, user_content, assistant_content,
                    tool_summary, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    session_id,
                    user_content,
                    assistant_content,
                    tool_summary,
                    max(0, int(token_count)),
                    created_at,
                ),
            )
            round_id = int(cursor.lastrowid)
            for sequence_index, artifact in enumerate(artifacts or []):
                connection.execute(
                    """
                    INSERT INTO round_artifacts(
                        round_id, sha256, sequence_index, artifact_type,
                        tool_name, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        round_id,
                        artifact["sha256"],
                        sequence_index,
                        artifact.get("artifact_type", "tool_result"),
                        artifact.get("tool_name"),
                        json.dumps(
                            artifact.get("metadata", {}),
                            ensure_ascii=False,
                            default=str,
                        ),
                    ),
                )
        return round_id

    @staticmethod
    def canonical_artifact_payload(payload: object) -> bytes:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    def put_context_artifact(
        self,
        artifact_type: str,
        payload: object,
        digest: str,
        token_count: int,
        metadata: dict | None = None,
    ) -> str:
        raw = self.canonical_artifact_payload(payload)
        sha256 = hashlib.sha256(raw).hexdigest()
        compressed = zlib.compress(raw, level=6)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO context_artifacts(
                    sha256, artifact_type, payload_blob, codec, digest,
                    token_count, metadata_json, created_at
                ) VALUES (?, ?, ?, 'zlib-json', ?, ?, ?, ?)
                """,
                (
                    sha256,
                    artifact_type,
                    compressed,
                    digest,
                    max(0, int(token_count)),
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    _utc_now(),
                ),
            )
        return sha256

    def get_context_artifact(self, sha256: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sha256, artifact_type, payload_blob, codec, digest,
                       token_count, metadata_json, created_at
                FROM context_artifacts
                WHERE sha256 = ?
                """,
                (sha256.lower(),),
            ).fetchone()
        if row is None:
            return None
        if row["codec"] != "zlib-json":
            raise ValueError(f"Unsupported artifact codec: {row['codec']}")
        payload = json.loads(zlib.decompress(row["payload_blob"]).decode("utf-8"))
        return {
            "sha256": row["sha256"],
            "artifact_type": row["artifact_type"],
            "payload": payload,
            "digest": row["digest"],
            "token_count": int(row["token_count"]),
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "created_at": row["created_at"],
        }

    def get_context_rounds(
        self,
        session_id: str,
        after_round_id: int = 0,
    ) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, round_uid, session_id, user_content, assistant_content,
                       tool_summary, token_count, created_at
                FROM conversation_rounds
                WHERE session_id = ? AND id > ?
                ORDER BY id ASC
                """,
                (session_id, max(0, int(after_round_id))),
            ).fetchall()
            result = []
            for row in rows:
                artifact_rows = connection.execute(
                    """
                    SELECT ra.sha256, ra.sequence_index, ra.artifact_type,
                           ra.tool_name, ra.metadata_json, ca.digest,
                           ca.token_count
                    FROM round_artifacts ra
                    JOIN context_artifacts ca ON ca.sha256 = ra.sha256
                    WHERE ra.round_id = ?
                    ORDER BY ra.sequence_index ASC
                    """,
                    (row["id"],),
                ).fetchall()
                item = dict(row)
                item["artifacts"] = [
                    {
                        "sha256": artifact["sha256"],
                        "sequence_index": int(artifact["sequence_index"]),
                        "artifact_type": artifact["artifact_type"],
                        "tool_name": artifact["tool_name"],
                        "metadata": json.loads(artifact["metadata_json"] or "{}"),
                        "digest": artifact["digest"],
                        "token_count": int(artifact["token_count"]),
                    }
                    for artifact in artifact_rows
                ]
                result.append(item)
        return result

    def get_active_memory_summary(self, session_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, session_id, content, token_count,
                       covers_through_round_id, previous_summary_id,
                       artifact_refs_json, created_at
                FROM memory_summaries
                WHERE session_id = ? AND is_active = 1
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["artifact_refs"] = json.loads(result.pop("artifact_refs_json") or "[]")
        return result

    def save_memory_summary(
        self,
        session_id: str,
        content: str,
        token_count: int,
        covers_through_round_id: int,
        previous_summary_id: int | None,
        artifact_refs: list[str],
    ) -> int:
        with self._connect() as connection:
            connection.execute(
                "UPDATE memory_summaries SET is_active = 0 WHERE session_id = ?",
                (session_id,),
            )
            cursor = connection.execute(
                """
                INSERT INTO memory_summaries(
                    session_id, content, token_count, covers_through_round_id,
                    previous_summary_id, artifact_refs_json, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    session_id,
                    content,
                    max(0, int(token_count)),
                    int(covers_through_round_id),
                    previous_summary_id,
                    json.dumps(sorted(set(artifact_refs))),
                    _utc_now(),
                ),
            )
        return int(cursor.lastrowid)

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
                connection.execute("DELETE FROM round_artifacts")
                connection.execute("DELETE FROM memory_summaries")
                connection.execute("DELETE FROM conversation_rounds")
                connection.execute("DELETE FROM conversation_messages")
                connection.execute("DELETE FROM context_artifacts")
            else:
                connection.execute(
                    """
                    DELETE FROM round_artifacts
                    WHERE round_id IN (
                        SELECT id FROM conversation_rounds WHERE session_id = ?
                    )
                    """,
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM memory_summaries WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM conversation_rounds WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM conversation_messages WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    """
                    DELETE FROM context_artifacts
                    WHERE sha256 NOT IN (SELECT DISTINCT sha256 FROM round_artifacts)
                    """
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
