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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_uid TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    event_type TEXT NOT NULL,
                    document_id INTEGER,
                    source_name TEXT,
                    occurred_at TEXT NOT NULL,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    received_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_events_user_time
                ON user_events(user_id, occurred_at DESC, id DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_events_paper
                ON user_events(user_id, document_id, source_name, occurred_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_paper_stats (
                    user_id TEXT NOT NULL,
                    paper_key TEXT NOT NULL,
                    document_id INTEGER,
                    source_name TEXT,
                    open_count INTEGER NOT NULL DEFAULT 0,
                    preview_open_count INTEGER NOT NULL DEFAULT 0,
                    pdf_open_count INTEGER NOT NULL DEFAULT 0,
                    evidence_click_count INTEGER NOT NULL DEFAULT 0,
                    scope_add_count INTEGER NOT NULL DEFAULT 0,
                    question_count INTEGER NOT NULL DEFAULT 0,
                    effective_read_seconds REAL NOT NULL DEFAULT 0,
                    last_opened_at TEXT,
                    last_read_at TEXT,
                    last_event_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, paper_key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_paper_stats_recent
                ON user_paper_stats(user_id, last_event_at DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_knowledge_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(user_id, normalized_query, reason)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_gaps_active
                ON user_knowledge_gaps(user_id, resolved, last_seen_at DESC)
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

    def get_recent_conversation_rounds(
        self,
        session_id: str,
        limit: int = 3,
    ) -> list[dict]:
        """Return recent complete turns with structured artifact metadata.

        The source router uses this instead of guessing paper identity from the
        assistant prose. Results are returned oldest-to-newest so relative turn
        indexes remain stable in the routing prompt.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, round_uid, session_id, user_content, assistant_content,
                       tool_summary, token_count, created_at
                FROM (
                    SELECT id, round_uid, session_id, user_content, assistant_content,
                           tool_summary, token_count, created_at
                    FROM conversation_rounds
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, max(1, int(limit))),
            ).fetchall()
            result = []
            for row in rows:
                artifact_rows = connection.execute(
                    """
                    SELECT ra.artifact_type, ra.tool_name, ra.metadata_json, ca.digest
                    FROM round_artifacts ra
                    JOIN context_artifacts ca ON ca.sha256 = ra.sha256
                    WHERE ra.round_id = ?
                    ORDER BY ra.sequence_index ASC
                    """,
                    (row["id"],),
                ).fetchall()
                sources = []
                for artifact in artifact_rows:
                    metadata = json.loads(artifact["metadata_json"] or "{}")
                    for source in metadata.get("sources") or []:
                        source = str(source)
                        if source and source not in sources:
                            sources.append(source)
                item = dict(row)
                item["retrieved_sources"] = sources
                item["artifact_digests"] = [
                    str(artifact["digest"] or "")
                    for artifact in artifact_rows
                    if artifact["digest"]
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

    def record_user_events(self, events: list[dict]) -> int:
        """Append idempotent frontend events and update rebuildable user aggregates."""
        inserted = 0
        with self._connect() as connection:
            for event in events:
                user_id = str(event.get("user_id") or "").strip()[:128]
                event_type = str(event.get("event_type") or "").strip()[:64]
                if not user_id or not event_type:
                    continue
                occurred_at = str(event.get("occurred_at") or _utc_now())[:64]
                properties = event.get("properties") or {}
                if not isinstance(properties, dict):
                    properties = {"value": properties}
                document_id = event.get("document_id")
                try:
                    document_id = int(document_id) if document_id is not None else None
                except (TypeError, ValueError):
                    document_id = None
                source_name = str(event.get("source_name") or "").strip()[:500] or None
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO user_events(
                        event_uid, user_id, session_id, event_type, document_id,
                        source_name, occurred_at, properties_json, received_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(event.get("event_uid") or uuid.uuid4().hex)[:128],
                        user_id,
                        str(event.get("session_id") or "")[:128] or None,
                        event_type,
                        document_id,
                        source_name,
                        occurred_at,
                        json.dumps(properties, ensure_ascii=False, default=str),
                        _utc_now(),
                    ),
                )
                if cursor.rowcount == 0:
                    continue
                inserted += 1
                self._update_user_paper_stats(
                    connection,
                    user_id=user_id,
                    event_type=event_type,
                    document_id=document_id,
                    source_name=source_name,
                    occurred_at=occurred_at,
                    properties=properties,
                )
                self._update_knowledge_gaps(
                    connection,
                    user_id=user_id,
                    event_type=event_type,
                    occurred_at=occurred_at,
                    properties=properties,
                )
        return inserted

    @classmethod
    def _update_user_paper_stats(
        cls,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        event_type: str,
        document_id: int | None,
        source_name: str | None,
        occurred_at: str,
        properties: dict,
    ) -> None:
        papers: list[tuple[int | None, str | None]] = []
        if document_id is not None or source_name:
            papers.append((document_id, source_name))
        if event_type == "question_submitted":
            for item in properties.get("papers", []):
                if isinstance(item, dict):
                    item_id = item.get("document_id")
                    try:
                        item_id = int(item_id) if item_id is not None else None
                    except (TypeError, ValueError):
                        item_id = None
                    item_source = str(item.get("source_name") or "").strip()[:500] or None
                    candidate = (item_id, item_source)
                    if candidate not in papers and (item_id is not None or item_source):
                        papers.append(candidate)

        for paper_document_id, paper_source_name in papers:
            paper_key = (
                f"source:{paper_source_name}"
                if paper_source_name
                else f"doc:{paper_document_id}"
            )
            open_count = int(event_type in {"paper_preview_open", "pdf_open"})
            preview_count = int(event_type == "paper_preview_open")
            pdf_count = int(event_type == "pdf_open")
            evidence_count = int(event_type == "evidence_clicked")
            scope_count = int(event_type == "paper_scope_added")
            question_count = int(event_type == "question_submitted")
            duration = 0.0
            if event_type == "paper_preview_close":
                try:
                    duration = min(21600.0, max(0.0, float(properties.get("duration_seconds", 0))))
                except (TypeError, ValueError):
                    duration = 0.0
            opened_at = occurred_at if open_count else None
            read_at = occurred_at if duration > 0 else None
            connection.execute(
                """
                INSERT INTO user_paper_stats(
                    user_id, paper_key, document_id, source_name, open_count,
                    preview_open_count, pdf_open_count, evidence_click_count,
                    scope_add_count, question_count, effective_read_seconds,
                    last_opened_at, last_read_at, last_event_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, paper_key) DO UPDATE SET
                    document_id = COALESCE(excluded.document_id, user_paper_stats.document_id),
                    source_name = COALESCE(excluded.source_name, user_paper_stats.source_name),
                    open_count = user_paper_stats.open_count + excluded.open_count,
                    preview_open_count = user_paper_stats.preview_open_count + excluded.preview_open_count,
                    pdf_open_count = user_paper_stats.pdf_open_count + excluded.pdf_open_count,
                    evidence_click_count = user_paper_stats.evidence_click_count + excluded.evidence_click_count,
                    scope_add_count = user_paper_stats.scope_add_count + excluded.scope_add_count,
                    question_count = user_paper_stats.question_count + excluded.question_count,
                    effective_read_seconds = user_paper_stats.effective_read_seconds + excluded.effective_read_seconds,
                    last_opened_at = CASE
                        WHEN excluded.last_opened_at IS NOT NULL AND (
                            user_paper_stats.last_opened_at IS NULL OR
                            excluded.last_opened_at > user_paper_stats.last_opened_at
                        ) THEN excluded.last_opened_at ELSE user_paper_stats.last_opened_at END,
                    last_read_at = CASE
                        WHEN excluded.last_read_at IS NOT NULL AND (
                            user_paper_stats.last_read_at IS NULL OR
                            excluded.last_read_at > user_paper_stats.last_read_at
                        ) THEN excluded.last_read_at ELSE user_paper_stats.last_read_at END,
                    last_event_at = CASE
                        WHEN excluded.last_event_at > user_paper_stats.last_event_at
                        THEN excluded.last_event_at ELSE user_paper_stats.last_event_at END
                """,
                (
                    user_id,
                    paper_key,
                    paper_document_id,
                    paper_source_name,
                    open_count,
                    preview_count,
                    pdf_count,
                    evidence_count,
                    scope_count,
                    question_count,
                    duration,
                    opened_at,
                    read_at,
                    occurred_at,
                ),
            )

    @staticmethod
    def _update_knowledge_gaps(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        event_type: str,
        occurred_at: str,
        properties: dict,
    ) -> None:
        query = str(properties.get("query") or properties.get("question") or "").strip()[:2000]
        normalized = " ".join(query.casefold().split())[:500]
        if not normalized:
            return
        reason = None
        if event_type == "search_completed" and int(properties.get("result_count") or 0) == 0:
            reason = "search_no_results"
        elif (
            event_type == "answer_completed"
            and int(properties.get("evidence_count") or 0) == 0
            and not bool(properties.get("use_agent"))
        ):
            reason = "insufficient_evidence"
        elif event_type == "answer_failed":
            reason = "answer_failed"

        if reason:
            connection.execute(
                """
                INSERT INTO user_knowledge_gaps(
                    user_id, normalized_query, query_text, reason,
                    occurrence_count, resolved, first_seen_at, last_seen_at,
                    metadata_json
                ) VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?)
                ON CONFLICT(user_id, normalized_query, reason) DO UPDATE SET
                    query_text = excluded.query_text,
                    occurrence_count = user_knowledge_gaps.occurrence_count + 1,
                    resolved = 0,
                    last_seen_at = excluded.last_seen_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    user_id,
                    normalized,
                    query,
                    reason,
                    occurred_at,
                    occurred_at,
                    json.dumps(properties, ensure_ascii=False, default=str),
                ),
            )
        elif event_type == "answer_completed" and int(properties.get("evidence_count") or 0) > 0:
            connection.execute(
                """
                UPDATE user_knowledge_gaps
                SET resolved = 1, last_seen_at = ?
                WHERE user_id = ? AND normalized_query = ?
                """,
                (occurred_at, user_id, normalized),
            )

    def get_user_profile(self, user_id: str, limit: int = 20) -> dict:
        user_id = str(user_id).strip()[:128]
        limit = min(100, max(1, int(limit)))
        with self._connect() as connection:
            paper_rows = connection.execute(
                """
                SELECT user_id, paper_key, document_id, source_name, open_count,
                       preview_open_count, pdf_open_count, evidence_click_count,
                       scope_add_count, question_count, effective_read_seconds,
                       last_opened_at, last_read_at, last_event_at
                FROM user_paper_stats
                WHERE user_id = ?
                ORDER BY (
                    open_count * 2 + pdf_open_count * 3 +
                    evidence_click_count * 4 + scope_add_count * 3 +
                    question_count * 2 + MIN(effective_read_seconds / 60.0, 30)
                ) DESC, last_event_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            gap_rows = connection.execute(
                """
                SELECT id, query_text, reason, occurrence_count, first_seen_at,
                       last_seen_at, metadata_json
                FROM user_knowledge_gaps
                WHERE user_id = ? AND resolved = 0
                ORDER BY occurrence_count DESC, last_seen_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT event_uid, session_id, event_type, document_id, source_name,
                       occurred_at, properties_json
                FROM user_events
                WHERE user_id = ?
                ORDER BY occurred_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()

        papers = []
        for row in paper_rows:
            item = dict(row)
            item["effective_read_seconds"] = round(float(item["effective_read_seconds"]), 1)
            item["interest_score"] = round(
                item["open_count"] * 2
                + item["pdf_open_count"] * 3
                + item["evidence_click_count"] * 4
                + item["scope_add_count"] * 3
                + item["question_count"] * 2
                + min(item["effective_read_seconds"] / 60.0, 30),
                2,
            )
            papers.append(item)
        gaps = []
        for row in gap_rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            gaps.append(item)
        recent_events = []
        for row in event_rows:
            item = dict(row)
            item["properties"] = json.loads(item.pop("properties_json") or "{}")
            recent_events.append(item)
        return {
            "user_id": user_id,
            "paper_stats": papers,
            "knowledge_gaps": gaps,
            "recent_events": recent_events,
        }

    def build_user_profile_context(self, user_id: str) -> str:
        profile = self.get_user_profile(user_id, limit=8)
        papers = profile["paper_stats"]
        gaps = profile["knowledge_gaps"]
        if not papers and not gaps:
            return ""
        lines = [
            "[USER RESEARCH PROFILE — behavioral signals, not proof of mastery]",
            "Use this only for personalization. Never claim the user understands a paper merely because it was opened.",
        ]
        if papers:
            lines.append("Likely interests/recent working papers:")
            for paper in papers[:5]:
                label = paper.get("source_name") or paper["paper_key"]
                lines.append(
                    f"- {label}: opens={paper['open_count']}, "
                    f"effective_read_minutes={paper['effective_read_seconds'] / 60:.1f}, "
                    f"evidence_clicks={paper['evidence_click_count']}, "
                    f"questions={paper['question_count']}, last={paper['last_event_at']}"
                )
        if gaps:
            lines.append("Potential unresolved knowledge needs:")
            for gap in gaps[:5]:
                lines.append(
                    f"- {gap['query_text']} (reason={gap['reason']}, repeats={gap['occurrence_count']})"
                )
        return "\n".join(lines)

    def clear_user_profile(self, user_id: str) -> None:
        user_id = str(user_id).strip()[:128]
        with self._connect() as connection:
            connection.execute("DELETE FROM user_events WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM user_paper_stats WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM user_knowledge_gaps WHERE user_id = ?", (user_id,))

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
