"""SQLite-backed metadata storage for indexed documents and chunks."""

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from src.utils.pdf_parser import Document
from src.utils.pdf_parser import normalize_title


class SQLiteDocumentStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    paper_title TEXT,
                    title_normalized TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    metadata_json TEXT NOT NULL,
                    vector_index INTEGER NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS index_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raw_chunk_id TEXT NOT NULL UNIQUE,
                    chunk_type TEXT NOT NULL DEFAULT 'table',
                    section_title TEXT,
                    caption TEXT,
                    raw_content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_page ON chunks(page, chunk_index)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_raw_chunks_id ON raw_chunks(raw_chunk_id)"
            )
            self._ensure_column(connection, "documents", "paper_title", "TEXT")
            self._ensure_column(connection, "documents", "title_normalized", "TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_title_normalized ON documents(title_normalized)"
            )

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table: str,
        column: str,
        column_type: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def has_chunks(self) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
            return bool(row and row["count"] > 0)

    def replace_all_documents(self, documents: Iterable[Document], reason: str = "save") -> None:
        docs = list(documents)
        with self._connect() as connection:
            connection.execute("BEGIN")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")

            source_to_document_id: dict[str, int] = {}
            for vector_index, doc in enumerate(docs):
                source_name = str(doc.metadata.get("source") or "unknown")
                paper_title = str(doc.metadata.get("paper_title") or Path(source_name).stem)
                title_normalized = normalize_title(paper_title)
                document_id = source_to_document_id.get(source_name)
                if document_id is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO documents (
                            source_name,
                            paper_title,
                            title_normalized,
                            status
                        )
                        VALUES (?, ?, ?, 'active')
                        """,
                        (source_name, paper_title, title_normalized),
                    )
                    document_id = int(cursor.lastrowid)
                    source_to_document_id[source_name] = document_id

                metadata_json = json.dumps(doc.metadata or {}, ensure_ascii=False)
                connection.execute(
                    """
                    INSERT INTO chunks (
                        document_id,
                        chunk_id,
                        content,
                        page,
                        chunk_index,
                        metadata_json,
                        vector_index
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        doc.chunk_id,
                        doc.content,
                        doc.metadata.get("page"),
                        doc.metadata.get("chunk_index"),
                        metadata_json,
                        vector_index,
                    ),
                )

            next_version = self.current_version(connection) + 1
            connection.execute(
                "INSERT INTO index_versions (version, reason) VALUES (?, ?)",
                (next_version, reason),
            )
            connection.commit()

    def load_documents(self) -> List[Document]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, content, metadata_json, vector_index
                FROM chunks
                ORDER BY vector_index ASC
                """
            ).fetchall()

        return [
            Document(
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                chunk_id=row["chunk_id"],
            )
            for row in rows
        ]

    def current_version(self, connection: sqlite3.Connection | None = None) -> int:
        owns_connection = connection is None
        if connection is None:
            connection = self._connect()
        try:
            row = connection.execute(
                "SELECT version FROM index_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return int(row["version"]) if row else 0
        finally:
            if owns_connection:
                connection.close()

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    d.id,
                    d.source_name,
                    d.paper_title,
                    d.title_normalized,
                    d.status,
                    d.created_at,
                    d.updated_at,
                    COUNT(c.id) AS chunk_count,
                    MIN(c.page) AS first_page,
                    MAX(c.page) AS last_page
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_document(self, document_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    d.id,
                    d.source_name,
                    d.paper_title,
                    d.title_normalized,
                    d.status,
                    d.created_at,
                    d.updated_at,
                    COUNT(c.id) AS chunk_count,
                    MIN(c.page) AS first_page,
                    MAX(c.page) AS last_page
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = ?
                GROUP BY d.id
                """,
                (document_id,),
            ).fetchone()

        return dict(row) if row else None

    def update_document_title(self, document_id: int, paper_title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET paper_title = ?, title_normalized = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (paper_title, normalize_title(paper_title), document_id),
            )

    def save_raw_chunk(
        self,
        raw_chunk_id: str,
        raw_content: str,
        chunk_type: str = "table",
        section_title: str = "",
        caption: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO raw_chunks (raw_chunk_id, chunk_type, section_title, caption, raw_content)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(raw_chunk_id) DO UPDATE SET
                    raw_content = excluded.raw_content,
                    section_title = excluded.section_title,
                    caption = excluded.caption
                """,
                (raw_chunk_id, chunk_type, section_title or "", caption or "", raw_content),
            )

    def get_raw_chunk(self, raw_chunk_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT raw_chunk_id, chunk_type, section_title, caption, raw_content "
                "FROM raw_chunks WHERE raw_chunk_id = ?",
                (raw_chunk_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_raw_chunks_by_prefix(self, prefix: str) -> None:
        """Remove all raw chunks whose id starts with prefix (used when re-indexing a paper)."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM raw_chunks WHERE raw_chunk_id LIKE ?",
                (prefix + "%",),
            )
