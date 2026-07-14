"""Persistent SQLite-backed TTL caches for local single-node deployments."""

import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Generic, Optional, TypeVar

import numpy as np

from src.utils.cache import CacheStats


K = TypeVar("K")
V = TypeVar("V")


class SQLiteTTLCache(Generic[K, V]):
    """A bounded persistent TTL cache namespaced inside one SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        namespace: str,
        max_size: int = 4096,
        ttl_seconds: int = 1800,
        value_codec: str = "json",
    ):
        if value_codec not in {"json", "ndarray"}:
            raise ValueError("value_codec must be 'json' or 'ndarray'")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.value_codec = value_codec
        self.stats = CacheStats()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, cache_key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_expiry
                ON cache_entries(namespace, expires_at)
                """
            )
            connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND expires_at <= ?",
                (self.namespace, time.time()),
            )

    def _encode(self, value: V) -> bytes:
        if self.value_codec == "ndarray":
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(value), allow_pickle=False)
            return buffer.getvalue()
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def _decode(self, raw: bytes) -> V:
        if self.value_codec == "ndarray":
            return np.load(io.BytesIO(raw), allow_pickle=False)  # type: ignore[return-value]
        return json.loads(raw.decode("utf-8"))  # type: ignore[return-value]

    def get(self, key: K) -> Optional[V]:
        now = time.time()
        cache_key = str(key)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT value, expires_at
                FROM cache_entries
                WHERE namespace = ? AND cache_key = ?
                """,
                (self.namespace, cache_key),
            ).fetchone()
            if row is None:
                self.stats.miss += 1
                return None
            if float(row[1]) <= now:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                    (self.namespace, cache_key),
                )
                self.stats.miss += 1
                self.stats.expired += 1
                return None
            connection.execute(
                """
                UPDATE cache_entries SET updated_at = ?
                WHERE namespace = ? AND cache_key = ?
                """,
                (now, self.namespace, cache_key),
            )

        try:
            value = self._decode(row[0])
        except (ValueError, TypeError, json.JSONDecodeError, OSError, EOFError):
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                    (self.namespace, cache_key),
                )
            self.stats.miss += 1
            return None

        self.stats.hit += 1
        return value

    def set(self, key: K, value: V) -> None:
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries(
                    namespace, cache_key, value, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at
                """,
                (
                    self.namespace,
                    str(key),
                    self._encode(value),
                    now + self.ttl_seconds,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND expires_at <= ?",
                (self.namespace, now),
            )
            row = connection.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
            excess = max(0, int(row[0]) - self.max_size)
            if excess:
                connection.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE rowid IN (
                        SELECT rowid FROM cache_entries
                        WHERE namespace = ?
                        ORDER BY updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (self.namespace, excess),
                )
                self.stats.evicted += excess

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM cache_entries WHERE namespace = ?",
                (self.namespace,),
            )
