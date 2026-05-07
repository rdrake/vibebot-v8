"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SAFE_RE = re.compile(r"[^a-z0-9_-]")

SCHEMA_VERSION = 1
_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def db_path_for_channel(base_dir: Path, channel: str) -> Path:
    """Return the per-channel SQLite path under ``base_dir``."""
    lowered = channel.lower()
    safe = _SAFE_RE.sub("_", lowered)
    digest = hashlib.sha256(channel.encode("utf-8")).hexdigest()[:8]
    return base_dir / f"{safe}_{digest}.db"


class VerseStore:
    """Per-channel verse SQLite store. Thread-local connection + WAL +
    per-store write lock. Mirrors plugins/llm/src/llm/persistence.py:160-229,
    with an added threading.Lock to serialise writers across IRC commands and
    (later) the loom callback."""

    def __init__(self, base_dir: Path, channel: str) -> None:
        self.path = db_path_for_channel(base_dir, channel)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.execute("SELECT 1")
                return conn
            except sqlite3.ProgrammingError:
                self._local.conn = None
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._local.conn = conn
        return conn

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        yield self._connect()

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _migrate(self) -> None:
        # NOTE: executescript() implicitly commits before running; do NOT wrap
        # in write_transaction. Mirrors persistence.py:225-229.
        conn = self._connect()
        conn.executescript(_SCHEMA_SQL)
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is None:
            with self.write_transaction() as wconn:
                wconn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
