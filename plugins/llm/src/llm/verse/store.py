"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

_SAFE_RE = re.compile(r"[^a-z0-9_-]")


class Entity(NamedTuple):
    id: int
    kind: str
    name: str
    summary: str
    status: str
    created_at: float
    updated_at: float


class Relation(NamedTuple):
    id: int
    from_id: int
    to_id: int
    kind: str
    note: str


class Event(NamedTuple):
    id: int
    ts: float
    summary: str
    entity_ids: tuple[int, ...]
    source: str


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

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(self, kind: str, name: str, summary: str = "") -> int:
        """Insert a new entity and return its id."""
        now = time.time()
        with self.write_transaction() as conn:
            cur = conn.execute(
                "INSERT INTO entities (kind, name, summary, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (kind, name, summary, now, now),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def get_entity(self, entity_id: int) -> Entity | None:
        """Return the Entity with the given id, or None."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at"
                " FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
        return Entity(*row) if row else None

    def find_entity_by_name(self, name: str, kind: str | None = None) -> Entity | None:
        """Case-insensitive name lookup. Optional kind filter. Returns first match by id."""
        kind_filter = "AND kind = ?" if kind is not None else ""
        sql = (
            "SELECT id, kind, name, summary, status, created_at, updated_at"
            " FROM entities"
            f" WHERE LOWER(name) = LOWER(?)"
            f" {kind_filter}"
            " ORDER BY id ASC LIMIT 1"
        )
        params = (name, kind) if kind is not None else (name,)
        with self.read_connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return Entity(*row) if row else None

    def set_status(self, entity_id: int, status: str) -> None:
        """Update entity status and updated_at. Silent no-op if entity_id not found."""
        now = time.time()
        with self.write_transaction() as conn:
            conn.execute(
                "UPDATE entities SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, entity_id),
            )

    def list_entities_by_kind(self, kind: str, status: str | None = "active") -> list[Entity]:
        """List entities of the given kind. Filter by status unless status is None."""
        status_filter = "AND status = ?" if status is not None else ""
        sql = (
            "SELECT id, kind, name, summary, status, created_at, updated_at"
            " FROM entities"
            f" WHERE kind = ? {status_filter}"
            " ORDER BY updated_at DESC, id DESC"
        )
        params = (kind, status) if status is not None else (kind,)
        with self.read_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Entity(*row) for row in rows]

    # ------------------------------------------------------------------
    # Attribute CRUD
    # ------------------------------------------------------------------

    def set_attribute(self, entity_id: int, key: str, value: str) -> None:
        """Upsert an attribute key/value for the given entity."""
        with self.write_transaction() as conn:
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) VALUES (?, ?, ?)"
                " ON CONFLICT(entity_id, key) DO UPDATE SET value = excluded.value",
                (entity_id, key, value),
            )

    def get_attribute(self, entity_id: int, key: str) -> str | None:
        """Return the attribute value for key, or None if not set."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT value FROM attributes WHERE entity_id = ? AND key = ?",
                (entity_id, key),
            ).fetchone()
        return row[0] if row else None

    def list_attributes(self, entity_id: int) -> dict[str, str]:
        """Return all attributes for entity as a dict. Empty dict if none."""
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT key, value FROM attributes WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()
        return dict(rows)

    # ------------------------------------------------------------------
    # Relation CRUD
    # ------------------------------------------------------------------

    def add_relation(self, from_id: int, to_id: int, kind: str, note: str = "") -> int:
        """Insert a relation and return its id."""
        with self.write_transaction() as conn:
            cur = conn.execute(
                "INSERT INTO relations (from_id, to_id, kind, note) VALUES (?, ?, ?, ?)",
                (from_id, to_id, kind, note),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def list_relations(
        self,
        from_id: int | None = None,
        to_id: int | None = None,
        kind: str | None = None,
    ) -> list[Relation]:
        """Return relations matching all provided filters, ordered by id ASC."""
        clauses_params = [
            ("from_id = ?", from_id),
            ("to_id = ?", to_id),
            ("kind = ?", kind),
        ]
        active = [(c, p) for c, p in clauses_params if p is not None]
        where = (" WHERE " + " AND ".join(c for c, _ in active)) if active else ""
        params = tuple(p for _, p in active)
        sql = f"SELECT id, from_id, to_id, kind, note FROM relations{where} ORDER BY id ASC"
        with self.read_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Relation(*row) for row in rows]

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def add_event(
        self,
        summary: str,
        entity_ids: Sequence[int],
        source: str,
    ) -> int:
        """Insert an event and return its id."""
        ts = time.time()
        encoded = json.dumps(list(entity_ids))
        with self.write_transaction() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
                (ts, summary, encoded, source),
            )
            assert cur.lastrowid is not None
            return cur.lastrowid

    def recent_events(
        self,
        limit: int = 10,
        exclude_sources: Sequence[str] = (),
    ) -> list[Event]:
        """Return events newest-first, optionally excluding given sources."""
        if exclude_sources:
            placeholders = ",".join("?" * len(exclude_sources))
            sql = (
                f"SELECT id, ts, summary, entity_ids, source FROM events"
                f" WHERE source NOT IN ({placeholders})"
                f" ORDER BY ts DESC, id DESC LIMIT ?"
            )
            params: tuple = (*exclude_sources, limit)
        else:
            sql = (
                "SELECT id, ts, summary, entity_ids, source FROM events"
                " ORDER BY ts DESC, id DESC LIMIT ?"
            )
            params = (limit,)
        with self.read_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Event(
                id=row[0],
                ts=row[1],
                summary=row[2],
                entity_ids=tuple(int(x) for x in json.loads(row[3])),
                source=row[4],
            )
            for row in rows
        ]
