"""Shared crosspoll-seed queue used by the loom across all verses.

One SQLite file at ``<data_dir>/_crosspoll.db``. Thread-local connection
+ WAL + per-store write lock — same pattern as ``VerseStore``. Source
verses enqueue seeds; receiver verses pull the oldest unconsumed seed
on their next loom cycle.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, NamedTuple

_LOG = logging.getLogger("llm.verse.crosspoll_store")
_SCHEMA_VERSION = 1


class CrosspollSeed(NamedTuple):
    id: int
    source_channel: str
    summary: str
    payload: dict[str, Any]
    created_at: float


class CrosspollStore:
    """Thread-safe shared store. One instance per plugin process."""

    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "_crosspoll.db"
        self._tls = threading.local()
        self._write_lock = threading.Lock()
        self._migrate()

    def _conn(self) -> sqlite3.Connection:
        c: sqlite3.Connection | None = getattr(self._tls, "conn", None)
        if c is not None:
            try:
                c.execute("SELECT 1")
                return c
            except sqlite3.ProgrammingError:
                self._tls.conn = None
        c = sqlite3.connect(self._path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        self._tls.conn = c
        return c

    def _migrate(self) -> None:
        # NOTE: executescript() implicitly commits before running; do NOT wrap
        # in write_transaction. Mirrors VerseStore._migrate.
        sql = files("llm.verse").joinpath("crosspoll_schema.sql").read_text()
        conn = self._conn()
        conn.executescript(sql)
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            with self.write_transaction() as wconn:
                wconn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (_SCHEMA_VERSION, time.time()),
                )

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        yield self._conn()

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._conn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ----- seed queue API -----

    def enqueue_seed(
        self,
        *,
        source_channel: str,
        summary: str,
        payload: dict[str, Any],
    ) -> int:
        """Append a seed and return its id."""
        now = time.time()
        with self.write_transaction() as conn:
            cur = conn.execute(
                "INSERT INTO crosspoll_seeds "
                "(source_channel, summary, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (source_channel, summary, json.dumps(payload), now),
            )
            row_id = cur.lastrowid
            if row_id is None:
                raise RuntimeError("INSERT did not produce a lastrowid")
            return int(row_id)

    def claim_seed_for(
        self,
        dest_channel: str,
        *,
        proposal_id: str,
    ) -> CrosspollSeed | None:
        """Atomically read-and-mark the oldest unconsumed seed.

        Performs the SELECT and the consumption-row INSERT inside a
        single ``write_transaction``. If the INSERT raises
        ``sqlite3.IntegrityError`` (another caller won the race for the
        same ``(seed_id, dest_channel)`` PK), the exception propagates
        out of the context manager — triggering ROLLBACK — and is
        caught by the outer try/except, which converts it back to a
        ``None`` return. Two concurrent receivers can therefore both
        call this and exactly one will get the seed; the loser sees
        ``None``.

        Excludes seeds whose ``source_channel == dest_channel`` so a
        verse cannot consume its own emissions.
        """
        try:
            with self.write_transaction() as conn:
                row = conn.execute(
                    "SELECT id, source_channel, summary, payload, created_at "
                    "FROM crosspoll_seeds "
                    "WHERE source_channel != ? "
                    "AND id NOT IN ("
                    "  SELECT seed_id FROM crosspoll_consumptions "
                    "  WHERE dest_channel = ?"
                    ") "
                    "ORDER BY created_at ASC, id ASC LIMIT 1",
                    (dest_channel, dest_channel),
                ).fetchone()
                if row is None:
                    return None
                seed_id, src, summary, payload_json, created_at = row
                conn.execute(
                    "INSERT INTO crosspoll_consumptions "
                    "(seed_id, dest_channel, consumed_at, proposal_id) "
                    "VALUES (?, ?, ?, ?)",
                    (seed_id, dest_channel, time.time(), proposal_id),
                )
        except sqlite3.IntegrityError:
            # Lost the race; ROLLBACK already happened in the
            # contextmanager's except branch.
            return None
        return CrosspollSeed(
            id=seed_id,
            source_channel=src,
            summary=summary,
            payload=json.loads(payload_json),
            created_at=created_at,
        )

    def release_claim(self, seed_id: int, dest_channel: str) -> bool:
        """Delete the consumption row for ``(seed_id, dest_channel)``.

        Used by the receiver consume hook when the local proposal insert
        fails after a successful claim — without this the consumption
        row is permanent and that seed is lost for the receiver forever.
        Returns True if a row was deleted; idempotent (missing row
        returns False without raising).
        """
        with self.write_transaction() as conn:
            cur = conn.execute(
                "DELETE FROM crosspoll_consumptions WHERE seed_id=? AND dest_channel=?",
                (seed_id, dest_channel),
            )
            return cur.rowcount > 0

    def next_unconsumed_for(self, dest_channel: str) -> CrosspollSeed | None:
        """Diagnostic-only: oldest seed not yet consumed by
        ``dest_channel``. Does **not** mark consumed — use
        ``claim_seed_for`` for the consume flow.
        """
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT id, source_channel, summary, payload, created_at "
                "FROM crosspoll_seeds "
                "WHERE source_channel != ? "
                "AND id NOT IN ("
                "  SELECT seed_id FROM crosspoll_consumptions "
                "  WHERE dest_channel = ?"
                ") "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                (dest_channel, dest_channel),
            ).fetchone()
        if row is None:
            return None
        return CrosspollSeed(
            id=row[0],
            source_channel=row[1],
            summary=row[2],
            payload=json.loads(row[3]),
            created_at=row[4],
        )

    def pending_count_for(self, dest_channel: str) -> int:
        """Count of seeds the destination has not yet consumed.

        Diagnostic only; not used in the cycle itself.
        """
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM crosspoll_seeds "
                "WHERE source_channel != ? AND id NOT IN ("
                "  SELECT seed_id FROM crosspoll_consumptions "
                "  WHERE dest_channel = ?"
                ")",
                (dest_channel, dest_channel),
            ).fetchone()
        return int(row[0])
