"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

_SAFE_RE = re.compile(r"[^a-z0-9_-]")


class Entity(NamedTuple):
    id: int
    kind: str
    name: str
    summary: str
    status: str
    created_at: float
    updated_at: float


class AvatarOptInResult(NamedTuple):
    entity_id: int
    place_name: str
    scene_text: str
    was_already_opted_in: bool


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


class Proposal(NamedTuple):
    id: str
    created_at: float
    cycle_id: str
    op: str
    payload: dict[str, Any]
    confidence: float
    provenance: str
    status: str
    reviewer: str | None
    reviewed_at: float | None


_VALID_PROPOSAL_STATUSES = ("pending", "approved", "rejected")


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

    # ------------------------------------------------------------------
    # Avatar link CRUD
    # ------------------------------------------------------------------

    def link_avatar(self, entity_id: int, nick: str, account: str | None = None) -> None:
        """Upsert an avatar_link row for entity_id."""
        with self.write_transaction() as conn:
            conn.execute(
                "INSERT INTO avatar_link (entity_id, nick, account) VALUES (?, ?, ?)"
                " ON CONFLICT(entity_id) DO UPDATE SET nick = excluded.nick,"
                " account = excluded.account",
                (entity_id, nick, account),
            )

    def find_avatar_by_nick(self, nick: str) -> int | None:
        """Case-insensitive nick lookup. Returns entity_id or None."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_id FROM avatar_link WHERE LOWER(nick) = LOWER(?)",
                (nick,),
            ).fetchone()
        return int(row[0]) if row else None

    def find_avatar_by_account(self, account: str) -> int | None:
        """Case-sensitive account lookup. Returns entity_id or None."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_id FROM avatar_link WHERE account = ?",
                (account,),
            ).fetchone()
        return int(row[0]) if row else None

    def unlink_avatar(self, entity_id: int) -> None:
        """Remove avatar link and retire the entity atomically."""
        now = time.time()
        with self.write_transaction() as conn:
            conn.execute(
                "DELETE FROM avatar_link WHERE entity_id = ?",
                (entity_id,),
            )
            conn.execute(
                "UPDATE entities SET status = 'retired', updated_at = ? WHERE id = ?",
                (now, entity_id),
            )

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

    # ------------------------------------------------------------------
    # Avatar opt-in
    # ------------------------------------------------------------------

    def opt_in_avatar(
        self,
        nick: str,
        account: str | None,
        instruct_text: str,
    ) -> AvatarOptInResult:
        """Opt a user into the verse, creating or reactivating their avatar.

        All DB work happens inside a single write_transaction to avoid
        re-entrant lock acquisition (threading.Lock is not reentrant).
        """
        now = time.time()

        with self.write_transaction() as conn:
            # ----------------------------------------------------------
            # 1. Find existing avatar: prefer account lookup, fall back to nick.
            # ----------------------------------------------------------
            entity_id: int | None = None
            entity_status: str | None = None

            if account is not None:
                row = conn.execute(
                    "SELECT al.entity_id, e.status"
                    " FROM avatar_link al"
                    " JOIN entities e ON e.id = al.entity_id"
                    " WHERE al.account = ?",
                    (account,),
                ).fetchone()
                if row:
                    entity_id, entity_status = int(row[0]), row[1]

            if entity_id is None:
                row = conn.execute(
                    "SELECT al.entity_id, e.status"
                    " FROM avatar_link al"
                    " JOIN entities e ON e.id = al.entity_id"
                    " WHERE LOWER(al.nick) = LOWER(?)",
                    (nick,),
                ).fetchone()
                if row:
                    entity_id, entity_status = int(row[0]), row[1]

            # ----------------------------------------------------------
            # 2. Determine avatar state and act accordingly.
            # ----------------------------------------------------------
            was_already_opted_in = False

            if entity_id is not None and entity_status == "active":
                # Active: return existing. Update nick in link in case IRC renamed.
                was_already_opted_in = True
                conn.execute(
                    "UPDATE avatar_link SET nick = ?, account = ? WHERE entity_id = ?",
                    (nick, account, entity_id),
                )

            elif entity_id is not None and entity_status == "retired":
                # Retired (soft pause): reactivate and update the link.
                conn.execute(
                    "UPDATE entities SET status = 'active', updated_at = ? WHERE id = ?",
                    (now, entity_id),
                )
                conn.execute(
                    "UPDATE avatar_link SET nick = ?, account = ? WHERE entity_id = ?",
                    (nick, account, entity_id),
                )

            else:
                # New avatar.
                cur = conn.execute(
                    "INSERT INTO entities (kind, name, summary, status, created_at, updated_at)"
                    " VALUES ('avatar', ?, ?, 'active', ?, ?)",
                    (nick, instruct_text, now, now),
                )
                assert cur.lastrowid is not None
                entity_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO avatar_link (entity_id, nick, account) VALUES (?, ?, ?)"
                    " ON CONFLICT(entity_id) DO UPDATE SET nick = excluded.nick,"
                    " account = excluded.account",
                    (entity_id, nick, account),
                )

            # ----------------------------------------------------------
            # 3. Pick or create the starter place.
            #    Pull all active places in Python — small count in early-PR1
            #    use, avoids json_each complexity, and keeps SQL surface small.
            # ----------------------------------------------------------
            place_rows = conn.execute(
                "SELECT id, name, summary, updated_at FROM entities"
                " WHERE kind = 'place' AND status = 'active'",
            ).fetchall()

            if not place_rows:
                # Create default clearing.
                cur2 = conn.execute(
                    "INSERT INTO entities (kind, name, summary, status, created_at, updated_at)"
                    " VALUES ('place', 'The Clearing',"
                    " 'A quiet woodland clearing where new stories begin.',"
                    " 'active', ?, ?)",
                    (now, now),
                )
                assert cur2.lastrowid is not None
                place_name = "The Clearing"
                place_summary = "A quiet woodland clearing where new stories begin."
            else:
                # For each active place, find the max event ts that references it.
                # Fetch event rows once; iterate in Python to find best place.
                event_rows = conn.execute(
                    "SELECT entity_ids, ts FROM events ORDER BY ts DESC",
                ).fetchall()

                # Build map: place_id -> latest_event_ts
                place_ids = {row[0] for row in place_rows}
                latest_ts: dict[int, float] = dict.fromkeys(place_ids, 0.0)
                for ev_entity_ids_json, ev_ts in event_rows:
                    try:
                        ev_entity_ids = json.loads(ev_entity_ids_json)
                    except (ValueError, TypeError):
                        continue
                    for eid_val in ev_entity_ids:
                        pid = int(eid_val)
                        if pid in latest_ts and ev_ts > latest_ts[pid]:
                            latest_ts[pid] = ev_ts

                # Sort: latest_event_ts DESC, updated_at DESC, id DESC
                best = sorted(
                    place_rows,
                    key=lambda r: (latest_ts[r[0]], r[3], r[0]),
                    reverse=True,
                )[0]
                place_name = best[1]
                place_summary = best[2]

            # ----------------------------------------------------------
            # 4. Upsert avatar's location attribute.
            # ----------------------------------------------------------
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) VALUES (?, 'location', ?)"
                " ON CONFLICT(entity_id, key) DO UPDATE SET value = excluded.value",
                (entity_id, place_name),
            )

            # ----------------------------------------------------------
            # 5. Build scene text.
            # ----------------------------------------------------------
            scene_text = (
                f"You step into {place_name}. {place_summary}"
                " Use @look to inspect things or @ask … to act."
            )

        return AvatarOptInResult(
            entity_id=entity_id,
            place_name=place_name,
            scene_text=scene_text,
            was_already_opted_in=was_already_opted_in,
        )

    # ------------------------------------------------------------------
    # Proposals CRUD
    # ------------------------------------------------------------------

    def add_proposal(
        self,
        *,
        cycle_id: str,
        op: str,
        payload: dict[str, Any],
        confidence: float,
        provenance: str = "",
        status: str = "pending",
        reviewer: str | None = None,
    ) -> str:
        """Insert a proposal and return its uuid id.

        When *status* is 'approved' or 'rejected', *reviewer* must be
        supplied and reviewed_at is set to now (this is how auto-apply
        records its audit row inside the same write_transaction as the
        mutation it just applied).
        """
        if status not in _VALID_PROPOSAL_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        if status != "pending" and not reviewer:
            raise ValueError("reviewer required when status != pending")
        pid = uuid.uuid4().hex
        now = time.time()
        reviewed_at = now if status != "pending" else None
        with self.write_transaction() as conn:
            conn.execute(
                "INSERT INTO proposals "
                "(id, created_at, cycle_id, op, payload, confidence, provenance, "
                " status, reviewer, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pid,
                    now,
                    cycle_id,
                    op,
                    json.dumps(payload),
                    confidence,
                    provenance,
                    status,
                    reviewer,
                    reviewed_at,
                ),
            )
        return pid

    def get_proposal(self, proposal_id: str) -> Proposal | None:
        """Return the Proposal with *proposal_id*, or None."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT id, created_at, cycle_id, op, payload, confidence, "
                "provenance, status, reviewer, reviewed_at "
                "FROM proposals WHERE id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            return None
        return Proposal(
            id=row[0],
            created_at=row[1],
            cycle_id=row[2],
            op=row[3],
            payload=json.loads(row[4]),
            confidence=row[5],
            provenance=row[6],
            status=row[7],
            reviewer=row[8],
            reviewed_at=row[9],
        )

    def list_proposals(
        self,
        *,
        status: str | None = None,
        cycle_id: str | None = None,
        limit: int = 100,
    ) -> list[Proposal]:
        """Return proposals newest-first, optionally filtered by status/cycle."""
        sql = (
            "SELECT id, created_at, cycle_id, op, payload, confidence, "
            "provenance, status, reviewer, reviewed_at FROM proposals"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.read_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            Proposal(
                id=r[0],
                created_at=r[1],
                cycle_id=r[2],
                op=r[3],
                payload=json.loads(r[4]),
                confidence=r[5],
                provenance=r[6],
                status=r[7],
                reviewer=r[8],
                reviewed_at=r[9],
            )
            for r in rows
        ]
