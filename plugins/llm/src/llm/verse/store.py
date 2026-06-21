"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

_log = logging.getLogger(__name__)

_SAFE_RE = re.compile(r"[^a-z0-9_-]")

# Attribute keys that drive entity lifecycle/identity. Proposals (loom / model
# output) must never set these — they are maintained only by the engine's own
# inline writers (bump_last_seen_ts, aging, compaction heartbeat, verse_move).
# Letting a model-proposed set_attribute write them would grant NPC immortality
# (last_seen_ts), toggle aging enrollment (auto_created), or relocate/retype an
# entity outside the guarded paths (location/status/kind).
_RESERVED_ATTRIBUTE_KEYS = frozenset(
    {"last_seen_ts", "auto_created", "status", "kind", "location", "pinned", "author_locked"}
)

_VALID_SOURCES = frozenset({"operator", "loom", "llm", "crosspoll", "avatar"})
_DESTRUCTIVE_OPS = frozenset({"delete_event", "delete_relation", "set_status", "set_pinned"})
_MATCH_STOPLIST = frozenset({"the", "and", "you", "him", "her", "they", "will", "are", "was"})


def _parse_entity_ids(raw: str, event_id: object) -> tuple[int, ...]:
    """Decode an event's stored entity_ids JSON, degrading to () on corruption.

    A single malformed row must not crash a whole recent_events() read (manual
    DB edits, partial writes, or older bad data could otherwise take the verse
    down). Bad rows are logged and treated as having no entities.
    """
    try:
        return tuple(int(x) for x in json.loads(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        _log.warning("event %s has invalid entity_ids %r; treating as empty", event_id, raw)
        return ()


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


class RelationView(NamedTuple):
    from_name: str
    to_name: str
    kind: str
    note: str


_VALID_PROPOSAL_STATUSES = ("pending", "approved", "rejected")


SCHEMA_VERSION = 3
_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


def db_path_for_channel(base_dir: Path, channel: str) -> Path:
    """Return the per-channel SQLite path under ``base_dir``."""
    lowered = channel.lower()
    safe = _SAFE_RE.sub("_", lowered)
    digest = hashlib.sha256(channel.encode("utf-8")).hexdigest()[:8]
    return base_dir / f"{safe}_{digest}.db"


def list_active_verses(base_dir: Path) -> list[Path]:
    """Return paths of all verse DB files in *base_dir*, sorted.

    The caller maps these back to channel names via the same
    db_path_for_channel sanitizer used at construction time.
    """
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob("*.db"))


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
        existing = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = existing[0] if existing and existing[0] is not None else None
        if current is None:
            with self.write_transaction() as wconn:
                wconn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
            return
        if current < 2:
            self._upgrade_v1_to_v2()
        if current < 3:
            self._upgrade_v2_to_v3()

    def _upgrade_v1_to_v2(self) -> None:
        """Rebuild events + proposals with widened CHECK constraints.

        SQLite cannot ALTER ... DROP CONSTRAINT, so use the 12-step
        table-rebuild: create _new with the v2 CHECK, copy rows, drop old,
        rename. Gated on schema_version < 2 by the caller; stamps version 2.
        """
        with self.write_transaction() as conn:
            conn.execute(
                "CREATE TABLE events_new ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, summary TEXT NOT NULL,"
                " entity_ids TEXT NOT NULL DEFAULT '[]',"
                " source TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll','operator','llm')))"
            )
            conn.execute(
                "INSERT INTO events_new (id, ts, summary, entity_ids, source) "
                "SELECT id, ts, summary, entity_ids, source FROM events"
            )
            conn.execute("DROP TABLE events")
            conn.execute("ALTER TABLE events_new RENAME TO events")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)")

            conn.execute(
                "CREATE TABLE proposals_new ("
                " id TEXT PRIMARY KEY, created_at REAL NOT NULL, cycle_id TEXT NOT NULL,"
                " op TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation',"
                "  'add_entity','crosspoll_seed','update_entity','set_status','edit_event',"
                "  'delete_event','delete_relation','set_pinned')),"
                " payload TEXT NOT NULL, confidence REAL NOT NULL, provenance TEXT NOT NULL DEFAULT '',"
                " status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),"
                " reviewer TEXT, reviewed_at REAL)"
            )
            conn.execute(
                "INSERT INTO proposals_new SELECT id, created_at, cycle_id, op, payload, "
                "confidence, provenance, status, reviewer, reviewed_at FROM proposals"
            )
            conn.execute("DROP TABLE proposals")
            conn.execute("ALTER TABLE proposals_new RENAME TO proposals")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at)"
            )
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (2, ?)", (time.time(),)
            )

    def _upgrade_v2_to_v3(self) -> None:
        """Additive: entity_alias + event_actor (created via schema.sql executescript
        on open). Backfill event_actor from the legacy events.entity_ids JSON blob,
        ELEMENT-WISE tolerant (keep valid existing ids, drop bad elements — never the
        all-or-nothing _parse_entity_ids). Idempotent via INSERT OR IGNORE on the PK.
        NOTE: the v1->v2 rebuild DROP TABLE events cascade-empties event_actor under
        foreign_keys=ON, but this backfill runs LAST so v1->v3 ends correct."""
        with self.write_transaction() as conn:
            existing = {r[0] for r in conn.execute("SELECT id FROM entities")}
            for ev_id, raw in conn.execute("SELECT id, entity_ids FROM events"):
                try:
                    decoded = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    decoded = []
                for x in decoded if isinstance(decoded, list) else []:
                    try:
                        eid = int(x)
                    except (TypeError, ValueError):
                        continue
                    if eid in existing:
                        conn.execute(
                            "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                            (ev_id, eid),
                        )
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (3, ?)", (time.time(),)
            )

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def _add_entity_inline(
        self,
        conn: sqlite3.Connection,
        kind: str,
        name: str,
        summary: str = "",
    ) -> int:
        """Insert a new entity on the caller's open ``conn`` and return its id."""
        now = time.time()
        cur = conn.execute(
            "INSERT INTO entities (kind, name, summary, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (kind, name, summary, now, now),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def add_entity(self, kind: str, name: str, summary: str = "") -> int:
        """Insert a new entity and return its id."""
        with self.write_transaction() as conn:
            return self._add_entity_inline(conn, kind, name, summary)

    def get_entity(self, entity_id: int) -> Entity | None:
        """Return the Entity with the given id, or None."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at"
                " FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()
        return Entity(*row) if row else None

    def entity_exists(self, entity_id: object) -> bool:
        """True iff *entity_id* coerces to int and a row with that id exists.

        Tolerates non-int input (returns False for None, strings, etc.) so
        callers validating LLM-emitted payloads don't have to pre-check
        types. ``add_relation`` and ``set_attribute`` payloads use this to
        drop proposals referencing nonexistent ids before they reach the
        operator queue.
        """
        if isinstance(entity_id, bool) or not isinstance(entity_id, int):
            return False
        eid = entity_id
        with self.read_connection() as conn:
            row = conn.execute("SELECT 1 FROM entities WHERE id = ?", (eid,)).fetchone()
        return row is not None

    def find_entity_by_name(
        self, name: str, kind: str | None = None, *, active_only: bool = False
    ) -> Entity | None:
        """Case-insensitive name lookup. Optional kind filter. Returns first match by id.

        When ``active_only`` is True, retired entities are excluded — use it
        when resolving an action target (a retired place/avatar/item is not a
        valid thing to interact with).
        """
        kind_filter = "AND kind = ?" if kind is not None else ""
        active_filter = "AND status = 'active'" if active_only else ""
        sql = (
            "SELECT id, kind, name, summary, status, created_at, updated_at"
            " FROM entities"
            f" WHERE LOWER(name) = LOWER(?)"
            f" {kind_filter}"
            f" {active_filter}"
            " ORDER BY id ASC LIMIT 1"
        )
        params = (name, kind) if kind is not None else (name,)
        with self.read_connection() as conn:
            row = conn.execute(sql, params).fetchone()
        return Entity(*row) if row else None

    def _find_active_entity_by_name_inline(
        self,
        conn: sqlite3.Connection,
        name: str,
    ) -> Entity | None:
        """Resolve a name with precedence avatar > npc > item > place,
        case-insensitive, restricted to status='active'. Caller-provided
        open conn (works under both read_connection and write_transaction).

        Used by record_user_event (in-tx, must avoid lock reentry) and by
        find_active_entity_by_name (out-of-tx, public)."""
        row = conn.execute(
            "SELECT id, kind, name, summary, status, created_at, updated_at"
            " FROM entities"
            " WHERE LOWER(name) = LOWER(?) AND status = 'active'"
            " ORDER BY"
            "   CASE kind"
            "     WHEN 'avatar' THEN 0"
            "     WHEN 'npc'    THEN 1"
            "     WHEN 'item'   THEN 2"
            "     WHEN 'place'  THEN 3"
            "     ELSE 4"
            "   END,"
            "   id ASC"
            " LIMIT 1",
            (name,),
        ).fetchone()
        return Entity(*row) if row else None

    def find_active_entity_by_name(self, name: str) -> Entity | None:
        """Public wrapper around _find_active_entity_by_name_inline."""
        with self.read_connection() as conn:
            return self._find_active_entity_by_name_inline(conn, name)

    def _add_alias_inline(self, conn: sqlite3.Connection, entity_id: int, alias: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO entity_alias (entity_id, alias) VALUES (?, ?)",
            (entity_id, alias),
        )

    def add_alias(self, entity_id: int, alias: str) -> None:
        with self.write_transaction() as conn:
            self._add_alias_inline(conn, entity_id, alias)

    def list_aliases(self, entity_id: int) -> list[str]:
        with self.read_connection() as conn:
            return [
                r[0]
                for r in conn.execute(
                    "SELECT alias FROM entity_alias WHERE entity_id=?", (entity_id,)
                )
            ]

    def find_entity_by_name_or_alias(self, name: str) -> Entity | None:
        """Active resolution: canonical name (kind-precedence) first, then alias."""
        with self.read_connection() as conn:
            ent = self._find_active_entity_by_name_inline(conn, name)
            if ent is not None:
                return ent
            row = conn.execute(
                "SELECT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at "
                "FROM entities e JOIN entity_alias al ON al.entity_id = e.id "
                "WHERE al.alias = ? COLLATE NOCASE AND e.status='active' ORDER BY e.id ASC LIMIT 1",
                (name,),
            ).fetchone()
        return Entity(*row) if row else None

    def _reactivate_auto_npc_inline(
        self,
        conn: sqlite3.Connection,
        name: str,
        ts: float,
    ) -> int | None:
        """If a retired auto-created npc exists by this name, reactivate it
        (status->active, refresh last_seen_ts) and return its id; else None.

        This is what stops aged-out NPCs from spawning duplicate rows when
        they are mentioned again: aging retires ``auto_created`` npcs, and a
        re-mention reuses the same id instead of inserting a fresh entity.
        Scoped to ``auto_created`` npcs so the avatar lifecycle (opted-out
        avatars) and deliberate operator retirements of canon are never
        silently undone. Caller-provided open conn (used inside
        record_user_event's write transaction)."""
        row = conn.execute(
            "SELECT e.id FROM entities e JOIN attributes a ON a.entity_id = e.id"
            " WHERE LOWER(e.name) = LOWER(?) AND e.kind = 'npc'"
            "   AND e.status = 'retired' AND a.key = 'auto_created' AND a.value = '1'"
            " ORDER BY e.updated_at DESC, e.id DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row is None:
            return None
        eid = int(row[0])
        self._set_status_inline(conn, eid, "active")
        self._set_attribute_inline(conn, eid, "last_seen_ts", str(ts))
        return eid

    def resolve_ref(self, ref: str) -> int:
        """Resolve an operator <ref> to an entity id.

        '#<int>' is always an id; anything else is a name (so an entity
        literally named '7' is addressable). Raises LookupError if unknown.
        """
        ref = ref.strip()
        if ref.startswith("#") and ref[1:].isdigit():
            eid = int(ref[1:])
            if self.get_entity(eid) is None:
                raise LookupError(f"no entity #{eid}")
            return eid
        ent = self.find_active_entity_by_name(ref)
        if ent is None:
            raise LookupError(f"no active entity named {ref!r}")
        return ent.id

    def _set_status_inline(
        self,
        conn: sqlite3.Connection,
        entity_id: int,
        status: str,
    ) -> None:
        """Update entity status + updated_at on the caller's open ``conn``."""
        now = time.time()
        conn.execute(
            "UPDATE entities SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, entity_id),
        )

    def set_status(self, entity_id: int, status: str) -> None:
        """Update entity status and updated_at. Silent no-op if entity_id not found."""
        with self.write_transaction() as conn:
            self._set_status_inline(conn, entity_id, status)

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

    def list_pinned_entities(self) -> list[Entity]:
        """Active entities carrying the 'pinned' attribute, deterministic order.

        Order: kind precedence (avatar, npc, place, faction, item) then name,
        so the roster prompt block is cache-stable.
        """
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at "
                "FROM entities e JOIN attributes a ON a.entity_id = e.id "
                "WHERE a.key='pinned' AND a.value='1' AND e.status='active' "
                "ORDER BY CASE e.kind WHEN 'avatar' THEN 0 WHEN 'npc' THEN 1 "
                "  WHEN 'place' THEN 2 WHEN 'faction' THEN 3 ELSE 4 END, e.name COLLATE NOCASE"
            ).fetchall()
        return [Entity(*row) for row in rows]

    def _set_author_locked_inline(
        self, conn: sqlite3.Connection, entity_id: int, locked: bool
    ) -> None:
        if locked:
            self._set_attribute_inline(conn, entity_id, "author_locked", "1")
        else:
            conn.execute(
                "DELETE FROM attributes WHERE entity_id=? AND key='author_locked'", (entity_id,)
            )

    def set_author_locked(self, entity_id: int, locked: bool) -> None:
        """Lock/unlock durable canon (always injected, aging-exempt, loom-protected)."""
        with self.write_transaction() as conn:
            self._set_author_locked_inline(conn, entity_id, locked)

    def list_canon_entities(self) -> list[Entity]:
        """Active entities that are durable canon: pinned (operator) OR author_locked.

        DISTINCT (an entity may carry both). Deterministic kind-then-name order
        (matches list_pinned_entities so the roster prompt block stays cache-stable).
        """
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at "
                "FROM entities e JOIN attributes a ON a.entity_id = e.id "
                "WHERE a.key IN ('pinned','author_locked') AND a.value='1' AND e.status='active' "
                "ORDER BY CASE e.kind WHEN 'avatar' THEN 0 WHEN 'npc' THEN 1 "
                "  WHEN 'place' THEN 2 WHEN 'faction' THEN 3 ELSE 4 END, e.name COLLATE NOCASE"
            ).fetchall()
        return [Entity(*row) for row in rows]

    def active_name_exists(self, name: str) -> bool:
        """True if some active entity already has this name (case-insensitive)."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM entities WHERE LOWER(name)=LOWER(?) AND status='active' LIMIT 1",
                (name,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Attribute CRUD
    # ------------------------------------------------------------------

    def _set_attribute_inline(
        self,
        conn: sqlite3.Connection,
        entity_id: int,
        key: str,
        value: str,
    ) -> None:
        """Upsert an attribute on the caller's open ``conn``."""
        conn.execute(
            "INSERT INTO attributes (entity_id, key, value) VALUES (?, ?, ?)"
            " ON CONFLICT(entity_id, key) DO UPDATE SET value = excluded.value",
            (entity_id, key, value),
        )

    def set_attribute(self, entity_id: int, key: str, value: str) -> None:
        """Upsert an attribute key/value for the given entity."""
        with self.write_transaction() as conn:
            self._set_attribute_inline(conn, entity_id, key, value)

    def bump_last_seen_ts(
        self,
        entity_ids: Sequence[int],
        *,
        ts: float,
    ) -> None:
        """Bump ``last_seen_ts`` on every id. Single ``write_transaction``.

        Used by loom ``apply_or_queue`` (which runs outside any open tx).
        Defensively skips ids that do not resolve to an ``entities`` row,
        because callers may pass LLM-emitted ids that never got
        validated. No-op for empty input.
        """
        if not entity_ids:
            return
        ts_str = str(ts)
        with self.write_transaction() as conn:
            for eid in entity_ids:
                row = conn.execute("SELECT 1 FROM entities WHERE id=?", (int(eid),)).fetchone()
                if row is None:
                    continue
                self._set_attribute_inline(conn, int(eid), "last_seen_ts", ts_str)

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

    def list_entities_with_attribute(
        self,
        *,
        key: str,
        value: str,
        status: str | None = "active",
    ) -> list[Entity]:
        """All entities with attribute (key=value), optionally filtered by
        entity status. Used by aging to find auto_created='1' entities."""
        sql = (
            "SELECT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at"
            " FROM entities e"
            " JOIN attributes a ON a.entity_id = e.id"
            " WHERE a.key = ? AND a.value = ?"
        )
        params: list[Any] = [key, value]
        if status is not None:
            sql += " AND e.status = ?"
            params.append(status)
        sql += " ORDER BY e.id ASC"
        with self.read_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Entity(*row) for row in rows]

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

    def _add_event_inline(
        self,
        conn: sqlite3.Connection,
        *,
        summary: str,
        entity_ids: Sequence[int],
        source: str,
        ts: float | None = None,
    ) -> int:
        """Insert an event on the caller's open ``conn`` and return its id.

        Single writer for both ``events`` and the ``event_actor`` join. The
        join is populated FK-safe (only ids that resolve to an ``entities``
        row), de-duped, via ``INSERT OR IGNORE``.
        """
        if ts is None:
            ts = time.time()
        ids = list(entity_ids)
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, json.dumps(ids), source),
        )
        assert cur.lastrowid is not None
        event_id = int(cur.lastrowid)
        for eid in dict.fromkeys(ids):
            if conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                    (event_id, eid),
                )
        return event_id

    def add_event(
        self,
        summary: str,
        entity_ids: Sequence[int],
        source: str,
    ) -> int:
        """Insert an event and return its id."""
        with self.write_transaction() as conn:
            return self._add_event_inline(
                conn, summary=summary, entity_ids=entity_ids, source=source
            )

    def record_user_event(
        self,
        *,
        actor_id: int,
        summary: str,
        actor_names: Sequence[str],
        now: Callable[[], float] = time.time,
    ) -> int:
        """Resolve actor_names to entity ids (auto-create as npc if unknown),
        bump last_seen_ts on each non-avatar, and write one event row — all
        in a single write_transaction.

        The caller's avatar id is the first entry of the event's entity_ids
        list; auto-created NPCs follow in actor_names order. source='avatar'
        (per design §3 — re-using the existing CHECK constraint, not adding
        a new value).

        Concurrency: safe across callers sharing one cached VerseStore
        instance per channel within one process. Multiple processes touching
        the same DB or multiple VerseStore instances for the same channel
        are NOT defended against (out of scope for v1).
        """
        ts = now()
        with self.write_transaction() as conn:
            actor_row = conn.execute(
                "SELECT kind, status FROM entities WHERE id = ?", (actor_id,)
            ).fetchone()
            if actor_row is None or actor_row[1] != "active":
                raise ValueError(f"record_user_event: actor_id={actor_id} not an active entity")

            ids: list[int] = [actor_id]
            for name in actor_names:
                entity = self._find_active_entity_by_name_inline(conn, name)
                if entity is None:
                    eid = self._reactivate_auto_npc_inline(conn, name, ts)
                    if eid is None:
                        eid = self._add_entity_inline(conn, "npc", name, "")
                        self._set_attribute_inline(conn, eid, "auto_created", "1")
                        self._set_attribute_inline(conn, eid, "last_seen_ts", str(ts))
                else:
                    eid = entity.id
                    if entity.kind != "avatar":
                        self._set_attribute_inline(conn, eid, "last_seen_ts", str(ts))
                ids.append(eid)

            return self._add_event_inline(
                conn, summary=summary, entity_ids=ids, source="avatar", ts=ts
            )

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
        *,
        require_active_entity: bool = False,
    ) -> list[Event]:
        """Return events newest-first, optionally excluding given sources.

        When ``require_active_entity`` is True, events whose every referenced
        entity is retired or deleted ("dead lore") are skipped, so a retired
        entity's name/id is not replayed into a prompt via the event log.
        Entity-less narration events are always kept. ``limit`` then counts
        surviving rows, so the SQL LIMIT is dropped and the cursor is scanned
        lazily newest-first until ``limit`` survivors are collected."""
        where = ""
        params: list = []
        if exclude_sources:
            placeholders = ",".join("?" * len(exclude_sources))
            where = f" WHERE source NOT IN ({placeholders})"
            params.extend(exclude_sources)
        base = (
            "SELECT id, ts, summary, entity_ids, source FROM events"
            f"{where} ORDER BY ts DESC, id DESC"
        )
        with self.read_connection() as conn:
            if not require_active_entity:
                rows = conn.execute(f"{base} LIMIT ?", (*params, limit)).fetchall()
                return [
                    Event(
                        id=row[0],
                        ts=row[1],
                        summary=row[2],
                        entity_ids=_parse_entity_ids(row[3], row[0]),
                        source=row[4],
                    )
                    for row in rows
                ]
            active_ids = {
                r[0] for r in conn.execute("SELECT id FROM entities WHERE status='active'")
            }
            out: list[Event] = []
            for row in conn.execute(base, params):
                ids = _parse_entity_ids(row[3], row[0])
                if ids and not any(i in active_ids for i in ids):
                    continue
                out.append(
                    Event(
                        id=row[0],
                        ts=row[1],
                        summary=row[2],
                        entity_ids=ids,
                        source=row[4],
                    )
                )
                if len(out) >= limit:
                    break
            return out

    def match_entities_in_text(self, text: str, limit: int = 12) -> list[Entity]:
        """Active entities whose name OR alias appears as a whole word in ``text``.

        Names/aliases <=2 chars are skipped. A name/alias that is a common English
        word (stoplist) only matches when it appears CAPITALIZED as a whole word
        (proper-noun usage) — so an NPC "Will" matches "Will, run!" but not
        "I will go". All other names match case-insensitively. Plain scan — the
        world is small."""
        low = text.lower()
        with self.read_connection() as conn:
            ent_rows = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at "
                "FROM entities WHERE status='active' ORDER BY id"
            ).fetchall()
            alias_rows = conn.execute(
                "SELECT al.entity_id, al.alias FROM entity_alias al "
                "JOIN entities e ON e.id=al.entity_id WHERE e.status='active'"
            ).fetchall()
        aliases: dict[int, list[str]] = {}
        for eid, al in alias_rows:
            aliases.setdefault(eid, []).append(al)

        def hit(token: str) -> bool:
            t = token.lower()
            if len(t) <= 2:
                return False
            if t in _MATCH_STOPLIST:
                proper = t[0].upper() + t[1:]
                return re.search(r"(?<!\w)" + re.escape(proper) + r"(?!\w)", text) is not None
            return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low) is not None

        out: list[Entity] = []
        for row in ent_rows:
            ent = Entity(*row)
            if any(hit(n) for n in (ent.name, *aliases.get(ent.id, [])) if n):
                out.append(ent)
            if len(out) >= limit:
                break
        return out

    def relations_for(self, entity_ids: Sequence[int], limit: int = 30) -> list[RelationView]:
        """One-hop relations touching any of ``entity_ids`` (either endpoint),
        both endpoints active. Ordered by relation id."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT ef.name, et.name, r.kind, r.note FROM relations r "
                f"JOIN entities ef ON ef.id=r.from_id JOIN entities et ON et.id=r.to_id "
                f"WHERE (r.from_id IN ({ph}) OR r.to_id IN ({ph})) "
                f"  AND ef.status='active' AND et.status='active' ORDER BY r.id LIMIT ?",
                (*entity_ids, *entity_ids, limit),
            ).fetchall()
        return [RelationView(*r) for r in rows]

    def events_for_entities(self, entity_ids: Sequence[int], limit: int = 8) -> list[Event]:
        """Recent events linking any of ``entity_ids`` (via ``event_actor``),
        restricted to events that still have >=1 ACTIVE actor (SQL-side filter).
        Newest first."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT ev.id, ev.ts, ev.summary, ev.entity_ids, ev.source FROM events ev "
                f"JOIN event_actor ea ON ea.event_id=ev.id WHERE ea.entity_id IN ({ph}) "
                f"  AND EXISTS (SELECT 1 FROM event_actor ea2 JOIN entities e2 ON e2.id=ea2.entity_id "
                f"              WHERE ea2.event_id=ev.id AND e2.status='active') "
                f"ORDER BY ev.ts DESC, ev.id DESC LIMIT ?",
                (*entity_ids, limit),
            ).fetchall()
        return [
            Event(
                id=r[0],
                ts=r[1],
                summary=r[2],
                entity_ids=_parse_entity_ids(r[3], r[0]),
                source=r[4],
            )
            for r in rows
        ]

    def _replace_events_with_source(
        self,
        *,
        delete_ids: Sequence[int],
        summary: str,
        entity_ids: Sequence[int],
        ts: float,
        source: str,
    ) -> int:
        """Atomic delete-then-insert. Returns the new event's id."""
        with self.write_transaction() as conn:
            if delete_ids:
                placeholders = ",".join("?" for _ in delete_ids)
                conn.execute(
                    f"DELETE FROM events WHERE id IN ({placeholders})",
                    tuple(delete_ids),
                )
            new_id = self._add_event_inline(
                conn, summary=summary, entity_ids=entity_ids, source=source, ts=ts
            )
            # Heartbeat: bump last_seen_ts on every entity referenced in
            # the digest. ``events.entity_ids`` is a JSON blob with no FK
            # enforcement, so we defensively skip ids that do not resolve
            # to an ``entities`` row (otherwise the attributes-FK would
            # fail).
            for eid in entity_ids:
                row = conn.execute("SELECT 1 FROM entities WHERE id=?", (int(eid),)).fetchone()
                if row is None:
                    continue
                self._set_attribute_inline(conn, int(eid), "last_seen_ts", str(ts))
            return new_id

    def replace_events_with_lore_digest(
        self,
        *,
        delete_ids: Sequence[int],
        summary: str,
        entity_ids: Sequence[int],
        ts: float,
    ) -> int:
        """Replace ``delete_ids`` with a single ``source='llm'`` digest event.

        All work happens inside one ``write_transaction``; on error the whole
        operation rolls back and the originals survive.
        """
        return self._replace_events_with_source(
            delete_ids=delete_ids,
            summary=summary,
            entity_ids=entity_ids,
            ts=ts,
            source="llm",
        )

    def events_older_than(self, *, cutoff_ts: float) -> list[Event]:
        """All events with ``ts < cutoff_ts``, oldest-first.

        Used by retention compaction to gather rows that will be replaced by
        a single lore-digest event. Lock-free read. ``entity_ids`` are
        normalised to ``int`` to match the existing ``recent_events``
        convention (``store.py:387``).
        """
        with self.read_connection() as conn:
            cur = conn.execute(
                "SELECT id, ts, summary, entity_ids, source FROM events "
                "WHERE ts < ? ORDER BY ts ASC, id ASC",
                (cutoff_ts,),
            )
            return [
                Event(
                    id=row[0],
                    ts=row[1],
                    summary=row[2],
                    entity_ids=_parse_entity_ids(row[3], row[0]),
                    source=row[4],
                )
                for row in cur.fetchall()
            ]

    # ------------------------------------------------------------------
    # Avatar opt-in
    # ------------------------------------------------------------------

    @staticmethod
    def _relink_avatar(
        conn: sqlite3.Connection, entity_id: int, nick: str, account: str | None
    ) -> None:
        """Point this avatar's link at *nick*/*account*, degrading on a nick clash.

        ``avatar_link.nick`` is UNIQUE NOT NULL. If *nick* is already held by
        another avatar's link (a cross-account collision — the live IRC user has
        renamed into a nick recorded on someone else's link), the full update
        would raise sqlite3.IntegrityError and crash opt-in. Degrade instead: the
        opting-in user is still served via their account, and the contested nick
        stays with its current owner rather than taking the whole command down.
        """
        try:
            conn.execute(
                "UPDATE avatar_link SET nick = ?, account = ? WHERE entity_id = ?",
                (nick, account, entity_id),
            )
        except sqlite3.IntegrityError:
            conn.execute(
                "UPDATE avatar_link SET account = ? WHERE entity_id = ?",
                (account, entity_id),
            )

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
                # ``COLLATE NOCASE`` (not LOWER(al.nick)) keeps the predicate
                # sargable so it seeks idx_avatar_link_nick_nocase instead of
                # scanning avatar_link under the write lock. Both are ASCII-only,
                # so the match semantics are identical for IRC nicks.
                row = conn.execute(
                    "SELECT al.entity_id, e.status"
                    " FROM avatar_link al"
                    " JOIN entities e ON e.id = al.entity_id"
                    " WHERE al.nick = ? COLLATE NOCASE",
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
                self._relink_avatar(conn, entity_id, nick, account)

            elif entity_id is not None and entity_status == "retired":
                # Retired (soft pause): reactivate and update the link.
                conn.execute(
                    "UPDATE entities SET status = 'active', updated_at = ? WHERE id = ?",
                    (now, entity_id),
                )
                self._relink_avatar(conn, entity_id, nick, account)

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
            # 3. Resolve the avatar's place.
            #    Re-opt-in of an ALREADY-ACTIVE avatar must be idempotent and
            #    must NOT relocate it. Steps 3 (pick the activity-best starter
            #    place) and 4 (UPSERT the 'location' attribute) only run for new
            #    or reactivated avatars; a re-opt-in reads the avatar's existing
            #    committed location instead. Without this guard, a user who
            #    opted in, moved (verse_move writes the same 'location'
            #    attribute), then re-ran @verseopt in was silently teleported to
            #    the "best" place — destroying committed game state behind a
            #    "you are already opted in" no-op.
            # ----------------------------------------------------------
            existing_location: str | None = None
            if was_already_opted_in:
                loc_row = conn.execute(
                    "SELECT value FROM attributes WHERE entity_id = ? AND key = 'location'",
                    (entity_id,),
                ).fetchone()
                existing_location = loc_row[0] if loc_row else None

            if existing_location is not None:
                # Idempotent re-opt-in: keep the avatar exactly where it is.
                place_name = existing_location
                summary_row = conn.execute(
                    "SELECT summary FROM entities WHERE kind = 'place' AND name = ?"
                    " ORDER BY (status = 'active') DESC LIMIT 1",
                    (existing_location,),
                ).fetchone()
                place_summary = summary_row[0] if summary_row and summary_row[0] else ""
            else:
                # New / reactivated avatar (or a legacy active avatar that never
                # had a 'location' attribute): pick or create the starter place.
                # Pull all active places in Python — small count in early-PR1
                # use, avoids json_each complexity, and keeps SQL surface small.
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
        proposal_id: str | None = None,
    ) -> str:
        """Insert a proposal and return its id.

        When *proposal_id* is None (default) a fresh uuid is generated.
        The crosspoll-receiver consume hook passes a caller-supplied id
        so the consumption row written by ``CrosspollStore.claim_seed_for``
        points at the same proposal record.

        When *status* is 'approved' or 'rejected', *reviewer* must be
        supplied and reviewed_at is set to now (this is how auto-apply
        records its audit row inside the same write_transaction as the
        mutation it just applied).
        """
        if status not in _VALID_PROPOSAL_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        if status != "pending" and not reviewer:
            raise ValueError("reviewer required when status != pending")
        pid = proposal_id if proposal_id is not None else uuid.uuid4().hex
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

    def _apply_op_inline(
        self,
        conn: sqlite3.Connection,
        *,
        op: str,
        payload: dict[str, Any],
        source: str,
    ) -> int | None:
        """Run the op-specific INSERT on *conn*. The caller owns the txn."""
        now = time.time()
        if source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        privileged = source == "operator"
        if op in _DESTRUCTIVE_OPS and not privileged:
            raise PermissionError(f"op {op!r} requires operator privilege")
        if op == "add_event":
            return self._add_event_inline(
                conn,
                summary=payload["summary"],
                entity_ids=payload.get("entity_ids", []),
                source=source,
                ts=now,
            )
        if op == "set_attribute":
            eid = payload["entity_id"]
            key = payload["key"]
            # Proposals must not touch lifecycle/identity keys (immortality,
            # aging enrollment, relocation) — those are engine-only.
            if key in _RESERVED_ATTRIBUTE_KEYS:
                raise ValueError(
                    f"attribute key {key!r} is reserved (lifecycle-controlled) "
                    "and cannot be set by a proposal"
                )
            # Validate active status, not mere existence: a proposal must not
            # mutate a retired/soft-deleted entity (auto-apply, human approval
            # of a since-retired proposal, and crosspoll all flow through here).
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise LookupError(f"entity_id {eid} does not exist")
            if row[0] == "retired":
                raise LookupError(f"entity_id {eid} is retired")
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(entity_id, key) DO UPDATE SET value=excluded.value",
                (eid, key, payload["value"]),
            )
            return None
        if op == "add_relation":
            from_id = payload["from_id"]
            to_id = payload["to_id"]
            # Validate both endpoints are active, not merely extant: the FK
            # enforces existence but not status, so a retired/soft-deleted
            # entity would slip through. Mirror the set_attribute guard so
            # auto-apply, human approval of a since-retired proposal, and
            # crosspoll all reject retired endpoints.
            for endpoint in (from_id, to_id):
                row = conn.execute("SELECT status FROM entities WHERE id=?", (endpoint,)).fetchone()
                if row is None:
                    raise LookupError(f"entity_id {endpoint} does not exist")
                if row[0] == "retired":
                    raise LookupError(f"entity_id {endpoint} is retired")
            cur = conn.execute(
                "INSERT INTO relations (from_id, to_id, kind, note) VALUES (?, ?, ?, ?)",
                (from_id, to_id, payload["kind"], payload.get("note", "")),
            )
            return cur.lastrowid
        if op == "add_entity":
            cur = conn.execute(
                "INSERT INTO entities (kind, name, summary, status, "
                "                       created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', ?, ?)",
                (
                    payload["kind"],
                    payload["name"],
                    payload.get("summary", ""),
                    now,
                    now,
                ),
            )
            return cur.lastrowid
        if op == "update_entity":
            eid = payload["entity_id"]
            if "kind" in payload:
                raise ValueError("update_entity cannot change kind")
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise LookupError(f"entity_id {eid} does not exist")
            sets, args = [], []
            if "name" in payload:
                sets.append("name=?")
                args.append(payload["name"])
            if "summary" in payload:
                sets.append("summary=?")
                args.append(payload["summary"])
            if not sets:
                raise ValueError("update_entity needs name and/or summary")
            sets.append("updated_at=?")
            args.append(now)
            args.append(eid)
            conn.execute(f"UPDATE entities SET {', '.join(sets)} WHERE id=?", args)
            return None
        if op == "set_status":
            eid = payload["entity_id"]
            new_status = payload["status"]
            if new_status not in ("active", "retired"):
                raise ValueError(f"invalid status: {new_status!r}")
            row = conn.execute("SELECT kind FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise LookupError(f"entity_id {eid} does not exist")
            if new_status == "retired" and row[0] == "avatar":
                conn.execute("DELETE FROM avatar_link WHERE entity_id=?", (eid,))
            conn.execute(
                "UPDATE entities SET status=?, updated_at=? WHERE id=?", (new_status, now, eid)
            )
            return None
        if op == "set_pinned":
            eid = payload["entity_id"]
            pinned = payload["pinned"]
            row = conn.execute("SELECT id FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise LookupError(f"entity_id {eid} does not exist")
            if pinned:
                conn.execute(
                    "INSERT INTO attributes (entity_id, key, value) VALUES (?, 'pinned', '1') "
                    "ON CONFLICT(entity_id, key) DO UPDATE SET value='1'",
                    (eid,),
                )
            else:
                conn.execute("DELETE FROM attributes WHERE entity_id=? AND key='pinned'", (eid,))
            return None
        if op == "edit_event":
            ev_id = payload["event_id"]
            cur = conn.execute(
                "UPDATE events SET summary=? WHERE id=?", (payload["summary"], ev_id)
            )
            if cur.rowcount == 0:
                raise LookupError(f"event_id {ev_id} does not exist")
            return None
        if op == "delete_event":
            cur = conn.execute("DELETE FROM events WHERE id=?", (payload["event_id"],))
            if cur.rowcount == 0:
                raise LookupError(f"event_id {payload['event_id']} does not exist")
            return None
        if op == "delete_relation":
            cur = conn.execute("DELETE FROM relations WHERE id=?", (payload["relation_id"],))
            if cur.rowcount == 0:
                raise LookupError(f"relation_id {payload['relation_id']} does not exist")
            return None
        raise ValueError(f"unknown op: {op!r}")

    def apply_and_record_proposal(
        self,
        *,
        cycle_id: str,
        op: str,
        payload: dict[str, Any],
        confidence: float,
        provenance: str,
        reviewer: str,
        source: str = "loom",
    ) -> str:
        """Atomically apply *op* and insert an approved proposal row.

        Returns the new proposal id. Either both rows are written or
        neither (the lock + write_transaction guarantee SQLite atomicity).
        """
        pid = uuid.uuid4().hex
        now = time.time()
        with self.write_transaction() as conn:
            self._apply_op_inline(conn, op=op, payload=payload, source=source)
            conn.execute(
                "INSERT INTO proposals "
                "(id, created_at, cycle_id, op, payload, confidence, provenance, "
                " status, reviewer, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'approved', ?, ?)",
                (
                    pid,
                    now,
                    cycle_id,
                    op,
                    json.dumps(payload),
                    confidence,
                    provenance,
                    reviewer,
                    now,
                ),
            )
        return pid

    def apply_direct(
        self,
        *,
        op: str,
        payload: dict[str, Any],
        source: str,
        provenance: str,
    ) -> int | None:
        """Apply *op* immediately and write an approved audit proposal row.

        For operator commands (source='operator') and the verse_edit tool
        (source='llm'). Unlike apply_and_record_proposal this carries no loom
        ceremony (cycle_id/confidence/reviewer are synthesized for audit only).
        Returns the new row id for creating ops, else None.
        """
        pid = uuid.uuid4().hex
        now = time.time()
        with self.write_transaction() as conn:
            result = self._apply_op_inline(conn, op=op, payload=payload, source=source)
            conn.execute(
                "INSERT INTO proposals "
                "(id, created_at, cycle_id, op, payload, confidence, provenance, "
                " status, reviewer, reviewed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'approved', ?, ?)",
                (pid, now, "direct", op, json.dumps(payload), 1.0, provenance, source, now),
            )
        return result

    def apply_proposal_and_mark(
        self,
        proposal_id: str,
        *,
        reviewer: str,
        event_source: str = "loom",
    ) -> None:
        """Atomically apply a pending proposal and flip its status to approved.

        ``event_source`` is the value written into ``events.source`` (or any
        other rows the op produces). Defaults to ``'loom'``; the crosspoll
        receive path passes ``'crosspoll'``.

        Raises ``LookupError`` if no such id, ``ValueError`` if already
        terminal.
        """
        with self.write_transaction() as conn:
            row = conn.execute(
                "SELECT op, payload, status FROM proposals WHERE id=?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"no proposal: {proposal_id!r}")
            op, payload_json, status = row
            if status != "pending":
                raise ValueError(f"proposal {proposal_id!r} already {status}; cannot apply")
            payload = json.loads(payload_json)
            self._apply_op_inline(conn, op=op, payload=payload, source=event_source)
            conn.execute(
                "UPDATE proposals SET status='approved', reviewer=?, reviewed_at=? WHERE id=?",
                (reviewer, time.time(), proposal_id),
            )

    def apply_proposal(
        self,
        *,
        op: str,
        payload: dict[str, Any],
        source: str = "loom",
    ) -> int | None:
        """Convert a proposal payload into rows via the single core dispatcher.

        Returns the new entity id for ``add_entity``, the new event id for
        ``add_event``, the new relation id for ``add_relation``, or
        ``None`` for the mutation ops. Raises ``ValueError`` for unknown
        ops or ``KeyError`` for missing payload keys.
        """
        with self.write_transaction() as conn:
            return self._apply_op_inline(conn, op=op, payload=payload, source=source)

    def update_proposal_status(self, proposal_id: str, *, status: str, reviewer: str) -> None:
        """Flip *proposal_id*'s status (audit fields written together)."""
        if status not in _VALID_PROPOSAL_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        with self.write_transaction() as conn:
            cur = conn.execute(
                "UPDATE proposals SET status=?, reviewer=?, reviewed_at=? WHERE id=?",
                (status, reviewer, time.time(), proposal_id),
            )
            if cur.rowcount == 0:
                raise LookupError(f"no proposal: {proposal_id!r}")

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
