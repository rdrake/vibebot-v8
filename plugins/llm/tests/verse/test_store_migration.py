import json
import sqlite3
import time
from pathlib import Path

from llm.verse.store import SCHEMA_VERSION, VerseStore, db_path_for_channel


def _make_v1_db(base: Path, channel: str) -> Path:
    """Hand-build a v1-schema DB: legacy CHECKs + version row = 1."""
    path = db_path_for_channel(base, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at REAL NOT NULL);
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK (kind IN ('avatar','npc','place','faction','item')),
            name TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, summary TEXT NOT NULL,
            entity_ids TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll')));
        CREATE TABLE proposals (
            id TEXT PRIMARY KEY, created_at REAL NOT NULL, cycle_id TEXT NOT NULL,
            op TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity')),
            payload TEXT NOT NULL, confidence REAL NOT NULL, provenance TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
            reviewer TEXT, reviewed_at REAL);
        """
    )
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (1, ?)", (time.time(),))
    conn.commit()
    conn.close()
    return path


def test_migration_v1_to_v2_widens_checks(tmp_path):
    _make_v1_db(tmp_path, "#chan")
    store = VerseStore(tmp_path, "#chan")  # triggers _migrate

    with store.read_connection() as conn:
        version = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    # A v1 DB now chains v1->v2->v3, so the latest stamped version is 3; the v2
    # CHECK-widening this test guards is still exercised by the INSERTs below.
    assert version == 3

    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, '[]', 'operator')",
            (time.time(), "op edit"),
        )
        conn.execute(
            "INSERT INTO proposals (id, created_at, cycle_id, op, payload, confidence) "
            "VALUES ('p1', ?, 'c', 'delete_event', '{}', 1.0)",
            (time.time(),),
        )


def test_migration_is_idempotent(tmp_path):
    _make_v1_db(tmp_path, "#chan")
    VerseStore(tmp_path, "#chan")
    store2 = VerseStore(tmp_path, "#chan")  # second open, must not double-apply
    with store2.read_connection() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=2").fetchone()[0]
    assert rows == 1


def _seed_v2_event(base, channel, entity_ids):
    """Add an entity + one event (with possibly-bad entity_ids) to a v1 DB so the
    v1->v2->v3 chain backfills event_actor element-wise."""
    path = _make_v1_db(base, channel)
    raw = sqlite3.connect(path)
    raw.execute(
        "INSERT INTO entities(id,kind,name,created_at,updated_at) VALUES (1,'npc','Harry',0,0)"
    )
    raw.execute(
        "INSERT INTO events(id,ts,summary,entity_ids,source) VALUES (1,0,'x',?, 'avatar')",
        (json.dumps(entity_ids),),
    )
    raw.commit()
    raw.close()
    return path


def test_fresh_db_is_v3_with_new_tables(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    with store.read_connection() as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert {"entity_alias", "event_actor"} <= names
    assert ver == SCHEMA_VERSION == 3


def test_v1_to_v3_chain_backfills_event_actor_elementwise(tmp_path):
    _seed_v2_event(tmp_path, "#chan", [1, 99999, "garbage"])
    store = VerseStore(tmp_path, "#chan")  # runs v1->v2->v3
    with store.read_connection() as conn:
        rows = conn.execute(
            "SELECT event_id, entity_id FROM event_actor ORDER BY entity_id"
        ).fetchall()
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert rows == [(1, 1)]  # keep valid existing id 1; drop 99999 + 'garbage'
    assert ver == 3


def test_v3_migration_idempotent_on_reopen(tmp_path):
    _seed_v2_event(tmp_path, "#chan", [1])
    VerseStore(tmp_path, "#chan")
    store2 = VerseStore(tmp_path, "#chan")  # second open must not double-apply
    with store2.read_connection() as conn:
        v3rows = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=3").fetchone()[0]
        ea = conn.execute("SELECT COUNT(*) FROM event_actor").fetchone()[0]
    assert v3rows == 1 and ea == 1  # INSERT OR IGNORE keeps the backfill a no-op


def _make_v2_db(base: Path, channel: str) -> Path:
    """Hand-build a v2-schema DB — the production state: widened CHECKs, version=2,
    populated entities/events, and NO event_actor/entity_alias tables yet (those are
    created by schema.sql executescript on open, then backfilled by v2->v3)."""
    path = db_path_for_channel(base, channel)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at REAL NOT NULL);
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK (kind IN ('avatar','npc','place','faction','item')),
            name TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, summary TEXT NOT NULL,
            entity_ids TEXT NOT NULL DEFAULT '[]',
            source TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll','operator','llm')));
        """
    )
    conn.execute(
        "INSERT INTO entities(id,kind,name,created_at,updated_at) VALUES (1,'npc','Harry',0,0)"
    )
    conn.execute(
        "INSERT INTO entities(id,kind,name,created_at,updated_at) VALUES (2,'npc','Toby',0,0)"
    )
    conn.execute(
        "INSERT INTO events(id,ts,summary,entity_ids,source) VALUES (1,0,'a',?, 'avatar')",
        (json.dumps([1, 2, 77777]),),
    )
    conn.execute(
        "INSERT INTO events(id,ts,summary,entity_ids,source) VALUES (2,0,'b',?, 'loom')",
        (json.dumps("not-a-list"),),
    )
    conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (2, ?)", (time.time(),))
    conn.commit()
    conn.close()
    return path


def test_v2_to_v3_only_backfills_and_is_idempotent(tmp_path):
    _make_v2_db(tmp_path, "#chan")
    VerseStore(tmp_path, "#chan")  # runs ONLY v2->v3 (the production path)
    store2 = VerseStore(tmp_path, "#chan")  # reopen must not double-apply
    with store2.read_connection() as conn:
        rows = conn.execute(
            "SELECT event_id, entity_id FROM event_actor ORDER BY event_id, entity_id"
        ).fetchall()
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        v3rows = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=3").fetchone()[0]
    # event 1 keeps valid existing ids 1,2; drops nonexistent 77777.
    # event 2's blob decodes to a non-list -> contributes no rows (list-guard).
    assert rows == [(1, 1), (1, 2)]
    assert ver == 3
    assert v3rows == 1  # idempotent: reopen did not re-stamp v3 nor duplicate rows
