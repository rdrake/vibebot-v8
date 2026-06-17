import sqlite3
import time
from pathlib import Path

from llm.verse.store import VerseStore, db_path_for_channel


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
    assert version == 2

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
