# Verse Universe Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let authorized users manipulate a channel's forest-verse universe (entities, places, events, relations) both manually (`@versedit`) and via an LLM tool, and — critically — make pinned canon actually reach the model on every verse turn.

**Architecture:** Thin command/tool layer over one validated mutation core (`VerseStore._apply_op_inline`). A single new capability `llm.verse.edit` gates both surfaces. A new "pinned roster" block in `build_verse_system_prompt` is the consumption layer that solves roster memory. A real schema migration widens two CHECK constraints.

**Tech Stack:** Python 3, SQLite (per-channel `VerseStore`), Limnoria plugin (`wrap()` commands), pytest. Spec: `docs/superpowers/specs/2026-06-17-verse-universe-editing-design.md`.

**Phasing (each phase ends green + committed; you can stop after Phase 2):**
- **Phase 1** — Schema migration v1→v2 (foundation; unblocks new ops/sources).
- **Phase 2** — Mutation core + pinned consumption layer (makes the roster stick).
- **Phase 3** — `@versedit` operator commands.
- **Phase 4** — `verse_edit` LLM tool + per-user gate.

**Conventions:** Verse store tests are pure-SQLite (construct `VerseStore(tmp_path, "#chan")`, no IRC). Run a single test with `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest <path>::<name> -v`. Run lint with `make lint && make typecheck`. Commit after every green step.

---

## Phase 1 — Schema migration v1 → v2

### Task 1: Widen CHECK constraints in the fresh-install DDL

**Files:**
- Modify: `plugins/llm/src/llm/verse/schema.sql:43` and `:65`

- [ ] **Step 1: Edit the `events.source` CHECK** (schema.sql:43)

Replace:
```sql
    source     TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll'))
```
with:
```sql
    source     TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll','operator','llm'))
```

- [ ] **Step 2: Edit the `proposals.op` CHECK** (schema.sql:65)

Replace:
```sql
    op          TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity')),
```
with:
```sql
    op          TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity','crosspoll_seed','update_entity','set_status','edit_event','delete_event','delete_relation','set_pinned')),
```
(`crosspoll_seed` is added too — it was a valid loom op but missing from the CHECK; a latent bug.)

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/src/llm/verse/schema.sql
git commit -m "feat(verse): widen events.source and proposals.op CHECKs for new ops"
```

### Task 2: Add versioned migration to `_migrate`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py:94` (SCHEMA_VERSION), `:159-170` (`_migrate`)
- Test: `tests/verse/test_store_migration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_store_migration.py`:
```python
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

    # New source + new op must now insert without IntegrityError.
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_migration.py -v`
Expected: FAIL — `test_migration_v1_to_v2_widens_checks` asserts version 2 but `_migrate` never upgrades (stays 1), and inserting `source='operator'` raises `IntegrityError` against the still-legacy table.

- [ ] **Step 3: Bump SCHEMA_VERSION** (store.py:94)

Replace `SCHEMA_VERSION = 1` with `SCHEMA_VERSION = 2`.

- [ ] **Step 4: Add the upgrade step to `_migrate`** (store.py:159)

Replace the body of `_migrate` with:
```python
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

    def _upgrade_v1_to_v2(self) -> None:
        """Rebuild events + proposals with widened CHECK constraints.

        SQLite cannot ALTER ... DROP CONSTRAINT, so use the 12-step
        table-rebuild: create _new with the v2 CHECK, copy rows, drop old,
        rename. Idempotent: gated on schema_version < 2 by the caller and
        stamps version 2 in the same transaction.
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
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_migration.py -v`
Expected: PASS (both tests).

- [ ] **Step 6: Run the full verse suite to confirm no regression** (fresh stores now stamp v2)

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse -v`
Expected: PASS.

- [ ] **Step 7: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/store.py tests/verse/test_store_migration.py
git commit -m "feat(verse): versioned schema migration v1->v2 (rebuild events/proposals)"
```

---

## Phase 2 — Mutation core + pinned consumption layer

### Task 3: Validated-source privilege + collapse the second dispatcher

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `_apply_op_inline` (~946), `apply_proposal` (~1098)
- Test: `tests/verse/test_store_privilege.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_store_privilege.py`:
```python
import pytest

from llm.verse.store import VerseStore

VALID_SOURCES = {"operator", "loom", "llm", "crosspoll", "avatar"}


def _store(tmp_path):
    return VerseStore(tmp_path, "#chan")


def test_invalid_source_rejected(tmp_path):
    store = _store(tmp_path)
    with store.write_transaction() as conn:
        with pytest.raises(ValueError, match="source"):
            store._apply_op_inline(conn, op="add_event", payload={"summary": "x", "entity_ids": []}, source="bogus")


def test_destructive_op_blocked_for_non_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="loom")
    with store.write_transaction() as conn:
        with pytest.raises(PermissionError):
            store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="llm")


def test_destructive_op_allowed_for_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="operator")
    with store.write_transaction() as conn:
        store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="operator")
    assert store.recent_events(limit=10) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_privilege.py -v`
Expected: FAIL — `_apply_op_inline` doesn't validate `source`, has no `delete_event` op, raising `ValueError("unknown op")` instead of `PermissionError`, and `source='operator'` violates the (now-migrated, but in a fresh store still-fine) constraint? No — fresh store is v2, so the failure is the missing op/validation.

- [ ] **Step 3: Add source validation + privilege at the top of `_apply_op_inline`** (store.py:946)

Add module constant near `_RESERVED_ATTRIBUTE_KEYS` (store.py:28):
```python
_VALID_SOURCES = frozenset({"operator", "loom", "llm", "crosspoll", "avatar"})
_DESTRUCTIVE_OPS = frozenset({"delete_event", "delete_relation", "set_status", "set_pinned"})
```
At the very start of `_apply_op_inline` body (after `now = time.time()`):
```python
        if source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        privileged = source == "operator"
        if op in _DESTRUCTIVE_OPS and not privileged:
            raise PermissionError(f"op {op!r} requires operator privilege")
```

- [ ] **Step 4: Add the new op branches** to `_apply_op_inline`, immediately before the final `raise ValueError(f"unknown op: {op!r}")`:
```python
        if op == "update_entity":
            eid = payload["entity_id"]
            if "kind" in payload:
                raise ValueError("update_entity cannot change kind")
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise LookupError(f"entity_id {eid} does not exist")
            sets, args = [], []
            if "name" in payload:
                sets.append("name=?"); args.append(payload["name"])
            if "summary" in payload:
                sets.append("summary=?"); args.append(payload["summary"])
            if not sets:
                raise ValueError("update_entity needs name and/or summary")
            sets.append("updated_at=?"); args.append(now); args.append(eid)
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
                # Clear the avatar link atomically so the user is not bricked
                # (record_user_event raises on a retired-but-linked actor).
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
```
Also add `"pinned"` to `_RESERVED_ATTRIBUTE_KEYS` (store.py:28) so `set_attribute` can never set it — only `set_pinned` can:
```python
_RESERVED_ATTRIBUTE_KEYS = frozenset({"last_seen_ts", "auto_created", "status", "kind", "location", "pinned"})
```

- [ ] **Step 5: Collapse `apply_proposal` onto the core** (store.py:1098)

Replace the whole body of `apply_proposal` with a single-dispatch delegation:
```python
    def apply_proposal(
        self,
        *,
        op: str,
        payload: dict[str, Any],
        source: str = "loom",
    ) -> int | None:
        """Convert a proposal payload into rows via the single core dispatcher."""
        with self.write_transaction() as conn:
            return self._apply_op_inline(conn, op=op, payload=payload, source=source)
```

- [ ] **Step 6: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_privilege.py -v`
Expected: PASS.

- [ ] **Step 7: Run full verse suite (the `apply_proposal` collapse touches the loom path)**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse -v`
Expected: PASS. If any loom test asserted `apply_proposal` returned without opening a transaction, fix the test to the new behavior (it now owns its txn).

- [ ] **Step 8: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/store.py tests/verse/test_store_privilege.py
git commit -m "feat(verse): new core ops + validated-source privilege; single dispatcher"
```

### Task 4: `apply_direct` audit helper

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (add method near `apply_and_record_proposal` ~1026)
- Test: `tests/verse/test_store_apply_direct.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_store_apply_direct.py`:
```python
from llm.verse.store import VerseStore


def test_apply_direct_applies_and_audits(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    new_id = store.apply_direct(
        op="add_entity",
        payload={"kind": "npc", "name": "Archie", "summary": "stinky"},
        source="operator",
        provenance="@versedit",
    )
    assert store.get_entity(new_id).name == "Archie"
    # An approved audit proposal row exists.
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT op, status, provenance FROM proposals ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row == ("add_entity", "approved", "@versedit")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_apply_direct.py -v`
Expected: FAIL — `AttributeError: 'VerseStore' object has no attribute 'apply_direct'`.

- [ ] **Step 3: Implement `apply_direct`** (after `apply_and_record_proposal`, ~store.py:1063)
```python
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
        (source='llm'). Unlike apply_and_record_proposal this carries no
        loom ceremony (cycle_id/confidence/reviewer are synthesized for audit
        only). Returns the new row id for creating ops, else None.
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_apply_direct.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/store.py tests/verse/test_store_apply_direct.py
git commit -m "feat(verse): apply_direct — immediate apply + approved audit row"
```

### Task 5: Pinned-roster store helpers + name-collision guard

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (add `list_pinned_entities`, `active_name_exists`)
- Test: `tests/verse/test_store_pinned.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_store_pinned.py`:
```python
from llm.verse.store import VerseStore


def test_list_pinned_returns_only_active_pinned(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    a = store.add_entity("npc", "Archie", "stinky")
    b = store.add_entity("npc", "Bob", "plain")
    store.apply_direct(op="set_pinned", payload={"entity_id": a, "pinned": True},
                       source="operator", provenance="t")
    pinned = store.list_pinned_entities()
    assert [e.name for e in pinned] == ["Archie"]
    # Retiring drops it from the pinned list.
    store.apply_direct(op="set_status", payload={"entity_id": a, "status": "retired"},
                       source="operator", provenance="t")
    assert store.list_pinned_entities() == []


def test_active_name_exists(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Archie")
    assert store.active_name_exists("archie") is True
    assert store.active_name_exists("nobody") is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_pinned.py -v`
Expected: FAIL — `AttributeError: ... 'list_pinned_entities'`.

- [ ] **Step 3: Implement the helpers** (store.py, near `list_entities_by_kind` ~300)
```python
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

    def active_name_exists(self, name: str) -> bool:
        """True if some active entity already has this name (case-insensitive)."""
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM entities WHERE LOWER(name)=LOWER(?) AND status='active' LIMIT 1",
                (name,),
            ).fetchone()
        return row is not None
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_pinned.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/store.py tests/verse/test_store_pinned.py
git commit -m "feat(verse): list_pinned_entities + active_name_exists helpers"
```

### Task 6: Roster block in `build_verse_system_prompt` (the consumption layer)

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py:428-520`
- Modify: `plugins/llm/src/llm/config.py` (add `verseRosterMaxChars`)
- Test: `tests/verse/test_verse_prompt_roster.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_verse_prompt_roster.py`:
```python
from llm.verse.avatar import build_verse_system_prompt
from llm.verse.store import VerseStore


def test_pinned_entities_appear_in_prompt(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.apply_direct(op="set_pinned", payload={"entity_id": archie, "pinned": True},
                       source="operator", provenance="t")
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" in prompt
    assert "Assgas Archie: Y11 windbag" in prompt


def test_roster_omitted_when_none_pinned(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" not in prompt


def test_roster_respects_char_cap(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    for i in range(30):
        e = store.add_entity("npc", f"Lad{i:02d}", "x" * 40)
        store.apply_direct(op="set_pinned", payload={"entity_id": e, "pinned": True},
                           source="operator", provenance="t")
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=200)
    roster = prompt.split("Established characters in this world:")[1]
    assert len(roster) <= 260  # cap + the truncation marker line
    assert "(roster truncated)" in roster
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_verse_prompt_roster.py -v`
Expected: FAIL — `build_verse_system_prompt` has no `roster_max_chars` parameter (TypeError) and emits no roster block.

- [ ] **Step 3: Add the `roster_max_chars` param + roster block** (avatar.py:428)

Change the signature:
```python
def build_verse_system_prompt(
    store: VerseStore,
    avatar_id: int,
    instruct_text: str,
    roster_max_chars: int = 600,
) -> str:
```
Before the final `parts = [...]` assignment (avatar.py:511), build the roster block:
```python
    # --- Established (pinned) characters — durable canon every turn ---
    pinned = store.list_pinned_entities()
    roster_lines: list[str] = []
    if pinned:
        used = 0
        truncated = False
        for e in pinned:
            line = f"- {e.name}: {e.summary}" if e.summary else f"- {e.name}"
            if used + len(line) + 1 > roster_max_chars:
                truncated = True
                break
            roster_lines.append(line)
            used += len(line) + 1
        if truncated:
            roster_lines.append("- (roster truncated)")
```
Then change the `parts` list to append the roster block when present:
```python
    parts = [
        identity_line,
        persona_line,
        scene_line,
        events_header,
        event_bullets,
        others_header,
        other_bullets,
    ]
    if roster_lines:
        parts.append("Established characters in this world:")
        parts.extend(roster_lines)
    return "\n".join(parts)
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_verse_prompt_roster.py -v`
Expected: PASS.

- [ ] **Step 5: Add the `verseRosterMaxChars` registry key** (config.py)

Find the verse config block (the `verseAutoEntityMaxNamesPerCall` entry, config.py:370) and add immediately after it, following the exact surrounding `registerChannelValue`/`conf` idiom used there:
```python
    "verseRosterMaxChars",
    registry.PositiveInteger(
        600,
        _("""Max characters of the pinned-roster block injected into every
        verse system prompt. Caps context cost for large rosters; pinned
        entities beyond the cap are dropped with a (roster truncated) marker."""),
    ),
```
(Match the registration mechanism of the adjacent keys exactly — same `registerChannelValue` wrapper, same group.)

- [ ] **Step 6: Wire the registry value into the verse call site**

Find where `build_verse_system_prompt` is called (`plugin.py:2642`, per the spec). Pass the channel's configured cap:
```python
        verse_system = build_verse_system_prompt(
            store,
            avatar_id,
            instruct_text,
            roster_max_chars=self.registryValue("verseRosterMaxChars", channel),
        )
```
(Adjust the surrounding variable names to the actual call site; the only change is adding the `roster_max_chars=` argument.)

- [ ] **Step 7: Run verse suite + lint**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse -v && make lint && make typecheck`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/src/llm/config.py plugins/llm/src/llm/plugin.py tests/verse/test_verse_prompt_roster.py
git commit -m "feat(verse): pinned-roster block in verse prompt (consumption layer)"
```

**Phase 2 done — pinned canon now reaches the model every turn. You can stop here and seed the 15 lads via `apply_direct` if you don't need the command/tool surface yet.**

---

## Phase 3 — `@versedit` operator commands

### Task 7: `llm.verse.edit` capability + `_resolve_ref` helper

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:91` (capability set)
- Modify: `plugins/llm/src/llm/verse/store.py` (`resolve_ref` wrapper using the in-txn resolver)
- Test: `tests/verse/test_store_resolve_ref.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_store_resolve_ref.py`:
```python
import pytest

from llm.verse.store import VerseStore


def test_resolve_ref_by_hash_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Bob")
    assert store.resolve_ref("#%d" % eid) == eid


def test_resolve_ref_by_name(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Bob")
    assert store.resolve_ref("Bob") == eid


def test_resolve_ref_numeric_name_is_not_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "7")  # literally named "7"
    assert store.resolve_ref("7") == eid          # name, not id
    assert store.resolve_ref("#%d" % eid) == eid  # explicit id form


def test_resolve_ref_unknown_raises(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    with pytest.raises(LookupError):
        store.resolve_ref("ghost")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_resolve_ref.py -v`
Expected: FAIL — no `resolve_ref` method.

- [ ] **Step 3: Implement `resolve_ref`** (store.py, near `find_active_entity_by_name` ~277)
```python
    def resolve_ref(self, ref: str) -> int:
        """Resolve an operator <ref> to an entity id.

        '#<int>' is always an id; anything else is a name (so an entity
        literally named '7' is addressable). Raises LookupError if unknown
        or the name is ambiguous-by-collision (active duplicates).
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
```

- [ ] **Step 4: Add the capability to the default set** (plugin.py:91)

Replace:
```python
    {"llm.ask", "llm.code", "llm.draw", "llm.verse", "llm.verse.gm", "owner", "admin", "trusted"}
```
with:
```python
    {"llm.ask", "llm.code", "llm.draw", "llm.verse", "llm.verse.gm", "llm.verse.edit", "owner", "admin", "trusted"}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_store_resolve_ref.py -v`
Expected: PASS.

- [ ] **Step 6: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/store.py plugins/llm/src/llm/plugin.py tests/verse/test_store_resolve_ref.py
git commit -m "feat(verse): resolve_ref (#id-or-name) + llm.verse.edit capability"
```

### Task 8: `@versedit` dispatcher command (add/pin/set/name/desc)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (new `versedit` method + registration near `versedump` ~5795 and the command-registration block ~357)
- Test: `tests/test_versedit_command.py`

> **Pattern note:** mirror `versedump` — register with `wrap(versedit, [("checkCapability","llm.verse.edit"), many("anything"), optional("channel")])`. The capability-in-wrap form (used by `verseapprove` at `plugin.py:6184`) evaluates `llm.verse.edit` against the **command's channel** (the target). Resolve the per-channel `VerseStore` via the same helper `versedump` uses (`self._get_or_create_verse_store(channel)` / the store accessor in that method). Parse the leading verb token in the body.

- [ ] **Step 1: Write the failing test**

Create `tests/test_versedit_command.py` (mirror the harness other command tests use — `PluginTestCase` / `feedMsg` per existing `tests/`; if the repo uses a verse-command fixture, follow it). Minimal behavioral test through the store the command writes to:
```python
from llm.verse.store import VerseStore


def test_versedit_add_then_pin(tmp_path):
    """Unit-level proxy for the dispatcher's add+pin handlers.

    The dispatcher body delegates to these store calls; this asserts the
    contract the command relies on. (A full feedMsg integration test lives
    alongside the other plugin command tests.)
    """
    store = VerseStore(tmp_path, "#chan")
    new_id = store.apply_direct(
        op="add_entity",
        payload={"kind": "npc", "name": "Archie", "summary": "stinky"},
        source="operator", provenance="@versedit add",
    )
    store.apply_direct(op="set_pinned", payload={"entity_id": new_id, "pinned": True},
                       source="operator", provenance="@versedit pin")
    assert [e.name for e in store.list_pinned_entities()] == ["Archie"]
```

- [ ] **Step 2: Run to verify it passes against the store contract**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/test_versedit_command.py -v`
Expected: PASS (this guards the store contract the dispatcher uses).

- [ ] **Step 3: Add the `versedit` method** (plugin.py, near `versedump` ~5795)
```python
    def versedit(self, irc, msg, args, rest, channel):
        """<verb> <args...> [#channel]

        Edit the verse universe. verbs: add, pin, unpin, set, name, desc,
        retire, restore, relate, unrelate, event, editevent, delevent, show.
        Requires the llm.verse.edit capability (checked against the target
        channel by wrap).
        """
        channel = channel or msg.channel
        if not channel:
            irc.error("Specify a channel.", prefixNick=False)
            return
        store = self._get_or_create_verse_store(channel)
        tokens = rest.split(None, 1)
        verb = tokens[0].lower() if tokens else ""
        body = tokens[1] if len(tokens) > 1 else ""
        try:
            reply = self._versedit_dispatch(store, verb, body)
        except (LookupError, ValueError, PermissionError) as exc:
            irc.error(str(exc), prefixNick=False)
            return
        irc.reply(reply, prefixNick=False)

    def _versedit_dispatch(self, store, verb, body):
        if verb == "add":
            # "<kind> <name>[ :: summary]" — kind is the first token.
            kind_rest = body.split(None, 1)
            if len(kind_rest) < 2:
                raise ValueError("usage: versedit add <kind> <name> [:: summary]")
            kind, name_part = kind_rest[0], kind_rest[1]
            name, summary = (name_part.split("::", 1) + [""])[:2]
            name, summary = name.strip(), summary.strip()
            if kind not in ("avatar", "npc", "place", "faction", "item"):
                raise ValueError("kind must be avatar|npc|place|faction|item")
            if store.active_name_exists(name):
                raise ValueError(f"an active entity named {name!r} already exists")
            new_id = store.apply_direct(
                op="add_entity", payload={"kind": kind, "name": name, "summary": summary},
                source="operator", provenance="@versedit add",
            )
            return f"added {kind} #{new_id}: {name}"
        if verb in ("pin", "unpin"):
            eid = store.resolve_ref(body.strip())
            store.apply_direct(op="set_pinned", payload={"entity_id": eid, "pinned": verb == "pin"},
                               source="operator", provenance=f"@versedit {verb}")
            return f"{verb}ned #{eid}"
        if verb == "set":
            parts = body.split(None, 2)
            if len(parts) < 3:
                raise ValueError("usage: versedit set <ref> <key> <value>")
            eid = store.resolve_ref(parts[0])
            store.apply_direct(op="set_attribute", payload={"entity_id": eid, "key": parts[1], "value": parts[2]},
                               source="operator", provenance="@versedit set")
            return f"set {parts[1]} on #{eid}"
        if verb == "name":
            ref, _, newname = body.partition(" ")
            newname = newname.strip()
            if not newname:
                raise ValueError("usage: versedit name <ref> <new-name>")
            eid = store.resolve_ref(ref)
            if store.active_name_exists(newname):
                raise ValueError(f"an active entity named {newname!r} already exists")
            store.apply_direct(op="update_entity", payload={"entity_id": eid, "name": newname},
                               source="operator", provenance="@versedit name")
            return f"renamed #{eid} -> {newname}"
        if verb == "desc":
            ref, _, summary = body.partition("::")
            eid = store.resolve_ref(ref.strip())
            store.apply_direct(op="update_entity", payload={"entity_id": eid, "summary": summary.strip()},
                               source="operator", provenance="@versedit desc")
            return f"updated summary of #{eid}"
        return self._versedit_dispatch_2(store, verb, body)
```

- [ ] **Step 4: Register the command** (plugin.py, in the command-metadata block near `versedump` registration ~357, and add the `wrap` at the end of the class body where the other `versedump = wrap(...)` lines live ~5920)
```python
    versedit = wrap(
        versedit,
        [("checkCapability", "llm.verse.edit"), "text", optional("channel")],
    )
```
Add a help/metadata entry mirroring the `versedump` entry at plugin.py:357 (same dict/list structure), description: `"Edit the verse universe (add/pin/set/name/desc/retire/restore/relate/event/...). Requires the llm.verse.edit capability."`

- [ ] **Step 5: Run + lint**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/test_versedit_command.py tests/verse -v && make lint && make typecheck`
Expected: PASS. (`_versedit_dispatch_2` is added in Task 9; until then the verbs it handles return an AttributeError — acceptable mid-phase, but to keep green, add a stub now:)
```python
    def _versedit_dispatch_2(self, store, verb, body):
        raise ValueError(f"unknown verb: {verb!r}")
```

- [ ] **Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py tests/test_versedit_command.py
git commit -m "feat(verse): @versedit dispatcher (add/pin/set/name/desc)"
```

### Task 9: `@versedit` remaining verbs (retire/restore/relate/unrelate/event/editevent/delevent/show)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (replace the `_versedit_dispatch_2` stub)
- Test: `tests/test_versedit_command.py` (extend)

- [ ] **Step 1: Extend the test**

Append to `tests/test_versedit_command.py`:
```python
def test_versedit_event_and_delete(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    a = store.add_entity("npc", "Archie")
    ev = store.apply_direct(op="add_event", payload={"summary": "Archie parps", "entity_ids": [a]},
                            source="operator", provenance="t")
    assert store.recent_events(limit=5)[0].summary == "Archie parps"
    store.apply_direct(op="delete_event", payload={"event_id": ev}, source="operator", provenance="t")
    assert store.recent_events(limit=5) == []


def test_versedit_retire_clears_avatar_link(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    res = store.opt_in_avatar(nick="bob", account="bob!acct", instruct_text="")
    store.apply_direct(op="set_status", payload={"entity_id": res.entity_id, "status": "retired"},
                       source="operator", provenance="t")
    assert store.find_avatar_by_nick("bob") is None
```

- [ ] **Step 2: Run to verify it passes (store contract)**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/test_versedit_command.py -v`
Expected: PASS.

- [ ] **Step 3: Replace `_versedit_dispatch_2`** with the full implementation:
```python
    def _versedit_dispatch_2(self, store, verb, body):
        if verb in ("retire", "restore"):
            eid = store.resolve_ref(body.strip())
            status = "retired" if verb == "retire" else "active"
            store.apply_direct(op="set_status", payload={"entity_id": eid, "status": status},
                               source="operator", provenance=f"@versedit {verb}")
            return f"{verb}d #{eid}"
        if verb == "relate":
            # "<ref> <kind> <ref> [:: note]"
            head, _, note = body.partition("::")
            parts = head.split()
            if len(parts) != 3:
                raise ValueError("usage: versedit relate <ref> <kind> <ref> [:: note]")
            from_id = store.resolve_ref(parts[0])
            to_id = store.resolve_ref(parts[2])
            rid = store.apply_direct(
                op="add_relation",
                payload={"from_id": from_id, "to_id": to_id, "kind": parts[1], "note": note.strip()},
                source="operator", provenance="@versedit relate",
            )
            return f"related #{from_id} -{parts[1]}-> #{to_id} (relation #{rid})"
        if verb == "unrelate":
            rid = int(body.strip())
            store.apply_direct(op="delete_relation", payload={"relation_id": rid},
                               source="operator", provenance="@versedit unrelate")
            return f"deleted relation #{rid}"
        if verb == "event":
            # "<summary> [@ id,id,...]"
            summary, _, ids_part = body.partition("@")
            entity_ids = [int(x) for x in ids_part.split(",") if x.strip().isdigit()] if ids_part else []
            new_id = store.apply_direct(op="add_event",
                                        payload={"summary": summary.strip(), "entity_ids": entity_ids},
                                        source="operator", provenance="@versedit event")
            return f"added event #{new_id}"
        if verb == "editevent":
            id_part, _, summary = body.partition("::")
            ev_id = int(id_part.strip())
            store.apply_direct(op="edit_event", payload={"event_id": ev_id, "summary": summary.strip()},
                               source="operator", provenance="@versedit editevent")
            return f"edited event #{ev_id}"
        if verb == "delevent":
            ev_id = int(body.strip())
            store.apply_direct(op="delete_event", payload={"event_id": ev_id},
                               source="operator", provenance="@versedit delevent")
            return f"deleted event #{ev_id}"
        if verb == "show":
            eid = store.resolve_ref(body.strip())
            ent = store.get_entity(eid)
            attrs = store.list_attributes(eid)
            return f"#{eid} {ent.kind} {ent.name} [{ent.status}] — {ent.summary} | attrs={attrs}"
        raise ValueError(f"unknown verb: {verb!r}")
```

- [ ] **Step 4: Run + lint + commit**

```bash
cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/test_versedit_command.py tests/verse -v && make lint && make typecheck
git add plugins/llm/src/llm/plugin.py tests/test_versedit_command.py
git commit -m "feat(verse): @versedit retire/restore/relate/event/show verbs"
```

---

## Phase 4 — `verse_edit` LLM tool

### Task 10: `validate_payload` extraction + new-op schemas

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py:123-199`
- Test: `tests/verse/test_loom_validate_payload.py`

- [ ] **Step 1: Write the failing test**

Create `tests/verse/test_loom_validate_payload.py`:
```python
from llm.verse.loom import validate_payload


def test_validate_payload_ok():
    assert validate_payload("add_entity", {"kind": "npc", "name": "Bob"}) is None


def test_validate_payload_missing_field():
    assert "name" in (validate_payload("add_entity", {"kind": "npc"}) or "")


def test_validate_payload_update_entity():
    assert validate_payload("update_entity", {"entity_id": 3, "summary": "x"}) is None
    assert validate_payload("update_entity", {"entity_id": "x", "summary": "y"}) is not None


def test_validate_payload_unknown_op():
    assert validate_payload("nope", {}) is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_loom_validate_payload.py -v`
Expected: FAIL — no `validate_payload`.

- [ ] **Step 3: Extend `_PAYLOAD_SCHEMA`** (loom.py:146, add entries inside the dict) and add constructive new ops:
```python
    "update_entity": (
        ("entity_id", _is_strict_int, "int"),
    ),
```
(Only `entity_id` is required; `name`/`summary` are optional and validated in the core. `set_status`/`delete_*` are NOT added here — they are operator-only and never reach this validator.)

- [ ] **Step 4: Extract `validate_payload`** (loom.py, after `_PAYLOAD_SCHEMA`)
```python
def validate_payload(op: str, payload: dict[str, Any]) -> str | None:
    """Return None if *payload* is valid for *op*, else a human reason string.

    Shared by parse_digest (loom) and the verse_edit tool so one schema
    governs both. Only constructive ops have entries; an op without a schema
    entry is rejected.
    """
    schema = _PAYLOAD_SCHEMA.get(op)
    if schema is None:
        return f"unknown or non-constructive op: {op!r}"
    for key, predicate, label in schema:
        if key not in payload:
            return f"missing {key}"
        if not predicate(payload[key]):
            return f"{key} not {label}"
    return None
```

- [ ] **Step 5: Refactor `parse_digest` to call it** (loom.py:189-199) — replace the inline predicate loop with:
```python
        bad_field = validate_payload(op, payload)
        if bad_field is not None:
            log.warning("loom proposal %d %s; dropped", i, bad_field)
            continue
```

- [ ] **Step 6: Run loom tests + new test**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_loom_validate_payload.py tests/verse -v -k "loom or validate"`
Expected: PASS (parse_digest behavior unchanged; new helper covered).

- [ ] **Step 7: Lint + commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/loom.py tests/verse/test_loom_validate_payload.py
git commit -m "refactor(verse): extract validate_payload; share loom/tool validation"
```

### Task 11: `verse_edit` tool — schema, gate, dispatch

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (tool schema list + dispatch, near `verse_storybook` ~3782/4012) or `verse/avatar.py` verse-tool dispatch (~524) — follow wherever `verse_record` is defined and dispatched.
- Modify: `plugins/llm/src/llm/profile.py` (add `verse_edit` to the verse profile's tool set, alongside `verse_record`)
- Test: `tests/verse/test_verse_edit_tool.py`

> **Integration note:** the triggering user's account is already threaded into the completion path (`service.py` account resolver `:224`, `account` field `:425/:449/:1875`). The dispatch must receive that account (or the invoking `msg.prefix`) so it can check `llm.verse.edit`. Use the same `ircdb.checkCapability(prefix_or_account, "llm.verse.edit")` form the commands use. Thread the account through to the verse-tool executor if not already present.

- [ ] **Step 1: Write the failing test** (pure dispatch-function test, no live IRC)

Create `tests/verse/test_verse_edit_tool.py`:
```python
from llm.verse.store import VerseStore
from llm.verse.avatar import dispatch_verse_edit  # new helper


def test_verse_edit_unauthorized_is_noop(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store, op="add_entity", payload={"kind": "npc", "name": "X"},
        authorized=False, account="nobody",
    )
    assert result["status"] == "refused"
    assert store.list_entities_by_kind("npc") == []


def test_verse_edit_authorized_applies(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store, op="add_entity", payload={"kind": "npc", "name": "Archie"},
        authorized=True, account="gm!acct",
    )
    assert result["status"] == "ok"
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Archie"]


def test_verse_edit_rejects_destructive_op(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store, op="delete_event", payload={"event_id": 1},
        authorized=True, account="gm!acct",
    )
    assert result["status"] == "error"  # not a constructive op / no schema
    assert "op" in result["detail"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_verse_edit_tool.py -v`
Expected: FAIL — no `dispatch_verse_edit`.

- [ ] **Step 3: Implement `dispatch_verse_edit`** (avatar.py, in the verse tool-dispatch section ~524)
```python
from .loom import validate_payload  # add to imports at top of avatar.py

_VERSE_EDIT_OPS = frozenset({"add_entity", "add_event", "set_attribute", "add_relation", "update_entity"})


def dispatch_verse_edit(store, *, op, payload, authorized, account):
    """Execute a verse_edit tool call. Constructive ops only; gated.

    Returns a JSON-able dict: {status: ok|refused|error, detail/id}.
    """
    if not authorized:
        return {"status": "refused", "detail": "not authorized to edit canon (needs llm.verse.edit)"}
    if op not in _VERSE_EDIT_OPS:
        return {"status": "error", "detail": f"op {op!r} not permitted via verse_edit"}
    reason = validate_payload(op, payload)
    if reason is not None:
        return {"status": "error", "detail": reason}
    try:
        new_id = store.apply_direct(op=op, payload=payload, source="llm", provenance=f"verse_edit:{account}")
    except (LookupError, ValueError, PermissionError) as exc:
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "id": new_id}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse/test_verse_edit_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Register the tool schema + wire dispatch.** Add a `verse_edit` entry to the verse tool schema list wherever `verse_record` is declared (mirror its `{"type":"function","function":{"name":..., "parameters": {...}}}` shape), with parameters:
```json
{
  "type": "object",
  "properties": {
    "op": {"type": "string", "enum": ["add_entity", "add_event", "set_attribute", "add_relation", "update_entity"]},
    "payload": {"type": "object"}
  },
  "required": ["op", "payload"]
}
```
In the verse tool executor (where `verse_record` tool calls are handled), add a branch for `verse_edit` that resolves `authorized = ircdb.checkCapability(invoking_prefix, "llm.verse.edit")` and calls `dispatch_verse_edit(store, op=args["op"], payload=args["payload"], authorized=authorized, account=invoking_account)`, returning the dict as the tool result. Add `verse_edit` to the verse profile's tool set in `profile.py` next to `verse_record`.

- [ ] **Step 6: Run full suite + lint**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/verse tests/test_versedit_command.py -v && make lint && make typecheck`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/src/llm/service.py plugins/llm/src/llm/profile.py tests/verse/test_verse_edit_tool.py
git commit -m "feat(verse): verse_edit LLM tool — constructive ops, per-user llm.verse.edit gate"
```

---

## Final verification

- [ ] **Full suite:** `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest tests/ -q`
- [ ] **Lint/type:** `make lint && make typecheck`
- [ ] **Manual smoke (optional, on a dev bot):** `@versedit add npc "Assgas Archie" :: Y11 windbag` → `@versedit pin Assgas Archie` → start a verse turn → confirm "Established characters in this world:" lists Archie.
- [ ] **Seed the 15 lads** (the original goal) via `@versedit add` + `@versedit pin` for each, or a one-off `apply_direct` loop.

## Spec coverage map

| Spec component | Tasks |
|----------------|-------|
| 1 — Consumption (pinned roster) | 5, 6 |
| 2 — Mutation core (ops, privilege, single dispatcher, validate_payload, apply_direct) | 3, 4, 10 |
| 3 — Operator commands `@versedit` | 7, 8, 9 |
| 4 — `verse_edit` tool + per-user gate | 10, 11 |
| 5 — Schema migration | 1, 2 |
| Soft-delete coherence (avatar link) | 3 (set_status), 9 (test) |
| Name-uniqueness / `#id` rule | 5, 7, 8 |
