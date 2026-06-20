# Verse Retention Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the verse model reliably remember canon — inject the full author-locked/pinned roster plus the scene-relevant cast, their relations, and their recent events into every verse turn, and let the conversational author lock canon just by talking.

**Architecture:** Minimal, in-place changes to the existing `llm` plugin's verse path (no new plugin, no data port). One additive SQLite migration (v2→v3) adds an `entity_alias` table and an `event_actor` join table. `build_verse_system_prompt` is reordered stable-first (canon roster in the cacheable prefix; all volatile/retrieved context after) and enriched with message-matched cast + 1-hop relations + active-only scene events. A `msg.prefix`-bound `verse_record` handler overlay promotes human-offered, reinforced names to `author_locked`. chat/code/draw are untouched.

**Tech Stack:** Python 3, Limnoria/Supybot plugin, SQLite (WAL, schema.sql + `schema_version` table), pytest (real SQLite fixtures, no mocks), litellm. Run tests with `cd plugins/llm && uv run pytest <path> -v` (or `make test`). Lint/typecheck: `cd plugins/llm && make lint && make typecheck`.

**Key files:**
- `plugins/llm/src/llm/verse/schema.sql` — additive DDL (new tables/indexes)
- `plugins/llm/src/llm/verse/store.py` — migration, new store methods, retrieval
- `plugins/llm/src/llm/verse/avatar.py` — `build_verse_system_prompt` reorder + retrieval injection
- `plugins/llm/src/llm/verse/aging.py` — extend pinned-exemption to `author_locked`
- `plugins/llm/src/llm/plugin.py` — thread message text into `_verse_route_for`; `verse_record` promotion overlay; `@canon` command; storybook canon-write; verseModel warning
- `plugins/llm/src/llm/config.py` — new channel knobs
- `plugins/llm/tests/verse/` — tests

**Conventions to follow (from the codebase):**
- Every mutator has a private `_<name>_inline(self, conn, ...)` twin that takes a caller-owned `conn`; the public method wraps it in `with self.write_transaction() as conn:`. `self._lock` is NOT reentrant — never call a public store method from inside an open `write_transaction`.
- Lifecycle flags are EAV rows in `attributes(entity_id, key, value)` (TEXT values), not columns. `pinned='1'`, `last_seen_ts='<float>'`, `auto_created='1'`, `location='<place>'`. Engine-only keys are in `_RESERVED_ATTRIBUTE_KEYS`.
- Additive migration: add `CREATE TABLE/INDEX IF NOT EXISTS` to `schema.sql` (fresh DBs), bump `SCHEMA_VERSION`, add `if current < N: self._upgrade_v(N-1)_to_v(N)()` in `_migrate`, and a new `_upgrade_*` method running in one `write_transaction` that ends by stamping `schema_version`.
- Tests use real SQLite (no mocks), class-grouped, `verse_db_dir`/`store` fixtures, assert on `Entity(...)` dataclass attrs. Prompt tests split the returned string on a header marker and assert on the segment.

---

## Task 1: Additive migration — `entity_alias` + `event_actor` tables (v2→v3)

**Files:**
- Modify: `plugins/llm/src/llm/verse/schema.sql` (append new tables/indexes)
- Modify: `plugins/llm/src/llm/verse/store.py:99` (`SCHEMA_VERSION = 2` → `3`), `:178-179` (`_migrate` dispatch), add `_upgrade_v2_to_v3` after `_upgrade_v1_to_v2` (`:225`)
- Test: `plugins/llm/tests/verse/test_store_migration.py`

- [ ] **Step 1: Write the failing test**

```python
# append to plugins/llm/tests/verse/test_store_migration.py
import sqlite3
import json
from llm.verse.store import VerseStore, SCHEMA_VERSION


def test_v3_tables_exist_on_fresh_db(tmp_path):
    store = VerseStore(str(tmp_path / "v.db"))
    with store.read_connection() as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"entity_alias", "event_actor"} <= names
    with store.read_connection() as conn:
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver == SCHEMA_VERSION == 3


def test_v2_db_upgrades_and_backfills_event_actor(tmp_path):
    # Build a v2 DB by hand: one entity, one event referencing it + a garbage id.
    path = str(tmp_path / "v2.db")
    raw = sqlite3.connect(path)
    raw.executescript(
        "CREATE TABLE schema_version(version INTEGER NOT NULL, applied_at REAL NOT NULL);"
        "CREATE TABLE entities(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,"
        " name TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active',"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL);"
        "CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " summary TEXT NOT NULL, entity_ids TEXT NOT NULL DEFAULT '[]', source TEXT NOT NULL);"
    )
    raw.execute("INSERT INTO entities(id,kind,name,created_at,updated_at) VALUES (1,'npc','Harry',0,0)")
    raw.execute("INSERT INTO events(id,ts,summary,entity_ids,source) VALUES (1,0,'x',?, 'avatar')",
                (json.dumps([1, 99999, "garbage"]),))
    raw.execute("INSERT INTO schema_version(version, applied_at) VALUES (2, 0)")
    raw.commit()
    raw.close()

    store = VerseStore(path)  # opening runs _migrate
    with store.read_connection() as conn:
        rows = conn.execute("SELECT event_id, entity_id FROM event_actor ORDER BY entity_id").fetchall()
    # Element-wise tolerant: keep valid existing id 1, drop 99999 (no such entity) and 'garbage'.
    assert rows == [(1, 1)]
    with store.read_connection() as conn:
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert ver == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store_migration.py -v -k "v3 or backfill"`
Expected: FAIL (`no such table: event_actor`, and `SCHEMA_VERSION == 2`).

- [ ] **Step 3: Append DDL to `schema.sql`**

Append to `plugins/llm/src/llm/verse/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS entity_alias (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_entity_alias_alias ON entity_alias(alias COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS event_actor (
    event_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_event_actor_entity ON event_actor(entity_id, event_id);
```

- [ ] **Step 4: Bump `SCHEMA_VERSION` and add the migration branch + method**

In `store.py:99` change `SCHEMA_VERSION = 2` to `SCHEMA_VERSION = 3`.

In `_migrate` (after `if current < 2: self._upgrade_v1_to_v2()`) add:

```python
        if current < 3:
            self._upgrade_v2_to_v3()
```

Add this method directly after `_upgrade_v1_to_v2`:

```python
    def _upgrade_v2_to_v3(self) -> None:
        """Additive: add entity_alias + event_actor (created by executescript on
        fresh/existing DBs via CREATE IF NOT EXISTS), then backfill event_actor
        from the legacy events.entity_ids JSON blob using an ELEMENT-WISE
        tolerant decode (keep valid existing ids, drop bad elements — never the
        all-or-nothing _parse_entity_ids). Idempotent: INSERT OR IGNORE on the
        (event_id, entity_id) PK makes a re-run a no-op. Ends by stamping v3."""
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store_migration.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add plugins/llm/src/llm/verse/schema.sql plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store_migration.py
git commit -m "feat(verse): add entity_alias + event_actor tables (schema v3) with tolerant backfill"
```

---

## Task 2: Write `event_actor` on every new event

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `_add_event_inline` (`:567-587`)
- Test: `plugins/llm/tests/verse/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/verse/test_store.py (new class)
class TestEventActorWrite:
    def test_add_event_populates_event_actor_for_existing_entities(self, store):
        a = store.add_entity("npc", "Harry")
        ev = store.add_event("Harry did a thing", [a, 99999], source="avatar")
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT entity_id FROM event_actor WHERE event_id=? ORDER BY entity_id", (ev,)
            ).fetchall()
        # 99999 has no entities row -> skipped (FK-safe); a is recorded.
        assert rows == [(a,)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestEventActorWrite -v`
Expected: FAIL (no `event_actor` rows written by `add_event`).

- [ ] **Step 3: Populate `event_actor` in `_add_event_inline`**

In `_add_event_inline`, after `assert cur.lastrowid is not None` and before `return cur.lastrowid`:

```python
        event_id = cur.lastrowid
        for eid in dict.fromkeys(entity_ids):  # de-dup, preserve order
            # entity_ids is an unconstrained list; event_actor has a real FK,
            # so only link ids that actually exist (mirror the bump_last_seen
            # defensive existence check).
            if conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                    (event_id, eid),
                )
        return event_id
```

(Remove the old `return cur.lastrowid`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestEventActorWrite -v`
Expected: PASS. Also run `cd plugins/llm && uv run pytest tests/verse/test_store.py -v` to confirm no regressions.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): populate event_actor join on every new event (FK-safe)"
```

---

## Task 3: `author_locked` flag + canon roster (pinned OR author_locked)

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `_RESERVED_ATTRIBUTE_KEYS` (`:28-30`); add `set_author_locked` (+inline), `list_canon_entities`
- Test: `plugins/llm/tests/verse/test_store_pinned.py`

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/verse/test_store_pinned.py
class TestAuthorLocked:
    def test_set_author_locked_and_list_canon(self, store):
        h = store.add_entity("npc", "Harry", "year 8")
        t = store.add_entity("npc", "Toby", "year 9")
        store.set_attribute(t, "pinned", "1")          # operator pin
        store.set_author_locked(h, True)               # author lock
        canon = store.list_canon_entities()
        names = {e.name for e in canon}
        assert names == {"Harry", "Toby"}              # union of pinned + author_locked

    def test_author_locked_reserved_against_proposal_writes(self, store):
        from llm.verse.store import _RESERVED_ATTRIBUTE_KEYS
        assert "author_locked" in _RESERVED_ATTRIBUTE_KEYS

    def test_unlock_removes_from_canon(self, store):
        h = store.add_entity("npc", "Harry")
        store.set_author_locked(h, True)
        store.set_author_locked(h, False)
        assert store.list_canon_entities() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store_pinned.py::TestAuthorLocked -v`
Expected: FAIL (`set_author_locked` / `list_canon_entities` undefined).

- [ ] **Step 3: Implement**

Add `"author_locked"` to `_RESERVED_ATTRIBUTE_KEYS`:

```python
_RESERVED_ATTRIBUTE_KEYS = frozenset(
    {"last_seen_ts", "auto_created", "status", "kind", "location", "pinned", "author_locked"}
)
```

Add to `VerseStore` (place near `list_pinned_entities`):

```python
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
        """Mark/unmark an entity as author-locked durable canon (always injected,
        aging-exempt, loom-protected). Reversible."""
        with self.write_transaction() as conn:
            self._set_author_locked_inline(conn, entity_id, locked)

    def list_canon_entities(self) -> list[Entity]:
        """Active entities that are durable canon: pinned (operator) OR
        author_locked (author). Deterministic kind-then-name order so the roster
        block stays cache-stable. Superset of list_pinned_entities."""
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at "
                "FROM entities e JOIN attributes a ON a.entity_id = e.id "
                "WHERE a.key IN ('pinned','author_locked') AND a.value='1' AND e.status='active' "
                "ORDER BY CASE e.kind WHEN 'avatar' THEN 0 WHEN 'npc' THEN 1 "
                "  WHEN 'place' THEN 2 WHEN 'faction' THEN 3 ELSE 4 END, e.name COLLATE NOCASE"
            ).fetchall()
        return [Entity(*row) for row in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store_pinned.py::TestAuthorLocked -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store_pinned.py
git commit -m "feat(verse): author_locked canon flag + list_canon_entities (pinned OR author_locked)"
```

---

## Task 4: Aging + loom exempt `author_locked` entities

**Files:**
- Modify: `plugins/llm/src/llm/verse/aging.py` (locate the pinned-exemption predicate; extend to `author_locked`)
- Test: `plugins/llm/tests/verse/test_verse_aging.py`

> **Locate first:** `grep -n "pinned" plugins/llm/src/llm/verse/aging.py`. Aging already exempts `pinned='1'` entities (memory: fix f8aaede). Extend that same predicate to also exempt `author_locked='1'`. If aging selects retirement candidates via a SQL `WHERE` that excludes pinned, widen it to exclude `author_locked` too.

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/verse/test_verse_aging.py
class TestAgingExemptsAuthorLocked:
    def test_author_locked_npc_not_retired(self, store):
        # An auto-created npc that has aged out would normally retire; author_locked must save it.
        import time as _t
        h = store.add_entity("npc", "Harry")
        store.set_attribute(h, "auto_created", "1")
        store.set_attribute(h, "last_seen_ts", str(0.0))   # ancient -> aging candidate
        store.set_author_locked(h, True)
        from llm.verse.aging import age_out_entities  # adjust to the real entry-point name
        age_out_entities(store, retain_days=1, now=lambda: _t.time())
        ent = store.get_entity(h)
        assert ent.status == "active"   # author_locked saved it from retirement
```

> Adjust the import/call to aging.py's actual public function and signature (read aging.py to confirm the name, e.g. `run_aging`/`age_out_entities` and its params).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_verse_aging.py::TestAgingExemptsAuthorLocked -v`
Expected: FAIL (the author_locked npc is retired).

- [ ] **Step 3: Extend the pinned-exemption to author_locked in aging.py**

Wherever aging checks for the `pinned` exemption, broaden it. If it reads pinned via an attribute join, change the predicate to include `author_locked`. Example (adapt to the real code):

```python
        # was: WHERE a.key='pinned' AND a.value='1'
        # now: WHERE a.key IN ('pinned','author_locked') AND a.value='1'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_verse_aging.py -v`
Expected: PASS (full file, to confirm no regression to existing pinned-exemption tests).

- [ ] **Step 5: Loom-protection — locate and extend**

Run `grep -n "pinned\|_RESERVED_ATTRIBUTE_KEYS\|set_attribute" plugins/llm/src/llm/verse/loom.py`. The loom applies proposals via the store's reserved-key guard, which already rejects writes to `pinned`/`status`. Adding `author_locked` to `_RESERVED_ATTRIBUTE_KEYS` (Task 3) means the loom **cannot set or clear `author_locked`** via a `set_attribute` proposal — the protection is already structural. Add a test asserting it:

```python
# add to plugins/llm/tests/verse/test_loom.py (or test_store_privilege.py)
def test_loom_cannot_write_author_locked(store):
    h = store.add_entity("npc", "Harry")
    import pytest
    with pytest.raises(ValueError):
        # the apply path used by loom proposals for op=set_attribute
        store.apply_direct(op="set_attribute",
                           payload={"entity_id": h, "key": "author_locked", "value": "1"})
```

> Adjust `apply_direct`'s call shape to the real signature (`grep -n "def apply_direct" store.py`). The point: a reserved-key write raises `ValueError`, so the loom can't forge author-lock.

- [ ] **Step 6: Run + commit**

Run: `cd plugins/llm && uv run pytest tests/verse/test_verse_aging.py tests/verse/test_loom.py -v`

```bash
git add plugins/llm/src/llm/verse/aging.py plugins/llm/tests/verse/test_verse_aging.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse): exempt author_locked from aging; reserved-key blocks loom forging it"
```

---

## Task 5: Alias storage + name-or-alias resolution

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — add `add_alias` (+inline), `list_aliases`, `find_entity_by_name_or_alias`
- Test: `plugins/llm/tests/verse/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
class TestAliases:
    def test_add_and_resolve_alias(self, store):
        t = store.add_entity("npc", "Toby")
        store.add_alias(t, "Tobes")
        found = store.find_entity_by_name_or_alias("tobes")  # case-insensitive
        assert found is not None and found.id == t

    def test_name_takes_precedence_over_alias(self, store):
        real = store.add_entity("npc", "Tobes")
        other = store.add_entity("npc", "Toby")
        store.add_alias(other, "Tobes")
        found = store.find_entity_by_name_or_alias("Tobes")
        assert found.id == real  # exact active name wins over an alias

    def test_list_aliases(self, store):
        t = store.add_entity("npc", "Toby")
        store.add_alias(t, "Tobes")
        store.add_alias(t, "T")
        assert set(store.list_aliases(t)) == {"Tobes", "T"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestAliases -v`
Expected: FAIL (methods undefined).

- [ ] **Step 3: Implement**

```python
    def _add_alias_inline(self, conn: sqlite3.Connection, entity_id: int, alias: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO entity_alias (entity_id, alias) VALUES (?, ?)",
            (entity_id, alias),
        )

    def add_alias(self, entity_id: int, alias: str) -> None:
        """Record a nickname/alias for an entity (case-insensitive lookup)."""
        with self.write_transaction() as conn:
            self._add_alias_inline(conn, entity_id, alias)

    def list_aliases(self, entity_id: int) -> list[str]:
        with self.read_connection() as conn:
            return [r[0] for r in conn.execute(
                "SELECT alias FROM entity_alias WHERE entity_id=?", (entity_id,)
            )]

    def find_entity_by_name_or_alias(self, name: str) -> Entity | None:
        """Active-entity resolution by canonical name (precedence) then alias.
        Exact active name wins; alias is the fallback so nicknames resolve."""
        with self.read_connection() as conn:
            ent = self._find_active_entity_by_name_inline(conn, name)
            if ent is not None:
                return ent
            row = conn.execute(
                "SELECT e.id, e.kind, e.name, e.summary, e.status, e.created_at, e.updated_at "
                "FROM entities e JOIN entity_alias al ON al.entity_id = e.id "
                "WHERE al.alias = ? COLLATE NOCASE AND e.status='active' "
                "ORDER BY e.id ASC LIMIT 1",
                (name,),
            ).fetchone()
        return Entity(*row) if row else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestAliases -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): entity aliases + name-or-alias resolution"
```

---

## Task 6: Scene retrieval — match cast, 1-hop relations, active-only scene events

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — add `match_entities_in_text`, `relations_for`, `events_for_entities`
- Test: `plugins/llm/tests/verse/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
class TestSceneRetrieval:
    def test_match_entities_in_text_by_name_and_alias(self, store):
        h = store.add_entity("npc", "Harry")
        t = store.add_entity("npc", "Toby")
        store.add_alias(t, "Tobes")
        store.add_entity("npc", "Andrew")  # not mentioned
        got = {e.id for e in store.match_entities_in_text("did Harry and Tobes fight?")}
        assert got == {h, t}

    def test_relations_for_one_hop(self, store):
        h = store.add_entity("npc", "Harry")
        t = store.add_entity("npc", "Toby")
        store.add_relation(h, t, "rival_of", "since year 7")
        rels = store.relations_for([h])
        assert any(r.from_name == "Harry" and r.to_name == "Toby" and r.kind == "rival_of"
                   for r in rels)

    def test_events_for_entities_active_only_via_join(self, store):
        h = store.add_entity("npc", "Harry")
        gone = store.add_entity("npc", "Ghost")
        store.add_event("Harry won", [h], source="avatar")
        store.add_event("Ghost faded", [gone], source="avatar")
        store.set_status(gone, "retired")
        evs = store.events_for_entities([h, gone], limit=10)
        sums = [e.summary for e in evs]
        assert "Harry won" in sums
        # Ghost retired -> its event excluded (active-only, SQL-side via event_actor)
        assert "Ghost faded" not in sums
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestSceneRetrieval -v`
Expected: FAIL (methods undefined; `relations_for` return shape needs `from_name`/`to_name`).

- [ ] **Step 3: Implement**

Add a small result type near the top of `store.py` (next to the other NamedTuples):

```python
class RelationView(NamedTuple):
    from_name: str
    to_name: str
    kind: str
    note: str
```

Methods on `VerseStore`:

```python
    def match_entities_in_text(self, text: str, limit: int = 12) -> list[Entity]:
        """Active entities whose canonical name OR alias appears as a token-ish
        substring in `text` (case-insensitive). Plain scan over active entities —
        the world is small (tens of entities); no FTS needed. Deterministic order."""
        lowered = f" {text.lower()} "
        out: list[Entity] = []
        seen: set[int] = set()
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at "
                "FROM entities WHERE status='active' ORDER BY id"
            ).fetchall()
            alias_rows = conn.execute(
                "SELECT al.entity_id, al.alias FROM entity_alias al "
                "JOIN entities e ON e.id=al.entity_id WHERE e.status='active'"
            ).fetchall()
        alias_by_id: dict[int, list[str]] = {}
        for eid, al in alias_rows:
            alias_by_id.setdefault(eid, []).append(al)
        for row in rows:
            ent = Entity(*row)
            names = [ent.name, *alias_by_id.get(ent.id, [])]
            if any(f" {n.lower()} " in lowered or n.lower() in text.lower().split()
                   for n in names if n):
                if ent.id not in seen:
                    out.append(ent)
                    seen.add(ent.id)
            if len(out) >= limit:
                break
        return out

    def relations_for(self, entity_ids: Sequence[int], limit: int = 30) -> list[RelationView]:
        """1-hop relations touching any of entity_ids, with both endpoint names,
        active endpoints only. Deterministic order."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT ef.name, et.name, r.kind, r.note "
                f"FROM relations r "
                f"JOIN entities ef ON ef.id = r.from_id "
                f"JOIN entities et ON et.id = r.to_id "
                f"WHERE (r.from_id IN ({ph}) OR r.to_id IN ({ph})) "
                f"  AND ef.status='active' AND et.status='active' "
                f"ORDER BY r.id LIMIT ?",
                (*entity_ids, *entity_ids, limit),
            ).fetchall()
        return [RelationView(*row) for row in rows]

    def events_for_entities(
        self, entity_ids: Sequence[int], limit: int = 8
    ) -> list[Event]:
        """Recent events whose actors (via event_actor join) include any of
        entity_ids, restricted to events that still have at least one ACTIVE
        actor. SQL-side active filter (no Python full-scan)."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT ev.id, ev.ts, ev.summary, ev.entity_ids, ev.source "
                f"FROM events ev "
                f"JOIN event_actor ea ON ea.event_id = ev.id "
                f"WHERE ea.entity_id IN ({ph}) "
                f"  AND EXISTS (SELECT 1 FROM event_actor ea2 JOIN entities e2 ON e2.id=ea2.entity_id "
                f"              WHERE ea2.event_id=ev.id AND e2.status='active') "
                f"ORDER BY ev.ts DESC, ev.id DESC LIMIT ?",
                (*entity_ids, limit),
            ).fetchall()
        return [
            Event(id=r[0], ts=r[1], summary=r[2], entity_ids=_parse_entity_ids(r[3], r[0]), source=r[4])
            for r in rows
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestSceneRetrieval -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): scene retrieval — match cast, 1-hop relations, active-only events via event_actor"
```

---

## Task 7: Reorder + enrich `build_verse_system_prompt` (cache-stable + retrieval)

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py` — `build_verse_system_prompt` (`:465-577`); add `message_text` param
- Modify: `plugins/llm/src/llm/plugin.py` — `_verse_route_for` to pass the message text (`:2660-2679`); its caller to supply `text`
- Test: `plugins/llm/tests/verse/test_verse_prompt_roster.py`

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/verse/test_verse_prompt_roster.py
def test_canon_roster_is_stable_prefix_and_scene_is_after(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    t = store.add_entity("npc", "Toby", "year 9")
    store.add_relation(h, t, "rival_of")
    from llm.verse.avatar import build_verse_system_prompt
    out = build_verse_system_prompt(store, avatar_id, "be a year 8 boy",
                                    roster_max_chars=4000,
                                    message_text="did Harry and Toby fight?")
    # Stable canon block comes BEFORE the volatile scene block.
    assert out.index("Established characters") < out.index("In play right now")
    # Author-locked Harry is in the canon roster; both appear in scene cast; relation surfaced.
    assert "Harry" in out and "Toby" in out and "rival_of" in out


def test_prefix_is_byte_identical_when_message_changes_but_canon_does_not(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    from llm.verse.avatar import build_verse_system_prompt, VERSE_SCENE_MARKER
    a = build_verse_system_prompt(store, avatar_id, "p", roster_max_chars=4000, message_text="hi Harry")
    b = build_verse_system_prompt(store, avatar_id, "p", roster_max_chars=4000, message_text="yo Toby")
    # Everything up to the scene marker (the cache-stable prefix) is identical.
    assert a.split(VERSE_SCENE_MARKER)[0] == b.split(VERSE_SCENE_MARKER)[0]
```

> Add a `store_with_avatar` fixture to `tests/verse/conftest.py` if absent: create a store, `opt_in_avatar`/`link_avatar` an avatar, yield `(store, avatar_id)`. Mirror the `_opt_in` helper used in `test_avatar.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_verse_prompt_roster.py -v -k "stable_prefix or byte_identical"`
Expected: FAIL (`message_text` param + `VERSE_SCENE_MARKER` + new layout don't exist).

- [ ] **Step 3: Rewrite `build_verse_system_prompt` stable-first with retrieval**

Add a module constant near the top of `avatar.py`:

```python
VERSE_SCENE_MARKER = "In play right now:"
```

Replace the body so it assembles **stable parts first** (identity, persona, canon roster) then the **volatile scene block** (after `VERSE_SCENE_MARKER`): the avatar's scene/location, active-only recent events, co-located avatars, then the message-matched cast, their 1-hop relations, and their recent events.

```python
def build_verse_system_prompt(
    store: VerseStore,
    avatar_id: int,
    instruct_text: str,
    roster_max_chars: int = 4000,
    message_text: str = "",
) -> str:
    avatar = store.get_entity(avatar_id)
    if avatar is None:
        raise ValueError("avatar not found")

    # ===== STABLE PREFIX (cacheable across turns) =====
    identity_line = f"You are {avatar.name}."
    persona_line = f"Persona: {instruct_text}" if instruct_text.strip() else "Persona: no persona set."

    canon = store.list_canon_entities()
    roster_lines: list[str] = []
    if canon:
        used = 0
        for e in canon:
            line = f"- {e.name}: {e.summary}" if e.summary else f"- {e.name}"
            if used + len(line) + 1 > roster_max_chars:
                roster_lines.append("- (roster truncated)")
                break
            roster_lines.append(line)
            used += len(line) + 1

    parts: list[str] = [identity_line, persona_line]
    if roster_lines:
        parts.append("Established characters in this world:")
        parts.extend(roster_lines)

    # ===== VOLATILE SCENE BLOCK (per-turn; not in the cached prefix) =====
    parts.append(VERSE_SCENE_MARKER)

    location = store.get_attribute(avatar_id, "location")
    place = (store.find_entity_by_name(location, kind="place", active_only=True)
             if location is not None else None)
    parts.append(f"Scene: You are at {place.name}. {place.summary}" if place
                 else "Scene: You are nowhere in particular.")

    # Avatar's own recent events — active-only (dead-lore filtered).
    own = [ev for ev in store.recent_events(limit=50, require_active_entity=True)
           if avatar_id in ev.entity_ids][:5]
    parts.append("Recent events involving you:")
    parts.extend([f"- {ev.summary}" for ev in own] or ["- (none yet)"])

    # Co-located other avatars (unchanged behaviour).
    others = []
    if location is not None:
        for a in store.list_entities_by_kind("avatar", status="active"):
            if a.id != avatar_id and store.get_attribute(a.id, "location") == location:
                others.append(a)
    parts.append("Other avatars present here:")
    parts.extend(
        [f"- {a.name}: {a.summary}" if a.summary else f"- {a.name}" for a in others]
        or ["- (no other avatars present)"]
    )

    # Message-matched cast (not already in canon roster), their relations + events.
    roster_ids = {e.id for e in canon}
    scene = [e for e in store.match_entities_in_text(message_text) if e.id != avatar_id]
    fresh = [e for e in scene if e.id not in roster_ids]
    if fresh:
        parts.append("Characters referenced in this scene:")
        parts.extend([f"- {e.name}: {e.summary}" if e.summary else f"- {e.name}" for e in fresh])

    rel_ids = list(roster_ids | {e.id for e in scene} | {avatar_id})
    rels = store.relations_for(rel_ids)
    if rels:
        parts.append("Known relationships:")
        parts.extend([f"- {r.from_name} {r.kind.replace('_', ' ')} {r.to_name}"
                      + (f" ({r.note})" if r.note else "") for r in rels])

    scene_events = store.events_for_entities([e.id for e in scene], limit=8)
    if scene_events:
        parts.append("Recent events involving them:")
        parts.extend([f"- {ev.summary}" for ev in scene_events])

    return "\n".join(parts)
```

- [ ] **Step 4: Thread `message_text` through the call site**

In `plugin.py`, change `_verse_route_for` to accept and pass the user's message text. Update its signature (e.g. `_verse_route_for(self, channel, account, nick, message_text="")`) and the `build_verse_system_prompt(...)` call to pass `message_text=message_text`. Update the caller (where `_verse_route_for`/`VerseRoute` is built, near `verse_model = self.registryValue("verseModel", ...)`) to pass the incoming `text`.

> Locate with `grep -n "_verse_route_for\|build_verse_system_prompt" plugins/llm/src/llm/plugin.py`. Pass the same `text` that becomes the user turn.

- [ ] **Step 5: Update the roster registry default**

In `config.py:379-388`, raise `verseRosterMaxChars` default from `600` to `4000` (the locked roster must never truncate a ~15-NPC cast). Update the function-signature default in `build_verse_system_prompt` to match (`4000`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/verse/test_verse_prompt_roster.py tests/verse/test_avatar.py -v`
Expected: PASS. Fix any existing prompt tests that asserted the OLD ordering (roster-last) — update them to the new stable-first layout (this is intended behaviour change, documented in the spec).

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/src/llm/plugin.py plugins/llm/src/llm/config.py plugins/llm/tests/verse/
git commit -m "feat(verse): stable-first verse prompt with scene cast + relations + active-only events"
```

---

## Task 8: Promotion-on-reinforcement — `msg.prefix`-bound `verse_record` overlay

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — add `bump_author_mention` returning the new count + promotion at threshold
- Modify: `plugins/llm/src/llm/plugin.py` — overlay a `msg.prefix`-bound `verse_record` handler in `combined_handlers` (next to `verse_edit`, `:3825-3857`)
- Modify: `plugins/llm/src/llm/config.py` — add `verseAuthorLockMentions` (channel, default 2)
- Test: `plugins/llm/tests/verse/test_store.py` + `plugins/llm/tests/test_plugin_verse.py`

- [ ] **Step 1: Write the failing store test**

```python
class TestAuthorMentionPromotion:
    def test_promotes_after_threshold(self, store):
        h = store.add_entity("npc", "Harry")
        assert store.bump_author_mention(h, threshold=2) == 1   # first human mention
        assert store.get_attribute(h, "author_locked") is None
        assert store.bump_author_mention(h, threshold=2) == 2   # second -> promote
        assert store.get_attribute(h, "author_locked") == "1"
```

- [ ] **Step 2: Run + fail**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestAuthorMentionPromotion -v`
Expected: FAIL (`bump_author_mention` undefined).

- [ ] **Step 3: Implement `bump_author_mention`**

```python
    def bump_author_mention(self, entity_id: int, *, threshold: int = 2) -> int:
        """Increment the human-reinforcement counter for an entity; at >=threshold
        set author_locked. Returns the new count. Used only for HUMAN-offered
        names by an authorized author (the caller enforces both). Stored as the
        reserved EAV key 'author_mentions'."""
        with self.write_transaction() as conn:
            row = conn.execute(
                "SELECT value FROM attributes WHERE entity_id=? AND key='author_mentions'",
                (entity_id,),
            ).fetchone()
            count = (int(row[0]) if row and str(row[0]).isdigit() else 0) + 1
            self._set_attribute_inline(conn, entity_id, "author_mentions", str(count))
            if count >= threshold:
                self._set_author_locked_inline(conn, entity_id, True)
            return count
```

Add `"author_mentions"` to `_RESERVED_ATTRIBUTE_KEYS`.

- [ ] **Step 4: Run + pass**

Run: `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestAuthorMentionPromotion -v`
Expected: PASS.

- [ ] **Step 5: Add the config knob**

In `config.py` (Verse section), add:

```python
conf.registerChannelValue(
    LLM,
    "verseAuthorLockMentions",
    registry.PositiveInteger(
        2,
        _("""How many times an authorized author must mention a human-offered name
        before it is promoted to durable author-locked canon (always remembered)."""),
    ),
)
```

- [ ] **Step 6: Write the plugin overlay test**

```python
# add to plugins/llm/tests/test_plugin_verse.py
def test_verse_record_overlay_promotes_human_offered_name(verse_plugin_ctx):
    """An authorized author (llm.verse.edit) who names a NEW character in their
    message, twice, gets it auto-locked; a model-invented name (absent from the
    user's message) never promotes."""
    ctx = verse_plugin_ctx  # fixture: plugin + channel + authed author msg + store + avatar
    # Turn 1: author says "Harry joined"; model records actors=["Harry"].
    ctx.run_verse_record(message_text="Harry joined us", actors=["Harry"])
    # Turn 2: same.
    ctx.run_verse_record(message_text="Harry scored again", actors=["Harry"])
    h = ctx.store.find_active_entity_by_name("Harry")
    assert ctx.store.get_attribute(h.id, "author_locked") == "1"
    # Model-invented name not in the user's message must NOT promote.
    ctx.run_verse_record(message_text="tell me a tale", actors=["Gandalf"])
    g = ctx.store.find_active_entity_by_name("Gandalf")
    assert ctx.store.get_attribute(g.id, "author_locked") is None
```

> Build `verse_plugin_ctx` to mirror existing `test_plugin_verse.py` harness patterns (it already exercises verse handlers). The helper must run the overlaid `verse_record` handler with a `msg` whose prefix has `llm.verse.edit`.

- [ ] **Step 7: Implement the overlay handler in plugin.py**

In the block that builds `combined_handlers` (where `verse_edit` is overlaid, `:3825-3857`), overlay `verse_record` with a wrapper that (a) runs the normal dispatch, then (b) for each recorded actor name that appears in the user's message text AND when the caller holds `llm.verse.edit`, bumps the author-mention counter:

```python
                base_record = combined_handlers.get("verse_record")
                authed_author = ircdb.checkCapability(msg.prefix, "llm.verse.edit")
                lock_threshold = self.registryValue("verseAuthorLockMentions", channel)
                user_text_lower = text.lower()

                def _record_with_promotion(args: dict, _base=base_record,
                                           _store=verse_route.store):
                    result = _base(args) if _base else None
                    if authed_author and isinstance(args, dict):
                        for raw in args.get("actors") or []:
                            if not isinstance(raw, str) or not raw.strip():
                                continue
                            name = raw.strip()
                            # human-offered: the author typed this name (or its alias) this turn
                            ent = _store.find_entity_by_name_or_alias(name)
                            mentioned = name.lower() in user_text_lower or (
                                ent is not None and ent.name.lower() in user_text_lower
                            )
                            if not mentioned:
                                continue
                            target = ent or _store.find_active_entity_by_name(name)
                            if target is not None:
                                _store.bump_author_mention(target.id, threshold=lock_threshold)
                    return result

                combined_handlers["verse_record"] = _record_with_promotion
```

> `text` (the user's message), `msg`, `channel`, `verse_route` are all in scope here (this is the same block that builds `verse_edit`/`verse_storybook` with `msg=msg`). The base `verse_record` handler is the avatar-bound one already in `combined_handlers` from `_build_verse_handlers_for_route`.

- [ ] **Step 8: Run + pass**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -k promot -v` and `cd plugins/llm && uv run pytest tests/verse/test_store.py::TestAuthorMentionPromotion -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/src/llm/plugin.py plugins/llm/src/llm/config.py plugins/llm/tests/
git commit -m "feat(verse): promote human-offered reinforced names to author_locked (gated on llm.verse.edit)"
```

---

## Task 9: `@canon` command (explicit lock/unlock/forget — human, invisible to Grok)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — add a `canon` command near `verse`/`look`/`who` (`:5790-5882`)
- Test: `plugins/llm/tests/test_commands.py`

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/test_commands.py (verse command group)
def test_canon_lock_and_forget(verse_command_ctx):
    ctx = verse_command_ctx
    ctx.store.add_entity("npc", "Harry")
    ctx.run("canon lock Harry")
    h = ctx.store.find_active_entity_by_name("Harry")
    assert ctx.store.get_attribute(h.id, "author_locked") == "1"
    ctx.run("canon forget Harry")
    assert ctx.store.get_attribute(h.id, "author_locked") is None
```

> Mirror the existing verse-command test harness (`@verse`, `@look` tests already exist in this file).

- [ ] **Step 2: Run + fail**

Run: `cd plugins/llm && uv run pytest tests/test_commands.py -k canon -v`
Expected: FAIL (no `canon` command).

- [ ] **Step 3: Implement the command**

Add a `canon` command gated on `llm.verse.edit` (the author capability), with subcommands `lock <name>` / `unlock <name>` / `forget <name>` (forget == unlock), resolving via `find_entity_by_name_or_alias` and calling `set_author_locked`. Follow the `wrap(...)` idiom used by `verse`/`look`:

```python
    def canon(self, irc, msg, args, channel, action, name):
        """<lock|unlock|forget> <name>

        Lock or release a character as durable canon (always remembered).
        Requires the llm.verse.edit capability.
        """
        store = self._get_or_create_verse_store(channel)
        ent = store.find_entity_by_name_or_alias(name)
        if ent is None:
            irc.error(f"No such character: {name}", prefixNick=False)
            return
        store.set_author_locked(ent.id, action == "lock")
        irc.replySuccess()
    canon = wrap(canon, [("checkCapability", "llm.verse.edit"),
                         ("literal", ("lock", "unlock", "forget")), "text"])
```

> Map `"forget"` to unlock inside the body if you prefer an explicit branch; the literal set keeps Grok-free determinism. Confirm `channel` injection matches the other verse commands' `wrap` spec (they use `("checkCapability","llm.verse")` first — here use `llm.verse.edit`).

- [ ] **Step 4: Run + pass / commit**

Run: `cd plugins/llm && uv run pytest tests/test_commands.py -k canon -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat(verse): @canon lock/unlock/forget command (author-gated, Grok-invisible)"
```

---

## Task 10: Guarantee canon-write on storybook turns

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_submit_storybook_job` / storybook handler (`:2809-2834`)
- Test: `plugins/llm/tests/test_storybook.py`

- [ ] **Step 1: Write the failing test**

```python
# add to plugins/llm/tests/test_storybook.py
def test_storybook_records_a_canon_event(storybook_ctx):
    """An illustrated turn short-circuits before any verse_record model step,
    so the handler itself must log a canon event for the story."""
    ctx = storybook_ctx  # fixture: plugin + verse channel + avatar + store
    before = len(ctx.store.recent_events(limit=100))
    ctx.fire_storybook(brief="the stinky lads raid the cafeteria")
    after = ctx.store.recent_events(limit=100)
    assert len(after) == before + 1
    assert "cafeteria" in after[0].summary.lower() or "storybook" in after[0].summary.lower()
```

- [ ] **Step 2: Run + fail**

Run: `cd plugins/llm && uv run pytest tests/test_storybook.py -k records_a_canon -v`
Expected: FAIL (no event written).

- [ ] **Step 3: Record canon in the storybook handler**

In the storybook `_call` handler, right after reserving the per-turn slot and before/after `_submit_storybook_job`, record a canon event attributed to the caller's avatar so the illustrated turn is not invisible to canon:

```python
            # Illustrated turns short-circuit before any verse_record model step,
            # so log the story as canon here (best-effort; never block the page).
            try:
                route = self._verse_route_for(channel, account, nick)
                if route is not None:
                    summary = (brief.strip()[:200] or "told an illustrated tale")
                    self._get_or_create_verse_store(channel).record_user_event(
                        actor_id=route.avatar_id, summary=summary, actor_names=[],
                    )
            except Exception:
                self.log.exception("storybook canon-record failed (non-fatal)")
            self._submit_storybook_job(channel=channel, nick=nick, persona=persona, brief=brief)
```

> `brief` is computed just above in the handler. Keep this best-effort and non-fatal — the illustrated page must still post even if recording fails.

- [ ] **Step 4: Run + pass / commit**

Run: `cd plugins/llm && uv run pytest tests/test_storybook.py -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_storybook.py
git commit -m "fix(verse): record a canon event on storybook turns (they short-circuit verse_record)"
```

---

## Task 11: Loud warning when verse silently rides a non-verse model

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — the `verseModel` read site (`:3702`)
- Test: `plugins/llm/tests/test_plugin_verse.py`

- [ ] **Step 1: Write the failing test**

```python
def test_empty_verse_model_logs_warning(verse_plugin_ctx, caplog):
    ctx = verse_plugin_ctx
    ctx.plugin.setRegistryValue("verseModel", "", channel=ctx.channel)
    import logging
    with caplog.at_level(logging.WARNING):
        ctx.trigger_verse_turn("hello")
    assert any("verseModel" in r.message and "assistantModel" in r.message
               for r in caplog.records)
```

- [ ] **Step 2: Run + fail**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -k verse_model_logs -v`
Expected: FAIL (no warning emitted).

- [ ] **Step 3: Emit the warning**

At `plugin.py:3702`, after reading `verse_model`:

```python
            verse_model = self.registryValue("verseModel", preflight.channel) or None
            if verse_model is None:
                self.log.warning(
                    "verse turn on channel=%s has empty verseModel; falling back to "
                    "assistantModel — set a non-reasoning verseModel or verse prose may "
                    "be cratered by a reasoning model",
                    preflight.channel,
                )
```

> Keep the existing fallback behaviour (do NOT hard-fail). This only surfaces the foot-gun. Rate-limiting the log (once per channel) is a nice-to-have, not required.

- [ ] **Step 4: Run + pass / commit**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -k verse_model -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "feat(verse): loud warning when verseModel is empty and falls back to assistantModel"
```

---

## Task 12: Characterize the denial/degrade retry re-seed (do NOT change behaviour)

**Files:**
- Test only: `plugins/llm/tests/test_service_completion.py`

> The spec (§9) treats the "re-seed rejected reply before retry" as deliberate, not a bug. Pin the current behaviour so a future refactor can A/B against it — do not modify service.py here.

- [ ] **Step 1: Write the characterization test**

```python
def test_verse_denial_retry_reseeds_rejected_reply_then_nudges(service_ctx):
    """CHARACTERIZATION (pins current behaviour): on a verse denial, the in-flight
    message list gets [assistant: rejected_reply, user: nudge] appended before the
    retry, and the corrected reply (not the rejected one) is what is returned."""
    ctx = service_ctx.verse_denial_then_recover()  # stub model: turn1 refuses, turn2 recovers
    result = ctx.run()
    assert result.content == ctx.recovered_text          # rejected text never returned
    assert ctx.seen_messages_before_retry[-2:] == [
        {"role": "assistant", "content": ctx.rejected_text},
        {"role": "user", "content": ctx.denial_nudge},
    ]
```

> Build on the existing `test_service_completion.py` stub-client harness (it already drives multi-step completions with canned model replies). Capture the in-flight `messages` snapshot at the retry boundary.

- [ ] **Step 2: Run to verify it passes against current code**

Run: `cd plugins/llm && uv run pytest tests/test_service_completion.py -k reseeds -v`
Expected: PASS (it characterizes existing behaviour). If it fails, fix the TEST to match reality — do NOT change service.py.

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/tests/test_service_completion.py
git commit -m "test(verse): characterize denial-retry re-seed behaviour (pin before any future change)"
```

---

## Task 13: Integration test — full retrieval path end-to-end

**Files:**
- Test: `plugins/llm/tests/verse/test_retrieval_integration.py` (new)

- [ ] **Step 1: Write the integration test**

```python
"""End-to-end: a locked roster member absent from the message still appears;
a scene-named member + relation + event appear; a retired entity does not."""
from llm.verse.avatar import build_verse_system_prompt, VERSE_SCENE_MARKER


def test_full_retrieval(store_with_avatar):
    store, avatar_id = store_with_avatar
    harry = store.add_entity("npc", "Harry", "year 8 ringleader")
    toby = store.add_entity("npc", "Toby", "year 9")
    ghost = store.add_entity("npc", "Ghost")
    store.set_author_locked(harry, True)          # locked roster (not named in message)
    store.add_alias(toby, "Tobes")
    store.add_relation(harry, toby, "rival_of")
    store.add_event("Toby nicked the register", [toby], source="avatar")
    store.add_event("Ghost vanished", [ghost], source="avatar")
    store.set_status(ghost, "retired")

    out = build_verse_system_prompt(store, avatar_id, "be a year 8 boy",
                                    roster_max_chars=4000,
                                    message_text="what's Tobes up to?")
    prefix, scene = out.split(VERSE_SCENE_MARKER)
    assert "Harry" in prefix                       # locked roster member, even unmentioned
    assert "Toby" in scene                          # resolved via alias 'Tobes'
    assert "rival_of".replace("_", " ") in scene    # 1-hop relation surfaced
    assert "register" in scene                      # Toby's event surfaced
    assert "Ghost" not in out                        # retired -> excluded everywhere
```

- [ ] **Step 2: Run + pass**

Run: `cd plugins/llm && uv run pytest tests/verse/test_retrieval_integration.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/tests/verse/test_retrieval_integration.py
git commit -m "test(verse): end-to-end retrieval integration (roster + alias + relation + event + dead-lore)"
```

---

## Task 14: Full suite + lint/typecheck gate

- [ ] **Step 1: Run the full verse + plugin test suite**

Run: `cd plugins/llm && uv run pytest tests/verse tests/test_plugin_verse.py tests/test_storybook.py tests/test_commands.py tests/test_service_completion.py -v`
Expected: PASS. Fix any prompt-ordering tests that asserted the old layout (intended change).

- [ ] **Step 2: Lint + typecheck**

Run: `cd plugins/llm && make lint && make typecheck`
Expected: clean.

- [ ] **Step 3: Confirm chat/code/draw untouched**

Run: `cd plugins/llm && uv run pytest tests/test_assistant.py tests/test_service_core.py -v`
Expected: PASS (no behaviour change to non-verse paths).

- [ ] **Step 4: Commit any fixups**

```bash
git add -A && git commit -m "test(verse): suite green + lint/typecheck clean for retention fix"
```

---

## Rollout (post-merge, operator)

Not code tasks — operator steps once merged and deployed:
1. The v2→v3 migration runs automatically on first open of #afternet's store (additive; backfills `event_actor`).
2. Optionally seed aliases for the known roster (`@canon` has no alias verb in v1; aliases accrue via future use or a one-off `add_alias` script). The existing pinned roster already shows via `list_canon_entities`.
3. Verse stays behind the existing per-channel `verseEnabled` flag; rollback is a registry flip / revert (no data migration to undo — v3 is additive).
4. Watch `docker logs vibebot` for the new `verseModel` warning and any migration errors.

## Coverage check (plan vs spec)
- Spec §3.1 retrieval → Tasks 6, 7, 13. §3.2 author_locked promotion → Tasks 3, 8, 9. §3.3 alias → Task 5. §3.4 event_actor → Tasks 1, 2, 6. §3.5 cache byte-freeze → Task 7. §3.6 generation fixes → Task 11 (+ §9 freq_penalty note: see Open Items). §3.7 storybook canon-write → Task 10. Aging/loom protection → Task 4. Re-seed characterize → Task 12.

## Open items (carry into review)
- **Measurement (spec §8):** no code task — the re-scoped gate is fc42's live ~5:1 benchmark, not a shadow/A-B subsystem. Optional lightweight observability: log the injected canon-entity ids per verse turn (one `self.log.info` near the `_verse_route_for` call) so recall can be eyeballed. Deferred unless review wants it in v1.
- **Test harness fixtures:** the plugin/service/command/storybook tests reference context fixtures (`verse_plugin_ctx`, `storybook_ctx`, `service_ctx`, `verse_command_ctx`, `store_with_avatar`) named illustratively — the executor must wire these to the REAL harness in `tests/conftest.py` / `tests/verse/conftest.py` (e.g. the existing `_opt_in` helper, the `test_plugin_verse.py` verse-handler harness). Store-level tests use the real `store`/`verse_db_dir` fixtures and stand as written.
- **frequency_penalty drop log (spec §9):** add a one-time loud log when a verse sampling param is dropped for the active provider, at `service.py:3906-3909`. Small; fold into Task 11 or a follow-up — it is observability, not behaviour.
- **`match_entities_in_text` matching quality:** the substring/token heuristic is deliberately simple; if it over/under-matches in shadow, tighten to word-boundary regex. Flagged for the plan red-team.
- **aging.py / loom.py exact predicates (Task 4):** the executor must read those files to place the edits precisely; tests pin the behaviour.
