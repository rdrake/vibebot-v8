# Verse Retention Fix — Implementation Plan (v2, post plan-red-team)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Executor discipline (a plan-red-team finding):** the embedded code/SQL/test snippets are verified against the real files, but you MUST still open and re-read each target region before editing — match real signatures, fixtures, and call sites. Treat snippets as precise guidance, not blind paste.

**Goal:** Make the verse model reliably remember canon — inject the full canon roster (pinned OR author-locked) plus the scene-relevant cast, their relations, and their recent events into every verse turn; let an operator/author lock canon explicitly via `@canon`.

**Architecture:** Minimal, in-place changes to the existing `llm` plugin's verse path (no new plugin, no data port). One additive SQLite migration (v2→v3) adds `entity_alias` and an `event_actor` join table. `build_verse_system_prompt` is reordered stable-first (canon roster in the cacheable prefix; volatile/retrieved context after) and enriched with message-matched cast + 1-hop relations + active-only scene events. chat/code/draw untouched.

**Tech Stack:** Python 3, Limnoria/Supybot plugin, SQLite (WAL; `schema.sql` + `schema_version` table; `SCHEMA_VERSION` constant), pytest (real SQLite, no mocks), litellm.

**Commands (verified — repo ROOT, there is NO `plugins/llm/Makefile`):**
- Single test: `uv run pytest plugins/llm/tests/verse/test_x.py::TestClass -v`
- Full gate (coverage `--cov-fail-under=93`): `make test`
- Lint/format/type: `make lint && make typecheck`
- Because `make test` enforces 93% coverage, every new code branch needs a test or the suite fails — Task 0 builds the shared fixtures first so new branches are reachable.

**Scope note (deferred after the plan-red-team):** Auto-lock-canon-by-talking (promotion-on-reinforcement) is **deferred to a fast-follow**. Tasks 6–7 inject the full roster + scene cast every turn, and the #afternet stinky-lads roster is *already pinned* — so the immediate "it forgot the lads" is fixed by retrieval alone. `author_locked` + `@canon` cover explicit locking. Talking-only auto-promotion is added later only if fc42's live ~5:1 benchmark shows it's needed.

**Spec deltas (deliberate, documented — do not claim §5/§9 compliance):**
- §5 cache: v1 keeps the canon roster as the *tail-stable* lead of the single system message (volatile content follows it), rather than splitting scene into a separate user-role message. The plan-red-team verified the roster lands in the cacheable byte-prefix this way; the user-role split is a later refinement.
- §9 verse model: v1 *warns loudly* when `verseModel` is empty (falls back to `assistantModel`) rather than hard-failing. Hard-fail + reasoning-model startup validation is deferred.

**Conventions (from the codebase):** `_<name>_inline(self, conn, ...)` twin per mutator; `self._lock` is NOT reentrant (never call a public store method inside an open `write_transaction`); lifecycle flags are EAV rows in `attributes` (TEXT values), engine-only keys in `_RESERVED_ATTRIBUTE_KEYS`; additive migration = `CREATE ... IF NOT EXISTS` in `schema.sql` + bump `SCHEMA_VERSION` + `if current < N:` branch + `_upgrade_*` in one `write_transaction` stamping `schema_version`; tests use real SQLite, class-grouped, assert on `Entity(...)` attrs.

---

## Task 0: Shared verse test fixtures (prerequisite — unblocks Tasks 2,3,5,6,13)

**Files:**
- Modify: `plugins/llm/tests/verse/conftest.py`

> The plan-red-team verified `tests/verse/conftest.py` has only `verse_db_dir` — there is **no** shared `store` fixture (the per-file ones in `test_avatar.py`/`test_verse_record.py` are local). New store-level test classes need it, or they error at collection ("fixture 'store' not found"), masking the real assertion.

- [ ] **Step 1: Add fixtures + make the test event-insert helper populate `event_actor`**

Append to `plugins/llm/tests/verse/conftest.py`:

```python
from llm.verse.store import VerseStore


@pytest.fixture
def store(verse_db_dir: Path):
    """A migrated VerseStore on a real per-test SQLite file."""
    return VerseStore(verse_db_dir, "#test")


@pytest.fixture
def store_with_avatar(store):
    """(store, avatar_id) — an opted-in avatar named 'me' for prompt/retrieval tests."""
    avatar_id = store.opt_in_avatar(nick="me", account="me-acct").entity_id
    return store, avatar_id
```

Update `insert_event_at` so test-seeded events also get `event_actor` rows (Task 1 adds the table; this keeps the test helper consistent with production after Task 2):

```python
    with store.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, _json.dumps(list(entity_ids)), source),
        )
        event_id = int(cur.lastrowid)
        for eid in dict.fromkeys(entity_ids):
            if conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                    (event_id, eid),
                )
        return event_id
```

> Confirm `opt_in_avatar`'s return type exposes `.entity_id` (store.py:861, `AvatarOptInResult`). If the field name differs, adjust.

- [ ] **Step 2: Smoke-check the fixtures import**

Run: `uv run pytest plugins/llm/tests/verse/ -v -k "nonexistent_smoke" ; echo "collection ok if no fixture errors"`
Expected: no collection errors (the `-k` matches nothing; we're checking conftest imports cleanly).

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/tests/verse/conftest.py
git commit -m "test(verse): shared store/store_with_avatar fixtures; event_actor in insert_event_at"
```

---

## Task 1: Additive migration — `entity_alias` + `event_actor` (v2→v3)

**Files:**
- Modify: `plugins/llm/src/llm/verse/schema.sql`
- Modify: `plugins/llm/src/llm/verse/store.py:99` (`SCHEMA_VERSION`), `_migrate` (`:178-179`), add `_upgrade_v2_to_v3` after `_upgrade_v1_to_v2` (`:225`)
- Test: `plugins/llm/tests/verse/test_store_migration.py`

- [ ] **Step 1: Write failing tests (mirror the existing `_make_v1_db` pattern)**

```python
# append to tests/verse/test_store_migration.py — reuses _make_v1_db + VerseStore(base, channel)
import json
from llm.verse.store import SCHEMA_VERSION


def _seed_v2_event(base, channel, entity_ids):
    """Add an entity + one event (with possibly-bad entity_ids) to a v1 DB so the
    v1->v2->v3 chain backfills event_actor element-wise."""
    path = _make_v1_db(base, channel)
    raw = sqlite3.connect(path)
    raw.execute("INSERT INTO entities(id,kind,name,created_at,updated_at) VALUES (1,'npc','Harry',0,0)")
    raw.execute("INSERT INTO events(id,ts,summary,entity_ids,source) VALUES (1,0,'x',?, 'avatar')",
                (json.dumps(entity_ids),))
    raw.commit(); raw.close()
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
        rows = conn.execute("SELECT event_id, entity_id FROM event_actor ORDER BY entity_id").fetchall()
        ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert rows == [(1, 1)]   # keep valid existing id 1; drop 99999 + 'garbage'
    assert ver == 3


def test_v3_migration_idempotent_on_reopen(tmp_path):
    _seed_v2_event(tmp_path, "#chan", [1])
    VerseStore(tmp_path, "#chan")
    store2 = VerseStore(tmp_path, "#chan")  # second open must not double-apply
    with store2.read_connection() as conn:
        v3rows = conn.execute("SELECT COUNT(*) FROM schema_version WHERE version=3").fetchone()[0]
        ea = conn.execute("SELECT COUNT(*) FROM event_actor").fetchone()[0]
    assert v3rows == 1 and ea == 1   # INSERT OR IGNORE keeps the backfill a no-op
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_store_migration.py -v -k "v3 or chain or fresh"`
Expected: FAIL (no `entity_alias`/`event_actor`; `SCHEMA_VERSION == 2`).

- [ ] **Step 3: Append DDL to `schema.sql`** (note `alias ... COLLATE NOCASE` so the PK dedups case-variants)

```sql
CREATE TABLE IF NOT EXISTS entity_alias (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias     TEXT NOT NULL COLLATE NOCASE,
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

- [ ] **Step 4: Bump version + add migration**

`store.py:99`: `SCHEMA_VERSION = 3`. In `_migrate` after the `if current < 2:` branch add `if current < 3: self._upgrade_v2_to_v3()`. Add after `_upgrade_v1_to_v2`:

```python
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
```

- [ ] **Step 5: Run → pass**

Run: `uv run pytest plugins/llm/tests/verse/test_store_migration.py -v`
Expected: PASS (incl. the existing v1→v2 tests).

- [ ] **Step 6: Commit**

```bash
git add plugins/llm/src/llm/verse/schema.sql plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store_migration.py
git commit -m "feat(verse): entity_alias + event_actor (schema v3), element-wise tolerant backfill"
```

---

## Task 2: Populate `event_actor` from ALL THREE event-insert sites

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `_add_event_inline` (`:567-587`); route `_replace_events_with_source` (`:770`) and `_apply_op_inline` add_event (`:1118`) through it
- Test: `plugins/llm/tests/verse/test_store.py`

> Plan-red-team MF1 (real data bug): there are THREE runtime `INSERT INTO events` sites — `_add_event_inline:581`, `_replace_events_with_source:770` (loom digest), `_apply_op_inline:1118` (every proposal/`verse_edit`/`apply_direct` add_event). Patching only one leaves author-authored canon invisible to `events_for_entities` (Task 6), which JOINs strictly on `event_actor`. Make `_add_event_inline` the single writer and route the other two through it.

- [ ] **Step 1: Write failing test (covers the apply_direct path specifically)**

```python
class TestEventActorAllSites:
    def test_add_event_populates_event_actor(self, store):
        a = store.add_entity("npc", "Harry")
        ev = store.add_event("Harry did a thing", [a, 99999], source="avatar")
        with store.read_connection() as conn:
            rows = conn.execute("SELECT entity_id FROM event_actor WHERE event_id=?", (ev,)).fetchall()
        assert rows == [(a,)]   # 99999 has no entities row -> FK-safe skip

    def test_apply_direct_add_event_populates_event_actor(self, store):
        a = store.add_entity("npc", "Harry")
        store.apply_direct(op="add_event",
                           payload={"summary": "Harry returned", "entity_ids": [a]},
                           source="operator", provenance="test")
        evs = store.events_for_entities([a], limit=10)   # JOINs on event_actor
        assert any("returned" in e.summary for e in evs)
```

> Confirm the `add_event` payload keys `_apply_op_inline` expects (`grep -n "op == \"add_event\"" store.py`); adjust `payload` to the real keys.

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestEventActorAllSites -v`
Expected: FAIL (apply_direct path writes no `event_actor`; `events_for_entities` undefined until Task 6 — run this test after Task 6 is in, or stub `events_for_entities`; see ordering note below).

> **Ordering:** `events_for_entities` is built in Task 6. Either implement Task 6 before Task 2's second assertion, or split: do Step 3 here, assert `event_actor` rows directly via SQL now, and add the `events_for_entities` assertion in Task 6. Recommended: assert via SQL here.

- [ ] **Step 3: Make `_add_event_inline` the single writer**

In `_add_event_inline`, after the INSERT, populate `event_actor` (FK-safe), then route the other two sites through it. `_add_event_inline` final form:

```python
    def _add_event_inline(self, conn, *, summary, entity_ids, source, ts=None) -> int:
        if ts is None:
            ts = time.time()
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, json.dumps(list(entity_ids)), source),
        )
        event_id = cur.lastrowid
        assert event_id is not None
        for eid in dict.fromkeys(entity_ids):
            if conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                    (event_id, eid),
                )
        return event_id
```

At `_replace_events_with_source:770` and `_apply_op_inline:1118`, replace the raw `conn.execute("INSERT INTO events ...")` with `self._add_event_inline(conn, summary=<s>, entity_ids=<ids>, source=<src>, ts=<ts>)`. **Read each site first** — map its local variable names (and whether it captures `lastrowid`) to the helper call; preserve any downstream use of the returned id.

- [ ] **Step 4: Run → pass**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestEventActorAllSites tests/verse/test_loom.py tests/verse/test_compaction.py -v`
(Run from repo root with the `plugins/llm/` prefix on each path.) Expected: PASS (loom/compaction exercise sites 770/1118).

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "fix(verse): populate event_actor from all 3 event-insert sites (single writer)"
```

---

## Task 3: `author_locked` flag + canon roster (pinned OR author_locked)

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `_RESERVED_ATTRIBUTE_KEYS` (`:28`); add `set_author_locked` (+inline), `list_canon_entities`
- Test: `plugins/llm/tests/verse/test_store_pinned.py`

- [ ] **Step 1: Write failing test**

```python
class TestAuthorLocked:
    def test_list_canon_unions_pinned_and_author_locked(self, store):
        h = store.add_entity("npc", "Harry", "year 8")
        t = store.add_entity("npc", "Toby", "year 9")
        store.set_attribute(t, "pinned", "1")
        store.set_author_locked(h, True)
        assert {e.name for e in store.list_canon_entities()} == {"Harry", "Toby"}

    def test_author_locked_is_reserved(self):
        from llm.verse.store import _RESERVED_ATTRIBUTE_KEYS
        assert "author_locked" in _RESERVED_ATTRIBUTE_KEYS

    def test_unlock_removes_from_canon(self, store):
        h = store.add_entity("npc", "Harry")
        store.set_author_locked(h, True)
        store.set_author_locked(h, False)
        assert store.list_canon_entities() == []
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_store_pinned.py::TestAuthorLocked -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (add `"author_locked"` to `_RESERVED_ATTRIBUTE_KEYS`; add methods near `list_pinned_entities`)

```python
    def _set_author_locked_inline(self, conn, entity_id: int, locked: bool) -> None:
        if locked:
            self._set_attribute_inline(conn, entity_id, "author_locked", "1")
        else:
            conn.execute("DELETE FROM attributes WHERE entity_id=? AND key='author_locked'", (entity_id,))

    def set_author_locked(self, entity_id: int, locked: bool) -> None:
        """Lock/unlock durable canon (always injected, aging-exempt, loom-protected)."""
        with self.write_transaction() as conn:
            self._set_author_locked_inline(conn, entity_id, locked)

    def list_canon_entities(self) -> list[Entity]:
        """Active entities that are durable canon: pinned (operator) OR author_locked.
        DISTINCT (an entity may carry both). Deterministic kind-then-name order."""
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

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/verse/test_store_pinned.py::TestAuthorLocked -v`

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store_pinned.py
git commit -m "feat(verse): author_locked canon flag + list_canon_entities"
```

---

## Task 4: Aging + loom exempt `author_locked`

**Files:**
- Modify: `plugins/llm/src/llm/verse/aging.py:43-44` (the Python pinned-exemption check)
- Test: `plugins/llm/tests/verse/test_verse_aging.py`, `plugins/llm/tests/verse/test_store_privilege.py`

> Plan-red-team MF4: aging's public fn is `age_auto_created_entities(store, *, retire_after_days, now)` (aging.py:22) and the exemption is a **Python check** at line 43 (`if store.get_attribute(entity.id, "pinned") == "1": continue`) — there is no SQL WHERE. Loom-protection is already structural: `author_locked` is reserved, so `_apply_op_inline` raises `ValueError` on a loom/proposal `set_attribute` to it.

- [ ] **Step 1: Write failing aging test**

```python
class TestAgingExemptsAuthorLocked:
    def test_author_locked_npc_not_retired(self, store):
        import time as _t
        h = store.add_entity("npc", "Harry")
        store.set_attribute(h, "auto_created", "1")
        store.set_attribute(h, "last_seen_ts", "0.0")     # ancient
        store.set_author_locked(h, True)
        from llm.verse.aging import age_auto_created_entities
        age_auto_created_entities(store, retire_after_days=1, now=lambda: _t.time())
        assert store.get_entity(h).status == "active"
```

> Confirm `age_auto_created_entities`'s retirement criterion (it reads `last_seen_ts` vs `retire_after_days`); the ancient `last_seen_ts` makes Harry a candidate.

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_verse_aging.py::TestAgingExemptsAuthorLocked -v`
Expected: FAIL (Harry retired).

- [ ] **Step 3: Widen the exemption (aging.py:43)**

```python
        if (store.get_attribute(entity.id, "pinned") == "1"
                or store.get_attribute(entity.id, "author_locked") == "1"):
            continue  # pinned = operator canon; author_locked = author canon; never auto-retire
```

- [ ] **Step 4: Add the loom-forging guard test (apply_direct needs source + provenance)**

```python
# tests/verse/test_store_privilege.py
def test_loom_cannot_forge_author_locked(store):
    import pytest
    h = store.add_entity("npc", "Harry")
    with pytest.raises(ValueError):
        store.apply_direct(op="set_attribute",
                           payload={"entity_id": h, "key": "author_locked", "value": "1"},
                           source="loom", provenance="test")
```

- [ ] **Step 5: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/verse/test_verse_aging.py plugins/llm/tests/verse/test_store_privilege.py -v`

```bash
git add plugins/llm/src/llm/verse/aging.py plugins/llm/tests/verse/test_verse_aging.py plugins/llm/tests/verse/test_store_privilege.py
git commit -m "feat(verse): aging exempts author_locked; reserved-key blocks loom forging it"
```

---

## Task 5: Aliases + name-or-alias resolution

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — `add_alias` (+inline), `list_aliases`, `find_entity_by_name_or_alias`
- Test: `plugins/llm/tests/verse/test_store.py`

- [ ] **Step 1: Write failing test**

```python
class TestAliases:
    def test_resolve_alias_case_insensitive(self, store):
        t = store.add_entity("npc", "Toby")
        store.add_alias(t, "Tobes")
        assert store.find_entity_by_name_or_alias("tobes").id == t

    def test_exact_active_name_beats_alias(self, store):
        real = store.add_entity("npc", "Tobes")
        other = store.add_entity("npc", "Toby")
        store.add_alias(other, "Tobes")
        assert store.find_entity_by_name_or_alias("Tobes").id == real

    def test_alias_pk_dedups_case_variants(self, store):
        t = store.add_entity("npc", "Toby")
        store.add_alias(t, "Tobes")
        store.add_alias(t, "tobes")   # COLLATE NOCASE PK -> same row
        assert store.list_aliases(t) == ["Tobes"]
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestAliases -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
    def _add_alias_inline(self, conn, entity_id: int, alias: str) -> None:
        conn.execute("INSERT OR IGNORE INTO entity_alias (entity_id, alias) VALUES (?, ?)",
                     (entity_id, alias))

    def add_alias(self, entity_id: int, alias: str) -> None:
        with self.write_transaction() as conn:
            self._add_alias_inline(conn, entity_id, alias)

    def list_aliases(self, entity_id: int) -> list[str]:
        with self.read_connection() as conn:
            return [r[0] for r in conn.execute(
                "SELECT alias FROM entity_alias WHERE entity_id=?", (entity_id,))]

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
                (name,)).fetchone()
        return Entity(*row) if row else None
```

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestAliases -v`

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): entity aliases (NOCASE) + name-or-alias resolution"
```

---

## Task 6: Scene retrieval — cast (word-boundary), 1-hop relations, active-only events

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` — add `RelationView`, `match_entities_in_text`, `relations_for`, `events_for_entities`
- Test: `plugins/llm/tests/verse/test_store.py`

> Plan-red-team SC2: replace the substring heuristic with a **word-boundary regex** + a min-length floor (skip names/aliases ≤2 chars) + a tiny stoplist, to avoid both punctuation misses ("Tobes?") and common-word false-positives (an NPC named "Will"/"Di").

- [ ] **Step 1: Write failing test**

```python
class TestSceneRetrieval:
    def test_match_word_boundary_and_alias(self, store):
        h = store.add_entity("npc", "Harry")
        t = store.add_entity("npc", "Toby"); store.add_alias(t, "Tobes")
        store.add_entity("npc", "Di")  # 2 chars -> skipped (too short)
        got = {e.id for e in store.match_entities_in_text("did Harry and Tobes fight? ask Di.")}
        assert got == {h, t}

    def test_match_handles_punctuation_and_avoids_common_words(self, store):
        w = store.add_entity("npc", "Will")
        assert store.match_entities_in_text("I will go") == []          # common-word, not the name
        assert {e.id for e in store.match_entities_in_text("Will, run!")} == {w}

    def test_relations_for_one_hop(self, store):
        h = store.add_entity("npc", "Harry"); t = store.add_entity("npc", "Toby")
        store.add_relation(h, t, "rival_of", "since year 7")
        assert any(r.from_name == "Harry" and r.to_name == "Toby" and r.kind == "rival_of"
                   for r in store.relations_for([h]))

    def test_events_for_entities_active_only(self, store):
        h = store.add_entity("npc", "Harry"); g = store.add_entity("npc", "Ghost")
        store.add_event("Harry won", [h], source="avatar")
        store.add_event("Ghost faded", [g], source="avatar")
        store.set_status(g, "retired")
        sums = [e.summary for e in store.events_for_entities([h, g], limit=10)]
        assert "Harry won" in sums and "Ghost faded" not in sums
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestSceneRetrieval -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (add `import re` at top of store.py if absent)

```python
class RelationView(NamedTuple):
    from_name: str
    to_name: str
    kind: str
    note: str


_MATCH_STOPLIST = frozenset({"the", "and", "you", "him", "her", "they", "will", "are", "was"})
```

```python
    def match_entities_in_text(self, text: str, limit: int = 12) -> list[Entity]:
        """Active entities whose name OR alias appears as a whole word in `text`
        (case-insensitive). Names/aliases <=2 chars or in the stoplist are skipped
        to avoid common-word false positives. Plain scan — the world is small."""
        low = text.lower()
        with self.read_connection() as conn:
            ent_rows = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at "
                "FROM entities WHERE status='active' ORDER BY id").fetchall()
            alias_rows = conn.execute(
                "SELECT al.entity_id, al.alias FROM entity_alias al "
                "JOIN entities e ON e.id=al.entity_id WHERE e.status='active'").fetchall()
        aliases: dict[int, list[str]] = {}
        for eid, al in alias_rows:
            aliases.setdefault(eid, []).append(al)

        def hit(token: str) -> bool:
            t = token.lower()
            if len(t) <= 2 or t in _MATCH_STOPLIST:
                return False
            return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low) is not None

        out: list[Entity] = []
        for row in ent_rows:
            ent = Entity(*row)
            if any(hit(n) for n in (ent.name, *aliases.get(ent.id, [])) if n):
                out.append(ent)
            if len(out) >= limit:
                break
        return out

    def relations_for(self, entity_ids, limit: int = 30) -> list[RelationView]:
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT ef.name, et.name, r.kind, r.note FROM relations r "
                f"JOIN entities ef ON ef.id=r.from_id JOIN entities et ON et.id=r.to_id "
                f"WHERE (r.from_id IN ({ph}) OR r.to_id IN ({ph})) "
                f"  AND ef.status='active' AND et.status='active' ORDER BY r.id LIMIT ?",
                (*entity_ids, *entity_ids, limit)).fetchall()
        return [RelationView(*r) for r in rows]

    def events_for_entities(self, entity_ids, limit: int = 8) -> list[Event]:
        """Recent events linking any of entity_ids (via event_actor), restricted to
        events that still have >=1 ACTIVE actor. SQL-side active filter."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self.read_connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT ev.id, ev.ts, ev.summary, ev.entity_ids, ev.source FROM events ev "
                f"JOIN event_actor ea ON ea.event_id=ev.id WHERE ea.entity_id IN ({ph}) "
                f"  AND EXISTS (SELECT 1 FROM event_actor ea2 JOIN entities e2 ON e2.id=ea2.entity_id "
                f"              WHERE ea2.event_id=ev.id AND e2.status='active') "
                f"ORDER BY ev.ts DESC, ev.id DESC LIMIT ?", (*entity_ids, limit)).fetchall()
        return [Event(id=r[0], ts=r[1], summary=r[2], entity_ids=_parse_entity_ids(r[3], r[0]), source=r[4])
                for r in rows]
```

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/verse/test_store.py::TestSceneRetrieval -v`

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): scene retrieval (word-boundary match, 1-hop relations, active-only events)"
```

---

## Task 7: Reorder + enrich `build_verse_system_prompt`

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py` — `build_verse_system_prompt` (`:465-577`) + `message_text` param + `VERSE_SCENE_MARKER`
- Modify: `plugins/llm/src/llm/plugin.py:2668` — add `message_text=message_text` to the existing call (the param already exists on `_verse_route_for`)
- Modify: `plugins/llm/src/llm/config.py:379-388` — raise `verseRosterMaxChars` default to `4000`
- Test: `plugins/llm/tests/verse/test_verse_prompt_roster.py`

> Plan-red-team MF7: `_verse_route_for(self, channel, nick, account, message_text)` ALREADY takes `message_text` (plugin.py:2638) and the caller passes `text` (plugin.py:3655). Do NOT change its signature — the only edit is forwarding `message_text` into `build_verse_system_prompt` at plugin.py:2668.
> MF8: name the breaking tests and fix the char-cap slice; the renderer emits `rival of` (`kind.replace('_',' ')`), so assertions must use `rival of`.

- [ ] **Step 1: Write failing tests**

```python
from llm.verse.avatar import build_verse_system_prompt, VERSE_SCENE_MARKER


def test_canon_first_scene_after(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8"); store.set_author_locked(h, True)
    t = store.add_entity("npc", "Toby", "year 9"); store.add_relation(h, t, "rival_of")
    out = build_verse_system_prompt(store, avatar_id, "be a year 8 boy",
                                    roster_max_chars=4000, message_text="did Harry and Toby fight?")
    assert out.index("Established characters") < out.index(VERSE_SCENE_MARKER)
    assert "Harry" in out and "Toby" in out and "rival of" in out


def test_prefix_byte_identical_when_only_message_changes(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8"); store.set_author_locked(h, True)
    a = build_verse_system_prompt(store, avatar_id, "p", roster_max_chars=4000, message_text="hi Harry")
    b = build_verse_system_prompt(store, avatar_id, "p", roster_max_chars=4000, message_text="yo Toby")
    assert a.split(VERSE_SCENE_MARKER)[0] == b.split(VERSE_SCENE_MARKER)[0]
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/verse/test_verse_prompt_roster.py -v -k "canon_first or byte_identical"`
Expected: FAIL.

- [ ] **Step 3: Rewrite `build_verse_system_prompt` stable-first** (use the full body from the v1 plan §Task 7 Step 3 — canon roster via `list_canon_entities`, `VERSE_SCENE_MARKER = "In play right now:"` constant, volatile scene/events(active-only)/others/matched-cast/relations/scene-events after the marker; relation line uses `r.kind.replace('_',' ')`).

```python
VERSE_SCENE_MARKER = "In play right now:"
```

> Implement exactly as in the prior plan revision's Task 7 Step 3 body. KEY invariants the tests pin: (a) identity+persona+`Established characters in this world:`+roster come BEFORE `VERSE_SCENE_MARKER`; (b) everything message-dependent comes after; (c) relation lines render `from kind-with-spaces to`.

- [ ] **Step 4: Forward `message_text` (one line, plugin.py:2668)**

Add `message_text=message_text,` to the existing `build_verse_system_prompt(store, avatar_id, persona, roster_max_chars=...)` call. Do NOT touch `_verse_route_for`'s signature.

- [ ] **Step 5: Raise the roster cap (config.py:379-388)** — change the `PositiveInteger(600, ...)` default to `PositiveInteger(4000, ...)`; update the func-signature default in `build_verse_system_prompt` to `4000`.

- [ ] **Step 6: Fix the named breaking tests**

`test_verse_prompt_roster.py::test_roster_respects_char_cap` (and any `TestSystemPrompt` ordering assertions): the roster is now BEFORE the scene block, so `prompt.split("Established characters in this world:")[1]` includes the whole scene tail. Re-slice between the roster header and `VERSE_SCENE_MARKER`:

```python
    seg = prompt.split("Established characters in this world:")[1].split(VERSE_SCENE_MARKER)[0]
    assert len(seg) <= <cap>
```

Run `uv run pytest plugins/llm/tests/verse/test_verse_prompt_roster.py plugins/llm/tests/verse/test_avatar.py -v` and fix every assertion that depended on the OLD ordering (intended change). Search those files for the old scene/roster ordering assumptions.

- [ ] **Step 7: Run → pass / commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/src/llm/plugin.py plugins/llm/src/llm/config.py plugins/llm/tests/verse/
git commit -m "feat(verse): stable-first verse prompt with scene cast + relations + active-only events"
```

---

## Task 8 (DEFERRED): auto-lock-canon-by-talking

Deferred to a fast-follow per the plan-red-team + the keep-it-simple decision. Tasks 6–7 + the already-pinned roster + `@canon` deliver the v1 retention win. When added later, the hardening is: `msg.prefix`-bound `verse_record` overlay gated on `llm.verse.edit`, word-boundary human-offered match, **alias-aware** target resolution (resolve to canonical id before the base `record_user_event` auto-creates a duplicate), promote only on a parsed `{status:ok}` event, de-dup actors per turn, single write transaction, threshold via a new `verseAuthorLockMentions` channel knob.

---

## Task 9: `@canon lock/unlock/forget` command (author-gated)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — add `canon` command near `verse`/`look`/`who` (`:5790-5882`)
- Test: `plugins/llm/tests/test_plugin_verse.py` (has the real verse-command harness)

> Plan-red-team MF6: verse commands take `(irc, msg, args[, target])` and resolve the channel **in-body** via `channel = self._check_verse_channel(irc, msg)` (plugin.py:5732). Do NOT put `channel` in the signature/wrap.

- [ ] **Step 1: Write failing test** — mirror the existing `@look`/`@verse` command test in `test_plugin_verse.py` (same harness: real plugin, `_check_verse_channel` returns the channel, `ircdb.checkCapability` granted). Assert: after `canon lock Harry`, `store.get_attribute(h.id, "author_locked") == "1"`; after `canon forget Harry`, it's `None`.

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/test_plugin_verse.py -k canon -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
    def canon(self, irc, msg, args, action, name):
        """<lock|unlock|forget> <name>

        Lock or release a character as durable canon (always remembered).
        Requires the llm.verse.edit capability.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return
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

(`action == "lock"` → lock; `unlock`/`forget` → unlock.)

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/test_plugin_verse.py -k canon -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "feat(verse): @canon lock/unlock/forget (author-gated, channel resolved in-body)"
```

---

## Task 10: Record canon on storybook turns (in `_submit_storybook_job`)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_submit_storybook_job` (`:2739`); add `account` param; update its 2 callers
- Test: `plugins/llm/tests/test_storybook.py`

> Plan-red-team MF5: storybook short-circuits before any `verse_record` model step, so record canon in the shared `_submit_storybook_job` (covers BOTH the tool handler and `@story`). Resolve the avatar inline (no route rebuild), guard non-None, best-effort.

- [ ] **Step 1: Write failing test** — using the existing `test_storybook.py` harness (service-level via the plugin), fire a storybook for a channel whose store has an avatar for the caller; assert exactly one new event recorded with the brief text. Mirror the file's existing storybook test setup.

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/test_storybook.py -k canon -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — add `account: str | None = None` to `_submit_storybook_job`'s keyword-only params; at the TOP of the method body (before the `submit`), best-effort record:

```python
        try:
            store = self._get_or_create_verse_store(channel)
            avatar_id = (store.find_avatar_by_account(account) if account else None) \
                or store.find_avatar_by_nick(nick)
            if avatar_id is not None:
                store.record_user_event(actor_id=avatar_id,
                                        summary=(brief.strip()[:200] or "told an illustrated tale"),
                                        actor_names=[])
        except Exception:
            self.log.exception("storybook canon-record failed (non-fatal) channel=%s", channel)
```

Update both callers to pass `account=`: the storybook tool handler (`_storybook_handler` `_call`, ~plugin.py:2834 — it has `account` in scope) and the `@story` command (plugin.py:4155 — pass its account if any, else `None`; a no-avatar `@story` simply skips the record via the guard).

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/test_storybook.py -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_storybook.py
git commit -m "fix(verse): record a canon event on storybook turns (tool + @story)"
```

---

## Task 11: Loud warning when `verseModel` is empty (falls back to assistantModel)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:3702`
- Test: `plugins/llm/tests/test_plugin_verse.py`

> Plan-red-team SC3: `self.log` is a supybot plugin logger `caplog` cannot capture — assert on `plugin.log.warning.call_args` (the documented pattern at `test_plugin_verse.py:2787`). Gate the warning **once per channel** (don't fire every turn on an unset channel). Spec-delta vs §9 (warn, not hard-fail) is documented in the header.

- [ ] **Step 1: Write failing test** — mirror the `test_plugin_verse.py:2787` logging-assertion pattern: with `verseModel` empty for the channel, drive a verse turn; assert `plugin.log.warning` was called with a message mentioning `verseModel` and `assistantModel`; drive a second turn on the same channel and assert it is NOT warned again (once-per-channel).

- [ ] **Step 2: Run → fail**

Run: `uv run pytest plugins/llm/tests/test_plugin_verse.py -k verse_model_warn -v`
Expected: FAIL.

- [ ] **Step 3: Implement** — at plugin.py:3702, after reading `verse_model`, with a per-instance `set` guard (e.g. `self._verse_model_warned`):

```python
            verse_model = self.registryValue("verseModel", preflight.channel) or None
            if verse_model is None and preflight.channel not in self._verse_model_warned:
                self._verse_model_warned.add(preflight.channel)
                self.log.warning(
                    "verse turn on channel=%s has empty verseModel; falling back to "
                    "assistantModel — set a non-reasoning verseModel or verse prose may be "
                    "cratered by a reasoning model", preflight.channel)
```

Initialize `self._verse_model_warned: set[str] = set()` in `__init__`.

- [ ] **Step 4: Run → pass / commit**

Run: `uv run pytest plugins/llm/tests/test_plugin_verse.py -k verse_model_warn -v`

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "feat(verse): warn-once-per-channel when verseModel empty and falls back"
```

---

## Task 12: Characterize the denial/degrade retry re-seed (no behaviour change)

**Files:**
- Test only: `plugins/llm/tests/test_service_completion.py`

> Pin current behaviour (service.py appends `[assistant: rejected, user: nudge]` before `continue`); do NOT modify service.py. Build on the existing stub-client harness in `test_service_completion.py` (it already drives multi-step completions). If the existing harness cannot snapshot the in-flight `messages`, downscope to asserting the corrected (not rejected) text is returned and add an inline comment that the re-seed is intentional.

- [ ] **Step 1: Write the characterization test** (against the real harness; corrected text returned, rejected text never delivered).
- [ ] **Step 2: Run → pass** (`uv run pytest plugins/llm/tests/test_service_completion.py -k reseed -v`). If it can't be expressed on the real harness, write the narrower "corrected-text-returned" assertion instead.
- [ ] **Step 3: Commit** (`test(verse): characterize denial-retry re-seed before any future change`).

---

## Task 13: End-to-end retrieval integration test

**Files:**
- Test: `plugins/llm/tests/verse/test_retrieval_integration.py` (new)

- [ ] **Step 1: Write** — uses `store_with_avatar`; locked roster member unmentioned still appears in the prefix; alias-resolved member + relation (`rival of`) + their event appear in the scene block; a retired entity appears nowhere. (Body as in the prior revision's Task 13, with `rival of` spacing and `VERSE_SCENE_MARKER` split.)
- [ ] **Step 2: Run → pass** (`uv run pytest plugins/llm/tests/verse/test_retrieval_integration.py -v`).
- [ ] **Step 3: Commit** (`test(verse): end-to-end retrieval integration`).

---

## Task 14: Full gate

- [ ] **Step 1:** `make test` (full suite + 93% coverage). Fix any prompt-ordering tests that asserted the old layout.
- [ ] **Step 2:** `make lint && make typecheck` → clean.
- [ ] **Step 3:** Confirm non-verse untouched: `uv run pytest plugins/llm/tests/test_assistant.py plugins/llm/tests/test_service_core.py -v`.
- [ ] **Step 4:** Commit any fixups.

---

## Rollout (operator, post-merge)
1. v2→v3 migration runs automatically on first open of each verse store (additive; backfills `event_actor`; idempotent). #afternet DB is ~2.9MB — backfill is a fast single transaction.
2. Verse stays behind the existing per-channel `verseEnabled` flag; rollback = registry flip / revert (v3 is additive, nothing to un-migrate).
3. Watch `docker logs vibebot` for the `verseModel` warning and any migration error on first open.
4. Optionally seed roster aliases later (no `@canon alias` verb in v1).

## Coverage (plan vs spec)
§3.1 retrieval → Tasks 6,7,13. §3.2 author_locked + explicit lock → Tasks 3,9 (auto-promotion deferred, Task 8). §3.3 alias → Task 5. §3.4 event_actor → Tasks 1,2,6. §3.5 cache (stable-first, documented delta) → Task 7. §3.6 generation (verseModel warn, documented delta) → Task 11. §3.7 storybook canon-write → Task 10. Aging/loom protection → Task 4. Re-seed characterize → Task 12.

## Deferred / fast-follow
Auto-lock-by-talking (Task 8); freq_penalty drop-log observability; lightweight injected-vs-referenced telemetry (§8); §5 user-role-message cache split; §9 hard-fail + reasoning-model startup validation; new-plugin / gen-core / story-fan-out / portraits (later phases per spec §11).
