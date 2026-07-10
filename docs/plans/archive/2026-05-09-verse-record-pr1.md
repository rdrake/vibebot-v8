# `verse_record` + Auto-Entity Aging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `verse_record` assistant tool plus auto-NPC aging from `docs/plans/2026-05-09-verse-record-design.md` (v2.1) — let opted-in verse members narrate canon involving entities other than themselves, auto-create unknown actors as NPCs, and soft-retire unmentioned auto-NPCs after `verseAutoEntityRetireDays` of silence.

**Architecture:**
- Two prerequisite refactors: `dispatch_verse_tool_call` is retrofitted to return a structured `VerseDispatchResult` (so error/success payloads reach the model — today the wrapper always returns `{"status":"ok"}`); the four public store mutators grow `_*_inline(conn, …)` helpers so `record_user_event` can do find-or-create-then-link inside one `write_transaction` without deadlocking the non-reentrant `threading.Lock`.
- One new pure aging helper (`verse/aging.py`); one new store method (`record_user_event`); one new tool spec entry (`make_verse_tool_specs(*, max_actors)` gains `verse_record`); one new dispatch branch; three heartbeat call-sites (record, loom apply, compaction-digest insert); two new registry keys; `compact_verse` returns a `CompactionOutcome` NamedTuple instead of a string so the friendlier compaction outcome message can render.

**Tech Stack:**
- Python 3.13+ (`uv`, `pytest`, `ruff`, `ty`).
- Real SQLite in `tmp_path` for tests. **No DB mocks.**
- Existing `dispatch_verse_tool_call` envelope (no new LiteLLM wiring).

**Reference design:** `docs/plans/2026-05-09-verse-record-design.md` (v2.1). Reference PR-plan format: `docs/plans/2026-05-08-forest-verse-pr3.md`.

**Working directory:** the live tree (`yolo to main` per repo convention) — no worktree split. Push directly to `main`. CI + Docker build are separate workflows; wait for both before restarting prod.

---

## Scope guard

This PR ships **only** the items in the design doc's §10 line-estimate table. Nothing else. If a task tempts you outside that list, **stop** — either it belongs (then add a Revisions entry to the design doc first) or it doesn't (then it's a separate PR).

PR ships **none** of:
- Schema migration (no new `events.source` value — reuses `'avatar'`).
- Kind inference for auto-create (always `npc`).
- `verse_unrecord` / edit-event / un-retire UI.
- Bulk auto-create rate-limit knobs.
- Item / place auto-create.
- Embedding-based recall changes.

---

## Files map

```
plugins/llm/src/llm/verse/
  avatar.py              MOD  VerseDispatchResult; dispatch returns it; verse_record branch;
                              make_verse_tool_specs(*, max_actors) gains verse_record entry
  store.py               MOD  inline-helper extraction (4 public mutators);
                              find_active_entity_by_name + inline; list_entities_with_attribute;
                              record_user_event; heartbeat in _replace_events_with_source;
                              public bump_last_seen_ts (used by loom)
  loom.py                MOD  apply_or_queue bumps last_seen_ts on applied / crosspoll_emitted
  compaction.py          MOD  CompactionOutcome NamedTuple replaces string return
  aging.py               NEW  AgingOutcome + age_auto_created_entities pure helper
plugins/llm/src/llm/
  config.py              MOD  verseAutoEntityRetireDays (default 14), verseAutoEntityMaxNamesPerCall (default 8)
  plugin.py              MOD  _run_compaction_pass calls aging per channel; new outcome string;
                              make_verse_extra_handlers callsite at :3281 plumbs max_actors
plugins/llm/tests/verse/
  test_verse_record.py   NEW  tests #1–13 (§7 of design)
  test_verse_aging.py    NEW  tests #1–9 (§7 of design — aging + heartbeat)
  test_avatar.py         MOD  4-set → 5-set; dispatch contract migration (Tests #14, #16)
  test_compaction.py     MOD  all 8 string-equality sites → NamedTuple .state (Test #15)
  test_store.py          MOD  inline-helper unit tests + new store-query unit tests
  test_loom.py           MOD  apply_or_queue heartbeat tests #8, #9
plugins/llm/tests/
  test_plugin.py         MOD  plugin tests #10–13 (compaction-pass wiring + max_actors plumbing)
  test_config.py         MOD  two new registry keys
docs/guide/operator/
  forest-verse.md        MOD  3 new H2 sections (member-driven, aging, compaction outcome)
CHANGELOG.md             MOD  unreleased entry
```

---

## Test → Task mapping (§7 of design doc)

| §7 Test | Phase / Task |
|---|---|
| #1 all-new actors create NPCs, caller's avatar first in entity_ids | 2.1 |
| #2 avatar precedence; no `auto_created`; no `last_seen_ts` bump on avatar | 2.2 |
| #3 existing NPC reused; `last_seen_ts` bumped | 2.3 |
| #4 repeated calls keep one row; `last_seen_ts` is most recent | 2.4 |
| #5 race with `time.sleep` injection — only one entity row | 2.5 |
| #6 retired entity not rehydrated — new active NPC created | 2.6 |
| #7 truncate to N after non-string filter (filter-then-slice) | 6.5 |
| #8 empty `summary` → `VerseDispatchResult(ok=False, …)` | 6.3 |
| #9 too-long `summary` → error; no truncation | 6.4 |
| #10 opt-out → re-opt-in three-row state machine | 2.9 |
| #11 case-insensitive `actors=["ANDREW"]` → avatar "andrew" | 2.8 |
| #12 empty/whitespace actor strings filtered before slicing | 6.5 |
| #13 retired `actor_id` raises | 2.10 |
| #14 4-set → 5-set in `test_avatar.py:617` + `_verse_names` literal | 6.1 |
| #15 8 string-equality sites in `test_compaction.py` migrate to `.state` | 5b.2 |
| #16 dispatch-contract migration (existing four tools) | 0a.1 |
| Aging #1–5 basic aging cases | 3.2 – 3.6 |
| Aging #6–7 compaction-digest heartbeat / truncation | 4.1 – 4.2 |
| Aging #8–9 loom apply heartbeat positive / negative | 4.3 – 4.4 |
| Plugin #10–12 compaction-pass wiring + per-channel scope + isolation | 5a.1 – 5a.3 |
| Plugin #13 `max_actors` registry → dispatch closure plumbing | 6.6 |

---

## Phase 0a — Dispatch contract retrofit

**Why first:** v2.1 design FATAL #1. Today `dispatch_verse_tool_call` (`verse/avatar.py:383`) returns `None`, and `make_verse_extra_handlers._handler._call` (`verse/avatar.py:454-460`) hard-codes `{"status":"ok","tool":name}`. Validation errors and `event_id` payloads from `verse_record` (Phase 6) would be invisible to the model under the current contract. Fix the contract before any code that depends on it.

### Task 0a.1: introduce `VerseDispatchResult` and migrate the four existing branches

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py:383-435` (`dispatch_verse_tool_call`).
- Modify: `plugins/llm/tests/verse/test_avatar.py` (existing dispatch tests — Test #16).

- [ ] **Step 1: failing test** — append to `plugins/llm/tests/verse/test_avatar.py`:

```python
class TestDispatchContract:
    def test_existing_tools_return_ok_result(self, store: VerseStore) -> None:
        """GIVEN any of the four existing tools WHEN dispatched THEN returns
        VerseDispatchResult(ok=True, payload={'status':'ok'}). The wrapper's
        observable JSON is unchanged so the model's tool-result payloads
        do not regress."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )
        alice_id = _opt_in(store)
        for name, args in [
            ("verse_act", {"verb": "speak"}),
            ("verse_move", {"place_name": "anywhere"}),
            ("verse_look", {}),
            ("verse_recall", {"query": "x"}),
        ]:
            result = dispatch_verse_tool_call(store, alice_id, name, args)
            assert isinstance(result, VerseDispatchResult)
            assert result.ok is True
            assert result.payload == {"status": "ok"}
            assert result.error is None

    def test_unknown_tool_returns_ok_with_warning(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown tool name still doesn't raise; result is ok=True with
        the same payload (preserves today's silent-skip behaviour)."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )
        alice_id = _opt_in(store)
        with caplog.at_level(logging.WARNING, logger="llm.verse.avatar"):
            result = dispatch_verse_tool_call(
                store, alice_id, "hallucinated_tool", {"x": 1}
            )
        assert isinstance(result, VerseDispatchResult)
        assert result.ok is True
        assert result.payload == {"status": "ok"}
```

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_avatar.py::TestDispatchContract -v` → fails with `ImportError: cannot import name 'VerseDispatchResult'`.

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/avatar.py`. Add the dataclass near the top (after the existing imports, before `make_verse_tool_specs`):

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class VerseDispatchResult:
    """Structured result of a verse tool dispatch.

    The four legacy tools (verse_act / verse_move / verse_look / verse_recall)
    return ok=True with payload={'status': 'ok'}, preserving the wrapper's
    historical observable JSON. New branches (verse_record) populate
    payload with tool-specific data on success or error with a model-
    facing string on failure.
    """
    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None
```

Change `dispatch_verse_tool_call` signature and every existing branch to return `VerseDispatchResult` instead of `None`:

```python
def dispatch_verse_tool_call(
    store: VerseStore,
    avatar_id: int,
    name: str,
    args: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> VerseDispatchResult:
    log = logger or _log
    _OK = VerseDispatchResult(ok=True, payload={"status": "ok"})
    try:
        if name == "verse_act":
            verb = args.get("verb")
            if not verb:
                log.warning("verse_act missing 'verb' arg (avatar=%s)", avatar_id)
                return _OK
            verse_act(store, avatar_id, verb, args.get("target"), args.get("details"))
            return _OK
        elif name == "verse_move":
            place = args.get("place_name")
            if not place:
                log.warning("verse_move missing 'place_name' arg (avatar=%s)", avatar_id)
                return _OK
            verse_move(store, avatar_id, place)
            return _OK
        elif name == "verse_look":
            verse_look(store, avatar_id, args.get("target"))
            return _OK
        elif name == "verse_recall":
            q = args.get("query")
            if q is None:
                log.warning("verse_recall missing 'query' arg (avatar=%s)", avatar_id)
                return _OK
            verse_recall(store, q)
            return _OK
        else:
            log.warning("unknown verse tool: %s (avatar=%s)", name, avatar_id)
            return _OK
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "verse tool dispatch failed: name=%s avatar=%s err=%s",
            name, avatar_id, exc,
        )
        return _OK
```

(Existing-tool failure preserves today's *swallow-and-skip* semantics — non-`verse_record` branches never return `ok=False`. Phase 6 will add the `verse_record` branch that does.)

- [ ] **Step 4: run** `uv run pytest plugins/llm/tests/verse/test_avatar.py::TestDispatchContract -v` → green.

- [ ] **Step 5: run** the full file `uv run pytest plugins/llm/tests/verse/test_avatar.py -v` → still green (no regressions).

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/verse/test_avatar.py
git commit -m "refactor(verse/avatar): dispatch_verse_tool_call returns VerseDispatchResult"
```

### Task 0a.2: `make_verse_extra_handlers` consumes `VerseDispatchResult`

**Why split:** the dispatch-result type and the wrapper's consumption of it are independently testable. Keeping them as separate commits makes a regression bisect easy.

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py:438-462` (`make_verse_extra_handlers`).
- Modify: `plugins/llm/tests/verse/test_avatar.py` (existing handler tests).

- [ ] **Step 1: failing test** — append:

```python
class TestHandlerConsumesResult:
    def test_handler_emits_payload_on_ok(self, store: VerseStore) -> None:
        """ok=True with custom payload — handler serialises payload as
        JSON, includes 'tool' key for backwards compat."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            make_verse_extra_handlers,
        )
        alice_id = _opt_in(store)
        handlers = make_verse_extra_handlers(store, alice_id)
        result = handlers["verse_act"]({"verb": "speak"})
        payload = json.loads(result.content)
        assert payload["status"] == "ok"
        assert payload["tool"] == "verse_act"

    def test_handler_emits_error_on_not_ok(
        self, store: VerseStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ok=False with error string — handler emits {'status':'error',
        'error': <error>} so the model sees a structured failure."""
        from llm.verse import avatar as avatar_mod

        def fake_dispatch(*a, **k):
            return avatar_mod.VerseDispatchResult(
                ok=False, error="summary required"
            )

        monkeypatch.setattr(avatar_mod, "dispatch_verse_tool_call", fake_dispatch)
        alice_id = _opt_in(store)
        handlers = avatar_mod.make_verse_extra_handlers(store, alice_id)
        result = handlers["verse_act"]({"verb": "speak"})
        payload = json.loads(result.content)
        assert payload["status"] == "error"
        assert payload["error"] == "summary required"
        assert payload["tool"] == "verse_act"
```

- [ ] **Step 2: run** → fails (handler still ignores the result).

- [ ] **Step 3: implement** — replace `_handler` body in `verse/avatar.py:454-460`:

```python
def _handler(name: str) -> Callable[[dict[str, Any]], _VerseToolResult]:
    def _call(args: dict[str, Any]) -> _VerseToolResult:
        result = dispatch_verse_tool_call(store, avatar_id, name, args, logger=log)
        if result.ok:
            payload = {"status": "ok", "tool": name}
            if result.payload:
                payload.update(result.payload)
            return _VerseToolResult(content=json.dumps(payload))
        return _VerseToolResult(
            content=json.dumps({
                "status": "error",
                "error": result.error or "unknown error",
                "tool": name,
            })
        )

    _call.__name__ = f"_verse_handler_{name}"
    return _call
```

(`payload.update(result.payload)` lets `verse_record` add its `event_id` field without losing `status`/`tool`. The `if result.payload` guard handles the legacy tools where payload is `{"status":"ok"}` — `payload.update` overwrites `status` with the same value, no harm done.)

- [ ] **Step 4: run** → green. Run the full `test_avatar.py` to confirm no regression: `uv run pytest plugins/llm/tests/verse/test_avatar.py -v`.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/verse/test_avatar.py
git commit -m "refactor(verse/avatar): handlers surface VerseDispatchResult to model"
```

### Phase 0a verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_avatar.py -v` → all green.
- [ ] `make lint && make typecheck` → clean.

---

## Phase 0b — Store mutator inline-helper extraction

**Why second:** v2.1 design FATAL #2. `write_transaction` uses `threading.Lock` (not RLock — see warning at `store.py:471-475`). `record_user_event` (Phase 2) needs to call `add_entity` / `set_attribute` / `add_event` from inside its own `write_transaction`; calling the public method would acquire the lock recursively and **deadlock**. Extract `_*_inline(conn, …)` helpers that take an open `conn` and skip locking; public methods become thin wrappers. The closest existing precedent is `opt_in_avatar` (`store.py:465-560`), which inlines all its DB work directly inside one `write_transaction() as conn` block.

**Behaviour change:** zero. All existing store tests stay green throughout this phase.

### Task 0b.1: extract `_add_entity_inline`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py:151-188` (`add_entity`).
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: failing test** — append a unit test pinning the inline helper's contract:

```python
class TestInlineHelpers:
    def test_add_entity_inline_runs_on_caller_conn(
        self, verse_db_dir: Path
    ) -> None:
        """Caller opens its own write_transaction, calls _add_entity_inline,
        and a sibling INSERT in the same tx — all without lock reentry."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        with store.write_transaction() as conn:
            eid = store._add_entity_inline(  # type: ignore[attr-defined]
                conn, "npc", "ghost", "a wisp of vapour"
            )
            # Sibling INSERT in the same tx proves we hold the same conn.
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) "
                "VALUES (?, 'inline_marker', '1')",
                (eid,),
            )
        assert store.find_entity_by_name("ghost", kind="npc") is not None
        assert store.get_attribute(eid, "inline_marker") == "1"
```

- [ ] **Step 2: run** → fails (`_add_entity_inline` does not exist).

- [ ] **Step 3: implement** — refactor `add_entity` (`store.py:151-188`). Move the SQL body into `_add_entity_inline(conn, kind, name, summary)`; `add_entity` becomes a thin wrapper. The before/after:

**Before** (current code):

```python
def add_entity(self, kind: str, name: str, summary: str = "") -> int:
    now = time.time()
    with self.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO entities (kind, name, summary, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'active', ?, ?)",
            (kind, name, summary, now, now),
        )
        return int(cur.lastrowid)  # plus the existing dedupe / kind-coerce logic
```

**After:**

```python
def _add_entity_inline(
    self,
    conn: sqlite3.Connection,
    kind: str,
    name: str,
    summary: str = "",
) -> int:
    """Insert an entity on the caller's open ``conn``. Caller is
    responsible for the surrounding write_transaction. Returns the
    new entity's id."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO entities (kind, name, summary, status, created_at, updated_at)"
        " VALUES (?, ?, ?, 'active', ?, ?)",
        (kind, name, summary, now, now),
    )
    assert cur.lastrowid is not None
    return int(cur.lastrowid)

def add_entity(self, kind: str, name: str, summary: str = "") -> int:
    with self.write_transaction() as conn:
        return self._add_entity_inline(conn, kind, name, summary)
```

(If the existing `add_entity` carries dedupe / coercion logic above the INSERT, mirror it inside `_add_entity_inline`. Read the current 151-188 body first; preserve every assertion the existing tests rely on.)

- [ ] **Step 4: run** `uv run pytest plugins/llm/tests/verse/test_store.py -v` → all existing add_entity tests stay green; `TestInlineHelpers::test_add_entity_inline_runs_on_caller_conn` is green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "refactor(verse/store): extract _add_entity_inline helper"
```

### Task 0b.2: extract `_set_attribute_inline`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py:231-239` (`set_attribute`).
- Modify: `plugins/llm/tests/verse/test_store.py` (extend `TestInlineHelpers`).

- [ ] **Step 1: failing test** — append:

```python
def test_set_attribute_inline_upserts_on_caller_conn(
    self, verse_db_dir: Path
) -> None:
    from llm.verse.store import VerseStore

    store = VerseStore(verse_db_dir, "#inline")
    eid = store.add_entity("npc", "moss", "")
    with store.write_transaction() as conn:
        store._set_attribute_inline(  # type: ignore[attr-defined]
            conn, eid, "k", "v1"
        )
        store._set_attribute_inline(  # type: ignore[attr-defined]
            conn, eid, "k", "v2"
        )
    assert store.get_attribute(eid, "k") == "v2"
```

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement** in `store.py:231-239`:

```python
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
    with self.write_transaction() as conn:
        self._set_attribute_inline(conn, entity_id, key, value)
```

- [ ] **Step 4: run** → green; full `test_store.py` green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "refactor(verse/store): extract _set_attribute_inline helper"
```

### Task 0b.3: extract `_add_event_inline`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py:296-…` (`add_event`).
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: failing test**:

```python
def test_add_event_inline_writes_on_caller_conn(
    self, verse_db_dir: Path
) -> None:
    from llm.verse.store import VerseStore

    store = VerseStore(verse_db_dir, "#inline")
    eid = store.add_entity("avatar", "alice", "")
    with store.write_transaction() as conn:
        ev_id = store._add_event_inline(  # type: ignore[attr-defined]
            conn,
            summary="alice waved",
            entity_ids=[eid],
            source="avatar",
        )
    events = store.recent_events(limit=10)
    assert any(e.id == ev_id and e.summary == "alice waved" for e in events)
```

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement** — refactor `add_event` body into `_add_event_inline(conn, *, summary, entity_ids, source, ts=None)`. Keep timestamp-default and JSON-encoding logic in the inline helper; `add_event` becomes the wrapper:

```python
def _add_event_inline(
    self,
    conn: sqlite3.Connection,
    *,
    summary: str,
    entity_ids: Sequence[int],
    source: str,
    ts: float | None = None,
) -> int:
    """Insert an event row on the caller's open ``conn``. Returns the
    new event's id."""
    if ts is None:
        ts = time.time()
    cur = conn.execute(
        "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
        (ts, summary, json.dumps(list(entity_ids)), source),
    )
    assert cur.lastrowid is not None
    return int(cur.lastrowid)

def add_event(
    self,
    *,
    summary: str,
    entity_ids: Sequence[int],
    source: str,
    ts: float | None = None,
) -> int:
    with self.write_transaction() as conn:
        return self._add_event_inline(
            conn, summary=summary, entity_ids=entity_ids, source=source, ts=ts
        )
```

(Preserve the existing public signature exactly — read `store.py:296` first to capture every kwarg.)

- [ ] **Step 4: run** → full `test_store.py` green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "refactor(verse/store): extract _add_event_inline helper"
```

### Task 0b.4: extract `_set_status_inline`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py:204-212` (`set_status`).

- [ ] **Step 1: failing test**:

```python
def test_set_status_inline_updates_on_caller_conn(
    self, verse_db_dir: Path
) -> None:
    from llm.verse.store import VerseStore

    store = VerseStore(verse_db_dir, "#inline")
    eid = store.add_entity("npc", "ghost", "")
    with store.write_transaction() as conn:
        store._set_status_inline(  # type: ignore[attr-defined]
            conn, eid, "retired"
        )
    fetched = store.find_entity_by_name("ghost")
    assert fetched is None  # find_entity_by_name filters retired in active-first mode
    # status reads via get_entity_by_id are scope of phase 1; assert via raw SQL here
    with store.read_connection() as conn:
        row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
    assert row[0] == "retired"
```

(`find_entity_by_name` does not currently filter retired — see design §2. The active-only filter ships in Phase 1. For *this* test, assert via raw SQL on the entities table; the test pins the inline helper's behaviour, not the lookup helper's.)

Re-check `find_entity_by_name`'s actual behaviour by reading `store.py:189-203` before writing the assertion. Adjust the inline-helper test's *negative* check to match: if `find_entity_by_name` already returns retired rows, the first `assert` becomes `assert fetched is not None and fetched.status was checked via raw SQL` — keep the SQL-verify as the source of truth.

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement**:

```python
def _set_status_inline(
    self,
    conn: sqlite3.Connection,
    entity_id: int,
    status: str,
) -> None:
    """Update entity status + updated_at on the caller's open ``conn``.
    Silent no-op if entity_id not found."""
    now = time.time()
    conn.execute(
        "UPDATE entities SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, entity_id),
    )

def set_status(self, entity_id: int, status: str) -> None:
    with self.write_transaction() as conn:
        self._set_status_inline(conn, entity_id, status)
```

- [ ] **Step 4: run** → green; full `test_store.py` green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "refactor(verse/store): extract _set_status_inline helper"
```

### Phase 0b verification

- [ ] `uv run pytest plugins/llm/tests/verse/ -v` → all green (no behaviour change to public API).
- [ ] `make lint && make typecheck` → clean.

---

## Phase 1 — New store queries

### Task 1.1: `find_active_entity_by_name` + inline variant

**Why:** v2.1 design §2 — today's `find_entity_by_name(name)` returns the first id ASC across **all kinds and statuses**, contradicting the spec'd `avatar > npc > item > place` precedence and silently rehydrating retired entities. Add a new helper restricted to `status='active'` with an explicit `CASE WHEN kind='avatar' THEN 0 …` ordering.

The legacy `find_entity_by_name(name, kind=...)` stays — `verse_act`'s movement/item lookups want a kind filter and don't care about precedence.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (alongside `find_entity_by_name` at `:189-203`).
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: failing test** — append a new test class:

```python
class TestFindActiveEntityByName:
    def test_avatar_wins_over_npc(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        npc_id = store.add_entity("npc", "Andrew", "")
        avatar_id = store.add_entity("avatar", "Andrew", "")
        result = store.find_active_entity_by_name("Andrew")
        assert result is not None and result.id == avatar_id
        # Sanity: legacy kind-filtered call still finds the npc
        legacy = store.find_entity_by_name("Andrew", kind="npc")
        assert legacy is not None and legacy.id == npc_id

    def test_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        eid = store.add_entity("avatar", "andrew", "")
        result = store.find_active_entity_by_name("ANDREW")
        assert result is not None and result.id == eid

    def test_skips_retired(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        eid = store.add_entity("npc", "ghost", "")
        store.set_status(eid, "retired")
        assert store.find_active_entity_by_name("ghost") is None

    def test_returns_none_when_no_match(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        assert store.find_active_entity_by_name("nobody") is None

    def test_inline_variant_runs_on_caller_conn(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        store.add_entity("npc", "moss", "")
        with store.read_connection() as conn:
            result = store._find_active_entity_by_name_inline(  # type: ignore[attr-defined]
                conn, "moss"
            )
        assert result is not None and result.kind == "npc"

    def test_npc_beats_item(self, verse_db_dir: Path) -> None:
        """avatar > npc > item > place precedence — verify mid-tier."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        item_id = store.add_entity("item", "shard", "")
        npc_id = store.add_entity("npc", "shard", "")
        result = store.find_active_entity_by_name("shard")
        assert result is not None and result.id == npc_id
        # Ensure the item still exists — we picked, not deleted.
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE id IN (?, ?)",
                (item_id, npc_id),
            ).fetchone()[0]
        assert count == 2
```

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement** — add to `verse/store.py` near `find_entity_by_name`:

```python
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
```

(Match `Entity` field order to the existing NamedTuple definition at the top of `store.py`. If `Entity` has more or fewer columns than the SELECT lists, adjust both consistently.)

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse/store): find_active_entity_by_name with avatar>npc>item>place precedence"
```

### Task 1.2: `list_entities_with_attribute(key, value, *, status)`

**Why:** Phase 3's aging helper does a single SQL `SELECT entities.id, attributes.value AS last_seen FROM entities JOIN attributes …` to find auto-created entities past cutoff. Encapsulate that as a store method so aging stays pure.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: failing test**:

```python
class TestListEntitiesWithAttribute:
    def test_returns_matching_entities(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        b = store.add_entity("npc", "beta", "")
        store.add_entity("npc", "gamma", "")  # no attribute
        store.set_attribute(a, "auto_created", "1")
        store.set_attribute(b, "auto_created", "1")
        rows = store.list_entities_with_attribute(
            key="auto_created", value="1", status="active"
        )
        assert {e.id for e in rows} == {a, b}

    def test_status_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        store.set_attribute(a, "auto_created", "1")
        store.set_status(a, "retired")
        active = store.list_entities_with_attribute(
            key="auto_created", value="1", status="active"
        )
        assert active == []

    def test_value_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        store.set_attribute(a, "auto_created", "0")
        rows = store.list_entities_with_attribute(
            key="auto_created", value="1", status="active"
        )
        assert rows == []
```

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement**:

```python
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
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse/store): list_entities_with_attribute query"
```

### Phase 1 verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_store.py -v` → green.

---

## Phase 2 — `record_user_event`

**File:** `plugins/llm/tests/verse/test_verse_record.py` (NEW). Each task below adds one test class.

All tests share a `record_user_event` setup fixture; put it once at the top of the new file:

```python
"""Tests for VerseStore.record_user_event (verse_record's atomic DB path)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#record")


def _opt_in(store: VerseStore, nick: str = "alice") -> int:
    """Convenience: opt a nick into the verse and return the avatar entity id."""
    result = store.opt_in_avatar(nick, account=None, instruct_text=f"{nick} instruct")
    return result.entity_id
```

### Task 2.1: happy path — all-new actors create NPCs (Test #1)

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (new method `record_user_event`).
- Create: `plugins/llm/tests/verse/test_verse_record.py`.

- [ ] **Step 1: failing test** — append:

```python
class TestRecordUserEvent:
    def test_all_new_names_create_npcs_and_link_event(
        self, store: VerseStore
    ) -> None:
        """GIVEN unknown actor names WHEN record_user_event called THEN
        each name becomes an active npc with auto_created='1', and the
        event row's entity_ids list starts with the caller's avatar."""
        alice_id = _opt_in(store)
        event_id = store.record_user_event(
            actor_id=alice_id,
            summary="stinky dan threw a guff grenade at Andrew",
            actor_names=["stinky dan", "Andrew"],
            now=lambda: 100.0,
        )
        # NPC rows exist
        dan = store.find_active_entity_by_name("stinky dan")
        andrew = store.find_active_entity_by_name("Andrew")
        assert dan is not None and dan.kind == "npc"
        assert andrew is not None and andrew.kind == "npc"
        # auto_created marker on each
        assert store.get_attribute(dan.id, "auto_created") == "1"
        assert store.get_attribute(andrew.id, "auto_created") == "1"
        # last_seen_ts = now
        assert store.get_attribute(dan.id, "last_seen_ts") == "100.0"
        assert store.get_attribute(andrew.id, "last_seen_ts") == "100.0"
        # Event row links caller first, then actors in order
        events = store.recent_events(limit=10)
        ev = next(e for e in events if e.id == event_id)
        assert list(ev.entity_ids) == [alice_id, dan.id, andrew.id]
        assert ev.source == "avatar"
        assert ev.summary == "stinky dan threw a guff grenade at Andrew"
```

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_verse_record.py -v` → fails (`AttributeError: 'VerseStore' object has no attribute 'record_user_event'`).

- [ ] **Step 3: implement** — add to `plugins/llm/src/llm/verse/store.py`, alongside the public mutators:

```python
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
    are NOT defended against (out of scope for v1)."""
    ts = now()
    with self.write_transaction() as conn:
        # Validate actor_id is an active avatar.
        actor_row = conn.execute(
            "SELECT kind, status FROM entities WHERE id = ?", (actor_id,)
        ).fetchone()
        if actor_row is None or actor_row[1] != "active":
            raise ValueError(
                f"record_user_event: actor_id={actor_id} not an active entity"
            )

        ids: list[int] = [actor_id]
        for name in actor_names:
            entity = self._find_active_entity_by_name_inline(conn, name)
            if entity is None:
                eid = self._add_entity_inline(conn, "npc", name, "")
                self._set_attribute_inline(conn, eid, "auto_created", "1")
                self._set_attribute_inline(conn, eid, "last_seen_ts", str(ts))
            else:
                eid = entity.id
                if entity.kind != "avatar":
                    self._set_attribute_inline(
                        conn, eid, "last_seen_ts", str(ts)
                    )
            ids.append(eid)

        return self._add_event_inline(
            conn,
            summary=summary,
            entity_ids=ids,
            source="avatar",
            ts=ts,
        )
```

(`Callable` may need to be imported from `collections.abc`. Check the file's existing imports first.)

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_verse_record.py
git commit -m "feat(verse/store): record_user_event happy path"
```

### Task 2.2: avatar precedence — no `auto_created` tag, no `last_seen_ts` bump (Test #2)

- [ ] **Step 1: failing test** — append to `TestRecordUserEvent`:

```python
def test_avatar_actor_not_tagged_or_bumped(self, store: VerseStore) -> None:
    """When an actor name resolves to an existing avatar, the avatar
    must NOT receive auto_created='1' (it wasn't auto-created) and must
    NOT have last_seen_ts bumped (avatars don't age)."""
    alice_id = _opt_in(store, "alice")
    andrew_id = _opt_in(store, "andrew")
    store.record_user_event(
        actor_id=alice_id,
        summary="alice greeted Andrew",
        actor_names=["andrew"],
        now=lambda: 100.0,
    )
    assert store.get_attribute(andrew_id, "auto_created") is None
    assert store.get_attribute(andrew_id, "last_seen_ts") is None
    # Sanity: event row links both
    events = store.recent_events(limit=10)
    assert any(list(e.entity_ids) == [alice_id, andrew_id] for e in events)
```

- [ ] **Step 2: run** → expect green (the `if entity.kind != "avatar"` guard already lives in the implementation from Task 2.1).

If the test fails because the implementation accidentally bumped, fix by tightening the `entity.kind != "avatar"` branch — the test pins the guard exists.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): avatar actor not tagged or heartbeat-bumped"
```

### Task 2.3: existing NPC reused — `last_seen_ts` bumped (Test #3)

- [ ] **Step 1: failing test**:

```python
def test_existing_npc_reused_and_heartbeat_updated(
    self, store: VerseStore
) -> None:
    """When an actor name matches an existing active NPC, no new row is
    created; last_seen_ts is updated to the call's `now`."""
    alice_id = _opt_in(store)
    # Pre-existing NPC (e.g. created by an earlier verse_record call).
    dan_id = store.add_entity("npc", "dan", "")
    store.set_attribute(dan_id, "auto_created", "1")
    store.set_attribute(dan_id, "last_seen_ts", "50.0")

    store.record_user_event(
        actor_id=alice_id,
        summary="dan returned",
        actor_names=["dan"],
        now=lambda: 200.0,
    )
    # No duplicate row
    rows = [e for e in store.list_entities_by_kind("npc") if e.name == "dan"]
    assert len(rows) == 1
    # Heartbeat updated
    assert store.get_attribute(dan_id, "last_seen_ts") == "200.0"
```

- [ ] **Step 2: run** → green (impl from 2.1 already covers this path).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): existing npc reused and heartbeat updated"
```

### Task 2.4: repeated calls keep one row, latest timestamp wins (Test #4)

- [ ] **Step 1: failing test**:

```python
def test_repeated_calls_one_row_latest_timestamp(
    self, store: VerseStore
) -> None:
    alice_id = _opt_in(store)
    for ts in (100.0, 200.0, 300.0):
        store.record_user_event(
            actor_id=alice_id,
            summary="dan reappears",
            actor_names=["dan"],
            now=lambda ts=ts: ts,
        )
    rows = [e for e in store.list_entities_by_kind("npc") if e.name == "dan"]
    assert len(rows) == 1
    assert store.get_attribute(rows[0].id, "last_seen_ts") == "300.0"
```

- [ ] **Step 2: run** → green.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): repeated calls keep one row, latest ts wins"
```

### Task 2.5: race with `time.sleep` injection (Test #5)

**Why:** v2.1 design §6 SIG #4. A bare `ThreadPoolExecutor.map` test passes trivially because the Python `_lock` serialises both threads before they reach SQLite contention. The realistic race window is **between** `_find_active_entity_by_name_inline` and `_add_entity_inline`. Inject a `time.sleep` between those two calls so two threads enter the find phase, both find nothing, then both try to insert — and the test still asserts only one entity row results.

**Reference patterns:** `plugins/llm/tests/verse/test_store.py:611-629` (`test_concurrent_opt_in_distinct_nicks_one_place`) for the `ThreadPoolExecutor` shape, and `plugins/llm/tests/verse/test_crosspoll_store.py:84-108` for the `threading.Barrier` synchronisation pattern. Use the barrier pattern here so both threads start their find at the same moment.

- [ ] **Step 1: failing test**:

```python
def test_concurrent_record_same_actor_one_row(
    self, store: VerseStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two threads race record_user_event for the same unknown actor.
    A `time.sleep(0.01)` is monkey-patched between find and insert so
    the contention window is real (without it, the Python lock
    serialises everything and the test passes trivially — the sleep
    IS the test).

    Both threads start at a Barrier so they enter find at the same
    instant. Exactly one entity row results."""
    import threading

    alice_id = _opt_in(store)

    real_find = store._find_active_entity_by_name_inline
    barrier = threading.Barrier(2)

    def slow_find(conn, name):  # noqa: ANN001
        result = real_find(conn, name)
        # Force the contention window: both threads finish find before
        # either inserts.
        time.sleep(0.01)
        return result

    monkeypatch.setattr(
        store, "_find_active_entity_by_name_inline", slow_find
    )

    results: list[int] = []

    def call(seq: int) -> None:
        barrier.wait()
        eid = store.record_user_event(
            actor_id=alice_id,
            summary=f"event {seq}",
            actor_names=["zorp"],
            now=lambda: 100.0 + seq,
        )
        results.append(eid)

    t1 = threading.Thread(target=call, args=(1,))
    t2 = threading.Thread(target=call, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(results) == 2
    zorps = [
        e for e in store.list_entities_by_kind("npc") if e.name == "zorp"
    ]
    assert len(zorps) == 1, (
        f"expected exactly one 'zorp' entity, got {len(zorps)}"
    )
```

- [ ] **Step 2: run** → expect either pass or fail depending on whether the impl from 2.1 holds the write_lock across the entire find-and-insert. If it fails, the impl must be tightened: confirm that `record_user_event`'s `with self.write_transaction()` block wraps both the find and the insert (it does in 2.1's snippet — both are inside the same `conn`).

If it still fails, debug: is `_find_active_entity_by_name_inline` being patched at the class level vs the instance level? The monkeypatch above uses `monkeypatch.setattr(store, …)` (instance), which works if the method is bound. Adjust if it doesn't take effect.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): race resolved inside one write_transaction"
```

### Task 2.6: retired entity not rehydrated (Test #6)

- [ ] **Step 1: failing test**:

```python
def test_retired_entity_not_rehydrated(self, store: VerseStore) -> None:
    """find_active_entity_by_name filters status='active', so a name
    matching a retired NPC must create a NEW active NPC, not rehydrate
    the old row."""
    alice_id = _opt_in(store)
    old_id = store.add_entity("npc", "ghost", "")
    store.set_status(old_id, "retired")

    store.record_user_event(
        actor_id=alice_id,
        summary="ghost reappeared",
        actor_names=["ghost"],
        now=lambda: 100.0,
    )
    ghosts = [
        e for e in store.list_entities_by_kind("npc")
        if e.name == "ghost"
    ]
    # One retired + one fresh active = two rows total
    statuses = sorted(e.status for e in ghosts)
    assert statuses == ["active", "retired"]
    new_ghost = next(e for e in ghosts if e.status == "active")
    assert new_ghost.id != old_id
    assert store.get_attribute(new_ghost.id, "auto_created") == "1"
```

(`list_entities_by_kind` may default to `status="active"` only — check the existing signature. If so, pass `status=None` to retrieve both rows, or use raw SQL to query.)

- [ ] **Step 2: run** → green (the `_find_active_entity_by_name_inline` precedence-aware lookup from Phase 1 already filters `status='active'`).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): retired entity not rehydrated by new mention"
```

### Task 2.8: case-insensitive (Test #11)

- [ ] **Step 1: failing test**:

```python
def test_actors_resolved_case_insensitively(self, store: VerseStore) -> None:
    """`actors=["ANDREW"]` resolves to existing avatar 'andrew'."""
    alice_id = _opt_in(store, "alice")
    andrew_id = _opt_in(store, "andrew")
    store.record_user_event(
        actor_id=alice_id,
        summary="alice waved at ANDREW",
        actor_names=["ANDREW"],
        now=lambda: 100.0,
    )
    events = store.recent_events(limit=5)
    latest = events[0]
    assert list(latest.entity_ids) == [alice_id, andrew_id]
    # Avatar still has no auto_created marker
    assert store.get_attribute(andrew_id, "auto_created") is None
```

- [ ] **Step 2: run** → green (the `LOWER(name) = LOWER(?)` SQL from Phase 1's helper).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): actor name matching is case-insensitive"
```

### Task 2.9: opt-out → re-opt-in three-row state (Test #10)

**Why:** v2.1 design §2 + §6 SIG #6. After opt-out, an auto-NPC may be created with the same name; re-opt-in creates a fresh avatar; subsequent mentions must resolve to the NEW avatar (not the orphan NPC), and the orphan ages out.

- [ ] **Step 1: failing test**:

```python
def test_opt_out_then_record_then_reopt_in_three_row_state(
    self, store: VerseStore
) -> None:
    alice_id = _opt_in(store, "alice")
    avatar_v1 = _opt_in(store, "andrew")
    store.unlink_avatar(avatar_v1)
    # ↑ retires avatar_v1 + drops avatar_link. State: row 1 retired.

    # verse_record mentions Andrew; auto-NPC is created. State: row 1
    # retired avatar, row 2 active npc.
    store.record_user_event(
        actor_id=alice_id,
        summary="Andrew was seen",
        actor_names=["Andrew"],
        now=lambda: 100.0,
    )
    npc = store.find_active_entity_by_name("Andrew")
    assert npc is not None and npc.kind == "npc"

    # andrew opts back in. State: row 1 retired, row 2 active npc, row
    # 3 active avatar.
    avatar_v2 = _opt_in(store, "andrew")
    assert avatar_v2 != avatar_v1
    # Subsequent record resolves to the NEW avatar (precedence).
    store.record_user_event(
        actor_id=alice_id,
        summary="Andrew is back",
        actor_names=["Andrew"],
        now=lambda: 200.0,
    )
    events = store.recent_events(limit=2)
    latest = events[0]
    assert avatar_v2 in latest.entity_ids
    assert npc.id not in latest.entity_ids
    # Avatar is not bumped (avatars don't bump)
    assert store.get_attribute(avatar_v2, "last_seen_ts") is None
    # Orphan NPC is also NOT bumped on this call (it didn't resolve);
    # its last_seen_ts stays at the original 100.0 — aging will retire.
    assert store.get_attribute(npc.id, "last_seen_ts") == "100.0"
```

- [ ] **Step 2: run** → green (precedence + avatar-no-bump from earlier tasks already cover the behaviour; this test just pins it).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): opt-out → record → re-opt-in three-row state"
```

### Task 2.10: retired actor_id raises (Test #13)

- [ ] **Step 1: failing test**:

```python
def test_retired_actor_id_raises(self, store: VerseStore) -> None:
    """An actor_id pointing at a retired avatar is a programming error
    (the caller is supposed to look up the live avatar via avatar_link).
    Mirror verse_act's 'avatar retired' guard."""
    alice_id = _opt_in(store, "alice")
    store.unlink_avatar(alice_id)
    with pytest.raises(ValueError, match="not an active entity"):
        store.record_user_event(
            actor_id=alice_id,
            summary="alice did something",
            actor_names=["bob"],
            now=lambda: 100.0,
        )
    # No event row written
    assert store.recent_events(limit=10) == []
    # No 'bob' NPC created — full rollback
    assert store.find_active_entity_by_name("bob") is None
```

- [ ] **Step 2: run** → green (the validation + raise from Task 2.1's impl).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): retired actor_id raises and rolls back"
```

### Phase 2 verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_verse_record.py -v` → all green.
- [ ] `uv run pytest plugins/llm/tests/verse/ -v` → no regressions.

---

## Phase 3 — Aging helper

**File:** `plugins/llm/src/llm/verse/aging.py` (NEW). Tests in `plugins/llm/tests/verse/test_verse_aging.py` (NEW).

### Task 3.1: skeleton — `AgingOutcome` + zero-day disables

**Files:**
- Create: `plugins/llm/src/llm/verse/aging.py`.
- Create: `plugins/llm/tests/verse/test_verse_aging.py`.

- [ ] **Step 1: failing test** — create `plugins/llm/tests/verse/test_verse_aging.py`:

```python
"""Tests for verse/aging.age_auto_created_entities."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#aging")


SECONDS_PER_DAY = 86400.0


class TestAgeAutoCreatedEntities:
    def test_retire_after_days_zero_disables(self, store: VerseStore) -> None:
        from llm.verse.aging import AgingOutcome, age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(
            store, retire_after_days=0, now=lambda: 1e9
        )
        assert outcome == AgingOutcome(scanned=0, retired=0)
        # Entity still active
        from llm.verse.store import Entity
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT status FROM entities WHERE id=?", (eid,)
            ).fetchone()
        assert row[0] == "active"
```

- [ ] **Step 2: run** → fails (`ImportError: No module named 'llm.verse.aging'`).

- [ ] **Step 3: implement** — create `plugins/llm/src/llm/verse/aging.py`:

```python
"""Soft-retire auto_created NPCs that have been silent past the cutoff.

A pure helper module. Runs in the compaction pass per channel. No
schedule of its own. ``retire_after_days <= 0`` disables — early
return."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import NamedTuple, Any

_LOG = logging.getLogger("llm.verse.aging")
_SECONDS_PER_DAY = 86400.0


class AgingOutcome(NamedTuple):
    scanned: int
    retired: int


def age_auto_created_entities(
    store: Any,
    *,
    retire_after_days: int,
    now: Callable[[], float],
) -> AgingOutcome:
    """Soft-retire auto_created='1' entities whose last_seen_ts is
    older than now - retire_after_days*86400. Skips kind='avatar'
    defensively. Returns counts (scanned, retired). retire_after_days<=0
    disables — returns (0, 0)."""
    if retire_after_days <= 0:
        return AgingOutcome(scanned=0, retired=0)

    cutoff = now() - retire_after_days * _SECONDS_PER_DAY
    candidates = store.list_entities_with_attribute(
        key="auto_created", value="1", status="active"
    )
    scanned = 0
    retired = 0
    for entity in candidates:
        if entity.kind == "avatar":
            continue  # defensive — auto_created on an avatar is a bug, not a target
        scanned += 1
        last_seen_str = store.get_attribute(entity.id, "last_seen_ts")
        if last_seen_str is None:
            continue  # no heartbeat → no decision; leave it
        try:
            last_seen = float(last_seen_str)
        except ValueError:
            _LOG.warning(
                "verse aging: malformed last_seen_ts on entity %s: %r",
                entity.id, last_seen_str,
            )
            continue
        if last_seen < cutoff:
            store.set_status(entity.id, "retired")
            retired += 1
    return AgingOutcome(scanned=scanned, retired=retired)
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/aging.py plugins/llm/tests/verse/test_verse_aging.py
git commit -m "feat(verse/aging): age_auto_created_entities skeleton + zero-disable"
```

### Task 3.2: retire past-cutoff (Aging Test #1)

- [ ] **Step 1: failing test** — append to `TestAgeAutoCreatedEntities`:

```python
def test_retires_past_cutoff(self, store: VerseStore) -> None:
    from llm.verse.aging import age_auto_created_entities

    eid = store.add_entity("npc", "ghost", "")
    store.set_attribute(eid, "auto_created", "1")
    store.set_attribute(eid, "last_seen_ts", "100.0")  # very stale
    now = 100.0 + 30 * SECONDS_PER_DAY  # 30 days later

    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: now
    )
    assert outcome.scanned == 1
    assert outcome.retired == 1
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT status FROM entities WHERE id=?", (eid,)
        ).fetchone()
    assert row[0] == "retired"
```

- [ ] **Step 2: run** → green.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/aging): retire past cutoff"
```

### Task 3.3: keep recent (Aging Test #2)

- [ ] **Step 1: failing test**:

```python
def test_keeps_recent(self, store: VerseStore) -> None:
    from llm.verse.aging import age_auto_created_entities

    eid = store.add_entity("npc", "moss", "")
    store.set_attribute(eid, "auto_created", "1")
    last_seen = 1000.0
    store.set_attribute(eid, "last_seen_ts", str(last_seen))
    now = last_seen + 5 * SECONDS_PER_DAY  # 5 days < 14-day cutoff

    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: now
    )
    assert outcome == (1, 0)
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT status FROM entities WHERE id=?", (eid,)
        ).fetchone()
    assert row[0] == "active"
```

- [ ] **Step 2: run** → green.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/aging): keep recent entries inside the window"
```

### Task 3.4: skip non-`auto_created` (Aging Test #3)

- [ ] **Step 1: failing test**:

```python
def test_skips_manually_created(self, store: VerseStore) -> None:
    """An NPC without auto_created='1' must never be touched, even if
    last_seen_ts is past cutoff."""
    from llm.verse.aging import age_auto_created_entities

    eid = store.add_entity("npc", "manual", "")
    # Note: NO auto_created attribute. last_seen_ts could exist (e.g. set by
    # some external tool) but aging shouldn't see it.
    store.set_attribute(eid, "last_seen_ts", "0.0")
    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: 1e9
    )
    assert outcome == (0, 0)
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT status FROM entities WHERE id=?", (eid,)
        ).fetchone()
    assert row[0] == "active"
```

- [ ] **Step 2: run** → green (the SQL JOIN on `attributes WHERE key='auto_created' AND value='1'` excludes this row).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/aging): skip non-auto_created entities"
```

### Task 3.5: skip avatars defensively (Aging Test #4)

- [ ] **Step 1: failing test**:

```python
def test_skips_avatar_kind_defensively(self, store: VerseStore) -> None:
    """Even if a bug somewhere stamps auto_created='1' on an avatar,
    aging must not retire it. The kind!='avatar' guard is defensive."""
    from llm.verse.aging import age_auto_created_entities

    avatar_id = store.add_entity("avatar", "alice", "")
    store.set_attribute(avatar_id, "auto_created", "1")
    store.set_attribute(avatar_id, "last_seen_ts", "0.0")
    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: 1e9
    )
    assert outcome == (0, 0)
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT status FROM entities WHERE id=?", (avatar_id,)
        ).fetchone()
    assert row[0] == "active"
```

- [ ] **Step 2: run** → green (the `if entity.kind == "avatar": continue` skip in 3.1's impl).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/aging): defensively skip kind='avatar'"
```

### Phase 3 verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_verse_aging.py -v` → 5 tests pass.

---

## Phase 4 — Heartbeat wiring

Three call-sites, narrowed per v2.1 design §4.2:

1. `record_user_event` — already bumps via Phase 2.
2. `_replace_events_with_source` (compaction-digest insert) — Tasks 4.1 + 4.2.
3. `apply_or_queue` (loom proposal application) — Tasks 4.3 + 4.4.

### Task 4.1: digest-insert heartbeat (Aging Test #6)

**Why:** v2.1 design §4.2 #3. The bump runs on the same `conn` that wrote the digest, atomically. Each entity in `union_ids[:_MAX_DIGEST_ENTITY_IDS]` gets `last_seen_ts` set to the digest's `now()`.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (`_replace_events_with_source`).
- Modify: `plugins/llm/tests/verse/test_verse_aging.py`.

- [ ] **Step 1: failing test** — append to `TestAgeAutoCreatedEntities`:

```python
def test_digest_insert_bumps_last_seen(self, store: VerseStore) -> None:
    """When _replace_events_with_source inserts a digest, every entity
    in entity_ids has last_seen_ts bumped to ts. The bump is on the
    same conn as the INSERT — atomic with the digest write.

    Setup: an auto-created NPC with a stale last_seen_ts. Run the
    digest insert mentioning that NPC. Aging next finds last_seen_ts
    fresh and keeps the entity."""
    from llm.verse.aging import age_auto_created_entities

    eid = store.add_entity("npc", "ghost", "")
    store.set_attribute(eid, "auto_created", "1")
    store.set_attribute(eid, "last_seen_ts", "0.0")  # stale

    digest_ts = 1000.0
    store.replace_events_with_lore_digest(
        delete_ids=[],
        summary="ghost remained",
        entity_ids=(eid,),
        ts=digest_ts,
    )
    # Heartbeat fired
    assert store.get_attribute(eid, "last_seen_ts") == str(digest_ts)
    # Aging now sees a fresh entity → keeps it
    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: digest_ts + SECONDS_PER_DAY
    )
    assert outcome == (1, 0)
    with store.read_connection() as conn:
        row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
    assert row[0] == "active"
```

- [ ] **Step 2: run** → fails (`_replace_events_with_source` does not yet bump).

- [ ] **Step 3: implement** — modify `_replace_events_with_source` (`store.py:391-434`). Add a single loop after the INSERT, before the function returns:

```python
def _replace_events_with_source(
    self,
    *,
    delete_ids: Sequence[int],
    summary: str,
    entity_ids: Sequence[int],
    ts: float,
    source: str,
) -> int:
    """Atomic delete-then-insert. Returns the new event's id.

    Also bumps last_seen_ts to ts for every entity in entity_ids. The
    bump runs on the same conn as the INSERT — atomic with the digest
    write — so verse aging sees a consistent view."""
    with self.write_transaction() as conn:
        if delete_ids:
            placeholders = ",".join("?" for _ in delete_ids)
            conn.execute(
                f"DELETE FROM events WHERE id IN ({placeholders})",
                tuple(delete_ids),
            )
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, json.dumps(list(entity_ids)), source),
        )
        assert cur.lastrowid is not None
        new_id = int(cur.lastrowid)
        # Heartbeat: every entity in this digest is "alive as of ts".
        for eid in entity_ids:
            self._set_attribute_inline(conn, int(eid), "last_seen_ts", str(ts))
        return new_id
```

(The `_set_attribute_inline` helper from Phase 0b is the right call — same `conn`, no lock reentry.)

- [ ] **Step 4: run** → green. Run `uv run pytest plugins/llm/tests/verse/test_compaction.py -v` to confirm no regression in the existing digest tests.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_verse_aging.py
git commit -m "feat(verse/store): digest insert bumps last_seen_ts on entity_ids"
```

### Task 4.2: truncated-out entity correctly retires (Aging Test #7)

**Why:** v2.1 design §6 SIG #11 + §7 #7. `_MAX_DIGEST_ENTITY_IDS` truncates the digest's `union_ids` list. Entities truncated out are *not* heartbeated by the digest insert — and they should age out next cycle. Test pins this behaviour AND imports the constant rather than hardcoding 32 (per design's Codex v2 SIG #6 fix).

- [ ] **Step 1: failing test**:

```python
def test_digest_truncated_entities_correctly_age(
    self, store: VerseStore
) -> None:
    """Setup: _MAX_DIGEST_ENTITY_IDS + 8 auto-created NPCs all with
    last_seen_ts=0. Insert a digest with all of them in entity_ids; the
    digest layer truncates to _MAX_DIGEST_ENTITY_IDS. Aging then runs
    and the 8 truncated-out NPCs are correctly retired (their
    last_seen_ts was never bumped because they didn't make the cut)."""
    from llm.verse.aging import age_auto_created_entities
    from llm.verse.compaction import _MAX_DIGEST_ENTITY_IDS

    n_total = _MAX_DIGEST_ENTITY_IDS + 8
    ids: list[int] = []
    for i in range(n_total):
        eid = store.add_entity("npc", f"ghost{i}", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        ids.append(eid)

    # Stand-in for compaction: truncate to cap and write the digest.
    digest_ts = 1000.0
    union_ids_truncated = ids[:_MAX_DIGEST_ENTITY_IDS]
    store.replace_events_with_lore_digest(
        delete_ids=[],
        summary="many ghosts",
        entity_ids=union_ids_truncated,
        ts=digest_ts,
    )

    # Aging fires past cutoff for everyone whose last_seen_ts was not bumped.
    outcome = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: digest_ts + 30 * SECONDS_PER_DAY
    )
    # Exactly the truncated-out 8 were retired.
    assert outcome.retired == 8
    # In-digest survivors stay active.
    survivors_status = []
    truncated_status = []
    with store.read_connection() as conn:
        for i, eid in enumerate(ids):
            row = conn.execute(
                "SELECT status FROM entities WHERE id=?", (eid,)
            ).fetchone()
            if i < _MAX_DIGEST_ENTITY_IDS:
                survivors_status.append(row[0])
            else:
                truncated_status.append(row[0])
    assert set(survivors_status) == {"active"}
    assert set(truncated_status) == {"retired"}
```

- [ ] **Step 2: run** → green (impl from 4.1 covers the in-digest bump; truncated-out get aged because their last_seen_ts is still 0.0).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/aging): digest-truncated entities correctly retire"
```

### Task 4.3: loom apply heartbeat — positive (Aging Test #8)

**Why:** v2.1 design §4.2 #2 + §6 SIG #7. When `apply_or_queue` lands a proposal as `applied` or `crosspoll_emitted`, every referenced entity_id is bumped. Bumps run *after* `apply_and_record_proposal` / `enqueue_seed`+`add_event` finish. We add a public `bump_last_seen_ts(entity_ids, *, ts)` method on `VerseStore` for loom to call (it doesn't share `apply_or_queue`'s concern about non-reentrant locks because it runs *outside* any open tx).

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (new public method `bump_last_seen_ts`).
- Modify: `plugins/llm/src/llm/verse/loom.py` (`apply_or_queue` at `:257`).
- Modify: `plugins/llm/tests/verse/test_verse_aging.py` and/or a new section in `test_loom.py`.

- [ ] **Step 1: failing test** — append to `TestAgeAutoCreatedEntities`:

```python
def test_loom_applied_proposal_bumps_last_seen(
    self, store: VerseStore
) -> None:
    """An apply_or_queue call landing as 'applied' bumps last_seen_ts
    on every entity_id referenced by the proposal payload."""
    from llm.verse.aging import age_auto_created_entities
    from llm.verse.loom import apply_or_queue, ParsedProposal

    eid = store.add_entity("npc", "ghost", "")
    store.set_attribute(eid, "auto_created", "1")
    store.set_attribute(eid, "last_seen_ts", "0.0")  # stale

    prop = ParsedProposal(
        op="add_event",
        payload={
            "summary": "ghost lurked",
            "entity_ids": [eid],
        },
        confidence=0.95,  # above default threshold → applies
        provenance={"source": "test"},
    )
    outcome = apply_or_queue(
        store, prop,
        cycle_id="cyc-1",
        threshold=0.7,
    )
    assert outcome.outcome == "applied"
    # Bump fired.
    last_seen = float(store.get_attribute(eid, "last_seen_ts") or "0")
    assert last_seen > 0.0
    # Aging would now keep this entity.
    keep = age_auto_created_entities(
        store, retire_after_days=14, now=lambda: last_seen + SECONDS_PER_DAY
    )
    assert keep.retired == 0
```

- [ ] **Step 2: run** → fails (no bump happens today).

- [ ] **Step 3: implement step (a)** — add a public bump method to `verse/store.py`:

```python
def bump_last_seen_ts(
    self,
    entity_ids: Sequence[int],
    *,
    ts: float,
) -> None:
    """Bump last_seen_ts on every id. Single write_transaction.
    Used by loom apply_or_queue and (indirectly) by record_user_event
    via the inline helper. No-op for empty input."""
    if not entity_ids:
        return
    with self.write_transaction() as conn:
        for eid in entity_ids:
            self._set_attribute_inline(conn, int(eid), "last_seen_ts", str(ts))
```

- [ ] **Step 3: implement step (b)** — modify `apply_or_queue` in `verse/loom.py:257`. Read the function body first to understand the exact return paths. Add a heartbeat call **only** in the two success paths (`applied` and `crosspoll_emitted`), passing `prop.payload.get("entity_ids", [])`:

```python
# Excerpt — apply_or_queue, applied branch (around the existing
# `return ApplyOutcome(outcome="applied")`):
import time as _time

if auto:
    store.apply_and_record_proposal(
        cycle_id=cycle_id,
        op=prop.op,
        payload=prop.payload,
        confidence=prop.confidence,
        provenance=prop.provenance,
        reviewer="loom",
    )
    _entity_ids = prop.payload.get("entity_ids") or []
    store.bump_last_seen_ts(list(_entity_ids), ts=_time.time())
    return ApplyOutcome(outcome="applied")
```

(And mirror the `_entity_ids = prop.payload.get("entity_ids") or []` + `store.bump_last_seen_ts(...)` two lines into the `crosspoll_emitted` branch — read the function to find the right line. Time source: `time.time()`. If the loom already has a `now`-injection convention, use that for consistency.)

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_verse_aging.py
git commit -m "feat(verse/loom): apply_or_queue bumps last_seen_ts on applied/crosspoll_emitted"
```

### Task 4.4: loom apply heartbeat — negative (Aging Test #9)

**Why:** v2.1 design §6 SIG #7. Low-confidence proposals (`queued`) and `rejected_invalid_refs` MUST NOT bump — keeping junk-entity proposals from extending NPC lifetimes via low-confidence model output is the whole point.

- [ ] **Step 1: failing test**:

```python
def test_loom_queued_proposal_does_not_bump(
    self, store: VerseStore
) -> None:
    """Below-threshold proposals queue rather than apply; no bump."""
    from llm.verse.loom import apply_or_queue, ParsedProposal

    eid = store.add_entity("npc", "ghost", "")
    store.set_attribute(eid, "auto_created", "1")
    store.set_attribute(eid, "last_seen_ts", "0.0")
    prop = ParsedProposal(
        op="add_event",
        payload={"summary": "maybe ghost", "entity_ids": [eid]},
        confidence=0.10,  # well below 0.7 threshold
        provenance={"source": "test"},
    )
    outcome = apply_or_queue(
        store, prop, cycle_id="cyc-q", threshold=0.7
    )
    assert outcome.outcome == "queued"
    # No bump
    assert store.get_attribute(eid, "last_seen_ts") == "0.0"

def test_loom_rejected_invalid_refs_does_not_bump(
    self, store: VerseStore
) -> None:
    """Proposals referencing nonexistent entity ids auto-reject; no bump
    (and obviously can't bump nonexistent ids — but the test pins the
    contract)."""
    from llm.verse.loom import apply_or_queue, ParsedProposal

    real_eid = store.add_entity("npc", "ghost", "")
    store.set_attribute(real_eid, "auto_created", "1")
    store.set_attribute(real_eid, "last_seen_ts", "0.0")
    nonexistent_id = real_eid + 999_999
    prop = ParsedProposal(
        op="add_event",
        payload={
            "summary": "phantom event",
            "entity_ids": [real_eid, nonexistent_id],
        },
        confidence=0.95,
        provenance={"source": "test"},
    )
    outcome = apply_or_queue(
        store, prop, cycle_id="cyc-r", threshold=0.7
    )
    # apply_or_queue auto-rejects on invalid refs (existing behaviour).
    # Outcome enum may vary — match whatever the existing impl returns
    # for invalid refs (likely 'rejected_invalid_refs' or 'queued' with
    # an auto-rejecter reviewer marker). The bump assertion is what
    # this test pins.
    assert outcome.outcome != "applied"
    assert store.get_attribute(real_eid, "last_seen_ts") == "0.0"
```

(If the second test's outcome enum doesn't match the actual return — read `_proposal_entity_refs_resolve` and the auto-reject branch in `apply_or_queue` to find what the function returns — adjust the assertion to whatever the real outcome is. The critical assertion is `last_seen_ts == "0.0"`, not the outcome name.)

- [ ] **Step 2: run** → green (Task 4.3 only added the bump in the `applied` and `crosspoll_emitted` branches, so other paths don't bump).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse/loom): queued/rejected proposals do not bump last_seen_ts"
```

### Phase 4 verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_verse_aging.py -v` → 9 tests pass.
- [ ] `uv run pytest plugins/llm/tests/verse/test_compaction.py -v` → no regressions in digest tests.
- [ ] `uv run pytest plugins/llm/tests/verse/test_loom.py -v` → no regressions.

---

## Phase 5a — Wire aging into the compaction pass

**Why:** v2.1 design §11 Step 5a. The plugin already runs `_run_compaction_pass` on a daily timer (see `plugin.py:4878`). After `compact_verse` returns, call `age_auto_created_entities` for the same channel. Step 5a deliberately does NOT change the outcome string yet — that's Step 5b. This split lets us land aging-without-UI changes first and keep blast radius small.

### Task 5a.0: register both registry keys

**Why up front:** The plugin code in 5a.1 needs `verseAutoEntityRetireDays` to read; Phase 6 needs `verseAutoEntityMaxNamesPerCall`. Easier to do both in one config commit.

**Files:**
- Modify: `plugins/llm/src/llm/config.py`.
- Modify: `plugins/llm/tests/test_config.py`.

- [ ] **Step 1: failing test** — append to `plugins/llm/tests/test_config.py`. Mirror `test_verse_event_retention_days_default` at `test_config.py:242-246` exactly (the existing convention: assert via `conf.supybot.plugins.LLM.<key>()` after the plugin has been loaded by `plugin_test_env`):

```python
class TestVerseAutoEntityKeys:
    """Two new per-channel registry keys for verse_record + aging."""

    def test_verse_auto_entity_retire_days_default_14(
        self, plugin_test_env
    ) -> None:
        """verseAutoEntityRetireDays defaults to 14 (per-channel)."""
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseAutoEntityRetireDays() == 14

    def test_verse_auto_entity_max_names_per_call_default_8(
        self, plugin_test_env
    ) -> None:
        """verseAutoEntityMaxNamesPerCall defaults to 8 (per-channel)."""
        import supybot.conf as conf

        assert conf.supybot.plugins.LLM.verseAutoEntityMaxNamesPerCall() == 8

    def test_verse_auto_entity_retire_days_zero_allowed(
        self, plugin_test_env
    ) -> None:
        """0 must be a valid value — design §5: '0 disables sweep'.
        NonNegativeInteger accepts 0; PositiveInteger would not."""
        import supybot.conf as conf

        conf.supybot.plugins.LLM.verseAutoEntityRetireDays.setValue(0)
        assert conf.supybot.plugins.LLM.verseAutoEntityRetireDays() == 0
```

(`plugin_test_env` is the existing fixture in `test_config.py` that loads the LLM plugin module. If the file uses a different fixture name, mirror the existing tests — `grep -n 'def test_verse_event_retention_days_default' plugins/llm/tests/test_config.py` to find the exact shape.)

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement** in `plugins/llm/src/llm/config.py` — mirror the registration form used by the existing `verseEvent*` and `verseLoom*` keys (channel-scoped `registerChannelValue`):

```python
conf.registerChannelValue(
    LLM,
    "verseAutoEntityRetireDays",
    registry.NonNegativeInteger(
        14,
        """Days of no reference before auto-created NPCs retire. 0 disables sweep.""",
    ),
)
conf.registerChannelValue(
    LLM,
    "verseAutoEntityMaxNamesPerCall",
    registry.PositiveInteger(
        8,
        """Hard cap on verse_record `actors` array length. The advertised
        tool spec's maxItems is set from this; dispatch enforces. Increase
        past 16 only if your verse routinely cites large casts.""",
    ),
)
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat(config): verseAutoEntityRetireDays + verseAutoEntityMaxNamesPerCall"
```

### Task 5a.1: `_run_compaction_pass` calls aging per channel (Plugin Test #10)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_run_compaction_pass` body around `:4878-4950`).
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: failing test** — append to `test_plugin.py`. Read an existing `_run_compaction_pass` test (one likely exists for the daily timer) and mirror the fixture / Irc-mock style. Stub `compact_verse` and `age_auto_created_entities`; assert one call per enabled channel:

```python
class TestRunCompactionPassCallsAging:
    def test_aging_called_once_per_enabled_channel(
        self, plugin, monkeypatch
    ):
        """_run_compaction_pass calls age_auto_created_entities once per
        channel returned by _verse_enabled_channels."""
        from llm.verse import aging as aging_mod
        from llm.verse import compaction as compaction_mod

        monkeypatch.setattr(
            plugin._plugin, "_verse_enabled_channels",
            lambda: ["#a", "#b"],
        )
        # Stub compact_verse to a no-op string return.
        monkeypatch.setattr(
            compaction_mod, "compact_verse",
            lambda *a, **kw: "skipped_disabled",
        )
        called = []
        monkeypatch.setattr(
            aging_mod, "age_auto_created_entities",
            lambda store, *, retire_after_days, now: (
                called.append((store, retire_after_days)),
                aging_mod.AgingOutcome(0, 0),
            )[1],
        )

        plugin._plugin._run_compaction_pass()

        assert len(called) == 2
        # One per channel
        stores = {id(c[0]) for c in called}
        assert len(stores) == 2
```

(Adapt the fixture name `plugin` / `plugin._plugin` to whatever the existing tests in `test_plugin.py` use. Read the file first — `grep -n 'def test_' plugins/llm/tests/test_plugin.py | head -40` to find the convention.)

- [ ] **Step 2: run** → fails (`_run_compaction_pass` doesn't yet call aging).

- [ ] **Step 3: implement** — modify `_run_compaction_pass` in `plugin.py:4878-…`. After the existing `compact_verse(...)` call (around `:4940`), add aging:

```python
# After the existing compact_verse(...) call:
from llm.verse.aging import age_auto_created_entities

retire_days = self.registryValue(
    "verseAutoEntityRetireDays", channel=channel,
)
try:
    aging_outcome = age_auto_created_entities(
        self._get_or_create_verse_store(channel),
        retire_after_days=retire_days,
        now=time.time,
    )
    self.log.info(
        "verse aging: channel=%s scanned=%s retired=%s",
        channel, aging_outcome.scanned, aging_outcome.retired,
    )
except Exception:
    self.log.exception(
        "verse aging failed for %s; continuing with next channel",
        channel,
    )
```

(The exact placement depends on the existing `try/except` structure of `_run_compaction_pass`. Read `:4878-4950` first; place the aging call inside the same per-channel `try/except` if there is one, or in a sibling `try/except` if compaction failure should not block aging. The Plugin Test #12 below pins isolation.)

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(plugin): _run_compaction_pass calls age_auto_created_entities per channel"
```

### Task 5a.2: per-channel registry scope (Plugin Test #11)

- [ ] **Step 1: failing test**:

```python
def test_aging_reads_retire_days_per_channel(self, plugin, monkeypatch):
    """The aging call reads verseAutoEntityRetireDays at the channel
    scope, not global. Verify by mocking registryValue and asserting
    the channel kwarg."""
    from llm.verse import aging as aging_mod
    from llm.verse import compaction as compaction_mod

    monkeypatch.setattr(
        plugin._plugin, "_verse_enabled_channels", lambda: ["#a"]
    )
    monkeypatch.setattr(
        compaction_mod, "compact_verse",
        lambda *a, **kw: "skipped_disabled",
    )
    captured = []
    real_registry = plugin._plugin.registryValue

    def spy(key, *, channel=None):
        captured.append((key, channel))
        return real_registry(key, channel=channel)

    monkeypatch.setattr(plugin._plugin, "registryValue", spy)
    monkeypatch.setattr(
        aging_mod, "age_auto_created_entities",
        lambda *a, **kw: aging_mod.AgingOutcome(0, 0),
    )

    plugin._plugin._run_compaction_pass()

    # The retire-days lookup must have been per-channel
    assert ("verseAutoEntityRetireDays", "#a") in captured
```

- [ ] **Step 2: run** → green if 5a.1's impl already passes `channel=channel` to `registryValue`.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/test_plugin.py
git commit -m "test(plugin): aging reads verseAutoEntityRetireDays per-channel"
```

### Task 5a.3: failure isolation (Plugin Test #12)

- [ ] **Step 1: failing test**:

```python
def test_aging_failure_in_one_channel_does_not_abort_others(
    self, plugin, monkeypatch
):
    """If aging raises for #a, #b still gets aged. Mirrors the
    existing compact_verse failure-isolation pattern."""
    from llm.verse import aging as aging_mod
    from llm.verse import compaction as compaction_mod

    monkeypatch.setattr(
        plugin._plugin, "_verse_enabled_channels",
        lambda: ["#a", "#b"],
    )
    monkeypatch.setattr(
        compaction_mod, "compact_verse",
        lambda *a, **kw: "skipped_disabled",
    )
    seen: list[str] = []

    def aging(store, *, retire_after_days, now):
        # We can't recover the channel from the store easily; use a
        # counter and check both channels were attempted.
        seen.append(id(store))
        if len(seen) == 1:
            raise RuntimeError("simulated aging failure")
        return aging_mod.AgingOutcome(0, 0)

    monkeypatch.setattr(
        aging_mod, "age_auto_created_entities", aging
    )
    plugin._plugin._run_compaction_pass()
    # Both channels' aging was attempted
    assert len(seen) == 2
```

- [ ] **Step 2: run** → if 5a.1 already wrapped aging in a `try/except` and continues, green. Otherwise, tighten the impl to add the wrap.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/test_plugin.py
git commit -m "test(plugin): aging failure in one channel does not abort others"
```

### Phase 5a verification

- [ ] `uv run pytest plugins/llm/tests/test_plugin.py -v -k 'aging or compaction'` → green.
- [ ] No prod-outcome-string changes yet — that's Phase 5b.

---

## Phase 5b — `CompactionOutcome` NamedTuple + new outcome string

**Why:** v2.1 design §4.3. `compact_verse` returning a string makes §8's friendlier outcome message unproducible without a contract change. v2.1 corrects v2's wrong test count: **eight** sites in `test_compaction.py` need migration, not four.

### Task 5b.1: introduce `CompactionOutcome` NamedTuple

**Files:**
- Modify: `plugins/llm/src/llm/verse/compaction.py:42-…` (`compact_verse` signature + every `return` site).

- [ ] **Step 1: failing test** — append a new test class to `test_compaction.py`:

```python
class TestCompactionOutcomeShape:
    def test_returns_namedtuple(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import CompactionOutcome, compact_verse
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#shape")
        outcome = compact_verse(
            store,
            retention_days=0,  # trips skipped_disabled
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 0.0,
        )
        assert isinstance(outcome, CompactionOutcome)
        assert outcome.state == "skipped_disabled"
        assert outcome.total_events == 0
        assert outcome.kept_in_digest == 0
```

- [ ] **Step 2: run** → fails (no NamedTuple).

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/compaction.py`. Add the NamedTuple at the top (near `_MAX_DIGEST_ENTITY_IDS` at line 36), and migrate every `return` in `compact_verse`:

```python
from typing import NamedTuple


class CompactionOutcome(NamedTuple):
    state: str           # 'compacted' | 'skipped_disabled' | 'skipped_below_floor' | 'skipped_no_events'
    total_events: int    # COUNT(*) FROM events at pass entry
    kept_in_digest: int  # len(union_ids[:_MAX_DIGEST_ENTITY_IDS]) when state=='compacted', else 0
```

Then replace each existing `return "skipped_disabled"` (and the rest) site-by-site:

```python
# Keep the existing kwargs (store, *, retention_days, min_keep_events,
# model, client, log_usage, now). Only the return type changes.
def compact_verse(
    store: Any,
    *,
    retention_days: int,
    min_keep_events: int,
    model: str,
    client: Any,
    log_usage: Callable[..., None],
    now: Callable[[], float],
) -> CompactionOutcome:
    if retention_days <= 0:
        return CompactionOutcome("skipped_disabled", 0, 0)

    with store.read_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if total < min_keep_events:
        return CompactionOutcome("skipped_below_floor", total, 0)

    cutoff_ts = now() - retention_days * SECONDS_PER_DAY
    olds = store.events_older_than(cutoff_ts=cutoff_ts)
    if not olds:
        return CompactionOutcome("skipped_no_events", total, 0)

    # ... existing batch / summarisation / replace_events_with_lore_digest path ...

    # Right before the existing `return "compacted"`:
    return CompactionOutcome("compacted", total, len(union_ids))
```

(Adjust to whatever the existing return-flow looks like — read `:42-150` first to confirm. The key invariant: every code path returns a `CompactionOutcome`, never a bare string.)

- [ ] **Step 4: run** → the new test is green; the existing assertion-on-string tests now FAIL. That's expected — Task 5b.2 migrates them.

- [ ] **Step 5: commit** (intentionally with the existing tests still red — bisect-friendly because the next commit fixes them):

```bash
git add plugins/llm/src/llm/verse/compaction.py plugins/llm/tests/verse/test_compaction.py
git commit -m "feat(verse/compaction): compact_verse returns CompactionOutcome NamedTuple

Existing string-equality tests in test_compaction.py are red until the
following commit migrates them. Splitting deliberately to keep the
contract change isolated."
```

### Task 5b.2: migrate every assertion-on-string site (Test #15)

**Why:** v2.1 enumerated all eight sites. Lines 47, 66, 92, 130, 176, 192, 225, 278. Update each `assert out == "..."` to `assert out.state == "..."`. Where the assertion is on a `compacted` outcome, also assert sensible values for `out.total_events` and `out.kept_in_digest` (numbers depend on the test's seeded events — pick `>= 1` for `total_events` in the compacted cases, exact `kept_in_digest` from the test's seed pattern).

- [ ] **Step 1: list the actual sites first**:

```bash
grep -n 'assert out\(1\|2\| ==\)' plugins/llm/tests/verse/test_compaction.py
```

Expected output (matches v2.1 design §7 #15):

```
47:        assert out == "skipped_disabled"
66:        assert out == "skipped_below_floor"
92:        assert out == "skipped_no_events"
130:        assert out == "compacted"
176:        assert out1 == "compacted"
192:        assert out2 == "skipped_no_events"
225:        assert out == "compacted"
278:        assert out == "compacted"
```

If the line numbers shifted because Task 5b.1 added a `TestCompactionOutcomeShape` class above, adjust accordingly.

- [ ] **Step 2: edit each site**. For example:

```python
# was:
assert out == "skipped_disabled"
# becomes:
assert out.state == "skipped_disabled"
```

For `compacted` cases (lines 130, 176, 225, 278), also assert the new fields. Example for line 130:

```python
# was:
assert out == "compacted"
# becomes:
assert out.state == "compacted"
assert out.total_events >= 1
assert out.kept_in_digest >= 0  # exact value depends on the test's entity seeding
```

(Read the surrounding test to know what exact `kept_in_digest` to assert. If the test seeds N entities into the digest and N <= 32, expect `kept_in_digest == N`; otherwise `kept_in_digest == 32`.)

- [ ] **Step 3: run** `uv run pytest plugins/llm/tests/verse/test_compaction.py -v` → all green.

- [ ] **Step 4: commit**

```bash
git add plugins/llm/tests/verse/test_compaction.py
git commit -m "test(verse/compaction): migrate 8 string-equality sites to CompactionOutcome.state"
```

### Task 5b.3: plugin renders friendlier outcome string

**Why:** v2.1 design §8. The compaction-pass log line / `@versecompact` reply gains aging counts. Today `_run_compaction_pass` does `self.log.info("verse compaction: channel=%s outcome=%s", channel, outcome)` with `outcome` being the raw string. After 5b.1 it's a NamedTuple — render it as a human-readable string here.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_run_compaction_pass`, `@versecompact` reply formatter — find both via `grep -n '"verse compaction:"' plugins/llm/src/llm/plugin.py`).
- Modify: `plugins/llm/tests/test_plugin.py` (one new test).

- [ ] **Step 1: failing test**:

```python
def test_compaction_outcome_message_includes_aging_counts(
    self, plugin, monkeypatch, caplog
):
    from llm.verse import aging as aging_mod
    from llm.verse import compaction as compaction_mod

    monkeypatch.setattr(
        plugin._plugin, "_verse_enabled_channels", lambda: ["#foo"]
    )
    monkeypatch.setattr(
        compaction_mod, "compact_verse",
        lambda *a, **kw: compaction_mod.CompactionOutcome(
            state="compacted", total_events=12, kept_in_digest=5,
        ),
    )
    monkeypatch.setattr(
        aging_mod, "age_auto_created_entities",
        lambda *a, **kw: aging_mod.AgingOutcome(scanned=7, retired=2),
    )

    with caplog.at_level("INFO"):
        plugin._plugin._run_compaction_pass()
    msgs = [r.getMessage() for r in caplog.records]
    matched = [
        m for m in msgs
        if "compaction outcome" in m
        and "compacted 12 events" in m
        and "aged 2 entities" in m
        and "kept 5" in m
    ]
    assert matched, f"no friendly outcome message in {msgs!r}"
```

- [ ] **Step 2: run** → fails (current log emits raw NamedTuple repr).

- [ ] **Step 3: implement** — replace the log line in `_run_compaction_pass`. Pseudocode (adjust to actual code shape):

```python
def _format_compaction_outcome(
    co: CompactionOutcome, ao: AgingOutcome
) -> str:
    aged_kept = ao.scanned - ao.retired
    if co.state == "compacted":
        head = f"compacted {co.total_events} events"
    elif co.state == "skipped_below_floor":
        head = (
            f"skipped (only {co.total_events} events; floor is "
            f"{min_keep_events})"  # min_keep_events from registry
        )
    elif co.state == "skipped_no_events":
        head = f"skipped (no events past retention; total {co.total_events})"
    elif co.state == "skipped_disabled":
        head = "skipped (retention disabled)"
    else:
        head = co.state
    return (
        f"{head}; aged {ao.retired} entities (kept {aged_kept})"
    )

# Then in _run_compaction_pass:
msg = _format_compaction_outcome(compaction_outcome, aging_outcome)
self.log.info("compaction outcome for %s: %s", channel, msg)
```

(Place `_format_compaction_outcome` as a module-level helper near `_run_compaction_pass`. It takes only what it needs — keeps signature simple. If `min_keep_events` is needed, accept it as a third arg.)

Also update `@versecompact`'s reply (search `plugin.py` around `:5621-5692` for the existing `irc.reply(...)` of the compaction outcome) to call the same formatter.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(plugin): friendlier compaction outcome message with aging counts"
```

### Phase 5b verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_compaction.py plugins/llm/tests/test_plugin.py -v` → green.
- [ ] `make lint && make typecheck` → clean.

---

## Phase 6 — `verse_record` tool spec + dispatch branch

This is the user-facing surface. Phases 0–5 set up the plumbing; Phase 6 adds the actual tool.

### Task 6.1: `make_verse_tool_specs(*, max_actors)` gains the fifth tool (Test #14)

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py:16-…` (`make_verse_tool_specs`).
- Modify: `plugins/llm/tests/verse/test_avatar.py` (the 4-set assertion at `:617`).

- [ ] **Step 1: failing test** — append:

```python
class TestVerseRecordToolSpec:
    def test_make_verse_tool_specs_returns_five_with_default_max(self) -> None:
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs()
        assert len(specs) == 5
        names = {s["function"]["name"] for s in specs}
        assert names == {
            "verse_act", "verse_move", "verse_look", "verse_recall",
            "verse_record",
        }
        record = next(
            s for s in specs if s["function"]["name"] == "verse_record"
        )
        params = record["function"]["parameters"]
        assert params["properties"]["actors"]["maxItems"] == 8
        assert params["required"] == ["summary"]

    def test_make_verse_tool_specs_max_actors_dynamic(self) -> None:
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs(max_actors=12)
        record = next(
            s for s in specs if s["function"]["name"] == "verse_record"
        )
        assert record["function"]["parameters"]["properties"]["actors"]["maxItems"] == 12
```

Also update the existing 4-set assertion in `test_avatar.py:617`:

```python
# was:
assert set(handlers.keys()) == {"verse_act", "verse_move", "verse_look", "verse_recall"}
# becomes:
assert set(handlers.keys()) == {
    "verse_act", "verse_move", "verse_look", "verse_recall", "verse_record",
}
```

- [ ] **Step 2: run** → fails (the new tests fail; the 4-set assertion at `:617` will fail once 6.1's impl makes it a 5-set, but the existing test counts on it being a 4-set — flip it now in the same edit).

- [ ] **Step 3: implement** — modify `make_verse_tool_specs` in `verse/avatar.py:16`. Add the kwarg and the fifth spec:

```python
def make_verse_tool_specs(*, max_actors: int = 8) -> list[dict]:
    """Return OpenAI/LiteLLM tool specs for the five verse tools.

    The tools are model-callable but only meaningful when the @ask path
    is verse-routed (see plugin._verse_route_for + C7d dispatch).
    `max_actors` is injected into verse_record's actors maxItems; pass
    the per-channel registry value `verseAutoEntityMaxNamesPerCall` so
    operators can raise/lower per channel.
    """
    return [
        # ... existing four entries unchanged ...
        {
            "type": "function",
            "function": {
                "name": "verse_record",
                "description": (
                    "Record an in-world event involving one or more named "
                    "actors. Use whenever a member narrates events that "
                    "aren't strictly about their own avatar (e.g. \"stinky "
                    "dan threw a guff grenade at Andrew\" — record actors="
                    "[\"stinky dan\",\"Andrew\"], the grenade stays in the "
                    "summary as prose). Names that don't match an existing "
                    "entity are auto-created as kind=npc. Items, places, "
                    "and weapons are NOT actors — only put characters/"
                    "people in the actors list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "What happened, in past tense, ≤200 chars. "
                                "The full prose narration including any "
                                "items, places, or weapons mentioned. e.g. "
                                "'stinky dan threw a guff grenade at Andrew'."
                            ),
                        },
                        "actors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": max_actors,
                            "description": (
                                "Names of CHARACTERS (people/npcs) central "
                                "to the event. Do NOT include items, "
                                "weapons, places, or abstractions."
                            ),
                        },
                    },
                    "required": ["summary"],
                },
            },
        },
    ]
```

Update the literal `_verse_names = {...}` at `verse/avatar.py:452` to a 5-set:

```python
_verse_names = {
    "verse_act", "verse_move", "verse_look", "verse_recall", "verse_record",
}
```

- [ ] **Step 4: run** → green for the new tests; the dispatch branch for `verse_record` is still missing, so adding it to `_verse_names` will route `verse_record` calls into `dispatch_verse_tool_call`'s `else` branch, which logs and returns `_OK`. That's fine for now — no `verse_record` calls fire until the next task wires the dispatch branch.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/verse/test_avatar.py
git commit -m "feat(verse/avatar): make_verse_tool_specs(*, max_actors) gains verse_record"
```

### Task 6.2: dispatch branch — happy path

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py` (`dispatch_verse_tool_call` body, after the existing `verse_recall` branch).

- [ ] **Step 1: failing test** — append to `test_verse_record.py`:

```python
class TestVerseRecordDispatch:
    def test_dispatch_happy_path(self, store: VerseStore) -> None:
        """dispatch_verse_tool_call routes 'verse_record' to
        record_user_event and surfaces event_id in the result payload."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )
        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(
            store, alice_id, "verse_record",
            {"summary": "alice waved", "actors": ["bob"]},
        )
        assert isinstance(result, VerseDispatchResult)
        assert result.ok is True
        assert result.error is None
        assert result.payload is not None
        assert result.payload["status"] == "ok"
        assert isinstance(result.payload["event_id"], int)
```

- [ ] **Step 2: run** → fails (the `else` branch in `dispatch_verse_tool_call` swallows `verse_record` and returns `_OK` without `event_id`).

- [ ] **Step 3: implement** — add a new `elif name == "verse_record":` branch to `dispatch_verse_tool_call` in `verse/avatar.py:383-…`, BEFORE the `else: log.warning("unknown verse tool…")`:

```python
elif name == "verse_record":
    return _dispatch_verse_record(
        store, avatar_id, args, log=log
    )
```

And the helper at module level:

```python
def _dispatch_verse_record(
    store: VerseStore,
    avatar_id: int,
    args: dict[str, Any],
    *,
    log: logging.Logger,
) -> VerseDispatchResult:
    summary = (args.get("summary") or "").strip()
    if not summary:
        return VerseDispatchResult(
            ok=False, error="summary required"
        )
    if len(summary) > 200:
        return VerseDispatchResult(
            ok=False,
            error=f"summary too long: {len(summary)} chars (max 200)",
        )
    raw = args.get("actors") or []
    if not isinstance(raw, list):
        return VerseDispatchResult(
            ok=False, error="actors must be an array"
        )
    max_actors = args.get("_max_actors", 8)
    # Filter THEN slice — order matters (raw=["alice", 42, "bob"] with
    # max=2 must yield ["alice","bob"], not ["alice"] because the 42
    # ate a slot).
    cleaned = [
        s.strip() for s in raw if isinstance(s, str) and s.strip()
    ]
    actors = cleaned[:max_actors]
    event_id = store.record_user_event(
        actor_id=avatar_id,
        summary=summary,
        actor_names=actors,
        now=time.time,
    )
    return VerseDispatchResult(
        ok=True, payload={"status": "ok", "event_id": event_id}
    )
```

(`time` may need to be imported; check the file's imports.)

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/verse/test_verse_record.py
git commit -m "feat(verse/avatar): dispatch verse_record to record_user_event"
```

### Task 6.3: dispatch — empty summary error (Test #8)

- [ ] **Step 1: failing test**:

```python
def test_dispatch_empty_summary_returns_error(
    self, store: VerseStore
) -> None:
    from llm.verse.avatar import dispatch_verse_tool_call

    alice_id = _opt_in(store)
    n_events_before = len(store.recent_events(limit=100))
    result = dispatch_verse_tool_call(
        store, alice_id, "verse_record",
        {"summary": "   ", "actors": ["bob"]},
    )
    assert result.ok is False
    assert result.error == "summary required"
    assert len(store.recent_events(limit=100)) == n_events_before
```

- [ ] **Step 2: run** → green (covered by 6.2's `if not summary` guard).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): empty summary returns error and writes nothing"
```

### Task 6.4: dispatch — too-long summary error (Test #9)

- [ ] **Step 1: failing test**:

```python
def test_dispatch_too_long_summary_returns_error(
    self, store: VerseStore
) -> None:
    from llm.verse.avatar import dispatch_verse_tool_call

    alice_id = _opt_in(store)
    summary = "x" * 250  # > 200
    result = dispatch_verse_tool_call(
        store, alice_id, "verse_record",
        {"summary": summary, "actors": []},
    )
    assert result.ok is False
    assert result.error is not None
    assert "too long" in result.error
    assert "250" in result.error
```

- [ ] **Step 2: run** → green.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): too-long summary returns error, no truncation"
```

### Task 6.5: dispatch — filter-then-slice + non-string filter (Tests #7, #12)

- [ ] **Step 1: failing test**:

```python
def test_dispatch_actors_filter_then_slice(self, store: VerseStore) -> None:
    """Mixed-type input ['alice', 42, 'bob'] with max_actors=2 yields
    actors ['alice', 'bob'], not ['alice'] (the 42 must NOT eat a
    slot)."""
    from llm.verse.avatar import dispatch_verse_tool_call

    alice_id = _opt_in(store)
    result = dispatch_verse_tool_call(
        store, alice_id, "verse_record",
        {
            "summary": "test mixed",
            "actors": ["alice_npc", 42, "bob_npc"],
            "_max_actors": 2,
        },
    )
    assert result.ok is True
    # Both alice_npc and bob_npc should exist as auto-NPCs
    assert store.find_active_entity_by_name("alice_npc") is not None
    assert store.find_active_entity_by_name("bob_npc") is not None

def test_dispatch_actors_empty_or_whitespace_filtered(
    self, store: VerseStore
) -> None:
    """actors=['', '  ', 'alice'] processes only 'alice'; no
    empty-name entities created."""
    from llm.verse.avatar import dispatch_verse_tool_call

    alice_id = _opt_in(store)
    result = dispatch_verse_tool_call(
        store, alice_id, "verse_record",
        {"summary": "filter ws", "actors": ["", "  ", "alice_w"]},
    )
    assert result.ok is True
    assert store.find_active_entity_by_name("alice_w") is not None
    assert store.find_active_entity_by_name("") is None
    assert store.find_active_entity_by_name("  ") is None

def test_dispatch_truncates_to_max_actors(self, store: VerseStore) -> None:
    from llm.verse.avatar import dispatch_verse_tool_call

    alice_id = _opt_in(store)
    raw = [f"npc{i}" for i in range(20)]
    result = dispatch_verse_tool_call(
        store, alice_id, "verse_record",
        {"summary": "many", "actors": raw, "_max_actors": 5},
    )
    assert result.ok is True
    # Only first 5 created
    for i in range(5):
        assert store.find_active_entity_by_name(f"npc{i}") is not None
    for i in range(5, 20):
        assert store.find_active_entity_by_name(f"npc{i}") is None
```

- [ ] **Step 2: run** → green (all three covered by 6.2's filter-then-slice).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_verse_record.py
git commit -m "test(verse_record): actors filter-then-slice, mixed types, truncation"
```

### Task 6.6: registry → max_actors plumbing through `plugin.py:3281` (Plugin Test #13)

**Why:** v2.1 design §1 — `make_verse_tool_specs` is called per assistant request. The `max_actors` value must flow from `verseAutoEntityMaxNamesPerCall` (per-channel). The natural call-site is wherever specs are built; the design also wants the dispatch closure (`make_verse_extra_handlers`) to know it (so the dispatcher can default `_max_actors` from the registry rather than from a hardcoded 8).

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (callsite at `:3281`).
- Modify: `plugins/llm/src/llm/verse/avatar.py` (`make_verse_extra_handlers` accepts `max_actors`).
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: failing test** — append to `test_plugin.py`. The driver pattern is **direct unit-style** rather than driving the full `assistant_request` path: extract a tiny helper from the plugin (or just call the wiring inline) so we can assert on the registry → `make_verse_extra_handlers` plumbing without setting up the full IRC + LLM stack.

```python
class TestMaxActorsRegistryPlumbing:
    def test_max_actors_flows_to_make_verse_extra_handlers(
        self, plugin, monkeypatch
    ):
        """When channel #x has verseAutoEntityMaxNamesPerCall=4, the
        plugin's verse-handler-building site calls
        make_verse_extra_handlers with max_actors=4. We unit-test the
        wiring directly rather than via the full assistant_request
        path: monkeypatch make_verse_extra_handlers to a spy, then
        invoke whatever method on the plugin builds verse handlers
        for a channel."""
        from llm.verse import avatar as avatar_mod

        captured: list[int] = []
        real_handlers = avatar_mod.make_verse_extra_handlers

        def spy_handlers(store, avatar_id, logger=None, *, max_actors=8):
            captured.append(max_actors)
            return real_handlers(store, avatar_id, logger=logger, max_actors=max_actors)

        monkeypatch.setattr(
            avatar_mod, "make_verse_extra_handlers", spy_handlers
        )
        # Mock registryValue so #x returns 4 for the new key, and
        # delegates everything else to the real registry.
        real_registry = plugin._plugin.registryValue

        def fake_registry(key, channel=None):
            if key == "verseAutoEntityMaxNamesPerCall" and channel == "#x":
                return 4
            return real_registry(key, channel=channel)

        monkeypatch.setattr(plugin._plugin, "registryValue", fake_registry)

        # Direct invocation of the wiring. After Task 6.6 lands, the
        # plugin exposes a small helper (or this test calls the same
        # lines that plugin.py:3281 calls). The helper signature:
        #     plugin._plugin._build_verse_handlers_for(channel="#x")
        #         -> dict[str, Callable]
        # If 6.6's implementation chooses a different name, update
        # this single call site.
        handlers = plugin._plugin._build_verse_handlers_for(channel="#x")
        assert handlers is not None
        assert 4 in captured, f"max_actors not plumbed; got {captured}"
```

This test pins a concrete contract: the plugin must expose a
`_build_verse_handlers_for(channel)` helper that wraps the wiring at
`plugin.py:3281`. Task 6.6's implementation introduces that helper —
which is also a cleaner shape than inlining the wiring inside
`assistant_request`'s closure. If that name is wrong for the codebase,
adjust both the implementation in step (b) below AND this test in
lock-step.

- [ ] **Step 2: run** → fails.

- [ ] **Step 3: implement step (a)** — modify `make_verse_extra_handlers` signature in `verse/avatar.py:438`:

```python
def make_verse_extra_handlers(
    store: VerseStore,
    avatar_id: int,
    logger: logging.Logger | None = None,
    *,
    max_actors: int = 8,
) -> dict[str, Callable[[dict[str, Any]], Any]]:
    """... existing docstring ...

    `max_actors` is the per-channel cap from verseAutoEntityMaxNamesPerCall.
    It's threaded into the verse_record dispatch branch via the args
    dict's '_max_actors' key — set inside _call before dispatch.
    """
    log = logger or _log
    _verse_names = {"verse_act", "verse_move", "verse_look", "verse_recall", "verse_record"}

    def _handler(name: str) -> Callable[[dict[str, Any]], _VerseToolResult]:
        def _call(args: dict[str, Any]) -> _VerseToolResult:
            # Inject closure-bound max_actors so the dispatch branch
            # for verse_record can read it without changing the
            # dispatch signature.
            args = dict(args)  # shallow copy — never mutate caller dict
            args.setdefault("_max_actors", max_actors)
            result = dispatch_verse_tool_call(
                store, avatar_id, name, args, logger=log
            )
            # ... rest of existing _call body (uses `result` from 0a.2) ...
```

(The `_max_actors` setdefault keeps Tasks 6.3–6.5 backward-compatible when tests pass `_max_actors` directly.)

- [ ] **Step 3: implement step (b)** — extract a helper method `_build_verse_handlers_for(channel)` on the plugin, and use it from the call-site at `plugin.py:3281`. Place the helper near the existing `_get_or_create_verse_store` (around `plugin.py:4952`):

```python
def _build_verse_handlers_for(
    self, channel: str
) -> dict[str, Callable[[dict[str, Any]], Any]] | None:
    """Build the verse extra-handlers dict for ``channel``, plumbing
    the per-channel verseAutoEntityMaxNamesPerCall into the dispatch
    closure. Returns None if the channel has no verse route.

    Extracted so the registry → max_actors plumbing is testable
    without setting up the full assistant_request stack."""
    verse_route = self._verse_route_for(channel)
    if verse_route is None:
        return None
    max_actors = self.registryValue(
        "verseAutoEntityMaxNamesPerCall", channel=channel,
    )
    return make_verse_extra_handlers(
        verse_route.store,
        verse_route.avatar_id,
        max_actors=max_actors,
    )
```

(`_verse_route_for` already exists on the plugin — reuse it. If the actual symbol name differs, search via `grep -n 'def _verse_route' plugins/llm/src/llm/plugin.py`.)

Then replace the inline wiring at `plugin.py:3281`:

```python
# was:
if verse_route is not None:
    verse_handlers = make_verse_extra_handlers(
        verse_route.store, verse_route.avatar_id
    )
    combined_handlers: dict | None = {
        **(bridge_handlers or {}),
        **verse_handlers,
    }
else:
    combined_handlers = bridge_handlers

# becomes:
verse_handlers = self._build_verse_handlers_for(channel)
if verse_handlers is not None:
    combined_handlers: dict | None = {
        **(bridge_handlers or {}),
        **verse_handlers,
    }
else:
    combined_handlers = bridge_handlers
```

If `make_verse_tool_specs` is also called nearby (specs are sent to the model alongside handlers — find via `grep -n make_verse_tool_specs plugins/llm/src/llm/plugin.py`), wrap that call into a sibling helper `_build_verse_tool_specs_for(channel)` that reads the same registry value and call it from the same site. Keeps the two pieces of plumbing symmetric.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/avatar.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(plugin): plumb verseAutoEntityMaxNamesPerCall into dispatch closure"
```

### Phase 6 verification

- [ ] `uv run pytest plugins/llm/tests/verse/test_verse_record.py plugins/llm/tests/verse/test_avatar.py plugins/llm/tests/test_plugin.py -v` → all green.
- [ ] `make lint && make typecheck` → clean.
- [ ] `uv run pytest plugins/llm -q` → no regressions across the plugin.

---

## Phase 7 — Operator guide + CHANGELOG

**Files:**
- Modify: `docs/guide/operator/forest-verse.md`.
- Modify: `CHANGELOG.md`.

### Task 7.1: operator guide — three new H2 sections

- [ ] **Step 1: edit** `docs/guide/operator/forest-verse.md`. Add three new H2 sections (anchors as called out in v2.1 design §13):

```markdown
## Member-driven worldbuilding (verse_record)

Opted-in members can narrate events involving entities other than their
own avatar. Example:

> `vibebot, stinky dan threw a guff grenade at Andrew`

The bot's assistant calls the new `verse_record` tool with
`actors=["stinky dan","Andrew"]`. The grenade stays in the summary as
prose — items / weapons / places are never auto-created as actors.

Names that match an existing entity link to it (precedence: avatar >
npc > item > place, case-insensitive, retired entities skipped). Names
that don't match are auto-created as `kind=npc` and tagged
`auto_created='1'`.

Cap: `verseAutoEntityMaxNamesPerCall` (default 8) limits the actors
array length. Raise per-channel if your verse routinely cites large
casts; raising past 16 invites high-cardinality flooding.

## Auto-created NPCs and aging

Auto-created NPCs without recent mentions are soft-retired by the daily
compaction pass. The "recent mentions" definition (heartbeat scope):

- A `verse_record` call mentioning the NPC.
- A loom-applied or crosspoll-emitted proposal referencing the NPC.
- A compaction digest event listing the NPC in its truncated entity
  union (capped at the first 32 ids per digest).

Other paths that touch entities — `verse_act`, `verse_move`,
`verse_look`, `verse_recall`, `add_relation`, `opt_in_avatar` — do
**not** count.

Knob: `verseAutoEntityRetireDays` (default 14, per-channel). Set to
`0` to disable aging entirely. If your verse loses cast members the
operator wanted to keep, raise the value or manually clear the
`auto_created` attribute on the entity (then it's a "real" NPC and
won't age).

A retired NPC with the same name as a future mention does **not**
rehydrate — a fresh active row is created. The orphan ages out under
the same policy.

## Compaction outcome reference

The compaction pass log line / `@versecompact` reply now reads:

```
compaction outcome for #foo: compacted 12 events; aged 2 entities (kept 5)
compaction outcome for #foo: skipped (only 7 events; floor is 20); aged 0 entities (kept 0)
```

`CompactionOutcome.state` values:

| state | meaning |
|---|---|
| `compacted` | Old events past `verseEventRetentionDays` were summarised into one digest event. |
| `skipped_disabled` | `verseEventRetentionDays <= 0` — retention is off. |
| `skipped_below_floor` | Total events count is below `verseEventCompactionFloor`; nothing yet to compact. |
| `skipped_no_events` | No events older than the retention cutoff. |

Aging counts (`aged N entities (kept M)`) come from `AgingOutcome.retired`
and `scanned - retired` respectively.
```

- [ ] **Step 2: commit**

```bash
git add docs/guide/operator/forest-verse.md
git commit -m "docs(operator): verse_record + auto-NPC aging + new compaction outcomes"
```

### Task 7.2: CHANGELOG entry

- [ ] **Step 1: edit** `CHANGELOG.md`. Under the existing "Unreleased" heading add:

```markdown
### Added
- `verse_record` assistant tool: opted-in verse members can narrate events
  involving entities other than themselves; unknown actors auto-create as
  `kind=npc`. (#XXXX — replace with actual PR/issue number once filed)
- `verseAutoEntityRetireDays` (default 14, per-channel): soft-retire
  auto-created NPCs after this many days without a heartbeat.
- `verseAutoEntityMaxNamesPerCall` (default 8, per-channel): hard cap on
  the `actors` array length advertised by the tool spec and enforced by
  dispatch.

### Changed
- `compact_verse` returns a `CompactionOutcome` NamedTuple instead of a
  string. Operator-visible compaction messages now include aging counts.
- `dispatch_verse_tool_call` returns a structured `VerseDispatchResult`
  so verse tools can surface error and payload data to the model. The
  four legacy tools' observable JSON is unchanged.
- `find_active_entity_by_name(name)` resolves names with the documented
  `avatar > npc > item > place` precedence and skips retired entities.
  The legacy `find_entity_by_name(name, kind=...)` is unchanged.
```

- [ ] **Step 2: commit**

```bash
git add CHANGELOG.md
git commit -m "changelog: verse_record + auto-NPC aging"
```

### Phase 7 verification

- [ ] No code changes — docs only.

---

## Phase 8 — Final task: dispatch re-review

**Before merging the implementation, run two reviews in parallel via the Agent tool:**

- [ ] **Reviewer 1: senior code review.**

```
Agent(
  subagent_type="general-purpose",
  description="Code review of verse_record implementation",
  prompt="""
You are a senior code reviewer.  Review the diff between origin/main and
HEAD on this branch in /Users/rdrake/workspace/afternet/vibebot-v8.

Source design: docs/plans/2026-05-09-verse-record-design.md (v2.1).
Plan: docs/plans/2026-05-09-verse-record-pr1.md.

Verify:
1. Each test in §7 of the design doc exists in the diff (16 tests + 9
   aging + 4 plugin = 29 specific test names; check the table in the
   plan's 'Test → Task mapping' section).
2. Each step from §11 of the design (0a, 0b, 1, 2, 3, 4, 5a, 5b, 6, 7,
   8, 9) has its commits in order (read `git log --oneline main..HEAD`).
3. The two FATAL findings from v2 (dispatch contract, write_transaction
   reentrancy) have observable resolutions in the diff: VerseDispatchResult
   exists and propagates through the wrapper; the four public mutators
   each have an _inline variant that takes a conn parameter.
4. No SIG resolution from v2 was dropped during implementation (cross-
   reference v2.1's Revisions section).
5. Code quality: do the new tests use real SQLite (no DB mocks)? Do they
   assert on observable behaviour rather than internal calls?
6. Out-of-scope guard: does the diff touch anything not listed in the
   plan's 'Files map' or §10 of the design? Flag anything outside.

Report findings as:
- BLOCKERS (must fix before merge)
- NITS (worth fixing if cheap)
- READY TO MERGE / NOT READY TO MERGE — explicit verdict
""",
)
```

- [ ] **Reviewer 2: Codex adversarial pass on actual code.**

```
Agent(
  subagent_type="codex:codex-rescue",
  description="Adversarial pass on verse_record implementation",
  prompt="""
This is Codex's third adversarial sweep on the verse_record + aging
work — first two passes were on the design, this one is on the actual
code. Working dir is /Users/rdrake/workspace/afternet/vibebot-v8;
diff is `git diff main..HEAD`.

Specifically look for:
1. Drift from docs/plans/2026-05-09-verse-record-design.md (v2.1) — any
   place the implementation chose differently than the design without
   updating the design.
2. New bugs introduced during implementation. Common drift modes:
   - The race-test-with-sleep-injection (Phase 2 Task 2.5) actually
     proves the SQLite-level race window, not just Python-lock
     serialisation.
   - The heartbeat in _replace_events_with_source runs on the same
     conn as the digest INSERT (atomic).
   - apply_or_queue's heartbeat fires ONLY in `applied` and
     `crosspoll_emitted`, never in `queued` / `rejected_invalid_refs`.
   - record_user_event's filter-then-slice handles ['alice', 42,
     'bob'] with max=2 as ['alice','bob'] (not ['alice']).
3. Test gaps: tests that look fine but won't catch the bug they claim
   to. E.g. a test that asserts on `outcome.state == 'compacted'` but
   doesn't assert on `outcome.total_events` would let a regression in
   the count silently land.
4. Schema-invariant violations: events.source still 'avatar' (not a
   new value); _MAX_DIGEST_ENTITY_IDS imported (not hardcoded 32).

Report:
- FATAL (would break prod or hide bugs in tests)
- SIGNIFICANT (worth a follow-up commit before merge)
- READY TO MERGE / NOT READY TO MERGE — explicit verdict
""",
)
```

- [ ] **Both reviewers must explicitly say "ready to merge" before pushing.**

- [ ] **Integrate findings as a final commit.** If a finding is significant enough to require a v3 of the design doc (e.g. a class of issue Codex's design-time passes missed), update the design doc first, then re-run both reviewers.

- [ ] **Suggested commit message for the integration commit:**

```bash
git add <files>
git commit -m "fix(verse-record): address re-review findings

- <finding 1>
- <finding 2>
"
```

---

## Phase 9 — Wait for CI + Docker, restart prod, validate

Per repo convention (recorded in `.claude/projects/.../memory/feedback_wait_for_docker.md`).

- [ ] `git push origin main`.
- [ ] Wait for CI workflow to go green: `gh run list --workflow=ci --limit=1`.
- [ ] **Wait for Docker build workflow** to go green (separate workflow): `gh run list --workflow="Build and Push Docker Image" --limit=1`.
- [ ] SSH to vibebot host and restart: `ssh -i ~/.ssh/id_rsa vibebot@rdrake.org systemctl --user restart vibebot`.
- [ ] Tail logs for clean startup, no AssertionErrors, no schema migration errors:

```bash
ssh -i ~/.ssh/id_rsa vibebot@rdrake.org docker logs vibebot --since 2m
```

Look for: no `AssertionError`, no `OperationalError`, the plugin reports both new registry keys at startup, no `unknown verse tool` warnings on existing tool calls.

- [ ] In a verse channel, send a test prompt: `vibebot, stinky dan threw a guff grenade at Andrew`. Verify:
  - The bot replies in-character.
  - `@versedump` shows new entities `stinky dan` and `Andrew` (or links to existing if either was an avatar).
  - `auto_created='1'` is set on the new NPCs.
  - `last_seen_ts` is populated.
- [ ] In a non-verse channel, verify the four legacy tools still work (regression check on the 0a contract migration).

---

## Self-review checklist (run before opening PR)

- [ ] Every §7 test from the design doc has an explicit task in this plan (29 tests total — see the Test→Task mapping table near the top).
- [ ] Every §11 step from the design (0a, 0b, 1, 2, 3, 4, 5a, 5b, 6, 7, 8, 9) has its own phase or task block above.
- [ ] No placeholder text remains: search the plan for `TBD`, `TODO`, `implement later`, `Similar to Task`, `…and so on`.
- [ ] Type and method names are consistent across tasks: `VerseDispatchResult`, `record_user_event`, `_find_active_entity_by_name_inline`, `find_active_entity_by_name`, `list_entities_with_attribute`, `bump_last_seen_ts`, `AgingOutcome`, `age_auto_created_entities`, `CompactionOutcome`, `verseAutoEntityRetireDays`, `verseAutoEntityMaxNamesPerCall`.
- [ ] `_MAX_DIGEST_ENTITY_IDS` is imported from `llm.verse.compaction`, never hardcoded as 32 or 40.
- [ ] The race test (Task 2.5) injects `time.sleep` between `_find_active_entity_by_name_inline` and `_add_entity_inline` — without injection it passes trivially.
- [ ] `events.source` stays `'avatar'`. No new schema migration.
- [ ] Phase 0a (dispatch contract) and 0b (inline-helper extraction) are completed BEFORE any task that depends on them (Phase 2 needs both; Phase 4.1 needs `_set_attribute_inline` from 0b.2; Phase 6 needs the `VerseDispatchResult` shape from 0a).
