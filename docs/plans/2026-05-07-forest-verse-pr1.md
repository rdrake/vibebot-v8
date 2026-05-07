# Forest-verse PR 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the verse store + avatar shim, remove the rpg plugin, and remove the old `forestNicks` and spontaneous-mode code paths in one PR. After this PR ships, opted-in users have a verse-aware `@ask` flow but the loom is not yet wired (PR 2).

**Architecture:**
- New package `plugins/llm/src/llm/verse/` with `store.py`, `avatar.py`, `schema.sql`.
- Verse store mirrors the thread-local + WAL + per-DB write-lock pattern from `plugins/llm/src/llm/persistence.py:160-216`.
- Avatar shim wraps `@ask` for opted-in users. Verb whitelist enumerated; off-list verbs land as event-only.
- `plugins/rpg/` is deleted in the same PR. `forestNicks` and spontaneous registry keys are deleted with their code paths.

**Tech Stack:**
- Python 3.13+ via `uv`, `pytest`, `ruff`, `ty`.
- Limnoria 2025+ plugin (`supybot`).
- SQLite stdlib, real DB files in tmp_path for tests (no mocks per `feedback_wait_for_docker.md` rule).

**Reference design:** `docs/plans/2026-05-07-forest-verse-design.md` (v3, clean break, no migration).

**Working directory:** `.worktrees/forest-verse-pr1` (branch `feat/forest-verse-pr1`). Baseline: 1748 tests passing, 93.9% coverage on `main`.

**Project rules to honor:**
- `make lint && make typecheck` runs after every Edit (memory note); commits also trigger pre-commit (ruff format + gitleaks + ty). Don't suppress.
- All tests use real SQLite files, not mocks.
- `uv run pytest …` not bare `pytest`.
- Frequent atomic commits; one task → one commit (or two: test+impl can be one commit when they belong together).

---

## Phase A — Verse store

### Task A1: Create the package skeleton

**Files:**
- Create: `plugins/llm/src/llm/verse/__init__.py` (empty module docstring)
- Create: `plugins/llm/src/llm/verse/schema.sql` (placeholder; populated in A3)
- Create: `plugins/llm/src/llm/verse/store.py` (placeholder)
- Create: `plugins/llm/src/llm/verse/avatar.py` (placeholder)
- Create: `plugins/llm/src/llm/verse/tests/__init__.py` (empty)
- Create: `plugins/llm/src/llm/verse/tests/conftest.py`

**Step 1: write `__init__.py`:**

```python
"""Forest-verse: per-channel structured world model and avatar shim."""
```

**Step 2: write `tests/conftest.py`:**

```python
"""Pytest fixtures for verse tests — real SQLite, no mocks."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    """Per-test directory for verse SQLite files."""
    d = tmp_path / "verse"
    d.mkdir()
    return d
```

**Step 3: commit**

```bash
git add plugins/llm/src/llm/verse/
git commit -m "feat(verse): scaffold package skeleton"
```

---

### Task A2: db filename sanitization helper

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`
- Test: `plugins/llm/src/llm/verse/tests/test_store.py` (new)

**Step 1: write the failing test** in `test_store.py`:

```python
"""Tests for the verse store."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.verse.store import db_path_for_channel


class TestDbPathForChannel:
    def test_lowercases_and_sanitizes(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#Foo")
        assert result.parent == verse_db_dir
        assert result.name.startswith("_foo_")
        assert result.suffix == ".db"

    def test_distinguishes_case_variants(self, verse_db_dir: Path) -> None:
        # #Foo and #foo must NOT collide on case-insensitive filesystems.
        upper = db_path_for_channel(verse_db_dir, "#Foo")
        lower = db_path_for_channel(verse_db_dir, "#foo")
        assert upper != lower

    def test_strips_funky_characters(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#foo!bar/baz")
        assert "!" not in result.name
        assert "/" not in result.name

    def test_idempotent(self, verse_db_dir: Path) -> None:
        a = db_path_for_channel(verse_db_dir, "#afnet")
        b = db_path_for_channel(verse_db_dir, "#afnet")
        assert a == b
```

**Step 2: run** `uv run pytest plugins/llm/src/llm/verse/tests/test_store.py -v` — should fail with ImportError.

**Step 3: implement** in `store.py`:

```python
"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE_RE = re.compile(r"[^a-z0-9_-]")


def db_path_for_channel(base_dir: Path, channel: str) -> Path:
    """Return the per-channel SQLite path under ``base_dir``.

    Lowercases the channel name, replaces non-``[a-z0-9_-]`` with ``_``,
    and appends an 8-char SHA-256 prefix of the *original* channel string
    to disambiguate collisions on case-insensitive filesystems.
    """
    lowered = channel.lower()
    safe = _SAFE_RE.sub("_", lowered)
    digest = hashlib.sha256(channel.encode("utf-8")).hexdigest()[:8]
    return base_dir / f"{safe}_{digest}.db"
```

**Step 4: run tests** — should pass.

**Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/src/llm/verse/tests/test_store.py
git commit -m "feat(verse): db_path_for_channel sanitizer"
```

---

### Task A3: schema.sql + initial connection

**Files:**
- Modify: `plugins/llm/src/llm/verse/schema.sql`
- Modify: `plugins/llm/src/llm/verse/store.py`
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`

**Step 1: write `schema.sql`** with the full data model from the design doc:

```sql
-- Forest-verse per-channel schema.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL CHECK (kind IN ('avatar','npc','place','faction','item')),
    name       TEXT NOT NULL,
    summary    TEXT NOT NULL DEFAULT '',
    status     TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','retired')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind, status);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);

CREATE TABLE IF NOT EXISTS attributes (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    PRIMARY KEY (entity_id, key)
);

CREATE INDEX IF NOT EXISTS idx_attributes_kv ON attributes(key, value);

CREATE TABLE IF NOT EXISTS relations (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    from_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    kind    TEXT NOT NULL,
    note    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id, kind);
CREATE INDEX IF NOT EXISTS idx_relations_to   ON relations(to_id, kind);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    summary    TEXT NOT NULL,
    entity_ids TEXT NOT NULL DEFAULT '[]',  -- JSON array
    source     TEXT NOT NULL CHECK (source IN ('avatar','loom','crosspoll'))
);

CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

CREATE TABLE IF NOT EXISTS avatar_link (
    entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
    nick      TEXT NOT NULL,
    account   TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_avatar_link_nick    ON avatar_link(nick);
CREATE UNIQUE INDEX IF NOT EXISTS idx_avatar_link_account ON avatar_link(account) WHERE account IS NOT NULL;

CREATE TABLE IF NOT EXISTS proposals (
    id          TEXT PRIMARY KEY,           -- UUID
    created_at  REAL NOT NULL,
    cycle_id    TEXT NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity')),
    payload     TEXT NOT NULL,              -- JSON
    confidence  REAL NOT NULL,
    provenance  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
    reviewer    TEXT,
    reviewed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at);
```

**Step 2: write the failing test**:

```python
class TestVerseStoreInit:
    def test_creates_db_with_schema(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        # Should be able to open and check schema_version exists
        with store.read_connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row is not None
            assert row[0] >= 1

    def test_idempotent_init(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#afnet")  # second open must not double-apply migrations
        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            assert count == 1
```

**Step 3: implement `VerseStore.__init__`, `_connect`, `_migrate`, `read_connection`** in `store.py`. Follow the pattern at `plugins/llm/src/llm/persistence.py:160-216` exactly:
- Thread-local connection (`threading.local()`).
- `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` on first connect.
- `_migrate` applies `schema.sql` only when `schema_version` is empty.
- `read_connection()` is a context manager that yields the thread-local conn (no commit).
- `SCHEMA_VERSION = 1`.

Sketch:

```python
import sqlite3
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

SCHEMA_VERSION = 1
_SCHEMA_SQL = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")


class VerseStore:
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
        with self.write_transaction() as conn:
            conn.executescript(_SCHEMA_SQL)
            existing = conn.execute("SELECT version FROM schema_version").fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
```

**Step 4: run tests** — should pass.

**Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/
git commit -m "feat(verse): store init with schema + WAL + thread-local conn"
```

---

### Task A4: entity CRUD

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`

**Step 1: write failing tests** for `add_entity`, `get_entity`, `find_entity_by_name`, `set_status`, `list_entities_by_kind`.

```python
class TestEntities:
    def test_add_and_get(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity(kind="avatar", name="alice", summary="a curious traveller")
        e = store.get_entity(eid)
        assert e.name == "alice"
        assert e.kind == "avatar"
        assert e.status == "active"

    def test_find_by_name_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity(kind="place", name="The Clearing", summary="")
        e = store.find_entity_by_name("the clearing")
        assert e is not None
        assert e.name == "The Clearing"

    def test_retire_soft_deletes(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity(kind="avatar", name="bob", summary="")
        store.set_status(eid, "retired")
        e = store.get_entity(eid)
        assert e.status == "retired"

    def test_list_entities_by_kind_filters_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        active = store.add_entity(kind="avatar", name="active", summary="")
        retired = store.add_entity(kind="avatar", name="retired", summary="")
        store.set_status(retired, "retired")
        names = [e.name for e in store.list_entities_by_kind("avatar", status="active")]
        assert names == ["active"]
```

**Step 2: run** — fails (no methods).

**Step 3: implement.** Add an `Entity` NamedTuple and the methods. Names map directly to columns. Use `lower(name) = lower(?)` for case-insensitive lookup. All writes through `write_transaction`.

**Step 4: run tests** — pass.

**Step 5: commit** — `feat(verse): entity CRUD`

---

### Task A5: attributes + relations CRUD

**Files:** same as A4.

Tests to add:
- `set_attribute(eid, key, value)` upserts.
- `get_attribute(eid, key)` returns value or None.
- `list_attributes(eid)` returns dict.
- `add_relation(from_id, to_id, kind, note="")` returns rel id.
- `list_relations(from_id=None, to_id=None, kind=None)` filters.

Implement the methods. Commit: `feat(verse): attributes + relations`.

---

### Task A6: events append + retrieval

**Files:** same.

Tests:
- `add_event(summary, entity_ids, source)` returns id, sets `ts` to current time.
- `recent_events(limit=10, exclude_sources=())` returns newest-first.
- `recent_events(exclude_sources=('crosspoll',))` skips crosspoll source (forward-prep for PR 3 but the API lands now).

Implement. JSON-encode `entity_ids`. Commit: `feat(verse): events append + retrieval`.

---

### Task A7: avatar_link CRUD + soft-delete avatar

**Files:** same.

Tests:
- `link_avatar(entity_id, nick, account=None)` upserts.
- `find_avatar_by_nick(nick)` returns entity_id or None (case-insensitive).
- `find_avatar_by_account(account)` returns entity_id or None.
- `unlink_avatar(entity_id)` removes link AND retires the entity.

Implement. Commit: `feat(verse): avatar_link CRUD`.

---

### Task A8: write-lock concurrency test

**Files:**
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`

**Step 1: write a real-threads concurrency test.**

```python
class TestConcurrency:
    def test_parallel_writers_serialize(self, verse_db_dir: Path) -> None:
        from concurrent.futures import ThreadPoolExecutor
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        N = 50

        def writer(i: int) -> int:
            return store.add_entity(kind="npc", name=f"npc{i}", summary="")

        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(writer, range(N)))

        assert len(set(ids)) == N
        rows = store.list_entities_by_kind("npc")
        assert len(rows) == N
```

**Step 2: run** — should pass given the existing `_lock` in `write_transaction`.

**Step 3: commit** — `test(verse): write-lock concurrency`.

---

### Task A9: opt-in starter scene helper

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`

Add a high-level helper:

```python
def opt_in_avatar(self, nick: str, account: str | None, instruct_text: str) -> AvatarOptInResult:
    """Atomically: ensure a starter place exists, create the avatar entity,
    set its location attribute to the place, link the IRC user, and return
    a one-paragraph scene description for the caller to send back."""
```

**`AvatarOptInResult`** = NamedTuple of `entity_id`, `place_name`, `scene_text`, `was_already_opted_in: bool`.

**Starter place:** if no entity of `kind='place'` and `status='active'` exists, create one named `"The Clearing"` with summary `"A quiet woodland clearing where new stories begin."`. Subsequent users land at the active place with the most events recently.

**Scene text format:** `"You step into <place name>. <place summary> Try `verse_act look around` to begin."`

Tests cover: first-user case (creates clearing), second-user case (lands in same place, no duplicate), idempotent re-opt-in (returns existing avatar with `was_already_opted_in=True`).

Commit: `feat(verse): opt-in starter scene`.

---

## Phase B — Avatar shim

### Task B1: verb whitelist + dispatch

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py`
- Test: `plugins/llm/src/llm/verse/tests/test_avatar.py` (new)

**Step 1: define the whitelist.**

```python
"""Verse-aware @ask shim: tools, system prompt builder, verb dispatch."""

from __future__ import annotations

from enum import Enum

class VerbEffect(Enum):
    EVENT_ONLY = "event_only"
    MOVE = "move"          # updates location attribute
    ITEM = "item"          # records take/drop/give event linking item
    SEARCH = "search"      # event_only but flagged to re-render scene


VERB_TABLE: dict[str, VerbEffect] = {
    # event-only
    "whisper": VerbEffect.EVENT_ONLY,
    "speak":   VerbEffect.EVENT_ONLY,
    "listen":  VerbEffect.EVENT_ONLY,
    "examine": VerbEffect.EVENT_ONLY,
    "wait":    VerbEffect.EVENT_ONLY,
    "signal":  VerbEffect.EVENT_ONLY,
    "gesture": VerbEffect.EVENT_ONLY,
    # movement
    "move":    VerbEffect.MOVE,
    "flee":    VerbEffect.MOVE,
    "follow":  VerbEffect.MOVE,
    # items (event only — no inventory ledger in v1)
    "take":    VerbEffect.ITEM,
    "drop":    VerbEffect.ITEM,
    "give":    VerbEffect.ITEM,
    # search
    "search":  VerbEffect.SEARCH,
}
```

**Step 2: write tests.**

```python
def test_whitelisted_verb_returns_effect():
    from llm.verse.avatar import VERB_TABLE, VerbEffect
    assert VERB_TABLE["whisper"] == VerbEffect.EVENT_ONLY
    assert VERB_TABLE["move"] == VerbEffect.MOVE

def test_unlisted_verb_not_in_table():
    from llm.verse.avatar import VERB_TABLE
    assert "teleport" not in VERB_TABLE
```

**Step 3: implement** (above).

**Step 4: run + commit** — `feat(verse): verb whitelist`.

---

### Task B2: `verse_act` handler

**Files:** same.

Behaviour:
- Inputs: `store: VerseStore`, `avatar_id: int`, `verb: str`, `target: str | None`, `details: str | None`.
- Always writes an `events` row with `source='avatar'`, `entity_ids=[avatar_id, target_id?]`.
- If verb is `MOVE` and target resolves to a place (or to another avatar's current place): update avatar's `location` attribute.
- If verb is `ITEM`: resolve target as existing item entity; record event linking; do NOT create new item entities.
- Off-list verbs: just write the event row, no side effects.
- Returns `ActResult(event_id, scene_shift_text)` — one short paragraph of "what happens next" for the model to narrate.

Tests cover each verb type with a real store. Commit: `feat(verse): verse_act handler`.

---

### Task B3: `verse_move`, `verse_look`, `verse_recall` handlers

**Files:** same.

Each is small:
- `verse_move(store, avatar_id, place_name)`: sets `location` attribute to the matching place. Errors if no such place.
- `verse_look(store, target=None)`: returns avatar's current location summary if target is None; else returns the entity's summary.
- `verse_recall(store, query)`: returns up to 5 recent events whose summary contains any token of `query` (case-insensitive substring match — RAG-lite, no embedding in v1).

Tests + commit: `feat(verse): verse_move / verse_look / verse_recall`.

---

### Task B4: system prompt builder + OOC escape detector

**Files:** same.

```python
def build_verse_system_prompt(
    store: VerseStore,
    avatar_id: int,
    instruct_text: str,
) -> str:
    """Compose the avatar's per-request system prompt."""
```

Returns a string with sections:

```
You are <avatar.name>. Persona: <instruct_text or "no persona set">.
Scene: <current location summary>.
Recent events involving you (last 5):
- <event 1>
- ...
Other avatars present: <names or "none">.
```

Plus an OOC escape detector:

```python
OOC_PREFIX = "(("
OOC_SUFFIX = "))"

def is_ooc(message: str) -> bool:
    s = message.strip()
    return s.startswith(OOC_PREFIX) and s.endswith(OOC_SUFFIX)
```

Tests + commit: `feat(verse): system prompt + OOC escape`.

---

## Phase C — Plugin wiring

### Task C1: registry — add new keys, remove forestNicks/spontaneous

**Files:**
- Modify: `plugins/llm/src/llm/config.py`
- Test: `plugins/llm/tests/test_config.py`

**Step 1: read** `config.py` around line 470 to find the `forestNicks` registration. Remove it.

**Step 2: search for spontaneous keys.** Grep `grep -n "spontaneous" plugins/llm/src/llm/config.py`. Remove `spontaneousEnabled`, `spontaneousChance`, `spontaneousCooldown`, `spontaneousSystemPrompt` registrations.

**Step 3: add new registry keys** per design doc §"Configuration":

| Key | Scope | Default |
|---|---|---|
| `verseEnabled` | per-channel bool | `False` |
| `verseEventRetentionDays` | per-channel int | `30` |
| `verseAutoApplyThreshold` | global float | `0.85` |
| `verseCrosspollAllowSend` | per-channel bool | `False` |
| `verseCrosspollAllowReceive` | per-channel bool | `False` |
| `verseCrosspollPerCycleLimit` | global int | `1` |
| `loomChannel` | global string | `""` |
| `loomModel` | global string | `gemini/gemini-flash-lite-latest` |
| `loomCycleInterval` | global int | `5` |
| `loomVerseCooldown` | global int | `20` |
| `loomBeatWindow` | global int | `90` |
| `loomTranscriptMaxLines` | global int | `40` |
| `loomTranscriptMaxChars` | global int | `8000` |

(Loom keys land here in PR 1 even though the orchestrator is PR 2 — that way operator config can be set up before PR 2 ships.)

**Step 4: update tests.** Any existing test that asserts on `forestNicks` or `spontaneous*` registry presence needs updating or removal. Add a test that the new keys exist with the documented defaults.

**Step 5: run** `uv run pytest plugins/llm/tests/test_config.py -v`. Fix until green.

**Step 6: commit** — `refactor(llm): swap forestNicks/spontaneous registry for verse/loom keys`.

---

### Task C2: capability registration

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (search for existing capability declarations — `llm.ask`, `llm.draw`, `llm.code` — and add `llm.verse`, `llm.verse.gm` alongside).

Test: a unit test that asserts the plugin declares both new capabilities.

Commit: `feat(verse): register llm.verse and llm.verse.gm capabilities`.

---

### Task C3: `@verseopt in/out` command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py` (add)

Behaviour:
- `@verseopt in` (in a channel where `verseEnabled=True`, capability `llm.verse`): call `store.opt_in_avatar(nick, account, instruct_text)`; reply with `result.scene_text` (split into lines if > `longReplyLineThreshold`). If `was_already_opted_in`, reply `"You are already opted in. Current scene: <scene>"`.
- `@verseopt out`: call `store.unlink_avatar`; reply `"Avatar retired. Use @verseopt in to rejoin."`.
- `@verseopt in` in a channel without `verseEnabled`: reply `"This channel doesn't have a verse. Ask the operator to set verseEnabled."`.
- Lacking capability: standard Limnoria capability denial.

Where to put the verse store: lazily-constructed `dict[channel, VerseStore]` on the plugin instance; `get_or_create_store(channel)` helper.

Tests for each branch using `pytest`-driven plugin instantiation (see how `test_plugin.py` does it for other commands).

Commit: `feat(verse): @verseopt in/out command`.

---

### Task C4: `@verse`, `@look`, `@who` commands

**Files:** same.

- `@verse` → returns avatar's current scene one-liner.
- `@look [target]` → no-target shows scene; with target shows entity description.
- `@who` → list of active avatars in the channel's verse with their current locations.

Tests + commit: `feat(verse): @verse / @look / @who commands`.

---

### Task C5: owner commands `@versedump`, `@versepurge`

**Files:** same.

- `@versedump #chan [--format=json|yaml]` — capability `llm.verse.gm`. JSON/YAML dump of all entities/attributes/relations/events. YAML is optional; JSON default.
- `@versepurge #chan` — first call: store a 60-second token, reply `"Confirm with @versepurge #chan <token>"`. Second call with matching token: delete the verse DB file. Token TTL enforced by timestamp comparison.

Tests cover happy path + token expiry. Commit: `feat(verse): owner commands versedump/versepurge`.

---

### Task C6: `@instruct` integration

**Files:** same.

When a user runs `@instruct <text>` in a verse-enabled channel where they have an avatar, atomically update both their stored `@instruct` text AND the avatar's `summary`. If they don't have an avatar (or aren't in a verse channel), `@instruct` keeps existing behaviour.

Remove `@avatar persona` if it was added — it isn't (per design v3 it doesn't exist).

Tests + commit: `feat(verse): @instruct double-writes avatar summary`.

---

### Task C7: route `@ask` through avatar shim

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (the `_is_forest_nick` site at line 2290 is going away; the dispatch site at line 3040 changes).
- Modify: `plugins/llm/src/llm/assistant.py` if needed (route profile selection).
- Tests: `plugins/llm/tests/test_assistant.py`, `plugins/llm/tests/test_plugin.py`.

Replace the old forest-nick check at line 3040 with verse routing:

```python
def _verse_route_for(self, channel, nick, account, message_text) -> VerseRoute | None:
    """Return a VerseRoute (avatar_id, system_prompt, tools, store) or None."""
    if not self.registryValue("verseEnabled", channel):
        return None
    if not _user_has_capability(self, "llm.verse", nick, account):
        return None  # capability fallthrough
    if is_ooc(message_text):
        return None
    store = self._get_or_create_verse_store(channel)
    avatar_id = store.find_avatar_by_account(account) or store.find_avatar_by_nick(nick)
    if avatar_id is None:
        return None
    instruct = self._get_user_instruct(nick, account)
    system_prompt = build_verse_system_prompt(store, avatar_id, instruct)
    tools = make_verse_tool_specs()
    return VerseRoute(avatar_id, system_prompt, tools, store)
```

When `_verse_route_for` returns a route:
- Bypass `assistantSystemPrompt` (channel persona).
- Bypass the line cap (set `forest`-style flag, mirroring whatever the old code did at line 3040 — borrow the bypass behaviour, drop the `forestNicks` lookup).
- Append the verse tools to the chat tool list.
- Run the completion with `assistantModel`.
- After reply rendered, dispatch any `verse_act`/`verse_move` tool calls.

Tests:
- `verseEnabled=False` → falls through to chat path.
- User without `llm.verse` → falls through to chat path (no error, no warning).
- OOC `((...))` → falls through to chat path.
- User with avatar → verse system prompt used, tools exposed, reply not line-capped.
- Tool call `verse_act` → mutation applied after reply.

Commit: `feat(verse): wire @ask through avatar shim`.

---

### Task C8: remove `_is_forest_nick` and the old forest path

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (remove `_is_forest_nick` at line 2290, remove `is_forest` usage at line 3040, remove any FOREST_SYSTEM_PROMPT or `forest` route profile).
- Modify: `plugins/llm/src/llm/assistant.py` (remove forest profile if present).
- Modify: `plugins/llm/tests/test_*.py` (delete or rewrite tests that asserted the old behavior).

Use `grep -nE "forest|FOREST" plugins/llm/src/llm/` to find every remaining reference. Remove. Keep the long-reply-line-cap-bypass behaviour, but drive it from `verseRoute is not None` rather than `is_forest`.

Run full test suite. Fix until green.

Commit: `refactor(llm): drop _is_forest_nick and old forest path`.

---

### Task C9: remove spontaneous mode

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (lines 473–590 — `_spontaneous_*` state; line 932 — registry check; any scheduled-event cancellation tied to spontaneous; the spontaneous evaluator method).
- Delete: `plugins/llm/tests/test_spontaneous.py`.
- Modify: `plugins/llm/tests/conftest.py` — remove spontaneous fixtures if any.

Use `grep -nE "spontaneous|Spontaneous|SPONTANEOUS" plugins/llm/src/llm/` to find every reference. Remove.

Run full test suite. Fix until green. Coverage may dip; if `--cov-fail-under=93` trips, that's a real signal — likely some real code branch lost coverage when its test was deleted. Investigate before lowering the threshold.

Commit: `refactor(llm): remove spontaneous mode (superseded by loom in PR 2)`.

---

## Phase D — RPG plugin removal

### Task D1: delete `plugins/rpg/` and update Makefile

**Files:**
- Delete: `plugins/rpg/` (entire directory).
- Modify: `Makefile` — remove `plugins/rpg/tests/` from `test:` and `test-all:` targets; remove `plugins/rpg/src/` from `typecheck:`.
- Modify: `pyproject.toml` if it lists `rpg` as a package or workspace member.
- Modify: `bot.conf` — remove any `plugins.RPG.*` lines (if present).
- Modify: any plugin loader / `botname.conf` that references RPG plugin loading.

```bash
git rm -r plugins/rpg/
```

**Update Makefile lines:**

```make
test:
	uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ -v -m "not slow" --cov --cov-report=term-missing --cov-fail-under=93

test-all:
	uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ -v --cov --cov-report=term-missing --cov-fail-under=93

typecheck:
	uv run ty check plugins/llm/src/ plugins/nickinmiddle/src/
```

Also check `scripts/` for any references to `plugins/rpg/`.

Run `make check`. Fix until green.

Commit: `chore: remove plugins/rpg/ (superseded by forest-verse)`.

---

## Phase E — Docs

### Task E1: collapse docs into `forest-verse.md`

**Files:**
- Delete: `docs/guide/operator/forest-mode.md`, `docs/guide/operator/spontaneous.md`.
- Create: `docs/guide/operator/forest-verse.md` (operator-facing: opt-in flow, capability gates, registry keys, `@verse*` commands, `@versedump`/`@versepurge`).
- Modify: `mkdocs.yml` — update nav: drop the two old pages, add the new one. Remove any `rpg` nav entry if present.
- Modify: `docs/guide/reference/commands.md` — remove rpg commands; add `@verse*` family.
- Modify: `docs/guide/index.md` — fix any cross-refs.

Keep the new doc to ~150 lines. Reference the design doc for deeper architecture details.

Commit: `docs: forest-verse operator guide; drop forest-mode and spontaneous docs`.

---

### Task E2: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`.

Add an `### Breaking` section under unreleased:

```markdown
### Breaking

- Removed the `plugins/rpg/` plugin and all its registry keys (`plugins.RPG.*`).
  Existing rpg state is **discarded, not migrated**. Users wanting structured
  storytelling should use the new forest-verse: see
  `docs/guide/operator/forest-verse.md`.
- Removed Forest mode (`plugins.LLM.forestNicks`). Existing rosters are
  discarded; users opt in fresh via `@verseopt in` in a channel where
  `verseEnabled=True`.
- Removed Spontaneous mode (`plugins.LLM.spontaneousEnabled` and friends).
  Replacement is the upcoming loom orchestrator (PR 2 of the forest-verse
  rollout); per-channel chatty-bot behaviour is no longer available in the
  interim.

### Added

- Forest-verse: per-channel SQLite entity graph + avatar shim. New commands
  `@verseopt`, `@verse`, `@look`, `@who`, plus owner commands `@versedump`,
  `@versepurge`. New capabilities `llm.verse` and `llm.verse.gm`.
```

Commit: `docs: changelog for forest-verse PR 1`.

---

## Phase F — Final verification

### Task F1: full check

Run:

```bash
make lint
make typecheck
make test
make syntax-check
```

All four green. Coverage ≥ 93%. Commit anything that floated up (e.g. ruff format fixups).

### Task F2: manual smoke

Not strictly required for PR merge, but run the bot locally if convenient:

```bash
uv run limnoria bot.conf
```

In an IRC client:
- Set `verseEnabled=True` for a channel.
- `@verseopt in` — should get a starter scene.
- `@verse` — should show the scene.
- `@ask hello` — should reply in-character (with avatar persona if `@instruct` set).
- `((@ask hello))` — should reply in normal chat mode.
- `@verseopt out` — should retire avatar.

If anything's off, fix and recommit (small commits).

### Task F3: open the PR

```bash
git push -u origin feat/forest-verse-pr1
gh pr create --title "feat: forest-verse PR 1 — store + avatar shim, drop rpg/forest/spontaneous" --body "$(cat <<'EOF'
## Summary

- Adds `plugins/llm/src/llm/verse/` with the entity-graph store and avatar shim.
- Wires `@ask` through the avatar shim for opted-in users in verse-enabled channels.
- Removes `plugins/rpg/` (data discarded, no migration).
- Removes `forestNicks` and spontaneous mode (data discarded).
- See `docs/plans/2026-05-07-forest-verse-design.md` v3 for the design.

## Test plan

- [ ] `make check` green on CI
- [ ] Coverage ≥ 93%
- [ ] Manual smoke: opt in, scene shows, `@ask` in-character, OOC escape works, opt out
- [ ] Owner commands: `@versedump`, `@versepurge` (token flow)
EOF
)"
```

---

## Wrap-up

Once merged, PR 2 follows: loom orchestrator + proposal queue. The verse will sit idle (mutated only by `verse_act`) until then.

If a task uncovers a design gap, *stop and ask* — don't paper over it. Note it in the PR description as follow-up.
