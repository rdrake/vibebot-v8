# Forest-verse PR 1 Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Land the verse store + avatar shim, remove the rpg plugin, and remove the old `forestNicks` and spontaneous-mode code paths in one PR. After this PR ships, opted-in users have a verse-aware `@ask` flow but the loom is not yet wired (PR 2).

**Architecture:**
- New package `plugins/llm/src/llm/verse/` with `store.py`, `avatar.py`, `schema.sql`.
- Verse store mirrors the thread-local + WAL pattern from `plugins/llm/src/llm/persistence.py:160–229`. The per-channel `threading.Lock` is a *new* addition on top of that pattern (the existing persistence layer doesn't have it; the verse store needs it because writes can race across IRC commands and the loom callback in PR 2).
- Avatar shim wraps `@ask` for opted-in users. Verb whitelist enumerated; off-list verbs land as event-only.
- `plugins/rpg/` is deleted in the same PR. `forestNicks` and spontaneous registry keys are deleted only in the same commit that removes their call sites.

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

**Scope guard for PR 1 (per design v3 Rollout):**

PR 1 ships **only**:
- `verse/` package: store + avatar + verb whitelist + tools.
- New commands: `@verseopt`, `@verse`, `@look`, `@who`, `@versedump`, `@versepurge`.
- New capabilities: `llm.verse`, `llm.verse.gm`.
- New registry keys: **only** `verseEnabled` (per-channel bool) and `verseEventRetentionDays` (per-channel int). Nothing else.
- Removal of `plugins/rpg/`, `forestNicks`, and spontaneous code paths.

PR 1 does **not** ship:
- `loomChannel`, `loomModel`, `loomCycleInterval`, `loomVerseCooldown`, `loomBeatWindow`, `loomTranscriptMaxLines`, `loomTranscriptMaxChars` — all reserved for PR 2.
- `verseAutoApplyThreshold`, `verseCrosspollAllowSend`, `verseCrosspollAllowReceive`, `verseCrosspollPerCycleLimit` — reserved for PR 2/3.
- The `proposals` table is created by PR 1's schema (so PR 2 doesn't need a migration), but no code path in PR 1 writes to it.

---

## Phase A — Verse store

### Task A1: Create the package skeleton

**Files:**
- Create: `plugins/llm/src/llm/verse/__init__.py` (one-line module docstring).
- Create: `plugins/llm/src/llm/verse/schema.sql` (placeholder; populated in A3).
- Create: `plugins/llm/src/llm/verse/store.py` (placeholder).
- Create: `plugins/llm/src/llm/verse/avatar.py` (placeholder).
- Create: `plugins/llm/src/llm/verse/tests/__init__.py` (empty).
- Create: `plugins/llm/src/llm/verse/tests/conftest.py`.

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

**Step 3: run** `make check` to confirm empty modules pass `ruff` and `ty` (they do; verified — `pyproject.toml`'s ruff/ty configs tolerate empty modules). If anything trips, fix tooling before adding feature code.

**Step 4: commit**

```bash
git add plugins/llm/src/llm/verse/
git commit -m "feat(verse): scaffold package skeleton"
```

---

### Task A2: db filename sanitization helper

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Test: `plugins/llm/src/llm/verse/tests/test_store.py` (new).

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

**Step 2: run** `uv run pytest plugins/llm/src/llm/verse/tests/test_store.py -v` — should fail with `ImportError`.

**Step 3: implement** in `store.py`:

```python
"""SQLite-backed verse store: entities, attributes, relations, events, proposals."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SAFE_RE = re.compile(r"[^a-z0-9_-]")


def db_path_for_channel(base_dir: Path, channel: str) -> Path:
    """Return the per-channel SQLite path under ``base_dir``."""
    lowered = channel.lower()
    safe = _SAFE_RE.sub("_", lowered)
    digest = hashlib.sha256(channel.encode("utf-8")).hexdigest()[:8]
    return base_dir / f"{safe}_{digest}.db"
```

**Step 4: run tests** — pass.

**Step 5: commit** — `feat(verse): db_path_for_channel sanitizer`.

---

### Task A3: schema.sql + connection / migration

**IMPORTANT pattern note:** SQLite's `executescript()` issues an implicit `COMMIT` before running the script, which **breaks** any surrounding `write_transaction` context. The existing pattern at `plugins/llm/src/llm/persistence.py:225–229` runs `executescript` directly off `_connect()`, *not* inside a write transaction. Mirror that pattern exactly.

**Files:**
- Modify: `plugins/llm/src/llm/verse/schema.sql`.
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`.

**Step 1: write `schema.sql`** with the full data model (entities, attributes, relations, events, avatar_link, proposals, schema_version). The proposals table is created here even though PR 1 has no writer for it — this avoids a migration in PR 2. Schema body is the same as v1 of this plan; reproduced here for completeness:

```sql
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
    entity_ids TEXT NOT NULL DEFAULT '[]',
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
    id          TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    cycle_id    TEXT NOT NULL,
    op          TEXT NOT NULL CHECK (op IN ('add_event','set_attribute','add_relation','add_entity')),
    payload     TEXT NOT NULL,
    confidence  REAL NOT NULL,
    provenance  TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
    reviewer    TEXT,
    reviewed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at);
```

(Partial index on `account IS NOT NULL` requires SQLite ≥ 3.8; project already targets 3.x stdlib so this is fine.)

**Step 2: write the failing test:**

```python
class TestVerseStoreInit:
    def test_creates_db_with_schema(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row is not None
            assert row[0] >= 1

    def test_idempotent_init(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#afnet")
        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            assert count == 1
```

**Step 3: implement** in `store.py`:

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
        # executescript() implicitly commits; do NOT wrap in write_transaction.
        # Mirrors plugins/llm/src/llm/persistence.py:225-229.
        conn = self._connect()
        conn.executescript(_SCHEMA_SQL)
        existing = conn.execute("SELECT version FROM schema_version").fetchone()
        if existing is None:
            with self.write_transaction() as wconn:
                wconn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
```

**Step 4: run tests** — pass.

**Step 5: commit** — `feat(verse): store init with schema + WAL + thread-local conn`.

---

### Task A4: entity CRUD

**Files:** as A3.

Tests for `add_entity`, `get_entity`, `find_entity_by_name` (case-insensitive), `set_status`, `list_entities_by_kind` (with status filter).

`Entity` as a `NamedTuple`. All writes through `write_transaction`. `find_entity_by_name` uses `LOWER(name) = LOWER(?)`.

Commit: `feat(verse): entity CRUD`.

---

### Task A5: attributes + relations CRUD

**Files:** as A3.

Tests:
- `set_attribute(eid, key, value)` upserts (use `INSERT … ON CONFLICT DO UPDATE`).
- `get_attribute(eid, key)` returns value or `None`.
- `list_attributes(eid)` returns dict.
- `add_relation(from_id, to_id, kind, note="")` returns rel id.
- `list_relations(from_id=None, to_id=None, kind=None)` filters.

Commit: `feat(verse): attributes + relations`.

---

### Task A6: events append + retrieval

**Files:** as A3.

Tests:
- `add_event(summary, entity_ids, source)` returns id, sets `ts` to current time.
- `recent_events(limit=10, exclude_sources=())` returns newest-first.
- `recent_events(exclude_sources=('crosspoll',))` skips crosspoll source (forward-prep for PR 3 but the API lands now).

Implement. JSON-encode `entity_ids`. Commit: `feat(verse): events append + retrieval`.

---

### Task A7: avatar_link CRUD + soft-delete avatar

**Files:** as A3.

Tests:
- `link_avatar(entity_id, nick, account=None)` upserts.
- `find_avatar_by_nick(nick)` returns entity_id or None (case-insensitive).
- `find_avatar_by_account(account)` returns entity_id or None.
- `unlink_avatar(entity_id)` removes link AND retires the entity (atomic, in one `write_transaction`).

Commit: `feat(verse): avatar_link CRUD`.

---

### Task A8: write-lock concurrency test

**Files:**
- Modify: `plugins/llm/src/llm/verse/tests/test_store.py`.

Real-threads test with `ThreadPoolExecutor(max_workers=8)` writing 50 entities; assert all unique IDs and all rows present.

Commit: `test(verse): write-lock concurrency`.

---

### Task A9: opt-in starter scene helper

**Files:** as A3.

```python
def opt_in_avatar(self, nick: str, account: str | None, instruct_text: str) -> AvatarOptInResult:
    """Atomically: ensure a starter place exists, create or revive the avatar
    entity, set its location attribute to the place, link the IRC user, and
    return a one-paragraph scene description."""
```

`AvatarOptInResult` = `NamedTuple(entity_id: int, place_name: str, scene_text: str, was_already_opted_in: bool)`.

**Behavior table:**

| State | Effect |
|---|---|
| No avatar exists for nick/account | Create new avatar entity, link, set location, return `was_already_opted_in=False`. |
| Active avatar exists for nick/account | Return existing avatar with `was_already_opted_in=True`. |
| Retired avatar exists for nick/account | Reactivate (`status='active'`), update link, return `was_already_opted_in=False`. |
| No active place exists | Create a default `place` entity named `"The Clearing"`, summary `"A quiet woodland clearing where new stories begin."`. |
| At least one active place exists | New avatar lands at the active place with the most recent `events` row referencing it; tie-break on most recent `updated_at`. |

**Concurrency test (real threads):** two threads call `opt_in_avatar` for distinct nicks at the same time on a brand-new store; assert exactly one place entity gets created and both avatars share its id.

**Scene text format:** `"You step into <place name>. <place summary> Use @look to inspect things or @ask … to act."` (Avoid mentioning `verse_act` — that's a model-callable tool, not a user-typeable command.)

Commit: `feat(verse): opt-in starter scene`.

---

## Phase B — Avatar shim

### Task B1: verb whitelist + dispatch

**Files:**
- Modify: `plugins/llm/src/llm/verse/avatar.py`.
- Test: `plugins/llm/src/llm/verse/tests/test_avatar.py` (new).

Whitelist collapses `SEARCH` into `EVENT_ONLY` (no separate flag — the design doc's "re-render scene" note is a model-side hint, not a store-side flag).

```python
class VerbEffect(Enum):
    EVENT_ONLY = "event_only"
    MOVE = "move"
    ITEM = "item"


VERB_TABLE: dict[str, VerbEffect] = {
    "whisper": VerbEffect.EVENT_ONLY,
    "speak":   VerbEffect.EVENT_ONLY,
    "listen":  VerbEffect.EVENT_ONLY,
    "examine": VerbEffect.EVENT_ONLY,
    "wait":    VerbEffect.EVENT_ONLY,
    "signal":  VerbEffect.EVENT_ONLY,
    "gesture": VerbEffect.EVENT_ONLY,
    "search":  VerbEffect.EVENT_ONLY,
    "move":    VerbEffect.MOVE,
    "flee":    VerbEffect.MOVE,
    "follow":  VerbEffect.MOVE,
    "take":    VerbEffect.ITEM,
    "drop":    VerbEffect.ITEM,
    "give":    VerbEffect.ITEM,
}
```

Tests assert each verb maps correctly, off-list verbs are absent. Commit: `feat(verse): verb whitelist`.

---

### Task B2: `verse_act` handler

**Files:** as B1.

Behaviour:
- Inputs: `store: VerseStore`, `avatar_id: int`, `verb: str`, `target: str | None`, `details: str | None`.
- Always writes an `events` row with `source='avatar'`, `entity_ids=[avatar_id, target_id?]`.
- `MOVE` verb: target must resolve to a `place` (or to another avatar's current `place`). On success, update avatar's `location` attribute. **On target-not-found**: write the event row anyway with no side effect, return `ActResult(event_id, scene_shift_text="You can't find that place.")`.
- `ITEM` verb: resolve target as existing item entity. On not-found: event written, no side effect, scene_shift acknowledges absence.
- Off-list verbs: event row only, no side effects, generic acknowledgement.
- Returns `ActResult(event_id: int, scene_shift_text: str)`.

**Failure-path tests (required):**
- `verse_act(store, avatar_id, "move", target="Nowhere")` — target doesn't exist → event row written, no `location` attribute change, scene_shift indicates failure.
- `verse_act(store, avatar_id, "give", target="Phantom Sword")` — item doesn't exist → event row written, no relation added.
- `verse_act(store, avatar_id, "teleport", target="moon")` — off-list verb → event row written, no side effect.
- `verse_act` on a retired avatar → `ValueError("avatar retired")` raised before any write.

Commit: `feat(verse): verse_act handler with failure paths`.

---

### Task B3: `verse_move`, `verse_look`, `verse_recall` handlers

**Files:** as B1.

- `verse_move(store, avatar_id, place_name)`: sets `location` attribute to the matching place. Raises `ValueError("no such place")` if no match.
- `verse_look(store, target=None)`: returns avatar's current location summary if target is None; else returns the entity's summary; `None` if not found.
- `verse_recall(store, query)`: returns up to 5 recent events whose summary contains any token of `query` (case-insensitive substring match).

**Note:** the design doc describes `verse_recall` as "RAG over events." PR 1 ships substring matching; embedding-based retrieval is a documented follow-up (`docs/plans/2026-05-07-forest-verse-design.md` §"Open follow-ups" — add a bullet for it as part of E2).

Tests + commit: `feat(verse): verse_move / verse_look / verse_recall`.

---

### Task B4: system prompt builder + OOC escape detector

**Files:** as B1.

```python
def build_verse_system_prompt(
    store: VerseStore,
    avatar_id: int,
    instruct_text: str,
) -> str: ...
```

Composes sections: avatar name, persona (`instruct_text` or `"no persona set"`), current scene, last 5 events involving the avatar, other avatars present.

**OOC detector:**

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

**Critical ordering rule for Phase C:** `forestNicks` and `spontaneous*` registry keys are *not* removed until C9 (the same task that removes their call sites). C1 only adds new keys. This avoids the startup crash from removing a registry key while its caller still tries to read it.

### Task C1: registry — add new keys ONLY

**Files:**
- Modify: `plugins/llm/src/llm/config.py`.
- Test: `plugins/llm/tests/test_config.py`.

**Step 1: add only these two registry keys:**

| Key | Scope | Default |
|---|---|---|
| `verseEnabled` | per-channel bool | `False` |
| `verseEventRetentionDays` | per-channel int | `30` |

Loom keys, crosspoll keys, and the auto-apply threshold do **not** ship in PR 1.

**Step 2: add tests** asserting both new keys exist with documented defaults.

**Step 3: do NOT remove** any existing keys yet. `forestNicks` and `spontaneous*` stay until C9.

**Step 4: commit** — `feat(verse): add verseEnabled and verseEventRetentionDays registry`.

---

### Task C2: capability registration

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (search for existing capability declarations alongside `llm.ask`, `llm.draw`, `llm.code` and add `llm.verse`, `llm.verse.gm`).
- Test: `plugins/llm/tests/test_plugin.py` — assert both new capabilities are declared.

Commit: `feat(verse): register llm.verse and llm.verse.gm capabilities`.

---

### Task C3: `@verseopt in/out` command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Test: `plugins/llm/tests/test_plugin.py`.

Behaviour:
- `@verseopt in` (channel with `verseEnabled=True`, capability `llm.verse`): call `store.opt_in_avatar(nick, account, instruct_text)`; reply with `result.scene_text`. If `was_already_opted_in`, prefix `"You are already opted in. "`.
- `@verseopt out`: call `store.unlink_avatar`; reply `"Avatar retired. Use @verseopt in to rejoin."`.
- `@verseopt in` in a channel without `verseEnabled`: reply `"This channel doesn't have a verse. Ask the operator to set verseEnabled."`.
- Lacking capability: standard Limnoria capability denial.

Where to put the verse store: lazily-constructed `dict[channel, VerseStore]` on the plugin instance; `_get_or_create_verse_store(channel)` helper. `data/verse/` directory used as `base_dir`.

Tests for each branch.

Commit: `feat(verse): @verseopt in/out command`.

---

### Task C4: `@verse`, `@look`, `@who` commands

**Files:** as C3.

- `@verse` → returns avatar's current scene one-liner.
- `@look [target]` → no-target shows scene; with target shows entity description; `"Nothing matches."` if not found.
- `@who` → list of active avatars in the channel's verse with their current locations; `"Nobody is opted in here yet."` if empty.

Commit: `feat(verse): @verse / @look / @who commands`.

---

### Task C5: owner commands `@versedump`, `@versepurge`

**Files:** as C3.

**`@versedump #chan [--format=json|yaml]`** — capability `llm.verse.gm`. JSON default (yaml optional via `pyyaml` if it's already a dependency; if not, JSON-only is fine).

**`@versepurge #chan`** — token storage spec:

- Tokens live in `self._versepurge_tokens: dict[str, tuple[str, float]]` on the plugin instance — keyed by `channel`, value is `(token, expires_at)`. **In-memory only**; resets on plugin reload or bot restart. Documented as such in the operator guide.
- First call: generates a 6-character token (`secrets.token_hex(3)`), stores `(token, time.time() + 60.0)`, replies `"Confirm with @versepurge <chan> <token> within 60s."`.
- Second call with `<chan> <token>`: if a current token exists for that channel, hasn't expired, and matches → close the verse store, delete the DB file, drop the token from the dict. Reply `"Verse for <chan> purged."`.
- If the second-call token doesn't match (or is expired): reply `"Token expired or invalid. Run @versepurge <chan> again to start over."` and clear any expired entry.
- If a first call comes in while an unexpired token already exists for the same channel: replace it (issue a fresh token, expire the old one). Replies note that the old token is now invalid.

**Tests:**
- Happy path (issue token, confirm within 60s, file deleted).
- Token expiry (issue, sleep past 60s, confirm fails — use a freezable clock fixture or pass an injectable `now()`).
- Mismatched token rejected.
- Re-issuing token within window invalidates the old one.
- Reload safety: clear the dict on plugin construction (already implicit since the plugin instance is fresh).

Commit: `feat(verse): owner commands versedump/versepurge with token spec`.

---

### Task C6: `@instruct` integration

**Files:** as C3, plus wherever `@instruct` is currently implemented (grep `def _instruct` or `class Instruct`).

When a user runs `@instruct <text>` in a verse-enabled channel where they have an active avatar, atomically update both their stored `@instruct` text AND the avatar's `summary` (single `write_transaction` covering both writes; if the verse write fails, the `@instruct` update should also roll back to keep them in sync — implement by writing instruct second after verse update succeeds, or use a sentinel on the way out). If they don't have an avatar (or aren't in a verse channel), `@instruct` keeps existing behaviour.

Tests + commit: `feat(verse): @instruct double-writes avatar summary`.

---

### Task C7a: introduce `_verse_route_for` returning None always

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Test: `plugins/llm/tests/test_plugin.py`.

Add the helper that always returns `None` for now. Wire it into the `@ask` path **without changing any existing behavior** — every existing test must still pass.

```python
def _verse_route_for(self, channel, nick, account, message_text) -> VerseRoute | None:
    return None  # Wired in C7b/c/d.
```

Step: write test asserting that with `_verse_route_for` always returning None, the existing chat path is unchanged for an opted-in user. Then add the dispatch hook that consults `_verse_route_for` — if it returns None, call the existing chat path.

Commit: `refactor(llm): hook _verse_route_for stub into @ask dispatch`.

---

### Task C7b: gating logic

**Files:** as C7a.

Replace the stub body with the real gating:

```python
def _verse_route_for(self, channel, nick, account, message_text) -> VerseRoute | None:
    if not self.registryValue("verseEnabled", channel):
        return None
    if not _user_has_capability(self, "llm.verse", nick, account):
        return None  # capability fallthrough
    if is_ooc(message_text):
        return None
    # Body in C7c.
    return None
```

Tests:
- `verseEnabled=False` → returns `None`, `@ask` runs chat path.
- User without `llm.verse` → returns `None`, no error.
- OOC `((...))` → returns `None`.
- All three preconditions satisfied → still returns `None` (until C7c).

Commit: `feat(verse): _verse_route_for gating logic`.

---

### Task C7c: system prompt + tool list assembly

**Files:** as C7a, plus `plugins/llm/src/llm/assistant.py` if a new profile is needed.

After the gating in C7b passes, look up the avatar:

```python
store = self._get_or_create_verse_store(channel)
avatar_id = store.find_avatar_by_account(account) or store.find_avatar_by_nick(nick)
if avatar_id is None:
    return None  # User opted into the channel but isn't in the verse → chat path.
instruct = self._get_user_instruct(nick, account)
system_prompt = build_verse_system_prompt(store, avatar_id, instruct)
tools = make_verse_tool_specs()
return VerseRoute(avatar_id, system_prompt, tools, store)
```

When a route is returned, the call site (in C7a's hook) must:
- Bypass `assistantSystemPrompt` (channel persona) — set system_prompt to verse_route's value.
- Set route profile to bypass the line cap (use the same plumbing the old `forest` profile used; this profile is renamed/redirected, not re-invented).
- Append the verse tools to the chat tool list.
- Run completion with `assistantModel`.

Tests:
- User with avatar → verse system prompt used.
- **`assistantSystemPrompt` is bypassed** — set a unique sentinel string in `assistantSystemPrompt`, run `@ask`, assert sentinel is *absent* from the prompt the LLM saw.
- Long reply not line-capped (mirror behavior of old forest profile in tests).
- Tools list includes verse tools.

Commit: `feat(verse): _verse_route_for system prompt + tools`.

---

### Task C7d: post-reply tool-call dispatch + failure handling

**Files:** as C7a.

After the model reply is rendered, dispatch any `verse_act`/`verse_move`/`verse_look`/`verse_recall` tool calls.

**Order of operations on tool-call errors:**
1. Reply text is rendered to the user *first*, regardless of mutation outcome.
2. Each tool call is applied in sequence.
3. If a tool call raises (validation error, `ValueError("avatar retired")`, target-not-found etc.), log at WARNING with the avatar id and verb, *do not* halt subsequent tool calls — continue applying the rest. Failed mutations leave no event row.
4. **Race with `@versepurge`:** if the verse store / DB file is gone by the time a tool call lands, log at WARNING and drop the call silently. The user's `@ask` reply still landed; the next `@ask` will return a "verse not initialized" error from `_get_or_create_verse_store` (which auto-recreates).

Tests:
- Successful `verse_act` → event row written.
- `verse_act` with bad target → reply still sent, no event, WARNING logged (use `caplog`).
- `verse_act` on a verse whose DB was just deleted → reply still sent, no exception bubbles up.
- Multiple tool calls in one reply, one fails mid-sequence → others still apply.

Commit: `feat(verse): post-reply tool-call dispatch with failure handling`.

---

### Task C8: drop `_is_forest_nick` and the full forest path

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — remove `_is_forest_nick` (currently `:2290`, definition); remove `is_forest` usage at the dispatch site (currently `:3040`); search for any other `forest` references.
- Modify: `plugins/llm/src/llm/assistant.py` — remove `PROFILE_FOREST` (`:30`), the `_IRC_OUTPUT_FORMAT_FOREST` block (`:148–170`), `FOREST_SYSTEM_PROMPT`, and every `PROFILE_FOREST` membership in `visible_in` sets (`:730, :758, :763, :768, :774`).
- Modify: `plugins/llm/src/llm/service.py` — remove the `PROFILE_FOREST` import (`:38`), the `FOREST_SYSTEM_PROMPT` reference (`:3083`), the profile→prompt mapping entry (`:3117`), and the `forest`-related comment in the token-cap routing (`:3162–3163`).
- Modify: `plugins/llm/tests/test_plugin.py` — drop or rewrite tests asserting old forest behavior. Search hits: around `:4001–4104` (per reviewer; verify with grep).
- Modify: `plugins/llm/tests/test_assistant.py` — drop or rewrite tests at `:1003–1034, 2364–2423, 2528–2538`.
- Modify: `plugins/llm/tests/test_service.py` — drop or rewrite tests at `:5380–5397`.
- Modify: `plugins/llm/tests/conftest.py` — drop the forest fixture at `:332` (verify with grep).

**Process:**
1. `grep -rnE "is_forest|forest|FOREST|PROFILE_FOREST" plugins/llm/src/ plugins/llm/tests/` — produce the actual list. Do not trust the line numbers above blindly; verify with grep in the worktree.
2. Remove definitions first.
3. Compile errors will identify every remaining call site; fix them by either deleting the call or routing it through `_verse_route_for`.
4. Replace the line-cap-bypass plumbing the old forest profile used: the verse route should set the same flag that disabled the line cap. Do not invent a new mechanism; reuse the existing one.
5. Run `make check` until green.

Commit: `refactor(llm): drop _is_forest_nick and PROFILE_FOREST plumbing`.

---

### Task C9: remove spontaneous mode AND the now-orphaned registry keys

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — remove `_spontaneous_*` state and methods (`:473–590`), the registry check (`:932`), and any scheduled-event cancellation tied to spontaneous.
- Modify: `plugins/llm/src/llm/config.py` — remove `forestNicks` (`:470`), `spontaneousEnabled` (`:326`), `spontaneousChance` (`:334`), `spontaneousCooldown` (`:341`), `spontaneousSystemPrompt` (`:346`). Also clean up references in module docstrings at `:131, :206, :721`.
- Delete: `plugins/llm/tests/test_spontaneous.py`.
- Modify: `plugins/llm/tests/conftest.py` — remove spontaneous fixtures at `:319–323, :445–448` (verify with grep).
- Modify: `plugins/llm/tests/test_plugin.py` — remove spontaneous tests at `:965–967, :1553–1577` (verify with grep).
- Modify: `README.md` — remove the spontaneous-mode block at `:184–192` (and the rpg block at `:303` if not already done in D1).

**Process:**
1. `grep -rnE "spontaneous|Spontaneous|SPONTANEOUS|forestNicks" plugins/llm/src/ plugins/llm/tests/ README.md` — produce the actual list.
2. Removal order: tests first (they depend on the code), then code, then registry. The registry must be removed *last* in this task to avoid temporarily-orphaned references.
3. Run `make check`. Coverage may dip — investigate before lowering the threshold. The cause is almost always a real loss; either restore a representative test in `test_plugin.py` against the new verse path, or accept that the floor needs a small targeted bump (don't lower the global gate).

**Coverage contingency:** if `make test` trips `--cov-fail-under=93` after this task and the cause is genuinely "verse code isn't yet exercised because some C7 branches are uncovered," add focused tests in `test_plugin.py` that hit those branches. Do **not** lower the threshold.

Commit: `refactor(llm): remove spontaneous mode and orphaned registry keys`.

---

## Phase D — RPG plugin removal

### Task D1: delete `plugins/rpg/` and update every reference

**Files to update (verified in the current worktree):**

| File | Change |
|---|---|
| `plugins/rpg/` | `git rm -r` |
| `pyproject.toml:9` | Remove `"rpg",` from `dependencies` |
| `pyproject.toml:15` | Remove `plugins/rpg` from `members = [...]` |
| `pyproject.toml:20` | Remove `rpg = { workspace = true }` |
| `pyproject.toml:47` | Remove `plugins/rpg/tests` from `testpaths` |
| `pyproject.toml:73` | Remove `plugins/rpg/src` from coverage `source` |
| `Makefile` | Remove `plugins/rpg/tests/` from `test:` and `test-all:`; remove `plugins/rpg/src/` from `typecheck:` |
| `Dockerfile:10` | Remove `COPY plugins/rpg/pyproject.toml plugins/rpg/` line |
| `.pre-commit-config.yaml:27` | Remove `plugins/rpg/src/` from the ty hook entry |
| `.github/dependabot.yml:20–22` | Remove the `rpg` dependency-name entry from the ignore list |
| `mkdocs.yml:56` | Remove `- RPG: plugins/rpg.md` from nav |
| `docs/guide/plugins/rpg.md` | `git rm` |
| `docs/guide/plugins/index.md:9–14` | Remove the rpg paragraph (verify with grep) |
| `AGENTS.md:11, 77, 108` | Remove the three rpg lines (verify with grep) |
| `README.md:303` | Remove the `plugins/rpg/` directory tree line |
| `bot.conf` | Remove any `plugins.RPG.*` lines and any `load RPG` line (use `grep -n "RPG\|rpg" bot.conf`); leave `bot.conf.bak` alone (it's a backup of an older state) |
| `uv.lock` | Will be regenerated on `uv sync`; commit the regenerated lockfile |

**Process:**
1. `grep -rln "plugins/rpg\|^rpg\| rpg\b\|RPG" --include="*.toml" --include="*.yml" --include="*.yaml" --include="*.md" --include="Dockerfile" --include="Makefile" --include="*.conf"` to verify the list above is exhaustive.
2. Apply the removals.
3. `uv sync` to regenerate the lockfile.
4. `make check` until green. If `make ci` in CI uses different paths, double-check those too.

Commit: `chore: remove plugins/rpg/ (superseded by forest-verse)`.

---

## Phase E — Docs

### Task E1: collapse docs into `forest-verse.md`

**Process:**
1. `grep -rln "spontaneous\|forestNicks\|forest-mode\|forest mode" docs/` — produce the list. Confirmed candidates as of the current worktree:
   - `docs/guide/operator/forest-mode.md` (delete)
   - `docs/guide/operator/spontaneous.md` (delete)
   - `docs/guide/operator/configuration.md:23, 40, 62, 78` (update)
   - `docs/guide/operator/tuning-monitoring.md:9, 36, 60–63, 224, 314–317` (update)
   - `docs/guide/operator/memory-promotion.md:80, 91` (update)
   - `docs/guide/index.md:28–29` (update — bullets for spontaneous + forest-mode)
   - `docs/guide/reference/commands.md` (drop rpg + old forest commands; add `@verse*` family)
   - `mkdocs.yml` nav (already touched in D1, but confirm `forest-mode.md`/`spontaneous.md` entries are gone and `forest-verse.md` is added).
2. Create `docs/guide/operator/forest-verse.md` (~150 lines): opt-in flow, capability gates, the two new registry keys (verseEnabled, verseEventRetentionDays), `@verse*` commands, `@versedump`/`@versepurge` token semantics. Reference the design doc for architecture.
3. Update each doc above to drop dead references and point to the new page.

Commit: `docs: forest-verse operator guide; drop forest-mode and spontaneous docs`.

---

### Task E2: CHANGELOG + design-doc follow-up

**Files:**
- Modify: `CHANGELOG.md`.
- Modify: `docs/plans/2026-05-07-forest-verse-design.md` — add a follow-up bullet under §"Open follow-ups": *"Embedding-based `verse_recall`. PR 1 ships substring matching."*

**CHANGELOG entry** under unreleased:

```markdown
### Breaking

- Removed `plugins/rpg/` and all its registry keys. Existing rpg state is
  **discarded, not migrated**. See `docs/guide/operator/forest-verse.md`.
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

```bash
make lint
make typecheck
make format-check
make syntax-check
make test
```

All green. Coverage ≥ 93%. Commit anything that floated up (e.g. ruff format fixups).

### Task F2: manual smoke

In an IRC client (after `uv run limnoria bot.conf`):
- Set `verseEnabled=True` for a channel.
- `@verseopt in` — get starter scene.
- `@verse` — shows scene.
- `@instruct You are a curious traveller` — sets persona AND avatar summary.
- `@ask hello` — replies in-character with avatar persona; channel `assistantSystemPrompt` not visible in tone.
- `@ask` as a user without `llm.verse` capability — replies as normal chat path (no error).
- `((@ask hello))` — replies in normal chat mode, OOC bypass works.
- `@look the clearing` — entity description.
- `@who` — current avatar listed.
- `@versepurge #chan` — token issued; `@versepurge #chan <wrong>` rejected; `@versepurge #chan <right>` purges; new `@verseopt in` recreates.
- `@verseopt out` — retires avatar.

### Task F3: open the PR

```bash
git push -u origin feat/forest-verse-pr1
gh pr create --title "feat: forest-verse PR 1 — store + avatar shim, drop rpg/forest/spontaneous" --body "$(cat <<'EOF'
## Summary

- Adds `plugins/llm/src/llm/verse/` with the entity-graph store and avatar shim.
- Wires `@ask` through the avatar shim for opted-in users in verse-enabled channels (capability fallthrough + OOC `((...))` escape).
- New commands: `@verseopt`, `@verse`, `@look`, `@who`, `@versedump`, `@versepurge`.
- New capabilities: `llm.verse`, `llm.verse.gm`.
- Removes `plugins/rpg/` (data discarded, no migration).
- Removes `forestNicks` and spontaneous mode (data discarded).
- See `docs/plans/2026-05-07-forest-verse-design.md` v3 for the design.

## Test plan

- [ ] `make check` green on CI
- [ ] Coverage ≥ 93%
- [ ] Manual smoke (see plan §F2)
EOF
)"
```

---

## Wrap-up

Once merged, PR 2 follows: loom orchestrator + proposal queue + the deferred `loom*` / crosspoll registry keys / `verseAutoApplyThreshold`. The verse will sit idle (mutated only by `verse_act`) until then.

If a task uncovers a design gap, *stop and ask* — don't paper over it. Note it in the PR description as follow-up.
