# Forest-verse PR 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Round out the forest-verse with crosspollination, retention compaction, and entity-id grounding so verses can borrow seeds from each other (opt-in both ends), the events log stays bounded by `verseEventRetentionDays`, and the digest model stops inventing entity ids.

**Architecture:**
- One new module `plugins/llm/src/llm/verse/crosspoll_store.py` owns a single shared SQLite file (`data/verse/_crosspoll.db`) that holds emitted crosspoll seeds plus per-receiver consumption rows. The per-channel verse DBs are untouched by crosspoll wiring.
- One new module `plugins/llm/src/llm/verse/compaction.py` owns the daily compaction job — a pure function over a `VerseStore` plus a `LoomModelClient` that summarises events older than `verseEventRetentionDays` into a single lore-digest event (source `'loom'`) and deletes the originals atomically.
- Existing `verse/loom.py` grows three small surgical changes: entity-id grounding in `build_verse_stable_block`, a new `'crosspoll_seed'` op in `parse_digest` / `apply_or_queue`, and a `consume_one_seed_for(channel)` helper invoked at the start of `Loom.tick` for receivers.
- Existing `verse/store.py` grows two retention helpers (`events_older_than(days)`, `replace_events_with_lore_digest(...)`) — both atomic via `write_transaction`.
- `plugin.py` registers the daily compaction timer alongside the loom timer, wires a `CrosspollStore` singleton into the `Loom` bridge, and adds one new owner command (`@versecompact #chan`).

**Tech stack:**
- Python 3.13+ via `uv`, `pytest`, `ruff`, `ty`.
- Limnoria 2025+ (`supybot`); `supybot.schedule.addEvent` for the daily compaction timer; same single-threaded scheduler invariants as the loom timer.
- `litellm.completion` for the compaction summarisation call, reusing the existing `LiteLLMLoomClient` (tagged `loom:compact`).
- Real SQLite in `tmp_path` for tests. **No mocks for the DB.** Recorded transcript JSON fixtures already in `plugins/llm/tests/verse/fixtures/` for digest tests; live model calls behind `VIBEBOT_TEST_LIVE=1`.

**Reference design:** `docs/plans/2026-05-07-forest-verse-design.md` (esp. §"Loom orchestrator", §"Cross-pollination", §"Configuration", §"Tests"). PR 1 is `docs/plans/2026-05-07-forest-verse-pr1.md`; PR 2 is `docs/plans/2026-05-07-forest-verse-pr2.md` and is the format model for this document.

**Working directory:** `.worktrees/forest-verse-pr3` (branch `feat/forest-verse-pr3`). Create with `superpowers:using-git-worktrees`. Push directly to `main` is fine; CI + Docker build are separate workflows — wait for both before restarting prod.

---

## Revisions

- **2026-05-08 v1** — initial draft after brainstorm.
- **2026-05-08 v4** — second adversarial pass + second friendly pass.
  1. **D6-pre promoted from a stub paragraph to a full TDD task.**
     Confirmed `VerseStore.add_proposal` (`store.py:556-600`) does NOT
     accept a `proposal_id` kwarg today; it always generates a uuid.
     The kwarg is unconditionally needed for the receiver consume hook
     to use the same id in both the consumption row and the local
     proposal. D6-pre is now a full red-green-commit task.
  2. **`claim_seed_for` rollback semantics tightened.** The
     `IntegrityError` catch is moved *outside* the `write_transaction`
     context manager so the failed claim triggers an explicit
     ROLLBACK. Caller-visible behaviour unchanged (still returns None
     on lost-race), but the transaction state machine is now correct
     by inspection rather than by relying on SQLite's per-statement
     rollback under a no-op COMMIT.
  3. **F3-pre tests extended to assert proposal status flip.** The
     v3 tests only checked `events.source`; v4 adds asserts for
     `proposals.status='approved'`, `proposals.reviewer`, and
     `proposals.reviewed_at` so the apply-and-mark contract is
     verified end-to-end.
  4. **Compaction drain rate documented.** The operator guide's
     "Retention compaction" section now includes the math:
     `_MAX_EVENTS_PER_PASS=200`, daily cadence → 200 events/day drain
     rate; a 10k backlog converges in ~50 days; verses producing
     >200 events/day past their retention window will not converge
     under this policy. Both knobs (`verseEventRetentionDays`,
     `verseCompactionDailyAt`) interact with the cap; operators can
     run `@versecompact` repeatedly to drain a backlog manually.
- **2026-05-08 v3** — applied feedback from the adversarial review (Codex,
  hostile pass).
  Significant fixes:
  1. **Race**: replaced `next_unconsumed_for` + `mark_consumed` consume
     flow with a single-TX atomic `claim_seed_for(dest_channel, *,
     proposal_id)` so two concurrent receivers cannot insert duplicate
     pending proposals for one seed. The split methods remain available
     for diagnostics only.
  2. **Compaction data loss**: `delete_ids` in `compact_verse` previously
     deleted ALL old events while the prompt only summarised the last
     200. Fixed: per-pass compaction handles the oldest 200 events,
     summarises those, deletes only those. Long backlogs drain over
     several daily runs.
  3. **Compaction prompt safety**: per-event summary text is now capped
     at 240 chars before joining; total bullet block capped at 16k chars.
  4. **Lore-digest entity-id cap**: the 32-id truncation now logs at
     INFO when it actually truncates, so operators can spot
     entity-heavy verses where the cap costs grounding.
  5. **Schema-invariant regression test**: explicit test that no
     `op='crosspoll_seed'` row ever enters `VerseStore.proposals`,
     guarding against future code paths that might write one.
  Nits: `events_older_than` normalises `entity_ids` via `int(...)` like
  `recent_events`; `@wrap` capability syntax aligned with existing
  PR 1/2 verse commands (`("checkCapability", "llm.verse.gm")` form);
  `@versecompact` capability test asserts on `irc.error` reply rather
  than `pytest.raises(Exception)`; the `VerseSnapshot` grep includes
  `plugins/llm/tests/verse/test_loom_integration.py`; adds a regression
  test that the bridge `snapshot()` carries `exclude_sources=("crosspoll",)`.
- **2026-05-08 v2** — applied feedback from the friendly structural review.
  Required-change fixes:
  1. Auto-reject path for `crosspoll_seed` invalid-refs no longer writes a
     `proposals` row — that would have failed the existing
     `proposals.op` CHECK constraint, which is unchanged in PR 3.
  2. C1 bridge snippet now uses the real `list_entities_by_kind(kind,
     status="active")` API; the previously-cited `list_active_entities`
     does not exist on `VerseStore`.
  3. New mandatory **Task F3-pre** threads `event_source` through
     `apply_proposal_and_mark`. `apply_proposal_and_mark` currently
     hard-codes `source="loom"` (`store.py:717`); the receiver flow
     **needs** `source='crosspoll'`. F3a's "conditional" status was
     illusory; promoting to a real task and reordering F3 to depend on it.
  4. D5's `TestDigestPhaseRoutesCrosspoll` and D6's
     `TestLoomTickConsumesSeed` test bodies now have full bridge/client
     stubs; the v1 `...` placeholders are gone.
  5. D5's `_digest_phase` snippet now passes `already_emitted=cycle.emitted_seeds`
     each loop iteration after the previous outcome's increment, so the
     per-cycle cap actually holds when a digest produces N>limit seeds.
  Plus nits: `_add_event_at` moved off the production class into a
  conftest helper; `LoomBridge` Protocol test uses `inspect.getmembers`;
  `register_daily_timer` threads `now` through consistently.

---

## Scope guard for PR 3

PR 3 ships **only**:

- `verse/crosspoll_store.py` (new): shared `_crosspoll.db` with `crosspoll_seeds` and `crosspoll_consumptions` tables; `CrosspollStore.enqueue_seed`, `claim_seed_for(dest_channel, *, proposal_id)` (atomic single-TX read-and-mark), and read-only diagnostic helpers `next_unconsumed_for(channel)` / `pending_count_for(channel)`. Real SQLite, thread-local conn + WAL.
- `verse/compaction.py` (new): pure helper `compact_verse(store, *, retention_days, min_keep_events, model, client, log_usage)` plus a thin scheduling driver `register_daily_timer(...)` / `cancel_daily_timer(...)`.
- `verse/store.py` (modify): `events_older_than(cutoff_ts) -> list[Event]`, `replace_events_with_lore_digest(*, before_ts, summary, entity_ids, ts) -> int` — both atomic.
- `verse/loom.py` (modify):
  - `build_verse_stable_block`: emit `(id=N)` suffix on every entity line.
  - `LOOM_STATIC_PREFIX`: documents the new `crosspoll_seed` op + tells the model that entity ids are surfaced inline.
  - `parse_digest`: accepts op `crosspoll_seed` (payload schema mirrors `add_event`).
  - `apply_or_queue`: routes `crosspoll_seed` to a new `_route_crosspoll_seed(...)` helper (gated by `verseCrosspollAllowSend` via the cfg passed in) which calls `CrosspollStore.enqueue_seed(...)` and writes one local audit `event` row (`source='loom'`). Per-cycle emit cap honoured here.
  - `Loom.tick`: after picking the focus verse and inside the same lock, if `verseCrosspollAllowReceive` is true for that verse, pull one seed via the bridge and insert a pending `add_event` proposal in the receiver's local proposals table.
  - `LoomBridge` Protocol grows `crosspoll_store(self) -> CrosspollStore | None`, `verse_allow_send(channel) -> bool`, `verse_allow_receive(channel) -> bool`.
- `config.py` (modify): five new registry keys (table below).
- `plugin.py` (modify): `_get_or_create_crosspoll_store()`; bridge implementations of the three new `LoomBridge` methods; `_compaction_*` state + `_register_compaction_timer()` / `_cancel_compaction_timer()`; one new owner command `@versecompact`; capability docs unchanged (re-uses `llm.verse.gm`).
- Tests under `plugins/llm/tests/verse/` and `plugins/llm/tests/test_plugin.py` and `plugins/llm/tests/test_config.py` for everything above.
- Docs: `docs/guide/operator/forest-verse.md` (crosspoll + compaction sections, `@versecompact`), `docs/guide/reference/commands.md` (one new entry), `CHANGELOG.md` unreleased.
- Design-doc cross-reference: `docs/plans/2026-05-07-forest-verse-design.md` §"Open follow-ups" loses the embedding-`verse_recall` bullet only if you add it explicitly here (we don't — it stays open).

PR 3 ships **none** of:

- Embedding-based `verse_recall` (separate PR; substring search remains).
- Persistence-on-`@config` for any registry key (cross-cutting Limnoria concern, separate PR).
- Web view at `/verse/<channel>` (separate PR).
- Gemini cache plumbing (`service.py` change, separate PR).
- Loom-cycle inspection dashboard.
- Backfill / migration of existing verses; the new `crosspoll_seeds` table starts empty and the lore digest is only produced when retention is exceeded.

If a task tempts you to touch anything outside this list, **stop**. Either it belongs in this list (then add it via a Revisions entry first) or it doesn't (then it's a separate PR).

---

## Files map

```
plugins/llm/src/llm/verse/
  crosspoll_store.py            NEW   shared CrosspollStore class + helpers
  crosspoll_schema.sql          NEW   embedded as importlib.resources for crosspoll DB init
  compaction.py                 NEW   compact_verse(...) + register_daily_timer(...)
  store.py                      MOD   add events_older_than, replace_events_with_lore_digest
  loom.py                       MOD   entity-id grounding; crosspoll_seed op; apply_or_queue
                                      routing; Loom.tick consume hook; bridge protocol grows
  schema.sql                    UNCHANGED
plugins/llm/src/llm/
  config.py                     MOD   five new registry keys
  plugin.py                     MOD   crosspoll store wiring, bridge methods, compaction
                                      timer, @versecompact command
plugins/llm/tests/verse/
  test_crosspoll_store.py       NEW
  test_compaction.py            NEW
  test_loom.py                  MOD   entity-id grounding; crosspoll_seed parse + route;
                                      tick consume hook
  test_store.py                 MOD   retention helpers
  fixtures/                     MOD   one new fixture for digests with crosspoll_seed
plugins/llm/tests/
  test_plugin.py                MOD   @versecompact; compaction timer; bridge wiring
  test_config.py                MOD   five new keys
docs/guide/operator/
  forest-verse.md               MOD   crosspoll + compaction sections + @versecompact
docs/guide/reference/
  commands.md                   MOD   @versecompact entry
CHANGELOG.md                    MOD   unreleased entry
```

---

## Phase A — VerseStore retention helpers

### Task A1: `events_older_than(cutoff_ts)`

A pure read helper that returns every event with `ts < cutoff_ts`, oldest-first. Used by compaction to gather the rows that will be summarised into a lore digest.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: conftest helper** — add a small test helper to `plugins/llm/tests/verse/conftest.py` (or extend an existing one). Tests need to seed events at a *fixed* `ts`; the production `add_event` always stamps with `time.time()`, so a test-only insert path is required.

```python
# in plugins/llm/tests/verse/conftest.py

def insert_event_at(store, *, summary: str, entity_ids, source: str, ts: float) -> int:
    """Test helper: insert an events row with a caller-specified ``ts``.

    Production code always stamps ``ts`` to ``time.time()``; this helper
    bypasses that for retention/compaction tests. Lives in conftest so
    no production class carries test-only methods.
    """
    import json as _json
    with store.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) "
            "VALUES (?, ?, ?, ?)",
            (ts, summary, _json.dumps(list(entity_ids)), source),
        )
        return int(cur.lastrowid)
```

(If `plugins/llm/tests/verse/conftest.py` does not yet exist, create it.)

- [ ] **Step 2: tests** — append a new test class to `plugins/llm/tests/verse/test_store.py`:

```python
class TestEventsOlderThan:
    def test_returns_oldest_first(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        ids: list[int] = []
        for ts in (10.0, 20.0, 30.0):
            ids.append(insert_event_at(
                store, summary=f"e{ts}", entity_ids=[], source="loom", ts=ts,
            ))
        rows = store.events_older_than(cutoff_ts=25.0)
        assert [r.id for r in rows] == [ids[0], ids[1]]
        assert [r.ts for r in rows] == [10.0, 20.0]

    def test_empty_when_no_events_below_cutoff(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=100.0)
        assert store.events_older_than(cutoff_ts=50.0) == []

    def test_includes_all_sources(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="a", entity_ids=[], source="avatar", ts=5.0)
        insert_event_at(store, summary="b", entity_ids=[], source="loom", ts=6.0)
        insert_event_at(store, summary="c", entity_ids=[], source="crosspoll", ts=7.0)
        rows = store.events_older_than(cutoff_ts=10.0)
        assert {r.source for r in rows} == {"avatar", "loom", "crosspoll"}
```

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_store.py::TestEventsOlderThan -v` → fail (no method).

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/store.py`, alongside `recent_events`:

```python
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
                entity_ids=tuple(int(x) for x in json.loads(row[3])),
                source=row[4],
            )
            for row in cur.fetchall()
        ]
```

- [ ] **Step 4: run** the tests again → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): events_older_than helper for retention compaction"
```

### Task A2: `replace_events_with_lore_digest`

A single atomic mutation: insert one lore-digest event with the supplied summary + ts + entity_ids and delete every event whose `id` was in the supplied list. Returns the new digest event's id. The caller (compaction) is responsible for picking the cutoff, building the summary text, and gathering the to-delete ids.

**Files:** as A1.

- [ ] **Step 1: tests** — append a new test class to `plugins/llm/tests/verse/test_store.py`:

```python
class TestReplaceEventsWithLoreDigest:
    def test_replaces_atomically_and_returns_new_id(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        old_ids = [
            insert_event_at(
                store, summary=f"e{i}", entity_ids=[],
                source="avatar", ts=float(i),
            )
            for i in range(5)
        ]
        new_id = store.replace_events_with_lore_digest(
            delete_ids=old_ids,
            summary="A digest of five small events.",
            entity_ids=(),
            ts=100.0,
        )
        assert new_id > 0
        # surviving rows: only the new digest event
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT id, summary, source FROM events ORDER BY id ASC"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == new_id
        assert rows[0][1] == "A digest of five small events."
        assert rows[0][2] == "loom"

    def test_rolls_back_on_invalid_source(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        oid = insert_event_at(
            store, summary="e", entity_ids=[], source="avatar", ts=1.0
        )
        # Force a CHECK violation by exercising the inner helper with a
        # source not in the events.source CHECK list.
        with pytest.raises(Exception):
            store._replace_events_with_source(  # type: ignore[attr-defined]
                delete_ids=[oid],
                summary="x",
                entity_ids=(),
                ts=2.0,
                source="not_a_real_source",
            )
        # original event still present, no digest row created
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id FROM events").fetchall()
        assert [r[0] for r in rows] == [oid]

    def test_no_delete_ids_still_inserts_digest(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="empty digest",
            entity_ids=(),
            ts=42.0,
        )
        assert new_id > 0

    def test_entity_ids_are_json_encoded(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="d",
            entity_ids=(1, 2, 3),
            ts=10.0,
        )
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_ids FROM events WHERE id=?", (new_id,)
            ).fetchone()
        assert json.loads(row[0]) == [1, 2, 3]
```

The second test uses a private helper `_replace_events_with_source` to exercise the rollback path — implement it as the shared inner method that takes `source` as a parameter; `replace_events_with_lore_digest` is a thin caller that hard-codes `source='loom'`.

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_store.py::TestReplaceEventsWithLoreDigest -v` → fail.

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/store.py`:

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
    """Atomic delete-then-insert. Returns the new event's id."""
    with self.write_transaction() as conn:
        if delete_ids:
            placeholders = ",".join("?" for _ in delete_ids)
            conn.execute(
                f"DELETE FROM events WHERE id IN ({placeholders})",
                tuple(delete_ids),
            )
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) "
            "VALUES (?, ?, ?, ?)",
            (ts, summary, json.dumps(list(entity_ids)), source),
        )
        return int(cur.lastrowid)

def replace_events_with_lore_digest(
    self,
    *,
    delete_ids: Sequence[int],
    summary: str,
    entity_ids: Sequence[int],
    ts: float,
) -> int:
    """Replace ``delete_ids`` with a single ``source='loom'`` digest event.

    All work happens inside one ``write_transaction``; on error the whole
    operation rolls back and the originals survive.
    """
    return self._replace_events_with_source(
        delete_ids=delete_ids,
        summary=summary,
        entity_ids=entity_ids,
        ts=ts,
        source="loom",
    )
```

- [ ] **Step 4: run** the tests again → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): replace_events_with_lore_digest atomic helper"
```

### Phase A verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/test_store.py -v` → all pass.
- [ ] `uv run pytest plugins/llm -q` → no regressions.

---

## Phase B — CrosspollStore (shared seed queue)

### Task B1: scaffold `verse/crosspoll_store.py` + `crosspoll_schema.sql`

A separate SQLite file at `data/verse/_crosspoll.db`. Distinct from per-channel DBs. Two tables: `crosspoll_seeds` (one row per emit), `crosspoll_consumptions` (one row per (seed_id, dest_channel) pull). Thread-local connection + WAL + write lock, mirroring the pattern in `plugins/llm/src/llm/verse/store.py`.

**Files:**
- Create: `plugins/llm/src/llm/verse/crosspoll_schema.sql`.
- Create: `plugins/llm/src/llm/verse/crosspoll_store.py`.
- Create: `plugins/llm/tests/verse/test_crosspoll_store.py`.

- [ ] **Step 1: schema** — write `plugins/llm/src/llm/verse/crosspoll_schema.sql`:

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crosspoll_seeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_channel  TEXT NOT NULL,
    summary         TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crosspoll_seeds_created ON crosspoll_seeds(created_at);

CREATE TABLE IF NOT EXISTS crosspoll_consumptions (
    seed_id      INTEGER NOT NULL REFERENCES crosspoll_seeds(id) ON DELETE CASCADE,
    dest_channel TEXT NOT NULL,
    consumed_at  REAL NOT NULL,
    proposal_id  TEXT NOT NULL,
    PRIMARY KEY (seed_id, dest_channel)
);
CREATE INDEX IF NOT EXISTS idx_crosspoll_consumptions_dest ON crosspoll_consumptions(dest_channel, consumed_at);
```

- [ ] **Step 2: tests** — write `plugins/llm/tests/verse/test_crosspoll_store.py`:

```python
from pathlib import Path
import pytest


@pytest.fixture
def crosspoll_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verse"
    d.mkdir()
    return d


class TestCrosspollStoreInit:
    def test_creates_db_file_on_first_use(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="hello", payload={})
        assert (crosspoll_dir / "_crosspoll.db").exists()

    def test_schema_version_recorded(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="hello", payload={})
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
        assert row[0] == 1
```

- [ ] **Step 3: run** `uv run pytest plugins/llm/tests/verse/test_crosspoll_store.py::TestCrosspollStoreInit -v` → fail.

- [ ] **Step 4: implement** `plugins/llm/src/llm/verse/crosspoll_store.py`:

```python
"""Shared crosspoll-seed queue used by the loom across all verses.

One SQLite file at ``<data_dir>/_crosspoll.db``. Thread-local connection
+ WAL + per-store write lock — same pattern as ``VerseStore``. Source
verses enqueue seeds; receiver verses pull the oldest unconsumed seed
on their next loom cycle.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, NamedTuple

_LOG = logging.getLogger("llm.verse.crosspoll_store")
_SCHEMA_VERSION = 1


class CrosspollSeed(NamedTuple):
    id: int
    source_channel: str
    summary: str
    payload: dict[str, Any]
    created_at: float


class CrosspollStore:
    """Thread-safe shared store. One instance per plugin process."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "_crosspoll.db"
        self._tls = threading.local()
        self._write_lock = threading.Lock()
        self._initialised = False

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._tls, "conn", None)
        if c is not None:
            return c
        c = sqlite3.connect(self._path, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        self._tls.conn = c
        if not self._initialised:
            with self._write_lock:
                if not self._initialised:
                    self._migrate(c)
                    self._initialised = True
        return c

    def _migrate(self, conn: sqlite3.Connection) -> None:
        sql = files("llm.verse").joinpath("crosspoll_schema.sql").read_text()
        conn.executescript(sql)
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (_SCHEMA_VERSION, time.time()),
            )

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        yield conn

    @contextmanager
    def write_transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # ----- seed queue API -----

    def enqueue_seed(
        self,
        *,
        source_channel: str,
        summary: str,
        payload: dict[str, Any],
    ) -> int:
        """Append a seed and return its id."""
        now = time.time()
        with self.write_transaction() as conn:
            cur = conn.execute(
                "INSERT INTO crosspoll_seeds "
                "(source_channel, summary, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (source_channel, summary, json.dumps(payload), now),
            )
            return int(cur.lastrowid)

    def claim_seed_for(
        self,
        dest_channel: str,
        *,
        proposal_id: str,
    ) -> CrosspollSeed | None:
        """Atomically read-and-mark the oldest unconsumed seed.

        Performs the SELECT and the consumption-row INSERT inside a
        single ``write_transaction``. If the INSERT raises
        ``sqlite3.IntegrityError`` (another caller won the race for the
        same ``(seed_id, dest_channel)`` PK), the exception propagates
        out of the context manager — triggering ROLLBACK — and is
        caught by the outer try/except, which converts it back to a
        ``None`` return. Two concurrent receivers can therefore both
        call this and exactly one will get the seed; the loser sees
        ``None``.

        Excludes seeds whose ``source_channel == dest_channel`` so a
        verse cannot consume its own emissions.
        """
        try:
            with self.write_transaction() as conn:
                row = conn.execute(
                    "SELECT id, source_channel, summary, payload, created_at "
                    "FROM crosspoll_seeds "
                    "WHERE source_channel != ? "
                    "AND id NOT IN ("
                    "  SELECT seed_id FROM crosspoll_consumptions "
                    "  WHERE dest_channel = ?"
                    ") "
                    "ORDER BY created_at ASC, id ASC LIMIT 1",
                    (dest_channel, dest_channel),
                ).fetchone()
                if row is None:
                    return None
                seed_id, src, summary, payload_json, created_at = row
                conn.execute(
                    "INSERT INTO crosspoll_consumptions "
                    "(seed_id, dest_channel, consumed_at, proposal_id) "
                    "VALUES (?, ?, ?, ?)",
                    (seed_id, dest_channel, time.time(), proposal_id),
                )
        except sqlite3.IntegrityError:
            # Lost the race; ROLLBACK already happened in the
            # contextmanager's except branch.
            return None
        return CrosspollSeed(
            id=seed_id,
            source_channel=src,
            summary=summary,
            payload=json.loads(payload_json),
            created_at=created_at,
        )

    def next_unconsumed_for(self, dest_channel: str) -> CrosspollSeed | None:
        """Diagnostic-only: oldest seed not yet consumed by
        ``dest_channel``. Does **not** mark consumed — use
        ``claim_seed_for`` for the consume flow.
        """
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT id, source_channel, summary, payload, created_at "
                "FROM crosspoll_seeds "
                "WHERE source_channel != ? "
                "AND id NOT IN ("
                "  SELECT seed_id FROM crosspoll_consumptions "
                "  WHERE dest_channel = ?"
                ") "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                (dest_channel, dest_channel),
            ).fetchone()
        if row is None:
            return None
        return CrosspollSeed(
            id=row[0],
            source_channel=row[1],
            summary=row[2],
            payload=json.loads(row[3]),
            created_at=row[4],
        )

    def pending_count_for(self, dest_channel: str) -> int:
        """Count of seeds the destination has not yet consumed.

        Diagnostic only; not used in the cycle itself.
        """
        with self.read_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM crosspoll_seeds "
                "WHERE source_channel != ? AND id NOT IN ("
                "  SELECT seed_id FROM crosspoll_consumptions "
                "  WHERE dest_channel = ?"
                ")",
                (dest_channel, dest_channel),
            ).fetchone()
        return int(row[0])
```

- [ ] **Step 5: run** the tests → green.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/crosspoll_schema.sql \
        plugins/llm/src/llm/verse/crosspoll_store.py \
        plugins/llm/tests/verse/test_crosspoll_store.py
git commit -m "feat(verse): scaffold CrosspollStore + crosspoll_schema"
```

### Task B2: enqueue + atomic claim round-trip

End-to-end coverage of the most common path: source emits, receiver claims (atomic), second claim returns None for that receiver, second destination can still claim.

**Files:**
- Modify: `plugins/llm/tests/verse/test_crosspoll_store.py`.

- [ ] **Step 1: tests** — append:

```python
class TestEnqueueAndClaim:
    def test_claim_returns_seed_to_other_channel_and_marks_consumed(
        self, crosspoll_dir: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        sid = store.enqueue_seed(
            source_channel="#a", summary="A whisper", payload={"n": 1}
        )
        seed = store.claim_seed_for("#b", proposal_id="p-1")
        assert seed is not None
        assert seed.id == sid
        assert seed.source_channel == "#a"
        assert seed.summary == "A whisper"
        assert seed.payload == {"n": 1}
        # Second claim from same dest returns None — already consumed.
        assert store.claim_seed_for("#b", proposal_id="p-2") is None

    def test_source_cannot_claim_its_own_seed(
        self, crosspoll_dir: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="x", payload={})
        assert store.claim_seed_for("#a", proposal_id="p") is None

    def test_claim_returns_oldest_first(self, crosspoll_dir: Path) -> None:
        import time as _t
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="first", payload={})
        _t.sleep(0.001)
        store.enqueue_seed(source_channel="#a", summary="second", payload={})
        seed = store.claim_seed_for("#b", proposal_id="p-b1")
        assert seed is not None and seed.id == s1
        seed2 = store.claim_seed_for("#b", proposal_id="p-b2")
        assert seed2 is not None and seed2.summary == "second"

    def test_two_destinations_can_each_claim_same_seed(
        self, crosspoll_dir: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="x", payload={})
        seed_b = store.claim_seed_for("#b", proposal_id="p-b")
        seed_c = store.claim_seed_for("#c", proposal_id="p-c")
        assert seed_b is not None and seed_c is not None
        assert seed_b.id == seed_c.id == s1
        # Each dest's second claim returns None (already consumed there).
        assert store.claim_seed_for("#b", proposal_id="p-b2") is None
        assert store.claim_seed_for("#c", proposal_id="p-c2") is None

    def test_concurrent_claims_one_winner(self, crosspoll_dir: Path) -> None:
        """Two threads try to claim the same seed for the same dest;
        exactly one wins, exactly one consumption row exists."""
        import threading
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="x", payload={})
        results: list = []
        barrier = threading.Barrier(2)

        def claim(pid: str) -> None:
            barrier.wait()
            results.append(store.claim_seed_for("#b", proposal_id=pid))

        t1 = threading.Thread(target=claim, args=("p-1",))
        t2 = threading.Thread(target=claim, args=("p-2",))
        t1.start(); t2.start(); t1.join(); t2.join()
        won = [r for r in results if r is not None]
        lost = [r for r in results if r is None]
        assert len(won) == 1 and len(lost) == 1
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM crosspoll_consumptions "
                "WHERE seed_id=? AND dest_channel=?",
                (s1, "#b"),
            ).fetchone()[0]
        assert count == 1

    def test_pending_count_reflects_unconsumed(
        self, crosspoll_dir: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="x", payload={})
        store.enqueue_seed(source_channel="#a", summary="y", payload={})
        assert store.pending_count_for("#b") == 2
        store.claim_seed_for("#b", proposal_id="p")
        assert store.pending_count_for("#b") == 1
```

- [ ] **Step 2: run** → all green (B1's implementation already covers these paths). If any fail, fix the implementation; do **not** weaken the tests.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_crosspoll_store.py
git commit -m "test(verse): crosspoll enqueue + atomic claim_seed_for coverage"
```

### Task B3: concurrent writes hold the lock

Mirrors the existing concurrency test in `plugins/llm/tests/verse/test_store.py`. Real threads. Real DB. No mocks.

**Files:** as B2.

- [ ] **Step 1: tests** — append:

```python
class TestCrosspollConcurrency:
    def test_concurrent_enqueue_serialises(self, crosspoll_dir: Path) -> None:
        import threading
        from llm.verse.crosspoll_store import CrosspollStore
        store = CrosspollStore(crosspoll_dir)
        N = 50
        errors: list[BaseException] = []

        def writer(i: int) -> None:
            try:
                store.enqueue_seed(
                    source_channel=f"#chan-{i % 4}",
                    summary=f"line-{i}",
                    payload={"i": i},
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM crosspoll_seeds"
            ).fetchone()[0]
        assert count == N
```

- [ ] **Step 2: run** → green.

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_crosspoll_store.py
git commit -m "test(verse): CrosspollStore concurrent-writer guarantee"
```

### Phase B verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/test_crosspoll_store.py -v` → all pass.
- [ ] `uv run pytest plugins/llm -q` → no regressions.

---

## Phase C — Loom prompt: entity-id grounding

### Task C1: `build_verse_stable_block` emits `(id=N)` per entity

The current snapshot lists `(kind, name)` only. Extend it to `(kind, name, id)` so the digest model can reference real ids. The `VerseSnapshot` NamedTuple's `top_entities` field changes from `list[tuple[str, str]]` to `list[tuple[str, str, int]]`. Bridge implementations that build snapshots are updated to pass the entity id.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — locate the existing `TestBuildVerseStableBlock` class in `plugins/llm/tests/verse/test_loom.py` (near the top of the file). Replace its body (or append new tests if no class exists yet) with:

```python
class TestBuildVerseStableBlock:
    def test_lists_entities_with_ids(self) -> None:
        from llm.verse.loom import VerseSnapshot, build_verse_stable_block
        snap = VerseSnapshot(
            channel="#afnet",
            summary="A wood at the edge of town.",
            top_entities=[
                ("place", "the brook", 4),
                ("avatar", "rin", 7),
            ],
            recent_events=["someone whispered"],
        )
        out = build_verse_stable_block(snap)
        assert "- place: the brook (id=4)" in out
        assert "- avatar: rin (id=7)" in out

    def test_recent_events_preserved(self) -> None:
        from llm.verse.loom import VerseSnapshot, build_verse_stable_block
        snap = VerseSnapshot(
            channel="#afnet",
            summary="x",
            top_entities=[],
            recent_events=["A", "B"],
        )
        out = build_verse_stable_block(snap)
        assert "- A" in out and "- B" in out
```

Search the entire `plugins/llm/` tree for any other place that constructs a `VerseSnapshot` with two-tuples in `top_entities` — every occurrence must be updated to three-tuples now. Do this in the same step; the diff is mechanical.

```bash
grep -rn "VerseSnapshot(" plugins/llm/
```

Confirmed call sites (as of PR 2):
- `plugins/llm/tests/verse/test_loom.py` — multiple test fixtures
- `plugins/llm/tests/verse/test_loom_integration.py` — at least line 70
- `plugins/llm/tests/test_plugin.py` — bridge tests
- `plugins/llm/src/llm/plugin.py` — production `Bridge.snapshot`

Update all of them in the same commit so the test suite is consistent.

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_loom.py::TestBuildVerseStableBlock -v` → fail (missing id column).

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/loom.py`:

```python
class VerseSnapshot(NamedTuple):
    channel: str
    summary: str
    top_entities: list[tuple[str, str, int]]
    """``(kind, name, id)`` triples."""
    recent_events: list[str]
    """Newest-first."""


def build_verse_stable_block(snap: VerseSnapshot) -> str:
    """Per-cycle prompt block reused across seed/beat/digest calls.

    Each entity line carries its numeric id so the digest model can
    reference real entities instead of inventing ids.
    """
    parts = [
        f"# Focus verse: {snap.channel}",
        f"# Summary: {snap.summary}",
        "# Active entities:",
    ]
    for kind, name, eid in snap.top_entities:
        parts.append(f"- {kind}: {name} (id={eid})")
    parts.append("# Recent events (newest first):")
    for ev in snap.recent_events:
        parts.append(f"- {ev}")
    return "\n".join(parts)
```

- [ ] **Step 4: bridge updates** — `plugin.py` contains a `Bridge` class implementing `LoomBridge`. Find its `snapshot(channel)` method (`grep -n "def snapshot" plugins/llm/src/llm/plugin.py`). The current implementation builds `top_entities` as `[(e.kind, e.name) for e in (*avatars, *places)]` where `avatars` and `places` come from `store.list_entities_by_kind(...)`. The change is mechanical — append `e.id` to each tuple:

```python
def snapshot(self, channel: str) -> VerseSnapshot:
    store = self._plugin._get_or_create_verse_store(channel)
    avatars = store.list_entities_by_kind("avatar", status="active")[:5]
    places = store.list_entities_by_kind("place", status="active")[:5]
    top_entities: list[tuple[str, str, int]] = [
        (e.kind, e.name, e.id) for e in (*avatars, *places)
    ]
    recent = [
        ev.summary
        for ev in store.recent_events(limit=10, exclude_sources=("crosspoll",))
    ]
    summary = self._verse_summary_for(store)  # existing helper from PR 2
    return VerseSnapshot(
        channel=channel,
        summary=summary,
        top_entities=top_entities,
        recent_events=recent,
    )
```

(`Entity` is a NamedTuple with an `id` field — `plugins/llm/src/llm/verse/store.py:20`. The exact slicing limits and source helper names are whatever the existing PR 2 bridge already uses; the **only** behaviour change is appending `e.id` to each entity tuple.)

- [ ] **Step 5: regression test** — append a single test to `plugins/llm/tests/test_plugin.py` (or wherever Bridge tests already live) confirming the `exclude_sources=("crosspoll",)` invariant survives the snapshot rewrite. The original PR 2 bridge had this; PR 3's rewrite must preserve it so receiver-side crosspoll events don't recursively seed the source verse.

```python
class TestBridgeSnapshotExcludesCrosspoll:
    def test_snapshot_does_not_surface_crosspoll_events(self, plugin) -> None:
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        store = plugin._get_or_create_verse_store("#afnet")
        # Two events, one regular and one crosspoll-sourced.
        store.add_event(summary="regular", entity_ids=[], source="loom")
        store.add_event(summary="from elsewhere", entity_ids=[], source="crosspoll")
        snap = plugin._loom_bridge.snapshot("#afnet")
        joined = "\n".join(snap.recent_events)
        assert "regular" in joined
        assert "from elsewhere" not in joined
```

- [ ] **Step 6: run** `uv run pytest plugins/llm/tests/verse/test_loom.py -v` and `uv run pytest plugins/llm/tests/test_plugin.py -k 'Snapshot or VerseSnapshot' -v` → all green.

- [ ] **Step 7: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py \
        plugins/llm/src/llm/plugin.py \
        plugins/llm/tests/verse/test_loom.py \
        plugins/llm/tests/verse/test_loom_integration.py \
        plugins/llm/tests/test_plugin.py
git commit -m "feat(verse/loom): ground entity ids in verse-stable block"
```

### Task C2: tell the model that ids exist (LOOM_STATIC_PREFIX)

A small documentation-only change inside the static prefix: a sentence telling the model that entity lines include their numeric id and that proposals should reuse those ids. Cache-friendly (the prefix is unchanged across cycles after this lands).

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append a small test class:

```python
class TestStaticPrefixMentionsEntityIds:
    def test_prefix_documents_id_inclusion(self) -> None:
        from llm.verse.loom import LOOM_STATIC_PREFIX
        # Two assertions — neither is fragile to whitespace.
        assert "(id=" in LOOM_STATIC_PREFIX
        assert "reuse" in LOOM_STATIC_PREFIX.lower() and "id" in LOOM_STATIC_PREFIX.lower()
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — locate the `LOOM_STATIC_PREFIX` triple-quoted string in `plugins/llm/src/llm/verse/loom.py` and append two new lines just before the trailing `"""`:

```
Each entity in the focus verse appears as `- kind: name (id=N)`. When you
reference an existing entity in `entity_ids`, `from_id`, `to_id`, or
`entity_id`, reuse the id you saw — do not invent ids.
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): document entity-id reuse in static prefix"
```

### Phase C verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm -q` → no regressions; `_proposal_entity_refs_resolve` should now reject fewer proposals in fixtures (acceptable change in counter values; if a test asserts a specific reject count, update it to reflect reality after grounding lands).

---

## Phase D — Loom: `crosspoll_seed` op + tick-time consume

### Task D1: extend `parse_digest` to accept `crosspoll_seed`

`crosspoll_seed`'s payload schema is identical to `add_event`'s (`summary: str`, `entity_ids: list[int]`). The validator path is fully reused.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append a test class to `plugins/llm/tests/verse/test_loom.py`:

```python
class TestParseDigestCrosspollSeed:
    def test_accepts_crosspoll_seed_op(self) -> None:
        from llm.verse.loom import parse_digest
        text = """
        [
          {
            "op": "crosspoll_seed",
            "payload": {"summary": "rumour from the brook", "entity_ids": [4]},
            "confidence": 0.6,
            "provenance": "transcript-line-2",
            "rationale": "ambient riffing"
          }
        ]
        """
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].op == "crosspoll_seed"
        assert out[0].payload["summary"] == "rumour from the brook"
        assert out[0].payload["entity_ids"] == [4]

    def test_rejects_crosspoll_seed_with_bad_payload(self) -> None:
        from llm.verse.loom import parse_digest
        text = '[{"op":"crosspoll_seed","payload":{"summary":"ok"},"confidence":0.5,"provenance":"p","rationale":"r"}]'
        # missing entity_ids
        out = parse_digest(text)
        assert out == []
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** in `plugins/llm/src/llm/verse/loom.py`:

Locate `_VALID_OPS` (`grep -n "_VALID_OPS" plugins/llm/src/llm/verse/loom.py`). It is a frozenset/tuple of the four current ops. Add `"crosspoll_seed"`. Then extend `_PAYLOAD_SCHEMA`:

```python
_PAYLOAD_SCHEMA: dict[str, tuple[tuple[str, Callable[[Any], bool], str], ...]] = {
    "add_event": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
    "set_attribute": (
        ("entity_id", _is_strict_int, "int"),
        ("key", lambda v: isinstance(v, str), "str"),
        ("value", lambda v: isinstance(v, str), "str"),
    ),
    "add_relation": (
        ("from_id", _is_strict_int, "int"),
        ("to_id", _is_strict_int, "int"),
        ("kind", lambda v: isinstance(v, str), "str"),
    ),
    "add_entity": (
        ("kind", lambda v: isinstance(v, str), "str"),
        ("name", lambda v: isinstance(v, str), "str"),
    ),
    "crosspoll_seed": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
}
```

- [ ] **Step 4: also document** — same file, just below the existing op list inside `LOOM_STATIC_PREFIX`, append the seed op to the documented op set:

Look for the line `op          — one of: add_event, set_attribute, add_relation, add_entity` and change to `op          — one of: add_event, set_attribute, add_relation, add_entity, crosspoll_seed`. Add one new `crosspoll_seed:` payload row in the description block, mirroring `add_event:`:

```
                  crosspoll_seed: summary (str), entity_ids (list[int])
                                  — emit only if this verse has crosspoll
                                    send permission; the seed will appear
                                    as a *proposal* in another verse for
                                    that operator to approve or reject.
```

- [ ] **Step 5: run** → green. Also run `uv run pytest plugins/llm/tests/verse/test_loom.py -v` to confirm no other parse tests regress.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): parse_digest accepts crosspoll_seed op"
```

### Task D2: extend `_proposal_entity_refs_resolve` for `crosspoll_seed`

`crosspoll_seed` carries `entity_ids` but those refer to entities in the **source** verse. The seed will be enqueued; entity-id validation against the source verse should still happen so we don't enqueue garbage.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append:

```python
class TestProposalEntityRefsResolveCrosspoll:
    def test_seed_refs_validate_against_source_store(self) -> None:
        from llm.verse.loom import ParsedProposal, _proposal_entity_refs_resolve

        class FakeStore:
            def __init__(self, known: set[int]) -> None:
                self.known = known
            def entity_exists(self, eid: int) -> bool:
                return eid in self.known

        store = FakeStore({4, 7})
        ok = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "x", "entity_ids": [4, 7]},
            confidence=0.5, provenance="p", rationale="r",
        )
        bad = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "x", "entity_ids": [99]},
            confidence=0.5, provenance="p", rationale="r",
        )
        assert _proposal_entity_refs_resolve(store, ok) is True
        assert _proposal_entity_refs_resolve(store, bad) is False
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — in `_proposal_entity_refs_resolve`, add a new branch for `crosspoll_seed` that mirrors `add_event`:

```python
if op == "crosspoll_seed":
    ids = payload.get("entity_ids") or []
    return all(store.entity_exists(eid) for eid in ids)
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): validate crosspoll_seed entity refs against source"
```

### Task D3: `apply_or_queue` routes `crosspoll_seed` through a new helper

`apply_or_queue` keeps its existing return-codes (`'applied'`, `'queued'`, `'rejected_invalid_refs'`) and gains `'crosspoll_emitted'` and `'crosspoll_skipped_disabled'` and `'crosspoll_skipped_limit'`. The dispatch is at the top: if the op is `crosspoll_seed`, run a separate path that:

1. Validates entity refs (already done by `_proposal_entity_refs_resolve`).
2. Checks `cfg.crosspoll_allow_send` — if False, return `'crosspoll_skipped_disabled'`. No DB write.
3. Checks the per-cycle emit counter — if it's at `cfg.crosspoll_per_cycle_limit`, return `'crosspoll_skipped_limit'`. No DB write.
4. Calls `crosspoll_store.enqueue_seed(...)`.
5. Writes one local audit `event` row with `source='loom'` summarising the emit (so it shows in `@verse` recents).
6. Increments the per-cycle emit counter.

The per-cycle counter lives on the `LoomCycle` (`emitted_seeds: int = 0`). The signature of `apply_or_queue` grows two parameters; the existing call site in `_digest_phase` passes them in.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append:

```python
class TestApplyOrQueueCrosspollSeed:
    def _seed(self, **over: Any) -> "ParsedProposal":
        from llm.verse.loom import ParsedProposal
        base = dict(
            op="crosspoll_seed",
            payload={"summary": "rumour", "entity_ids": []},
            confidence=0.6, provenance="p", rationale="r",
        )
        base.update(over)
        return ParsedProposal(**base)

    def test_disabled_send_returns_skipped(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.enqueued: list[tuple[str, str]] = []
            def enqueue_seed(self, *, source_channel, summary, payload):
                self.enqueued.append((source_channel, summary))
                return 1

        cx = FakeCross()
        result = apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx,
            source_channel="#afnet",
            allow_send=False,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert result.outcome == "crosspoll_skipped_disabled"
        assert cx.enqueued == []

    def test_at_limit_returns_skipped(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        class FakeCross:
            def enqueue_seed(self, **kw): return 0
        result = apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=FakeCross(),
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=1,
        )
        assert result.outcome == "crosspoll_skipped_limit"

    def test_emits_seed_writes_audit_event_and_increments(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.calls: list[dict] = []
            def enqueue_seed(self, *, source_channel, summary, payload):
                self.calls.append({
                    "source_channel": source_channel,
                    "summary": summary,
                    "payload": payload,
                })
                return 42

        cx = FakeCross()
        result = apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx,
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=2,
            already_emitted=0,
        )
        assert result.outcome == "crosspoll_emitted"
        assert result.seed_id == 42
        assert cx.calls == [{
            "source_channel": "#afnet",
            "summary": "rumour",
            "payload": {"summary": "rumour", "entity_ids": []},
        }]
        # one audit event present, source='loom'
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT summary, source FROM events ORDER BY id ASC"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "loom"
        assert "crosspoll" in rows[0][0].lower()

    def test_invalid_refs_rejected_before_send_check(
        self, verse_db_dir: Path
    ) -> None:
        # entity_ids=[99] doesn't resolve in this verse; we must hit the
        # rejected_invalid_refs branch even when allow_send=True. The
        # existing proposals.op CHECK rejects 'crosspoll_seed', so this
        # path must NOT call store.add_proposal.
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        class FakeCross:
            def enqueue_seed(self, **kw):
                raise AssertionError("must not be called")
        result = apply_or_queue(
            store, self._seed(payload={"summary": "x", "entity_ids": [99]}),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=FakeCross(),
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert result.outcome == "rejected_invalid_refs"
        # No proposals row was written (CHECK would have rejected it).
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_no_outcome_writes_crosspoll_seed_to_proposals(
        self, verse_db_dir: Path
    ) -> None:
        """Schema-invariant regression test: across every apply_or_queue
        outcome for op='crosspoll_seed', no proposals row with
        op='crosspoll_seed' is ever written. The proposals.op CHECK
        constraint accepts only the four PR 1/2 ops; if a future code
        path adds add_proposal(op='crosspoll_seed') it will explode at
        runtime, and this test will catch it."""
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.calls = 0
            def enqueue_seed(self, *, source_channel, summary, payload):
                self.calls += 1
                return self.calls

        cx = FakeCross()
        # Drive every outcome we care about: emit, skip-disabled,
        # skip-limit, rejected-invalid-refs.
        outcomes = []
        outcomes.append(apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx, source_channel="#afnet",
            allow_send=True, per_cycle_limit=1, already_emitted=0,
        ).outcome)
        outcomes.append(apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx, source_channel="#afnet",
            allow_send=False, per_cycle_limit=1, already_emitted=0,
        ).outcome)
        outcomes.append(apply_or_queue(
            store, self._seed(),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx, source_channel="#afnet",
            allow_send=True, per_cycle_limit=1, already_emitted=1,
        ).outcome)
        outcomes.append(apply_or_queue(
            store, self._seed(payload={"summary": "x", "entity_ids": [99]}),
            cycle_id="c-1", threshold=0.85,
            crosspoll_store=cx, source_channel="#afnet",
            allow_send=True, per_cycle_limit=1, already_emitted=0,
        ).outcome)
        # Sanity: we drove the four distinct branches.
        assert set(outcomes) == {
            "crosspoll_emitted", "crosspoll_skipped_disabled",
            "crosspoll_skipped_limit", "rejected_invalid_refs",
        }
        # The schema-invariant assertion:
        with store.read_connection() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE op='crosspoll_seed'"
            ).fetchone()[0]
        assert bad == 0
```

Adapt the result type to whatever return-shape you choose. The test file expects `result.outcome` and `result.seed_id` — implement a small NamedTuple to carry both.

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — in `plugins/llm/src/llm/verse/loom.py`:

```python
class ApplyOutcome(NamedTuple):
    outcome: str
    """One of: applied, queued, rejected_invalid_refs,
    crosspoll_emitted, crosspoll_skipped_disabled, crosspoll_skipped_limit."""
    seed_id: int | None = None


def apply_or_queue(
    store: Any,
    prop: ParsedProposal,
    *,
    cycle_id: str,
    threshold: float,
    crosspoll_store: Any | None = None,
    source_channel: str | None = None,
    allow_send: bool = False,
    per_cycle_limit: int = 0,
    already_emitted: int = 0,
) -> ApplyOutcome:
    """Always insert a proposal row OR enqueue a crosspoll seed.

    Crosspoll seeds bypass the per-channel proposals table and instead
    go to the shared crosspoll queue. An audit event row with
    ``source='loom'`` is written locally so the emit shows up in
    ``@verse`` recents.

    Note: the existing ``proposals.op`` CHECK constraint
    (``schema.sql``: ``op IN ('add_event','set_attribute','add_relation','add_entity')``)
    does **not** include ``'crosspoll_seed'``. We do not write a
    ``proposals`` row for any ``crosspoll_seed`` outcome — auto-rejects,
    skips, and successful emits all stay out of the proposals table.
    The local ``events`` audit row covers the success case.
    """
    if not _proposal_entity_refs_resolve(store, prop):
        if prop.op == "crosspoll_seed":
            # Cannot insert into proposals (CHECK constraint).
            # Just drop with a log line; the source verse loses nothing
            # because no real seed was emitted.
            return ApplyOutcome(outcome="rejected_invalid_refs")
        store.add_proposal(
            cycle_id=cycle_id,
            op=prop.op,
            payload=prop.payload,
            confidence=prop.confidence,
            provenance=prop.provenance,
            status="rejected",
            reviewer="auto-validator",
        )
        return ApplyOutcome(outcome="rejected_invalid_refs")

    if prop.op == "crosspoll_seed":
        if not allow_send:
            return ApplyOutcome(outcome="crosspoll_skipped_disabled")
        if already_emitted >= per_cycle_limit:
            return ApplyOutcome(outcome="crosspoll_skipped_limit")
        assert crosspoll_store is not None and source_channel is not None
        seed_id = crosspoll_store.enqueue_seed(
            source_channel=source_channel,
            summary=prop.payload["summary"],
            payload=prop.payload,
        )
        store.add_event(
            summary=f"crosspoll seed emitted: {prop.payload['summary']}",
            entity_ids=prop.payload.get("entity_ids") or [],
            source="loom",
        )
        return ApplyOutcome(outcome="crosspoll_emitted", seed_id=seed_id)

    auto = prop.op != "add_entity" and prop.confidence >= threshold
    if auto:
        store.apply_and_record_proposal(
            cycle_id=cycle_id,
            op=prop.op,
            payload=prop.payload,
            confidence=prop.confidence,
            provenance=prop.provenance,
            reviewer="loom",
        )
        return ApplyOutcome(outcome="applied")
    store.add_proposal(
        cycle_id=cycle_id,
        op=prop.op,
        payload=prop.payload,
        confidence=prop.confidence,
        provenance=prop.provenance,
    )
    return ApplyOutcome(outcome="queued")
```

Update the existing `_digest_phase` call site to pass the new kwargs and to track `cycle.emitted_seeds`. Add `emitted_seeds: int = 0` to `LoomCycle`. Each `apply_or_queue` call's `outcome == "crosspoll_emitted"` increments `cycle.emitted_seeds` (under the cycle lock).

- [ ] **Step 4: existing tests** — every existing call to `apply_or_queue` in `plugins/llm/tests/verse/test_loom.py` needs the new kwargs. The defaults (`crosspoll_store=None, source_channel=None, allow_send=False, per_cycle_limit=0, already_emitted=0`) keep the existing behaviour for non-crosspoll ops, so old tests can be left alone *except* anything that asserts the bare string return value (`assert result == "applied"`); change those to `assert result.outcome == "applied"`. Use `grep -n "apply_or_queue" plugins/llm/tests/verse/test_loom.py` to find every call and update each one.

- [ ] **Step 5: run** `uv run pytest plugins/llm/tests/verse/test_loom.py -v` → all green.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): apply_or_queue routes crosspoll_seed to shared queue"
```

### Task D4: `LoomConfig` + bridge protocol grow crosspoll fields

The `LoomConfig` dataclass grows `crosspoll_per_cycle_limit: int`. The `LoomBridge` Protocol grows three methods so the loom can ask the plugin per-channel and global state:

```python
def crosspoll_store(self) -> Any | None: ...
def verse_allow_send(self, channel: str) -> bool: ...
def verse_allow_receive(self, channel: str) -> bool: ...
```

`LoomConfig` is built once per cycle from registry values; `LoomBridge` methods are called inside `Loom.tick` and the digest phase.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append:

```python
class TestLoomConfigCrosspollDefault:
    def test_per_cycle_limit_defaults_present_in_dataclass(self) -> None:
        from llm.verse.loom import LoomConfig
        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="gemini/gemini-flash-lite-latest",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        assert cfg.crosspoll_per_cycle_limit == 1


class TestLoomBridgeProtocolHasCrosspoll:
    def test_protocol_documents_three_new_methods(self) -> None:
        import inspect
        from llm.verse.loom import LoomBridge
        members = {n for n, _ in inspect.getmembers(LoomBridge)}
        for name in ("crosspoll_store", "verse_allow_send", "verse_allow_receive"):
            assert name in members, f"LoomBridge missing {name}"
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — extend `LoomConfig` and `LoomBridge` Protocol:

```python
@dataclass(frozen=True, slots=True)
class LoomConfig:
    network: str
    loom_channel: str
    bot_nicks: tuple[str, ...]
    model: str
    cycle_interval_s: int
    verse_cooldown_s: int
    beat_window_s: int
    transcript_max_lines: int
    transcript_max_chars: int
    auto_apply_threshold: float
    crosspoll_per_cycle_limit: int = 1


class LoomBridge(Protocol):
    def list_candidate_channels(self) -> list[str]: ...
    def candidate_weight(self, channel: str) -> int: ...
    def snapshot(self, channel: str) -> VerseSnapshot: ...
    def post_to_loom_channel(self, text: str) -> bool: ...
    def schedule_after(self, delay_s: float, fn: Callable[[], None], name: str) -> None: ...
    def submit(self, label: str, fn: Callable[[], None]) -> None: ...
    def now(self) -> float: ...
    def store_for(self, channel: str) -> Any: ...
    def log_usage(
        self, *, channel: str, op: str, model: str, usage: LoomCallUsage
    ) -> None: ...
    # new in PR 3:
    def crosspoll_store(self) -> Any | None: ...
    def verse_allow_send(self, channel: str) -> bool: ...
    def verse_allow_receive(self, channel: str) -> bool: ...
```

Every `LoomBridge`-implementing class now has to provide these. There are two: the production `Bridge` in `plugin.py` and at least one fake in `plugins/llm/tests/verse/test_loom.py`. The production version reads `verseCrosspollAllowSend` / `verseCrosspollAllowReceive` from registry (Phase F adds the keys); for now, return `False` for both. Fakes return whatever the test wants.

- [ ] **Step 4: existing fakes** — `grep -n "class.*LoomBridge\|class FakeBridge\|class TestBridge" plugins/llm/tests/verse/test_loom.py`. Each fake adds three small stubs:

```python
def crosspoll_store(self) -> Any | None:
    return None
def verse_allow_send(self, channel: str) -> bool:
    return False
def verse_allow_receive(self, channel: str) -> bool:
    return False
```

- [ ] **Step 5: run** → green.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py \
        plugins/llm/src/llm/plugin.py \
        plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): LoomConfig + bridge gain crosspoll plumbing"
```

### Task D5: digest phase passes crosspoll plumbing into `apply_or_queue`

Inside `_digest_phase`, after `parse_digest`, the loop over proposals invokes `apply_or_queue` for each. PR 3 wires the crosspoll args:

```python
cycle.emitted_seeds  # increments per crosspoll_emitted outcome
```

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: fixture** — write `plugins/llm/tests/verse/fixtures/digests/two_seeds.json`:

```json
[
  {
    "op": "crosspoll_seed",
    "payload": {"summary": "first whisper", "entity_ids": []},
    "confidence": 0.6,
    "provenance": "transcript-line-1",
    "rationale": "ambient"
  },
  {
    "op": "crosspoll_seed",
    "payload": {"summary": "second whisper", "entity_ids": []},
    "confidence": 0.6,
    "provenance": "transcript-line-2",
    "rationale": "ambient"
  }
]
```

If `plugins/llm/tests/verse/conftest.py` does not already have a `fixture_text` helper, add one:

```python
def fixture_text(name: str) -> str:
    """Return the text contents of a file under
    ``plugins/llm/tests/verse/fixtures/<name>``."""
    return (
        Path(__file__).parent / "fixtures" / name
    ).read_text(encoding="utf-8")
```

(`Path` import must be added to conftest if missing.)

- [ ] **Step 2: tests** — append to `plugins/llm/tests/verse/test_loom.py`:

```python
class TestDigestPhaseRoutesCrosspoll:
    def test_emit_caps_at_per_cycle_limit(
        self, verse_db_dir: Path
    ) -> None:
        """Two seeds in one digest, limit=1 → only first enqueued."""
        from llm.verse.loom import (
            LoomCycle, apply_or_queue, parse_digest,
        )
        from llm.verse.store import VerseStore
        from .conftest import fixture_text

        store = VerseStore(verse_db_dir, "#afnet")
        digest = fixture_text("digests/two_seeds.json")
        proposals = parse_digest(digest)
        assert len(proposals) == 2

        enqueued: list[str] = []

        class FakeCross:
            def enqueue_seed(self, *, source_channel, summary, payload):
                enqueued.append(summary)
                return len(enqueued)

        cx = FakeCross()
        cycle = LoomCycle(
            cycle_id="c1", channel="#afnet", started_at=0.0,
            verse_stable_block="block",
        )
        for p in proposals:
            r = apply_or_queue(
                store, p,
                cycle_id=cycle.cycle_id, threshold=0.85,
                crosspoll_store=cx,
                source_channel=cycle.channel,
                allow_send=True,
                per_cycle_limit=1,
                already_emitted=cycle.emitted_seeds,
            )
            if r.outcome == "crosspoll_emitted":
                cycle.emitted_seeds += 1

        assert cycle.emitted_seeds == 1
        assert enqueued == ["first whisper"]
        # second seed was skipped, NOT silently re-enqueued
        assert "second whisper" not in enqueued
```

- [ ] **Step 3: run** → fail (existing code doesn't yet pass crosspoll plumbing through).

- [ ] **Step 4: implement** — modify `_digest_phase` in `plugins/llm/src/llm/verse/loom.py`. Inside the loop over proposals; **each iteration must observe the running counter** so the cap holds across N>1 seeds in a single digest:

```python
def _digest_phase(self, cycle: LoomCycle) -> None:
    # ... existing seeding and call to client.call(op="digest", ...)
    proposals = parse_digest(content)
    cx = self._bridge.crosspoll_store()
    allow_send = self._bridge.verse_allow_send(cycle.channel)
    store = self._bridge.store_for(cycle.channel)
    for p in proposals:
        # Snapshot the running emit counter under the lock for this
        # iteration only — the cap is enforced against the count *before*
        # this proposal is evaluated.
        with self._lock:
            already = cycle.emitted_seeds
        outcome = apply_or_queue(
            store, p,
            cycle_id=cycle.cycle_id,
            threshold=self._cfg.auto_apply_threshold,
            crosspoll_store=cx,
            source_channel=cycle.channel,
            allow_send=allow_send,
            per_cycle_limit=self._cfg.crosspoll_per_cycle_limit,
            already_emitted=already,
        )
        if outcome.outcome == "crosspoll_emitted":
            with self._lock:
                cycle.emitted_seeds += 1
        # existing per-outcome logging continues here
```

- [ ] **Step 5: run** → green.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py \
        plugins/llm/tests/verse/fixtures/digests/two_seeds.json \
        plugins/llm/tests/verse/conftest.py
git commit -m "feat(verse/loom): digest phase routes crosspoll seeds with per-cycle cap"
```

### Task D6: `Loom.tick` consumes one seed for receivers (atomic claim)

At cycle start, after picking the focus verse, if the bridge says `verse_allow_receive(channel)` is True and a `crosspoll_store()` exists, atomically claim one seed via `crosspoll_store.claim_seed_for(channel, proposal_id=<pre-generated-uuid>)`. The single-TX claim writes the consumption row inside the same transaction as the SELECT, so two concurrent receivers cannot both insert a duplicate local proposal. If the claim returns a seed, insert a pending `add_event` proposal into the channel's local `proposals` table — using the **same** pre-generated `proposal_id` (so the consumption row's `proposal_id` column points at a real proposal). Payload is `{"summary": seed.summary, "entity_ids": []}` (entity_ids stripped — the source's ids are not valid in the receiver).

Failure mode: if the local `add_proposal` insert fails after a successful claim, the consumption row is left dangling — that's a single-seed loss for this receiver, logged but not retried. Acceptable given the cheap retry path (next cycle picks up a different seed).

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Modify: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Step 1: tests** — append the full test class. Reuses `LoomCallUsage` and `VerseSnapshot` from PR 2's existing fakes; if the existing test file already defines a `FakeBridge`, this test can sub-class or re-instantiate; otherwise the in-line `FakeBridge` below is self-contained:

```python
class TestLoomTickConsumesSeed:
    def test_receiver_pulls_one_seed_inserts_proposal(
        self, verse_db_dir: Path
    ) -> None:
        from typing import Any
        from llm.verse.loom import (
            Loom, LoomCallUsage, LoomConfig, VerseSnapshot,
        )
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeSeed:
            id = 7
            source_channel = "#other"
            summary = "incoming whisper"
            payload: dict[str, Any] = {}
            created_at = 0.0

        class FakeCross:
            def __init__(self) -> None:
                self._available: list[Any] = [FakeSeed()]
                self.claims: list[tuple[int, str, str]] = []
            def claim_seed_for(self, ch: str, *, proposal_id: str) -> Any | None:
                if not self._available:
                    return None
                seed = self._available.pop(0)
                self.claims.append((seed.id, ch, proposal_id))
                return seed
            # next_unconsumed_for / mark_consumed not used in consume path

        class FakeClient:
            def call(self, *, op, model, messages):
                # Return empty content — seed phase short-circuits and
                # the cycle finalises before any beats get scheduled.
                return "", LoomCallUsage(0, 0, 0.0)

        cx = FakeCross()

        class FakeBridge:
            def list_candidate_channels(self) -> list[str]:
                return ["#afnet"]
            def candidate_weight(self, channel: str) -> int:
                return 1
            def snapshot(self, channel: str) -> VerseSnapshot:
                return VerseSnapshot(
                    channel=channel, summary="x",
                    top_entities=[], recent_events=[],
                )
            def post_to_loom_channel(self, text: str) -> bool:
                return True
            def schedule_after(self, delay_s, fn, name):
                pass  # tests never execute beat 2
            def submit(self, label, fn):
                fn()  # run worker inline
            def now(self) -> float:
                return 1000.0
            def store_for(self, channel: str) -> Any:
                return store
            def log_usage(self, *, channel, op, model, usage):
                pass
            def crosspoll_store(self) -> Any | None:
                return cx
            def verse_allow_send(self, channel: str) -> bool:
                return False
            def verse_allow_receive(self, channel: str) -> bool:
                return True

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="gemini/gemini-flash-lite-latest",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        loom.tick()

        assert cx.claims and cx.claims[0][1] == "#afnet"
        assert cx.claims[0][0] == 7
        # The proposal_id used in the claim matches the inserted proposal.
        proposal_id_claimed = cx.claims[0][2]
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT id, op, status, payload FROM proposals"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == proposal_id_claimed
        assert rows[0][1] == "add_event"
        assert rows[0][2] == "pending"
        import json
        assert json.loads(rows[0][3])["summary"] == "incoming whisper"

    def test_no_pull_when_receive_disabled(self, verse_db_dir: Path) -> None:
        # Same wiring as above but with verse_allow_receive=False;
        # expect no consume, no proposal. (Inline-copy the bridge or
        # parameterize.)
        from typing import Any
        from llm.verse.loom import (
            Loom, LoomCallUsage, LoomConfig, VerseSnapshot,
        )
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def claim_seed_for(self, ch, *, proposal_id):
                raise AssertionError("must not be called when receive disabled")

        cx = FakeCross()

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        class FakeBridge:
            def list_candidate_channels(self): return ["#afnet"]
            def candidate_weight(self, channel): return 1
            def snapshot(self, channel):
                return VerseSnapshot(channel=channel, summary="x",
                                      top_entities=[], recent_events=[])
            def post_to_loom_channel(self, text): return True
            def schedule_after(self, *a, **kw): pass
            def submit(self, label, fn): fn()
            def now(self): return 1000.0
            def store_for(self, channel): return store
            def log_usage(self, **kw): pass
            def crosspoll_store(self): return cx
            def verse_allow_send(self, channel): return False
            def verse_allow_receive(self, channel): return False

        cfg = LoomConfig(
            network="afnet", loom_channel="#forest", bot_nicks=(),
            model="m", cycle_interval_s=300, verse_cooldown_s=1200,
            beat_window_s=90, transcript_max_lines=40,
            transcript_max_chars=8000, auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        loom.tick()

        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — at the end of `Loom.tick`'s critical section in `plugins/llm/src/llm/verse/loom.py`, just after `self._active = cycle` and *before* `self._bridge.submit("loom:seed", ...)`, add the consume hook:

```python
self._maybe_consume_one_seed_for(cycle.channel)
```

And implement the helper on `Loom`:

```python
def _maybe_consume_one_seed_for(self, channel: str) -> None:
    """If this verse opts into receiving, atomically claim one pending
    seed and insert it as a pending ``add_event`` proposal in the
    receiver's table.

    The consume flow is:
      1. Pre-generate ``proposal_id`` (uuid).
      2. ``claim_seed_for`` writes the consumption row in one TX. If
         this caller wins the claim, it returns the seed; otherwise
         None (seed gone, or another receiver claimed first).
      3. If we won, insert the local proposal with the pre-generated
         id. If the proposal insert fails for any reason, the
         consumption row is left dangling — that's a one-seed loss
         for this receiver, logged but not retried.

    Failures otherwise are logged at WARNING and swallowed; a different
    seed will be picked up on the next cycle.
    """
    cx = self._bridge.crosspoll_store()
    if cx is None or not self._bridge.verse_allow_receive(channel):
        return
    proposal_id = uuid.uuid4().hex
    try:
        seed = cx.claim_seed_for(channel, proposal_id=proposal_id)
    except Exception:
        self._log.exception("crosspoll: claim_seed_for failed")
        return
    if seed is None:
        return
    store = self._bridge.store_for(channel)
    try:
        store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": seed.summary, "entity_ids": []},
            confidence=0.0,
            provenance=(
                f"crosspoll from {seed.source_channel} (seed-id={seed.id})"
            ),
            proposal_id=proposal_id,
        )
    except Exception:
        self._log.exception(
            "crosspoll: claimed seed %s but proposal insert failed; "
            "consumption row at proposal_id=%s is now dangling",
            seed.id, proposal_id,
        )
```

**Note:** `add_proposal` must accept a caller-supplied `proposal_id` for the consume hook to work. **Task D6-pre** below adds it. Confirmed `VerseStore.add_proposal` (`plugins/llm/src/llm/verse/store.py:556-600`) does not accept an id kwarg today — it always generates `pid = uuid.uuid4().hex` at line 578. D6-pre is therefore mandatory, not conditional.

### Task D6-pre: `VerseStore.add_proposal` accepts caller-supplied `proposal_id`

The default behaviour (kwarg omitted → generate uuid) is preserved so every existing call site stays untouched. The receiver consume hook in D6 passes the same id it gave to `claim_seed_for` so the crosspoll consumption row's `proposal_id` column points at a real proposal.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: tests** — append:

```python
class TestAddProposalAcceptsId:
    def test_default_generates_uuid_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0, provenance="t",
        )
        # 32-char lowercase hex (uuid4 .hex)
        assert isinstance(pid, str) and len(pid) == 32
        assert all(c in "0123456789abcdef" for c in pid)

    def test_caller_supplied_id_is_used_verbatim(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid_in = "deadbeef" * 4  # 32 chars
        pid_out = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0, provenance="t",
            proposal_id=pid_in,
        )
        assert pid_out == pid_in
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT id FROM proposals WHERE id=?", (pid_in,),
            ).fetchone()
        assert row is not None and row[0] == pid_in

    def test_caller_supplied_duplicate_id_raises(
        self, verse_db_dir: Path
    ) -> None:
        import sqlite3
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = "abcd" * 8
        store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0, provenance="t",
            proposal_id=pid,
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.add_proposal(
                cycle_id="c-1", op="add_event",
                payload={"summary": "y", "entity_ids": []},
                confidence=0.0, provenance="t",
                proposal_id=pid,
            )
```

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_store.py::TestAddProposalAcceptsId -v` → fail.

- [ ] **Step 3: implement** — modify `add_proposal` in `plugins/llm/src/llm/verse/store.py`:

```python
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
    The crosspoll-receiver consume hook passes a caller-supplied id so
    the consumption row written by ``CrosspollStore.claim_seed_for``
    points at the same proposal record.

    When *status* is 'approved' or 'rejected', *reviewer* must be
    supplied and reviewed_at is set to now.
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
```

- [ ] **Step 4: run** the tests again → green. Run the broader suite to confirm no existing call site broke: `uv run pytest plugins/llm -q`.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): add_proposal accepts caller-supplied proposal_id"
```

The consume call is inside the same lock as the `tick` body but **outside** the critical-section sub-block that mutates `_active`. The `add_proposal` write happens against `store` directly (its own lock). To keep cyclomatic clarity, place the `_maybe_consume_one_seed_for` call after the `with self._lock:` block exits but before `self._bridge.submit("loom:seed", ...)`. That way a long DB write doesn't hold the loom-cycle lock.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): tick consumes one crosspoll seed for receivers"
```

### Phase D verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/test_loom.py -v` → all pass.
- [ ] `uv run pytest plugins/llm -q` → no regressions.

---

## Phase E — Compaction job

### Task E1: pure `compact_verse(...)` helper

A pure function: takes a `VerseStore`, a retention window, a min-keep floor, a model name, a `LoomModelClient`, and a `log_usage` callback. Returns one of `'compacted'`, `'skipped_no_events'`, `'skipped_below_floor'`, `'skipped_disabled'`. Uses `events_older_than` and `replace_events_with_lore_digest`. The summary text is a one-shot loom call tagged `loom:compact`.

The helper does not know about `schedule.addEvent` or `verseEnabled`. The caller (the timer driver in E3) walks per-channel registries and calls the helper.

**Files:**
- Create: `plugins/llm/src/llm/verse/compaction.py`.
- Create: `plugins/llm/tests/verse/test_compaction.py`.

- [ ] **Step 1: tests** — write `plugins/llm/tests/verse/test_compaction.py`:

```python
from pathlib import Path
import pytest


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verse"
    d.mkdir()
    return d


class _FakeClient:
    def __init__(self, content: str = "A digest of past events.") -> None:
        self.content = content
        self.calls: list[dict] = []
    def call(self, *, op: str, model: str, messages: list[dict[str, str]]):
        from llm.verse.loom import LoomCallUsage
        self.calls.append({"op": op, "model": model, "messages": messages})
        return self.content, LoomCallUsage(prompt_tokens=10, completion_tokens=20, cost=0.0)


class TestCompactVerse:
    def test_skips_when_retention_zero(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store,
            retention_days=0,
            min_keep_events=20,
            model="gemini/gemini-flash-lite-latest",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out == "skipped_disabled"

    def test_skips_when_below_min_keep(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        # one old event, but min_keep=20 means we never compact
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store, retention_days=30, min_keep_events=20,
            model="m", client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 100_000_000.0,
        )
        assert out == "skipped_below_floor"

    def test_skips_when_no_old_events(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse
        from .conftest import insert_event_at
        store = VerseStore(verse_db_dir, "#afnet")
        # 25 recent events, none past retention
        for i in range(25):
            insert_event_at(
                store, summary=f"e{i}", entity_ids=[], source="loom",
                ts=1_000_000.0 - i,
            )
        out = compact_verse(
            store, retention_days=30, min_keep_events=20,
            model="m", client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out == "skipped_no_events"

    def test_compacts_old_events_into_single_digest(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse
        from .conftest import insert_event_at
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        # 25 old events (≥30 days back) + 25 fresh
        for i in range(25):
            insert_event_at(
                store, summary=f"old{i}", entity_ids=[], source="avatar",
                ts=now - 60 * SECONDS_PER_DAY,
            )
        for i in range(25):
            insert_event_at(
                store, summary=f"new{i}", entity_ids=[], source="avatar",
                ts=now - 1.0,
            )
        usage_calls: list[dict] = []
        client = _FakeClient(content="Past events: a wood, a brook, a whisper.")
        out = compact_verse(
            store, retention_days=30, min_keep_events=20,
            model="gemini/gemini-flash-lite-latest",
            client=client,
            log_usage=lambda **kw: usage_calls.append(kw),
            now=lambda: now,
        )
        assert out == "compacted"
        # 25 fresh + 1 digest
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT summary, source FROM events ORDER BY ts ASC"
            ).fetchall()
        assert len(rows) == 26
        # the digest is the oldest now
        assert rows[0][1] == "loom"
        assert "Past events" in rows[0][0]
        # client tag was loom:compact
        assert client.calls and client.calls[0]["op"] == "compact"
        # log_usage fired once
        assert len(usage_calls) == 1
        assert usage_calls[0]["op"] == "compact"

    def test_long_backlog_only_deletes_what_was_summarised(
        self, verse_db_dir: Path
    ) -> None:
        """If there are 500 old events and the per-pass cap is 200,
        exactly 200 originals are deleted; 300 survive for the next pass.
        Regression test for the v1 plan bug where ALL olds were deleted
        but only the last 200 were shown to the model."""
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse, _MAX_EVENTS_PER_PASS
        from .conftest import insert_event_at
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        # 500 old events, all way past retention
        for i in range(500):
            insert_event_at(
                store, summary=f"old{i}", entity_ids=[],
                source="avatar", ts=now - 60 * SECONDS_PER_DAY - i,
            )
        client = _FakeClient(content="A long-ago digest.")
        out = compact_verse(
            store, retention_days=30, min_keep_events=20,
            model="m", client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert out == "compacted"
        assert _MAX_EVENTS_PER_PASS == 200  # contract guard
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        # 500 originals - 200 deleted + 1 new digest = 301
        assert count == 500 - _MAX_EVENTS_PER_PASS + 1

    def test_per_event_summary_cap_truncates_long_summaries(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import (
            compact_verse, _MAX_SUMMARY_CHARS_PER_EVENT,
        )
        from .conftest import insert_event_at
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        long_summary = "x" * 5000
        # 25 events all with the giant summary
        for _ in range(25):
            insert_event_at(
                store, summary=long_summary, entity_ids=[],
                source="avatar", ts=now - 60 * SECONDS_PER_DAY,
            )
        client = _FakeClient()
        compact_verse(
            store, retention_days=30, min_keep_events=20,
            model="m", client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        # No bullet line in the prompt should exceed the cap (+ "- "
        # prefix + newline overhead).
        assert client.calls
        bullets = client.calls[0]["messages"][1]["content"]
        for line in bullets.splitlines():
            # "- " prefix + content + optional ellipsis ⇒ under cap+4
            assert len(line) <= _MAX_SUMMARY_CHARS_PER_EVENT + 4

    def test_entity_ids_truncation_logs_when_capped(
        self, verse_db_dir: Path, caplog
    ) -> None:
        import logging
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse, _MAX_DIGEST_ENTITY_IDS
        from .conftest import insert_event_at
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        # 25 old events, each referencing a unique entity id beyond the
        # union cap.
        for i in range(25):
            insert_event_at(
                store, summary=f"e{i}", entity_ids=list(range(i * 4, i * 4 + 4)),
                source="avatar", ts=now - 60 * SECONDS_PER_DAY,
            )
        with caplog.at_level(logging.INFO, logger="llm.verse.compaction"):
            compact_verse(
                store, retention_days=30, min_keep_events=20,
                model="m", client=_FakeClient(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        assert any(
            "entity_ids truncated" in r.message for r in caplog.records
        )
        # Resulting digest event has exactly _MAX_DIGEST_ENTITY_IDS ids.
        import json as _json
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_ids FROM events WHERE source='loom' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert len(_json.loads(row[0])) == _MAX_DIGEST_ENTITY_IDS

    def test_client_failure_aborts_without_data_loss(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        from llm.verse.compaction import compact_verse
        from .conftest import insert_event_at
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store, summary=f"old{i}", entity_ids=[], source="avatar",
                ts=now - 60 * SECONDS_PER_DAY,
            )
        class Bomb:
            def call(self, **kw):
                raise RuntimeError("model down")
        with pytest.raises(RuntimeError):
            compact_verse(
                store, retention_days=30, min_keep_events=20,
                model="m", client=Bomb(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        # all 25 still present
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 25
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** `plugins/llm/src/llm/verse/compaction.py`:

```python
"""Daily retention compaction for forest-verse channels.

A pure helper (`compact_verse`) and a thin scheduling driver
(`register_daily_timer` / `cancel_daily_timer`). The helper is the unit
of work; the driver picks the fire time, walks verse-enabled channels,
and invokes the helper. Failures abort the helper but never the timer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

_LOG = logging.getLogger("llm.verse.compaction")
SECONDS_PER_DAY = 86400

# Per-pass tunables. These are intentionally constants — operators can
# tune retention via verseEventRetentionDays / verseCompactionMinKeepEvents;
# the rest are safety-net caps.
_MAX_EVENTS_PER_PASS = 200
"""Hard cap on how many old events one compaction pass touches. Long
backlogs drain across multiple daily runs; one pass writes one
digest event covering at most this many originals."""

_MAX_SUMMARY_CHARS_PER_EVENT = 240
"""Per-event truncation in the bullet block; longer summaries are
elided with an ellipsis. Stops a single pathological event from
blowing past the cheap model's context."""

_MAX_BULLET_BLOCK_CHARS = 16000
"""Hard cap on the user-message bullet block, after per-event
truncation. Very few cases should hit this (200 × 240 = 48k worst
case, but realistic averages drop well below); when it does fire we
trim from the front (oldest) so the *newest* of the old events stay
in the prompt."""

_MAX_DIGEST_ENTITY_IDS = 32
"""Cap on the lore-digest event's entity_ids array. Beyond this we
log INFO + drop the rest; entity-heavy verses lose some grounding
in their digest event but the events table remains the canonical
truth."""


def compact_verse(
    store: Any,
    *,
    retention_days: int,
    min_keep_events: int,
    model: str,
    client: Any,
    log_usage: Callable[..., None],
    now: Callable[[], float],
) -> str:
    """Compact a single verse. Returns one of:

    - ``'compacted'`` — old events replaced by one digest event
    - ``'skipped_disabled'`` — ``retention_days <= 0``
    - ``'skipped_below_floor'`` — fewer than ``min_keep_events`` total
      events in the store
    - ``'skipped_no_events'`` — no events older than the retention window

    Per-pass behaviour: a single call processes at most
    ``_MAX_EVENTS_PER_PASS`` (200) old events. If the verse has a long
    backlog, additional daily runs drain it incrementally; the digest
    event written by *this* pass covers only the events the LLM
    actually saw.
    """
    if retention_days <= 0:
        return "skipped_disabled"

    with store.read_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if total < min_keep_events:
        return "skipped_below_floor"

    cutoff_ts = now() - retention_days * SECONDS_PER_DAY
    olds = store.events_older_than(cutoff_ts=cutoff_ts)
    if not olds:
        return "skipped_no_events"

    # Process the OLDEST batch first. This guarantees forward progress:
    # even if the verse keeps receiving new events past the retention
    # window, the floor on the events-older-than query keeps shrinking.
    batch = olds[:_MAX_EVENTS_PER_PASS]

    def _truncated(s: str) -> str:
        if len(s) <= _MAX_SUMMARY_CHARS_PER_EVENT:
            return s
        return s[: _MAX_SUMMARY_CHARS_PER_EVENT - 1] + "…"

    bullet_lines = [f"- {_truncated(e.summary)}" for e in batch]
    bullets = "\n".join(bullet_lines)
    if len(bullets) > _MAX_BULLET_BLOCK_CHARS:
        # Trim oldest bullets first; newest of the batch stay in.
        while bullet_lines and len("\n".join(bullet_lines)) > _MAX_BULLET_BLOCK_CHARS:
            bullet_lines.pop(0)
        bullets = "\n".join(bullet_lines)
        _LOG.info(
            "verse compaction: bullet block trimmed to %d chars over "
            "%d-event batch",
            len(bullets), len(bullet_lines),
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chronicler. Summarise the bullet list of past "
                "events into one paragraph (≤120 words). Do not invent "
                "details; only summarise what is in the list."
            ),
        },
        {"role": "user", "content": bullets},
    ]
    content, usage = client.call(op="compact", model=model, messages=messages)
    summary = (content or "").strip() or "A period of unrecorded events passed."

    delete_ids = [e.id for e in batch]
    union_ids: list[int] = []
    seen: set[int] = set()
    for ev in batch:
        for eid in ev.entity_ids:
            if eid not in seen:
                seen.add(eid)
                union_ids.append(eid)

    if len(union_ids) > _MAX_DIGEST_ENTITY_IDS:
        _LOG.info(
            "verse compaction: digest entity_ids truncated %d → %d "
            "(union over %d events); rest dropped",
            len(union_ids), _MAX_DIGEST_ENTITY_IDS, len(batch),
        )
        union_ids = union_ids[:_MAX_DIGEST_ENTITY_IDS]

    store.replace_events_with_lore_digest(
        delete_ids=delete_ids,
        summary=summary,
        entity_ids=union_ids,
        ts=now(),
    )
    log_usage(op="compact", model=model, usage=usage)
    return "compacted"


def register_daily_timer(
    *,
    schedule_module: Any,
    fire_at_local: str,
    callback: Callable[[], None],
    name: str = "llm_verse_compact",
    now: Callable[[], float] | None = None,
) -> None:
    """Register a single-shot ``schedule.addEvent`` for the next time the
    local clock reaches ``fire_at_local`` (HH:MM). The callback re-arms
    itself at the end of its run; this function is called once at plugin
    load.

    If a timer with ``name`` is already registered, it is cancelled first
    so duplicate registrations (e.g. on a plugin reload) cannot crash
    ``schedule.addEvent``'s name-uniqueness check.
    """
    import time as _time
    now_fn: Callable[[], float] = now if now is not None else _time.time
    cancel_daily_timer(schedule_module=schedule_module, name=name)
    fire_at = _next_local_time(fire_at_local, now=now_fn)
    schedule_module.addEvent(callback, fire_at, name=name)


def cancel_daily_timer(*, schedule_module: Any, name: str = "llm_verse_compact") -> None:
    try:
        schedule_module.removeEvent(name)
    except KeyError:
        pass


def _next_local_time(hhmm: str, *, now: Callable[[], float]) -> float:
    """Return the next epoch second whose local time is ``hhmm``.

    Falls back to one hour from now if ``hhmm`` is malformed.
    """
    import time
    try:
        h, m = (int(x) for x in hhmm.split(":", 1))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
    except ValueError:
        _LOG.warning("verseCompactionDailyAt malformed (%r); deferring 1h", hhmm)
        return now() + 3600.0
    cur = time.localtime(now())
    candidate = time.mktime((cur.tm_year, cur.tm_mon, cur.tm_mday, h, m, 0,
                              cur.tm_wday, cur.tm_yday, cur.tm_isdst))
    if candidate <= now():
        candidate += SECONDS_PER_DAY
    return candidate
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/compaction.py plugins/llm/tests/verse/test_compaction.py
git commit -m "feat(verse): compact_verse helper + daily-timer driver"
```

### Task E2: `_next_local_time` covers boundary cases

A small dedicated test class so the boundary logic (today vs tomorrow, malformed input, midnight) is verified independently of the larger flow.

**Files:**
- Modify: `plugins/llm/tests/verse/test_compaction.py`.

- [ ] **Step 1: tests** — append:

```python
class TestNextLocalTime:
    def test_returns_today_when_hhmm_in_future(self) -> None:
        import time as _t
        from llm.verse.compaction import _next_local_time
        # construct a "now" at local 03:00, ask for 10:00
        struct = _t.struct_time((2026, 5, 8, 3, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert out > now_ts
        assert (out - now_ts) < 86400  # under one day away

    def test_returns_tomorrow_when_hhmm_already_passed(self) -> None:
        import time as _t
        from llm.verse.compaction import _next_local_time
        struct = _t.struct_time((2026, 5, 8, 14, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert (out - now_ts) > 0
        assert (out - now_ts) < 86400  # under one day away

    def test_malformed_hhmm_falls_back_to_one_hour(self) -> None:
        from llm.verse.compaction import _next_local_time
        out = _next_local_time("not-a-time", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0

    def test_out_of_range_hhmm_falls_back(self) -> None:
        from llm.verse.compaction import _next_local_time
        out = _next_local_time("25:99", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0
```

- [ ] **Step 2: run** → green (the function is already implemented in E1).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_compaction.py
git commit -m "test(verse/compaction): _next_local_time boundary cases"
```

### Task E3: timer driver + plugin wiring

`plugin.py` registers the daily timer at plugin load. The callback walks every channel for which `verseEnabled=True`, calls `compact_verse(...)`, then re-registers itself for the next day.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: tests** — append:

```python
class TestCompactionTimerWiring:
    def test_plugin_registers_compaction_timer_at_load(self, plugin) -> None:
        # The plugin's __init__ should call register_daily_timer; the
        # registered name should be 'llm_verse_compact'.
        # We assert via the schedule-event registry the plugin keeps.
        assert plugin._compaction_timer_name == "llm_verse_compact"

    def test_compaction_callback_walks_verse_enabled_channels(
        self, plugin, irc, monkeypatch
    ) -> None:
        # Two channels: #afnet (enabled), #other (disabled).
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        plugin.registryValue("verseEnabled", "#other", value=False)
        called_for: list[str] = []
        def fake_compact(store, **kw):
            called_for.append(store._channel)  # type: ignore[attr-defined]
            return "skipped_no_events"
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse", fake_compact
        )
        plugin._run_compaction_pass()
        assert called_for == ["#afnet"]

    def test_compaction_failure_does_not_abort_remaining_channels(
        self, plugin, monkeypatch
    ) -> None:
        plugin.registryValue("verseEnabled", "#a", value=True)
        plugin.registryValue("verseEnabled", "#b", value=True)
        seen: list[str] = []
        def maybe_bomb(store, **kw):
            seen.append(store._channel)  # type: ignore[attr-defined]
            if store._channel == "#a":  # type: ignore[attr-defined]
                raise RuntimeError("fail")
            return "skipped_no_events"
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse", maybe_bomb
        )
        plugin._run_compaction_pass()
        assert "#a" in seen and "#b" in seen
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** in `plugins/llm/src/llm/plugin.py`:

Locate the section near the existing `addPeriodicEvent` calls in `__init__` (search for `llm_file_cleanup`, `llm_pending_tasks`). Add:

```python
self._compaction_timer_name = "llm_verse_compact"
self._register_compaction_timer()
```

And implement on the plugin class:

```python
def _register_compaction_timer(self) -> None:
    from llm.verse.compaction import register_daily_timer
    fire_at = self.registryValue("verseCompactionDailyAt") or "03:00"
    try:
        register_daily_timer(
            schedule_module=schedule,
            fire_at_local=fire_at,
            callback=self._compaction_tick,
            name=self._compaction_timer_name,
        )
    except Exception:
        self.log.exception("verse: failed to register compaction timer")

def _cancel_compaction_timer(self) -> None:
    from llm.verse.compaction import cancel_daily_timer
    cancel_daily_timer(
        schedule_module=schedule, name=self._compaction_timer_name
    )

def _compaction_tick(self) -> None:
    """Single firing of the daily timer: do work, then re-arm."""
    try:
        self._run_compaction_pass()
    finally:
        # Always re-arm — a failed pass shouldn't kill the timer.
        self._register_compaction_timer()

def _run_compaction_pass(self) -> None:
    from llm.verse import compaction as _compaction
    retention_days_default = 30
    min_keep = int(self.registryValue("verseCompactionMinKeepEvents") or 20)
    model = self.registryValue("loomModel") or "gemini/gemini-flash-lite-latest"
    client = self._get_or_create_loom_client()  # existing helper from PR 2
    for channel in self._verse_enabled_channels():
        if not self.registryValue("verseEnabled", channel):
            continue
        store = self._get_or_create_verse_store(channel)
        retention_days = int(
            self.registryValue("verseEventRetentionDays", channel)
            or retention_days_default
        )
        try:
            outcome = _compaction.compact_verse(
                store,
                retention_days=retention_days,
                min_keep_events=min_keep,
                model=model,
                client=client,
                log_usage=lambda *, op, model, usage: self._log_loom_usage(
                    channel=channel, op=op, model=model, usage=usage
                ),
                now=time.time,
            )
            self.log.info(
                "verse compaction: channel=%s outcome=%s", channel, outcome
            )
        except Exception:
            self.log.exception(
                "verse compaction failed for %s; continuing", channel
            )
```

`_verse_enabled_channels()` — if the plugin already has a helper that walks verse-enabled channels (it likely does — search for `verseEnabled` in `plugin.py`), reuse it. Otherwise add it: it iterates `self._verse_stores` (the cache) plus inspects the registry for any not-yet-seen channels. Don't over-engineer; an in-process "channels we've ever instantiated a verse for" plus "channels listed in the plugin's joined-channels registry" is enough.

`_log_loom_usage` likewise reuses the existing PR 2 helper that logs `loom:seed/beat/digest`. The same pattern applies; pass `op="compact"`.

In the plugin's existing `die()` / cleanup path (search `schedule.removeEvent("llm_file_cleanup")`), add `self._cancel_compaction_timer()`.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(verse): plugin wires daily compaction timer"
```

### Task E4: `@versecompact #chan` owner command

Manual trigger so the operator can verify compaction without waiting for the daily timer.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: tests** — append. Capability + reply-capture patterns mirror the existing `@verseapprove` tests in this same file (see `plugin.py:5286-5326` for the wrap pattern).

```python
class TestVersecompactCommand:
    def test_compacts_named_channel(self, plugin, irc, msg_in_channel) -> None:
        # Concrete import path: this test lives in
        # plugins/llm/tests/test_plugin.py; the helper lives in
        # plugins/llm/tests/verse/conftest.py.
        from plugins.llm.tests.verse.conftest import insert_event_at
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        store = plugin._get_or_create_verse_store("#afnet")
        SECONDS_PER_DAY = 86400
        now = 100_000_000.0
        for i in range(25):
            insert_event_at(
                store, summary=f"old{i}", entity_ids=[], source="avatar",
                ts=now - 60 * SECONDS_PER_DAY,
            )
        msg = msg_in_channel(irc, "#afnet", "owner",
                              "@versecompact #afnet")
        plugin.versecompact(irc, msg, [], "#afnet")
        replies = irc.captured_replies()
        assert any("compacted" in r.lower() for r in replies)

    def test_requires_capability(self, plugin, irc, msg_in_channel) -> None:
        """A user without ``llm.verse.gm`` is rejected with an irc.error
        reply. Mirrors how the existing ``versepurge`` and ``verseapprove``
        commands surface a missing capability — neither raises; both
        emit an ``irc.error`` line. The wrap-time
        ``("checkCapability", "llm.verse.gm")`` arg also covers this at
        the framework level for normal command dispatch."""
        msg = msg_in_channel(irc, "#afnet", "stranger",
                              "@versecompact #afnet")
        plugin.versecompact(irc, msg, [], "#afnet")
        replies = irc.captured_replies()
        assert any(
            "capability" in r.lower() for r in replies
        ), f"expected capability denial, got {replies!r}"

    def test_disabled_verse_says_so(self, plugin, irc, msg_in_channel) -> None:
        plugin.registryValue("verseEnabled", "#afnet", value=False)
        msg = msg_in_channel(irc, "#afnet", "owner",
                              "@versecompact #afnet")
        plugin.versecompact(irc, msg, [], "#afnet")
        replies = irc.captured_replies()
        assert any("verseEnabled" in r for r in replies)
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — add to `plugins/llm/src/llm/plugin.py` near the other verse owner commands (search for `def versepurge`):

```python
def versecompact(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    channel: str,
) -> None:
    """<channel>

    Manually run retention compaction for <channel>. Requires
    capability llm.verse.gm.
    """
    if not self.registryValue("verseEnabled", channel):
        irc.reply(
            f"verseEnabled is False for {channel}; nothing to compact.",
            prefixNick=False,
        )
        return
    from llm.verse import compaction as _compaction
    store = self._get_or_create_verse_store(channel)
    retention_days = int(
        self.registryValue("verseEventRetentionDays", channel) or 30
    )
    min_keep = int(self.registryValue("verseCompactionMinKeepEvents") or 20)
    model = (
        self.registryValue("loomModel") or "gemini/gemini-flash-lite-latest"
    )
    client = self._get_or_create_loom_client()  # existing helper from PR 2
    try:
        outcome = _compaction.compact_verse(
            store,
            retention_days=retention_days,
            min_keep_events=min_keep,
            model=model,
            client=client,
            log_usage=lambda *, op, model, usage: self._log_loom_usage(
                channel=channel, op=op, model=model, usage=usage
            ),
            now=time.time,
        )
    except Exception as exc:
        self.log.exception("@versecompact failed for %s", channel)
        irc.error(
            f"compaction failed for {channel}: {type(exc).__name__}",
            prefixNick=False,
        )
        return
    irc.reply(f"compaction outcome for {channel}: {outcome}", prefixNick=False)

versecompact = wrap(
    versecompact,
    [
        ("checkCapability", "llm.verse.gm"),
        "channel",
    ],
)
```

Register the command in the same `COMMAND_REGISTRY` dict where the other verse commands sit (`grep -n "versepurge\|verseproposals\|versedump\|verseapprove" plugins/llm/src/llm/plugin.py` to find the block — `name="..."` entries near line 346); add an entry mirroring `verseapprove`'s shape.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(verse): @versecompact owner command"
```

### Phase E verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/test_compaction.py -v` → all pass.
- [ ] `uv run pytest plugins/llm/tests/test_plugin.py::TestVersecompactCommand -v` → all pass.
- [ ] `uv run pytest plugins/llm -q` → no regressions.

---

## Phase F — Config + bridge wiring + docs + CHANGELOG

### Task F1: registry keys

Five new keys. Defaults match the design doc. Tests assert presence + default values; existing PR 1/2 keys are not touched.

**Files:**
- Modify: `plugins/llm/src/llm/config.py`.
- Modify: `plugins/llm/tests/test_config.py`.

| Key | Scope | Type | Default |
|---|---|---|---|
| `verseCrosspollAllowSend` | per-channel bool | `False` |
| `verseCrosspollAllowReceive` | per-channel bool | `False` |
| `verseCrosspollPerCycleLimit` | global int | `1` |
| `verseCompactionDailyAt` | global str | `"03:00"` |
| `verseCompactionMinKeepEvents` | global int | `20` |

- [ ] **Step 1: tests** — append to `plugins/llm/tests/test_config.py`:

```python
class TestPR3RegistryKeys:
    def test_verse_crosspoll_allow_send_per_channel_false(self, plugin):
        assert plugin.registryValue("verseCrosspollAllowSend", "#anywhere") is False

    def test_verse_crosspoll_allow_receive_per_channel_false(self, plugin):
        assert plugin.registryValue("verseCrosspollAllowReceive", "#anywhere") is False

    def test_verse_crosspoll_per_cycle_limit_global_one(self, plugin):
        assert plugin.registryValue("verseCrosspollPerCycleLimit") == 1

    def test_verse_compaction_daily_at_default(self, plugin):
        assert plugin.registryValue("verseCompactionDailyAt") == "03:00"

    def test_verse_compaction_min_keep_events_default(self, plugin):
        assert plugin.registryValue("verseCompactionMinKeepEvents") == 20
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — extend `plugins/llm/src/llm/config.py`. Locate the verse block (`grep -n "verseEnabled\|verseAutoApplyThreshold" plugins/llm/src/llm/config.py`). After the existing PR 1/2 keys, add:

```python
conf.registerChannelValue(
    LLM, "verseCrosspollAllowSend",
    registry.Boolean(False, _("""When True, this channel may emit crosspoll
        seeds from its loom digest into the shared crosspoll queue. Default
        False (off).""")),
)
conf.registerChannelValue(
    LLM, "verseCrosspollAllowReceive",
    registry.Boolean(False, _("""When True, on each loom cycle this channel
        may pull one queued crosspoll seed from another verse and insert it
        as a pending proposal for the operator to approve or reject. Default
        False (off).""")),
)
conf.registerGlobalValue(
    LLM, "verseCrosspollPerCycleLimit",
    registry.PositiveInteger(1, _("""Maximum crosspoll seeds a single loom
        digest may emit per cycle. Excess seeds are dropped with a
        warning.""")),
)
conf.registerGlobalValue(
    LLM, "verseCompactionDailyAt",
    registry.String("03:00", _("""Local-time HH:MM at which the daily
        verse-event-retention compaction job fires. Empty or malformed
        values defer the next run by one hour.""")),
)
conf.registerGlobalValue(
    LLM, "verseCompactionMinKeepEvents",
    registry.NonNegativeInteger(20, _("""Floor on total event count below
        which a verse is left alone by compaction. Prevents thrashing
        small verses.""")),
)
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat(verse): five PR 3 registry keys"
```

### Task F2: production `Bridge.verse_allow_send` / `verse_allow_receive` / `crosspoll_store`

Wire the three new bridge methods to real registry / store reads. The crosspoll store singleton lives on the plugin alongside the verse store cache.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: tests** — append:

```python
class TestBridgeCrosspollWiring:
    def test_verse_allow_send_reads_registry(self, plugin) -> None:
        plugin.registryValue("verseCrosspollAllowSend", "#a", value=True)
        plugin.registryValue("verseCrosspollAllowSend", "#b", value=False)
        bridge = plugin._loom_bridge  # set up at plugin load in PR 2
        assert bridge.verse_allow_send("#a") is True
        assert bridge.verse_allow_send("#b") is False

    def test_verse_allow_receive_reads_registry(self, plugin) -> None:
        plugin.registryValue("verseCrosspollAllowReceive", "#a", value=True)
        plugin.registryValue("verseCrosspollAllowReceive", "#b", value=False)
        bridge = plugin._loom_bridge
        assert bridge.verse_allow_receive("#a") is True
        assert bridge.verse_allow_receive("#b") is False

    def test_crosspoll_store_is_a_singleton(self, plugin) -> None:
        bridge = plugin._loom_bridge
        a = bridge.crosspoll_store()
        b = bridge.crosspoll_store()
        assert a is b
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — in `plugins/llm/src/llm/plugin.py`, locate the `Bridge` class (or however the existing `LoomBridge` impl is named — `grep -n "class .*LoomBridge\|class Bridge" plugins/llm/src/llm/plugin.py`). Add three methods:

```python
def crosspoll_store(self):
    return self._plugin._get_or_create_crosspoll_store()

def verse_allow_send(self, channel: str) -> bool:
    return bool(self._plugin.registryValue("verseCrosspollAllowSend", channel))

def verse_allow_receive(self, channel: str) -> bool:
    return bool(self._plugin.registryValue("verseCrosspollAllowReceive", channel))
```

And on the plugin class:

```python
def _get_or_create_crosspoll_store(self):
    from llm.verse.crosspoll_store import CrosspollStore
    if getattr(self, "_crosspoll_store", None) is None:
        self._crosspoll_store = CrosspollStore(self._verse_data_dir())
    return self._crosspoll_store
```

`_verse_data_dir()` — reuse whatever helper PR 1 added that returns the verse dir; or compose `Path(conf.supybot.directories.data()) / "verse"`. Whichever pattern PR 1 already uses.

- [ ] **Step 4: also update `LoomConfig` build** — the place that constructs `LoomConfig` from registry (`grep -n "LoomConfig(" plugins/llm/src/llm/plugin.py`) gains one line:

```python
crosspoll_per_cycle_limit=int(self.registryValue("verseCrosspollPerCycleLimit") or 1),
```

- [ ] **Step 5: run** → green.

- [ ] **Step 6: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(verse): production bridge wires crosspoll + per-cycle limit"
```

### Task F3-pre: thread `event_source` through `apply_proposal_and_mark` (mandatory)

`apply_proposal_and_mark` currently hard-codes `source="loom"` (`plugins/llm/src/llm/verse/store.py:717` calls `self._apply_op_inline(conn, op=op, payload=payload, source="loom")`). The receiver-side crosspoll flow needs `source='crosspoll'` on the resulting event row, so this kwarg must be plumbed through. `apply_proposal` already accepts a `source` kwarg (`store.py:728`); only `apply_proposal_and_mark` is missing it.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Modify: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: tests** — append:

```python
class TestApplyProposalAndMarkEventSource:
    def test_default_source_is_loom_and_proposal_marked_approved(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0, provenance="t",
        )
        store.apply_proposal_and_mark(pid, reviewer="op")
        with store.read_connection() as conn:
            ev_row = conn.execute(
                "SELECT source FROM events WHERE summary='x'"
            ).fetchone()
            pr_row = conn.execute(
                "SELECT status, reviewer, reviewed_at FROM proposals "
                "WHERE id=?",
                (pid,),
            ).fetchone()
        assert ev_row[0] == "loom"
        # apply_proposal_and_mark contract: status flipped, reviewer
        # recorded, reviewed_at populated.
        assert pr_row[0] == "approved"
        assert pr_row[1] == "op"
        assert pr_row[2] is not None and pr_row[2] > 0

    def test_event_source_crosspoll_and_proposal_marked_approved(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="crosspoll-recv", op="add_event",
            payload={"summary": "incoming", "entity_ids": []},
            confidence=0.0, provenance="crosspoll from #other",
        )
        store.apply_proposal_and_mark(
            pid, reviewer="op", event_source="crosspoll"
        )
        with store.read_connection() as conn:
            ev_row = conn.execute(
                "SELECT source FROM events WHERE summary='incoming'"
            ).fetchone()
            pr_row = conn.execute(
                "SELECT status, reviewer FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
        assert ev_row[0] == "crosspoll"
        assert pr_row[0] == "approved"
        assert pr_row[1] == "op"

    def test_already_approved_raises_and_does_not_double_apply(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "once", "entity_ids": []},
            confidence=0.0, provenance="t",
        )
        store.apply_proposal_and_mark(pid, reviewer="op")
        with pytest.raises(ValueError):
            store.apply_proposal_and_mark(
                pid, reviewer="op", event_source="crosspoll"
            )
        # Only one event row; no double-apply. The 'crosspoll' source
        # was rejected because the proposal was already terminal.
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE summary='once'"
            ).fetchone()[0]
        assert count == 1
```

- [ ] **Step 2: run** → fail.

- [ ] **Step 3: implement** — modify `apply_proposal_and_mark` in `plugins/llm/src/llm/verse/store.py`:

```python
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
            raise ValueError(
                f"proposal {proposal_id!r} already {status}; cannot apply"
            )
        payload = json.loads(payload_json)
        self._apply_op_inline(conn, op=op, payload=payload, source=event_source)
        conn.execute(
            "UPDATE proposals SET status='approved', reviewer=?, reviewed_at=? "
            "WHERE id=?",
            (reviewer, time.time(), proposal_id),
        )
```

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): apply_proposal_and_mark accepts event_source kwarg"
```

### Task F3-pre-2: `@verseapprove` infers crosspoll source from cycle_id

PR 2's `@verseapprove` command calls `apply_proposal_and_mark(pid, reviewer=...)` without an `event_source` kwarg. For receiver-side crosspoll proposals (which D6's consume hook inserts with `cycle_id="crosspoll-recv"`), the resulting event must carry `source='crosspoll'`. Smallest viable fix: `@verseapprove` reads the proposal's `cycle_id`, and if it starts with `"crosspoll-"`, passes `event_source="crosspoll"`. Otherwise default `'loom'` continues unchanged.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`.
- Modify: `plugins/llm/tests/test_plugin.py`.

- [ ] **Step 1: tests** — append:

```python
class TestVerseapproveCrosspollSource:
    def test_approve_crosspoll_proposal_writes_crosspoll_event(
        self, plugin, irc, msg_in_channel
    ) -> None:
        plugin.registryValue("verseEnabled", "#beta", value=True)
        store = plugin._get_or_create_verse_store("#beta")
        pid = store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": "from elsewhere", "entity_ids": []},
            confidence=0.0,
            provenance="crosspoll from #alpha (seed-id=1)",
        )
        msg = msg_in_channel(irc, "#beta", "owner",
                              f"@verseapprove {pid}")
        plugin.verseapprove(irc, msg, [], pid)
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT source FROM events WHERE summary='from elsewhere'"
            ).fetchone()
        assert row is not None and row[0] == "crosspoll"

    def test_approve_loom_proposal_still_writes_loom_event(
        self, plugin, irc, msg_in_channel
    ) -> None:
        plugin.registryValue("verseEnabled", "#beta", value=True)
        store = plugin._get_or_create_verse_store("#beta")
        pid = store.add_proposal(
            cycle_id="loom-c1", op="add_event",
            payload={"summary": "regular event", "entity_ids": []},
            confidence=0.5, provenance="t",
        )
        msg = msg_in_channel(irc, "#beta", "owner",
                              f"@verseapprove {pid}")
        plugin.verseapprove(irc, msg, [], pid)
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT source FROM events WHERE summary='regular event'"
            ).fetchone()
        assert row is not None and row[0] == "loom"
```

- [ ] **Step 2: run** → fail (existing command doesn't pass `event_source`).

- [ ] **Step 3: implement** — locate the existing `verseapprove` method in `plugins/llm/src/llm/plugin.py` (PR 2 added it — `grep -n "def verseapprove" plugins/llm/src/llm/plugin.py`). PR 2's body looks up the proposal via the existing `_load_proposal` helper, calls `store.apply_proposal_and_mark(...)`, and replies with `f"Approved {p.id[:8]} ({p.op})."`.

The change in PR 3 is **minimal**: derive `event_source` from the already-loaded `Proposal.cycle_id` and pass it through. Concretely, inside the existing body, between the proposal-load and the `apply_proposal_and_mark` call:

```python
# Existing PR 2 lines look something like:
#   p = self._load_proposal(store, proposal_id)        # returns Proposal | None
#   if p is None: ...                                   # existing handling
#   if p.status != "pending": ...                       # existing handling
#
# Change: derive event_source and thread it through.
event_source = (
    "crosspoll" if p.cycle_id.startswith("crosspoll-") else "loom"
)
try:
    store.apply_proposal_and_mark(
        proposal_id,
        reviewer=self._resolve_identity(irc, msg).key,
        event_source=event_source,
    )
except (LookupError, ValueError) as exc:
    self.log.exception("verseapprove apply failed: %s", proposal_id)
    irc.error(f"Apply failed: {exc}.", prefixNick=False)
    return
# Existing reply format preserved — do NOT change to a different format:
irc.reply(f"Approved {p.id[:8]} ({p.op}).", prefixNick=False)
```

Three rules:
1. Do **not** rename or replace existing helpers (`_load_proposal`, `_proposal_target_store`, `_resolve_identity`). Reuse them exactly as PR 2 wrote them.
2. Do **not** alter the reply format, the argument signature (`proposal_id`, `channel_arg: str | None = None`), or the wrap registration.
3. Only two new statements: the `event_source = ...` line and the `event_source=event_source` kwarg in the `apply_proposal_and_mark` call.

- [ ] **Step 4: run** → green.

- [ ] **Step 5: commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(verse): @verseapprove infers event_source from cycle_id"
```

### Task F3: end-to-end happy path test

A single coarse-grained test that exercises the full crosspoll flow: source verse digest emits a seed → seed lands in shared queue → receiver verse's next tick pulls and creates a pending proposal → operator approves → event row materialises with `source='crosspoll'`.

Depends on F3-pre's `event_source` kwarg being in place.

**Files:**
- Modify: `plugins/llm/tests/verse/test_loom.py` (add a single end-to-end test class — keep it adjacent to the existing PR 2 end-to-end tests).

- [ ] **Step 1: tests** — append:

```python
class TestCrosspollEndToEnd:
    def test_seed_emitted_then_consumed_then_approved(
        self, tmp_path: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        from llm.verse.loom import (
            Loom, LoomConfig, ParsedProposal, apply_or_queue,
        )
        from llm.verse.store import VerseStore

        verse_dir = tmp_path / "verse"
        verse_dir.mkdir()
        cx = CrosspollStore(verse_dir)
        src_store = VerseStore(verse_dir, "#alpha")
        rcv_store = VerseStore(verse_dir, "#beta")

        # Source emits one crosspoll_seed via apply_or_queue.
        seed_prop = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "a rumour from alpha", "entity_ids": []},
            confidence=0.7, provenance="t-1", rationale="ambient",
        )
        out = apply_or_queue(
            src_store, seed_prop,
            cycle_id="c-src", threshold=0.85,
            crosspoll_store=cx,
            source_channel="#alpha",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert out.outcome == "crosspoll_emitted"

        # Receiver atomically claims the seed (consumption row + read in
        # one TX), then inserts the local pending proposal with the same id.
        import uuid as _uuid
        proposal_id = _uuid.uuid4().hex
        seed = cx.claim_seed_for("#beta", proposal_id=proposal_id)
        assert seed is not None and seed.source_channel == "#alpha"
        rcv_store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": seed.summary, "entity_ids": []},
            confidence=0.0,
            provenance=f"crosspoll from #alpha (seed-id={seed.id})",
            proposal_id=proposal_id,
        )

        # Operator approves; receiver event row gets source='crosspoll'.
        rcv_store.apply_proposal_and_mark(
            proposal_id, reviewer="op", event_source="crosspoll"
        )
        with rcv_store.read_connection() as conn:
            rows = conn.execute(
                "SELECT summary, source FROM events"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "a rumour from alpha"
        assert rows[0][1] == "crosspoll"

        # Second claim returns None — already consumed for this dest.
        assert cx.claim_seed_for("#beta", proposal_id="p-x") is None
```

- [ ] **Step 2: run** → green (depends on F3-pre).

- [ ] **Step 3: commit**

```bash
git add plugins/llm/tests/verse/test_loom.py
git commit -m "test(verse): end-to-end crosspoll emit→consume→approve"
```

### Task F4: docs — `docs/guide/operator/forest-verse.md`

Extend the existing PR 1/PR 2 operator guide with two new sections plus document the new command. Match the existing tone (compact operator prose; no narrative). The file exists from PR 1 and was extended by PR 2's "Loom orchestrator" section; PR 3 inserts the new sections relative to those headings — confirm by `grep -n "## Loom orchestrator\|## Owner commands" docs/guide/operator/forest-verse.md` before editing.

**Files:**
- Modify: `docs/guide/operator/forest-verse.md`.

- [ ] **Step 1: add** a "Cross-pollination" section after the "Loom orchestrator" section. Body — exactly:

```markdown
## Cross-pollination

Two verses can exchange seeds — short rumours that flow from one
channel's loom digest to another's pending-proposals queue, where the
receiving operator decides whether to canonise them. **Both ends must
opt in:** the source needs `verseCrosspollAllowSend=True`; the receiver
needs `verseCrosspollAllowReceive=True`. Defaults are `False` everywhere.

`verseCrosspollPerCycleLimit` (global, default `1`) caps how many seeds
a source verse's digest may emit per loom cycle. Seeds in excess are
dropped with a warning.

Receivers pull at most one seed per loom cycle, oldest first. A seed
becomes a pending `add_event` proposal in the receiver's verse; approve
or reject it with `@verseapprove` / `@versereject` as usual. Approved
seeds materialise as events with `source='crosspoll'`.

A verse cannot consume its own emissions.
```

- [ ] **Step 2: add** a "Retention compaction" section just before "Owner commands". Body:

```markdown
## Retention compaction

Once a day at `verseCompactionDailyAt` (global, default `"03:00"`
local time), the plugin walks every channel where `verseEnabled=True`
and replaces the **oldest 200** events past `verseEventRetentionDays`
(per-channel, default `30`) with a single lore-digest event. The
summary is produced by the same cheap model the loom uses
(`loomModel`), tagged `loom:compact` in `@usage`.

`verseCompactionMinKeepEvents` (global, default `20`) sets a floor:
verses with fewer than that many total events are skipped. This keeps
small verses from thrashing.

### Drain rate and backlog math

A single compaction pass touches at most **200 events** — a safety
cap so one model call cannot blow past the cheap model's context
window. Practical implications:

- A backlog of 10,000 events past the retention window converges in
  about **50 daily runs** (~50 days).
- A verse that produces **more than 200 events/day past its retention
  window** will not converge under the daily cap; the events table
  grows unboundedly. If you see this, lower
  `verseEventRetentionDays`, or run `@versecompact #channel`
  repeatedly to drain a backlog manually (each invocation processes
  another 200-event batch).
- Realistic verses (avatar-driven) produce on the order of 1-10
  events/day, so the cap rarely matters.

Failures are logged at WARNING and never block the timer; the next
day's run will retry.
```

- [ ] **Step 3: add** the new command in the existing "Owner commands" list:

```markdown
### `@versecompact #channel`

Manually run retention compaction for `#channel`. Useful for testing or
forcing a digest before the daily timer fires. Requires capability
`llm.verse.gm`. Reports the outcome (`compacted`, `skipped_no_events`,
`skipped_below_floor`, `skipped_disabled`).
```

- [ ] **Step 4: registry-keys table** — append five rows for the five new keys with the same column shape as the existing table.

- [ ] **Step 5: commit**

```bash
git add docs/guide/operator/forest-verse.md
git commit -m "docs(verse): crosspoll, compaction, @versecompact in operator guide"
```

### Task F5: docs — `docs/guide/reference/commands.md`

Add a single command-reference entry for `@versecompact` mirroring the existing `@versepurge` entry.

**Files:**
- Modify: `docs/guide/reference/commands.md`.

- [ ] **Step 1: locate** the verse command section (`grep -n "@versedump\|@versepurge\|@verseproposals" docs/guide/reference/commands.md`).

- [ ] **Step 2: insert** an entry, modeled on `@versepurge`'s format:

```markdown
### versecompact

Manually run retention compaction for the named channel.
Requires capability `llm.verse.gm`.

```
@versecompact #channel
```

See [Forest-verse — Retention compaction](../operator/forest-verse.md#retention-compaction)
for what compaction does.
```

- [ ] **Step 3: commit**

```bash
git add docs/guide/reference/commands.md
git commit -m "docs: command reference for @versecompact"
```

### Task F6: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`.

- [ ] **Step 1: insert** under the unreleased "Added" header:

```markdown
- Forest-verse: cross-pollination between verses (`verseCrosspollAllowSend`,
  `verseCrosspollAllowReceive`, `verseCrosspollPerCycleLimit`); seeds queue
  in a shared `_crosspoll.db` and arrive in receivers as pending proposals.
- Forest-verse: daily retention compaction summarises events older than
  `verseEventRetentionDays` into a single lore-digest event
  (`verseCompactionDailyAt`, `verseCompactionMinKeepEvents`). New owner
  command `@versecompact #channel` runs it on demand.
- Forest-verse: loom prompt now grounds entity ids inline so the digest
  model stops inventing them.
```

- [ ] **Step 2: commit**

```bash
git add CHANGELOG.md
git commit -m "docs: CHANGELOG entries for forest-verse PR 3"
```

### Phase F verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm -q` → all pass.
- [ ] `uv run mkdocs build --strict` → no warnings (operator guide + commands ref render).

---

## Out of scope / deferred to a later PR

- **Embedding-based `verse_recall`.** Substring search continues to ship.
- **Persistence-on-`@config`.** Cross-cutting Limnoria registry concern; affects every key, not just verse.
- **Web view at `/verse/<channel>`.** Read-only HTML inspector; separate frontend track.
- **Gemini cache plumbing in `service.py`.** A different layer than the verse code; logs `cached_tokens` once that's wired.
- **Loom-cycle inspection dashboard.** Useful when tuning beat windows; defer until cycles are demonstrably running with crosspoll on.
- **Documenting the post-PR2 micro-fixes** (`loomCaptureTranscript`, orphan-rejection auto-validator, registry-rewire) in the operator guide. They landed post-PR2 and are not yet documented; ratification is a small standalone docs PR.

---

## Open follow-ups (not blocking PR 3)

- **Crosspoll seed garbage collection.** The shared `crosspoll_seeds` table grows monotonically; a row is "consumed" by destinations but never deleted. Acceptable for v1 (each row is small); revisit if the shared DB grows unbounded. A simple "delete seeds older than N days *and* consumed by ≥M destinations" cron would handle it.
- **Compaction model choice.** PR 3 uses `loomModel`. If operators want a stronger model for the digest, a `compactionModel` registry key with fallback to `loomModel` is the logical extension.
- **Per-channel crosspoll allowlist.** Today: any AllowReceive=True channel can pull from any AllowSend=True channel. A future "only receive from these channels" allowlist gives operators finer-grained social-graph control if it becomes necessary.
- **Crosspoll seed entity IDs.** Today seeds drop entity_ids on the receiver side (source's ids are not valid in the receiver). A future "soft entity matching" pass could let the receiver's loom propose `add_entity` follow-ups when a seed mentions an unknown name.

---

## Scope guard reminder

Re-read the "Scope guard for PR 3" block above before opening the PR.
The PR description should copy that block verbatim.

---

## Final review checklist

- [ ] `make check` green on the full repo.
- [ ] `uv run pytest plugins/llm -q` passes; coverage on `plugins/llm/src/llm/verse/` ≥ 90 %.
- [ ] `git diff main...HEAD --stat` shows only files listed in **Files** sections of the tasks above. No drive-by edits.
- [ ] CHANGELOG entry added.
- [ ] Operator guide and commands reference updated; `mkdocs build --strict` is clean.
- [ ] Open the PR with title `feat: forest-verse PR 3 — crosspoll + retention compaction + entity-id grounding` and a body that copies the §"Scope guard for PR 3" block above.
- [ ] After merge: wait for both the GitHub Actions run **and** the GHCR Docker build to publish before `systemctl --user restart vibebot` over SSH (per `feedback_wait_for_docker.md`). Crosspoll is opt-in and defaults off, so existing operators see no behaviour change at deploy time. Compaction begins firing daily at `verseCompactionDailyAt` (default `03:00`) once the new image is live.
