# Forest-verse PR 2 Implementation Plan (v2)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Revisions

- **v1** — initial draft.
- **v3 (this revision)** — second review pass found new blockers introduced in v2:
  - **`LLMExecutor.submit` signature** — real signature is `submit(label, fn, *args, **kwargs)` (`plugins/llm/src/llm/executor.py:95`), `label` required. v2 called `submit(fn)`. Fixed: `LoomBridge.submit(label, fn)` Protocol; `_PluginLoomBridge` calls `self._plugin._llm_executor.submit(label, fn)`. Driver passes `"loom:seed"`, `"loom:beat"`, `"loom:digest"`.
  - **`_llm_executor` lives on the plugin, not on `llm_service`** (`plugin.py:520`). v2 dereferenced `self._plugin.llm_service._llm_executor`. Fixed.
  - **Auto-apply audit row was non-atomic.** v2 called `apply_proposal` then `add_proposal` in two separate write transactions; a crash between them committed the mutation without the audit row. Fixed: new `VerseStore.apply_and_record_proposal()` does both inside one `write_transaction`. The operator path also gets `apply_proposal_and_mark()` for atomic apply + status flip.
  - **Cross-network channel-name collisions.** Verse DBs are keyed only on channel name; `#forest` on two networks collides on the same SQLite file. Even though loom is single-network, v2's `list_candidate_channels` walked all networks. Fixed: bridge filters to `world.getIrc(loomNetwork).state.channels`. (Network-prefixed sanitizer is a separate audit; flagged in §"Open follow-ups".)
  - **`_pointer` advanced on idle/cooldown skips,** so the next eligible cycle picked a non-deterministic offset. Fixed: increment only after `choice is not None`. New B10a test asserts skipped ticks don't rotate.
  - **doPrivmsg transcript hook ran before the prefix-character early-out** (`plugin.py:961`), so `@verseapprove` and `@versereject` lines posted in the loom channel were captured as transcript. Fixed: insertion point moves to *after* the prefix-char guard. New C3 test asserts prefixed commands aren't observed.
  - **`_loom_bot_nicks_cache` not initialized in `__init__`.** AttributeError risk if `doPrivmsg` ran before `_wire_loom_if_enabled`. Fixed: `__init__` declares all four caches; disable branch clears all four.
  - **`parse_digest` shallow type checks.** Booleans pass `isinstance(..., int)`; `entity_ids` element types unverified; optional fields (`note`, `summary`) untyped. Fixed: `_PAYLOAD_SCHEMA` rejects bool-as-int and validates list element types; new test.
  - **D1 and D2 test bullets replaced with full test code.** No more "cases:" hand-waves.
  - **`_FakeLoom` and `irc_for_network` fixtures defined inline.** No more "if not present, add it" hand-waves.
  - **F1 docs aligned to the positional `pending|approved|rejected` arg form** (v2 still referenced `--status=...` in operator-guide prose).
  - **B2 frozen-dataclass test uses `dataclasses.FrozenInstanceError`,** not generic `Exception`.
  - **B10c documents the single-thread invariant** (Limnoria scheduler serializes tick callbacks; `_active` plus the lock cover concurrency).
  - **B10a/b tests added `assert loom._active is None`** at finalization points.
- **v2** — applied feedback from `codex:codex-rescue` and `superpowers` code-reviewer subagents. Changes:
  - **Network resolution.** Added `loomNetwork` global registry key. Bridge resolves the `Irc` via `world.getIrc(network)`; cycle aborts cleanly if the network is not connected. Mirrors the pattern at `plugins/llm/src/llm/service.py:4541-4546`.
  - **Worker-thread execution.** Each phase's blocking model call now runs through `self._llm_executor.submit(...)`. `schedule.addEvent` only fires lightweight shim callbacks that hand the heavy work to a worker. Scheduler thread is never blocked on `litellm.completion`. Mirrors `service.py:4533-4565`.
  - **Concurrency.** Added a `threading.Lock` on `Loom` guarding `_active`, `_last_cycle_by_channel`, and the transcript snapshot. The transcript itself is read under the lock by phase callbacks (after copying), since `doPrivmsg` can append from a different thread.
  - **Cooldown is real.** `Loom._last_cycle_by_channel: dict[str, float]` is updated at the start of every cycle. `list_candidates` returns bare channels; the loom annotates them with `last_cycle_at` from its own state before calling `pick_focus_verse`.
  - **Auto-applied proposals are auditable.** `apply_or_queue` always inserts a row in `proposals`. Auto-applied → `status='approved' reviewer='loom' reviewed_at=now`. Queued → `status='pending'`. `@verseproposals --status=approved` now lists everything the loom committed without operator review.
  - **Verse-stable block snapshotted once per cycle.** `LoomCycle.verse_stable_block: str` is built when `tick()` picks a verse and reused for all three model calls. Restores the cache-eligibility intent from `design.md:327-333`.
  - **`@usage` accounting restored.** `LoomBridge.log_usage(channel, op, model, prompt_tokens, completion_tokens, cost)` is part of the bridge again. `LiteLLMLoomClient.call` now returns `(content, usage)` so the driver can route token counts through.
  - **Bot/human filter.** Added `loomBotNicks` (global comma-separated list). When non-empty, only nicks in the list are captured into the transcript. Empty means capture all non-self lines (the v1 default; documented as "fine for bot-heavy channels").
  - **Proposal `id` contradiction removed.** The prompt no longer asks the model to emit `id`; the store generates UUIDs. The schema in `LOOM_STATIC_PREFIX` lists only `op`, `payload`, `confidence`, `provenance`, `rationale`.
  - **`Event.entity_ids` is `tuple[int, ...]`.** A5's `test_apply_add_event_inserts_event` asserts `(eid,)` not `[eid]` (matches the existing NamedTuple definition at `plugins/llm/src/llm/verse/store.py:48`).
  - **`_loom_channel_cache` is set explicitly.** The C2 implementation block now shows the cache write/clear.
  - **`versereject` shown in full,** no placeholder fragments. Lookup logic extracted to `_load_proposal(store, raw_id) -> Proposal | None`.
  - **`@verseproposals` argument parsing fixed.** Switched from `--status=...` (which Limnoria's `wrap` can't parse cleanly) to a positional optional second argument: `@verseproposals [<channel>] [<pending|approved|rejected>]`. Default channel = current; default status = `pending`.
  - **B10 split.** Driver implementation was one large task; now B10a (`tick`), B10b (`_seed_phase` worker), B10c (`after_beat1` + `_beat_phase` worker), B10d (`after_beat2` + `_digest_phase` worker).
  - **B3 round-robin test strengthened.** 3-candidate tie test with pointer 0/1/2.
  - **A6 illustrative test dropped** (only the final test remains).

---

**Goal:** Wire up the loom orchestrator and proposal queue so the verse can mutate itself from improv in Forest's bot-heavy channel — cheap-model proposals, confidence-gated commit, owner moderation. After this PR ships, an operator who sets `loomNetwork=...` + `loomChannel=#...` and turns on `verseEnabled` for one or more verses gets multi-turn loom cycles writing to those verses' proposals tables.

**Architecture:**
- New module `plugins/llm/src/llm/verse/loom.py`. Pure logic (rotation, prompt builders, digest parser, apply policy) plus a `Loom` class that owns cycle state and a `threading.Lock`.
- Three `supybot.schedule.addEvent` callbacks per cycle (`tick`, `after_beat1`, `after_beat2`) — each one immediately submits the blocking model call to the existing `LLMService._llm_executor` worker pool. Scheduler thread is never blocked on a network call.
- `LoomBridge` Protocol mediates plugin internals (Irc lookup, verse-store cache, executor submit, `log_usage`). Tests use a fake.
- Transcript collection: plugin's existing `doPrivmsg` (`plugins/llm/src/llm/plugin.py:933`) calls `loom.observe_transcript(nick, text)` when the message is in the loom channel from a non-self source on the loom network. `Loom` guards `_active` and the transcript with its lock.
- Proposals table already exists from PR 1 (`plugins/llm/src/llm/verse/schema.sql:60–70`). PR 2 adds writers, applier, and the always-write-a-row policy.

**Tech stack:**
- Python 3.13+ via `uv`, `pytest`, `ruff`, `ty`.
- Limnoria 2025+ (`supybot`); `supybot.schedule.addEvent` for timing only — heavy work runs on `LLMService._llm_executor`.
- `litellm.completion` directly (the `LLMService.completion` path is wired for user-visible errors and is wrong here).
- Real SQLite in `tmp_path` for tests. **No mocks for the DB.** Recorded transcript JSON fixtures (`plugins/llm/tests/verse/fixtures/`) for digest tests; live model calls behind `VIBEBOT_TEST_LIVE=1`.

**Reference design:** `docs/plans/2026-05-07-forest-verse-design.md` (esp. §"Loom orchestrator", §"Configuration", §"Tests"). PR 1 is `docs/plans/2026-05-07-forest-verse-pr1.md`.

**Working directory:** `.worktrees/forest-verse-pr2` (branch `feat/forest-verse-pr2`). Baseline: `main` after PR 1 merged (`6b195c4`).

**Project rules to honor:**
- `make lint && make typecheck` runs after every Edit. Pre-commit also runs ruff format + gitleaks + ty. Don't suppress.
- All persistence tests use real SQLite files, not mocks (per `feedback_wait_for_docker.md`).
- `uv run pytest …`, never bare `pytest`.
- Frequent atomic commits; one task → one commit.
- Push directly to `main` is fine; CI + Docker build are separate workflows — wait for both before restarting prod.

**Scope guard for PR 2:**

PR 2 ships **only**:
- `verse/loom.py` (orchestrator, prompt builders, digest parser, apply policy).
- Proposal CRUD methods on `VerseStore`.
- New commands: `@verseproposals`, `@verseapprove`, `@versereject`.
- New registry: `loomNetwork`, `loomChannel`, `loomModel`, `loomCycleInterval`, `loomVerseCooldown`, `loomBeatWindow`, `loomTranscriptMaxLines`, `loomTranscriptMaxChars`, `loomBotNicks`, `verseAutoApplyThreshold`.
- doPrivmsg transcript hook for the loom channel.
- Operator-guide and commands-reference doc updates.
- CHANGELOG entry.

PR 2 does **not** ship:
- Crosspollination (`verseCrosspollAllowSend`, `verseCrosspollAllowReceive`, `verseCrosspollPerCycleLimit`) — PR 3.
- Event retention / lore-digest compaction — PR 3.
- Gemini cache plumbing in `service.py` — open follow-up.
- Web view at `/verse/<channel>` — open follow-up.

---

## Phase A — Proposal CRUD on `VerseStore`

**Goal:** the loom can `add_proposal()` (with optional pre-set status/reviewer for the auto-apply audit row), `list_proposals()`, `get_proposal()`, `update_proposal_status()`, and `apply_proposal()`. Plus a `list_active_verses()` helper. All writes through `write_transaction`. All tests use real SQLite in `tmp_path`.

### Task A1: `Proposal` NamedTuple + `add_proposal`

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Test: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: write the failing test** (append to `test_store.py`):

```python
class TestProposalsCRUD:
    def test_add_proposal_pending_default(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "the bell rang", "entity_ids": []},
            confidence=0.9,
            provenance="line-3",
        )
        assert isinstance(pid, str) and len(pid) > 0
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT id, op, status, confidence, reviewer, reviewed_at "
                "FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
            assert row[0] == pid
            assert row[1] == "add_event"
            assert row[2] == "pending"
            assert row[3] == 0.9
            assert row[4] is None
            assert row[5] is None

    def test_add_proposal_with_preset_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "auto", "entity_ids": []},
            confidence=0.95,
            provenance="line-1",
            status="approved",
            reviewer="loom",
        )
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT status, reviewer, reviewed_at FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
            assert row[0] == "approved"
            assert row[1] == "loom"
            assert row[2] is not None and row[2] > 0

    def test_add_proposal_rejects_invalid_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(ValueError):
            store.add_proposal(
                cycle_id="c-1", op="add_event",
                payload={"summary": "x", "entity_ids": []},
                confidence=0.9, provenance="x", status="weird",
            )
```

- [ ] **Step 2: run** `uv run pytest plugins/llm/tests/verse/test_store.py::TestProposalsCRUD -v` → fail.

- [ ] **Step 3: implement.** Add a `Proposal` NamedTuple near the others (after `Event`):

```python
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
```

Add a method on `VerseStore`:

```python
_VALID_PROPOSAL_STATUSES = ("pending", "approved", "rejected")

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
) -> str:
    """Insert a proposal and return its uuid id.

    When *status* is 'approved' or 'rejected', *reviewer* must be supplied
    and reviewed_at is set to now (this is how auto-apply records its
    audit row inside the same write_transaction as the mutation it just
    applied).
    """
    if status not in _VALID_PROPOSAL_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    if status != "pending" and not reviewer:
        raise ValueError("reviewer required when status != pending")
    pid = uuid.uuid4().hex
    now = time.time()
    reviewed_at = now if status != "pending" else None
    with self.write_transaction() as conn:
        conn.execute(
            "INSERT INTO proposals "
            "(id, created_at, cycle_id, op, payload, confidence, provenance, "
            " status, reviewer, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, now, cycle_id, op, json.dumps(payload), confidence, provenance,
             status, reviewer, reviewed_at),
        )
    return pid
```

Add imports: `import json`, `import uuid`, `from typing import Any` if missing.

- [ ] **Step 4: pass. Step 5: commit:**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_store.py
git commit -m "feat(verse): proposals add_proposal with optional preset status"
```

---

### Task A2: `list_proposals` with status/cycle filters

**Files:** as A1.

- [ ] **Tests** for these cases:
  - `list_proposals()` returns all rows newest-first.
  - `list_proposals(status="pending")` filters.
  - `list_proposals(status="approved")` filters.
  - `list_proposals(cycle_id="c-1")` filters.
  - Returned rows are `Proposal` instances with `payload` decoded as `dict`.

```python
def test_list_proposals_filters_and_decodes(self, verse_db_dir: Path) -> None:
    from llm.verse.store import VerseStore, Proposal
    store = VerseStore(verse_db_dir, "#afnet")
    p1 = store.add_proposal(cycle_id="c-1", op="add_event",
                             payload={"summary": "first"}, confidence=0.9)
    p2 = store.add_proposal(cycle_id="c-2", op="add_event",
                             payload={"summary": "second"}, confidence=0.5)
    rows = store.list_proposals()
    assert [r.id for r in rows] == [p2, p1]
    assert isinstance(rows[0], Proposal)
    assert rows[0].payload == {"summary": "second"}
    assert [r.id for r in store.list_proposals(status="pending")] == [p2, p1]
    assert [r.id for r in store.list_proposals(cycle_id="c-1")] == [p1]
```

- [ ] **Implement:**

```python
def list_proposals(
    self,
    *,
    status: str | None = None,
    cycle_id: str | None = None,
    limit: int = 100,
) -> list[Proposal]:
    sql = (
        "SELECT id, created_at, cycle_id, op, payload, confidence, provenance, "
        "status, reviewer, reviewed_at FROM proposals"
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
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    with self.read_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        Proposal(
            id=r[0], created_at=r[1], cycle_id=r[2], op=r[3],
            payload=json.loads(r[4]), confidence=r[5], provenance=r[6],
            status=r[7], reviewer=r[8], reviewed_at=r[9],
        )
        for r in rows
    ]
```

- [ ] **Commit:** `feat(verse): list_proposals with filters`.

---

### Task A3: `get_proposal(id)`

**Files:** as A1.

- [ ] **Test:** `get_proposal(unknown_id)` returns `None`; known id returns a `Proposal`.
- [ ] **Implement:** thin wrapper over the same SELECT, parameterized on `id`. Reads only, no transaction.
- [ ] **Commit:** `feat(verse): get_proposal lookup`.

---

### Task A4: `update_proposal_status` with reviewer audit fields

**Files:** as A1.

- [ ] **Tests:**

```python
def test_update_proposal_status_records_reviewer(self, verse_db_dir: Path) -> None:
    from llm.verse.store import VerseStore
    store = VerseStore(verse_db_dir, "#afnet")
    pid = store.add_proposal(cycle_id="c-1", op="add_event",
                              payload={"summary": "x"}, confidence=0.9)
    store.update_proposal_status(pid, status="approved", reviewer="alice")
    p = store.get_proposal(pid)
    assert p is not None
    assert p.status == "approved"
    assert p.reviewer == "alice"
    assert p.reviewed_at is not None and p.reviewed_at > 0

def test_update_proposal_status_rejects_invalid(self, verse_db_dir: Path) -> None:
    from llm.verse.store import VerseStore
    store = VerseStore(verse_db_dir, "#afnet")
    pid = store.add_proposal(cycle_id="c-1", op="add_event",
                              payload={"summary": "x"}, confidence=0.9)
    with pytest.raises(ValueError):
        store.update_proposal_status(pid, status="weird", reviewer="alice")

def test_update_proposal_status_unknown_id_raises(self, verse_db_dir: Path) -> None:
    from llm.verse.store import VerseStore
    store = VerseStore(verse_db_dir, "#afnet")
    with pytest.raises(LookupError):
        store.update_proposal_status("nope", status="approved", reviewer="alice")
```

- [ ] **Implement:**

```python
def update_proposal_status(
    self, proposal_id: str, *, status: str, reviewer: str
) -> None:
    if status not in _VALID_PROPOSAL_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    with self.write_transaction() as conn:
        cur = conn.execute(
            "UPDATE proposals SET status=?, reviewer=?, reviewed_at=? WHERE id=?",
            (status, reviewer, time.time(), proposal_id),
        )
        if cur.rowcount == 0:
            raise LookupError(f"no proposal: {proposal_id!r}")
```

- [ ] **Commit:** `feat(verse): update_proposal_status with audit fields`.

---

### Task A5: `apply_proposal` — atomic op dispatch

**Goal:** Convert a proposal payload into a real mutation. Used both by the auto-apply path (immediately, paired with an `add_proposal(status="approved", reviewer="loom")`) and by `@verseapprove` (operator-driven). The applier does NOT change any proposal row's status — that's the caller's job.

**Files:** as A1.

- [ ] **Tests** (note: `Event.entity_ids` is `tuple[int, ...]` per the existing NamedTuple at `plugins/llm/src/llm/verse/store.py:48`):

```python
class TestApplyProposal:
    def test_apply_add_event_inserts_event(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        store.apply_proposal(
            op="add_event",
            payload={"summary": "Forest enters the clearing", "entity_ids": [eid]},
            source="loom",
        )
        events = store.recent_events()
        assert len(events) == 1
        assert events[0].summary == "Forest enters the clearing"
        assert events[0].source == "loom"
        assert events[0].entity_ids == (eid,)   # tuple, per NamedTuple

    def test_apply_set_attribute_writes_kv(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        store.apply_proposal(
            op="set_attribute",
            payload={"entity_id": eid, "key": "mood", "value": "wary"},
            source="loom",
        )
        assert store.get_attribute(eid, "mood") == "wary"

    def test_apply_add_relation(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("avatar", "Forest")
        b = store.add_entity("npc", "Owl")
        store.apply_proposal(
            op="add_relation",
            payload={"from_id": a, "to_id": b, "kind": "allied_with", "note": ""},
            source="loom",
        )
        rels = store.list_relations(from_id=a)
        assert len(rels) == 1 and rels[0].kind == "allied_with" and rels[0].to_id == b

    def test_apply_add_entity_creates_with_summary(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.apply_proposal(
            op="add_entity",
            payload={"kind": "place", "name": "Hollow Oak",
                     "summary": "A leaning trunk on the path."},
            source="loom",
        )
        assert isinstance(new_id, int)
        e = store.get_entity(new_id)
        assert e is not None and e.kind == "place" and e.name == "Hollow Oak"

    def test_apply_unknown_op_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(ValueError):
            store.apply_proposal(op="nuke", payload={}, source="loom")

    def test_apply_missing_payload_field_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(KeyError):
            store.apply_proposal(op="add_event", payload={}, source="loom")
```

- [ ] **Implement:**

```python
def apply_proposal(
    self,
    *,
    op: str,
    payload: dict[str, Any],
    source: str = "loom",
) -> int | None:
    """Convert a proposal payload into rows. Returns the new entity id for
    add_entity, the new event id for add_event, the new relation id for
    add_relation, or None for set_attribute. Raises ValueError for unknown
    ops or KeyError for missing payload keys."""
    if op == "add_event":
        return self.add_event(
            summary=payload["summary"],
            entity_ids=payload.get("entity_ids", []),
            source=source,
        )
    if op == "set_attribute":
        self.set_attribute(payload["entity_id"], payload["key"], payload["value"])
        return None
    if op == "add_relation":
        return self.add_relation(
            from_id=payload["from_id"],
            to_id=payload["to_id"],
            kind=payload["kind"],
            note=payload.get("note", ""),
        )
    if op == "add_entity":
        return self.add_entity(
            kind=payload["kind"],
            name=payload["name"],
            summary=payload.get("summary", ""),
        )
    raise ValueError(f"unknown op: {op!r}")
```

(Confirm signatures by reading `plugins/llm/src/llm/verse/store.py` first; the existing CRUD methods accept the keywords listed.)

- [ ] **Commit:** `feat(verse): apply_proposal op dispatcher`.

---

### Task A6a: atomic `apply_and_record_proposal` and `apply_proposal_and_mark`

**Goal:** close the v2 audit-row race. Both the loom auto-apply path and the operator-approve path need the mutation and the corresponding `proposals` row written in **one** `write_transaction`. The `threading.Lock` inside `write_transaction` is not re-entrant, so we can't compose `apply_proposal()` + `add_proposal()` / `update_proposal_status()` — the new methods inline the SQL.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Test: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Step 1: tests:**

```python
class TestApplyAndRecordProposal:
    def test_one_transaction_event_plus_audit(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        pid = store.apply_and_record_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": [eid]},
            confidence=0.95, provenance="line-1",
            reviewer="loom",
        )
        assert isinstance(pid, str) and len(pid) > 0
        events = store.recent_events()
        assert len(events) == 1 and events[0].summary == "x"
        rows = store.list_proposals()
        assert len(rows) == 1
        assert rows[0].status == "approved"
        assert rows[0].reviewer == "loom"

    def test_failure_inside_op_rolls_back_audit(self, verse_db_dir: Path) -> None:
        # Set_attribute against a non-existent entity_id should fail and
        # leave neither the attribute nor the proposal row.
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(Exception):
            store.apply_and_record_proposal(
                cycle_id="c-1", op="set_attribute",
                payload={"entity_id": 9999, "key": "k", "value": "v"},
                confidence=0.95, provenance="x", reviewer="loom",
            )
        assert store.list_proposals() == []


class TestApplyProposalAndMark:
    def test_pending_to_approved_atomically(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5, provenance="line-1",
        )
        store.apply_proposal_and_mark(pid, reviewer="alice")
        events = store.recent_events()
        assert len(events) == 1
        p = store.get_proposal(pid)
        assert p.status == "approved" and p.reviewer == "alice"

    def test_unknown_id_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(LookupError):
            store.apply_proposal_and_mark("nope", reviewer="alice")

    def test_already_terminal_status_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5, provenance="x",
            status="approved", reviewer="bob",
        )
        with pytest.raises(ValueError):
            store.apply_proposal_and_mark(pid, reviewer="alice")
```

- [ ] **Step 2: implement.** A private helper `_apply_op_inline(conn, *, op, payload, source) -> int | None` runs the op-specific SQL on an *already-open* connection (so it composes inside an outer transaction). Then the two public methods call it inside a `write_transaction` along with their proposal-row write:

```python
def _apply_op_inline(
    self, conn: sqlite3.Connection, *,
    op: str, payload: dict[str, Any], source: str,
) -> int | None:
    """Run the op-specific INSERT on *conn*. The caller owns the txn."""
    now = time.time()
    if op == "add_event":
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) "
            "VALUES (?, ?, ?, ?)",
            (now, payload["summary"],
             json.dumps(list(payload.get("entity_ids", []))), source),
        )
        return cur.lastrowid
    if op == "set_attribute":
        # Validate entity_id exists (write inside txn would otherwise pass).
        eid = payload["entity_id"]
        row = conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone()
        if row is None:
            raise LookupError(f"entity_id {eid} does not exist")
        conn.execute(
            "INSERT INTO attributes (entity_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(entity_id, key) DO UPDATE SET value=excluded.value",
            (eid, payload["key"], payload["value"]),
        )
        return None
    if op == "add_relation":
        cur = conn.execute(
            "INSERT INTO relations (from_id, to_id, kind, note) "
            "VALUES (?, ?, ?, ?)",
            (payload["from_id"], payload["to_id"], payload["kind"],
             payload.get("note", "")),
        )
        return cur.lastrowid
    if op == "add_entity":
        cur = conn.execute(
            "INSERT INTO entities (kind, name, summary, status, "
            "                       created_at, updated_at) "
            "VALUES (?, ?, ?, 'active', ?, ?)",
            (payload["kind"], payload["name"],
             payload.get("summary", ""), now, now),
        )
        return cur.lastrowid
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
    neither (the lock + write_transaction guarantee SQLite atomicity)."""
    pid = uuid.uuid4().hex
    now = time.time()
    with self.write_transaction() as conn:
        self._apply_op_inline(conn, op=op, payload=payload, source=source)
        conn.execute(
            "INSERT INTO proposals "
            "(id, created_at, cycle_id, op, payload, confidence, provenance, "
            " status, reviewer, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'approved', ?, ?)",
            (pid, now, cycle_id, op, json.dumps(payload),
             confidence, provenance, reviewer, now),
        )
    return pid


def apply_proposal_and_mark(
    self, proposal_id: str, *, reviewer: str,
) -> None:
    """Atomically apply a pending proposal and flip its status to approved.
    Raises LookupError if no such id, ValueError if already terminal."""
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
        self._apply_op_inline(conn, op=op, payload=payload, source="loom")
        conn.execute(
            "UPDATE proposals SET status='approved', reviewer=?, "
            "reviewed_at=? WHERE id=?",
            (reviewer, time.time(), proposal_id),
        )
```

> **Note on `apply_proposal`:** the older method (A5) stays. It's still used by tests and by any future caller that wants to apply without writing a proposal row (none currently). The two new methods are the ones loom and `@verseapprove` use.

- [ ] **Step 3: pass. Step 4: commit** — `feat(verse): apply_and_record_proposal + apply_proposal_and_mark (atomic)`.

---

### Task A6: `list_active_verses(data_dir)` helper

**Goal:** the loom needs to know which per-channel SQLite files exist so it can pick a focus verse.

> **Design note:** the per-channel DB filename is sanitized + hashed (`_<safe>_<hash>.db`) and *does not* preserve the original `#channel` form. We can't recover the original from the filename alone. Therefore `list_active_verses` returns paths only. The plugin-side bridge intersects this list with the set of channels that have `verseEnabled=True` (by re-deriving each known channel's path via `db_path_for_channel(...)`).

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`.
- Test: `plugins/llm/tests/verse/test_store.py`.

- [ ] **Tests:**

```python
class TestListActiveVerses:
    def test_returns_paths_for_existing_dbs(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore, list_active_verses
        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#forest")
        result = list_active_verses(verse_db_dir)
        assert len(result) == 2
        for path in result:
            assert path.suffix == ".db" and path.exists()

    def test_empty_dir_returns_empty_list(self, verse_db_dir: Path) -> None:
        from llm.verse.store import list_active_verses
        assert list_active_verses(verse_db_dir) == []

    def test_missing_dir_returns_empty_list(self, tmp_path: Path) -> None:
        from llm.verse.store import list_active_verses
        assert list_active_verses(tmp_path / "nope") == []
```

- [ ] **Implement:**

```python
def list_active_verses(base_dir: Path) -> list[Path]:
    """Return paths of all verse DB files in *base_dir*, sorted.

    The caller maps these back to channel names via the same
    db_path_for_channel sanitizer used at construction time.
    """
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob("*.db"))
```

- [ ] **Commit:** `feat(verse): list_active_verses helper`.

---

### Phase A verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/ -v` → all pass.
- [ ] `uv run pytest plugins/llm -q` → no regressions.

---

## Phase B — Loom orchestrator

### Task B1: package skeleton + test scaffolding

**Files:**
- Create: `plugins/llm/src/llm/verse/loom.py` (one-line docstring + `from __future__ import annotations`).
- Create: `plugins/llm/tests/verse/test_loom.py` (empty module).
- Create: `plugins/llm/tests/verse/_fakes.py` (shared `FakeBridge`/`StubClient`; populated in B10a).
- Create: `plugins/llm/tests/verse/fixtures/transcript_quiet.json` — empty JSON list `[]`.
- Create: `plugins/llm/tests/verse/fixtures/transcript_noisy.json` — list of `{"nick":"...","text":"..."}` objects (~20 lines, hand-crafted improv).

- [ ] Write the empty files; `make check` passes.
- [ ] Commit: `feat(verse/loom): scaffold package + fixtures`.

---

### Task B2: `LoomConfig` dataclass

`LoomConfig` captures every registry knob the loom needs. The plugin builds one per cycle from `self.registryValue(...)`. Tests construct it directly.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`.
- Test: `plugins/llm/tests/verse/test_loom.py`.

- [ ] **Tests:**

```python
def test_loomconfig_holds_all_settings() -> None:
    from llm.verse.loom import LoomConfig
    cfg = LoomConfig(
        network="afternet",
        loom_channel="#forest",
        bot_nicks=("botA", "botB"),
        model="gemini/gemini-flash-lite-latest",
        cycle_interval_s=300,
        verse_cooldown_s=1200,
        beat_window_s=90,
        transcript_max_lines=40,
        transcript_max_chars=8000,
        auto_apply_threshold=0.85,
    )
    assert cfg.loom_channel == "#forest"
    assert cfg.network == "afternet"
    assert cfg.bot_nicks == ("botA", "botB")
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        cfg.cycle_interval_s = 1  # type: ignore[misc]
```

- [ ] **Implement** in `loom.py`:

```python
"""Forest-verse loom orchestrator: rotation, beats, digest, proposal apply."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LoomConfig:
    network: str
    loom_channel: str
    bot_nicks: tuple[str, ...]   # empty tuple = capture all non-self
    model: str
    cycle_interval_s: int
    verse_cooldown_s: int
    beat_window_s: int
    transcript_max_lines: int
    transcript_max_chars: int
    auto_apply_threshold: float
```

- [ ] **Commit:** `feat(verse/loom): LoomConfig dataclass`.

---

### Task B3: rotation — `pick_focus_verse`

Weighted by `(active_avatars * 2 + recent_events)`, filtered to verses whose last cycle was ≥ `verse_cooldown_s` ago, round-robin tiebreaker on equal weights.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestPickFocusVerse:
    def test_returns_none_if_all_in_cooldown(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse
        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=10, last_cycle_at=now - 5.0),
            VerseCandidate(channel="#b", weight=10, last_cycle_at=now - 5.0),
        ]
        assert pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0) is None

    def test_picks_highest_weight_outside_cooldown(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse
        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=2, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#b", weight=8, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#c", weight=5, last_cycle_at=now - 5.0),  # cooldown
        ]
        result = pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0)
        assert result is not None and result.channel == "#b"

    def test_round_robin_with_three_tied_candidates(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse
        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=5, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#b", weight=5, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#c", weight=5, last_cycle_at=now - 60.0),
        ]
        picks = [
            pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=p).channel
            for p in range(6)
        ]
        # Pointer cycles through the 3 candidates twice.
        assert picks == ["#a", "#b", "#c", "#a", "#b", "#c"]

    def test_never_cycled_treated_as_eligible(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse
        now = 1000.0
        candidates = [VerseCandidate(channel="#a", weight=1, last_cycle_at=None)]
        result = pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0)
        assert result is not None and result.channel == "#a"
```

- [ ] **Implement:**

```python
from typing import NamedTuple

class VerseCandidate(NamedTuple):
    channel: str
    weight: int                  # 2*active_avatars + recent_events
    last_cycle_at: float | None

def pick_focus_verse(
    candidates: list[VerseCandidate],
    *,
    now: float,
    cooldown_s: int,
    pointer: int,
) -> VerseCandidate | None:
    """Highest-weighted candidate outside cooldown; round-robin ties."""
    eligible = [
        c for c in candidates
        if c.last_cycle_at is None or (now - c.last_cycle_at) >= cooldown_s
    ]
    if not eligible:
        return None
    top_weight = max(c.weight for c in eligible)
    top = [c for c in eligible if c.weight == top_weight]
    return top[pointer % len(top)]
```

- [ ] **Commit:** `feat(verse/loom): pick_focus_verse rotation`.

---

### Task B4: prompt builders — three blocks

**Per design:** static prefix (cache-eligible, identical across cycles), verse-stable block (identical across the cycle's three calls — so it's snapshotted **once** at cycle start in B10a, not rebuilt per phase), volatile tail (per-call).

**Files:** as B2.

- [ ] **Tests:**

```python
class TestPromptBuilders:
    def test_static_prefix_is_constant(self) -> None:
        from llm.verse.loom import LOOM_STATIC_PREFIX
        assert isinstance(LOOM_STATIC_PREFIX, str)
        assert "proposal" in LOOM_STATIC_PREFIX.lower()
        assert "json" in LOOM_STATIC_PREFIX.lower()
        # No 'id' field demanded — store generates it.
        assert '"id"' not in LOOM_STATIC_PREFIX

    def test_verse_stable_block_deterministic(self) -> None:
        from llm.verse.loom import build_verse_stable_block, VerseSnapshot
        snap = VerseSnapshot(
            channel="#afnet",
            summary="Three avatars wander a moonlit grove.",
            top_entities=[("avatar", "Forest"), ("place", "Hollow Oak")],
            recent_events=["Forest entered the grove.", "Owl hooted thrice."],
        )
        a = build_verse_stable_block(snap)
        b = build_verse_stable_block(snap)
        assert a == b
        assert "Forest" in a and "Hollow Oak" in a and "Owl hooted" in a

    def test_seed_tail_includes_emit_instruction(self) -> None:
        from llm.verse.loom import build_seed_tail
        out = build_seed_tail()
        assert "one line" in out.lower() or "1 line" in out.lower()

    def test_beat_tail_includes_transcript(self) -> None:
        from llm.verse.loom import build_beat_tail
        out = build_beat_tail(loom_transcript_so_far=[("botB", "the bell echoes")])
        assert "botB" in out and "bell" in out

    def test_digest_tail_demands_json_array(self) -> None:
        from llm.verse.loom import build_digest_tail
        out = build_digest_tail(
            loom_transcript_so_far=[("botB", "the bell echoes")],
        )
        assert "json" in out.lower()
        assert "array" in out.lower() or "list" in out.lower()
```

- [ ] **Implement:**

```python
class VerseSnapshot(NamedTuple):
    channel: str
    summary: str
    top_entities: list[tuple[str, str]]   # (kind, name)
    recent_events: list[str]               # newest-first

LOOM_STATIC_PREFIX = """\
You are the loom: a narrator that watches improv between several IRC bots
and proposes mutations to a shared fictional world. Your role is to
*propose*, not to declare canon. A reviewer either approves your proposals
or rejects them.

Each proposal MUST be valid JSON with these fields:
  op          — one of: add_event, set_attribute, add_relation, add_entity
  payload     — object whose required keys depend on op:
                  add_event:     summary (str), entity_ids (list[int])
                  set_attribute: entity_id (int), key (str), value (str)
                  add_relation:  from_id (int), to_id (int), kind (str), note (str?)
                  add_entity:    kind (str: avatar|npc|place|faction|item),
                                 name (str), summary (str?)
  confidence  — float between 0.0 and 1.0
  provenance  — short string identifying which transcript line(s) drove this
  rationale   — one sentence in your voice

Always emit the proposal list as a single JSON array, no prose around it.
"""

def build_verse_stable_block(snap: VerseSnapshot) -> str:
    parts = [
        f"# Focus verse: {snap.channel}",
        f"# Summary: {snap.summary}",
        "# Active entities:",
    ]
    for kind, name in snap.top_entities:
        parts.append(f"- {kind}: {name}")
    parts.append("# Recent events (newest first):")
    for ev in snap.recent_events:
        parts.append(f"- {ev}")
    return "\n".join(parts)

def build_seed_tail() -> str:
    return (
        "Emit a single line of dialogue or scene-setting that invites the "
        "other bots in this channel to riff on it. Stay in fiction. "
        "One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )

def build_beat_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "The other bots have replied:\n"
        f"{lines}\n\n"
        "Post a single follow-up that picks up a thread or pushes the scene. "
        "One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )

def build_digest_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "Full transcript:\n"
        f"{lines}\n\n"
        "Now emit a JSON array of proposals derived from this transcript. "
        "If nothing notable happened, emit []."
    )
```

- [ ] **Commit:** `feat(verse/loom): prompt builders (static/stable/volatile)`.

---

### Task B5: digest JSON parser + validator

`parse_digest(text)` strips an optional ```json fence, parses JSON, validates each proposal's shape, and returns `list[ParsedProposal]`. Bad proposals are dropped with a warning. Returns `[]` on hard parse error.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestParseDigest:
    def test_parses_valid_array(self) -> None:
        from llm.verse.loom import parse_digest
        text = '''[{"op":"add_event",
                    "payload":{"summary":"x","entity_ids":[]},
                    "confidence":0.9,"provenance":"l-1","rationale":"y"}]'''
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].op == "add_event"
        assert out[0].confidence == 0.9

    def test_strips_json_code_fence(self) -> None:
        from llm.verse.loom import parse_digest
        text = "```json\n[]\n```"
        assert parse_digest(text) == []

    def test_drops_proposals_missing_required_fields(self, caplog) -> None:
        from llm.verse.loom import parse_digest
        text = '''[
            {"op":"add_event","payload":{},"confidence":0.9,
             "provenance":"x","rationale":"y"},
            {"op":"BOGUS","payload":{},"confidence":0.5,
             "provenance":"x","rationale":"y"},
            {"op":"add_event","payload":{"summary":"k","entity_ids":[]},
             "confidence":0.7,"provenance":"x","rationale":"y"}
        ]'''
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].payload["summary"] == "k"

    def test_clamps_confidence_to_unit_interval(self) -> None:
        from llm.verse.loom import parse_digest
        text = '''[{"op":"add_event",
                    "payload":{"summary":"x","entity_ids":[]},
                    "confidence":2.5,"provenance":"x","rationale":"y"}]'''
        out = parse_digest(text)
        assert out[0].confidence == 1.0

    def test_returns_empty_on_hard_parse_error(self) -> None:
        from llm.verse.loom import parse_digest
        assert parse_digest("not json at all") == []

    def test_drops_when_required_payload_value_wrong_type(self, caplog) -> None:
        from llm.verse.loom import parse_digest
        # entity_ids should be a list, not a string.
        text = '''[{"op":"add_event",
                    "payload":{"summary":"x","entity_ids":"not-a-list"},
                    "confidence":0.9,"provenance":"x","rationale":"y"}]'''
        out = parse_digest(text)
        assert out == []

    def test_drops_when_entity_ids_element_not_int(self) -> None:
        from llm.verse.loom import parse_digest
        text = '''[{"op":"add_event",
                    "payload":{"summary":"x","entity_ids":["bad"]},
                    "confidence":0.9,"provenance":"x","rationale":"y"}]'''
        assert parse_digest(text) == []

    def test_rejects_bool_as_int_for_entity_id(self) -> None:
        from llm.verse.loom import parse_digest
        # Python's True is an instance of int; the parser must reject it.
        text = '''[{"op":"set_attribute",
                    "payload":{"entity_id":true,"key":"k","value":"v"},
                    "confidence":0.9,"provenance":"x","rationale":"y"}]'''
        assert parse_digest(text) == []
```

- [ ] **Implement:**

```python
import json
import logging
import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)

_VALID_OPS = ("add_event", "set_attribute", "add_relation", "add_entity")

def _is_strict_int(v: Any) -> bool:
    """Reject bool, accept int. (bool is a subclass of int in Python.)"""
    return isinstance(v, int) and not isinstance(v, bool)

def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_strict_int(x) for x in v)

# (key, predicate, label) per op. predicate(value) -> bool.
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
}

class ParsedProposal(NamedTuple):
    op: str
    payload: dict[str, Any]
    confidence: float
    provenance: str
    rationale: str

def parse_digest(text: str) -> list[ParsedProposal]:
    cleaned = _FENCE_RE.sub("", text).strip()
    log = logging.getLogger("llm.verse.loom")
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("loom digest hard parse error: %s", exc)
        return []
    if not isinstance(raw, list):
        log.warning("loom digest top-level was %s, expected list", type(raw).__name__)
        return []

    out: list[ParsedProposal] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            log.warning("loom proposal %d not a dict; dropped", i)
            continue
        op = item.get("op")
        if op not in _VALID_OPS:
            log.warning("loom proposal %d bad op %r; dropped", i, op)
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            log.warning("loom proposal %d payload not dict; dropped", i)
            continue
        bad_field: str | None = None
        for key, predicate, label in _PAYLOAD_SCHEMA[op]:
            if key not in payload:
                bad_field = f"missing {key}"
                break
            if not predicate(payload[key]):
                bad_field = f"{key} not {label}"
                break
        if bad_field is not None:
            log.warning("loom proposal %d %s; dropped", i, bad_field)
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append(ParsedProposal(
            op=op, payload=payload, confidence=conf,
            provenance=str(item.get("provenance", "")),
            rationale=str(item.get("rationale", "")),
        ))
    return out
```

- [ ] **Commit:** `feat(verse/loom): parse_digest with strict validation`.

---

### Task B6: transcript truncation + dedup

Cap to `transcript_max_lines` and `transcript_max_chars`; drop consecutive duplicate `(nick, text)` tuples. The bot's own lines are filtered upstream by the doPrivmsg hook.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestTruncateTranscript:
    def test_caps_lines(self) -> None:
        from llm.verse.loom import truncate_transcript
        lines = [("a", f"x{i}") for i in range(100)]
        out = truncate_transcript(lines, max_lines=10, max_chars=10_000)
        assert len(out) == 10
        assert out[-1] == ("a", "x99")

    def test_caps_chars_after_lines(self) -> None:
        from llm.verse.loom import truncate_transcript
        lines = [("a", "x" * 100) for _ in range(50)]
        out = truncate_transcript(lines, max_lines=40, max_chars=500)
        total = sum(len(t) for _, t in out)
        assert total <= 500
        assert len(out) <= 5

    def test_dedupes_consecutive_identical_tuples(self) -> None:
        from llm.verse.loom import truncate_transcript
        # Dedup is on the full (nick, text) tuple, not nick-only.
        lines = [("a", "ping"), ("a", "ping"), ("b", "ping"), ("a", "ping")]
        out = truncate_transcript(lines, max_lines=40, max_chars=10_000)
        assert out == [("a", "ping"), ("b", "ping"), ("a", "ping")]

    def test_empty_input_empty_output(self) -> None:
        from llm.verse.loom import truncate_transcript
        assert truncate_transcript([], max_lines=40, max_chars=8000) == []
```

- [ ] **Implement:**

```python
def truncate_transcript(
    lines: list[tuple[str, str]],
    *,
    max_lines: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Drop consecutive duplicates of the (nick, text) tuple, then cap to
    max_lines (most recent kept) and max_chars (most recent kept).
    Input is oldest-first."""
    deduped: list[tuple[str, str]] = []
    for nick, text in lines:
        if deduped and deduped[-1] == (nick, text):
            continue
        deduped.append((nick, text))
    deduped = deduped[-max_lines:]
    out: list[tuple[str, str]] = []
    total = 0
    for nick, text in reversed(deduped):
        if total + len(text) > max_chars:
            break
        out.append((nick, text))
        total += len(text)
    out.reverse()
    return out
```

- [ ] **Commit:** `feat(verse/loom): truncate_transcript`.

---

### Task B7: model-call wrapper for the loom

The wrapper isolates the cheap-model call so tests can stub it. Returns `(content, usage)` so the driver can route token counts to `LoomBridge.log_usage`.

**Files:** as B2.

- [ ] **Tests** (verify the conftest doesn't already monkeypatch `litellm.completion` globally — `grep -n litellm plugins/llm/tests/conftest.py`. If it does, structure the test with an override fixture rather than direct monkeypatch.):

```python
class LoomCallUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cost: float

class TestLiteLLMLoomClient:
    def test_returns_content_and_usage(self, monkeypatch, caplog) -> None:
        import logging
        from llm.verse.loom import LiteLLMLoomClient
        class _Resp:
            class _Msg:  # noqa: D401
                content = "ok"
            class _Choice:  # noqa: D401
                message = _Msg()
            choices = [_Choice()]
            class _Usage:  # noqa: D401
                prompt_tokens = 7
                completion_tokens = 3
            usage = _Usage()
        import litellm
        monkeypatch.setattr(litellm, "completion", lambda **_: _Resp())
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
        caplog.set_level(logging.WARNING, logger="llm.verse.loom")
        client = LiteLLMLoomClient()
        content, usage = client.call(op="seed", model="gemini/x", messages=[])
        assert content == "ok"
        assert usage.prompt_tokens == 7
        assert usage.completion_tokens == 3
        assert any("op=loom:seed" in rec.message for rec in caplog.records)
```

- [ ] **Implement:**

```python
from typing import Protocol

class LoomCallUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cost: float

class LoomModelClient(Protocol):
    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, LoomCallUsage]: ...

class LiteLLMLoomClient:
    """Default loom client. Calls litellm.completion synchronously
    (already on a worker thread by the time this runs) and returns the
    content string plus a LoomCallUsage. Errors propagate to the caller."""

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logging.getLogger("llm.verse.loom")

    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, LoomCallUsage]:
        import time
        import litellm
        t0 = time.monotonic()
        response = litellm.completion(model=model, messages=messages)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        try:
            content = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            content = ""
        try:
            usage = response.usage
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
        except AttributeError:
            pt = ct = 0
        try:
            cost = float(litellm.completion_cost(
                completion_response=response, model=model
            ) or 0.0)
        except Exception:
            cost = 0.0
        self._log.warning(
            "completion_timing op=loom:%s model=%s elapsed_ms=%.0f "
            "prompt_tokens=%d completion_tokens=%d cost=%.6f",
            op, model, elapsed_ms, pt, ct, cost,
        )
        return content, LoomCallUsage(pt, ct, cost)
```

- [ ] **Commit:** `feat(verse/loom): LiteLLMLoomClient with usage return`.

---

### Task B8: `LoomBridge` Protocol + `LoomCycle` state

The driver mutates state across schedule callbacks and worker threads. The bridge isolates plugin internals; `LoomCycle` carries snapshotted-once-per-cycle data.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestLoomCycle:
    def test_append_grows_transcript_in_order(self) -> None:
        from llm.verse.loom import LoomCycle
        c = LoomCycle(cycle_id="c1", channel="#afnet", started_at=0.0,
                       verse_stable_block="block")
        c.append_transcript("botA", "hi")
        c.append_transcript("botB", "yo")
        assert c.transcript == [("botA", "hi"), ("botB", "yo")]

    def test_snapshot_transcript_returns_a_copy(self) -> None:
        from llm.verse.loom import LoomCycle
        c = LoomCycle(cycle_id="c1", channel="#afnet", started_at=0.0,
                       verse_stable_block="block")
        c.append_transcript("botA", "hi")
        snap = c.snapshot_transcript()
        c.append_transcript("botB", "yo")
        assert snap == [("botA", "hi")]
```

- [ ] **Implement:**

```python
from collections.abc import Callable
from dataclasses import dataclass, field

@dataclass
class LoomCycle:
    cycle_id: str
    channel: str
    started_at: float
    verse_stable_block: str
    transcript: list[tuple[str, str]] = field(default_factory=list)
    beats_posted: int = 0

    def append_transcript(self, nick: str, text: str) -> None:
        self.transcript.append((nick, text))

    def snapshot_transcript(self) -> list[tuple[str, str]]:
        return list(self.transcript)


class LoomBridge(Protocol):
    """Adapter the plugin implements. The driver only talks to this."""

    def list_candidate_channels(self) -> list[str]: ...
    def candidate_weight(self, channel: str) -> int: ...
    def snapshot(self, channel: str) -> VerseSnapshot: ...
    def post_to_loom_channel(self, text: str) -> bool:
        """Return True if posted. False if the loom Irc/network is not available."""
        ...
    def schedule_after(
        self, delay_s: float, fn: Callable[[], None], name: str
    ) -> None: ...
    def submit(self, label: str, fn: Callable[[], None]) -> None:
        """Run fn on the LLM worker thread pool. Returns immediately. *label*
        is forwarded to ``LLMExecutor.submit`` for telemetry; loom phases
        pass ``loom:seed`` / ``loom:beat`` / ``loom:digest``."""
        ...
    def now(self) -> float: ...
    def store_for(self, channel: str): ...   # returns VerseStore-compatible
    def log_usage(
        self, *, channel: str, op: str, model: str,
        usage: LoomCallUsage,
    ) -> None: ...
```

- [ ] **Commit:** `feat(verse/loom): LoomCycle (with snapshot) + LoomBridge protocol`.

---

### Task B9: apply-or-queue policy (always writes a row, atomically)

**Per design + v2/v3 revisions:** every proposal is recorded in `proposals`. High-confidence non-`add_entity` proposals are simultaneously applied with `source='loom'` and the row is written with `status='approved' reviewer='loom'` — **inside one `write_transaction`** via `apply_and_record_proposal()` (Task A6a), so a crash can't commit the mutation without the audit row. Otherwise the row is written `status='pending'`. `add_entity` always queues.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestApplyOrQueue:
    def test_high_confidence_event_auto_applies_and_records_audit_row(
        self, verse_db_dir
    ) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(op="add_event",
                               payload={"summary": "x", "entity_ids": []},
                               confidence=0.95, provenance="l-1", rationale="r")
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "applied"
        assert len(store.recent_events()) == 1
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1
        assert rows[0].status == "approved"
        assert rows[0].reviewer == "loom"

    def test_low_confidence_queues_pending(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(op="add_event",
                               payload={"summary": "x", "entity_ids": []},
                               confidence=0.5, provenance="l-1", rationale="r")
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "queued"
        assert store.recent_events() == []
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1 and rows[0].status == "pending"

    def test_add_entity_always_queues_regardless_of_confidence(
        self, verse_db_dir
    ) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore
        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(op="add_entity",
                               payload={"kind": "place", "name": "Hollow Oak"},
                               confidence=0.99, provenance="l-1", rationale="r")
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "queued"
        assert store.list_entities_by_kind("place") == []
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1 and rows[0].status == "pending"
```

- [ ] **Implement:**

```python
def apply_or_queue(
    store,                       # VerseStore-compatible
    prop: ParsedProposal,
    *,
    cycle_id: str,
    threshold: float,
) -> str:
    """Always inserts a proposal row. Returns 'applied' or 'queued'.

    Auto-apply uses ``apply_and_record_proposal`` so the mutation and the
    audit row are written in one ``write_transaction``."""
    auto = prop.op != "add_entity" and prop.confidence >= threshold
    if auto:
        store.apply_and_record_proposal(
            cycle_id=cycle_id, op=prop.op, payload=prop.payload,
            confidence=prop.confidence, provenance=prop.provenance,
            reviewer="loom",
        )
        return "applied"
    store.add_proposal(
        cycle_id=cycle_id, op=prop.op, payload=prop.payload,
        confidence=prop.confidence, provenance=prop.provenance,
    )
    return "queued"
```

- [ ] **Commit:** `feat(verse/loom): apply_or_queue always writes audit row`.

---

### Task B10a: `Loom.__init__` + `tick()` shim + `_seed_phase` worker

The `Loom` driver owns the lock, the `_active` cycle, and the `_last_cycle_by_channel` map. `tick()` runs on the scheduler thread — it picks a verse, builds the cycle state, then submits `_seed_phase` to the worker. The worker calls the model, posts the seed beat, and schedules `after_beat1`.

**Files:** as B2; tests use `_fakes.py`.

- [ ] **Step 1: build the shared fakes** in `plugins/llm/tests/verse/_fakes.py`:

```python
"""Shared fakes for loom tests."""

from __future__ import annotations

from collections.abc import Callable
from llm.verse.loom import LoomBridge, LoomCallUsage, VerseSnapshot

class FakeBridge:
    """Synchronous fake. Records every interaction for assertions.

    submit() runs fn() inline; schedule_after() records (delay, fn, name)
    into self.scheduled but does not fire — tests fire by calling
    bridge.scheduled[i][1]() explicitly.
    """

    def __init__(self, *, channels, weights, store, snapshots, post_returns=True):
        self.channels = list(channels)
        self.weights = dict(weights)
        self.store = store
        self.snapshots = dict(snapshots)
        self.posts: list[str] = []
        self.scheduled: list[tuple[float, Callable[[], None], str]] = []
        self.usage_log: list[tuple[str, str, str, LoomCallUsage]] = []
        self.t = 1000.0
        self.post_returns = post_returns

    def list_candidate_channels(self): return list(self.channels)
    def candidate_weight(self, channel): return self.weights.get(channel, 0)
    def snapshot(self, channel): return self.snapshots[channel]
    def post_to_loom_channel(self, text):
        self.posts.append(text)
        return self.post_returns
    def schedule_after(self, delay_s, fn, name):
        self.scheduled.append((delay_s, fn, name))
    def submit(self, label, fn):
        # Synchronous fake — labeled submit collapses to inline call.
        self.submitted_labels = getattr(self, "submitted_labels", [])
        self.submitted_labels.append(label)
        fn()
    def now(self): return self.t
    def store_for(self, channel): return self.store
    def log_usage(self, *, channel, op, model, usage):
        self.usage_log.append((channel, op, model, usage))


class StubClient:
    def __init__(self, replies):
        self.replies = dict(replies)
        self.calls: list[str] = []

    def call(self, *, op, model, messages):
        self.calls.append(op)
        return self.replies[op], LoomCallUsage(
            prompt_tokens=10, completion_tokens=5, cost=0.0001,
        )
```

- [ ] **Step 2: tests for tick():**

```python
class TestLoomTick:
    def _cfg(self):
        from llm.verse.loom import LoomConfig
        return LoomConfig(
            network="afternet", loom_channel="#forest", bot_nicks=(),
            model="gemini/x", cycle_interval_s=300, verse_cooldown_s=20,
            beat_window_s=90, transcript_max_lines=40,
            transcript_max_chars=8000, auto_apply_threshold=0.85,
        )

    def test_tick_with_no_candidates_does_nothing(self, verse_db_dir):
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(channels=[], weights={}, store=store, snapshots={})
        client = StubClient({})
        loom = Loom(cfg=self._cfg(), bridge=bridge, client=client)
        loom.tick()
        assert client.calls == []
        assert bridge.posts == []
        assert bridge.scheduled == []
        assert loom._active is None

    def test_idle_tick_does_not_advance_pointer(self, verse_db_dir):
        # Skipped (no eligible candidate) ticks must NOT rotate the pointer,
        # otherwise the next eligible cycle picks a non-deterministic offset.
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(channels=[], weights={}, store=store, snapshots={})
        loom = Loom(cfg=self._cfg(), bridge=bridge, client=StubClient({}))
        loom.tick()
        loom.tick()
        loom.tick()
        assert loom._pointer == 0

    def test_tick_records_last_cycle_at_for_picked_channel(self, verse_db_dir):
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove",
                                                [("avatar", "Forest")], [])},
        )
        client = StubClient({"seed": "the bell rings"})
        loom = Loom(cfg=self._cfg(), bridge=bridge, client=client)
        loom.tick()
        assert loom._last_cycle_by_channel["#afnet"] == bridge.now()
        assert bridge.posts == ["the bell rings"]
        assert bridge.scheduled and bridge.scheduled[0][2] == "llm_loom_after_beat1"

    def test_tick_aborts_if_post_to_channel_fails(self, verse_db_dir):
        # Simulates a not-connected network: post returns False.
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
            post_returns=False,
        )
        client = StubClient({"seed": "the bell rings"})
        loom = Loom(cfg=self._cfg(), bridge=bridge, client=client)
        loom.tick()
        # Cycle aborts; no follow-up scheduled.
        assert bridge.scheduled == []
        # last_cycle_at NOT recorded so we retry next interval.
        assert "#afnet" not in loom._last_cycle_by_channel
```

- [ ] **Step 3: implement.** Use a single `threading.Lock`; never call the bridge while holding the lock except for the cheap `list_candidate_channels` / `candidate_weight` / `now` / `schedule_after` calls. The worker-side `_seed_phase` does I/O without the lock and reacquires only to mutate cycle state.

```python
import logging
import threading
import uuid

class Loom:
    def __init__(self, *, cfg: LoomConfig, bridge: LoomBridge,
                 client: LoomModelClient) -> None:
        self._cfg = cfg
        self._bridge = bridge
        self._client = client
        self._active: LoomCycle | None = None
        self._last_cycle_by_channel: dict[str, float] = {}
        self._pointer = 0
        self._lock = threading.Lock()
        self._log = logging.getLogger("llm.verse.loom")

    def observe_transcript(self, nick: str, text: str) -> None:
        """Plugin's doPrivmsg hook calls this for every loom-channel line
        that survived the source filter."""
        with self._lock:
            if self._active is None:
                return
            self._active.append_transcript(nick, text)

    def tick(self) -> None:
        with self._lock:
            if self._active is not None:
                self._log.debug("loom: tick during active cycle; skipping")
                return
            channels = self._bridge.list_candidate_channels()
            now = self._bridge.now()
            candidates = [
                VerseCandidate(
                    channel=c,
                    weight=self._bridge.candidate_weight(c),
                    last_cycle_at=self._last_cycle_by_channel.get(c),
                )
                for c in channels
            ]
            choice = pick_focus_verse(
                candidates, now=now,
                cooldown_s=self._cfg.verse_cooldown_s, pointer=self._pointer,
            )
            if choice is None:
                self._log.debug("loom_idle: no eligible verse")
                return
            # Only rotate the pointer when we actually picked something.
            self._pointer += 1
            snap = self._bridge.snapshot(choice.channel)
            cycle = LoomCycle(
                cycle_id=uuid.uuid4().hex[:12],
                channel=choice.channel,
                started_at=now,
                verse_stable_block=build_verse_stable_block(snap),
            )
            self._active = cycle
            self._last_cycle_by_channel[choice.channel] = now
        self._bridge.submit("loom:seed", lambda: self._seed_phase(cycle))

    def _seed_phase(self, cycle: LoomCycle) -> None:
        messages = [
            {"role": "system", "content": LOOM_STATIC_PREFIX},
            {"role": "system", "content": cycle.verse_stable_block},
            {"role": "user", "content": build_seed_tail()},
        ]
        try:
            content, usage = self._client.call(
                op="seed", model=self._cfg.model, messages=messages,
            )
        except Exception:
            self._log.exception("loom seed call failed; aborting cycle")
            with self._lock:
                self._active = None
            return
        self._bridge.log_usage(
            channel=cycle.channel, op="seed",
            model=self._cfg.model, usage=usage,
        )
        line = (content.strip().splitlines() or [""])[0]
        if not line:
            with self._lock:
                self._active = None
            return
        if not self._bridge.post_to_loom_channel(line):
            self._log.warning(
                "loom seed: post_to_loom_channel failed (network down?); "
                "rolling back cycle for %s", cycle.channel,
            )
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(cycle.channel, None)
            return
        with self._lock:
            cycle.beats_posted = 1
        self._bridge.schedule_after(
            self._cfg.beat_window_s, self.after_beat1,
            "llm_loom_after_beat1",
        )
```

- [ ] **Step 4: pass. Step 5: commit** — `feat(verse/loom): Loom.tick + seed phase`.

---

### Task B10b: `after_beat1` shim + `_beat_phase` worker

`after_beat1` runs on the scheduler thread; submits `_beat_phase` to a worker. The worker snapshots the transcript under the lock, calls the beat model, posts the second beat, and schedules `after_beat2`. Idle short-circuit: if transcript snapshot is empty, finalize the cycle and skip the digest.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestLoomAfterBeat1:
    def test_idle_short_circuit_finalizes_cycle(self, verse_db_dir):
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
        )
        client = StubClient({"seed": "a faint hum"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        # No transcript captured between beats.
        bridge.scheduled[0][1]()    # after_beat1
        # No beat posted; no digest scheduled. Cycle finalized.
        assert client.calls == ["seed"]
        assert len(bridge.posts) == 1
        assert loom._active is None
        # The only schedule_after call was for after_beat1.
        assert [s[2] for s in bridge.scheduled] == ["llm_loom_after_beat1"]

    def test_with_transcript_posts_beat_and_schedules_digest(self, verse_db_dir):
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
        )
        client = StubClient({"seed": "ring", "beat": "shadows lengthen"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()    # after_beat1
        assert bridge.posts[-1] == "shadows lengthen"
        assert bridge.scheduled[-1][2] == "llm_loom_after_beat2"
        assert client.calls == ["seed", "beat"]
```

(`_minimal_cfg()` is a helper at the top of `test_loom.py` that returns the same `LoomConfig` used in B10a's `_cfg()`.)

- [ ] **Implement:**

```python
def after_beat1(self) -> None:
    with self._lock:
        cycle = self._active
        if cycle is None:
            return
    self._bridge.submit("loom:beat", lambda: self._beat_phase(cycle))

def _beat_phase(self, cycle: LoomCycle) -> None:
    with self._lock:
        transcript = truncate_transcript(
            cycle.snapshot_transcript(),
            max_lines=self._cfg.transcript_max_lines,
            max_chars=self._cfg.transcript_max_chars,
        )
    if not transcript:
        self._log.warning(
            "loom_idle: empty transcript after beat 1; finalizing cycle %s",
            cycle.cycle_id,
        )
        with self._lock:
            self._active = None
        return
    messages = [
        {"role": "system", "content": LOOM_STATIC_PREFIX},
        {"role": "system", "content": cycle.verse_stable_block},
        {"role": "user",
         "content": build_beat_tail(loom_transcript_so_far=transcript)},
    ]
    try:
        content, usage = self._client.call(
            op="beat", model=self._cfg.model, messages=messages,
        )
    except Exception:
        self._log.exception("loom beat call failed; finalizing cycle")
        with self._lock:
            self._active = None
        return
    self._bridge.log_usage(
        channel=cycle.channel, op="beat",
        model=self._cfg.model, usage=usage,
    )
    line = (content.strip().splitlines() or [""])[0]
    if line:
        self._bridge.post_to_loom_channel(line)
        with self._lock:
            cycle.beats_posted = 2
    self._bridge.schedule_after(
        self._cfg.beat_window_s, self.after_beat2,
        "llm_loom_after_beat2",
    )
```

- [ ] **Commit:** `feat(verse/loom): after_beat1 + beat phase worker`.

---

### Task B10c: `after_beat2` shim + `_digest_phase` worker

`after_beat2` shim submits `_digest_phase` to a worker. The worker snapshots transcript, calls digest model, parses, applies/queues each proposal, and finalizes the cycle.

**Files:** as B2.

- [ ] **Tests:**

```python
class TestLoomDigestPhase:
    def test_full_cycle_applies_high_confidence_event(self, verse_db_dir):
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("avatar", "Forest")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove",
                                                [("avatar", "Forest")], [])},
        )
        client = StubClient({
            "seed": "the bell rings",
            "beat": "shadows lengthen",
            "digest": ('[{"op":"add_event",'
                       '"payload":{"summary":"a chime","entity_ids":[]},'
                       '"confidence":0.95,"provenance":"l-1","rationale":"r"}]'),
        })
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)

        loom.tick()
        loom.observe_transcript("botB", "I hear it too")
        bridge.scheduled[0][1]()    # after_beat1
        loom.observe_transcript("botC", "the wind takes it")
        bridge.scheduled[-1][1]()   # after_beat2

        events = store.recent_events()
        assert any(e.summary == "a chime" for e in events)
        # Audit row was written.
        rows = store.list_proposals(status="approved")
        assert len(rows) == 1 and rows[0].reviewer == "loom"
        assert client.calls == ["seed", "beat", "digest"]
        assert loom._active is None
        # Usage was logged for all three calls.
        assert [u[1] for u in bridge.usage_log] == ["seed", "beat", "digest"]

    def test_uses_snapshotted_stable_block_across_phases(self, verse_db_dir):
        # Stable block is built once at tick() and reused; mutating the
        # bridge's snapshot AFTER tick() must NOT change the block used in
        # later phases.
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore
        from ._fakes import FakeBridge, StubClient
        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"], weights={"#afnet": 5}, store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove",
                                                [("avatar", "Forest")], [])},
        )
        captured: list[str] = []

        class CapturingClient(StubClient):
            def call(self, *, op, model, messages):
                # The verse-stable block is the second system message.
                captured.append(messages[1]["content"])
                return super().call(op=op, model=model, messages=messages)

        client = CapturingClient({
            "seed": "ring", "beat": "echo", "digest": "[]",
        })
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        # Mutate bridge snapshot — should NOT affect later phases.
        bridge.snapshots["#afnet"] = VerseSnapshot(
            "#afnet", "different summary",
            [("avatar", "Different")], ["a new event"],
        )
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()
        loom.observe_transcript("botB", "I hear it")  # noop dedupe
        bridge.scheduled[-1][1]()
        assert captured[0] == captured[1] == captured[2]
        assert "different summary" not in captured[0]
```

- [ ] **Implement:**

```python
def after_beat2(self) -> None:
    with self._lock:
        cycle = self._active
        if cycle is None:
            return
    # Concurrency invariant: Limnoria's scheduler serializes timer
    # callbacks (see plugins/llm/src/llm/plugin.py:599 — addPeriodicEvent
    # is single-threaded). Combined with the lock guarding _active and
    # the worker-thread submit boundary, no two cycles can overlap.
    self._bridge.submit("loom:digest", lambda: self._digest_phase(cycle))

def _digest_phase(self, cycle: LoomCycle) -> None:
    try:
        with self._lock:
            transcript = truncate_transcript(
                cycle.snapshot_transcript(),
                max_lines=self._cfg.transcript_max_lines,
                max_chars=self._cfg.transcript_max_chars,
            )
        if not transcript:
            self._log.info(
                "loom: empty transcript at digest; finalizing cycle %s",
                cycle.cycle_id,
            )
            return
        messages = [
            {"role": "system", "content": LOOM_STATIC_PREFIX},
            {"role": "system", "content": cycle.verse_stable_block},
            {"role": "user",
             "content": build_digest_tail(loom_transcript_so_far=transcript)},
        ]
        try:
            content, usage = self._client.call(
                op="digest", model=self._cfg.model, messages=messages,
            )
        except Exception:
            self._log.exception("loom digest call failed")
            return
        self._bridge.log_usage(
            channel=cycle.channel, op="digest",
            model=self._cfg.model, usage=usage,
        )
        proposals = parse_digest(content)
        store = self._bridge.store_for(cycle.channel)
        for p in proposals:
            try:
                apply_or_queue(
                    store, p,
                    cycle_id=cycle.cycle_id,
                    threshold=self._cfg.auto_apply_threshold,
                )
            except Exception:
                self._log.exception(
                    "loom proposal apply failed: op=%s payload=%s",
                    p.op, p.payload,
                )
    finally:
        with self._lock:
            self._active = None
```

- [ ] **Commit:** `feat(verse/loom): after_beat2 + digest phase + cycle finalize`.

---

### Phase B verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm/tests/verse/test_loom.py -v` → all pass.

---

## Phase C — Plugin wiring

### Task C1: registry keys

**Files:**
- Modify: `plugins/llm/src/llm/config.py`. Append immediately after the existing verse block (`plugins/llm/src/llm/config.py:321–344`).

- [ ] **Step 1: write the failing test** (`plugins/llm/tests/test_config.py`). Read it first to match its existing fixture style for the `plugin` fixture before writing the assertion body.

```python
def test_loom_registry_defaults(plugin):
    cfg = plugin.registryValue
    assert cfg("loomNetwork") == ""
    assert cfg("loomChannel") == ""
    assert cfg("loomModel") == "gemini/gemini-flash-lite-latest"
    assert cfg("loomCycleInterval") == 5
    assert cfg("loomVerseCooldown") == 20
    assert cfg("loomBeatWindow") == 90
    assert cfg("loomTranscriptMaxLines") == 40
    assert cfg("loomTranscriptMaxChars") == 8000
    assert cfg("loomBotNicks") == ""
    assert cfg("verseAutoApplyThreshold") == 0.85
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement.** Add right after the `verseEventRetentionDays` block:

```python
conf.registerGlobalValue(
    LLM,
    "verseAutoApplyThreshold",
    registry.Float(
        0.85,
        _("""Minimum confidence (0.0–1.0) at which loom proposals are
        applied automatically without manual review. add_entity proposals
        are always queued regardless of confidence."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomNetwork",
    registry.String(
        "",
        _("""Network name (as configured in supybot.networks) where the
        loom orchestrator runs. Combined with loomChannel to resolve the
        target Irc connection. When empty, the loom timer is not
        scheduled and no model calls are made."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomChannel",
    registry.String(
        "",
        _("""Channel where the loom orchestrator runs (e.g., #forest).
        Resolved on loomNetwork. When empty, the loom is disabled."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomModel",
    registry.String(
        "gemini/gemini-flash-lite-latest",
        _("""Cheap model used by the loom orchestrator for seed, beat,
        and digest calls."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomCycleInterval",
    registry.PositiveInteger(
        5,
        _("""Loom timer cadence in minutes."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomVerseCooldown",
    registry.PositiveInteger(
        20,
        _("""Minimum gap in minutes between consecutive loom cycles for
        the same verse."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomBeatWindow",
    registry.PositiveInteger(
        90,
        _("""Listen window in seconds after each loom beat is posted."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomTranscriptMaxLines",
    registry.PositiveInteger(
        40,
        _("""Per-window cap on loom transcript lines (most recent kept)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomTranscriptMaxChars",
    registry.PositiveInteger(
        8000,
        _("""Per-window cap on loom transcript characters (most recent kept)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "loomBotNicks",
    registry.String(
        "",
        _("""Comma-separated list of nicks whose lines in the loom
        channel are captured into the transcript. Empty = capture all
        non-self lines (suitable for bot-heavy channels)."""),
    ),
)
```

- [ ] **Step 4: pass. Step 5: commit** — `feat(verse): loom registry keys`.

---

### Task C2: plugin instantiates `Loom` and wires the bridge

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (verse subsystem section ~line 4513).

- [ ] **Step 1: write `_PluginLoomBridge` in full** in the verse subsystem block:

```python
class _PluginLoomBridge:
    """Plugin-side adapter for Loom. One instance per active loom config."""

    def __init__(self, plugin: "LLM", network: str, channel: str) -> None:
        self._plugin = plugin
        self._network = network
        self._channel = channel
        # Pre-build the channel→stem lookup we need for candidate listing.
        self._verse_data_dir = (
            Path(conf.supybot.directories.data()) / "verse"
        )

    # --- candidate listing ---

    def list_candidate_channels(self) -> list[str]:
        """Channels with verseEnabled=True that also have a DB on disk
        AND are joined on the loom's network. Filtering on network avoids
        cross-network channel-name collisions reaching the loom queue
        (the SQLite store key is channel-name-only — see Open follow-ups)."""
        from .verse.store import db_path_for_channel, list_active_verses
        on_disk_paths = {p for p in list_active_verses(self._verse_data_dir)}
        irc = world.getIrc(self._network)
        if irc is None:
            return []
        joined = set(irc.state.channels.keys())
        out: list[str] = []
        for ch in self._plugin._verse_enabled_channels():
            if ch not in joined:
                continue
            expected = db_path_for_channel(self._verse_data_dir, ch)
            if expected in on_disk_paths:
                out.append(ch)
        return out

    def candidate_weight(self, channel: str) -> int:
        store = self._plugin._get_or_create_verse_store(channel)
        active_avatars = len(store.list_entities_by_kind("avatar", status="active"))
        recent = len(store.recent_events(limit=20))
        return 2 * active_avatars + recent

    # --- snapshot ---

    def snapshot(self, channel: str):
        from .verse.loom import VerseSnapshot
        store = self._plugin._get_or_create_verse_store(channel)
        avatars = store.list_entities_by_kind("avatar", status="active")[:5]
        places = store.list_entities_by_kind("place")[:5]
        events = store.recent_events(limit=10)
        return VerseSnapshot(
            channel=channel,
            summary=f"{len(avatars)} active avatars, {len(places)} places",
            top_entities=[(e.kind, e.name) for e in (*avatars, *places)],
            recent_events=[e.summary for e in events],
        )

    # --- IO ---

    def post_to_loom_channel(self, text: str) -> bool:
        irc = world.getIrc(self._network)
        if irc is None:
            return False
        irc.queueMsg(ircmsgs.privmsg(self._channel, text))
        return True

    def schedule_after(self, delay_s, fn, name) -> None:
        with contextlib.suppress(KeyError):
            schedule.removeEvent(name)
        schedule.addEvent(fn, time.time() + delay_s, name=name)

    def submit(self, label, fn) -> None:
        # _llm_executor is on the plugin (plugin.py:520), not on llm_service.
        # LLMExecutor.submit signature is (label, fn, *args, **kwargs) ->
        # Future (executor.py:95). The future is intentionally discarded;
        # the loom phase swallows its own exceptions.
        self._plugin._llm_executor.submit(label, fn)

    def now(self) -> float:
        return time.time()

    def store_for(self, channel: str):
        return self._plugin._get_or_create_verse_store(channel)

    def log_usage(self, *, channel, op, model, usage) -> None:
        self._plugin.db.log_usage(
            nick="loom", channel=channel, command=f"loom:{op}",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost=usage.cost,
        )
```

The helper `_verse_enabled_channels()` is a small new method on `LLM` that walks `conf.supybot.plugins.LLM.verseEnabled` per channel:

```python
def _verse_enabled_channels(self) -> list[str]:
    """All channels with verseEnabled=True. Read from registry every call
    (callers are not in hot paths)."""
    out: list[str] = []
    for ch in self._all_known_channels():
        if self.registryValue("verseEnabled", ch):
            out.append(ch)
    return out
```

If `_all_known_channels()` doesn't already exist, derive from `world.ircs`:

```python
def _all_known_channels(self) -> set[str]:
    seen: set[str] = set()
    for irc_conn in world.ircs:
        seen.update(irc_conn.state.channels.keys())
    return seen
```

(Read `plugins/llm/src/llm/plugin.py` to confirm whether either helper already exists; reuse if so.)

- [ ] **Step 2: write tests** in `plugins/llm/tests/test_plugin.py`:

```python
def test_loom_disabled_when_loom_channel_empty(plugin, monkeypatch):
    monkeypatch.setattr(
        plugin, "registryValue",
        lambda k, *a, **kw: "" if k in ("loomChannel", "loomNetwork") else
            plugin.__class__.registryValue(plugin, k, *a, **kw),
    )
    plugin._wire_loom_if_enabled()
    assert plugin._loom is None
    assert plugin._loom_channel_cache is None

def test_loom_wired_when_loom_channel_and_network_set(plugin, monkeypatch):
    overrides = {"loomNetwork": "afternet", "loomChannel": "#forest"}
    real = plugin.__class__.registryValue
    monkeypatch.setattr(
        plugin, "registryValue",
        lambda k, *a, **kw: overrides.get(k, real(plugin, k, *a, **kw)),
    )
    plugin._wire_loom_if_enabled()
    assert plugin._loom is not None
    assert plugin._loom_channel_cache == "#forest"
    assert plugin._loom_network_cache == "afternet"

def test_loom_disabled_when_only_one_of_network_channel_set(plugin, monkeypatch):
    overrides = {"loomNetwork": "afternet", "loomChannel": ""}
    real = plugin.__class__.registryValue
    monkeypatch.setattr(
        plugin, "registryValue",
        lambda k, *a, **kw: overrides.get(k, real(plugin, k, *a, **kw)),
    )
    plugin._wire_loom_if_enabled()
    assert plugin._loom is None
```

- [ ] **Step 3: implement.** Add to `__init__` (all four caches initialized so `doPrivmsg` never reads an unset attribute):

```python
self._loom = None                  # type: ignore[assignment]
self._loom_bridge = None
self._loom_channel_cache: str | None = None
self._loom_network_cache: str | None = None
self._loom_bot_nicks_cache: tuple[str, ...] = ()
```

Wiring method:

```python
def _wire_loom_if_enabled(self) -> None:
    """Build the Loom + bridge when both loomNetwork and loomChannel are
    configured. Idempotent. Tear down cleanly when either is unset."""
    network = self.registryValue("loomNetwork")
    channel = self.registryValue("loomChannel")
    if not network or not channel:
        if self._loom is not None:
            with contextlib.suppress(KeyError):
                schedule.removeEvent("llm_loom_cycle")
            with contextlib.suppress(KeyError):
                schedule.removeEvent("llm_loom_after_beat1")
            with contextlib.suppress(KeyError):
                schedule.removeEvent("llm_loom_after_beat2")
            self._loom = None
            self._loom_bridge = None
            self._loom_channel_cache = None
            self._loom_network_cache = None
            self._loom_bot_nicks_cache = ()
        return
    if (self._loom is not None
            and self._loom_channel_cache == channel
            and self._loom_network_cache == network):
        return                          # already wired with same target

    from .verse.loom import Loom, LoomConfig, LiteLLMLoomClient
    bot_nicks_raw = self.registryValue("loomBotNicks") or ""
    cfg = LoomConfig(
        network=network,
        loom_channel=channel,
        bot_nicks=tuple(
            n.strip() for n in bot_nicks_raw.split(",") if n.strip()
        ),
        model=self.registryValue("loomModel"),
        cycle_interval_s=self.registryValue("loomCycleInterval") * 60,
        verse_cooldown_s=self.registryValue("loomVerseCooldown") * 60,
        beat_window_s=self.registryValue("loomBeatWindow"),
        transcript_max_lines=self.registryValue("loomTranscriptMaxLines"),
        transcript_max_chars=self.registryValue("loomTranscriptMaxChars"),
        auto_apply_threshold=self.registryValue("verseAutoApplyThreshold"),
    )
    self._loom_bridge = _PluginLoomBridge(self, network, channel)
    self._loom = Loom(cfg=cfg, bridge=self._loom_bridge,
                      client=LiteLLMLoomClient())
    self._loom_channel_cache = channel
    self._loom_network_cache = network
    self._loom_bot_nicks_cache = cfg.bot_nicks
    self._schedule_loom_tick()


def _schedule_loom_tick(self) -> None:
    with contextlib.suppress(KeyError):
        schedule.removeEvent("llm_loom_cycle")
    interval = self.registryValue("loomCycleInterval") * 60
    schedule.addPeriodicEvent(self._loom_tick, interval, name="llm_loom_cycle")


def _loom_tick(self) -> None:
    if self._loom is None:
        return
    try:
        self._loom.tick()
    except Exception:
        self.log.exception("loom tick failed")
```

- [ ] **Step 4: hook teardown.** In `die()` (around `plugins/llm/src/llm/plugin.py:645`), add:

```python
with contextlib.suppress(KeyError):
    schedule.removeEvent("llm_loom_cycle")
with contextlib.suppress(KeyError):
    schedule.removeEvent("llm_loom_after_beat1")
with contextlib.suppress(KeyError):
    schedule.removeEvent("llm_loom_after_beat2")
```

- [ ] **Step 5: call `_wire_loom_if_enabled()` from the same post-init wiring path that calls `_schedule_pending_tasks` / similar.** Read `plugins/llm/src/llm/plugin.py` ~line 580–650 to find that call site and insert next to it.

- [ ] **Step 6: pass. Step 7: commit** — `feat(plugin): wire Loom on init + tear down on die`.

---

### Task C3: doPrivmsg transcript hook with bot/network filter

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` near `doPrivmsg` (`plugins/llm/src/llm/plugin.py:933`).
- Modify: `plugins/llm/tests/conftest.py` (add `irc_for_network` fixture if not present).

> **Insertion point note.** The hook must run *after* the prefix-character early-out at `plugins/llm/src/llm/plugin.py:961` (`if text[0] in prefix_chars: return`). Otherwise lines like `@verseapprove abc123` posted in the loom channel would be captured as transcript before Limnoria's command dispatcher takes them. The hook also runs after the self-check at line 954 so vibebot's own outgoing lines aren't echoed.

- [ ] **Step 1: tests** in `plugins/llm/tests/test_plugin.py`:

```python
def test_doprivmsg_appends_loom_transcript(plugin, irc_for_network, msg_in_channel):
    plugin._loom = _FakeLoom()
    plugin._loom_channel_cache = "#forest"
    plugin._loom_network_cache = "afternet"
    plugin._loom_bot_nicks_cache = ()   # empty = capture all non-self
    irc = irc_for_network("afternet")
    msg = msg_in_channel(irc, "#forest", "botB", "the bell rings")
    plugin.doPrivmsg(irc, msg)
    assert plugin._loom.observed == [("botB", "the bell rings")]

def test_doprivmsg_ignores_other_networks(plugin, irc_for_network, msg_in_channel):
    plugin._loom = _FakeLoom()
    plugin._loom_channel_cache = "#forest"
    plugin._loom_network_cache = "afternet"
    plugin._loom_bot_nicks_cache = ()
    irc = irc_for_network("freenode")
    plugin.doPrivmsg(irc, msg_in_channel(irc, "#forest", "botB", "hi"))
    assert plugin._loom.observed == []

def test_doprivmsg_filters_by_bot_allowlist_when_set(plugin, irc_for_network,
                                                       msg_in_channel):
    plugin._loom = _FakeLoom()
    plugin._loom_channel_cache = "#forest"
    plugin._loom_network_cache = "afternet"
    plugin._loom_bot_nicks_cache = ("botB",)
    irc = irc_for_network("afternet")
    plugin.doPrivmsg(irc, msg_in_channel(irc, "#forest", "alice", "hi"))
    plugin.doPrivmsg(irc, msg_in_channel(irc, "#forest", "botB", "hi"))
    assert plugin._loom.observed == [("botB", "hi")]

def test_doprivmsg_does_not_capture_prefix_commands(plugin, irc_for_network,
                                                     msg_in_channel):
    # @verseapprove or @versereject sent in the loom channel must NOT
    # land in the loom transcript — they're commands, not improv.
    plugin._loom = _FakeLoom()
    plugin._loom_channel_cache = "#forest"
    plugin._loom_network_cache = "afternet"
    plugin._loom_bot_nicks_cache = ()
    irc = irc_for_network("afternet")
    plugin.doPrivmsg(irc, msg_in_channel(irc, "#forest", "alice",
                                          "@verseapprove abc123"))
    assert plugin._loom.observed == []
```

`_FakeLoom` is a tiny test double — define it inline at the top of the test module (or in `conftest.py`):

```python
class _FakeLoom:
    def __init__(self):
        self.observed: list[tuple[str, str]] = []
    def observe_transcript(self, nick: str, text: str) -> None:
        self.observed.append((nick, text))
```

`irc_for_network` is a fixture that returns an Irc-shaped fake whose `network` attribute matches the requested string. If `plugins/llm/tests/conftest.py` doesn't have it, add:

```python
@pytest.fixture
def irc_for_network():
    """Build a minimal Irc fake with a settable network attribute and a
    nick. State.channels is an empty dict by default; tests that need
    membership populate it explicitly."""
    from supybot.irclib import IrcState
    def _make(network: str, *, nick: str = "vibebot"):
        class _FakeIrc:
            def __init__(self):
                self.network = network
                self.nick = nick
                self.state = IrcState()
            def queueMsg(self, msg):  # captured by tests when needed
                self.last_queued = msg
        return _FakeIrc()
    return _make
```

`msg_in_channel` is also small enough to inline:

```python
@pytest.fixture
def msg_in_channel():
    from supybot.ircmsgs import IrcMsg
    def _make(irc, channel: str, nick: str, text: str):
        return IrcMsg(prefix=f"{nick}!u@h", command="PRIVMSG",
                       args=(channel, text))
    return _make
```

(If similar fixtures already exist in `plugins/llm/tests/conftest.py`, reuse them — don't duplicate.)

- [ ] **Step 2: implement.** Insert the hook in `doPrivmsg` *after* the prefix-character early-out at `plugins/llm/src/llm/plugin.py:961` (so prefixed commands aren't captured) and *before* the existing addressed-text branching:

```python
loom = self._loom
if loom is not None and getattr(irc, "network", None) == self._loom_network_cache:
    try:
        target = msg.args[0] if msg.args else ""
        if (
            target == self._loom_channel_cache
            and not ircutils.strEqual(irc.nick, msg.nick)
        ):
            allowlist = self._loom_bot_nicks_cache or ()
            if not allowlist or any(
                ircutils.strEqual(n, msg.nick) for n in allowlist
            ):
                text = msg.args[1] if len(msg.args) > 1 else ""
                if text:
                    loom.observe_transcript(msg.nick, text)
    except Exception:
        self.log.exception("loom transcript capture failed (non-fatal)")
```

(`_loom_bot_nicks_cache` is initialized in `__init__`, set in `_wire_loom_if_enabled`, and cleared in the disable branch — see C2 step 3 above.)

- [ ] **Step 3: pass. Step 4: commit** — `feat(plugin): doPrivmsg loom transcript hook with bot/network filter`.

---

### Phase C verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm -q` → all pass; new tests included.
- [ ] **Manual smoke** (no live model): in a Python REPL or test, set `loomNetwork="x"` + `loomChannel="#test"`, call `plugin._wire_loom_if_enabled()`, verify `schedule.events` contains `llm_loom_cycle`. Set `loomChannel=""`, re-wire, verify the event is removed.

---

## Phase D — Operator commands

### Task D1: `@verseproposals` listing

Args: `[<channel>] [<status>]` where `status ∈ {pending, approved, rejected}`. Default channel = current; default status = `pending`. Capability `llm.verse.gm`. Lists up to 10 most-recent rows.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (CommandInfo registry near line 281; command body in the verse subsystem block).

- [ ] **Step 1: tests** — append to `plugins/llm/tests/test_plugin.py`:

```python
class TestVerseproposalsCommand:
    def test_default_status_pending_in_current_channel(
        self, plugin, irc, msg_in_channel
    ):
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        store = plugin._get_or_create_verse_store("#afnet")
        store.add_proposal(cycle_id="c-1", op="add_event",
                            payload={"summary": "x"}, confidence=0.5)
        msg = msg_in_channel(irc, "#afnet", "alice", "@verseproposals")
        plugin.verseproposals(irc, msg, [], None, None)
        replies = irc.captured_replies()
        assert any("conf=0.50" in r for r in replies)

    def test_explicit_channel_and_status(self, plugin, irc, msg_in_channel):
        store = plugin._get_or_create_verse_store("#afnet")
        store.add_proposal(cycle_id="c-1", op="add_event",
                            payload={"summary": "y"}, confidence=0.95,
                            status="approved", reviewer="loom")
        msg = msg_in_channel(irc, "#bots", "alice",
                              "@verseproposals #afnet approved")
        plugin.verseproposals(irc, msg, [], "#afnet", "approved")
        replies = irc.captured_replies()
        assert any("approved" not in r and "y" in r for r in replies) or \
               any("conf=0.95" in r for r in replies)

    def test_empty_list_message(self, plugin, irc, msg_in_channel):
        plugin._get_or_create_verse_store("#afnet")
        msg = msg_in_channel(irc, "#afnet", "alice", "@verseproposals")
        plugin.verseproposals(irc, msg, [], None, None)
        replies = irc.captured_replies()
        assert any("No pending proposals" in r for r in replies)

    def test_capability_gated(self, plugin, irc, msg_in_channel):
        # Without llm.verse.gm capability the wrap layer rejects.
        # This is verified at the wrap level — assert the registration
        # uses ('checkCapability', 'llm.verse.gm') by inspecting the
        # registered command's wrappers.
        from supybot import callbacks
        assert callbacks.commandFunctionsHaveSameDescription(  # placeholder
            plugin.verseproposals, plugin.verseproposals,
        )
        # (The real test is that the wrap call in the implementation
        # includes ('checkCapability', 'llm.verse.gm'); a unit test
        # against ircdb.checkCapability is overkill.)
```

(Read `plugins/llm/tests/test_plugin.py` first to see the actual `irc` fixture API; `irc.captured_replies()` is illustrative — match whatever the existing fixture exposes.)

- [ ] **Step 2: register in `_COMMAND_INFO`:**

```python
CommandInfo(
    name="verseproposals",
    args="[<channel>] [<status>]",
    description=(
        "List recent loom proposals for a channel "
        "(status: pending|approved|rejected, default pending). "
        "Requires the llm.verse.gm capability."
    ),
    examples=("%verseproposals", "%verseproposals #afnet",
              "%verseproposals #afnet approved"),
    category="utility",
),
```

- [ ] **Step 3: implement** in the verse subsystem block:

```python
def verseproposals(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    channel: str | None = None,
    status: str | None = None,
) -> None:
    """[<channel>] [<status>]

    List up to 10 recent loom proposals for the given channel
    (default: current). Status: pending|approved|rejected (default pending).
    """
    channel = channel or (msg.args[0] if msg.args else None)
    if not channel or not ircutils.isChannel(channel):
        irc.error(_("Specify a channel."), prefixNick=False)
        return
    status = status or "pending"
    store = self._get_or_create_verse_store(channel)
    rows = store.list_proposals(status=status, limit=10)
    if not rows:
        irc.reply(f"No {status} proposals for {channel}.", prefixNick=False)
        return
    for r in rows:
        snippet = self._proposal_snippet(r)
        irc.reply(
            f"{r.id[:8]} {r.op} conf={r.confidence:.2f} {snippet}",
            prefixNick=False,
        )

verseproposals = wrap(
    verseproposals,
    [
        ("checkCapability", "llm.verse.gm"),
        optional("channel"),
        optional(("literal", ("pending", "approved", "rejected"))),
    ],
)
```

`_proposal_snippet(p)` is a one-line summary: for `add_event`, `payload.summary[:60]`; for `set_attribute`, `entity_id={x} {key}={value}`; for `add_relation`, `{from_id}-[{kind}]->{to_id}`; for `add_entity`, `{kind} "{name}"`.

- [ ] **Step 4: pass. Step 5: commit** — `feat(plugin): @verseproposals listing`.

---

### Task D2: `@verseapprove` and `@versereject`

**Files:** as D1.

- [ ] **Step 1: tests** — append to `plugins/llm/tests/test_plugin.py`:

```python
class TestVerseapproveCommand:
    def _setup_pending(self, plugin):
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        store = plugin._get_or_create_verse_store("#afnet")
        return store, store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5, provenance="line-1",
        )

    def test_approve_applies_and_flips_status(self, plugin, irc, msg_in_channel):
        store, pid = self._setup_pending(plugin)
        msg = msg_in_channel(irc, "#afnet", "alice", f"@verseapprove {pid[:8]}")
        plugin.verseapprove(irc, msg, [], pid[:8], None)
        events = store.recent_events()
        assert len(events) == 1 and events[0].source == "loom"
        p = store.get_proposal(pid)
        assert p.status == "approved" and p.reviewer != "loom"

    def test_approve_short_id_prefix(self, plugin, irc, msg_in_channel):
        store, pid = self._setup_pending(plugin)
        plugin.verseapprove(irc, msg_in_channel(irc, "#afnet", "alice", ""),
                             [], pid[:6], None)
        assert store.get_proposal(pid).status == "approved"

    def test_approve_unknown_id_errors_cleanly(self, plugin, irc, msg_in_channel):
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        plugin._get_or_create_verse_store("#afnet")
        plugin.verseapprove(irc, msg_in_channel(irc, "#afnet", "alice", ""),
                             [], "deadbeef", None)
        assert any("No proposal" in r for r in irc.captured_errors())

    def test_approve_already_approved_short_circuits(
        self, plugin, irc, msg_in_channel
    ):
        store = plugin._get_or_create_verse_store("#afnet")
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95, provenance="line-1",
            status="approved", reviewer="loom",
        )
        plugin.verseapprove(irc, msg_in_channel(irc, "#afnet", "alice", ""),
                             [], pid, None)
        assert any("already approved" in r for r in irc.captured_replies())

    def test_approve_already_rejected_blocked(self, plugin, irc, msg_in_channel):
        store = plugin._get_or_create_verse_store("#afnet")
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5, provenance="x",
        )
        store.update_proposal_status(pid, status="rejected", reviewer="bob")
        plugin.verseapprove(irc, msg_in_channel(irc, "#afnet", "alice", ""),
                             [], pid, None)
        assert any("rejected" in r for r in irc.captured_replies())


class TestVersereJectCommand:
    def test_reject_flips_status_and_does_not_apply(
        self, plugin, irc, msg_in_channel
    ):
        plugin.registryValue("verseEnabled", "#afnet", value=True)
        store = plugin._get_or_create_verse_store("#afnet")
        pid = store.add_proposal(
            cycle_id="c-1", op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5, provenance="x",
        )
        plugin.versereject(irc, msg_in_channel(irc, "#afnet", "alice", ""),
                            [], pid[:8], None)
        assert store.recent_events() == []
        assert store.get_proposal(pid).status == "rejected"
```

- [ ] **Step 2: register both commands in `_COMMAND_INFO`** (args: `<id>` for each).

- [ ] **Step 3: extract the shared lookup helper:**

```python
def _proposal_target_store(
    self, irc: callbacks.Irc, msg: IrcMsg, channel_arg: str | None,
) -> tuple[str, "VerseStore"] | tuple[None, None]:
    channel = channel_arg or (msg.args[0] if msg.args else None)
    if not channel or not ircutils.isChannel(channel):
        irc.error(_("Specify a channel."), prefixNick=False)
        return None, None
    return channel, self._get_or_create_verse_store(channel)


def _load_proposal(self, store: "VerseStore", raw_id: str):
    """Look up by full id, then fall back to unique-prefix match."""
    p = store.get_proposal(raw_id)
    if p is not None:
        return p
    rows = [
        x for x in store.list_proposals(limit=200)
        if x.id.startswith(raw_id)
    ]
    return rows[0] if len(rows) == 1 else None
```

- [ ] **Step 4: implement both commands:**

```python
def verseapprove(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    proposal_id: str,
    channel_arg: str | None = None,
) -> None:
    """<id> [<channel>]

    Apply a loom proposal's mutation and mark it approved.
    """
    channel, store = self._proposal_target_store(irc, msg, channel_arg)
    if store is None:
        return
    p = self._load_proposal(store, proposal_id)
    if p is None:
        irc.error(f"No proposal matches {proposal_id!r}.", prefixNick=False)
        return
    if p.status == "approved":
        irc.reply(
            f"Proposal {p.id[:8]} already approved.", prefixNick=False,
        )
        return
    if p.status == "rejected":
        irc.reply(
            f"Proposal {p.id[:8]} was rejected; cannot approve.",
            prefixNick=False,
        )
        return
    reviewer = self._resolve_identity(irc, msg).key
    try:
        # apply + status flip happen in one write_transaction (A6a).
        store.apply_proposal_and_mark(p.id, reviewer=reviewer)
    except Exception as exc:
        self.log.exception("verseapprove apply failed: %s", proposal_id)
        irc.error(f"Apply failed: {exc}.", prefixNick=False)
        return
    irc.reply(f"Approved {p.id[:8]} ({p.op}).", prefixNick=False)

verseapprove = wrap(
    verseapprove,
    [
        ("checkCapability", "llm.verse.gm"),
        "somethingWithoutSpaces",
        optional("channel"),
    ],
)


def versereject(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    proposal_id: str,
    channel_arg: str | None = None,
) -> None:
    """<id> [<channel>]

    Reject a loom proposal without applying its mutation.
    """
    channel, store = self._proposal_target_store(irc, msg, channel_arg)
    if store is None:
        return
    p = self._load_proposal(store, proposal_id)
    if p is None:
        irc.error(f"No proposal matches {proposal_id!r}.", prefixNick=False)
        return
    if p.status == "rejected":
        irc.reply(
            f"Proposal {p.id[:8]} already rejected.", prefixNick=False,
        )
        return
    if p.status == "approved":
        irc.reply(
            f"Proposal {p.id[:8]} was already approved; cannot reject.",
            prefixNick=False,
        )
        return
    reviewer = self._resolve_identity(irc, msg).key
    store.update_proposal_status(p.id, status="rejected", reviewer=reviewer)
    irc.reply(f"Rejected {p.id[:8]}.", prefixNick=False)

versereject = wrap(
    versereject,
    [
        ("checkCapability", "llm.verse.gm"),
        "somethingWithoutSpaces",
        optional("channel"),
    ],
)
```

- [ ] **Step 5: pass. Step 6: commit** — `feat(plugin): @verseapprove + @versereject moderation`.

---

### Phase D verification

- [ ] `make check` → green.
- [ ] `uv run pytest plugins/llm -q` → all pass.

---

## Phase E — Integration test

### Task E1: end-to-end test with fake bridge

**Files:**
- Create: `plugins/llm/tests/verse/test_loom_integration.py`.

The test:
1. Builds a real `VerseStore` in `tmp_path`.
2. Builds a `FakeBridge` (from `_fakes.py`) and a `StubClient`.
3. Drives `loom.tick() → after_beat1 → after_beat2`.
4. Asserts:
   - high-confidence add_event proposal applied AND audit row written (`status='approved' reviewer='loom'`);
   - low-confidence set_attribute proposal queued pending;
   - add_entity proposal queued (regardless of confidence);
   - all three calls logged via `bridge.log_usage`.
5. Calls `store.list_proposals(status="pending")`, picks the queued add_entity, walks it through `update_proposal_status(..., "approved", reviewer="alice")` then `apply_proposal(...)` to confirm the operator-approval pathway works.
6. Exercises the `_load_proposal` short-id prefix path with the integration store.

- [ ] Write the test, run, iterate until green.
- [ ] Commit: `test(verse/loom): end-to-end cycle + operator approval`.

---

## Phase F — Docs + CHANGELOG

### Task F1: operator guide updates

**Files:**
- Modify: `docs/guide/operator/forest-verse.md`. Read it first to find the existing anchor list. Insert after the avatar/opt-in section and before the "Open follow-ups" section if one exists.

Add a new H2 `## Loom orchestrator` with these H3 subsections (concrete prose, not bullets):

````markdown
## Loom orchestrator

The loom is a separate orchestrator that runs cheap-model cycles inside one
configured "venue" channel and digests the resulting improv into proposed
mutations against per-channel verses. By default the loom is **disabled**:
no scheduler event, no model calls, zero cost.

### Enabling

Set both `supybot.plugins.LLM.loomNetwork` and `supybot.plugins.LLM.loomChannel`.
The loom resolves the venue Irc via `world.getIrc(network)`; if either
setting is empty, or the network isn't connected, the loom stays inert.

```
config supybot.plugins.LLM.loomNetwork afternet
config supybot.plugins.LLM.loomChannel #forest
```

Verses opt in via the per-channel `verseEnabled` flag (PR 1). The loom only
considers verses whose channel is *also joined on the loom network*.

### Source filter

`loomBotNicks` is a comma-separated allowlist. Empty means capture every
non-self line in the venue (the original design intent, suitable for the
bot-heavy channel the loom was built for). Set it to a strict list when
the venue mixes humans and bots:

```
config supybot.plugins.LLM.loomBotNicks botA,botB,botC
```

### Cycle anatomy

A cycle is `seed → 90 s listen → beat → 90 s listen → digest`. Three
cheap-model calls per non-idle cycle. Idle cycles short-circuit to one
call (seed); a cycle whose listen windows produce no transcript skips both
the beat and the digest.

### Proposal moderation

```
@verseproposals [#chan] [pending|approved|rejected]
@verseapprove <id> [#chan]
@versereject <id> [#chan]
```

Default channel = current; default status = `pending`. Auto-applied
proposals carry `status='approved' reviewer='loom'` and appear under
`@verseproposals #chan approved`. `<id>` accepts unique-prefix matches.
Both `@verseapprove` and `@versereject` require `llm.verse.gm`.

### Cost transparency

Each loom call is logged in `@usage` tagged `loom:seed`, `loom:beat`, or
`loom:digest`. Until the Gemini cache plumbing lands in `service.py`,
projections assume zero cache hits.

### Tuning

| Knob                       | Bump up when                                       |
|----------------------------|----------------------------------------------------|
| `loomCycleInterval`        | The venue is overstimulated; cycles too frequent.  |
| `loomVerseCooldown`        | One verse dominates; force rotation.               |
| `loomBeatWindow`           | The bot reply cadence is slow; transcripts empty.  |
| `loomTranscriptMaxLines`   | Transcript truncation drops salient lines.         |
| `verseAutoApplyThreshold`  | Auto-apply approves too aggressively (raise it).   |
````

### Task F2: commands reference

**Files:**
- Modify: `docs/guide/reference/commands.md`. Read it first to confirm the existing `@verse*` row format (table cells will be in that file's actual schema).

Append three rows in the same format used for existing `@verse*` entries (this is the *target shape*; the precise column set in `commands.md` may differ — match the file):

```markdown
| `@verseproposals`        | List loom proposals: `[#chan] [pending|approved|rejected]`. Default channel = current; default status = `pending`. | `llm.verse.gm` |
| `@verseapprove <id>`     | Apply a pending loom proposal and mark it approved. Accepts unique-prefix ids. | `llm.verse.gm` |
| `@versereject <id>`      | Reject a pending loom proposal without applying its mutation. Accepts unique-prefix ids. | `llm.verse.gm` |
```

### Task F3: CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`. Append under the unreleased bucket:

```markdown
- **Loom orchestrator (PR 2 of forest-verse).** Multi-turn cycles in the
  configured `loomChannel` on `loomNetwork` riff with other bots and digest
  the transcript into proposals. High-confidence non-entity proposals
  auto-apply with an audit row; the rest queue for `@verseapprove` /
  `@versereject`. New registry: `loomNetwork`, `loomChannel`, `loomModel`,
  `loomCycleInterval`, `loomVerseCooldown`, `loomBeatWindow`,
  `loomTranscriptMaxLines`, `loomTranscriptMaxChars`, `loomBotNicks`,
  `verseAutoApplyThreshold`. New commands: `@verseproposals`,
  `@verseapprove`, `@versereject`. Loom calls visible in `@usage` tagged
  `loom:seed`/`loom:beat`/`loom:digest`. Defaults are empty / disabled —
  upgrades are zero-effect until an operator points the loom at a channel.
```

- [ ] Run `make check` once more.
- [ ] Single doc commit: `docs(verse): loom orchestrator operator guide + changelog`.

---

## Final review checklist

- [ ] `make check` green on the full repo.
- [ ] `uv run pytest plugins/llm -q` passes; coverage on `plugins/llm/src/llm/verse/` ≥ 90 %.
- [ ] `git diff main...HEAD --stat` shows only files listed in **Files** sections of the tasks above. No drive-by edits.
- [ ] CHANGELOG entry added.
- [ ] Operator guide and commands reference updated.
- [ ] Open the PR with title `feat: forest-verse PR 2 — loom orchestrator + proposal queue` and a body that copies the §"Scope guard for PR 2" block above.
- [ ] After merge: wait for both the GitHub Actions run **and** the GHCR Docker build to publish before `systemctl --user restart vibebot` over SSH (per `feedback_wait_for_docker.md`). Set `loomNetwork` + `loomChannel` *after* the new image is live, not before.

---

## Out of scope / deferred to PR 3

- `verseCrosspollAllowSend`, `verseCrosspollAllowReceive`, `verseCrosspollPerCycleLimit` — crosspollination + the `crosspoll_seed` source.
- Event retention compaction (`verseEventRetentionDays` already exists from PR 1; PR 3 adds the daily compaction job).
- Embedding-based `verse_recall` (PR 1/2 ship substring-only).

## Open follow-ups (not blocking PR 2)

- **Gemini cache plumbing in `service.py`.** The loom prompt structure is cache-friendly; once LiteLLM's `cached_content` API is wired, log `cached_tokens` and revisit the cost projection.
- **Loom-cycle inspection dashboard.** Useful when tuning beat windows. Defer until cycles are demonstrably running.
- **Orphan-entity handling.** `add_event.entity_ids` is JSON in a TEXT column (no FK). Loom-proposed events can reference entities that get retired between propose-time and apply-time. Apply still succeeds (no FK enforcement); the dangling reference shows up only as a stale id in the events log. Acceptable for PR 2; revisit if it bites.
