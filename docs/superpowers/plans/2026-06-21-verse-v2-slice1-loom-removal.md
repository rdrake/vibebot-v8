# Verse v2 — Slice 1: Loom Removal, Compaction Decouple, Loom-Data Purge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the #idlerpg loom subsystem and its crosspoll appendage, decouple the kept compaction subsystem from the loom, and purge loom/crosspoll exhaust from prod #afternet canon — leaving fc42's authored lore (including the ~10 compaction digests) intact.

**Architecture:** Two parts. **Part A** (Tasks 1–11) is in-place code/config removal plus compaction decoupling, done in dependency order so every commit leaves `make test` green: *relocate before delete*. We first relocate the two symbols compaction/`@versedit` depend on (the payload validator → new `verse/validation.py`; the LLM completion client → `verse/compaction.py`, renamed neutrally), then delete the loom plugin wiring, the loom-moderation commands, the loom/crosspoll source files and tests, and finally the now-dead store helpers and 14 config keys. **No schema migration** — `'loom'`/`'crosspoll'` stay as harmless dead enum values; `SCHEMA_VERSION` stays 3. **Part B** (Task 12) is a tested, one-time `purge_loom_data` run once against prod after a WAL-safe backup, in a single `write_transaction`.

**Tech Stack:** Python 3, Limnoria/Supybot plugin (`plugins/llm/src/llm/`), SQLite (WAL, per-channel verse stores), `litellm`, pytest + coverage (`--cov-fail-under=93`), `uv`, `make lint`/`make typecheck`/`make test`.

**Source design:** `docs/superpowers/specs/2026-06-21-verse-v2-slice1-loom-removal-design.md` (read it before starting).

---

## Orientation for the implementer

- **Repo root** (worktree): `/Users/rdrake/workspace/afternet/vibebot-v8/.claude/worktrees/bridge-cse_013BojRduM1ko6Wqm5eiKKsD`. All paths below are repo-relative.
- **Plugin source:** `plugins/llm/src/llm/` — key files `plugin.py` (large), `config.py`, `verse/loom.py`, `verse/compaction.py`, `verse/store.py`, `verse/avatar.py`.
- **Tests:** `plugins/llm/tests/` and `plugins/llm/tests/verse/`. There is a `store` pytest fixture (used by `tests/verse/test_verse_aging.py` and `tests/verse/test_compaction.py`) — reuse it for new tests.
- **Run the full suite + coverage gate:** `make test` (→ `uv run pytest plugins/llm/tests/ plugins/nickinmiddle/tests/ -v -m "not slow" --cov --cov-report=term-missing --cov-fail-under=93`).
- **Run one file:** `uv run pytest plugins/llm/tests/verse/test_validation.py -v`.
- **After every Edit**, `make lint && make typecheck` runs automatically (project hook); keep them clean.
- **Commit style:** Conventional Commits (`feat(verse):`, `refactor(verse):`, `test(verse):`, `chore(verse):`). Pushing to `main` is fine, but this plan is executed on the worktree branch; do not push unless asked.
- **Line numbers are hints only.** They drift as earlier tasks edit the same file. Locate each symbol by name (grep), not by line number.
- **Keep / delete boundary — burn this in:** KEEP `apply_direct`, the `proposals` table, `@versedit`, `@canon`, `@versecompact`, the daily compaction timer, and `compact_verse`. DELETE the loom orchestrator, crosspoll, the proposal *machinery* (`add_proposal` / `apply_and_record_proposal` / `apply_proposal_and_mark`), and the `@verseproposals`/`@verseapprove`/`@versereject` moderation trio.

## File Structure

**New files**
- `plugins/llm/src/llm/verse/validation.py` — payload validator (`validate_payload` + `_PAYLOAD_SCHEMA` + predicates), relocated out of `loom.py`. One responsibility: validate a proposal/edit op payload.
- `plugins/llm/src/llm/verse/purge.py` — one-time loom-data purge (`list_loom_digest_candidates` + `purge_loom_data`). One responsibility: destructive, channel-specific cleanup.
- `plugins/llm/tests/verse/test_validation.py` — relocated from `test_loom_validate_payload.py`.
- `plugins/llm/tests/verse/test_purge.py` — purge unit tests.

**Modified**
- `verse/compaction.py` — gains the relocated, renamed completion client (`LiteLLMVerseClient` / `VerseCallUsage` / `VerseModelClient`).
- `verse/avatar.py` — import `validate_payload` from `.validation`.
- `verse/store.py` — `replace_events_with_lore_digest` stamps `source='llm'`; delete 3 loom proposal helpers + `bump_last_seen_ts`; reword `apply_direct` docstring.
- `plugin.py` — delete loom lifecycle wiring, `_PluginLoomBridge`, crosspoll wiring, the moderation trio; repoint compaction to `verseCompactionModel` + relocated client.
- `config.py` — add `verseCompactionModel`; remove 14 loom-family keys.
- Several test files de-loomed (Tasks 3, 7) or deleted (Tasks 5, 8).

**Deleted**
- `verse/loom.py`, `verse/crosspoll_store.py`, `verse/crosspoll_schema.sql`.
- `tests/verse/test_loom.py`, `tests/verse/test_loom_integration.py`, `tests/verse/test_crosspoll_store.py`, `tests/test_plugin_loom.py`, `tests/verse/test_loom_validate_payload.py` (relocated).

---

## Task 1: Add the `verseCompactionModel` config key

Additive and independent — do it first so Task 3 can repoint the compaction call sites onto a key that already exists at test time.

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (near the existing `loomModel` registration, ~line 423)

- [ ] **Step 1: Write the failing test**

Add to `plugins/llm/tests/verse/test_compaction.py` (a new test; keep existing tests):

```python
def test_verse_compaction_model_key_registered():
    """The new global key exists with the old loomModel default."""
    from supybot import conf

    import llm.config  # noqa: F401  (registers keys as a side effect)

    val = conf.supybot.plugins.LLM.verseCompactionModel()
    assert val == "gemini/gemini-flash-lite-latest"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py::test_verse_compaction_model_key_registered -v`
Expected: FAIL — `NonExistentRegistryEntry` (key not registered yet).

- [ ] **Step 3: Register the key**

In `config.py`, immediately after the `loomModel` registration block, add (copies the `loomModel` idiom — global `registry.String`):

```python
conf.registerGlobalValue(
    LLM,
    "verseCompactionModel",
    registry.String(
        "gemini/gemini-flash-lite-latest",
        _("""Cheap model used by retention compaction to summarise old
        verse events into a single lore-digest. Formerly shared the loom's
        ``loomModel`` key; split out when the loom was removed so compaction
        owns its own model setting."""),
    ),
)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py::test_verse_compaction_model_key_registered -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/verse/test_compaction.py
git commit -m "feat(verse): add verseCompactionModel config key (split from loomModel)"
```

---

## Task 2: Relocate `validate_payload` into `verse/validation.py`

Copy the validator (and the schema + predicates it closes over) into a new module that does not depend on the loom; repoint `avatar.py`; relocate its test. `loom.py` keeps its own copy for now — it is deleted wholesale in Task 8, so we never need an intermediate gutted-loom state.

**Files:**
- Create: `plugins/llm/src/llm/verse/validation.py`
- Modify: `plugins/llm/src/llm/verse/avatar.py` (import at ~line 13)
- Create: `plugins/llm/tests/verse/test_validation.py`
- Delete: `plugins/llm/tests/verse/test_loom_validate_payload.py`

- [ ] **Step 1: Create the relocated test (failing — module doesn't exist yet)**

Create `plugins/llm/tests/verse/test_validation.py` (content of `test_loom_validate_payload.py`, repointed to the new module):

```python
from llm.verse.validation import validate_payload


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

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest plugins/llm/tests/verse/test_validation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.verse.validation'`.

- [ ] **Step 3: Create `verse/validation.py`**

```python
"""Shared payload validation for verse proposal/edit ops.

Relocated from the (removed) loom module so the verse_edit tool (avatar.py)
and the @versedit path can validate op payloads without depending on the
loom subsystem. One schema governs all constructive ops; an op with no
schema entry is rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _is_strict_int(v: Any) -> bool:
    """Reject bool, accept int. (bool is a subclass of int in Python.)"""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_strict_int(x) for x in v)


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
    "update_entity": (("entity_id", _is_strict_int, "int"),),
}


def validate_payload(op: str, payload: dict[str, Any]) -> str | None:
    """Return None if *payload* is valid for *op*, else a human reason string.

    Only constructive ops have entries; an op without a schema entry is
    rejected. Used by the verse_edit tool (avatar.py).
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

> Note: the `crosspoll_seed` entry is kept verbatim as harmless dead data — `avatar.py`'s `_VERSE_EDIT_OPS` allowlist already excludes it, and removing it would diverge from the source without benefit.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest plugins/llm/tests/verse/test_validation.py -v`
Expected: PASS (all 4).

- [ ] **Step 5: Repoint `avatar.py`**

In `plugins/llm/src/llm/verse/avatar.py`, change the import (currently `from .loom import validate_payload`):

```python
from .validation import validate_payload
```

- [ ] **Step 6: Delete the old test file**

```bash
git rm plugins/llm/tests/verse/test_loom_validate_payload.py
```

- [ ] **Step 7: Run the verse suite + the avatar path**

Run: `uv run pytest plugins/llm/tests/verse/ plugins/llm/tests/test_plugin_verse.py -v`
Expected: PASS (avatar still validates; `loom.py` still has its own copy, untouched).

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/verse/validation.py plugins/llm/src/llm/verse/avatar.py plugins/llm/tests/verse/test_validation.py
git commit -m "refactor(verse): relocate validate_payload to verse/validation.py"
```

---

## Task 3: Relocate the completion client into `verse/compaction.py`

`compact_verse` needs a client whose only implementation lives in `loom.py` (`LiteLLMLoomClient`, return type `LoomCallUsage`, Protocol `LoomModelClient`). Copy all three into `compaction.py` — compaction's sole surviving consumer — renamed neutrally, then repoint the production importers and the surviving tests. `loom.py` keeps its own copies until Task 8.

**Files:**
- Modify: `plugins/llm/src/llm/verse/compaction.py` (imports + append 3 classes)
- Modify: `plugins/llm/src/llm/plugin.py` (`_run_compaction_pass`, `versecompact`)
- Modify: `plugins/llm/tests/verse/test_compaction.py` (`_FakeClient`)
- Modify: `plugins/llm/tests/verse/_fakes.py` (`VerseCallUsage` import)
- Modify: `plugins/llm/tests/test_plugin_verse.py` (4 patch sites)

- [ ] **Step 1: Write the failing test**

Add to `plugins/llm/tests/verse/test_compaction.py`:

```python
def test_relocated_client_types_importable_from_compaction():
    from llm.verse.compaction import (
        LiteLLMVerseClient,
        VerseCallUsage,
        VerseModelClient,
    )

    usage = VerseCallUsage(prompt_tokens=1, completion_tokens=2, cost=0.0)
    assert usage.completion_tokens == 2
    # Structural check only. Do NOT use isinstance(LiteLLMVerseClient(),
    # VerseModelClient): VerseModelClient is a plain Protocol (not
    # @runtime_checkable), so isinstance() would raise TypeError, not pass.
    assert callable(getattr(LiteLLMVerseClient, "call", None))
    assert hasattr(VerseModelClient, "call")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py::test_relocated_client_types_importable_from_compaction -v`
Expected: FAIL — `ImportError` (names not in `compaction`).

- [ ] **Step 3: Add `time` + `Protocol` to compaction's imports**

In `compaction.py`, the current top imports are:

```python
import contextlib
import logging
from collections.abc import Callable
from typing import Any, NamedTuple
```

Change to add `time` and `Protocol`:

```python
import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol
```

- [ ] **Step 4: Append the three relocated classes to `compaction.py`**

Add at the end of `compaction.py`. This is a faithful copy of the loom client with **only the three type names changed** (`LiteLLMLoomClient`→`LiteLLMVerseClient`, `LoomCallUsage`→`VerseCallUsage`, `LoomModelClient`→`VerseModelClient`) and the default logger name pointed at this module. The internal log/usage strings are left verbatim (they feed `@usage` accounting keyed on `op=loom:…`; changing them would silently re-key the usage DB — out of scope here):

```python
class VerseCallUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cost: float


class VerseModelClient(Protocol):
    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, VerseCallUsage]: ...


class LiteLLMVerseClient:
    """Default verse completion client (used by retention compaction).

    Calls ``litellm.completion`` synchronously (already on a worker thread
    by the time this runs) and returns the content string plus a
    ``VerseCallUsage``. Errors propagate to the caller.
    """

    def __init__(
        self,
        log: logging.Logger | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        self._log = log or logging.getLogger("llm.verse.compaction")
        self._api_key = api_key or None

    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, VerseCallUsage]:
        import litellm

        t0 = time.monotonic()
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        response = litellm.completion(model=model, messages=messages, **kwargs)
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
            cost = float(litellm.completion_cost(completion_response=response, model=model) or 0.0)
        except Exception:
            cost = 0.0
        # Sanity clamp: litellm.completion_cost falls back to a token count
        # for models without pricing data (observed in prod for
        # gemini-flash-lite-latest, returning ~365). Anything over $1 for a
        # single short cheap-model call is implausible — assume the
        # accounting is wrong and zero it out so @usage isn't polluted with
        # nonsense. This is a soft clamp; remove once litellm pricing
        # catches up or once we add explicit pricing tables.
        if cost > 1.0:
            self._log.warning(
                f"loom completion_cost returned implausible value {cost!r} "
                f"for model={model}; clamping to 0.0 (likely missing "
                "pricing data in litellm)"
            )
            cost = 0.0
        # Match service.py:_log_completion_timing's f-string convention.
        # %-args formatting was partially failing under the bot's runtime
        # logger setup (some args substituted, %d ones not) — see #66.
        self._log.warning(
            f"completion_timing op=loom:{op} model={model} "
            f"elapsed_ms={elapsed_ms:.0f} "
            f"prompt_tokens={pt} completion_tokens={ct} cost={cost:.6f}"
        )
        return content, VerseCallUsage(pt, ct, cost)
```

- [ ] **Step 5: Run the new test to verify it passes**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py::test_relocated_client_types_importable_from_compaction -v`
Expected: PASS.

- [ ] **Step 6: Repoint `_run_compaction_pass` in `plugin.py`**

Locate `_run_compaction_pass` (grep `def _run_compaction_pass`). It currently has, at the top, `from llm.verse import compaction as _compaction` and `from llm.verse.loom import LiteLLMLoomClient`, and later:

```python
        model = self.registryValue("loomModel") or "gemini/gemini-flash-lite-latest"
        loom_api_key = self.registryValue("assistantApiKey") or None
        client = LiteLLMLoomClient(api_key=loom_api_key)
```

Make three edits:
1. Delete the line `from llm.verse.loom import LiteLLMLoomClient`.
2. Change `self.registryValue("loomModel")` → `self.registryValue("verseCompactionModel")`.
3. Change `client = LiteLLMLoomClient(api_key=loom_api_key)` → `client = _compaction.LiteLLMVerseClient(api_key=loom_api_key)`.

Result:

```python
        model = self.registryValue("verseCompactionModel") or "gemini/gemini-flash-lite-latest"
        loom_api_key = self.registryValue("assistantApiKey") or None
        client = _compaction.LiteLLMVerseClient(api_key=loom_api_key)
```

- [ ] **Step 7: Repoint `versecompact` in `plugin.py`**

Locate `def versecompact` (grep). It has `from llm.verse import compaction as _compaction`, `from llm.verse.loom import LiteLLMLoomClient`, and the same `loomModel`/`LiteLLMLoomClient` trio. Apply the identical three edits:
1. Delete `from llm.verse.loom import LiteLLMLoomClient`.
2. `self.registryValue("loomModel")` → `self.registryValue("verseCompactionModel")`.
3. `client = LiteLLMLoomClient(api_key=loom_api_key)` → `client = _compaction.LiteLLMVerseClient(api_key=loom_api_key)`.

(Leave the `_log_usage` `nick="loom"` / `command=f"loom:{op}"` labels untouched — usage-DB keys, out of scope.)

- [ ] **Step 8: Repoint `_FakeClient` in `test_compaction.py`**

Change the inner import (currently `from llm.verse.loom import LoomCallUsage`) and its use:

```python
class _FakeClient:
    def __init__(self, content: str = "A digest of past events.") -> None:
        self.content = content
        self.calls: list[dict] = []

    def call(self, *, op: str, model: str, messages: list[dict[str, str]]):
        from llm.verse.compaction import VerseCallUsage

        self.calls.append({"op": op, "model": model, "messages": messages})
        return self.content, VerseCallUsage(prompt_tokens=10, completion_tokens=20, cost=0.0)
```

- [ ] **Step 9: Repoint `_fakes.py`**

In `plugins/llm/tests/verse/_fakes.py`, the line `from llm.verse.loom import LoomCallUsage, LoomConfig, VerseSnapshot` mixes a still-needed type (`LoomCallUsage`→`VerseCallUsage`) with loom-only fakes (`LoomConfig`, `VerseSnapshot`) used by `make_loom_config`/`make_snapshot`/`FakeBridge` (all removed in Task 8). For now, split the import so the usage type comes from `compaction` and the loom-only names still resolve (they are deleted in Task 8):

```python
from llm.verse.compaction import VerseCallUsage
from llm.verse.loom import LoomConfig, VerseSnapshot
```

Then replace every `LoomCallUsage(...)` construction in this file with `VerseCallUsage(...)` (grep `LoomCallUsage` within `_fakes.py`; same field names, drop-in).

- [ ] **Step 10: Repoint the 4 patch sites in `test_plugin_verse.py`**

There are 4 `mocker.patch("llm.verse.loom.LiteLLMLoomClient", ...)` calls (in `test_compaction_compacts_old_events`, `test_failure_in_compact_verse_replies_error`, `test_registry_lookup_failure_uses_defaults`, `test_zero_retention_and_min_keep_are_honoured`). The plugin now constructs `_compaction.LiteLLMVerseClient`, so the patch target must change. Replace each:

```python
        mocker.patch(
            "llm.verse.compaction.LiteLLMVerseClient",
            return_value=...,   # keep each call's existing return_value
        )
```

(Preserve each site's existing `return_value=` — `_FakeClient()` for the first, `mocker.MagicMock()` for the other three.)

- [ ] **Step 11: Run the affected suites**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py plugins/llm/tests/test_plugin_verse.py -v`
Expected: PASS — compaction runs against the relocated client + `verseCompactionModel`.

- [ ] **Step 12: Commit**

```bash
git add plugins/llm/src/llm/verse/compaction.py plugins/llm/src/llm/plugin.py plugins/llm/tests/verse/test_compaction.py plugins/llm/tests/verse/_fakes.py plugins/llm/tests/test_plugin_verse.py
git commit -m "refactor(verse): relocate completion client to compaction.py, repoint to verseCompactionModel"
```

---

## Task 4: Stamp compaction digests `source='llm'`

`source='loom'` is overloaded — it tags both #idlerpg junk and real compaction lore-digests. Make future digests unambiguous by stamping `'llm'` (already a valid `events.source` value — no migration). This must land before the Part B purge so new digests are never caught by `DELETE … WHERE source='loom'`.

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py` (`replace_events_with_lore_digest`, ~line 981)

- [ ] **Step 1: Write the failing test**

Add to `plugins/llm/tests/verse/test_compaction.py`:

```python
def test_lore_digest_is_stamped_llm(store):
    """Compaction digests are now source='llm', not 'loom'."""
    e1 = store.add_entity("npc", "Aldous", "")
    ev1 = store.apply_direct(
        op="add_event",
        payload={"summary": "old deed one", "entity_ids": [e1]},
        source="avatar",
        provenance="test",
    )
    ev2 = store.apply_direct(
        op="add_event",
        payload={"summary": "old deed two", "entity_ids": [e1]},
        source="avatar",
        provenance="test",
    )
    store.replace_events_with_lore_digest(
        delete_ids=[ev1, ev2],
        summary="A digest of Aldous's old deeds.",
        entity_ids=[e1],
        ts=1000.0,
    )
    with store.read_connection() as conn:
        rows = conn.execute(
            "SELECT source FROM events WHERE summary LIKE 'A digest%'"
        ).fetchall()
    assert rows and all(r[0] == "llm" for r in rows)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py::test_lore_digest_is_stamped_llm -v`
Expected: FAIL — digest is stamped `'loom'`.

- [ ] **Step 3: Change the source + docstring**

In `replace_events_with_lore_digest`, change the docstring line and the `source=` argument from `"loom"` to `"llm"`:

```python
    def replace_events_with_lore_digest(
        self,
        *,
        delete_ids: Sequence[int],
        summary: str,
        entity_ids: Sequence[int],
        ts: float,
    ) -> int:
        """Replace ``delete_ids`` with a single ``source='llm'`` digest event.

        All work happens inside one ``write_transaction``; on error the whole
        operation rolls back and the originals survive.
        """
        return self._replace_events_with_source(
            delete_ids=delete_ids,
            summary=summary,
            entity_ids=entity_ids,
            ts=ts,
            source="llm",
        )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest plugins/llm/tests/verse/test_compaction.py -v`
Expected: PASS. If any *existing* compaction test asserts the digest is `source='loom'`, update that assertion to `'llm'` in this same step (grep `'loom'` in `test_compaction.py` and `test_plugin_verse.py`).

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/verse/test_compaction.py
git commit -m "feat(verse): stamp compaction lore-digests source='llm' (disambiguate from loom junk)"
```

---

## Task 5: Delete the loom lifecycle wiring, `_PluginLoomBridge`, and crosspoll wiring from `plugin.py`

Remove the plugin-side loom orchestrator plumbing in one coherent commit, plus the crosspoll store accessor and the bridge. Delete the loom *plugin* test that covers exactly this wiring.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Delete: `plugins/llm/tests/test_plugin_loom.py`

> All deletions below are located by symbol name. Line numbers are pre-Task-5 hints. After deleting, grep `loom` and `crosspoll` in `plugin.py` and confirm the only survivors are the moderation trio (Task 6 deletes those) and the cosmetic `nick="loom"`/`command="loom:…"` usage labels inside `versecompact`/`_run_compaction_pass`.

- [ ] **Step 1: Delete `test_plugin_loom.py` first (it tests the wiring we're removing)**

```bash
git rm plugins/llm/tests/test_plugin_loom.py
```

- [ ] **Step 2: Delete the loom + crosspoll `__init__` fields**

Remove the loom cache block (~744–749):

```python
        # Forest-verse loom orchestrator (PR 2). All four caches must be
        # initialised before doPrivmsg can run so the transcript hook never
        # reads an unset attribute.
        self._loom = None
        self._loom_bridge = None
        self._loom_channel_cache: str | None = None
        self._loom_network_cache: str | None = None
        self._loom_bot_nicks_cache: tuple[str, ...] = ()
        self._loom_capture_transcript_cache: bool = True
```

Remove the crosspoll singleton fields (~733–734):

```python
        # Process-wide CrosspollStore singleton (lazily created on first use).
        self._crosspoll_store: CrosspollStore | None = None
        self._crosspoll_store_lock = threading.Lock()
```

Also remove the `CrosspollStore` `TYPE_CHECKING` import (grep `CrosspollStore` near the top of the file, ~line 76) — it only existed for the `_crosspoll_store` annotation.

- [ ] **Step 3: Delete the loom config-change hook loop + `_on_loom_config_change`**

Remove the registration loop (~810–823) AND the method (~844–859). **Do not touch** the compaction-timer arming (~line 831, `schedule.add…` for the daily compaction pass) sitting between them — keep it. Grep the surrounding lines and excise only the loom callback loop and the `def _on_loom_config_change` method:

```python
        # Re-wire the loom on live config changes. Without this, an operator
        # who sets loomNetwork + loomChannel via @config has to @reload LLM
        # before the orchestrator notices.
        for _key in (
            "loomNetwork",
            "loomChannel",
            "loomModel",
            "loomCycleInterval",
            "loomVerseCooldown",
            "loomBeatWindow",
            "loomTranscriptMaxLines",
            "loomTranscriptMaxChars",
            "loomBotNicks",
            "loomCaptureTranscript",
            "verseAutoApplyThreshold",
        ):
            getattr(conf.supybot.plugins.LLM, _key).addCallback(self._on_loom_config_change)
```

```python
    def _on_loom_config_change(self, *args: object) -> None:
        """Re-wire the loom when any loom-* registry value changes at runtime.

        ``_wire_loom_if_enabled`` is idempotent: if the (network, channel)
        tuple is unchanged it short-circuits, so callbacks for
        ``loomBotNicks`` etc. correctly tear down + rebuild only when the
        target identity flips.
        """
        # Force a rebuild even when (network, channel) didn't change so
        # bot-nicks / cycle-interval changes take effect.
        self._loom_channel_cache = None
        self._loom_network_cache = None
        try:
            self._wire_loom_if_enabled()
        except Exception:
            self.log.exception("loom re-wire failed (non-fatal)")
```

- [ ] **Step 4: Delete the loom teardown in `die()`**

Remove (~889–891):

```python
        # Loom orchestrator teardown (reactive trigger; single beat window).
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_loom_after_chime")
```

- [ ] **Step 5: Delete the loom transcript-capture hook in `doPrivmsg`**

Remove the whole block (~1233–1255) starting `# Forest-verse loom transcript hook (PR 2).` through the `except Exception: self.log.exception("loom transcript capture failed (non-fatal)")`.

- [ ] **Step 6: Delete `_wire_loom_if_enabled` AND its call site**

Remove the entire method (grep `def _wire_loom_if_enabled`, ~5434–5520), including its `from .verse.loom import LiteLLMLoomClient, Loom, LoomConfig` line and the `schedule.removeEvent("llm_loom_after_chime")` teardown inside it.

**Also delete the unconditional call site `self._wire_loom_if_enabled()` in `__init__` (~plugin.py:805).** Grep `_wire_loom_if_enabled(` and remove every call. The second call (inside `_on_loom_config_change`) is already gone with Step 3, but the `__init__` call is NOT — if left, `__init__` will `AttributeError` at plugin construction, which every plugin test triggers in Step 10.

- [ ] **Step 7: Delete `_get_or_create_crosspoll_store`**

Remove the method (grep `def _get_or_create_crosspoll_store`, ~5696–5711).

- [ ] **Step 8: Delete the `_PluginLoomBridge` class**

Remove the whole class (grep `class _PluginLoomBridge`, ~6710–6808), including its `from .verse.loom import VerseSnapshot` import (~6747).

- [ ] **Step 9: Find any remaining call sites of the deleted symbols**

Run: `grep -n "_wire_loom_if_enabled\|_PluginLoomBridge\|_get_or_create_crosspoll_store\|_loom\b\|observe_transcript\|llm_loom_after_chime" plugins/llm/src/llm/plugin.py`
Expected: no matches (a stray call to `_wire_loom_if_enabled()` in `__init__`/`__call__`/`do376` startup must also be removed — grep and delete it if present).

- [ ] **Step 10: Run the plugin suites**

Run: `uv run pytest plugins/llm/tests/ -v -m "not slow" -k "plugin or verse"`
Expected: PASS. (`loom.py` still imports cleanly; its own tests in `test_loom.py` still run and pass — they're deleted in Task 8.)

- [ ] **Step 11: Commit**

```bash
git add -A plugins/llm/src/llm/plugin.py
git rm --cached plugins/llm/tests/test_plugin_loom.py  # already staged by Step 1 git rm
git commit -m "refactor(verse): remove loom lifecycle wiring, _PluginLoomBridge, crosspoll plumbing from plugin"
```

---

## Task 6: Delete the loom-moderation command trio from `plugin.py`

`@verseproposals`/`@verseapprove`/`@versereject` only ever reviewed loom-produced `status='pending'` proposals (the loom was the sole producer). Delete the commands, their `wrap` registrations, their `COMMAND_REGISTRY` help entries, and their private helpers — together, so the help-sync test stays green.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: tests referencing these commands (grep — likely `tests/test_plugin_verse.py`)

- [ ] **Step 1: Delete the three commands + their `wrap` registrations**

Remove `def verseproposals` + `verseproposals = wrap(...)` (~6477–6526), `def verseapprove` + `verseapprove = wrap(...)` (~6528–6573, includes the `event_source = "crosspoll" if … else "loom"` line), `def versereject` + `versereject = wrap(...)` (~6575–6614).

- [ ] **Step 2: Delete the private helpers**

Remove `_proposal_snippet` (~6438–6452), `_proposal_target_store` (~6454–6464), `_load_proposal` (~6466–6472), and the `_VERSEPROPOSALS_MAX_LIMIT = 50` constant (~6475). Also remove `_VERSEPROPOSALS_DEFAULT_LIMIT` if grep shows it now has no remaining reader.

- [ ] **Step 3: Delete the `COMMAND_REGISTRY` entries**

Remove the three `CommandInfo(name="verseproposals"…)`, `CommandInfo(name="verseapprove"…)`, `CommandInfo(name="versereject"…)` blocks (~377–410).

- [ ] **Step 4: Delete the orphaned test classes + dispatch-list entries**

These commands have dedicated test classes that exist solely to exercise them — delete them **wholesale** (they have no value without the commands), don't try to "adjust" them:
- In `plugins/llm/tests/test_plugin_verse.py`: delete the entire `TestVerseproposalsCommand`, `TestVerseapproveRejectCommands`, and `TestVerseapproveCrosspollSource` classes (the last tests the now-deleted `event_source="crosspoll"` branch; it also mocks `store.apply_proposal_and_mark`, which Task 9 deletes).
- In `plugins/llm/tests/test_plugin_dispatch.py`: remove `"verseproposals"`, `"verseapprove"`, `"versereject"` from the dispatch / help-sync assertion (~lines 683–685) so the `COMMAND_REGISTRY` ↔ command consistency test stays green.

Then confirm nothing dangling remains:
Run: `grep -rn "verseproposals\|verseapprove\|versereject" plugins/llm/tests/`
Expected: no matches. (Class names/line numbers are hints — locate by grep, delete by class.)

- [ ] **Step 5: Confirm `@versedit` and `@canon` survive**

Run: `grep -n "def versedit\|def canon\|def versecompact" plugins/llm/src/llm/plugin.py`
Expected: all three still present (KEEP).

- [ ] **Step 6: Run the plugin + help-sync suites**

Run: `uv run pytest plugins/llm/tests/ -v -m "not slow" -k "plugin or verse or help or command_registry"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "refactor(verse): remove loom-moderation commands (verseproposals/verseapprove/versereject)"
```

---

## Task 7: De-loom `test_verse_aging.py`

Five aging tests seed state via the loom's `apply_or_queue`. Replace those seedings with equivalent direct store inserts so the tests no longer import `loom.py` (deleted next task) while still exercising the aging behaviour they target (`last_seen_ts` bumps).

**Files:**
- Modify: `plugins/llm/tests/verse/test_verse_aging.py`

- [ ] **Step 1: Understand what each test asserts**

The 5 tests assert that applying a proposal bumps `last_seen_ts` (or that a *queued*/*rejected* one does not). With the loom gone, the behaviour under test is the store's own write path. Rewrite each to call the store directly and set `last_seen_ts` the way the live `apply_direct` path does (or assert the no-bump cases by simply not writing).

- [ ] **Step 2: Rewrite seeding site 1 (`test_loom_applied_proposal_bumps_last_seen`)**

Replace:

```python
        from llm.verse.loom import ParsedProposal, apply_or_queue

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "ghost lurked", "entity_ids": [eid]},
            confidence=0.95,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(
            store,
            prop,
            cycle_id="cyc-1",
            threshold=0.7,
        )
```

with a direct apply + explicit heartbeat (mirrors what the loom path did — write the event, then bump `last_seen_ts`):

```python
        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        store.apply_direct(
            op="add_event",
            payload={"summary": "ghost lurked", "entity_ids": [eid]},
            source="llm",
            provenance="test",
        )
        store.set_attribute(eid, "last_seen_ts", str(time.time()))
```

Keep the existing assertion (`last_seen_ts > 0.0`). Add `import time` at the top of the test module if not already present.

> Rationale: the aging logic being tested reads `last_seen_ts`; the loom merely produced the write. The direct apply + heartbeat reproduces the *state* the loom created without importing it. The "applied → bump" semantics are preserved.

- [ ] **Step 3: DELETE seeding site 2 (`test_loom_queued_proposal_does_not_bump`)**

This test only asserted that the loom *queued* (did not apply) a low-confidence proposal, so `last_seen_ts` stayed at the floor. With no loom and no proposal queue, any rewrite collapses to a tautology (set `0.0`, read back `0.0`) that exercises no aging logic. **Delete the test method outright** — do not rewrite it. The "applied → bump" direction is still covered by sites 1/3/4; the "no write → no bump" direction adds nothing once the queue is gone. Note the deletion in the commit message.

- [ ] **Step 4: Rewrite seeding sites 3, 4, 5**

Apply the same pattern:
- Site 3 (`set_attribute` applied → bump): `store.apply_direct(op="set_attribute", payload={"entity_id": eid, "key": "mood", "value": "wary"}, source="llm", provenance="test")` then `store.set_attribute(eid, "last_seen_ts", str(time.time()))`; keep the bump assertion.
- Site 4 (`add_relation` bumps both endpoints): `store.apply_direct(op="add_relation", payload={"from_id": a, "to_id": b, "kind": "ally"}, source="llm", provenance="test")` then bump both `a` and `b`; keep the `last_seen_ts > 0.0` assertion on `a`.
- Site 5 (invalid refs → no bump): the loom rejected events referencing a nonexistent id. The store's `apply_direct(op="add_event", …)` with a bad id will raise or no-op depending on `_apply_op_inline`'s validation — wrap in `pytest.raises`/`contextlib.suppress` and assert `last_seen_ts` stays `0.0` (no heartbeat written). First read `_apply_op_inline` (grep in `store.py`) to confirm whether a bad entity id raises `ValueError`/`LookupError` or is silently skipped, then assert accordingly. The behaviour under test is "rejected ⇒ no bump."

- [ ] **Step 5: Confirm no loom import remains**

Run: `grep -n "loom" plugins/llm/tests/verse/test_verse_aging.py`
Expected: no matches.

- [ ] **Step 6: Run the aging suite**

Run: `uv run pytest plugins/llm/tests/verse/test_verse_aging.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/tests/verse/test_verse_aging.py
git commit -m "test(verse): de-loom aging tests (direct store inserts replace apply_or_queue)"
```

---

## Task 8: Delete `loom.py`, the crosspoll files, and their tests

Everything importing `loom.py`/`crosspoll_store.py` has been repointed (Tasks 2–3) or de-loomed (Tasks 5–7). Delete the source + remaining tests wholesale, and drop the loom-only fakes.

**Files:**
- Delete: `plugins/llm/src/llm/verse/loom.py`, `plugins/llm/src/llm/verse/crosspoll_store.py`, `plugins/llm/src/llm/verse/crosspoll_schema.sql`
- Delete: `plugins/llm/tests/verse/test_loom.py`, `plugins/llm/tests/verse/test_loom_integration.py`, `plugins/llm/tests/verse/test_crosspoll_store.py`
- Modify: `plugins/llm/tests/verse/_fakes.py`

- [ ] **Step 1: Confirm nothing in `src/` still imports loom/crosspoll**

Run: `grep -rn "verse.loom\|verse\.crosspoll\|from .loom\|from .crosspoll\|import loom\|crosspoll_store" plugins/llm/src/`
Expected: no matches. (If any remain, fix them before deleting — do not proceed with a dangling import.)

- [ ] **Step 2: Drop the loom-only fakes from `_fakes.py`**

In `plugins/llm/tests/verse/_fakes.py`, remove `from llm.verse.loom import LoomConfig, VerseSnapshot` (added back temporarily in Task 3 Step 9), and delete the now-unused fakes `make_loom_config`, `make_snapshot`, and `FakeBridge` (grep them; confirm via `grep -rn "make_loom_config\|make_snapshot\|FakeBridge" plugins/llm/tests/` that their only callers were the loom tests being deleted in Step 3). Keep `VerseCallUsage` (imported from `compaction`).

- [ ] **Step 3: Delete the source + test files**

```bash
git rm plugins/llm/src/llm/verse/loom.py \
       plugins/llm/src/llm/verse/crosspoll_store.py \
       plugins/llm/src/llm/verse/crosspoll_schema.sql \
       plugins/llm/tests/verse/test_loom.py \
       plugins/llm/tests/verse/test_loom_integration.py \
       plugins/llm/tests/verse/test_crosspoll_store.py
```

- [ ] **Step 4: Full collection + run**

Run: `uv run pytest plugins/llm/tests/ -m "not slow" -q`
Expected: PASS, no collection/import errors. If `_fakes.py` still references a removed name, fix it now.

- [ ] **Step 5: Commit**

```bash
git add -A plugins/llm/tests/verse/_fakes.py
git commit -m "refactor(verse): delete loom.py + crosspoll store/schema + their tests"
```

---

## Task 9: Delete the dead store proposal helpers + `bump_last_seen_ts`; reword `apply_direct` docstring

With the loom and the moderation trio gone, three `store.py` methods have no callers. Delete them and `bump_last_seen_ts` (loom-only), and drop the stale `apply_and_record_proposal` mention from the kept `apply_direct` docstring. **Keep `apply_direct` and the `proposals` table.**

**Files:**
- Modify: `plugins/llm/src/llm/verse/store.py`

- [ ] **Step 1: Confirm zero callers**

Run: `grep -rn "add_proposal\|apply_and_record_proposal\|apply_proposal_and_mark\|bump_last_seen_ts" plugins/llm/src/ plugins/llm/tests/`
Expected: matches only inside `store.py` definitions themselves (and possibly dedicated unit tests — handle those in Step 3). If any `src/` *caller* remains, stop and investigate.

- [ ] **Step 2: Delete the four methods**

Remove from `store.py`: `add_proposal` (~1243–1293), `apply_and_record_proposal` (~1445–1482), `apply_proposal_and_mark` (~1512–1543), and `bump_last_seen_ts` (~566–587). **Do not** touch `apply_direct`, `_apply_op_inline`, `_set_attribute_inline`, `replace_events_with_lore_digest`, or the `proposals`-table read helpers (`get_proposal`/`list_proposals`/`update_proposal_status`) unless Step 5 shows they are now uncovered.

- [ ] **Step 3: Delete the dedicated store tests for the removed methods**

`test_store.py` has whole classes that exist only to test these methods — delete them **wholesale** (locate by grep, delete by class):
- Every class/test calling `add_proposal` (the `TestProposal*` / `TestProposalsCRUD` / `TestAddProposal*` group, ~lines 813–979).
- `TestApplyAndRecordProposal` (~lines 1060–1170).
- `TestApplyProposalAndMark` (~lines 1171–1236).
- `TestApplyProposalAndMarkEventSource` (~lines 1559–1625).

**Keep** the read-helper tests `test_get_proposal_*`, `test_list_proposals_*`, `test_update_proposal_status_*` — those methods are NOT deleted (and now provide the only coverage for them once the moderation commands are gone, so deleting them would drop coverage).

Verify no callers remain in the suite:
Run: `grep -rn "add_proposal\|apply_and_record_proposal\|apply_proposal_and_mark\|bump_last_seen_ts" plugins/llm/tests/`
Expected: no matches.

- [ ] **Step 4: Reword the `apply_direct` docstring**

Change the middle of `apply_direct`'s docstring (it names the deleted `apply_and_record_proposal`):

```python
        For operator commands (source='operator') and the verse_edit tool
        (source='llm'). This path carries no cycle tracking
        (cycle_id/confidence/reviewer are synthesized for audit only).
        Returns the new row id for creating ops, else None.
```

- [ ] **Step 5: Leave the dead enum values (NO migration)**

Do **not** modify `_VALID_SOURCES` (it may still list `'loom'`/`'crosspoll'`) or the schema `CHECK` constraints / `proposals.op` enum. Per the design §5.7 these stay as harmless dead values so `SCHEMA_VERSION` remains 3 and the Part B purge's `WHERE source IN ('loom','crosspoll')` read filter still works.

- [ ] **Step 6: Run the store + verse suites and check coverage locally**

Run: `uv run pytest plugins/llm/tests/verse/ plugins/llm/tests/test_plugin_verse.py --cov=llm.verse.store --cov-report=term-missing -q`
Expected: PASS. Note any newly-uncovered lines in `store.py`; the global gate is checked in Task 11.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/verse/store.py plugins/llm/tests/
git commit -m "refactor(verse): delete dead proposal helpers + bump_last_seen_ts from store"
```

---

## Task 10: Remove the 14 loom-family config keys

Nothing reads these any more (verified by the Task 5/6 deletions). Remove all 14 registrations.

**Files:**
- Modify: `plugins/llm/src/llm/config.py`

- [ ] **Step 1: Confirm no readers remain**

Run: `grep -rn "loomNetwork\|loomChannel\|loomModel\|loomCycleInterval\|loomVerseCooldown\|loomBeatWindow\|loomTranscriptMaxLines\|loomTranscriptMaxChars\|loomBotNicks\|loomCaptureTranscript\|verseCrosspollAllowSend\|verseCrosspollAllowReceive\|verseCrosspollPerCycleLimit\|verseAutoApplyThreshold" plugins/llm/src/`
Expected: matches only in `config.py` (the registrations themselves). **`loomModel` must NOT appear in `plugin.py`** — if it does, Task 3 missed a site; fix before continuing. (`verseCompactionModel` is the only model key plugin reads.)

- [ ] **Step 2: Delete the 10 `loom*` registrations**

Remove the `conf.registerGlobalValue(LLM, "loomNetwork", …)` … through `loomCaptureTranscript` blocks (~401–505). **Keep** the `verseCompactionModel` block added in Task 1 (it sits right after where `loomModel` was).

- [ ] **Step 3: Delete the 3 crosspoll registrations**

Remove `verseCrosspollAllowSend` (~507–516), `verseCrosspollAllowReceive` (~518–528), `verseCrosspollPerCycleLimit` (~530–538).

- [ ] **Step 4: Delete `verseAutoApplyThreshold`**

Remove the `conf.registerGlobalValue(LLM, "verseAutoApplyThreshold", …)` block (~390–399). (Loom-only; its sole consumer was `_wire_loom_if_enabled`. Easy to miss in a crosspoll-only grep.)

- [ ] **Step 5: Confirm exactly 14 keys are gone and `verseCompactionModel` remains**

Run: `grep -n "loom\|Crosspoll\|verseAutoApplyThreshold\|verseCompactionModel" plugins/llm/src/llm/config.py`
Expected: only `verseCompactionModel` matches.

- [ ] **Step 6: Run the full suite**

Run: `make test`
Expected: PASS, coverage ≥ 93%.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/config.py
git commit -m "chore(verse): remove 14 loom-family config keys"
```

---

## Task 11: Regression sweep — coverage gate + clean start with stale keys

Prove the de-loomed plugin starts cleanly even when a prod `bot.conf` still carries stale `loom*` registry values, and that the coverage gate holds.

**Files:**
- Modify: `plugins/llm/tests/test_plugin_verse.py` (add a de-loomed-boot smoke test beside the existing plugin-boot tests)

- [ ] **Step 1: Write a de-loomed-boot smoke test (real fixture pattern, NOT file I/O)**

Goal (spec §5.5/§7): the de-loomed plugin constructs cleanly and reads `verseCompactionModel`, even when a stale prod registry still answers `loom*` keys. Do **not** use `registry.open_registry` — it loads the global registry module and proves nothing about the plugin. Instead reuse the plugin-construction helper already used in `test_plugin_verse.py` (grep `make_registry_side_effect` in `plugins/llm/tests/conftest.py` and how the plugin-boot tests build a plugin), supplying a registry whose lookups also return stale `loom*` values:

```python
def test_deloomed_plugin_boots_with_stale_loom_keys(mocker):
    """Construct the plugin with a registry that STILL answers stale loom*
    keys; it must build and resolve verseCompactionModel without error.

    Adapt the construction to the existing make_registry_side_effect helper
    (arg names per conftest.py). The point: nothing in the de-loomed plugin
    reads loom* keys, so their lingering presence is inert; verseCompactionModel
    is the only model key compaction now reads.
    """
    overrides = {
        "loomChannel": "#dead",           # stale, must be harmless
        "loomModel": "gemini/gemini-flash-lite-latest",
        "verseAutoApplyThreshold": 0.85,
        "verseCompactionModel": "gemini/gemini-flash-lite-latest",
    }
    plugin = _construct_plugin_with_registry(mocker, overrides)  # per existing helper
    assert plugin.registryValue("verseCompactionModel") == "gemini/gemini-flash-lite-latest"
```

> The "stale keys in a real `bot.conf` don't fatal" guarantee is Limnoria's (it ignores registry entries with no registered definition); that half is verified for real at rollout step 1, not in a unit test. This test covers *our* half: the de-loomed plugin neither reads nor trips over `loom*` keys.

- [ ] **Step 2: Run it**

Run: `uv run pytest plugins/llm/tests/test_plugin_verse.py -k deloomed_plugin_boots -v`
Expected: PASS.

- [ ] **Step 3: Full gate**

Run: `make test`
Expected: PASS, `--cov-fail-under=93` satisfied. If coverage dipped below 93% because a kept helper lost its only (loom-test) coverage, add a focused unit test for that helper (e.g. a direct `apply_direct`/`get_proposal` test) rather than lowering the gate.

- [ ] **Step 4: Lint + typecheck**

Run: `make lint && make typecheck`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/tests/test_plugin_verse.py
git commit -m "test(verse): smoke-test de-loomed plugin boot with stale loom* keys present"
```

---

## Task 12 (Part B): Implement and test `purge_loom_data`

A tested, one-time purge for a single verse store. Pure code here — it is **not** run against prod in this task (that's the rollout runbook below). One responsibility, one `write_transaction`, inline SQL only.

**Files:**
- Create: `plugins/llm/src/llm/verse/purge.py`
- Create: `plugins/llm/tests/verse/test_purge.py`

- [ ] **Step 1: Write the failing tests**

Create `plugins/llm/tests/verse/test_purge.py`. Reuse the `store` fixture from the existing verse tests (grep `def store` under `plugins/llm/tests/` — it lives in a `conftest.py`; the fixture is injected by name).

```python
import json

import pytest

from llm.verse.purge import list_loom_digest_candidates, purge_loom_data


def _add_event(store, summary, entity_ids, source):
    """Seed one event with an event_actor link, returning its id."""
    return store.apply_direct(
        op="add_event",
        payload={"summary": summary, "entity_ids": list(entity_ids)},
        source=source,
        provenance="test",
    )


def test_apply_direct_writes_event_actor(store):
    """Sanity: the seeding path actually creates event_actor rows.

    The purge's orphan logic reads event_actor; if add_event didn't write it,
    every other assertion here would be meaningless.
    """
    eid = store.add_entity("npc", "probe", "")
    _add_event(store, "probe acted", [eid], "avatar")
    with store.read_connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM event_actor WHERE entity_id=?", (eid,)
        ).fetchone()[0]
    assert n >= 1


def test_purge_removes_idlerpg_junk_keeps_canon(store):
    # --- canon: a pinned roster entity with mixed-source events ---
    freddie = store.add_entity("npc", "Farty Freddie", "")
    store.set_attribute(freddie, "pinned", "1")
    keep_ev = _add_event(store, "Freddie's real deed", [freddie], "avatar")
    _add_event(store, "freddie defeats jspiros in combat", [freddie], "loom")

    # --- an authored operator event (no entity link) ---
    _add_event(store, "The Cathedral Siege", [], "operator")

    # --- a compaction lore-digest mis-stamped 'loom' (>300 chars) ---
    chronicler = store.add_entity("npc", "Stinky Sebastian", "")
    store.set_attribute(chronicler, "pinned", "1")
    digest_summary = "Chronicler fc42 recounts the anarchic reign of the Stinky Lads. " + (
        "Poo Pete and Assripping Alex schemed through the long winter, while " * 6
    )
    assert len(digest_summary) > 300
    digest_ev = _add_event(store, digest_summary, [chronicler], "loom")

    # --- orphan auto-NPC: only loom events ---
    blaat = store.add_entity("npc", "blaat", "")
    store.set_attribute(blaat, "auto_created", "1")
    _add_event(store, "blaat defeats jspiros in combat", [blaat], "loom")

    # --- NON-orphan auto-NPC: has a surviving (avatar) event ---
    survivor = store.add_entity("npc", "survivor-npc", "")
    store.set_attribute(survivor, "auto_created", "1")
    _add_event(store, "survivor did a real thing", [survivor], "avatar")
    _add_event(store, "survivor combat junk", [survivor], "loom")

    # --- digest review: operator confirms the digest id(s) ---
    candidates = list_loom_digest_candidates(store, min_chars=300)
    cand_ids = [cid for cid, _ in candidates]
    assert digest_ev in cand_ids

    result = purge_loom_data(store, digest_ids=[digest_ev])

    # digest re-stamped + survives
    assert result.digests_restamped == 1
    with store.read_connection() as conn:
        src = conn.execute(
            "SELECT source FROM events WHERE id=?", (digest_ev,)
        ).fetchone()
        assert src is not None and src[0] == "llm"

    # all loom/crosspoll events gone
    with store.read_connection() as conn:
        leftover = conn.execute(
            "SELECT COUNT(*) FROM events WHERE source IN ('loom','crosspoll')"
        ).fetchone()[0]
    assert leftover == 0

    # canon intact
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (freddie,)).fetchone()
        assert conn.execute("SELECT 1 FROM events WHERE id=?", (keep_ev,)).fetchone()
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (chronicler,)).fetchone()
        assert conn.execute(
            "SELECT 1 FROM events WHERE summary='The Cathedral Siege'"
        ).fetchone()

    # orphan deleted, non-orphan survivor kept
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (blaat,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (survivor,)).fetchone()
    assert result.entities_deleted == 1


def test_purge_dual_linkage_guard_spares_json_only_reference(store):
    """An auto-NPC referenced only via a SURVIVING event's entity_ids JSON
    (no event_actor row) must NOT be deleted, even if event_actor says
    loom-only."""
    npc = store.add_entity("npc", "json-ghost", "")
    store.set_attribute(npc, "auto_created", "1")
    # A loom event links it via event_actor (would mark it orphan)...
    _add_event(store, "json-ghost combat", [npc], "loom")
    # ...but a surviving operator event references it ONLY via entity_ids JSON,
    # with no event_actor row (legacy linkage). Insert raw.
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?,?,?,?)",
            (1.0, "legacy json-only mention", json.dumps([npc]), "operator"),
        )

    purge_loom_data(store, digest_ids=[])

    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (npc,)).fetchone()


def test_purge_no_digests_is_safe(store):
    """digest_ids=() re-stamps nothing and still purges junk."""
    n = store.add_entity("npc", "blaat", "")
    store.set_attribute(n, "auto_created", "1")
    _add_event(store, "blaat combat", [n], "loom")
    result = purge_loom_data(store, digest_ids=[])
    assert result.digests_restamped == 0
    assert result.events_deleted == 1


def test_purge_spares_autocreated_entity_whose_only_event_is_a_restamped_digest(store):
    """The re-stamp (step 0) runs BEFORE orphan computation (step 1): an
    UNPINNED auto-NPC whose sole event is a reviewed digest keeps a surviving
    ('llm') link after the re-stamp and is therefore NOT deleted. This is the
    case the big test misses (it uses a pinned chronicler, which survives for a
    different reason) — it proves the ordering matters."""
    npc = store.add_entity("npc", "lonely-chronicled-npc", "")
    store.set_attribute(npc, "auto_created", "1")
    long_summary = "A chronicle of the lonely npc's deeds. " + (
        "It wandered far and its doings were recorded at length. " * 8
    )
    assert len(long_summary) > 300
    digest = _add_event(store, long_summary, [npc], "loom")

    result = purge_loom_data(store, digest_ids=[digest])

    assert result.digests_restamped == 1
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (npc,)).fetchone()
        src = conn.execute("SELECT source FROM events WHERE id=?", (digest,)).fetchone()
        assert src is not None and src[0] == "llm"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest plugins/llm/tests/verse/test_purge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.verse.purge'`.

> If `test_apply_direct_writes_event_actor` fails (event_actor not written by `add_event`), STOP and read `_apply_op_inline` in `store.py`: the purge's orphan logic depends on `event_actor`. If `add_event` writes only the `entity_ids` JSON, the orphan SQL must instead/also parse that JSON. Adjust the design before proceeding and note it in the commit. (Per the v1 work, `event_actor` was backfilled and is written on add — this test makes that assumption explicit.)

- [ ] **Step 3: Implement `verse/purge.py`**

```python
"""One-time #idlerpg loom/crosspoll exhaust purge for a single verse store.

NOT a migration: destructive, channel-specific, invoked explicitly ONCE
against prod #afternet after a WAL-safe backup (see the slice-1 design doc
§6). Never auto-runs. Everything happens in ONE ``write_transaction`` with
direct SQL so it respects the store's non-reentrant write lock — never call
public store methods from inside it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

_LOG = logging.getLogger("llm.verse.purge")


class PurgeResult(NamedTuple):
    events_deleted: int
    entities_deleted: int
    digests_restamped: int


def list_loom_digest_candidates(store: Any, *, min_chars: int = 300) -> list[tuple[int, str]]:
    """Return ``(id, summary)`` for ``source='loom'`` events long enough to be
    compaction lore-digests rather than #idlerpg combat lines.

    Read-only. The operator REVIEWS this list and passes the confirmed ids to
    :func:`purge_loom_data` as ``digest_ids`` — we never re-stamp on length
    alone (a long combat brag could clear the threshold).
    """
    with store.read_connection() as conn:
        rows = conn.execute(
            "SELECT id, summary FROM events WHERE source='loom' "
            "AND length(summary) >= ? ORDER BY id",
            (min_chars,),
        ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def purge_loom_data(store: Any, *, digest_ids: Sequence[int] = ()) -> PurgeResult:
    """Delete #idlerpg loom/crosspoll events + their orphaned auto-NPCs.

    ``digest_ids`` are the reviewed compaction lore-digest event ids (from
    :func:`list_loom_digest_candidates`). They are re-stamped ``source='llm'``
    FIRST, so they survive the event delete and so their actors are not
    counted as loom-only orphans.

    One ``write_transaction``: on any error the whole op rolls back.
    """
    digest_id_list = [int(x) for x in digest_ids]
    with store.write_transaction() as conn:
        # 0. Protect reviewed compaction digests (only flip rows still 'loom').
        restamped = 0
        if digest_id_list:
            placeholders = ",".join("?" for _ in digest_id_list)
            cur = conn.execute(
                f"UPDATE events SET source='llm' "
                f"WHERE id IN ({placeholders}) AND source='loom'",
                digest_id_list,
            )
            restamped = cur.rowcount

        # 1. Compute orphans BEFORE deleting events (event_actor cascades in
        #    step 2). Orphan = auto_created, not pinned/author_locked, has >=1
        #    actor link, and NONE of its actor links point to a surviving
        #    (non-loom/crosspoll) event.
        orphan_rows = conn.execute(
            """
            SELECT e.id FROM entities e
            WHERE EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='auto_created' AND a.value='1'
            )
            AND NOT EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='pinned' AND a.value='1'
            )
            AND NOT EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='author_locked' AND a.value='1'
            )
            AND EXISTS (
                SELECT 1 FROM event_actor ea WHERE ea.entity_id=e.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM event_actor ea JOIN events ev ON ev.id=ea.event_id
                WHERE ea.entity_id=e.id AND ev.source NOT IN ('loom','crosspoll')
            )
            """
        ).fetchall()
        orphan_ids = {int(r[0]) for r in orphan_rows}

        # Dual-linkage guard: an entity can also be referenced by a surviving
        # event via the legacy events.entity_ids JSON without an event_actor
        # row. Never delete such an entity even if event_actor says orphan.
        if orphan_ids:
            referenced_json: set[int] = set()
            for (blob,) in conn.execute(
                "SELECT entity_ids FROM events WHERE source NOT IN ('loom','crosspoll')"
            ).fetchall():
                try:
                    for eid in json.loads(blob or "[]"):
                        if isinstance(eid, int):
                            referenced_json.add(eid)
                except (ValueError, TypeError):
                    # A corrupt entity_ids blob on a SURVIVING event must not
                    # silently lose its protective effect in a destructive op.
                    # We skip the row (its ids aren't added to the protected
                    # set) but WARN so the operator reviews before trusting the
                    # counts — an entity that should have been spared could
                    # otherwise be deleted unnoticed.
                    _LOG.warning(
                        "purge: unparseable entity_ids on a surviving event; "
                        "skipping it for dual-linkage protection"
                    )
                    continue
            orphan_ids -= referenced_json

        # 2. Delete loom/crosspoll events (cascades event_actor; the legacy
        #    entity_ids JSON lives on the deleted row, so both linkages go).
        cur = conn.execute("DELETE FROM events WHERE source IN ('loom','crosspoll')")
        events_deleted = cur.rowcount

        # 3. Delete orphaned auto-NPCs (cascades attributes/relations/
        #    entity_alias/event_actor).
        entities_deleted = 0
        if orphan_ids:
            id_list = sorted(orphan_ids)
            placeholders = ",".join("?" for _ in id_list)
            cur = conn.execute(
                f"DELETE FROM entities WHERE id IN ({placeholders})", id_list
            )
            entities_deleted = cur.rowcount

    return PurgeResult(events_deleted, entities_deleted, restamped)
```

- [ ] **Step 4: Run the purge tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/verse/test_purge.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Full gate**

Run: `make test && make lint && make typecheck`
Expected: PASS, coverage ≥ 93%, clean.

- [ ] **Step 6: Commit**

```bash
git add plugins/llm/src/llm/verse/purge.py plugins/llm/tests/verse/test_purge.py
git commit -m "feat(verse): add tested one-time purge_loom_data (Part B)"
```

---

## Rollout runbook (operator — NOT a code task)

Execute only after the whole branch above is merged and deployed. Destructive prod actions need explicit confirmation per project rules.

1. **Deploy Part A** (merge → auto-deploy on Docker green). Confirm a clean start: `docker logs vibebot` shows no fatal on stale `loom*` registry keys and **no** empty-`loomBotNicks` WARN (the loom is gone). Confirm the daily compaction timer arms (grep the boot log for the compaction schedule).
2. **WAL-safe backup** of prod `_afternet_2de47b99.db` (+ `-wal`/`-shm`) — copy to a timestamped path. (Host has no `sqlite3` binary; use host `python3`'s stdlib `sqlite3`, or `scp` the DB local and run via `PYTHONPATH=plugins/llm/src uv run python`.)
3. **Review the digest ids** against a local copy first (the bot must be running for this read-only step is fine; or use the backup from step 2):

   ```bash
   scp -i ~/.ssh/id_rsa "vibebot@rdrake.org:/config/data/verse/_afternet_2de47b99.db*" /tmp/verse-afnet/
   ```
   ```python
   # from repo root: PYTHONPATH=plugins/llm/src uv run python
   from llm.verse.store import VerseStore
   from llm.verse.purge import list_loom_digest_candidates
   store = VerseStore("/tmp/verse-afnet", "#afternet")   # same (data_dir, channel) the bot uses
   for cid, summary in list_loom_digest_candidates(store, min_chars=300):
       print(cid, repr(summary[:120]))
   ```
   Read every row and confirm each is a chronicle digest, not a long combat brag. Record the confirmed id list, e.g. `[1234, 1240, …]`.

4. **Run the purge once** against the **prod** DB, with the bot stopped to avoid WAL contention (mirror the stop→write→start procedure used for `bot.conf` edits — see the `reference_verse_prod_db_ops` memory for the exact paths/one-off-container path):

   ```python
   # PYTHONPATH=plugins/llm/src uv run python, pointed at the PROD verse data dir
   from llm.verse.store import VerseStore
   from llm.verse.purge import purge_loom_data
   store = VerseStore("<prod verse data dir>", "#afternet")
   result = purge_loom_data(store, digest_ids=[1234, 1240])   # the reviewed ids
   print(result)   # PurgeResult(events_deleted=…, entities_deleted=…, digests_restamped=…)
   ```
   `digests_restamped` MUST equal the number of ids you reviewed. Log the full result. Restart the bot.

   **Operator cautions (from the adversarial purge review):**
   - After the run, scan the log for any `purge: unparseable entity_ids on surviving event id=…` WARNING. A corrupt JSON blob is skipped for dual-linkage protection (warned, not fatal); if one appears, manually confirm the entity it referenced wasn't wrongly deleted before trusting the counts.
   - The dual-linkage guard only protects entity ids that appear as **integers** in a surviving event's `entity_ids` JSON. The #afternet store writes int arrays (and `event_actor` was backfilled in v1), so this is safe in practice; if you suspect any hand-inserted rows with string ids, confirm before running.
5. **Spot-check a verse turn for fc42:** roster + authored events + the chronicle digests present; no #idlerpg combat lines.

**Rollback:** revert the PR + restore the DB backup.

---

## Self-Review (run after writing, before the red-team)

**Spec coverage** — each design section maps to a task:
- §5.1 delete loom/crosspoll source → Task 8. §5.2 relocate validate_payload → Task 2; relocate client → Task 3. §5.3 plugin wiring + moderation trio + `_PluginLoomBridge` → Tasks 5–6 (KEEP `@versedit`/`@versecompact`). §5.4 store helpers + `bump_last_seen_ts` + digest source + docstring → Tasks 4, 9. §5.5 config keys + `verseCompactionModel` → Tasks 1, 10. §5.6 de-loomed tests → Tasks 3, 7, 8. §5.7 no migration → Task 9 Step 5. §5.8 coverage → Task 11. §6 purge → Task 12. §7 test plan → Tasks 11–12. §8 rollout → runbook.
- **Ordering invariant:** relocate (1–3) → digest source (4) → delete plugin wiring (5–6) → de-loom aging test (7) → delete loom/crosspoll (8) → delete dead store helpers (9) → remove config keys (10) → regression (11) → purge code (12). Every commit leaves `make test` green because each delete is preceded by the relocation/repoint that frees its consumers.

**Type/name consistency:** relocated client is `LiteLLMVerseClient`/`VerseCallUsage`/`VerseModelClient` everywhere (compaction.py defn, plugin repoint, `_fakes`, `_FakeClient`, patch target `llm.verse.compaction.LiteLLMVerseClient`). Validator is `llm.verse.validation.validate_payload`. New config key is `verseCompactionModel` (global `registry.String`). Purge API is `purge_loom_data(store, *, digest_ids)` + `list_loom_digest_candidates(store, *, min_chars)` → `PurgeResult(events_deleted, entities_deleted, digests_restamped)`.

**Load-bearing assumption — VERIFIED by red-team:** the purge orphan logic and Task 7's heartbeat rewrite assume `apply_direct(op="add_event")` writes `event_actor` rows. Confirmed real: `apply_direct → _apply_op_inline → _add_event_inline` (store.py:687–699) writes both the `events` row and `event_actor` joins. The `test_apply_direct_writes_event_actor` sanity test + STOP instruction remain as a guard. The dual-linkage guard covers the legacy-JSON-only edge regardless.

## Red-team outcome (task wl60fx910 — 45 agents, find→verify)

A 7-dimension adversarial red-team of this plan (every finding code-cited, then independently verified) surfaced **11 confirmed defects** (38 raised, 27 refuted). All 11 are folded in above:

1. **[blocker]** `isinstance()` on a non-`@runtime_checkable` Protocol would `TypeError` → Task 3 test now uses a structural `hasattr`/`callable` check.
2. **[high]** Task 5 missed the `self._wire_loom_if_enabled()` call site in `__init__` (plugin.py:805) → Step 6 now deletes it explicitly.
3–4. **[high]** Tasks 6 & 9 under-specified which test classes to delete → both steps now name the classes (`TestVerseproposalsCommand`/`TestVerseapproveRejectCommands`/`TestVerseapproveCrosspollSource`; `TestApplyAndRecordProposal`/`TestApplyProposalAndMark`/`TestApplyProposalAndMarkEventSource`/`TestProposal*`) + the `test_plugin_dispatch.py` help-sync list.
5. **[high]** Task 11's `registry.open_registry` test proved nothing → replaced with a real plugin-boot smoke test via `make_registry_side_effect`.
6. **[high]** Part B had no concrete operator invocation → rollout runbook now has the exact `VerseStore`/`purge_loom_data` REPL incantation.
7. **[medium]** Purge's JSON-parse `except` was silent in a destructive context → now `_LOG.warning`s.
8. **[medium]** Task 7's rewritten queued-proposal test was a tautology → now deleted outright; plus a new purge test for the unpinned-digest-only ordering case.

The 27 refuted findings **validated the purge core**: FK `ON DELETE CASCADE` is enforced (`PRAGMA foreign_keys=ON` in store.py `_connect`), `_add_event_inline` writes `event_actor`, attribute booleans are TEXT `'1'`, `apply_direct(op="add_event")` returns the new event id, and the single-`write_transaction` atomicity holds.
