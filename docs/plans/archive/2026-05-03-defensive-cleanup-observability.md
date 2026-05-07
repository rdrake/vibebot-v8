# Defensive-Code Cleanup & Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace string-sniffing tool-result handlers with structured returns, fix logging-severity inconsistencies that hide production failures, standardize the `memories` command on `irc.error`, and remove framework-attribute defensiveness *only where the invariant is verifiable and tested*.

**Architecture:** This plan is a series of small, narrow edits. Each task either upgrades observability or removes a guard whose invariant is verifiable, with a regression test pinning the new behavior. Plan C is independent of Plans A and B and can land in any order, but landing it after Plan B avoids touching the same lines twice.

**Tech Stack:** Python 3.12+, Limnoria callbacks, pytest. Lint with `make lint`, types with `make typecheck`, tests with `make test`. Coverage floor is **93%**.

**Plan note:** The original draft included broad framework-guard removal across `plugin.py` and `service.py`. After review, those edits are **narrowed** to ones where (a) Limnoria's contract is verifiable from its public API, and (b) a regression test pins the invariant. Removals like `bot_nick = active_irc.nick`, `doPrivmsg`'s `args[1]` guard, and `world.startedAt` type narrowing are **not in this plan** — the cost of a stray `IndexError`/`AttributeError` in production exceeds the cleanup payoff.

---

### Task 1: Replace string-sniffing tool-result handlers with structured returns

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py:980, 997, 1041` and the callback signatures they consume
- Modify: `plugins/llm/src/llm/plugin.py` callback producers (see inventory below)
- Test: `plugins/llm/tests/test_assistant.py`, `plugins/llm/tests/test_plugin.py`

**Verified producer/consumer inventory:**

| Consumer (`assistant.py`) | Method name | Consumes callback | Producer (`plugin.py`) |
|---|---|---|---|
| `_tool_cleanup_memories` (~979) | `_cleanup_fn` | `_run_memory_cleanup` (~2302) |
| `_tool_set_reminder` (~992) | `_set_reminder_fn` | `_remind_set_for_assistant` (~3670) |
| `_tool_generate_image` (**not `_tool_draw`**) (~1034) | `_draw_fn` | `_draw_for_assistant` (~1811) |

`_remind_delete_for_assistant` (~3694) is consumed by `cancel_pending_task_fn` (the `_tool_cancel_pending_task` closure at plugin.py:~3191), where its return string is parsed by substring match `"not found" not in message.lower()`. Include this producer in the migration.

**Verification before edits:** Search for any other callers of the four producer methods to confirm only the assistant path consumes them. Internal `_remind_*_for_assistant` and `_draw_for_assistant` should be assistant-only by their names; verify with:

```bash
grep -n "_run_memory_cleanup\|_remind_set_for_assistant\|_remind_delete_for_assistant\|_draw_for_assistant" plugins/llm/src/llm/
```

If any caller besides the assistant tool path uses the string return value, that caller must be updated too — the type change is breaking.

**Step 1: Define the structured return type**

In `assistant.py`, near the other named tuples:

```python
class ToolCallbackResult(NamedTuple):
    """Structured result from plugin-side callbacks invoked by tool handlers.

    ``ok=False`` means the operation failed; ``message`` is human-readable
    text safe to surface to the LLM (no secrets, no internal tracebacks).
    """
    ok: bool
    message: str
```

**Step 2: Write failing tests**

```python
def test_tool_cleanup_returns_err_when_callback_says_not_ok(make_assistant):
    a = make_assistant(
        cleanup_fn=lambda *args, **kw: ToolCallbackResult(False, "boom"),
    )
    out = a._tool_cleanup_memories({"target": "alice"})
    assert json.loads(out)["status"] == "error"


def test_tool_cleanup_does_not_misclassify_success_with_error_word(make_assistant):
    a = make_assistant(
        cleanup_fn=lambda *args, **kw: ToolCallbackResult(
            True, "Removed 3 errors from your memories"
        ),
    )
    out = a._tool_cleanup_memories({"target": "alice"})
    assert json.loads(out)["status"] == "ok"
```

Add analogous tests for `_tool_set_reminder` and `_tool_generate_image`. Use the existing test fixtures and helpers — read `test_assistant.py` and `conftest.py` first.

**Step 3: Run, confirm fail**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -k tool_ -v
```

**Step 4: Update the three handlers in `assistant.py`**

```python
# _tool_cleanup_memories (~979)
result = self._cleanup_fn(target)
if not result.ok:
    return self._err(result.message)
return self._ok(result.message)

# _tool_set_reminder (~997)
result = self._set_reminder_fn(text)
if not result.ok:
    return self._err(result.message)
return self._ok(result.message)

# _tool_generate_image (~1041)
result = self._draw_fn(prompt)
if not result.ok:
    return self._err(result.message)
return self._ok(result.message)
```

**Step 5: Update producers in `plugin.py`**

Each producer must return a `ToolCallbackResult` instead of a string. Construct the result based on the actual operation outcome (not by parsing your own message):

```python
# _run_memory_cleanup
def _run_memory_cleanup(self, nick, channel) -> ToolCallbackResult:
    try:
        summary = self._service.cleanup_memories(nick, channel=channel)
    except Exception as e:
        self.log.exception("memory cleanup failed for %s: %s", nick, self._sanitize(str(e)))
        return ToolCallbackResult(False, "Memory cleanup failed.")
    if summary.error:
        return ToolCallbackResult(False, summary.error)
    return ToolCallbackResult(True, summary.message)
```

(Match the existing logic — read each producer first.)

**Step 6: Update `cancel_pending_task_fn` to use the structured result**

At `plugin.py:~3191`:

```python
result = self._remind_delete_for_assistant(caller, task_id, irc=react_irc, msg=react_msg)
return {
    "status": "ok" if result.ok else "error",
    ...
    "message": result.message,
}
```

**Step 7: Run all affected test files**

```bash
uv run pytest plugins/llm/tests/test_assistant.py \
              plugins/llm/tests/test_plugin.py \
              plugins/llm/tests/test_reminders.py \
              plugins/llm/tests/test_commands.py -v
```
Expected: PASS.

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "refactor(assistant): structured ToolCallbackResult; drop string-sniffing"
```

---

### Task 2: Upgrade silent or low-severity exception handlers

**Files:**
- Modify: `plugins/llm/src/llm/service.py:249` (`ExtractionResult` type — add `error` field)
- Modify: `plugins/llm/src/llm/service.py:3834` (`extract_memories` silent except)
- Modify: `plugins/llm/src/llm/service.py:2535` (`_ask_completion` debug-log + silent return)
- Modify: `plugins/llm/src/llm/service.py:3073` (`assistant_completion` `.error` → `.exception`)
- Modify: `plugins/llm/src/llm/plugin.py:712` (`_deliver_pending_result` bare except)

**Verified state of `ExtractionResult`:** currently `class ExtractionResult(NamedTuple): add: list[str] = []`. **Has no `error` field.** Step 1 must add it before Step 4 references it.

**Verified state of `_ask_completion`:** at `service.py:2527` it calls `litellm.completion(...)` directly, **not** `_completion_with_tool_fallback`. Tests must patch `litellm.completion` (or, more cleanly, mock the method at module level via `monkeypatch.setattr(service.litellm, "completion", ...)`).

**Step 1: Extend `ExtractionResult`**

```python
class ExtractionResult(NamedTuple):
    """Result of memory extraction."""
    add: list[str] = []
    error: str | None = None
```

**Step 2: Write failing tests**

```python
def test_extract_memories_logs_and_records_error_on_exception(service, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr("plugins.llm.src.llm.service.litellm.completion", boom)
    # If extract_memories goes through a different code path, patch the right
    # litellm entry point used by service.extract_memories.

    with caplog.at_level("ERROR", logger="LLM"):
        result = service.extract_memories(...)

    assert result.error is not None
    assert "boom" not in result.error  # sanitized
    assert any("extract_memories failed" in r.message for r in caplog.records)


def test_ask_completion_logs_at_info_on_failure(service, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("nope")
    monkeypatch.setattr("plugins.llm.src.llm.service.litellm.completion", boom)

    with caplog.at_level("INFO", logger="LLM"):
        out = service._ask_completion("sys", "user", channel=None)
    assert out is None
    assert any("Ask completion failed" in r.message for r in caplog.records)
```

Adjust the `monkeypatch.setattr` target to match how `litellm` is imported in `service.py` (it is imported at module top, so the path above should work — verify before running).

**Step 3: Run, confirm fail**

```bash
uv run pytest plugins/llm/tests/test_service.py -k "extract_memories_logs or ask_completion_logs" -v
```

**Step 4: Patch the four sites**

```python
# service.py extract_memories (~3834)
except Exception as e:
    self.log.exception("extract_memories failed: %s", self._sanitize(str(e)))
    return ExtractionResult(error=self._sanitize(str(e)))

# service.py _ask_completion (~2535)  -- INFO, not WARNING (graceful degradation)
except Exception as e:
    self.log.info("Ask completion failed: %s", self._sanitize(str(e)))
    return None

# service.py assistant_completion (~3073)  -- already logs at .error; upgrade to .exception
except Exception as e:
    self._log_server_headers(e)
    self.log.exception("assistant_completion failed: %s", self._sanitize(str(e)))
    return AssistantResult(
        content="Sorry, something went wrong.",
        error=self._sanitize(str(e)),
    )

# plugin.py _deliver_pending_result inner queueMsg (~712)
except Exception as e:
    self.log.warning(
        "queueMsg failed for task_id=%s: %s",
        r.task_id, self._sanitize(str(e)) if hasattr(self, "_sanitize") else e,
    )
    delivered = False
```

**Severity rationale:**
- `extract_memories` → `.exception()` (records traceback): novel failures need the stack to diagnose.
- `_ask_completion` → `.info()` (not `.warning()`): this is a graceful-degradation helper used by summarization and similar non-critical paths; promoting to warning would spam logs on transient LLM hiccups.
- `assistant_completion` → `.exception()`: most complex code path, needs traceback.
- `_deliver_pending_result` queueMsg → `.warning()`: delivery failure with task identity is the actionable info.

**Always sanitize:** every error string surfaced to the user (or written to `result.error`) must pass through `self._sanitize(...)` to redact API keys per AGENTS.md security invariants.

**Step 5: Run tests**

```bash
uv run pytest plugins/llm/tests/test_service.py plugins/llm/tests/test_plugin.py -v
```

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "fix(observability): upgrade severity, add tracebacks, sanitize error fields"
```

---

### Task 3: Add log breadcrumb for reminder-validation rejections

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:3661-3668` (`_remind_set`)

**Step 1: Add an info-level log on the failure branch**

```python
def _remind_set(self, irc, msg, caller, text):
    result = self._schedule_reminder(irc, msg, caller, text)
    if result.ok:
        self._ack(irc, msg, "⏰", result.message, prefixNick=True)
        return
    self.log.info(
        "remind_set blocked nick=%s reason=%s",
        caller.key, result.message,
    )
    self._react(irc, msg, "❌")
    irc.error(_(result.message))
```

**Step 2: Run reminder tests**

```bash
uv run pytest plugins/llm/tests/test_reminders.py -v
```

**Step 3: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "feat(plugin): log validation rejections from _remind_set"
```

---

### Task 4: Standardize `memories` command on `irc.error()` for failure paths

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:2825, 2839` (the `irc.reply("Usage: …")` failure replies)
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Verify each candidate line is a failure path**

Read the surrounding context for each `irc.reply("Usage: ...")` call in the `memories` command group. Only convert lines where the function returns immediately after with no further work — those are the failure-mode replies. Lines that print usage on success (e.g., `memories help`) keep `irc.reply`.

**Step 2: Convert failure-path replies**

```python
# Before
irc.reply("Usage: memories delete <id> [<id> ...]", prefixNick=False)
return

# After
irc.error("Usage: memories delete <id> [<id> ...]")
return
```

**Step 3: Add a test pinning the prefix**

```python
def test_memories_delete_bad_arg_uses_irc_error(run_command):
    out = run_command("memories", "delete")  # missing required id
    assert any(r.startswith("Error:") for r in out.errors)
```

**Step 4: Run tests**

```bash
uv run pytest plugins/llm/tests/test_commands.py -k memories -v
```

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "fix(plugin): standardize memories command on irc.error for failures"
```

---

### Task 5: Targeted false-guard removal in `service.py`

**Scope (narrowed from original draft):** only the guards where Limnoria's API contract is verifiable from its module surface, not from incidental runtime state. Excluded from this task: `bot_nick = active_irc.nick` (active_irc provenance unverified), `doPrivmsg`'s `args[1]` guard (defense against malformed server messages), `world.startedAt` type narrowing (no measurable benefit, low risk).

**Files:**
- Modify: `plugins/llm/src/llm/service.py:458` — `isinstance(key, str)` on a `String` registry value
- Modify: `plugins/llm/src/llm/service.py:1174-1185` — duplicate `hasattr(response, "_hidden_params")` checked twice on the same object

**Step 1: Patch each site**

```python
# 458
if key:
    result = result.replace(key, "[REDACTED]")

# 1174-1185 — collapse to single read
hidden = getattr(response, "_hidden_params", None) or {}
if hidden.get("vertex_ai_grounding_metadata"):
    ...
if hidden.get("citations") or ...:
    ...
```

**Step 2: Run tests**

```bash
uv run pytest plugins/llm/tests/test_service.py -v
```

**Step 3: Commit**

```bash
git add plugins/llm/src/llm/service.py
git commit -m "refactor(service): drop redundant defensive guards on registry value and hidden_params"
```

---

### Task 6: Targeted false-guard removal in support files

**Scope:** only `context.py:354` where the dict producer is in the same file and the key is unconditionally set. The `limnoria_bridge.py` log getattrs and `narrator.py` per-call check are deferred — see footnotes.

**Files:**
- Modify: `plugins/llm/src/llm/context.py:354` — `msg.get("nick", "")`

**Step 1: Verify producers**

`add_channel_message` (~line 304) constructs channel-message dicts as `{"nick": nick, "role": role, "content": content}`. Confirm by reading `context.py:300-360` and checking that **every** code path that writes to the channel-messages list sets the `"nick"` key. If any path omits it, do not proceed.

**Step 2: Patch**

```python
# Before
if msg.get("nick", "").lower() != exclude_lower:

# After
if msg["nick"].lower() != exclude_lower:
```

**Step 3: Run tests**

```bash
uv run pytest plugins/llm/tests/test_context.py -v
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/context.py
git commit -m "refactor(context): drop msg.get('nick', '') guard; key is producer-guaranteed"
```

**Deferred:**
- `limnoria_bridge.py:297-298` `getattr(msg, "nick", "?")` — Removing this is fine if `dispatch()` later accesses `msg.channel` unguarded (currently does at line 331). Defer to a follow-up because the payoff is purely cosmetic and the audit must verify every caller of `dispatch()`.
- `narrator.py` per-call truthiness — moving to `__init__` is a behavior change (loses runtime config-reload semantics). Rejected; keep per-call.

---

### Task 7: Final preflight

```bash
make preflight
```
Expected: PASS, coverage ≥ 93%.

If a task removed branches that previously had implicit coverage, add small focused tests; do not lower the floor.
