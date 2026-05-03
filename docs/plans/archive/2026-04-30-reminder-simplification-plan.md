---
status: revised-after-code-review
date: 2026-04-30
---

> **Revision note (round 1, Claude code-reviewer):** A1 split into A1a/A1b.
> A2 line numbers corrected, `_for_assistant` excluded, `prefixNick` parameterized.
> A5 spelled out to guard both `irc.error` AND `db.log_usage` behind `silent`.
> A6 narrowed to `chain_id` only (TTL → B0.6). New B0.5 (in-flight migration),
> B0.6 (TTL drop), B1/B2/B4 reshaped for two-column recurrence (numeric +
> RRULE).
>
> **Revision note (round 2, Codex second opinion):** Five blockers fixed.
> (1) `python-dateutil` made an explicit direct dep before B4 (not transitive).
> (2) **B0.5 option (b) hardened with a strict routing gate** — structured
> rows take mechanical path ONLY; legacy rows take LLM-tool path ONLY. New
> task B3.5 asserts mutual exclusion. (3) B5 reply suppression switched from
> phrase-matching to a strict signal: assistant loop now exposes
> `last_successful_tool` metadata; suppression fires only on strictly
> empty/whitespace text after a reminder mutation. (4) B4 RRULE computation
> moved to timezone-aware UTC `datetime` objects; DST-boundary test added.
> (5) Architecture summary below rewritten to match the revised task split.

# Reminder Plumbing Simplification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collapse the redundancy that accumulated across the 21-commit reminder burst (commits `cf8ad30`..`74772e6`) into two coherent shapes — first mechanical refactors, then a structural redesign of recurrence/watch/[silent] semantics.

**Architecture:** Two sequenced PRs.

**PR A (mechanical, no behavior change):**
1. Replace the 8-tuple in-memory reminder store with the existing `ReminderRow` NamedTuple (A1a source + A1b ~44 fixture sites).
2. Extract `_ack` helper for IRCv3 reaction-with-text-fallback (four `remind()` sites; `_for_assistant` reactions intentionally untouched).
3. Extract `_reminder_fns(caller, irc, msg)` kwargs builder for the four duplicated tool-callback lambda blocks.
4. Revert `set_reminder`/`list_reminders`/`delete_reminder`/`cancel_all_reminders` visibility to `{chat, remind_action}` only.
5. Replace the `_check_rate_limit_silent` clone with a `silent: bool` parameter that guards both `irc.error` AND `db.log_usage(status="rate_limited")`.
6. Drop **only** the `chain_id` column (v11 migration); `chain_started_at` and the 30-day TTL stay intact in PR A.
7. Finish the `Identity` migration in callers that still launder accounts as `nick: str`.
8. Delete commit-narrating comments.

**PR B (structural):**
1. Add `python-dateutil` as a direct runtime dep on `plugins/llm`.
2. Decide in-flight reminder migration strategy (B0.5; recommended option (b) — graceful degradation).
3. Drop the 30-day chain TTL and `chain_started_at` column (B0.6); chain_position cap is the sole runaway guard.
4. Add **two** structured recurrence columns — `recurrence_seconds INTEGER` (numeric cadences) and `recurrence_rrule TEXT` (RFC 5545 calendar cadences) — plus `watch_mode INTEGER`. Exactly one recurrence column is non-null per recurring row; both null for one-shot.
5. Parser populates the structured fields directly; no more parenthetical embedding into `action_prompt`.
6. Recurring reminders reschedule mechanically at fire time (no LLM tool call). Numeric path computes `now + recurrence_seconds`; RRULE path uses timezone-aware UTC `dateutil.rrule.rrulestr(...).after(...)`.
7. **Strict transition gate (B3.5):** structured rows take mechanical path ONLY; legacy parenthetical-encoded rows take LLM-tool path ONLY. No row ever takes both — no double-reschedule.
8. Move post-tool silence from a `[silent]` chat-prompt contract to reply-path suppression keyed on a structured `last_successful_tool` signal from the assistant loop. `[silent]` retained only as the watch-mode no-news sentinel inside the reminder delivery closure.

**Tech Stack:** Python 3.14, Limnoria (Supybot), SQLite (`PRAGMA user_version` migrations), pytest with `pytest-mock`.

**Hard constraints:**
- Behavior preserved across PR A — only test-fixture *construction* (not assertion shape) changes; ~44 fixture sites build `_reminders[name] = (...)` 5-tuples relying on the `len(data) > 3` legacy guard, and they must be rewritten to `ReminderRow(...)` in Task A1b. No production behavior change in PR A.
- `ReminderRow` is a `NamedTuple` (`persistence.py:24-38`), not a dataclass — attribute access works identically; do not convert it to `@dataclass`.
- `Identity` lives at `plugin.py:70` (not `context.py`).
- Each PR ships green: `make test-all` passes at every commit boundary.
- **Migration safety procedure** (applies to A6, B0.6, B1 — all `DROP COLUMN`/`ADD COLUMN`):
  1. Stop the running bot (`systemctl --user stop vibebot`).
  2. Snapshot the SQLite DB file (`cp llm.db llm.db.pre-vN.bak`).
  3. Deploy the new image. The bot's startup applies the migration via `PRAGMA user_version`. Only one bot instance touches the DB at a time — no migrator-vs-runtime race.
  4. If startup fails: stop the bot, restore the snapshot, downgrade the image, restart. Document the failure for analysis.
  Each task that does a schema change must include this procedure in its commit message footer or link to this section.
- **Concurrency policy for `cancel_all_reminders` vs in-flight fires:** "clear wins." Snapshot the user's pending reminders under `_reminders_lock`; cancel each via `_cancel_reminder` which removes both the in-memory entry and the scheduler event under the same lock. If a fire is *currently mid-execution* (closure already running outside the lock), it completes — but its mechanical or LLM-tool reschedule MUST check the in-memory map and skip rescheduling if the original event is no longer present. B3.5 and B4 must enforce this; add a test in B3.5.
- **Granularity disclaimer:** the "2-5 minute step" guidance from `superpowers:writing-plans` is aspirational. A1b (44 fixture sites), B4 (mechanical reschedule + RRULE + tests), and B5 (assistant-loop metadata + suppression + tests) are larger. Treat their step lists as units of work, not stopwatch budgets.
- No NickServ terminology in user-facing strings (AfterNet has no NickServ — see memory).
- Existing reminder rows in production DBs must keep working — schema migrations are forward-only.
- Don't skip pre-commit hooks. Don't push to main without local green tests.

---

# PR A — Mechanical refactors

## Task A0: Snapshot baseline test count

**Why:** PR A is meant to be behavior-preserving. Capture the test count and pass-rate before changes so any drop is visible.

**Files:** none modified.

**Step 1: Run the LLM plugin test suite**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest plugins/llm/ -q 2>&1 | tail -5`
Expected: All tests pass. Record the test count in your notes (e.g. "247 passed").

**Step 2: No commit** — this is a baseline read.

---

## Task A1a: Replace `_reminders` 8-tuple with `ReminderRow` (source-side)

**Why:** `persistence.ReminderRow` (NamedTuple) already has every field the in-memory `_reminders` dict stores. The positional 8-tuple at `plugin.py:406-409` forces `data[0]`/`data[2]`/`data[4]` indexing throughout, requires `_stored_reminder_identity` to extract the identity, and triggers a `len(data) > 3` "legacy tuples" defensive guard at `plugin.py:2886` for tuples this module owns. Switching to `ReminderRow` removes ~40 lines including one helper.

> **Granularity note:** A1 was split into A1a (source code) and A1b (test fixtures, ~44 sites). Run A1a first; tests will fail at the boundary because fixtures still build 5-tuples. A1b fixes them. Both must land in the same PR but should be separate commits.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:406-409` (declaration block)
- Modify: `plugins/llm/src/llm/plugin.py:1264` (write site in `_reload_reminders`)
- Modify: `plugins/llm/src/llm/plugin.py:3060` (write site in `_remind_set`)
- Modify: `plugins/llm/src/llm/plugin.py:2832-2843` (delete `_stored_reminder_identity` helper)
- Modify: `plugins/llm/src/llm/plugin.py:2867` AND `plugin.py:2909` (both callers of `_stored_reminder_identity` — reviewer found two, not one)
- Modify: `plugins/llm/src/llm/plugin.py:2870-2893` (`_format_reminders` indexing + legacy guard)
- Modify: `plugins/llm/src/llm/plugin.py:2906-2911` (loop using indexing)
- Modify: `plugins/llm/src/llm/plugin.py:1042-1221` (`_make_reminder_delivery_closure` — check for `_reminders[event_name]` indexing)

**Step 1: Read `ReminderRow` to confirm field order**

Open `persistence.py:24-38`. Fields: `id, event_name, nick, channel, message, action_prompt, account, fire_at, created_at, chain_id, chain_position, chain_started_at`. The in-memory 8-tuple stores `(nick, channel, message, action_prompt, account, chain_id, chain_position, chain_started_at)` — a strict subset. Reuse `ReminderRow` directly; fill `id`, `event_name`, `fire_at`, `created_at` (event_name = dict key; the rest are in scope at write time).

**Step 2: Update declaration at `plugin.py:406-409`**

Replace:

```python
        self._reminders: dict[
            str,
            tuple[str, str, str, str, str | None, str, int, float],
        ] = {}
```

with:

```python
        self._reminders: dict[str, ReminderRow] = {}
```

Add `from llm.persistence import ReminderRow` (or `persistence.ReminderRow` — match the import style already in use).

**Step 3: Update both write sites**

At `plugin.py:1264` (`_reload_reminders`) and `plugin.py:3060` (`_remind_set`), replace the 8-element tuple literals with `ReminderRow(...)` keyword-arg constructions. All values already in scope.

**Step 4: Delete `_stored_reminder_identity` and inline both callers**

Delete the helper at `plugin.py:2832-2843`. Replace **both** callers — `plugin.py:2867` (in `_get_user_reminders`) and `plugin.py:2909` (in `_find_user_reminder`, which the reviewer caught) — with `Identity(raw_nick=data.nick, account=data.account).matches(caller)`.

**Step 5: Replace positional indexing in `_format_reminders`**

At `plugin.py:2870-2893`: `data[2]` → `data.message`, `data[3] if len(data) > 3 else ""` → `data.action_prompt`. Delete the legacy-tuple defensive comment.

**Step 6: Audit remaining positional indexing**

Run: `grep -n 'data\[' plugins/llm/src/llm/plugin.py`
Convert each hit where `data` came from `_reminders` to attribute access.

**Step 7: Lint and typecheck (skip pytest until A1b)**

Run: `uv run ruff check plugins/llm/ && uv run ty check plugins/llm/`
Expected: clean. Tests will be broken until A1b — that is expected and the next commit fixes them.

**Step 8: Commit**

```bash
git add plugins/llm/src/
git commit -m "refactor(llm): use ReminderRow for in-memory reminder store (src)

Replaces the 8-tuple _reminders dict with the existing ReminderRow
NamedTuple. Removes _stored_reminder_identity helper and the
len(data) > 3 legacy guard for tuples this module owns. Test fixtures
follow in next commit."
```

(The pre-commit hook may fail at this commit if it runs the test suite — if so, use `--no-verify` only after confirming the failure is exclusively the fixture-shape mismatch, then immediately do A1b and verify the hook passes there. Coordinate with the user before bypassing a hook.)

---

## Task A1b: Mass-rewrite test fixtures from 5-tuple to `ReminderRow`

**Why:** ~44 sites across `test_plugin.py`, `test_commands.py`, `test_reminders.py` construct `_reminders` entries as **5-element tuples** like `("testnick", "#test", "msg", "", None)` — they relied on the `len(data) > 3` defensive guard A1a removed. They all need to become `ReminderRow(...)` constructions.

**Files:**
- Modify: `plugins/llm/tests/test_plugin.py`
- Modify: `plugins/llm/tests/test_commands.py`
- Modify: `plugins/llm/tests/test_reminders.py`
- Modify: any other `plugins/llm/tests/*.py` matched by the grep below

**Step 1: Enumerate every fixture site**

Run: `grep -rn '_reminders\[' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/tests/`
Expected: ~44 hits. For each, note the file:line and the tuple shape. Common shapes are 5-tuple `(nick, channel, message, action_prompt, account)` and 8-tuple (full new shape).

**Step 2: Add a small fixture helper**

To avoid repeating `ReminderRow(...)` with default values across 44 sites, add a builder to `plugins/llm/tests/conftest.py`:

```python
from llm.persistence import ReminderRow


def make_reminder_row(
    *,
    event_name: str = "evt",
    nick: str = "testnick",
    channel: str = "#test",
    message: str = "",
    action_prompt: str = "",
    account: str | None = None,
    fire_at: float = 0.0,
    chain_id: str = "chain",
    chain_position: int = 1,
    chain_started_at: float = 0.0,
    id: int = 0,
    created_at: float = 0.0,
) -> ReminderRow:
    return ReminderRow(
        id=id,
        event_name=event_name,
        nick=nick,
        channel=channel,
        message=message,
        action_prompt=action_prompt,
        account=account,
        fire_at=fire_at,
        created_at=created_at,
        chain_id=chain_id,
        chain_position=chain_position,
        chain_started_at=chain_started_at,
    )
```

(After PR B Task B1 adds new columns, this helper grows. After Task B0.6 drops `chain_id`/`chain_started_at`, it shrinks. The helper localizes that churn.)

**Step 3: Convert each call site**

Pattern:

```python
# Before
plugin._reminders["evt1"] = ("testnick", "#test", "msg", "", None)

# After
plugin._reminders["evt1"] = make_reminder_row(
    nick="testnick", channel="#test", message="msg",
)
```

If a test asserts on tuple-positional access (`plugin._reminders["evt1"][0]`), convert to attribute access (`plugin._reminders["evt1"].nick`).

**Step 4: Run the suite**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest plugins/llm/ -q 2>&1 | tail -5`
Expected: same count as Task A0 baseline, all passing.

**Step 5: Lint**

Run: `uv run ruff check plugins/llm/`

**Step 6: Commit**

```bash
git add plugins/llm/tests/
git commit -m "test(llm): rewrite ~44 _reminders fixture sites for ReminderRow

Companion to the A1a refactor — fixtures previously built 5-tuples that
relied on the len(data) > 3 legacy guard. Adds make_reminder_row helper
in conftest to localize future column churn."
```

---

## Task A2: Extract `_ack(irc, msg, emoji, fallback_text)` helper

**Why:** Multiple `@remind` subcommand sites do "try `_react`, on failure `irc.reply`" with an emoji + fallback text. One helper centralizes the convention.

> **Reviewer correction:** The original plan cited `plugin.py:3084-3088, 3134-3148, 3160-3175, 3219-3237`. Wrong. Lines `3084-3088` are inside `_schedule_reminder` and have no `_react`. The actual react-with-fallback sites are inside `remind()` only — see Step 1. The `_for_assistant` helpers (`3124`, `3144`, `3149`, `3170`) call `_react` **without** a text fallback because the chat path is supposed to stay `[silent]` after a reaction; `_ack` MUST NOT be applied there or behavior changes. Also: the four `remind()` sites use varying `prefixNick` (the `_remind_set` reaction at `3098-3099` does NOT pass `prefixNick=False`, while the delete/clear sites at `3223-3225`, `3230-3231`, `3235-3237` do). `_ack` accepts `prefixNick` as a parameter to preserve each site's behavior exactly.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (add `_ack` near `_react`; convert call sites in `remind()` and `_remind_set`)

**Step 1: Confirm the exact call sites in code**

Read `plugin.py` directly. The five sites with the react-then-fallback pattern are:

| Site | Line | Emoji | Fallback | `prefixNick` |
|------|------|-------|----------|--------------|
| `_remind_set` success | 3098-3099 | ⏰ | `result.message` | default (True) |
| `remind()` delete success | 3223-3225 | 👍 | `f"Cancelled {deleted} {label}."` | False |
| `remind()` clear empty | 3230-3231 | 👌 | `_("No reminders to clear.")` | False |
| `remind()` clear success | 3235-3237 | 👍 | `f"Cleared {len(...)} {label}."` | False |

Plus `_remind_set` failure at `3101-3102`: this is **not** a react-with-fallback — it reacts ❌ AND calls `irc.error()` unconditionally. Leave it alone.

The `_for_assistant` reactions at `3124, 3144, 3149, 3170`: also leave alone. They have no fallback by design (chat profile stays `[silent]`).

**Step 2: Find `_react` to confirm return type**

Run: `grep -n 'def _react' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/src/llm/plugin.py`
Read the body. Confirm it returns `bool` (True if the reaction was sent; False if no msgid / no message-tags cap).

**Step 3: Write `_ack`**

Add immediately after `_react`:

```python
def _ack(
    self,
    irc: callbacks.Irc,
    msg: ircmsgs.IrcMsg,
    emoji: str,
    fallback_text: str,
    *,
    prefixNick: bool = False,
) -> None:
    """React with `emoji`; fall back to text if the server can't carry it.

    `prefixNick` mirrors the kwarg on `irc.reply` — pass True when the call
    site previously called `irc.reply(text)` with the default prefix, False
    when it explicitly disabled prefixing.
    """
    if not self._react(irc, msg, emoji):
        irc.reply(fallback_text, prefixNick=prefixNick)
```

(No `success: bool` — the emoji is site-specific, not a binary 👍/❌.)

**Step 4: Convert each site**

- `3098-3099`: `self._ack(irc, msg, "⏰", result.message, prefixNick=True)` — preserves the original default.
- `3223-3225`: `self._ack(irc, msg, "👍", f"Cancelled {deleted} {label}.")` — `prefixNick=False` is the helper default.
- `3230-3231`: `self._ack(irc, msg, "👌", _("No reminders to clear."))`. Make sure the `return` immediately after still fires.
- `3235-3237`: `self._ack(irc, msg, "👍", f"Cleared {len(user_reminders)} {label}.")`.

Do NOT touch the failure path at `3101-3102` (unconditional `irc.error`).

Do NOT touch any of the `_for_assistant` reaction calls.

**Step 5: Run tests**

Run: `uv run pytest plugins/llm/tests/test_reminders.py -q 2>&1 | tail -5`
Expected: pass. Reaction tests rely on `_react` being called with the right emoji — that's preserved.

**Step 6: Lint**

Run: `uv run ruff check plugins/llm/`

**Step 7: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): extract _ack helper for react-with-text-fallback

Four sites in remind() (set/delete/clear/clear-empty) collapsed to one
helper. _for_assistant reactions intentionally untouched — they have no
fallback because the chat path is contracted to stay [silent] after a
successful reminder tool call."
```

---

## Task A3: Extract `_reminder_fns(caller, irc, msg)` kwargs builder

**Why:** Four sites (`plugin.py:1153-1163`, `2262-2269`, `2377-2384`, `2473-2480`) build the same four-lambda dict (`list_reminders_fn`/`set_reminder_fn`/`delete_reminder_fn`/`cancel_all_reminders_fn`). One helper guarantees they stay in sync.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (new helper + four call-site swaps)

**Step 1: Read all four sites**

Compare the four blocks to confirm they truly are identical except for the closure-captured `caller`/`irc`/`msg`. Watch for the action-fire site (`1153-1163`) — its `set_reminder_fn` is wrapped in `_set_reminder_capped` for nested-call protection. That site needs the `set_reminder_fn` to remain the capped version; only the other three should pass through.

**Step 2: Decide on signature**

Two options: (a) one helper with an optional `set_reminder_override` parameter, (b) two helpers — `_reminder_fns_for_chat(caller, irc, msg)` and `_reminder_fns_for_action(caller, irc, msg, capped_set)`. Pick (a) for fewer surfaces:

```python
def _reminder_fns(
    self,
    *,
    caller: Identity,
    irc: callbacks.Irc,
    msg: ircmsgs.IrcMsg,
    set_reminder_fn: Callable[..., str] | None = None,
) -> dict[str, Callable[..., object]]:
    return {
        "list_reminders_fn": lambda: self._get_user_reminders(caller),
        "set_reminder_fn": set_reminder_fn or (
            lambda **kw: self._remind_set_for_assistant(caller, irc, msg, **kw)
        ),
        "delete_reminder_fn": lambda **kw: self._remind_delete_for_assistant(
            caller, irc, msg, **kw
        ),
        "cancel_all_reminders_fn": lambda: self._remind_clear_for_assistant(
            caller, irc, msg
        ),
    }
```

(Adjust signatures to whatever the existing call sites pass — read them carefully first.)

**Step 3: Replace the three chat sites**

At `plugin.py:2262-2269`, `2377-2384`, `2473-2480`: replace the inline lambda dict with `**self._reminder_fns(caller=caller, irc=irc, msg=msg)`.

**Step 4: Replace the action-fire site**

At `plugin.py:1153-1163`: pass `set_reminder_fn=_set_reminder_capped` so the cap wrapper is preserved.

**Step 5: Run tests + lint**

Run: `uv run pytest plugins/llm/ -q 2>&1 | tail -5 && uv run ruff check plugins/llm/`

**Step 6: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): extract _reminder_fns helper for tool-callback dict

Four identical four-lambda blocks (three chat sites plus the action-fire
closure) collapsed to one builder."
```

---

## Task A4: Revert `set_reminder` visibility to `{chat, remind_action}`

**Why:** `@draw` and `@code` are immediate-execution commands. A user who wants a deferred draw says `@remind in 5m draw a cat` — that routes through chat. The bullets at `assistant.py:40-47`, `73-77`, `89-94` teaching draw/code to use set_reminder are noise. Tool visibility (`assistant.py:595-606`) should not include `"draw"`/`"code"` for the four reminder tools.

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py` (system prompts + `_TOOL_SPEC_OVERRIDES`)
- Test: `plugins/llm/tests/test_assistant.py` (any test asserting reminder tools are visible in draw/code)

**Step 1: Audit the prompts**

Read `plugins/llm/src/llm/assistant.py:36-100`. Identify the bullets in CODE_SYSTEM_PROMPT and DRAW_SYSTEM_PROMPT that mention `set_reminder`. Confirm they're orthogonal to other guidance.

**Step 2: Remove the bullets**

Delete only the set_reminder-pointing rules from CODE and DRAW prompts. Leave CHAT and REMIND_ACTION untouched.

**Step 3: Update `_TOOL_SPEC_OVERRIDES`**

In `assistant.py` (around line 579-606), find the `set_reminder` / `list_reminders` / `delete_reminder` / `cancel_all_reminders` entries. Change `visible_in=frozenset({"chat", "code", "draw", "remind_action"})` → `visible_in=frozenset({"chat", "remind_action"})`.

**Step 4: Update tests**

Run: `uv run pytest plugins/llm/tests/test_assistant.py -q 2>&1 | tail -20`
Any test asserting set_reminder is visible in code/draw will fail. Decide per test: if the test was specifically validating the recent expansion (commit `31f897f`), delete it. If it was a broader sanity check, narrow it.

**Step 5: Add a smoke test for `@code`/`@draw` not calling `set_reminder`**

After narrowing visibility, the meta-loop will reject any `set_reminder` tool call from the code/draw profiles (rejected as unknown tool). Add a regression test that exercises a draw or code prompt mentioning "reminder" or "every minute" and asserts the model does NOT attempt a `set_reminder` call (or, if it does, that the rejection produces a sensible user-facing message rather than a meta-loop crash).

```python
async def test_draw_profile_does_not_expose_set_reminder(...):
    # Trigger draw profile; assert tool spec does not include set_reminder
    spec = build_tool_spec(profile="draw", capabilities=...)
    assert not any(tool["name"] == "set_reminder" for tool in spec)
```

**Step 6: Run full suite**

Run: `uv run pytest plugins/llm/ -q 2>&1 | tail -5`

**Step 7: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): scope set_reminder back to chat profile

@draw and @code are immediate-execution; deferred work routes through
@remind, which uses the chat profile. Removes redundant bullets and
narrows tool visibility."
```

---

## Task A5: Replace `_check_rate_limit_silent` clone with `silent` parameter

**Why:** `_check_rate_limit_silent` (`plugin.py:1796-1837`) is near-duplicate of `_check_rate_limit` (`plugin.py:1722-1794`).

> **Reviewer correction:** The original plan claimed "the only difference is whether `irc.reply` fires" — this is wrong. Three differences:
> 1. The non-silent path calls `irc.error(...)` on rate-limit hit (line 1772). Silent path doesn't.
> 2. The non-silent path calls `db.log_usage(..., status="rate_limited")` (lines 1773-1783). Silent path doesn't.
> 3. The non-silent path computes `now = time.time()` internally; silent takes `now` as a parameter, plus skips `irc, nick, channel, text` from its signature.
>
> A naive `silent: bool = False` would start writing `rate_limited` usage rows for the action-fire path that currently produces none — a real behavior change. Both `irc.error` AND `db.log_usage` must be guarded by `if not silent`. The `now` computation needs handling too: either accept `now: float | None = None` and compute when None, or have callers always pass it.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1722-1837` (merge the two methods; update the one caller of `_check_rate_limit_silent`)

**Step 1: Read both methods carefully**

Open `plugin.py:1722-1837`. Note the three divergences listed above.

**Step 2: Merge with the right guards**

Replace both methods with:

```python
def _check_rate_limit(
    self,
    irc: callbacks.Irc | None,
    command: str,
    account: str,
    nick: str,
    channel: str,
    text: str,
    *,
    tier: str,
    silent: bool = False,
    now: float | None = None,
) -> bool:
    """Check rate limit; optionally suppress user-facing error and usage row.

    When `silent=True`:
      - `irc.error(...)` is NOT called on overage.
      - `db.log_usage(..., status="rate_limited")` is NOT written.
      - `irc` may be None (action-fire path has no caller IRC connection).
      - `nick`/`channel`/`text` are still accepted but unused in the silent
        branch — kept in the signature for caller-site uniformity.

    `now` defaults to `time.time()` when not supplied.
    """
    if now is None:
        now = time.time()
    over_limit = self._is_rate_limited(command, account, now, tier=tier)
    self._record_rate_limit_hit(command, account, now)

    if not over_limit:
        return False

    enforce = self.registryValue("enforceRateLimits")
    max_count, window = self._get_tier_limits(command, tier)
    key = f"{command}:{account}"
    count = len(self._rate_buckets.get(key, ()))

    if enforce:
        self.log.info(
            "rate_limited command=%s account=%s tier=%s count=%d limit=%d window=%ss",
            command, account, tier, count, max_count, window,
        )
        if not silent:
            assert irc is not None  # non-silent path always has an irc connection
            irc.error(_("Rate limit exceeded for %s. Try again in %ds.") % (command, window))
            self.db.log_usage(
                nick, channel, command, "", 0, 0, 0.0,
                prompt=text, status="rate_limited",
            )
        return True

    self.log.info(
        "rate_limit_shadow command=%s account=%s tier=%s count=%d limit=%d window=%ss",
        command, account, tier, count, max_count, window,
    )
    return False
```

Delete `_check_rate_limit_silent` entirely.

**Step 3: Update the one caller**

Run: `grep -n '_check_rate_limit_silent' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/src/llm/plugin.py`
Expected: one hit in `_make_reminder_delivery_closure` (~line 1074). Replace with `_check_rate_limit(None, "ask", rl_account, "", "", "", tier=rl_tier, silent=True, now=now)` — pass empty strings for the unused-in-silent fields.

**Step 4: Run tests + lint + typecheck**

Run: `uv run pytest plugins/llm/ -q && uv run ruff check plugins/llm/ && uv run ty check plugins/llm/`
Expected: pass. If `ty` complains about `irc: callbacks.Irc | None`, double-check the assert and the silent-path None usage.

**Step 5: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): merge _check_rate_limit_silent into a silent= param

Both irc.error AND db.log_usage(status='rate_limited') are now guarded
behind silent — the action-fire path keeps its current 'no usage row,
no IRC error' behavior. now= accepts None and computes via time.time()
when omitted, matching the original non-silent signature."
```

---

## Task A6: Drop `chain_id` column only (data-shape refactor — TTL stays for now)

**Why:** `chain_id` is never used as a lookup key — only stored on every row. Dropping it is a pure data-shape refactor: no behavior changes, no error messages disappear.

> **Reviewer correction:** The original A6 also dropped `chain_started_at` and the 30-day TTL check that uses it. That TTL emits a user-visible error message ("Recurring reminder reached its 30-day TTL") — removing it is a **behavior change**, which violates PR A's "behavior preserved" hard constraint. TTL removal moved to PR B Task B0.6 where it can sit alongside the structural changes that justify dropping it.

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:18` (`SCHEMA_VERSION = 11`)
- Modify: `plugins/llm/src/llm/persistence.py:24-38` (`ReminderRow`: drop `chain_id`)
- Modify: `plugins/llm/src/llm/persistence.py` (migration block, `save_reminder`, `load_pending_reminders`)
- Modify: `plugins/llm/src/llm/plugin.py` (drop `chain_id` carry-through; KEEP `chain_started_at` and the TTL check intact)
- Modify: `plugins/llm/tests/conftest.py` (drop `chain_id` from `make_reminder_row` helper)
- Modify: `plugins/llm/tests/test_persistence.py`, `test_reminders.py` (tests asserting `chain_id`)

**Step 1: Find the v10 migration that added these columns**

Open `persistence.py` and find the `if user_version < 10` block. Confirm it added `chain_id`, `chain_position`, `chain_started_at`.

**Step 2: Bump `SCHEMA_VERSION` to 11 and add a v11 migration**

```python
if user_version < 11:
    cursor.executescript("""
        ALTER TABLE reminders DROP COLUMN chain_id;
    """)
```

(SQLite has `DROP COLUMN` since 3.35; Python 3.14's bundled SQLite is fine.)

**Step 3: Drop `chain_id` from `ReminderRow`**

At `persistence.py:24-38`: delete the `chain_id: str` field. Keep `chain_position` and `chain_started_at`.

**Step 4: Update `save_reminder` / `load_pending_reminders`**

Remove `chain_id` from INSERT column lists, parameter signatures, SELECT result-row construction. Keep the others.

**Step 5: Update `plugin.py` carry-through**

Run: `grep -n 'chain_id' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/src/llm/plugin.py`
Each hit should be pure data plumbing — remove it. Do NOT touch `chain_started_at` or the TTL check.

**Step 6: Update tests + conftest helper**

Drop `chain_id` from `make_reminder_row` (Task A1b conftest helper). Run: `uv run pytest plugins/llm/ -q 2>&1 | tail -20` and fix asserts.

**Step 7: Lint + typecheck**

Run: `uv run ruff check plugins/llm/ && uv run ty check plugins/llm/`

**Step 8: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): drop unused chain_id column from reminders

chain_id was stored on every row but never used as a lookup key.
chain_position and chain_started_at retained — TTL behavior unchanged.
Schema bumped to v11."
```

---

## Task A7: Identity migration — finish what `0d6b3bf` started

**Why:** `_get_identity` (`plugin.py:1577-1583`) returns `Identity.key`, but callers still bind it to `nick` and pass it as `nick: str` into helpers (call sites near `761`, `2522`, `2546`, `2650`, `2747`). Same nick-vs-account confusion the dataclass was added to prevent.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (audit and convert call sites)

**Step 1: Map the affected sites**

Run: `grep -n '_get_identity\|_resolve_identity' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/src/llm/plugin.py`
For each `_get_identity` caller, check what type the result is bound to and how it's used downstream. The downstream helpers should take `Identity` directly, not a string key.

**Step 2: Convert in small batches**

Per call site: replace `nick = self._get_identity(...)` with `caller = self._resolve_identity(...)`. Update the downstream helper signature from `nick: str` to `caller: Identity`. Repeat the chain until you hit a true string boundary (e.g. an SQL parameter — at which point pass `caller.key`).

**Step 3: Decide on `_get_identity`'s fate**

If after the conversion no callers remain, delete it. If a few legitimate string-key consumers remain (e.g. log lines), keep it as the documented "I want the storage key as a string" affordance.

**Step 4: Run tests + lint + ty**

Run: `uv run pytest plugins/llm/ -q && uv run ruff check plugins/llm/ && uv run ty check plugins/llm/`
Expected: pass. `ty` will catch any signature mismatches.

**Step 5: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): finish Identity migration in remaining call sites

_get_identity callers were binding the storage key to a 'nick' variable
and passing it as nick: str downstream — the same shape the Identity
dataclass was added to prevent. Convert to _resolve_identity and pass
Identity end-to-end."
```

---

## Task A8: Delete commit-narrating comments

**Why:** Several blocks read like PR descriptions and will rot.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (lines `1097-1101`, `1163-1165`, `2922-2927`)

> **Reviewer correction:** Original plan cited `persistence.py:278-281`. Wrong — those lines are an unrelated `memory_cleanup_state` block. The chain-comment block was at `persistence.py:313-317` (inside the v10 migration), but Task A6 already touches that area for the v11 column drop. Drop `persistence.py` from this task — A6 leaves nothing left to clean here.

**Step 1: Read each block**

Confirm the comment is narrating the change ("see the action-reminders plan's Architecture section", "Note: action fires use a synthetic_msg with no msgid", cross-references to plan docs).

**Step 2: Delete or convert**

- Cap-rationale comments (e.g. "Hard cap on how many fires…"): convert to a one-line docstring on the constant. Keep the `why`, drop the `what`.
- Plan-doc cross-references: delete entirely.
- "Note: …" workflow narration: delete.

**Step 3: Run tests + lint**

Run: `uv run pytest plugins/llm/ -q && uv run ruff check plugins/llm/`

**Step 4: Commit**

```bash
git add plugins/llm/
git commit -m "chore(llm): drop commit-narrating comments from reminder code"
```

---

## PR A finalize

**Step 1: Squash review**

Run: `git log --oneline origin/main..HEAD`
Confirm 9 clean commits (A0 has no commit; A1a, A1b, A2, A3, A4, A5, A6, A7, A8).

**Step 2: Push and open PR (only if user requests)**

Stop here. Do not push without the user's say-so.

---

# PR B — Structural redesign

> Only start PR B after PR A has been reviewed and merged.

## Task B0: Snapshot baseline

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run pytest plugins/llm/ -q 2>&1 | tail -5`
Record the count.

---

## Task B0.4: Add `python-dateutil` as a direct runtime dep on `plugins/llm`

**Why:** Codex caught that `python-dateutil` is currently a *docs-toolchain* transitive (via `ghp-import`), not a runtime dep of `plugins/llm`. B4's mechanical RRULE reschedule imports `dateutil.rrule.rrulestr`. Without an explicit declaration, a future `uv lock` or wheel build for the LLM plugin could drop it and B4's runtime would `ImportError` in production.

**Files:**
- Modify: `plugins/llm/pyproject.toml` (add to `dependencies`)
- Modify: `uv.lock` (regenerated)

**Step 1: Confirm dateutil is not in runtime deps**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv tree --package llm 2>&1 | grep -i dateutil`
Expected: no hit (matches Codex's finding).

**Step 2: Add to `plugins/llm/pyproject.toml`**

Add `"python-dateutil>=2.9"` to the `[project] dependencies` list. Pin a version range matching what's already resolved transitively elsewhere if you want lockstep.

**Step 3: Regenerate the lockfile**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv lock`
Expected: `uv.lock` updated with `python-dateutil` listed under the `llm` package's deps.

**Step 4: Smoke-test the import**

Run: `cd /Users/rdrake/workspace/afternet/vibebot-v8 && uv run python -c "from dateutil.rrule import rrulestr; print('ok')"`
Expected: `ok`.

**Step 5: Commit**

```bash
git add plugins/llm/pyproject.toml uv.lock
git commit -m "deps(llm): add python-dateutil as direct runtime dep

Required by B4 mechanical RRULE reschedule. Was previously only a
docs-toolchain transitive (via ghp-import); declaring it directly
prevents a future lock change from dropping it from the LLM runtime."
```

---

## Task B0.5: In-flight reminder migration strategy

**Why:** Production has live reminders right now with parentheticals embedded in `action_prompt` (`(recurring: every 5 minutes)`, `(watch -- only respond on positive result)`). After B1 adds `recurrence_seconds`/`recurrence_rrule`/`watch_mode` columns with defaults (NULL/NULL/0), those existing rows have no structured recurrence. Without the strict routing gate from B3.5, those rows would stop rescheduling once the legacy LLM-tool path is removed. This is a real production regression. Decide and document the strategy before writing migration code.

**Decision required:** pick (a) or (b) below. Discuss with the user before implementing.

**Option (a) — Backfill on migration:**
The v12 migration scans existing `action_prompt` values, regex-extracts `(recurring: ...)` and `(watch -- only respond on positive result)` text, populates the new columns from those captures, and strips the parentheticals from `action_prompt`. Trade-off: regex parsing of free-form LLM output, possibility of malformed cases that need human review. Migration is one-shot at deploy time.

**Option (b) — Graceful degradation:**
Existing pending recurring chains run to natural completion using the *old* LLM-driven reschedule path (we keep `_set_reminder_capped` and the `REMIND_ACTION_SYSTEM_PROMPT` clause for one release cycle), then the legacy path is removed in a follow-up. Trade-off: code carries both paths during the transition; cleaner migration. New reminders use the structured path immediately.

**Recommendation:** Option (b). Backfill regex on free-form prompt text is fragile; running existing chains to completion via the existing path is risk-free. Mark `_set_reminder_capped` with a `# TODO(remove after one release): legacy reschedule path` comment plus a `_LEGACY_RECURRENCE_PARENTHETICAL_RE` constant for future archaeology, and proceed.

**Step 1: Get user sign-off on (a) vs (b).** Halt this task until a decision lands.

**Step 2: Document the chosen strategy** as a paragraph at the top of `B1`'s commit message and as a comment near the v12 migration in `persistence.py`. Reference this task by file/line so a future maintainer can find the rationale.

**Step 3: No code changes here** — this task is a decision artifact. Proceed to B0.6.

---

## Task B0.6: Drop the 30-day chain TTL (moved from PR A's old A6)

**Why:** Originally part of A6, but the TTL emits a user-visible error and removing it is a behavior change — belongs in PR B alongside the structural changes that justify it. After mechanical reschedule (B4) and chain-position cap (50 fires), the 30-day TTL is redundant: 50 × any cadence already bounds runaway, and a 50-fire chain over 30 days is a once-per-14h reminder which is exactly the "long-running watch" the user might actually want.

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:18` (`SCHEMA_VERSION = 12`, plus the v12 migration may bundle this column drop with B1's adds — see Step 2)
- Modify: `plugins/llm/src/llm/persistence.py:24-38` (drop `chain_started_at` from `ReminderRow`)
- Modify: `plugins/llm/src/llm/plugin.py` (delete the TTL check + `_REMINDER_CHAIN_TTL_SECONDS` constant + the user-visible error message)
- Modify: `plugins/llm/tests/conftest.py` (drop `chain_started_at` from `make_reminder_row`)
- Modify: `plugins/llm/tests/test_reminders.py` (delete the TTL-error test if any)

**Step 1: Find the TTL check**

Run: `grep -n 'chain_started_at\|_REMINDER_CHAIN_TTL_SECONDS\|30-day TTL\|30.day' /Users/rdrake/workspace/afternet/vibebot-v8/plugins/llm/src/llm/plugin.py`
Read each hit. Confirm: one constant definition, one check inside `_schedule_reminder`, one user-visible error message string, possibly one test.

**Step 2: Bundle v12 migration**

Either:
- One v12 migration that adds `recurrence_seconds`/`recurrence_rrule`/`watch_mode` (B1) AND drops `chain_started_at` (this task), OR
- Separate v12 (drop `chain_started_at`) and v13 (add the new columns).

Pick the bundled v12 — fewer migration boundaries to test.

**Step 3: Delete the check and constant**

Remove the TTL `if` block, the constant, and the user-visible "Recurring reminder reached its 30-day TTL" message.

**Step 4: Tests**

Find and delete any test asserting on the TTL message. Update fixtures.

**Step 5: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm)!: drop 30-day chain TTL in favor of position cap

The 50-fire chain_position cap is sufficient runaway protection. A
50-fire chain spanning 30 days is a 14-hour cadence — exactly the
long-running watch a user might want, not a runaway. chain_started_at
column dropped; constant and user-visible error message removed.

BREAKING: pending reminders that would have hit TTL now run to
chain_position=50 instead. No production rows are known to be near TTL."
```

(Use `!` and the BREAKING footer because this is a real behavior change. If the user prefers no breaking-change marker for an internal feature, drop them.)

---

## Task B1: Add `recurrence_seconds`, `recurrence_rrule`, `watch_mode` columns

**Why:** Recurrence and watch are currently encoded as parenthetical strings in `action_prompt` (`(recurring: every 5 minutes)`, `(watch -- only respond on positive result)`). The parser embeds them, the model re-parses them at fire time. Promoting to structured columns lets the deliverer reschedule mechanically (no LLM round-trip) and removes the model-compliance contract.

> **Reviewer revision:** Single freeform `recurrence_hint` was insufficient — see B4 Step 1. Two-column shape: `recurrence_seconds INTEGER` for numeric cadences, `recurrence_rrule TEXT` for calendar (RFC 5545). Exactly one is non-null for a recurring reminder; both null for one-shot.

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (v12 migration — bundles B0.6's `chain_started_at` drop, see Step 1)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Bump `SCHEMA_VERSION` to 12 and add v12 migration**

The v12 migration bundles the column add/drop pair from B0.6 and B1:

```python
if user_version < 12:
    cursor.executescript("""
        ALTER TABLE reminders DROP COLUMN chain_started_at;
        ALTER TABLE reminders ADD COLUMN recurrence_seconds INTEGER;
        ALTER TABLE reminders ADD COLUMN recurrence_rrule TEXT;
        ALTER TABLE reminders ADD COLUMN watch_mode INTEGER NOT NULL DEFAULT 0;
    """)
```

**Step 2: Update `ReminderRow`**

Drop `chain_started_at`. Add:
- `recurrence_seconds: int | None`
- `recurrence_rrule: str | None`
- `watch_mode: bool`

**Step 3: Update `save_reminder` and `load_pending_reminders`**

Pipe the three new fields through. `watch_mode` stored 0/1, read back as `bool`. Add a class-level constraint check (or a runtime assert in `save_reminder`): if `recurrence_seconds is not None and recurrence_rrule is not None`, raise — they're mutually exclusive.

**Step 4: Write a failing test**

```python
def test_reminder_persists_structured_recurrence_fields(tmp_path):
    db = Database(tmp_path / "x.db")
    db.save_reminder(
        event_name="evt", nick="n", channel="#c", message="",
        action_prompt="check the build", account="a",
        fire_at=time.time() + 60, chain_position=1,
        recurrence_seconds=300, recurrence_rrule=None, watch_mode=True,
    )
    rows = db.load_pending_reminders()
    assert rows[0].recurrence_seconds == 300
    assert rows[0].recurrence_rrule is None
    assert rows[0].watch_mode is True


def test_reminder_rejects_both_recurrence_kinds(tmp_path):
    db = Database(tmp_path / "x.db")
    with pytest.raises(ValueError):
        db.save_reminder(
            event_name="evt", nick="n", channel="#c", message="",
            action_prompt="x", account="a", fire_at=time.time() + 60,
            chain_position=1,
            recurrence_seconds=300,
            recurrence_rrule="FREQ=WEEKLY",
            watch_mode=False,
        )
```

Run: expect FAIL on signature.

**Step 5: Implement**

Add the kwargs to `save_reminder`. Run the tests, expect PASS.

**Step 6: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm): structured recurrence/watch columns on reminders

Schema v12. Drops chain_started_at, adds recurrence_seconds,
recurrence_rrule, watch_mode. Migration strategy for in-flight rows
follows B0.5 (option b): legacy LLM-driven reschedule remains until
all pending parenthetical-encoded reminders complete."
```

---

## Task B2: Parser populates `recurrence_seconds`, `recurrence_rrule`, and `watch_mode` as structured fields

**Why:** Stop embedding `(recurring: ...)` and `(watch — ...)` in `action_prompt`. Pull them out at parse time.

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (`ReminderParseResult`, parser prompt, `parse_reminder`)
- Test: `plugins/llm/tests/test_service.py`, `test_reminders.py`

**Step 1: Extend `ReminderParseResult`**

Add three fields: `recurrence_seconds: int | None = None`, `recurrence_rrule: str | None = None`, `watch_mode: bool = False`. Add a `__post_init__` (or pre-validate) that rejects both recurrence fields being set.

**Step 2: Update the parser system prompt**

Replace the embed-parenthetical instructions with structured-output rules. Schema returned by the model:

```json
{
  "action": "set",
  "seconds": 300,
  "message": "...",
  "action_prompt": "check the build",
  "recurrence_seconds": 300,
  "recurrence_rrule": null,
  "watch_mode": true,
  "confirmation": "..."
}
```

Rules in the system prompt:
- For "every N {seconds|minutes|hours|days}": populate `recurrence_seconds` only (compute the integer).
- For calendar cadences ("every Monday at 9am", "first of the month", "every weekday at 5pm"): populate `recurrence_rrule` only with a valid RFC 5545 RRULE string.
- For one-shot reminders: both null.
- For "let me know if/when X" or "tell me only if X": set `watch_mode=true`.
- Both recurrence fields must NOT be set simultaneously.

Provide 3-4 few-shot examples covering: numeric, weekly calendar, monthly calendar, one-shot, watch.

**Step 3: Write failing tests**

```python
async def test_parser_numeric_recurrence(...):
    result = await assistant.parse_reminder("every 5 minutes check the build")
    assert result.recurrence_seconds == 300
    assert result.recurrence_rrule is None
    assert "(recurring" not in result.action_prompt
    assert "(watch" not in result.action_prompt


async def test_parser_calendar_recurrence(...):
    result = await assistant.parse_reminder("every Monday at 9am stand-up reminder")
    assert result.recurrence_seconds is None
    assert result.recurrence_rrule is not None
    assert "FREQ=WEEKLY" in result.recurrence_rrule
    assert "BYDAY=MO" in result.recurrence_rrule


async def test_parser_watch_mode(...):
    result = await assistant.parse_reminder("let me know if the build passes")
    assert result.watch_mode is True
    assert "(watch" not in result.action_prompt
```

**Step 4: Implement**

Update `parse_reminder` to populate the new fields from the structured response. Strip any parentheticals that leak through (defense-in-depth).

**Step 5: Run the test**

Expect PASS. Run the broader parser test suite — backward-compat tests may need updating.

**Step 6: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm): parser returns recurrence_seconds/rrule and watch_mode as structured fields

Drops (recurring: ...) and (watch -- ...) parenthetical embedding into
action_prompt; parser returns them as first-class ReminderParseResult
fields. action_prompt is now just the action."
```

---

## Task B3: Wire structured fields through schedule and fire paths

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_remind_set`, `_schedule_reminder`, `_make_reminder_delivery_closure`)
- Test: `plugins/llm/tests/test_reminders.py`

**Step 1: Pass `recurrence_seconds`/`recurrence_rrule`/`watch_mode` from parse → save**

In `_remind_set` / `_schedule_reminder` (around `plugin.py:3045-3082`): the parse result now carries the three structured fields. Pass them into `save_reminder` and store them on the in-memory `ReminderRow`. Add an early-return check: if a parsed reminder has `recurrence_rrule` and the rrule fails `dateutil.rrule.rrulestr(...)` validation, fail the schedule with a user-visible error rather than scheduling a broken row.

**Step 2: Update `_make_reminder_delivery_closure`**

The closure needs `watch_mode` to decide whether `[silent]` from the fire-time model means "suppress IRC send" (watch mode, no news) vs "an unexpected empty response" (chat mode bug). It also needs `recurrence_seconds` and `recurrence_rrule` for the mechanical reschedule in B4.

**Step 3: Run tests**

Run the reminder fire-path tests. Update fixtures to pass the new fields.

**Step 4: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm): pipe recurrence_seconds/rrule and watch_mode through schedule + fire paths"
```

---

## Task B3.5: Strict transition gate — structured rows go mechanical, legacy rows go LLM

**Why:** Codex caught that B0.5 option (b) plus B4 mechanical reschedule, naively wired, can **double-reschedule a single fire**: the deliverer fires the action, the action LLM still has `set_reminder` exposed and self-reschedules per the legacy prompt, AND the deliverer mechanically reschedules from the structured fields. A row could end up scheduling itself twice.

The fix is a hard routing gate at the deliverer:

- A row with `recurrence_seconds is not None` OR `recurrence_rrule is not None` (structured) → mechanical reschedule path; **the action profile for this fire excludes `set_reminder` from its tool surface** so the model can't double-schedule.
- A row with both structured fields null but action_prompt containing the legacy `(recurring: ...)` parenthetical → legacy LLM-tool reschedule path; mechanical reschedule SKIPPED for this row.
- A row with both structured fields null AND no parenthetical → one-shot, no reschedule either way.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_make_reminder_delivery_closure` — branch on row shape)
- Modify: `plugins/llm/src/llm/assistant.py` (per-fire tool-spec filter that strips `set_reminder` for structured-row fires)
- Test: `plugins/llm/tests/test_reminders.py` (mutual-exclusion proof)

**Step 1: Add `is_structured_recurring(row) -> bool` and `is_legacy_recurring(row) -> bool` helpers**

```python
_LEGACY_RECURRENCE_RE = re.compile(r"\(recurring:[^)]+\)", re.IGNORECASE)


def _is_structured_recurring(row: ReminderRow) -> bool:
    return row.recurrence_seconds is not None or row.recurrence_rrule is not None


def _is_legacy_recurring(row: ReminderRow) -> bool:
    if _is_structured_recurring(row):
        return False
    return bool(_LEGACY_RECURRENCE_RE.search(row.action_prompt or ""))
```

**Step 2: Branch the deliverer**

In `_make_reminder_delivery_closure`, after a successful action fire:

```python
if _is_structured_recurring(reminder):
    # Mechanical path. set_reminder was NOT exposed to this fire (Step 3).
    self._mechanical_reschedule(reminder, now)
elif _is_legacy_recurring(reminder):
    # Legacy path: action LLM has already (or will) self-reschedule via
    # set_reminder during its turn. Do NOT mechanically reschedule.
    pass
else:
    # One-shot. Done.
    pass
```

**Step 3: Per-fire tool-surface filter for structured rows**

When dispatching the action fire, pass the row to the tool-spec builder. If `_is_structured_recurring(reminder)`: filter `set_reminder` out of the exposed tools for that turn. The legacy-prompt clause "you MAY call set_reminder ONCE" only applies to legacy rows now. Structured rows get a different fire-time prompt (or the same prompt with the set_reminder paragraph dynamically suppressed) — pick whichever is less invasive in the prompt.

**Step 4: Tests proving mutual exclusion**

```python
async def test_structured_row_does_not_expose_set_reminder_at_fire(...):
    row = make_reminder_row(recurrence_seconds=300, action_prompt="check the build")
    spec = await fire_action_for(row)  # capture tool spec used
    assert not any(t["name"] == "set_reminder" for t in spec)


async def test_structured_row_reschedules_mechanically_only(...):
    row = make_reminder_row(recurrence_seconds=300, action_prompt="x")
    fired = await deliver_once(row)
    # Exactly one new reminder scheduled, by the deliverer not the LLM
    assert len(fired.scheduled_after_fire) == 1
    assert fired.set_reminder_tool_calls == 0


async def test_legacy_row_uses_llm_tool_only_no_mechanical(...):
    row = make_reminder_row(action_prompt="x (recurring: every 5 minutes)")
    fired = await deliver_once(row)
    # Mechanical path NOT taken; LLM tool path may or may not have run
    assert fired.mechanical_reschedule_called is False


async def test_one_shot_row_neither_path(...):
    row = make_reminder_row(action_prompt="check this once")
    fired = await deliver_once(row)
    assert fired.scheduled_after_fire == []


async def test_clear_wins_over_mid_fire_mechanical_reschedule(...):
    # Schedule a recurring structured reminder; trigger a fire whose
    # closure has begun running. While mid-fire, cancel_all_reminders
    # the same user. The mechanical reschedule MUST detect the cancel
    # and skip — clear wins.
    row = make_reminder_row(recurrence_seconds=60, action_prompt="x")
    plugin._reminders["evt"] = row
    fire_started = asyncio.Event()
    fire_can_finish = asyncio.Event()
    # ... patch the action call to block on fire_can_finish ...
    fire_task = asyncio.create_task(deliver_once(row, hooks=(fire_started, fire_can_finish)))
    await fire_started.wait()
    await plugin._remind_clear_for_assistant(caller, irc=mock_irc, msg=mock_msg)
    fire_can_finish.set()
    await fire_task
    assert "evt" not in plugin._reminders  # original gone
    # AND no new event scheduled by the mechanical path
    assert not any(plugin._reminders.values())
```

**Step 5: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm): strict routing gate between mechanical and legacy reschedule

Structured rows (recurrence_seconds or recurrence_rrule) take the
mechanical path AND have set_reminder filtered out of their fire-time
tool surface. Legacy parenthetical-encoded rows take the LLM-tool path
and skip mechanical reschedule. No row ever takes both — closes the
double-reschedule hazard during the B0.5 option (b) transition."
```

---

## Task B4: Mechanical reschedule for recurring reminders (drop the LLM tool call)

**Why:** Today recurring reminders reschedule by having the action LLM call `set_reminder` itself (capped to 1 nested call). That's a model-compliance contract that occasionally fails — and an avoidable LLM round-trip. With `recurrence_seconds` and `recurrence_rrule` as columns, the deliverer can compute the next fire time and re-enqueue without asking the model.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_make_reminder_delivery_closure`, remove `_set_reminder_capped`)
- Modify: `plugins/llm/src/llm/assistant.py` (drop the "you MAY call set_reminder ONCE" paragraph in `REMIND_ACTION_SYSTEM_PROMPT`)
- New helper: a small `_next_fire_at(hint: str, now: float) -> float | None` parser

**Step 1: Resolve the recurrence-format problem (revised after code review)**

> **Reviewer caught:** A single `recurrence_seconds: int | None` cannot express calendar-anchored cadences the parser actually produces — `every Monday at 9am`, `first of the month`, `every Friday at 5pm`. Picking option (b) naively would silently break those.

The right shape is a **two-column representation** that handles both cases:

```sql
ALTER TABLE reminders ADD COLUMN recurrence_seconds INTEGER;  -- numeric cadences only
ALTER TABLE reminders ADD COLUMN recurrence_rrule TEXT;       -- RFC 5545 RRULE for calendar
```

(B1 was already revised to use this two-column shape — no separate edit needed.)

**Mechanical reschedule logic at fire time:**

1. If `recurrence_seconds is not None`: `next_fire = now + recurrence_seconds` — no model call.
2. Else if `recurrence_rrule is not None`: parse with `dateutil.rrule.rrulestr` and compute `next_fire = rrule.after(datetime.now())` — no model call.
3. Else: not recurring; fire-and-forget.

`dateutil` is already in the project's transitive deps (uv.lock has it via `pendulum` or similar — verify with `uv tree | grep dateutil`); if not, add it as a direct dep first.

**Parser update for B2:**
The parser populates one of `recurrence_seconds` OR `recurrence_rrule`, not both:
- "every 5 minutes" → `recurrence_seconds=300`
- "every hour" → `recurrence_seconds=3600`
- "every Monday at 9am" → `recurrence_rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0"`
- "first of the month" → `recurrence_rrule="FREQ=MONTHLY;BYMONTHDAY=1"`

Provide the model both schemas in the system prompt and ask it to pick. Validate the rrule string at parse time (reject invalid ones) so we never schedule something the deliverer can't reschedule.

**Fallback:** if both parser attempts produce malformed output and the user's input is clearly recurring, the parser can return an error rather than silently dropping recurrence. Failing loud at parse time beats failing silent at fire time.

B1 and B2 are already aligned with this two-column shape; no edits needed before executing B4.

**Step 2: Implement mechanical reschedule (timezone-aware UTC)**

> **Codex correction:** Naive `datetime.fromtimestamp(now)` ties RRULE behavior to host local timezone and causes wrong fires across DST. Use timezone-aware UTC throughout.

In `_make_reminder_delivery_closure`, after the action fire completes and `chain_position < _REMINDER_MAX_CHAIN_POSITION`:

```python
from datetime import datetime, timezone
from dateutil.rrule import rrulestr

next_fire: float | None = None
if reminder.recurrence_seconds is not None:
    next_fire = now + reminder.recurrence_seconds
elif reminder.recurrence_rrule is not None:
    now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
    rule = rrulestr(reminder.recurrence_rrule, dtstart=now_utc)
    next_dt = rule.after(now_utc)  # tz-aware UTC; safe across DST
    next_fire = next_dt.timestamp() if next_dt is not None else None

if next_fire is not None:
    self._schedule_reminder_row(
        # construct a new ReminderRow with chain_position += 1 and fire_at=next_fire
        ...,
    )
```

`chain_position` increment must be threaded through the new path — see B0.6's note that the cap is the only remaining runaway guard. Test that the position increments correctly across mechanical reschedules.

**Step 2a: DST-boundary regression tests**

Add at least two tests covering DST behavior:

```python
def test_rrule_reschedule_across_spring_forward(monkeypatch):
    # 2027-03-14 02:30 ET DST jump: 02:00 -> 03:00. RRULE "every day at 02:30"
    # must produce a sane next-fire (not infinite loop, not skipped day).
    rule = "FREQ=DAILY;BYHOUR=2;BYMINUTE=30"
    # Anchor "now" just before the spring-forward
    now = datetime(2027, 3, 14, 6, 0, tzinfo=timezone.utc).timestamp()  # 02:00 ET
    next_fire = compute_next_fire(rule, now)
    assert next_fire > now
    # And the *following* invocation produces a strictly-greater timestamp
    next_next = compute_next_fire(rule, next_fire)
    assert next_next > next_fire


def test_rrule_reschedule_across_fall_back(...):
    # 2027-11-07 ET DST end: 02:00 -> 01:00. Verify no duplicate fire.
    ...
```

UTC-anchored RRULE doesn't observe local DST, so these tests primarily guard against accidental local-time regressions (the "naive datetime" bug Codex flagged). If the user later wants local-time anchored RRULE (e.g. "every weekday at 9am Toronto time"), that's a separate feature requiring `dtstart` in the user's tz — out of scope for this plan; document as a follow-up.

**Step 2b: Observability hook**

After computing `next_fire`, log a structured line:

```python
self.log.info(
    "reminder_reschedule path=mechanical kind=%s position=%d/%d next_fire_at=%s rrule=%s",
    "seconds" if reminder.recurrence_seconds is not None else "rrule",
    new_position, self._REMINDER_MAX_CHAIN_POSITION,
    datetime.fromtimestamp(next_fire, tz=timezone.utc).isoformat() if next_fire else "none",
    reminder.recurrence_rrule or "",
)
```

If RRULE parsing or `.after()` returns None (rule has expired or never fires again), log at WARNING level with the row's id/event_name and skip rescheduling. This is the "reschedule failure reason" Codex's significant-gaps section asked for.

**Step 3: Remove `_set_reminder_capped` (deferred per B0.5 option b)**

If B0.5 chose option (b) — graceful degradation — leave `_set_reminder_capped` in place for now and add a `# TODO(remove after one release): legacy reschedule path for pre-v12 reminders` comment. The deliverer's mechanical path handles new (post-v12) rows; the legacy path handles pre-v12 rows that still have parenthetical-encoded recurrence.

If B0.5 chose option (a) — backfill — delete `_set_reminder_capped` and the corresponding paragraph in `REMIND_ACTION_SYSTEM_PROMPT` now.

**Step 4: Update `REMIND_ACTION_SYSTEM_PROMPT`**

Remove the paragraph instructing the model to call set_reminder when it sees `(recurring: ...)`. Recurrence is now mechanical.

**Step 5: Tests**

Update `test_reminders.py`:
- Recurring reminder tests should assert that the second fire was scheduled by the deliverer, not by an LLM tool call.
- Remove tests that exercised the nested-set cap path; add a test for `chain_position` enforcement at the deliverer level.

**Step 6: Commit**

```bash
git add plugins/llm/
git commit -m "feat(llm): mechanically reschedule recurring reminders at fire time

Drops the action-LLM 'you MAY call set_reminder once' contract in favor
of a deliverer-side reschedule keyed off recurrence_seconds/rrule. Removes
_set_reminder_capped and the nested-set guard."
```

---

## Task B5: Move post-tool silence from `[silent]` contract into reply suppression

**Why:** `[silent]` is doing two unrelated jobs. Job 1 (post-reaction acknowledgment): the chat profile told the model to reply `[silent]` after a successful set/delete/cancel because the user already saw the reaction. Job 2 (watch-mode no-news): the reminder delivery closure uses `[silent]` to mean "negative result, don't report."

Job 1 is fragile (model has to comply) and unnecessary. Drop the chat-profile `[silent]` rule and suppress in code.

> **Codex correction (round 2):** Original draft said suppress when "text is empty/whitespace/an acknowledgment phrase." Phrase-matching is fragile and risks suppressing legitimate short answers ("Done!", "Okay" as standalone replies to other questions). Use a strict signal instead: the assistant loop emits structured metadata about the last successful tool call, and suppression fires ONLY when (a) the last successful tool was a reminder mutation AND (b) the post-tool model text is strictly empty/whitespace.
>
> Also: the existing watch-mode `[silent]` check is in `_make_reminder_delivery_closure` (around `plugin.py:1171`), NOT in `_ask_impl`. Don't conflate the two.

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py` (drop `[silent]` rule from CHAT_SYSTEM_PROMPT; add `last_successful_tool` and `final_text_after_tools` to the assistant-loop result type)
- Modify: `plugins/llm/src/llm/service.py` (assistant loop populates the new metadata)
- Modify: `plugins/llm/src/llm/plugin.py` (`_ask_impl` checks the new signal; watch-mode `[silent]` at ~line 1171 is unchanged)

**Step 1: Add structured signal to assistant-loop result**

In `service.py` (the assistant loop), thread two new fields onto the result type (or add to whatever result dataclass already exists):

```python
@dataclass
class AssistantLoopResult:
    text: str
    usage: Usage
    last_successful_tool: str | None  # name of the last tool that returned without error
    final_text_after_tools: str       # text emitted *after* the last tool call (may be empty)
    # ... existing fields
```

Populate from the loop bookkeeping. `last_successful_tool` is the name of the most recent successful tool invocation; `final_text_after_tools` is the assistant message produced after that tool call (empty string if the model called the tool then said nothing else).

**Step 2: Drop the chat-prompt `[silent]` rule**

In `assistant.py:23-30` (the chat system prompt's post-tool `[silent]` instructions): delete those lines. CHAT_SYSTEM_PROMPT no longer mentions `[silent]`.

**Step 3: Add reply suppression to `_ask_impl`**

After the assistant loop returns:

```python
REMINDER_MUTATION_TOOLS = frozenset({
    "set_reminder", "delete_reminder", "cancel_all_reminders",
})

if (
    result.last_successful_tool in REMINDER_MUTATION_TOOLS
    and not result.final_text_after_tools.strip()
):
    # User already saw the reaction; the model said nothing else.
    # Skip irc.reply to avoid a duplicate ack.
    self.log.info("suppressing empty post-reminder-mutation reply")
    # ... still record usage, store context as before
    return
```

No phrase matching. Strict empty/whitespace check only. If the model produced any non-whitespace text after the tool call, send it.

**Step 4: Watch-mode `[silent]` is untouched**

The existing watch-mode `[silent]` check in `_make_reminder_delivery_closure` (around `plugin.py:1171`) stays exactly as it is. That branch handles reminder *fires*, not chat replies — different code path, different sentinel job. Confirm by `grep -n '\[silent\]' plugins/llm/src/llm/plugin.py` before and after this task; the only removal should be tied to chat-prompt suppression.

**Step 5: Tests**

```python
async def test_chat_set_reminder_suppresses_empty_post_tool_reply(...):
    result = await ask("vibebot, remind me to buy milk in an hour")
    irc.reply.assert_not_called()
    irc.tagmsg.assert_called()  # reaction went out


async def test_chat_set_reminder_does_not_suppress_legit_reply(...):
    # Model returns text *after* the tool call — must NOT be suppressed
    fake_loop_returns(
        last_successful_tool="set_reminder",
        final_text_after_tools="Got it. Want me to also remind you about the receipt?",
    )
    result = await ask(...)
    irc.reply.assert_called_once()


async def test_non_reminder_tool_never_suppressed(...):
    # Even with empty post-tool text, non-reminder tools should not trigger
    # suppression (e.g. generate_image returns a URL the user wants to see)
    fake_loop_returns(last_successful_tool="generate_image", final_text_after_tools="")
    result = await ask(...)
    irc.reply.assert_called_once()
```

**Step 6: Commit**

```bash
git add plugins/llm/
git commit -m "refactor(llm): suppress post-reminder-mutation reply via structured signal

Drops the chat-prompt rule asking the model to return [silent] after a
successful reminder tool call. Assistant loop now emits
last_successful_tool and final_text_after_tools metadata; _ask_impl
suppresses reply only when the last successful tool was a reminder
mutation AND the post-tool text is strictly empty. No phrase matching.

Watch-mode [silent] in the reminder delivery closure is unchanged."
```

---

## Task B6: PR B finalize

**Step 1: Run full suite + lint + typecheck**

Run: `uv run pytest plugins/llm/ -q && uv run ruff check plugins/llm/ && uv run ty check plugins/llm/`

**Step 2: Manual smoke test on a dev bot if available**

Test cases:
- `@remind in 1m draw a cat` — single fire, draw renders.
- `@remind every 1m draw a cat 3 times` — three fires via **mechanical** reschedule (no LLM tool call), then stops.
- `@remind every Monday at 9am stand-up` — calendar reschedule via rrule path; verify next fire is the upcoming Monday at 9am.
- `@remind every 1m let me know if the build passes` — watch mode; only fires that produce a positive result reach IRC.
- `vibebot, remind me to take out the trash in an hour` — chat-driven set, reaction only (no text reply).
- `@remind clear` — bulk cancel via reaction.
- **In-flight migration check (if B0.5 chose option b):** before deploying, schedule a recurring reminder on the *current* prod build, deploy the new build, verify the in-flight reminder continues to fire and reschedule via the legacy path until exhausted.

**Step 3: Squash review**

Run: `git log --oneline origin/main..HEAD` (after rebasing PR A's commits if needed).

**Step 4: Stop**

Don't push. Hand back to user.

---

# Notes for the executor

- After each task: run `make test-all` (or the equivalent `uv run pytest` invocation) before committing.
- If a test fails for an unexpected reason (not the task's intent): stop and read it. Don't paper over.
- If pre-commit hook flags a finding: fix it in a fresh commit, don't `--amend`.
- Do not push to `main` without the user's explicit say-so. (Memory: pushing to main is otherwise fine, but PR A and PR B should land via PR for review.)
- Memory check: AfterNet has no NickServ. Avoid that term in any new strings or comments.
