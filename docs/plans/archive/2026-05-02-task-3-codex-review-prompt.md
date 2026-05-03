# Codex review prompt — Task 3 implementation plan

You are doing a code-review pass on a Claude-Code-authored implementation plan for a new feature in the `vibebot-v8` Limnoria IRC bot repository (Python 3.14, Limnoria, LiteLLM, SQLite + WAL). The plan is the source of truth for what will be built; it has not been implemented yet. Your job is to find anything wrong, missing, or risky **before code is written** so we can fix the plan, not the code.

## Inputs

Repo root: read the user's local checkout at `/Users/rdrake/workspace/afternet/vibebot-v8`. All paths below are relative to that root.

Read in full:

1. **The plan you are reviewing:**
   `docs/plans/2026-05-02-task-3-schedule-llm-task-implementation-plan.md`

2. **The design plan it implements (Task 3 section is mandatory; skim the rest for context):**
   `docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md`

3. **The canonical exemplar plan whose structure Task 3 mirrors:**
   `docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md`

4. **Repo conventions:**
   `AGENTS.md`, `CLAUDE.md`, `.claude/settings.json`

5. **The existing reminder system (the architectural template Task 3 mirrors):**
   - `plugins/llm/src/llm/service.py` — `parse_reminder` (~2029), `_schedule_reminder` (~3538), `_mechanical_reschedule` (~3369), all `schedule.addEvent(...)` call sites
   - `plugins/llm/src/llm/plugin.py` — `_reminder_fns` (~3218), the four `assistant_request` call sites that splat it (~2530, 2658, 2767, 2865), the reminder fire path (~1151-1224), `_next_rrule_fire` (~3350), `Identity.matches` and its uses
   - `plugins/llm/src/llm/persistence.py` — `SCHEMA_VERSION` (currently 12), `save_reminder`, `load_pending_reminders`, the migration ladder shape
   - `plugins/llm/src/llm/assistant.py` — `ASSISTANT_TOOLS`, `_TOOL_SPEC_OVERRIDES`, `AssistantToolExecutor` and its `_tool_*` handlers, `ToolSpec.denial_reason`
   - `plugins/llm/src/llm/limnoria_bridge.py` — `MUTATING_COMMANDS`, `enumerate_commands`, `dispatch`, the bridge's `BufferingIrcProxy`

6. **Limnoria internals the plan relies on:**
   - `.venv/lib/python3.14/site-packages/supybot/schedule.py` — `addEvent`, `addPeriodicEvent`, `removeEvent`
   - `.venv/lib/python3.14/site-packages/supybot/plugins/Scheduler/plugin.py` — pickle persistence, `_makeCommandFunction`
   - `.venv/lib/python3.14/site-packages/supybot/ircmsgs.py` — `IrcMsg.__reduce__`, `tag()`, `tagged()`

## What the plan proposes (one-line summary)

A native LLM tool `schedule_llm_task` (with companion `list_scheduled_llm_tasks` / `cancel_scheduled_llm_task`) that schedules a future LLM-with-tools invocation. Persistence: new SQLite table `scheduled_llm_tasks` (schema v13). Scheduling: raw `supybot.schedule.addEvent` + DB-backed restore. Fire-time dispatch: rehydrate `IrcMsg` from the persisted wire string, set `msg.tags["llm_schedule_depth"]=1`, call `LLMService.assistant_request` directly (NOT `LLM.ask`, to avoid `MetaSynchronized` from the scheduler thread). Depth cap of 1, per-creator budget via new `bridgeScheduledTaskLimit` channel registry value.

## What I want from you

A concrete, prioritised review.

### 1. Verify load-bearing claims by reading the source

For each of the following, confirm against the actual source files. If a claim is wrong, say which line you read and what the truth is:

- **Pickle behaviour.** The plan asserts `IrcMsg.__reduce__` rebuilds from `str(self)` only and `msg.tags` is lost on pickle. Confirm against `supybot/ircmsgs.py:363-364` and either run the Step 0.4 verification snippet (preferred) or re-derive from the source.
- **Reminder fire-path threading.** The plan asserts `plugin.py:1196` calls `assistant_request` directly and avoids `LLM.ask` to skip the `MetaSynchronized` RLock. Confirm by reading `plugin.py:1140-1225` and Limnoria's `@wrap` / command-dispatch path. Is the RLock claim accurate? Is there a subtler hazard the plan misses?
- **`_next_rrule_fire` reuse.** The plan reduces `_compute_next_fire` to a single delegation. Confirm `_next_rrule_fire(rule_str, now)` exists at `plugin.py:3350` with that exact signature and semantics. Flag any difference (e.g., the existing helper uses `dtstart=now`, but a recurring schedule's first fire was at `row.fire_at` — does `now` vs. `row.fire_at` matter for `rule.after()` correctness?).
- **`Identity.matches` semantics.** The plan uses `Identity(raw_nick=row.creator_nick, account=row.account).matches(caller)` for owner checks. Confirm the matching policy matches what `_get_user_reminders` (`plugin.py:3266-3286`) does, and that the asymmetry of `.matches` (if any — read its body) doesn't break the cancel/list path.
- **Schema migration shape.** Confirm the v13 migration block follows the existing `if current_version < N:` pattern in `persistence.py`. Confirm `SCHEMA_VERSION` is set in exactly one place. Confirm the new indexes match the existing index naming convention.
- **`_reminder_fns(pass_irc_msg_to_callbacks=False)`.** The fire path uses this kwarg. Confirm it exists and behaves as documented (read `plugin.py:3218-3264`).

### 2. Find missing pieces

Look for things the plan should specify but does not:

- **Cancellation race.** Between `schedule.removeEvent(name)` and the fire callable starting to execute (the scheduler may have already popped the event but not yet called it), what happens? The plan's `get_scheduled_llm_task(name) is None` guard at fire-time handles a deleted DB row, but is there a cleaner way?
- **Restart-restore + re-add collision.** `schedule.addEvent` raises `AssertionError("An event with the same name has already been scheduled.")` if the name already exists (`schedule.py:88`). Plugin reload (without process restart) hits this. The plan has a try/except for it in `restore_scheduled_llm_tasks` — is that sufficient, and does it match the Scheduler plugin's `_restoreEvents` pattern at `Scheduler/plugin.py:114-117`?
- **`world.flushers` interaction.** The Scheduler plugin registers itself with `world.flushers` for pickle persistence. We don't, because we use SQLite — but is anything else in our path expecting our events to be in `Scheduler.events`? (Check `@scheduler list` behaviour; the plan's operational test step assumes our events show up there. Will they?)
- **`assistant_request` exclude_tools.** The plan does NOT exclude `set_reminder` at fire time, but the existing reminder fire path does (`plugin.py:1192-1194`) for structured rows to prevent double-reschedule. Should the scheduled-task fire path exclude any tool? If not, why not — and document the reasoning in the plan.
- **`uuid.uuid4().hex[:12]` collision under tests.** Plan acknowledges no retry. Is there a test that seeds RNG / freezes time that would make the truncated UUID collide? If so, does the plan's Pre-flight 0.3 catch it?
- **`Identity` import path.** The plan uses `Identity(...)` in `service.py`. Is `Identity` importable from there? Check the existing imports in `service.py`.
- **`ircmsgs` import in `_dispatch_scheduled_task`.** The plan's snippet uses `ircmsgs.privmsg(...)`. Confirm `ircmsgs` is imported (or should be added) in `service.py`.
- **Channel-vs-PM detection.** `_dispatch_scheduled_task` uses `row.channel.startswith(("#", "&"))`. The codebase has `ircutils.isChannel` (used at `plugin.py:1151, 1168`). The plan should prefer the canonical helper.

### 3. Quality / risk pass

- **Hot-path bloat.** `_dispatch_scheduled_task` calls `_gather_history`, `_get_user_memories`, `db.get_instruction`, `resolve_setting` per fire. For a task firing every 5 minutes, is the per-fire cost bounded? Compare against the reminder fire path which does the same — is there a cached or lazy variant?
- **Memory leak via closures.** Each schedule's fire closure captures `event_name`, `network`, and `self` (the `LLMService`). Reschedule creates a fresh closure (`_make_scheduled_llm_task_callback` is called again in `_maybe_reschedule_or_clean`). Confirm no growing reference chain (e.g., the new closure capturing the old closure transitively).
- **`AssistantRequestContext` field shape.** The plan synthesises one with specific field names (`entry_route`, `profile`, `nick`, `raw_nick`, `account`, `channel`, `is_private`, `is_owner`, `capabilities`). Confirm against the existing dataclass definition that no field is missing or misnamed.
- **`profile="remind_action"` vs a new `profile="scheduled_llm_task"`.** The plan piggybacks on the `remind_action` profile so the depth tag + tool surface matches. Is that the right semantic, or should we add a distinct profile? (Note `ASSISTANT_TOOL_SPECS`'s `visible_in` is what gates which tools appear.)
- **Fire-time silent suppression.** The plan suppresses output when `result.content` is empty or `[silent]`. The reminder path also handles `[silent]`. Confirm we want the same suppression here, and confirm the action prompt says nothing about `[silent]` for non-watch schedules (it could surprise a debugger).

### 4. Smaller polish

- Tests: each test imports `pytest`, `time`, `mocker` — confirm they exist where the test would land (`grep -l "import pytest" plugins/llm/tests/`).
- Style: do the proposed snippets pass `make lint` and `make typecheck` as written? Any obvious type errors (`Callable[..., dict]` vs. `Callable[..., dict[str, Any]]`, missing `from __future__ import annotations`, etc.)?
- Naming: `bridgeScheduledTaskLimit` is the registry key. Is "bridge" the right prefix, given Task 3 is a native tool that doesn't go through the bridge? (The `bridge*` prefix has been used for related-but-not-identical settings; pick one and justify.)
- The plan has a few `**Note...**`/`**Background...**` paragraphs that overlap with the code comments above the same blocks. Flag any redundancy worth pruning to keep the plan readable at ~2700 lines.

### 5. Final scoring

End with three sections:

1. **Blockers** — things that must be fixed before implementation starts (with file path + concrete suggested edit).
2. **Should-fix** — things that should be fixed but aren't strictly blocking.
3. **Nice-to-have** — pure polish.

If you find no blockers, say so explicitly.

## Output

Write your review to **`docs/plans/2026-05-02-task-3-codex-review.md`** (overwrite if it exists). Use this skeleton:

```markdown
# Codex review — Task 3 implementation plan

**Reviewed:** docs/plans/2026-05-02-task-3-schedule-llm-task-implementation-plan.md
**Date:** <YYYY-MM-DD>
**Codex model:** <model id>

## Verified claims
<bulleted, citing line numbers from the actual source files>

## Findings

### Blockers
<numbered; each has: location in plan (unique substring), the issue, suggested edit>

### Should-fix
<same shape>

### Nice-to-have
<same shape>

## Notes for the human
<anything that didn't fit above; <300 words>
```

Keep the review terse. Do not narrate what you read; just report what's wrong. If a section has no findings, write "None." and move on.

When done, print the path of the file you wrote so the user can pipe it back into the original Claude session.
