# Codex review — Task 3 implementation plan

**Reviewed:** docs/plans/2026-05-02-task-3-schedule-llm-task-implementation-plan.md
**Date:** 2026-05-02
**Codex model:** gpt-5

## Verified claims

- `IrcMsg.__reduce__` is exactly `(self.__class__, (str(self),))` at `.venv/lib/python3.14/site-packages/supybot/ircmsgs.py:363-364`; `tag()`/`tagged()` use internal `msg.tags` at `ircmsgs.py:366-375`. The Step 0.4 pickle snippet passes: internal tags are lost, nick/args are preserved.
- `schedule.addEvent` rejects duplicate names at `.venv/lib/python3.14/site-packages/supybot/schedule.py:80-93`; one-shot events are popped before callback execution at `schedule.py:146-151`, so a callback may already be running when `removeEvent()` races it.
- The Scheduler plugin persists only its own `self.events` through `world.flushers` at `Scheduler/plugin.py:54-56,121-127`; `@scheduler list` reads that dict at `Scheduler/plugin.py:273-290`. Raw `supybot.schedule.addEvent(...)` events will not show there.
- The reminder action path calls `LLMService.assistant_request` directly at `plugins/llm/src/llm/plugin.py:1196-1224`, but it also manually checks rate limits at `plugin.py:1134-1148`, sanitizes output at `plugin.py:1107-1109`, excludes `set_reminder` for structured rows at `plugin.py:1189-1194`, and logs usage at `plugin.py:1238-1259`.
- The RLock claim is overstated. `@wrap` parses arguments and calls the wrapped function at `.venv/.../supybot/commands.py:1232-1253`; the `MetaSynchronized` lock is on `Commands.__call__`, `callCommand`, and `invalidCommand` at `.venv/.../supybot/callbacks.py:1437-1447`. Normal command dispatch may spawn a thread for `threaded` plugins at `callbacks.py:1205-1213`; directly calling `ask(...)` would bypass that dispatch path.
- `_next_rrule_fire(rule_str, now)` exists with the planned signature at `plugins/llm/src/llm/plugin.py:3350-3367` and uses `rrulestr(rule_str, dtstart=now_utc).after(now_utc)`.
- `Identity.matches` uses account-to-account when both sides have an account, otherwise raw-nick fallback, all case-insensitive, at `plugins/llm/src/llm/plugin.py:110-120`. `_get_user_reminders` uses `Identity(raw_nick=data.nick, account=data.account).matches(caller)` at `plugin.py:3266-3286`.
- `SCHEMA_VERSION` is defined once at `plugins/llm/src/llm/persistence.py:18`; migrations use `if current_version < N:` blocks and stamp `PRAGMA user_version` at `persistence.py:342-362`.
- `_reminder_fns(pass_irc_msg_to_callbacks=False)` exists at `plugins/llm/src/llm/plugin.py:3218-3264`; it suppresses `irc/msg` only for delete/clear reactions, while `set_reminder_fn` still receives the synthetic `irc,msg`.
- `AssistantRequestContext` fields match the plan at `plugins/llm/src/llm/service.py:273-285`. `service.py` already imports `ircmsgs` and `ircutils` at `service.py:23-24`, but not `uuid`, `sqlite3`, `schedule`, `ScheduledLlmTaskRow`, or runtime `Identity`.

## Findings

### Blockers

1. Location in plan: `parsed = self.parse_reminder(when_natural, channel=channel)`
   The existing parser is not a time-only parser. Its prompt says vague requests missing time or message must clarify (`service.py:2077-2082`), so `when_natural="in 60s"` can fail even when `prompt` is valid. Suggested edit: either call `parse_reminder(f"{when_natural} {prompt}", ...)` and ignore parsed `message/action_prompt`, or add an explicit parser mode for schedule-only parsing and test it without mocks.

2. Location in plan: `result = self.assistant_request(`
   The scheduled fire path bypasses `_run_preflight` and does not mirror the reminder path's manual rate-limit check or usage logging. This contradicts the config help claiming fires count against the user's normal ask bucket (`bridgeScheduledTaskLimit` help) and makes recurring tasks invisible to `@usage`. Suggested edit: before `assistant_request`, call `_check_rate_limit` like `plugin.py:1134-1148`; after the result, call `db.log_usage(...)` like `plugin.py:1238-1259` with a new command/status such as `scheduled_llm_task`.

3. Location in plan: `irc.queueMsg(ircmsgs.privmsg(target, response))`
   Generated fire-time output is sent without `sanitize_output`, unlike reminders (`plugin.py:1107-1109`), violating the repo's IRC command-injection invariant. Suggested edit: sanitize before queuing and preferably use a plugin-owned send helper that also handles long/multiline replies.

4. Location in plan: `self._maybe_reschedule_or_clean(row, db)`
   Cancellation during an in-flight recurring fire can be lost. The row is loaded before the LLM call; if cancel deletes the DB row while the LLM runs, `_maybe_reschedule_or_clean(row, db)` will still re-add the event. Suggested edit: before rescheduling, re-check `db.get_scheduled_llm_task(row.event_name)` and skip if missing, matching the reminder clear-wins-over-mid-fire guard at `plugin.py:3422-3431`.

5. Location in plan: `row.event_name, _network_from_msg(row.wire_msg)`
   `_network_from_msg` is undefined, and `_maybe_reschedule_or_clean` has no `network` parameter even though `_make_scheduled_llm_task_callback` needs it. First recurring fire will fail. Suggested edit: persist a `network` column, or pass `network` through `fire()` into `_maybe_reschedule_or_clean(row, db, network)`.

6. Location in plan: `profile="remind_action"` and `**plugin._reminder_fns(`
   The new scheduled fire path does not pass `_scheduled_llm_task_fns` into `assistant_request`, so `schedule_llm_task` will be visible in the `remind_action` profile but unconfigured, returning "Scheduling is not configured" instead of the planned depth-cap refusal. Suggested edit: add `**plugin._scheduled_llm_task_fns(...)` to `_dispatch_scheduled_task`, and also fix D3 to include the actual existing reminder-fire call site at `plugin.py:1218`, not just the chat/code/g/draw sites at `2530/2658/2767/2865`.

7. Location in plan: `Set to 0 to disable scheduling entirely` and `if limit > 0:`
   The implementation treats `0` as unlimited, not disabled. Suggested edit: use `registry.NonNegativeInteger`, reject create when `limit == 0`, and enforce `existing >= limit` only for positive limits.

8. Location in plan: `caller = Identity(raw_nick=row.creator_nick, account=row.account)`
   `service.py` cannot import `Identity` from `plugin.py` at module load without worsening the existing import cycle (`plugin.py` imports `LLMService` before defining `Identity`). Suggested edit: move `Identity` to a neutral module, or keep scheduled-task owner/list/cancel wiring in `plugin.py`; if using a local import inside service methods, state that explicitly and test import/load.

### Should-fix

1. Location in plan: `count_scheduled_llm_tasks_for(account, nick)`
   The SQL list/count helpers compare `account` and `creator_nick` exactly, while `Identity.matches` is case-insensitive (`plugin.py:118-120`). Suggested edit: normalize on insert or query with `lower(...)`; add tests for account/nick case drift.

2. Location in plan: `Maximum number of active LLM-scheduled tasks per creator in this channel`
   The count/list helper does not filter by channel, so the channel-scoped registry value enforces a global-per-user limit. Suggested edit: either add `channel` to the query/count API or change the help text to say the cap is per creator across all channels.

3. Location in plan: `During the wait, run @scheduler list — the new event should appear`
   Raw `schedule.addEvent` events do not populate `Scheduler.events`, and `@scheduler list` reads only `Scheduler.events` (`Scheduler/plugin.py:273-290`). Suggested edit: remove this smoke assertion, or explicitly add a Scheduler-plugin integration layer and accept its persistence/list semantics.

4. Location in plan: `except AssertionError: ... already scheduled; skip`
   Restore returns `len(rows)` even when duplicates were skipped and does not add skipped events to any operator-visible dict (unlike Scheduler `_restoreEvents`, which writes `self.events[name] = event` at `Scheduler/plugin.py:113-117`). Suggested edit: return restored count vs skipped count separately and test plugin reload.

5. Location in plan: `schedule.addEvent(callback, fire_at, name=event_name)`
   Create saves the DB row before scheduling; if `addEvent` raises, the DB row is orphaned. Suggested edit: catch `Exception`/`AssertionError` around `addEvent`, delete the row, and return an error envelope.

6. Location in plan: `response == "[silent]"`
   The plan suppresses `[silent]` for every scheduled task even though only reminder watch-mode documents that sentinel (`plugin.py:1225-1231`). Suggested edit: either persist/use `watch_mode` for scheduled tasks or suppress `[silent]` only for an explicit watch schedule; document the behavior.

7. Location in plan: `row.channel.startswith(("#", "&"))`
   The codebase already uses `ircutils.isChannel(channel)` for this (`plugin.py:1151,1168`). Suggested edit: use `ircutils.isChannel` in service snippets and tests.

8. Location in plan: `def test_schedule_llm_task_creates_db_row_and_schedules_event(llm_service, db, mocker)`
   The proposed test fixtures do not exist; current shared fixtures are `make_service` and `test_db` (`plugins/llm/tests/conftest.py:80-85,391`). `test_service.py` also lacks module imports for `time` and `ReminderParseResult` (`test_service.py:1-15`). Suggested edit: rewrite snippets to current fixture/import style before implementation.

9. Location in plan: `call.kwargs.get("name") == "overdue_ev" or call.args[2] == "overdue_ev"`
   The restore test can index `call.args[2]` when `name` was passed as a keyword and the first clause is false. Suggested edit: compute `name = call.kwargs.get("name") if "name" in call.kwargs else call.args[2]` once.

10. Location in plan: `requires an authenticated account`
   `schedule_llm_task` has `require_account=True`, but direct service calls with `account=None` still schedule if invoked outside `AssistantToolExecutor`. Suggested edit: enforce the account requirement in `LLMService.schedule_llm_task` too, or explicitly state the service method is not a security boundary.

### Nice-to-have

1. Location in plan: `profile="remind_action"`
   Consider a distinct `profile="scheduled_llm_task"` with the same visible tools. It would make logs, prompts, and usage easier to distinguish without overloading reminder semantics.

2. Location in plan: `bridgeScheduledTaskLimit`
   The `bridge` prefix is defensible as Phase 2 bridge-adjacent work, but this is a native tool. Either rename to `scheduledTaskLimit` before shipping or add one sentence explaining why the bridge prefix is intentional.

3. Location in plan: `IrcMsg(s=self.wire_msg)`
   Add an explicit test that IRCv3 `server_tags` in the wire string survive rehydration, not just internal `msg.tags` loss.

4. Location in plan: `Background — schema:` and repeated code comments
   Several background/note paragraphs duplicate adjacent snippet comments. Prune the repeated pickle/closure explanations after the blockers above are fixed to keep the plan easier to execute.

## Notes for the human

The plan's pivot away from Scheduler pickle is sound, and the basic DB-backed restore shape matches the existing reminder architecture. The risky part is that the scheduled-task fire path is not just "assistant_request directly"; the existing reminder path has accumulated manual guardrails because it bypasses normal command preflight. The implementation plan needs those guardrails copied deliberately, plus a parser fix for the split `when_natural`/`prompt` tool schema, before code starts.
