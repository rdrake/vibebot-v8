# Async LLM Concurrency

Date: 2026-05-06 (revised after Codex + code-reviewer review)

## Problem

Concurrent queries to the bot back up into a multi-minute serial backlog
during busy periods. Two root causes compound:

1. **Main-thread blocking.** Every LLM call that fires from
   `supybot.schedule.addEvent` runs on the main IRC event-loop thread.
   While the call is in flight (10–60 s typical for tool-using agents),
   the bot cannot read or dispatch any new IRC traffic. Spontaneous
   replies, post-reply memory extraction, watch-mode reminders,
   `schedule_llm_task` fires, and the safety-poll pending-task check
   all sit on this path.

2. **Unbounded command-path concurrency.** `Plugin.threaded = True` and
   `_allow_concurrent()` (plugin.py:1372) already release Limnoria's
   per-plugin RLock around the LLM call site, so command-path queries
   *can* run in parallel — but with no upper bound. A burst of 30 user
   queries spawns 30 simultaneous provider calls, hits provider rate
   limits, and amplifies the symptom in (1) because each individual
   call also gets slower.

Symptom: a burst of five queries during an active spontaneous-reply or
memory-extraction window can take minutes to clear.

## Goals

- Move every blocking LLM call off the main IRC event-loop thread.
- Cap **all** simultaneous outbound LLM calls (command path **and**
  background) at a single configurable bound, to protect provider
  rate limits and the host. The cap applies regardless of which
  thread originates the call.
- Preserve current ordering guarantees:
  - Reply for a given user command is sent in response order to that
    command (already true; bounded concurrency does not change this).
  - Background work (memory extraction, watch fires) does not need to
    serialize globally, only avoid clobbering shared state — which
    requires explicit locks added in this PR.
- Keep plugin reload / `die()` fast and **safe**: do not block on
  in-flight LLM calls, and prevent post-`die()` workers from
  resurrecting state that a replacement plugin instance has restored.

## Non-goals

- Switching the codebase to `asyncio` or `litellm.acompletion`. The
  rest of the plugin is sync and Limnoria's dispatch is thread-based;
  introducing an event loop would add an async boundary without
  unlocking anything bounded threads can't.
- Tiered pools / reserved slots for command-path priority. Single cap
  for v1; document the starvation edge case and revisit if observed.
- Per-conversation serialization queues. Existing context lock and
  per-channel cooldowns are sufficient.

## Current state — call-site survey

Call sites that issue LLM calls today, grouped by thread of origin.

### Already off the main thread (Limnoria `CommandThread` workers)

- `LLM.ask` / `LLM.code` / `LLM.draw` (plugin.py:2790, 2909, 2993) —
  Limnoria spawns a `CommandThread` per call (`callbacks.py:1207–1213`),
  and the handler calls `_allow_concurrent()` before the LLM dispatch.
- `LLM.invalidCommand` → `_ask_impl` (plugin.py:1455 → 2790) —
  Limnoria threads `invalidCommand` when any callback declares
  `threaded = True` (`callbacks.py:1098`), so addressed natural-language
  traffic ("vibebot, …") is handled in a worker too.
- `LLM.remind` (plugin.py:3803) — same `_allow_concurrent()` pattern.

These are correctly off the main thread but **not bounded**.

### Currently on the main IRC thread (must move)

| Call site                                  | File:line                       | Notes |
|--------------------------------------------|---------------------------------|-------|
| Spontaneous reply (`_evaluate`)            | plugin.py:864–917 / `addEvent` 923 | Calls `llm_service.completion`; logs usage; queues IRC reply. |
| Memory extraction (`_extract_memories_bg`) | plugin.py:~2480–2570            | Calls `llm_service.completion`; mutates DB; may invoke `_run_memory_cleanup`. |
| Memory cleanup (`_run_memory_cleanup`)     | **plugin.py:2590** (was wrong in v1 of this plan: not service.py:4114) | Reachable from worker and main thread; serializes on `_cleanup_lock`. |
| Watch-mode reminder fire (`_deliver`)      | plugin.py:1065 (closure), body 1141–1306, finally 1307–1310 | Calls `assistant_request` end-to-end inside `schedule.addEvent` callback. Iterates `world.ircs` with `break`. Mechanical reschedule today only on success path (1273). |
| Scheduled LLM task fire                    | service.py:4536 `fire` / 4561 `_dispatch_scheduled_task` | Calls `assistant_request`. `_maybe_reschedule_or_clean` runs unconditionally after dispatch (4557). |
| Safety poll (`_check_pending_tasks`)       | plugin.py:620–652; periodic at 504; event wakeup at 613 | Inspects DB; touches `_next_wakeup_time` (580, 629); the LLM-touching delivery path runs inline. |

These are the migration targets.

## Design

### Component 1 — `LLMExecutor`

New module: `plugins/llm/src/llm/executor.py`.

```python
class LLMExecutor:
    """Owns the global cap for all LLM I/O — both command-path and
    background work share the same bound.

    Two surfaces:

      * ``permit()`` — context manager. Acquires one slot from the
        bounded semaphore and releases on exit. Use this from a thread
        that already exists (Limnoria CommandThread on the command
        path) so the slot bounds the call without spawning a worker.

      * ``submit(label, fn, *args, **kw) -> Future`` — runs ``fn`` on
        a pool worker after acquiring a slot. Use this from the main
        IRC thread (in a ``schedule.addEvent`` shim) so the main
        thread returns immediately.

    Both surfaces share one ``BoundedSemaphore(max_concurrency)``. The
    pool's ``max_workers`` is ``max_concurrency`` so the pool itself
    cannot exceed the cap.
    """

    def __init__(self, max_concurrency: int, log: Logger) -> None: ...

    @contextmanager
    def permit(self) -> Iterator[None]: ...

    def submit(self, label: str, fn, *args, **kw) -> Future: ...

    def running(self) -> int: ...   # holding a permit
    def queued(self) -> int: ...    # submitted but waiting for a permit
    def shutdown(self) -> None: ...

    @property
    def closing(self) -> bool: ...  # True after shutdown() called
```

Behavior of the `submit` wrapper:

- Captures `request_id.get()` at submit time and reapplies it inside
  the worker via `request_id.set(captured)`. **Critical**: stores the
  returned `Token` and `request_id.reset(token)` in the wrapper's
  `finally`, so pool worker reuse never leaks a stale id.
- Acquires the semaphore inside the worker (not at submit) so
  `queued()` reflects backlog accurately.
- Outer try/except logs `label` plus exception via `log.exception`.
- Increments `_running` after permit acquisition; decrements in
  `finally`. `_queued` increments at `submit` and decrements when the
  worker acquires its permit.

`shutdown()`:
- Sets `self._closing = True`.
- Calls `self._executor.shutdown(cancel_futures=True, wait=False)`.
- Does **not** drain workers — see Component 3 for the worker-side
  `closing` gate that prevents post-`die()` state mutation.

### Component 2 — Plugin wiring

- Construct in `LLM.__init__` after `self.llm_service` is built:
  `self._llm_executor = LLMExecutor(max_concurrency=N, log=self.log)`.
- New registry key `supybot.plugins.LLM.maxConcurrentLLMCalls`
  default **`16`** (raised from the brainstorm's 8). Rationale: a
  single user can register multiple watch-mode reminders, plus the
  bot has spontaneous + memory + safety-poll background slots; 8 is
  empirically tight. Operators on small hosts can lower it.
- `Plugin.die()` ordering (replaces today's plugin.py:529–570 sequence):
  1. `self._llm_executor.shutdown()` — sets `closing=True` and cancels
     queued futures.
  2. `schedule.removeEvent("llm_file_cleanup")` and
     `schedule.removeEvent("llm_pending_tasks")`.
  3. `self.db.delete_expired_reminders()` then **defer** `self.db.close()`
     until after a brief drain — see "Worker DB lifetime" below.

### Component 3 — `closing` gate (worker safety on reload)

In-flight workers may try to call `irc.queueMsg`, `db.log_usage`,
`schedule.addEvent` (recurring reminder reschedule), or mutate
`self._reminders` after `die()` returns. With `wait=False` shutdown,
a replacement plugin instance can be constructing on the main thread
while the old plugin's worker is still running.

**Gate**: each migrated worker checks `self._llm_executor.closing` at
specific commit points and short-circuits:

- Before `irc.queueMsg(...)` — silently drop.
- Before `schedule.addEvent(...)` for recurring reschedules — drop;
  the replacement plugin's `restore_scheduled_llm_tasks()` /
  `_reload_reminders()` paths re-register from DB on startup.
- Before `db.log_usage` — drop. Acceptable lossage for a shutdown.
- Before `self._reminders` mutation — drop; replacement reloaded the
  dict from DB.

### Component 4 — Migrate the command path (NEW vs v1 plan)

The command path stays on Limnoria `CommandThread`s but **must**
acquire a permit so it counts against the cap.

```python
# plugin.py — inside _ask_impl (and code, draw, remind equivalents):
with self._trace_request("ask", nick, channel):
    history, channel_history = self._gather_history(nick, channel)
    ...
    with self._allow_concurrent(), self._llm_executor.permit():
        result = self.llm_service.assistant_request(...)
        response, should_log = self._dispatch_assistant_reply(...)
```

`_allow_concurrent()` releases Limnoria's per-plugin RLock so other
command threads can enter the dispatcher. `permit()` then bounds
concurrent LLM calls. If the semaphore is full, the CommandThread
blocks on `permit()` — that is the desired backpressure.

Affected sites: plugin.py:2790, 2909, 2993, 3803.

### Component 5 — Migrate the main-thread call sites

Pattern: each `schedule.addEvent(callback, when, name=...)` whose
`callback` performs an LLM call becomes `schedule.addEvent` of a
*tiny* main-thread shim that submits to the executor.

```python
# Before:
schedule.addEvent(_evaluate, time.time() + 0.5, name=event_name)

# After:
def _enqueue() -> None:
    if self._llm_executor.closing:
        return
    self._llm_executor.submit(f"spontaneous:{channel}", _evaluate)

schedule.addEvent(_enqueue, time.time() + 0.5, name=event_name)
```

Per-site decisions:

#### Spontaneous reply (plugin.py:864–923)

- Straight submit. Cooldown bookkeeping (`_spontaneous_cooldowns`) and
  `_spontaneous_events.add(event_name)` happen on the main thread at
  schedule time, before the executor submit, so the cooldown and the
  set's add-then-discard pair behave as today.
- `_spontaneous_events` mutations from `_evaluate` (line 919, on
  worker after this PR) and from `__init__`/`die()` (line 559) need a
  lock. Add `self._spontaneous_events_lock = threading.Lock()`.

#### Memory extraction (plugin.py:~2480–2570)

- Straight submit of the entire `_extract_memories_bg` closure, not
  just lines 2540–2570 (the closure body that promotes candidates and
  triggers cleanup is inside the same closure).
- **No recursive submit**: `_run_memory_cleanup` (plugin.py:2590) is
  called inline within the same worker. Submitting it back into the
  executor risks deadlock when the pool is saturated (every worker
  blocked on a future its own pool needs to run). The plan **forbids**
  any worker calling `executor.submit(...)`. Worker-internal LLM
  calls go straight through `llm_service` with the worker's existing
  permit covering them.

#### Watch-mode reminder fire (plugin.py:1065 closure, body 1141–1306)

The most fragile site. Spelled out:

1. **`active_irc` is precomputed on the main thread.** The shim
   resolves `for irc_conn in world.ircs: ... break` to a single
   reference (or `None`), passes it to the worker. The worker never
   touches `world.ircs`. (Avoids mutation racing with reconnects.)
2. **Rate-limit precheck stays on the main thread shim.** Calls
   `_check_rate_limit(silent=True)`. If blocked, send the user-visible
   skip notice via `irc.queueMsg` from the main thread, do not submit.
3. **Worker runs the inner `try`** (today's lines 1141–1306) — history
   gather, memory + instruction load, system prompt build,
   `assistant_request`, response dispatch, usage log.
4. **Worker `finally`** (replaces today's 1307–1310 + 1273–1286 split)
   does, in order:
   - `_mechanical_reschedule(...)` if `is_structured` — happens
     **regardless** of whether the inner try raised.
   - `self._reminders.pop(event_name, None)` under `_reminders_lock`.
   - `self.db.delete_reminder(event_name)`.
   - All four wrapped in the `closing` gate per Component 3.
5. **Behavior change (explicit, deliberate):** today, an exception
   inside the inner try (LLM timeout, transient provider error)
   skips `_mechanical_reschedule` and the watch chain dies after one
   bad fire. After this change, the chain survives a single failed
   fire. This is the intended product behavior for watch mode and
   must be covered by a regression test that asserts both the
   pre-change "skip" behavior is gone and the new "reschedule on
   exception" behavior is present.

#### Scheduled LLM task fire (service.py:4536–4559)

- Shim: submit `_dispatch_scheduled_task` to the executor.
- Worker `finally` runs `_maybe_reschedule_or_clean` (today's line
  4557 unconditional path), under the `closing` gate.
- `_check_rate_limit` precheck (today at service.py:4609) stays in
  the worker (it has access to the captured row); guarded by the new
  `_rate_buckets_lock`.

#### Safety poll (`_check_pending_tasks`, plugin.py:620–652)

This is the trickiest because the existing poll structure has two
new hazards once concurrent:

1. The periodic schedule reschedules after the **shim** returns
   (supybot `schedule.py:116`), not after the **worker** finishes —
   so a slow worker can be running when the next tick fires. Result:
   two polls running concurrently, both calling
   `claim_due_pending_tasks` and racing on `_next_wakeup_time`.
2. The event-wakeup path (plugin.py:613) targets the same callable.

**Mitigation**: an in-flight guard `self._safety_poll_inflight =
threading.Event()`. The shim:

```python
def _enqueue_safety_poll() -> None:
    if self._llm_executor.closing:
        return
    if self._safety_poll_inflight.is_set():
        return  # previous poll still running; skip this tick
    self._safety_poll_inflight.set()
    fut = self._llm_executor.submit("safety_poll", self._run_safety_poll)
    fut.add_done_callback(lambda _f: self._safety_poll_inflight.clear())
```

Per-task granularity is rejected for v1: claim-on-main / process-on-
worker would require restructuring `claim_due_pending_tasks` and
`release_pending_task`. The single-poll-at-a-time guard is sufficient
for the observed load.

### Component 6 — `irc.queueMsg` thread-safety

Limnoria's `IrcMsgQueue.enqueue` (`irclib.py:245`) holds no explicit
lock and the duplicate check `if msg in self` followed by `enqueue`
is a TOCTOU. Today every `queueMsg` call originates from main thread
or a `CommandThread` under the per-plugin RLock — never concurrent.
After this PR, N workers can call `queueMsg` simultaneously.

**Decision: marshal worker output through a small plugin-side lock.**
Cheap insurance, no change to upstream Limnoria.

```python
# plugin.py
self._irc_send_lock = threading.Lock()

def _safe_queue(self, irc, msg) -> None:
    if self._llm_executor.closing:
        return
    with self._irc_send_lock:
        irc.queueMsg(msg)
```

Worker-thread call sites (spontaneous, memory cleanup notices,
reminder fires, scheduled task fires, safety poll deliveries) call
`self._safe_queue` instead of `irc.queueMsg`. Command-path call sites
remain on `irc.reply` (Limnoria internally serializes those through
the same queue, just from a single thread today).

### Component 7 — Lock additions

- `self._rate_buckets_lock = threading.Lock()` — wraps every read/
  mutate of `self._rate_buckets` in `_is_rate_limited` (plugin.py:
  ~2192) and `_record_rate_limit_hit` (~2216). Required for both
  reminder-fire and scheduled-task-fire paths now running on workers.
- `self._spontaneous_events_lock = threading.Lock()` — see Component 5.
- `self._reminders_lock` already exists; audit every `_reminders`
  access in this PR to confirm no missing-lock cases now that more
  concurrent writers exist.
- `self._irc_send_lock` per Component 6.

### Component 8 — Worker DB lifetime

`LLMDatabase` uses `threading.local` connections (persistence.py:167).
On `die()`:

- `self.db.close()` today closes only the main thread's local
  connection. Worker-thread connections are released when those
  threads exit.
- After this PR, worker threads from the executor pool are long-lived
  (one per slot), so connections accumulate up to `max_concurrency`
  per plugin lifetime. Acceptable.
- `die()` ordering: shutdown the executor **first** (Component 2),
  but do **not** call `self.db.close()` on the main thread until at
  least one drain attempt — `self._llm_executor._executor.shutdown(
  wait=True, timeout=2)` in a separate call after the cancel. Two
  seconds is enough to let a finishing worker write its last
  `log_usage`, then close. Workers still running after the drain
  are abandoned; their `closing` gate prevents post-close DB writes.

### Component 9 — Observability

- Submit log: `llm_executor submit label=… running=… queued=… max=…`.
- Done log: `llm_executor done label=… elapsed_ms=…`.
- Saturation warn log when `running >= max` for >5s.
- Add `running=… queued=… max=…` to `completion_timing`.
- `%status` (or equivalent) gains `llm_executor: running/queued/max`.

## Risks and edge cases

- **Pool starvation under reminder + spontaneous bursts.** With
  default 16 and a steady-state of 4 watch-mode reminders + memory
  extraction + spontaneous + safety poll, ~7 slots are routinely
  occupied. A burst of user commands can still saturate. Mitigation
  in this PR is documentation and the operator-tunable knob.
  Follow-up: tiered slots (out of scope).
- **`schedule.removeEvent` race.** Upstream pops `events` before
  taking the lock (`schedule.py:95`). The plan does not assert
  "schedule.addEvent is safe from any thread" — instead, all worker
  reschedules sit behind the `closing` gate, and `die()` removes
  events **after** the executor's cancel-futures completes. This
  doesn't fully eliminate the upstream race but sequences our
  callbacks so we don't aggravate it.
- **Behavior change in watch-mode rescheduling.** Documented in
  Component 5 step 5; covered by regression test.
- **Recursive `submit()` from a worker is forbidden.** Static check:
  add a Ruff custom rule or a runtime assertion in `submit()` that
  inspects the calling thread name (`llm-worker-*`) and raises if
  invoked from a worker. Cheap guardrail.

## Testing

Unit:
- `LLMExecutor`: exception swallowing, request-id token reset on
  worker reuse, running/queued counters under load, shutdown does
  not block, `closing` flag flips.
- Lock additions: `_rate_buckets_lock` race test (N workers calling
  `_check_rate_limit` for same account — assert no `KeyError` from
  pop-then-get, no lost timestamps).
- `_safe_queue` serializes (sanity test under N concurrent calls).

Integration / regression:
- Each migrated call site asserts (a) `schedule.addEvent` is invoked
  with the shim, (b) the shim submits to a fake executor, (c) the
  original side effects (DB mutations, IRC sends) happen inside the
  submitted task, (d) the `closing` gate short-circuits when set.
- Watch-mode regression: assistant_request raises in the worker —
  user gets fallback message, **chain still reschedules**. (Behavior
  change.)
- Watch-mode cancel during fire: worker is mid-fire on chain X;
  main-thread `cancel_pending_task` for chain X — assert the cancel
  wins and the worker's reschedule is suppressed.
- Scheduled LLM task: dispatch raises, reschedule still fires.
- Safety poll: in-flight guard prevents overlapping polls; `Event` is
  cleared on worker exception too.
- Reload mid-flight: a slow fake task is submitted, `Plugin.die()` is
  called, replacement plugin instance is constructed; assert the old
  worker's post-completion `db.log_usage` and reschedule attempts
  short-circuit and do not write to the new plugin's DB.

Concurrency / stress:
- N > max_concurrency tasks submitted simultaneously; cap honored,
  all complete, `queued()` reaches > 0.
- Concurrent `queueMsg` from N workers (proves `_irc_send_lock`
  serializes; no message corruption).
- Recursive-`submit` guardrail fires when called from worker.

Coverage: ≥90% target for the new `executor.py`. Existing 80% floor
must hold.

## Rollout

Single PR. Behavior changes observable in logs (`llm_executor` lines)
and via `%status`. Registry change: new `maxConcurrentLLMCalls` key,
default 16. Pre-flight: `make preflight`.

Module placement: `plugins/llm/src/llm/executor.py` (plugin-local).
The RPG plugin doesn't issue LLM calls today; if that changes,
promote to a workspace-shared location.
