# Async LLM Concurrency Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the multi-minute query backlog by moving every blocking
LLM call off the main IRC event-loop thread and applying a single global
concurrency cap to all LLM I/O.

**Architecture:** New `LLMExecutor` (BoundedSemaphore + bounded
ThreadPoolExecutor) exposes `permit()` for the command path and
`submit()` for main-thread shims. Every `schedule.addEvent` callback
that issues LLM calls becomes a tiny shim that submits to the executor
and returns immediately. New locks (`_rate_buckets_lock`,
`_spontaneous_events_lock`, `_irc_send_lock`) and a `closing` gate
make the worker side reload-safe.

**Tech Stack:** Python 3.12–3.14, Limnoria/supybot, LiteLLM, pytest,
ruff, ty. Uses `concurrent.futures.ThreadPoolExecutor`,
`threading.{BoundedSemaphore, Lock, Event}`, `contextvars`.

**Reference design:** `docs/plans/2026-05-06-async-llm-concurrency.md`

**Conventions:**

- After every Python edit: `make lint && make typecheck`.
- Before considering a task done: `make test` (use `make test-all`
  when slow/concurrency tests are touched).
- Final task before merge: `make preflight`.
- Commit messages follow `type(scope): summary` (see `git log --oneline`).

---

## Task 1: Create `LLMExecutor` module with full test coverage

**Files:**
- Create: `plugins/llm/src/llm/executor.py`
- Create: `plugins/llm/tests/test_executor.py`

**Step 1: Write failing tests**

Create `plugins/llm/tests/test_executor.py`:

```python
"""Tests for the LLMExecutor — global cap on LLM I/O concurrency."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from llm.executor import LLMExecutor, RecursiveSubmitError
from llm.tracing import request_id


class TestLLMExecutorBasics:
    def test_submit_runs_function(self) -> None:
        """GIVEN an executor WHEN submit a function THEN function runs and result returned."""
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            fut = ex.submit("test", lambda: 42)
            assert fut.result(timeout=2) == 42
        finally:
            ex.shutdown()

    def test_permit_acquires_and_releases(self) -> None:
        """GIVEN an executor WHEN entering permit twice in nested with THEN second blocks until first releases."""
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            entered = threading.Event()
            release = threading.Event()

            def hold() -> None:
                with ex.permit():
                    entered.set()
                    release.wait(timeout=2)

            t = threading.Thread(target=hold)
            t.start()
            assert entered.wait(timeout=1)
            assert ex.running() == 1

            with_acquired = threading.Event()

            def try_acquire() -> None:
                with ex.permit():
                    with_acquired.set()

            t2 = threading.Thread(target=try_acquire)
            t2.start()
            assert not with_acquired.wait(timeout=0.2), "second permit should block"

            release.set()
            t.join(timeout=2)
            assert with_acquired.wait(timeout=1)
            t2.join(timeout=2)
        finally:
            ex.shutdown()

    def test_submit_swallows_exceptions(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN an executor WHEN submitted function raises THEN exception logged with label, no crash."""
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            with caplog.at_level(logging.ERROR):
                fut = ex.submit("bad-task", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
                with pytest.raises(RuntimeError):
                    fut.result(timeout=2)
            assert any("bad-task" in r.getMessage() for r in caplog.records)
        finally:
            ex.shutdown()

    def test_running_and_queued_counters(self) -> None:
        """GIVEN max=1 WHEN submit 3 tasks THEN running=1, queued=2 mid-flight."""
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            release = threading.Event()

            def slow() -> None:
                release.wait(timeout=5)

            futs = [ex.submit(f"t{i}", slow) for i in range(3)]
            # Give pool time to pick up the first task.
            for _ in range(50):
                if ex.running() == 1:
                    break
                time.sleep(0.02)
            assert ex.running() == 1
            assert ex.queued() == 2
            release.set()
            for f in futs:
                f.result(timeout=2)
        finally:
            ex.shutdown()

    def test_request_id_propagation_with_reset(self) -> None:
        """GIVEN submit captures request_id WHEN worker runs THEN id is set, AND reset after so pool reuse doesn't leak."""
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            seen: list[str | None] = []

            def capture() -> None:
                seen.append(request_id.get())

            tok = request_id.set("trace-A")
            try:
                ex.submit("a", capture).result(timeout=2)
            finally:
                request_id.reset(tok)

            # Submit again with NO request_id set on caller. Worker thread is reused.
            ex.submit("b", capture).result(timeout=2)

            assert seen[0] == "trace-A"
            # Pool worker must NOT carry "trace-A" into the second submission.
            assert seen[1] != "trace-A"
        finally:
            ex.shutdown()

    def test_shutdown_sets_closing_flag(self) -> None:
        """GIVEN an executor WHEN shutdown THEN closing flag is True."""
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        assert ex.closing is False
        ex.shutdown()
        assert ex.closing is True

    def test_recursive_submit_guard_raises(self) -> None:
        """GIVEN running inside a worker WHEN submit called THEN RecursiveSubmitError raised."""
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            captured: list[Exception] = []

            def inner() -> None:
                try:
                    ex.submit("inner", lambda: None)
                except RecursiveSubmitError as e:
                    captured.append(e)

            ex.submit("outer", inner).result(timeout=2)
            assert captured, "expected RecursiveSubmitError from worker thread"
        finally:
            ex.shutdown()
```

**Step 2: Run to confirm failure**

Run: `cd plugins/llm && uv run pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm.executor'`

**Step 3: Implement the module**

Create `plugins/llm/src/llm/executor.py`:

```python
"""Global concurrency cap for all LLM I/O in the plugin.

Both command-path and background callers funnel through here so a
single ``maxConcurrentLLMCalls`` knob bounds outbound provider load
regardless of which thread originated the call.

Two surfaces:

* ``permit()`` — context manager. Use from a thread that already
  exists (Limnoria CommandThread). Acquires/releases one slot from
  the bounded semaphore.
* ``submit(label, fn, *args, **kw) -> Future`` — runs ``fn`` on a
  pool worker after acquiring a slot. Use from the main IRC thread
  inside a ``schedule.addEvent`` shim so the main thread returns
  immediately.

Workers must NEVER call :meth:`submit` recursively — saturating the
pool would deadlock. ``RecursiveSubmitError`` is raised at runtime as
a guardrail.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from logging import Logger
from typing import Any

from .tracing import request_id

_WORKER_THREAD_PREFIX = "llm-worker"


class RecursiveSubmitError(RuntimeError):
    """Raised when LLMExecutor.submit is called from a pool worker."""


class LLMExecutor:
    def __init__(self, max_concurrency: int, log: Logger) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._max = max_concurrency
        self._log = log
        self._semaphore = threading.BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix=_WORKER_THREAD_PREFIX,
        )
        self._running = 0
        self._queued = 0
        self._counter_lock = threading.Lock()
        self._closing = False

    @property
    def max_concurrency(self) -> int:
        return self._max

    @property
    def closing(self) -> bool:
        return self._closing

    def running(self) -> int:
        with self._counter_lock:
            return self._running

    def queued(self) -> int:
        with self._counter_lock:
            return self._queued

    @contextmanager
    def permit(self) -> Iterator[None]:
        self._semaphore.acquire()
        with self._counter_lock:
            self._running += 1
        try:
            yield
        finally:
            with self._counter_lock:
                self._running -= 1
            self._semaphore.release()

    def submit(self, label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        if threading.current_thread().name.startswith(_WORKER_THREAD_PREFIX):
            raise RecursiveSubmitError(
                f"submit('{label}') called from worker thread; nested LLM calls "
                "must run inline within the existing permit"
            )
        captured_rid = request_id.get()
        with self._counter_lock:
            self._queued += 1
        submit_t = time.monotonic()
        self._log.info(
            "llm_executor submit label=%s running=%s queued=%s max=%s",
            label,
            self._running,
            self._queued,
            self._max,
        )

        def _run() -> Any:
            self._semaphore.acquire()
            with self._counter_lock:
                self._queued -= 1
                self._running += 1
            token = request_id.set(captured_rid) if captured_rid else None
            start = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except Exception:
                self._log.exception("llm_executor task raised: label=%s", label)
                raise
            finally:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                self._log.info(
                    "llm_executor done label=%s elapsed_ms=%s queued_ms=%s",
                    label,
                    elapsed_ms,
                    int((start - submit_t) * 1000),
                )
                if token is not None:
                    request_id.reset(token)
                with self._counter_lock:
                    self._running -= 1
                self._semaphore.release()

        return self._executor.submit(_run)

    def shutdown(self) -> None:
        self._closing = True
        self._executor.shutdown(cancel_futures=True, wait=False)
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_executor.py -v`
Expected: 7 passed.

Also run: `make lint && make typecheck` (from repo root).

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/executor.py plugins/llm/tests/test_executor.py
git commit -m "feat(llm): add LLMExecutor for bounded concurrent LLM I/O"
```

---

## Task 2: Add `maxConcurrentLLMCalls` registry value

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (append a new `conf.registerGlobalValue` block; pattern matches `memoryCleanupInterval` at lines 289–297).
- Modify: `plugins/llm/tests/test_config.py` if it exhaustively asserts the registry key set; otherwise no test needed (registry registration is exercised by plugin construction).

**Step 1: Add the registry key**

Append after the last `registerGlobalValue` block in `config.py`:

```python
conf.registerGlobalValue(
    LLM,
    "maxConcurrentLLMCalls",
    registry.PositiveInteger(
        16,
        _("""Maximum number of simultaneous outbound LLM calls (across the
        command path and background work — spontaneous replies, memory
        extraction, watch-mode reminders, scheduled tasks). Lower this on
        small hosts or when the provider rate-limits aggressively."""),
    ),
)
```

**Step 2: Verify**

Run: `make lint && make typecheck`
Run: `cd plugins/llm && uv run pytest tests/test_config.py -v`
Expected: existing tests pass.

**Step 3: Commit**

```bash
git add plugins/llm/src/llm/config.py
git commit -m "feat(llm): register maxConcurrentLLMCalls config (default 16)"
```

---

## Task 3: Wire `LLMExecutor` into plugin lifecycle

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `LLM.__init__` (~423–517), `LLM.die` (529–570).
- Modify: `plugins/llm/tests/test_plugin.py` — add lifecycle test.

**Step 1: Write failing test**

Add to `tests/test_plugin.py`:

```python
class TestLLMExecutorLifecycle:
    def test_plugin_constructs_executor(self, plugin: "MagicMock") -> None:
        """GIVEN plugin construction THEN _llm_executor is set with configured max."""
        from llm.executor import LLMExecutor
        assert isinstance(plugin._llm_executor, LLMExecutor)
        assert plugin._llm_executor.max_concurrency >= 1

    def test_die_shuts_down_executor(self, plugin: "MagicMock") -> None:
        """GIVEN running plugin WHEN die called THEN executor.closing is True."""
        plugin.die()
        assert plugin._llm_executor.closing is True
```

(If `plugin` fixture doesn't exist or returns a MagicMock with the
plugin already constructed, adapt to whatever `conftest.py` provides;
look at how existing tests instantiate `LLM`.)

**Step 2: Verify test fails**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestLLMExecutorLifecycle -v`
Expected: FAIL with `AttributeError` on `_llm_executor`.

**Step 3: Wire in `__init__`**

In `plugin.py`, after `self.llm_service = LLMService(self)` (~line 432),
add:

```python
        # Global concurrency cap for all LLM I/O. See
        # docs/plans/2026-05-06-async-llm-concurrency.md
        from .executor import LLMExecutor
        self._llm_executor = LLMExecutor(
            max_concurrency=self.registryValue("maxConcurrentLLMCalls"),
            log=self.log,
        )
```

In `die()` (line 529), shutdown the executor **first**, then proceed
with existing teardown:

```python
    def die(self) -> None:
        """Clean up when plugin is unloaded."""
        # Shutdown the executor before mutating shared state so workers
        # see closing=True at their commit points.
        if hasattr(self, "_llm_executor"):
            self._llm_executor.shutdown()

        # Brief drain — give already-running workers a chance to flush
        # final db.log_usage / queueMsg calls before we close the DB.
        if hasattr(self, "_llm_executor"):
            self._llm_executor._executor.shutdown(wait=True, timeout=2.0)
        # ... existing body unchanged below ...
```

**Step 4: Verify tests pass**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestLLMExecutorLifecycle -v`
Expected: PASS.

Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): wire LLMExecutor into plugin lifecycle"
```

---

## Task 4: Add `_rate_buckets_lock` around rate-limit dict mutations

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `__init__` (~458 add lock), `_is_rate_limited` (~2192), `_record_rate_limit_hit` (~2216).
- Modify: `plugins/llm/tests/test_plugin.py` — add concurrency test.

**Step 1: Write failing test**

Add to `tests/test_plugin.py`:

```python
class TestRateBucketsConcurrency:
    def test_concurrent_rate_limit_check_does_not_raise(self, plugin: "MagicMock") -> None:
        """GIVEN N threads checking rate limit for same account WHEN concurrent THEN no KeyError."""
        import threading
        errors: list[Exception] = []

        def hammer() -> None:
            try:
                for _ in range(200):
                    plugin._record_rate_limit_hit("ask", "alice")
                    plugin._is_rate_limited("ask", "alice", limit=10000, window_s=60)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
```

**Step 2: Verify test fails (or is flaky without lock)**

Run repeatedly: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestRateBucketsConcurrency -v --count=10`
Expected: at least one failure under load (TOCTOU on `pop` then `setdefault`).
If `--count` plugin not available, just run several times.

**Step 3: Add lock and use it**

In `plugin.py` `__init__` around line 458 (next to existing
`self._rate_buckets`):

```python
        self._rate_buckets: dict[str, collections.deque[float]] = {}
        self._rate_buckets_lock = threading.Lock()
```

Wrap every read or mutation of `_rate_buckets` in `_is_rate_limited`
and `_record_rate_limit_hit` with `with self._rate_buckets_lock:`.
Read both functions in full first; the wrap covers the entire
deque-touching body, not just one operation.

**Step 4: Verify tests pass**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestRateBucketsConcurrency -v`
Expected: PASS reliably.

Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "fix(llm): add _rate_buckets_lock for concurrent rate-limit access"
```

---

## Task 5: Add `_spontaneous_events_lock`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `__init__` (~467), `_evaluate` finally (line 919), `__init__` event-add (922), `die` cleanup (559).

**Step 1: Add lock**

In `__init__` next to `self._spontaneous_events`:

```python
        self._spontaneous_events: set[str] = set()
        self._spontaneous_events_lock = threading.Lock()
```

**Step 2: Wrap mutations**

Every `self._spontaneous_events.add(...)`, `discard(...)`, `clear()`,
and `for event_name in list(...)` iteration must be inside
`with self._spontaneous_events_lock:`.

**Step 3: Run tests**

Run: `make lint && make typecheck && make test`.
Expected: PASS.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "fix(llm): add _spontaneous_events_lock for worker-thread safety"
```

---

## Task 6: Add `_safe_queue` helper for thread-safe `irc.queueMsg`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `__init__`, new helper near `_send_long_reply`.
- Create: test in `plugins/llm/tests/test_plugin.py`.

**Step 1: Write failing test**

```python
class TestSafeQueue:
    def test_safe_queue_drops_when_closing(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN executor is closing WHEN _safe_queue called THEN queueMsg NOT called."""
        plugin._llm_executor.shutdown()
        irc = mocker.MagicMock()
        plugin._safe_queue(irc, mocker.sentinel.msg)
        irc.queueMsg.assert_not_called()

    def test_safe_queue_calls_queuemsg(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN normal state WHEN _safe_queue called THEN queueMsg called once."""
        irc = mocker.MagicMock()
        plugin._safe_queue(irc, mocker.sentinel.msg)
        irc.queueMsg.assert_called_once_with(mocker.sentinel.msg)
```

**Step 2: Verify failure**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestSafeQueue -v`
Expected: FAIL — `_safe_queue` AttributeError.

**Step 3: Implement**

In `__init__`:

```python
        self._irc_send_lock = threading.Lock()
```

Add method on `LLM` (near other send helpers, ~line 1797):

```python
    def _safe_queue(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Thread-safe wrapper around irc.queueMsg for worker-thread sends.

        Limnoria's IrcMsgQueue.enqueue mutates internal state without an
        explicit lock; with the new executor pool, multiple workers may
        call queueMsg concurrently. This serializes them on the plugin
        side and short-circuits cleanly when the plugin is shutting
        down.
        """
        if self._llm_executor.closing:
            return
        with self._irc_send_lock:
            irc.queueMsg(msg)
```

**Step 4: Verify**

Run the new tests, `make lint && make typecheck`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): add _safe_queue for thread-safe worker IRC sends"
```

---

## Task 7: Bound the command path with `executor.permit()`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_ask_impl` (around line 2790), `code` (2909), `draw` (2993), `remind` (3803).
- Modify: tests in `plugins/llm/tests/test_commands.py`.

**Step 1: Write failing test**

Add to `tests/test_commands.py` (or appropriate file — see existing
command-test fixtures):

```python
class TestCommandPathPermit:
    def test_ask_acquires_permit(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN ask command WHEN executed THEN executor.permit acquired around assistant_request."""
        # Reduce the cap to 1 so we can detect acquisition by saturation.
        plugin._llm_executor = mocker.MagicMock()
        permit_cm = mocker.MagicMock()
        plugin._llm_executor.permit.return_value = permit_cm
        permit_cm.__enter__.return_value = None
        permit_cm.__exit__.return_value = None
        # ... call into _ask_impl with a stubbed assistant_request ...
        plugin._llm_executor.permit.assert_called()
```

(Adapt to existing test fixtures. The intent: verify `permit()` is
entered before the LLM call.)

**Step 2: Verify failure**

Run the new test. Expected: FAIL.

**Step 3: Wrap the LLM call sites**

For each of `_ask_impl` (around line 2790), `code` (2909), `draw`
(2993), `remind` (3803): change

```python
with self._allow_concurrent():
    result = self.llm_service.assistant_request(...)
```

to

```python
with self._allow_concurrent(), self._llm_executor.permit():
    result = self.llm_service.assistant_request(...)
```

The `permit` MUST be the inner context: `_allow_concurrent` releases
Limnoria's plugin RLock so other commands can dispatch; `permit`
then bounds the LLM I/O. If it were the outer one, blocking on the
semaphore would still hold the RLock.

**Step 4: Verify**

Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat(llm): bound command path with LLMExecutor.permit()"
```

---

## Task 8: Migrate spontaneous reply to executor

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_schedule_spontaneous` (lines 848–923) — wrap `_evaluate` submission.
- Modify: `plugins/llm/tests/test_plugin.py`.

**Step 1: Write failing test**

```python
class TestSpontaneousMigration:
    def test_schedule_spontaneous_uses_executor(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN spontaneous fired WHEN scheduler invokes shim THEN shim submits to executor (does not run _evaluate inline)."""
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_spontaneous(mocker.MagicMock(), "#chan", "alice", "hi")

        # The callable passed to addEvent must be the shim, not _evaluate itself.
        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_called_once()
        label = plugin._llm_executor.submit.call_args[0][0]
        assert label.startswith("spontaneous:")
```

**Step 2: Verify failure**

Run the new test. Expected: FAIL — addEvent currently passes `_evaluate`.

**Step 3: Convert to shim**

Change line 923 from:

```python
schedule.addEvent(_evaluate, time.time() + 0.5, name=event_name)
```

to:

```python
def _enqueue() -> None:
    if self._llm_executor.closing:
        return
    self._llm_executor.submit(f"spontaneous:{channel}", _evaluate)

schedule.addEvent(_enqueue, time.time() + 0.5, name=event_name)
```

Inside `_evaluate`, replace `irc.queueMsg(...)` calls (lines 893,
898–900) with `self._safe_queue(irc, ...)`.

**Step 4: Verify**

Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): submit spontaneous replies via LLMExecutor"
```

---

## Task 9: Migrate memory extraction to executor

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_schedule_memory_extraction` and the `_extract_memories_bg` closure (the body around 2480–2570; the addEvent at 2570).

**Step 1: Write failing test**

```python
class TestMemoryExtractionMigration:
    def test_extraction_submitted_to_executor(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")

        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_called_once()
        assert plugin._llm_executor.submit.call_args[0][0].startswith("memory_extract:")
```

**Step 2: Verify failure**

Run. Expected: FAIL.

**Step 3: Convert to shim**

At line 2570, replace:

```python
schedule.addEvent(_extract_memories_bg, time.time() + 0.1, name=event_name)
```

with:

```python
def _enqueue() -> None:
    if self._llm_executor.closing:
        return
    self._llm_executor.submit(f"memory_extract:{nick}", _extract_memories_bg)

schedule.addEvent(_enqueue, time.time() + 0.1, name=event_name)
```

The inline `_run_memory_cleanup` call inside `_extract_memories_bg`
(line 2564) stays — runs on the same worker, covered by the same
permit. No recursive submit.

Worker-thread `irc` calls inside `_extract_memories_bg` (if any —
audit) use `_safe_queue`.

**Step 4: Verify**

Run tests; `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): submit memory extraction via LLMExecutor"
```

---

## Task 10: Migrate watch-mode reminder fire (highest care)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_make_reminder_delivery_closure` at line 1065, the `_deliver` body 1141–1310.

**Step 1: Write failing tests**

```python
class TestWatchModeReminderMigration:
    def test_dispatch_runs_on_worker(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN reminder _deliver invoked WHEN action_prompt set THEN inner LLM dispatch submitted to executor."""
        # Build a synthetic reminder, invoke _deliver via _make_reminder_delivery_closure.
        # Assert _llm_executor.submit was called with label starting with "reminder:".
        ...

    def test_chain_reschedules_on_exception(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN structured (recurring) reminder WHEN inner assistant_request raises THEN _mechanical_reschedule still called."""
        # NEW BEHAVIOR — today the chain dies on exception. After this PR, it survives.
        ...

    def test_active_irc_resolved_on_main_thread(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN _deliver called WHEN world.ircs is empty in worker THEN main-thread captured the irc reference."""
        # Verify the worker does NOT iterate world.ircs.
        ...

    def test_rate_limit_skip_does_not_submit(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN rate limit blocks WHEN _deliver called THEN executor.submit NOT called and skip notice queued."""
        ...

    def test_finally_runs_under_closing_gate(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN closing=True WHEN worker finally fires THEN no _mechanical_reschedule, no _reminders mutation, no db.delete_reminder."""
        ...
```

(These need the existing reminder test fixtures — see how
`test_plugin.py` builds `make_reminder_row` from `conftest.py` and
adapt.)

**Step 2: Verify failures**

Run the new tests. Expected: FAIL.

**Step 3: Refactor `_deliver`**

Read the entire closure body 1141–1310 first. The refactor:

1. **Extract a worker-side function** `_fire_reminder_action(...)` that
   takes `active_irc`, the captured row, and the LLM-touching body.
2. **Main-thread shim** does:
   - Resolve `active_irc` from `world.ircs` (the loop+break is
     replaced by a single resolution).
   - Run the rate-limit precheck (today's call inside the closure).
     If blocked, `_safe_queue` the skip notice and return.
   - `self._llm_executor.submit(f"reminder:{event_name}", _fire_reminder_action, active_irc, ...)`.
3. **Worker `_fire_reminder_action`** does:
   - The inner try (today's 1141–1306) — history gather, system
     prompt, `assistant_request`, response dispatch via
     `_safe_queue`, usage logging.
   - The `finally` ALWAYS runs:
     - If `is_structured`: `_mechanical_reschedule(...)` under
       the `closing` gate.
     - `with self._reminders_lock: self._reminders.pop(event_name, None)`
       under `closing` gate.
     - `self.db.delete_reminder(event_name)` under `closing` gate.

Note the behavior change: today a raised exception in the inner try
skips `_mechanical_reschedule` entirely (the call is at line 1273,
inside the same try). The new contract reschedules on exception; the
test in step 1 covers it.

**Step 4: Verify**

Run reminder tests: `cd plugins/llm && uv run pytest tests/ -k "reminder" -v`.
Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): submit watch-mode reminder fires via LLMExecutor

Behavior change: a single failed fire (LLM timeout, transient provider
error) no longer kills a recurring watch chain. Rescheduling now runs
in the worker finally regardless of inner-try outcome."
```

---

## Task 11: Migrate scheduled LLM task fire

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — `_make_scheduled_llm_task_callback` (4526–4559), `_dispatch_scheduled_task` (4561+).

**Step 1: Write failing test**

In `tests/test_service.py`:

```python
class TestScheduledLLMTaskMigration:
    def test_fire_submits_to_executor(self, llm_service: "MagicMock", mocker: "MockerFixture") -> None:
        ...
    def test_reschedule_runs_after_dispatch_even_on_exception(self, llm_service: "MagicMock", mocker: "MockerFixture") -> None:
        ...
```

**Step 2: Verify failure.**

**Step 3: Refactor `fire()`**

Today's `fire()` (4536–4559) does:

```python
def fire() -> None:
    row = db.get_scheduled_llm_task(event_name)
    ...
    self._dispatch_scheduled_task(irc, msg, row)
    self._maybe_reschedule_or_clean(row, db)
```

After:

```python
def fire() -> None:
    if self.plugin._llm_executor.closing:
        return
    row = db.get_scheduled_llm_task(event_name)
    if row is None:
        return
    irc = world.getIrc(row.network) or (world.ircs[0] if world.ircs else None)
    if irc is None:
        return
    msg = row.rehydrate_msg()
    msg.tag("llm_schedule_depth", 1)

    def _worker() -> None:
        try:
            self._dispatch_scheduled_task(irc, msg, row)
        finally:
            if not self.plugin._llm_executor.closing:
                self._maybe_reschedule_or_clean(row, db)

    self.plugin._llm_executor.submit(
        f"scheduled_task:{event_name}", _worker
    )
```

Audit `_dispatch_scheduled_task` and replace any `irc.queueMsg(...)`
with `self.plugin._safe_queue(irc, ...)`.

**Step 4: Verify.**

`make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "refactor(llm): submit scheduled LLM task fires via LLMExecutor"
```

---

## Task 12: Migrate safety poll with in-flight guard

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_check_pending_tasks` (620–652), `__init__` for the `Event`.

**Step 1: Write failing test**

```python
class TestSafetyPollGuard:
    def test_overlapping_poll_is_skipped(self, plugin: "MagicMock", mocker: "MockerFixture") -> None:
        """GIVEN previous poll still running WHEN periodic shim fires THEN second poll is skipped."""
        plugin._safety_poll_inflight.set()
        plugin._llm_executor = mocker.MagicMock()
        # Call the shim directly.
        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()
```

**Step 2: Verify failure.**

**Step 3: Add Event and shim**

In `__init__`:

```python
        self._safety_poll_inflight = threading.Event()
```

Replace the periodic registration (line 504) and event registration
(line 613) to point at a new `_enqueue_safety_poll` method:

```python
    def _enqueue_safety_poll(self) -> None:
        if self._llm_executor.closing:
            return
        if self._safety_poll_inflight.is_set():
            return
        self._safety_poll_inflight.set()
        fut = self._llm_executor.submit("safety_poll", self._check_pending_tasks)
        fut.add_done_callback(lambda _f: self._safety_poll_inflight.clear())
```

Update both schedule.addPeriodicEvent (504) and schedule.addEvent
(613) call sites to pass `self._enqueue_safety_poll`.

`_check_pending_tasks` body itself does not need changes beyond
replacing `irc.queueMsg` (if any reachable from this path) with
`self._safe_queue`.

**Step 4: Verify.**

`make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): submit safety-poll via LLMExecutor with in-flight guard"
```

---

## Task 13: Add `%status` executor field

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — find the existing `status`/`stats` command (search `def status` or grep for `"status"` near `irc.reply`).

**Step 1: Locate the status command**

Run: `grep -n "def status\|llm_status\|registryValue.*status" plugins/llm/src/llm/plugin.py`

**Step 2: Add executor field**

Append `running/queued/max` to the existing status output:

```python
ex = self._llm_executor
parts.append(f"executor: {ex.running()}/{ex.queued()}/{ex.max_concurrency}")
```

**Step 3: Add a test**

In `tests/test_commands.py` or `tests/test_plugin.py`:

```python
def test_status_includes_executor_field(self, plugin: "MagicMock", ...) -> None:
    """GIVEN status command WHEN invoked THEN reply contains executor: running/queued/max."""
```

**Step 4: Verify.**

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): expose executor running/queued/max in %status"
```

---

## Task 14: Stress + reload regression tests

**Files:**
- Create: `plugins/llm/tests/test_executor_stress.py`.

**Step 1: Write the suite**

```python
"""Stress and regression tests for the LLM executor migration.

These exercise scenarios both reviewers flagged as risky:
- N>max simultaneous submissions cap honored, all complete
- _safe_queue serializes under N concurrent workers
- request_id propagation correct under load (no leakage between tasks)
- Plugin reload mid-flight: workers' post-completion writes short-circuit
- Recursive submit raises from worker
- Concurrent rate-limit checks race-free
"""

import threading
import time
from concurrent.futures import wait

import pytest

# ... fixtures from conftest ...


def test_cap_honored_under_burst(plugin) -> None:
    ex = plugin._llm_executor
    max_seen = [0]
    release = threading.Event()

    def task() -> None:
        with threading.Lock():
            max_seen[0] = max(max_seen[0], ex.running())
        release.wait(timeout=5)

    futs = [ex.submit(f"t{i}", task) for i in range(ex.max_concurrency * 3)]
    time.sleep(0.1)
    assert max_seen[0] <= ex.max_concurrency
    release.set()
    wait(futs, timeout=5)


def test_safe_queue_serializes(plugin, mocker) -> None:
    """N workers calling _safe_queue concurrently should not interleave."""
    irc = mocker.MagicMock()
    calls: list[int] = []

    def task(i: int) -> None:
        plugin._safe_queue(irc, mocker.MagicMock(idx=i))

    threads = [threading.Thread(target=task, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert irc.queueMsg.call_count == 50


def test_reload_mid_flight_drops_post_completion_writes(plugin, mocker) -> None:
    """Slow task is in-flight; plugin.die() called; worker finishes; writes are no-ops."""
    # Submit a task that blocks on an event, call die() while it's running,
    # release the event, assert the worker's tail-end calls (db.log_usage,
    # _safe_queue) short-circuit on closing=True.
    ...


def test_recursive_submit_from_worker_raises(plugin) -> None:
    from llm.executor import RecursiveSubmitError
    ex = plugin._llm_executor
    captured: list[Exception] = []

    def inner() -> None:
        try:
            ex.submit("inner", lambda: None)
        except RecursiveSubmitError as e:
            captured.append(e)

    ex.submit("outer", inner).result(timeout=2)
    assert captured
```

**Step 2: Run**

`cd plugins/llm && uv run pytest tests/test_executor_stress.py -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_executor_stress.py
git commit -m "test(llm): stress + reload regression tests for LLM executor"
```

---

## Task 15: Documentation update

**Files:**
- Modify: `docs/guide/` — find the operator-facing tuning page; document `maxConcurrentLLMCalls`.
- Modify: `README.md` if there's a tunables section.

**Step 1: Locate**

Run: `rg -n "registryValue|maxConcurrent|tuning" docs/guide/ README.md`

**Step 2: Add operator-facing notes**

Document:
- What the cap does and when to raise/lower it.
- The watch-mode reschedule behavior change ("a single failed fire
  no longer kills the chain").
- The new `executor: running/queued/max` field in `%status`.

**Step 3: Verify**

Run: `make docs` (per AGENTS.md when changing docs).

**Step 4: Commit**

```bash
git add docs/guide/ README.md
git commit -m "docs(llm): document maxConcurrentLLMCalls and watch-reschedule change"
```

---

## Task 16: Final preflight + push

**Step 1:** Run `make preflight` from the repo root.
Expected: all checks pass.

**Step 2:** If preflight fails, fix the specific failures (do not
skip hooks; follow the `_release_save`-style pattern of the codebase
for any new shared-state access).

**Step 3:** Push to main (project allows direct pushes per
`feedback_branch_protection`).

```bash
git push origin main
```

**Step 4:** Wait for the Docker build workflow to complete (per
`feedback_wait_for_docker`), then `systemctl --user restart vibebot`
on the prod host.

---

## Reference

Design doc: `docs/plans/2026-05-06-async-llm-concurrency.md`

Key invariants the implementation must preserve:
1. The cap binds command-path AND background calls (one
   `BoundedSemaphore`).
2. No worker calls `executor.submit()` (recursive-submit guardrail
   raises).
3. `closing` gate is checked before any worker-thread mutation of
   `_reminders`, `db.log_usage`, `irc.queueMsg`, or `schedule.addEvent`.
4. Watch-mode reschedule runs in the worker `finally` regardless of
   inner-try outcome (intentional behavior change).
5. `_rate_buckets` access is locked.
6. `_spontaneous_events` access is locked.
7. `irc.queueMsg` from any worker goes through `_safe_queue`.
