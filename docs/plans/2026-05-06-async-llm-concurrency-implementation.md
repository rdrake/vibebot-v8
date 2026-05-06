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
`_spontaneous_events_lock`, `_irc_send_lock`) and a `closing` gate make
the worker side reload-safe.

**Tech Stack:** Python 3.12–3.14, Limnoria/supybot, LiteLLM, pytest,
ruff, ty. Uses `concurrent.futures.{ThreadPoolExecutor, wait}`,
`threading.{BoundedSemaphore, Lock, Event}`, `contextvars`.

**Reference design:** `docs/plans/2026-05-06-async-llm-concurrency.md`

**Conventions:**

- After every Python edit: `make lint && make typecheck`.
- Before considering a task done: `make test` (use `make test-all`
  when slow/concurrency tests are touched).
- Final task before merge: `make preflight`.
- Commit messages follow `type(scope): summary` (see `git log --oneline`).

**Test fixtures:** the canonical fixture is `plugin_env` (defined in
`plugins/llm/tests/conftest.py:155`) which yields a tuple
`(plugin, mock_irc, mock_msg)`. Every test below destructures it.

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
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            fut = ex.submit("test", lambda: 42)
            assert fut.result(timeout=2) == 42
        finally:
            ex.shutdown()

    def test_permit_acquires_and_releases(self) -> None:
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

            second_acquired = threading.Event()

            def try_acquire() -> None:
                with ex.permit():
                    second_acquired.set()

            t2 = threading.Thread(target=try_acquire)
            t2.start()
            assert not second_acquired.wait(timeout=0.2)

            release.set()
            t.join(timeout=2)
            assert second_acquired.wait(timeout=1)
            t2.join(timeout=2)
        finally:
            ex.shutdown()

    def test_submit_swallows_and_logs_exceptions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            with caplog.at_level(logging.ERROR):
                fut = ex.submit("bad", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
                with pytest.raises(RuntimeError):
                    fut.result(timeout=2)
            assert any("bad" in r.getMessage() for r in caplog.records)
        finally:
            ex.shutdown()

    def test_running_and_queued_counters(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        try:
            release = threading.Event()
            futs = [ex.submit(f"t{i}", release.wait, 5) for i in range(3)]
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

            ex.submit("b", capture).result(timeout=2)
            assert seen[0] == "trace-A"
            assert seen[1] != "trace-A"
        finally:
            ex.shutdown()

    def test_shutdown_sets_closing_flag(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        assert ex.closing is False
        ex.shutdown()
        assert ex.closing is True

    def test_recursive_submit_from_worker_raises(self) -> None:
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        try:
            captured: list[Exception] = []

            def inner() -> None:
                try:
                    ex.submit("inner", lambda: None)
                except RecursiveSubmitError as e:
                    captured.append(e)

            ex.submit("outer", inner).result(timeout=2)
            assert captured
        finally:
            ex.shutdown()

    def test_drain_waits_for_in_flight(self) -> None:
        """drain(timeout) waits for in-flight tasks; finished within budget => True."""
        ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
        release = threading.Event()
        ex.submit("slow", release.wait, 5)
        time.sleep(0.05)
        # Schedule release after a short delay; drain should return True.
        threading.Timer(0.2, release.set).start()
        assert ex.drain(timeout=2.0) is True
        ex.shutdown()

    def test_drain_returns_false_on_timeout(self) -> None:
        ex = LLMExecutor(max_concurrency=1, log=logging.getLogger("test"))
        release = threading.Event()
        ex.submit("very-slow", release.wait, 30)
        time.sleep(0.05)
        assert ex.drain(timeout=0.2) is False
        release.set()
        ex.shutdown()
```

**Step 2: Run to confirm failure**

Run: `cd plugins/llm && uv run pytest tests/test_executor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.executor'`.

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
a guardrail. Note: the guard fires at ``submit()`` call time (outside
the worker function body), not inside the worker wrapper. Keep it
that way — putting it inside the wrapper would not catch the case
where ``submit`` is called from worker context but never executes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
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
        # Track in-flight futures so drain() can wait without re-shutting down.
        self._inflight: set[Future[Any]] = set()
        self._inflight_lock = threading.Lock()

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
                f"submit('{label}') called from worker thread; nested LLM "
                "calls must run inline within the existing permit"
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

        fut = self._executor.submit(_run)
        with self._inflight_lock:
            self._inflight.add(fut)
        fut.add_done_callback(self._discard_inflight)
        return fut

    def _discard_inflight(self, fut: Future[Any]) -> None:
        with self._inflight_lock:
            self._inflight.discard(fut)

    def drain(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for in-flight tasks to finish.

        Returns True if all in-flight tasks completed within the budget,
        False otherwise. Does NOT shut down the executor — callers do
        that via :meth:`shutdown` (typically before drain so queued
        futures are cancelled and only the actually-running tasks are
        awaited).
        """
        with self._inflight_lock:
            futs = set(self._inflight)
        if not futs:
            return True
        done, not_done = wait(futs, timeout=timeout)
        return not not_done

    def shutdown(self) -> None:
        """Mark closing and cancel queued futures. Idempotent."""
        if self._closing:
            return
        self._closing = True
        self._executor.shutdown(cancel_futures=True, wait=False)
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_executor.py -v`
Expected: 9 passed.

Run: `make lint && make typecheck` (from repo root).

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/executor.py plugins/llm/tests/test_executor.py
git commit -m "feat(llm): add LLMExecutor for bounded concurrent LLM I/O"
```

---

## Task 2: Add `maxConcurrentLLMCalls` registry value (+ test conftest)

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (append a new
  `conf.registerGlobalValue` block; pattern matches
  `memoryCleanupInterval` at lines 289–297).
- Modify: `plugins/llm/tests/conftest.py` — add the key to
  `make_registry_side_effect` defaults (line 279) so plugin
  construction in tests sees an int, not the empty-string fallback
  at line 369.

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

**Step 2: Update test conftest defaults**

In `plugins/llm/tests/conftest.py`, add to the `defaults` dict in
`make_registry_side_effect` (alongside other plugin-level init keys
near line 286–293):

```python
        # Async LLM concurrency cap
        "maxConcurrentLLMCalls": 16,
```

Without this, the empty-string fallback at line 369 would propagate
into `LLMExecutor(max_concurrency="")` and crash plugin construction
in any test that builds a real `LLM` instance.

**Step 3: Verify**

Run: `make lint && make typecheck`.
Run: `cd plugins/llm && uv run pytest tests/ -v -k "config or plugin_env"`.
Expected: existing tests still pass.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat(llm): register maxConcurrentLLMCalls (default 16) and wire test default"
```

---

## Task 3: Wire `LLMExecutor` into plugin lifecycle

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `LLM.__init__` (~423–517),
  `LLM.die` (529–570).
- Modify: `plugins/llm/tests/test_plugin.py` — add lifecycle tests.

**Step 1: Write failing tests**

Add to `tests/test_plugin.py`:

```python
class TestLLMExecutorLifecycle:
    def test_plugin_constructs_executor(self, plugin_env) -> None:
        from llm.executor import LLMExecutor
        plugin, _irc, _msg = plugin_env
        assert isinstance(plugin._llm_executor, LLMExecutor)
        assert plugin._llm_executor.max_concurrency == 16

    def test_die_shuts_down_executor(self, plugin_env) -> None:
        plugin, _irc, _msg = plugin_env
        plugin.die()
        assert plugin._llm_executor.closing is True
```

**Step 2: Verify test fails**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestLLMExecutorLifecycle -v`
Expected: FAIL with `AttributeError: '... ' object has no attribute '_llm_executor'`.

**Step 3: Wire in `__init__`**

In `plugin.py`, the constructor sets `self.log` at line 433–434
**after** `self.llm_service = LLMService(self)` at line 432. The
executor uses `self.log`, so insert the executor construction
**after** `self.log = log.getPluginLogger("LLM")` and the
`addFilter`/`_apply_log_level` calls (around line 437):

```python
        # Apply configured log level to plugin and service loggers
        self._apply_log_level()

        # Global concurrency cap for all LLM I/O. See
        # docs/plans/2026-05-06-async-llm-concurrency.md
        from .executor import LLMExecutor
        self._llm_executor = LLMExecutor(
            max_concurrency=self.registryValue("maxConcurrentLLMCalls"),
            log=self.log,
        )

        self.startup_time = time.time()  # existing line continues
```

In `die()` (line 529), shutdown the executor first, drain briefly,
**then** proceed with the existing teardown:

```python
    def die(self) -> None:
        """Clean up when plugin is unloaded."""
        # Shutdown the executor before mutating shared state so workers
        # see closing=True at their commit points. Brief drain gives
        # already-running workers a chance to flush final
        # db.log_usage / queueMsg writes before we close the DB.
        if hasattr(self, "_llm_executor"):
            self._llm_executor.shutdown()
            self._llm_executor.drain(timeout=2.0)

        # ... existing body unchanged below ...
        if hasattr(self, "db"):
            self.db.delete_expired_reminders()
            self.db.close()
```

Note: `drain()` uses `concurrent.futures.wait` on captured futures —
NOT a second call to `_executor.shutdown(timeout=…)`, which is not a
real signature in CPython 3.12–3.14.

**Step 4: Verify**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestLLMExecutorLifecycle -v`
Expected: PASS.
Run: `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): wire LLMExecutor into plugin lifecycle with drain on die"
```

---

## Task 4: Add `_rate_buckets_lock` around all rate-limit dict accesses

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — lock declaration ~458;
  `_is_rate_limited` (line 2170); `_record_rate_limit_hit` (2208);
  `_check_rate_limit` (2223) — note the read at line 2276 inside this
  method also touches `_rate_buckets` and must be under the lock.
- Modify: `plugins/llm/tests/test_plugin.py` — add concurrency test.

**Step 1: Write failing test**

Real signatures (from `plugin.py:2170, 2208`):
- `_is_rate_limited(self, command: str, account: str, now: float, *, tier: str) -> bool`
- `_record_rate_limit_hit(self, command: str, account: str, now: float) -> None`

```python
class TestRateBucketsConcurrency:
    def test_concurrent_rate_limit_check_does_not_raise(self, plugin_env) -> None:
        import threading
        import time
        plugin, _irc, _msg = plugin_env
        errors: list[Exception] = []

        def hammer() -> None:
            try:
                now = time.time()
                for _ in range(200):
                    plugin._record_rate_limit_hit("ask", "alice", now)
                    plugin._is_rate_limited("ask", "alice", now, tier="registered")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
```

**Step 2: Verify failure (or flake)**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestRateBucketsConcurrency -v` (rerun several times — TOCTOU is probabilistic).

**Step 3: Add lock and use it**

In `__init__` next to `self._rate_buckets` (line 458):

```python
        self._rate_buckets: dict[str, collections.deque[float]] = {}
        self._rate_buckets_lock = threading.Lock()
```

Wrap every read/mutate of `_rate_buckets` with
`with self._rate_buckets_lock:`. The three sites to cover:

- `_is_rate_limited` body (lines 2192–2206) — wrap from
  `key = f"{command}:{account}"` to the final return.
- `_record_rate_limit_hit` body (lines 2216–2221) — wrap entire body.
- `_check_rate_limit` line 2276 read
  `count = len(self._rate_buckets.get(key, ()))` — short critical
  section, just wrap the one line.

**Step 4: Verify**

Run the new test reliably. `make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "fix(llm): add _rate_buckets_lock for concurrent rate-limit access"
```

---

## Task 5: Add `_spontaneous_events_lock`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — lock at ~467, mutations
  at lines 559 (die clear), 919 (worker discard), 922 (main add).

**Step 1: Add lock**

In `__init__` next to `self._spontaneous_events` (line 467):

```python
        self._spontaneous_events: set[str] = set()
        self._spontaneous_events_lock = threading.Lock()
```

**Step 2: Wrap mutations**

Every site that touches `self._spontaneous_events`:

- Line 559 (die):
  ```python
          if hasattr(self, "_spontaneous_events"):
              with self._spontaneous_events_lock:
                  events = list(self._spontaneous_events)
                  self._spontaneous_events.clear()
              for event_name in events:
                  ...
  ```
  (Iterate outside the lock to avoid holding it across
  `schedule.removeEvent` calls.)
- Line 919 (worker `_evaluate` finally):
  ```python
              with self._spontaneous_events_lock:
                  self._spontaneous_events.discard(event_name)
  ```
- Line 922 (main thread, add at schedule time):
  ```python
          with self._spontaneous_events_lock:
              self._spontaneous_events.add(event_name)
  ```

**Step 3: Run tests**

`make lint && make typecheck && make test`.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "fix(llm): add _spontaneous_events_lock for worker-thread safety"
```

---

## Task 6: Add `_safe_queue` helper for thread-safe `irc.queueMsg`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `__init__` lock; new
  helper near `_send_long_reply` (~1797).
- Modify: `plugins/llm/tests/test_plugin.py`.

**Step 1: Write failing test**

```python
class TestSafeQueue:
    def test_safe_queue_drops_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor.shutdown()
        target_irc = mocker.MagicMock()
        plugin._safe_queue(target_irc, mocker.sentinel.msg)
        target_irc.queueMsg.assert_not_called()

    def test_safe_queue_calls_queuemsg(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        target_irc = mocker.MagicMock()
        plugin._safe_queue(target_irc, mocker.sentinel.msg)
        target_irc.queueMsg.assert_called_once_with(mocker.sentinel.msg)
```

**Step 2: Verify failure**

Run: `cd plugins/llm && uv run pytest tests/test_plugin.py::TestSafeQueue -v`
Expected: FAIL — `_safe_queue` AttributeError.

**Step 3: Implement**

In `__init__`:

```python
        self._irc_send_lock = threading.Lock()
```

Add method on `LLM` (near other send helpers):

```python
    def _safe_queue(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Thread-safe wrapper around irc.queueMsg for worker-thread sends.

        Limnoria's IrcMsgQueue.enqueue (irclib.py:245) mutates internal
        state without an explicit lock and the dedup check is TOCTOU.
        With the new executor pool, multiple workers may call queueMsg
        concurrently — serialize them on the plugin side and short-circuit
        cleanly when the plugin is shutting down.
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
- Modify: `plugins/llm/src/llm/plugin.py` — `_ask_impl` (around line
  2790), `code` (2909), `draw` (2993), `remind` (3803).
- Modify: `plugins/llm/tests/test_commands.py`.

**Note on `remind`:** the call at line 3803 wraps a parser
(`parse_reminder`), not `assistant_request`. The action-fire LLM call
(the one inside watch-mode reminders) lives in
`_make_reminder_delivery_closure` and is migrated by Task 10. Wrap
this site with `permit` anyway — even the parser counts toward the
global cap.

**Step 1: Write failing test**

In `tests/test_commands.py`:

```python
class TestCommandPathPermit:
    def test_ask_acquires_permit(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        permit_cm = mocker.MagicMock()
        permit_cm.__enter__ = mocker.MagicMock(return_value=None)
        permit_cm.__exit__ = mocker.MagicMock(return_value=None)
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.permit.return_value = permit_cm
        # Stub assistant_request to a benign result so we hit permit and return.
        plugin.llm_service.assistant_request = mocker.MagicMock(
            return_value=mocker.MagicMock(content="ok", error=None)
        )
        # Invoke _ask_impl through the test fixture's normal path.
        # (Adapt to the fixture shape used by other command tests in this file.)
        ...
        plugin._llm_executor.permit.assert_called()
```

**Step 2: Verify failure**

Run the new test. Expected: FAIL.

**Step 3: Wrap the LLM call sites**

For each of `_ask_impl` (~2790), `code` (~2909), `draw` (~2993),
`remind` (~3803): change

```python
with self._allow_concurrent():
    result = self.llm_service.assistant_request(...)
```

to

```python
with self._allow_concurrent(), self._llm_executor.permit():
    result = self.llm_service.assistant_request(...)
```

`permit` is the **inner** context manager. Python enters
`_allow_concurrent` first (releasing Limnoria's plugin RLock), then
`permit` (which may block on a saturated semaphore). If the order
were reversed, blocking on the semaphore would still hold the plugin
RLock and serialize every other command behind it.

For `remind` (3803), the existing line is
`with self._trace_request("remind", caller.key, channel), self._allow_concurrent():` —
extend it to
`with self._trace_request("remind", caller.key, channel), self._allow_concurrent(), self._llm_executor.permit():`.

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
- Modify: `plugins/llm/src/llm/plugin.py` — `_schedule_spontaneous`
  (lines 848–923).
- Modify: `plugins/llm/tests/test_plugin.py`.

**Step 1: Write failing tests**

```python
class TestSpontaneousMigration:
    def test_addEvent_callback_is_shim_not_evaluate(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False

        plugin._schedule_spontaneous(irc, "#chan", "alice", "hi")

        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_called_once()
        label = plugin._llm_executor.submit.call_args[0][0]
        assert label.startswith("spontaneous:")

    def test_shim_short_circuits_when_closing(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True

        plugin._schedule_spontaneous(irc, "#chan", "alice", "hi")
        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_not_called()
```

**Step 2: Verify failure**

Run new tests. Expected: FAIL — `addEvent` currently passes `_evaluate`.

**Step 3: Convert to shim**

The current code at the bottom of `_schedule_spontaneous` is:

```python
event_name = f"llm_spontaneous_{uuid.uuid4().hex[:8]}"
self._spontaneous_events.add(event_name)
schedule.addEvent(_evaluate, time.time() + 0.5, name=event_name)
```

Change to (preserving the order so `event_name` is bound before
`_enqueue` is *defined*, not just before it runs):

```python
event_name = f"llm_spontaneous_{uuid.uuid4().hex[:8]}"
with self._spontaneous_events_lock:  # from Task 5
    self._spontaneous_events.add(event_name)

def _enqueue() -> None:
    if self._llm_executor.closing:
        return
    self._llm_executor.submit(f"spontaneous:{channel}", _evaluate)

schedule.addEvent(_enqueue, time.time() + 0.5, name=event_name)
```

Inside `_evaluate`, audit and replace worker-thread writes:

- Line 893 (`irc.queueMsg(ircmsgs.action(...))`) → `self._safe_queue(irc, ircmsgs.action(...))`.
- Line 898–900 (`irc.queueMsg(ircmsgs.privmsg(...))`) → `self._safe_queue(irc, ircmsgs.privmsg(...))`.
- Line 902–912 (`self.db.log_usage(...)`) — guard with closing
  check: `if self._llm_executor.closing: return` after the PASS
  check, so a shutting-down plugin doesn't write a usage row that
  the replacement plugin will not see.
- Line 914–915 (`self._schedule_memory_extraction(...)`) — same
  closing check before scheduling. The memory-extraction shim itself
  also short-circuits on closing (Task 9), but stopping at the
  source avoids the round-trip.

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
- Modify: `plugins/llm/src/llm/plugin.py` — `_schedule_memory_extraction`
  closure body and `addEvent` at line 2570.

**Step 1: Write failing test**

```python
class TestMemoryExtractionMigration:
    def test_extraction_submitted_to_executor(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")
        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_called_once()
        assert plugin._llm_executor.submit.call_args[0][0].startswith("memory_extract:")

    def test_extraction_short_circuits_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True
        addEvent = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")
        callback = addEvent.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_not_called()
```

**Step 2: Verify failure**

Run. Expected: FAIL.

**Step 3: Convert to shim**

At line 2570, replace:

```python
event_name = f"llm_memory_{uuid.uuid4().hex[:8]}"
schedule.addEvent(_extract_memories_bg, time.time() + 0.1, name=event_name)
```

with:

```python
event_name = f"llm_memory_{uuid.uuid4().hex[:8]}"

def _enqueue() -> None:
    if self._llm_executor.closing:
        return
    self._llm_executor.submit(f"memory_extract:{nick}", _extract_memories_bg)

schedule.addEvent(_enqueue, time.time() + 0.1, name=event_name)
```

The inline `self._run_memory_cleanup(nick, channel)` call inside
`_extract_memories_bg` (line 2564) stays — runs on the same worker,
no recursive submit. (`_run_memory_cleanup` is at plugin.py:2590, NOT
service.py:4114 as the v1 design erroneously cited.)

Audit the closure body for worker-thread writes — the closure does
`self.db.add_memory_candidate`, `self.db.save_memory`, etc., which
are DB-only and use the thread-local connection. No `irc.queueMsg`
in this path.

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
- Modify: `plugins/llm/src/llm/plugin.py` —
  `_make_reminder_delivery_closure` at line 1065; the inner `_deliver`
  body 1141–1310; the `_send` closure at 1111; mechanical reschedule
  at 1273; finally at 1307–1310.

**Step 1: Write failing tests**

```python
class TestWatchModeReminderMigration:
    def test_dispatch_runs_on_worker_via_submit(self, plugin_env, mocker) -> None:
        """Action-prompt reminder _deliver submits to executor."""
        ...

    def test_chain_reschedules_on_inner_exception(self, plugin_env, mocker) -> None:
        """NEW BEHAVIOR — assistant_request raises ⇒ _mechanical_reschedule still called."""
        ...

    def test_active_irc_resolved_on_main_thread(self, plugin_env, mocker) -> None:
        """Worker does NOT iterate world.ircs."""
        ...

    def test_rate_limit_skip_does_not_submit(self, plugin_env, mocker) -> None:
        """Precheck blocks ⇒ executor.submit NOT called and skip notice queued."""
        ...

    def test_finally_short_circuits_on_closing(self, plugin_env, mocker) -> None:
        """closing=True at finally entry ⇒ no reschedule, no _reminders mutation, no DB delete."""
        ...

    def test_skip_notice_preserves_nick_prefix_and_collapse(
        self, plugin_env, mocker
    ) -> None:
        """Today's _send prepends 'nick: ' and runs _collapse_for_irc — preserve both."""
        ...
```

(Use the existing `make_reminder_row` fixture from
`conftest.py:379+` and look at existing reminder tests for the
build-then-invoke pattern.)

**Step 2: Verify failures**

Run new tests. Expected: FAIL.

**Step 3: Refactor**

Read the entire closure body 1065–1310 first.

a) **Extract a shared sender helper** so both the main-thread skip
   notice and the worker reply use the same nick-prefix + collapse:

   ```python
   def _send_reminder_text(
       self,
       irc: callbacks.Irc,
       target: str,
       nick: str,
       text: str,
   ) -> None:
       """Match the inner _send closure semantics for reminder output."""
       prefixed = f"{nick}: {text}" if nick else text
       collapsed = self._collapse_for_irc(prefixed) or prefixed
       self._safe_queue(irc, ircmsgs.privmsg(target, collapsed))
   ```

b) **Main-thread shim** (replaces today's body of `_deliver`):
   - Resolve `active_irc = next((c for c in world.ircs), None)` —
     no more `for/break` pattern.
   - If `active_irc is None`, log and return.
   - Run rate-limit precheck via
     `self._check_rate_limit(active_irc, "ask", rl_account, ..., silent=True)`.
   - If blocked, call `self._send_reminder_text(active_irc, target, nick,
     "(action skipped — daily ask limit reached)")` and return.
   - `self._llm_executor.submit(f"reminder:{event_name}",
     self._fire_reminder_action, active_irc, captured_args...)`.

c) **Worker `_fire_reminder_action(active_irc, ...)`**:

   ```python
   def _fire_reminder_action(self, active_irc, ..., is_structured, event_name, ...) -> None:
       try:
           # today's lines 1141–1306: history gather, system prompt,
           # assistant_request, response dispatch via _send_reminder_text.
           # No iteration of world.ircs; use captured active_irc.
           ...
       finally:
           # Capture closing ONCE so a flip mid-finally does not silently drop
           # half the cleanup.
           closing = self._llm_executor.closing
           if not closing and is_structured:
               self._mechanical_reschedule(...)
           if not closing:
               with self._reminders_lock:
                   self._reminders.pop(event_name, None)
               self.db.delete_reminder(event_name)
   ```

d) **Behavior change (deliberate):** today an exception in the inner
   try (LLM timeout, transient provider error) skips
   `_mechanical_reschedule` and the watch chain dies after one bad
   fire. After this change the chain survives a single failed fire.
   The regression test `test_chain_reschedules_on_inner_exception`
   asserts the new behavior.

e) **`irc.queueMsg` audit**: every queueMsg site reachable from the
   worker uses `_safe_queue` (or `_send_reminder_text` which itself
   uses `_safe_queue`). The fallback-error queueMsg path inside the
   `except Exception` at line 1294 (`_send` for the fallback message)
   also routes through the helper.

**Step 4: Verify**

Run reminder tests:
`cd plugins/llm && uv run pytest tests/ -k "reminder" -v`.
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
- Modify: `plugins/llm/src/llm/service.py` —
  `_make_scheduled_llm_task_callback` (4526–4559),
  `_dispatch_scheduled_task` (4561+),
  reschedule helper `_maybe_reschedule_or_clean` (~4557 caller).

**Worker-reachable `irc.queueMsg` / `db.log_usage` sites in this
path** (must move to `_safe_queue` / closing-gate):
- `service.py:4594` — auto-cancel on capability revoke.
- `service.py:4620` — rate-limit skip notice.
- `service.py:4699` — main reply.
- `service.py:4680–4694` — `plugin.db.log_usage` (gate with `closing`).

**Step 1: Write failing tests**

In `tests/test_service.py`:

```python
class TestScheduledLLMTaskMigration:
    def test_fire_submits_to_executor(self, ...) -> None:
        ...

    def test_reschedule_runs_after_dispatch_even_on_exception(self, ...) -> None:
        ...

    def test_closing_short_circuits(self, ...) -> None:
        ...
```

**Step 2: Verify failure.**

**Step 3: Refactor `fire()`**

Today's `fire()` (4536–4559) does:

```python
def fire() -> None:
    row = db.get_scheduled_llm_task(event_name)
    if row is None:
        return
    irc = ...
    msg = row.rehydrate_msg()
    msg.tag("llm_schedule_depth", 1)
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
            closing = self.plugin._llm_executor.closing
            if not closing:
                self._maybe_reschedule_or_clean(row, db)

    self.plugin._llm_executor.submit(
        f"scheduled_task:{event_name}", _worker
    )
```

In `_dispatch_scheduled_task`, replace the three `irc.queueMsg`
sites (4594, 4620, 4699) with `self.plugin._safe_queue(irc, ...)`
and gate the `db.log_usage` block (4680–4694) with
`if not self.plugin._llm_executor.closing:`.

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
- Modify: `plugins/llm/src/llm/plugin.py` — `_check_pending_tasks`
  (620–652); `__init__` for the `Event`; both schedule registrations
  at 504 (periodic) and 613 (event wakeup); `_deliver_pending_result`
  worker-reachable queueMsg sites at 720–729; `_schedule_queue_wakeup`
  call from worker at 649–650; delivery-retry schedule at 759–760.

**Step 1: Write failing tests**

```python
class TestSafetyPollGuard:
    def test_overlapping_poll_is_skipped(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._safety_poll_inflight.set()

        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()

    def test_flag_clears_after_worker_completes(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        # Use a real LLMExecutor so add_done_callback fires.
        plugin._enqueue_safety_poll()
        # Wait briefly for the future to complete.
        time.sleep(0.5)
        assert not plugin._safety_poll_inflight.is_set()

    def test_flag_clears_on_synchronous_submit_failure(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._llm_executor.submit.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            plugin._enqueue_safety_poll()
        assert not plugin._safety_poll_inflight.is_set()
```

**Step 2: Verify failure.**

**Step 3: Add Event and shim**

In `__init__`:

```python
        self._safety_poll_inflight = threading.Event()
```

Add the shim:

```python
    def _enqueue_safety_poll(self) -> None:
        if self._llm_executor.closing:
            return
        if self._safety_poll_inflight.is_set():
            return
        self._safety_poll_inflight.set()
        try:
            fut = self._llm_executor.submit("safety_poll", self._check_pending_tasks)
        except Exception:
            # Synchronous submit failure must not leave the flag stuck set.
            self._safety_poll_inflight.clear()
            raise
        fut.add_done_callback(lambda _f: self._safety_poll_inflight.clear())
```

Replace both schedule registrations to use the shim:
- Line 504: `self._enqueue_safety_poll` instead of `self._check_pending_tasks`.
- Line 613: same.

In `_check_pending_tasks` body (now running on a worker):
- Line 649–650 (`self._schedule_queue_wakeup()`): gate with
  `if not self._llm_executor.closing:`.
- Worker calls into `_deliver_pending_result` (lines 720, 723, 728)
  use `self._safe_queue(irc_conn, ...)` instead of
  `irc_conn.queueMsg(...)`.
- Delivery retry that does `schedule.addEvent` from worker (line
  759–760): gate with `if not self._llm_executor.closing:`.

**Step 4: Verify.**

`make lint && make typecheck && make test`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): submit safety-poll via LLMExecutor with in-flight guard"
```

---

## Task 13: Surface executor stats in `usage` command

The plan originally targeted a "status" command; there is no such
command. The closest user-facing diagnostic surface is `usage`
(plugin.py:3183). Add an executor line to the global `usage` output
(reachable via `%usage` with no args).

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `_usage_global` (3218).
- Modify: `plugins/llm/tests/test_commands.py` (or test_plugin.py if
  that's where usage tests live — `grep -n "def test.*usage"
  plugins/llm/tests/`).

**Step 1: Write failing test**

```python
class TestUsageExecutorField:
    def test_global_usage_includes_executor(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        # Invoke usage with no args via the existing test harness.
        ...
        # Assert one of the reply messages contains "executor: 0/0/16"
        # (running/queued/max).
```

**Step 2: Verify failure.**

**Step 3: Append the field**

In `_usage_global`, after the existing parts list, add:

```python
        ex = self._llm_executor
        parts.append(
            f"executor: {ex.running()}/{ex.queued()}/{ex.max_concurrency}"
        )
```

**Step 4: Verify.**

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat(llm): surface executor running/queued/max in %usage output"
```

---

## Task 14: Stress + reload regression tests

**Files:**
- Create: `plugins/llm/tests/test_executor_stress.py`.

**Step 1: Write the suite**

```python
"""Stress and regression tests for the LLM executor migration."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import wait

import pytest
from llm.executor import LLMExecutor, RecursiveSubmitError


def test_cap_honored_under_burst() -> None:
    ex = LLMExecutor(max_concurrency=4, log=logging.getLogger("test"))
    try:
        max_seen = [0]
        seen_lock = threading.Lock()  # SHARED, not per-task — bug from v1 plan.
        release = threading.Event()

        def task() -> None:
            with seen_lock:
                max_seen[0] = max(max_seen[0], ex.running())
            release.wait(timeout=5)

        futs = [ex.submit(f"t{i}", task) for i in range(ex.max_concurrency * 3)]
        # Give pool time to fill.
        time.sleep(0.1)
        assert max_seen[0] <= ex.max_concurrency
        release.set()
        done, not_done = wait(futs, timeout=5)
        assert not not_done, "all tasks must finish"
        assert all(f.exception() is None for f in done)
    finally:
        ex.shutdown()


def test_safe_queue_serializes_under_load(plugin_env, mocker) -> None:
    plugin, _irc, _msg = plugin_env
    target = mocker.MagicMock()

    def call() -> None:
        plugin._safe_queue(target, mocker.MagicMock())

    threads = [threading.Thread(target=call) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert target.queueMsg.call_count == 50


def test_recursive_submit_from_worker_raises() -> None:
    ex = LLMExecutor(max_concurrency=2, log=logging.getLogger("test"))
    try:
        captured: list[Exception] = []

        def inner() -> None:
            try:
                ex.submit("inner", lambda: None)
            except RecursiveSubmitError as e:
                captured.append(e)

        ex.submit("outer", inner).result(timeout=2)
        assert captured
    finally:
        ex.shutdown()


def test_reload_drops_post_completion_writes(plugin_env, mocker) -> None:
    """Slow worker is in-flight; die() called; worker tail-end writes short-circuit."""
    plugin, irc, _msg = plugin_env
    started = threading.Event()
    release = threading.Event()
    db_writes: list[object] = []

    plugin.db.log_usage = mocker.MagicMock(side_effect=lambda *a, **kw: db_writes.append(a))

    def slow_task() -> None:
        started.set()
        release.wait(timeout=5)
        # Simulated tail-end behaviour gated by closing.
        if plugin._llm_executor.closing:
            return
        plugin._safe_queue(irc, mocker.sentinel.msg)
        plugin.db.log_usage("alice", "#chan", "ask", "model", 1, 1, 0.0)

    plugin._llm_executor.submit("slow", slow_task)
    assert started.wait(timeout=2)

    plugin.die()
    release.set()
    time.sleep(0.5)

    irc.queueMsg.assert_not_called()
    assert not db_writes, "post-die() writes must be suppressed by closing gate"
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
- Modify: `docs/guide/` — find the operator-facing tuning page (run
  `rg -l "registryValue|tuning|maxPromptLength" docs/guide/`).
- Modify: `README.md` if there's a tunables section.

**Step 1: Locate**

Run: `rg -n "registryValue|maxConcurrent|tuning" docs/guide/ README.md`

**Step 2: Add operator-facing notes**

Document:
- `maxConcurrentLLMCalls`: what it bounds and when to raise/lower.
- The watch-mode reschedule behavior change ("a single failed fire
  no longer kills the chain").
- The new `executor: running/queued/max` field in `%usage` output.

**Step 3: Verify**

If the Makefile has a `docs` target (`grep '^docs:' Makefile`), run
`make docs`. If not, just check the page renders by reading the
modified file.

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
skip hooks).

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
3. `closing` gate is checked before any worker-thread
   `irc.queueMsg`, `db.log_usage`, `schedule.addEvent` reschedule, or
   `_reminders` mutation.
4. Watch-mode reschedule runs in the worker `finally` regardless of
   inner-try outcome (intentional behavior change). `closing` is
   captured into a local once at finally entry.
5. `_rate_buckets` access is locked at all three sites
   (`_is_rate_limited`, `_record_rate_limit_hit`, `_check_rate_limit`
   line 2276).
6. `_spontaneous_events` access is locked at all three sites (die,
   worker discard, main add).
7. `irc.queueMsg` from any worker goes through `_safe_queue`. Sites
   to migrate are enumerated per task: plugin.py 893, 898, 1117 (via
   `_send_reminder_text`), 720, 723, 728; service.py 4594, 4620, 4699.
