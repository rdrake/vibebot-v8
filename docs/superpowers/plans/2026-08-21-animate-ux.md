# @animate progress and delivery UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hold an IRCv3 typing indicator for the whole video render, and give the delivered clip line back the nick and prompt it lost, without ever truncating the URL.

**Architecture:** The typing state is *derived*, not tracked. One plugin-owned daemon thread asks `pending_tasks` every four seconds which animate rows are still `pending` (rendering), sends `+typing=active` to those targets, and sends one `+typing=done` to any target that dropped out since the last pass. Nothing increments or decrements, so at-least-once delivery, `delivery_failed`, restarts, and `@reload` all stop being cases to handle. The delivery line is rebuilt with a byte budget reserved for the URL first.

**Tech Stack:** Python 3.14, Limnoria (supybot), sqlite3, pytest, uv, ruff, ty.

**Design doc:** `docs/plans/2026-08-21-animate-ux.md` — read it first, especially the "Red-team findings" table, which records what was rejected and why.

## Global Constraints

- Run everything through `uv run` (`uv run pytest`, `uv run ruff`, `uv run ty`). Never bare `python3`.
- A pre-commit hook runs `ruff format`, `ruff check`, gitleaks, and `ty`. Let it run; do not `--no-verify`.
- CHANGELOG.md is git-cliff generated: commit, then `git cliff -o CHANGELOG.md`, then `git add CHANGELOG.md && git commit --amend --no-edit`. Never `git commit -C ORIG_HEAD`.
- **Do not push.** Pushing to main auto-deploys to production. Leave commits local; the human decides when to deploy.
- supybot's logger drops `%d`/`%s` args on some paths. For any log line carrying a number, use an f-string, not `%`-args.
- Never call `claim_due_pending_tasks` from the typing path. It opens `BEGIN IMMEDIATE` and leases the rows it returns (`persistence.py:1102`), which would steal work from the pending-task poller.
- Full suite must stay green: `uv run pytest plugins/llm/tests -q`. Baseline at time of writing: 3292 passed.

## File Structure

| File | Responsibility in this change |
| --- | --- |
| `plugins/llm/src/llm/persistence.py` | New read-only reader: which reply targets have a clip still rendering |
| `plugins/llm/src/llm/plugin.py` | The refresher pass, its thread lifecycle, `die()` cleanup, and the delivery line formatter |
| `plugins/llm/src/llm/service.py` | `_begin_typing` gains an optional `suppress_done_if` predicate |
| `plugins/llm/tests/test_animate.py` | Tests for the refresher, the predicate, and the delivery line |
| `plugins/llm/tests/test_persistence.py` | Tests for the new reader |

---

### Task 1: Read-only reader for rendering clips

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (add a method after `load_pending_tasks`, ~line 1362)
- Test: `plugins/llm/tests/test_persistence.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `LLMDatabase.active_animate_targets(now: float, max_age_seconds: float) -> list[str]` — distinct `reply_target` values for animate rows whose `delivery_state` is `'pending'` and whose `submitted_at` is newer than `now - max_age_seconds`. Task 2 calls this.

- [ ] **Step 1: Write the failing tests**

Append to `plugins/llm/tests/test_persistence.py` (the `test_db` fixture is in `conftest.py:211`):

```python
class TestActiveAnimateTargets:
    """Which targets have a clip still rendering, for the typing refresher."""

    def _animate_row(self, test_db, target: str, submitted_at: float, nick: str = "alice") -> int:
        return test_db.save_pending_task(
            task_type="animate",
            nick=nick,
            reply_target=target,
            is_channel=True,
            prompt_preview="a corgi riding a unicorn",
            model="wan",
            request_data='{"job_id": "video_gen_abc"}',
            submitted_at=submitted_at,
            expires_at=submitted_at + 1800,
            next_attempt_at=submitted_at + 10,
        )

    def test_rendering_clip_is_listed(self, test_db) -> None:
        now = time.time()
        self._animate_row(test_db, "#chan", now)
        assert test_db.active_animate_targets(now, 360) == ["#chan"]

    def test_ready_clip_is_excluded(self, test_db) -> None:
        """'ready' means the clip exists and is awaiting delivery — stop typing."""
        now = time.time()
        task_id = self._animate_row(test_db, "#chan", now)
        test_db.update_delivery_attempt(
            task_id=task_id,
            delivery_state="ready",
            last_delivery_error="",
            delivery_attempt_count=0,
            next_attempt_at=now,
        )
        assert test_db.active_animate_targets(now, 360) == []

    def test_delivery_failed_is_excluded(self, test_db) -> None:
        """The fourth end state: ten failed sends must not leave the bot typing."""
        now = time.time()
        task_id = self._animate_row(test_db, "#chan", now)
        test_db.update_delivery_attempt(
            task_id=task_id,
            delivery_state="delivery_failed",
            last_delivery_error="IRC delivery failed",
            delivery_attempt_count=10,
            next_attempt_at=now,
        )
        assert test_db.active_animate_targets(now, 360) == []

    def test_stale_row_is_excluded(self, test_db) -> None:
        """A job may stay pending for 1800s; nobody watches the bot type that long."""
        now = time.time()
        self._animate_row(test_db, "#chan", now - 400)
        assert test_db.active_animate_targets(now, 360) == []

    def test_other_task_types_are_excluded(self, test_db) -> None:
        now = time.time()
        test_db.save_pending_task(
            task_type="draw",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="a corgi",
            model="dall-e-3",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 60,
            next_attempt_at=now,
        )
        assert test_db.active_animate_targets(now, 360) == []

    def test_two_clips_on_one_target_collapse(self, test_db) -> None:
        """DISTINCT: one target, one typing indicator, however many jobs."""
        now = time.time()
        self._animate_row(test_db, "#chan", now, nick="alice")
        self._animate_row(test_db, "#chan", now, nick="bob")
        assert test_db.active_animate_targets(now, 360) == ["#chan"]
```

`import time` is already at the top of `test_persistence.py`; confirm before adding.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_persistence.py -q -k ActiveAnimateTargets`
Expected: FAIL — `AttributeError: 'LLMDatabase' object has no attribute 'active_animate_targets'`

- [ ] **Step 3: Write the implementation**

Add to `plugins/llm/src/llm/persistence.py`, directly after `load_pending_tasks`:

```python
    def active_animate_targets(self, now: float, max_age_seconds: float) -> list[str]:
        """Reply targets with a video still rendering.

        The typing refresher's whole state, read fresh each pass instead of
        tracked in memory: a row is ``pending`` while the box renders and
        ``ready`` once the clip is in hand, so this is exactly "clips still
        rendering". Deriving it means nothing has to be released, which is
        what makes at-least-once delivery and ``delivery_failed`` harmless
        here.

        Read-only on purpose. ``claim_due_pending_tasks`` accepts a
        ``delivery_state`` filter and looks like the right tool, but it opens
        BEGIN IMMEDIATE and leases the rows it returns — calling it from a
        four-second loop would steal work from the pending-task poller.

        Args:
            now: Current epoch seconds.
            max_age_seconds: Ignore rows submitted longer ago than this. A job
                can stay pending for ``animateExpiry`` (1800s by default);
                typing should give up long before that.

        Returns:
            Distinct reply targets, oldest submission first.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT reply_target FROM pending_tasks "
            "WHERE task_type = 'animate' AND delivery_state = 'pending' "
            "AND submitted_at > ? AND reply_target <> '' "
            "ORDER BY reply_target",
            (now - max_age_seconds,),
        ).fetchall()
        return [str(row[0]) for row in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_persistence.py -q -k ActiveAnimateTargets`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(persistence): read which reply targets have a clip rendering"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

### Task 2: The refresh pass

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — constants near `_IMAGE_USAGE_COMMAND` (~line 102), lock in `__init__` beside `_irc_send_lock` (~line 845), method beside `_animate_for_assistant` (~line 4738)
- Test: `plugins/llm/tests/test_animate.py`

**Interfaces:**
- Consumes: `LLMDatabase.active_animate_targets(now, max_age_seconds)` from Task 1.
- Produces:
  - `LLM._RENDER_TYPING_MAX_AGE: float = 360.0`
  - `LLM._RENDER_TYPING_INTERVAL: float = 4.0`
  - `LLM._render_typing_active: set[tuple[str, str]]` — `(network, target)` pairs typed on the last pass, guarded by `LLM._render_typing_lock`
  - `LLM._typing_refresh_pass() -> None` — one pass; Task 3 calls it in a loop
  - `LLM._render_typing_holds(target: str) -> bool` — True when any network holds `target`; Task 4's predicate

- [ ] **Step 1: Write the failing tests**

Append to `plugins/llm/tests/test_animate.py`:

```python
class TestTypingRefreshPass:
    """One pass of the render-typing refresher.

    State is derived from pending_tasks every pass rather than tracked, so
    there is no release to get wrong — see docs/plans/2026-08-21-animate-ux.md.
    """

    def _irc(self, mocker, network: str = "afternet", channels=("#chan",)):
        irc = mocker.MagicMock()
        irc.network = network
        irc.state.channels = dict.fromkeys(channels, mocker.MagicMock())
        return irc

    def test_active_target_is_typed(self, plugin_env, mocker) -> None:
        """GIVEN a rendering clip WHEN a pass runs THEN the target gets active."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "#chan", "active")
        assert plugin._render_typing_holds("#chan")

    def test_target_dropping_out_gets_one_done(self, plugin_env, mocker) -> None:
        """The clip landed: exactly one done, and not repeated next pass."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])

        plugin.db.active_animate_targets.return_value = ["#chan"]
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.reset_mock()

        plugin.db.active_animate_targets.return_value = []
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "#chan", "done")
        assert not plugin._render_typing_holds("#chan")

        plugin.llm_service.send_typing_indicator.reset_mock()
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.assert_not_called()

    def test_channel_the_bot_has_left_is_skipped(self, plugin_env, mocker) -> None:
        """No membership, no typing — mirrors the delivery path's resolution."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker, channels=("#other",))
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_not_called()
        assert not plugin._render_typing_holds("#chan")

    def test_pm_target_uses_the_first_connection(self, plugin_env, mocker) -> None:
        """A PM has no channel membership to check."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker, channels=())
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["alice"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "alice", "active")

    def test_holds_are_keyed_by_network(self, plugin_env, mocker) -> None:
        """The key is (network, target), and resolution stops at the first hit.

        Keying on the bare target would merge #chan on one network with #chan
        on another; stopping at the first connection carrying it matches how
        the delivery path picks a connection.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        a = self._irc(mocker, network="afternet")
        b = self._irc(mocker, network="other")
        mocker.patch("llm.plugin.world.ircs", [a, b])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        # Resolution stops at the first connection carrying the target, same
        # as the delivery path, so only one network is held.
        assert plugin._render_typing_active == {("afternet", "#chan")}

    def test_db_failure_does_not_raise(self, plugin_env, mocker) -> None:
        """A read failure must not kill the refresher thread."""
        plugin, _mock_irc, _mock_msg = plugin_env
        mocker.patch("llm.plugin.world.ircs", [])
        plugin.db.active_animate_targets.side_effect = RuntimeError("db is gone")

        plugin._typing_refresh_pass()  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k TypingRefreshPass`
Expected: FAIL — `AttributeError: ... has no attribute '_typing_refresh_pass'`

- [ ] **Step 3: Add the constants**

In `plugins/llm/src/llm/plugin.py`, in the `LLM` class body beside the other class constants (search for `_STATUS_MAX_SOURCES`):

```python
    # Render-typing refresher. The interval matches _begin_typing's keepalive
    # because the constraint is the same: clients expire +typing=active after
    # roughly six seconds. The max age is a deliberate ceiling on how long the
    # bot will appear to type — a job can stay pending for animateExpiry
    # (1800s), and nobody should watch that.
    _RENDER_TYPING_INTERVAL: float = 4.0
    _RENDER_TYPING_MAX_AGE: float = 360.0
```

- [ ] **Step 4: Add the state and lock**

In `LLM.__init__`, directly after `self._irc_send_lock = threading.Lock()`:

```python
        # (network, target) pairs typed on the most recent refresher pass, so
        # the next pass can send exactly one +typing=done to whatever dropped
        # out. Written by the refresher thread, read by _begin_typing's
        # suppress predicate on worker threads — hence the lock, following
        # every other mutable map in this class.
        self._render_typing_active: set[tuple[str, str]] = set()
        self._render_typing_lock = threading.Lock()
```

- [ ] **Step 5: Write the pass**

Add to `plugins/llm/src/llm/plugin.py` directly above `_animate_for_assistant`:

```python
    def _render_typing_holds(self, target: str) -> bool:
        """True when the render refresher is typing at ``target`` right now."""
        with self._render_typing_lock:
            return any(held == target for _network, held in self._render_typing_active)

    def _typing_refresh_pass(self) -> None:
        """Send one round of +typing for every clip still rendering.

        The whole state comes from the database each pass (see
        ``active_animate_targets``), so a restart, a reload, a redelivered
        row, or a job that fails ten delivery attempts all resolve themselves
        on the next tick rather than needing a code path each.

        Resolves the connection per pass instead of capturing one: a zombie
        Irc makes queueMsg return False rather than raise, so a captured
        object would silently stop typing while looking fine. Mirrors the
        delivery path's resolution (channel membership, else first
        connection).

        Never raises. This runs on a daemon thread whose death would be
        invisible.
        """
        try:
            targets = self.db.active_animate_targets(time.time(), self._RENDER_TYPING_MAX_AGE)
        except Exception:
            self.log.exception("render typing: pending-task read failed")
            return

        active: set[tuple[str, str]] = set()
        for target in targets:
            is_channel = ircutils.isChannel(target)
            for irc_conn in world.ircs:
                if is_channel and target not in irc_conn.state.channels:
                    continue
                try:
                    self.llm_service.send_typing_indicator(irc_conn, target, "active")
                except Exception:
                    self.log.exception("render typing: active send failed")
                else:
                    active.add((irc_conn.network, target))
                break

        with self._render_typing_lock:
            stale = self._render_typing_active - active
            self._render_typing_active = active

        for network, target in stale:
            for irc_conn in world.ircs:
                if irc_conn.network != network:
                    continue
                try:
                    self.llm_service.send_typing_indicator(irc_conn, target, "done")
                except Exception:
                    self.log.exception("render typing: done send failed")
                break
```

`time`, `world`, and `ircutils` are already imported in `plugin.py`; confirm rather than re-adding.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k TypingRefreshPass`
Expected: 6 passed

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest plugins/llm/tests -q && uv run ruff check plugins/llm && uv run ty check plugins/llm/src/llm`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_animate.py
git commit -m "feat(animate): one refresher pass for clips still rendering"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

### Task 3: The refresher thread

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `__init__` (start), `die()` (~line 957, stop), `_animate_for_assistant` (~line 4738, wake)
- Test: `plugins/llm/tests/test_animate.py`

**Interfaces:**
- Consumes: `_typing_refresh_pass()` and `_RENDER_TYPING_INTERVAL` from Task 2.
- Produces:
  - `LLM._render_typing_wake: threading.Event` — set by a submission to wake the loop
  - `LLM._render_typing_stop: threading.Event` — set by `die()`
  - `LLM._render_typing_loop() -> None` — the thread body

- [ ] **Step 1: Write the failing tests**

Append to `plugins/llm/tests/test_animate.py`:

```python
class TestTypingRefresherLifecycle:
    """The thread idles instead of polling, and dies with the plugin."""

    def test_submission_wakes_the_refresher(self, plugin_env, mocker) -> None:
        """GIVEN a queued clip WHEN the tool callback runs THEN the loop wakes."""
        from llm.service import VideoResult

        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="Rendering your video.", job_id="job-1", queued=True
        )
        plugin._render_typing_wake.clear()

        plugin._animate_for_assistant(
            mock_irc, mock_msg, "a corgi", nick="alice", channel="#test", account="acct"
        )

        assert plugin._render_typing_wake.is_set()

    def test_rejected_submission_does_not_wake(self, plugin_env, mocker) -> None:
        """Nothing is rendering, so nothing should type."""
        from llm.service import VideoResult

        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="Error: video server rejected the request.", error="rejected"
        )
        plugin._render_typing_wake.clear()

        plugin._animate_for_assistant(
            mock_irc, mock_msg, "a corgi", nick="alice", channel="#test", account="acct"
        )

        assert not plugin._render_typing_wake.is_set()

    def test_loop_runs_passes_until_the_set_empties(self, plugin_env, mocker) -> None:
        """Woken → pass; empty set → back to blocking, wake flag cleared."""
        plugin, _mock_irc, _mock_msg = plugin_env
        calls = []

        def fake_pass() -> None:
            calls.append(1)
            # Two passes with work, then nothing left to type.
            plugin._render_typing_active = set() if len(calls) >= 2 else {("afternet", "#chan")}

        mocker.patch.object(plugin, "_typing_refresh_pass", side_effect=fake_pass)
        plugin._RENDER_TYPING_INTERVAL = 0.01
        plugin._render_typing_wake.set()

        plugin._render_typing_loop(max_cycles=1)

        assert len(calls) == 2
        assert not plugin._render_typing_wake.is_set()

    def test_die_stops_the_thread(self, plugin_env, mocker) -> None:
        """An orphaned daemon thread would keep typing into a dead plugin."""
        plugin, _mock_irc, _mock_msg = plugin_env
        assert plugin._render_typing_thread.is_alive()

        plugin.die()

        assert plugin._render_typing_stop.is_set()
        plugin._render_typing_thread.join(timeout=2.0)
        assert not plugin._render_typing_thread.is_alive()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k TypingRefresherLifecycle`
Expected: FAIL — `AttributeError: ... has no attribute '_render_typing_wake'`

- [ ] **Step 3: Add the events and start the thread**

In `LLM.__init__`, directly after the `_render_typing_lock` added in Task 2:

```python
        # The refresher blocks on _render_typing_wake instead of querying the
        # database every four seconds forever; a submission sets it. Set once
        # here so a restart mid-render picks the job back up from the
        # database on the first pass.
        self._render_typing_wake = threading.Event()
        self._render_typing_wake.set()
        self._render_typing_stop = threading.Event()
        self._render_typing_thread = threading.Thread(
            target=self._render_typing_loop,
            name="animate-typing-refresher",
            daemon=True,
        )
        self._render_typing_thread.start()
```

- [ ] **Step 4: Write the loop**

Add to `plugins/llm/src/llm/plugin.py` directly above `_render_typing_holds`:

```python
    def _render_typing_loop(self, max_cycles: int | None = None) -> None:
        """Refresh typing while clips render; block when none are.

        Two nested waits rather than a flat poll: the outer one parks the
        thread on ``_render_typing_wake`` so an idle bot does no database work
        at all, and the inner one paces the passes while something is
        actually rendering. A pass that ends with nothing held drops back to
        the outer wait.

        ``max_cycles`` bounds the outer loop for tests; production passes
        None and runs until ``die()``.
        """
        cycles = 0
        while not self._render_typing_stop.is_set():
            if not self._render_typing_wake.wait(timeout=1.0):
                continue
            if self._render_typing_stop.is_set():
                return
            while not self._render_typing_stop.is_set():
                self._typing_refresh_pass()
                with self._render_typing_lock:
                    still_typing = bool(self._render_typing_active)
                if not still_typing:
                    self._render_typing_wake.clear()
                    break
                self._render_typing_stop.wait(timeout=self._RENDER_TYPING_INTERVAL)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
```

- [ ] **Step 5: Wake on submission**

In `_animate_for_assistant`, replace the closing `return` statement so a queued job wakes the refresher. The method currently ends:

```python
        return _ToolCallbackResult(not bool(result.error), result.content)
```

Replace with:

```python
        if not result.error:
            # A clip is on the box now, so the refresher has something to do.
            # Only on success: a rejected submission means nothing is
            # rendering and typing would be a lie.
            self._render_typing_wake.set()
        return _ToolCallbackResult(not bool(result.error), result.content)
```

- [ ] **Step 6: Stop the thread in die()**

In `die()`, directly after the executor shutdown block (`self._llm_executor.drain(timeout=2.0)`):

```python
        # Stop the render-typing refresher. It cannot send +typing=done on the
        # way out — _safe_queue drops sends once shutdown has begun — and it
        # does not need to: clients expire the state after about six seconds.
        if hasattr(self, "_render_typing_stop"):
            self._render_typing_stop.set()
            self._render_typing_wake.set()
            self._render_typing_thread.join(timeout=2.0)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k TypingRefresherLifecycle`
Expected: 4 passed

- [ ] **Step 8: Run the full suite**

Run: `uv run pytest plugins/llm/tests -q`
Expected: all pass. If unrelated tests hang, suspect the new thread: every `plugin_env` now starts one. Confirm `die()` is reached, or that the thread is a daemon.

- [ ] **Step 9: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_animate.py
git commit -m "feat(animate): hold the typing indicator for the whole render"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

### Task 4: Suppress the planner's done

**Files:**
- Modify: `plugins/llm/src/llm/service.py:2458-2499` (`_begin_typing`)
- Modify: `plugins/llm/src/llm/plugin.py` — the two animate call sites of `_begin_typing`
- Test: `plugins/llm/tests/test_animate.py`

**Interfaces:**
- Consumes: `LLM._render_typing_holds(target)` from Task 2.
- Produces: `_begin_typing(irc, msg, *, refresh=4.0, suppress_done_if=None)` where `suppress_done_if: Callable[[str], bool] | None` receives the target and returns True to skip the final `+typing=done`.

- [ ] **Step 1: Write the failing tests**

Append to `plugins/llm/tests/test_animate.py`:

```python
class TestPlannerDoneSuppression:
    """The planner's stopper must not cancel the render's indicator.

    The planner turn ends seconds after the render refresher starts on the
    same target; without this its +typing=done shows as a visible flicker.
    """

    def _irc(self, mocker):
        irc = mocker.MagicMock()
        irc.state.capabilities_ack = {"message-tags"}
        return irc

    def test_done_is_skipped_when_the_render_holds_the_target(
        self, make_service, mocker
    ) -> None:
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"))
        send = mocker.patch.object(service, "send_typing_indicator")

        stop = service._begin_typing(irc, msg, suppress_done_if=lambda target: True)
        stop()

        assert "done" not in [call.args[2] for call in send.call_args_list]

    def test_done_is_sent_when_nothing_holds_the_target(self, make_service, mocker) -> None:
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"))
        send = mocker.patch.object(service, "send_typing_indicator")

        stop = service._begin_typing(irc, msg, suppress_done_if=lambda target: False)
        stop()

        assert "done" in [call.args[2] for call in send.call_args_list]

    def test_default_still_sends_done(self, make_service, mocker) -> None:
        """Every other caller passes no predicate and must be unaffected."""
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"))
        send = mocker.patch.object(service, "send_typing_indicator")

        service._begin_typing(irc, msg)()

        assert "done" in [call.args[2] for call in send.call_args_list]

    def test_a_failing_predicate_does_not_swallow_done(self, make_service, mocker) -> None:
        """If the check itself breaks, err toward stopping the indicator."""
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"))
        send = mocker.patch.object(service, "send_typing_indicator")

        def boom(target: str) -> bool:
            raise RuntimeError("lock is gone")

        service._begin_typing(irc, msg, suppress_done_if=boom)()

        assert "done" in [call.args[2] for call in send.call_args_list]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k PlannerDoneSuppression`
Expected: FAIL — `TypeError: _begin_typing() got an unexpected keyword argument 'suppress_done_if'`

- [ ] **Step 3: Add the parameter**

In `plugins/llm/src/llm/service.py`, change the signature of `_begin_typing`:

```python
    def _begin_typing(
        self,
        irc: Irc | None,
        msg: IrcMsg | None,
        *,
        refresh: float = 4.0,
        suppress_done_if: Callable[[str], bool] | None = None,
    ) -> Callable[[], None]:
```

Extend the docstring with:

```
        ``suppress_done_if`` is checked when the stopper runs: when it returns
        True for this target, the final ``+typing=done`` is skipped because
        somebody else is still legitimately typing there. The @animate paths
        pass it so a planner turn ending mid-render does not cancel the
        render's own indicator.
```

- [ ] **Step 4: Use it in the stopper**

Replace the `stopper` body in `_begin_typing`:

```python
        def stopper() -> None:
            stop.set()
            thread.join(timeout=1.0)
            try:
                if suppress_done_if is not None and suppress_done_if(target):
                    return
            except Exception:
                # A broken predicate must not strand the indicator on; fall
                # through and send done.
                self.log.exception("typing suppress predicate failed")
            try:
                self.send_typing_indicator(irc, target, "done")
            except Exception:
                self.log.exception("typing done send failed")
```

- [ ] **Step 5: Pass it from the animate paths**

In `plugins/llm/src/llm/plugin.py`, in the `animate` command, replace:

```python
        stop_typing = self.llm_service._begin_typing(irc, msg)
```

with:

```python
        stop_typing = self.llm_service._begin_typing(
            irc, msg, suppress_done_if=self._render_typing_holds
        )
```

Do the same at the chat path's call site in `_ask_impl` — find it with:

```bash
grep -n "_begin_typing(" plugins/llm/src/llm/plugin.py
```

Only the `animate` command and the `_ask_impl` / `_dispatch_with_verse_routing` site that can host a `generate_video` call need the predicate. Leave `draw`, `story`, and the rest untouched: passing it there would be harmless but implies a relationship that does not exist.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k PlannerDoneSuppression`
Expected: 4 passed

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest plugins/llm/tests -q && uv run ruff check plugins/llm && uv run ty check plugins/llm/src/llm`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_animate.py
git commit -m "fix(animate): keep the planner's done from cancelling the render's typing"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

### Task 5: Delivery line carries its context

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:2186-2193` (the `animate` branch of the delivery path) and a new formatter beside `_collapse_for_irc` (~line 3461)
- Test: `plugins/llm/tests/test_animate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `LLM._format_animate_delivery(nick: str, prompt: str, url: str, target: str) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `plugins/llm/tests/test_animate.py`:

```python
class TestAnimateDeliveryLine:
    """The delivered clip says who asked and for what, and never loses the URL."""

    _URL = "https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4"

    def test_line_carries_nick_prompt_and_url(self, plugin_env) -> None:
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery(
            "rdrake", "a corgi riding a unicorn", self._URL, "#chan"
        )

        assert line == f'rdrake: your video is ready! "a corgi riding a unicorn" → {self._URL}'

    def test_long_prompt_never_costs_the_url(self, plugin_env) -> None:
        """The URL is the one indispensable part; it is budgeted first."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery("rdrake", "corgi " * 200, self._URL, "#chan")

        assert line.endswith(self._URL)
        assert len(line.encode("utf-8")) <= 400

    def test_multibyte_prompt_is_budgeted_in_bytes(self, plugin_env) -> None:
        """100 characters of emoji is 400 bytes — a character count is not a budget."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery("rdrake", "\U0001f600" * 100, self._URL, "#chan")

        assert line.endswith(self._URL)
        assert len(line.encode("utf-8")) <= 400

    def test_prompt_is_dropped_when_it_cannot_fit(self, plugin_env) -> None:
        """Better a bare-but-attributed line than a truncated link."""
        plugin, _mock_irc, _mock_msg = plugin_env
        long_url = "https://paste.boxlabs.uk/img/" + ("x" * 340) + ".mp4"

        line = plugin._format_animate_delivery("rdrake", "a corgi riding a unicorn", long_url, "#chan")

        assert line == f"rdrake: your video is ready! → {long_url}"

    def test_formatting_codes_are_stripped_from_the_echo(self, plugin_env) -> None:
        """A requester does not get to colour the bot's output."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery(
            "rdrake", "\x02bold\x03,4red\x0f corgi", self._URL, "#chan"
        )

        assert "\x02" not in line and "\x03" not in line
        assert "boldred corgi" in line
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k AnimateDeliveryLine`
Expected: FAIL — `AttributeError: ... has no attribute '_format_animate_delivery'`

- [ ] **Step 3: Write the formatter**

Add to `plugins/llm/src/llm/plugin.py` beside `_collapse_for_irc`:

```python
    def _format_animate_delivery(self, nick: str, prompt: str, url: str, target: str) -> str:
        """One wire-line clip delivery, with the URL budgeted first.

        The link is the only part that must survive. ``prompt_preview`` is
        capped at 100 *characters*, which can be several hundred bytes, and
        the pending-delivery path does no length fitting — so a long prompt in
        front of the URL would let Limnoria's wire-limit truncation eat the
        link. Everything else gives way to it.

        Formatting codes are stripped from the echoed prompt: ``sanitize_output``
        deliberately keeps them, and a delivery line is not the place to let a
        requester colour the bot's output.
        """
        allowed = (
            conf.get(conf.supybot.reply.mores.length, channel=target)
            if target and ircutils.isChannel(target)
            else conf.supybot.reply.mores.length()
        ) or 400

        head = f"{nick}: your video is ready!"
        tail = f" → {url}"
        bare = f"{head}{tail}"

        clean = ircutils.stripFormatting(prompt or "").strip()
        if not clean:
            return bare

        # ' ""' is the quoting the prompt would add on top of the bare line.
        budget = allowed - len(bare.encode("utf-8")) - 3
        if budget <= 0:
            return bare

        clipped = truncate_to_word_boundary(clean, budget)
        while clipped and len(clipped.encode("utf-8")) > budget:
            clipped = clipped[:-1].rstrip()
        if not clipped:
            return bare
        return f'{head} "{clipped}"{tail}'
```

`conf`, `ircutils`, and `truncate_to_word_boundary` are already imported in `plugin.py`; confirm rather than re-adding.

- [ ] **Step 4: Use it in the delivery path**

In `plugins/llm/src/llm/plugin.py`, replace the `animate` branch of the delivery path:

```python
            elif r.task_type == "animate":
                # Bare URL, exactly as @draw answers an image request. The
                # nick and the prompt used to be spelled out here because a
                # deferred line had nothing tying it to a request; the
                # +draft/reply tag attached below carries that now, and
                # repeating it in the body just restates what the client is
                # already showing above the message.
                text = content
```

with:

```python
            elif r.task_type == "animate":
                # Nick and prompt, matching draw and code. The +draft/reply
                # tag below still threads this under the request — that is
                # additive, not a substitute: a client that does not render
                # replies would otherwise get a naked URL two minutes later
                # attached to nothing. Reverses 84dbb67 deliberately; see
                # docs/plans/2026-08-21-animate-ux.md.
                text = self._format_animate_delivery(nick, prompt_preview, content, target)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_animate.py -q -k AnimateDeliveryLine`
Expected: 5 passed

- [ ] **Step 6: Check the existing delivery tests**

Run: `uv run pytest plugins/llm/tests/test_plugin_delivery.py -q`
Expected: pass. If a test asserts the bare-URL animate delivery, it is pinning the behaviour this task deliberately changes — update the assertion and say so in the commit message.

- [ ] **Step 7: Run the full suite and lint**

Run: `uv run pytest plugins/llm/tests -q && uv run ruff check plugins/llm && uv run ty check plugins/llm/src/llm`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_animate.py
git commit -m "feat(animate): deliver the clip with its nick and prompt"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/guide/user/ai-commands.md` (the `@animate` section, ~line 123)
- Modify: `docs/plans/README.md` (move the design entry to shipped)
- Move: `docs/plans/2026-08-21-animate-ux.md` → `docs/plans/archive/`

- [ ] **Step 1: Update the user guide**

In the `@animate` section of `docs/guide/user/ai-commands.md`, after the paragraph beginning "Rendering takes a minute or two", add:

```markdown
While the clip renders the bot shows as typing, so a working render looks
different from a failed one without another line in the channel. When it is
ready the link arrives with your nick and what you asked for, threaded under
your original request.
```

- [ ] **Step 2: Check the style gate**

Run: `vale docs/guide/user/ai-commands.md`
Expected: 0 errors. Warnings and suggestions are advisory; the em-dash and passive-voice ones fire throughout this file already. Vale is local-only and not in CI.

- [ ] **Step 3: Archive the design doc**

```bash
git mv docs/plans/2026-08-21-animate-ux.md docs/plans/archive/
```

Then edit `docs/plans/README.md`: remove the "@animate progress and delivery UX" bullet from the Active list.

- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs(animate): document the render typing indicator and delivery line"
git cliff -o CHANGELOG.md && git add CHANGELOG.md && git commit --amend --no-edit
```

---

## Manual verification (after deploy)

The human deploys; do not push. Once the new revision is live:

1. `ssh -i ~/.ssh/id_rsa vibebot@rdrake.org "docker inspect vibebot --format '{{.Config.Labels}}'"` — confirm the revision.
2. Send `vibebot animate a corgi riding a unicorn` in `#afternet`.
3. Watch the client: the bot should show as typing continuously for roughly two minutes, not for four seconds.
4. Confirm the delivered line reads `rdrake: your video is ready! "…" → https://…mp4` and is still threaded under the request.
5. `docker logs vibebot --since <ts> 2>&1 | grep -E 'assistant_step|render typing'` — `tool_calls=1` on step 1, and no `render typing:` exception lines.

## Self-review notes

Checked against `docs/plans/2026-08-21-animate-ux.md`:

- Derived-from-database state → Task 1 + Task 2.
- Read-only reader, not `claim_due_pending_tasks` → Task 1, in the docstring and the Global Constraints.
- Live `irc` resolution, network-scoped keys → Task 2, `_typing_refresh_pass`.
- One batched thread, idles on an Event → Task 3.
- `die()` cleanup → Task 3, Step 6.
- `suppress_done_if` predicate → Task 4.
- Byte-budgeted delivery, formatting stripped → Task 5.
- Every spec testing bullet maps to a test: the database-state cases and exclusions (Task 1), restart recovery (Task 3's wake-set-at-init, exercised by `test_loop_runs_passes_until_the_set_empties`), overlapping submissions (Task 1's DISTINCT test plus Task 2's per-network test), one `done` per drop-out (Task 2), missing connection (Task 2), `die()` (Task 3), planner suppression (Task 4), and the delivery line including the multi-byte case (Task 5).
