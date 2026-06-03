# Reactive Loom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip the loom from a timer-driven host (it seeds, bots react to us) into a reactive participant (a bot/user speaks → we chime in once, capped to once per interval).

**Architecture:** `observe_transcript` becomes the trigger on the IRC driver thread — cheap form-or-append under a lock. When due (≥ `cycle_interval_s` since the last chime-in), it forms a cycle recording the triggering line as `transcript[0]` and offloads the heavy work (verse pick, snapshot, chime-in LLM call) to the worker via `submit("loom:open", …)`. After one beat window, digest runs unchanged. The periodic timer, seed phase, and second beat are removed.

**Tech Stack:** Python 3.14, pytest, Limnoria plugin, litellm. Loom orchestrator in `plugins/llm/src/llm/verse/loom.py`.

**Reference spec:** `docs/superpowers/specs/2026-06-03-reactive-loom-design.md`

---

## File Structure

- `plugins/llm/src/llm/verse/loom.py` — orchestrator. Remove `tick`, `_seed_phase`, `_beat_phase`, `after_beat1`, `after_beat2`, `build_seed_tail`, `build_beat_tail`. Add `_last_chime_at` state, rewrite `observe_transcript`, add `_open_and_chime`, `after_chime`, `build_chimein_tail`. Keep `_digest_phase`, `pick_focus_verse`, `truncate_transcript`, `apply_or_queue`, `parse_digest`, crosspoll send/receive.
- `plugins/llm/src/llm/plugin.py` — remove `_schedule_loom_tick` + `_loom_tick` + the call that registers the periodic timer; update two scheduler-teardown blocks to drop `llm_loom_cycle` and rename `llm_loom_after_beat1/2` → `llm_loom_after_chime`.
- `plugins/llm/src/llm/config.py` — update help text on `loomCycleInterval` and `loomCaptureTranscript`. Defaults unchanged.
- `plugins/llm/tests/verse/test_loom.py` — replace seed/beat/tick test classes with reactive-trigger tests.
- `plugins/llm/tests/verse/test_loom_integration.py` — rewrite the end-to-end driver to the reactive flow.
- `plugins/llm/tests/verse/_fakes.py` — no change (its `submit` already runs inline and records labels; `StubClient` is keyed by op).

---

## Task 1: `build_chimein_tail` prompt builder

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py` (add near the other `build_*_tail` functions, ~line 100)
- Test: `plugins/llm/tests/verse/test_loom.py`

- [ ] **Step 1: Write the failing test**

Add to `test_loom.py` (top-level, near the other prompt-builder tests):

```python
from llm.verse.loom import build_chimein_tail


class TestBuildChimeinTail:
    def test_frames_lines_as_spontaneous_and_forbids_json(self) -> None:
        tail = build_chimein_tail(
            loom_transcript_so_far=[("botB", "the bell rings"), ("botC", "i hear it")]
        )
        assert "botB: the bell rings" in tail
        assert "botC: i hear it" in tail
        # Framed as the others speaking unprompted, not replying to us.
        assert "unprompted" in tail
        assert "replied" not in tail
        # Same guardrails as seed/beat: one line, no JSON.
        assert "Do NOT emit JSON" in tail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && python -m pytest tests/verse/test_loom.py::TestBuildChimeinTail -v`
Expected: FAIL — `ImportError: cannot import name 'build_chimein_tail'`

- [ ] **Step 3: Write minimal implementation**

In `loom.py`, add after `build_seed_tail` (it will be deleted in Task 2; placement is temporary):

```python
def build_chimein_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "You are idling in this channel, watching. The others just spoke, "
        "unprompted:\n"
        f"{lines}\n\n"
        "Chime in with a single line that picks up on what they're doing. "
        "Stay in fiction. One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && python -m pytest tests/verse/test_loom.py::TestBuildChimeinTail -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py
git commit -m "feat(verse/loom): add build_chimein_tail prompt builder"
```

---

## Task 2: Reactive orchestration in `loom.py`

This is one atomic rewrite of the orchestration: the trigger, the worker open+chime phase, and the digest hand-off all change together, and the old `tick`/seed/beat methods are removed in the same commit. Tests are written first.

**Files:**
- Modify: `plugins/llm/src/llm/verse/loom.py`
- Test: `plugins/llm/tests/verse/test_loom.py`

- [ ] **Step 1: Write the failing tests**

Add a new test class to `test_loom.py`. These use the existing `FakeBridge` / `StubClient` from `_fakes.py` and the existing `verse_db_dir` fixture + `_make_store` helper used by the current tests (mirror how the current seed/beat tests build `store` and `snapshots`; reuse the same `VerseSnapshot` construction). The `StubClient` is now keyed by `"chimein"` and `"digest"`.

```python
class TestReactiveTrigger:
    def _bridge(self, store, *, post_returns=True):
        snap = VerseSnapshot(
            channel="#forest",
            summary="a quiet grove",
            top_entities=[("place", "grove", 1)],
            recent_events=["the bell rang"],
        )
        return FakeBridge(
            channels=["#forest"],
            weights={"#forest": 5},
            store=store,
            snapshots={"#forest": snap},
            post_returns=post_returns,
        )

    def test_first_line_opens_cycle_and_posts_single_chimein(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "the bell still hums", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "the bell rings")

        # Exactly one post (the chime-in); a digest is scheduled, not posted.
        assert bridge.posts == ["the bell still hums"]
        assert bridge.scheduled[-1][2] == "llm_loom_after_chime"
        assert client.calls == ["chimein"]

    def test_chimein_transcript_includes_triggering_line(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "the bell rings")
        # The chime-in user message must contain the spontaneous first line.
        # FakeBridge does not capture messages, so assert via a capturing client.
        assert "the bell rings" in client.last_user_content

    def test_second_line_within_interval_is_ignored(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")   # opens + chimes (inline)
        bridge.scheduled[-1][1]()                   # after_chime -> digest -> _active=None
        bridge.t += 10                              # still < cycle_interval_s
        loom.observe_transcript("botC", "second")   # within interval -> ignored

        assert client.calls == ["chimein", "digest"]   # no second chime-in
        assert bridge.posts == ["ok"]

    def test_line_after_interval_opens_new_cycle(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")
        bridge.scheduled[-1][1]()                   # finalize cycle 1
        bridge.t += _make_cfg().cycle_interval_s + 1
        loom.observe_transcript("botC", "second")   # now due again

        assert bridge.posts == ["ok", "ok"]
        assert client.calls == ["chimein", "digest", "chimein"]

    def test_lines_during_active_cycle_append_not_retrigger(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")    # opens cycle, posts chimein
        loom.observe_transcript("botC", "second")   # active cycle -> append only
        # No new chime-in posted; second line waits for digest.
        assert bridge.posts == ["ok"]
        bridge.scheduled[-1][1]()                    # digest sees both lines
        assert "second" in client.last_user_content  # digest user content

    def test_no_eligible_verse_rolls_back_and_stays_due(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        bridge.channels = []          # nothing to pick
        bridge.weights = {}
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")     # forms, worker finds no verse
        assert bridge.posts == []
        assert client.calls == []
        # Still due: restoring channels and firing a new line opens a cycle.
        bridge.channels = ["#forest"]
        bridge.weights = {"#forest": 5}
        loom.observe_transcript("botC", "second")
        assert bridge.posts == ["ok"]

    def test_post_failure_rolls_back_and_stays_due(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store, post_returns=False)
        client = StubClient({"chimein": "ok", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")     # chime-in post fails
        assert bridge.scheduled == []                # no digest scheduled
        # Cooldown rolled back: a new line (post now works) opens a cycle.
        bridge.post_returns = True
        loom.observe_transcript("botC", "second")
        # FakeBridge.post_to_loom_channel appends *then* returns its status,
        # so the failed first attempt is recorded too: two appends total.
        assert bridge.posts == ["ok", "ok"]
        assert bridge.scheduled[-1][2] == "llm_loom_after_chime"  # second armed digest

    def test_chimein_call_exception_finalizes_cycle(self, verse_db_dir) -> None:
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)

        class BoomClient(StubClient):
            def call(self, *, op, model, messages):
                if op == "chimein":
                    self.calls.append(op)   # record before raising
                    raise RuntimeError("boom")
                return super().call(op=op, model=model, messages=messages)

        client = BoomClient({"chimein": "x", "digest": "[]"})
        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "first")
        assert bridge.posts == []
        assert bridge.scheduled == []
        # Rolled back -> still due.
        loom.observe_transcript("botC", "second")
        assert client.calls.count("chimein") == 2

    def test_trigger_path_does_not_snapshot_on_driver_thread(self, verse_db_dir) -> None:
        # The cheap trigger path forms the cycle and offloads; snapshot must
        # happen inside the submitted worker, not before submit() is called.
        store = _make_store(verse_db_dir)
        bridge = self._bridge(store)
        client = StubClient({"chimein": "ok", "digest": "[]"})

        order = []
        orig_snapshot = bridge.snapshot
        orig_submit = bridge.submit

        def tracking_snapshot(channel):
            order.append("snapshot")
            return orig_snapshot(channel)

        def tracking_submit(label, fn):
            order.append(f"submit:{label}")
            return orig_submit(label, fn)

        bridge.snapshot = tracking_snapshot
        bridge.submit = tracking_submit

        loom = Loom(cfg=_make_cfg(), bridge=bridge, client=client)
        loom.observe_transcript("botB", "first")

        # submit:loom:open is recorded before the first snapshot call.
        assert order[0] == "submit:loom:open"
        assert "snapshot" in order
        assert order.index("submit:loom:open") < order.index("snapshot")
```

Note: this task also requires a small `StubClient` capability — capturing the last user-message content — so `test_chimein_transcript_includes_triggering_line` and the digest-content assertions work. Add `last_user_content` to `StubClient` in `_fakes.py`:

```python
    def call(self, *, op, model, messages):
        self.calls.append(op)
        self.last_user_content = messages[-1]["content"]
        return self.replies[op], LoomCallUsage(prompt_tokens=10, completion_tokens=5, cost=0.0001)
```

Also confirm `_make_cfg()` and `_make_store(verse_db_dir)` helpers exist at the top of `test_loom.py`; if the current file builds `LoomConfig` / store inline per-test instead, add module-level helpers `_make_cfg()` (returning a `LoomConfig` with `cycle_interval_s=300`, `verse_cooldown_s=1200`, `beat_window_s=90`, `transcript_max_lines=40`, `transcript_max_chars=8000`, `auto_apply_threshold=0.8`, `bot_nicks=()`, `model="cheap"`, `network="afternet"`, `loom_channel="#forest"`) and `_make_store(verse_db_dir)` mirroring the existing per-test store construction.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && python -m pytest tests/verse/test_loom.py::TestReactiveTrigger -v`
Expected: FAIL — `_open_and_chime`/`after_chime` not defined, `observe_transcript` still only appends, `last_user_content` missing.

- [ ] **Step 3: Add `last_user_content` to `StubClient`**

Apply the `_fakes.py` edit shown above (capture `messages[-1]["content"]`).

- [ ] **Step 4: Implement the reactive orchestration**

In `loom.py`, in `Loom.__init__`, add the new field after `self._active`:

```python
        self._last_chime_at: float | None = None
```

Replace `observe_transcript` with:

```python
    def observe_transcript(self, nick: str, text: str) -> None:
        """Trigger entry point. Runs on the IRC driver thread (doPrivmsg),
        so the path stays cheap: lock, timestamp compare, list append.

        If a cycle is active, append. Otherwise, if at least
        ``cycle_interval_s`` has elapsed since the last chime-in, form a
        cycle recording this line as ``transcript[0]`` and offload the
        heavy verse-pick + snapshot + chime-in to the LLM worker.
        """
        with self._lock:
            if self._active is not None:
                self._active.append_transcript(nick, text)
                return
            now = self._bridge.now()
            if (
                self._last_chime_at is not None
                and (now - self._last_chime_at) < self._cfg.cycle_interval_s
            ):
                return
            prev_last_chime = self._last_chime_at
            cycle = LoomCycle(
                cycle_id=uuid.uuid4().hex[:12],
                channel="",
                started_at=now,
                verse_stable_block="",
                transcript=[(nick, text)],
            )
            self._active = cycle
            self._last_chime_at = now
        # Outside the lock: heavy DB work must not block the driver thread.
        self._bridge.submit(
            "loom:open", lambda: self._open_and_chime(cycle, prev_last_chime)
        )
```

Add the worker open+chime method (replaces `_seed_phase`):

```python
    def _open_and_chime(self, cycle: LoomCycle, prev_last_chime: float | None) -> None:
        now = self._bridge.now()
        channels = self._bridge.list_candidate_channels()
        candidates = [
            VerseCandidate(
                channel=c,
                weight=self._bridge.candidate_weight(c),
                last_cycle_at=self._last_cycle_by_channel.get(c),
            )
            for c in channels
        ]
        choice = pick_focus_verse(
            candidates,
            now=now,
            cooldown_s=self._cfg.verse_cooldown_s,
            pointer=self._pointer,
        )
        if choice is None:
            self._log.debug("loom: no eligible verse at chime-in; rolling back")
            with self._lock:
                self._active = None
                self._last_chime_at = prev_last_chime
            return
        snap = self._bridge.snapshot(choice.channel)
        with self._lock:
            self._pointer += 1
            cycle.channel = choice.channel
            cycle.verse_stable_block = build_verse_stable_block(snap)
            self._last_cycle_by_channel[choice.channel] = now
            transcript = truncate_transcript(
                cycle.snapshot_transcript(),
                max_lines=self._cfg.transcript_max_lines,
                max_chars=self._cfg.transcript_max_chars,
            )
        # Crosspoll receive (unchanged), outside the cycle lock.
        self._maybe_consume_one_seed_for(choice.channel)
        messages = [
            {"role": "system", "content": LOOM_STATIC_PREFIX},
            {"role": "system", "content": cycle.verse_stable_block},
            {"role": "user", "content": build_chimein_tail(loom_transcript_so_far=transcript)},
        ]
        try:
            content, usage = self._client.call(
                op="chimein", model=self._cfg.model, messages=messages
            )
        except Exception:
            self._log.exception("loom chime-in call failed; aborting cycle")
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(choice.channel, None)
                self._last_chime_at = prev_last_chime
            return
        self._bridge.log_usage(
            channel=choice.channel, op="chimein", model=self._cfg.model, usage=usage
        )
        line = (content.strip().splitlines() or [""])[0]
        if not line:
            with self._lock:
                self._active = None
            return
        if not self._bridge.post_to_loom_channel(line):
            self._log.warning(
                "loom chime-in: post_to_loom_channel failed (network down?); "
                "rolling back cycle for %s",
                choice.channel,
            )
            with self._lock:
                self._active = None
                self._last_cycle_by_channel.pop(choice.channel, None)
                self._last_chime_at = prev_last_chime
            return
        with self._lock:
            cycle.beats_posted = 1
        self._bridge.schedule_after(
            self._cfg.beat_window_s,
            self.after_chime,
            "llm_loom_after_chime",
        )
```

Add the digest hand-off (replaces `after_beat2`):

```python
    def after_chime(self) -> None:
        with self._lock:
            cycle = self._active
            if cycle is None:
                return
        self._bridge.submit("loom:digest", lambda: self._digest_phase(cycle))
```

Delete these now-obsolete members entirely: `build_seed_tail`, `build_beat_tail`, `Loom.tick`, `Loom._seed_phase`, `Loom.after_beat1`, `Loom._beat_phase`, `Loom.after_beat2`. Keep `_digest_phase`, `_maybe_consume_one_seed_for`, `_release_claim_with_retry` unchanged.

- [ ] **Step 5: Delete obsolete tests**

Remove from `test_loom.py` the tests that exercise the removed methods: every test calling `loom.tick()`, `loom.after_beat1()`, `loom.after_beat2()`, and any asserting `llm_loom_after_beat1`/`llm_loom_after_beat2` or `usage_log == [..., "seed", "beat", ...]`. The pure-function tests (`parse_digest`, `truncate_transcript`, `pick_focus_verse`, `apply_or_queue`) stay.

- [ ] **Step 6: Run the full loom unit suite**

Run: `cd plugins/llm && python -m pytest tests/verse/test_loom.py -v`
Expected: PASS (new `TestReactiveTrigger` + `TestBuildChimeinTail` + retained pure-function tests).

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/verse/loom.py plugins/llm/tests/verse/test_loom.py plugins/llm/tests/verse/_fakes.py
git commit -m "feat(verse/loom): reactive trigger replaces timer-driven seed/beat"
```

---

## Task 3: Remove the periodic timer from `plugin.py`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`

- [ ] **Step 1: Drop the timer registration call**

In `_wire_loom_if_enabled`, remove the final line `self._schedule_loom_tick()` (≈ line 5061). The loom now arms itself via `observe_transcript`; no periodic event is registered.

- [ ] **Step 2: Delete `_schedule_loom_tick` and `_loom_tick`**

Remove both methods (≈ lines 5063–5075):

```python
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

- [ ] **Step 3: Update teardown block #1 (≈ lines 834–840)**

Replace:

```python
        # Loom orchestrator teardown (PR 2).
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_loom_cycle")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_loom_after_beat1")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_loom_after_beat2")
```

with:

```python
        # Loom orchestrator teardown (reactive trigger; single beat window).
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_loom_after_chime")
```

- [ ] **Step 4: Update teardown block #2 (≈ lines 5006–5012)**

Replace:

```python
            if self._loom is not None:
                with contextlib.suppress(KeyError):
                    schedule.removeEvent("llm_loom_cycle")
                with contextlib.suppress(KeyError):
                    schedule.removeEvent("llm_loom_after_beat1")
                with contextlib.suppress(KeyError):
                    schedule.removeEvent("llm_loom_after_beat2")
```

with:

```python
            if self._loom is not None:
                with contextlib.suppress(KeyError):
                    schedule.removeEvent("llm_loom_after_chime")
```

- [ ] **Step 5: Verify no dangling references**

Run: `grep -rn "llm_loom_cycle\|llm_loom_after_beat\|_schedule_loom_tick\|_loom_tick\|\.tick()" plugins/llm/src`
Expected: no output.

- [ ] **Step 6: Run the plugin tests**

Run: `cd plugins/llm && python -m pytest tests/test_plugin.py -v`
Expected: PASS. The `TestDoPrivmsgLoomHook` tests are unaffected (the doPrivmsg → `observe_transcript` hook is unchanged). If any test references `_loom_tick`/`_schedule_loom_tick`, delete it (grep in Step 5 confirms none in src; re-grep tests: `grep -rn "_loom_tick\|_schedule_loom_tick" plugins/llm/tests` and remove matches).

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "refactor(plugin): retire loom periodic timer; reactive trigger only"
```

---

## Task 4: Update config help text

**Files:**
- Modify: `plugins/llm/src/llm/config.py`

- [ ] **Step 1: Update `loomCycleInterval` help (≈ line 427)**

Replace:

```python
        _("""Loom timer cadence in minutes."""),
```

with:

```python
        _("""Minimum gap in minutes between loom chime-ins. The loom is
        reactive: it waits for another user or bot to speak, then chimes
        in once, and will not chime in again until this interval has
        elapsed."""),
```

- [ ] **Step 2: Update `loomCaptureTranscript` help (≈ lines 484–490)**

Replace:

```python
        _("""When True (default), the loom captures non-self lines from
        loomChannel into its transcript and drives beat + digest calls
        from that content. When False, the loom still posts seed lines
        (for ambient flavor) but ignores all channel chatter — every
        cycle finalizes via the empty-transcript short-circuit, no
        proposals are generated. Useful when the venue is too noisy or
        too off-topic to feed the model."""),
```

with:

```python
        _("""When True (default), the loom captures non-self lines from
        loomChannel; the first such line after the cycle interval triggers
        a chime-in and the transcript drives the digest. When False, the
        loom is fully inert — it only ever acts on captured lines, so it
        neither chimes in nor generates proposals. Set False to disable
        the loom's participation without unsetting loomChannel."""),
```

- [ ] **Step 3: Run config tests**

Run: `cd plugins/llm && python -m pytest tests/test_config.py -v`
Expected: PASS (`test_config.py` asserts only the default *values* `loomCycleInterval() == 5` and `loomCaptureTranscript() is True`, not help text).

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py
git commit -m "docs(config): loom help text reflects reactive trigger"
```

---

## Task 5: Rewrite the integration test

**Files:**
- Modify: `plugins/llm/tests/verse/test_loom_integration.py`

- [ ] **Step 1: Rewrite the end-to-end driver**

The current test drives `tick → after_beat1 → after_beat2` with `StubClient` replies keyed `seed`/`beat`/`digest` and asserts `usage_log == ["seed","beat","digest"]`. Rewrite to the reactive flow.

Replace the cycle-driving section (current lines ≈ 78–109) with:

```python
    client = StubClient(
        {
            "chimein": "a chime echoes in answer",
            "digest": (
                '[{"op":"add_event",'
                '"payload":{"summary":"a chime echoes","entity_ids":[]},'
                '"confidence":0.95,"provenance":"botB","rationale":"the bell"}]'
            ),
        }
    )
    loom = Loom(cfg=cfg, bridge=bridge, client=client)

    # A bot speaks spontaneously -> loom forms a cycle, chimes in (inline via
    # FakeBridge.submit), and schedules the digest.
    loom.observe_transcript("botB", "I hear it too")
    loom.observe_transcript("botC", "the wind takes it")  # appended to cycle
    bridge.scheduled[-1][1]()  # after_chime -> digest

    events = store.recent_events(limit=10)
    assert any(e.summary == "a chime echoes" and e.source == "loom" for e in events)
    assert [u[1] for u in bridge.usage_log] == ["chimein", "digest"]
```

Update the module docstring (line 3) from `Loom.tick → after_beat1 → after_beat2` to `observe_transcript → after_chime → digest`.

- [ ] **Step 2: Run the integration test**

Run: `cd plugins/llm && python -m pytest tests/verse/test_loom_integration.py -v`
Expected: PASS — the digest applies the high-confidence `add_event` and writes a `source='loom'` event row.

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/tests/verse/test_loom_integration.py
git commit -m "test(verse/loom): integration drives reactive chime-in flow"
```

---

## Task 6: Full verification

- [ ] **Step 1: Run the whole loom-related suite**

Run: `cd plugins/llm && python -m pytest tests/verse/ tests/test_plugin.py tests/test_config.py -v`
Expected: PASS, no skips related to loom.

- [ ] **Step 2: Lint + typecheck**

Run: `make lint && make typecheck`
Expected: clean (the Edit hook also runs these, but confirm the whole tree).

- [ ] **Step 3: Final grep for stale loom vocabulary**

Run: `grep -rn "seed_phase\|beat_phase\|after_beat\|loom_cycle\|build_seed_tail\|build_beat_tail" plugins/llm/src plugins/llm/tests | grep -v "\.pyc"`
Expected: no output.

- [ ] **Step 4: Commit any cleanup**

```bash
git add -A && git commit -m "chore(verse/loom): finalize reactive loom" || echo "nothing to commit"
```

---

## Notes for the implementer

- **Cooldown clock:** `_last_chime_at` is stamped when the cycle *forms* (the inbound line arrives), not when the chime-in posts. A cycle is short (one chime + one beat window), so forming-time ≈ chime-time. This is intentional and simpler.
- **Reload behavior:** `_last_chime_at` initializes to `None`, so right after `@reload`/config change the loom is due and chimes in on the next line. This is an accepted liveness signal (per the spec's user review).
- **Rollback paths** (no eligible verse, chime-in call exception, post failure) all restore `_last_chime_at = prev_last_chime` so a failed attempt does not consume the interval. Lines appended to an aborted forming cycle are dropped — rare and acceptable, matching today's aborted-seed behavior.
- **Driver-thread discipline:** keep `snapshot`/`list_candidate_channels`/`candidate_weight` out of `observe_transcript`. They belong in `_open_and_chime`, which runs on the LLM worker. `test_trigger_path_does_not_snapshot_on_driver_thread` guards this.
- **Pointer:** `self._pointer` increments only on a successful verse pick (inside the lock in `_open_and_chime`), so a no-eligible-verse rollback does not advance round-robin — matching the old `tick`'s idle short-circuit.
