# Preserve Stripped Replies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure a newly generated reply that matches a persisted duplicate cluster triggers the existing repetition retry even though that cluster is excluded from the model prompt.

**Architecture:** Separate the history pipeline into comparison preparation and prompt preparation. Denials and degraded replies are removed before collecting `prior_replies`; repeated clusters are removed afterward, so they remain retry anchors without remaining prompt exemplars.

**Tech Stack:** Python 3.12+, pytest, pytest-mock, Ruff, ty

## Global Constraints

- Keep prompt de-poisoning and retry detection as separate data-flow steps.
- Degraded replies and verse denials must not anchor either the prompt or repetition detection.
- Chat considers personal and channel history; verse considers only personal history and retains its tighter history window.
- Do not change the retry budget or best-effort response behavior.

---

## File Structure

- Modify `plugins/llm/tests/test_assistant.py`: add the persisted duplicate-cluster regression test.
- Modify `plugins/llm/src/llm/service.py`: capture clean comparison replies before removing repeated clusters from prompt history.

### Task 1: Preserve duplicate-cluster replies for retry comparison

**Files:**
- Modify: `plugins/llm/tests/test_assistant.py:2681`
- Modify: `plugins/llm/src/llm/service.py:4100-4136`

**Interfaces:**
- Consumes: `_strip_verse_denials(history)`, `_strip_degraded(history)`, `_strip_repeated_replies(history)`, `_trim_history_window(history, max_messages)`, `Role.ASSISTANT`.
- Produces: unchanged `assistant_completion(...) -> AssistantResult` behavior except that persisted duplicate clusters remain in the local `prior_replies: list[str]` comparison set.

- [ ] **Step 1: Write the failing regression test**

Add this test to `TestRepeatReplyGuard` after `test_chat_retries_and_recovers_when_reply_parrots_history`:

```python
def test_chat_retries_reply_matching_stripped_duplicate_cluster(
    self, service: LLMService, mocker: MockerFixture
) -> None:
    """GIVEN persisted duplicate replies WHEN the model repeats them THEN
    they stay out of the prompt but remain anchors for the retry guard."""
    responses = [
        self._text_response(mocker, self.COMET_A),
        self._text_response(mocker, self.FRESH),
    ]
    seen: list[list] = []

    def fake_completion(**kwargs: object) -> MagicMock:
        seen.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
        return responses[len(seen) - 1]

    mocker.patch("llm.service.litellm.completion", side_effect=fake_completion)
    mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

    result = service.assistant_completion(
        prompt="how's it going",
        nick="rdrake",
        channel="#afternet",
        db=mocker.MagicMock(),
        context=mocker.MagicMock(),
        bot_nick="vibebot",
        history=[
            {"role": "user", "content": "how's it going"},
            {"role": "assistant", "content": self.COMET_A},
            {"role": "user", "content": "same question tomorrow"},
            {"role": "assistant", "content": self.COMET_B},
        ],
    )

    assert result.content == self.FRESH
    assert result.error is None
    assert len(seen) == 2
    assert all(
        m.get("content") not in {self.COMET_A, self.COMET_B}
        for m in seen[0]
    )
    assert any(
        m.get("role") == "user"
        and str(m.get("content", "")) == _REPEAT_RETRY_NUDGE
        for m in seen[1]
    )
```

Add `_REPEAT_RETRY_NUDGE` to the existing module-level imports from `llm.service`, and remove the local import in the neighboring test.

- [ ] **Step 2: Run the test to verify it fails for the reviewed reason**

Run:

```bash
uv run pytest plugins/llm/tests/test_assistant.py::TestRepeatReplyGuard::test_chat_retries_reply_matching_stripped_duplicate_cluster -q
```

Expected: FAIL because `result.content` is `COMET_A` and only one completion call is recorded; the duplicate cluster was stripped before `prior_replies` was collected.

- [ ] **Step 3: Implement the minimal history-pipeline fix**

In `assistant_completion`, replace the single combined de-poisoning pass with these ordered phases:

```python
if route_profile == PROFILE_VERSE:
    history = _strip_verse_denials(history)
    history = _strip_degraded(history)
    channel_history = None
else:
    history = _strip_degraded(history)
    channel_history = _strip_degraded(channel_history)

prior_replies = [
    str(m.get("content", ""))
    for m in [*(history or []), *(channel_history or [])]
    if m.get("role") == Role.ASSISTANT
]

history = _strip_repeated_replies(history)
if route_profile == PROFILE_VERSE:
    history = _trim_history_window(history, _VERSE_HISTORY_MAX_MESSAGES)
else:
    channel_history = _strip_repeated_replies(channel_history)
```

Retain the existing verse channel-history rationale and update nearby comments to state that repetition anchors are captured after denial/degraded filtering but before duplicate clusters are excluded from the model prompt.

- [ ] **Step 4: Run the regression test and repeat-guard tests**

Run:

```bash
uv run pytest plugins/llm/tests/test_assistant.py::TestRepeatReplyGuard -q
```

Expected: all `TestRepeatReplyGuard` tests PASS.

- [ ] **Step 5: Run Python quality checks**

Run:

```bash
make lint
make typecheck
make preflight
```

Expected: all commands exit 0. If `make preflight` is too expensive or fails outside the changed area, record the exact command and failure in the handoff.

- [ ] **Step 6: Review and commit the implementation**

Run:

```bash
git diff --check
git diff -- plugins/llm/src/llm/service.py plugins/llm/tests/test_assistant.py
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_assistant.py docs/superpowers/plans/2026-07-18-preserve-stripped-replies.md
git commit -m "fix(assistant): retain stripped replies for retry detection"
```

Expected: the diff contains only the regression test, ordered history-pipeline change, and this plan; commit hooks pass.
