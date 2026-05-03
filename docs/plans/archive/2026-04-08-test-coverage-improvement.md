# Test Coverage Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close test coverage gaps across all LLM plugin modules, fix the unclosed database warning noise, and raise overall coverage from 91.66% toward 95%+.

**Architecture:** Work bottom-up from infrastructure (conftest fixture) through data layer (persistence, context) to business logic (service) to command layer (plugin). Each task is self-contained and independently committable.

**Tech Stack:** pytest, pytest-mock, unittest.mock, sqlite3, LLMDatabase, LLMService

---

### Task 1: Fix unclosed database warnings in test infrastructure

**Files:**
- Modify: `plugins/llm/tests/conftest.py`
- Modify: `plugins/llm/tests/test_context.py:419-423`

**Step 1: Add a `test_db` fixture to conftest.py**

Add after the `make_service` fixture (line 208):

```python
@pytest.fixture
def test_db(tmp_path: Path) -> Generator[LLMDatabase, None, None]:
    """Create a test database with automatic cleanup."""
    from llm.persistence import LLMDatabase

    db = LLMDatabase(str(tmp_path / "test.db"))
    yield db
    db.close()
```

Also add `Path` to the imports from `pathlib` and `LLMDatabase` import inside the fixture (deferred to avoid import overhead for tests that don't need it).

**Step 2: Update `TestPersistentContext._make_ctx` to accept and use a db parameter**

In `test_context.py`, change `_make_ctx` (line 419-423) to:

```python
def _make_ctx(
    self, tmp_path: Path, db: LLMDatabase | None = None
) -> tuple[ConversationContext, LLMDatabase]:
    if db is None:
        db = LLMDatabase(str(tmp_path / "test.db"))
    config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
    ctx = ConversationContext(config, db=db)
    return ctx, db
```

Then update all `TestPersistentContext` test methods that call `_make_ctx` to add `db.close()` at the end, or restructure to use a class-level fixture. A simpler approach: add a `test_db` fixture usage in the class and pass it to `_make_ctx`.

**Step 3: Update `test_persistence.py` to close databases**

Add `db.close()` calls in test methods or convert to use the `test_db` fixture.

**Step 4: Run tests and verify warnings are gone**

Run: `make test 2>&1 | grep -c "ResourceWarning"`
Expected: 0 (or drastically reduced)

**Step 5: Commit**

```bash
git add plugins/llm/tests/conftest.py plugins/llm/tests/test_context.py plugins/llm/tests/test_persistence.py
git commit -m "test: fix unclosed database warnings with proper cleanup"
```

---

### Task 2: Cover `context.py` gaps (update_config, __repr__, prune with db)

**Files:**
- Modify: `plugins/llm/tests/test_context.py`

Coverage targets: lines 107-108, 112, 144, 167-169, 176

**Step 1: Write tests for `update_config` and `__repr__`**

Add to `test_context.py`:

```python
class TestContextConfigUpdate:
    """Test runtime configuration updates."""

    def test_update_config_changes_defaults(self) -> None:
        """GIVEN context WHEN update_config called THEN new config is used."""
        config = ContextConfig(max_messages=10, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)
        new_config = ContextConfig(max_messages=50, timeout_minutes=60, enabled=True)
        ctx.update_config(new_config)
        assert ctx.config.max_messages == 50
        assert ctx.config.timeout_minutes == 60

    def test_repr_shows_state(self) -> None:
        """GIVEN context with data WHEN repr called THEN shows counts."""
        config = ContextConfig(max_messages=10, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)
        ctx.add_message("user1", "#chan", "user", "Hello")
        result = repr(ctx)
        assert "conversations=1" in result
        assert "enabled=True" in result
```

**Step 2: Write test for `_prune_expired` with database deletion (lines 167-169)**

```python
class TestPruneWithDatabase:
    """Test that expired conversations are deleted from database."""

    def test_prune_deletes_expired_from_db(self, test_db, tmp_path: Path) -> None:
        """GIVEN expired conversation in db WHEN prune runs THEN db row deleted."""
        config = ContextConfig(max_messages=10, timeout_minutes=0, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        # Add a message (will persist to db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        assert len(test_db.load_conversations()) == 1

        # Force prune (timeout_minutes=0 means everything is expired)
        import time
        time.sleep(0.01)
        ctx.get_stats()  # Triggers force-prune

        assert len(test_db.load_conversations()) == 0
```

**Step 3: Write test for `_is_expired` returning True when disabled (line 144)**

```python
    def test_is_expired_when_disabled(self) -> None:
        """GIVEN disabled context WHEN _is_expired checked THEN returns True."""
        config = ContextConfig(max_messages=10, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)
        conv = Conversation()
        assert ctx._is_expired(conv, config) is True
```

**Step 4: Run tests**

Run: `make test -- -k test_context -v`
Expected: All new tests PASS

**Step 5: Commit**

```bash
git add plugins/llm/tests/test_context.py
git commit -m "test: cover context update_config, repr, and db prune paths"
```

---

### Task 3: Cover `persistence.py` gaps (rollback paths, empty results, zero-cost rank)

**Files:**
- Modify: `plugins/llm/tests/test_persistence.py`

Coverage targets: lines 621-623, 676-685, 776-784, 827, 956, 1016, 1073, 1114, 1212-1216

**Step 1: Write test for `load_pending_tasks` with type filter (line 827)**

```python
class TestPendingTaskFiltering:
    """Test pending task queries with filters."""

    def test_load_pending_tasks_with_type_filter(self, test_db) -> None:
        """GIVEN tasks of different types WHEN filtered THEN only matching returned."""
        test_db.save_pending_task(
            task_type="ask", nick="u", reply_target="#c", is_channel=True,
            prompt_preview="q1", model="m", request_data="{}", submitted_at=1.0,
            expires_at=999.0, next_attempt_at=1.0,
        )
        test_db.save_pending_task(
            task_type="draw", nick="u", reply_target="#c", is_channel=True,
            prompt_preview="q2", model="m", request_data="{}", submitted_at=2.0,
            expires_at=999.0, next_attempt_at=2.0,
        )
        ask_tasks = test_db.load_pending_tasks(task_type="ask")
        assert len(ask_tasks) == 1
        assert ask_tasks[0].task_type == "ask"
```

**Step 2: Write test for `delete_expired_pending_tasks` with actual rows (lines 776-784)**

```python
    def test_delete_expired_pending_tasks(self, test_db) -> None:
        """GIVEN expired pending task WHEN delete_expired called THEN removed."""
        test_db.save_pending_task(
            task_type="ask", nick="u", reply_target="#c", is_channel=True,
            prompt_preview="q", model="m", request_data="{}", submitted_at=1.0,
            expires_at=10.0, next_attempt_at=1.0,
        )
        expired = test_db.delete_expired_pending_tasks(now=20.0)
        assert len(expired) == 1
        assert len(test_db.load_pending_tasks()) == 0
```

**Step 3: Write tests for usage summary edge cases (lines 956, 1016, 1073, 1114)**

```python
class TestUsageEdgeCases:
    """Test usage queries with empty/edge-case data."""

    def test_get_usage_summary_empty_db(self, test_db) -> None:
        """GIVEN no usage records WHEN summary queried THEN returns zeros."""
        summary = test_db.get_usage_summary()
        assert summary.total_requests == 0

    def test_get_usage_summary_with_since_filter(self, test_db) -> None:
        """GIVEN usage records WHEN filtered by since THEN only recent included."""
        test_db.log_usage("u", "#c", "m", 10, 5, 0.01)
        summary = test_db.get_usage_summary(since=time.time() + 1000)
        assert summary.total_requests == 0

    def test_get_usage_summary_for_channel_empty(self, test_db) -> None:
        """GIVEN no usage WHEN channel summary queried THEN returns zeros."""
        summary = test_db.get_usage_summary_for_channel("#nonexistent")
        assert summary.total_requests == 0

    def test_get_usage_summary_for_nick_empty(self, test_db) -> None:
        """GIVEN no usage WHEN nick summary queried THEN returns zeros."""
        summary = test_db.get_usage_summary_for_nick("nobody")
        assert summary.total_requests == 0

    def test_get_usage_by_nick_with_since(self, test_db) -> None:
        """GIVEN usage data WHEN queried with since THEN filters correctly."""
        test_db.log_usage("u", "#c", "m", 10, 5, 0.01)
        result = test_db.get_usage_by_nick(since=time.time() + 1000)
        assert len(result) == 0
```

**Step 4: Write test for zero-cost rank edge case (lines 1212-1216)**

```python
    def test_get_nick_rank_zero_cost_with_usage(self, test_db) -> None:
        """GIVEN nick with zero-cost usage WHEN ranked THEN gets valid rank."""
        test_db.log_usage("u", "#c", "m", 10, 5, 0.0)
        rank = test_db.get_nick_rank("u")
        assert rank.rank == 1
        assert rank.total == 1

    def test_get_nick_rank_no_usage(self, test_db) -> None:
        """GIVEN nick with no usage WHEN ranked THEN rank is 0."""
        rank = test_db.get_nick_rank("nobody")
        assert rank.rank == 0
```

**Step 5: Run tests**

Run: `make test -- -k test_persistence -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add plugins/llm/tests/test_persistence.py
git commit -m "test: cover persistence rollback, empty results, and zero-cost rank"
```

---

### Task 4: Cover `service.py` — context building and uptime edge cases

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 461, 500-514, 605-606

**Step 1: Write tests for `_get_channel_topic` with no state (line 461)**

```python
class TestContextBuilding:
    """Test context message construction."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_get_channel_topic_no_state(self) -> None:
        """GIVEN irc with no state WHEN get_channel_topic THEN returns None."""
        irc = self.mocker.Mock(spec=[])  # No 'state' attribute
        assert self.service._get_channel_topic(irc, "#test") is None

    def test_get_uptime_negative_returns_none(self) -> None:
        """GIVEN start time in future WHEN get_uptime_info THEN returns None."""
        import time
        with self.mocker.patch("llm.service.world") as mock_world:
            mock_world.startedAt = time.time() + 99999
            assert self.service._get_uptime_info() is None

    def test_build_context_includes_uptime(self) -> None:
        """GIVEN valid irc+msg WHEN build_context_message THEN includes date."""
        irc = self.mocker.Mock()
        irc.state.channels = {"#test": self.mocker.Mock(topic="A topic")}
        msg = self.mocker.Mock()
        msg.args = ["#test"]
        msg.prefix = "user!user@host"

        with self.mocker.patch("llm.service.ircutils") as mock_ircutils:
            mock_ircutils.isChannel.return_value = True
            mock_ircutils.nickFromHostmask.return_value = "user"
            result = self.service._build_context_message(irc, msg)

        assert result is not None
        assert "Date:" in result["content"]

    def test_build_context_none_without_irc(self) -> None:
        """GIVEN no irc WHEN build_context_message THEN returns None."""
        assert self.service._build_context_message(None, None) is None
```

**Step 2: Run tests**

Run: `make test -- -k "TestContextBuilding" -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover service context building and uptime edge cases"
```

---

### Task 5: Cover `service.py` — Gemini tool fallback and usage extraction

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 786-791, 955-956

**Step 1: Write tests for Gemini tool fallback (lines 786-791)**

```python
class TestCompletionWithToolFallback:
    """Test _completion_with_tool_fallback retry logic."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_retries_without_tools_on_invalid_argument(self) -> None:
        """GIVEN tools cause INVALID_ARGUMENT WHEN called THEN retries without tools."""
        import litellm

        mock_completion = self.mocker.patch("llm.service.litellm.completion")
        bad_request = litellm.BadRequestError(
            message="INVALID_ARGUMENT: tools not supported",
            model="gemini/gemini-2.0-flash",
            llm_provider="gemini",
        )
        success_response = self.mocker.Mock()
        success_response.choices = [self.mocker.Mock()]
        mock_completion.side_effect = [bad_request, success_response]

        result = self.service._completion_with_tool_fallback(
            model="gemini/gemini-2.0-flash",
            messages=[{"role": "user", "content": "test"}],
            api_key="key",
            timeout=30,
            tools=[{"type": "function"}],
        )
        assert result == success_response
        assert mock_completion.call_count == 2
        # Second call should NOT have tools
        second_call_kwargs = mock_completion.call_args_list[1]
        assert "tools" not in second_call_kwargs.kwargs
```

**Step 2: Write test for `_extract_usage` exception handling (lines 955-956)**

```python
    def test_extract_usage_handles_attribute_error(self) -> None:
        """GIVEN response with broken usage attr WHEN extracted THEN returns zeros."""
        response = self.mocker.Mock()
        type(response).usage = self.mocker.PropertyMock(side_effect=TypeError)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)

        pt, ct, cost = self.service._extract_usage(response, "gpt-4")
        assert pt == 0
        assert ct == 0
```

**Step 3: Run and commit**

Run: `make test -- -k "TestCompletionWithToolFallback" -v`

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover Gemini tool fallback and usage extraction errors"
```

---

### Task 6: Cover `service.py` — pending task retry paths

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 1043-1044, 1088-1090, 1205-1244, 1334-1347, 1399-1408

**Step 1: Write tests for `_stash_timeout` with no database (lines 1043-1044)**

```python
class TestStashTimeout:
    """Test timeout stashing for pending tasks."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_stash_timeout_no_db_returns_false(self) -> None:
        """GIVEN no database WHEN stash_timeout called THEN returns False."""
        self.mock_plugin.db = None
        result = self.service._stash_timeout(
            task_type="ask", nick="u", reply_target="#c",
            is_channel=True, prompt="test", model="m",
            request_data={"prompt": "test"}, submitted_at=1.0,
        )
        assert result is False
```

**Step 2: Write test for `_delete_stashed_task` (lines 1088-1090)**

```python
    def test_delete_stashed_task_with_none_db(self) -> None:
        """GIVEN None db WHEN delete_stashed_task THEN no error."""
        LLMService._delete_stashed_task(None, 1)  # Should not raise

    def test_delete_stashed_task_with_none_id(self) -> None:
        """GIVEN None task_id WHEN delete_stashed_task THEN no error."""
        LLMService._delete_stashed_task(self.mocker.Mock(), None)  # Should not raise
```

**Step 3: Write tests for `_retry_image` edge cases (lines 1205-1244)**

```python
class TestRetryImage:
    """Test image retry for pending tasks."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _make_task(self):
        return self.mocker.Mock(
            task_type="draw", nick="user", reply_target="#chan",
            is_channel=1, prompt_preview="test prompt", model="dall-e-3",
        )

    def test_retry_image_malformed_data(self) -> None:
        """GIVEN request_data without prompt WHEN retried THEN fails terminal."""
        result = self.service._retry_image(self._make_task(), {"not_prompt": "x"})
        assert result.status == "failed_terminal"
        assert "Malformed" in result.reason

    def test_retry_image_no_api_key(self) -> None:
        """GIVEN no API key configured WHEN retried THEN fails terminal."""
        self.mock_plugin.registryValue.side_effect = lambda key, *a: (
            "" if key == "drawApiKey" else 30
        )
        result = self.service._retry_image(self._make_task(), {"prompt": "cat"})
        assert result.status == "failed_terminal"
        assert "API key" in result.reason

    def test_retry_image_content_blocked(self) -> None:
        """GIVEN blocked content WHEN retried THEN fails terminal."""
        self.mocker.patch.object(self.service, "_attempt_image_generation", return_value=None)
        result = self.service._retry_image(self._make_task(), {"prompt": "cat"})
        assert result.status == "failed_terminal"
        assert "blocked" in result.reason.lower()
```

**Step 4: Write tests for `check_pending_tasks` edge cases (lines 1334-1347, 1399-1408)**

```python
class TestCheckPendingTasks:
    """Test check_pending_tasks dispatch and delivery."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _make_db(self):
        db = self.mocker.Mock()
        self.mock_plugin.db = db
        return db

    def test_unknown_task_type_marks_terminal(self) -> None:
        """GIVEN task with unknown type WHEN checked THEN marked failed_terminal."""
        db = self._make_db()
        task = self.mocker.Mock(
            id=1, task_type="unknown", nick="u", reply_target="#c",
            is_channel=1, prompt_preview="q", model="m",
            request_data='{"prompt":"test"}', result_payload=None,
            delivery_state="pending",
        )
        db.claim_due_pending_tasks.side_effect = [
            [task],  # Phase 1: provider
            [],      # Phase 2: delivery
        ]
        db.delete_expired_pending_tasks.return_value = []
        self.service.check_pending_tasks({"#c"})
        db.update_task_for_delivery.assert_called_once()
        call_args = db.update_task_for_delivery.call_args
        assert "Unknown task type" in call_args[0][2]

    def test_malformed_json_marks_terminal(self) -> None:
        """GIVEN task with bad JSON WHEN checked THEN marked failed."""
        db = self._make_db()
        task = self.mocker.Mock(
            id=1, task_type="ask", nick="u", reply_target="#c",
            is_channel=1, prompt_preview="q", model="m",
            request_data="not json!", result_payload=None,
            delivery_state="pending",
        )
        db.claim_due_pending_tasks.side_effect = [[task], []]
        db.delete_expired_pending_tasks.return_value = []
        self.service.check_pending_tasks({"#c"})
        db.update_task_for_delivery.assert_called_once()

    def test_delivery_undeliverable_channel_deferred(self) -> None:
        """GIVEN task for missing channel WHEN delivery phase THEN deferred."""
        db = self._make_db()
        task = self.mocker.Mock(
            id=1, task_type="ask", nick="u", reply_target="#gone",
            is_channel=1, prompt_preview="q", model="m",
            request_data='{}', result_payload='{"status":"completed","content":"hi"}',
            delivery_state="ready",
        )
        db.claim_due_pending_tasks.side_effect = [
            [],     # Phase 1: no provider tasks
            [task], # Phase 2: delivery
        ]
        db.delete_expired_pending_tasks.return_value = []
        results = self.service.check_pending_tasks({"#other"})
        db.release_pending_task.assert_called_once()
        assert len(results) == 0
```

**Step 5: Run and commit**

Run: `make test -- -k "TestStash or TestRetryImage or TestCheckPending" -v`

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover pending task retry, stashing, and delivery edge cases"
```

---

### Task 7: Cover `service.py` — image generation and rewrite paths

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 1898, 1926-1928, 1969-1970, 2009-2010, 2016-2017, 2069, 2132-2144, 2163-2166

**Step 1: Write tests for `_rewrite_prompt_for_safety` empty response (line 1898)**

```python
class TestImageGeneration:
    """Test image generation paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_rewrite_empty_response_returns_none(self) -> None:
        """GIVEN LLM returns empty rewrite WHEN rewriting THEN returns None."""
        mock_response = self.mocker.Mock()
        mock_response.choices = [self.mocker.Mock()]
        mock_response.choices[0].message.content = ""
        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result, pt, ct, cost = self.service._rewrite_prompt_for_safety(
            "bad prompt", "blocked", [], "#chan"
        )
        assert result is None
```

**Step 2: Write tests for `_attempt_image_generation` b64_json path (lines 1969-1970)**

```python
    def test_attempt_image_b64_json_save_failure(self) -> None:
        """GIVEN b64_json data but save fails WHEN attempted THEN returns error."""
        response = self.mocker.Mock()
        response.data = [self.mocker.Mock(url=None, b64_json="base64data")]
        response.data[0].url = None
        response.usage = None
        self.mocker.patch("llm.service.litellm.image_generation", return_value=response)
        self.mocker.patch.object(self.service, "save_image_to_http", return_value=None)
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))

        result = self.service._attempt_image_generation("cat", "dall-e-3", 30)
        assert result is not None
        assert result.error is not None
```

**Step 3: Write tests for `image_generation` timeout stashing (line 2069)**

```python
    def test_image_generation_timeout_not_stashed(self) -> None:
        """GIVEN timeout but stashing fails WHEN generating THEN returns error."""
        import litellm
        self.mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm.Timeout(message="timeout", model="dall-e-3", llm_provider="openai"),
        )
        self.mocker.patch.object(self.service, "_stash_timeout", return_value=False)
        result = self.service.image_generation("a cat")
        assert result.error is not None
```

**Step 4: Write test for content policy error in rewrite loop (lines 2132-2144)**

```python
    def test_image_generation_rewrite_loop_non_content_error(self) -> None:
        """GIVEN non-content error during rewrite retry WHEN generating THEN stops."""
        # First attempt: content blocked (returns None)
        self.mocker.patch.object(
            self.service, "_attempt_image_generation",
            side_effect=[None, RuntimeError("network error")],
        )
        self.mocker.patch.object(
            self.service, "_rewrite_prompt_for_safety",
            return_value=("rewritten prompt", 10, 5, 0.01),
        )
        self.mocker.patch.object(self.service, "_is_content_safety_error", return_value=False)
        result = self.service.image_generation("a cat")
        assert result.error is not None
```

**Step 5: Write test for outer exception handler (lines 2163-2166)**

```python
    def test_image_generation_unexpected_exception(self) -> None:
        """GIVEN unexpected error WHEN generating THEN returns error gracefully."""
        self.mocker.patch.object(
            self.service, "validate_prompt",
            side_effect=RuntimeError("unexpected"),
        )
        result = self.service.image_generation("a cat")
        assert result.error is not None
```

**Step 6: Write test for xai model kwargs (lines 1926-1928)**

```python
    def test_attempt_image_xai_model_sets_kwargs(self) -> None:
        """GIVEN xai model WHEN attempted THEN passes aspect_ratio and quality."""
        response = self.mocker.Mock()
        img = self.mocker.Mock(url="https://example.com/img.png", b64_json=None)
        response.data = [img]
        response.usage = None
        mock_gen = self.mocker.patch("llm.service.litellm.image_generation", return_value=response)
        self.mocker.patch.object(self.service, "_extract_usage", return_value=(0, 0, 0.0))
        self.mocker.patch.object(self.service, "_download_and_save_image", return_value=None)

        self.service._attempt_image_generation("cat", "xai/grok-2-image", 30)
        call_kwargs = mock_gen.call_args.kwargs
        assert call_kwargs.get("aspect_ratio") == "9:16"
```

**Step 7: Run and commit**

Run: `make test -- -k "TestImageGeneration" -v`

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover image generation rewrite loop and error paths"
```

---

### Task 8: Cover `service.py` — HTTP file management and cleanup

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 2216-2223, 2328-2330, 2355-2357, 2472-2492, 2540-2546

**Step 1: Write tests for `get_http_paths` localhost fallback (lines 2216-2223)**

```python
class TestHTTPFileManagement:
    """Test HTTP file storage and cleanup."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(httpRoot="", httpUrlBase="")

    def test_get_http_paths_localhost_fallback(self) -> None:
        """GIVEN no public URL WHEN get_http_paths THEN falls back to localhost."""
        self.mocker.patch("llm.service.conf.supybot.directories.data.web.dirize", return_value="/tmp/web")
        self.mocker.patch("llm.service.conf.supybot.servers.http.publicUrl", return_value="")
        self.mocker.patch("llm.service.conf.supybot.servers.http.port", return_value=8080)

        root, url = self.service.get_http_paths()
        assert "localhost:8080" in url
```

**Step 2: Write tests for `save_code_to_http` OSError (lines 2328-2330)**

```python
    def test_save_code_to_http_oserror(self) -> None:
        """GIVEN write fails WHEN save_code_to_http THEN returns None."""
        self.mocker.patch.object(self.service, "get_http_paths", return_value=("/nonexistent/path", "http://x"))
        self.mocker.patch("llm.service.Path.mkdir", side_effect=OSError("no space"))
        result = self.service.save_code_to_http("# hello world")
        assert result is None
```

**Step 3: Write tests for `_cleanup_old_files` (lines 2472-2492)**

```python
    def test_cleanup_old_files_deletes_old(self, tmp_path: Path) -> None:
        """GIVEN old files WHEN cleanup runs THEN old files deleted."""
        import time

        # Create an "old" file
        old_file = tmp_path / "code_old.html"
        old_file.write_text("old")
        # Backdate modification time
        import os
        old_time = time.time() - (25 * 3600)  # 25 hours old
        os.utime(old_file, (old_time, old_time))

        # Create a "new" file
        new_file = tmp_path / "code_new.html"
        new_file.write_text("new")

        self.service._cleanup_old_files(str(tmp_path), max_age_hours=24, max_files=100)
        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_old_files_caps_recent(self, tmp_path: Path) -> None:
        """GIVEN too many recent files WHEN cleanup THEN oldest recent deleted."""
        for i in range(5):
            (tmp_path / f"code_{i}.html").write_text(f"content{i}")

        self.service._cleanup_old_files(str(tmp_path), max_age_hours=9999, max_files=2)
        remaining = list(tmp_path.glob("*.html"))
        assert len(remaining) == 2

    def test_cleanup_nonexistent_dir_noop(self) -> None:
        """GIVEN nonexistent directory WHEN cleanup THEN no error."""
        self.service._cleanup_old_files("/nonexistent/path", max_age_hours=24, max_files=100)
```

**Step 4: Run and commit**

Run: `make test -- -k "TestHTTPFileManagement" -v`

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover HTTP file management and cleanup edge cases"
```

---

### Task 9: Cover `service.py` — message building and memory cleanup validation

**Files:**
- Modify: `plugins/llm/tests/test_service.py`

Coverage targets: lines 2540-2579, 2622-2624, 2695-2723

**Step 1: Write tests for `_build_messages` with images (lines 2540, 2542, 2546)**

```python
class TestBuildMessages:
    """Test _build_messages construction."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_build_messages_with_images(self) -> None:
        """GIVEN image URLs WHEN building messages THEN multimodal format used."""
        msgs = self.service._build_messages("describe this", ["https://img.png"])
        last_msg = msgs[-1]
        assert isinstance(last_msg["content"], list)
        assert any(p["type"] == "image_url" for p in last_msg["content"])

    def test_build_messages_with_channel_history(self) -> None:
        """GIVEN channel history WHEN building THEN includes summary."""
        history = [{"nick": "alice", "role": "user", "content": "hello"}]
        msgs = self.service._build_messages("hi", None, channel_history=history)
        assert any("channel discussion" in str(m.get("content", "")).lower() for m in msgs)
```

**Step 2: Write test for `_format_channel_history` truncation (lines 2571-2574)**

```python
    def test_format_channel_history_truncates_long(self) -> None:
        """GIVEN long message WHEN formatted THEN truncated with ellipsis."""
        history = [{"nick": "alice", "content": "x" * 1000}]
        result = self.service._format_channel_history(history)
        assert result.endswith("...")
        assert len(result) < 1000
```

**Step 3: Write test for `extract_memories` API key fallback (lines 2622-2624)**

```python
class TestExtractMemories:
    """Test memory extraction."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(memoryApiKey="")

    def test_extract_memories_falls_back_to_ask_key(self) -> None:
        """GIVEN no memoryApiKey WHEN extracting THEN uses askApiKey."""
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock()]
        response.choices[0].message.content = '{"add": ["likes cats"]}'
        mock_completion = self.mocker.patch("llm.service.litellm.completion", return_value=response)

        result = self.service.extract_memories("nick", "#chan", "I like cats", "Cool!", [])
        assert result.add == ["likes cats"]
        # Verify askApiKey was used
        call_kwargs = mock_completion.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"  # The TEST_API_KEY from conftest
```

**Step 4: Write tests for `cleanup_memories` validation (lines 2695-2723)**

```python
class TestCleanupMemoriesValidation:
    """Test cleanup_memories input validation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _mock_cleanup_response(self, parsed: dict) -> None:
        import json
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock()]
        response.choices[0].message.content = json.dumps(parsed)
        self.mocker.patch("llm.service.litellm.completion", return_value=response)

    def _make_rows(self, count: int):
        from llm.persistence import MemoryRow
        return [MemoryRow(id=i, nick="u", fact=f"fact{i}", source_channel="#c", created_at=0.0) for i in range(count)]

    def test_not_a_dict(self) -> None:
        """GIVEN LLM returns non-dict WHEN cleanup THEN error."""
        response = self.mocker.Mock()
        response.choices = [self.mocker.Mock()]
        response.choices[0].message.content = '"just a string"'
        self.mocker.patch("llm.service.litellm.completion", return_value=response)
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert result.error is not None
        assert "not a JSON object" in result.error

    def test_drop_not_list(self) -> None:
        """GIVEN drop is not a list WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": "not a list", "merge": []})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "must be arrays" in result.error

    def test_invalid_drop_index(self) -> None:
        """GIVEN out-of-range drop index WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [99], "merge": []})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "Invalid drop index" in result.error

    def test_invalid_merge_entry(self) -> None:
        """GIVEN non-dict merge entry WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [], "merge": ["not a dict"]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "Invalid merge entry" in result.error

    def test_merge_needs_indices(self) -> None:
        """GIVEN merge with no indices WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "at least 2 indices" in result.error

    def test_merge_empty_text(self) -> None:
        """GIVEN merge with empty text WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [0, 1], "text": ""}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "non-empty" in result.error

    def test_duplicate_indices(self) -> None:
        """GIVEN same index in drop and merge WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [0], "merge": [{"indices": [0, 1], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "Duplicate" in result.error

    def test_merge_index_out_of_range(self) -> None:
        """GIVEN merge index >= len WHEN cleanup THEN error."""
        self._mock_cleanup_response({"drop": [], "merge": [{"indices": [0, 99], "text": "merged"}]})
        result = self.service.cleanup_memories("u", "#c", self._make_rows(3))
        assert "out of range" in result.error
```

**Step 5: Run and commit**

Run: `make test -- -k "TestBuildMessages or TestExtractMemories or TestCleanupMemoriesValidation" -v`

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test: cover message building, memory extraction fallback, and cleanup validation"
```

---

### Task 10: Cover `plugin.py` — HTTP callback and build info

**Files:**
- Modify: `plugins/llm/tests/test_plugin.py`

Coverage targets: lines 264-265, 852-853

**Step 1: Write test for HTTP callback OSError handler (lines 264-265)**

```python
class TestHTTPCallbackErrors:
    """Test HTTP callback error handling."""

    def test_doget_oserror_nested_broken_pipe(self, mocker) -> None:
        """GIVEN OSError in doGet WHEN send_response also fails THEN silenced."""
        from llm.plugin import LLMHTTPCallback

        handler = mocker.Mock()
        handler.path = "/llm/nonexistent.html"
        callback = LLMHTTPCallback.__new__(LLMHTTPCallback)
        callback.plugin = mocker.Mock()
        callback.plugin.registryValue.return_value = "/tmp/nonexistent"

        # OSError triggers send_response(500) which also raises BrokenPipeError
        handler.send_response.side_effect = BrokenPipeError
        # Should not raise
        callback.doGet(handler, "")
```

**Step 2: Write test for `_get_build_info` git failure (lines 852-853)**

```python
class TestBuildInfo:
    """Test build info retrieval."""

    def test_build_info_git_failure(self, mocker, mock_irc) -> None:
        """GIVEN git not available WHEN build_info THEN returns version only."""
        import subprocess
        from llm.plugin import LLM
        from tests.conftest import plugin_init_patches

        patches = plugin_init_patches(mocker)
        mocker.patch("subprocess.check_output", side_effect=FileNotFoundError)

        plugin = LLM(mock_irc)
        info = plugin._get_build_info()
        assert info.startswith("v")
        assert "(" not in info  # No git SHA
```

**Step 3: Run and commit**

Run: `make test -- -k "TestHTTPCallback or TestBuildInfo" -v`

```bash
git add plugins/llm/tests/test_plugin.py
git commit -m "test: cover HTTP callback error nesting and build info git failure"
```

---

### Task 11: Cover `plugin.py` — pending task delivery and spontaneous participation

**Files:**
- Modify: `plugins/llm/tests/test_commands.py`

Coverage targets: lines 482-493, 541-552, 654, 738, 756-757

**Step 1: Write tests for `_check_pending_tasks` exception handling (lines 482-493)**

```python
class TestCheckPendingTasksPlugin:
    """Test plugin-level pending task dispatch."""

    def test_delivery_failure_logged_not_raised(self, mocker, mock_irc) -> None:
        """GIVEN delivery raises WHEN check_pending_tasks THEN error logged."""
        from llm.plugin import LLM
        from tests.conftest import plugin_init_patches

        patches = plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        mock_result = mocker.Mock(task_id=1, nick="u", status="completed")
        plugin.llm_service.check_pending_tasks.return_value = [mock_result]
        mocker.patch.object(plugin, "_deliver_pending_result", side_effect=RuntimeError("oops"))

        plugin._check_pending_tasks()  # Should not raise
        plugin.log.error.assert_called()
```

**Step 2: Write test for spontaneous CTCP filtering (line 654) and exception recovery (line 756)**

These are in the doPrivmsg handler for spontaneous participation. The tests need to exercise the spontaneous code path where context is disabled (line 654) and where the evaluation closure catches an exception (line 756). These are deeply coupled to IRC message flow and may be better tested via integration-style tests. Write a focused unit test for the spontaneous evaluation:

```python
class TestSpontaneousEdgeCases:
    """Test spontaneous participation edge cases."""

    def test_spontaneous_skips_when_context_disabled(self, mocker, mock_irc) -> None:
        """GIVEN context disabled WHEN message received THEN spontaneous skipped."""
        from llm.plugin import LLM
        from tests.conftest import plugin_init_patches, make_registry_side_effect

        patches = plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({
                "contextEnabled": False,
                "spontaneousEnabled": True,
            })
        )

        msg = mocker.Mock()
        msg.args = ["#test", "hello everyone"]
        msg.prefix = "user!user@host"
        msg.nick = "user"
        msg.tagged.return_value = None
        msg.command = "PRIVMSG"

        # Should not schedule spontaneous evaluation
        plugin.doPrivmsg(mock_irc, msg)
        assert not plugin.llm_service.completion.called
```

**Step 3: Run and commit**

Run: `make test -- -k "TestCheckPendingTasksPlugin or TestSpontaneousEdgeCases" -v`

```bash
git add plugins/llm/tests/test_commands.py
git commit -m "test: cover pending task delivery errors and spontaneous edge cases"
```

---

### Task 12: Run `make preflight` and verify coverage improvement

**Step 1: Run full test suite with coverage**

Run: `make preflight`
Expected: All checks pass, coverage > 93%

**Step 2: Review remaining uncovered lines**

Run: `make test 2>&1 | grep "Missing"`
Check which lines are still uncovered and assess whether they're worth covering (some deep IRC protocol paths may be impractical to unit test).

**Step 3: Final commit if any adjustments needed**

```bash
git add -A
git commit -m "test: finalize test coverage improvements"
```
