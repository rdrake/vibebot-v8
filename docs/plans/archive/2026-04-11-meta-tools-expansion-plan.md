# Meta Tools Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the meta command from 9 to 15 tools, adding usage queries, memory cleanup, and reminders.

**Architecture:** Usage and cleanup tools are added directly to `MetaToolExecutor`. Reminder tools use a callable pattern — the plugin passes pre-bound functions to the executor so all scheduler/dict/DB coordination stays in the plugin. Two new plugin helper methods (`_remind_set_for_meta`, `_remind_delete_for_meta`) return strings instead of calling `irc.reply()`.

**Tech Stack:** Existing `LLMDatabase`, `ConversationContext`, Limnoria scheduler, `MetaToolExecutor` callable pattern

**Design Doc:** `docs/plans/2026-04-11-meta-tools-expansion-design.md`

**Review fixes incorporated:**
- Update `META_SYSTEM_PROMPT` to mention usage, cleanup, and reminders
- Update "not meta" help text in plugin.py
- Mock `sanitize_output` as pass-through in reminder helper tests
- Add `_MetaSynchronized_rlock` to reminder helper test fixture
- Only add `cleanup_fn` to `meta_completion` in Task 2; defer reminder callables to Task 4
- Add reminder integration test in Task 5
- Move `datetime` import to module level in meta.py

---

### Task 1: Add Usage Tools and Update System Prompt

**Files:**
- Modify: `plugins/llm/src/llm/meta.py`
- Modify: `plugins/llm/tests/test_meta.py`

**Step 1: Update META_SYSTEM_PROMPT**

In `plugins/llm/src/llm/meta.py`, update the system prompt (around line 18)
to mention the new capabilities. Change the `NOT_META` rule from:

```
"memories, or conversation context, respond with exactly: NOT_META\n"
```

to:

```
"memories, conversation context, usage statistics, memory cleanup, "
"or reminders, respond with exactly: NOT_META\n"
```

**Step 2: Move datetime import to module level**

Add at the top of `meta.py` (with the other stdlib imports):

```python
from datetime import UTC, datetime
```

**Step 3: Add usage tool schemas to META_TOOLS**

In `meta.py`, add after the `forget_context` tool definition (before the
closing `]` of `META_TOOLS`):

```python
    {
        "type": "function",
        "function": {
            "name": "get_usage",
            "description": (
                "Get the user's API usage statistics for the current month "
                "(request count, tokens, cost)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_usage",
            "description": (
                "Get API usage statistics for the current channel this month "
                "(request count, tokens, cost)."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
```

**Step 4: Add executor handlers and helper**

In `meta.py`, add to `MetaToolExecutor`:

```python
    def _tool_get_usage(self, _args: dict[str, Any]) -> str:
        since = self._month_start()
        summary = self.db.get_usage_summary_for_nick(self.nick, since=since)
        return json.dumps({
            "requests": summary.total_requests,
            "prompt_tokens": summary.total_prompt_tokens,
            "completion_tokens": summary.total_completion_tokens,
            "cost": round(summary.total_cost, 4),
        })

    def _tool_get_channel_usage(self, _args: dict[str, Any]) -> str:
        since = self._month_start()
        summary = self.db.get_usage_summary_for_channel(
            self.channel, since=since
        )
        return json.dumps({
            "requests": summary.total_requests,
            "prompt_tokens": summary.total_prompt_tokens,
            "completion_tokens": summary.total_completion_tokens,
            "cost": round(summary.total_cost, 4),
        })

    @staticmethod
    def _month_start() -> float:
        """Return Unix timestamp for midnight UTC on the 1st of this month."""
        now = datetime.now(UTC)
        return datetime(now.year, now.month, 1, tzinfo=UTC).timestamp()
```

**Step 5: Write the tests**

Add to `TestMetaToolExecutor` in `plugins/llm/tests/test_meta.py`:

```python
    def test_get_usage(
        self, executor: MetaToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN get_usage tool WHEN called THEN returns user's usage summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=47,
            total_prompt_tokens=12000,
            total_completion_tokens=3000,
            total_cost=0.12,
        )
        result = executor.execute("get_usage", {})
        assert "47" in result
        assert "0.12" in result
        mock_db.get_usage_summary_for_nick.assert_called_once()

    def test_get_channel_usage(
        self, executor: MetaToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN get_channel_usage tool WHEN called THEN returns channel summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=200,
            total_prompt_tokens=50000,
            total_completion_tokens=10000,
            total_cost=0.85,
        )
        result = executor.execute("get_channel_usage", {})
        assert "200" in result
        assert "0.85" in result
        mock_db.get_usage_summary_for_channel.assert_called_once()
```

**Step 6: Update tool count test**

```python
    def test_tool_count(self) -> None:
        assert len(META_TOOLS) == 11
```

**Step 7: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/meta.py plugins/llm/tests/test_meta.py
git commit -m "feat(meta): add usage tools and update system prompt for new capabilities"
```

---

### Task 2: Add Memory Cleanup Tool (with callable pattern)

**Files:**
- Modify: `plugins/llm/src/llm/meta.py` (tool schema, executor __init__, handler)
- Modify: `plugins/llm/src/llm/service.py` (add cleanup_fn to meta_completion)
- Modify: `plugins/llm/src/llm/plugin.py` (pass callable at call sites)
- Modify: `plugins/llm/tests/test_meta.py`

Only `cleanup_fn` is added to `meta_completion` in this task. Reminder
callables are deferred to Task 4 to keep commits self-contained.

**Step 1: Write the failing tests**

Update the executor fixture in `TestMetaToolExecutor` to accept
`cleanup_fn`, and add the test:

```python
    @pytest.fixture
    def mock_cleanup_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = "Before: 8 | dropped: 2, merged: 4 → 2 | after: 4"
        return fn

    @pytest.fixture
    def executor(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> MetaToolExecutor:
        return MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            cleanup_fn=mock_cleanup_fn,
        )
```

Test:

```python
    def test_cleanup_memories(
        self, executor: MetaToolExecutor, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN cleanup_memories tool WHEN called THEN runs cleanup callable."""
        result = executor.execute("cleanup_memories", {})
        mock_cleanup_fn.assert_called_once()
        assert "Before: 8" in result

    def test_cleanup_memories_not_available(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN no cleanup_fn WHEN cleanup_memories called THEN returns error."""
        executor = MetaToolExecutor(
            db=mock_db, context=mock_context, nick="testuser", channel="#test"
        )
        result = executor.execute("cleanup_memories", {})
        assert "not available" in result.lower() or "error" in result.lower()
```

**Step 2: Add tool schema**

In `meta.py`, add after `get_channel_usage`:

```python
    {
        "type": "function",
        "function": {
            "name": "cleanup_memories",
            "description": (
                "Run automatic memory cleanup — deduplicates, merges related "
                "facts, and removes low-quality entries. Requires at least 2 "
                "stored memories."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
```

**Step 3: Update MetaToolExecutor to accept cleanup_fn**

Update `__init__` in `meta.py`. Add `Callable` import under `TYPE_CHECKING`:

```python
if TYPE_CHECKING:
    from collections.abc import Callable

    from .context import ConversationContext
    from .persistence import LLMDatabase
```

Update constructor:

```python
    def __init__(
        self,
        *,
        db: LLMDatabase,
        context: ConversationContext,
        nick: str,
        channel: str,
        cleanup_fn: Callable[[], str] | None = None,
    ) -> None:
        self.db = db
        self.context = context
        self.nick = nick
        self.channel = channel
        self._cleanup_fn = cleanup_fn
```

Add handler:

```python
    def _tool_cleanup_memories(self, _args: dict[str, Any]) -> str:
        if self._cleanup_fn is None:
            return json.dumps({"error": "Memory cleanup is not available."})
        result = self._cleanup_fn()
        return json.dumps({"status": "ok", "message": result})
```

**Step 4: Update service.py meta_completion — add cleanup_fn only**

In `plugins/llm/src/llm/service.py`, update `meta_completion` signature
to add `cleanup_fn`:

```python
    def meta_completion(
        self,
        prompt: str,
        *,
        nick: str,
        channel: str,
        db: LLMDatabase,
        context: ConversationContext,
        bot_nick: str,
        api_key: str | None = None,
        model_override: str | None = None,
        cleanup_fn: Callable[[], str] | None = None,
    ) -> MetaResult:
```

Add `Callable` to the `TYPE_CHECKING` imports in service.py:

```python
if TYPE_CHECKING:
    from collections.abc import Callable
    ...
```

Update executor construction:

```python
            executor = MetaToolExecutor(
                db=db, context=context, nick=nick, channel=channel,
                cleanup_fn=cleanup_fn,
            )
```

**Step 5: Update plugin.py call sites**

Both `meta_completion` call sites (in `invalidCommand` and `meta` command)
need `cleanup_fn`. Add to both:

```python
            cleanup_fn=lambda: self._run_memory_cleanup(
                preflight.nick, preflight.channel
            ),
```

**Step 6: Update tool count test**

```python
        assert len(META_TOOLS) == 12
```

**Step 7: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/meta.py plugins/llm/src/llm/service.py \
    plugins/llm/src/llm/plugin.py plugins/llm/tests/test_meta.py
git commit -m "feat(meta): add cleanup_memories tool with callable pattern"
```

---

### Task 3: Add Reminder Helper Methods to Plugin

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (two new private methods)
- Modify: `plugins/llm/tests/test_meta.py`

**Step 1: Write the failing tests**

Add a new test class to `test_meta.py`. Important: the fixture must include
`_MetaSynchronized_rlock` (needed by `_allow_concurrent()`) and mock
`sanitize_output` as a pass-through (it returns its input, not a MagicMock):

```python
class TestReminderMetaHelpers:
    """Tests for plugin reminder helper methods used by meta."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": True})
        )
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda s: s
        plugin.db = mocker.MagicMock()
        plugin._reminders = {}
        plugin._reminders_lock = threading.Lock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_remind_set_for_meta_success(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN valid reminder text WHEN _remind_set_for_meta THEN returns confirmation."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="check the build",
            confirmation="I'll remind you in 1 hour",
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "check the build in 1 hour")

        assert "remind" in result.lower() or "hour" in result.lower()
        assert plugin.db.save_reminder.called

    def test_remind_set_for_meta_with_note(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder with note WHEN _remind_set_for_meta THEN includes note."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="deploy",
            confirmation="I'll remind you in 1 hour",
            note="assuming Eastern time",
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "deploy in 1 hour")

        assert "Eastern" in result

    def test_remind_set_for_meta_parse_failure(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unparseable reminder WHEN _remind_set_for_meta THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "maybe sometime")

        assert "could not" in result.lower()

    def test_remind_set_for_meta_too_short(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder < 10 seconds WHEN _remind_set_for_meta THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="now",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "remind me now")

        assert "10 second" in result.lower() or "at least" in result.lower()

    def test_remind_set_for_meta_too_long(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder > 7 days WHEN _remind_set_for_meta THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=700000,
            message="later",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "remind me in 2 weeks")

        assert "7 day" in result.lower()

    def test_remind_set_for_meta_clarify(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN clarify action WHEN _remind_set_for_meta THEN returns clarification."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When exactly should I remind you?",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_meta(mock_irc, msg, "testuser", "remind me")

        assert "when" in result.lower()

    def test_remind_delete_for_meta_success(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """GIVEN valid reminder ID WHEN _remind_delete_for_meta THEN deletes."""
        event_name = "llm_remind_abc123def456"
        plugin._reminders = {event_name: ("testuser", "#test", "check build")}
        mocker.patch("llm.plugin.schedule.removeEvent")

        result = plugin._remind_delete_for_meta("testuser", "abc123def456")

        assert "delete" in result.lower() or "cancel" in result.lower()
        assert event_name not in plugin._reminders

    def test_remind_delete_for_meta_not_found(self, plugin) -> None:
        """GIVEN unknown reminder ID WHEN _remind_delete_for_meta THEN error."""
        plugin._reminders = {}

        result = plugin._remind_delete_for_meta("testuser", "nonexistent")

        assert "not found" in result.lower()
```

**Step 2: Implement the helper methods**

In `plugins/llm/src/llm/plugin.py`, add after `_remind_set` (around
line 2421):

```python
    def _remind_set_for_meta(
        self, irc: callbacks.Irc, msg: IrcMsg, nick: str, text: str
    ) -> str:
        """Parse and schedule a reminder, returning a result string for meta.

        Same logic as _remind_set() but returns a string instead of
        calling irc.reply().
        """
        channel = self._get_channel(msg)

        with self._trace_request("remind", nick, channel):
            with self._allow_concurrent():
                result = self.llm_service.parse_reminder(text, channel)

        if result.action == "clarify":
            return result.confirmation

        if result.seconds is None:
            return "Could not determine when to set the reminder."

        if result.seconds < 10:
            return "Reminder must be at least 10 seconds from now."

        if result.seconds > 604800:  # 7 days
            return "Reminder can't be more than 7 days out."

        reminder_message = result.message or text
        event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
        deliver = self._make_reminder_delivery_closure(
            nick, channel, reminder_message, event_name
        )

        try:
            schedule.addEvent(
                deliver, time.time() + result.seconds, name=event_name
            )
            with self._reminders_lock:
                self._reminders[event_name] = (nick, channel, reminder_message)

            self.db.save_reminder(
                event_name, nick, channel, reminder_message,
                time.time() + result.seconds,
            )

            reply = self.llm_service.sanitize_output(result.confirmation)
            if result.note:
                reply = (
                    f"{reply} ({self.llm_service.sanitize_output(result.note)})"
                )
            return reply
        except Exception as e:
            self.log.error("Failed to schedule reminder: %s", e)
            return "Failed to set reminder."

    def _remind_delete_for_meta(self, nick: str, reminder_id: str) -> str:
        """Delete a reminder by ID, returning a result string for meta."""
        target = self._find_user_reminder(nick, reminder_id)
        if target is None:
            return f"Reminder {reminder_id} not found."

        with contextlib.suppress(KeyError):
            schedule.removeEvent(target)
        with self._reminders_lock:
            self._reminders.pop(target, None)
        self.db.delete_reminder(target)
        return f"Deleted reminder {reminder_id}."
```

**Step 3: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_meta.py
git commit -m "feat(meta): add _remind_set_for_meta and _remind_delete_for_meta helpers"
```

---

### Task 4: Add Reminder Tools to Meta Executor

**Files:**
- Modify: `plugins/llm/src/llm/meta.py` (tool schemas, executor __init__, handlers)
- Modify: `plugins/llm/src/llm/service.py` (add reminder callables to meta_completion)
- Modify: `plugins/llm/src/llm/plugin.py` (pass callables at call sites, update help text)
- Modify: `plugins/llm/tests/test_meta.py`

**Step 1: Write the failing tests**

Update the `TestMetaToolExecutor` executor fixture to its final form with
all callables:

```python
    @pytest.fixture
    def mock_list_reminders_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = [
            ("llm_remind_abc123", ("testuser", "#test", "check build")),
            ("llm_remind_def456", ("testuser", "#test", "deploy app")),
        ]
        return fn

    @pytest.fixture
    def mock_set_reminder_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = "I'll remind you in 1 hour"
        return fn

    @pytest.fixture
    def mock_delete_reminder_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = "Deleted reminder abc123."
        return fn

    @pytest.fixture
    def executor(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_reminders_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_delete_reminder_fn: MagicMock,
    ) -> MetaToolExecutor:
        return MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            cleanup_fn=mock_cleanup_fn,
            list_reminders_fn=mock_list_reminders_fn,
            set_reminder_fn=mock_set_reminder_fn,
            delete_reminder_fn=mock_delete_reminder_fn,
        )
```

Tests:

```python
    def test_list_reminders(
        self, executor: MetaToolExecutor, mock_list_reminders_fn: MagicMock
    ) -> None:
        """GIVEN list_reminders tool WHEN called THEN returns formatted reminders."""
        result = executor.execute("list_reminders", {})
        mock_list_reminders_fn.assert_called_once()
        assert "check build" in result
        assert "deploy app" in result
        assert "abc123" in result

    def test_list_reminders_empty(
        self, executor: MetaToolExecutor, mock_list_reminders_fn: MagicMock
    ) -> None:
        """GIVEN no reminders WHEN list_reminders THEN returns empty message."""
        mock_list_reminders_fn.return_value = []
        result = executor.execute("list_reminders", {})
        assert "no" in result.lower() or "[]" in result

    def test_set_reminder(
        self, executor: MetaToolExecutor, mock_set_reminder_fn: MagicMock
    ) -> None:
        """GIVEN set_reminder tool WHEN called THEN schedules via callable."""
        result = executor.execute(
            "set_reminder", {"text": "check build in 1 hour"}
        )
        mock_set_reminder_fn.assert_called_once_with("check build in 1 hour")
        assert "remind" in result.lower() or "hour" in result.lower()

    def test_delete_reminder(
        self, executor: MetaToolExecutor, mock_delete_reminder_fn: MagicMock
    ) -> None:
        """GIVEN delete_reminder tool WHEN called THEN deletes via callable."""
        result = executor.execute("delete_reminder", {"id": "abc123"})
        mock_delete_reminder_fn.assert_called_once_with("abc123")
        assert "delete" in result.lower()

    def test_delete_reminder_not_found(
        self, executor: MetaToolExecutor, mock_delete_reminder_fn: MagicMock
    ) -> None:
        """GIVEN nonexistent reminder WHEN delete_reminder THEN returns error."""
        mock_delete_reminder_fn.return_value = "Reminder xyz not found."
        result = executor.execute("delete_reminder", {"id": "xyz"})
        assert "not found" in result.lower()
```

**Step 2: Add tool schemas**

In `meta.py`, add after `cleanup_memories`:

```python
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List the user's pending reminders with IDs and messages.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Set a reminder using natural language time. "
                "Examples: 'check build in 30 minutes', 'deploy tomorrow at 3pm'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Reminder text with time, "
                            "e.g. 'check build in 1 hour'."
                        ),
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_reminder",
            "description": "Delete a reminder by its short hex ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The reminder's hex ID (e.g. 'abc123def456').",
                    },
                },
                "required": ["id"],
            },
        },
    },
```

**Step 3: Update executor __init__ to final form**

Replace the constructor in `meta.py` with the full version:

```python
    def __init__(
        self,
        *,
        db: LLMDatabase,
        context: ConversationContext,
        nick: str,
        channel: str,
        cleanup_fn: Callable[[], str] | None = None,
        list_reminders_fn: Callable[[], list] | None = None,
        set_reminder_fn: Callable[[str], str] | None = None,
        delete_reminder_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.db = db
        self.context = context
        self.nick = nick
        self.channel = channel
        self._cleanup_fn = cleanup_fn
        self._list_reminders_fn = list_reminders_fn
        self._set_reminder_fn = set_reminder_fn
        self._delete_reminder_fn = delete_reminder_fn
```

**Step 4: Add executor handlers**

```python
    def _tool_list_reminders(self, _args: dict[str, Any]) -> str:
        if self._list_reminders_fn is None:
            return json.dumps({"error": "Reminders are not available."})
        reminders = self._list_reminders_fn()
        if not reminders:
            return json.dumps(
                {"reminders": [], "message": "No pending reminders."}
            )
        return json.dumps({
            "reminders": [
                {
                    "id": name.split("_")[-1],
                    "message": data[2],
                    "channel": data[1],
                }
                for name, data in reminders
            ],
        })

    def _tool_set_reminder(self, args: dict[str, Any]) -> str:
        if self._set_reminder_fn is None:
            return json.dumps({"error": "Reminders are not available."})
        text = args["text"]
        result = self._set_reminder_fn(text)
        return json.dumps({"status": "ok", "message": result})

    def _tool_delete_reminder(self, args: dict[str, Any]) -> str:
        if self._delete_reminder_fn is None:
            return json.dumps({"error": "Reminders are not available."})
        reminder_id = args["id"]
        result = self._delete_reminder_fn(reminder_id)
        if "not found" in result.lower():
            return json.dumps({"error": result})
        return json.dumps({"status": "ok", "message": result})
```

**Step 5: Update service.py — add reminder callables to meta_completion**

Add the three new parameters to `meta_completion` signature:

```python
        cleanup_fn: Callable[[], str] | None = None,
        list_reminders_fn: Callable[[], list] | None = None,
        set_reminder_fn: Callable[[str], str] | None = None,
        delete_reminder_fn: Callable[[str], str] | None = None,
```

Update executor construction:

```python
            executor = MetaToolExecutor(
                db=db, context=context, nick=nick, channel=channel,
                cleanup_fn=cleanup_fn,
                list_reminders_fn=list_reminders_fn,
                set_reminder_fn=set_reminder_fn,
                delete_reminder_fn=delete_reminder_fn,
            )
```

**Step 6: Update plugin.py call sites with all callables**

Both `meta_completion` call sites need the full set. Update both:

```python
        result: MetaResult = self.llm_service.meta_completion(
            prompt=text,
            nick=preflight.nick,
            channel=preflight.channel,
            db=self.db,
            context=self.context,
            bot_nick=irc.nick,
            cleanup_fn=lambda: self._run_memory_cleanup(
                preflight.nick, preflight.channel
            ),
            list_reminders_fn=lambda: self._get_user_reminders(
                preflight.nick
            ),
            set_reminder_fn=lambda t: self._remind_set_for_meta(
                irc, msg, preflight.nick, t
            ),
            delete_reminder_fn=lambda rid: self._remind_delete_for_meta(
                preflight.nick, rid
            ),
        )
```

**Step 7: Update "not meta" help text in plugin.py**

In the `meta` command method, update the NOT_META reply:

```python
            irc.reply(
                _(
                    "I can manage your instructions, memories, "
                    "context, usage stats, and reminders. "
                    "Try: @meta list my memories"
                ),
                prefixNick=False,
            )
```

Also update the `meta` `CommandInfo` examples in `COMMAND_REGISTRY` to
include usage and reminder examples:

```python
        examples=(
            "%meta always respond in haiku",
            "%meta what are my memories?",
            "%meta how much have I used this month?",
            "%meta remind me to deploy in 2 hours",
            "%meta show my reminders",
        ),
```

**Step 8: Update tool count test**

```python
        assert len(META_TOOLS) == 15
```

**Step 9: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 10: Commit**

```bash
git add plugins/llm/src/llm/meta.py plugins/llm/src/llm/service.py \
    plugins/llm/src/llm/plugin.py plugins/llm/tests/test_meta.py
git commit -m "feat(meta): add reminder tools (list, set, delete) and update help text"
```

---

### Task 5: Integration Tests

**Files:**
- Modify: `plugins/llm/tests/test_meta.py`

**Step 1: Add integration tests**

Append to `TestMetaIntegration`:

```python
    def test_get_usage_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user asks about usage WHEN meta handles it THEN returns stats."""
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        db.log_usage("testuser", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("testuser", "#test", "ask", "gpt-4", 200, 100, 0.02)

        svc, _plugin = self._make_service(mocker)

        tool_call = mocker.MagicMock()
        tool_call.id = "call_usage"
        tool_call.function.name = "get_usage"
        tool_call.function.arguments = "{}"

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "You've made 2 requests costing $0.03."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        from llm.context import ContextConfig, ConversationContext

        result = svc.meta_completion(
            prompt="how much have I used?",
            nick="testuser",
            channel="#test",
            db=db,
            context=ConversationContext(ContextConfig(
                max_messages=20, timeout_minutes=5, channel_max_messages=10,
            )),
            bot_nick="VibeBot",
        )

        assert result.is_meta is True
        assert "2" in result.content
        db.close()

    def test_cleanup_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN cleanup_fn callable WHEN meta calls it THEN cleanup runs."""
        svc, _plugin = self._make_service(mocker)

        tool_call = mocker.MagicMock()
        tool_call.id = "call_cleanup"
        tool_call.function.name = "cleanup_memories"
        tool_call.function.arguments = "{}"

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Cleaned up your memories."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        from llm.context import ContextConfig, ConversationContext

        cleanup_fn = mocker.MagicMock(
            return_value="Before: 5 | dropped: 1 | after: 4"
        )

        result = svc.meta_completion(
            prompt="clean up my memories",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(ContextConfig(
                max_messages=20, timeout_minutes=5, channel_max_messages=10,
            )),
            bot_nick="VibeBot",
            cleanup_fn=cleanup_fn,
        )

        assert result.is_meta is True
        cleanup_fn.assert_called_once()

    def test_set_reminder_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN set_reminder callable WHEN meta calls it THEN reminder set."""
        svc, _plugin = self._make_service(mocker)

        tool_call = mocker.MagicMock()
        tool_call.id = "call_remind"
        tool_call.function.name = "set_reminder"
        tool_call.function.arguments = '{"text": "deploy in 2 hours"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Reminder set: deploy (in 2 hours)."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        from llm.context import ContextConfig, ConversationContext

        set_reminder_fn = mocker.MagicMock(
            return_value="I'll remind you in 2 hours"
        )

        result = svc.meta_completion(
            prompt="remind me to deploy in 2 hours",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(ContextConfig(
                max_messages=20, timeout_minutes=5, channel_max_messages=10,
            )),
            bot_nick="VibeBot",
            set_reminder_fn=set_reminder_fn,
        )

        assert result.is_meta is True
        set_reminder_fn.assert_called_once_with("deploy in 2 hours")
```

**Step 2: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_meta.py
git commit -m "test(meta): add integration tests for usage, cleanup, and reminder tools"
```

---

### Task 6: Final Preflight and Cleanup

**Step 1: Run full preflight**

Run: `make preflight`
Expected: All checks pass — lint, format, typecheck, tests (80%+ coverage)

**Step 2: Verify test coverage**

Run: `make test -- --cov-report=term-missing`
Check that `meta.py` has good coverage.

**Step 3: Review all changes**

Run: `git diff main --stat`

**Step 4: Commit if needed**

```bash
git commit -m "chore(meta): final cleanup for tools expansion"
```
