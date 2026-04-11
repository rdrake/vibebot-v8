"""Tests for the meta command tool definitions and executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.meta import META_TOOLS, MetaToolExecutor
from llm.plugin import LLM
from llm.service import LLMService, MetaResult

from .conftest import make_registry_side_effect, plugin_init_patches

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestMetaTools:
    """GIVEN the META_TOOLS list WHEN inspected THEN it has correct structure."""

    def test_tool_count(self) -> None:
        """GIVEN META_TOOLS WHEN counted THEN has expected number of tools."""
        assert len(META_TOOLS) == 15

    def test_tools_have_function_format(self) -> None:
        """GIVEN each tool WHEN checked THEN follows OpenAI function calling schema."""
        for tool in META_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


class TestMetaToolExecutor:
    """Tests for MetaToolExecutor dispatching tool calls."""

    @pytest.fixture
    def mock_db(self, mocker: MockerFixture) -> MagicMock:
        db = mocker.MagicMock()
        db.get_instruction.return_value = "respond in haiku"
        db.get_memories.return_value = [
            mocker.MagicMock(id=1, fact="likes Python"),
            mocker.MagicMock(id=2, fact="owns a cat"),
        ]
        db.save_instruction.return_value = None
        db.delete_instruction.return_value = True
        db.save_memory.return_value = 3
        db.delete_memory.return_value = True
        db.update_memory.return_value = True
        db.delete_all_memories.return_value = 2
        return db

    @pytest.fixture
    def mock_context(self, mocker: MockerFixture) -> MagicMock:
        ctx = mocker.MagicMock()
        ctx.clear.return_value = True
        return ctx

    @pytest.fixture
    def mock_cleanup_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = "Before: 8 | dropped: 2, merged: 4 \u2192 2 | after: 4"
        return fn

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

    def test_get_instruction(self, executor: MetaToolExecutor) -> None:
        """GIVEN get_instruction tool WHEN called THEN returns current instruction."""
        result = executor.execute("get_instruction", {})
        assert "respond in haiku" in result

    def test_set_instruction(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN set_instruction tool WHEN called THEN saves instruction."""
        result = executor.execute("set_instruction", {"text": "be brief"})
        mock_db.save_instruction.assert_called_once_with("testuser", "be brief")
        assert "ok" in result.lower()

    def test_clear_instruction(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_instruction tool WHEN called THEN deletes instruction."""
        result = executor.execute("clear_instruction", {})
        mock_db.delete_instruction.assert_called_once_with("testuser")
        assert "clear" in result.lower()

    def test_list_memories(self, executor: MetaToolExecutor) -> None:
        """GIVEN list_memories tool WHEN called THEN returns formatted memories."""
        result = executor.execute("list_memories", {})
        assert "likes Python" in result
        assert "owns a cat" in result

    def test_save_memory(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN save_memory tool WHEN called THEN saves to db."""
        result = executor.execute("save_memory", {"text": "prefers vim"})
        mock_db.save_memory.assert_called_once_with("testuser", "prefers vim", "#test")
        assert "saved" in result.lower() or "3" in result

    def test_delete_memory(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN delete_memory tool WHEN called THEN deletes by ID."""
        result = executor.execute("delete_memory", {"id": 1})
        mock_db.delete_memory.assert_called_once_with("testuser", 1)
        assert "delete" in result.lower()

    def test_delete_memory_not_found(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN delete_memory tool WHEN ID not found THEN returns error."""
        mock_db.delete_memory.return_value = False
        result = executor.execute("delete_memory", {"id": 999})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_update_memory(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN update_memory tool WHEN called THEN updates in db."""
        result = executor.execute("update_memory", {"id": 1, "text": "loves Python"})
        mock_db.update_memory.assert_called_once_with("testuser", 1, "loves Python")
        assert "update" in result.lower()

    def test_clear_memories(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_memories tool WHEN called THEN deletes all."""
        result = executor.execute("clear_memories", {})
        mock_db.delete_all_memories.assert_called_once_with("testuser")
        assert "2" in result  # count returned

    def test_forget_context(self, executor: MetaToolExecutor, mock_context: MagicMock) -> None:
        """GIVEN forget_context tool WHEN called THEN clears context for channel."""
        result = executor.execute("forget_context", {})
        mock_context.clear.assert_called_once_with("testuser", "#test")
        assert "clear" in result.lower()

    def test_unknown_tool(self, executor: MetaToolExecutor) -> None:
        """GIVEN unknown tool name WHEN called THEN returns error."""
        result = executor.execute("launch_missiles", {})
        assert "error" in result.lower() or "unknown" in result.lower()

    def test_executor_catches_exceptions(
        self, executor: MetaToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN tool raises exception WHEN executed THEN returns error string."""
        mock_db.get_memories.side_effect = RuntimeError("db error")
        result = executor.execute("list_memories", {})
        assert "error" in result.lower()

    def test_execute_denies_when_capability_missing(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_reminders_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_delete_reminder_fn: MagicMock,
    ) -> None:
        """GIVEN missing tool capability WHEN executed THEN dispatch denies it server-side."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            capabilities=frozenset(),
            cleanup_fn=mock_cleanup_fn,
            list_reminders_fn=mock_list_reminders_fn,
            set_reminder_fn=mock_set_reminder_fn,
            delete_reminder_fn=mock_delete_reminder_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.lower() or "capability" in result.lower()
        mock_db.get_memories.assert_not_called()

    def test_execute_denies_when_route_profile_not_visible(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_reminders_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_delete_reminder_fn: MagicMock,
    ) -> None:
        """GIVEN a hidden route profile WHEN executed THEN dispatch denies it server-side."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="draw",
            cleanup_fn=mock_cleanup_fn,
            list_reminders_fn=mock_list_reminders_fn,
            set_reminder_fn=mock_set_reminder_fn,
            delete_reminder_fn=mock_delete_reminder_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.lower() or "profile" in result.lower()
        mock_db.get_memories.assert_not_called()

    def test_get_usage(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
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

    def test_get_channel_usage(self, executor: MetaToolExecutor, mock_db: MagicMock) -> None:
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

    def test_cleanup_memories(self, executor: MetaToolExecutor, mock_cleanup_fn: MagicMock) -> None:
        """GIVEN cleanup_memories tool WHEN called THEN runs cleanup callable."""
        result = executor.execute("cleanup_memories", {})
        mock_cleanup_fn.assert_called_once_with("testuser")
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

    def test_list_memories_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN listing another user's memories THEN allowed."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        mock_db.get_memories.assert_called_with("someone")
        assert "someone" in result

    def test_list_memories_other_user_denied(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN non-owner WHEN listing another user's memories THEN denied."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        assert "owner" in result.lower()
        mock_db.get_memories.assert_not_called()

    def test_delete_memory_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN owner WHEN deleting another user's memory THEN allowed."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
        )
        executor.execute("delete_memory", {"id": 5, "nick": "someone"})
        mock_db.delete_memory.assert_called_once_with("someone", 5)

    def test_clear_memories_other_user_denied(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN non-owner WHEN clearing another user's memories THEN denied."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("clear_memories", {"nick": "someone"})
        assert "owner" in result.lower()
        mock_db.delete_all_memories.assert_not_called()

    def test_cleanup_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN cleaning up another user's memories THEN allowed."""
        executor = MetaToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        executor.execute("cleanup_memories", {"nick": "someone"})
        mock_cleanup_fn.assert_called_once_with("someone")

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
        result = executor.execute("set_reminder", {"text": "check build in 1 hour"})
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


# =========================================================================
# meta_completion() service-level tests
# =========================================================================


class TestMetaCompletion:
    """Tests for LLMService.meta_completion() tool-calling loop."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(metaModel="gpt-4")
        return svc

    def test_text_response_no_tools(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM returns text WHEN no tool calls THEN returns text."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done \u2014 instruction set."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            return_value=mock_response,
        )
        mocker.patch(
            "llm.service.litellm.completion_cost",
            return_value=0.001,
        )

        result = service.meta_completion(
            prompt="set my instruction to haiku",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 instruction set."
        assert result.is_meta is True
        assert result.error is None

    def test_not_meta_sentinel(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM returns NOT_META WHEN not config THEN is_meta=False."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "NOT_META"
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            return_value=mock_response,
        )

        result = service.meta_completion(
            prompt="what is the capital of France",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.is_meta is False

    def test_not_meta_exact_match(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM mentions NOT_META in longer text WHEN checked THEN not sentinel."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "I returned NOT_META because this isn't config."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            return_value=mock_response,
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.meta_completion(
            prompt="explain NOT_META",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        # Substring match would incorrectly flag this; exact match should not
        assert result.is_meta is True

    def test_tool_call_then_text(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM calls a tool then responds WHEN executed THEN tool runs."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "set_instruction"
        tool_call.function.arguments = '{"text": "respond in haiku"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Done \u2014 I'll respond in haiku."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mock_completion = mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        db = mocker.MagicMock()
        db.save_instruction.return_value = None

        result = service.meta_completion(
            prompt="always respond in haiku",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 I'll respond in haiku."
        assert result.is_meta is True
        db.save_instruction.assert_called_once_with("testuser", "respond in haiku")
        assert mock_completion.call_count == 2

    def test_max_steps_exceeded(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM keeps calling tools WHEN max steps hit THEN error."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_loop"
        tool_call.function.name = "list_memories"
        tool_call.function.arguments = "{}"

        loop_response = mocker.MagicMock()
        loop_choice = mocker.MagicMock()
        loop_choice.message.content = None
        loop_choice.message.tool_calls = [tool_call]
        loop_choice.message.role = "assistant"
        loop_response.choices = [loop_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            return_value=loop_response,
        )

        db = mocker.MagicMock()
        db.get_memories.return_value = []

        result = service.meta_completion(
            prompt="do something",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.error is not None

    def test_api_error_returns_error_result(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN LLM API fails WHEN called THEN returns error result."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("API down"),
        )

        result = service.meta_completion(
            prompt="list my memories",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.error is not None
        assert result.is_meta is True

    def test_parallel_tool_calls(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN LLM returns multiple tool calls WHEN executed THEN all run."""
        call_1 = mocker.MagicMock()
        call_1.id = "call_del_1"
        call_1.function.name = "delete_memory"
        call_1.function.arguments = '{"id": 14}'

        call_2 = mocker.MagicMock()
        call_2.id = "call_del_2"
        call_2.function.name = "delete_memory"
        call_2.function.arguments = '{"id": 27}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [call_1, call_2]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Deleted 2 memories."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        db = mocker.MagicMock()
        db.delete_memory.return_value = True

        result = service.meta_completion(
            prompt="delete memories 14 and 27",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Deleted 2 memories."
        assert db.delete_memory.call_count == 2

    def test_cost_is_populated(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN meta completion WHEN successful THEN cost is calculated."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            return_value=mock_response,
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.005)

        result = service.meta_completion(
            prompt="get instruction",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.cost > 0


# =========================================================================
# Plugin-level command tests
# =========================================================================


class TestMetaCommand:
    """Tests for the @meta IRC command in plugin.py."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        """Create an LLM plugin with mocked dependencies."""
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": True})
        )
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        return plugin

    def test_meta_calls_service(self, plugin, mocker: MockerFixture, mock_irc: MagicMock) -> None:
        """GIVEN @meta command WHEN invoked THEN calls meta_completion."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test", "@meta set my instruction"]
        msg.time = float("inf")  # Not a replay

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="Done \u2014 instruction set.",
            is_meta=True,
        )
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )

        plugin.meta(mock_irc, msg, ["set", "my", "instruction"])

        plugin.llm_service.meta_completion.assert_called_once()

    def test_meta_disabled(self, plugin, mocker: MockerFixture, mock_irc: MagicMock) -> None:
        """GIVEN metaEnabled=False WHEN @meta invoked THEN error reply."""
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": False})
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test", "@meta do something"]
        msg.time = float("inf")

        plugin.meta(mock_irc, msg, ["do", "something"])

        mock_irc.reply.assert_called()

    def test_meta_not_meta_does_not_echo_sentinel(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN explicit @meta WHEN NOT_META returned THEN helpful message."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]
        msg.time = float("inf")

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="NOT_META",
            is_meta=False,
        )
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )

        plugin.meta(mock_irc, msg, ["what", "is", "Python"])

        # Should NOT echo "NOT_META" — should give a helpful message
        call_args = mock_irc.reply.call_args[0][0]
        assert "NOT_META" not in call_args

    def test_meta_uses_ask_rate_limit(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN @meta command WHEN preflight runs THEN uses ask rate limit."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]
        msg.time = float("inf")

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="Done.",
            is_meta=True,
        )
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )

        plugin.meta(mock_irc, msg, ["set", "instruction"])

        # Preflight should be called with command="ask" for rate limiting
        plugin._run_preflight.assert_called_once()
        # Fourth positional arg is the command name
        call_args = plugin._run_preflight.call_args[0]
        assert call_args[3] == "ask"


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

    def test_remind_delete_for_meta_success(self, plugin, mocker: MockerFixture) -> None:
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


class TestInvalidCommandMetaFallback:
    """Tests for invalidCommand routing through the shared ask-style path."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": True})
        )
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        return plugin

    def test_not_meta_falls_through_to_ask(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN invalidCommand runs THEN _ask_impl called directly."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["what", "is", "python"])

        plugin._ask_impl.assert_called_once()
        plugin.llm_service.meta_completion.assert_not_called()

    def test_meta_disabled_skips_to_ask(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN metaEnabled=False WHEN unknown command THEN still uses _ask_impl."""
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": False})
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["hello", "there"])

        plugin._ask_impl.assert_called_once()
        plugin.llm_service.meta_completion.assert_not_called()

    def test_meta_handled_does_not_call_ask(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN routed THEN meta is not consulted."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#test",
                account=None,
            )
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["always", "respond", "in", "haiku"])

        plugin._ask_impl.assert_called_once()
        plugin.llm_service.meta_completion.assert_not_called()


# =========================================================================
# Integration tests (real DB + context)
# =========================================================================


class TestMetaIntegration:
    """End-to-end integration tests for the meta feature."""

    def test_set_instruction_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user says 'always respond in haiku' WHEN routed through meta
        THEN instruction is saved and confirmation returned."""
        from llm.context import ContextConfig, ConversationContext
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=5,
            channel_max_messages=10,
        )
        context = ConversationContext(config)

        svc, _plugin = self._make_service(mocker)

        # Simulate: tool call -> set_instruction -> text response
        tool_call = mocker.MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "set_instruction"
        tool_call.function.arguments = '{"text": "always respond in haiku"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Done \u2014 I'll respond in haiku."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        result = svc.meta_completion(
            prompt="always respond in haiku",
            nick="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )

        assert result.is_meta is True
        assert "haiku" in result.content.lower()
        assert db.get_instruction("testuser") == "always respond in haiku"

        db.close()

    def test_list_and_delete_memories_via_meta(self, mocker: MockerFixture) -> None:
        """GIVEN user has memories WHEN meta deletes by topic THEN removed."""
        from llm.context import ContextConfig, ConversationContext
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=5,
            channel_max_messages=10,
        )
        context = ConversationContext(config)

        id1 = db.save_memory("testuser", "likes cats", "#test")
        id2 = db.save_memory("testuser", "owns two cats", "#test")
        _id3 = db.save_memory("testuser", "uses vim", "#test")

        svc, _plugin = self._make_service(mocker)

        # Simulate: list -> delete two cat memories -> confirmation
        list_call = mocker.MagicMock()
        list_call.id = "call_list"
        list_call.function.name = "list_memories"
        list_call.function.arguments = "{}"

        r1 = mocker.MagicMock()
        c1 = mocker.MagicMock()
        c1.message.content = None
        c1.message.tool_calls = [list_call]
        c1.message.role = "assistant"
        r1.choices = [c1]

        del_call_1 = mocker.MagicMock()
        del_call_1.id = "call_del_1"
        del_call_1.function.name = "delete_memory"
        del_call_1.function.arguments = f'{{"id": {id1}}}'

        del_call_2 = mocker.MagicMock()
        del_call_2.id = "call_del_2"
        del_call_2.function.name = "delete_memory"
        del_call_2.function.arguments = f'{{"id": {id2}}}'

        r2 = mocker.MagicMock()
        c2 = mocker.MagicMock()
        c2.message.content = None
        c2.message.tool_calls = [del_call_1, del_call_2]
        c2.message.role = "assistant"
        r2.choices = [c2]

        r3 = mocker.MagicMock()
        c3 = mocker.MagicMock()
        c3.message.content = "Deleted 2 memories about cats."
        c3.message.tool_calls = None
        r3.choices = [c3]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[r1, r2, r3],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        result = svc.meta_completion(
            prompt="delete any memories about cats",
            nick="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )

        assert result.is_meta is True
        assert "cat" in result.content.lower()

        remaining = db.get_memories("testuser")
        assert len(remaining) == 1
        assert remaining[0].fact == "uses vim"

        db.close()

    def test_usage_logged_with_meta_command(self) -> None:
        """GIVEN a meta call WHEN usage logged THEN recorded as 'meta'."""
        from llm.persistence import LLMDatabase

        db = LLMDatabase(":memory:")
        db.log_usage(
            nick="testuser",
            channel="#test",
            command="meta",
            model="gemini/gemini-2.0-flash",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
        )

        summary = db.get_usage_summary_for_nick("testuser")
        assert summary.total_requests == 1

        db.close()

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
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
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

        cleanup_fn = mocker.MagicMock(return_value="Before: 5 | dropped: 1 | after: 4")

        result = svc.meta_completion(
            prompt="clean up my memories",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
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

        set_reminder_fn = mocker.MagicMock(return_value="I'll remind you in 2 hours")

        result = svc.meta_completion(
            prompt="remind me to deploy in 2 hours",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=ConversationContext(
                ContextConfig(
                    max_messages=20,
                    timeout_minutes=5,
                    channel_max_messages=10,
                )
            ),
            bot_nick="VibeBot",
            set_reminder_fn=set_reminder_fn,
        )

        assert result.is_meta is True
        set_reminder_fn.assert_called_once_with("deploy in 2 hours")

    @staticmethod
    def _make_service(mocker: MockerFixture) -> tuple:
        """Create an LLMService with meta config defaults."""
        plugin = mocker.MagicMock()
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaModel": "gpt-4"})
        )
        plugin.log = mocker.Mock()
        return LLMService(plugin), plugin
