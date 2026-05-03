"""Tests for the meta command tool definitions and executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.assistant import (
    ASSISTANT_TOOL_SPECS,
    ASSISTANT_TOOLS,
    CHAT_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    DRAW_SYSTEM_PROMPT,
    REMIND_ACTION_SYSTEM_PROMPT,
    AssistantToolExecutor,
    ToolResult,
    get_tools_for_profile,
)
from llm.plugin import LLM, Identity
from llm.service import LLMService

from .conftest import make_registry_side_effect, make_reminder_row, plugin_init_patches

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestMetaTools:
    """GIVEN the ASSISTANT_TOOLS list WHEN inspected THEN it has correct structure."""

    def test_tool_count(self) -> None:
        """GIVEN ASSISTANT_TOOLS WHEN counted THEN has expected number of tools."""
        assert len(ASSISTANT_TOOLS) == 21

    def test_tools_have_function_format(self) -> None:
        """GIVEN each tool WHEN checked THEN follows OpenAI function calling schema."""
        for tool in ASSISTANT_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn


class TestAssistantToolExecutor:
    """Tests for AssistantToolExecutor dispatching tool calls."""

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
    def mock_list_pending_tasks_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = [
            {
                "kind": "reminder",
                "id": "abc123",
                "channel": "#test",
                "description": "check build",
            },
            {
                "kind": "reminder",
                "id": "def456",
                "channel": "#test",
                "description": "deploy app",
            },
        ]
        return fn

    @pytest.fixture
    def mock_set_reminder_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = "I'll remind you in 1 hour"
        return fn

    @pytest.fixture
    def mock_cancel_pending_task_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = {
            "status": "ok",
            "kind": "reminder",
            "id": "abc123",
            "message": "Deleted reminder abc123.",
        }
        return fn

    @pytest.fixture
    def mock_cancel_all_pending_tasks_fn(self, mocker: MockerFixture) -> MagicMock:
        fn = mocker.MagicMock()
        fn.return_value = {
            "reminders_message": "Cancelled 2 reminders.",
            "scheduled_tasks_cancelled": 0,
        }
        return fn

    @pytest.fixture
    def executor(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
        mock_cancel_all_pending_tasks_fn: MagicMock,
    ) -> AssistantToolExecutor:
        return AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
            cancel_all_pending_tasks_fn=mock_cancel_all_pending_tasks_fn,
        )

    def test_get_instruction(self, executor: AssistantToolExecutor) -> None:
        """GIVEN get_instruction tool WHEN called THEN returns current instruction."""
        result = executor.execute("get_instruction", {})
        assert "respond in haiku" in result.content

    def test_set_instruction(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN set_instruction tool WHEN called THEN saves instruction."""
        result = executor.execute("set_instruction", {"text": "be brief"})
        mock_db.save_instruction.assert_called_once_with("testuser", "be brief")
        assert "ok" in result.content.lower()

    def test_clear_instruction(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_instruction tool WHEN called THEN deletes instruction."""
        result = executor.execute("clear_instruction", {})
        mock_db.delete_instruction.assert_called_once_with("testuser")
        assert "clear" in result.content.lower()

    def test_list_memories(self, executor: AssistantToolExecutor) -> None:
        """GIVEN list_memories tool WHEN called THEN returns formatted memories."""
        result = executor.execute("list_memories", {})
        assert "likes Python" in result.content
        assert "owns a cat" in result.content

    def test_save_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN save_memory tool WHEN called THEN saves to db."""
        result = executor.execute("save_memory", {"text": "prefers vim"})
        mock_db.save_memory.assert_called_once_with("testuser", "prefers vim", "#test")
        assert "saved" in result.content.lower() or "3" in result.content

    def test_delete_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN delete_memory tool WHEN called THEN deletes by ID."""
        result = executor.execute("delete_memory", {"id": 1})
        mock_db.delete_memory.assert_called_once_with("testuser", 1)
        assert "delete" in result.content.lower()

    def test_delete_memory_not_found(
        self, executor: AssistantToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN delete_memory tool WHEN ID not found THEN returns error."""
        mock_db.delete_memory.return_value = False
        result = executor.execute("delete_memory", {"id": 999})
        assert "not found" in result.content.lower() or "error" in result.content.lower()

    def test_update_memory(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN update_memory tool WHEN called THEN updates in db."""
        result = executor.execute("update_memory", {"id": 1, "text": "loves Python"})
        mock_db.update_memory.assert_called_once_with("testuser", 1, "loves Python")
        assert "update" in result.content.lower()

    def test_clear_memories(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN clear_memories tool WHEN called THEN deletes all."""
        result = executor.execute("clear_memories", {})
        mock_db.delete_all_memories.assert_called_once_with("testuser")
        assert "2" in result.content  # count returned

    def test_forget_context(self, executor: AssistantToolExecutor, mock_context: MagicMock) -> None:
        """GIVEN forget_context tool WHEN called THEN clears context for channel."""
        result = executor.execute("forget_context", {})
        mock_context.clear.assert_called_once_with("testuser", "#test")
        assert "clear" in result.content.lower()

    def test_unknown_tool(self, executor: AssistantToolExecutor) -> None:
        """GIVEN unknown tool name WHEN called THEN returns error."""
        result = executor.execute("launch_missiles", {})
        assert "error" in result.content.lower() or "unknown" in result.content.lower()

    def test_executor_catches_exceptions(
        self, executor: AssistantToolExecutor, mock_db: MagicMock
    ) -> None:
        """GIVEN tool raises exception WHEN executed THEN returns error string."""
        mock_db.get_memories.side_effect = RuntimeError("db error")
        result = executor.execute("list_memories", {})
        assert "error" in result.content.lower()

    def test_execute_denies_when_capability_missing(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN missing tool capability WHEN executed THEN dispatch denies it server-side."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            capabilities=frozenset(),
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.content.lower() or "capability" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_execute_denies_when_route_profile_not_visible(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
        mock_cleanup_fn: MagicMock,
        mock_list_pending_tasks_fn: MagicMock,
        mock_set_reminder_fn: MagicMock,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN a hidden route profile WHEN executed THEN dispatch denies it server-side."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="draw",
            cleanup_fn=mock_cleanup_fn,
            list_pending_tasks_fn=mock_list_pending_tasks_fn,
            set_reminder_fn=mock_set_reminder_fn,
            cancel_pending_task_fn=mock_cancel_pending_task_fn,
        )

        result = executor.execute("list_memories", {})

        assert "not allowed" in result.content.lower() or "profile" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_get_usage(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN get_usage tool WHEN called THEN returns user's usage summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=47,
            total_prompt_tokens=12000,
            total_completion_tokens=3000,
            total_cost=0.12,
        )
        result = executor.execute("get_usage", {})
        assert "47" in result.content
        assert "0.12" in result.content
        mock_db.get_usage_summary_for_nick.assert_called_once()

    def test_get_channel_usage(self, executor: AssistantToolExecutor, mock_db: MagicMock) -> None:
        """GIVEN get_channel_usage tool WHEN called THEN returns channel summary."""
        from llm.persistence import UsageSummary

        mock_db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=200,
            total_prompt_tokens=50000,
            total_completion_tokens=10000,
            total_cost=0.85,
        )
        result = executor.execute("get_channel_usage", {})
        assert "200" in result.content
        assert "0.85" in result.content
        mock_db.get_usage_summary_for_channel.assert_called_once()

    def test_cleanup_memories(
        self, executor: AssistantToolExecutor, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN cleanup_memories tool WHEN called THEN runs cleanup callable."""
        result = executor.execute("cleanup_memories", {})
        mock_cleanup_fn.assert_called_once_with("testuser")
        assert "Before: 8" in result.content

    def test_cleanup_memories_not_available(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN no cleanup_fn WHEN cleanup_memories called THEN returns error."""
        executor = AssistantToolExecutor(
            db=mock_db, context=mock_context, nick="testuser", channel="#test"
        )
        result = executor.execute("cleanup_memories", {})
        assert "not available" in result.content.lower() or "error" in result.content.lower()

    def test_list_memories_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN listing another user's memories THEN allowed."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        mock_db.get_memories.assert_called_with("someone")
        assert "someone" in result.content

    def test_list_memories_other_user_denied(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN non-owner WHEN listing another user's memories THEN denied."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("list_memories", {"nick": "someone"})
        assert "owner" in result.content.lower()
        mock_db.get_memories.assert_not_called()

    def test_delete_memory_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """GIVEN owner WHEN deleting another user's memory THEN allowed."""
        executor = AssistantToolExecutor(
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
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="regular",
            channel="#test",
        )
        result = executor.execute("clear_memories", {"nick": "someone"})
        assert "owner" in result.content.lower()
        mock_db.delete_all_memories.assert_not_called()

    def test_cleanup_other_user_as_owner(
        self, mock_db: MagicMock, mock_context: MagicMock, mock_cleanup_fn: MagicMock
    ) -> None:
        """GIVEN owner WHEN cleaning up another user's memories THEN allowed."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="owner",
            channel="#test",
            is_owner=True,
            cleanup_fn=mock_cleanup_fn,
        )
        executor.execute("cleanup_memories", {"nick": "someone"})
        mock_cleanup_fn.assert_called_once_with("someone")

    def test_list_pending_tasks(
        self, executor: AssistantToolExecutor, mock_list_pending_tasks_fn: MagicMock
    ) -> None:
        """GIVEN list_pending_tasks tool WHEN called THEN returns merged list."""
        result = executor.execute("list_pending_tasks", {})
        mock_list_pending_tasks_fn.assert_called_once()
        assert "check build" in result.content
        assert "deploy app" in result.content
        assert "abc123" in result.content

    def test_list_pending_tasks_empty(
        self, executor: AssistantToolExecutor, mock_list_pending_tasks_fn: MagicMock
    ) -> None:
        """GIVEN no tasks WHEN list_pending_tasks THEN returns empty message."""
        mock_list_pending_tasks_fn.return_value = []
        result = executor.execute("list_pending_tasks", {})
        assert "no" in result.content.lower() or "[]" in result.content

    def test_set_reminder(
        self, executor: AssistantToolExecutor, mock_set_reminder_fn: MagicMock
    ) -> None:
        """GIVEN set_reminder tool WHEN called THEN schedules via callable."""
        result = executor.execute("set_reminder", {"text": "check build in 1 hour"})
        mock_set_reminder_fn.assert_called_once_with("check build in 1 hour")
        assert "remind" in result.content.lower() or "hour" in result.content.lower()

    def test_cancel_pending_task(
        self, executor: AssistantToolExecutor, mock_cancel_pending_task_fn: MagicMock
    ) -> None:
        """GIVEN cancel_pending_task tool WHEN called THEN dispatches via callable."""
        result = executor.execute("cancel_pending_task", {"id": "abc123"})
        mock_cancel_pending_task_fn.assert_called_once_with("abc123")
        assert "delete" in result.content.lower()

    def test_cancel_pending_task_not_found(
        self,
        executor: AssistantToolExecutor,
        mock_cancel_pending_task_fn: MagicMock,
    ) -> None:
        """GIVEN nonexistent task WHEN cancel_pending_task THEN returns error."""
        mock_cancel_pending_task_fn.return_value = {
            "status": "error",
            "kind": "reminder",
            "id": "xyz",
            "message": "Reminder xyz not found.",
        }
        result = executor.execute("cancel_pending_task", {"id": "xyz"})
        assert "not found" in result.content.lower()

    def test_cancel_all_pending_tasks(
        self,
        executor: AssistantToolExecutor,
        mock_cancel_all_pending_tasks_fn: MagicMock,
    ) -> None:
        """GIVEN cancel_all_pending_tasks tool WHEN called THEN dispatches to callable."""
        result = executor.execute("cancel_all_pending_tasks", {})
        mock_cancel_all_pending_tasks_fn.assert_called_once_with()
        assert "2" in result.content
        assert "ok" in result.content.lower()

    def test_cancel_all_pending_tasks_unavailable(
        self,
        mock_db: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """GIVEN no callable WHEN cancel_all_pending_tasks THEN returns error."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
        )
        result = executor.execute("cancel_all_pending_tasks", {})
        assert "not available" in result.content.lower()

    # -- Task 6: Structured returns and new callables ----------------------

    def test_executor_accepts_search_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts search_fn callable."""
        search_fn = mocker.MagicMock(return_value=ToolResult(content="results"))
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            search_fn=search_fn,
        )
        assert executor._search_fn is search_fn

    def test_executor_accepts_fetch_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts fetch_fn callable."""
        fetch_fn = mocker.MagicMock(return_value=ToolResult(content="page"))
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            fetch_fn=fetch_fn,
        )
        assert executor._fetch_fn is fetch_fn

    def test_executor_accepts_code_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """AssistantToolExecutor accepts code_fn callable."""
        code_fn = mocker.MagicMock(return_value=ToolResult(content="code"))
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            code_fn=code_fn,
        )
        assert executor._code_fn is code_fn

    def test_execute_returns_tool_result(self, executor: AssistantToolExecutor) -> None:
        """execute() returns ToolResult for existing tools."""
        result = executor.execute("get_instruction", {})
        assert isinstance(result, ToolResult)
        assert "respond in haiku" in result.content

    def test_execute_denied_returns_tool_result(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """execute() returns ToolResult for denied tools."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            capabilities=frozenset(),
        )
        result = executor.execute("list_memories", {})
        assert isinstance(result, ToolResult)
        assert "capability" in result.content.lower() or "not allowed" in result.content.lower()

    def test_executor_tracks_grounding_used(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """Executor sets grounding_used when tool returns grounding_used=True."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(content="web results", grounding_used=True),
        )
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        assert executor.grounding_used is False
        executor.execute("search_web", {"query": "test"})
        assert executor.grounding_used is True

    def test_executor_accumulates_cost(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """Executor accumulates cost from ToolResult."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(
                content="results",
                prompt_tokens=100,
                completion_tokens=50,
                cost=0.01,
            ),
        )
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        executor.execute("search_web", {"query": "a"})
        executor.execute("search_web", {"query": "b"})
        assert executor.accumulated_prompt_tokens == 200
        assert executor.accumulated_completion_tokens == 100
        assert executor.accumulated_cost == pytest.approx(0.02)

    # -- Task 7: Tool handlers --------------------------------------------

    def test_tool_search_web_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """search_web handler calls search_fn with query."""
        search_fn = mocker.MagicMock(
            return_value=ToolResult(content="web results"),
        )
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            search_fn=search_fn,
        )
        result = executor.execute("search_web", {"query": "python async"})
        search_fn.assert_called_once_with("python async")
        assert result.content == "web results"

    def test_tool_search_web_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """search_web returns error when search_fn is None."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
        )
        result = executor.execute("search_web", {"query": "test"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()

    def test_tool_fetch_url_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """fetch_url handler calls fetch_fn with url."""
        fetch_fn = mocker.MagicMock(
            return_value=ToolResult(content="page content"),
        )
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
            fetch_fn=fetch_fn,
        )
        result = executor.execute("fetch_url", {"url": "https://example.com"})
        fetch_fn.assert_called_once_with("https://example.com")
        assert result.content == "page content"

    def test_tool_fetch_url_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """fetch_url returns error when fetch_fn is None."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="chat",
        )
        result = executor.execute("fetch_url", {"url": "https://example.com"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()

    def test_tool_generate_code_calls_fn(
        self, mock_db: MagicMock, mock_context: MagicMock, mocker: MockerFixture
    ) -> None:
        """generate_code handler calls code_fn with prompt."""
        code_fn = mocker.MagicMock(
            return_value=ToolResult(content="https://paste.example/abc"),
        )
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="code",
            capabilities=frozenset({"llm.ask", "llm.code"}),
            code_fn=code_fn,
        )
        result = executor.execute("generate_code", {"prompt": "fizzbuzz in rust"})
        code_fn.assert_called_once_with("fizzbuzz in rust")
        assert result.content == "https://paste.example/abc"

    def test_tool_generate_code_no_fn_returns_error(
        self, mock_db: MagicMock, mock_context: MagicMock
    ) -> None:
        """generate_code returns error when code_fn is None."""
        executor = AssistantToolExecutor(
            db=mock_db,
            context=mock_context,
            nick="testuser",
            channel="#test",
            route_profile="code",
            capabilities=frozenset({"llm.ask", "llm.code"}),
        )
        result = executor.execute("generate_code", {"prompt": "hello world"})
        assert "unavailable" in result.content.lower() or "error" in result.content.lower()


# =========================================================================
# assistant_completion() service-level tests
# =========================================================================


class TestMetaCompletion:
    """Tests for LLMService.assistant_completion() tool-calling loop."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
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

        result = service.assistant_completion(
            prompt="set my instruction to haiku",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 instruction set."
        assert result.error is None

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

        result = service.assistant_completion(
            prompt="always respond in haiku",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Done \u2014 I'll respond in haiku."
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

        result = service.assistant_completion(
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

        result = service.assistant_completion(
            prompt="list my memories",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.error is not None

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

        result = service.assistant_completion(
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

        result = service.assistant_completion(
            prompt="get instruction",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.cost > 0

    def test_assistant_completion_accepts_system_prompt(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion uses provided system_prompt instead of META_SYSTEM_PROMPT."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_completion)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            system_prompt="You are a helpful assistant named {bot_nick}.",
        )

        assert captured_messages[0]["role"] == "system"
        assert "helpful assistant" in captured_messages[0]["content"]
        assert "VibeBot" in captured_messages[0]["content"]

    def test_assistant_completion_defaults_to_chat_prompt(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion uses CHAT_SYSTEM_PROMPT when no system_prompt given."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        captured_messages: list = []

        def capture_completion(**kwargs: object) -> object:
            captured_messages.extend(kwargs.get("messages", []))  # type: ignore[union-attr]
            return mock_response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_completion)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert captured_messages[0]["role"] == "system"
        assert captured_messages[0]["content"] == CHAT_SYSTEM_PROMPT.format(bot_nick="VibeBot")

    def test_assistant_completion_passes_search_fn_to_executor(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion passes search_fn/fetch_fn/code_fn to AssistantToolExecutor."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        mock_executor_cls = mocker.patch("llm.assistant.AssistantToolExecutor")

        sentinel_search = mocker.Mock()
        sentinel_fetch = mocker.Mock()
        sentinel_code = mocker.Mock()

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            search_fn=sentinel_search,
            fetch_fn=sentinel_fetch,
            code_fn=sentinel_code,
        )

        call_kwargs = mock_executor_cls.call_args[1]
        assert call_kwargs["search_fn"] is sentinel_search
        assert call_kwargs["fetch_fn"] is sentinel_fetch
        assert call_kwargs["code_fn"] is sentinel_code

    def test_explicit_search_prompt_forces_search_web_tool_choice(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """assistant_completion forces search_web first for explicit current-info prompts."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_search"
        tool_call.function.name = "search_web"
        tool_call.function.arguments = '{"query": "latest nefarious 2 release"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Nefarious 2 details..."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mock_completion = mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.assistant_completion(
            prompt="search for the latest nefarious 2 release",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            search_fn=lambda _query: ToolResult(content="search results"),
        )

        assert result.content == "Nefarious 2 details..."
        first_kwargs = mock_completion.call_args_list[0].kwargs
        assert first_kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "search_web"},
        }
        assert "tool_choice" not in mock_completion.call_args_list[1].kwargs

    def test_meta_result_includes_grounding_used(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """AssistantResult.grounding_used reflects executor state after tool calls."""
        # First response: tool call; second response: text
        tool_call = mocker.MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "web_search"
        tool_call.function.arguments = '{"query": "test"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Here are results."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        # Mock the executor to set grounding_used when execute is called
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = True
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mock_executor.execute.return_value = ToolResult(
            content='{"results": []}', grounding_used=True
        )
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        result = service.assistant_completion(
            prompt="search for test",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.grounding_used is True

    def test_meta_result_includes_leaf_tool_costs(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """AssistantResult totals include costs from leaf tool calls."""
        # First response: tool call; second response: text
        tool_call = mocker.MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "web_search"
        tool_call.function.arguments = '{"query": "test"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Here are results."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        # Mock executor with accumulated leaf costs
        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 200
        mock_executor.accumulated_completion_tokens = 100
        mock_executor.accumulated_cost = 0.05
        mock_executor.execute.return_value = ToolResult(content='{"results": []}')
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        # _extract_usage returns (10, 5, 0.001) for each LLM call
        mocker.patch.object(service, "_extract_usage", return_value=(10, 5, 0.001))

        result = service.assistant_completion(
            prompt="search for test",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        # LLM loop: 2 calls * (10, 5, 0.001) = (20, 10, 0.002)
        # Executor: (200, 100, 0.05)
        # Total: (220, 110, 0.052)
        assert result.prompt_tokens == 220
        assert result.completion_tokens == 110
        assert result.cost == pytest.approx(0.052)

    @pytest.mark.parametrize(
        ("route_profile", "expected_task_type"),
        [("chat", "ask"), ("code", "code")],
    )
    def test_timeout_stashes_for_chat_and_code(
        self,
        service: LLMService,
        mocker: MockerFixture,
        route_profile: str,
        expected_task_type: str,
    ) -> None:
        """Timeout in assistant_completion stashes via _stash_timeout for ask/code routes."""
        import litellm as litellm_module

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm_module.Timeout(
                message="timed out", model="gpt-4", llm_provider="openai"
            ),
        )
        stash_mock = mocker.patch.object(service, "_stash_timeout", return_value=True)

        result = service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=route_profile,
        )

        stash_mock.assert_called_once()
        call_kwargs = stash_mock.call_args.kwargs
        assert call_kwargs["task_type"] == expected_task_type
        assert call_kwargs["prompt"] == "hello"
        assert "messages" in call_kwargs["request_data"]
        assert result.error is None
        assert "deliver the answer when ready" in result.content

    def test_timeout_without_expiry_returns_error(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """When _stash_timeout returns False (expiry disabled), assistant returns an error."""
        import litellm as litellm_module

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm_module.Timeout(
                message="timed out", model="gpt-4", llm_provider="openai"
            ),
        )
        mocker.patch.object(service, "_stash_timeout", return_value=False)

        result = service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile="chat",
        )

        assert result.error is not None
        assert "something went wrong" in result.content.lower()

    def test_timeout_stash_messages_unaffected_by_tool_loop(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Stashed request_data captures the initial messages, not the tool-loop-mutated list."""
        import litellm as litellm_module

        # First call returns a tool_calls response (mutates `messages`),
        # second call raises Timeout. The stash should still see the
        # pre-loop messages list.
        tool_call = mocker.MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "list_memories"
        tool_call.function.arguments = "{}"

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_response.choices = [first_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[
                first_response,
                litellm_module.Timeout(message="timed out", model="gpt-4", llm_provider="openai"),
            ],
        )
        stash_mock = mocker.patch.object(service, "_stash_timeout", return_value=True)

        db = mocker.MagicMock()
        db.get_memories.return_value = []

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=db,
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile="chat",
        )

        stashed_messages = stash_mock.call_args.kwargs["request_data"]["messages"]
        # Pre-loop snapshot has system + user (no assistant tool_calls or tool result)
        roles = [m["role"] for m in stashed_messages]
        assert "tool" not in roles
        assert not any(m.get("tool_calls") for m in stashed_messages)

    def test_extra_tools_appended_to_profile_tools(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """extra_tools are included in the tools kwarg passed to litellm.completion."""
        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mock_completion = mocker.patch(
            "llm.service.litellm.completion",
            return_value=mock_response,
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        fake_schema = {
            "type": "function",
            "function": {
                "name": "run_limnoria_command",
                "description": "Run a Limnoria command",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_tools=[fake_schema],
        )

        assert mock_completion.call_count == 1
        called_tools = mock_completion.call_args.kwargs["tools"]
        assert fake_schema in called_tools

    def test_extra_handlers_dispatched_before_executor(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """When a tool name is in extra_handlers, the handler runs instead of the executor."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_bridge"
        tool_call.function.name = "run_limnoria_command"
        tool_call.function.arguments = '{"plugin": "Misc", "command": "ping"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Pong."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "reply": "pong"}')
        )

        service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        handler.assert_called_once_with({"plugin": "Misc", "command": "ping"})
        # executor.execute must NOT have been called for this tool name
        for call in mock_executor.execute.call_args_list:
            assert call.args[0] != "run_limnoria_command"

    def test_extra_handlers_error_envelope_does_not_set_last_successful_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Handler returning an error envelope leaves last_successful_tool as None."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_bridge"
        tool_call.function.name = "run_limnoria_command"
        tool_call.function.arguments = '{"plugin": "Misc", "command": "ping"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Sorry, that failed."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"error": "denied: Misc.ping"}')
        )

        result = service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        assert result.last_successful_tool is None

    def test_extra_handlers_ok_envelope_sets_last_successful_tool(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Handler returning a success envelope sets last_successful_tool on the result."""
        tool_call = mocker.MagicMock()
        tool_call.id = "call_bridge"
        tool_call.function.name = "run_limnoria_command"
        tool_call.function.arguments = '{"plugin": "Misc", "command": "ping"}'

        first_response = mocker.MagicMock()
        first_choice = mocker.MagicMock()
        first_choice.message.content = None
        first_choice.message.tool_calls = [tool_call]
        first_choice.message.role = "assistant"
        first_response.choices = [first_choice]

        second_response = mocker.MagicMock()
        second_choice = mocker.MagicMock()
        second_choice.message.content = "Pong."
        second_choice.message.tool_calls = None
        second_response.choices = [second_choice]

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[first_response, second_response],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        mock_executor = mocker.MagicMock()
        mock_executor.grounding_used = False
        mock_executor.accumulated_prompt_tokens = 0
        mock_executor.accumulated_completion_tokens = 0
        mock_executor.accumulated_cost = 0.0
        mocker.patch("llm.assistant.AssistantToolExecutor", return_value=mock_executor)

        handler = mocker.MagicMock(
            return_value=ToolResult(content='{"status": "ok", "reply": "pong"}')
        )

        result = service.assistant_completion(
            prompt="ping",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            extra_handlers={"run_limnoria_command": handler},
        )

        assert result.last_successful_tool == "run_limnoria_command"


# =========================================================================
# Plugin-level command tests
# =========================================================================


class TestReminderMetaHelpers:
    """Tests for plugin reminder helper methods used by meta."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect({}))
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda s: s
        plugin.db = mocker.MagicMock()
        plugin._reminders = {}
        plugin._reminders_lock = threading.Lock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_remind_set_for_assistant_success(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN valid reminder text WHEN _remind_set_for_assistant THEN returns confirmation."""
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

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "check the build in 1 hour"
        )

        assert "remind" in result.lower() or "hour" in result.lower()
        assert plugin.db.save_reminder.called

    def test_remind_set_for_assistant_with_note(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder with note WHEN _remind_set_for_assistant THEN includes note."""
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

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "deploy in 1 hour"
        )

        assert "Eastern" in result

    def test_remind_set_for_assistant_parse_failure(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unparseable reminder WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "maybe sometime"
        )

        assert "could not" in result.lower()

    def test_remind_set_for_assistant_too_short(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder < 10 seconds WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="now",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "remind me now"
        )

        assert "10 second" in result.lower() or "at least" in result.lower()

    def test_remind_set_for_assistant_too_long(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN reminder > 7 days WHEN _remind_set_for_assistant THEN returns error."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=700000,
            message="later",
            confirmation="OK",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "remind me in 2 weeks"
        )

        assert "7 day" in result.lower()

    def test_remind_set_for_assistant_clarify(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN clarify action WHEN _remind_set_for_assistant THEN returns clarification."""
        from llm.service import ReminderParseResult

        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When exactly should I remind you?",
        )

        msg = mocker.MagicMock()
        msg.args = ["#test"]

        result = plugin._remind_set_for_assistant(
            mock_irc, msg, Identity(raw_nick="testuser", account=None), "remind me"
        )

        assert "when" in result.lower()

    def test_remind_delete_for_assistant_success(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN valid reminder ID WHEN _remind_delete_for_assistant THEN deletes."""
        event_name = "llm_remind_abc123def456"
        plugin._reminders = {
            event_name: make_reminder_row(
                event_name=event_name,
                nick="testuser",
                channel="#test",
                message="check build",
            )
        }
        mocker.patch("llm.plugin.schedule.removeEvent")

        result = plugin._remind_delete_for_assistant(
            Identity(raw_nick="testuser", account=None), "abc123def456"
        )

        assert "delete" in result.lower() or "cancel" in result.lower()
        assert event_name not in plugin._reminders

    def test_remind_delete_for_assistant_not_found(self, plugin) -> None:
        """GIVEN unknown reminder ID WHEN _remind_delete_for_assistant THEN error."""
        plugin._reminders = {}

        result = plugin._remind_delete_for_assistant(
            Identity(raw_nick="testuser", account=None), "nonexistent"
        )

        assert "not found" in result.lower()


class TestDrawForMeta:
    """Tests for _draw_for_assistant helper used by meta generate_image tool."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect())
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_draw_for_assistant_does_not_log_usage(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """_draw_for_assistant does not call db.log_usage.

        Usage logging is consolidated in the outer command wrapper via
        _store_context_and_log_usage; leaf tool handlers must not log
        independently to avoid double-counting.
        """
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="dall-e-3",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a cat")

        assert result == "https://img.example/cat.png"
        plugin.db.log_usage.assert_not_called()

    def test_draw_for_assistant_returns_content(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """_draw_for_assistant returns the image result content string."""
        from llm.service import ImageResult

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/sunset.png",
            model="dall-e-3",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
        )

        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mock_irc, msg, "a sunset")

        assert result == "https://img.example/sunset.png"
        plugin.llm_service.image_generation.assert_called_once_with(
            "a sunset", irc=mock_irc, msg=msg
        )


class TestCodeForAssistant:
    """Tests for _code_for_assistant helper used by meta generate_code tool."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        import threading

        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"codeSystemPrompt": "You are a coder."})
        )
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._MetaSynchronized_rlock = threading.RLock()
        return plugin

    def test_code_for_assistant_returns_url(self, plugin) -> None:
        """_code_for_assistant saves code to HTTP and returns URL."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="print('hello')",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "https://example.com/code/abc123"

        result = plugin._code_for_assistant("write hello world", "#test")

        import json

        data = json.loads(result.content)
        assert data["url"] == "https://example.com/code/abc123"
        plugin.llm_service.save_code_to_http.assert_called_once_with("print('hello')")

    def test_code_for_assistant_includes_code_in_result(self, plugin) -> None:
        """_code_for_assistant includes code content for context."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="def foo(): pass",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "https://example.com/code/xyz"

        result = plugin._code_for_assistant("write a function", "#test")

        import json

        data = json.loads(result.content)
        assert data["code"] == "def foo(): pass"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.cost == 0.001

    def test_code_for_assistant_handles_error(self, plugin) -> None:
        """_code_for_assistant returns sanitized error ToolResult on failure."""
        plugin.llm_service.completion.side_effect = RuntimeError("API down")

        result = plugin._code_for_assistant("write something", "#test")

        import json

        data = json.loads(result.content)
        assert "error" in data
        # Internal exception text must NOT leak into the tool payload
        assert "API down" not in data["error"]
        assert data["error"] == "Code generation failed."

    def test_code_for_assistant_handles_completion_error(self, plugin) -> None:
        """_code_for_assistant returns error ToolResult when completion has error."""
        from llm.service import CompletionResult

        plugin.llm_service.completion.return_value = CompletionResult(
            content="",
            error="Content blocked",
        )

        result = plugin._code_for_assistant("bad prompt", "#test")

        import json

        data = json.loads(result.content)
        assert data["error"] == "Content blocked"


class TestInvalidCommandRouting:
    """Tests for invalidCommand routing directly through chat profile."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture, mock_irc: MagicMock):  # type: ignore[no-untyped-def]
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect({}))
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        return plugin

    def test_invalid_command_routes_through_ask_impl(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN invalidCommand THEN routes to _ask_impl with chat profile."""
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
        call_kwargs = plugin._ask_impl.call_args
        assert call_kwargs[1]["entry_route"] == "invalid_command"

    def test_invalid_command_no_meta_dispatch(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN invalidCommand THEN does not call _run_meta."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()
        plugin._run_meta = mocker.MagicMock()
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

        plugin._run_meta.assert_not_called()
        plugin.llm_service.assistant_completion.assert_not_called()

    def test_invalid_command_still_checks_capability(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN user without llm.ask capability WHEN invalidCommand THEN returns early."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin._ask_impl = mocker.MagicMock()

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["always", "respond", "in", "haiku"])

        plugin._ask_impl.assert_not_called()


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

        result = svc.assistant_completion(
            prompt="always respond in haiku",
            nick="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )
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

        result = svc.assistant_completion(
            prompt="delete any memories about cats",
            nick="testuser",
            channel="#test",
            db=db,
            context=context,
            bot_nick="VibeBot",
        )
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

        result = svc.assistant_completion(
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

        svc.assistant_completion(
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

        svc.assistant_completion(
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
        set_reminder_fn.assert_called_once_with("deploy in 2 hours")

    @staticmethod
    def _make_service(mocker: MockerFixture) -> tuple:
        """Create an LLMService with meta config defaults."""
        plugin = mocker.MagicMock()
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"assistantModel": "gpt-4"})
        )
        plugin.log = mocker.Mock()
        return LLMService(plugin), plugin


class TestToolResult:
    """Tests for the ToolResult frozen dataclass."""

    def test_defaults(self) -> None:
        """GIVEN ToolResult with only content WHEN created THEN defaults are correct."""
        result = ToolResult(content="search result")
        assert result.content == "search result"
        assert result.grounding_used is False
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cost == 0.0

    def test_all_fields_set(self) -> None:
        """GIVEN ToolResult with all fields WHEN created THEN all values stored."""
        result = ToolResult(
            content="fetched page",
            grounding_used=True,
            prompt_tokens=200,
            completion_tokens=100,
            cost=0.005,
        )
        assert result.content == "fetched page"
        assert result.grounding_used is True
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.cost == 0.005

    def test_frozen(self) -> None:
        """GIVEN a ToolResult WHEN attempting mutation THEN raises FrozenInstanceError."""
        result = ToolResult(content="immutable")
        with pytest.raises(AttributeError):
            result.content = "changed"  # type: ignore[misc]


class TestProfileSystemPrompts:
    """GIVEN per-profile system prompts WHEN inspected THEN they have correct content."""

    def test_chat_system_prompt_no_not_meta(self) -> None:
        """GIVEN CHAT_SYSTEM_PROMPT WHEN checked THEN does not contain NOT_META."""
        assert "NOT_META" not in CHAT_SYSTEM_PROMPT

    def test_chat_system_prompt_has_bot_nick_placeholder(self) -> None:
        """GIVEN CHAT_SYSTEM_PROMPT WHEN checked THEN contains {bot_nick} placeholder."""
        assert "{bot_nick}" in CHAT_SYSTEM_PROMPT

    def test_code_system_prompt_mentions_generate_code(self) -> None:
        """GIVEN CODE_SYSTEM_PROMPT WHEN checked THEN mentions generate_code tool."""
        assert "generate_code" in CODE_SYSTEM_PROMPT

    def test_draw_system_prompt_mentions_generate_image(self) -> None:
        """GIVEN DRAW_SYSTEM_PROMPT WHEN checked THEN mentions generate_image tool."""
        assert "generate_image" in DRAW_SYSTEM_PROMPT

    def test_remind_action_prompt_omits_set_reminder_for_structured_rows(self) -> None:
        """GIVEN structured-row prompt WHEN checked THEN no set_reminder paragraph.

        Structured rows reschedule mechanically; the action LLM must NOT see
        a 'you MAY call set_reminder' rule. The set_reminder tool is also
        filtered from the tool surface for those fires (see plugin.py).
        """
        assert "set_reminder" not in REMIND_ACTION_SYSTEM_PROMPT
        assert "Recurrence is handled mechanically" in REMIND_ACTION_SYSTEM_PROMPT

    def test_remind_action_prompt_has_bot_nick_placeholder(self) -> None:
        """GIVEN REMIND_ACTION_SYSTEM_PROMPT WHEN checked THEN contains {bot_nick}."""
        assert "{bot_nick}" in REMIND_ACTION_SYSTEM_PROMPT


class TestToolSpecVisibility:
    """GIVEN tool specs WHEN inspected THEN visibility and capability are correct."""

    def test_search_web_visible_in_chat(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "chat" in specs["search_web"].visible_in

    def test_search_web_visible_in_code(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "code" in specs["search_web"].visible_in

    def test_search_web_not_visible_in_draw(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert "draw" not in specs["search_web"].visible_in

    def test_fetch_url_visible_in_chat_code_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["fetch_url"].visible_in == frozenset({"chat", "code", "remind_action"})

    def test_generate_code_capability_is_llm_code(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_code"].capability == "llm.code"

    def test_generate_code_visible_in_chat_and_code_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_code"].visible_in == frozenset({"chat", "code", "remind_action"})

    def test_generate_image_visible_in_chat_draw_and_remind_action(self) -> None:
        specs = {s.name: s for s in ASSISTANT_TOOL_SPECS}
        assert specs["generate_image"].visible_in == frozenset({"chat", "draw", "remind_action"})

    def test_profile_tools_remind_action_includes_search_fetch_code_image(self) -> None:
        """Action reminders need the union of @ask + @draw tool surfaces."""
        tools = get_tools_for_profile("remind_action")
        names = {t["function"]["name"] for t in tools}
        for required in ("search_web", "fetch_url", "generate_code", "generate_image"):
            assert required in names, f"{required} missing from remind_action profile"
        # Sanity: also includes ordinary chat tools (defaults to chat+remind_action)
        assert "list_pending_tasks" in names
        assert "set_reminder" in names

    def test_profile_tools_chat_includes_search(self) -> None:
        tools = get_tools_for_profile("chat")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" in names
        assert "generate_code" in names
        assert "generate_image" in names

    def test_profile_tools_draw_includes_generate_image(self) -> None:
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        assert "generate_image" in names

    def test_profile_tools_draw_excludes_search(self) -> None:
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" not in names
        assert "generate_code" not in names

    def test_profile_tools_code_includes_search_and_code(self) -> None:
        tools = get_tools_for_profile("code")
        names = {t["function"]["name"] for t in tools}
        assert "search_web" in names
        assert "fetch_url" in names
        assert "generate_code" in names

    def test_set_reminder_visible_in_chat_and_remind_action(self) -> None:
        """set_reminder routes through chat (deferred entry point) and remind_action
        (self-rescheduling). Immediate-execution profiles (@draw, @code) defer via
        chat using @remind, so they should NOT expose the reminder tools."""
        for profile in ("chat", "remind_action"):
            tools = get_tools_for_profile(profile)
            names = {t["function"]["name"] for t in tools}
            assert "set_reminder" in names, f"set_reminder missing from {profile} profile"

    def test_reminder_tools_not_visible_in_draw_profile(self) -> None:
        """@draw is immediate-execution; deferred draws go via @remind (chat profile)."""
        tools = get_tools_for_profile("draw")
        names = {t["function"]["name"] for t in tools}
        for tool in (
            "set_reminder",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
        ):
            assert tool not in names, f"{tool} should not be visible in draw profile"

    def test_reminder_tools_not_visible_in_code_profile(self) -> None:
        """@code is immediate-execution; deferred code generation goes via @remind."""
        tools = get_tools_for_profile("code")
        names = {t["function"]["name"] for t in tools}
        for tool in (
            "set_reminder",
            "list_pending_tasks",
            "cancel_pending_task",
            "cancel_all_pending_tasks",
        ):
            assert tool not in names, f"{tool} should not be visible in code profile"


class TestScheduleLlmTaskFamily:
    """Phase 2 Task 3 / C1 — schedule_llm_task tool registration."""

    def test_assistant_tools_includes_schedule_llm_task_family(self) -> None:
        names = {t["function"]["name"] for t in ASSISTANT_TOOLS}
        assert "schedule_llm_task" in names
        # Listing/cancelling for both reminders and scheduled tasks now
        # goes through the unified pending-task tool surface.
        assert "list_pending_tasks" in names
        assert "cancel_pending_task" in names
        assert "cancel_all_pending_tasks" in names

        by_name = {t["function"]["name"]: t for t in ASSISTANT_TOOLS}
        sch = by_name["schedule_llm_task"]
        # Description must call out the @ask-with-tools shape and contrast
        # with set_reminder so the LLM picks the right tool.
        desc = sch["function"]["description"].lower()
        assert "@ask" in desc or "ask " in desc
        assert "set_reminder" in desc
        assert "tool" in desc

        params = sch["function"]["parameters"]
        assert "when_natural" in params["properties"]
        assert "prompt" in params["properties"]
        assert set(params["required"]) >= {"when_natural", "prompt"}

    def test_schedule_llm_task_specs_overrides_applied(self) -> None:
        """C2: ToolSpec overrides give schedule_llm_task require_account=True;
        list/cancel inherit defaults (llm.ask, no account, chat+remind_action)."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY

        sch = ASSISTANT_TOOL_REGISTRY["schedule_llm_task"]
        assert sch.capability == "llm.ask"
        assert sch.require_account is True
        assert sch.visible_in == frozenset({"chat", "remind_action"})

        lst = ASSISTANT_TOOL_REGISTRY["list_pending_tasks"]
        assert lst.capability == "llm.ask"
        assert lst.require_account is False
        assert lst.visible_in == frozenset({"chat", "remind_action"})

        can = ASSISTANT_TOOL_REGISTRY["cancel_pending_task"]
        assert can.capability == "llm.ask"
        assert can.require_account is False
        assert can.visible_in == frozenset({"chat", "remind_action"})

    def test_executor_accepts_pending_task_fns(self, mocker: MockerFixture) -> None:
        """C3: AssistantToolExecutor accepts the unified pending-task fn kwargs."""
        schedule_fn = mocker.MagicMock()
        list_fn = mocker.MagicMock()
        cancel_fn = mocker.MagicMock()
        cancel_all_fn = mocker.MagicMock()

        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
            list_pending_tasks_fn=list_fn,
            cancel_pending_task_fn=cancel_fn,
            cancel_all_pending_tasks_fn=cancel_all_fn,
        )
        assert ex._schedule_llm_task_fn is schedule_fn
        assert ex._list_pending_tasks_fn is list_fn
        assert ex._cancel_pending_task_fn is cancel_fn
        assert ex._cancel_all_pending_tasks_fn is cancel_all_fn

    def test_tool_schedule_llm_task_calls_callback_and_returns_json(
        self, mocker: MockerFixture
    ) -> None:
        import json

        schedule_fn = mocker.MagicMock(
            return_value={
                "status": "ok",
                "event_name": "llm_task_abc",
                "fire_at": 1700000000.0,
                "message": "Scheduled.",
            }
        )

        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
        )

        out = ex.execute(
            "schedule_llm_task",
            {"when_natural": "in 60s", "prompt": "ping me"},
        )
        parsed = json.loads(out.content)
        assert parsed["status"] == "ok"
        assert parsed["event_name"] == "llm_task_abc"
        schedule_fn.assert_called_once_with(
            when_natural="in 60s", prompt="ping me", reply_target=None
        )

    def test_tool_list_pending_tasks_returns_merged(self, mocker: MockerFixture) -> None:
        import json

        list_fn = mocker.MagicMock(
            return_value=[
                {
                    "kind": "reminder",
                    "id": "abc123",
                    "channel": "#t",
                    "description": "check build",
                },
                {
                    "kind": "scheduled_task",
                    "id": "llm_task_ev2",
                    "when": "2026-05-09T13:00:00Z",
                    "channel": "#t",
                    "description": "weekly digest",
                    "recurrence": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9",
                },
            ]
        )
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            list_pending_tasks_fn=list_fn,
        )
        out = ex.execute("list_pending_tasks", {})
        parsed = json.loads(out.content)
        assert len(parsed["tasks"]) == 2
        kinds = {t["kind"] for t in parsed["tasks"]}
        assert kinds == {"reminder", "scheduled_task"}

    def test_tool_cancel_pending_task_passes_id(self, mocker: MockerFixture) -> None:
        cancel_fn = mocker.MagicMock(
            return_value={
                "status": "ok",
                "kind": "scheduled_task",
                "id": "llm_task_abc",
                "message": "Cancelled.",
            }
        )
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            cancel_pending_task_fn=cancel_fn,
        )
        out = ex.execute("cancel_pending_task", {"id": "llm_task_abc"})
        assert "ok" in out.content.lower()
        cancel_fn.assert_called_once_with("llm_task_abc")


class TestExecutorCoverageGaps:
    """Targeted tests for branches not exercised by the main suite."""

    def test_denial_reason_require_account_without_account(self) -> None:
        """ToolSpec.denial_reason rejects when require_account=True and account is None."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY

        spec = ASSISTANT_TOOL_REGISTRY["schedule_llm_task"]
        reason = spec.denial_reason(
            route_profile="chat",
            capabilities=frozenset({"llm.ask"}),
            account=None,
        )
        assert reason is not None
        assert "authenticated account" in reason

    def test_execute_missing_handler_returns_error(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute() falls through to the missing-handler branch when handler_name is bogus."""
        from llm.assistant import ASSISTANT_TOOL_REGISTRY, ToolSpec

        bogus = ToolSpec(
            name="bogus_handler_tool",
            schema={"name": "bogus_handler_tool", "description": "x", "parameters": {}},
            handler_name="_tool_does_not_exist",
            capability="llm.ask",
        )
        monkeypatch.setitem(ASSISTANT_TOOL_REGISTRY, "bogus_handler_tool", bogus)

        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
        )
        result = ex.execute("bogus_handler_tool", {})
        assert "Unknown tool implementation" in result.content

    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("get_instruction", {"nick": "someone"}),
            ("set_instruction", {"nick": "someone", "text": "x"}),
            ("clear_instruction", {"nick": "someone"}),
            ("save_memory", {"nick": "someone", "text": "x"}),
            ("delete_memory", {"nick": "someone", "id": 1}),
            ("update_memory", {"nick": "someone", "id": 1, "text": "x"}),
        ],
    )
    def test_per_tool_owner_only_denials(
        self,
        mocker: MockerFixture,
        tool_name: str,
        args: dict,
    ) -> None:
        """Non-owner targeting another nick is denied for owner-gated tools."""
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="regular",
            channel="#t",
        )
        result = ex.execute(tool_name, args)
        assert "owner" in result.content.lower()

    def test_forget_context_when_nothing_to_clear(self, mocker: MockerFixture) -> None:
        """forget_context returns the no-op message when context.clear() returns False."""
        ctx = mocker.MagicMock()
        ctx.clear.return_value = False
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=ctx,
            nick="n",
            channel="#t",
        )
        result = ex.execute("forget_context", {})
        assert "No context to clear" in result.content

    def test_cleanup_memories_error_result_is_routed_as_error(self, mocker: MockerFixture) -> None:
        """cleanup_fn returning a string with 'failed'/'error' produces an error envelope."""
        cleanup_fn = mocker.MagicMock(return_value="Cleanup failed: db error")
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            cleanup_fn=cleanup_fn,
        )
        result = ex.execute("cleanup_memories", {})
        assert '"error"' in result.content
        assert "Cleanup failed" in result.content

    @pytest.mark.parametrize(
        "tool_name,args",
        [
            ("list_pending_tasks", {}),
            ("set_reminder", {"text": "in 1h ping"}),
            ("cancel_pending_task", {"id": "abc"}),
            ("cancel_all_pending_tasks", {}),
        ],
    )
    def test_pending_task_tools_unavailable_when_no_fn(
        self, mocker: MockerFixture, tool_name: str, args: dict
    ) -> None:
        """Pending-task tools return 'not available' when their callback is None."""
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
        )
        result = ex.execute(tool_name, args)
        assert "not available" in result.content.lower()

    def test_set_reminder_error_result_is_routed_as_error(self, mocker: MockerFixture) -> None:
        """set_reminder_fn returning 'Could not parse...' becomes an error envelope."""
        set_fn = mocker.MagicMock(return_value="Could not parse time.")
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            set_reminder_fn=set_fn,
        )
        result = ex.execute("set_reminder", {"text": "garbage"})
        assert '"error"' in result.content
        assert "Could not parse" in result.content

    def test_schedule_llm_task_unconfigured(self, mocker: MockerFixture) -> None:
        """schedule_llm_task without a callback returns 'not configured'."""
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
        )
        result = ex.execute(
            "schedule_llm_task",
            {"when_natural": "in 1h", "prompt": "ping"},
        )
        assert "not configured" in result.content.lower()

    @pytest.mark.parametrize(
        "args,expected_substring",
        [
            ({"when_natural": "", "prompt": "ping"}, "when_natural is required"),
            ({"when_natural": "in 1h", "prompt": ""}, "prompt is required"),
        ],
    )
    def test_schedule_llm_task_validates_required_args(
        self,
        mocker: MockerFixture,
        args: dict,
        expected_substring: str,
    ) -> None:
        """schedule_llm_task rejects empty when_natural / prompt before calling fn."""
        schedule_fn = mocker.MagicMock()
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            account="acct",
            schedule_llm_task_fn=schedule_fn,
        )
        result = ex.execute("schedule_llm_task", args)
        assert expected_substring in result.content
        schedule_fn.assert_not_called()

    def test_cancel_pending_task_requires_id(self, mocker: MockerFixture) -> None:
        """cancel_pending_task rejects empty id before calling fn."""
        cancel_fn = mocker.MagicMock()
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask"}),
            cancel_pending_task_fn=cancel_fn,
        )
        result = ex.execute("cancel_pending_task", {"id": ""})
        assert "id is required" in result.content
        cancel_fn.assert_not_called()

    def test_generate_image_unavailable_when_no_draw_fn(self, mocker: MockerFixture) -> None:
        """generate_image without a draw_fn returns 'not available'."""
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert "not available" in result.content.lower()

    def test_generate_image_rejects_empty_prompt(self, mocker: MockerFixture) -> None:
        """generate_image rejects whitespace-only prompts before calling draw_fn."""
        draw_fn = mocker.MagicMock()
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "   "})
        assert "prompt is required" in result.content.lower()
        draw_fn.assert_not_called()

    def test_generate_image_propagates_draw_fn_error(self, mocker: MockerFixture) -> None:
        """draw_fn returning 'Error...' is mapped to an error envelope."""
        draw_fn = mocker.MagicMock(return_value="Error: quota exceeded")
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert '"error"' in result.content
        assert "quota exceeded" in result.content

    def test_generate_image_success(self, mocker: MockerFixture) -> None:
        """draw_fn returning a normal URL produces an ok envelope."""
        draw_fn = mocker.MagicMock(return_value="https://example.com/cat.png")
        ex = AssistantToolExecutor(
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            nick="n",
            channel="#t",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="acct",
            draw_fn=draw_fn,
        )
        result = ex.execute("generate_image", {"prompt": "a cat"})
        assert '"ok"' in result.content
        assert "example.com/cat.png" in result.content
