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
        assert len(META_TOOLS) == 9

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
    def executor(self, mock_db: MagicMock, mock_context: MagicMock) -> MetaToolExecutor:
        return MetaToolExecutor(db=mock_db, context=mock_context, nick="testuser", channel="#test")

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
        mock_choice.message.content = "Done — instruction set."
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

        assert result.content == "Done — instruction set."
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
        second_choice.message.content = "Done — I'll respond in haiku."
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

        assert result.content == "Done — I'll respond in haiku."
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

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="Done — instruction set.",
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


class TestInvalidCommandMetaFallback:
    """Tests for invalidCommand routing through meta then to ask."""

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
        """GIVEN unknown command WHEN meta returns NOT_META THEN ask called."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="NOT_META",
            is_meta=False,
        )
        plugin.ask = mocker.MagicMock()
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

        plugin.ask.assert_called_once()

    def test_meta_disabled_skips_to_ask(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN metaEnabled=False WHEN unknown command THEN straight to ask."""
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaEnabled": False})
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin.ask = mocker.MagicMock()
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._is_old_message = mocker.MagicMock(return_value=False)

        plugin.invalidCommand(mock_irc, msg, ["hello", "there"])

        plugin.ask.assert_called_once()
        plugin.llm_service.meta_completion.assert_not_called()

    def test_meta_handled_does_not_call_ask(
        self, plugin, mocker: MockerFixture, mock_irc: MagicMock
    ) -> None:
        """GIVEN unknown command WHEN meta handles it THEN ask NOT called."""
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.nick = "testuser"
        msg.args = ["#test"]

        plugin.llm_service.meta_completion.return_value = MetaResult(
            content="Instruction set to haiku.",
            is_meta=True,
            model="gpt-4",
        )
        plugin.ask = mocker.MagicMock()
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

        plugin.ask.assert_not_called()
        mock_irc.reply.assert_called()


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
        second_choice.message.content = "Done — I'll respond in haiku."
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

    @staticmethod
    def _make_service(mocker: MockerFixture) -> tuple:
        """Create an LLMService with meta config defaults."""
        plugin = mocker.MagicMock()
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"metaModel": "gpt-4"})
        )
        plugin.log = mocker.Mock()
        return LLMService(plugin), plugin
