"""Tests that call actual plugin command methods (not reimplementations).

These tests exercise the real ask, code, draw, forget, usage,
and remind methods on a properly-initialised LLM plugin instance
with mocked dependencies.

Unlike the _call_ask / _call_code / _call_draw helpers in test_plugin.py
which reimplement command logic, these tests invoke the actual methods so
regressions in the real command code are caught.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from llm.persistence import UsageBreakdown, UsageSummary
from llm.plugin import LLM
from llm.service import AssistantResult, CompletionResult, ReminderParseResult

from .conftest import make_registry_side_effect, make_reminder_row

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class TestAskCommand:
    """Tests for the real LLM.ask method."""

    def test_ask_routes_through_assistant_request(self, plugin_env):
        """GIVEN ask WHEN executed THEN it uses the shared assistant_request facade."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Hello from unified assistant",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        request_context = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert request_context.profile == "chat"
        assert request_context.entry_route == "ask"
        mock_irc.reply.assert_called_once_with("Hello from unified assistant", prefixNick=False)

    def test_ask_replies_with_completion_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN a normal prompt WHEN ask is called THEN irc.reply receives the completion content."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hello from AI",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["What", "is", "Python?"])

        mock_irc.reply.assert_called_once_with("Hello from AI", prefixNick=False)

    def test_ask_stores_context_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN a successful completion WHEN ask is called THEN conversation context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response text",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Context should have both user and assistant messages
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "response text"

    def test_ask_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN completion with cost WHEN ask completes THEN usage is logged in db."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.005,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            100,
            50,
            0.005,
            prompt="hello",
            status="success",
            error_detail="",
        )

    def test_ask_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN a message older than startup WHEN ask called THEN no reply is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100  # before startup

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_ask_prepends_grounding_icon_when_used(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN ask completes THEN reply has globe icon prefix."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="searched result",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["search", "something"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")
        assert "searched result" in reply_text

    def test_ask_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN completion returns an error WHEN ask completes THEN context is NOT stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # No context should be stored because result has an error
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_ask_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN ask completes THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="reply",
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_ask_sends_action_for_me_response(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN sends IRC action."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me shrugs",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")

        plugin.ask(mock_irc, mock_msg, ["how", "are", "you?"])

        mock_irc.reply.assert_not_called()
        mock_action.assert_called_once_with("#test", "shrugs")
        mock_irc.queueMsg.assert_called_once_with(mock_action.return_value)

    def test_ask_normal_response_uses_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds normally WHEN ask called THEN uses irc.reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="The capital is Paris.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["what", "is", "the", "capital?"])

        mock_irc.reply.assert_called_once_with("The capital is Paris.", prefixNick=False)

    def test_ask_reminder_mutation_with_empty_text_suppresses_reply(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN successful set_reminder + empty post-tool text WHEN ask called THEN no irc.reply, but usage logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=20,
            completion_tokens=2,
            cost=0.0002,
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "in", "1m"])

        # Reaction is the user-visible ack; no duplicate reply.
        mock_irc.reply.assert_not_called()
        mock_irc.queueMsg.assert_not_called()
        # Suppression isn't an error path.
        mock_irc.error.assert_not_called()
        # Usage still logged so the suppressed path isn't free.
        plugin.db.log_usage.assert_called_once()

    def test_ask_reminder_mutation_with_text_does_not_suppress(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN successful set_reminder + follow-up text WHEN ask called THEN reply IS sent."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        follow_up = "Got it. Want me to also remind you about the receipt?"
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content=follow_up,
            grounding_used=False,
            prompt_tokens=20,
            completion_tokens=12,
            cost=0.0003,
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools=follow_up,
        )

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "in", "1m"])

        mock_irc.reply.assert_called_once_with(follow_up, prefixNick=False)
        mock_irc.error.assert_not_called()

    def test_ask_non_reminder_tool_with_empty_text_does_not_suppress(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN non-mutation tool + empty text WHEN ask called THEN falls through to empty-response error."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=15,
            completion_tokens=0,
            cost=0.0001,
            model="gpt-4",
            last_successful_tool="search_web",
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["what", "is", "happening"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_called_once()

    def test_ask_no_tool_called_with_empty_text_falls_through_to_error(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN no tool call + empty text WHEN ask called THEN empty-response error fires."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.0001,
            model="gpt-4",
            last_successful_tool=None,
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_called_once()

    def test_ask_action_stores_context_with_star_prefix(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN context stores * BotNick text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me thinks about it",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["hmm"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "* testbot thinks about it"

    def test_ask_action_with_grounding_icon(self, plugin_env, mocker: MockerFixture):
        """GIVEN /me response with grounding WHEN ask called THEN action includes globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me looks it up",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["search", "for", "it"])

        mock_action.assert_called_once_with("#test", "\U0001f310 looks it up")

    def test_ask_bare_me_not_treated_as_action(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with '/me' with no trailing text WHEN ask called THEN uses reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["test"])

        mock_irc.reply.assert_called_once_with("/me", prefixNick=False)

    def test_ask_star_botname_treated_as_action(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with '* BotNick ...' WHEN ask called THEN sent as IRC action."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="* testbot nudges a pair of glasses toward you",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["test"])

        mock_action.assert_called_once_with("#test", "nudges a pair of glasses toward you")


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


class TestCodeCommand:
    """Tests for the real LLM.code method."""

    def _make_code_result(self, **overrides):
        """Build a AssistantResult with sensible defaults for code tests."""
        defaults = {
            "content": "Here is your code — http://x/code.html",
            "grounding_used": False,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.001,
            "model": "gpt-4",
        }
        defaults.update(overrides)
        return AssistantResult(**defaults)  # ty: ignore[invalid-argument-type]

    def test_code_routes_through_assistant_request(self, plugin_env, mocker: MockerFixture):
        """GIVEN code WHEN executed THEN it uses assistant_request with code profile."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result()

        plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        ctx = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert ctx.profile == "code"
        assert ctx.entry_route == "code"

    def test_code_replies_with_planner_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns summary with URL WHEN code called THEN reply contains it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="Fibonacci function — http://localhost:8080/llm/code_abc.html",
        )

        plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://localhost:8080/llm/code_abc.html" in reply_text
        assert "Fibonacci function" in reply_text

    def test_code_stores_context(self, plugin_env, mocker: MockerFixture):
        """GIVEN code command succeeds WHEN executed THEN conversation context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="code output",
        )

        plugin.code(mock_irc, mock_msg, ["generate", "something"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_code_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN code completion with cost WHEN code completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
        )

        plugin.code(mock_irc, mock_msg, ["assign"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "code",
            "gpt-4",
            50,
            20,
            0.003,
            prompt="assign",
            status="success",
            error_detail="",
        )

    def test_code_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN an old message WHEN code called THEN nothing happens."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_code_grounding_icon_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN code response returned THEN reply has globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="summary — http://x/c.html",
            grounding_used=True,
        )

        plugin.code(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")

    def test_code_preserves_preflight(self, plugin_env, mocker: MockerFixture):
        """GIVEN code WHEN executed THEN preflight checks still run."""
        plugin, mock_irc, mock_msg = plugin_env
        # Make preflight block the request (e.g. rate limited)
        mocker.patch.object(
            plugin,
            "_run_preflight",
            return_value=mocker.MagicMock(blocked=True),
        )

        plugin.code(mock_irc, mock_msg, ["test"])

        plugin.llm_service.assistant_request.assert_not_called()
        mock_irc.reply.assert_not_called()

    def test_code_user_instruction_layers_on_facade_prompt(self, plugin_env):
        """GIVEN user has instruction WHEN code called THEN planner sees CODE_SYSTEM_PROMPT.

        Regression: a user instruction must not replace the assistant facade
        prompt with the (inner-call) registry codeSystemPrompt — otherwise
        the planner stops calling generate_code and the pastebin breaks.
        """
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Captain Picard."
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result()

        plugin.code(mock_irc, mock_msg, ["fibonacci"])

        system_prompt = plugin.llm_service.assistant_request.call_args.kwargs["system_prompt"]
        assert "Picard" in system_prompt
        assert "generate_code" in system_prompt


# ---------------------------------------------------------------------------
# _send_long_reply (multiline batch helper)
# ---------------------------------------------------------------------------


class TestSendLongReply:
    """Tests for _send_long_reply — multiline-or-paginated reply helper."""

    @pytest.fixture(autouse=True)
    def _restore_experimental_extensions(self):
        """Save/restore experimentalExtensions across each test."""
        import supybot.conf as supy_conf

        knob = supy_conf.supybot.protocols.irc.experimentalExtensions
        original = knob()
        try:
            yield
        finally:
            knob.setValue(original)

    def test_short_text_uses_irc_reply(self, plugin_env):
        """GIVEN one-line short text WHEN sent THEN goes via irc.reply (no batch)."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.capabilities_ack = {"draft/multiline"}

        plugin._send_long_reply(mock_irc, mock_msg, "hello world")

        mock_irc.reply.assert_called_once_with("hello world", prefixNick=False)
        mock_irc.queueMultilineBatches.assert_not_called()

    def test_multiline_text_uses_multiline_batch_when_supported(self, plugin_env, mocker):
        """GIVEN \\n in text AND multiline negotiated WHEN sent THEN batch path used."""
        import supybot.conf as supy_conf

        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.capabilities_ack = {"draft/multiline"}
        supy_conf.supybot.protocols.irc.experimentalExtensions.setValue(True)

        plugin._send_long_reply(mock_irc, mock_msg, "line one\nline two\nline three")

        mock_irc.queueMultilineBatches.assert_called_once()
        call = mock_irc.queueMultilineBatches.call_args
        assert call.kwargs.get("concat") is False
        msgs = call.args[0]
        assert len(msgs) == 3
        mock_irc.reply.assert_not_called()

    def test_multiline_falls_back_when_cap_not_negotiated(self, plugin_env, mocker):
        """GIVEN \\n in text AND multiline NOT negotiated THEN falls back to irc.reply."""
        import supybot.conf as supy_conf

        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.capabilities_ack = set()  # no draft/multiline
        supy_conf.supybot.protocols.irc.experimentalExtensions.setValue(True)

        plugin._send_long_reply(mock_irc, mock_msg, "line one\nline two")

        mock_irc.reply.assert_called_once_with("line one\nline two", prefixNick=False)
        mock_irc.queueMultilineBatches.assert_not_called()

    def test_multiline_falls_back_when_experimental_disabled(self, plugin_env, mocker):
        """GIVEN multiline cap acked but experimentalExtensions off THEN falls back."""
        import supybot.conf as supy_conf

        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.capabilities_ack = {"draft/multiline"}
        supy_conf.supybot.protocols.irc.experimentalExtensions.setValue(False)

        plugin._send_long_reply(mock_irc, mock_msg, "line one\nline two")

        mock_irc.reply.assert_called_once_with("line one\nline two", prefixNick=False)
        mock_irc.queueMultilineBatches.assert_not_called()


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


class TestDrawCommand:
    """Tests for the real LLM.draw method (thin wrapper over assistant facade)."""

    def _make_draw_result(self, **overrides):
        """Build a AssistantResult with sensible defaults for draw tests."""
        defaults = {
            "content": "Here is your image: http://img.example/gen.png",
            "grounding_used": False,
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "cost": 0.02,
            "model": "dall-e-3",
        }
        defaults.update(overrides)
        return AssistantResult(**defaults)  # ty: ignore[invalid-argument-type]

    def test_draw_routes_through_assistant_request(self, plugin_env, mocker: MockerFixture):
        """@draw calls assistant_request with draw profile."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        plugin.llm_service.assistant_request.assert_called_once()
        ctx = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert ctx.profile == "draw"
        assert ctx.entry_route == "draw"

    def test_draw_requires_account(self, plugin_env, mocker: MockerFixture):
        """@draw still requires authenticated account."""
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default in plugin_env)

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        mock_irc.error.assert_called_once()
        plugin.llm_service.assistant_request.assert_not_called()

    def test_draw_passes_recent_context(self, plugin_env, mocker: MockerFixture):
        """@draw fetches history with the max-age window set on the context calls."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.context = mocker.MagicMock()
        plugin.context.get_messages.return_value = [{"role": "user", "content": "hi"}]
        plugin.context.get_channel_messages.return_value = []

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        get_msgs_kwargs = plugin.context.get_messages.call_args.kwargs
        assert get_msgs_kwargs["max_age_seconds"] == 60
        get_ch_kwargs = plugin.context.get_channel_messages.call_args.kwargs
        assert get_ch_kwargs["max_age_seconds"] == 60

        assistant_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert assistant_kwargs["history"] == [{"role": "user", "content": "hi"}]
        assert assistant_kwargs["memories"] == []

    def test_draw_skips_context_when_max_age_zero(self, plugin_env, mocker: MockerFixture):
        """@draw with drawContextMaxAgeSeconds=0 skips context entirely."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        original = plugin.registryValue.side_effect

        def registry(key: str, *args: object) -> object:
            if key == "drawContextMaxAgeSeconds":
                return 0
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin.context = mocker.MagicMock()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        plugin.context.get_messages.assert_not_called()
        plugin.context.get_channel_messages.assert_not_called()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs["history"] == []
        assert kwargs["channel_history"] == []

    def test_draw_replies_with_planner_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns image URL WHEN draw called THEN reply contains it."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Here is your image: http://img.example/gen.png",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://img.example/gen.png" in reply_text

    def test_draw_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw with cost WHEN draw completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            10,
            0,
            0.04,
            prompt="a cat",
            status="success",
            error_detail="",
        )

    def test_draw_logs_usage_even_with_zero_cost(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw with zero cost/tokens WHEN draw succeeds THEN usage is still logged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="success",
            error_detail="",
        )

    def test_draw_logs_usage_on_content_blocked(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw content blocked WHEN draw completes THEN usage logged with status=content_blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: content blocked",
            error="Error: content blocked",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="content_blocked",
            error_detail="Error: content blocked",
        )

    def test_draw_logs_usage_on_generic_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw that errors (non-content) WHEN draw completes THEN usage logged with status=error."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: timeout exceeded",
            error="Error: timeout exceeded",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="error",
            error_detail="Error: timeout exceeded",
        )

    def test_draw_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN old message WHEN draw called THEN no reply."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        mock_irc.reply.assert_not_called()

    def test_draw_stores_context_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw succeeds WHEN executed THEN personal and channel context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "a sunset"
        assert messages[1]["role"] == "assistant"

    def test_draw_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw returns error WHEN executed THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        plugin.draw(mock_irc, mock_msg, ["bad", "prompt"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0

    def test_draw_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN draw succeeds THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0

    def test_draw_grounding_icon_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN draw response returned THEN reply has globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="image — http://x/img.png",
            grounding_used=True,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")

    def test_draw_preserves_preflight(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw WHEN preflight blocks THEN assistant_request is not called."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(
            plugin,
            "_run_preflight",
            return_value=mocker.MagicMock(blocked=True),
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.llm_service.assistant_request.assert_not_called()
        mock_irc.reply.assert_not_called()


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForgetCommand:
    """Tests for the real LLM.forget method."""

    def test_forget_clears_existing_context(self, plugin_env):
        """GIVEN user has context WHEN forget called THEN context cleared and confirmed."""
        plugin, mock_irc, mock_msg = plugin_env

        # Pre-populate context
        plugin.context.add_message("testnick", "#test", "user", "hi")
        plugin.context.add_message("testnick", "#test", "assistant", "hello")

        plugin.forget(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower() or "fresh" in reply_text.lower()

        # Context should be empty
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    def test_forget_reports_cleared_even_when_empty(self, plugin_env):
        """GIVEN user has no context WHEN forget called THEN still says cleared."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.forget(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower()

    def test_forget_defaults_to_current_channel(self, plugin_env):
        """GIVEN user in a channel WHEN forget called without args THEN clears current channel."""
        plugin, mock_irc, mock_msg = plugin_env

        # Add context for the current channel (#test from fixture)
        plugin.context.add_message("testnick", "#test", "user", "hello")
        plugin.context.add_message("testnick", "#test", "assistant", "hi")
        assert len(plugin.context.get_messages("testnick", "#test")) == 2

        # Also add context for another channel (should not be affected)
        plugin.context.add_message("testnick", "#other", "user", "keep me")

        plugin.forget(mock_irc, mock_msg, [])

        # Current channel context should be cleared
        assert len(plugin.context.get_messages("testnick", "#test")) == 0
        # Other channel context should be untouched
        assert len(plugin.context.get_messages("testnick", "#other")) == 1


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


class TestUsageCommand:
    """Tests for the real LLM.usage method (dual-mode: channel + PM)."""

    # -- PM mode (global stats, admin only) --

    def test_usage_pm_shows_today_and_month_stats(self, plugin_env, mocker: MockerFixture):
        """GIVEN admin via PM WHEN usage called THEN response includes today and monthly stats."""
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=10,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_cost=0.05,
        )
        plugin.db.get_usage_by_nick.return_value = [
            UsageBreakdown(
                name="testnick",
                total_requests=5,
                total_prompt_tokens=500,
                total_completion_tokens=250,
                total_cost=0.03,
            )
        ]
        plugin.db.get_usage_by_channel.return_value = [
            UsageBreakdown(
                name="#test",
                total_requests=10,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_cost=0.05,
            )
        ]

        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Today:" in reply_text
        assert "This month:" in reply_text
        assert "Top users:" in reply_text
        assert "Top channels:" in reply_text
        assert "Context:" in reply_text
        assert "conversations" in reply_text
        # Sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_usage_pm_with_no_top_users_or_channels(self, plugin_env, mocker: MockerFixture):
        """GIVEN no usage data via PM WHEN usage called THEN response omits top users/channels."""
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cost=0.0,
        )
        plugin.db.get_usage_by_nick.return_value = []
        plugin.db.get_usage_by_channel.return_value = []

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Top users:" not in reply_text
        assert "Top channels:" not in reply_text

    def test_usage_pm_requires_admin(self, plugin_env, mocker: MockerFixture):
        """GIVEN non-admin via PM WHEN usage called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "admin" in error_text.lower()

    # -- Channel mode (personal + channel stats, any user) --

    def test_usage_channel_shows_channel_and_personal_stats(self, plugin_env):
        """GIVEN user in channel WHEN usage called THEN shows channel and personal stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        # msg.channel is already "#test" from fixture
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=45,
            total_prompt_tokens=5000,
            total_completion_tokens=2500,
            total_cost=0.0292,
        )
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=12,
            total_prompt_tokens=1200,
            total_completion_tokens=600,
            total_cost=0.0139,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=5)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=8)

        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        # Channel stats
        assert "#test this month:" in reply_text
        assert "45 requests" in reply_text
        assert "rank 1/5 channels" in reply_text
        # Personal stats
        assert "You:" in reply_text
        assert "12 requests" in reply_text
        assert "rank 1/8 users" in reply_text
        # Context info
        assert "Context:" in reply_text
        # Not sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is not True
        assert mock_irc.reply.call_args.kwargs.get("prefixNick") is False

    def test_usage_channel_works_without_admin(self, plugin_env, mocker: MockerFixture):
        """GIVEN non-admin in channel WHEN usage called THEN stats shown (no error)."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=0)

        # No admin capability needed — should still work
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        mock_irc.error.assert_not_called()

    def test_usage_channel_zero_data_graceful(self, plugin_env):
        """GIVEN no usage data WHEN usage called in channel THEN shows zeros without rank."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=0)

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#test this month: $0.0000" in reply_text
        assert "You: $0.0000" in reply_text
        # rank should not appear when rank=0
        assert "rank" not in reply_text
        # Context should show empty when no messages
        assert "Context: empty" in reply_text

    def test_usage_channel_shows_context_message_count(self, plugin_env):
        """GIVEN user with active context WHEN usage called THEN shows message count and expiry."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(10, 500, 250, 0.01)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 250, 125, 0.005)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        # Add some messages to the user's context
        plugin.context.add_message("testnick", "#test", "user", "Hello")
        plugin.context.add_message("testnick", "#test", "assistant", "Hi there")

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Context: 2/" in reply_text
        assert "msgs" in reply_text
        assert "expires in" in reply_text

    # -- Target nick mode --

    def test_usage_strips_irc_status_prefix_from_nick(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with @ prefix WHEN usage called THEN prefix stripped before lookup."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(7, 800, 400, 0.01)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, ["@Larry"])

        # Should query for "Larry", not "@Larry"
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "Larry"
        assert "Larry" in mock_irc.reply.call_args[0][0]

    def test_usage_handles_nick_with_brackets(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with brackets WHEN usage called THEN raw arg parsed correctly."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(3, 300, 150, 0.005)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=4)

        plugin.usage(mock_irc, mock_msg, ["Rubin[F]"])

        # Should query DB with the full bracket nick
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "Rubin[F]"
        assert "Rubin[F]" in mock_irc.reply.call_args[0][0]

    def test_usage_resolves_target_nick_to_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN target nick with NickServ account WHEN usage called THEN queries by account."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        # Target nick "OldNick" resolves to account "RealAccount"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="RealAccount")
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 500, 250, 0.01)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=2, total=6)

        plugin.usage(mock_irc, mock_msg, ["OldNick"])

        # DB should be queried with the account name
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "RealAccount"
        # But display should still show the original nick typed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "OldNick" in reply_text

    def test_usage_with_nick_in_channel(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target in channel WHEN usage called THEN shows that nick's channel stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=7,
            total_prompt_tokens=800,
            total_completion_tokens=400,
            total_cost=0.0100,
        )
        plugin.db.get_nick_rank.return_value = UsageRank(rank=3, total=10)

        plugin.usage(mock_irc, mock_msg, ["othernick"])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "othernick" in reply_text
        assert "in #test" in reply_text
        assert "7 requests" in reply_text
        assert "rank 3/10 users" in reply_text
        # Scoped to current channel
        plugin.db.get_usage_summary_for_nick.assert_called_once()
        call_kwargs = plugin.db.get_usage_summary_for_nick.call_args
        assert call_kwargs[0][0] == "othernick"
        assert call_kwargs[1]["channel"] == "#test"

    def test_usage_with_nick_via_pm(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target via PM WHEN usage called THEN shows that nick's global stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=20,
            total_prompt_tokens=2000,
            total_completion_tokens=1000,
            total_cost=0.0500,
        )
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, ["othernick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "othernick" in reply_text
        assert "in #" not in reply_text  # no channel scope
        assert "20 requests" in reply_text

    # -- Target channel mode --

    def test_usage_with_channel_target(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target WHEN usage called THEN shows that channel's stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=100,
            total_prompt_tokens=10000,
            total_completion_tokens=5000,
            total_cost=0.1234,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=2, total=8)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#other"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#other this month:" in reply_text
        assert "100 requests" in reply_text
        assert "rank 2/8 channels" in reply_text
        plugin.db.get_usage_summary_for_channel.assert_called_once()
        assert plugin.db.get_usage_summary_for_channel.call_args[0][0] == "#other"

    def test_usage_with_channel_target_via_pm(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target via PM WHEN usage called THEN shows channel stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=50,
            total_prompt_tokens=5000,
            total_completion_tokens=2500,
            total_cost=0.0750,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#somechan"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#somechan this month:" in reply_text
        assert "50 requests" in reply_text


# ---------------------------------------------------------------------------
# remind
# ---------------------------------------------------------------------------


class TestRemindSetCommand:
    """Tests for the real LLM.remind method (set subcommand)."""

    def test_remind_schedules_reminder_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN valid reminder WHEN remind called THEN reminder is scheduled and persisted."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note=None,
        )

        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 30m check the build"])

        # Should schedule the event
        mock_add_event.assert_called_once()
        # Should persist to database
        plugin.db.save_reminder.assert_called_once()
        # Should reply with confirmation
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Reminder set" in reply_text

    def test_remind_includes_note_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN reminder with timezone note WHEN remind called THEN reply includes note."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="meeting",
            confirmation="Reminder set for 1 hour from now.",
            note="Assuming UTC timezone",
        )

        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 1h meeting"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Assuming UTC timezone" in reply_text

    def test_remind_handles_clarification(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse returns clarify WHEN remind called THEN asks for clarification."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )

        plugin.remind(mock_irc, mock_msg, ["something vague"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "When should I remind you?" in reply_text

    def test_remind_rejects_too_short_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration < 10 seconds WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["in 5s test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remind_rejects_too_long_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration > 7 days WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=604801,  # >7 days
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["in 8d test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "7 days" in error_text

    def test_remind_rejects_none_seconds(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result has seconds=None WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.error.call_args[0][0]
        assert "could not determine" in reply_text.lower()

    def test_remind_handles_schedule_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN schedule.addEvent raises WHEN remind called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke"))
        plugin.remind(mock_irc, mock_msg, ["in 1m test"])

        mock_irc.error.assert_called_once()


# ---------------------------------------------------------------------------
# remind list
# ---------------------------------------------------------------------------


class TestRemindListCommand:
    """Tests for the real LLM.remind method (list subcommand)."""

    def test_remind_list_shows_pending_reminders(self, plugin_env):
        """GIVEN user has reminders WHEN remind list called THEN formatted list is shown."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="check build",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="call Bob",
            )

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#1:" in reply_text or "#2:" in reply_text
        assert "check build" in reply_text
        assert "call Bob" in reply_text

    def test_remind_list_shows_no_pending_message(self, plugin_env):
        """GIVEN user has no reminders WHEN remind list called THEN reports none."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending" in reply_text.lower()

    def test_remind_no_args_defaults_to_list(self, plugin_env):
        """GIVEN no arguments WHEN remind called THEN defaults to list."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending" in reply_text.lower()

    def test_remind_list_only_shows_own_reminders(self, plugin_env):
        """GIVEN reminders from different users WHEN remind list called THEN only shows own."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="my reminder",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="otheruser",
                channel="#test",
                message="not mine",
            )

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "my reminder" in reply_text
        assert "not mine" not in reply_text


# ---------------------------------------------------------------------------
# remind delete
# ---------------------------------------------------------------------------


class TestRemindDeleteCommand:
    """Tests for the real LLM.remind method (delete subcommand)."""

    def test_remind_delete_cancels_own_reminder(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns a reminder WHEN remind delete called THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mock_remove = mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["delete 42"])

        # Should remove from schedule
        mock_remove.assert_called_once_with(event_name)
        # Should remove from internal dict
        assert event_name not in plugin._reminders
        # Should delete from database
        plugin.db.delete_reminder.assert_called_once_with(event_name)
        # Should confirm
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_delete_rejects_nonexistent_reminder(self, plugin_env):
        """GIVEN no matching reminder WHEN remind delete called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["delete 999"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "no matching" in error_text.lower()

    def test_remind_delete_rejects_other_users_reminder(self, plugin_env):
        """GIVEN reminder owned by another user WHEN remind delete called THEN error."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_5"] = make_reminder_row(
                event_name="llm_remind_100_5",
                nick="otheruser",
                channel="#test",
                message="their reminder",
            )

        plugin.remind(mock_irc, mock_msg, ["delete 5"])

        mock_irc.error.assert_called_once()

    def test_remind_delete_handles_missing_schedule_event_gracefully(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN reminder exists but schedule event is gone WHEN remind delete THEN no crash."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_7"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mocker.patch("llm.plugin.schedule.removeEvent", side_effect=KeyError("gone"))
        plugin.remind(mock_irc, mock_msg, ["delete 7"])

        # Still confirmed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_del_shorthand_works(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns a reminder WHEN remind del called THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["del 42"])

        assert event_name not in plugin._reminders
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_delete_multiple_ids(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns multiple reminders WHEN remind delete with multiple IDs THEN all cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="first",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="second",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["delete 1 2"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "2 reminders" in reply_text.lower()


# ---------------------------------------------------------------------------
# remind clear
# ---------------------------------------------------------------------------


class TestRemindClearCommand:
    """Tests for the real LLM.remind method (clear subcommand)."""

    def test_remind_clear_removes_all_own_reminders(self, plugin_env, mocker: MockerFixture):
        """GIVEN user has reminders WHEN remind clear called THEN all are removed."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="first",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="second",
            )
            plugin._reminders["llm_remind_100_3"] = make_reminder_row(
                event_name="llm_remind_100_3",
                nick="otheruser",
                channel="#test",
                message="not mine",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "2 reminders" in reply_text.lower()
        # Other user's reminder should remain
        assert "llm_remind_100_3" in plugin._reminders

    def test_remind_clear_reports_no_reminders(self, plugin_env):
        """GIVEN user has no reminders WHEN remind clear called THEN reports none."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no reminders" in reply_text.lower()

    def test_remind_clear_singular_label(self, plugin_env, mocker: MockerFixture):
        """GIVEN user has one reminder WHEN remind clear called THEN uses singular label."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="only one",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "1 reminder." in reply_text


# ---------------------------------------------------------------------------
# Account-based identity
# ---------------------------------------------------------------------------


class TestAccountBasedIdentity:
    """Tests for NickServ account-based identity resolution across commands."""

    @pytest.fixture
    def account_env(self, plugin_env, mocker: MockerFixture):
        """Extend plugin_env so the calling user has a NickServ account."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="MyAccount")
        return plugin, mock_irc, mock_msg

    # -- Usage logging under account --

    def test_ask_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN ask completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
        )

    def test_code_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN code completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["assign"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "code",
            "gpt-4",
            50,
            20,
            0.003,
            prompt="assign",
            status="success",
            error_detail="",
        )

    def test_draw_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN draw completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "draw",
            "dall-e-3",
            10,
            0,
            0.04,
            prompt="a cat",
            status="success",
            error_detail="",
        )

    # -- Context storage under account --

    def test_ask_stores_context_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN ask completes THEN context stored under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Context keyed by account, not nick
        messages = plugin.context.get_messages("MyAccount", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # No context under the raw nick
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    def test_draw_stores_context_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN draw completes THEN context stored under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        messages = plugin.context.get_messages("MyAccount", "#test")
        assert len(messages) == 2
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    # -- Usage query resolves calling user to account --

    def test_usage_channel_resolves_caller_to_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with account WHEN usage called in channel THEN queries by account."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = account_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(10, 1000, 500, 0.01)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 500, 250, 0.005)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, [])

        # Personal stats should query by account name
        plugin.db.get_usage_summary_for_nick.assert_called_once()
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "MyAccount"

    # -- Fallback when no account --

    def test_ask_falls_back_to_nick_when_no_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN user without NickServ account WHEN ask completes THEN usage logged under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default in plugin_env fixture)
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
        )

    def test_ask_falls_back_to_nick_on_keyerror(self, plugin_env, mocker: MockerFixture):
        """GIVEN nickToAccount raises KeyError WHEN ask completes THEN usage logged under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("unknown"))
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
        )

    def test_context_stored_under_nick_when_no_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN user without account WHEN ask completes THEN context stored under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2

    def test_forget_clears_context_under_account(self, account_env):
        """GIVEN user with account has context WHEN forget called THEN account context cleared."""
        plugin, mock_irc, mock_msg = account_env

        # Pre-populate context under account name
        plugin.context.add_message("MyAccount", "#test", "user", "hi")
        plugin.context.add_message("MyAccount", "#test", "assistant", "hello")

        plugin.forget(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower() or "fresh" in reply_text.lower()
        assert len(plugin.context.get_messages("MyAccount", "#test")) == 0

    # -- Lazy nick→account migration --

    def test_migrate_called_when_nick_differs_from_account(self, account_env):
        """GIVEN nick != account WHEN identity resolved THEN migrate_nick called."""
        plugin, mock_irc, _ = account_env
        plugin.db.migrate_nick.return_value = 3

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "MyAccount"
        plugin.db.migrate_nick.assert_called_once_with("testnick", "MyAccount")

    def test_migrate_not_called_when_nick_matches_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick == account (case-insensitive) WHEN resolved THEN no migration."""
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="testnick")

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "testnick"
        plugin.db.migrate_nick.assert_not_called()

    def test_migrate_called_only_once_per_nick(self, account_env):
        """GIVEN nick already migrated WHEN resolved again THEN no second DB call."""
        plugin, mock_irc, _ = account_env
        plugin.db.migrate_nick.return_value = 0

        plugin._resolve_nick_to_identity(mock_irc, "testnick")
        plugin._resolve_nick_to_identity(mock_irc, "testnick")

        plugin.db.migrate_nick.assert_called_once()

    def test_migrate_not_called_when_no_account(self, plugin_env):
        """GIVEN nickToAccount returns None WHEN resolved THEN no migration."""
        plugin, mock_irc, _ = plugin_env

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "testnick"
        plugin.db.migrate_nick.assert_not_called()


# ---------------------------------------------------------------------------
# Additional coverage: edge cases for usage, remind, code
# ---------------------------------------------------------------------------


class TestUsageEdgeCases:
    """Additional edge-case tests for usage command flows."""

    def test_usage_for_nick_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=5)

        plugin.usage(mock_irc, mock_msg, ["somenick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "somenick" in reply_text
        assert "rank" not in reply_text

    def test_usage_for_channel_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#empty"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#empty this month:" in reply_text
        assert "rank" not in reply_text


class TestRemindEdgeCases:
    """Additional edge-case tests for remind command."""

    def test_remind_rejects_negative_seconds_via_min_check(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration < 0 WHEN remind called THEN caught by the <10s check."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=-100,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["yesterday test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remind_uses_input_text_when_no_message(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result with no message WHEN remind called THEN uses original text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message=None,  # No message extracted
            confirmation="Reminder set for 1 minute.",
        )

        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 1m something"])

        # The save_reminder call should use the original text as fallback
        save_call = plugin.db.save_reminder.call_args
        assert save_call[0][3] == "in 1m something"  # message arg = original text


class TestReloadRemindersEdgeCases:
    """Tests for _reload_reminders error handling."""

    def test_reload_reminders_handles_schedule_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN future reminder WHEN schedule.addEvent fails THEN reminder cleaned from DB."""
        import time as time_module

        from llm.persistence import ReminderRow

        plugin, mock_irc, mock_msg = plugin_env
        future_time = time_module.time() + 3600
        created = time_module.time()
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_broken_1",
            nick="testuser",
            channel="#test",
            message="test",
            action_prompt="",
            account=None,
            fire_at=future_time,
            created_at=created,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        )

        plugin.db.load_pending_reminders.return_value = [reminder]

        mocker.patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke"))
        plugin._reload_reminders(mock_irc)

        # Failed reminder should be deleted from DB
        plugin.db.delete_reminder.assert_called_with("llm_remind_broken_1")
        # Should NOT be stored in _reminders dict
        assert "llm_remind_broken_1" not in plugin._reminders


class TestCodeEdgeCases:
    """Additional edge-case tests for code command."""

    def test_code_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN code called THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="print('hi')",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_code_empty_response_triggers_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns empty content WHEN code called THEN irc.error is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.0,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["generate"])

        mock_irc.error.assert_called_once()
        mock_irc.reply.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting: draw respects per-command rate limits
# ---------------------------------------------------------------------------


class TestResolveTier:
    """Tests for _resolve_tier user classification."""

    def test_owner_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with owner capability WHEN _resolve_tier THEN returns 'owner'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "owner",
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == "owner"

    def test_admin_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with admin (not owner) WHEN _resolve_tier THEN returns 'admin'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap in ("admin", "trusted"),
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == "admin"

    def test_trusted_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with trusted (not admin) WHEN _resolve_tier THEN returns 'trusted'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted",
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == "trusted"

    def test_registered_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN identified user without trusted WHEN _resolve_tier THEN returns 'registered'."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "some_account"
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        assert plugin._resolve_tier(mock_irc, mock_msg) == "registered"

    def test_unregistered_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN unidentified user WHEN _resolve_tier THEN returns 'unregistered'."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        assert plugin._resolve_tier(mock_irc, mock_msg) == "unregistered"


class TestRateLimitIntegration:
    """Test that commands respect per-command, per-tier rate limits."""

    def test_draw_rate_limited_when_enforced(self, plugin_env, mocker: MockerFixture):
        """GIVEN rate limit exceeded and enforced WHEN draw called THEN blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"

        # Override to enforce rate limits with low threshold
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "enforceRateLimits": True,
                "drawRateLimitCount": 2,
                "drawRateLimitWindow": 60,
            }.get(key, "")
        )

        # Fill the bucket
        now = time.time()
        plugin._record_rate_limit_hit("draw", "test_account", now - 5)
        plugin._record_rate_limit_hit("draw", "test_account", now - 2)

        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.assistant_request.assert_not_called()

    def test_draw_over_threshold_logs_shadow_when_not_enforced(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN enforce=False and over limit WHEN draw called THEN request runs and shadow log is emitted."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "enforceRateLimits": False,
                "drawRateLimitCount": 1,
                "drawRateLimitWindow": 60,
                "drawContextMaxAgeSeconds": 0,
            }.get(key, "")
        )
        plugin._record_rate_limit_hit("draw", "test_account", time.time() - 2)

        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()
        assert any(
            "rate_limit_shadow" in c.args[0] for c in plugin.log.info.call_args_list if c.args
        )

    def test_ask_succeeds_under_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN user under ask rate limit WHEN ask called THEN request succeeds."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="hello",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()

    def test_code_succeeds_under_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN user under code rate limit WHEN code called THEN request succeeds."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="print('hi') — http://x/code.html",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()

    def test_owner_exempt_from_rate_limits(self, plugin_env, mocker: MockerFixture):
        """GIVEN owner user over limit WHEN draw called THEN not blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "owner_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "drawRateLimitCount": 1,
                    "drawRateLimitWindow": 60,
                }
            )
        )
        # Fill bucket way past limit
        now = time.time()
        for _ in range(10):
            plugin._record_rate_limit_hit("draw", "owner_account", now - 1)

        # Mock: user is owner
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: True,  # owner has all caps
        )
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()

    def test_trusted_gets_relaxed_limits(self, plugin_env, mocker: MockerFixture):
        """GIVEN trusted user within trusted limit but over registered limit WHEN draw called THEN allowed."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "trusted_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "drawRateLimitCount": 2,  # registered: 2
                    "drawRateLimitWindow": 60,
                    "drawTrustedRateLimitCount": 10,  # trusted: 10
                    "drawTrustedRateLimitWindow": 60,
                }
            )
        )
        # 5 hits — over registered limit (2), under trusted limit (10)
        now = time.time()
        for _ in range(5):
            plugin._record_rate_limit_hit("draw", "trusted_account", now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted" or cap.startswith("llm."),
        )
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()

    def test_ask_rate_limited_for_unregistered(self, plugin_env, mocker: MockerFixture):
        """GIVEN unregistered user over unreg limit WHEN ask called THEN blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "askUnregRateLimitCount": 2,
                    "askUnregRateLimitWindow": 60,
                }
            )
        )
        now = time.time()
        # Use nick as bucket key for unregistered users
        nick = "testnick"
        for _ in range(3):
            plugin._record_rate_limit_hit("ask", nick, now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]

    def test_zero_count_disables_rate_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN trusted tier with count=0 WHEN ask called many times THEN never blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "trusted_account"
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="hello",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "askTrustedRateLimitCount": 0,  # 0 = disabled
                    "askTrustedRateLimitWindow": 60,
                }
            )
        )
        now = time.time()
        for _ in range(100):
            plugin._record_rate_limit_hit("ask", "trusted_account", now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted" or cap.startswith("llm."),
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.error.assert_not_called()
        mock_irc.reply.assert_called_once()


# ---------------------------------------------------------------------------
# instruct
# ---------------------------------------------------------------------------


class TestInstructCommand:
    """Tests for the %instruct command."""

    def test_instruct_sets_instruction(self, plugin_env):
        """GIVEN text WHEN instruct called THEN saves to DB and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.instruct(mock_irc, mock_msg, ["You are Captain Picard."])
        plugin.db.save_instruction.assert_called_once_with("testnick", "You are Captain Picard.")
        mock_irc.reply.assert_called_once()
        assert "set" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_no_args_shows_current(self, plugin_env):
        """GIVEN no text and existing instruction WHEN instruct called THEN shows it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Picard."
        plugin.instruct(mock_irc, mock_msg, [])
        mock_irc.reply.assert_called_once()
        assert "Picard" in mock_irc.reply.call_args.args[0]

    def test_instruct_no_args_no_instruction(self, plugin_env):
        """GIVEN no text and no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = None
        plugin.instruct(mock_irc, mock_msg, [])
        mock_irc.reply.assert_called_once()
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_removes(self, plugin_env):
        """GIVEN 'clear' WHEN instruct called THEN deletes and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = True
        plugin.instruct(mock_irc, mock_msg, ["clear"])
        plugin.db.delete_instruction.assert_called_once_with("testnick")
        mock_irc.reply.assert_called_once()
        assert "cleared" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_when_none_set(self, plugin_env):
        """GIVEN 'clear' with no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = False
        plugin.instruct(mock_irc, mock_msg, ["clear"])
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()


# ---------------------------------------------------------------------------
# ask + instruct integration
# ---------------------------------------------------------------------------


class TestAskWithInstruction:
    """Tests for %ask using user instructions from %instruct."""

    def test_ask_prepends_user_instruction(self, plugin_env):
        """GIVEN user has instruction WHEN ask called THEN instruction prepended to system prompt."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Captain Picard."
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Make it so.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])
        call_kwargs = plugin.llm_service.completion.call_args.kwargs
        assert "Picard" in call_kwargs["system_prompt"]

    def test_ask_no_instruction_uses_default(self, plugin_env):
        """GIVEN no instruction WHEN ask called THEN no system_prompt override."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = None
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hello!",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])
        call_kwargs = plugin.llm_service.completion.call_args.kwargs
        assert call_kwargs.get("system_prompt") is None


# ---------------------------------------------------------------------------
# _check_pending_tasks
# ---------------------------------------------------------------------------


class TestCheckPendingTasks:
    """Tests for the _check_pending_tasks polling loop."""

    def test_delivery_exception_does_not_propagate(self, mock_irc, mocker: MockerFixture) -> None:
        """GIVEN _deliver_pending_result raises WHEN _check_pending_tasks runs THEN no exception propagates and error is logged."""
        from llm.service import PendingTaskResult

        from .conftest import plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)

        # Provide a deliverable result
        mock_result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="testnick",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="Hello from AI",
            task_id=42,
        )
        mocker.patch("llm.plugin.world.ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = [mock_result]
        mocker.patch.object(
            plugin, "_deliver_pending_result", side_effect=RuntimeError("delivery failed")
        )

        # Should NOT raise
        plugin._check_pending_tasks()

        plugin.log.error.assert_called()
        # Verify the error message mentions the task_id
        error_args = plugin.log.error.call_args[0]
        assert "42" in str(error_args)

    def test_delivery_exception_does_not_block_other_results(
        self, mock_irc, mocker: MockerFixture
    ) -> None:
        """GIVEN two results and first delivery raises WHEN _check_pending_tasks runs THEN second result is still delivered."""
        from llm.service import PendingTaskResult

        from .conftest import plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)

        result_a = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="Reply A",
            task_id=1,
        )
        result_b = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="bob",
            reply_target="#test",
            is_channel=True,
            prompt_preview="world",
            model="gpt-4",
            content="Reply B",
            task_id=2,
        )

        mocker.patch("llm.plugin.world.ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = [result_a, result_b]

        call_count = 0

        def deliver_side_effect(r: PendingTaskResult) -> None:
            nonlocal call_count
            call_count += 1
            if r.task_id == 1:
                raise RuntimeError("delivery failed for first")

        mocker.patch.object(plugin, "_deliver_pending_result", side_effect=deliver_side_effect)

        plugin._check_pending_tasks()

        # Both results were attempted
        assert call_count == 2
