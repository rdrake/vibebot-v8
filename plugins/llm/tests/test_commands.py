"""Tests that call actual plugin command methods (not reimplementations).

These tests exercise the real ask, code, draw, forget, llmkeys, usage,
remindme, reminders, and unremind methods on a properly-initialised LLM
plugin instance with mocked dependencies.

Unlike the _call_ask / _call_code / _call_draw helpers in test_plugin.py
which reimplement command logic, these tests invoke the actual methods so
regressions in the real command code are caught.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.persistence import UsageBreakdown, UsageSummary
from llm.plugin import LLM
from llm.service import CompletionResult, ImageResult, ReminderParseResult

from .conftest import make_registry_side_effect

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin_env(mocker: MockerFixture):
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    registry = make_registry_side_effect()

    mock_irc = mocker.MagicMock()
    mock_irc.nick = "testbot"
    mock_irc.state = mocker.MagicMock()
    mock_irc.state.channels = {"#test": mocker.MagicMock(topic="Test topic")}
    mock_irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

    mock_msg = mocker.MagicMock()
    mock_msg.prefix = "testnick!user@host"
    mock_msg.args = ("#test", "test message")
    mock_msg.time = time.time() + 100  # future time -- not ZNC playback
    mock_msg.channel = "#test"
    mock_msg.nick = "testnick"

    mocker.patch.object(LLM, "registryValue", side_effect=registry)
    mocker.patch("llm.plugin.LLMService")
    mocker.patch("llm.plugin.LLMDatabase")
    mocker.patch("llm.plugin.log")
    mocker.patch("llm.plugin.httpserver")
    mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    mocker.patch("llm.plugin.schedule.removeEvent")
    mocker.patch("llm.plugin.schedule.addEvent")

    plugin = LLM(mock_irc)
    # After __init__, swap registryValue to a plain MagicMock so
    # each test can override specific keys while keeping defaults.
    plugin.registryValue = mocker.MagicMock(side_effect=registry)

    # Provide the MetaSynchronized RLock that _allow_concurrent expects.
    plugin._MetaSynchronized_rlock = threading.RLock()

    # sanitize_output is a passthrough in tests (the mock would return MagicMock).
    plugin.llm_service.sanitize_output.side_effect = lambda x: x

    # migrate_nick returns an int (0 = nothing to migrate) by default.
    plugin.db.migrate_nick.return_value = 0

    # is_user_flagged returns False by default (user not flagged)
    plugin.db.is_user_flagged.return_value = False

    return plugin, mock_irc, mock_msg


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class TestAskCommand:
    """Tests for the real LLM.ask method."""

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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["search", "something"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")
        assert "searched result" in reply_text

    def test_ask_with_images_sends_processing_message(self, plugin_env, mocker: MockerFixture):
        """GIVEN prompt with image URL WHEN ask is called THEN processing message is sent first."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = ["http://img.example/pic.jpg"]
        plugin.llm_service.completion.return_value = CompletionResult(
            content="I see an image",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["describe", "http://img.example/pic.jpg"])

        # First call is the "Processing with N image(s)..." message
        assert mock_irc.reply.call_count == 2
        first_reply = mock_irc.reply.call_args_list[0]
        assert "image" in first_reply[0][0].lower()

    def test_ask_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN completion returns an error WHEN ask completes THEN context is NOT stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


class TestCodeCommand:
    """Tests for the real LLM.code method."""

    def test_code_replies_with_url_and_preview(self, plugin_env, mocker: MockerFixture):
        """GIVEN code generation succeeds WHEN code called THEN reply has preview and URL."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="def hello(): pass",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = (
            "http://localhost:8080/llm/code_abc.html"
        )
        plugin.llm_service.summarize.return_value = None  # fallback to truncation
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://localhost:8080/llm/code_abc.html" in reply_text

    def test_code_uses_ai_summary_when_available(self, plugin_env, mocker: MockerFixture):
        """GIVEN summarize returns a summary WHEN code called THEN reply uses AI summary."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)",
            prompt_tokens=10,
            completion_tokens=20,
            cost=0.002,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "http://localhost:8080/llm/code_x.html"
        plugin.llm_service.summarize.return_value = "Recursive Fibonacci function"
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["fibonacci"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Recursive Fibonacci function" in reply_text

    def test_code_falls_back_to_irc_on_save_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN save_code_to_http returns None WHEN code called THEN raw response is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="print('hello')",
            prompt_tokens=5,
            completion_tokens=3,
            cost=0.0,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = None

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["print", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "print('hello')" in reply_text

    def test_code_stores_context(self, plugin_env, mocker: MockerFixture):
        """GIVEN code command succeeds WHEN executed THEN conversation context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="code output",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "http://x/code.html"
        plugin.llm_service.summarize.return_value = None
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["generate", "something"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_code_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN code completion with cost WHEN code completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = None

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_code_grounding_icon_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN code saved to URL THEN reply has globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="code",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "http://x/c.html"
        plugin.llm_service.summarize.return_value = "summary"
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


class TestDrawCommand:
    """Tests for the real LLM.draw method."""

    def test_draw_requires_nickserv_auth(self, plugin_env, mocker: MockerFixture):
        """GIVEN unidentified user WHEN draw called THEN error about NickServ identification."""
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default in plugin_env)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "NickServ" in error_text
        plugin.llm_service.image_generation.assert_not_called()

    def test_draw_replies_with_image_url(self, plugin_env, mocker: MockerFixture):
        """GIVEN image generation succeeds WHEN draw called THEN irc.reply has image URL."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        mock_irc.reply.assert_called_once_with("http://img.example/gen.png")

    def test_draw_shows_rewritten_prompt_when_present(self, plugin_env, mocker: MockerFixture):
        """GIVEN image result has rewritten_prompt WHEN draw called THEN reply includes it."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
            rewritten_prompt="A beautiful sunset over mountains",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["sunset"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Rewritten" in reply_text
        assert "A beautiful sunset over mountains" in reply_text
        assert "http://img.example/gen.png" in reply_text

    def test_draw_truncates_long_rewritten_prompt(self, plugin_env, mocker: MockerFixture):
        """GIVEN rewritten_prompt is >200 chars WHEN draw called THEN prompt is truncated."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        long_prompt = "A" * 250
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
            rewritten_prompt=long_prompt,
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "..." in reply_text

    def test_draw_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw with cost WHEN draw completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="Error: content blocked",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
            error="Error: content blocked",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="Error: timeout exceeded",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
            error="Error: timeout exceeded",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["sunset"])

        mock_irc.reply.assert_not_called()

    def test_draw_stores_context_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw succeeds WHEN executed THEN personal and channel context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "a sunset"
        assert messages[1]["role"] == "assistant"
        assert "[Generated image:" in messages[1]["content"]

    def test_draw_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw returns error WHEN executed THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["bad", "prompt"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0

    def test_draw_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN draw succeeds THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0


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

    def test_forget_reports_no_context(self, plugin_env):
        """GIVEN user has no context WHEN forget called THEN reports no context."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.forget(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "no" in reply_text.lower()

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
# llmkeys
# ---------------------------------------------------------------------------


class TestLlmkeysCommand:
    """Tests for the real LLM.llmkeys method."""

    def test_llmkeys_shows_key_status_privately(self, plugin_env, mocker: MockerFixture):
        """GIVEN admin user WHEN llmkeys called THEN key status sent as private reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.safe_key_display.return_value = "tes...(10 chars hidden)"

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.llmkeys(mock_irc, mock_msg, [])

        # Should be sent privately
        mock_irc.reply.assert_called_once()
        assert mock_irc.reply.call_args.kwargs.get("private") is True

        # Should call safe_key_display for all 4 keys (ask, code, draw, animate)
        assert plugin.llm_service.safe_key_display.call_count == 4

    def test_llmkeys_response_contains_all_key_types(self, plugin_env, mocker: MockerFixture):
        """GIVEN admin WHEN llmkeys called THEN response mentions ask, code, draw."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.safe_key_display.return_value = "abc...(5 chars hidden)"

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.llmkeys(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "ask=" in reply_text
        assert "code=" in reply_text
        assert "draw=" in reply_text
        assert "animate=" in reply_text


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


class TestUsageCommand:
    """Tests for the real LLM.usage method (dual-mode: channel + PM)."""

    @pytest.fixture(autouse=True)
    def _mock_addressed(self, mocker: MockerFixture):
        """Mock callbacks.addressed so _extract_raw_arg doesn't hit real Limnoria."""
        mocker.patch("llm.plugin.callbacks.addressed", return_value=None)

    # -- PM mode (global stats, admin only) --

    def test_usage_pm_shows_today_and_month_stats(self, plugin_env, mocker: MockerFixture):
        """GIVEN admin via PM WHEN usage called THEN response includes today and monthly stats."""
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Today:" in reply_text
        assert "This month:" in reply_text
        assert "Top users:" in reply_text
        assert "Top channels:" in reply_text
        # Sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_usage_pm_with_no_top_users_or_channels(self, plugin_env, mocker: MockerFixture):
        """GIVEN no usage data via PM WHEN usage called THEN response omits top users/channels."""
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

    # -- Target nick mode --

    def test_usage_strips_irc_status_prefix_from_nick(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with @ prefix WHEN usage called THEN prefix stripped before lookup."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(7, 800, 400, 0.01)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage @Larry")
        plugin.usage(mock_irc, mock_msg, [])

        # Should query for "Larry", not "@Larry"
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "Larry"
        assert "Larry" in mock_irc.reply.call_args[0][0]

    def test_usage_handles_nick_with_brackets(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with brackets WHEN usage called THEN raw arg parsed correctly."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(3, 300, 150, 0.005)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=4)

        # Mock the raw message extraction to return the bracket nick intact
        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage Rubin[F]")
        plugin.usage(mock_irc, mock_msg, [])

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

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage OldNick")
        plugin.usage(mock_irc, mock_msg, [])

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

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage othernick")
        plugin.usage(mock_irc, mock_msg, [])

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

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage othernick")
        plugin.usage(mock_irc, mock_msg, [])

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

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage #other")
        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, [])

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

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage #somechan")
        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#somechan this month:" in reply_text
        assert "50 requests" in reply_text


# ---------------------------------------------------------------------------
# remindme
# ---------------------------------------------------------------------------


class TestRemindmeCommand:
    """Tests for the real LLM.remindme method."""

    def test_remindme_schedules_reminder_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN valid reminder WHEN remindme called THEN reminder is scheduled and persisted."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note=None,
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remindme(mock_irc, mock_msg, ["in", "30m", "check", "the", "build"])

        # Should schedule the event
        mock_add_event.assert_called_once()
        # Should persist to database
        plugin.db.save_reminder.assert_called_once()
        # Should reply with confirmation
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Reminder set" in reply_text

    def test_remindme_includes_note_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN reminder with timezone note WHEN remindme called THEN reply includes note."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="meeting",
            confirmation="Reminder set for 1 hour from now.",
            note="Assuming UTC timezone",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remindme(mock_irc, mock_msg, ["in", "1h", "meeting"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Assuming UTC timezone" in reply_text

    def test_remindme_handles_clarification(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse returns clarify WHEN remindme called THEN asks for clarification."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.remindme(mock_irc, mock_msg, ["something", "vague"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "When should I remind you?" in reply_text

    def test_remindme_rejects_too_short_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration < 10 seconds WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.remindme(mock_irc, mock_msg, ["in", "5s", "test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remindme_rejects_too_long_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration > 7 days WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=604801,  # >7 days
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.remindme(mock_irc, mock_msg, ["in", "8d", "test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "7 days" in error_text

    def test_remindme_rejects_none_seconds(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result has seconds=None WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.remindme(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "couldn't determine" in reply_text.lower()

    def test_remindme_handles_schedule_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN schedule.addEvent raises WHEN remindme called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke"))
        plugin.remindme(mock_irc, mock_msg, ["in", "1m", "test"])

        mock_irc.error.assert_called_once()


# ---------------------------------------------------------------------------
# reminders
# ---------------------------------------------------------------------------


class TestRemindersCommand:
    """Tests for the real LLM.reminders method."""

    def test_reminders_lists_pending_reminders(self, plugin_env):
        """GIVEN user has reminders WHEN reminders called THEN formatted list is shown."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = ("testnick", "#test", "check build")
            plugin._reminders["llm_remind_100_2"] = ("testnick", "#test", "call Bob")

        plugin.reminders(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#1:" in reply_text or "#2:" in reply_text
        assert "check build" in reply_text
        assert "call Bob" in reply_text

    def test_reminders_shows_no_pending_message(self, plugin_env):
        """GIVEN user has no reminders WHEN reminders called THEN reports none."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.reminders(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending" in reply_text.lower()

    def test_reminders_only_shows_own_reminders(self, plugin_env):
        """GIVEN reminders from different users WHEN reminders called THEN only shows own."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = ("testnick", "#test", "my reminder")
            plugin._reminders["llm_remind_100_2"] = ("otheruser", "#test", "not mine")

        plugin.reminders(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "my reminder" in reply_text
        assert "not mine" not in reply_text


# ---------------------------------------------------------------------------
# unremind
# ---------------------------------------------------------------------------


class TestUnremindCommand:
    """Tests for the real LLM.unremind method."""

    def test_unremind_cancels_own_reminder(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns a reminder WHEN unremind called with ID THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = ("testnick", "#test", "my reminder")

        mock_remove = mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.unremind(mock_irc, mock_msg, ["42"])

        # Should remove from schedule
        mock_remove.assert_called_once_with(event_name)
        # Should remove from internal dict
        assert event_name not in plugin._reminders
        # Should delete from database
        plugin.db.delete_reminder.assert_called_once_with(event_name)
        # Should confirm
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_unremind_rejects_nonexistent_reminder(self, plugin_env):
        """GIVEN no matching reminder WHEN unremind called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.unremind(mock_irc, mock_msg, ["999"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "not found" in error_text.lower()

    def test_unremind_rejects_other_users_reminder(self, plugin_env):
        """GIVEN reminder owned by another user WHEN unremind called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_5"] = ("otheruser", "#test", "their reminder")

        plugin.unremind(mock_irc, mock_msg, ["5"])

        mock_irc.error.assert_called_once()

    def test_unremind_handles_missing_schedule_event_gracefully(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN reminder exists but schedule event is gone WHEN unremind called THEN no crash."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_7"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = ("testnick", "#test", "my reminder")

        mocker.patch("llm.plugin.schedule.removeEvent", side_effect=KeyError("gone"))
        # Should not raise
        plugin.unremind(mock_irc, mock_msg, ["7"])

        # Still confirmed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()


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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.completion.return_value = CompletionResult(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = None

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.callbacks.addressed", return_value=None)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
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
# Additional coverage: edge cases for usage, remindme, code
# ---------------------------------------------------------------------------


class TestUsageEdgeCases:
    """Additional edge-case tests for usage command flows."""

    @pytest.fixture(autouse=True)
    def _mock_addressed(self, mocker: MockerFixture):
        """Mock callbacks.addressed so _extract_raw_arg doesn't hit real Limnoria."""
        mocker.patch("llm.plugin.callbacks.addressed", return_value=None)

    def test_usage_for_nick_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=5)

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage somenick")
        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "somenick" in reply_text
        assert "rank" not in reply_text

    def test_usage_for_channel_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage #empty")
        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#empty this month:" in reply_text
        assert "rank" not in reply_text


class TestExtractRawArgEdgeCases:
    """Tests for _extract_raw_arg edge cases."""

    def test_extract_raw_arg_command_not_in_payload(self, mocker: MockerFixture):
        """GIVEN payload without the command WHEN _extract_raw_arg THEN returns None."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_msg = mocker.MagicMock()

        mocker.patch("llm.plugin.callbacks.addressed", return_value="help something")
        result = LLM._extract_raw_arg(mock_irc, mock_msg, "usage")

        assert result is None


class TestRemindmeEdgeCases:
    """Additional edge-case tests for remindme command."""

    def test_remindme_rejects_negative_seconds_via_min_check(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN duration < 0 WHEN remindme called THEN caught by the <10s check."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=-100,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.remindme(mock_irc, mock_msg, ["yesterday", "test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remindme_uses_input_text_when_no_message(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result with no message WHEN remindme called THEN uses original text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message=None,  # No message extracted
            confirmation="Reminder set for 1 minute.",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remindme(mock_irc, mock_msg, ["in", "1m", "something"])

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
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_broken_1",
            nick="testuser",
            channel="#test",
            message="test",
            fire_at=future_time,
            created_at=time_module.time(),
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
        plugin.llm_service.completion.return_value = CompletionResult(
            content="print('hi')",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = None

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_code_truncates_long_preview_without_summary(self, plugin_env, mocker: MockerFixture):
        """GIVEN long code and no AI summary WHEN code called THEN preview is truncated."""
        plugin, mock_irc, mock_msg = plugin_env
        long_code = "x" * 200  # > CODE_PREVIEW_MAX_LEN (60)
        plugin.llm_service.completion.return_value = CompletionResult(
            content=long_code,
            prompt_tokens=10,
            completion_tokens=50,
            cost=0.002,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "http://x/code.html"
        plugin.llm_service.summarize.return_value = None  # no AI summary
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["generate"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "..." in reply_text
        assert "http://x/code.html" in reply_text
        # Preview should be truncated to ~60 chars
        preview_part = reply_text.split(" — ")[0]
        assert len(preview_part) <= 61  # 57 + "..."


# ---------------------------------------------------------------------------
# flag / unflag / flagged admin commands
# ---------------------------------------------------------------------------


class TestFlagCommands:
    """Tests for the flag, unflag, and flagged admin commands."""

    def test_flag_flags_user(self, plugin_env, mocker: MockerFixture):
        """GIVEN identified target WHEN flag called THEN db.flag_user called and reply sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="target_account")
        plugin.db.flag_user.return_value = True

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flag(mock_irc, mock_msg, ["baduser", "spamming"])

        plugin.db.flag_user.assert_called_once_with(
            "target_account", "spamming", auto_flagged=False
        )
        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Flagged" in reply_text
        assert "baduser" in reply_text
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_flag_rejects_unidentified_target(self, plugin_env, mocker: MockerFixture):
        """GIVEN nickToAccount returns None WHEN flag called THEN error sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flag(mock_irc, mock_msg, ["unknown", "testing"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "NickServ" in error_text
        plugin.db.flag_user.assert_not_called()

    def test_flag_handles_already_flagged(self, plugin_env, mocker: MockerFixture):
        """GIVEN already flagged user WHEN flag called THEN reply says already flagged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="target_account")
        plugin.db.flag_user.return_value = False

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flag(mock_irc, mock_msg, ["baduser", "spamming"])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "already flagged" in reply_text

    def test_flag_handles_nick_to_account_keyerror(self, plugin_env, mocker: MockerFixture):
        """GIVEN nickToAccount raises KeyError WHEN flag called THEN error sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("not found"))

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flag(mock_irc, mock_msg, ["ghost", "testing"])

        mock_irc.error.assert_called_once()
        plugin.db.flag_user.assert_not_called()

    def test_unflag_unflags_user(self, plugin_env, mocker: MockerFixture):
        """GIVEN flagged user WHEN unflag called THEN db.unflag_user called and reply sent."""
        plugin, mock_irc, mock_msg = plugin_env

        def nick_to_account(nick):
            if nick == "baduser":
                return "target_account"
            return "admin_account"

        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=nick_to_account)
        plugin.db.unflag_user.return_value = True

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.unflag(mock_irc, mock_msg, ["baduser"])

        plugin.db.unflag_user.assert_called_once_with("target_account", "admin_account")
        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Unflagged" in reply_text
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_unflag_rejects_unidentified(self, plugin_env, mocker: MockerFixture):
        """GIVEN nickToAccount returns None WHEN unflag called THEN error sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.unflag(mock_irc, mock_msg, ["unknown"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "NickServ" in error_text
        plugin.db.unflag_user.assert_not_called()

    def test_unflag_handles_not_flagged(self, plugin_env, mocker: MockerFixture):
        """GIVEN user not flagged WHEN unflag called THEN reply says not flagged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="target_account")
        plugin.db.unflag_user.return_value = False

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.unflag(mock_irc, mock_msg, ["gooduser"])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "not currently flagged" in reply_text

    def test_flagged_lists_users(self, plugin_env, mocker: MockerFixture):
        """GIVEN flagged users exist WHEN flagged called THEN lists them."""
        from llm.persistence import FlaggedUserRow

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_flagged_users.return_value = [
            FlaggedUserRow(
                id=1,
                account="alice",
                flagged_at=time.time(),
                reason="spamming",
                auto_flagged=1,
                resolved_at=None,
                resolved_by=None,
            ),
            FlaggedUserRow(
                id=2,
                account="bob",
                flagged_at=time.time(),
                reason="abuse",
                auto_flagged=0,
                resolved_at=None,
                resolved_by=None,
            ),
        ]

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flagged(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "alice (auto): spamming" in reply_text
        assert "bob (manual): abuse" in reply_text
        assert " | " in reply_text
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_flagged_empty(self, plugin_env, mocker: MockerFixture):
        """GIVEN no flagged users WHEN flagged called THEN reports no flagged users."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_flagged_users.return_value = []

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.flagged(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "No flagged users" in reply_text
        assert mock_irc.reply.call_args.kwargs.get("private") is True


# ---------------------------------------------------------------------------
# Rate limiting: draw/animate respect per-command rate limits
# ---------------------------------------------------------------------------


class TestRateLimitIntegration:
    """Test that draw/animate commands respect per-command rate limits."""

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

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.image_generation.assert_not_called()

    def test_animate_rate_limited_when_enforced(self, plugin_env, mocker: MockerFixture):
        """GIVEN rate limit exceeded and enforced WHEN animate called THEN blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "enforceRateLimits": True,
                "animateRateLimitCount": 1,
                "animateRateLimitWindow": 600,
            }.get(key, "")
        )

        now = time.time()
        plugin._record_rate_limit_hit("animate", "test_account", now - 2)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.animate(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.video_generation.assert_not_called()

    def test_draw_over_threshold_logs_shadow_when_not_enforced(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN enforce=False and over limit WHEN draw called THEN request runs and shadow log is emitted."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
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
            }.get(key, "")
        )
        plugin._record_rate_limit_hit("draw", "test_account", time.time() - 2)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.image_generation.assert_called_once()
        assert any(
            "rate_limit_shadow" in c.args[0] for c in plugin.log.info.call_args_list if c.args
        )

    def test_ask_not_rate_limited(self, plugin_env, mocker: MockerFixture):
        """GIVEN rate limits enforced WHEN ask called THEN no rate check applied."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="hello",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()

    def test_code_not_rate_limited(self, plugin_env, mocker: MockerFixture):
        """GIVEN rate limits enforced WHEN code called THEN no rate check is applied."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="print('hi')",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.save_code_to_http.return_value = "http://x/code.html"
        plugin.llm_service.summarize.return_value = "small summary"

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
