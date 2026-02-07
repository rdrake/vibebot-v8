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
from unittest.mock import MagicMock, patch

import pytest
from llm.persistence import UsageBreakdown, UsageSummary
from llm.plugin import LLM
from llm.service import CompletionResult, ImageResult, ReminderParseResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Default config values returned by registryValue in tests
_DEFAULT_CONFIG: dict[str, object] = {
    "httpRoot": "/tmp/llm-test-web",
    "httpUrlBase": "http://localhost:8080/llm",
    "databasePath": "",
    "contextMaxMessages": 20,
    "contextTimeoutMinutes": 30,
    "contextEnabled": True,
    "channelContextMaxMessages": 10,
    "contextTrackAllMessages": False,
    "askApiKey": "test-key",
    "askModel": "gpt-4",
    "askSystemPrompt": "You are helpful.",
    "codeApiKey": "test-key",
    "codeModel": "gpt-4",
    "codeSystemPrompt": "You write code.",
    "drawApiKey": "test-key",
    "drawModel": "dall-e-3",
    "drawTimeout": 60,
    "drawAutoRewriteMax": 2,
    "timeout": 30,
    "maxPromptLength": 10000,
    "commandPrefixes": [".", "/"],
    "fileCleanupAge": 24,
    "fileCleanupMax": 100,
}


def _registry(key: str, *args: object) -> object:
    """Simulate registryValue look-ups for tests."""
    return _DEFAULT_CONFIG.get(key, "")


@pytest.fixture
def plugin_env():
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    mock_irc = MagicMock()
    mock_irc.nick = "testbot"
    mock_irc.state = MagicMock()
    mock_irc.state.channels = {"#test": MagicMock(topic="Test topic")}
    mock_irc.state.capabilities_ack = set()

    mock_msg = MagicMock()
    mock_msg.prefix = "testnick!user@host"
    mock_msg.args = ("#test", "test message")
    mock_msg.time = time.time() + 100  # future time -- not ZNC playback
    mock_msg.channel = "#test"
    mock_msg.nick = "testnick"

    with (
        patch.object(LLM, "registryValue", side_effect=_registry),
        patch("llm.plugin.LLMService"),
        patch("llm.plugin.LLMDatabase"),
        patch("llm.plugin.log"),
        patch("llm.plugin.httpserver"),
        patch("llm.plugin.schedule.addPeriodicEvent"),
        patch("llm.plugin.schedule.removeEvent"),
        patch("llm.plugin.schedule.addEvent"),
    ):
        plugin = LLM(mock_irc)
        # After __init__, swap registryValue to a plain MagicMock so
        # each test can override specific keys while keeping defaults.
        plugin.registryValue = MagicMock(side_effect=_registry)

    # Provide the MetaSynchronized RLock that _allow_concurrent expects.
    plugin._MetaSynchronized_rlock = threading.RLock()

    return plugin, mock_irc, mock_msg


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class TestAskCommand:
    """Tests for the real LLM.ask method."""

    def test_ask_replies_with_completion_content(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["What", "is", "Python?"])

        mock_irc.reply.assert_called_once_with("Hello from AI", prefixNick=False)

    def test_ask_stores_context_on_success(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["hello"])

        # Context should have both user and assistant messages
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "response text"

    def test_ask_logs_usage(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick", "#test", "ask", "gpt-4", 100, 50, 0.005
        )

    def test_ask_skips_znc_playback(self, plugin_env):
        """GIVEN a message older than startup WHEN ask called THEN no reply is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100  # before startup

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_ask_prepends_grounding_icon_when_used(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["search", "something"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")
        assert "searched result" in reply_text

    def test_ask_with_images_sends_processing_message(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["describe", "http://img.example/pic.jpg"])

        # First call is the "Processing with N image(s)..." message
        assert mock_irc.reply.call_count == 2
        first_reply = mock_irc.reply.call_args_list[0]
        assert "image" in first_reply[0][0].lower()

    def test_ask_does_not_store_context_on_error(self, plugin_env):
        """GIVEN completion returns an error WHEN ask completes THEN context is NOT stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["hello"])

        # No context should be stored because result has an error
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_ask_skips_context_when_disabled(self, plugin_env):
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

        def disabled_registry(key, *args):
            if key == "contextEnabled":
                return False
            return _registry(key, *args)

        plugin.registryValue = MagicMock(side_effect=disabled_registry)

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


class TestCodeCommand:
    """Tests for the real LLM.code method."""

    def test_code_replies_with_url_and_preview(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://localhost:8080/llm/code_abc.html" in reply_text

    def test_code_uses_ai_summary_when_available(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["fibonacci"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Recursive Fibonacci function" in reply_text

    def test_code_falls_back_to_irc_on_save_failure(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["print", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "print('hello')" in reply_text

    def test_code_stores_context(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["generate", "something"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_code_logs_usage(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["assign"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick", "#test", "code", "gpt-4", 50, 20, 0.003
        )

    def test_code_skips_znc_playback(self, plugin_env):
        """GIVEN an old message WHEN code called THEN nothing happens."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_code_grounding_icon_in_reply(self, plugin_env):
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.code(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


class TestDrawCommand:
    """Tests for the real LLM.draw method."""

    def test_draw_replies_with_image_url(self, plugin_env):
        """GIVEN image generation succeeds WHEN draw called THEN irc.reply has image URL."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        mock_irc.reply.assert_called_once_with("http://img.example/gen.png")

    def test_draw_shows_rewritten_prompt_when_present(self, plugin_env):
        """GIVEN image result has rewritten_prompt WHEN draw called THEN reply includes it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
            rewritten_prompt="A beautiful sunset over mountains",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["sunset"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Rewritten" in reply_text
        assert "A beautiful sunset over mountains" in reply_text
        assert "http://img.example/gen.png" in reply_text

    def test_draw_truncates_long_rewritten_prompt(self, plugin_env):
        """GIVEN rewritten_prompt is >200 chars WHEN draw called THEN prompt is truncated."""
        plugin, mock_irc, mock_msg = plugin_env
        long_prompt = "A" * 250
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
            rewritten_prompt=long_prompt,
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "..." in reply_text

    def test_draw_logs_usage(self, plugin_env):
        """GIVEN draw with cost WHEN draw completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick", "#test", "draw", "dall-e-3", 10, 0, 0.04
        )

    def test_draw_skips_usage_logging_when_zero_cost(self, plugin_env):
        """GIVEN draw with zero cost and tokens WHEN draw completes THEN no usage logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_not_called()

    def test_draw_skips_znc_playback(self, plugin_env):
        """GIVEN old message WHEN draw called THEN no reply."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.draw(mock_irc, mock_msg, ["sunset"])

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

    def test_llmkeys_shows_key_status_privately(self, plugin_env):
        """GIVEN admin user WHEN llmkeys called THEN key status sent as private reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.safe_key_display.return_value = "tes...(10 chars hidden)"

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.llmkeys(mock_irc, mock_msg, [])

        # Should be sent privately
        mock_irc.reply.assert_called_once()
        assert mock_irc.reply.call_args.kwargs.get("private") is True

        # Should call safe_key_display for all 3 keys
        assert plugin.llm_service.safe_key_display.call_count == 3

    def test_llmkeys_response_contains_all_key_types(self, plugin_env):
        """GIVEN admin WHEN llmkeys called THEN response mentions ask, code, draw."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.safe_key_display.return_value = "abc...(5 chars hidden)"

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.llmkeys(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "ask=" in reply_text
        assert "code=" in reply_text
        assert "draw=" in reply_text


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


class TestUsageCommand:
    """Tests for the real LLM.usage method."""

    def test_usage_shows_today_and_month_stats(self, plugin_env):
        """GIVEN admin WHEN usage called THEN response includes today and monthly stats."""
        plugin, mock_irc, mock_msg = plugin_env
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

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Today:" in reply_text
        assert "This month:" in reply_text
        assert "Top users:" in reply_text
        assert "Top channels:" in reply_text
        # Sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_usage_with_no_top_users_or_channels(self, plugin_env):
        """GIVEN no usage data WHEN usage called THEN response omits top users/channels."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cost=0.0,
        )
        plugin.db.get_usage_by_nick.return_value = []
        plugin.db.get_usage_by_channel.return_value = []

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Top users:" not in reply_text
        assert "Top channels:" not in reply_text


# ---------------------------------------------------------------------------
# remindme
# ---------------------------------------------------------------------------


class TestRemindmeCommand:
    """Tests for the real LLM.remindme method."""

    def test_remindme_schedules_reminder_on_success(self, plugin_env):
        """GIVEN valid reminder WHEN remindme called THEN reminder is scheduled and persisted."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note=None,
        )

        with (
            patch("llm.plugin.ircdb.checkCapability", return_value=True),
            patch("llm.plugin.schedule.addEvent") as mock_add_event,
        ):
            plugin.remindme(mock_irc, mock_msg, ["in", "30m", "check", "the", "build"])

        # Should schedule the event
        mock_add_event.assert_called_once()
        # Should persist to database
        plugin.db.save_reminder.assert_called_once()
        # Should reply with confirmation
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Reminder set" in reply_text

    def test_remindme_includes_note_in_reply(self, plugin_env):
        """GIVEN reminder with timezone note WHEN remindme called THEN reply includes note."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="meeting",
            confirmation="Reminder set for 1 hour from now.",
            note="Assuming UTC timezone",
        )

        with (
            patch("llm.plugin.ircdb.checkCapability", return_value=True),
            patch("llm.plugin.schedule.addEvent"),
        ):
            plugin.remindme(mock_irc, mock_msg, ["in", "1h", "meeting"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Assuming UTC timezone" in reply_text

    def test_remindme_handles_clarification(self, plugin_env):
        """GIVEN parse returns clarify WHEN remindme called THEN asks for clarification."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.remindme(mock_irc, mock_msg, ["something", "vague"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "When should I remind you?" in reply_text

    def test_remindme_rejects_too_short_duration(self, plugin_env):
        """GIVEN duration < 10 seconds WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="test",
            confirmation="ok",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.remindme(mock_irc, mock_msg, ["in", "5s", "test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remindme_rejects_too_long_duration(self, plugin_env):
        """GIVEN duration > 7 days WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=604801,  # >7 days
            message="test",
            confirmation="ok",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.remindme(mock_irc, mock_msg, ["in", "8d", "test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "7 days" in error_text

    def test_remindme_rejects_none_seconds(self, plugin_env):
        """GIVEN parse result has seconds=None WHEN remindme called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
            message="test",
            confirmation="ok",
        )

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.remindme(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "couldn't determine" in reply_text.lower()

    def test_remindme_handles_schedule_failure(self, plugin_env):
        """GIVEN schedule.addEvent raises WHEN remindme called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message="test",
            confirmation="ok",
        )

        with (
            patch("llm.plugin.ircdb.checkCapability", return_value=True),
            patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke")),
        ):
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

    def test_unremind_cancels_own_reminder(self, plugin_env):
        """GIVEN user owns a reminder WHEN unremind called with ID THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = ("testnick", "#test", "my reminder")

        with patch("llm.plugin.schedule.removeEvent") as mock_remove:
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

    def test_unremind_handles_missing_schedule_event_gracefully(self, plugin_env):
        """GIVEN reminder exists but schedule event is gone WHEN unremind called THEN no crash."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_7"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = ("testnick", "#test", "my reminder")

        with patch("llm.plugin.schedule.removeEvent", side_effect=KeyError("gone")):
            # Should not raise
            plugin.unremind(mock_irc, mock_msg, ["7"])

        # Still confirmed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()
