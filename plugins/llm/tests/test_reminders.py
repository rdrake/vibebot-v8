"""Tests for reminder commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.service import ReminderParseResult

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestReminderCommands:
    """Tests for the consolidated remind command."""

    def test_remind_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for remind THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "remind")
        assert callable(LLM.remind)

    def test_remind_docstring_shows_natural_language_examples(self) -> None:
        """GIVEN remind command WHEN checking docstring THEN shows examples."""
        from llm.plugin import LLM

        doc = LLM.remind.__doc__ or ""
        assert "natural language" in doc.lower() or "in 30 minutes" in doc


class TestReminderParseResult:
    """Tests for ReminderParseResult NamedTuple."""

    def test_schedule_result(self) -> None:
        """GIVEN schedule action WHEN creating result THEN stores all fields."""
        result = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note="Assuming UTC timezone.",
        )
        assert result.action == "schedule"
        assert result.seconds == 1800
        assert result.message == "check the build"
        assert result.confirmation == "Reminder set for 30 minutes from now."
        assert result.note == "Assuming UTC timezone."

    def test_clarify_result(self) -> None:
        """GIVEN clarify action WHEN creating result THEN stores confirmation."""
        result = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )
        assert result.action == "clarify"
        assert result.seconds is None
        assert result.message is None
        assert result.confirmation == "When should I remind you?"

    def test_default_values(self) -> None:
        """GIVEN minimal args WHEN creating result THEN defaults are correct."""
        result = ReminderParseResult(action="clarify")
        assert result.seconds is None
        assert result.message is None
        assert result.confirmation == ""
        assert result.note is None


class TestReminderHelperMethods:
    """Tests for reminder helper methods on the plugin."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture) -> MagicMock:
        """Create a plugin instance with reminder support."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        return plugin

    def test_plugin_initializes_reminders_dict(self, plugin: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN _reminders dict exists."""
        assert hasattr(plugin, "_reminders")
        assert isinstance(plugin._reminders, dict)
        assert len(plugin._reminders) == 0

    # Tests for _get_user_reminders

    def test_get_user_reminders_empty(self, plugin: MagicMock) -> None:
        """GIVEN no reminders WHEN queried THEN returns empty list."""
        result = plugin._get_user_reminders("testnick")
        assert result == []

    def test_get_user_reminders_filters_by_nick(self, plugin: MagicMock) -> None:
        """GIVEN reminders from multiple users WHEN queried THEN filters."""
        plugin._reminders["llm_remind_1_100"] = ("testnick", "#channel", "my msg")
        plugin._reminders["llm_remind_2_200"] = ("othernick", "#channel", "their msg")

        result = plugin._get_user_reminders("testnick")
        assert len(result) == 1
        assert result[0][1][2] == "my msg"

    # Tests for _format_reminders

    def test_format_reminders_single(self, plugin: MagicMock) -> None:
        """GIVEN single reminder WHEN formatted THEN shows ID and message."""
        reminders = [("llm_remind_123_456", ("nick", "#chan", "test message"))]
        result = plugin._format_reminders(reminders)
        assert "#456" in result
        assert "test message" in result

    def test_format_reminders_multiple(self, plugin: MagicMock) -> None:
        """GIVEN multiple reminders WHEN formatted THEN pipe-separated."""
        reminders = [
            ("llm_remind_1_100", ("nick", "#chan", "first")),
            ("llm_remind_2_200", ("nick", "#chan", "second")),
        ]
        result = plugin._format_reminders(reminders)
        assert "#100" in result
        assert "#200" in result
        assert " | " in result

    def test_format_reminders_truncates_long_message(self, plugin: MagicMock) -> None:
        """GIVEN long message WHEN formatted THEN truncates."""
        long_msg = "x" * 100
        reminders = [("llm_remind_1_100", ("nick", "#chan", long_msg))]
        result = plugin._format_reminders(reminders)
        assert "..." in result
        assert len(result) < len(long_msg)

    # Tests for _find_user_reminder

    def test_find_user_reminder_found(self, plugin: MagicMock) -> None:
        """GIVEN matching reminder WHEN searched THEN returns event name."""
        plugin._reminders["llm_remind_123_456"] = ("testnick", "#channel", "msg")
        result = plugin._find_user_reminder("testnick", "456")
        assert result == "llm_remind_123_456"

    def test_find_user_reminder_not_found(self, plugin: MagicMock) -> None:
        """GIVEN no matching reminder WHEN searched THEN returns None."""
        result = plugin._find_user_reminder("testnick", "999")
        assert result is None

    def test_find_user_reminder_wrong_owner(self, plugin: MagicMock) -> None:
        """GIVEN reminder owned by other WHEN searched THEN returns None."""
        plugin._reminders["llm_remind_123_456"] = ("othernick", "#channel", "msg")
        result = plugin._find_user_reminder("testnick", "456")
        assert result is None

    # Test database attribute

    def test_plugin_has_db_attribute(self, plugin: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN db attribute exists."""
        assert hasattr(plugin, "db")

    # Test plugin cleanup

    def test_plugin_die_cleans_up_reminders(
        self,
        plugin: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """GIVEN plugin with reminders WHEN die called THEN removes all."""
        from llm.plugin import LLM

        mock_remove_event = mocker.patch("llm.plugin.schedule.removeEvent")

        # Add some reminders
        plugin._reminders["llm_remind_1_100"] = ("user1", "#channel", "msg1")
        plugin._reminders["llm_remind_2_200"] = ("user2", "#channel", "msg2")

        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        mocker.patch("llm.plugin.httpserver.unhook")
        plugin.die()

        # Should have removed both reminder events
        assert mock_remove_event.call_count >= 2
        assert len(plugin._reminders) == 0


class TestParseReminderService:
    """Tests for parse_reminder method in LLMService."""

    @pytest.fixture
    def mock_plugin(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock plugin for service tests."""
        plugin = mocker.MagicMock()
        plugin.registryValue.side_effect = lambda key, *args: {
            "askApiKey": "test-api-key",
            "askModel": "gemini/gemini-2.0-flash",
            "timeout": 30,
        }.get(key, "")
        return plugin

    @pytest.fixture
    def service(self, mock_plugin: MagicMock, mocker: MockerFixture) -> MagicMock:
        """Create an LLMService instance."""
        from llm.service import LLMService

        mocker.patch("llm.service.log")
        return LLMService(mock_plugin)

    def test_parse_reminder_empty_text(self, service: MagicMock) -> None:
        """GIVEN empty text WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("")
        assert result.action == "clarify"
        assert (
            "what to remind" in result.confirmation.lower()
            or "tell me" in result.confirmation.lower()
        )

    def test_parse_reminder_whitespace_only(self, service: MagicMock) -> None:
        """GIVEN whitespace-only text WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("   \n\t  ")
        assert result.action == "clarify"

    def test_parse_reminder_too_long(self, service: MagicMock) -> None:
        """GIVEN text over 500 chars WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("x" * 501)
        assert result.action == "clarify"
        assert "too long" in result.confirmation.lower()

    def test_parse_reminder_exactly_500_chars_accepted(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN text at exactly 500 chars WHEN parsing THEN proceeds to API call."""
        text = "x" * 500
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": 60, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response
        result = service.parse_reminder(text)
        assert result.action == "schedule"

    def test_parse_reminder_no_api_key(self, mock_plugin: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN no API key WHEN parsing THEN returns clarify with error."""
        from llm.service import LLMService

        mock_plugin.registryValue.side_effect = lambda key, *args: ""
        mocker.patch("llm.service.log")
        service = LLMService(mock_plugin)

        result = service.parse_reminder("in 30 minutes test")
        assert result.action == "clarify"
        assert "API key" in result.confirmation

    def test_parse_reminder_schedule_success(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN valid LLM response WHEN parsing THEN returns schedule result."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = (
            '{"action": "schedule", "seconds": 1800, '
            '"message": "check build", "confirmation": "Reminder set for 30m."}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 30 minutes check build")

        assert result.action == "schedule"
        assert result.seconds == 1800
        assert result.message == "check build"
        assert "30m" in result.confirmation

    def test_parse_reminder_clarify_response(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN clarify LLM response WHEN parsing THEN returns clarify result."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "clarify", "confirmation": "When should I remind you?"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("remind me")

        assert result.action == "clarify"
        assert "When" in result.confirmation

    def test_parse_reminder_invalid_json(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN invalid JSON WHEN parsing THEN returns clarify with error."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't understand" in result.confirmation.lower()

    def test_parse_reminder_strips_markdown_fences(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN JSON with markdown fences WHEN parsing THEN strips them."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = (
            '```json\n{"action": "schedule", "seconds": 60, '
            '"message": "test", "confirmation": "Set!"}\n```'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 1 minute test")

        assert result.action == "schedule"
        assert result.seconds == 60

    def test_parse_reminder_with_note(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN response with note WHEN parsing THEN includes note."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = (
            '{"action": "schedule", "seconds": 3600, "message": "meeting", '
            '"confirmation": "Reminder set for 3pm.", "note": "Assuming EST"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("at 3pm meeting")

        assert result.action == "schedule"
        assert result.note == "Assuming EST"

    def test_parse_reminder_negative_seconds_rejected(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN negative seconds WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": -100, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("yesterday test")

        assert result.action == "clarify"

    def test_parse_reminder_api_error(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN API error WHEN parsing THEN returns clarify with error."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_completion.side_effect = Exception("API error")

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't parse" in result.confirmation.lower()

    def test_parse_reminder_zero_seconds_rejected(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN zero seconds WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": 0, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("now test")

        assert result.action == "clarify"

    def test_parse_reminder_missing_seconds(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN missing seconds field WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "schedule", "message": "test", "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("sometime test")

        assert result.action == "clarify"

    def test_parse_reminder_uses_text_as_fallback_message(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN no message in response WHEN parsing THEN uses input text."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "schedule", "seconds": 60, "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 1 minute original text")

        assert result.action == "schedule"
        assert result.message == "in 1 minute original text"

    def test_parse_reminder_with_non_gemini_model(self, mocker: MockerFixture) -> None:
        """GIVEN non-Gemini model WHEN parsing THEN works without Gemini tools."""
        from llm.service import LLMService

        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.side_effect = lambda key, *args: {
            "askApiKey": "test-api-key",
            "askModel": "openai/gpt-4",  # Non-Gemini model
            "timeout": 30,
        }.get(key, "")

        mocker.patch("llm.service.log")
        service = LLMService(mock_plugin)

        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": 300, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 5 minutes test")

        assert result.action == "schedule"
        assert result.seconds == 300
        # Verify no Gemini-specific kwargs were passed
        call_kwargs = mock_completion.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "safety_settings" not in call_kwargs

    def test_parse_reminder_strips_fences_without_trailing_backticks(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN markdown fence without closing WHEN parsing THEN handles gracefully."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        # Fence without proper closing
        mock_response.choices[
            0
        ].message.content = '```json\n{"action": "schedule", "seconds": 120, "message": "test", "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 2 minutes test")

        assert result.action == "schedule"
        assert result.seconds == 120


class TestReminderEventNaming:
    """Tests for uuid-based reminder event naming (Fix 3)."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture) -> MagicMock:
        """Create a plugin instance."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        return plugin

    def test_event_name_uses_uuid_format(self, plugin: MagicMock) -> None:
        """GIVEN reminder delivery closure WHEN created THEN event name uses uuid hex."""
        import re

        closure = plugin._make_reminder_delivery_closure(
            "nick", "#chan", "msg", "llm_remind_abc123"
        )
        assert callable(closure)
        # The event name format should be llm_remind_ + 12 hex chars
        assert re.match(r"^[a-f0-9]{12}$", "abc123def456")

    def test_no_reminder_counter_attribute(self, plugin: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN no _reminder_counter attribute."""
        assert not hasattr(plugin, "_reminder_counter")


class TestReminderDeliveryClosure:
    """Tests for _make_reminder_delivery_closure (Fix 6)."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture) -> MagicMock:
        """Create a plugin instance."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        return plugin

    def test_delivery_cleans_up_on_success(self, plugin: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN delivery closure WHEN queueMsg succeeds THEN cleans up reminder."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]

        event_name = "llm_remind_test123"
        plugin._reminders[event_name] = ("nick", "#chan", "test msg")

        deliver = plugin._make_reminder_delivery_closure("nick", "#chan", "test msg", event_name)
        deliver()

        assert event_name not in plugin._reminders
        plugin.db.delete_reminder.assert_called_with(event_name)
        mock_irc.queueMsg.assert_called_once()

    def test_delivery_via_pm_sends_to_user_nick(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN reminder set via PM WHEN delivered THEN sends to user nick, not bot nick."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        event_name = "llm_remind_pm_test"
        # channel="vibebot" simulates PM: _get_channel returns the bot's own nick
        plugin._reminders[event_name] = ("rdrake", "vibebot", "eat a sandwich")

        deliver = plugin._make_reminder_delivery_closure(
            "rdrake", "vibebot", "eat a sandwich", event_name
        )
        deliver()

        # Should deliver to user's nick, not the bot's nick
        sent_msg = mock_irc.queueMsg.call_args[0][0]
        assert sent_msg.args[0] == "rdrake"
        assert "eat a sandwich" in sent_msg.args[1]

    def test_delivery_in_channel_sends_to_channel(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN reminder set in channel WHEN delivered THEN sends to channel."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        event_name = "llm_remind_chan_test"
        plugin._reminders[event_name] = ("rdrake", "#test", "check build")

        deliver = plugin._make_reminder_delivery_closure(
            "rdrake", "#test", "check build", event_name
        )
        deliver()

        sent_msg = mock_irc.queueMsg.call_args[0][0]
        assert sent_msg.args[0] == "#test"

    def test_delivery_cleans_up_even_on_error(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN delivery closure WHEN queueMsg raises THEN still cleans up reminder."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_irc.queueMsg.side_effect = RuntimeError("send failed")
        mock_world.ircs = [mock_irc]

        event_name = "llm_remind_test456"
        plugin._reminders[event_name] = ("nick", "#chan", "test msg")

        deliver = plugin._make_reminder_delivery_closure("nick", "#chan", "test msg", event_name)

        with pytest.raises(RuntimeError, match="send failed"):
            deliver()

        # Cleanup should still happen despite the error
        assert event_name not in plugin._reminders
        plugin.db.delete_reminder.assert_called_with(event_name)
