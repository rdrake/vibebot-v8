"""Tests for reminder duration parsing and reminder commands."""

from unittest.mock import MagicMock, patch

import pytest
from llm.service import ReminderParseResult, format_duration, parse_duration


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_seconds(self) -> None:
        """GIVEN seconds format WHEN parsed THEN returns correct seconds."""
        assert parse_duration("30s") == 30
        assert parse_duration("1s") == 1

    def test_minutes(self) -> None:
        """GIVEN minutes format WHEN parsed THEN returns correct seconds."""
        assert parse_duration("5m") == 300
        assert parse_duration("30m") == 1800

    def test_hours(self) -> None:
        """GIVEN hours format WHEN parsed THEN returns correct seconds."""
        assert parse_duration("2h") == 7200
        assert parse_duration("24h") == 86400

    def test_days(self) -> None:
        """GIVEN days format WHEN parsed THEN returns correct seconds."""
        assert parse_duration("1d") == 86400
        assert parse_duration("7d") == 604800

    def test_case_insensitive(self) -> None:
        """GIVEN uppercase unit WHEN parsed THEN works correctly."""
        assert parse_duration("5M") == 300
        assert parse_duration("2H") == 7200
        assert parse_duration("1D") == 86400

    def test_invalid_formats(self) -> None:
        """GIVEN invalid formats WHEN parsed THEN returns None."""
        assert parse_duration("30") is None
        assert parse_duration("abc") is None
        assert parse_duration("") is None
        assert parse_duration("5x") is None
        assert parse_duration("m5") is None
        assert parse_duration("-5m") is None

    def test_whitespace_handling(self) -> None:
        """GIVEN whitespace around duration WHEN parsed THEN handles correctly."""
        assert parse_duration(" 30m ") == 1800
        assert parse_duration("  5h  ") == 18000


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_simple_units(self) -> None:
        """GIVEN simple duration WHEN formatted THEN returns single unit."""
        assert format_duration(45) == "45s"
        assert format_duration(300) == "5m"
        assert format_duration(3600) == "1h"
        assert format_duration(86400) == "1d"

    def test_compound_durations(self) -> None:
        """GIVEN compound duration WHEN formatted THEN returns multiple units."""
        assert format_duration(3660) == "1h 1m"
        assert format_duration(90000) == "1d 1h"
        assert format_duration(3661) == "1h 1m 1s"

    def test_zero(self) -> None:
        """GIVEN zero WHEN formatted THEN returns 0s."""
        assert format_duration(0) == "0s"

    def test_negative(self) -> None:
        """GIVEN negative WHEN formatted THEN returns 0s."""
        assert format_duration(-10) == "0s"


class TestReminderCommands:
    """Tests for remindme, reminders, and unremind commands."""

    def test_remindme_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for remindme THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "remindme")
        assert callable(LLM.remindme)

    def test_reminders_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for reminders THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "reminders")
        assert callable(LLM.reminders)

    def test_unremind_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for unremind THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "unremind")
        assert callable(LLM.unremind)

    def test_remindme_docstring_shows_natural_language_examples(self) -> None:
        """GIVEN remindme command WHEN checking docstring THEN shows examples."""
        from llm.plugin import LLM

        doc = LLM.remindme.__doc__ or ""
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
    def mock_irc(self) -> MagicMock:
        """Create a mock IRC object."""
        irc = MagicMock()
        irc.nick = "testbot"
        irc.state = MagicMock()
        irc.state.channels = {}
        irc.state.capabilities_ack = set()
        return irc

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock) -> MagicMock:
        """Create a plugin instance with reminder support."""
        from llm.plugin import LLM

        with (
            patch.object(
                LLM,
                "registryValue",
                side_effect=lambda key, *args: {
                    "httpRoot": "",
                    "contextMaxMessages": 20,
                    "contextTimeoutMinutes": 30,
                    "contextEnabled": True,
                    "channelContextMaxMessages": 10,
                }.get(key, ""),
            ),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
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

    # Test plugin cleanup

    @patch("llm.plugin.schedule.removeEvent")
    def test_plugin_die_cleans_up_reminders(
        self,
        mock_remove_event: MagicMock,
        plugin: MagicMock,
    ) -> None:
        """GIVEN plugin with reminders WHEN die called THEN removes all."""
        from llm.plugin import LLM

        # Add some reminders
        plugin._reminders["llm_remind_1_100"] = ("user1", "#channel", "msg1")
        plugin._reminders["llm_remind_2_200"] = ("user2", "#channel", "msg2")

        with (
            patch.object(LLM.__bases__[0], "die", return_value=None),
            patch("llm.plugin.httpserver.unhook"),
        ):
            plugin.die()

        # Should have removed both reminder events
        assert mock_remove_event.call_count >= 2
        assert len(plugin._reminders) == 0


class TestParseReminderService:
    """Tests for parse_reminder method in LLMService."""

    @pytest.fixture
    def mock_plugin(self) -> MagicMock:
        """Create a mock plugin for service tests."""
        plugin = MagicMock()
        plugin.registryValue.side_effect = lambda key, *args: {
            "askApiKey": "test-api-key",
            "askModel": "gemini/gemini-2.0-flash",
            "timeout": 30,
        }.get(key, "")
        return plugin

    @pytest.fixture
    def service(self, mock_plugin: MagicMock) -> MagicMock:
        """Create an LLMService instance."""
        from llm.service import LLMService

        with patch("llm.service.log"):
            return LLMService(mock_plugin)

    def test_parse_reminder_no_api_key(self, mock_plugin: MagicMock) -> None:
        """GIVEN no API key WHEN parsing THEN returns clarify with error."""
        from llm.service import LLMService

        mock_plugin.registryValue.side_effect = lambda key, *args: ""
        with patch("llm.service.log"):
            service = LLMService(mock_plugin)

        result = service.parse_reminder("in 30 minutes test")
        assert result.action == "clarify"
        assert "API key" in result.confirmation

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_schedule_success(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN valid LLM response WHEN parsing THEN returns schedule result."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
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

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_clarify_response(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN clarify LLM response WHEN parsing THEN returns clarify result."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "clarify", "confirmation": "When should I remind you?"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("remind me")

        assert result.action == "clarify"
        assert "When" in result.confirmation

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_invalid_json(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN invalid JSON WHEN parsing THEN returns clarify with error."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't understand" in result.confirmation.lower()

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_strips_markdown_fences(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN JSON with markdown fences WHEN parsing THEN strips them."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '```json\n{"action": "schedule", "seconds": 60, '
            '"message": "test", "confirmation": "Set!"}\n```'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 1 minute test")

        assert result.action == "schedule"
        assert result.seconds == 60

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_with_note(self, mock_completion: MagicMock, service: MagicMock) -> None:
        """GIVEN response with note WHEN parsing THEN includes note."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '{"action": "schedule", "seconds": 3600, "message": "meeting", '
            '"confirmation": "Reminder set for 3pm.", "note": "Assuming EST"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("at 3pm meeting")

        assert result.action == "schedule"
        assert result.note == "Assuming EST"

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_negative_seconds_rejected(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN negative seconds WHEN parsing THEN returns clarify."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": -100, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("yesterday test")

        assert result.action == "clarify"

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_api_error(self, mock_completion: MagicMock, service: MagicMock) -> None:
        """GIVEN API error WHEN parsing THEN returns clarify with error."""
        mock_completion.side_effect = Exception("API error")

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't parse" in result.confirmation.lower()

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_zero_seconds_rejected(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN zero seconds WHEN parsing THEN returns clarify."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = (
            '{"action": "schedule", "seconds": 0, "message": "test", "confirmation": "Set!"}'
        )
        mock_completion.return_value = mock_response

        result = service.parse_reminder("now test")

        assert result.action == "clarify"

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_missing_seconds(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN missing seconds field WHEN parsing THEN returns clarify."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "schedule", "message": "test", "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("sometime test")

        assert result.action == "clarify"

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_uses_text_as_fallback_message(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN no message in response WHEN parsing THEN uses input text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"action": "schedule", "seconds": 60, "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 1 minute original text")

        assert result.action == "schedule"
        assert result.message == "in 1 minute original text"

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_with_non_gemini_model(self, mock_completion: MagicMock) -> None:
        """GIVEN non-Gemini model WHEN parsing THEN works without Gemini tools."""
        from llm.service import LLMService

        mock_plugin = MagicMock()
        mock_plugin.registryValue.side_effect = lambda key, *args: {
            "askApiKey": "test-api-key",
            "askModel": "openai/gpt-4",  # Non-Gemini model
            "timeout": 30,
        }.get(key, "")

        with patch("llm.service.log"):
            service = LLMService(mock_plugin)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
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

    @patch("llm.service.litellm.completion")
    def test_parse_reminder_strips_fences_without_trailing_backticks(
        self, mock_completion: MagicMock, service: MagicMock
    ) -> None:
        """GIVEN markdown fence without closing WHEN parsing THEN handles gracefully."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # Fence without proper closing
        mock_response.choices[
            0
        ].message.content = '```json\n{"action": "schedule", "seconds": 120, "message": "test", "confirmation": "Set!"}'
        mock_completion.return_value = mock_response

        result = service.parse_reminder("in 2 minutes test")

        assert result.action == "schedule"
        assert result.seconds == 120
