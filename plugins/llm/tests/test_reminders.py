"""Tests for reminder duration parsing and reminder commands."""

from unittest.mock import MagicMock, patch

import pytest
from llm.service import format_duration, parse_duration


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
    """Tests for remind, reminders, and unremind commands."""

    def test_remind_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for remind THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "remind")
        assert callable(LLM.remind)

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

    def test_remind_docstring_shows_duration_examples(self) -> None:
        """GIVEN remind command WHEN checking docstring THEN shows duration format."""
        from llm.plugin import LLM

        doc = LLM.remind.__doc__ or ""
        assert "<duration>" in doc
        assert "30s" in doc or "5m" in doc or "2h" in doc


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

    # Tests for _validate_remind_duration

    def test_validate_duration_valid_minutes(self, plugin: MagicMock) -> None:
        """GIVEN valid minutes WHEN validated THEN returns seconds."""
        seconds, error = plugin._validate_remind_duration("30m")
        assert seconds == 1800
        assert error == ""

    def test_validate_duration_valid_hours(self, plugin: MagicMock) -> None:
        """GIVEN valid hours WHEN validated THEN returns seconds."""
        seconds, error = plugin._validate_remind_duration("2h")
        assert seconds == 7200
        assert error == ""

    def test_validate_duration_invalid_format(self, plugin: MagicMock) -> None:
        """GIVEN invalid format WHEN validated THEN returns error."""
        seconds, error = plugin._validate_remind_duration("invalid")
        assert seconds is None
        assert "Invalid" in error

    def test_validate_duration_too_short(self, plugin: MagicMock) -> None:
        """GIVEN duration under 10s WHEN validated THEN returns error."""
        seconds, error = plugin._validate_remind_duration("5s")
        assert seconds is None
        assert "10 seconds" in error

    def test_validate_duration_too_long(self, plugin: MagicMock) -> None:
        """GIVEN duration over 7 days WHEN validated THEN returns error."""
        seconds, error = plugin._validate_remind_duration("8d")
        assert seconds is None
        assert "7 days" in error

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
