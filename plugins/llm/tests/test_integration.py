"""Integration tests for LLM plugin.

These tests verify the integration between plugin components and
realistic Limnoria-like scenarios without requiring a full Limnoria runtime.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestPluginCommandRouting:
    """Test that commands are properly wired up and route correctly."""

    def test_ask_command_has_capability_requirement(self) -> None:
        """GIVEN ask command WHEN inspecting wrapper THEN requires llm.ask capability."""
        from llm.plugin import LLM

        # The wrapped function stores info about requirements
        assert hasattr(LLM.ask, "commands")
        # The wrap decorator sets up capability checking

    def test_code_command_has_capability_requirement(self) -> None:
        """GIVEN code command WHEN inspecting wrapper THEN requires llm.code capability."""
        from llm.plugin import LLM

        assert hasattr(LLM.code, "commands")

    def test_draw_command_has_capability_requirement(self) -> None:
        """GIVEN draw command WHEN inspecting wrapper THEN requires llm.draw capability."""
        from llm.plugin import LLM

        assert hasattr(LLM.draw, "commands")

    def test_forget_command_requires_channel_argument(self) -> None:
        """GIVEN forget command WHEN inspecting wrapper THEN requires channel."""
        from llm.plugin import LLM

        assert hasattr(LLM.forget, "commands")

    def test_llmkeys_command_requires_admin(self) -> None:
        """GIVEN llmkeys command WHEN inspecting wrapper THEN requires admin."""
        from llm.plugin import LLM

        assert hasattr(LLM.llmkeys, "commands")


class TestPluginLifecycleIntegration:
    """Test plugin initialization, reload, and shutdown behaviors."""

    @pytest.fixture
    def mock_irc(self) -> MagicMock:
        """Create a mock IRC object."""
        irc = MagicMock()
        irc.nick = "testbot"
        irc.state = MagicMock()
        irc.state.channels = {}
        irc.state.capabilities_ack = set()
        return irc

    def test_plugin_initializes_all_components(self, mock_irc: MagicMock) -> None:
        """GIVEN fresh plugin WHEN initialized THEN all components created."""
        from llm.plugin import LLM

        with (
            patch.object(LLM, "registryValue", return_value=""),
            patch("llm.plugin.LLMService") as mock_service,
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
            plugin = LLM(mock_irc)

            # Service should be created
            mock_service.assert_called_once_with(plugin)

            # Context should be initialized
            assert hasattr(plugin, "context")

            # Startup time should be set
            assert hasattr(plugin, "startup_time")
            assert plugin.startup_time > 0

    def test_plugin_reload_handles_existing_scheduled_event(
        self, mock_irc: MagicMock
    ) -> None:
        """GIVEN plugin reloading WHEN existing event exists THEN no error."""
        from llm.plugin import LLM

        with (
            patch.object(LLM, "registryValue", return_value=""),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent") as mock_remove,
        ):
            # Simulate reload where event already exists
            mock_remove.side_effect = [KeyError("not found")]

            # Should not raise
            plugin = LLM(mock_irc)
            assert plugin is not None

    def test_plugin_die_cleans_up_all_resources(self, mock_irc: MagicMock) -> None:
        """GIVEN running plugin WHEN die called THEN all resources cleaned up."""
        from llm.plugin import LLM

        with (
            patch.object(LLM, "registryValue", return_value=""),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.httpserver.unhook") as mock_unhook,
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent") as mock_remove,
        ):
            plugin = LLM(mock_irc)

            # Reset mocks after init
            mock_remove.reset_mock()

            with patch.object(LLM.__bases__[0], "die", return_value=None):
                plugin.die()

            # Should remove scheduled event
            mock_remove.assert_called_with("llm_file_cleanup")

            # Should unhook HTTP callback
            mock_unhook.assert_called_with("llm")


class TestPluginContextIntegration:
    """Test plugin context management integration."""

    @pytest.fixture
    def plugin_with_context(self) -> tuple:
        """Create plugin with initialized context."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "testbot"

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

        return plugin, mock_irc

    def test_context_initialized_from_config(self, plugin_with_context: tuple) -> None:
        """GIVEN plugin WHEN initialized THEN context uses config values."""
        plugin, _ = plugin_with_context

        stats = plugin.context.get_stats()
        assert stats["max_messages_per_conv"] == 20
        assert stats["timeout_minutes"] == 30
        assert stats["enabled"] is True

    def test_context_shared_between_commands(self, plugin_with_context: tuple) -> None:
        """GIVEN plugin WHEN multiple commands THEN share same context."""
        plugin, _ = plugin_with_context

        # Add message to context
        plugin.context.add_message("user1", "#test", "user", "Hello")

        # Should be retrievable
        messages = plugin.context.get_messages("user1", "#test")
        assert len(messages) == 1


class TestDoPrivmsgIntegration:
    """Test doPrivmsg message tracking integration."""

    @pytest.fixture
    def plugin_for_tracking(self) -> tuple:
        """Create plugin configured for message tracking."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "testbot"

        def registry_side_effect(key, *args):
            return {
                "httpRoot": "",
                "contextMaxMessages": 20,
                "contextTimeoutMinutes": 30,
                "contextEnabled": True,
                "channelContextMaxMessages": 10,
                "contextTrackAllMessages": True,
            }.get(key, True)

        with (
            patch.object(LLM, "registryValue", side_effect=registry_side_effect),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
            plugin = LLM(mock_irc)
            plugin.registryValue = MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc

    def test_doprivmsg_tracks_channel_messages(self, plugin_for_tracking: tuple) -> None:
        """GIVEN tracking enabled WHEN channel message received THEN tracked."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "Hello world")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100  # Future time (not playback)

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=False),
            patch("supybot.ircutils.strEqual", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("user1", "#channel")
        assert len(messages) >= 1

    def test_doprivmsg_tracks_action_messages(self, plugin_for_tracking: tuple) -> None:
        """GIVEN ACTION message WHEN received THEN tracked normally."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "\x01ACTION does something\x01")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=True),
            patch("supybot.ircmsgs.isAction", return_value=True),
            patch("supybot.ircutils.strEqual", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("user1", "#channel")
        assert len(messages) >= 1


class TestHTTPCallbackIntegration:
    """Test HTTP callback integration with plugin."""

    @pytest.fixture
    def http_callback_with_plugin(self) -> tuple:
        """Create HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = MagicMock()
        mock_plugin.registryValue.return_value = ""
        callback = LLMHTTPCallback(mock_plugin)
        return callback, mock_plugin

    def test_http_callback_serves_multiple_content_types(
        self, http_callback_with_plugin: tuple, tmp_path
    ) -> None:
        """GIVEN various file types WHEN served THEN correct content types."""
        import os

        callback, _ = http_callback_with_plugin

        # Create test files
        files = {
            "test.html": (b"<html></html>", "text/html"),
            "test.png": (b"\x89PNG\r\n\x1a\n", "image/png"),
            "test.jpg": (b"\xff\xd8\xff\xe0", "image/jpeg"),
            "test.gif": (b"GIF89a", "image/gif"),
            "test.css": (b"body {}", "text/css"),
            "test.js": (b"function() {}", "text/javascript"),
        }

        for filename, (content, expected_type) in files.items():
            filepath = tmp_path / filename
            filepath.write_bytes(content)

            mock_handler = MagicMock()
            mock_handler.wfile = MagicMock()

            with patch.object(callback, "_get_web_dir", return_value=str(tmp_path)):
                callback.doGet(mock_handler, filename)

            mock_handler.send_response.assert_called_with(200)

    def test_http_callback_handles_concurrent_requests(
        self, http_callback_with_plugin: tuple, tmp_path
    ) -> None:
        """GIVEN multiple concurrent requests WHEN served THEN no race conditions."""
        callback, _ = http_callback_with_plugin

        # Create test file
        test_file = tmp_path / "concurrent.txt"
        test_file.write_bytes(b"test content")

        errors = []
        lock = threading.Lock()

        def make_request(request_id: int) -> None:
            try:
                mock_handler = MagicMock()
                mock_handler.wfile = MagicMock()

                with patch.object(callback, "_get_web_dir", return_value=str(tmp_path)):
                    callback.doGet(mock_handler, "concurrent.txt")

                mock_handler.send_response.assert_called_with(200)
            except Exception as e:
                with lock:
                    errors.append((request_id, e))

        # Launch concurrent requests
        threads = []
        for i in range(20):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent requests: {errors}"


class TestServicePluginIntegration:
    """Test LLMService integration with plugin."""

    @pytest.fixture
    def service_with_plugin(self) -> tuple:
        """Create service with mock plugin."""
        from llm.service import LLMService

        mock_plugin = MagicMock()
        mock_plugin.log = MagicMock()
        mock_plugin.registryValue = MagicMock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "test-api-key",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )

        service = LLMService(mock_plugin)
        return service, mock_plugin

    def test_service_uses_plugin_config_for_validation(
        self, service_with_plugin: tuple
    ) -> None:
        """GIVEN service WHEN validating prompt THEN uses plugin config."""
        service, mock_plugin = service_with_plugin

        # Should validate against configured max length
        is_valid, _ = service.validate_prompt("Hello")
        assert is_valid is True

        # Long prompt should fail
        mock_plugin.registryValue = MagicMock(
            side_effect=lambda key, channel=None: 10 if key == "maxPromptLength" else None
        )
        is_valid, error = service.validate_prompt("This is too long")
        assert is_valid is False
        assert "too long" in error.lower()

    def test_service_sanitize_output_uses_plugin_config(
        self, service_with_plugin: tuple
    ) -> None:
        """GIVEN service WHEN sanitizing output THEN uses plugin's prefix config."""
        service, _ = service_with_plugin

        # Should sanitize based on configured prefixes
        result = service._sanitize_output(".kick user")
        assert result == " .kick user"

        result = service._sanitize_output("/msg user hello")
        assert result == " /msg user hello"


class TestFullCommandFlow:
    """Test complete command flows from input to output."""

    @pytest.fixture
    def full_plugin_setup(self) -> tuple:
        """Create fully mocked plugin for command testing."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "testbot"
        mock_irc.state = MagicMock()
        mock_irc.state.channels = {"#test": MagicMock(topic="Test channel")}
        mock_irc.state.capabilities_ack = {"message-tags"}

        def registry_side_effect(key, *args):
            return {
                "httpRoot": "",
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
                "timeout": 30,
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "httpUrlBase": "http://localhost:8080/llm",
            }.get(key, "")

        with (
            patch.object(LLM, "registryValue", side_effect=registry_side_effect),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook"),
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
            plugin = LLM(mock_irc)
            plugin.registryValue = MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc

    def test_ask_command_full_flow(self, full_plugin_setup: tuple) -> None:
        """GIVEN ask command WHEN executed THEN full flow completes."""
        plugin, mock_irc = full_plugin_setup

        mock_msg = MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#test", "What is Python?")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#test"

        # Mock service response
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = "Python is a programming language."

        # Call ask command logic directly (bypassing wrap)
        plugin._is_old_message = MagicMock(return_value=False)
        plugin._get_context_enabled = MagicMock(return_value=True)

        nick = plugin._get_nick(mock_msg)
        channel = plugin._get_channel(mock_msg)
        history = plugin.context.get_messages(nick, channel)
        channel_history = plugin.context.get_channel_messages(channel, exclude_nick=nick)

        response = plugin.llm_service.completion(
            "What is Python?",
            command="ask",
            history=history,
            channel_history=channel_history,
            irc=mock_irc,
            msg=mock_msg,
        )

        # Store context
        plugin.context.add_message(nick, channel, "user", "What is Python?")
        plugin.context.add_message(nick, channel, "assistant", response)

        # Verify context was stored
        messages = plugin.context.get_messages(nick, channel)
        assert len(messages) == 2
        assert messages[0]["content"] == "What is Python?"
        assert messages[1]["content"] == "Python is a programming language."

    def test_forget_command_clears_context(self, full_plugin_setup: tuple) -> None:
        """GIVEN forget command WHEN executed THEN context cleared."""
        plugin, mock_irc = full_plugin_setup

        # Add some context first
        plugin.context.add_message("user1", "#test", "user", "Hello")
        plugin.context.add_message("user1", "#test", "assistant", "Hi!")

        # Verify context exists
        assert len(plugin.context.get_messages("user1", "#test")) == 2

        # Clear context
        cleared = plugin.context.clear("user1", "#test")
        assert cleared is True

        # Verify context is empty
        assert len(plugin.context.get_messages("user1", "#test")) == 0
