"""Integration tests for LLM plugin.

These tests verify the integration between plugin components and
realistic Limnoria-like scenarios without requiring a full Limnoria runtime.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestPluginContextIntegration:
    """Test plugin context management integration."""

    @pytest.fixture
    def plugin_with_context(self, mock_irc: MagicMock, mocker: MockerFixture) -> tuple:
        """Create plugin with initialized context."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker, mock_database=False)
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
    def plugin_for_tracking(self, mock_irc: MagicMock, mocker: MockerFixture) -> tuple:
        """Create plugin configured for message tracking."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect({"contextTrackAllMessages": True})

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc

    def test_doprivmsg_tracks_channel_messages(
        self, plugin_for_tracking: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN tracking enabled WHEN channel message received THEN tracked."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "Hello world")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100  # Future time (not playback)

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("user1", "#channel")
        assert len(messages) >= 1

    def test_doprivmsg_tracks_action_messages(
        self, plugin_for_tracking: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ACTION message WHEN received THEN tracked normally."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "\x01ACTION does something\x01")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=True)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("user1", "#channel")
        assert len(messages) >= 1


class TestHTTPCallbackIntegration:
    """Test HTTP callback integration with plugin."""

    @pytest.fixture
    def http_callback_with_plugin(self, mocker: MockerFixture) -> tuple:
        """Create HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = ""
        callback = LLMHTTPCallback(mock_plugin)
        return callback, mock_plugin

    def test_http_callback_serves_multiple_content_types(
        self, http_callback_with_plugin: tuple, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN various file types WHEN served THEN correct content types."""

        callback, _ = http_callback_with_plugin
        mocker.patch.object(callback, "_get_web_dir", return_value=str(tmp_path))

        # Create test files
        files = {
            "test.html": (b"<html></html>", "text/html"),
            "test.png": (b"\x89PNG\r\n\x1a\n", "image/png"),
            "test.jpg": (b"\xff\xd8\xff\xe0", "image/jpeg"),
            "test.gif": (b"GIF89a", "image/gif"),
            "test.css": (b"body {}", "text/css"),
            "test.js": (b"function() {}", "text/javascript"),
        }

        for filename, (content, _expected_type) in files.items():
            filepath = tmp_path / filename
            filepath.write_bytes(content)

            mock_handler = mocker.MagicMock()
            mock_handler.wfile = mocker.MagicMock()

            callback.doGet(mock_handler, filename)

            mock_handler.send_response.assert_called_with(200)

    def test_http_callback_handles_concurrent_requests(
        self, http_callback_with_plugin: tuple, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN multiple concurrent requests WHEN served THEN no race conditions."""
        callback, _ = http_callback_with_plugin
        mocker.patch.object(callback, "_get_web_dir", return_value=str(tmp_path))

        # Create test file
        test_file = tmp_path / "concurrent.txt"
        test_file.write_bytes(b"test content")

        errors = []
        lock = threading.Lock()

        def make_request(request_id: int) -> None:
            try:
                mock_handler = mocker.MagicMock()
                mock_handler.wfile = mocker.MagicMock()

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
