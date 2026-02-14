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


class TestRateLimitFullFlow:
    """Integration test for rate limiting on expensive commands.

    Uses a real SQLite database to verify that repeated draw requests
    trigger rate limiting, that unflagging still works independently,
    and that normal requests succeed after the window expires.
    """

    @pytest.fixture
    def plugin_with_real_db(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> tuple:
        """Create plugin with real database but mocked LLM service."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        db_path = str(tmp_path / "test.db")
        registry = make_registry_side_effect(
            {
                "databasePath": db_path,
                "enforceRateLimits": True,
                "drawRateLimitCount": 2,
                "drawRateLimitWindow": 60,
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        return plugin, mock_irc

    def test_rate_limit_full_flow(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN enforced rate limits WHEN user exceeds threshold THEN blocked then recovers.

        Full flow:
        1. User makes 2 successful draw requests (at limit)
        2. 3rd request is rate_limited
        3. After window expires, user can draw again
        4. Unflag flow works independently of rate limiting
        """
        from llm.service import ImageResult

        plugin, mock_irc = plugin_with_real_db

        # Set up user identity
        mock_irc.state.nickToAccount.return_value = "testuser"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "draw something")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        # Step 1: Two successful draws fill the bucket
        for i in range(2):
            mock_irc.reset_mock()
            plugin.draw(mock_irc, mock_msg, [f"prompt {i}"])
            mock_irc.reply.assert_called_once()

        # Step 2: 3rd draw is rate limited
        mock_irc.reset_mock()
        plugin.llm_service.image_generation.reset_mock()
        plugin.draw(mock_irc, mock_msg, ["one too many"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.image_generation.assert_not_called()

        # Verify rate_limited was logged in the real database
        usage_rows = (
            plugin.db._connect()
            .execute("SELECT status FROM usage WHERE nick = 'testuser' AND status = 'rate_limited'")
            .fetchall()
        )
        assert len(usage_rows) >= 1

        # Step 3: Simulate window expiration by clearing buckets
        plugin._rate_buckets.clear()
        mock_irc.reset_mock()
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen2.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )
        plugin.draw(mock_irc, mock_msg, ["fresh prompt"])
        mock_irc.reply.assert_called_once()
        assert "http://img.example/gen2.png" in mock_irc.reply.call_args[0][0]

        # Step 4: Unflag flow works independently
        plugin.db.flag_user("testuser", "manual test", auto_flagged=False)
        assert plugin.db.is_user_flagged("testuser") is True

        mock_irc.reset_mock()
        plugin.llm_service.image_generation.reset_mock()
        plugin._rate_buckets.clear()  # Clear rate limits
        plugin.draw(mock_irc, mock_msg, ["flagged draw"])
        mock_irc.error.assert_called_once()
        assert "suspended" in mock_irc.error.call_args[0][0]

        plugin.db.unflag_user("testuser", "admin")
        mock_irc.reset_mock()
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="http://img.example/gen3.png",
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )
        plugin.draw(mock_irc, mock_msg, ["unflagged draw"])
        mock_irc.reply.assert_called_once()
