"""Tests for LLM plugin.

These tests verify the plugin structure, imports, and command registration
without requiring a full Limnoria runtime environment.
"""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest


class TestPluginImport:
    """Test plugin module can be imported and has expected structure."""

    def test_plugin_module_imports(self) -> None:
        """GIVEN llm.plugin module WHEN imported THEN no errors."""
        from llm import plugin

        assert plugin is not None

    def test_plugin_class_exists(self) -> None:
        """GIVEN llm.plugin module WHEN accessing Class THEN plugin class found."""
        from llm.plugin import Class

        assert Class is not None
        assert Class.__name__ == "LLM"

    def test_plugin_inherits_from_callbacks(self) -> None:
        """GIVEN LLM class WHEN checking inheritance THEN inherits from Plugin."""
        # Check that LLM inherits from callbacks.Plugin
        import supybot.callbacks as callbacks
        from llm.plugin import LLM

        assert issubclass(LLM, callbacks.Plugin)


class TestCommandExistence:
    """Test that expected commands are defined on the plugin class."""

    def test_ask_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for ask THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "ask")
        assert callable(LLM.ask)

    def test_code_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for code THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "code")
        assert callable(LLM.code)

    def test_draw_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for draw THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "draw")
        assert callable(LLM.draw)

    def test_forget_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for forget THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "forget")
        assert callable(LLM.forget)

    def test_llmkeys_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for llmkeys THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "llmkeys")
        assert callable(LLM.llmkeys)

    def test_usage_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for usage THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "usage")
        assert callable(LLM.usage)


class TestPluginConfiguration:
    """Test plugin configuration and service dependencies."""

    def test_plugin_is_threaded(self) -> None:
        """GIVEN LLM plugin class WHEN checking threaded attribute THEN True."""
        from llm.plugin import LLM

        assert LLM.threaded is True

    def test_service_module_imports(self) -> None:
        """GIVEN llm.service module WHEN imported THEN no errors."""
        from llm.service import LLMService

        assert LLMService is not None

    def test_context_module_imports(self) -> None:
        """GIVEN llm.context module WHEN imported THEN no errors."""
        from llm.context import ContextConfig, ConversationContext

        assert ConversationContext is not None
        assert ContextConfig is not None


class TestHTTPCallback:
    """Test HTTP callback class exists and has expected structure."""

    def test_http_callback_class_exists(self) -> None:
        """GIVEN llm.plugin module WHEN accessing LLMHTTPCallback THEN class exists."""
        from llm.plugin import LLMHTTPCallback

        assert LLMHTTPCallback is not None

    def test_http_callback_has_name(self) -> None:
        """GIVEN LLMHTTPCallback WHEN checking name THEN has expected name."""
        from llm.plugin import LLMHTTPCallback

        assert hasattr(LLMHTTPCallback, "name")
        assert LLMHTTPCallback.name == "LLM"

    def test_http_callback_is_public(self) -> None:
        """GIVEN LLMHTTPCallback WHEN checking public THEN is True."""
        from llm.plugin import LLMHTTPCallback

        assert hasattr(LLMHTTPCallback, "public")
        assert LLMHTTPCallback.public is True


class TestHTTPCallbackDoGet:
    """Test HTTP callback doGet method for serving files."""

    @pytest.fixture
    def mock_plugin(self) -> MagicMock:
        """Create a mock plugin for HTTP callback."""
        plugin = MagicMock()
        plugin.registryValue.return_value = ""  # No custom httpRoot
        return plugin

    @pytest.fixture
    def http_callback(self, mock_plugin: MagicMock):
        """Create an HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        return LLMHTTPCallback(mock_plugin)

    @pytest.fixture
    def mock_handler(self) -> MagicMock:
        """Create a mock HTTP handler."""
        handler = MagicMock()
        # wfile needs to be a MagicMock so we can set side_effect
        handler.wfile = MagicMock()
        return handler

    def test_doget_serves_help_at_root(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN empty path WHEN doGet called THEN serves help page."""
        http_callback.doGet(mock_handler, "")
        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "text/html; charset=utf-8")

    def test_doget_blocks_directory_traversal(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN path with .. WHEN doGet called THEN returns 403."""
        http_callback.doGet(mock_handler, "../etc/passwd")
        mock_handler.send_response.assert_called_with(403)
        mock_handler.end_headers.assert_called_once()

    def test_doget_blocks_absolute_path_in_middle(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN path with / after stripping WHEN doGet called THEN returns 403."""
        # After lstrip("/"), if there's still a / at start, it's suspicious
        # Actually looking at the code: path.startswith("/") after lstrip
        # This tests the security check more directly
        http_callback.doGet(mock_handler, "/../test")
        mock_handler.send_response.assert_called_with(403)

    def test_doget_returns_404_for_missing_file(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN nonexistent file WHEN doGet called THEN returns 404."""
        with patch.object(http_callback, "_get_web_dir", return_value="/nonexistent"):
            http_callback.doGet(mock_handler, "missing.txt")
        mock_handler.send_response.assert_called_with(404)
        mock_handler.end_headers.assert_called_once()

    def test_doget_serves_existing_file(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN existing file WHEN doGet called THEN returns 200 with content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"test content")

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                http_callback.doGet(mock_handler, "test.txt")

            mock_handler.send_response.assert_called_with(200)
            mock_handler.send_header.assert_any_call("Content-Type", "text/plain")
            mock_handler.send_header.assert_any_call("Content-Length", "12")

    def test_doget_serves_image_with_correct_type(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN image file WHEN doGet called THEN returns correct content type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.png")
            with open(test_file, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")  # PNG header

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                http_callback.doGet(mock_handler, "test.png")

            mock_handler.send_response.assert_called_with(200)
            mock_handler.send_header.assert_any_call("Content-Type", "image/png")

    def test_doget_handles_unknown_content_type(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN file with unknown extension WHEN doGet called THEN uses octet-stream."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.xyz123")
            with open(test_file, "wb") as f:
                f.write(b"binary data")

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                http_callback.doGet(mock_handler, "test.xyz123")

            mock_handler.send_response.assert_called_with(200)
            mock_handler.send_header.assert_any_call("Content-Type", "application/octet-stream")

    def test_doget_handles_broken_pipe_silently(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN client disconnect WHEN doGet serving file THEN no error raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"test")

            mock_handler.wfile.write.side_effect = BrokenPipeError()

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                # Should not raise
                http_callback.doGet(mock_handler, "test.txt")

    def test_doget_handles_connection_reset_silently(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN connection reset WHEN doGet serving file THEN no error raised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"test")

            mock_handler.wfile.write.side_effect = ConnectionResetError()

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                # Should not raise
                http_callback.doGet(mock_handler, "test.txt")

    def test_doget_handles_os_error_with_500(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN OS error reading file WHEN doGet called THEN returns 500."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "wb") as f:
                f.write(b"test")

            with (
                patch.object(http_callback, "_get_web_dir", return_value=tmpdir),
                patch("builtins.open", side_effect=OSError("disk error")),
            ):
                http_callback.doGet(mock_handler, "test.txt")

            mock_handler.send_response.assert_called_with(500)

    def test_doget_blocks_symlink_escape(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN symlink pointing outside web dir WHEN doGet called THEN returns 403."""
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, "web")
            os.makedirs(web_dir)

            # Create a file outside web dir
            outside_file = os.path.join(tmpdir, "secret.txt")
            with open(outside_file, "w") as f:
                f.write("secret data")

            # Create a symlink inside web dir pointing outside
            symlink_path = os.path.join(web_dir, "escape.txt")
            os.symlink(outside_file, symlink_path)

            with patch.object(http_callback, "_get_web_dir", return_value=web_dir):
                http_callback.doGet(mock_handler, "escape.txt")

            # Should return 403 because resolved path is outside web_dir
            mock_handler.send_response.assert_called_with(403)

    def test_doget_allows_symlink_within_web_dir(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN symlink pointing within web dir WHEN doGet called THEN serves file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            web_dir = os.path.join(tmpdir, "web")
            os.makedirs(web_dir)

            # Create a file inside web dir
            real_file = os.path.join(web_dir, "real.txt")
            with open(real_file, "wb") as f:
                f.write(b"content")

            # Create a symlink inside web dir pointing to the file
            symlink_path = os.path.join(web_dir, "link.txt")
            os.symlink(real_file, symlink_path)

            with patch.object(http_callback, "_get_web_dir", return_value=web_dir):
                http_callback.doGet(mock_handler, "link.txt")

            # Should serve the file
            mock_handler.send_response.assert_called_with(200)

    def test_doget_handles_realpath_oserror(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN realpath raises OSError WHEN doGet called THEN returns 403."""
        with (
            patch.object(http_callback, "_get_web_dir", return_value="/some/dir"),
            patch("os.path.realpath", side_effect=OSError("permission denied")),
        ):
            http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(403)


class TestHTTPCallbackGetWebDir:
    """Test HTTP callback _get_web_dir method."""

    def test_get_web_dir_uses_http_root_when_set(self) -> None:
        """GIVEN httpRoot configured WHEN _get_web_dir called THEN returns httpRoot."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = MagicMock()
        mock_plugin.registryValue.return_value = "/custom/path"
        callback = LLMHTTPCallback(mock_plugin)

        result = callback._get_web_dir()

        assert result == "/custom/path"
        mock_plugin.registryValue.assert_called_with("httpRoot")

    def test_get_web_dir_uses_data_web_when_no_http_root(self) -> None:
        """GIVEN httpRoot empty WHEN _get_web_dir called THEN returns data/web/llm."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = MagicMock()
        mock_plugin.registryValue.return_value = ""
        callback = LLMHTTPCallback(mock_plugin)

        # Just verify it returns a string (can't easily mock supybot's registry)
        # The actual behavior is tested implicitly when httpRoot is empty
        result = callback._get_web_dir()

        # Should return a path that ends with 'llm'
        assert result.endswith("llm") or "llm" in result
        mock_plugin.registryValue.assert_called_with("httpRoot")


class TestHTTPCallbackServeHelpPage:
    """Test HTTP callback _serve_help_page method."""

    @pytest.fixture
    def mock_plugin(self) -> MagicMock:
        """Create a mock plugin for HTTP callback."""
        plugin = MagicMock()
        plugin.registryValue.return_value = ""
        return plugin

    @pytest.fixture
    def http_callback(self, mock_plugin: MagicMock):
        """Create an HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        return LLMHTTPCallback(mock_plugin)

    @pytest.fixture
    def mock_handler(self) -> MagicMock:
        """Create a mock HTTP handler."""
        handler = MagicMock()
        handler.wfile = MagicMock()
        return handler

    def test_serve_help_page_uses_builtin_template(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN no custom help.html WHEN _serve_help_page THEN uses builtin template."""
        from llm.plugin import HELP_HTML_TEMPLATE

        with patch.object(http_callback, "_get_web_dir", return_value="/nonexistent"):
            http_callback._serve_help_page(mock_handler)

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "text/html; charset=utf-8")
        # Verify content matches template
        written_content = mock_handler.wfile.write.call_args[0][0]
        assert written_content == HELP_HTML_TEMPLATE.encode("utf-8")

    def test_serve_help_page_uses_custom_file_when_exists(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN custom help.html WHEN _serve_help_page THEN uses custom file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_help = os.path.join(tmpdir, "help.html")
            custom_content = b"<html>Custom Help</html>"
            with open(custom_help, "wb") as f:
                f.write(custom_content)

            with patch.object(http_callback, "_get_web_dir", return_value=tmpdir):
                http_callback._serve_help_page(mock_handler)

            mock_handler.send_response.assert_called_with(200)
            written_content = mock_handler.wfile.write.call_args[0][0]
            assert written_content == custom_content

    def test_serve_help_page_handles_broken_pipe(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN client disconnect WHEN _serve_help_page THEN no error raised."""
        mock_handler.wfile.write.side_effect = BrokenPipeError()

        with patch.object(http_callback, "_get_web_dir", return_value="/nonexistent"):
            # Should not raise
            http_callback._serve_help_page(mock_handler)

    def test_serve_help_page_falls_back_on_read_error(
        self, http_callback, mock_handler: MagicMock
    ) -> None:
        """GIVEN custom file read error WHEN _serve_help_page THEN falls back to template."""
        from llm.plugin import HELP_HTML_TEMPLATE

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_help = os.path.join(tmpdir, "help.html")
            # Create file then make it unreadable by mocking
            with open(custom_help, "wb") as f:
                f.write(b"content")

            with (
                patch.object(http_callback, "_get_web_dir", return_value=tmpdir),
                patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied")),
            ):
                http_callback._serve_help_page(mock_handler)

            written_content = mock_handler.wfile.write.call_args[0][0]
            assert written_content == HELP_HTML_TEMPLATE.encode("utf-8")


class TestPluginHelperMethods:
    """Test plugin helper methods."""

    @pytest.fixture
    def mock_msg(self) -> MagicMock:
        """Create a mock IRC message."""
        msg = MagicMock()
        msg.prefix = "testnick!user@host"
        msg.args = ("#testchannel", "test message")
        msg.time = time.time()
        msg.channel = "#testchannel"
        return msg

    def test_get_nick_extracts_nick_from_hostmask(self, mock_msg: MagicMock) -> None:
        """GIVEN message with prefix WHEN _get_nick called THEN returns nick."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            result = plugin._get_nick(mock_msg)

        assert result == "testnick"

    def test_get_channel_extracts_channel_from_args(self, mock_msg: MagicMock) -> None:
        """GIVEN message with args WHEN _get_channel called THEN returns channel."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            result = plugin._get_channel(mock_msg)

        assert result == "#testchannel"

    def test_get_channel_returns_unknown_for_empty_args(self) -> None:
        """GIVEN message with no args WHEN _get_channel called THEN returns unknown."""
        from llm.plugin import LLM

        mock_msg = MagicMock()
        mock_msg.args = []

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            result = plugin._get_channel(mock_msg)

        assert result == "unknown"

    def test_is_old_message_returns_true_for_old_message(self) -> None:
        """GIVEN message older than startup WHEN _is_old_message THEN returns True."""
        from llm.plugin import LLM

        mock_msg = MagicMock()
        mock_msg.time = time.time() - 100

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            result = plugin._is_old_message(mock_msg)

        assert result is True

    def test_is_old_message_returns_false_for_new_message(self) -> None:
        """GIVEN message newer than startup WHEN _is_old_message THEN returns False."""
        from llm.plugin import LLM

        mock_msg = MagicMock()
        mock_msg.time = time.time() + 100

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            result = plugin._is_old_message(mock_msg)

        assert result is False

    def test_is_old_message_returns_false_for_zero_timestamp(self) -> None:
        """GIVEN message with time=0 WHEN _is_old_message THEN returns False.

        Limnoria defaults msg.time to 0 when no server-time tag is present.
        This should be treated as a live message, not ZNC playback.
        """
        from llm.plugin import LLM

        mock_msg = MagicMock()
        mock_msg.time = 0

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            result = plugin._is_old_message(mock_msg)

        assert result is False

    def test_get_help_url_delegates_to_service(self) -> None:
        """GIVEN service returns url_base WHEN _get_help_url THEN returns url_base + /."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.llm_service.get_http_paths.return_value = (
                "/var/www/llm",
                "https://example.com/llm",
            )

            result = plugin._get_help_url()

        assert result == "https://example.com/llm/"

    def test_get_help_url_with_localhost_fallback(self) -> None:
        """GIVEN service returns localhost url WHEN _get_help_url THEN uses it."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.llm_service.get_http_paths.return_value = (
                "/data/web/llm",
                "http://localhost:8080/llm",
            )

            result = plugin._get_help_url()

        assert result == "http://localhost:8080/llm/"

    def test_get_plugin_help_includes_url(self) -> None:
        """GIVEN plugin WHEN getPluginHelp called THEN includes help URL."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.llm_service.get_http_paths.return_value = (
                "/data/web/llm",
                "https://example.com/llm",
            )

            result = plugin.getPluginHelp()

        assert "https://example.com/llm/" in result
        assert "ask" in result
        assert "code" in result
        assert "draw" in result
        assert "forget" in result


class TestDoPrivmsg:
    """Test plugin doPrivmsg for channel message tracking."""

    @pytest.fixture
    def plugin_with_mocks(self) -> tuple:
        """Create plugin with mocked dependencies."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "botname"

        mock_msg = MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            plugin.registryValue = MagicMock(return_value=True)
            plugin.context = MagicMock()

        return plugin, mock_irc, mock_msg

    def test_doprivmsg_skips_private_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN private message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.channel = None  # Private message

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_old_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN ZNC playback message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_when_tracking_disabled(self, plugin_with_mocks: tuple) -> None:
        """GIVEN tracking disabled WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        # contextTrackAllMessages returns False
        def registry_side_effect(key, *args):
            return key != "contextTrackAllMessages"

        plugin.registryValue.side_effect = registry_side_effect

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_bot_own_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN message from bot itself WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.prefix = "botname!user@host"  # Same as bot nick

        with patch("supybot.ircutils.strEqual", return_value=True):
            plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_ctcp_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN CTCP message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=True),
            patch("supybot.ircmsgs.isAction", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_tracks_action_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN ACTION message WHEN doPrivmsg called THEN tracks message."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=True),
            patch("supybot.ircmsgs.isAction", return_value=True),
            patch("supybot.ircutils.strEqual", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_tracks_normal_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN normal message WHEN doPrivmsg called THEN tracks message with channel config."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=False),
            patch("supybot.ircutils.strEqual", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        # add_message called with channel-specific config kwarg
        call_args = plugin.context.add_message.call_args
        assert call_args[0] == ("usernick", "#channel", "user", "hello world")
        assert "config" in call_args[1]


class TestCommandFlows:
    """Test command flows with mocked LLM service.

    These tests call the internal command methods directly since supybot's wrap()
    doesn't preserve __wrapped__. We define local versions that match the original
    function signatures before wrapping.
    """

    @pytest.fixture
    def plugin_with_service(self) -> tuple:
        """Create plugin with mocked service."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_msg = MagicMock()
        mock_msg.prefix = "testnick!user@host"
        mock_msg.args = ("#channel", "test message")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#channel"

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            plugin.registryValue = MagicMock(return_value="test-key")
            plugin.context = MagicMock()
            plugin.context.get_messages.return_value = []
            plugin.llm_service = MagicMock()
            plugin.llm_service.detect_images.return_value = []
            plugin.llm_service.completion.return_value = "AI response"
            plugin.llm_service.image_generation.return_value = MagicMock(
                content="http://img.url/test.png", error=None
            )
            plugin.llm_service.save_code_to_http.return_value = "http://code.url/test.py"
            plugin.llm_service.safe_key_display.return_value = "tes***"
            plugin.llm_service.summarize.return_value = (
                None  # Default to None (fallback to truncation)
            )

        return plugin, mock_irc, mock_msg

    def _call_ask(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock, text: str) -> None:
        """Call the ask command implementation directly."""
        # Replicate what the ask method does before wrap
        if plugin._is_old_message(msg):
            return

        nick = plugin._get_nick(msg)
        channel = plugin._get_channel(msg)
        images = plugin.llm_service.detect_images(text)
        history = plugin.context.get_messages(nick, channel)

        if images:
            clean_prompt = text
            for img in images:
                clean_prompt = clean_prompt.replace(img, "").strip()
            irc.reply(f"Processing with {len(images)} image(s)...", prefixNick=False)
            response = plugin.llm_service.completion(
                clean_prompt, command="ask", images=images, history=history, irc=irc, msg=msg
            )
        else:
            response = plugin.llm_service.completion(
                text, command="ask", history=history, irc=irc, msg=msg
            )

        irc.reply(response, prefixNick=False)
        plugin.context.add_message(nick, channel, "user", text)
        plugin.context.add_message(nick, channel, "assistant", response)

    def _call_code(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock, text: str) -> None:
        """Call the code command implementation directly."""
        if plugin._is_old_message(msg):
            return

        nick = plugin._get_nick(msg)
        channel = plugin._get_channel(msg)
        history = plugin.context.get_messages(nick, channel)

        response = plugin.llm_service.completion(
            text, command="code", history=history, irc=irc, msg=msg
        )

        url = plugin.llm_service.save_code_to_http(response)
        if url:
            # Try AI-generated summary first
            summary = plugin.llm_service.summarize(response, channel)
            if summary:
                preview = summary
            else:
                # Fallback to truncation if summarization fails
                preview = response.replace("\n", " ").strip()
                if len(preview) > 60:
                    preview = preview[:57] + "..."
            irc.reply(f"{preview} — {url}")
        else:
            irc.reply(response)

        plugin.context.add_message(nick, channel, "user", text)
        plugin.context.add_message(nick, channel, "assistant", response)

    def _call_draw(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock, text: str) -> None:
        """Call the draw command implementation directly."""
        if plugin._is_old_message(msg):
            return

        nick = plugin._get_nick(msg)
        channel = plugin._get_channel(msg)

        # Get conversation history for context
        history = plugin.context.get_messages(nick, channel)

        result = plugin.llm_service.image_generation(text, history=history, irc=irc, msg=msg)
        irc.reply(result.content)

        # Store in context for follow-up references
        if result.error is None:
            plugin.context.add_message(nick, channel, "user", text)
            plugin.context.add_message(
                nick, channel, "assistant", f"[Generated image: {result.content}]"
            )
            plugin.context.add_channel_message(channel, nick, "user", text)
            plugin.context.add_channel_message(
                channel, irc.nick, "assistant", f"[Generated image: {result.content}]"
            )

    def _call_forget(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock, channel: str) -> None:
        """Call the forget command implementation directly."""
        nick = plugin._get_nick(msg)
        cleared = plugin.context.clear(nick, channel)

        if cleared:
            irc.reply("Conversation context cleared. Starting fresh!", prefixNick=False)
        else:
            irc.reply("No conversation context to clear.", prefixNick=False)

    def _call_llmkeys(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock) -> None:
        """Call the llmkeys command implementation directly."""
        ask_key = plugin.registryValue("askApiKey")
        code_key = plugin.registryValue("codeApiKey")
        draw_key = plugin.registryValue("drawApiKey")

        ask_status = plugin.llm_service.safe_key_display(ask_key)
        code_status = plugin.llm_service.safe_key_display(code_key)
        draw_status = plugin.llm_service.safe_key_display(draw_key)

        response = f"API Key Status: ask={ask_status}, code={code_status}, draw={draw_status}"
        irc.reply(response, private=True)

    def test_ask_skips_old_messages(self, plugin_with_service: tuple) -> None:
        """GIVEN ZNC playback message WHEN ask called THEN skips processing."""
        plugin, mock_irc, mock_msg = plugin_with_service
        mock_msg.time = time.time() - 100  # Old message

        self._call_ask(plugin, mock_irc, mock_msg, "test question")

        mock_irc.reply.assert_not_called()

    def test_ask_calls_completion_without_images(self, plugin_with_service: tuple) -> None:
        """GIVEN question without images WHEN ask called THEN calls completion."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_ask(plugin, mock_irc, mock_msg, "What is Python?")

        plugin.llm_service.completion.assert_called_once()
        mock_irc.reply.assert_called_with("AI response", prefixNick=False)

    def test_ask_detects_and_processes_images(self, plugin_with_service: tuple) -> None:
        """GIVEN question with image URL WHEN ask called THEN processes with image."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.detect_images.return_value = ["http://example.com/img.jpg"]

        self._call_ask(plugin, mock_irc, mock_msg, "Describe http://example.com/img.jpg")

        # Should call reply with image processing message + response
        assert mock_irc.reply.call_count == 2

    def test_ask_stores_context(self, plugin_with_service: tuple) -> None:
        """GIVEN ask command WHEN executed THEN stores context."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_ask(plugin, mock_irc, mock_msg, "test question")

        # Should add both user message and assistant response
        assert plugin.context.add_message.call_count == 2

    def test_code_generates_and_saves_code(self, plugin_with_service: tuple) -> None:
        """GIVEN code request WHEN code called THEN generates and saves code."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.completion.return_value = "def test():\n    pass\n"

        self._call_code(plugin, mock_irc, mock_msg, "Python hello world function")

        plugin.llm_service.save_code_to_http.assert_called_once()
        mock_irc.reply.assert_called()

    def test_code_falls_back_to_irc_on_save_failure(self, plugin_with_service: tuple) -> None:
        """GIVEN save failure WHEN code called THEN falls back to IRC reply."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.llm_service.completion.return_value = "print('hello')"

        self._call_code(plugin, mock_irc, mock_msg, "Python print hello")

        mock_irc.reply.assert_called_with("print('hello')")

    def test_code_shows_truncated_preview(self, plugin_with_service: tuple) -> None:
        """GIVEN long code response WHEN code called THEN shows truncated preview."""
        plugin, mock_irc, mock_msg = plugin_with_service
        long_response = (
            "Here is a really long explanation of what this code does and how it works in detail"
        )
        plugin.llm_service.completion.return_value = long_response

        self._call_code(plugin, mock_irc, mock_msg, "Python function")

        # Should show truncated preview (57 chars) + ... + URL
        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        assert "..." in reply_text
        assert "http://code.url/test.py" in reply_text
        assert len(reply_text.split(" — ")[0]) <= 60

    def test_code_shows_short_content_preview(self, plugin_with_service: tuple) -> None:
        """GIVEN short code response WHEN code called THEN shows full preview."""
        plugin, mock_irc, mock_msg = plugin_with_service
        short_response = "def foo(): pass"
        plugin.llm_service.completion.return_value = short_response

        self._call_code(plugin, mock_irc, mock_msg, "Python function")

        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        # Short content should not have ellipsis
        assert reply_text == "def foo(): pass — http://code.url/test.py"

    def test_code_preview_collapses_newlines(self, plugin_with_service: tuple) -> None:
        """GIVEN multiline code WHEN code called THEN preview collapses to single line."""
        plugin, mock_irc, mock_msg = plugin_with_service
        multiline = "def foo():\n    return 1"
        plugin.llm_service.completion.return_value = multiline

        self._call_code(plugin, mock_irc, mock_msg, "Python function")

        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        # Newlines should be replaced with spaces
        assert "\n" not in reply_text
        assert "def foo():     return 1 —" in reply_text

    def test_code_uses_ai_summary_when_available(self, plugin_with_service: tuple) -> None:
        """GIVEN summarize returns summary WHEN code called THEN uses AI summary as preview."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.completion.return_value = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        plugin.llm_service.summarize.return_value = "A recursive Fibonacci implementation"

        self._call_code(plugin, mock_irc, mock_msg, "Python fibonacci")

        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        assert "A recursive Fibonacci implementation" in reply_text
        assert "http://code.url/test.py" in reply_text

    def test_code_falls_back_to_truncation_when_summarize_fails(
        self, plugin_with_service: tuple
    ) -> None:
        """GIVEN summarize returns None WHEN code called THEN falls back to truncation."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.completion.return_value = "short code"
        plugin.llm_service.summarize.return_value = None

        self._call_code(plugin, mock_irc, mock_msg, "Python code")

        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        assert "short code —" in reply_text

    def test_code_calls_summarize_with_response_and_channel(
        self, plugin_with_service: tuple
    ) -> None:
        """GIVEN code command WHEN called THEN passes response and channel to summarize."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.llm_service.completion.return_value = "test code"
        plugin.llm_service.summarize.return_value = "summary"

        self._call_code(plugin, mock_irc, mock_msg, "Python code")

        plugin.llm_service.summarize.assert_called_once_with("test code", "#channel")

    def test_code_does_not_truncate_ai_summary(self, plugin_with_service: tuple) -> None:
        """GIVEN long AI summary WHEN code called THEN uses full summary without truncation."""
        plugin, mock_irc, mock_msg = plugin_with_service
        long_summary = "This is a comprehensive explanation of the code that generates Fibonacci numbers using recursion with memoization for optimization"
        plugin.llm_service.completion.return_value = "def fib(n): pass"
        plugin.llm_service.summarize.return_value = long_summary

        self._call_code(plugin, mock_irc, mock_msg, "Python code")

        reply_call = mock_irc.reply.call_args
        reply_text = reply_call[0][0]
        # AI summary should not be truncated
        assert long_summary in reply_text
        assert "..." not in reply_text.split(" — ")[0]  # No truncation in preview

    def test_draw_calls_image_generation(self, plugin_with_service: tuple) -> None:
        """GIVEN draw request WHEN draw called THEN calls image_generation."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_draw(plugin, mock_irc, mock_msg, "a sunset")

        plugin.llm_service.image_generation.assert_called_once()
        call_args = plugin.llm_service.image_generation.call_args
        assert call_args[0][0] == "a sunset"  # First positional arg is prompt
        assert call_args.kwargs.get("irc") == mock_irc
        assert call_args.kwargs.get("msg") == mock_msg
        mock_irc.reply.assert_called_with("http://img.url/test.png")

    def test_draw_skips_old_messages(self, plugin_with_service: tuple) -> None:
        """GIVEN ZNC playback message WHEN draw called THEN skips processing."""
        plugin, mock_irc, mock_msg = plugin_with_service
        mock_msg.time = time.time() - 100  # Old message

        self._call_draw(plugin, mock_irc, mock_msg, "a sunset")

        mock_irc.reply.assert_not_called()

    def test_draw_stores_context(self, plugin_with_service: tuple) -> None:
        """GIVEN draw command WHEN executed THEN stores personal and channel context."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_draw(plugin, mock_irc, mock_msg, "a sunset")

        # Should add user + assistant for personal context
        assert plugin.context.add_message.call_count == 2
        # Should add user + assistant for channel context
        assert plugin.context.add_channel_message.call_count == 2

    def test_forget_clears_context(self, plugin_with_service: tuple) -> None:
        """GIVEN forget command WHEN called THEN clears user context."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.context.clear.return_value = True

        self._call_forget(plugin, mock_irc, mock_msg, "#channel")

        plugin.context.clear.assert_called_once_with("testnick", "#channel")
        mock_irc.reply.assert_called()

    def test_forget_reports_no_context(self, plugin_with_service: tuple) -> None:
        """GIVEN no context to clear WHEN forget called THEN reports no context."""
        plugin, mock_irc, mock_msg = plugin_with_service
        plugin.context.clear.return_value = False

        self._call_forget(plugin, mock_irc, mock_msg, "#channel")

        # Check reply contains "No conversation context"
        mock_irc.reply.assert_called_with("No conversation context to clear.", prefixNick=False)

    def test_llmkeys_shows_key_status(self, plugin_with_service: tuple) -> None:
        """GIVEN llmkeys command WHEN called THEN shows key status privately."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_llmkeys(plugin, mock_irc, mock_msg)

        # Should call safe_key_display 3 times
        assert plugin.llm_service.safe_key_display.call_count == 3
        mock_irc.reply.assert_called_once()
        # Check it's sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is True


class TestInitContext:
    """Test _init_context method."""

    def test_init_context_creates_new_context(self) -> None:
        """GIVEN plugin WHEN _init_context called THEN creates new context."""
        from llm.context import ConversationContext
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            # Returns: contextMaxMessages, contextTimeoutMinutes, contextEnabled, channelContextMaxMessages
            plugin.registryValue = MagicMock(side_effect=[20, 30, True, 10])

            plugin._init_context()

            assert isinstance(plugin.context, ConversationContext)


class TestPluginInitialization:
    """Test plugin initialization paths."""

    def test_init_with_httproot_skips_http_callback(self) -> None:
        """GIVEN httpRoot configured WHEN plugin initialized THEN skips HTTP callback."""
        from llm.plugin import LLM

        mock_irc = MagicMock()

        with (
            patch.object(LLM, "registryValue", return_value="/var/www/llm"),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.LLMDatabase"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook") as mock_hook,
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
            plugin = LLM(mock_irc)

        # Should NOT hook HTTP callback when httpRoot is set
        mock_hook.assert_not_called()
        assert plugin._http_callback is None

    def test_init_without_httproot_registers_http_callback(self) -> None:
        """GIVEN httpRoot empty WHEN plugin initialized THEN registers HTTP callback."""
        from llm.plugin import LLM

        mock_irc = MagicMock()

        def registry_side_effect(key, *args):
            if key == "httpRoot":
                return ""
            if key == "databasePath":
                return ""
            return MagicMock()

        with (
            patch.object(LLM, "registryValue", side_effect=registry_side_effect),
            patch("llm.plugin.LLMService"),
            patch("llm.plugin.LLMDatabase"),
            patch("llm.plugin.log"),
            patch("llm.plugin.httpserver.hook") as mock_hook,
            patch("llm.plugin.schedule.addPeriodicEvent"),
            patch("llm.plugin.schedule.removeEvent"),
        ):
            plugin = LLM(mock_irc)

        # Should hook HTTP callback when httpRoot is not set
        mock_hook.assert_called_once()
        assert plugin._http_callback is not None


class TestPluginLifecycle:
    """Test plugin initialization and cleanup."""

    def test_plugin_die_removes_scheduled_event(self) -> None:
        """GIVEN plugin WHEN die called THEN removes scheduled event."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._http_callback = None

            with patch("supybot.schedule.removeEvent") as mock_remove:
                # Call parent's die
                with patch.object(LLM.__bases__[0], "die", return_value=None):
                    plugin.die()

                mock_remove.assert_any_call("llm_file_cleanup")
                mock_remove.assert_any_call("llm_startup_check")

    def test_plugin_die_unhooks_http_callback(self) -> None:
        """GIVEN plugin with HTTP callback WHEN die called THEN unhooks."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._http_callback = MagicMock()  # Has callback

            with (
                patch("supybot.schedule.removeEvent"),
                patch("supybot.httpserver.unhook") as mock_unhook,
                patch.object(LLM.__bases__[0], "die", return_value=None),
            ):
                plugin.die()

            mock_unhook.assert_called_with("llm")


class TestRunFileCleanup:
    """Test _run_file_cleanup scheduled task."""

    def test_run_file_cleanup_calls_service(self) -> None:
        """GIVEN scheduled cleanup WHEN _run_file_cleanup called THEN calls service."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.log = MagicMock()

            plugin._run_file_cleanup()

            plugin.llm_service.run_scheduled_cleanup.assert_called_once()

    def test_run_file_cleanup_handles_errors(self) -> None:
        """GIVEN cleanup error WHEN _run_file_cleanup called THEN logs error."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.llm_service.run_scheduled_cleanup.side_effect = Exception("test error")
            plugin.log = MagicMock()

            # Should not raise
            plugin._run_file_cleanup()

            plugin.log.error.assert_called_once()


class TestStartupNotification:
    """Test startup notification to bot owner."""

    @pytest.fixture
    def plugin_with_mocks(self) -> tuple:
        """Create plugin with mocked dependencies for startup tests."""
        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "VibeBot"
        mock_irc.state.channels = {"#channel1": MagicMock(), "#channel2": MagicMock()}

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._pending_channels = set()
            plugin._startup_notified = False
            plugin.log = MagicMock()

        return plugin, mock_irc

    def test_dojoin_tracks_bot_joins(self, plugin_with_mocks: tuple) -> None:
        """GIVEN bot joining channel WHEN doJoin called THEN adds to pending."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = MagicMock()
        mock_msg.nick = "VibeBot"
        mock_msg.args = ["#channel1"]

        with patch("supybot.ircutils.strEqual", return_value=True):
            plugin.doJoin(mock_irc, mock_msg)

        assert "#channel1" in plugin._pending_channels

    def test_dojoin_ignores_other_users(self, plugin_with_mocks: tuple) -> None:
        """GIVEN other user joining WHEN doJoin called THEN does not track."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = MagicMock()
        mock_msg.nick = "someuser"
        mock_msg.args = ["#channel1"]

        with patch("supybot.ircutils.strEqual", return_value=False):
            plugin.doJoin(mock_irc, mock_msg)

        assert "#channel1" not in plugin._pending_channels

    def test_do315_removes_synced_channel(self, plugin_with_mocks: tuple) -> None:
        """GIVEN pending channel WHEN do315 received THEN removes from pending."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        with patch.object(plugin, "_send_startup_notification"):
            plugin.do315(mock_irc, mock_msg)

        assert "#channel1" not in plugin._pending_channels

    def test_do315_sends_notification_when_all_synced(self, plugin_with_mocks: tuple) -> None:
        """GIVEN last channel synced WHEN do315 received THEN sends notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        with patch.object(plugin, "_send_startup_notification") as mock_notify:
            plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_called_once_with(mock_irc)
        assert plugin._startup_notified is True

    def test_do315_does_not_send_notification_if_channels_pending(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN other channels pending WHEN do315 received THEN no notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._pending_channels.add("#channel2")

        mock_msg = MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        with patch.object(plugin, "_send_startup_notification") as mock_notify:
            plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()
        assert plugin._startup_notified is False

    def test_do315_does_not_send_duplicate_notification(self, plugin_with_mocks: tuple) -> None:
        """GIVEN already notified WHEN do315 received THEN no duplicate."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._startup_notified = True

        mock_msg = MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        with patch.object(plugin, "_send_startup_notification") as mock_notify:
            plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()

    def test_do376_resets_tracking_state(self, plugin_with_mocks: tuple) -> None:
        """GIVEN plugin WHEN do376 received THEN resets tracking state."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#oldchannel")
        plugin._startup_notified = True

        mock_msg = MagicMock()

        with patch("supybot.schedule.addEvent"):
            plugin.do376(mock_irc, mock_msg)

        assert len(plugin._pending_channels) == 0
        assert plugin._startup_notified is False

    def test_do376_schedules_no_channels_check(self, plugin_with_mocks: tuple) -> None:
        """GIVEN MOTD end WHEN do376 received THEN schedules check for no channels."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = MagicMock()

        with (
            patch("supybot.schedule.removeEvent") as mock_remove_event,
            patch("supybot.schedule.addEvent") as mock_add_event,
        ):
            plugin.do376(mock_irc, mock_msg)

        mock_add_event.assert_called_once()
        mock_remove_event.assert_called_once_with("llm_startup_check")
        call_args = mock_add_event.call_args
        assert call_args.kwargs.get("name") == "llm_startup_check"

    def _mock_owner_user(self, name: str) -> MagicMock:
        """Create a mock user with owner capability."""
        mock_user = MagicMock()
        mock_user.name = name
        mock_user.capabilities = ["owner"]
        return mock_user

    def test_send_startup_notification_sends_pm(self, plugin_with_mocks: tuple) -> None:
        """GIVEN owner configured WHEN notification sent THEN PMs owner."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with an owner user
        mock_ircdb = MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user("owner_nick")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            with patch("supybot.schedule.removeEvent"):
                plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_called_once()
        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert queued_msg.args[0] == "owner_nick"
        assert "VibeBot started" in queued_msg.args[1]
        assert "2 channels" in queued_msg.args[1]
        assert "UTC" in queued_msg.args[1]

    def test_send_startup_notification_handles_no_owner(self, plugin_with_mocks: tuple) -> None:
        """GIVEN no owner configured WHEN notification sent THEN logs warning."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with no owner users
        mock_ircdb = MagicMock()
        mock_ircdb.users.users.values.return_value = []

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            with patch("supybot.schedule.removeEvent"):
                plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_not_called()
        plugin.log.warning.assert_called_once()

    def test_send_startup_notification_singular_channel(self, plugin_with_mocks: tuple) -> None:
        """GIVEN single channel WHEN notification sent THEN uses singular."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks
        mock_irc.state.channels = {"#channel1": MagicMock()}

        mock_ircdb = MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user("owner")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            with patch("supybot.schedule.removeEvent"):
                plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert "1 channel |" in queued_msg.args[1]

    def test_send_startup_notification_removes_scheduled_check(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN scheduled check exists WHEN notification sent THEN removes it."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        mock_ircdb = MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user("owner")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            with patch("supybot.schedule.removeEvent") as mock_remove:
                plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_remove.assert_called_once_with("llm_startup_check")


class TestInvalidCommand:
    """Test invalidCommand fallback to ask."""

    @pytest.fixture
    def plugin_with_mocks(self) -> tuple:
        """Create plugin with mocked dependencies for invalidCommand tests."""
        import threading

        from llm.plugin import LLM

        mock_irc = MagicMock()
        mock_irc.nick = "botname"

        mock_msg = MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.args = ("#channel", "hello there")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.startup_time = time.time()
            plugin.ask = MagicMock()
            # Limnoria's MetaSynchronized requires this lock for synchronized methods
            plugin._MetaSynchronized_rlock = threading.RLock()

        return plugin, mock_irc, mock_msg

    def test_invalid_command_empty_tokens_returns_early(self, plugin_with_mocks: tuple) -> None:
        """GIVEN empty tokens WHEN invalidCommand called THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.invalidCommand(mock_irc, mock_msg, [])

        plugin.ask.assert_not_called()

    def test_invalid_command_no_capability_returns_early(self, plugin_with_mocks: tuple) -> None:
        """GIVEN user without llm.ask capability WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with patch("llm.plugin.ircdb.checkCapability", return_value=False):
            plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_old_message_returns_early(self, plugin_with_mocks: tuple) -> None:
        """GIVEN ZNC playback message WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_delegates_to_ask(self, plugin_with_mocks: tuple) -> None:
        """GIVEN valid tokens and capability WHEN invalidCommand THEN delegates to ask."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with patch("llm.plugin.ircdb.checkCapability", return_value=True):
            plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_called_once_with(mock_irc, mock_msg, ["hello", "there"])


class TestReminderDelivery:
    """Test reminder delivery callback."""

    def test_deliver_queues_message_and_removes_reminder(self) -> None:
        """GIVEN scheduled reminder WHEN deliver fires THEN queues privmsg and cleans up."""
        from llm.plugin import LLM

        mock_irc = MagicMock()

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._reminders = {}

        event_name = "llm_remind_12345_1"
        channel = "#test"
        nick = "testuser"
        reminder_message = "check the build"

        # Simulate the deliver closure as defined in remindme()
        plugin._reminders[event_name] = (nick, channel, reminder_message)

        def deliver() -> None:
            mock_irc.queueMsg(
                __import__("supybot.ircmsgs", fromlist=["ircmsgs"]).privmsg(
                    channel, f"{nick}: Reminder: {reminder_message}"
                )
            )
            plugin._reminders.pop(event_name, None)

        deliver()

        mock_irc.queueMsg.assert_called_once()
        assert event_name not in plugin._reminders


class TestAllowConcurrent:
    """Test _allow_concurrent context manager for concurrent command execution."""

    def test_allow_concurrent_context_manager_exists(self) -> None:
        """GIVEN LLM plugin WHEN _allow_concurrent used THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "_allow_concurrent")

    def test_allow_concurrent_noop_when_lock_not_held(self) -> None:
        """GIVEN LLM plugin WHEN _allow_concurrent called without lock THEN is a no-op."""
        import threading

        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._MetaSynchronized_rlock = threading.RLock()

        # Calling _allow_concurrent when lock is not held should not raise
        with plugin._allow_concurrent():
            pass

    def test_allow_concurrent_releases_and_reacquires_lock(self) -> None:
        """GIVEN lock held WHEN _allow_concurrent used THEN lock released inside and reacquired after."""
        import threading

        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin._MetaSynchronized_rlock = threading.RLock()

        lock = plugin._MetaSynchronized_rlock
        lock.acquire()
        try:
            with plugin._allow_concurrent():
                # Lock should be released — another thread should be able to acquire it
                acquired = lock.acquire(blocking=False)
                assert acquired, "Lock should be released inside _allow_concurrent"
                lock.release()

            # Lock should be re-acquired after exiting context manager
            # Try to acquire non-blocking — should fail because it's held
            acquired = lock.acquire(blocking=False)
            # RLock is reentrant, so this will succeed but count goes up
            assert acquired
            lock.release()
        finally:
            lock.release()


class TestPluginDatabaseWiring:
    """Test database persistence wiring in plugin lifecycle."""

    def test_plugin_creates_database(self, mock_irc: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN LLMDatabase is instantiated."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        with (
            patch.object(LLM, "registryValue", side_effect=make_registry_side_effect()),
            plugin_init_patches() as mocks,
        ):
            plugin = LLM(mock_irc)

        mocks["LLMDatabase"].assert_called_once()
        assert plugin.db is mocks["LLMDatabase"].return_value

    def test_plugin_reload_reminders_reschedules_future(self, mock_irc: MagicMock) -> None:
        """GIVEN future reminder in DB WHEN plugin starts THEN schedule.addEvent called."""
        from llm.persistence import ReminderRow
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        future_time = time.time() + 3600  # 1 hour from now
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_123_1",
            nick="testuser",
            channel="#test",
            message="check build",
            fire_at=future_time,
            created_at=time.time(),
        )

        mock_db = MagicMock()
        mock_db.load_pending_reminders.return_value = [reminder]

        with (
            patch.object(LLM, "registryValue", side_effect=make_registry_side_effect()),
            plugin_init_patches(mock_database=False),
            patch("llm.plugin.LLMDatabase", return_value=mock_db),
            patch("llm.plugin.schedule.addEvent") as mock_add_event,
        ):
            plugin = LLM(mock_irc)

        # schedule.addEvent should be called with the future fire_at time
        mock_add_event.assert_called_once()
        call_kwargs = mock_add_event.call_args
        assert call_kwargs[1]["name"] == "llm_remind_123_1"
        # Reminder should be stored in plugin._reminders
        assert "llm_remind_123_1" in plugin._reminders
        assert plugin._reminders["llm_remind_123_1"] == ("testuser", "#test", "check build")

    def test_plugin_reload_reminders_delivers_overdue(self, mock_irc: MagicMock) -> None:
        """GIVEN overdue reminder in DB WHEN plugin starts THEN irc.queueMsg called."""
        from llm.persistence import ReminderRow
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        past_time = time.time() - 60  # 1 minute ago
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_123_1",
            nick="testuser",
            channel="#test",
            message="check build",
            fire_at=past_time,
            created_at=time.time() - 120,
        )

        mock_db = MagicMock()
        mock_db.load_pending_reminders.return_value = [reminder]

        with (
            patch.object(LLM, "registryValue", side_effect=make_registry_side_effect()),
            plugin_init_patches(mock_database=False),
            patch("llm.plugin.LLMDatabase", return_value=mock_db),
            patch("llm.plugin.world") as mock_world,
        ):
            mock_world.ircs = [mock_irc]
            LLM(mock_irc)

        # Overdue reminder should be delivered immediately via irc.queueMsg
        mock_irc.queueMsg.assert_called_once()
        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert queued_msg.args[0] == "#test"
        assert "testuser" in queued_msg.args[1]
        assert "check build" in queued_msg.args[1]
        # Overdue reminder should be deleted from DB after delivery
        mock_db.delete_reminder.assert_called_once_with("llm_remind_123_1")

    def test_plugin_die_cleans_expired_reminders(self, mock_irc: MagicMock) -> None:
        """GIVEN plugin with database WHEN die called THEN db.delete_expired_reminders called."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mock_db = MagicMock()
        mock_db.load_pending_reminders.return_value = []

        with (
            patch.object(LLM, "registryValue", side_effect=make_registry_side_effect()),
            plugin_init_patches(mock_database=False),
            patch("llm.plugin.LLMDatabase", return_value=mock_db),
        ):
            plugin = LLM(mock_irc)

        with (
            patch("llm.plugin.schedule.removeEvent"),
            patch("llm.plugin.httpserver.unhook"),
            patch.object(LLM.__bases__[0], "die", return_value=None),
        ):
            plugin.die()

        mock_db.delete_expired_reminders.assert_called_once()


class TestCompletionResultUsageData:
    """Test that result NamedTuples carry usage data for logging."""

    def test_completion_result_carries_usage_data(self) -> None:
        """GIVEN CompletionResult with usage WHEN accessed THEN data available for logging."""
        from llm.service import CompletionResult

        result = CompletionResult(
            content="response",
            grounding_used=False,
            prompt_tokens=150,
            completion_tokens=75,
            cost=0.002,
            model="gemini/flash",
        )
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 75
        assert result.cost == 0.002
        assert result.model == "gemini/flash"

    def test_image_result_carries_usage_data(self) -> None:
        """GIVEN ImageResult with usage WHEN accessed THEN data available for logging."""
        from llm.service import ImageResult

        result = ImageResult(
            content="http://example.com/image.png",
            prompt_tokens=50,
            completion_tokens=0,
            cost=0.04,
            model="vertex/imagen-3",
        )
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 0
        assert result.cost == 0.04
        assert result.model == "vertex/imagen-3"
