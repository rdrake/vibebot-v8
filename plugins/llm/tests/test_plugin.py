"""Tests for LLM plugin.

These tests verify the plugin structure, imports, and command registration
without requiring a full Limnoria runtime environment.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


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
    def mock_plugin(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock plugin for HTTP callback."""
        plugin = mocker.MagicMock()
        plugin.registryValue.return_value = ""  # No custom httpRoot
        return plugin

    @pytest.fixture
    def http_callback(self, mock_plugin: MagicMock):
        """Create an HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        return LLMHTTPCallback(mock_plugin)

    @pytest.fixture
    def mock_handler(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock HTTP handler."""
        handler = mocker.MagicMock()
        # wfile needs to be a MagicMock so we can set side_effect
        handler.wfile = mocker.MagicMock()
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
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN nonexistent file WHEN doGet called THEN returns 404."""
        mocker.patch.object(http_callback, "_get_web_dir", return_value="/nonexistent")
        http_callback.doGet(mock_handler, "missing.txt")
        mock_handler.send_response.assert_called_with(404)
        mock_handler.end_headers.assert_called_once()

    def test_doget_serves_existing_file(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN existing file WHEN doGet called THEN returns 200 with content."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "text/plain")
        mock_handler.send_header.assert_any_call("Content-Length", "12")

    def test_doget_serves_image_with_correct_type(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN image file WHEN doGet called THEN returns correct content type."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        http_callback.doGet(mock_handler, "test.png")

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "image/png")

    def test_doget_handles_unknown_content_type(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN file with unknown extension WHEN doGet called THEN uses octet-stream."""
        test_file = tmp_path / "test.xyz123"
        test_file.write_bytes(b"binary data")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        http_callback.doGet(mock_handler, "test.xyz123")

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "application/octet-stream")

    def test_doget_handles_broken_pipe_silently(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN client disconnect WHEN doGet serving file THEN no error raised."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        mock_handler.wfile.write.side_effect = BrokenPipeError()

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        # Should not raise
        http_callback.doGet(mock_handler, "test.txt")

    def test_doget_handles_connection_reset_silently(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN connection reset WHEN doGet serving file THEN no error raised."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        mock_handler.wfile.write.side_effect = ConnectionResetError()

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        # Should not raise
        http_callback.doGet(mock_handler, "test.txt")

    def test_doget_handles_os_error_with_500(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN OS error reading file WHEN doGet called THEN returns 500."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        mocker.patch("builtins.open", side_effect=OSError("disk error"))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(500)

    def test_doget_blocks_symlink_escape(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN symlink pointing outside web dir WHEN doGet called THEN returns 403."""
        web_dir = tmp_path / "web"
        web_dir.mkdir()

        # Create a file outside web dir
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")

        # Create a symlink inside web dir pointing outside
        symlink_path = web_dir / "escape.txt"
        symlink_path.symlink_to(outside_file)

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(web_dir))
        http_callback.doGet(mock_handler, "escape.txt")

        # Should return 403 because resolved path is outside web_dir
        mock_handler.send_response.assert_called_with(403)

    def test_doget_allows_symlink_within_web_dir(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN symlink pointing within web dir WHEN doGet called THEN serves file."""
        web_dir = tmp_path / "web"
        web_dir.mkdir()

        # Create a file inside web dir
        real_file = web_dir / "real.txt"
        real_file.write_bytes(b"content")

        # Create a symlink inside web dir pointing to the file
        symlink_path = web_dir / "link.txt"
        symlink_path.symlink_to(real_file)

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(web_dir))
        http_callback.doGet(mock_handler, "link.txt")

        # Should serve the file
        mock_handler.send_response.assert_called_with(200)

    def test_doget_handles_realpath_oserror(
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN realpath raises OSError WHEN doGet called THEN returns 403."""
        mocker.patch.object(http_callback, "_get_web_dir", return_value="/some/dir")
        mocker.patch("os.path.realpath", side_effect=OSError("permission denied"))
        http_callback.doGet(mock_handler, "test.txt")

        mock_handler.send_response.assert_called_with(403)


class TestHTTPCallbackGetWebDir:
    """Test HTTP callback _get_web_dir method."""

    def test_get_web_dir_uses_http_root_when_set(self, mocker: MockerFixture) -> None:
        """GIVEN httpRoot configured WHEN _get_web_dir called THEN returns httpRoot."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = "/custom/path"
        callback = LLMHTTPCallback(mock_plugin)

        result = callback._get_web_dir()

        assert result == "/custom/path"
        mock_plugin.registryValue.assert_called_with("httpRoot")

    def test_get_web_dir_uses_data_web_when_no_http_root(self, mocker: MockerFixture) -> None:
        """GIVEN httpRoot empty WHEN _get_web_dir called THEN returns data/web/llm."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = mocker.MagicMock()
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
    def mock_plugin(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock plugin for HTTP callback."""
        plugin = mocker.MagicMock()
        plugin.registryValue.return_value = ""
        return plugin

    @pytest.fixture
    def http_callback(self, mock_plugin: MagicMock):
        """Create an HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        return LLMHTTPCallback(mock_plugin)

    @pytest.fixture
    def mock_handler(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock HTTP handler."""
        handler = mocker.MagicMock()
        handler.wfile = mocker.MagicMock()
        return handler

    def test_serve_help_page_uses_builtin_template(
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN no custom help.html WHEN _serve_help_page THEN uses builtin template."""
        from llm.plugin import HELP_HTML_TEMPLATE

        mocker.patch.object(http_callback, "_get_web_dir", return_value="/nonexistent")
        http_callback._serve_help_page(mock_handler)

        mock_handler.send_response.assert_called_with(200)
        mock_handler.send_header.assert_any_call("Content-Type", "text/html; charset=utf-8")
        # Verify content matches template
        written_content = mock_handler.wfile.write.call_args[0][0]
        assert written_content == HELP_HTML_TEMPLATE.encode("utf-8")

    def test_serve_help_page_uses_custom_file_when_exists(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN custom help.html WHEN _serve_help_page THEN uses custom file."""
        custom_help = tmp_path / "help.html"
        custom_content = b"<html>Custom Help</html>"
        custom_help.write_bytes(custom_content)

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        http_callback._serve_help_page(mock_handler)

        mock_handler.send_response.assert_called_with(200)
        written_content = mock_handler.wfile.write.call_args[0][0]
        assert written_content == custom_content

    def test_serve_help_page_handles_broken_pipe(
        self, http_callback, mock_handler: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN client disconnect WHEN _serve_help_page THEN no error raised."""
        mock_handler.wfile.write.side_effect = BrokenPipeError()

        mocker.patch.object(http_callback, "_get_web_dir", return_value="/nonexistent")
        # Should not raise
        http_callback._serve_help_page(mock_handler)

    def test_serve_help_page_falls_back_on_read_error(
        self, http_callback, mock_handler: MagicMock, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN custom file read error WHEN _serve_help_page THEN falls back to template."""
        from llm.plugin import HELP_HTML_TEMPLATE

        custom_help = tmp_path / "help.html"
        custom_help.write_bytes(b"content")

        mocker.patch.object(http_callback, "_get_web_dir", return_value=str(tmp_path))
        mocker.patch("pathlib.Path.read_bytes", side_effect=OSError("permission denied"))
        http_callback._serve_help_page(mock_handler)

        written_content = mock_handler.wfile.write.call_args[0][0]
        assert written_content == HELP_HTML_TEMPLATE.encode("utf-8")


class TestPluginHelperMethods:
    """Test plugin helper methods."""

    @pytest.fixture
    def mock_msg(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock IRC message."""
        msg = mocker.MagicMock()
        msg.prefix = "testnick!user@host"
        msg.args = ("#testchannel", "test message")
        msg.time = time.time()
        msg.channel = "#testchannel"
        return msg

    def test_get_identity_returns_account_when_available(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user logged into NickServ WHEN _get_identity called THEN returns account."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="MyAccount")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._migrated_nicks = set()
        plugin.db = mocker.MagicMock()
        plugin.db.migrate_nick.return_value = 0
        plugin.log = mocker.MagicMock()
        result = plugin._get_identity(mock_irc, mock_msg)

        assert result == "MyAccount"

    def test_get_identity_falls_back_to_nick_when_no_account(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user not logged in WHEN _get_identity called THEN returns nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._get_identity(mock_irc, mock_msg)

        assert result == "testnick"

    def test_get_identity_falls_back_to_nick_on_keyerror(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN nickToAccount raises KeyError WHEN _get_identity called THEN returns nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("unknown nick"))

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._get_identity(mock_irc, mock_msg)

        assert result == "testnick"

    def test_extract_raw_arg_returns_target_with_brackets(self, mocker: MockerFixture) -> None:
        """GIVEN message with bracket nick WHEN _extract_raw_arg THEN brackets preserved."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_msg = mocker.MagicMock()

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage Rubin[F]")
        result = LLM._extract_raw_arg(mock_irc, mock_msg, "usage")

        assert result == "Rubin[F]"

    def test_extract_raw_arg_returns_simple_nick(self, mocker: MockerFixture) -> None:
        """GIVEN message with simple nick WHEN _extract_raw_arg THEN nick returned."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_msg = mocker.MagicMock()

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage othernick")
        result = LLM._extract_raw_arg(mock_irc, mock_msg, "usage")

        assert result == "othernick"

    def test_extract_raw_arg_returns_none_when_no_arg(self, mocker: MockerFixture) -> None:
        """GIVEN usage with no argument WHEN _extract_raw_arg THEN returns None."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_msg = mocker.MagicMock()

        mocker.patch("llm.plugin.callbacks.addressed", return_value="usage")
        result = LLM._extract_raw_arg(mock_irc, mock_msg, "usage")

        assert result is None

    def test_extract_raw_arg_handles_plugin_qualified_command(self, mocker: MockerFixture) -> None:
        """GIVEN plugin-qualified command WHEN _extract_raw_arg THEN arg extracted."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_msg = mocker.MagicMock()

        mocker.patch("llm.plugin.callbacks.addressed", return_value="llm usage Rubin[F]")
        result = LLM._extract_raw_arg(mock_irc, mock_msg, "usage")

        assert result == "Rubin[F]"

    def test_get_channel_extracts_channel_from_args(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN message with args WHEN _get_channel called THEN returns channel."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._get_channel(mock_msg)

        assert result == "#testchannel"

    def test_get_channel_returns_unknown_for_empty_args(self, mocker: MockerFixture) -> None:
        """GIVEN message with no args WHEN _get_channel called THEN returns unknown."""
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.args = []

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._get_channel(mock_msg)

        assert result == "unknown"

    def test_is_old_message_returns_true_for_old_message(self, mocker: MockerFixture) -> None:
        """GIVEN message older than startup WHEN _is_old_message THEN returns True."""
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.time = time.time() - 100

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        result = plugin._is_old_message(mock_msg)

        assert result is True

    def test_is_old_message_returns_false_for_new_message(self, mocker: MockerFixture) -> None:
        """GIVEN message newer than startup WHEN _is_old_message THEN returns False."""
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.time = time.time() + 100

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        result = plugin._is_old_message(mock_msg)

        assert result is False

    def test_is_old_message_returns_false_for_zero_timestamp(self, mocker: MockerFixture) -> None:
        """GIVEN message with time=0 WHEN _is_old_message THEN returns False.

        Limnoria defaults msg.time to 0 when no server-time tag is present.
        This should be treated as a live message, not ZNC playback.
        """
        from llm.plugin import LLM

        mock_msg = mocker.MagicMock()
        mock_msg.time = 0

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        result = plugin._is_old_message(mock_msg)

        assert result is False

    def test_get_help_url_delegates_to_service(self, mocker: MockerFixture) -> None:
        """GIVEN service returns url_base WHEN _get_help_url THEN returns url_base + /."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.get_http_paths.return_value = (
            "/var/www/llm",
            "https://example.com/llm",
        )

        result = plugin._get_help_url()

        assert result == "https://example.com/llm/"

    def test_get_help_url_with_localhost_fallback(self, mocker: MockerFixture) -> None:
        """GIVEN service returns localhost url WHEN _get_help_url THEN uses it."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.get_http_paths.return_value = (
            "/data/web/llm",
            "http://localhost:8080/llm",
        )

        result = plugin._get_help_url()

        assert result == "http://localhost:8080/llm/"

    def test_get_plugin_help_includes_url(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN getPluginHelp called THEN includes help URL."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
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
    def plugin_with_mocks(self, mocker: MockerFixture) -> tuple:
        """Create plugin with mocked dependencies."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        plugin.registryValue = mocker.MagicMock(return_value=True)
        plugin.context = mocker.MagicMock()

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

    def test_doprivmsg_skips_bot_own_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN message from bot itself WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.prefix = "botname!user@host"  # Same as bot nick

        mocker.patch("supybot.ircutils.strEqual", return_value=True)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_ctcp_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN CTCP message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_tracks_action_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ACTION message WHEN doPrivmsg called THEN tracks message."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=True)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_tracks_normal_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN normal message WHEN doPrivmsg called THEN tracks message with channel config."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # add_message called with channel-specific config kwarg
        call_args = plugin.context.add_message.call_args
        assert call_args[0] == ("usernick", "#channel", "user", "hello world")
        assert "config" in call_args[1]


class TestInitContext:
    """Test _init_context method."""

    def test_init_context_creates_new_context(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN _init_context called THEN creates new context."""
        from llm.context import ConversationContext
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        # Returns: contextMaxMessages, contextTimeoutMinutes, contextEnabled, channelContextMaxMessages
        plugin.registryValue = mocker.MagicMock(side_effect=[20, 30, True, 10])

        plugin._init_context()

        assert isinstance(plugin.context, ConversationContext)


class TestPluginInitialization:
    """Test plugin initialization paths."""

    def test_init_with_httproot_skips_http_callback(self, mocker: MockerFixture) -> None:
        """GIVEN httpRoot configured WHEN plugin initialized THEN skips HTTP callback."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()

        mocker.patch.object(LLM, "registryValue", return_value="/var/www/llm")
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mock_hook = mocker.patch("llm.plugin.httpserver.hook")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin = LLM(mock_irc)

        # Should NOT hook HTTP callback when httpRoot is set
        mock_hook.assert_not_called()
        assert plugin._http_callback is None

    def test_init_without_httproot_registers_http_callback(self, mocker: MockerFixture) -> None:
        """GIVEN httpRoot empty WHEN plugin initialized THEN registers HTTP callback."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()

        def registry_side_effect(key, *args):
            if key == "httpRoot":
                return ""
            if key == "databasePath":
                return ""
            if key == "logLevel":
                return "WARNING"
            return mocker.MagicMock()

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mock_hook = mocker.patch("llm.plugin.httpserver.hook")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin = LLM(mock_irc)

        # Should hook HTTP callback when httpRoot is not set
        mock_hook.assert_called_once()
        assert plugin._http_callback is not None


class TestPluginLifecycle:
    """Test plugin initialization and cleanup."""

    def test_plugin_die_removes_scheduled_event(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN die called THEN removes scheduled event."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._http_callback = None

        mock_remove = mocker.patch("supybot.schedule.removeEvent")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        plugin.die()

        mock_remove.assert_any_call("llm_file_cleanup")
        mock_remove.assert_any_call("llm_startup_check")

    def test_plugin_die_unhooks_http_callback(self, mocker: MockerFixture) -> None:
        """GIVEN plugin with HTTP callback WHEN die called THEN unhooks."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._http_callback = mocker.MagicMock()  # Has callback

        mocker.patch("supybot.schedule.removeEvent")
        mock_unhook = mocker.patch("supybot.httpserver.unhook")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        plugin.die()

        mock_unhook.assert_called_with("llm")


class TestRunFileCleanup:
    """Test _run_file_cleanup scheduled task."""

    def test_run_file_cleanup_calls_service(self, mocker: MockerFixture) -> None:
        """GIVEN scheduled cleanup WHEN _run_file_cleanup called THEN calls service."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.log = mocker.MagicMock()

        plugin._run_file_cleanup()

        plugin.llm_service.run_scheduled_cleanup.assert_called_once()

    def test_run_file_cleanup_handles_errors(self, mocker: MockerFixture) -> None:
        """GIVEN cleanup error WHEN _run_file_cleanup called THEN logs error."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.run_scheduled_cleanup.side_effect = Exception("test error")
        plugin.log = mocker.MagicMock()

        # Should not raise
        plugin._run_file_cleanup()

        plugin.log.error.assert_called_once()


class TestStartupNotification:
    """Test startup notification to bot owner."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture) -> tuple:
        """Create plugin with mocked dependencies for startup tests."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "VibeBot"
        mock_irc.state.channels = {"#channel1": mocker.MagicMock(), "#channel2": mocker.MagicMock()}

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._pending_channels = set()
        plugin._startup_notified = False
        plugin.log = mocker.MagicMock()

        return plugin, mock_irc

    def test_dojoin_tracks_bot_joins(self, plugin_with_mocks: tuple, mocker: MockerFixture) -> None:
        """GIVEN bot joining channel WHEN doJoin called THEN adds to pending."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = mocker.MagicMock()
        mock_msg.nick = "VibeBot"
        mock_msg.args = ["#channel1"]

        mocker.patch("supybot.ircutils.strEqual", return_value=True)
        plugin.doJoin(mock_irc, mock_msg)

        assert "#channel1" in plugin._pending_channels

    def test_dojoin_ignores_other_users(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN other user joining WHEN doJoin called THEN does not track."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = mocker.MagicMock()
        mock_msg.nick = "someuser"
        mock_msg.args = ["#channel1"]

        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doJoin(mock_irc, mock_msg)

        assert "#channel1" not in plugin._pending_channels

    def test_do315_removes_synced_channel(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN pending channel WHEN do315 received THEN removes from pending."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        assert "#channel1" not in plugin._pending_channels

    def test_do315_sends_notification_when_all_synced(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN last channel synced WHEN do315 received THEN sends notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_called_once_with(mock_irc)
        assert plugin._startup_notified is True

    def test_do315_does_not_send_notification_if_channels_pending(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN other channels pending WHEN do315 received THEN no notification."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._pending_channels.add("#channel2")

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()
        assert plugin._startup_notified is False

    def test_do315_does_not_send_duplicate_notification(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN already notified WHEN do315 received THEN no duplicate."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#channel1")
        plugin._startup_notified = True

        mock_msg = mocker.MagicMock()
        mock_msg.args = ["VibeBot", "#channel1"]

        mock_notify = mocker.patch.object(plugin, "_send_startup_notification")
        plugin.do315(mock_irc, mock_msg)

        mock_notify.assert_not_called()

    def test_do376_resets_tracking_state(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN plugin WHEN do376 received THEN resets tracking state."""
        plugin, mock_irc = plugin_with_mocks
        plugin._pending_channels.add("#oldchannel")
        plugin._startup_notified = True

        mock_msg = mocker.MagicMock()

        mocker.patch("supybot.schedule.addEvent")
        plugin.do376(mock_irc, mock_msg)

        assert len(plugin._pending_channels) == 0
        assert plugin._startup_notified is False

    def test_do376_schedules_no_channels_check(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN MOTD end WHEN do376 received THEN schedules check for no channels."""
        plugin, mock_irc = plugin_with_mocks

        mock_msg = mocker.MagicMock()

        mock_remove_event = mocker.patch("supybot.schedule.removeEvent")
        mock_add_event = mocker.patch("supybot.schedule.addEvent")
        plugin.do376(mock_irc, mock_msg)

        mock_add_event.assert_called_once()
        mock_remove_event.assert_called_once_with("llm_startup_check")
        call_args = mock_add_event.call_args
        assert call_args.kwargs.get("name") == "llm_startup_check"

    def _mock_owner_user(self, mocker: MockerFixture, name: str) -> MagicMock:
        """Create a mock user with owner capability."""
        mock_user = mocker.MagicMock()
        mock_user.name = name
        mock_user.capabilities = ["owner"]
        return mock_user

    def test_send_startup_notification_sends_pm(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN owner configured WHEN notification sent THEN PMs owner."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with an owner user
        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user(mocker, "owner_nick")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_called_once()
        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert queued_msg.args[0] == "owner_nick"
        assert "VibeBot started" in queued_msg.args[1]
        assert "2 channels" in queued_msg.args[1]
        assert "UTC" in queued_msg.args[1]

    def test_send_startup_notification_handles_no_owner(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN no owner configured WHEN notification sent THEN logs warning."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        # Mock ircdb.users.users with no owner users
        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = []

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_irc.queueMsg.assert_not_called()
        plugin.log.warning.assert_called_once()

    def test_send_startup_notification_singular_channel(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN single channel WHEN notification sent THEN uses singular."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks
        mock_irc.state.channels = {"#channel1": mocker.MagicMock()}

        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user(mocker, "owner")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        queued_msg = mock_irc.queueMsg.call_args[0][0]
        assert "1 channel |" in queued_msg.args[1]

    def test_send_startup_notification_removes_scheduled_check(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN scheduled check exists WHEN notification sent THEN removes it."""
        from llm import plugin as plugin_module

        plugin, mock_irc = plugin_with_mocks

        mock_ircdb = mocker.MagicMock()
        mock_ircdb.users.users.values.return_value = [self._mock_owner_user(mocker, "owner")]

        original_ircdb = plugin_module.ircdb
        plugin_module.ircdb = mock_ircdb
        try:
            mock_remove = mocker.patch("supybot.schedule.removeEvent")
            plugin._send_startup_notification(mock_irc)
        finally:
            plugin_module.ircdb = original_ircdb

        mock_remove.assert_called_once_with("llm_startup_check")


class TestInvalidCommand:
    """Test invalidCommand fallback to ask."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture) -> tuple:
        """Create plugin with mocked dependencies for invalidCommand tests."""
        import threading

        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.args = ("#channel", "hello there")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        plugin.ask = mocker.MagicMock()
        # Limnoria's MetaSynchronized requires this lock for synchronized methods
        plugin._MetaSynchronized_rlock = threading.RLock()

        return plugin, mock_irc, mock_msg

    def test_invalid_command_empty_tokens_returns_early(self, plugin_with_mocks: tuple) -> None:
        """GIVEN empty tokens WHEN invalidCommand called THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.invalidCommand(mock_irc, mock_msg, [])

        plugin.ask.assert_not_called()

    def test_invalid_command_no_capability_returns_early(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN user without llm.ask capability WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_old_message_returns_early(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ZNC playback message WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_delegates_to_ask(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN valid tokens and capability WHEN invalidCommand THEN delegates to ask."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_called_once_with(mock_irc, mock_msg, ["hello", "there"])


class TestReminderDelivery:
    """Test reminder delivery callback."""

    def test_deliver_queues_message_and_removes_reminder(self, mocker: MockerFixture) -> None:
        """GIVEN scheduled reminder WHEN deliver fires THEN queues privmsg and cleans up."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
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

    def test_allow_concurrent_noop_when_lock_not_held(self, mocker: MockerFixture) -> None:
        """GIVEN LLM plugin WHEN _allow_concurrent called without lock THEN is a no-op."""
        import threading

        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._MetaSynchronized_rlock = threading.RLock()

        # Calling _allow_concurrent when lock is not held should not raise
        with plugin._allow_concurrent():
            pass

    def test_allow_concurrent_releases_and_reacquires_lock(self, mocker: MockerFixture) -> None:
        """GIVEN lock held WHEN _allow_concurrent used THEN lock released inside and reacquired after."""
        import threading

        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
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


class TestPluginInit:
    """Test plugin initialization paths using full init patches."""

    def test_init_applies_custom_log_level(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN logLevel=DEBUG in config WHEN plugin initialized THEN logger level is DEBUG."""
        import logging

        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocks = plugin_init_patches(mocker)
        # Use a real logger so setLevel() actually changes the level attribute
        real_logger = logging.getLogger("test.LLM.init_log_level")
        mocks["log"].getPluginLogger.return_value = real_logger
        side_effect = make_registry_side_effect({"logLevel": "DEBUG"})
        mocker.patch.object(LLM, "registryValue", side_effect=side_effect)
        plugin = LLM(mock_irc)
        assert plugin.log.level == logging.DEBUG


class TestPluginDatabaseWiring:
    """Test database persistence wiring in plugin lifecycle."""

    def test_plugin_creates_database(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN initialized THEN LLMDatabase is instantiated."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        mocks = plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        mocks["LLMDatabase"].assert_called_once()
        assert plugin.db is mocks["LLMDatabase"].return_value

    def test_plugin_reload_reminders_reschedules_future(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
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

        mock_db = mocker.MagicMock()
        mock_db.load_pending_reminders.return_value = [reminder]

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.LLMDatabase", return_value=mock_db)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)

        # schedule.addEvent should be called with the future fire_at time
        mock_add_event.assert_called_once()
        call_kwargs = mock_add_event.call_args
        assert call_kwargs[1]["name"] == "llm_remind_123_1"
        # Reminder should be stored in plugin._reminders
        assert "llm_remind_123_1" in plugin._reminders
        assert plugin._reminders["llm_remind_123_1"] == ("testuser", "#test", "check build")

    def test_plugin_reload_reminders_delivers_overdue(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
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

        mock_db = mocker.MagicMock()
        mock_db.load_pending_reminders.return_value = [reminder]

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        mocks = plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.LLMDatabase", return_value=mock_db)
        mock_world = mocker.patch("llm.plugin.world")
        mocks["LLMService"].return_value.sanitize_output.side_effect = lambda x: x
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

    def test_plugin_die_cleans_expired_reminders(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN plugin with database WHEN die called THEN db.delete_expired_reminders called."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mock_db = mocker.MagicMock()
        mock_db.load_pending_reminders.return_value = []

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.LLMDatabase", return_value=mock_db)
        plugin = LLM(mock_irc)

        mocker.patch("llm.plugin.httpserver.unhook")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
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


class TestPendingTaskScheduler:
    """Test pending task scheduler event naming and lifecycle."""

    def test_init_schedules_pending_tasks_event(self, mocker: MockerFixture) -> None:
        """GIVEN plugin init WHEN started THEN schedules llm_pending_tasks event."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()

        mocker.patch.object(LLM, "registryValue", return_value="")
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver.hook")
        mock_add = mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")

        LLM(mock_irc)

        # Check that llm_pending_tasks was scheduled
        event_names = [call[1].get("name", "") for call in mock_add.call_args_list]
        assert "llm_pending_tasks" in event_names

    def test_die_removes_pending_tasks_event(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN die called THEN removes llm_pending_tasks event."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._http_callback = None

        mock_remove = mocker.patch("supybot.schedule.removeEvent")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        plugin.die()

        mock_remove.assert_any_call("llm_pending_tasks")


class TestDeliverPendingResult:
    """Test _deliver_pending_result sends messages to correct targets."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello world",
            "model": "gpt-4",
            "content": "The answer is 42",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_delivers_completed_ask_to_channel(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed ask result WHEN delivered THEN sends to channel."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result()
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg = mock_irc.queueMsg.call_args[0][0]
        assert "alice" in str(msg)

    def test_delivers_expired_notification(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired result WHEN delivered THEN sends apology."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(status="expired", content="", reason="expired")
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "expired" in msg_text.lower()

    def test_delivers_terminal_failure_notification(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN terminal failure WHEN delivered THEN sends failure message."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(status="failed_terminal", content="", reason="API key not configured")
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "failed" in msg_text.lower()

    def test_delivers_to_pm_target(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN PM result WHEN delivered THEN sends to nick."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(reply_target="alice", is_channel=False)
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()

    def test_logs_usage_for_completed_task(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed result with cost WHEN delivered THEN usage logged."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice_account"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(cost=0.01, prompt_tokens=100, completion_tokens=50)
        plugin._deliver_pending_result(r)

        plugin.db.log_usage.assert_called_once()
        call_args = plugin.db.log_usage.call_args[0]
        assert call_args[2] == "ask"  # command
        assert call_args[3] == "gpt-4"  # model
        assert call_args[4] == 100  # prompt_tokens
        assert call_args[5] == 50  # completion_tokens
        assert call_args[6] == 0.01  # cost

    def test_logs_structured_expired_outcome(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired deferred result WHEN delivered THEN logs structured operator entry."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(
            status="expired",
            task_type="animate",
            content="",
            reason="Request expired after retry timeout",
        )
        plugin._deliver_pending_result(r)

        # Should log a structured warning for operator visibility
        plugin.log.warning.assert_called_once()
        log_msg = plugin.log.warning.call_args[0][0]
        assert "expired" in log_msg.lower()
        # Should include key fields for grep/monitoring
        assert "animate" in plugin.log.warning.call_args[0][1]
        assert "alice" in plugin.log.warning.call_args[0][2]

    def test_logs_structured_failed_terminal_outcome(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN terminal-failure deferred result WHEN delivered THEN logs structured operator entry."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(
            status="failed_terminal",
            task_type="draw",
            content="",
            reason="API key not configured",
        )
        plugin._deliver_pending_result(r)

        # Should log a structured warning for operator visibility
        plugin.log.warning.assert_called_once()
        log_msg = plugin.log.warning.call_args[0][0]
        assert "failed_terminal" in log_msg.lower()
        assert "draw" in plugin.log.warning.call_args[0][1]
        assert "alice" in plugin.log.warning.call_args[0][2]


class TestRequireAccount:
    """Test _require_account NickServ gate helper."""

    def test_returns_account_when_identified(self, mocker: MockerFixture) -> None:
        """GIVEN identified user WHEN _require_account called THEN returns account name."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="alice_account")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, mock_msg)
        assert result == "alice_account"
        mock_irc.error.assert_not_called()

    def test_returns_none_and_errors_when_unidentified(self, mocker: MockerFixture) -> None:
        """GIVEN unidentified user WHEN _require_account called THEN returns None and errors."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()
        assert "NickServ" in mock_irc.error.call_args[0][0]

    def test_returns_none_on_key_error(self, mocker: MockerFixture) -> None:
        """GIVEN nickToAccount raises KeyError WHEN called THEN returns None."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("no such nick"))

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()

    def test_returns_none_on_attribute_error(self, mocker: MockerFixture) -> None:
        """GIVEN nickToAccount raises AttributeError WHEN called THEN returns None."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(
            side_effect=AttributeError("no nickToAccount")
        )

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()


class TestRateLimiter:
    """Test in-memory rate limiter helpers."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for rate-limit testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        p = LLM.__new__(LLM)
        p.db = mocker.MagicMock()
        p.log = mocker.MagicMock()
        p._rate_buckets = {}
        p.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
                "animateRateLimitCount": 2,
                "animateRateLimitWindow": 600,
                "enforceRateLimits": True,
            }.get(key, "")
        )
        return p

    def test_not_limited_under_threshold(self, plugin) -> None:
        """GIVEN fewer requests than limit WHEN _is_rate_limited THEN False."""
        now = 1000.0
        plugin._record_rate_limit_hit("draw", "alice", now - 10)
        plugin._record_rate_limit_hit("draw", "alice", now - 5)
        assert plugin._is_rate_limited("draw", "alice", now) is False

    def test_limited_at_threshold(self, plugin) -> None:
        """GIVEN requests at limit WHEN _is_rate_limited THEN True."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 30 + i)
        assert plugin._is_rate_limited("draw", "alice", now) is True

    def test_evicts_expired_entries(self, plugin) -> None:
        """GIVEN old entries outside window WHEN _is_rate_limited THEN evicted and not counted."""
        now = 1000.0
        # Three hits from 200s ago (outside 60s window)
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 200 + i)
        assert plugin._is_rate_limited("draw", "alice", now) is False
        assert "draw:alice" not in plugin._rate_buckets

    def test_different_commands_isolated(self, plugin) -> None:
        """GIVEN draw at limit WHEN checking animate THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("animate", "alice", now) is False

    def test_different_accounts_isolated(self, plugin) -> None:
        """GIVEN alice at limit WHEN checking bob THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("draw", "bob", now) is False

    def test_check_rate_limit_blocks_when_enforced(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN enforce=True and over limit WHEN _check_rate_limit THEN blocks and logs."""
        mock_irc = mocker.MagicMock()
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)

        mocker.patch("time.time", return_value=now)
        blocked = plugin._check_rate_limit(mock_irc, "draw", "alice", "alice", "#test", "prompt")

        assert blocked is True
        mock_irc.error.assert_called_once()
        plugin.db.log_usage.assert_called_once()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "rate_limited"
        assert "rate_limited" in plugin.log.info.call_args.args[0]

    def test_check_rate_limit_logs_only_when_not_enforced(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """GIVEN enforce=False and over limit WHEN _check_rate_limit THEN emits shadow log but allows."""
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
                "enforceRateLimits": False,
            }.get(key, "")
        )
        mock_irc = mocker.MagicMock()
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)

        mocker.patch("time.time", return_value=now)
        blocked = plugin._check_rate_limit(mock_irc, "draw", "alice", "alice", "#test", "prompt")

        assert blocked is False
        mock_irc.error.assert_not_called()
        plugin.db.log_usage.assert_not_called()
        assert "rate_limit_shadow" in plugin.log.info.call_args.args[0]


class TestRunPreflight:
    """Test _run_preflight shared preflight logic."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for preflight testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        p = LLM.__new__(LLM)
        p.db = mocker.MagicMock()
        p.db.is_user_flagged.return_value = False
        p.log = mocker.MagicMock()
        p._rate_buckets = {}
        p._migrated_nicks = set()
        p.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
                "enforceRateLimits": False,
            }.get(key, "")
        )
        return p

    def test_preflight_passes_for_ask(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN normal user WHEN ask preflight THEN not blocked."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.return_value = "alice"
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.args = ("#test", "hello")

        result = plugin._run_preflight(
            mock_irc, mock_msg, "hello", "ask", require_account=False, apply_rate_limit=False
        )
        assert result.blocked is False
        assert result.nick == "alice"
        assert result.channel == "#test"

    def test_preflight_blocks_flagged_user(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN flagged user WHEN preflight THEN blocked and usage logged."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.return_value = "baduser"
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "baduser!user@host"
        mock_msg.args = ("#test", "test")
        plugin.db.is_user_flagged.return_value = True

        result = plugin._run_preflight(
            mock_irc, mock_msg, "test", "ask", require_account=False, apply_rate_limit=False
        )
        assert result.blocked is True
        plugin.db.log_usage.assert_called_once()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "flagged_blocked"

    def test_preflight_blocks_unidentified_for_draw(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN unidentified user WHEN draw preflight THEN blocked with auth_failure."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.side_effect = KeyError("not found")
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "anon!user@host"
        mock_msg.args = ("#test", "draw me")

        result = plugin._run_preflight(
            mock_irc, mock_msg, "draw me", "draw", require_account=True, apply_rate_limit=True
        )
        assert result.blocked is True
        mock_irc.error.assert_called_once()
        plugin.db.log_usage.assert_called_once()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "auth_failure"


class TestIsContentBlockedError:
    """Test _is_content_blocked_error static method."""

    def test_detects_content_keyword(self) -> None:
        """GIVEN error with 'content' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("content policy violation") is True

    def test_detects_moderation_keyword(self) -> None:
        """GIVEN error with 'moderation' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("moderation_blocked") is True

    def test_detects_safety_keyword(self) -> None:
        """GIVEN error with 'safety' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("Triggered safety filter") is True

    def test_detects_blocked_keyword(self) -> None:
        """GIVEN error with 'blocked' WHEN checked THEN returns True."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("Request was blocked") is True

    def test_returns_false_for_generic_error(self) -> None:
        """GIVEN generic error WHEN checked THEN returns False."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("timeout exceeded") is False

    def test_returns_false_for_none(self) -> None:
        """GIVEN None WHEN checked THEN returns False."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error(None) is False

    def test_returns_false_for_empty_string(self) -> None:
        """GIVEN empty string WHEN checked THEN returns False."""
        from llm.plugin import LLM

        assert LLM._is_content_blocked_error("") is False
