"""Tests for LLM plugin.

These tests verify the plugin structure, imports, and command registration
without requiring a full Limnoria runtime environment.
"""

from __future__ import annotations

import inspect
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

    def test_doget_returns_404_at_root(self, http_callback, mock_handler: MagicMock) -> None:
        """GIVEN empty path WHEN doGet called THEN returns 404 (help is on GitHub Pages)."""
        http_callback.doGet(mock_handler, "")
        mock_handler.send_response.assert_called_with(404)
        mock_handler.end_headers.assert_called_once()

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
        msg.server_tags = {}  # default: no IRCv3 account-tag
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

    def test_get_plugin_help_uses_help_url_config(self, mocker: MockerFixture) -> None:
        """GIVEN helpUrl configured WHEN getPluginHelp called THEN uses config URL."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.registryValue = mocker.MagicMock(return_value="https://rdrake.github.io/vibebot-v8/")

        result = plugin.getPluginHelp()

        assert "https://rdrake.github.io/vibebot-v8/" in result
        assert "ask" in result


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
        mock_msg.server_tags = {}  # default: no IRCv3 account-tag

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        plugin.registryValue = mocker.MagicMock(return_value=True)
        plugin.context = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._migrated_nicks = set()
        plugin._spontaneous_cooldowns = {}
        plugin._spontaneous_events = set()

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
        plugin.db = None  # _init_context now passes db=self.db to ConversationContext

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


class TestInFilter:
    """Test inFilter sanitisation of control characters and unbalanced brackets."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture) -> object:
        """Create a bare LLM instance for inFilter tests."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        return LLM.__new__(LLM)

    @staticmethod
    def _privmsg(text: str, channel: str = "#test") -> object:
        """Build a minimal PRIVMSG, bypassing Limnoria's argument validation.

        Uses the raw-string constructor so we can inject control
        characters that the keyword constructor would reject.
        """
        import supybot.ircmsgs as ircmsgs

        return ircmsgs.IrcMsg(s=f":n!u@h PRIVMSG {channel} :{text}\r\n")

    def test_normal_text_passes_through(self, plugin: object) -> None:
        """GIVEN plain text WHEN inFilter THEN message unchanged."""
        msg = self._privmsg("hello world")
        result = plugin.inFilter(None, msg)
        assert result.args[1] == "hello world"

    def test_strips_esc_byte(self, plugin: object) -> None:
        """GIVEN text with ESC byte WHEN inFilter THEN ESC removed."""
        msg = self._privmsg("before\x1bafter")
        result = plugin.inFilter(None, msg)
        assert "\x1b" not in result.args[1]
        assert result.args[1] == "beforeafter"

    def test_ansi_escape_sequence_with_bracket(self, plugin: object) -> None:
        """GIVEN ANSI escape \\x1b[6n WHEN inFilter THEN does not crash tokenizer."""
        from supybot import callbacks

        msg = self._privmsg("\x1b[6n cursor position check")
        result = plugin.inFilter(None, msg)
        # Should not raise SyntaxError
        callbacks.tokenize(result.args[1])

    def test_unbalanced_open_bracket_escaped(self, plugin: object) -> None:
        """GIVEN unmatched [ WHEN inFilter THEN brackets replaced with full-width."""
        msg = self._privmsg("explain array[0")
        result = plugin.inFilter(None, msg)
        assert "[" not in result.args[1]
        assert "\uff3b" in result.args[1]

    def test_balanced_brackets_preserved(self, plugin: object) -> None:
        """GIVEN matched brackets WHEN inFilter THEN original brackets kept."""
        msg = self._privmsg("run [echo hello]")
        result = plugin.inFilter(None, msg)
        assert result.args[1] == "run [echo hello]"

    def test_non_privmsg_passes_through(self, plugin: object) -> None:
        """GIVEN non-PRIVMSG WHEN inFilter THEN returned unchanged."""
        import supybot.ircmsgs as ircmsgs

        msg = ircmsgs.join("#test")
        result = plugin.inFilter(None, msg)
        assert result is msg

    def test_strips_null_bytes(self, plugin: object) -> None:
        """GIVEN text with null bytes WHEN inFilter THEN nulls removed."""
        msg = self._privmsg("hello\x00world")
        result = plugin.inFilter(None, msg)
        assert result.args[1] == "helloworld"

    def test_preserves_tabs(self, plugin: object) -> None:
        """GIVEN text with tab WHEN inFilter THEN preserved."""
        msg = self._privmsg("col1\tcol2")
        result = plugin.inFilter(None, msg)
        assert result.args[1] == "col1\tcol2"

    def test_original_crash_message(self, plugin: object) -> None:
        r"""GIVEN the exact message that caused the crash WHEN inFilter THEN tokenizable."""
        from supybot import callbacks

        text = (
            r"do this but don't fuck it up suggests sending \x1b[6n"
            " to see if the terminal force-injects its cursor position"
            " into his input buffer."
        )
        msg = self._privmsg(text)
        result = plugin.inFilter(None, msg)
        # Must not raise SyntaxError
        callbacks.tokenize(result.args[1])


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
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin.context = mocker.MagicMock()
        plugin.registryValue = mocker.MagicMock(return_value=True)
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
        """GIVEN valid tokens WHEN invalidCommand THEN delegates to _ask_impl."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#channel",
                account=None,
            )
        )
        plugin._ask_impl = mocker.MagicMock()
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin._ask_impl.assert_called_once()
        plugin._run_preflight.assert_called_once()

    def test_invalid_command_does_not_call_meta(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN valid tokens WHEN invalidCommand THEN does not call _run_meta."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._run_preflight = mocker.MagicMock(
            return_value=mocker.MagicMock(
                blocked=False,
                nick="testuser",
                channel="#channel",
                account=None,
            )
        )
        plugin._run_meta = mocker.MagicMock()
        plugin._ask_impl = mocker.MagicMock()
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin._run_meta.assert_not_called()


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

        # Simulate the deliver closure as defined in remind()
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
        mock_irc.state.nickToAccount.return_value = "alice"
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
        mock_irc.state.nickToAccount.return_value = "alice"
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
            task_type="draw",
            content="",
            reason="Request expired after retry timeout",
        )
        plugin._deliver_pending_result(r)

        # Should log a structured warning for operator visibility
        plugin.log.warning.assert_called_once()
        log_msg = plugin.log.warning.call_args[0][0]
        assert "expired" in log_msg.lower()
        # Should include key fields for grep/monitoring
        assert "draw" in plugin.log.warning.call_args[0][1]
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


class TestDeliveryRetry:
    """Test delivery retry with bounded backoff and per-result error isolation."""

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
        plugin._next_wakeup_time = None
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults including task_id."""
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
            "task_id": 42,
            "delivery_attempt_count": 0,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_successful_delivery_deletes_task(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN delivery succeeds WHEN queueMsg works THEN task deleted from DB."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        plugin.db.delete_pending_task.assert_called_once_with(42)

    def test_delivery_failure_retries_with_backoff(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN queueMsg raises WHEN delivering THEN delivery retried with backoff."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("IRC connection lost")
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        # Should NOT delete, should update delivery state
        plugin.db.delete_pending_task.assert_not_called()
        plugin.db.update_delivery_attempt.assert_called_once()
        call_args = plugin.db.update_delivery_attempt.call_args
        assert call_args[1]["task_id"] == 42
        assert call_args[1]["delivery_state"] == "retrying"
        assert call_args[1]["delivery_attempt_count"] == 1

    def test_delivery_backoff_formula(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN 3 prior delivery failures WHEN failing THEN next backoff is capped at 120s."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("connection reset")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mocker.patch("llm.plugin.time.time", return_value=1000000.0)

        # 3 prior failures means this failure is attempt 4:
        # delay = min(15 * 2^(4-1), 120) = 120
        r = self._make_result(task_id=42, delivery_attempt_count=3)
        plugin._deliver_pending_result(r)

        call_args = plugin.db.update_delivery_attempt.call_args[1]
        assert call_args["delivery_attempt_count"] == 4
        assert call_args["next_attempt_at"] == 1000000.0 + 120

    def test_delivery_exhaustion_marks_failed(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN 10 delivery failures WHEN delivering THEN set delivery_failed, retain row."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("persistent failure")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        # Task already at delivery_attempt_count = 9 (this is the 10th attempt)
        r = self._make_result(task_id=42, delivery_attempt_count=9)
        plugin._deliver_pending_result(r)

        plugin.db.update_delivery_attempt.assert_called_once()
        call_args = plugin.db.update_delivery_attempt.call_args[1]
        assert call_args["delivery_attempt_count"] == 10
        assert call_args["delivery_state"] == "delivery_failed"
        # Exhausted rows should not schedule another automatic wakeup.
        mock_wakeup.assert_not_called()

    def test_batch_cascade_isolation(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN batch of 3 results WHEN second delivery fails THEN first and third still delivered."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        call_count = 0

        def flaky_queue(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("IRC send failed")

        mock_irc.queueMsg.side_effect = flaky_queue
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        results = [
            self._make_result(task_id=1, nick="alice"),
            self._make_result(task_id=2, nick="bob"),
            self._make_result(task_id=3, nick="charlie"),
        ]

        # Simulate the loop in _check_pending_tasks
        plugin.llm_service.check_pending_tasks.return_value = results
        plugin._check_pending_tasks()

        # All 3 should be attempted, not just the first
        assert mock_irc.queueMsg.call_count == 3
        # Tasks 1 and 3 should be deleted (delivered successfully)
        delete_calls = plugin.db.delete_pending_task.call_args_list
        assert len(delete_calls) == 2
        deleted_ids = {c[0][0] for c in delete_calls}
        assert deleted_ids == {1, 3}
        # Task 2 should be retried
        plugin.db.update_delivery_attempt.assert_called_once()
        assert plugin.db.update_delivery_attempt.call_args[1]["task_id"] == 2

    def test_ephemeral_results_no_db_operations(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired result with no task_id WHEN delivered THEN no DB delete/update."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(status="expired", task_id=None, content="", reason="expired")
        plugin._deliver_pending_result(r)

        # Should deliver message but not touch DB
        mock_irc.queueMsg.assert_called_once()
        plugin.db.delete_pending_task.assert_not_called()
        plugin.db.update_delivery_attempt.assert_not_called()


class TestScheduleQueueWakeup:
    """Test event-driven queue wakeup scheduling (Phase 2)."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for wakeup testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._next_wakeup_time = None
        return plugin

    def test_no_tasks_does_nothing(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN empty queue WHEN _schedule_queue_wakeup called THEN no event scheduled."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        plugin.db.get_next_due_time.return_value = None

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_not_called()
        assert plugin._next_wakeup_time is None

    def test_schedules_at_next_due_time(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN task due at T WHEN _schedule_queue_wakeup called THEN one-shot event at T."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin.db.get_next_due_time.return_value = 1060.0

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_called_once()
        call_args = mock_schedule.addEvent.call_args
        assert call_args[1]["name"] == "llm_queue_wakeup"
        assert call_args[0][1] == 1060.0  # at= parameter
        assert plugin._next_wakeup_time == 1060.0

    def test_replaces_if_earlier(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN scheduled wakeup at T=100 WHEN new due time T=50 THEN reschedule to T=50."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=10.0)
        plugin._next_wakeup_time = 100.0
        plugin.db.get_next_due_time.return_value = 50.0

        plugin._schedule_queue_wakeup()

        mock_schedule.removeEvent.assert_any_call("llm_queue_wakeup")
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 50.0

    def test_keeps_earlier_existing(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN scheduled wakeup at T=50 WHEN new due time T=100 THEN keep T=50."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=10.0)
        plugin._next_wakeup_time = 50.0
        plugin.db.get_next_due_time.return_value = 100.0

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_not_called()
        assert plugin._next_wakeup_time == 50.0

    def test_past_due_schedules_immediately(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN task due in the past WHEN _schedule_queue_wakeup called THEN schedule at now+1."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin.db.get_next_due_time.return_value = 900.0  # in the past

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_called_once()
        call_args = mock_schedule.addEvent.call_args
        # Should schedule at now + 1, not in the past
        assert call_args[0][1] == 1001.0

    def test_explicit_at_time_bypasses_db(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN explicit at_time WHEN _schedule_queue_wakeup(at_time=T) THEN uses T, no DB query."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)

        plugin._schedule_queue_wakeup(at_time=1030.0)

        plugin.db.get_next_due_time.assert_not_called()
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 1030.0

    def test_clears_stale_wakeup_in_past(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN existing wakeup already in the past WHEN new due time THEN replace it."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin._next_wakeup_time = 900.0  # already past
        plugin.db.get_next_due_time.return_value = 1060.0

        plugin._schedule_queue_wakeup()

        mock_schedule.removeEvent.assert_any_call("llm_queue_wakeup")
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 1060.0


class TestSafetyPollInterval:
    """Test that the safety poll runs at 5-minute intervals (Phase 2)."""

    def test_safety_poll_interval_is_300_seconds(self, mocker: MockerFixture) -> None:
        """GIVEN plugin init WHEN addPeriodicEvent called for pending tasks THEN interval is 300."""
        from llm.plugin import LLM

        mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.conf")
        mocker.patch("llm.plugin.world")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin.registryValue = mocker.MagicMock(return_value="")
        plugin._http_callback = None
        plugin._reminders = {}
        plugin._reminders_lock = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin._apply_log_level = mocker.MagicMock()
        plugin._next_wakeup_time = None

        # Check that the constant is defined
        assert hasattr(LLM, "_SAFETY_POLL_INTERVAL")
        assert LLM._SAFETY_POLL_INTERVAL == 300


class TestWakeupTriggers:
    """Test that wakeup is triggered from all queue mutation points (Phase 2)."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for wakeup trigger testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._next_wakeup_time = None
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
            "prompt_preview": "hello",
            "model": "gpt-4",
            "content": "answer",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
            "task_id": 42,
            "delivery_attempt_count": 0,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_check_pending_tasks_reschedules_after_batch(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """GIVEN batch completes WHEN _check_pending_tasks finishes THEN _schedule_queue_wakeup called."""
        import supybot.world as world_mod

        mocker.patch.object(world_mod, "ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = []
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        plugin._check_pending_tasks()

        mock_wakeup.assert_called_once()

    def test_check_pending_tasks_clears_stale_wakeup(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN active wakeup WHEN _check_pending_tasks runs THEN _next_wakeup_time cleared first."""
        import supybot.world as world_mod

        mocker.patch.object(world_mod, "ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = []
        plugin._next_wakeup_time = 999.0

        # Use real _schedule_queue_wakeup but mock schedule module
        mocker.patch("llm.plugin.schedule")
        plugin.db.get_next_due_time.return_value = None

        plugin._check_pending_tasks()

        # Wakeup time should be cleared since no pending tasks
        assert plugin._next_wakeup_time is None

    def test_delivery_retry_triggers_wakeup(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN delivery fails WHEN _deliver_pending_result retries THEN wakeup scheduled at retry time."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("IRC send failed")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        # Should schedule wakeup at the retry time (now + backoff)
        mock_wakeup.assert_called_once_with(at_time=1000.0 + 15)

    def test_stash_triggers_wakeup(self, mocker: MockerFixture) -> None:
        """GIVEN a request times out WHEN _stash_timeout succeeds THEN wakeup scheduled."""
        from llm.service import LLMService

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = 3600  # expiry
        mock_plugin.db.save_pending_task.return_value = 1

        service = LLMService.__new__(LLMService)
        service.plugin = mock_plugin
        service.log = mocker.MagicMock()

        now = 1000.0
        result = service._stash_timeout(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt="hello",
            model="gpt-4",
            request_data={"messages": []},
            submitted_at=now,
        )

        assert result is True
        mock_plugin._schedule_queue_wakeup.assert_called_once_with(at_time=now)


class TestRequireAccount:
    """Test _require_account account-identification gate helper."""

    def test_returns_account_when_identified(self, mocker: MockerFixture) -> None:
        """GIVEN identified user WHEN _require_account called THEN returns account name."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="alice_account")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}

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
        mock_msg.server_tags = {}

        result = plugin._require_account(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()
        err_text = mock_irc.error.call_args[0][0]
        assert "NickServ" not in err_text
        assert "identified" in err_text.lower()

    def test_returns_none_on_key_error(self, mocker: MockerFixture) -> None:
        """GIVEN nickToAccount raises KeyError WHEN called THEN returns None."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("no such nick"))

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "alice!user@host"
        mock_msg.server_tags = {}

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
        mock_msg.server_tags = {}

        result = plugin._require_account(mock_irc, mock_msg)
        assert result is None
        mock_irc.error.assert_called_once()


class TestRequireAccountUsesResolver:
    """_require_account must read account-tag via _account_from_msg."""

    def test_returns_tag_when_present(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._require_account(mock_irc, mock_msg) == "tag_acct"
        mock_irc.error.assert_not_called()

    def test_returns_none_and_errors_when_unidentified(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._require_account(mock_irc, mock_msg) is None
        mock_irc.error.assert_called_once()
        err_text = mock_irc.error.call_args[0][0]
        assert "NickServ" not in err_text
        assert "identified" in err_text.lower()


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
                "askRateLimitCount": 15,
                "askRateLimitWindow": 60,
                "drawRateLimitCount": 3,
                "drawRateLimitWindow": 60,
                "enforceRateLimits": True,
            }.get(key, 0)
        )
        return p

    def test_not_limited_under_threshold(self, plugin) -> None:
        """GIVEN fewer requests than limit WHEN _is_rate_limited THEN False."""
        now = 1000.0
        plugin._record_rate_limit_hit("draw", "alice", now - 10)
        plugin._record_rate_limit_hit("draw", "alice", now - 5)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is False

    def test_limited_at_threshold(self, plugin) -> None:
        """GIVEN requests at limit WHEN _is_rate_limited THEN True."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 30 + i)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is True

    def test_evicts_expired_entries(self, plugin) -> None:
        """GIVEN old entries outside window WHEN _is_rate_limited THEN evicted and not counted."""
        now = 1000.0
        # Three hits from 200s ago (outside 60s window)
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 200 + i)
        assert plugin._is_rate_limited("draw", "alice", now, tier="registered") is False
        assert "draw:alice" not in plugin._rate_buckets

    def test_different_commands_isolated(self, plugin) -> None:
        """GIVEN draw at limit WHEN checking ask THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("ask", "alice", now, tier="registered") is False

    def test_different_accounts_isolated(self, plugin) -> None:
        """GIVEN alice at limit WHEN checking bob THEN not limited."""
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)
        assert plugin._is_rate_limited("draw", "bob", now, tier="registered") is False

    def test_check_rate_limit_blocks_when_enforced(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN enforce=True and over limit WHEN _check_rate_limit THEN blocks and logs."""
        mock_irc = mocker.MagicMock()
        now = 1000.0
        for i in range(3):
            plugin._record_rate_limit_hit("draw", "alice", now - 10 + i)

        mocker.patch("time.time", return_value=now)
        blocked = plugin._check_rate_limit(
            mock_irc, "draw", "alice", "alice", "#test", "prompt", tier="registered"
        )

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
        blocked = plugin._check_rate_limit(
            mock_irc, "draw", "alice", "alice", "#test", "prompt", tier="registered"
        )

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
        # Default: registered user (no owner/admin/trusted capabilities)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        p = LLM.__new__(LLM)
        p.db = mocker.MagicMock()
        p.log = mocker.MagicMock()
        p._rate_buckets = {}
        p._migrated_nicks = set()
        p.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "askRateLimitCount": 15,
                "askRateLimitWindow": 60,
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
        mock_msg.server_tags = {}

        result = plugin._run_preflight(mock_irc, mock_msg, "hello", "ask", require_account=False)
        assert result.blocked is False
        assert result.nick == "alice"
        assert result.channel == "#test"

    def test_preflight_blocks_unidentified_for_draw(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN unidentified user WHEN draw preflight THEN blocked with auth_failure."""
        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount.side_effect = KeyError("not found")
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "anon!user@host"
        mock_msg.args = ("#test", "draw me")
        mock_msg.server_tags = {}

        result = plugin._run_preflight(mock_irc, mock_msg, "draw me", "draw", require_account=True)
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


class TestCommandRegistry:
    """Tests for the command metadata registry."""

    def test_registry_contains_all_commands(self) -> None:
        """GIVEN command registry WHEN checked THEN contains all user-facing commands."""
        from llm.plugin import COMMAND_REGISTRY

        names = {cmd.name for cmd in COMMAND_REGISTRY}
        expected = {
            "ask",
            "code",
            "draw",
            "forget",
            "memories",
            "instruct",
            "remind",
            "usage",
        }
        assert names == expected

    def test_registry_entries_have_required_fields(self) -> None:
        """GIVEN command registry WHEN checked THEN all entries have name, args, description."""
        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert cmd.name, "name is required"
            assert cmd.description, "description is required"
            assert cmd.category in ("generation", "memory", "utility")

    def test_registry_entries_have_examples(self) -> None:
        """GIVEN command registry WHEN checked THEN all entries have at least one example."""
        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert cmd.examples, f"{cmd.name} needs at least one example"


class TestGetPluginHelp:
    """Tests for getPluginHelp() generation from COMMAND_REGISTRY."""

    def test_get_plugin_help_lists_all_commands(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN getPluginHelp called THEN lists all registered commands."""
        from llm.plugin import COMMAND_REGISTRY, LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.registryValue = mocker.MagicMock(return_value="https://example.com/help")

        help_text = plugin.getPluginHelp()
        for cmd in COMMAND_REGISTRY:
            assert cmd.name in help_text, f"{cmd.name} missing from help"


class TestHTTPCallbackOSErrorWithBrokenPipe:
    """Test HTTP callback OSError handler when sending 500 also fails."""

    def test_oserror_then_broken_pipe_silenced(self, tmp_path, mocker: MockerFixture) -> None:
        """GIVEN file open raises OSError WHEN sending 500 also raises BrokenPipeError THEN no exception propagates."""
        from llm.plugin import LLMHTTPCallback

        callback = LLMHTTPCallback.__new__(LLMHTTPCallback)
        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = str(tmp_path)
        callback._plugin = mock_plugin

        # Create a real file so path resolution succeeds and is_file() returns True
        test_file = tmp_path / "somefile.html"
        test_file.write_bytes(b"<html>test</html>")

        handler = mocker.MagicMock()
        handler.wfile = mocker.MagicMock()

        # Patch builtins.open to raise OSError (hits line 260)
        mocker.patch("builtins.open", side_effect=OSError("disk error"))
        # Then handler.send_response raises BrokenPipeError (hits line 264)
        handler.send_response.side_effect = BrokenPipeError("client gone")

        # Should not raise
        callback.doGet(handler, "somefile.html")

    def test_oserror_then_connection_reset_silenced(self, tmp_path, mocker: MockerFixture) -> None:
        """GIVEN file open raises OSError WHEN sending 500 raises ConnectionResetError THEN no exception propagates."""
        from llm.plugin import LLMHTTPCallback

        callback = LLMHTTPCallback.__new__(LLMHTTPCallback)
        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = str(tmp_path)
        callback._plugin = mock_plugin

        test_file = tmp_path / "somefile.html"
        test_file.write_bytes(b"<html>test</html>")

        handler = mocker.MagicMock()
        handler.wfile = mocker.MagicMock()

        mocker.patch("builtins.open", side_effect=OSError("disk error"))
        handler.send_response.side_effect = ConnectionResetError("reset")

        # Should not raise
        callback.doGet(handler, "somefile.html")


class TestGetBuildInfoGitFailure:
    """Test _get_build_info when git is not available."""

    def test_git_not_found_returns_version_without_sha(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN git not installed WHEN _get_build_info called THEN returns version without SHA."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        plugin_init_patches(mocker)
        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin = LLM(mock_irc)

        mocker.patch("subprocess.check_output", side_effect=FileNotFoundError("git"))
        result = plugin._get_build_info()

        assert result.startswith("v")
        assert "(" not in result

    def test_git_subprocess_error_returns_version_without_sha(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN git fails with SubprocessError WHEN _get_build_info called THEN returns version without SHA."""
        import subprocess

        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        plugin_init_patches(mocker)
        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin = LLM(mock_irc)

        mocker.patch(
            "subprocess.check_output",
            side_effect=subprocess.SubprocessError("git failed"),
        )
        result = plugin._get_build_info()

        assert result.startswith("v")
        assert "(" not in result


class TestDeliverPendingResultCodeBranch:
    """Test _deliver_pending_result code branch with HTTP URL."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
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

    def test_code_result_with_url_sends_code_is_ready(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed code result WHEN save_code_to_http returns URL THEN sends 'code is ready' message."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        plugin.llm_service.save_code_to_http.return_value = "http://example.com/code_abc.html"

        r = self._make_result(
            task_type="code",
            nick="alice",
            content="print('hello')",
            prompt_preview="hello world",
            task_id=1,
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg = mock_irc.queueMsg.call_args[0][0]
        msg_text = str(msg)
        assert "code is ready" in msg_text
        assert "http://example.com/code_abc.html" in msg_text
        assert "alice" in msg_text

    def test_code_result_without_url_sends_content(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed code result WHEN save_code_to_http returns None THEN sends raw content."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        plugin.llm_service.save_code_to_http.return_value = None

        r = self._make_result(
            task_type="code",
            nick="alice",
            content="print('hello')",
            prompt_preview="hello world",
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "print('hello')" in msg_text
        assert "code is ready" not in msg_text


class TestDeliverPendingResultUnknownStatus:
    """Test _deliver_pending_result with an unknown status."""

    def test_unknown_status_returns_early_no_message(self, mocker: MockerFixture) -> None:
        """GIVEN result with unknown status WHEN _deliver_pending_result called THEN returns early with no IRC message."""
        from llm.plugin import LLM
        from llm.service import PendingTaskResult

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()

        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = PendingTaskResult(
            status="weird",
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="some content",
            reason="",
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_not_called()
        mock_irc.sendMsg.assert_not_called()


class TestCommandRegistryCompleteness:
    """Drift-prevention: ensures registry stays in sync with actual commands."""

    def test_all_wrapped_commands_in_registry(self) -> None:
        """GIVEN plugin class WHEN checking command methods THEN all are in registry.

        This test prevents adding a new command to plugin.py without updating
        the command registry. It uses the same introspection as Limnoria's
        isCommandMethod() to find all commands.
        """
        from llm.plugin import COMMAND_REGISTRY, LLM
        from supybot.callbacks import canonicalName

        registry_names = {cmd.name for cmd in COMMAND_REGISTRY}
        command_args = ["self", "irc", "msg", "args"]

        for name in dir(LLM):
            if name.startswith("_"):
                continue
            if name != canonicalName(name):
                continue  # filters getPluginHelp, invalidCommand, inFilter, etc.
            obj = getattr(LLM, name, None)
            if not inspect.isfunction(obj):
                continue
            if inspect.getargs(obj.__code__)[0] == command_args:
                assert name in registry_names, (
                    f"Command '{name}' is registered with Limnoria but missing from "
                    f"COMMAND_REGISTRY. Add it to keep help in sync."
                )


class TestAccountFromMsg:
    """Two-layer account resolver: server_tags then state cache."""

    def test_layer1_account_tag_wins(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "tag_acct"

    def test_layer2_state_cache_when_no_tag(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_returns_none_when_unknown(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_keyerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = KeyError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_attributeerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = AttributeError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_empty_string_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        # account-tag value of "" or "*" means "logged out" per IRCv3 — treat as no tag.
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": ""}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_star_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "*"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"


class TestResolveTierUsesResolver:
    def test_registered_tier_via_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "registered"

    def test_unregistered_when_no_tag_no_cache(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "unregistered"


class TestPreflightOptionalAccountTag:
    """When require_account=False, account-tag should still populate the account."""

    def test_optional_path_picks_up_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        result = plugin._run_preflight(
            mock_irc, mock_msg, text="hi", command="ask", require_account=False
        )
        assert result.account == "tag_acct"


class TestMaybeMigrateNickCasemap:
    def test_rfc1459_brackets_treated_as_same(self, plugin_env, mocker: MockerFixture):
        plugin, _, _ = plugin_env
        plugin.db.migrate_nick = mocker.MagicMock(return_value=0)
        # In RFC1459, "[" lowers to "{". toLower("Foo[") == "foo{".
        plugin._maybe_migrate_nick("Foo[", "foo{")
        plugin.db.migrate_nick.assert_not_called()

    def test_distinct_account_still_migrates(self, plugin_env, mocker: MockerFixture):
        plugin, _, _ = plugin_env
        plugin.db.migrate_nick = mocker.MagicMock(return_value=1)
        plugin._maybe_migrate_nick("Foo", "BarAccount")
        plugin.db.migrate_nick.assert_called_once_with("Foo", "BarAccount")
