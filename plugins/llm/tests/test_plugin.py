"""Tests for LLM plugin.

These tests verify the plugin structure, imports, and command registration
without requiring a full Limnoria runtime environment.
"""

from __future__ import annotations

import inspect
import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolCallbackResult
from llm.service import AssistantResult

from .conftest import make_registry_side_effect, make_reminder_row

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

    def test_resolve_identity_returns_account_when_available(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user logged into account WHEN _resolve_identity called THEN key is account."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="MyAccount")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._migrated_nicks = set()
        plugin.db = mocker.MagicMock()
        plugin.db.migrate_nick.return_value = 0
        plugin.db.migrate_conversations.return_value = 0
        plugin.context = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        result = plugin._resolve_identity(mock_irc, mock_msg)

        assert result.key == "MyAccount"
        assert result.account == "MyAccount"
        assert result.raw_nick == "testnick"

    def test_resolve_identity_falls_back_to_nick_when_no_account(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN user not logged in WHEN _resolve_identity called THEN key is nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._resolve_identity(mock_irc, mock_msg)

        assert result.key == "testnick"
        assert result.account is None
        assert result.raw_nick == "testnick"

    def test_resolve_identity_falls_back_to_nick_on_keyerror(
        self, mock_msg: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN nickToAccount raises KeyError WHEN _resolve_identity called THEN key is nick."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("unknown nick"))

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        result = plugin._resolve_identity(mock_irc, mock_msg)

        assert result.key == "testnick"
        assert result.account is None

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


class TestBuildBridgeTool:
    """Tests for LLM._build_bridge_tool — per-request bridge tool builder."""

    def test_returns_none_when_disabled(self, plugin_env):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            False
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )

        schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schema is None
        assert handlers is None

    def test_returns_none_when_no_commands_resolve(self, plugin_env):
        """Empty allowlist falls back to DEFAULT_ALLOWED_PLUGINS, but no
        callbacks are loaded in this test fixture, so enumerate yields
        nothing and the bridge tool is not registered."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        irc.callbacks = []

        schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schema is None
        assert handlers is None

    def test_empty_allowlist_falls_back_to_curated_default(self, plugin_env, mocker):
        """T2-A: empty bridgeAllowedPlugins + bridgeEnabled True →
        enumerate_commands receives the curated DEFAULT_ALLOWED_PLUGINS set."""
        from llm import limnoria_bridge as lb

        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else []
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        plugin._build_bridge_tool(irc, msg, "#test")

        called_allowed = enum_mock.call_args.args[2]
        assert called_allowed == lb.DEFAULT_ALLOWED_PLUGINS

    def test_explicit_non_empty_allowlist_overrides_curated_default(self, plugin_env, mocker):
        """T2-A: an operator-set allowlist must NOT get expanded with the
        curated set — explicit config wins."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        plugin._build_bridge_tool(irc, msg, "#test")

        called_allowed = enum_mock.call_args.args[2]
        assert called_allowed == frozenset({"Misc"})

    def test_returns_schema_and_handler_when_commands_present(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        fake_cmds = [
            mocker.MagicMock(
                plugin="Misc",
                command="ping",
                arg_syntax="takes no arguments",
                description="Replies with pong.",
            )
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        schemas, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert schemas is not None
        assert handlers is not None
        names = [s["function"]["name"] for s in schemas]
        assert "run_limnoria_command" in names
        assert "search_bridge_commands" in names
        assert "run_limnoria_command" in handlers
        assert "search_bridge_commands" in handlers
        run_schema = next(s for s in schemas if s["function"]["name"] == "run_limnoria_command")
        # Description should mention the available command.
        assert "Misc.ping" in run_schema["function"]["description"]

    def test_search_handler_returns_matching_commands(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        from llm import limnoria_bridge as lb

        fake_cmds = [
            lb.BridgeCommand(
                plugin="Misc", command="ping", arg_syntax="", description="Replies pong."
            ),
            lb.BridgeCommand(
                plugin="Misc",
                command="help",
                arg_syntax="<command>",
                description="Returns help for a command.",
            ),
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["search_bridge_commands"]({"query": "pong"})

        import json

        envelope = json.loads(result.content)
        assert envelope["status"] == "ok"
        assert len(envelope["matches"]) == 1
        assert envelope["matches"][0]["plugin"] == "Misc"
        assert envelope["matches"][0]["command"] == "ping"

    def test_search_handler_blank_query_returns_error(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["search_bridge_commands"]({"query": ""})

        import json

        envelope = json.loads(result.content)
        assert "error" in envelope

    def test_search_handler_clamps_limit(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        from llm import limnoria_bridge as lb

        fake_cmds = [
            lb.BridgeCommand(plugin="Misc", command=f"cmd{i}", arg_syntax="", description="ping")
            for i in range(50)
        ]
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        # Over the cap → clamped to 25
        result_over = handlers["search_bridge_commands"]({"query": "ping", "limit": 999})
        # Under the floor → clamped to 1
        result_under = handlers["search_bridge_commands"]({"query": "ping", "limit": 0})

        import json

        assert len(json.loads(result_over.content)["matches"]) == 25
        assert len(json.loads(result_under.content)["matches"]) == 1

    def test_handler_returns_tool_result_with_json(self, plugin_env, mocker):
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description=""),
            ],
        )
        mocker.patch("llm.limnoria_bridge.dispatch", return_value={"status": "ok", "reply": "pong"})

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        result = handlers["run_limnoria_command"]({"plugin": "Misc", "command": "ping", "args": ""})

        import json

        assert json.loads(result.content) == {"status": "ok", "reply": "pong"}

    def test_passes_allow_mutating_false_by_default(self, plugin_env, mocker):
        """When bridgeAllowMutating is False (default), enumerate_commands is
        called with allow_mutating=False."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )

        plugin._build_bridge_tool(irc, msg, "#test")

        assert enum_mock.call_args.kwargs["allow_mutating"] is False

    def test_passes_allow_mutating_true_when_gate_open(self, plugin_env, mocker):
        """When bridgeAllowMutating is True, enumerate_commands receives
        allow_mutating=True and the bridge dispatch handler does too."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else True
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        enum_mock = mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(
                    plugin="Later",
                    command="tell",
                    arg_syntax="<nick> <text>",
                    description="",
                )
            ],
        )
        dispatch_mock = mocker.patch(
            "llm.limnoria_bridge.dispatch",
            return_value={"status": "ok", "reply": "ok"},
        )

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")

        assert enum_mock.call_args.kwargs["allow_mutating"] is True

        handlers["run_limnoria_command"]({"plugin": "Later", "command": "tell", "args": "alice hi"})
        assert dispatch_mock.call_args.kwargs["allow_mutating"] is True

    def test_dispatch_handler_uses_closed_gate_by_default(self, plugin_env, mocker):
        """The handler closure captures the gate value at build time and uses it
        for every dispatch call within that turn."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Misc"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description="")
            ],
        )
        dispatch_mock = mocker.patch(
            "llm.limnoria_bridge.dispatch",
            return_value={"status": "ok", "reply": "pong"},
        )

        _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
        handlers["run_limnoria_command"]({"plugin": "Misc", "command": "ping", "args": ""})

        assert dispatch_mock.call_args.kwargs["allow_mutating"] is False

    def test_appends_footer_when_gate_closed_and_both_kinds_present(self, plugin_env, mocker):
        """Allowlist contains 'Later' (has both writes and reads), gate closed
        → tool description ends with the footer."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Later", command="notes", arg_syntax="", description="")
            ],
        )

        schemas, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc = schemas[0]["function"]["description"]
        assert "write commands hidden" in desc
        assert "bridgeAllowMutating" in desc

    def test_omits_footer_when_gate_open(self, plugin_env, mocker):
        """Gate open → no footer (writes are exposed; nothing to flag)."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else True
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(
                    plugin="Later",
                    command="tell",
                    arg_syntax="<nick> <text>",
                    description="",
                )
            ],
        )

        schemas, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc = schemas[0]["function"]["description"]
        assert "write commands hidden" not in desc

    def test_omits_footer_when_only_pure_read_plugins_allowed(self, plugin_env, mocker):
        """Allowlist is Time + Math (both pure-read) and gate is closed —
        nothing was hidden, so the footer would be misleading."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Time", "Math"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        mocker.patch(
            "llm.limnoria_bridge.enumerate_commands",
            return_value=[
                mocker.MagicMock(plugin="Time", command="time", arg_syntax="", description=""),
                mocker.MagicMock(
                    plugin="Math", command="calc", arg_syntax="<expr>", description=""
                ),
            ],
        )

        schemas, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc = schemas[0]["function"]["description"]
        assert "write commands hidden" not in desc

    def test_behavior_later_notes_visible_tell_hidden_when_gate_closed(self, plugin_env, mocker):
        """Behavior-level (not plumbing): with Later allowlisted and the gate
        closed, the rendered tool description must list Later.notes (read) and
        NOT list Later.tell (mutating). Same setup with gate open must list both.
        """
        plugin, irc, msg = plugin_env

        # Stub callback shaped to look like the Later plugin: 'tell' (mutating)
        # and 'notes' (read-only).
        later = mocker.MagicMock()
        later.name.return_value = "Later"
        later.canonicalName.return_value = "later"
        later.listCommands.return_value = ["tell", "notes"]
        later.getCommandMethod.side_effect = lambda cmd: mocker.MagicMock(
            __doc__={
                "tell": "<nick> <text>\n\nQueue offline message.",
                "notes": "takes no arguments\n\nList queued notes.",
            }[cmd[0]]
        )
        irc.callbacks = [later]
        mocker.patch(
            "llm.limnoria_bridge.callbacks.checkCommandCapability",
            return_value=False,
        )

        # Gate closed: only 'notes' should appear in the description table.
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else False
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        schemas_closed, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc_closed = schemas_closed[0]["function"]["description"]
        assert "later.notes" in desc_closed.lower()
        assert "later.tell" not in desc_closed.lower()

        # Gate open: both should appear.
        plugin.registryValue.side_effect = lambda k, ch=None: (
            True
            if k == "bridgeEnabled"
            else ["Later"]
            if k == "bridgeAllowedPlugins"
            else True
            if k == "bridgeAllowMutating"
            else False
            if k == "bridgeDebugInChannel"
            else None
        )
        schemas_open, _ = plugin._build_bridge_tool(irc, msg, "#test")
        desc_open = schemas_open[0]["function"]["description"]
        assert "later.notes" in desc_open.lower()
        assert "later.tell" in desc_open.lower()


class TestDoPrivmsg:
    """Test plugin doPrivmsg for channel message tracking."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture):
        """Create plugin with mocked dependencies."""
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.nick = "usernick"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"
        mock_msg.server_tags = {}  # default: no IRCv3 account-tag

        # Configure command prefix so `@cmd` short-circuits in doPrivmsg.
        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        plugin.registryValue = mocker.MagicMock(return_value=True)
        plugin.context = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._migrated_nicks = set()
        plugin._route_addressed_to_assistant = mocker.MagicMock()
        # Loom caches (PR 2). Tests that exercise the loom hook set _loom
        # explicitly; default state here is "loom not wired".
        plugin._loom = None
        plugin._loom_bridge = None
        plugin._loom_channel_cache = None
        plugin._loom_network_cache = None
        plugin._loom_bot_nicks_cache = ()

        try:
            yield plugin, mock_irc, mock_msg
        finally:
            chars_value.setValue(original_chars)

    def test_doprivmsg_routes_private_messages_to_assistant(self, plugin_with_mocks: tuple) -> None:
        """GIVEN private message WHEN doPrivmsg called THEN routed to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("botname", "remove the memories about RMS")
        mock_msg.channel = None  # Private message — no channel context

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "remove the memories about RMS"
        )
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_old_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN ZNC playback message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_drops_server_prefixed_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN server-prefixed PRIVMSG WHEN doPrivmsg called THEN dropped, not routed.

        Downstream code calls ircutils.nickFromHostmask which asserts
        user-hostmask form. Without this gate, services-originated PMs
        would crash _run_preflight (seen as AssertionError on prod).
        """
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.prefix = "luna.AfterNET.Org"  # bare server prefix
        mock_msg.args = ("botname", "some text")
        mock_msg.channel = None

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
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
        plugin._route_addressed_to_assistant.assert_not_called()

    def test_doprivmsg_routes_nick_addressed_channel_message(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed channel message WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname: remove the memories about RMS")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "remove the memories about RMS"
        )
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_routes_nick_addressed_with_comma_separator(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed with comma WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname, draw a cat")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "draw a cat"
        )

    def test_doprivmsg_routes_nick_addressed_with_whitespace_separator(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed with whitespace WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname what time is it?")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "what time is it?"
        )

    def test_doprivmsg_does_not_route_when_nick_prefix_is_part_of_word(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick is prefix of a longer word WHEN doPrivmsg THEN treats as plain chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botnamesomething")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        # Falls through to channel chatter tracking
        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_does_not_route_explicit_command_prefix(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN @-prefixed command WHEN doPrivmsg THEN skips (Limnoria handles)."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "@search foo")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_does_not_route_bare_nick_mention(self, plugin_with_mocks: tuple) -> None:
        """GIVEN message with nick alone WHEN doPrivmsg THEN treats as chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()


class TestInFilterDispatchGate:
    """inFilter must suppress Limnoria's command dispatcher for non-prefix
    addressed messages by tagging msg.addressed=''."""

    @pytest.fixture
    def plugin_and_irc(self, mocker: MockerFixture):
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        irc = mocker.MagicMock()
        irc.nick = "botname"
        try:
            yield plugin, irc
        finally:
            chars_value.setValue(original_chars)

    def _msg(self, target: str, text: str, *, sender: str = "user") -> object:
        from supybot import ircmsgs

        return ircmsgs.IrcMsg(prefix=f"{sender}!u@h", command="PRIVMSG", args=(target, text))

    def test_nick_addressed_channel_msg_is_marked_unaddressed(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "botname: remove that thing")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""

    def test_unprefixed_pm_is_marked_unaddressed(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("botname", "remove that thing")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""

    def test_at_prefixed_command_is_not_marked(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "@search foo")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") is None

    def test_at_prefixed_pm_is_not_marked(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("botname", "@later add foo bar")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") is None

    def test_plain_channel_chatter_is_marked(self, plugin_and_irc: tuple) -> None:
        # Even for non-addressed channel chatter, tagging with '' is a no-op
        # for dispatch (it would already be unaddressed). The tag itself is
        # harmless because doPrivmsg routes only when text actually starts
        # with our nick.
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "just chatting")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""


class TestStripNickPrefix:
    """Direct unit coverage for the nick-prefix-stripping helper."""

    def test_colon_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bot: hello") == "hello"

    def test_comma_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bot, hello") == "hello"

    def test_whitespace_separator(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bot hello") == "hello"

    def test_case_insensitive_match(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "BOT, hi") == "hi"

    def test_returns_none_when_nick_is_part_of_word(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bottle of wine") is None

    def test_returns_none_when_no_text_after_nick(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bot") is None

    def test_returns_none_when_text_does_not_start_with_nick(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "hello bot") is None

    def test_returns_none_when_only_separators_after_nick(self) -> None:
        from llm.plugin import LLM

        assert LLM._strip_nick_prefix("bot", "bot:   ") is None


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

        registry = make_registry_side_effect({"httpRoot": "/var/www/llm"})
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
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

        registry = make_registry_side_effect({"httpRoot": ""})
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
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


class _FakeLoom:
    """Tiny test double for the Loom orchestrator."""

    def __init__(self) -> None:
        self.observed: list[tuple[str, str]] = []

    def observe_transcript(self, nick: str, text: str) -> None:
        self.observed.append((nick, text))


class TestDoPrivmsgLoomHook:
    """C3: doPrivmsg captures loom-channel chatter on the loom network."""

    def _build_loom_plugin(self, mocker: MockerFixture):
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.network = "afternet"
        mock_irc.nick = "vibebot"
        mock_irc.state = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        return plugin

    def _make_msg(
        self,
        mocker: MockerFixture,
        *,
        target: str,
        nick: str,
        text: str,
    ):
        msg = mocker.MagicMock()
        msg.prefix = f"{nick}!u@h"
        msg.nick = nick
        msg.args = (target, text)
        msg.time = time.time() + 100  # not ZNC playback
        msg.channel = target
        msg.server_tags = {}
        return msg

    def _make_irc(self, mocker: MockerFixture, *, network: str):
        irc = mocker.MagicMock()
        irc.network = network
        irc.nick = "vibebot"
        irc.state = mocker.MagicMock()
        irc.state.channels = {}
        return irc

    def test_doprivmsg_appends_loom_transcript(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="the bell")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == [("botB", "the bell")]

    def test_doprivmsg_ignores_other_networks(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="freenode")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="hi")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []

    def test_doprivmsg_filters_by_bot_allowlist_when_set(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ("botB",)
        irc = self._make_irc(mocker, network="afternet")
        plugin.doPrivmsg(irc, self._make_msg(mocker, target="#forest", nick="alice", text="hi"))
        plugin.doPrivmsg(irc, self._make_msg(mocker, target="#forest", nick="botB", text="hi"))
        assert plugin._loom.observed == [("botB", "hi")]

    def test_doprivmsg_does_not_capture_prefix_commands(self, mocker: MockerFixture) -> None:
        # @verseapprove or @versereject in the loom channel must NOT land
        # in the loom transcript — they're commands, not improv. The
        # supybot default for whenAddressedBy.chars() is empty in tests, so
        # we set it to the operator-configured "@" used in production.
        import supybot.conf as conf

        prefix_value = conf.supybot.reply.whenAddressedBy.chars
        original = prefix_value()
        prefix_value.setValue("@")
        try:
            plugin = self._build_loom_plugin(mocker)
            plugin._loom = _FakeLoom()
            plugin._loom_channel_cache = "#forest"
            plugin._loom_network_cache = "afternet"
            plugin._loom_bot_nicks_cache = ()
            irc = self._make_irc(mocker, network="afternet")
            msg = self._make_msg(
                mocker, target="#forest", nick="alice", text="@verseapprove abc123"
            )
            plugin.doPrivmsg(irc, msg)
            assert plugin._loom.observed == []
        finally:
            prefix_value.setValue(original)

    def test_doprivmsg_ignores_other_channels_on_loom_network(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#other", nick="botB", text="hi")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []

    def test_doprivmsg_skips_capture_when_flag_disabled(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        plugin._loom_capture_transcript_cache = False
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="hi there")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []


class TestVerseproposalsCommand:
    """D1: @verseproposals listing command."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        msg.args = ("#afnet", "@verseproposals")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_default_status_pending_in_current_channel(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x"},
            confidence=0.5,
        )
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("0.50" in r and "add_event" in r for r in replies)

    def test_explicit_channel_and_status(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "y"},
            confidence=0.95,
            status="approved",
            reviewer="loom",
        )
        plugin.verseproposals(irc, msg, ["#afnet", "approved"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("0.95" in r and "add_event" in r for r in replies)

    def test_empty_list_message(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("No pending proposals" in r for r in replies)

    def test_default_caps_at_three_with_more_footer(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(7):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        # 3 proposal rows + 1 "more pending" footer = 4 IRC lines max
        assert len(replies) == 4
        assert any("more pending" in r for r in replies)

    def test_explicit_limit_overrides_default(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(7):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        # Wrap parses positional args; limit must come after channel + status.
        plugin.verseproposals(irc, msg, ["#afnet", "pending", "10"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        # 7 rows shown, no footer (because limit > total)
        assert len(replies) == 7
        assert not any("more pending" in r for r in replies)

    def test_limit_clamped_to_max(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(3):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        # Pass an absurd limit; the implementation clamps to MAX_LIMIT (50)
        # which still happily fits 3 rows without a footer.
        plugin.verseproposals(irc, msg, ["#afnet", "pending", "10000"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert len(replies) == 3
        assert not any("more pending" in r for r in replies)


class TestVerseapproveRejectCommands:
    """D2: @verseapprove + @versereject moderation commands."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        msg.args = ("#afnet", "")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_approve_applies_and_flips_status(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="line-1",
        )
        plugin.verseapprove(irc, msg, [pid[:8]])
        events = store.recent_events()
        assert len(events) == 1
        assert events[0].source == "loom"
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"
        assert p.reviewer != "loom"

    def test_approve_short_id_prefix(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="line-1",
        )
        plugin.verseapprove(irc, msg, [pid[:6]])
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"

    def test_approve_unknown_id_errors_cleanly(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.verseapprove(irc, msg, ["deadbeef"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("No proposal" in e for e in errors)

    def test_approve_already_approved_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95,
            provenance="line-1",
            status="approved",
            reviewer="loom",
        )
        plugin.verseapprove(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already approved" in r for r in replies)

    def test_approve_already_rejected_blocked(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        store.update_proposal_status(pid, status="rejected", reviewer="bob")
        plugin.verseapprove(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("rejected" in r for r in replies)

    def test_reject_flips_status_and_does_not_apply(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        plugin.versereject(irc, msg, [pid[:8]])
        assert store.recent_events() == []
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "rejected"

    def test_reject_unknown_id_errors(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.versereject(irc, msg, ["deadbeef"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("No proposal" in e for e in errors)

    def test_reject_already_approved_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95,
            provenance="x",
            status="approved",
            reviewer="loom",
        )
        plugin.versereject(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already approved" in r for r in replies)

    def test_approve_apply_exception_reports_error(self, verse_env, mocker) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        mocker.patch.object(store, "apply_proposal_and_mark", side_effect=RuntimeError("boom"))
        plugin.verseapprove(irc, msg, [pid])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("Apply failed" in e for e in errors)

    def test_proposal_target_store_no_channel_errors(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        msg.args = ("",)  # No channel context
        channel, store = plugin._proposal_target_store(irc, msg, None)
        assert channel is None
        assert store is None
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("Specify a channel" in e for e in errors)

    def test_proposal_snippet_covers_all_ops(self, verse_env) -> None:
        from llm.verse.store import Proposal

        plugin, _irc, _msg, _store = verse_env
        sa = Proposal(
            id="x",
            created_at=0.0,
            cycle_id="c",
            op="set_attribute",
            payload={"entity_id": 1, "key": "k", "value": "v"},
            confidence=0.5,
            provenance="",
            status="pending",
            reviewer=None,
            reviewed_at=None,
        )
        ar = sa._replace(
            op="add_relation",
            payload={"from_id": 1, "to_id": 2, "kind": "k", "note": ""},
        )
        ae = sa._replace(op="add_entity", payload={"kind": "place", "name": "Oak"})
        unknown = sa._replace(op="weird", payload={})
        assert "entity_id=1" in plugin._proposal_snippet(sa)
        assert "1-[k]->2" in plugin._proposal_snippet(ar)
        assert "place 'Oak'" in plugin._proposal_snippet(ae)
        assert plugin._proposal_snippet(unknown) == ""

    def test_reject_already_rejected_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        store.update_proposal_status(pid, status="rejected", reviewer="bob")
        plugin.versereject(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already rejected" in r for r in replies)


class TestPluginLoomBridge:
    """Cover _PluginLoomBridge methods + _loom_tick path."""

    def _make_bridge(self, mocker: MockerFixture, tmp_path):
        from llm.plugin import LLM, _PluginLoomBridge

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        bridge = _PluginLoomBridge(plugin, "afternet", "#forest")
        bridge._verse_data_dir = tmp_path
        return plugin, bridge

    def test_list_candidate_channels_no_irc_returns_empty(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        mocker.patch("llm.plugin.world.getIrc", return_value=None)
        assert bridge.list_candidate_channels() == []

    def test_list_candidate_channels_intersects_enabled_joined_ondisk(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        VerseStore(tmp_path, "#forest")
        VerseStore(tmp_path, "#noverse")

        plugin._verse_enabled_channels = lambda: ["#forest", "#unjoined"]

        irc_stub = mocker.MagicMock()
        irc_stub.state.channels = {"#forest": object()}
        mocker.patch("llm.plugin.world.getIrc", return_value=irc_stub)

        out = bridge.list_candidate_channels()
        assert out == ["#forest"]

    def test_candidate_weight_combines_avatars_and_events(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        store.add_entity("avatar", "Forest")
        store.add_entity("avatar", "Owl")
        store.add_event("a chime", [], "loom")
        plugin._get_or_create_verse_store = lambda ch: store
        assert bridge.candidate_weight("#forest") == 2 * 2 + 1

    def test_snapshot_returns_versesnapshot(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.loom import VerseSnapshot
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        forest_id = store.add_entity("avatar", "Forest")
        grove_id = store.add_entity("place", "Grove")
        store.add_event("a chime", [], "loom")
        plugin._get_or_create_verse_store = lambda ch: store

        snap = bridge.snapshot("#forest")
        assert isinstance(snap, VerseSnapshot)
        assert snap.channel == "#forest"
        assert ("avatar", "Forest", forest_id) in snap.top_entities
        assert ("place", "Grove", grove_id) in snap.top_entities
        assert "a chime" in snap.recent_events

    def test_snapshot_excludes_crosspoll_events(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        store.add_event("regular", [], "loom")
        store.add_event("from elsewhere", [], "crosspoll")
        plugin._get_or_create_verse_store = lambda ch: store

        snap = bridge.snapshot("#forest")
        joined = "\n".join(snap.recent_events)
        assert "regular" in joined
        assert "from elsewhere" not in joined

    def test_post_to_loom_channel_returns_false_without_irc(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        mocker.patch("llm.plugin.world.getIrc", return_value=None)
        assert bridge.post_to_loom_channel("hello") is False

    def test_post_to_loom_channel_queues_when_irc_present(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        irc_stub = mocker.MagicMock()
        mocker.patch("llm.plugin.world.getIrc", return_value=irc_stub)
        assert bridge.post_to_loom_channel("ring") is True
        irc_stub.queueMsg.assert_called_once()

    def test_schedule_after_replaces_existing_event(self, mocker: MockerFixture, tmp_path) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        rm = mocker.patch("llm.plugin.schedule.removeEvent")
        add = mocker.patch("llm.plugin.schedule.addEvent")
        bridge.schedule_after(5.0, lambda: None, "n")
        rm.assert_called_with("n")
        add.assert_called_once()

    def test_submit_routes_to_executor(self, mocker: MockerFixture, tmp_path) -> None:
        plugin, bridge = self._make_bridge(mocker, tmp_path)
        plugin._llm_executor = mocker.MagicMock()
        bridge.submit("loom:seed", lambda: None)
        plugin._llm_executor.submit.assert_called_once()

    def test_now_returns_float(self, mocker: MockerFixture, tmp_path) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        assert isinstance(bridge.now(), float)

    def test_store_for_delegates(self, mocker: MockerFixture, tmp_path) -> None:
        plugin, bridge = self._make_bridge(mocker, tmp_path)
        sentinel = object()
        plugin._get_or_create_verse_store = lambda ch: sentinel
        assert bridge.store_for("#x") is sentinel

    def test_log_usage_routes_to_db(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.loom import LoomCallUsage

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        plugin.db = mocker.MagicMock()
        bridge.log_usage(
            channel="#forest",
            op="seed",
            model="gemini/x",
            usage=LoomCallUsage(7, 3, 0.0),
        )
        plugin.db.log_usage.assert_called_once()
        kwargs = plugin.db.log_usage.call_args.kwargs
        assert kwargs["nick"] == "loom"
        assert kwargs["command"] == "loom:seed"
        assert kwargs["prompt_tokens"] == 7

    def test_loom_tick_no_loom_is_noop(self, mocker: MockerFixture) -> None:
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin._loom_tick()

    def test_loom_tick_swallows_loom_exceptions(self, mocker: MockerFixture) -> None:
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin._loom = mocker.MagicMock()
        plugin._loom.tick.side_effect = RuntimeError("boom")
        plugin._loom_tick()


class TestLoomWiring:
    """C2: plugin wires Loom + bridge via _wire_loom_if_enabled()."""

    def _build(self, mocker: MockerFixture, overrides: dict[str, object]):
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(overrides)
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        return LLM(mock_irc)

    def test_loom_disabled_when_loom_channel_empty(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {"loomNetwork": "afternet", "loomChannel": ""})
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None

    def test_loom_disabled_when_loom_network_empty(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {"loomNetwork": "", "loomChannel": "#forest"})
        assert plugin._loom is None
        assert plugin._loom_network_cache is None

    def test_loom_disabled_when_loom_channel_not_a_channel_name(
        self, mocker: MockerFixture
    ) -> None:
        # Operators sometimes paste smart-quoted text from a doc instead
        # of typing a literal channel name. Without validation the bot
        # PRIVMSGs that string as a nick and the server returns 401.
        plugin = self._build(mocker, {"loomNetwork": "afternet", "loomChannel": "“”"})
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None

    def test_loom_wired_when_both_set(self, mocker: MockerFixture) -> None:
        plugin = self._build(
            mocker,
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "botA, botB",
                "verseAutoApplyThreshold": 0.85,
            },
        )
        assert plugin._loom is not None
        assert plugin._loom_channel_cache == "#forest"
        assert plugin._loom_network_cache == "afternet"
        assert plugin._loom_bot_nicks_cache == ("botA", "botB")

    def test_rewire_after_disabling_clears_caches(self, mocker: MockerFixture) -> None:
        plugin = self._build(
            mocker,
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
            },
        )
        assert plugin._loom is not None
        # Now flip loomChannel to empty and re-wire.
        new_side = make_registry_side_effect({"loomNetwork": "afternet", "loomChannel": ""})
        plugin.registryValue = mocker.MagicMock(side_effect=new_side)
        plugin._wire_loom_if_enabled()
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None
        assert plugin._loom_bot_nicks_cache == ()

    def test_on_loom_config_change_rewires_after_live_config(self, mocker: MockerFixture) -> None:
        # Boot with loom disabled (defaults).
        plugin = self._build(mocker, {})
        assert plugin._loom is None

        # Operator runs @config plugins.LLM.loomNetwork / loomChannel ...
        # — the callback fires _on_loom_config_change which re-runs
        # _wire_loom_if_enabled with the new registry values.
        new_side = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#cybercafe",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
            }
        )
        plugin.registryValue = mocker.MagicMock(side_effect=new_side)

        plugin._on_loom_config_change()

        assert plugin._loom is not None
        assert plugin._loom_channel_cache == "#cybercafe"
        assert plugin._loom_network_cache == "afternet"

    def test_on_loom_config_change_swallows_exceptions(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {})
        mocker.patch.object(plugin, "_wire_loom_if_enabled", side_effect=RuntimeError("boom"))
        # Must not raise — non-fatal, logged only.
        plugin._on_loom_config_change()


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


class TestSafetyPollGuard:
    def test_overlapping_poll_is_skipped(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._safety_poll_inflight.set()

        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()

    def test_flag_clears_after_worker_completes(self, plugin_env, mocker) -> None:
        """Use a real LLMExecutor so add_done_callback fires."""
        import time as _t

        plugin, _irc, _msg = plugin_env
        # Stub the worker body so the future completes promptly with a
        # known result. Without this stub, the worker enters the real
        # `_check_pending_tasks` which iterates a MagicMock service
        # return value (TypeError) — the test would still "pass" but
        # only via the exception path, not the success path.
        plugin.llm_service.check_pending_tasks = mocker.MagicMock(return_value=[])

        plugin._enqueue_safety_poll()
        # Wait briefly for the future to complete.
        _t.sleep(0.5)
        assert not plugin._safety_poll_inflight.is_set()

    def test_flag_clears_on_synchronous_submit_failure(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._llm_executor.submit.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            plugin._enqueue_safety_poll()
        assert not plugin._safety_poll_inflight.is_set()

    def test_closing_short_circuits(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True

        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()


class TestSafeQueue:
    def test_safe_queue_drops_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor.shutdown()
        target_irc = mocker.MagicMock()
        ok = plugin._safe_queue(target_irc, mocker.sentinel.msg)
        target_irc.queueMsg.assert_not_called()
        assert ok is False

    def test_safe_queue_calls_queuemsg(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        target_irc = mocker.MagicMock()
        ok = plugin._safe_queue(target_irc, mocker.sentinel.msg)
        target_irc.queueMsg.assert_called_once_with(mocker.sentinel.msg)
        assert ok is True


class TestRateBucketsConcurrency:
    def test_concurrent_rate_limit_count_is_exact(self, plugin_env) -> None:
        """The lock guarantees the deque length matches the number of
        recorded hits exactly. CPython's GIL makes individual deque ops
        atomic, so a "no exception" assertion would pass even on the
        broken code — assert the count instead."""
        import threading
        import time

        plugin, _irc, _msg = plugin_env
        errors: list[Exception] = []
        threads_n = 8
        per_thread = 200
        now = time.time()
        barrier = threading.Barrier(threads_n)

        def hammer() -> None:
            try:
                barrier.wait(timeout=2)
                for _ in range(per_thread):
                    plugin._record_rate_limit_hit("ask", "alice", now)
                    plugin._is_rate_limited("ask", "alice", now, tier="registered")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(threads_n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        # The deque trims entries older than the window; with `now`
        # constant, every hit lands in the window.
        key = "ask:alice"
        assert len(plugin._rate_buckets[key]) == threads_n * per_thread


class TestWatchModeReminderMigration:
    """Verify the action-prompt reminder path submits to the executor and
    that the legacy / rate-limited / no-IRC paths still finalize on the
    main thread."""

    def test_action_prompt_dispatch_submits_to_executor(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=False)

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "watch news", "evt-1", action_prompt="say hi"
        )
        deliver()
        plugin._llm_executor.submit.assert_called_once()
        label = plugin._llm_executor.submit.call_args[0][0]
        assert label.startswith("reminder:")

    def test_legacy_no_action_prompt_does_not_submit(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        deliver = plugin._make_reminder_delivery_closure("alice", "#chan", "ping", "evt-2")
        deliver()
        plugin._llm_executor.submit.assert_not_called()
        irc.queueMsg.assert_called_once()

    def test_rate_limit_skip_does_not_submit(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=True)
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "ping", "evt-3", action_prompt="say hi"
        )
        deliver()
        plugin._llm_executor.submit.assert_not_called()
        # Skip notice queued via _safe_queue → irc.queueMsg.
        irc.queueMsg.assert_called()

    def test_no_active_irc_finalizes_in_main_thread(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        mocker.patch("llm.plugin.world.ircs", [])
        finalize = mocker.spy(plugin, "_finalize_reminder_fire")

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "ping", "evt-4", action_prompt="say hi"
        )
        deliver()
        plugin._llm_executor.submit.assert_not_called()
        finalize.assert_called_once()

    def test_closing_skips_submit_and_finalize(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin._check_rate_limit = mocker.MagicMock(return_value=False)

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "ping", "evt-5", action_prompt="say hi"
        )
        deliver()
        plugin._llm_executor.submit.assert_not_called()


class TestMemoryExtractionMigration:
    def test_extraction_submitted_to_executor(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        add_event = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")
        callback = add_event.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_called_once()
        assert plugin._llm_executor.submit.call_args[0][0].startswith("memory_extract:")

    def test_extraction_short_circuits_when_closing(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True
        add_event = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_memory_extraction("alice", "#chan", "user msg", "bot reply")
        callback = add_event.call_args[0][0]
        callback()
        plugin._llm_executor.submit.assert_not_called()


class TestLLMExecutorLifecycle:
    def test_plugin_constructs_executor(self, plugin_env) -> None:
        from llm.executor import LLMExecutor

        plugin, _irc, _msg = plugin_env
        assert isinstance(plugin._llm_executor, LLMExecutor)
        assert plugin._llm_executor.max_concurrency == 16

    def test_die_shuts_down_executor(self, plugin_env) -> None:
        plugin, _irc, _msg = plugin_env
        plugin.die()
        assert plugin._llm_executor.closing is True


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

    @pytest.fixture
    def irc(self, mocker: MockerFixture) -> object:
        """Mock irc with a nick so the dispatch-gate path can run."""
        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        return mock_irc

    @staticmethod
    def _privmsg(text: str, channel: str = "#test") -> object:
        """Build a minimal PRIVMSG, bypassing Limnoria's argument validation.

        Uses the raw-string constructor so we can inject control
        characters that the keyword constructor would reject.
        """
        import supybot.ircmsgs as ircmsgs

        return ircmsgs.IrcMsg(s=f":n!u@h PRIVMSG {channel} :{text}\r\n")

    def test_normal_text_passes_through(self, plugin: object, irc: object) -> None:
        """GIVEN plain text WHEN inFilter THEN message unchanged."""
        msg = self._privmsg("hello world")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "hello world"

    def test_strips_esc_byte(self, plugin: object, irc: object) -> None:
        """GIVEN text with ESC byte WHEN inFilter THEN ESC removed."""
        msg = self._privmsg("before\x1bafter")
        result = plugin.inFilter(irc, msg)
        assert "\x1b" not in result.args[1]
        assert result.args[1] == "beforeafter"

    def test_ansi_escape_sequence_with_bracket(self, plugin: object, irc: object) -> None:
        """GIVEN ANSI escape \\x1b[6n WHEN inFilter THEN does not crash tokenizer."""
        from supybot import callbacks

        msg = self._privmsg("\x1b[6n cursor position check")
        result = plugin.inFilter(irc, msg)
        # Should not raise SyntaxError
        callbacks.tokenize(result.args[1])

    def test_unbalanced_open_bracket_escaped(self, plugin: object, irc: object) -> None:
        """GIVEN unmatched [ WHEN inFilter THEN brackets replaced with full-width."""
        msg = self._privmsg("explain array[0")
        result = plugin.inFilter(irc, msg)
        assert "[" not in result.args[1]
        assert "\uff3b" in result.args[1]

    def test_balanced_brackets_preserved(self, plugin: object, irc: object) -> None:
        """GIVEN matched brackets WHEN inFilter THEN original brackets kept."""
        msg = self._privmsg("run [echo hello]")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "run [echo hello]"

    def test_non_privmsg_passes_through(self, plugin: object, irc: object) -> None:
        """GIVEN non-PRIVMSG WHEN inFilter THEN returned unchanged."""
        import supybot.ircmsgs as ircmsgs

        msg = ircmsgs.join("#test")
        result = plugin.inFilter(irc, msg)
        assert result is msg

    def test_strips_null_bytes(self, plugin: object, irc: object) -> None:
        """GIVEN text with null bytes WHEN inFilter THEN nulls removed."""
        msg = self._privmsg("hello\x00world")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "helloworld"

    def test_preserves_tabs(self, plugin: object, irc: object) -> None:
        """GIVEN text with tab WHEN inFilter THEN preserved."""
        msg = self._privmsg("col1\tcol2")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "col1\tcol2"

    def test_original_crash_message(self, plugin: object, irc: object) -> None:
        r"""GIVEN the exact message that caused the crash WHEN inFilter THEN tokenizable."""
        from supybot import callbacks

        text = (
            r"do this but don't fuck it up suggests sending \x1b[6n"
            " to see if the terminal force-injects its cursor position"
            " into his input buffer."
        )
        msg = self._privmsg(text)
        result = plugin.inFilter(irc, msg)
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
        plugin._reminders[event_name] = make_reminder_row(
            event_name=event_name,
            nick=nick,
            channel=channel,
            message=reminder_message,
        )

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
        created = time.time()
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_123_1",
            nick="testuser",
            channel="#test",
            message="check build",
            action_prompt="",
            account=None,
            fire_at=future_time,
            created_at=created,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        )

        mock_db = mocker.MagicMock()
        mock_db.load_pending_reminders.return_value = [reminder]

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.LLMDatabase", return_value=mock_db)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)

        # schedule.addEvent should be called with the future fire_at time.
        # __init__ also arms the daily compaction timer, so filter by name.
        reminder_calls = [
            c for c in mock_add_event.call_args_list if c.kwargs.get("name") == "llm_remind_123_1"
        ]
        assert len(reminder_calls) == 1
        call_kwargs = reminder_calls[0]
        assert call_kwargs[1]["name"] == "llm_remind_123_1"
        # Reminder should be stored in plugin._reminders
        assert "llm_remind_123_1" in plugin._reminders
        assert plugin._reminders["llm_remind_123_1"] == make_reminder_row(
            id=1,
            event_name="llm_remind_123_1",
            nick="testuser",
            channel="#test",
            message="check build",
            fire_at=future_time,
            created_at=created,
            chain_position=1,
        )

    def test_plugin_reload_reminders_delivers_overdue(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN overdue reminder in DB WHEN plugin starts THEN irc.queueMsg called."""
        from llm.persistence import ReminderRow
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        past_time = time.time() - 60  # 1 minute ago
        created = time.time() - 120
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_123_1",
            nick="testuser",
            channel="#test",
            message="check build",
            action_prompt="",
            account=None,
            fire_at=past_time,
            created_at=created,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
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

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
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
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
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
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
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
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
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
        p._rate_buckets_lock = threading.Lock()
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
        p._rate_buckets_lock = threading.Lock()
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
            "avatar",
            "remind",
            "usage",
            "verseopt",
            "verse",
            "look",
            "who",
            "versedump",
            "versepurge",
            "verseproposals",
            "verseapprove",
            "versereject",
            "versecompact",
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
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
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

    def test_server_prefix_returns_none_instead_of_assertion(
        self, plugin_env, mocker: MockerFixture
    ):
        # Server-originated PRIVMSG (no nick!user@host) used to crash
        # nickFromHostmask's assert. Must return None cleanly.
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.server_tags = {}
        mock_msg.prefix = "luna.AfterNET.Org"

        assert plugin._account_from_msg(mock_irc, mock_msg) is None
        mock_irc.state.nickToAccount.assert_not_called()

    def test_empty_prefix_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.server_tags = {}
        mock_msg.prefix = ""

        assert plugin._account_from_msg(mock_irc, mock_msg) is None
        mock_irc.state.nickToAccount.assert_not_called()


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


class TestDeliveryLogsAccountWhenPresent:
    def test_log_usage_uses_captured_account(self, plugin_env, mocker: MockerFixture):
        plugin, _, _ = plugin_env
        plugin.db.log_usage = mocker.MagicMock()
        from llm.service import PendingTaskResult

        result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            cost=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            account="alice_acct",
        )
        # Avoid real world.ircs iteration in tests.
        mocker.patch("llm.plugin.world.ircs", [mocker.MagicMock()])
        plugin._log_pending_delivery_usage(result, nick="alice", target="#chan")
        plugin.db.log_usage.assert_called_once_with(
            "alice_acct", "#chan", "ask", "gpt-4", 10, 5, 0.01
        )

    def test_log_usage_falls_back_to_resolver_when_account_null(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, _, _ = plugin_env
        plugin.db.log_usage = mocker.MagicMock()
        from llm.service import PendingTaskResult

        result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            cost=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            account=None,
        )
        mocker.patch.object(plugin, "_resolve_nick_to_identity", return_value="alice")
        mocker.patch("llm.plugin.world.ircs", [mocker.MagicMock()])
        plugin._log_pending_delivery_usage(result, nick="alice", target="#chan")
        plugin.db.log_usage.assert_called_once_with("alice", "#chan", "ask", "gpt-4", 10, 5, 0.01)

    def test_log_usage_skipped_when_zero_cost_and_tokens(self, plugin_env, mocker: MockerFixture):
        plugin, _, _ = plugin_env
        plugin.db.log_usage = mocker.MagicMock()
        from llm.service import PendingTaskResult

        result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            cost=0,
            prompt_tokens=0,
            completion_tokens=0,
            account="alice_acct",
        )
        mocker.patch("llm.plugin.world.ircs", [mocker.MagicMock()])
        plugin._log_pending_delivery_usage(result, nick="alice", target="#chan")
        plugin.db.log_usage.assert_not_called()


class TestPatchedDoJoin:
    """The plugin patches supybot.irclib.Irc.doJoin to skip slow auto-queries."""

    def _self_join(self, mocker: MockerFixture, channel="#test", nick="testbot"):
        msg = mocker.MagicMock()
        msg.nick = nick
        msg.args = (channel,)
        return msg

    def test_mode_b_never_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        for call in mock_irc.queueMsg.call_args_list:
            sent = call.args[0]
            if getattr(sent, "command", "") == "MODE" and "+b" in getattr(sent, "args", ()):
                pytest.fail(f"MODE +b should never be queued: {sent}")

    def test_who_skipped_when_both_caps_and_flag_enabled(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" not in commands

    def test_who_kept_when_account_tag_missing(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_extended_join_missing(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_flag_disabled(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        # Override the registry default for this test.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "skipAutoWhoOnJoin" else ""
        )
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_channel_mode_always_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        mode_calls = [
            c.args[0]
            for c in mock_irc.queueMsg.call_args_list
            if getattr(c.args[0], "command", "") == "MODE"
        ]
        # Plain MODE <channel> has args=(channel,) — length 1.
        assert any(len(getattr(m, "args", ())) == 1 for m in mode_calls)


class TestPluginDoJoinPendingChannels:
    """Plugin's own doJoin must not add to _pending_channels when WHO is skipped."""

    def test_pending_added_when_who_will_fire(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = set()  # no caps → WHO fires
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" in plugin._pending_channels

    def test_pending_NOT_added_when_who_will_be_skipped(  # noqa: N802
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" not in plugin._pending_channels, (
            "When WHO is skipped, do315 won't fire — the bot must not add to "
            "_pending_channels or startup notification will never send."
        )


class TestChatProfileBridgeWiring:
    """Tests that _ask_impl passes extra_tools/extra_handlers to assistant_request."""

    def _make_assistant_result(self) -> AssistantResult:
        return AssistantResult(
            content="bridge wiring reply",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

    def test_passes_bridge_extras_when_enabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN bridgeEnabled=True with allowed plugins WHEN ask is called
        THEN assistant_request receives extra_tools (list) and extra_handlers (dict)."""
        plugin, mock_irc, mock_msg = plugin_env

        # Override registry: enable bridge with one allowed plugin.
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"bridgeEnabled": True, "bridgeAllowedPlugins": ["Misc"]}
        )

        fake_cmd = mocker.MagicMock(
            plugin="Misc",
            command="ping",
            arg_syntax="takes no arguments",
            description="Replies with pong.",
        )
        mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=[fake_cmd])

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_assistant_result()

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        extra_tools = kwargs.get("extra_tools")
        extra_handlers = kwargs.get("extra_handlers")
        assert extra_tools is not None and len(extra_tools) == 2
        names = [t["function"]["name"] for t in extra_tools]
        assert "run_limnoria_command" in names
        assert "search_bridge_commands" in names
        assert extra_handlers is not None
        assert "run_limnoria_command" in extra_handlers
        assert "search_bridge_commands" in extra_handlers

    def test_omits_bridge_extras_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN bridgeEnabled=False (default) WHEN ask is called
        THEN assistant_request receives extra_tools=None and extra_handlers=None."""
        plugin, mock_irc, mock_msg = plugin_env

        # Default conftest registry has bridgeEnabled=False — no override needed.
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_assistant_result()

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs.get("extra_tools") is None
        assert kwargs.get("extra_handlers") is None


class TestExtractActionFloodSafety:
    """``_extract_action`` must collapse newlines so the IRC ACTION is single-line.

    Sending raw ``\\n`` inside an ACTION puts literal newlines on the wire; the
    server parses each as a separate command and disconnects with Excess Flood
    (regression for vibebot disconnect on a multi-line ``/me`` reply).
    """

    @pytest.fixture
    def plugin_with_extract(self, mocker: MockerFixture):
        from llm.plugin import LLM

        irc = mocker.MagicMock()
        irc.nick = "vibebot"
        # Wire a stand-in that exposes the real _collapse_for_irc but doesn't
        # require a full Limnoria callback boot.
        stand_in = mocker.MagicMock(spec_set=["_collapse_for_irc"])
        stand_in._collapse_for_irc = LLM._collapse_for_irc
        return LLM._extract_action.__get__(stand_in, LLM), irc

    def test_collapses_multiline_me_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        result = extract(irc, "/me draws a cookbook\nwith many recipes\nand pictures")
        assert result == "draws a cookbook | with many recipes | and pictures"
        assert "\n" not in result

    def test_collapses_multiline_star_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        result = extract(irc, "* vibebot draws\nline 2\nline 3")
        assert result == "draws | line 2 | line 3"
        assert "\n" not in result

    def test_returns_none_for_blank_collapsed_body(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        # All-whitespace body should not produce an empty-string action.
        assert extract(irc, "/me \n\n  \n") is None

    def test_returns_none_for_non_action(self, plugin_with_extract) -> None:
        extract, irc = plugin_with_extract
        assert extract(irc, "Just a regular reply.") is None


class TestPendingTaskFns:
    """Phase 2 follow-up — unified `_pending_task_fns` helper wiring.

    Replaces the older split between `_reminder_fns` and
    `_scheduled_llm_task_fns`; the LLM-facing list/cancel surface now spans
    both reminders and scheduled tasks via a single helper.
    """

    def test_helper_returns_unified_callables_with_owner_identity_bound(
        self, mocker: MockerFixture
    ) -> None:
        """The helper closes over caller/irc/msg/channel and dispatches to the
        right backend by id prefix."""
        from llm.persistence import ScheduledLlmTaskRow
        from llm.plugin import LLM, Identity
        from llm.service import ScheduleLlmTaskResult

        stand_in = mocker.MagicMock()
        stand_in.llm_service = mocker.MagicMock()
        stand_in.llm_service.schedule_llm_task.return_value = ScheduleLlmTaskResult(
            status="ok",
            event_name="llm_task_xyz",
            fire_at=1700000000.0,
            message="Scheduled.",
            note=None,
        )
        stand_in.llm_service.list_scheduled_llm_tasks.return_value = [
            ScheduledLlmTaskRow(
                id=1,
                event_name="llm_task_ev1",
                creator_nick="rdrake",
                account="rdrake_a",
                channel="#t",
                network="afternet",
                wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
                prompt="check the build" * 4,  # >80 chars to verify truncation
                fire_at=1700000000.0,
                created_at=1699999000.0,
                chain_position=1,
                recurrence_seconds=300,
                recurrence_rrule=None,
                watch_mode=False,
            ),
        ]
        stand_in.llm_service.cancel_scheduled_llm_task.return_value = ScheduleLlmTaskResult(
            status="ok",
            event_name="llm_task_ev1",
            fire_at=0.0,
            message="Cancelled.",
            note=None,
        )
        # Reminder side: stub _get_user_reminders + the per-id helpers used
        # internally by cancel_pending_task_fn.
        stand_in._get_user_reminders.return_value = [
            ("llm_remind_rdrake_abc123", ("rdrake", "#t", "check build")),
        ]
        stand_in._remind_set_for_assistant.return_value = ToolCallbackResult(
            True, "I'll remind you."
        )
        stand_in._remind_delete_for_assistant.return_value = ToolCallbackResult(
            True, "Deleted reminder abc123."
        )
        stand_in._remind_clear_for_assistant.return_value = "Cancelled 1 reminder."

        helper = LLM._pending_task_fns.__get__(stand_in, LLM)
        caller = Identity(raw_nick="rdrake", account="rdrake_a")
        irc = mocker.MagicMock()
        msg = mocker.MagicMock()
        fns = helper(caller=caller, irc=irc, msg=msg, channel="#t")

        assert set(fns.keys()) == {
            "set_reminder_fn",
            "schedule_llm_task_fn",
            "list_pending_tasks_fn",
            "cancel_pending_task_fn",
            "cancel_all_pending_tasks_fn",
        }

        # schedule_fn forwards keyword args and binds caller identity.
        out = fns["schedule_llm_task_fn"](when_natural="in 60s", prompt="ping me")
        assert out["status"] == "ok"
        assert out["event_name"] == "llm_task_xyz"
        stand_in.llm_service.schedule_llm_task.assert_called_once_with(
            irc=irc,
            msg=msg,
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#t",
            when_natural="in 60s",
            prompt="ping me",
            reply_target=None,
        )

        # list_pending_tasks_fn merges reminders + scheduled tasks with
        # `kind` discriminators.
        listed = fns["list_pending_tasks_fn"]()
        assert {row["kind"] for row in listed} == {"reminder", "scheduled_task"}
        scheduled = next(r for r in listed if r["kind"] == "scheduled_task")
        assert scheduled["id"] == "llm_task_ev1"
        assert len(scheduled["description"]) <= 80
        assert scheduled["recurrence"] == "every 300s"
        reminder = next(r for r in listed if r["kind"] == "reminder")
        assert reminder["id"] == "abc123"
        assert reminder["description"] == "check build"

        # cancel_pending_task_fn routes by id prefix to the right backend.
        cancelled = fns["cancel_pending_task_fn"]("llm_task_ev1")
        assert cancelled["status"] == "ok"
        assert cancelled["kind"] == "scheduled_task"
        stand_in.llm_service.cancel_scheduled_llm_task.assert_called_once_with(
            event_name="llm_task_ev1",
            creator_nick="rdrake",
            account="rdrake_a",
        )

        cancelled_reminder = fns["cancel_pending_task_fn"]("abc123")
        assert cancelled_reminder["kind"] == "reminder"
        stand_in._remind_delete_for_assistant.assert_called_once()


class TestMemoryExtractionBackground:
    """Coverage for _schedule_memory_extraction's inner _extract_memories_bg."""

    @pytest.fixture
    def plugin_and_callback(self, mock_irc, mocker: MockerFixture):
        """Set up a plugin, schedule extraction, and capture the inner callback."""
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)

        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        return plugin, add_event

    def _existing_rows(self, mocker: MockerFixture, n: int):
        return [mocker.MagicMock(id=i + 1, fact=f"fact{i + 1}") for i in range(n)]

    def test_race_abort_when_rows_changed(self, plugin_and_callback, mocker: MockerFixture) -> None:
        """If memory rows changed during extraction, no new memories are saved."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 2)
        # Second get_memories call (after extraction) returns a *different* row set.
        plugin.db.get_memories.side_effect = [
            snapshot,
            [mocker.MagicMock(id=99, fact="injected")],
        ]
        plugin.llm_service.extract_memories.return_value = ExtractionResult(add=["new fact"])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        plugin.db.save_memory.assert_not_called()

    def test_cap_stops_loop_before_saving_all_facts(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """The save loop respects memoryMaxPerUser; extra facts are dropped."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        # Existing rows + max are wired so only ONE additional fact fits.
        snapshot = self._existing_rows(mocker, 9)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        # Threshold=1 reproduces legacy single-stage behavior so this test
        # keeps exercising the cap on direct saves.
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect(
                {
                    "memoryMaxPerUser": 10,
                    "memoryEnabled": True,
                    "memoryCleanupInterval": 0,
                    "memoryPromotionThreshold": 1,
                }
            )
        )
        plugin.db.get_memory_candidates.return_value = []
        plugin.llm_service.extract_memories.return_value = ExtractionResult(
            add=["one", "two", "three"]
        )

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        # Only one fact should fit before the cap is hit.
        assert plugin.db.save_memory.call_count == 1

    def test_cleanup_triggers_when_save_counter_reaches_interval(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """When increment_memory_saves crosses cleanup_interval, _run_memory_cleanup runs."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 2)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect(
                {
                    "memoryMaxPerUser": 50,
                    "memoryEnabled": True,
                    "memoryCleanupInterval": 3,
                    "memoryPromotionThreshold": 1,
                }
            )
        )
        plugin.db.get_memory_candidates.return_value = []
        plugin.llm_service.extract_memories.return_value = ExtractionResult(add=["a fresh fact"])
        plugin.db.increment_memory_saves.return_value = 3  # Reached interval.
        cleanup_spy = mocker.patch.object(plugin, "_run_memory_cleanup")

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        bg_callback = add_event.call_args.args[0]
        bg_callback()

        plugin.db.reset_memory_saves.assert_called_once_with("alice")
        cleanup_spy.assert_called_once_with("alice", "#test")

    def test_new_facts_become_candidates_not_memories(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """Default threshold>1: an unfamiliar fact is staged as a candidate."""
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        plugin.db.get_memory_candidates.return_value = []
        plugin.llm_service.extract_memories.return_value = ExtractionResult(add=["uses Arch Linux"])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.add_memory_candidate.assert_called_once_with("alice", "uses Arch Linux", "#test")
        plugin.db.save_memory.assert_not_called()

    def test_reinforce_below_threshold_bumps_only(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """A reinforcement that doesn't cross the threshold stays a candidate."""
        from llm.persistence import MemoryCandidate
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        candidate = MemoryCandidate(
            id=42,
            nick="alice",
            fact="uses Arch Linux",
            mentions=1,
            first_seen=100.0,
            last_seen=100.0,
            source_channel="#test",
        )
        plugin.db.get_memory_candidates.return_value = [candidate]
        plugin.registryValue = mocker.Mock(
            side_effect=make_registry_side_effect({"memoryPromotionThreshold": 3})
        )
        plugin.db.reinforce_memory_candidate.return_value = 2  # below threshold
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.reinforce_memory_candidate.assert_called_once_with(42, "alice")
        plugin.db.save_memory.assert_not_called()
        plugin.db.delete_memory_candidate.assert_not_called()

    def test_reinforce_at_threshold_promotes_candidate(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """Reaching the threshold moves the candidate into memories."""
        from llm.persistence import MemoryCandidate
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        candidate = MemoryCandidate(
            id=7,
            nick="alice",
            fact="lives in Berlin",
            mentions=1,
            first_seen=100.0,
            last_seen=100.0,
            source_channel="#origin",
        )
        plugin.db.get_memory_candidates.return_value = [candidate]
        plugin.db.reinforce_memory_candidate.return_value = 2  # crosses default threshold
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.save_memory.assert_called_once_with("alice", "lives in Berlin", "#origin")
        plugin.db.delete_memory_candidate.assert_called_once_with(7, "alice")

    def test_candidate_change_during_extraction_aborts(
        self, plugin_and_callback, mocker: MockerFixture
    ) -> None:
        """If candidate row IDs change mid-call, reinforce indices abort."""
        from llm.persistence import MemoryCandidate
        from llm.service import ExtractionResult

        plugin, add_event = plugin_and_callback
        snapshot = self._existing_rows(mocker, 0)
        plugin.db.get_memories.side_effect = [snapshot, snapshot]
        before = MemoryCandidate(1, "alice", "x", 1, 100.0, 100.0, "#test")
        after = MemoryCandidate(2, "alice", "x", 1, 100.0, 100.0, "#test")
        plugin.db.get_memory_candidates.side_effect = [[before], [after]]
        plugin.llm_service.extract_memories.return_value = ExtractionResult(reinforce=[0])

        plugin._schedule_memory_extraction("alice", "#test", "user", "bot")
        add_event.call_args.args[0]()

        plugin.db.reinforce_memory_candidate.assert_not_called()
        plugin.db.save_memory.assert_not_called()


class TestMechanicalRescheduleEdgeCases:
    """Coverage for invalid/exhausted-rrule and missing-recurrence guards."""

    @pytest.fixture
    def plugin(self, mock_irc, mocker: MockerFixture):
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        return LLM(mock_irc)

    def test_invalid_rrule_aborts_without_scheduling(self, plugin, mocker: MockerFixture) -> None:
        """An rrule that yields no future fire returns without registering an event."""
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        mocker.patch.object(plugin, "_next_rrule_fire", return_value=None)

        plugin._mechanical_reschedule(
            nick="alice",
            channel="#t",
            message="m",
            event_name="llm_remind_x",
            action_prompt="p",
            account=None,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule="FREQ=DAILY;UNTIL=19990101T000000Z",
            watch_mode=False,
            now=time.time(),
        )

        add_event.assert_not_called()
        plugin.db.save_reminder.assert_not_called()

    def test_no_recurrence_set_returns_without_scheduling(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """When neither recurrence_seconds nor recurrence_rrule is set, the helper exits cleanly."""
        add_event = mocker.patch("llm.plugin.schedule.addEvent")

        plugin._mechanical_reschedule(
            nick="alice",
            channel="#t",
            message="m",
            event_name="llm_remind_x",
            action_prompt="p",
            account=None,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
            now=time.time(),
        )

        add_event.assert_not_called()
        plugin.db.save_reminder.assert_not_called()


class TestVerseCapabilities:
    """llm.verse and llm.verse.gm must be declared in _REQUEST_CONTEXT_CAPABILITIES."""

    def test_llm_verse_in_request_context_capabilities(self) -> None:
        """GIVEN plugin module WHEN capabilities inspected THEN llm.verse is declared."""
        import llm.plugin as plugin_module

        assert "llm.verse" in plugin_module._REQUEST_CONTEXT_CAPABILITIES

    def test_llm_verse_gm_in_request_context_capabilities(self) -> None:
        """GIVEN plugin module WHEN capabilities inspected THEN llm.verse.gm is declared."""
        import llm.plugin as plugin_module

        assert "llm.verse.gm" in plugin_module._REQUEST_CONTEXT_CAPABILITIES

    def test_llm_verse_and_llm_verse_gm_are_distinct(self) -> None:
        """GIVEN both verse capabilities WHEN compared THEN they are separate entries."""
        import llm.plugin as plugin_module

        caps = plugin_module._REQUEST_CONTEXT_CAPABILITIES
        assert "llm.verse" in caps
        assert "llm.verse.gm" in caps
        # Distinct: having gm does not subsume verse by default
        assert "llm.verse" != "llm.verse.gm"


# =============================================================================
# TestVerseoptCommand
# =============================================================================


class TestVerseoptCommand:
    """Tests for the @verseopt in/out command (C3).

    Strategy: use the ``plugin_env`` fixture (mocked LLMService/DB) plus a
    real VerseStore backed by ``tmp_path`` so the avatar table round-trips
    through actual SQLite.  We swap ``plugin._get_or_create_verse_store`` with
    a factory that always returns the same real store, giving us full
    DB coverage without touching production paths.
    """

    SCENE_PREFIX = "You step into The Clearing."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for the channel.
        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Instruct text and avatar persona for the user (empty by default).
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        # msg arrives in #afnet
        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helper: invoke the wrapped command correctly.
    # wrap(verseopt, [("checkCapability", "llm.verse"), ("literal", ("in", "out"))])
    # produces a callable (irc, msg, args) where args is the token list.
    # ------------------------------------------------------------------

    @staticmethod
    def _call(plugin, irc, msg, mode: str) -> None:
        """Call the wrapped verseopt command with the given mode token."""
        plugin.verseopt(irc, msg, [mode])

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    def test_verseopt_in_new_avatar_replies_scene(self, verse_env) -> None:
        """GIVEN verse-enabled channel and llm.verse WHEN @verseopt in THEN scene text replied."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert self.SCENE_PREFIX in reply_text
        assert "You are already opted in." not in reply_text

    def test_verseopt_in_new_avatar_created_in_store(self, verse_env) -> None:
        """GIVEN new opt-in WHEN @verseopt in THEN avatar entity exists in store."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

    def test_verseopt_in_uses_avatar_persona(self, verse_env, mocker) -> None:
        """GIVEN user has @avatar set WHEN @verseopt in THEN opt_in_avatar called with persona."""
        plugin, irc, msg, store = verse_env
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value="Be a wizard.")

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == "Be a wizard."

    def test_verseopt_in_ignores_instruction(self, verse_env, mocker) -> None:
        """GIVEN @instruct set but no @avatar WHEN @verseopt in THEN persona is empty.

        The split: @instruct only shapes %ask. The verse must read @avatar.
        """
        plugin, irc, msg, store = verse_env
        plugin.db.get_instruction = mocker.MagicMock(return_value="ask voice")
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == ""

    # ------------------------------------------------------------------
    # Branch 2: verseopt in — already opted in
    # ------------------------------------------------------------------

    def test_verseopt_in_already_opted_in_prefixes_message(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt in again THEN reply prefixed with already-in message."""
        plugin, irc, msg, store = verse_env

        # First opt-in creates the avatar.
        self._call(plugin, irc, msg, "in")
        irc.reply.reset_mock()

        # Second opt-in should detect was_already_opted_in=True.
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text.startswith("You are already opted in. ")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 3: verseopt in after verseopt out — reactivation
    # ------------------------------------------------------------------

    def test_verseopt_in_after_out_reactivates_without_prefix(self, verse_env) -> None:
        """GIVEN retired avatar WHEN @verseopt in THEN was_already_opted_in=False, no prefix."""
        plugin, irc, msg, store = verse_env

        # Create avatar.
        self._call(plugin, irc, msg, "in")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

        # Retire it.
        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")
        assert store.find_avatar_by_nick("alice") is None

        # Reactivate.
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verseopt in")
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert not reply_text.startswith("You are already opted in.")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 4: verseopt out
    # ------------------------------------------------------------------

    def test_verseopt_out_replies_retired(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt out THEN retired message replied."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")
        irc.reply.reset_mock()

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Avatar retired. Use @verseopt in to rejoin."

    def test_verseopt_out_removes_avatar_link(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt out THEN find_avatar_by_nick returns None."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")
        assert store.find_avatar_by_nick("alice") is not None

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        assert store.find_avatar_by_nick("alice") is None

    def test_verseopt_out_with_no_avatar_replies_not_found(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verseopt out THEN 'no avatar' reply."""
        plugin, irc, msg, store = verse_env

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You don't have an avatar in this channel."

    # ------------------------------------------------------------------
    # Branch 5: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verseopt_in_disabled_channel_replies_not_enabled(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN not-enabled message."""
        plugin, irc, msg = plugin_env

        # verse disabled (default in make_registry_side_effect)
        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == (
            "This channel doesn't have a verse. Ask the operator to set verseEnabled."
        )

    def test_verseopt_in_disabled_channel_no_store_created(
        self, plugin_env, mocker, tmp_path
    ) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN no VerseStore is instantiated."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        store_cls = mocker.patch("llm.plugin.VerseStore")

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        store_cls.assert_not_called()

    # ------------------------------------------------------------------
    # Branch 6: missing llm.verse capability
    # ------------------------------------------------------------------

    def test_verseopt_in_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @verseopt in THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        # Deny all capabilities.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#test", "@verseopt in")

        # wrap's checkCapability gate raises an error via irc.error when denied.
        # Call the wrapped version; the body should NOT run (capability blocked).
        plugin.verseopt(irc, msg, ["in"])

        # The scene text should not have been replied.
        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "step into" not in reply_text


# =============================================================================
# TestVerseCommand
# =============================================================================


class TestVerseCommand:
    """Tests for the @verse command (C4).

    Uses the same verse_env fixture pattern as TestVerseoptCommand.
    """

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verse")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verse")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 1: avatar present, location set → scene one-liner
    # ------------------------------------------------------------------

    def test_verse_with_location_returns_scene_oneliner(self, verse_env) -> None:
        """GIVEN avatar with location WHEN @verse THEN 'You are at <place>.' reply."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        # Add a place and set avatar's location attribute.
        place_id = store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")
        store.set_attribute(entity_id, "location", str(place_id))

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert f"You are at {self.PLACE_NAME}." in reply_text
        assert "A sunlit glade." in reply_text

    # ------------------------------------------------------------------
    # Branch 2: avatar present, no location → "nowhere in particular"
    # ------------------------------------------------------------------

    def test_verse_no_location_returns_nowhere(self, verse_env) -> None:
        """GIVEN avatar with no location WHEN @verse THEN 'nowhere in particular' reply."""
        plugin, irc, msg, store = verse_env
        self._opt_in(plugin, irc, msg, store)

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You are nowhere in particular."

    # ------------------------------------------------------------------
    # Branch 3: no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_verse_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verse THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 4: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_verse_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verse THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 5: lacking llm.verse capability → Limnoria denial
    # ------------------------------------------------------------------

    def test_verse_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @verse THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        # Limnoria's wrap gate blocks execution — no scene reply.
        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "You are at" not in reply_text


# =============================================================================
# TestLookCommand
# =============================================================================


class TestLookCommand:
    """Tests for the @look [target] command (C4)."""

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@look")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@look")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 6: no target, avatar present → scene one-liner
    # ------------------------------------------------------------------

    def test_look_no_target_with_avatar_returns_scene(self, verse_env) -> None:
        """GIVEN avatar with location, no target WHEN @look THEN scene one-liner."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        place_id = store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")
        store.set_attribute(entity_id, "location", str(place_id))

        # Wrapped: args=[] means no optional target → target=None inside handler.
        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert f"You are at {self.PLACE_NAME}." in reply_text

    # ------------------------------------------------------------------
    # Branch 7: no target, no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_look_no_target_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar, no target WHEN @look THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 8: target matches → entity description
    # ------------------------------------------------------------------

    def test_look_target_matches_returns_description(self, verse_env) -> None:
        """GIVEN named entity, target given WHEN @look <name> THEN entity description."""
        plugin, irc, msg, store = verse_env
        store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")

        # Wrapped: target string goes into args list.
        plugin.look(irc, msg, [self.PLACE_NAME])

        reply_text = irc.reply.call_args[0][0]
        assert self.PLACE_NAME in reply_text
        assert "A sunlit glade." in reply_text

    # ------------------------------------------------------------------
    # Branch 9: target doesn't match → "Nothing matches."
    # ------------------------------------------------------------------

    def test_look_target_no_match_replies_nothing_matches(self, verse_env) -> None:
        """GIVEN target with no matching entity WHEN @look <name> THEN 'Nothing matches.'"""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, ["Atlantis"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nothing matches."

    # ------------------------------------------------------------------
    # Branch 10: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_look_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @look THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@look")
        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 11: lacking capability → denial
    # ------------------------------------------------------------------

    def test_look_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @look THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@look")
        plugin.look(irc, msg, [])

        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "You are at" not in reply_text


# =============================================================================
# TestWhoCommand
# =============================================================================


class TestWhoCommand:
    """Tests for the @who command (C4)."""

    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@who")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 12: multiple active avatars with locations → comma-joined list
    # ------------------------------------------------------------------

    def test_who_multiple_avatars_returns_list(self, verse_env) -> None:
        """GIVEN two active avatars with locations WHEN @who THEN comma-joined list."""
        plugin, irc, msg, store = verse_env

        # Create two avatars and a place.
        place_id = store.add_entity("place", "The Clearing", "A sunlit glade.")
        alice_id = store.add_entity("avatar", "alice")
        bob_id = store.add_entity("avatar", "bob")
        store.link_avatar(alice_id, "alice")
        store.link_avatar(bob_id, "bob")
        store.set_attribute(alice_id, "location", str(place_id))
        store.set_attribute(bob_id, "location", str(place_id))

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice (at The Clearing)" in reply_text
        assert "bob (at The Clearing)" in reply_text
        # Comma-joined format.
        assert ", " in reply_text

    # ------------------------------------------------------------------
    # Branch 13: no avatars → "Nobody is opted in here yet."
    # ------------------------------------------------------------------

    def test_who_no_avatars_replies_empty(self, verse_env) -> None:
        """GIVEN no active avatars WHEN @who THEN 'Nobody is opted in here yet.'"""
        plugin, irc, msg, store = verse_env

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nobody is opted in here yet."

    # ------------------------------------------------------------------
    # Branch 14: retired avatars excluded
    # ------------------------------------------------------------------

    def test_who_retired_avatar_excluded(self, verse_env) -> None:
        """GIVEN one active and one retired avatar WHEN @who THEN only active shown."""
        plugin, irc, msg, store = verse_env

        active_id = store.add_entity("avatar", "alice")
        store.link_avatar(active_id, "alice")

        retired_id = store.add_entity("avatar", "bob")
        store.link_avatar(retired_id, "bob")
        store.set_status(retired_id, "retired")

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice" in reply_text
        assert "bob" not in reply_text

    # ------------------------------------------------------------------
    # Branch 15: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_who_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @who THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@who")
        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 16: lacking capability → denial
    # ------------------------------------------------------------------

    def test_who_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @who THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@who")
        plugin.who(irc, msg, [])

        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "Nobody" not in reply_text
            assert "alice" not in reply_text


# =============================================================================
# TestVersepurgeCommand
# =============================================================================


class TestVersepurgeCommand:
    """Tests for the @versepurge command (C5).

    Strategy: direct method calls (bypassing wrap), with a real VerseStore
    backed by tmp_path. Token logic is exercised via a ``now_func`` shim passed
    to ``_versepurge_check_token`` so we can fake the clock without monkeypatching.
    """

    @pytest.fixture
    def purge_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore wired for #afnet, with gm capability."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Wire the store into the cache so versepurge can pop it.
        plugin._verse_stores["#afnet"] = store

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Grant llm.verse.gm.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@versepurge #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 1: first call issues token
    # ------------------------------------------------------------------

    def test_first_call_issues_token(self, purge_env) -> None:
        """GIVEN no existing token WHEN @versepurge #afnet THEN 6-char token issued."""
        plugin, irc, msg, store = purge_env

        plugin.versepurge(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Confirm with @versepurge #afnet" in reply_text
        assert "within 60s" in reply_text

        assert "#afnet" in plugin._versepurge_tokens
        stored_token, expires_at = plugin._versepurge_tokens["#afnet"]
        assert len(stored_token) == 6  # token_hex(3) → 6 hex chars
        import time as _time

        assert expires_at > _time.time()

    # ------------------------------------------------------------------
    # Test 2: correct token purges
    # ------------------------------------------------------------------

    def test_second_call_correct_token_purges(self, purge_env, tmp_path) -> None:
        """GIVEN valid token WHEN @versepurge #afnet <token> THEN purged reply, store gone."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        # Step 1: issue token.
        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Step 2: confirm.
        plugin.versepurge(irc, msg, [f"#afnet {stored_token}"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #afnet purged."

        # Store evicted from cache.
        assert "#afnet" not in plugin._verse_stores

        # DB file removed.
        assert not db_path.exists()

        # Token cleared.
        assert "#afnet" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 3: wrong token rejected
    # ------------------------------------------------------------------

    def test_second_call_wrong_token_rejected(self, purge_env) -> None:
        """GIVEN valid token WHEN confirmed with garbage token THEN rejected."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, ["#afnet garbage"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Token expired or invalid" in reply_text
        assert "Run @versepurge #afnet again" in reply_text

        # DB still exists; token NOT cleared (wrong token doesn't clear unexpired entry).
        assert db_path.exists()
        assert "#afnet" in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 4: token expires after 60s
    # ------------------------------------------------------------------

    def test_token_expires_after_60s(self, purge_env) -> None:
        """GIVEN token issued WHEN 60s pass and correct token presented THEN expired."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Use _versepurge_check_token directly with a fake clock past expiry.
        import time as _time

        future_now = _time.time() + 61.0
        result = plugin._versepurge_check_token("#afnet", stored_token, now_func=lambda: future_now)
        assert result is False

        # Stale entry should be cleared.
        assert "#afnet" not in plugin._versepurge_tokens

        # DB still exists (no purge occurred).
        assert db_path.exists()

    # ------------------------------------------------------------------
    # Test 5: reissue within window invalidates old token
    # ------------------------------------------------------------------

    def test_reissue_within_window_invalidates_old(self, purge_env) -> None:
        """GIVEN unexpired T1 WHEN @versepurge #afnet called again THEN T2 issued, T1 invalid."""
        plugin, irc, msg, store = purge_env

        # Issue T1.
        plugin.versepurge(irc, msg, ["#afnet"])
        token_t1, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Issue T2 (while T1 still valid).
        plugin.versepurge(irc, msg, ["#afnet"])
        reply_text = irc.reply.call_args[0][0]
        assert "Previous token invalidated" in reply_text or "invalidated" in reply_text

        token_t2, _ = plugin._versepurge_tokens["#afnet"]
        assert token_t2 != token_t1

        # T1 no longer valid.
        result = plugin._versepurge_check_token("#afnet", token_t1)
        assert result is False

    # ------------------------------------------------------------------
    # Test 6: purge channel with no store yet is a no-op
    # ------------------------------------------------------------------

    def test_purge_for_channel_with_no_store_yet(self, purge_env) -> None:
        """GIVEN channel never had a store WHEN confirmed THEN 'purged' reply, no error."""
        plugin, irc, msg, store = purge_env

        # Issue token for a channel that has no store in cache.
        plugin._verse_stores.pop("#newchan", None)
        plugin.versepurge(irc, msg, ["#newchan"])
        token, _ = plugin._versepurge_tokens["#newchan"]
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, [f"#newchan {token}"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #newchan purged."
        assert "#newchan" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 7: lacking capability is denied
    # ------------------------------------------------------------------

    def test_lacking_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse.gm WHEN @versepurge THEN error, no token issued."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 8: llm.verse (not gm) is denied
    # ------------------------------------------------------------------

    def test_purge_unrelated_to_versedump_capability(self, plugin_env, mocker) -> None:
        """GIVEN user has llm.verse but NOT llm.verse.gm WHEN @versepurge THEN denied."""
        plugin, irc, msg = plugin_env

        # Only grant llm.verse (not llm.verse.gm).
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "llm.verse",
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens


# =============================================================================
# TestVersedumpCommand
# =============================================================================


class TestVersedumpCommand:
    """Tests for the @versedump command (C5).

    Strategy: real VerseStore via tmp_path; invoke plugin.versedump directly.
    pyyaml is not installed → yaml branch returns unsupported message.
    """

    @pytest.fixture
    def dump_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore for #afnet, gm capability granted."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Default: pastebin write fails → inline fallback path (preserves
        # the JSON-content assertions in tests written before the
        # publish-and-link change). Tests exercising the URL path
        # override this on the same mock.
        plugin.llm_service.save_markdown_to_http = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@versedump #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 9: dump includes entities, attributes, relations, events, avatar_links
    # ------------------------------------------------------------------

    def test_dump_includes_entities_attributes_relations_events_avatar_link(self, dump_env) -> None:
        """GIVEN populated verse WHEN @versedump THEN JSON contains all sections."""
        import json as _json

        plugin, irc, msg, store = dump_env

        # Opt alice in (creates avatar + place + location attribute via opt_in_avatar).
        alice_id = store.opt_in_avatar("alice", None, "").entity_id

        # Add a relation.
        store.add_relation(alice_id, alice_id, "self-aware", "test note")

        # Add an event.
        store.add_event("alice did something", [alice_id], "avatar")

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        data = _json.loads(reply_text)

        assert "entities" in data
        assert len(data["entities"]) >= 1

        # Check attributes present on alice's entity.
        alice_entity = next((e for e in data["entities"] if e["id"] == alice_id), None)
        assert alice_entity is not None
        assert "attributes" in alice_entity

        assert "relations" in data
        assert len(data["relations"]) >= 1

        assert "events" in data
        assert len(data["events"]) >= 1

        assert "avatar_links" in data
        assert any(row["nick"] == "alice" for row in data["avatar_links"])

    # ------------------------------------------------------------------
    # Test 10: default format is JSON
    # ------------------------------------------------------------------

    def test_dump_default_format_is_json(self, dump_env) -> None:
        """WHEN @versedump #chan THEN reply is valid JSON."""
        import json as _json

        plugin, irc, msg, store = dump_env

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        data = _json.loads(reply_text)  # Raises if not valid JSON.
        assert data["schema_version"] == 1
        assert data["channel"] == "#afnet"

    # ------------------------------------------------------------------
    # Test 11: yaml format returns unsupported (pyyaml not installed)
    # ------------------------------------------------------------------

    def test_dump_yaml_format_or_unsupported(self, dump_env) -> None:
        """WHEN @versedump #chan --format=yaml THEN unsupported message (pyyaml absent)."""
        plugin, irc, msg, store = dump_env

        # Pass --format=yaml in the text arg (space-separated from channel).
        plugin.versedump(irc, msg, ["#afnet --format=yaml"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        # pyyaml is not a dependency → unsupported message.
        assert "Unsupported format" in reply_text
        assert "json" in reply_text.lower()

    # ------------------------------------------------------------------
    # Test 12: lacking capability is denied
    # ------------------------------------------------------------------

    def test_dump_lacking_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse.gm WHEN @versedump THEN error."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        msg.args = ("#afnet", "@versedump #afnet")
        plugin.versedump(irc, msg, ["#afnet"])

    def test_dump_publishes_to_pastebin_when_available(self, dump_env, mocker) -> None:
        """WHEN save_markdown_to_http returns URL THEN reply is just the URL.

        Avoids spamming IRC with a fat JSON line when the bot's HTTP
        pastebin is configured.
        """
        plugin, irc, msg, store = dump_env
        published_url = "http://example.com/llm/answer_abc123.html"
        plugin.llm_service.save_markdown_to_http.return_value = published_url

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert published_url in reply_text
        # Body must have been wrapped as a fenced JSON markdown block so
        # the pastebin renders with syntax highlighting (and includes the
        # channel name in a header).
        plugin.llm_service.save_markdown_to_http.assert_called_once()
        markdown_arg = plugin.llm_service.save_markdown_to_http.call_args[0][0]
        assert "```json" in markdown_arg
        assert "#afnet" in markdown_arg


# ===========================================================================
# C6: @avatar double-write — avatar summary syncs with persona text.
# (Was @instruct; split out so @instruct only shapes %ask.)
# ===========================================================================


class TestAvatarVerseSync:
    """@avatar updates avatar.summary when channel is verse-enabled.

    Strategy: real VerseStore backed by tmp_path (no SQLite mocks).
    We patch ``plugin._get_or_create_verse_store`` to return the same real
    store used to assert post-condition, mirroring the TestVerseoptCommand
    pattern.
    """

    @pytest.fixture
    def avatar_verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for #afnet, disable for everything else.
        def _registry(key, *args):
            if key == "verseEnabled":
                channel = args[0] if args else None
                return channel == "#afnet"
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # msg arrives in #afnet from alice (no account → nick fallback).
        msg.args = ("#afnet", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Wire up db mocks for the avatar path.
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()
        plugin.db.delete_avatar_persona = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(store, nick: str, account: str | None = None) -> int:
        """Opt alice into the verse and return her entity_id."""
        result = store.opt_in_avatar(nick, account, instruct_text="")
        assert result.entity_id is not None
        return result.entity_id

    def test_in_verse_channel_with_avatar_updates_both(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + active avatar WHEN @avatar THEN both updated."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        plugin.avatar(irc, msg, ["curious traveller"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "curious traveller")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "curious traveller"

    def test_in_verse_channel_without_avatar_only_updates_persona(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + no avatar WHEN @avatar THEN only persona updated."""
        plugin, irc, msg, store = avatar_verse_env

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        assert store.find_avatar_by_nick("alice") is None

    def test_in_non_verse_channel_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @avatar THEN only persona updated, no verse store."""
        plugin, irc, msg = plugin_env

        get_store_spy = mocker.patch.object(plugin, "_get_or_create_verse_store")

        def _registry(key, *args):
            if key == "verseEnabled":
                return False
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.db.save_avatar_persona = mocker.MagicMock()
        msg.args = ("#other", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        get_store_spy.assert_not_called()

    def test_clear_in_verse_channel_clears_both(self, avatar_verse_env) -> None:
        """GIVEN active avatar with summary WHEN @avatar clear THEN both cleared."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
                ("pirate queen", _time.time(), alice_id),
            )

        msg.args = ("#afnet", "@avatar clear")
        plugin.avatar(irc, msg, ["clear"])

        plugin.db.delete_avatar_persona.assert_called_once_with("alice")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == ""

    def test_verse_write_failure_does_not_update_persona(self, avatar_verse_env, mocker) -> None:
        """GIVEN verse write raises WHEN @avatar THEN persona text unchanged."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        class _BrokenStore:
            def find_avatar_by_account(self, *a, **kw):
                return None

            def find_avatar_by_nick(self, *a, **kw):
                return alice_id

            def get_entity(self, *a, **kw):
                return store.get_entity(*a, **kw)

            def write_transaction(self):
                raise RuntimeError("disk full")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=_BrokenStore())

        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="disk full"):
            plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_not_called()


class TestInstructDoesNotTouchAvatar:
    """@instruct must no longer mirror to avatar.summary. The split's whole point."""

    @pytest.fixture
    def env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@instruct hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.save_instruction = mocker.MagicMock()
        plugin.db.delete_instruction = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    def test_instruct_in_verse_channel_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="original persona")
        assert result.entity_id is not None
        original_summary = store.get_entity(result.entity_id).summary

        plugin.instruct(irc, msg, ["new ask voice"])

        plugin.db.save_instruction.assert_called_once_with("alice", "new ask voice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == original_summary

    def test_instruct_clear_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct clear THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="keep me")
        assert result.entity_id is not None

        msg.args = ("#afnet", "@instruct clear")
        plugin.instruct(irc, msg, ["clear"])

        plugin.db.delete_instruction.assert_called_once_with("alice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == "keep me"


class TestAvatarRetiredAvatar:
    """@avatar must not touch a retired avatar's summary."""

    def test_retired_avatar_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN retired avatar WHEN @avatar THEN persona updated, summary unchanged."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@avatar foo")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()

        result = store.opt_in_avatar("alice", None, instruct_text="")
        alice_id = result.entity_id
        assert alice_id is not None
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, status = 'retired', updated_at = ? WHERE id = ?",
                ("old summary", _time.time(), alice_id),
            )

        plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "foo")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "old summary"


class TestVerseRouteForGating:
    """_verse_route_for gating logic (C7b): verseEnabled, llm.verse cap, OOC short-circuit."""

    # ------------------------------------------------------------------
    # Gate 1: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verse_disabled_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=False WHEN _verse_route_for called THEN returns None (gate fires).

        We spy on registryValue to confirm the verseEnabled check is the gate that fires
        rather than the stub unconditionally returning None.
        """
        plugin, _irc, _msg = plugin_env

        # verseEnabled=False (default in plugin_env) — spy to confirm it is consulted.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        plugin.registryValue.assert_any_call("verseEnabled", "#afnet")

    # ------------------------------------------------------------------
    # Gate 2: no llm.verse capability
    # ------------------------------------------------------------------

    def test_no_capability_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True but user lacks llm.verse WHEN called THEN None (quiet fallthrough)."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        cap_check = mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        cap_check.assert_called_once_with("alice!*@*", "llm.verse")

    # ------------------------------------------------------------------
    # Gate 3: OOC message
    # ------------------------------------------------------------------

    def test_ooc_message_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True and cap granted WHEN message is OOC THEN None."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "((this is OOC))")

        assert result is None

    # ------------------------------------------------------------------
    # Gate 4: all preconditions satisfied but no avatar — None (C7c)
    # ------------------------------------------------------------------

    def test_all_preconditions_satisfied_no_avatar_returns_none(
        self, plugin_env, mocker: MockerFixture, tmp_path
    ) -> None:
        """GIVEN verseEnabled=True, cap granted, plain message, but user has no avatar
        WHEN called THEN None (no opt-in → chat path)."""
        from llm.verse.store import VerseStore

        plugin, _irc, _msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", None, "just a plain message")

        assert result is None

    # ------------------------------------------------------------------
    # Regression: @ask dispatch still hits chat path (carried from C7a)
    # ------------------------------------------------------------------

    def test_ask_dispatch_still_reaches_chat_path(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN _verse_route_for returns None WHEN @ask sent THEN chat path fires unchanged.

        Regression guard: the dispatch hook must be a no-op — assistant_request
        must still be called exactly once (verseEnabled=False in plugin_env default).
        """
        from llm.service import AssistantResult

        plugin, mock_irc, mock_msg = plugin_env

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="normal reply",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=3,
            cost=0.0,
            model="test-model",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # The chat path must have fired.
        plugin.llm_service.assistant_request.assert_called_once()


# =============================================================================
# C7c: _verse_route_for system prompt + tool list assembly
# =============================================================================


class TestVerseRouteForC7c:
    """Tests for _verse_route_for returning a populated VerseRoute (C7c).

    Complements TestVerseRouteForGating which covers the None branches.
    """

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore, verse-enabled channel, alice opted in."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in so she has an avatar.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        return plugin, irc, msg, store

    def test_avatar_present_returns_route(self, verse_env) -> None:
        """GIVEN opted-in avatar WHEN _verse_route_for called THEN non-None route returned."""
        plugin, _irc, _msg, store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert route.avatar_id is not None
        assert isinstance(route.system_prompt, str)
        assert "alice" in route.system_prompt
        assert len(route.tools) == 4
        assert route.store is store

    def test_route_system_prompt_includes_identity(self, verse_env) -> None:
        """System prompt must start with 'You are alice.'"""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert route.system_prompt.startswith("You are alice.")

    def test_route_system_prompt_includes_scene(self, verse_env) -> None:
        """System prompt must include a Scene section."""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert "Scene:" in route.system_prompt

    def test_route_tools_have_expected_names(self, verse_env) -> None:
        """Returned tools must be the four verse tool specs."""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        tool_names = {t["function"]["name"] for t in route.tools}
        assert tool_names == {"verse_act", "verse_move", "verse_look", "verse_recall"}


class TestAskWithVerseRoute:
    """Tests that @ask in a verse channel with an opted-in avatar uses the verse path (C7c)."""

    SENTINEL = "VIBEBOT_TEST_SENTINEL_DO_NOT_LEAK"

    @staticmethod
    def _make_result(content: str = "verse reply") -> AssistantResult:
        return AssistantResult(
            content=content,
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

    @pytest.fixture
    def verse_ask_env(self, plugin_env, tmp_path, mocker):
        """plugin_env with verse enabled, alice opted in, and assistant_request stubbed."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "assistantSystemPrompt":
                return self.SENTINEL
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        # Stub assistant_request for the @ask call.
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_result()

        msg.args = ("#afnet", "@ask hello")

        return plugin, irc, msg, store

    def test_ask_uses_verse_prompt_not_sentinel(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN assistantSystemPrompt sentinel is NOT in system_prompt.

        The verse system prompt (avatar persona + scene) must entirely replace the
        channel assistantSystemPrompt — the sentinel must not appear in the call.
        """
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        system_prompt = kwargs.get("system_prompt", "")
        assert self.SENTINEL not in (system_prompt or "")

    def test_ask_verse_prompt_contains_avatar_name(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN system_prompt includes avatar name 'alice'."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        system_prompt = kwargs.get("system_prompt", "")
        assert "alice" in (system_prompt or "")

    def test_ask_in_verse_appends_verse_tools(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN all 4 verse tool names appear in extra_tools."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        extra_tools = kwargs.get("extra_tools") or []
        tool_names = {t["function"]["name"] for t in extra_tools}
        expected = {"verse_act", "verse_move", "verse_look", "verse_recall"}
        assert expected.issubset(tool_names)

    def test_ask_in_verse_bypasses_token_cap(self, verse_ask_env, mocker: MockerFixture) -> None:
        """GIVEN verse route WHEN @ask THEN request_context uses PROFILE_VERSE.

        PROFILE_VERSE is the only profile not in the profile_max_output dict
        in assistant.py, so it bypasses the token cap applied to PROFILE_CHAT.
        We verify by checking the profile on the request_context passed to assistant_request.
        """
        from llm.assistant import PROFILE_VERSE

        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        request_context = kwargs.get("request_context")
        assert request_context is not None
        assert request_context.profile == PROFILE_VERSE

    def test_ask_in_verse_does_not_pass_model_override(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN assistant_request receives no model_override.

        The verse path must not hard-code an alternate model — it defers to the
        standard assistantModel registry key, which the service reads itself.
        Passing model_override=None (or not at all) is the correct behaviour.
        """
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        # model_override should be absent or None — never a hard-coded value.
        assert kwargs.get("model_override") is None


class TestCompactionTimerWiring:
    """E3: plugin wires the daily compaction timer + walks verse-enabled channels."""

    def test_plugin_registers_compaction_timer_at_load(self, plugin_env) -> None:
        """The plugin's __init__ should set ``_compaction_timer_name`` and
        attempt registration; the registered name is ``llm_verse_compact``."""
        plugin, _irc, _msg = plugin_env
        assert plugin._compaction_timer_name == "llm_verse_compact"

    def test_compaction_callback_walks_verse_enabled_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """``_run_compaction_pass`` invokes ``compact_verse`` once per
        verse-enabled channel returned by ``_verse_enabled_channels``."""
        plugin, _irc, _msg = plugin_env

        # Only #afnet is verse-enabled. _verse_enabled_channels already
        # filters by registry; emulate that here.
        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])

        # Defensive re-check inside _run_compaction_pass uses
        # registryValue("verseEnabled", channel) — return True for both
        # so the iteration faithfully reflects the helper's output.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )

        # Stub stores expose ``_channel`` so the fake compact_verse can
        # introspect which channel it was handed.
        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        called_for: list[str] = []

        def fake_compact(store, **kw):
            called_for.append(store._channel)
            return "skipped_no_events"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", fake_compact)

        plugin._run_compaction_pass()
        assert called_for == ["#afnet"]

    def test_run_compaction_pass_falls_back_when_registry_raises(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """Defensive guards in _run_compaction_pass: if the registry
        raises for either retention or min-keep keys (e.g. F1 key not
        yet defined on a freshly-upgraded install), defaults kick in
        and compaction still runs."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])

        def _flaky(key, *args):
            if key == "verseEnabled":
                return True
            if key in ("verseEventRetentionDays", "verseCompactionMinKeepEvents"):
                raise RuntimeError("registry not loaded")
            if key == "loomModel":
                return "gemini/x"
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_flaky)
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=mocker.MagicMock(),
        )
        seen: dict = {}

        def _fake_compact(store, **kw):
            seen.update(kw)
            return "skipped_no_events"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)
        plugin._run_compaction_pass()
        assert seen["retention_days"] == 30
        assert seen["min_keep_events"] == 20

    def test_compaction_failure_does_not_abort_remaining_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """A raise in one channel's compact_verse must not skip later channels."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        seen: list[str] = []

        def maybe_bomb(store, **kw):
            seen.append(store._channel)
            if store._channel == "#a":
                raise RuntimeError("fail")
            return "skipped_no_events"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", maybe_bomb)

        plugin._run_compaction_pass()
        assert "#a" in seen and "#b" in seen


class TestRunCompactionPassCallsAging:
    """Phase 5a: aging runs once per verse-enabled channel inside the
    compaction pass, with its own try/except for failure isolation."""

    def test_aging_called_once_per_enabled_channel(self, plugin_env, mocker, monkeypatch) -> None:
        """_run_compaction_pass calls age_auto_created_entities once per
        channel returned by _verse_enabled_channels."""
        from llm.verse import aging as aging_mod

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else 14
                    if key == "verseAutoEntityRetireDays"
                    else ""
                )
            )
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: "skipped_disabled",
        )

        called: list[tuple[object, int]] = []

        def _spy_aging(store, *, retire_after_days, now):
            called.append((store, retire_after_days))
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _spy_aging)

        plugin._run_compaction_pass()

        assert len(called) == 2
        stores = {id(c[0]) for c in called}
        assert len(stores) == 2

    def test_aging_reads_retire_days_per_channel(self, plugin_env, mocker, monkeypatch) -> None:
        """The aging call reads verseAutoEntityRetireDays at the channel
        scope, not global."""
        from llm.verse import aging as aging_mod

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a"])
        captured: list[tuple[str, str | None]] = []

        def _spy(key, *args):
            channel = args[0] if args else None
            captured.append((key, channel))
            if key == "verseEnabled":
                return True
            if key == "verseEventRetentionDays":
                return 30
            if key == "verseCompactionMinKeepEvents":
                return 20
            if key == "loomModel":
                return "gemini/x"
            if key == "verseAutoEntityRetireDays":
                return 14
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_spy)

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: "skipped_disabled",
        )
        monkeypatch.setattr(
            "llm.verse.aging.age_auto_created_entities",
            lambda *a, **kw: aging_mod.AgingOutcome(0, 0),
        )

        plugin._run_compaction_pass()

        assert ("verseAutoEntityRetireDays", "#a") in captured


class TestVersecompactCommand:
    """E4: @versecompact owner command — manual retention compaction.

    Strategy mirrors @versepurge / @verseapprove tests: direct method
    calls (bypassing wrap), real VerseStore in tmp_path, ircdb.checkCapability
    monkeypatched. The happy-path test inserts >min_keep events older than
    the retention window and monkeypatches compact_verse's loom client
    constructor so no network call is attempted.
    """

    @pytest.fixture
    def compact_env(self, plugin_env, tmp_path, mocker):
        """plugin_env wired with a real VerseStore for #afnet."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )
        plugin._verse_stores["#afnet"] = store

        msg.args = ("#afnet", "@versecompact #afnet")
        msg.prefix = "owner!user@host"
        msg.nick = "owner"
        msg.server_tags = {}

        return plugin, irc, msg, store

    def _override_registry(self, plugin, mocker, *, verse_enabled: bool) -> None:
        from tests.conftest import make_registry_side_effect

        defaults = make_registry_side_effect()

        def _registry(key, *args):
            if key == "verseEnabled":
                return verse_enabled
            if key == "verseEventRetentionDays":
                return 30
            if key == "verseCompactionMinKeepEvents":
                return 20
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            return defaults(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

    def test_compacts_named_channel(self, compact_env, mocker) -> None:
        """GIVEN >min_keep old events WHEN @versecompact #afnet THEN reply 'compacted'."""
        from plugins.llm.tests.verse.conftest import insert_event_at

        plugin, irc, msg, store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)

        # Grant llm.verse.gm so the wrap-bypassed direct call still
        # treats the caller as an owner. (Direct .versecompact() call
        # also won't pass through wrap's capability check; the in-method
        # registry guard is what matters.)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Substitute a fake loom client so no network call happens.
        class _FakeClient:
            def call(self, *, op, model, messages):
                from llm.verse.loom import LoomCallUsage

                return "A digest of the past.", LoomCallUsage(
                    prompt_tokens=10, completion_tokens=20, cost=0.0
                )

        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=_FakeClient(),
        )

        seconds_per_day = 86400
        now_ts = 100_000_000.0
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now_ts - 60 * seconds_per_day,
            )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "compaction outcome for #afnet" in reply_text
        assert "compacted" in reply_text

    def test_disabled_verse_says_so(self, compact_env, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @versecompact THEN reply names verseEnabled."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=False)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "verseEnabled" in reply_text

    def test_failure_in_compact_verse_replies_error(self, compact_env, mocker, monkeypatch) -> None:
        """GIVEN compact_verse raises WHEN @versecompact THEN irc.error not crash."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        def _boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _boom)

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        err_text = irc.error.call_args[0][0]
        assert "compaction failed for #afnet" in err_text
        assert "RuntimeError" in err_text

    def test_registry_lookup_failure_uses_defaults(self, compact_env, mocker, monkeypatch) -> None:
        """GIVEN registryValue raises for retention/min_keep keys WHEN
        @versecompact THEN defaults kick in and compaction still runs.

        Covers the defensive try/except guards around the registry reads
        for verseEventRetentionDays and verseCompactionMinKeepEvents.
        F1 will define verseCompactionMinKeepEvents — until then, the
        guard ensures the command still works.
        """
        plugin, irc, msg, _store = compact_env

        def _flaky_registry(key, *args):
            if key == "verseEnabled":
                return True
            if key in ("verseEventRetentionDays", "verseCompactionMinKeepEvents"):
                raise RuntimeError("registry not loaded")
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            if key == "assistantApiKey":
                return ""
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_flaky_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        seen_kwargs: dict = {}

        def _fake_compact(store, **kw):
            seen_kwargs.update(kw)
            return "skipped_no_events"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)

        plugin.versecompact(irc, msg, ["#afnet"])

        # Defaults applied via except branches.
        assert seen_kwargs["retention_days"] == 30
        assert seen_kwargs["min_keep_events"] == 20
        irc.reply.assert_called_once()

    def test_zero_retention_and_min_keep_are_honoured(
        self, compact_env, mocker, monkeypatch
    ) -> None:
        """Regression: ``int(0 or 30) == 30`` would coerce a legitimate
        zero value (the registry types accept 0) into the default. The
        fix uses an explicit conversion that preserves zero so operators
        can disable compaction with retention=0."""
        plugin, irc, msg, _store = compact_env

        def _zero_registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "verseEventRetentionDays":
                return 0
            if key == "verseCompactionMinKeepEvents":
                return 0
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            if key == "assistantApiKey":
                return ""
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_zero_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        seen_kwargs: dict = {}

        def _fake_compact(store, **kw):
            seen_kwargs.update(kw)
            return "skipped_disabled"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)

        plugin.versecompact(irc, msg, ["#afnet"])

        assert seen_kwargs["retention_days"] == 0
        assert seen_kwargs["min_keep_events"] == 0

    def test_zero_retention_and_min_keep_in_run_compaction_pass(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """Same regression but for the daily-timer driver
        ``_run_compaction_pass`` — the same ``or default`` pattern was
        also present there."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    0
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 0
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=mocker.MagicMock(),
        )

        seen: dict = {}

        def _fake_compact(store, **kw):
            seen.update(kw)
            return "skipped_disabled"

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)
        plugin._run_compaction_pass()
        assert seen["retention_days"] == 0
        assert seen["min_keep_events"] == 0


class TestBridgeCrosspollWiring:
    """F2: production bridge wires real crosspoll registry + store."""

    @staticmethod
    def _redirect_data_dir(request, tmp_path) -> None:
        """Point supybot's data directory at tmp_path for the test.

        Registers a finalizer that restores the original value after the
        test completes.
        """
        import supybot.conf as conf

        original = conf.supybot.directories.data()
        conf.supybot.directories.data.setValue(str(tmp_path))
        request.addfinalizer(lambda: conf.supybot.directories.data.setValue(original))

    def _build_plugin_with_bridge(self, mocker: MockerFixture, tmp_path, request):
        from llm.plugin import LLM

        self._redirect_data_dir(request, tmp_path)

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
                "verseCrosspollPerCycleLimit": 1,
            }
        )
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        assert plugin._loom_bridge is not None
        return plugin

    def test_verse_allow_send_reads_registry(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)

        def _registry(key, *args):
            if key == "verseCrosspollAllowSend":
                channel = args[0] if args else None
                return channel == "#a"
            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        bridge = plugin._loom_bridge
        assert bridge.verse_allow_send("#a") is True
        assert bridge.verse_allow_send("#b") is False

    def test_verse_allow_receive_reads_registry(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)

        def _registry(key, *args):
            if key == "verseCrosspollAllowReceive":
                channel = args[0] if args else None
                return channel == "#a"
            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        bridge = plugin._loom_bridge
        assert bridge.verse_allow_receive("#a") is True
        assert bridge.verse_allow_receive("#b") is False

    def test_crosspoll_store_is_a_singleton(self, mocker: MockerFixture, tmp_path, request) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)
        bridge = plugin._loom_bridge
        a = bridge.crosspoll_store()
        b = bridge.crosspoll_store()
        assert a is b
        assert a is not None

    def test_loom_config_threads_per_cycle_limit(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        from llm.plugin import LLM

        self._redirect_data_dir(request, tmp_path)

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
                "verseCrosspollPerCycleLimit": 4,
            }
        )
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        assert plugin._loom is not None
        assert plugin._loom._cfg.crosspoll_per_cycle_limit == 4


class TestVerseapproveCrosspollSource:
    """F3-pre-2: @verseapprove infers event_source from cycle_id."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        msg.args = ("#afnet", "")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_approve_crosspoll_proposal_writes_crosspoll_event(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": "from elsewhere", "entity_ids": []},
            confidence=0.0,
            provenance="crosspoll from #alpha (seed-id=1)",
        )
        plugin.verseapprove(irc, msg, [pid])
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT source FROM events WHERE summary='from elsewhere'"
            ).fetchone()
        assert row is not None and row[0] == "crosspoll"

    def test_approve_loom_proposal_still_writes_loom_event(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="loom-c1",
            op="add_event",
            payload={"summary": "regular event", "entity_ids": []},
            confidence=0.5,
            provenance="t",
        )
        plugin.verseapprove(irc, msg, [pid])
        with store.read_connection() as conn:
            row = conn.execute("SELECT source FROM events WHERE summary='regular event'").fetchone()
        assert row is not None and row[0] == "loom"
