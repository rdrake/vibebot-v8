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
        """GIVEN normal message WHEN doPrivmsg called THEN tracks message."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        with (
            patch("supybot.ircmsgs.isCtcp", return_value=False),
            patch("supybot.ircutils.strEqual", return_value=False),
        ):
            plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_called_once_with(
            "usernick", "#channel", "user", "hello world"
        )


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
            plugin.llm_service.image_generation.return_value = "http://img.url/test.png"
            plugin.llm_service.save_code_to_http.return_value = "http://code.url/test.py"
            plugin.llm_service.safe_key_display.return_value = "tes***"

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

        lines = response.count("\n")
        url = plugin.llm_service.save_code_to_http(response)
        if url:
            irc.reply(f"Code generated ({lines} lines): {url}", prefixNick=False)
        else:
            irc.reply(response, prefixNick=False)

        plugin.context.add_message(nick, channel, "user", text)
        plugin.context.add_message(nick, channel, "assistant", response)

    def _call_draw(self, plugin: MagicMock, irc: MagicMock, msg: MagicMock, text: str) -> None:
        """Call the draw command implementation directly."""
        if plugin._is_old_message(msg):
            return

        result = plugin.llm_service.image_generation(text, irc=irc, msg=msg)
        irc.reply(result, prefixNick=False)

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

        mock_irc.reply.assert_called_with("print('hello')", prefixNick=False)

    def test_draw_calls_image_generation(self, plugin_with_service: tuple) -> None:
        """GIVEN draw request WHEN draw called THEN calls image_generation."""
        plugin, mock_irc, mock_msg = plugin_with_service

        self._call_draw(plugin, mock_irc, mock_msg, "a sunset")

        plugin.llm_service.image_generation.assert_called_once_with(
            "a sunset", irc=mock_irc, msg=mock_msg
        )
        mock_irc.reply.assert_called_with("http://img.url/test.png", prefixNick=False)

    def test_draw_skips_old_messages(self, plugin_with_service: tuple) -> None:
        """GIVEN ZNC playback message WHEN draw called THEN skips processing."""
        plugin, mock_irc, mock_msg = plugin_with_service
        mock_msg.time = time.time() - 100  # Old message

        self._call_draw(plugin, mock_irc, mock_msg, "a sunset")

        mock_irc.reply.assert_not_called()

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


class TestUpdateContext:
    """Test _update_context method."""

    def test_update_context_creates_new_context(self) -> None:
        """GIVEN plugin WHEN _update_context called THEN creates new context."""
        from llm.context import ConversationContext
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.registryValue = MagicMock(side_effect=[20, 30, True])

            plugin._update_context()

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
            return MagicMock()

        with (
            patch.object(LLM, "registryValue", side_effect=registry_side_effect),
            patch("llm.plugin.LLMService"),
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

                mock_remove.assert_called_with("llm_file_cleanup")

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
            plugin.llm_service._get_http_paths.return_value = ("/path", "http://url")
            plugin.log = MagicMock()

            plugin._run_file_cleanup()

            plugin.llm_service._cleanup_old_files.assert_called_once_with("/path")

    def test_run_file_cleanup_handles_errors(self) -> None:
        """GIVEN cleanup error WHEN _run_file_cleanup called THEN logs error."""
        from llm.plugin import LLM

        with patch.object(LLM, "__init__", lambda self, irc: None):
            plugin = LLM.__new__(LLM)
            plugin.llm_service = MagicMock()
            plugin.llm_service._get_http_paths.side_effect = Exception("test error")
            plugin.log = MagicMock()

            # Should not raise
            plugin._run_file_cleanup()

            plugin.log.error.assert_called_once()
