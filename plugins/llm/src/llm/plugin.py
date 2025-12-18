"""LLM plugin implementation."""

from __future__ import annotations

import contextlib
import mimetypes
import os
import time
from typing import TYPE_CHECKING

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.httpserver as httpserver
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.schedule as schedule
from supybot.commands import wrap
from supybot.i18n import PluginInternationalization

from .context import ContextConfig, ConversationContext
from .service import LLMService

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

_ = PluginInternationalization("LLM")


class LLMHTTPCallback(httpserver.SupyHTTPServerCallback):
    """HTTP callback to serve LLM-generated files (images, code)."""

    name = "LLM"
    public = True

    def __init__(self, plugin: LLM) -> None:
        """Initialize with plugin reference."""
        super().__init__()
        self._plugin = plugin

    def _get_web_dir(self) -> str:
        """Get the web directory for LLM files."""
        http_root = self._plugin.registryValue("httpRoot")
        if http_root:
            return http_root
        return conf.supybot.directories.data.web.dirize("llm")

    def doGet(self, handler: httpserver.RequestHandler, path: str) -> None:  # noqa: N802
        """Serve static files from LLM web directory."""
        # Remove leading slash
        path = path.lstrip("/")

        # Security: prevent directory traversal
        if ".." in path or path.startswith("/"):
            handler.send_response(403)
            handler.end_headers()
            return

        web_dir = self._get_web_dir()
        filepath = os.path.join(web_dir, path)

        # Check file exists and is within web_dir
        if not os.path.isfile(filepath):
            handler.send_response(404)
            handler.end_headers()
            return

        # Determine content type
        content_type, _ = mimetypes.guess_type(filepath)
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            with open(filepath, "rb") as f:
                content = f.read()

            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(content)))
            handler.end_headers()
            handler.wfile.write(content)
        except OSError:
            handler.send_response(500)
            handler.end_headers()


class LLM(callbacks.Plugin):
    """AI-powered commands using LiteLLM.

    Provides ask, code, draw commands with conversation context
    and multi-provider support.
    """

    threaded = True  # Commands run in threads for non-blocking I/O

    def __init__(self, irc: callbacks.Irc) -> None:
        """Initialize plugin.

        Args:
            irc: IRC connection instance
        """
        super().__init__(irc)
        self.llm_service = LLMService(self)
        self.log = log.getPluginLogger("LLM")
        self.startup_time = time.time()  # Track startup for ZNC playback filtering

        # Initialize conversation context
        self._update_context()

        # Only register HTTP callback if using Limnoria's built-in web directory
        # (i.e., httpRoot is not configured). When httpRoot is set, an external
        # web server (e.g., nginx) is expected to serve files from that path.
        if not self.registryValue("httpRoot"):
            self._http_callback = LLMHTTPCallback(self)
            httpserver.hook("llm", self._http_callback)
        else:
            self._http_callback = None

        # Schedule periodic file cleanup (runs every hour)
        # Defensive: remove any existing event first (handles plugin reloads)
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_file_cleanup")

        schedule.addPeriodicEvent(
            self._run_file_cleanup,
            3600,  # 1 hour in seconds
            name="llm_file_cleanup",
            now=False,  # Don't run immediately on startup
        )

    def die(self) -> None:
        """Clean up when plugin is unloaded."""
        # Remove scheduled cleanup event
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_file_cleanup")

        # Only unhook HTTP callback if we registered
        if self._http_callback is not None:
            httpserver.unhook("llm")
        super().die()

    def _run_file_cleanup(self) -> None:
        """Scheduled cleanup of old generated files."""
        try:
            http_root, _ = self.llm_service._get_http_paths()
            self.llm_service._cleanup_old_files(http_root)
            self.log.debug("Scheduled file cleanup completed")
        except Exception as e:
            self.log.error(f"Scheduled file cleanup failed: {e}")

    def doPrivmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Monitor channel messages for enhanced context (opt-in feature).

        When contextTrackAllMessages is enabled, this captures all channel
        messages to provide richer context for the ask command.

        Note: Disabled by default for privacy since messages are sent to
        third-party LLM providers.
        """
        # Only track if enabled (disabled by default for privacy)
        channel = msg.channel
        if not channel:
            return  # Skip private messages

        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        if not self.registryValue("contextTrackAllMessages", channel):
            return

        # Skip CTCP messages (except ACTION)
        if ircmsgs.isCtcp(msg) and not ircmsgs.isAction(msg):
            return

        # Skip bot's own messages
        if ircutils.strEqual(irc.nick, msg.nick):
            return

        # Skip if context is disabled
        if not self.registryValue("contextEnabled", channel):
            return

        nick = self._get_nick(msg)
        message_text = msg.args[1] if len(msg.args) > 1 else ""

        # Store in conversation context for richer follow-up questions
        self.context.add_message(nick, channel, "user", message_text)

    def _update_context(self) -> None:
        """Update context manager from current config."""
        config = ContextConfig(
            max_messages=self.registryValue("contextMaxMessages"),
            timeout_minutes=self.registryValue("contextTimeoutMinutes"),
            enabled=self.registryValue("contextEnabled"),
        )
        self.context = ConversationContext(config)

    def _get_nick(self, msg: IrcMsg) -> str:
        """Extract nick from IRC message.

        Args:
            msg: IRC message

        Returns:
            User's nick
        """
        return ircutils.nickFromHostmask(msg.prefix)

    def _get_channel(self, msg: IrcMsg) -> str:
        """Extract channel from IRC message.

        Args:
            msg: IRC message

        Returns:
            Channel name
        """
        return msg.args[0] if msg.args else "unknown"

    def _is_old_message(self, msg: IrcMsg) -> bool:
        """Check if message predates bot startup (ZNC playback).

        Args:
            msg: IRC message

        Returns:
            True if message is older than bot startup time
        """
        return msg.time < self.startup_time

    def ask(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<question>

        Ask the AI a question. Supports conversation context (follow-up questions)
        and vision (include image URLs in your question).

        Examples:
          %ask What is the capital of France?
          %ask Describe this: https://example.com/image.jpg
          %ask And what about Germany?  (follow-up using context)
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        try:
            nick = self._get_nick(msg)
            channel = self._get_channel(msg)

            # Detect images for vision
            images = self.llm_service.detect_images(text)

            # Get conversation history
            history = self.context.get_messages(nick, channel)

            if images:
                # Clean prompt by removing image URLs
                clean_prompt = text
                for img in images:
                    clean_prompt = clean_prompt.replace(img, "").strip()

                irc.reply(_("Processing with %d image(s)...") % len(images), prefixNick=False)
                response = self.llm_service.completion(
                    clean_prompt, command="ask", images=images, history=history, irc=irc, msg=msg
                )
            else:
                response = self.llm_service.completion(
                    text, command="ask", history=history, irc=irc, msg=msg
                )

            # Reply first, then store context (so user gets response even if context fails)
            irc.reply(response, prefixNick=False)

            # Store conversation in context
            self.context.add_message(nick, channel, "user", text)
            self.context.add_message(nick, channel, "assistant", response)
        except Exception:
            # Only log - completion() already returns error strings transparently
            self.log.exception("Error in ask command")

    ask = wrap(ask, [("checkCapability", "llm.ask"), "text"])

    def code(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<request>

        Generate code based on your request. Long code is saved to HTTP link.
        Supports conversation context for iterating on code.

        Examples:
          %code Python function to calculate fibonacci numbers
          %code Now add memoization to that
          %code JavaScript async fetch with error handling
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        try:
            nick = self._get_nick(msg)
            channel = self._get_channel(msg)

            # Get conversation history for iterating on code
            history = self.context.get_messages(nick, channel)

            response = self.llm_service.completion(
                text, command="code", history=history, irc=irc, msg=msg
            )

            # Reply first, then store context
            lines = response.count("\n")
            url = self.llm_service.save_code_to_http(response)
            if url:
                irc.reply(_("Code generated (%d lines): %s") % (lines, url), prefixNick=False)
            else:
                # Fallback to IRC paging if save failed
                irc.reply(response, prefixNick=False)

            # Store conversation in context
            self.context.add_message(nick, channel, "user", text)
            self.context.add_message(nick, channel, "assistant", response)
        except Exception:
            # Only log - completion() already returns error strings transparently
            self.log.exception("Error in code command")

    code = wrap(code, [("checkCapability", "llm.code"), "text"])

    def draw(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<prompt>

        Generate an image from a text description.
        This command does NOT use conversation context.

        Examples:
          %draw A sunset over mountains in watercolor style
          %draw Cyberpunk city with neon lights
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        try:
            # Typing indicator sent by service - no "Generating..." message needed
            result = self.llm_service.image_generation(text, irc=irc, msg=msg)
            irc.reply(result, prefixNick=False)
        except Exception:
            # Only log - image_generation() already returns error strings transparently
            self.log.exception("Error in draw command")

    draw = wrap(draw, [("checkCapability", "llm.draw"), "text"])

    def forget(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str,
    ) -> None:
        """[<channel>]

        Clear your conversation context (memory) for the current or specified channel.
        Use this to start fresh.
        """
        try:
            nick = self._get_nick(msg)

            cleared = self.context.clear(nick, channel)

            if cleared:
                irc.reply(_("Conversation context cleared. Starting fresh!"), prefixNick=False)
            else:
                irc.reply(_("No conversation context to clear."), prefixNick=False)
        except Exception:
            self.log.exception("Error in forget command")
            irc.reply(_("Error clearing context."), prefixNick=False)

    forget = wrap(forget, ["channel"])

    def llmkeys(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        Check API key configuration status (admin only). Shows first 3 characters only.

        This is a diagnostic command to verify keys are configured without exposing them.
        """
        try:
            # Get all API keys
            ask_key = self.registryValue("askApiKey")
            code_key = self.registryValue("codeApiKey")
            draw_key = self.registryValue("drawApiKey")

            # Safely display each key
            ask_status = self.llm_service.safe_key_display(ask_key)
            code_status = self.llm_service.safe_key_display(code_key)
            draw_status = self.llm_service.safe_key_display(draw_key)

            # Build response
            response = _("API Key Status: ask=%s, code=%s, draw=%s") % (
                ask_status,
                code_status,
                draw_status,
            )

            # Send as private message for extra security
            irc.reply(response, private=True)
        except Exception:
            self.log.exception("Error checking API keys")
            irc.reply(_("Error checking API key status."), private=True)

    llmkeys = wrap(llmkeys, ["admin"])


Class = LLM
