"""LLM plugin implementation."""

from __future__ import annotations

import contextlib
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.httpserver as httpserver
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.schedule as schedule
from supybot.commands import optional, wrap
from supybot.i18n import PluginInternationalization

from .context import ContextConfig, ConversationContext, Role
from .service import CODE_PREVIEW_MAX_LEN, CODE_PREVIEW_TRUNCATE_LEN, LLMService

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

_ = PluginInternationalization("LLM")

# Icon shown when Google grounding/search was used in the response
GROUNDING_ICON = "\U0001f310"  # 🌐 (globe with meridians)


HELP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Bot Commands</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    padding: 20px;
    background: #272822;
    color: #f8f8f2;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    line-height: 1.6;
    max-width: 800px;
    margin: 0 auto;
}
h1 { color: #f8f8f2; margin-bottom: 0.5em; }
h2 { color: #a6e22e; margin-top: 1.5em; border-bottom: 1px solid #49483e; padding-bottom: 0.3em; }
h3 { color: #66d9ef; margin-top: 1.2em; }
code {
    font-family: 'SF Mono', 'Fira Code', Consolas, 'Liberation Mono', monospace;
    font-size: 14px;
    background: #1e1e1e;
    padding: 2px 6px;
    border-radius: 3px;
}
pre {
    background: #1e1e1e;
    padding: 16px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1em 0;
}
pre code { padding: 0; background: none; }
.command { color: #e6db74; font-weight: bold; }
.param { color: #fd971f; }
.example { color: #75715e; font-style: italic; }
ul { margin: 0.5em 0; padding-left: 1.5em; }
li { margin: 0.3em 0; }
a { color: #66d9ef; }
.note {
    background: #3e3d32;
    border-left: 3px solid #a6e22e;
    padding: 10px 15px;
    margin: 1em 0;
    border-radius: 0 6px 6px 0;
}
@media (max-width: 600px) {
    body { padding: 15px; }
    pre { padding: 12px; font-size: 13px; }
}
</style>
</head>
<body>
<h1>LLM Bot Commands</h1>
<p>AI-powered IRC bot commands using LiteLLM.</p>

<h2>Commands</h2>

<h3><code class="command">%ask</code> <span class="param">&lt;question&gt;</span></h3>
<p>Ask the AI a question. Supports conversation context (follow-up questions) and vision (include image URLs).</p>
<pre><code><span class="example">%ask What is the capital of France?</span>
<span class="example">%ask Describe this: https://example.com/image.jpg</span>
<span class="example">%ask And what about Germany?</span>  <span class="example">(follow-up using context)</span></code></pre>

<h3><code class="command">%code</code> <span class="param">&lt;request&gt;</span></h3>
<p>Generate code based on your request. Code is saved to an HTTP link with syntax highlighting.</p>
<pre><code><span class="example">%code Python function to calculate fibonacci numbers</span>
<span class="example">%code Now add memoization to that</span>
<span class="example">%code JavaScript async fetch with error handling</span></code></pre>

<h3><code class="command">%draw</code> <span class="param">&lt;prompt&gt;</span></h3>
<p>Generate an image from a text description.</p>
<pre><code><span class="example">%draw A sunset over mountains in watercolor style</span>
<span class="example">%draw A cyberpunk cityscape at night</span></code></pre>

<h3><code class="command">%forget</code> <span class="param">[channel]</span></h3>
<p>Clear your conversation context (memory) for the current or specified channel. Use this to start fresh.</p>
<pre><code><span class="example">%forget</span>
<span class="example">%forget #channel</span></code></pre>

<h2>Features</h2>
<ul>
<li><strong>Conversation Context</strong> &ndash; The bot remembers recent exchanges for natural follow-up questions</li>
<li><strong>Vision Support</strong> &ndash; Include image URLs in <code>%ask</code> for image analysis</li>
<li><strong>Syntax Highlighting</strong> &ndash; Generated code is displayed with full highlighting</li>
<li><strong>Multi-Provider</strong> &ndash; Supports various AI providers via LiteLLM</li>
</ul>

<h2>Configuration</h2>
<div class="note">
Configuration is managed by the bot operator via Limnoria's config system.
Commands require the appropriate capability (e.g., <code>llm.ask</code>).
</div>

<p>Key settings include:</p>
<ul>
<li><strong>Model selection</strong> &ndash; Different models for ask/code/draw commands</li>
<li><strong>System prompts</strong> &ndash; Customize bot personality per command</li>
<li><strong>Context settings</strong> &ndash; Configure conversation memory limits</li>
</ul>

</body>
</html>"""


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

    def _serve_help_page(self, handler: httpserver.RequestHandler) -> None:
        """Serve the help documentation page.

        Serves custom help.html from web dir if it exists,
        otherwise falls back to built-in HELP_HTML_TEMPLATE.
        """
        web_dir = Path(self._get_web_dir())
        custom_help = web_dir / "help.html"

        # Try custom help.html first
        if custom_help.is_file():
            try:
                content = custom_help.read_bytes()
            except OSError:
                content = HELP_HTML_TEMPLATE.encode("utf-8")
        else:
            content = HELP_HTML_TEMPLATE.encode("utf-8")

        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/html; charset=utf-8")
            handler.send_header("Content-Length", str(len(content)))
            handler.end_headers()
            handler.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected

    def doGet(self, handler: httpserver.RequestHandler, path: str) -> None:  # noqa: N802
        """Serve static files from LLM web directory."""
        # Remove leading slash
        path = path.lstrip("/")

        # Serve help page at root
        if path == "":
            self._serve_help_page(handler)
            return

        # Security: prevent directory traversal (early check before path operations)
        if ".." in path or path.startswith("/"):
            handler.send_response(403)
            handler.end_headers()
            return

        web_dir = Path(self._get_web_dir())
        filepath = web_dir / path

        # Security: resolve symlinks and verify path is under web root
        try:
            resolved_web_dir = web_dir.resolve()
            resolved_filepath = filepath.resolve()

            # Ensure resolved path is under web directory (Python 3.9+)
            if not resolved_filepath.is_relative_to(resolved_web_dir):
                handler.send_response(403)
                handler.end_headers()
                return
        except (OSError, ValueError):
            handler.send_response(403)
            handler.end_headers()
            return

        # Check file exists
        if not resolved_filepath.is_file():
            handler.send_response(404)
            handler.end_headers()
            return

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(resolved_filepath))
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            with open(resolved_filepath, "rb") as f:
                content = f.read()

            handler.send_response(200)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(content)))
            handler.end_headers()
            handler.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected - this is normal, ignore silently
            pass
        except OSError:
            try:
                handler.send_response(500)
                handler.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass


class LLM(callbacks.Plugin):
    """AI-powered commands using LiteLLM.

    Provides ask, code, draw commands with multi-provider support.
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
        self._init_context()

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
            self.llm_service.run_scheduled_cleanup()
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
        self.context.add_message(nick, channel, Role.USER, message_text)
        self.context.add_channel_message(channel, nick, Role.USER, message_text)

    def _init_context(self) -> None:
        """Initialize context manager from current config (called once at startup)."""
        config = ContextConfig(
            max_messages=self.registryValue("contextMaxMessages"),
            timeout_minutes=self.registryValue("contextTimeoutMinutes"),
            enabled=self.registryValue("contextEnabled"),
            channel_max_messages=self.registryValue("channelContextMaxMessages"),
        )
        self.context = ConversationContext(config)

    def _get_help_url(self) -> str:
        """Get the URL to the web help documentation.

        Uses same URL construction pattern as service._get_http_paths().

        Returns:
            Full URL to help page (e.g., http://localhost:8080/llm/)
        """
        url_base = self.registryValue("httpUrlBase")

        if not url_base:
            public_url = conf.supybot.servers.http.publicUrl()
            if public_url:
                url_base = public_url.rstrip("/") + "/llm"
            else:
                port = conf.supybot.servers.http.port()
                url_base = f"http://localhost:{port}/llm"

        return f"{url_base}/"

    def getPluginHelp(self) -> str:  # noqa: N802
        """Return plugin help with dynamic documentation URL.

        Overrides Limnoria's default to include web docs URL.
        """
        url = self._get_help_url()
        return (
            _(
                "AI-powered commands using LiteLLM. "
                "Commands: ask, code, draw, forget. "
                "Full documentation: %s"
            )
            % url
        )

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

    def _get_context_enabled(self, channel: str) -> bool:
        """Check if context is enabled for a channel.

        Args:
            channel: Channel name

        Returns:
            True if context is enabled for this channel
        """
        return self.registryValue("contextEnabled", channel)

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

        nick = self._get_nick(msg)
        channel = self._get_channel(msg)

        # Detect images for vision
        images = self.llm_service.detect_images(text)

        # Get conversation history (personal + shared channel) if context enabled
        if self._get_context_enabled(channel):
            history = self.context.get_messages(nick, channel)
            channel_history = self.context.get_channel_messages(channel, exclude_nick=nick)
        else:
            history, channel_history = [], []

        if images:
            # Clean prompt by removing image URLs
            clean_prompt = text
            for img in images:
                clean_prompt = clean_prompt.replace(img, "").strip()

            irc.reply(_("Processing with %d image(s)...") % len(images), prefixNick=False)
            result = self.llm_service.completion(
                clean_prompt,
                command="ask",
                images=images,
                history=history,
                channel_history=channel_history,
                irc=irc,
                msg=msg,
            )
        else:
            result = self.llm_service.completion(
                text,
                command="ask",
                history=history,
                channel_history=channel_history,
                irc=irc,
                msg=msg,
            )

        # Format response with grounding icon if search was used
        response = result.content
        display_response = f"{GROUNDING_ICON} {response}" if result.grounding_used else response

        # Reply first, then store context (so user gets response even if context fails)
        irc.reply(display_response, prefixNick=False)

        # Store conversation context if enabled for this channel
        if self._get_context_enabled(channel):
            # Store in personal context (without icon for clean history)
            self.context.add_message(nick, channel, Role.USER, text)
            self.context.add_message(nick, channel, Role.ASSISTANT, response)

            # Store in shared channel context (allows group conversation flow)
            self.context.add_channel_message(channel, nick, Role.USER, text)
            self.context.add_channel_message(channel, irc.nick, Role.ASSISTANT, response)

    ask = wrap(ask, [("checkCapability", "llm.ask"), "text"])

    def code(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<request>

        Generate code based on your request. Code is saved to HTTP link.
        Supports conversation context for iterating on code.

        Examples:
          %code Python function to calculate fibonacci numbers
          %code Now add memoization to that
          %code JavaScript async fetch with error handling
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        nick = self._get_nick(msg)
        channel = self._get_channel(msg)

        # Get conversation history (personal + shared channel) if context enabled
        if self._get_context_enabled(channel):
            history = self.context.get_messages(nick, channel)
            channel_history = self.context.get_channel_messages(channel, exclude_nick=nick)
        else:
            history, channel_history = [], []

        result = self.llm_service.completion(
            text,
            command="code",
            history=history,
            channel_history=channel_history,
            irc=irc,
            msg=msg,
        )

        response = result.content
        grounding_prefix = f"{GROUNDING_ICON} " if result.grounding_used else ""

        # Reply first, then store context
        url = self.llm_service.save_code_to_http(response)
        if url:
            # Try AI-generated summary first
            summary = self.llm_service.summarize(response, channel)
            if summary:
                preview = self.llm_service._sanitize_output(summary)
            else:
                # Fallback to truncation if summarization fails
                preview = response.replace("\n", " ").strip()
                if len(preview) > CODE_PREVIEW_MAX_LEN:
                    preview = preview[:CODE_PREVIEW_TRUNCATE_LEN] + "..."
                preview = self.llm_service._sanitize_output(preview)
            irc.reply(f"{grounding_prefix}{preview} — {url}", prefixNick=False)
        else:
            # Fallback to IRC paging if save failed
            display_response = f"{grounding_prefix}{response}" if grounding_prefix else response
            irc.reply(display_response, prefixNick=False)

        # Store conversation context if enabled for this channel
        if self._get_context_enabled(channel):
            # Store in personal context (without icon for clean history)
            self.context.add_message(nick, channel, Role.USER, text)
            self.context.add_message(nick, channel, Role.ASSISTANT, response)

            # Store in shared channel context (allows group conversation flow)
            self.context.add_channel_message(channel, nick, Role.USER, text)
            self.context.add_channel_message(channel, irc.nick, Role.ASSISTANT, response)

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

        Examples:
          %draw A sunset over mountains in watercolor style
          %draw A cyberpunk cityscape at night
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        # Typing indicator sent by service - no "Generating..." message needed
        result = self.llm_service.image_generation(text, irc=irc, msg=msg)
        irc.reply(result, prefixNick=False)

    draw = wrap(draw, [("checkCapability", "llm.draw"), "text"])

    def forget(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str | None,
    ) -> None:
        """[<channel>]

        Clear your conversation context (memory) for the current or specified channel.
        Use this to start fresh.
        """
        nick = self._get_nick(msg)
        # Default to current channel if not specified
        if channel is None:
            channel = self._get_channel(msg)
        cleared = self.context.clear(nick, channel)

        if cleared:
            irc.reply(_("Conversation context cleared. Starting fresh!"), prefixNick=False)
        else:
            irc.reply(_("No conversation context to clear."), prefixNick=False)

    forget = wrap(forget, [optional("channel")])

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

    llmkeys = wrap(llmkeys, ["admin"])


Class = LLM
