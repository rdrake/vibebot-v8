"""LLM plugin implementation."""

from __future__ import annotations

import collections
import contextlib
import logging
import mimetypes
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.httpserver as httpserver
import supybot.ircdb as ircdb
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils
import supybot.log as log
import supybot.schedule as schedule
from supybot import world
from supybot.commands import optional, wrap
from supybot.i18n import PluginInternationalization

from .context import ContextConfig, ConversationContext, Role
from .persistence import LLMDatabase
from .service import (
    CODE_PREVIEW_MAX_LEN,
    CODE_PREVIEW_TRUNCATE_LEN,
    CompletionResult,
    ImageResult,
    LLMService,
    VideoResult,
)
from .tracing import TraceFilter, generate_request_id, request_id

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

_ = PluginInternationalization("LLM")

# Icon shown when Google grounding/search was used in the response
GROUNDING_ICON = "\U0001f310"  # 🌐 (globe with meridians)


class PreflightResult(NamedTuple):
    """Result of the shared command preflight check.

    ``blocked`` is True when the command should not proceed (the preflight
    already sent the appropriate error reply and logged usage).
    """

    blocked: bool
    nick: str  # account-resolved identity for logging
    channel: str
    account: str | None  # NickServ account, or None if unidentified


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

<h3><code class="command">%animate</code> <span class="param">&lt;prompt&gt;</span></h3>
<p>Generate a short video from a text description. Also available as <code>%video</code>. Requires NickServ identification.</p>
<pre><code><span class="example">%animate A cat playing with a ball of yarn</span>
<span class="example">%animate A timelapse of a sunset over the ocean</span></code></pre>

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
        self.log.addFilter(TraceFilter())

        # Apply configured log level to plugin and service loggers
        self._apply_log_level()

        self.startup_time = time.time()  # Track startup for ZNC playback filtering
        self.build_info = self._get_build_info()

        # Initialize conversation context
        self._init_context()

        # Initialize database for persistence
        db_path = self.registryValue("databasePath")
        if not db_path:
            db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
        self.db = LLMDatabase(db_path)

        # Track nicks already migrated to account-based identity this session
        self._migrated_nicks: set[str] = set()

        # In-memory per-command rate-limit buckets: "{command}:{account}" -> deque of timestamps
        self._rate_buckets: dict[str, collections.deque[float]] = {}

        # Reminder storage: event_name -> (nick, channel, message)
        self._reminders: dict[str, tuple[str, str, str]] = {}
        self._reminders_lock = threading.Lock()

        # Reload persisted reminders from database
        self._reload_reminders(irc)

        # Startup notification tracking
        self._pending_channels: set[str] = set()
        self._startup_notified: bool = False

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

        # Safety poll for pending tasks (5-minute fallback for event-driven wakeups)
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_pending_tasks")

        schedule.addPeriodicEvent(
            self._check_pending_tasks,
            self._SAFETY_POLL_INTERVAL,
            name="llm_pending_tasks",
            now=False,
        )

        # Event-driven queue wakeup state
        self._next_wakeup_time: float | None = None
        self._schedule_queue_wakeup()  # rebuild from DB on startup

        # Register callback for live log level changes
        conf.supybot.plugins.LLM.logLevel.addCallback(self._on_log_level_change)

    def _apply_log_level(self) -> None:
        """Set plugin logger levels from the logLevel config value."""
        level_name = self.registryValue("logLevel")
        level = getattr(logging, level_name, logging.WARNING)
        self.log.setLevel(level)
        self.llm_service.log.setLevel(level)

    def _on_log_level_change(self, *args: object) -> None:
        """Called when logLevel config changes at runtime."""
        self._apply_log_level()

    def die(self) -> None:
        """Clean up when plugin is unloaded."""
        # Clean up expired reminders from database
        if hasattr(self, "db"):
            self.db.delete_expired_reminders()

        # Remove scheduled cleanup event
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_file_cleanup")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_pending_tasks")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_queue_wakeup")

        # Remove all reminder events (guard for tests that mock __init__)
        if hasattr(self, "_reminders"):
            with self._reminders_lock:
                for event_name in list(self._reminders.keys()):
                    with contextlib.suppress(KeyError):
                        schedule.removeEvent(event_name)
                self._reminders.clear()

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
            self.log.error("Scheduled file cleanup failed: %s", e)

    def _schedule_queue_wakeup(self, at_time: float | None = None) -> None:
        """Schedule a one-shot wakeup for the next due queue task.

        If *at_time* is given it is used directly; otherwise the earliest
        ``next_attempt_at`` is queried from the database.  A wakeup is only
        scheduled when it would fire earlier than any existing one.

        Args:
            at_time: Optional explicit wakeup timestamp.  When provided the
                database is not queried.
        """
        if at_time is None:
            at_time = self.db.get_next_due_time()
        if not isinstance(at_time, (int, float)):
            return

        now = time.time()

        # Clamp past-due timestamps to now + 1 so Limnoria doesn't discard them
        effective = max(at_time, now + 1)

        # Skip if an existing wakeup is already earlier and still in the future
        if (
            self._next_wakeup_time is not None
            and self._next_wakeup_time <= effective
            and self._next_wakeup_time > now
        ):
            return

        # Replace any existing wakeup
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_queue_wakeup")

        schedule.addEvent(
            self._check_pending_tasks,
            effective,
            name="llm_queue_wakeup",
        )
        self._next_wakeup_time = effective

    def _check_pending_tasks(self) -> None:
        """Poll pending tasks and deliver completed/failed/expired results.

        Each result is delivered independently so that one delivery failure
        does not cascade to the rest of the batch.
        """
        try:
            # The wakeup that triggered this call has fired; clear it so
            # _schedule_queue_wakeup can schedule the next one.
            self._next_wakeup_time = None

            # Build set of channels the bot is currently in
            deliverable_channels: set[str] = set()
            for irc_conn in world.ircs:
                deliverable_channels.update(irc_conn.state.channels.keys())

            results = self.llm_service.check_pending_tasks(deliverable_channels)

            for r in results:
                try:
                    self._deliver_pending_result(r)
                except Exception as e:
                    self.log.error(
                        "Delivery failed for task_id=%s nick=%s: %s",
                        r.task_id,
                        r.nick,
                        e,
                    )

            # Schedule the next wakeup based on remaining queue state
            self._schedule_queue_wakeup()
        except Exception as e:
            self.log.error("Pending task check failed: %s", e)

    # Safety poll interval (seconds) — fallback for event-driven wakeups
    _SAFETY_POLL_INTERVAL = 300  # 5 minutes

    # Delivery retry constants: 15 * 2^attempt, capped at 120s, max 10 attempts
    _DELIVERY_BASE_BACKOFF = 15
    _DELIVERY_MAX_BACKOFF = 120
    _DELIVERY_MAX_ATTEMPTS = 10

    def _deliver_pending_result(self, r) -> None:
        """Deliver a single pending task result to the correct target.

        Sends the message to the original channel or PM nick.  For results
        with a ``task_id`` (from the durable delivery queue), acknowledges
        successful delivery by deleting the row, or retries with bounded
        exponential backoff on failure.

        Args:
            r: PendingTaskResult from check_pending_tasks.
        """
        target = r.reply_target
        nick = r.nick
        prompt_preview = self.llm_service.sanitize_output(r.prompt_preview)

        if r.status == "expired":
            text = f'{nick}: sorry, your {r.task_type} request "{prompt_preview}" expired.'
            self.log.warning(
                "Deferred task expired: task_type=%s nick=%s target=%s prompt=%s",
                r.task_type,
                nick,
                target,
                prompt_preview[:50],
            )
        elif r.status == "failed_terminal":
            reason = self.llm_service.sanitize_output(r.reason)[:200]
            text = f'{nick}: sorry, your {r.task_type} request "{prompt_preview}" failed: {reason}'
            self.log.warning(
                "Deferred task failed_terminal: task_type=%s nick=%s target=%s reason=%s",
                r.task_type,
                nick,
                target,
                reason[:100],
            )
        elif r.status == "completed":
            content = self.llm_service.sanitize_output(r.content)
            if r.task_type == "code":
                # Try to save code to HTTP URL
                url = self.llm_service.save_code_to_http(r.content)
                if url:
                    text = f'{nick}: your code is ready! "{prompt_preview}" \u2192 {url}'
                else:
                    text = f"{nick}: {content}"
            elif r.task_type == "draw":
                text = f'{nick}: your image is ready! "{prompt_preview}" \u2192 {content}'
            elif r.task_type == "animate":
                text = f'{nick}: your video is ready! "{prompt_preview}" \u2192 {content}'
            else:
                # ask or fallback
                text = f"{nick}: {content}"
        else:
            return

        # Try to deliver via IRC
        delivered = False
        try:
            for irc_conn in world.ircs:
                if r.is_channel:
                    if target in irc_conn.state.channels:
                        irc_conn.queueMsg(ircmsgs.privmsg(target, text))
                        delivered = True
                        break
                else:
                    # PM delivery — use first available connection
                    irc_conn.queueMsg(ircmsgs.privmsg(target, text))
                    delivered = True
                    break
        except Exception:
            delivered = False

        # Acknowledge or retry delivery for durable results
        if r.task_id is not None:
            if delivered:
                self.db.delete_pending_task(r.task_id)
            else:
                now = time.time()
                attempt = max(r.delivery_attempt_count, 0) + 1
                delay = min(
                    self._DELIVERY_BASE_BACKOFF * (2 ** (attempt - 1)),
                    self._DELIVERY_MAX_BACKOFF,
                )
                state = "delivery_failed" if attempt >= self._DELIVERY_MAX_ATTEMPTS else "retrying"
                retry_at = now + delay
                self.db.update_delivery_attempt(
                    task_id=r.task_id,
                    delivery_state=state,
                    last_delivery_error="IRC delivery failed",
                    delivery_attempt_count=attempt,
                    next_attempt_at=retry_at,
                )
                if state != "delivery_failed":
                    self._schedule_queue_wakeup(at_time=retry_at)

        # Log usage for completed tasks
        if r.status == "completed" and delivered and (r.cost > 0 or r.prompt_tokens > 0):
            for irc_conn in world.ircs:
                identity = self._resolve_nick_to_identity(irc_conn, nick)
                self.db.log_usage(
                    identity,
                    target,
                    r.task_type,
                    r.model,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.cost,
                )
                break

    def doPrivmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Monitor channel messages for enhanced context (opt-in feature).

        When contextTrackAllMessages is enabled, this captures all channel
        messages to provide richer context for the ask command.

        Note: Disabled by default for privacy since messages are sent to
        third-party LLM providers.
        """
        channel = msg.channel
        if not channel:
            return  # Skip private messages

        # Check config first (cheapest checks)
        if not self.registryValue("contextEnabled", channel):
            return
        if not self.registryValue("contextTrackAllMessages", channel):
            return

        # Then message checks
        if self._is_old_message(msg):
            return
        if ircmsgs.isCtcp(msg) and not ircmsgs.isAction(msg):
            return
        if ircutils.strEqual(irc.nick, msg.nick):
            return

        nick = self._get_identity(irc, msg)
        message_text = msg.args[1] if len(msg.args) > 1 else ""

        # Store in conversation context for richer follow-up questions
        ctx_cfg = self._get_context_config(channel)
        self.context.add_message(nick, channel, Role.USER, message_text, config=ctx_cfg)
        self.context.add_channel_message(channel, nick, Role.USER, message_text, config=ctx_cfg)

    def doJoin(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Track channels the bot is joining for startup notification.

        When the bot joins a channel, we add it to _pending_channels.
        The channel is removed when we receive do315 (end of WHO).
        """
        if ircutils.strEqual(irc.nick, msg.nick):
            channel = msg.args[0]
            self._pending_channels.add(channel)

    def do315(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Handle end of WHO reply (channel sync complete).

        When a channel finishes syncing (WHO complete), remove it from
        pending channels. When all channels are synced and we haven't
        notified yet, send the startup notification.
        """
        channel = msg.args[1]
        self._pending_channels.discard(channel)

        if not self._pending_channels and not self._startup_notified:
            self._send_startup_notification(irc)
            self._startup_notified = True

    def do376(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Handle end of MOTD (connection established).

        Reset startup tracking state on reconnection so we send a fresh
        notification. Also handles case where bot has no channels configured.
        """
        self._pending_channels.clear()
        self._startup_notified = False

        # If no channels are configured, send notification immediately
        # (we need to check after a short delay to allow channel joins to start)
        def check_no_channels() -> None:
            if not self._pending_channels and not self._startup_notified:
                self._send_startup_notification(irc)
                self._startup_notified = True

        # Schedule check after 2 seconds to allow join commands to be processed
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")
        schedule.addEvent(check_no_channels, time.time() + 2, name="llm_startup_check")

    def _send_startup_notification(self, irc: callbacks.Irc) -> None:
        """Send startup notification PM to bot owner.

        Message format: VibeBot started | v8 | N channel(s) | YYYY-MM-DD HH:MM:SS UTC
        """
        # Remove the scheduled check event if it exists
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_startup_check")

        # Find users with owner capability.
        users_mod = getattr(ircdb, "users", None)
        users_map = getattr(users_mod, "users", {})
        owners = [user.name for user in users_map.values() if "owner" in user.capabilities]
        if not owners:
            self.log.warning("No bot owner configured, skipping startup notification")
            return

        owner = owners[0]
        channel_count = len(irc.state.channels)
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        plural = "s" if channel_count != 1 else ""
        message = f"VibeBot started | v8 | {channel_count} channel{plural} | {timestamp}"

        irc.queueMsg(ircmsgs.privmsg(owner, message))
        self.log.info("Startup notification sent to %s", owner)

    @staticmethod
    def _get_build_info() -> str:
        """Get version and git commit SHA for context prompt.

        Returns:
            Build string like "v0.1.0 (abc1234)" or just "v0.1.0" if git unavailable.
        """
        from . import __version__

        try:
            sha = subprocess.check_output(  # noqa: S603
                ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return f"v{__version__} ({sha})"
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return f"v{__version__}"

    def _init_context(self) -> None:
        """Initialize context manager with global defaults (called once at startup).

        Per-channel overrides are read at query time via ``_get_context_config``.
        """
        config = ContextConfig(
            max_messages=self.registryValue("contextMaxMessages"),
            timeout_minutes=self.registryValue("contextTimeoutMinutes"),
            enabled=self.registryValue("contextEnabled"),
            channel_max_messages=self.registryValue("channelContextMaxMessages"),
        )
        self.context = ConversationContext(config)

    def _get_context_config(self, channel: str) -> ContextConfig:
        """Read channel-specific context configuration.

        Args:
            channel: IRC channel name (passed to ``registryValue``
                for per-channel overrides)

        Returns:
            ContextConfig with channel-specific values
        """
        return ContextConfig(
            max_messages=self.registryValue("contextMaxMessages", channel),
            timeout_minutes=self.registryValue("contextTimeoutMinutes", channel),
            enabled=self.registryValue("contextEnabled", channel),
            channel_max_messages=self.registryValue("channelContextMaxMessages", channel),
        )

    def _make_reminder_delivery_closure(
        self, nick: str, channel: str, message: str, event_name: str
    ):
        """Create a reminder delivery closure with error handling.

        Wraps delivery in try/finally so cleanup (removing from _reminders
        and database) always happens even if queueMsg raises.

        Args:
            nick: User's nick
            channel: Channel to deliver to
            message: Reminder message
            event_name: Scheduler event name for cleanup

        Returns:
            Callable for use with schedule.addEvent
        """
        lock = self._reminders_lock

        def _deliver() -> None:
            try:
                for active_irc in world.ircs:
                    safe_message = self.llm_service.sanitize_output(message)
                    active_irc.queueMsg(
                        ircmsgs.privmsg(channel, f"{nick}: Reminder: {safe_message}")
                    )
                    break
            finally:
                with lock:
                    self._reminders.pop(event_name, None)
                self.db.delete_reminder(event_name)

        return _deliver

    def _reload_reminders(self, irc: callbacks.Irc) -> None:
        """Reload persisted reminders from database on startup.

        Reschedules future reminders and delivers overdue ones immediately.
        Reminders more than 24h overdue are cleaned up by the database layer.
        """
        pending = self.db.load_pending_reminders()
        now = time.time()

        for reminder in pending:
            nick = reminder.nick
            channel = reminder.channel
            message = reminder.message
            event_name = reminder.event_name

            deliver = self._make_reminder_delivery_closure(nick, channel, message, event_name)

            if reminder.fire_at <= now:
                # Overdue — deliver immediately
                deliver()
            else:
                # Future — reschedule
                try:
                    schedule.addEvent(deliver, reminder.fire_at, name=event_name)
                    with self._reminders_lock:
                        self._reminders[event_name] = (nick, channel, message)
                except Exception as e:
                    self.log.error("Failed to reload reminder %s: %s", event_name, e)
                    self.db.delete_reminder(event_name)

        if pending:
            self.log.info("Reloaded %s reminder(s) from database", len(pending))

    @contextlib.contextmanager
    def _allow_concurrent(self):
        """Temporarily release the MetaSynchronized RLock for concurrent commands.

        Limnoria's Commands base class wraps callCommand() with an RLock,
        which serializes all command execution per-plugin. This releases the
        lock around blocking I/O (LLM API calls) so multiple commands can
        run concurrently.

        Uses RLock._release_save()/_acquire_restore() — the same mechanism
        threading.Condition uses internally.

        WARNING: These are private CPython implementation details (prefixed
        with ``_``) and are not guaranteed by the Python language spec. They
        are stable in CPython 3.12-3.14 and used by threading.Condition, but
        could break on alternative interpreters or future CPython versions.
        """
        lock = self._MetaSynchronized_rlock
        try:
            saved = lock._release_save()
        except RuntimeError:
            # Lock not held (e.g., direct call in tests) — just proceed
            yield
            return
        try:
            yield
        finally:
            lock._acquire_restore(saved)

    @contextlib.contextmanager
    def _trace_request(self, command: str, nick: str, channel: str):
        """Set a unique trace ID for the duration of a command invocation.

        All log messages emitted while the context manager is active
        will be prefixed with [trace_id] by TraceFilter.
        """
        rid = generate_request_id()
        token = request_id.set(rid)
        self.log.info("%s from %s/%s", command, channel, nick)
        try:
            yield rid
        finally:
            self.log.info("%s complete: %s/%s", command, channel, nick)
            request_id.reset(token)

    def _get_help_url(self) -> str:
        """Get the URL to the web help documentation.

        Delegates to service.get_http_paths() for consistent URL construction.

        Returns:
            Full URL to help page (e.g., http://localhost:8080/llm/)
        """
        _, url_base = self.llm_service.get_http_paths()
        return f"{url_base}/"

    def getPluginHelp(self) -> str:  # noqa: N802
        """Return plugin help with dynamic documentation URL.

        Overrides Limnoria's default to include web docs URL.
        """
        url = self._get_help_url()
        return (
            _(
                "AI-powered commands using LiteLLM. "
                "Commands: ask, code, draw, animate (video), forget. "
                "Full documentation: %s"
            )
            % url
        )

    def invalidCommand(  # noqa: N802
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        tokens: list[str],
    ) -> None:
        """Handle unrecognized commands as 'ask' by default.

        When someone says "vibebot hello there" without a command,
        treat it as "%ask hello there".
        """
        if not tokens:
            return

        # Check if user has ask capability
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return

        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        # Reconstruct the prompt from tokens and call ask
        # wrap() replaces ask's signature at runtime; ty sees the pre-wrap params
        self.ask(irc, msg, tokens[:])  # ty: ignore[missing-argument]

    def _resolve_nick_to_identity(self, irc: callbacks.Irc, nick: str) -> str:
        """Resolve a plain nick to its NickServ account, falling back to nick.

        AfterNet supports ``account-notify``, so Limnoria caches NickServ
        account names in ``irc.state.nicksToAccounts``.  Using the account
        name means usage stats, conversation context, and reminders follow
        the user across nick changes.

        When the account differs from the nick, old usage rows logged under
        the raw nick are lazily migrated to the account name (once per nick
        per session) so that ``%usage`` reports include historical data.

        Args:
            irc: IRC connection (provides account lookup via ``state``)
            nick: Plain IRC nick (no hostmask)

        Returns:
            NickServ account name, or the original nick as fallback.
        """
        try:
            account = irc.state.nickToAccount(nick)
            if account:
                self._maybe_migrate_nick(nick, account)
                return account
        except (KeyError, AttributeError):
            pass
        return nick

    def _maybe_migrate_nick(self, old_nick: str, account: str) -> None:
        """Migrate old nick-based usage rows to the account, once per session.

        Skips the DB call entirely when the nick and account are the same
        (case-insensitive) or when we've already attempted migration for
        this nick.

        Args:
            old_nick: The user's current IRC nick.
            account: The resolved NickServ account name.
        """
        if old_nick.lower() == account.lower():
            return
        key = old_nick.lower()
        if key in self._migrated_nicks:
            return
        self._migrated_nicks.add(key)
        count = self.db.migrate_nick(old_nick, account)
        if count > 0:
            self.log.info("Migrated %d usage row(s) from %s to %s", count, old_nick, account)

    def _get_identity(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
        """Return account name if the user is logged in, else fall back to nick.

        Extracts the nick from *msg.prefix* and delegates to
        :meth:`_resolve_nick_to_identity`.

        Args:
            irc: IRC connection (provides account lookup via ``state``)
            msg: IRC message

        Returns:
            NickServ account name, or the user's current nick as fallback.
        """
        nick = ircutils.nickFromHostmask(msg.prefix)
        return self._resolve_nick_to_identity(irc, nick)

    def _require_account(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Require NickServ identification. Returns account name or None.

        If the user is not identified, sends an error reply and returns None.
        Callers should ``return`` immediately when None is returned.
        """
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        try:
            account = irc.state.nickToAccount(raw_nick)
        except (KeyError, AttributeError):
            account = None
        if not account:
            irc.error(_("You must be identified with NickServ to use this command."))
            return None
        return account

    def _run_preflight(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        command: str,
        *,
        require_account: bool,
        apply_rate_limit: bool,
    ) -> PreflightResult:
        """Shared preflight check for all commands.

        Runs the following sequence:
        1. Account resolution (required or optional depending on command).
        2. Flagged-user block check.
        3. Per-command rate-limit check (draw/animate only).

        When any check fails the method sends the appropriate IRC error,
        logs usage with the blocked status, and returns ``blocked=True``.

        Args:
            irc: IRC connection.
            msg: IRC message.
            text: User's prompt text (for usage logging).
            command: Command name (ask, code, draw, animate).
            require_account: If True, NickServ identification is mandatory.
            apply_rate_limit: If True, check per-command rate limit.

        Returns:
            PreflightResult with blocked=False if the command should proceed.
        """
        channel = self._get_channel(msg)

        # --- account resolution ---
        if require_account:
            account = self._require_account(irc, msg)
            if account is None:
                nick = ircutils.nickFromHostmask(msg.prefix)
                self.db.log_usage(
                    nick,
                    channel,
                    command,
                    "",
                    0,
                    0,
                    0.0,
                    prompt=text,
                    status="auth_failure",
                )
                return PreflightResult(blocked=True, nick=nick, channel=channel, account=None)
            nick = self._resolve_nick_to_identity(irc, ircutils.nickFromHostmask(msg.prefix))
        else:
            raw_nick = ircutils.nickFromHostmask(msg.prefix)
            try:
                account = irc.state.nickToAccount(raw_nick)
            except (KeyError, AttributeError):
                account = None
            nick = self._get_identity(irc, msg)

        # --- flagged check ---
        if self._check_flagged(irc, msg, account):
            self.db.log_usage(
                nick,
                channel,
                command,
                "",
                0,
                0,
                0.0,
                prompt=text,
                status="flagged_blocked",
            )
            return PreflightResult(blocked=True, nick=nick, channel=channel, account=account)

        # --- rate limit check (expensive commands only) ---
        if (
            apply_rate_limit
            and account
            and self._check_rate_limit(irc, command, account, nick, channel, text)
        ):
            return PreflightResult(blocked=True, nick=nick, channel=channel, account=account)

        return PreflightResult(blocked=False, nick=nick, channel=channel, account=account)

    def _is_rate_limited(self, command: str, account: str, now: float) -> bool:
        """Check if a user exceeds the per-command rate limit.

        Evicts timestamps outside the configured window before checking.

        Args:
            command: Command name (draw or animate).
            account: NickServ account name.
            now: Current time (seconds since epoch).

        Returns:
            True if the user has exceeded the rate limit.
        """
        max_count = self.registryValue(f"{command}RateLimitCount")
        window = self.registryValue(f"{command}RateLimitWindow")
        cutoff = now - window

        key = f"{command}:{account}"
        bucket = self._rate_buckets.get(key)
        if bucket is None:
            return False

        # Evict expired entries
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        # Clean up idle keys so bucket map cannot grow forever.
        if not bucket:
            self._rate_buckets.pop(key, None)
            return False

        return len(bucket) >= max_count

    def _record_rate_limit_hit(self, command: str, account: str, now: float) -> None:
        """Record a request timestamp in the rate-limit bucket.

        Args:
            command: Command name.
            account: NickServ account name.
            now: Current time.
        """
        key = f"{command}:{account}"
        bucket = self._rate_buckets.get(key)
        if bucket is None:
            bucket = collections.deque()
            self._rate_buckets[key] = bucket
        bucket.append(now)

    def _check_rate_limit(
        self,
        irc: callbacks.Irc,
        command: str,
        account: str,
        nick: str,
        channel: str,
        text: str,
    ) -> bool:
        """Check rate limit and send error if exceeded.

        Always records the request timestamp. When enforceRateLimits is True
        and the user is over the limit, sends an error reply and logs
        ``status="rate_limited"``.

        Args:
            irc: IRC connection.
            command: Command name.
            account: NickServ account name.
            nick: Resolved identity for logging.
            channel: Channel name.
            text: Prompt text for logging.

        Returns:
            True if the request should be blocked.
        """
        now = time.time()
        over_limit = self._is_rate_limited(command, account, now)

        # Always record the hit (so the window tracks correctly)
        self._record_rate_limit_hit(command, account, now)

        if over_limit:
            enforce = self.registryValue("enforceRateLimits")
            max_count = self.registryValue(f"{command}RateLimitCount")
            window = self.registryValue(f"{command}RateLimitWindow")
            key = f"{command}:{account}"
            count = len(self._rate_buckets.get(key, ()))
            if enforce:
                self.log.info(
                    "rate_limited command=%s account=%s count=%d limit=%d window=%ss",
                    command,
                    account,
                    count,
                    max_count,
                    window,
                )
                irc.error(
                    _("Rate limit exceeded for %s. Please wait before trying again.") % command
                )
                self.db.log_usage(
                    nick,
                    channel,
                    command,
                    "",
                    0,
                    0,
                    0.0,
                    prompt=text,
                    status="rate_limited",
                )
                return True
            self.log.info(
                "rate_limit_shadow command=%s account=%s count=%d limit=%d window=%ss",
                command,
                account,
                count,
                max_count,
                window,
            )
        return False

    @staticmethod
    def _is_content_blocked_error(error: str | None) -> bool:
        """Return True if an error string indicates a content safety block.

        Checks for common keywords that LLM providers use when content
        is rejected for safety/moderation reasons.

        Args:
            error: Error message string from the LLM service, or None.

        Returns:
            True if the error looks like a content safety block.
        """
        if not error:
            return False
        lower = error.lower()
        return (
            "content" in lower or "moderation" in lower or "safety" in lower or "blocked" in lower
        )

    @staticmethod
    def _month_start_ts() -> float:
        """Return the UNIX timestamp for midnight UTC on the 1st of the current month."""
        return (
            datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        )

    def _check_flagged(self, irc: callbacks.Irc, msg: IrcMsg, account: str | None) -> bool:
        """Check if a user account is flagged for abuse.

        Returns True (and sends error) if the user should be blocked.
        Returns False if the user is clear to proceed.
        Unidentified users (account=None) are not checked.
        """
        if account is None:
            return False
        if self.db.is_user_flagged(account):
            irc.error(_("Your account has been suspended. Contact a bot admin."))
            return True
        return False

    def _get_channel(self, msg: IrcMsg) -> str:
        """Extract channel from IRC message.

        Args:
            msg: IRC message

        Returns:
            Channel name
        """
        return msg.args[0] if msg.args else "unknown"

    @staticmethod
    def _extract_raw_arg(irc: callbacks.Irc, msg: IrcMsg, command: str) -> str | None:
        """Extract the raw argument for a command from the original message.

        Limnoria's tokenizer treats ``[…]`` as nested-command syntax, so a
        nick like ``Rubin[F]`` is evaluated as a nested command before the
        command method ever sees it.  This bypasses the tokenizer entirely
        by reading the addressed text from the raw IRC message.

        Args:
            irc: IRC connection (needed by ``callbacks.addressed``)
            msg: IRC message
            command: Command name to find (e.g. ``"usage"``)

        Returns:
            The raw text after the command name, or None if absent.
        """
        payload = callbacks.addressed(irc, msg)
        if not payload:
            return None
        # payload is e.g. "usage Rubin[F]" or "llm usage Rubin[F]"
        idx = payload.lower().find(command)
        if idx < 0:
            return None
        after = payload[idx + len(command) :].strip()
        return after or None

    def _is_old_message(self, msg: IrcMsg) -> bool:
        """Check if message predates bot startup (ZNC playback).

        Args:
            msg: IRC message

        Returns:
            True if message is older than bot startup time
        """
        if msg.time == 0:
            return False  # No timestamp = live message (not ZNC playback)
        return msg.time < self.startup_time

    def _get_context_enabled(self, channel: str) -> bool:
        """Check if context is enabled for a channel.

        Args:
            channel: Channel name

        Returns:
            True if context is enabled for this channel
        """
        return self.registryValue("contextEnabled", channel)

    def _store_context_and_log_usage(
        self,
        nick: str,
        channel: str,
        command: str,
        text: str,
        response: str,
        result: CompletionResult | ImageResult | VideoResult,
        irc: callbacks.Irc,
        msg: IrcMsg,
    ) -> None:
        """Store conversation context and log API usage for a command.

        Shared between all commands (ask, code, draw, animate).

        Args:
            nick: User's nick
            channel: Channel name
            command: Command name ("ask", "code", "draw", or "animate")
            text: Original user input
            response: Text to store in context (e.g. LLM response or
                ``"[Generated image: <url>]"``)
            result: Result with usage metadata
            irc: IRC connection instance
            msg: IRC message
        """
        # Store conversation context if enabled and no error occurred
        if result.error is None and self._get_context_enabled(channel):
            ctx_cfg = self._get_context_config(channel)
            self.context.add_message(nick, channel, Role.USER, text, config=ctx_cfg)
            self.context.add_message(nick, channel, Role.ASSISTANT, response, config=ctx_cfg)
            self.context.add_channel_message(channel, nick, Role.USER, text, config=ctx_cfg)
            self.context.add_channel_message(
                channel, irc.nick, Role.ASSISTANT, response, config=ctx_cfg
            )

        # Determine status
        if result.error is None:
            status = "success"
        elif self._is_content_blocked_error(result.error):
            status = "content_blocked"
        else:
            status = "error"
        error_detail = (result.error or "")[:200]
        self.db.log_usage(
            nick,
            channel,
            command,
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost,
            prompt=text,
            status=status,
            error_detail=error_detail,
        )

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

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "ask",
            require_account=False,
            apply_rate_limit=False,
        )
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        with self._trace_request("ask", nick, channel):
            # Detect images for vision
            images = self.llm_service.detect_images(text)

            # Get conversation history (personal + shared channel) if context enabled
            if self._get_context_enabled(channel):
                ctx_cfg = self._get_context_config(channel)
                history = self.context.get_messages(nick, channel, config=ctx_cfg)
                channel_history = self.context.get_channel_messages(
                    channel, exclude_nick=nick, config=ctx_cfg
                )
            else:
                history, channel_history = [], []

            with self._allow_concurrent():
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
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                display_response = (
                    f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                )

                # Reply first, then store context (so user gets response even if context fails)
                self.log.info("replying to %s/%s", channel, nick)
                irc.reply(display_response, prefixNick=False)

            self._store_context_and_log_usage(
                nick, channel, "ask", text, response, result, irc, msg
            )

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

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "code",
            require_account=False,
            apply_rate_limit=False,
        )
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        with self._trace_request("code", nick, channel):
            # Get conversation history (personal + shared channel) if context enabled
            if self._get_context_enabled(channel):
                ctx_cfg = self._get_context_config(channel)
                history = self.context.get_messages(nick, channel, config=ctx_cfg)
                channel_history = self.context.get_channel_messages(
                    channel, exclude_nick=nick, config=ctx_cfg
                )
            else:
                history, channel_history = [], []

            with self._allow_concurrent():
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
                        preview = self.llm_service.sanitize_output(summary)
                    else:
                        # Fallback to truncation if summarization fails
                        preview = response.replace("\n", " ").strip()
                        if len(preview) > CODE_PREVIEW_MAX_LEN:
                            preview = preview[:CODE_PREVIEW_TRUNCATE_LEN] + "..."
                        preview = self.llm_service.sanitize_output(preview)
                    self.log.info("replying to %s/%s", channel, nick)
                    irc.reply(f"{grounding_prefix}{preview} — {url}")
                else:
                    # Fallback to IRC paging if save failed
                    display_response = (
                        f"{grounding_prefix}{response}" if grounding_prefix else response
                    )
                    self.log.info("replying to %s/%s", channel, nick)
                    irc.reply(display_response)

            self._store_context_and_log_usage(
                nick, channel, "code", text, response, result, irc, msg
            )

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

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "draw",
            require_account=True,
            apply_rate_limit=True,
        )
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        with self._trace_request("draw", nick, channel):
            # Typing indicator sent by service - no "Generating..." message needed
            with self._allow_concurrent():
                result = self.llm_service.image_generation(text, irc=irc, msg=msg)
                self.log.info("replying to %s/%s", channel, nick)
                sanitized_content = self.llm_service.sanitize_output(result.content)
                if result.rewritten_prompt:
                    prompt_preview = self.llm_service.sanitize_output(result.rewritten_prompt)
                    if len(prompt_preview) > 200:
                        prompt_preview = prompt_preview[:197] + "..."
                    irc.reply(
                        _("[Rewritten: %s] %s") % (prompt_preview, sanitized_content),
                    )
                else:
                    irc.reply(sanitized_content)

            self._store_context_and_log_usage(
                nick,
                channel,
                "draw",
                text,
                f"[Generated image: {result.content}]",
                result,
                irc,
                msg,
            )

    draw = wrap(draw, [("checkCapability", "llm.draw"), "text"])

    def animate(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<prompt>

        Generate a short video from a text description.
        Requires NickServ identification.

        Examples:
          %animate A cat playing with a ball of yarn
          %animate A timelapse of a sunset over the ocean
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        pf = self._run_preflight(
            irc,
            msg,
            text,
            "animate",
            require_account=True,
            apply_rate_limit=True,
        )
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        with self._trace_request("animate", nick, channel):
            with self._allow_concurrent():
                result = self.llm_service.video_generation(text, irc=irc, msg=msg)
                self.log.info("replying to %s/%s", channel, nick)
                sanitized_content = self.llm_service.sanitize_output(result.content)
                irc.reply(sanitized_content)

            self._store_context_and_log_usage(
                nick,
                channel,
                "animate",
                text,
                f"[Generated video: {result.content}]",
                result,
                irc,
                msg,
            )

    animate = wrap(animate, [("checkCapability", "llm.animate"), "text"])

    # Alias: %video works the same as %animate
    video = animate

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
        nick = self._get_identity(irc, msg)
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
        animate_key = self.registryValue("animateApiKey")

        # Safely display each key
        ask_status = self.llm_service.safe_key_display(ask_key)
        code_status = self.llm_service.safe_key_display(code_key)
        draw_status = self.llm_service.safe_key_display(draw_key)
        animate_status = self.llm_service.safe_key_display(animate_key)

        # Build response
        response = _("API Key Status: ask=%s, code=%s, draw=%s, animate=%s") % (
            ask_status,
            code_status,
            draw_status,
            animate_status,
        )

        # Send as private message for extra security
        irc.reply(response, private=True)

    llmkeys = wrap(llmkeys, ["admin"])

    def flag(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        nick: str,
        reason: str,
    ) -> None:
        """<nick> <reason>

        Flag a user account for abuse. Resolves nick to NickServ account.
        Flagged users are blocked from using bot commands.
        """
        try:
            target_account = irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            target_account = None
        if not target_account:
            irc.error(
                _("Cannot resolve %s to a NickServ account. User must be online and identified.")
                % nick
            )
            return

        created = self.db.flag_user(target_account, reason, auto_flagged=False)
        if created:
            irc.reply(_("Flagged %s (%s).") % (nick, target_account), private=True)
        else:
            irc.reply(_("%s is already flagged.") % target_account, private=True)

    flag = wrap(flag, ["admin", "nick", "text"])

    def unflag(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        nick: str,
    ) -> None:
        """<nick>

        Remove the abuse flag from a user account.
        """
        try:
            target_account = irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            target_account = None
        if not target_account:
            irc.error(_("Cannot resolve %s to a NickServ account.") % nick)
            return

        admin_account = self._get_identity(irc, msg)
        result = self.db.unflag_user(target_account, admin_account)
        if result:
            irc.reply(_("Unflagged %s (%s).") % (nick, target_account), private=True)
        else:
            irc.reply(_("%s is not currently flagged.") % target_account, private=True)

    unflag = wrap(unflag, ["admin", "nick"])

    def flagged(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        List all currently flagged user accounts.
        """
        users = self.db.get_flagged_users()
        if not users:
            irc.reply(_("No flagged users."), private=True)
            return

        lines = []
        for u in users:
            flag_type = "auto" if u.auto_flagged else "manual"
            lines.append(f"{u.account} ({flag_type}): {u.reason}")
        irc.reply(" | ".join(lines), private=True)

    flagged = wrap(flagged, ["admin"])

    def usage(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """[<nick or #channel>]

        Show API usage statistics.

        No argument in a channel: shows channel stats and your personal stats.
        No argument via PM: shows global overview (admin only).
        <nick>: shows that user's stats (scoped to current channel if in one).
        <#channel>: shows that channel's stats.
        """
        # Extract the raw target from the IRC message instead of using
        # Limnoria's tokenized args.  The tokenizer treats "[" as nested-
        # command syntax, so "Rubin[F]" is evaluated as a nested command
        # before the args reach us — _extract_raw_arg bypasses that.
        target: str | None = self._extract_raw_arg(irc, msg, "usage")
        args[:] = []  # consume all tokens

        # Strip IRC status prefixes (@op, +voice, %halfop) from nick targets
        if target and not ircutils.isChannel(target):
            target = target.lstrip("@+%")
        if target and ircutils.isChannel(target):
            self._usage_for_channel(irc, msg, target)
        elif target:
            self._usage_for_nick(irc, msg, target)
        elif msg.channel:
            self._usage_channel(irc, msg)
        else:
            # PM mode requires admin
            if not ircdb.checkCapability(msg.prefix, "admin"):
                irc.error(_("You need the 'admin' capability to view global usage stats."))
                return
            self._usage_global(irc, msg)

    def _usage_global(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Show global usage overview via PM (admin only)."""
        # Today: midnight UTC
        today_midnight = (
            datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        )

        # This month: first of month midnight UTC
        month_start = self._month_start_ts()

        today = self.db.get_usage_summary(since=today_midnight)
        month = self.db.get_usage_summary(since=month_start)
        top_users = self.db.get_usage_by_nick(since=month_start, limit=5)
        top_channels = self.db.get_usage_by_channel(since=month_start, limit=5)

        # Format response
        parts = []
        parts.append(f"Today: ${today.total_cost:.4f} ({today.total_requests} requests)")
        parts.append(f"This month: ${month.total_cost:.4f} ({month.total_requests} requests)")

        if top_users:
            user_parts = [f"{u.name} ${u.total_cost:.4f}" for u in top_users]
            parts.append(f"Top users: {', '.join(user_parts)}")

        if top_channels:
            chan_parts = [f"{c.name} ${c.total_cost:.4f}" for c in top_channels]
            parts.append(f"Top channels: {', '.join(chan_parts)}")

        irc.reply(" | ".join(parts), private=True)

    def _usage_channel(self, irc: callbacks.Irc, msg: IrcMsg) -> None:
        """Show channel and personal usage stats in-channel."""
        channel = msg.channel
        nick = self._get_identity(irc, msg)

        # This month: first of month midnight UTC
        month_start = self._month_start_ts()

        chan_summary = self.db.get_usage_summary_for_channel(channel, since=month_start)
        nick_summary = self.db.get_usage_summary_for_nick(nick, since=month_start, channel=channel)
        chan_rank = self.db.get_channel_rank(channel, since=month_start)
        nick_rank = self.db.get_nick_rank(nick, since=month_start, channel=channel)

        # Format channel part
        chan_part = f"{channel} this month: ${chan_summary.total_cost:.4f}"
        chan_part += f" ({chan_summary.total_requests} requests"
        if chan_rank.rank > 0:
            chan_part += f", rank {chan_rank.rank}/{chan_rank.total} channels"
        chan_part += ")"

        # Format personal part
        nick_part = f"You: ${nick_summary.total_cost:.4f}"
        nick_part += f" ({nick_summary.total_requests} requests"
        if nick_rank.rank > 0:
            nick_part += f", rank {nick_rank.rank}/{nick_rank.total} users"
        nick_part += ")"

        irc.reply(f"{chan_part} | {nick_part}", prefixNick=False)

    def _usage_for_nick(self, irc: callbacks.Irc, msg: IrcMsg, nick: str) -> None:
        """Show usage stats for a specific nick.

        Resolves the target nick to a NickServ account before querying the
        database, so ``%usage OldNick`` finds stats logged under the account.
        The display still uses the nick the caller typed.
        """
        channel = msg.channel

        # Resolve target nick → account for the DB query
        identity = self._resolve_nick_to_identity(irc, nick)

        month_start = self._month_start_ts()

        nick_summary = self.db.get_usage_summary_for_nick(
            identity, since=month_start, channel=channel
        )
        nick_rank = self.db.get_nick_rank(identity, since=month_start, channel=channel)

        scope = f" in {channel}" if channel else ""
        nick_part = f"{nick}{scope} this month: ${nick_summary.total_cost:.4f}"
        nick_part += f" ({nick_summary.total_requests} requests"
        if nick_rank.rank > 0:
            nick_part += f", rank {nick_rank.rank}/{nick_rank.total} users"
        nick_part += ")"

        irc.reply(nick_part, prefixNick=False)

    def _usage_for_channel(self, irc: callbacks.Irc, msg: IrcMsg, channel: str) -> None:
        """Show usage stats for a specific channel."""
        month_start = self._month_start_ts()

        chan_summary = self.db.get_usage_summary_for_channel(channel, since=month_start)
        chan_rank = self.db.get_channel_rank(channel, since=month_start)

        chan_part = f"{channel} this month: ${chan_summary.total_cost:.4f}"
        chan_part += f" ({chan_summary.total_requests} requests"
        if chan_rank.rank > 0:
            chan_part += f", rank {chan_rank.rank}/{chan_rank.total} channels"
        chan_part += ")"

        irc.reply(chan_part, prefixNick=False)

    # Reminder helper methods (testable without Limnoria wrap decorator)

    def _get_user_reminders(self, nick: str) -> list[tuple[str, tuple[str, str, str]]]:
        """Get reminders belonging to a specific user.

        Args:
            nick: User's nick

        Returns:
            List of (event_name, (nick, channel, message)) tuples
        """
        with self._reminders_lock:
            return [
                (name, data)
                for name, data in self._reminders.items()
                if data[0].lower() == nick.lower()
            ]

    def _format_reminders(self, reminders: list[tuple[str, tuple[str, str, str]]]) -> str:
        """Format reminders list for display.

        Args:
            reminders: List of (event_name, (nick, channel, message)) tuples

        Returns:
            Formatted string for IRC display
        """
        parts = []
        for name, (_, _, message) in reminders:
            # Truncate long messages
            preview = message[:40] + "..." if len(message) > 40 else message
            # Extract ID from event name
            reminder_id = name.split("_")[-1]
            parts.append(f"#{reminder_id}: {preview}")
        return " | ".join(parts)

    def _find_user_reminder(self, nick: str, reminder_id: str) -> str | None:
        """Find a reminder event name by ID for a specific user.

        Args:
            nick: User's nick
            reminder_id: Reminder ID (last part of event name)

        Returns:
            Event name if found and owned by user, None otherwise
        """
        with self._reminders_lock:
            for name, (owner, _, _) in self._reminders.items():
                if name.endswith(f"_{reminder_id}") and owner.lower() == nick.lower():
                    return name
            return None

    def remindme(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<natural language reminder>

        Set a reminder using natural language.

        Examples:
          %remindme in 30 minutes check the build
          %remindme tomorrow at 3pm call Bob
          %remindme in 2 hours meeting starts
          %remindme next Tuesday morning dentist appointment
        """
        nick = self._get_identity(irc, msg)
        channel = self._get_channel(msg)

        with self._trace_request("remindme", nick, channel):
            # Parse reminder using LLM (release lock during blocking API call)
            with self._allow_concurrent():
                result = self.llm_service.parse_reminder(text, channel)

            # Handle clarification requests
            if result.action == "clarify":
                irc.reply(result.confirmation)
                return

            # Validate duration limits
            if result.seconds is None:
                irc.reply(_("I couldn't determine when to remind you. Please try again."))
                return

            if result.seconds < 10:
                irc.error(_("Reminder must be at least 10 seconds from now."))
                return

            if result.seconds > 604800:  # 7 days
                irc.error(_("Reminder can't be more than 7 days out."))
                return

            # Schedule the reminder
            reminder_message = result.message or text
            event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
            deliver = self._make_reminder_delivery_closure(
                nick, channel, reminder_message, event_name
            )

            try:
                schedule.addEvent(deliver, time.time() + result.seconds, name=event_name)
                with self._reminders_lock:
                    self._reminders[event_name] = (nick, channel, reminder_message)

                # Persist to database
                self.db.save_reminder(
                    event_name,
                    nick,
                    channel,
                    reminder_message,
                    time.time() + result.seconds,
                )

                # Build reply with optional note
                reply = self.llm_service.sanitize_output(result.confirmation)
                if result.note:
                    reply = f"{reply} ({self.llm_service.sanitize_output(result.note)})"
                irc.reply(reply)
            except Exception as e:
                self.log.error("Failed to schedule reminder: %s", e)
                irc.error(_("Failed to set reminder."))

    remindme = wrap(remindme, ["text"])

    def reminders(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        List your pending reminders.
        """
        nick = self._get_identity(irc, msg)
        user_reminders = self._get_user_reminders(nick)

        if not user_reminders:
            irc.reply(_("You have no pending reminders."))
            return

        irc.reply(self._format_reminders(user_reminders))

    reminders = wrap(reminders, [])

    def unremind(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        reminder_id: str,
    ) -> None:
        """<id>

        Cancel a reminder by ID (shown in %reminders).
        """
        nick = self._get_identity(irc, msg)
        target = self._find_user_reminder(nick, reminder_id)

        if not target:
            irc.error(_("Reminder not found or not yours."))
            return

        with contextlib.suppress(KeyError):
            schedule.removeEvent(target)
        with self._reminders_lock:
            self._reminders.pop(target, None)
        self.db.delete_reminder(target)
        irc.reply(_("Reminder cancelled."))

    unremind = wrap(unremind, ["somethingWithoutSpaces"])


Class = LLM
