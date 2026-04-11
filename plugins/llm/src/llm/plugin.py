"""LLM plugin implementation."""

from __future__ import annotations

import collections
import contextlib
import logging
import mimetypes
import random
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
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
    MetaResult,
)
from .tracing import TraceFilter, generate_request_id, request_id

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

_ = PluginInternationalization("LLM")

# Icon shown when Google grounding/search was used in the response
GROUNDING_ICON = "\U0001f310"  # 🌐 (globe with meridians)

# Commands that support long-term memory extraction
_MEMORY_COMMANDS = frozenset({"ask", "code"})

# C0 control characters except TAB (\x09), LF (\x0a), CR (\x0d).
# Includes ESC (\x1b) which starts ANSI sequences like \x1b[6n whose
# brackets crash Limnoria's nested-command tokenizer.
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PreflightResult(NamedTuple):
    """Result of the shared command preflight check.

    ``blocked`` is True when the command should not proceed (the preflight
    already sent the appropriate error reply and logged usage).
    """

    blocked: bool
    nick: str  # account-resolved identity for logging
    channel: str
    account: str | None  # NickServ account, or None if unidentified


@dataclass(frozen=True)
class CommandInfo:
    """Metadata for a user-facing command, used to generate help."""

    name: str
    args: str
    description: str
    examples: tuple[str, ...]
    category: str  # "generation", "memory", "utility"


COMMAND_REGISTRY: tuple[CommandInfo, ...] = (
    CommandInfo(
        name="ask",
        args="<question>",
        description=(
            "Ask the AI a question. Supports conversation context "
            "(follow-up questions) and vision (include image URLs)."
        ),
        examples=(
            "%ask What is the capital of France?",
            "%ask Describe this: https://example.com/image.jpg",
            "%ask And what about Germany?  (follow-up using context)",
        ),
        category="generation",
    ),
    CommandInfo(
        name="code",
        args="<request>",
        description=(
            "Generate code based on your request. "
            "Code is saved to an HTTP link with syntax highlighting."
        ),
        examples=(
            "%code Python function to calculate fibonacci numbers",
            "%code Now add memoization to that",
        ),
        category="generation",
    ),
    CommandInfo(
        name="draw",
        args="<prompt>",
        description="Generate an image from a text description.",
        examples=(
            "%draw A sunset over mountains in watercolor style",
            "%draw A cyberpunk cityscape at night",
        ),
        category="generation",
    ),
    CommandInfo(
        name="forget",
        args="[channel]",
        description=(
            "Clear your volatile memory (conversation context) "
            "for the current or specified channel."
        ),
        examples=("%forget", "%forget #channel"),
        category="memory",
    ),
    CommandInfo(
        name="memories",
        args="[del <id> | edit <id> <text> | clear | cleanup]",
        description=(
            "Manage your non-volatile memory (stored facts the bot "
            "remembers about you across conversations)."
        ),
        examples=(
            "%memories",
            "%memories delete 3",
            "%memories edit 5 corrected fact",
            "%memories clear",
        ),
        category="memory",
    ),
    CommandInfo(
        name="instruct",
        args="[<instruction> | clear]",
        description=(
            "Set persistent instructions that shape how %ask responds to you. "
            "Your instruction is prepended to the system prompt."
        ),
        examples=(
            "%instruct You are Captain Picard. Respond in character.",
            "%instruct Respond only in haiku",
            "%instruct clear",
            "%instruct",
        ),
        category="memory",
    ),
    CommandInfo(
        name="remind",
        args="[<text> | list | del <id> | clear]",
        description="Set and manage reminders using natural language.",
        examples=(
            "%remind in 30 minutes check the build",
            "%remind list",
            "%remind delete abc1",
            "%remind clear",
        ),
        category="utility",
    ),
    CommandInfo(
        name="usage",
        args="[nick | #channel]",
        description="Show API usage statistics.",
        examples=("%usage", "%usage someone", "%usage #channel"),
        category="utility",
    ),
    CommandInfo(
        name="meta",
        args="<request>",
        description=(
            "Manage your settings with natural language (instructions, memories, context)."
        ),
        examples=(
            "%meta always respond in haiku",
            "%meta what are my memories?",
            "%meta delete any memories about cats",
            "%meta clear my conversation context",
        ),
        category="utility",
    ),
)


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

        # No index page — help docs are on GitHub Pages
        if path == "":
            handler.send_response(404)
            handler.end_headers()
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

        # Initialize database for persistence (before context, which loads from DB)
        db_path = self.registryValue("databasePath")
        if not db_path:
            db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
        self.db = LLMDatabase(db_path)

        # Initialize conversation context (loads persisted conversations from DB)
        self._init_context()

        # Track nicks already migrated to account-based identity this session
        self._migrated_nicks: set[str] = set()

        # In-memory per-command rate-limit buckets: "{command}:{account}" -> deque of timestamps
        self._rate_buckets: dict[str, collections.deque[float]] = {}

        # Reminder storage: event_name -> (nick, channel, message)
        self._reminders: dict[str, tuple[str, str, str]] = {}
        self._reminders_lock = threading.Lock()

        # Spontaneous participation cooldown tracking: channel -> last_fire_timestamp
        self._spontaneous_cooldowns: dict[str, float] = {}

        # Pending spontaneous schedule events (cancelled on unload)
        self._spontaneous_events: set[str] = set()

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

        # Cancel pending spontaneous events and clear cooldowns
        if hasattr(self, "_spontaneous_events"):
            for event_name in list(self._spontaneous_events):
                with contextlib.suppress(KeyError):
                    schedule.removeEvent(event_name)
            self._spontaneous_events.clear()
        if hasattr(self, "_spontaneous_cooldowns"):
            self._spontaneous_cooldowns.clear()

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

    def inFilter(self, irc: callbacks.Irc, msg: IrcMsg) -> IrcMsg:  # noqa: N802
        """Sanitize PRIVMSG text before Limnoria's tokenizer processes it.

        Limnoria's command tokenizer interprets ``[…]`` as nested-command
        syntax and raises ``SyntaxError`` on unmatched brackets.  Messages
        containing ANSI escape sequences (e.g. ``\\x1b[6n``) or casual
        bracket use (e.g. ``array[0``) crash the tokenizer before
        ``invalidCommand`` ever runs.

        This filter:
        1. Strips C0 control characters (except TAB/LF/CR) — removes the
           ESC byte from ANSI sequences.
        2. Replaces ``[`` and ``]`` with full-width equivalents when brackets
           are unbalanced — prevents the tokenizer crash while keeping the
           text readable for the LLM.
        """
        if msg.command != "PRIVMSG" or len(msg.args) < 2:
            return msg

        text = msg.args[1]
        cleaned = _CTRL_CHAR_RE.sub("", text)

        # Escape unbalanced brackets that would crash the tokenizer
        if cleaned.count("[") != cleaned.count("]"):
            cleaned = cleaned.replace("[", "\uff3b").replace("]", "\uff3d")

        if cleaned != text:
            return ircmsgs.IrcMsg(msg=msg, args=(msg.args[0], cleaned))
        return msg

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

        display_nick = msg.nick
        identity = self._get_identity(irc, msg)
        message_text = msg.args[1] if len(msg.args) > 1 else ""

        # Store in conversation context for richer follow-up questions
        # Use display nick for channel context (what the LLM sees) so it
        # addresses people by their visible IRC name, not their account name.
        ctx_cfg = self._get_context_config(channel)
        self.context.add_message(
            identity, channel, Role.USER, message_text, config=ctx_cfg, persist=False
        )
        self.context.add_channel_message(
            channel, display_nick, Role.USER, message_text, config=ctx_cfg
        )

        # Spontaneous participation
        if self.registryValue("spontaneousEnabled", channel):
            cooldown_minutes = self.registryValue("spontaneousCooldown", channel)
            last_spontaneous = self._spontaneous_cooldowns.get(channel, 0)
            if time.time() - last_spontaneous >= cooldown_minutes * 60:
                chance = self.registryValue("spontaneousChance", channel)
                if random.randint(1, 100) <= chance:
                    self._spontaneous_cooldowns[channel] = time.time()
                    self._schedule_spontaneous(irc, channel, identity, message_text)

    def _schedule_spontaneous(
        self, irc: callbacks.Irc, channel: str, trigger_nick: str, trigger_text: str
    ) -> None:
        """Schedule a spontaneous reply evaluation.

        Queues a short-delayed event that reads channel history, asks the LLM
        whether it wants to participate, and sends a message if the LLM does
        not PASS.

        Args:
            irc: IRC connection object
            channel: Channel to potentially respond in
            trigger_nick: Identity of the user whose message triggered this
            trigger_text: The message text that triggered this
        """

        def _evaluate() -> None:
            try:
                channel_msgs = self.context.get_channel_messages(channel)
                if not channel_msgs:
                    return

                api_key = self.registryValue("spontaneousApiKey")
                if not api_key:
                    api_key = self.registryValue("askApiKey")
                if not api_key:
                    return

                model = self.registryValue("spontaneousModel", channel)
                system_prompt = self.registryValue("spontaneousSystemPrompt", channel)

                prompt = "Respond to the conversation above, or say PASS."
                result = self.llm_service.completion(
                    prompt,
                    command="ask",
                    channel_history=channel_msgs,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    model_override=model,
                )

                if result.error or result.content.strip().upper() == "PASS":
                    return

                response = result.content
                action_text = self._extract_action(irc, response)
                if action_text:
                    irc.queueMsg(ircmsgs.action(channel, action_text))
                else:
                    irc.queueMsg(ircmsgs.privmsg(channel, response))

                self.db.log_usage(
                    irc.nick,
                    channel,
                    "spontaneous",
                    result.model,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.cost,
                    prompt="[spontaneous]",
                    status="success",
                )

                # Extract memories from the triggering user's message
                self._schedule_memory_extraction(trigger_nick, channel, trigger_text, response)
            except Exception:
                log.exception("Spontaneous evaluation failed for %s", channel)
            finally:
                self._spontaneous_events.discard(event_name)

        event_name = f"llm_spontaneous_{uuid.uuid4().hex[:8]}"
        self._spontaneous_events.add(event_name)
        schedule.addEvent(_evaluate, time.time() + 0.5, name=event_name)

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
        self.context = ConversationContext(config, db=self.db)

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
            channel: Channel to deliver to (or nick for PM delivery)
            message: Reminder message
            event_name: Scheduler event name for cleanup

        Returns:
            Callable for use with schedule.addEvent
        """
        lock = self._reminders_lock
        # If the command was sent via PM, channel is the bot's own nick.
        # Deliver to the user's nick instead.
        target = channel if ircutils.isChannel(channel) else nick

        def _deliver() -> None:
            try:
                for active_irc in world.ircs:
                    safe_message = self.llm_service.sanitize_output(message)
                    active_irc.queueMsg(
                        ircmsgs.privmsg(target, f"{nick}: Reminder: {safe_message}")
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

    def getPluginHelp(self) -> str:  # noqa: N802
        """Return plugin help with documentation URL."""
        url = self.registryValue("helpUrl")
        names = ", ".join(cmd.name for cmd in COMMAND_REGISTRY)
        return _("AI-powered commands using LiteLLM. Commands: %s. Full documentation: %s") % (
            names,
            url,
        )

    def invalidCommand(  # noqa: N802
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        tokens: list[str],
    ) -> None:
        """Handle unrecognized commands: try meta first, fall back to ask.

        When someone says "vibebot always respond in haiku" without a command:
        1. If metaEnabled, route through meta handler
        2. If meta returns NOT_META (not a config request), fall through to ask
        3. If metaEnabled is False, go straight to ask
        """
        if not tokens:
            return

        # Check if user has ask capability
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return

        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        channel = self._get_channel(msg)

        # Try meta handler first (if enabled)
        if self.registryValue("metaEnabled", channel):
            text = " ".join(tokens)
            # Use "ask" for rate limiting — meta shares the ask tier
            preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
            if preflight.blocked:
                return  # Rate limited — do not fall through to ask

            result: MetaResult = self.llm_service.meta_completion(
                prompt=text,
                nick=preflight.nick,
                channel=preflight.channel,
                db=self.db,
                context=self.context,
                bot_nick=irc.nick,
            )

            if result.is_meta:
                # Meta handled it — relay the response
                if result.content:
                    irc.reply(result.content, prefixNick=False)
                self.db.log_usage(
                    preflight.nick,
                    preflight.channel,
                    "meta",
                    result.model,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.cost,
                )
                return

            # NOT_META — fall through to ask below

        # Fall through to ask (original behavior)
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

    def _extract_action(self, irc: callbacks.Irc, response: str) -> str | None:
        """Return action text if *response* looks like an IRC action, else ``None``.

        Recognises both ``/me does something`` and ``* BotNick does something``.
        """
        if response.startswith("/me ") and len(response) > 4:
            return response[4:]
        star_prefix = f"* {irc.nick} "
        if response.startswith(star_prefix) and len(response) > len(star_prefix):
            return response[len(star_prefix) :]
        return None

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
    ) -> PreflightResult:
        """Shared preflight check for all commands.

        Runs the following sequence:
        1. Account resolution (required or optional depending on command).
        2. Flagged-user block check.
        3. Tier resolution (owner/admin exempt, then trusted/registered/unregistered).
        4. Per-command, per-tier rate-limit check.

        When any check fails the method sends the appropriate IRC error,
        logs usage with the blocked status, and returns ``blocked=True``.

        Args:
            irc: IRC connection.
            msg: IRC message.
            text: User's prompt text (for usage logging).
            command: Command name (ask, code, draw).
            require_account: If True, NickServ identification is mandatory.

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

        # --- tier-based rate limit check ---
        tier = self._resolve_tier(irc, msg)
        # Owner and admin are always exempt from rate limits
        if tier not in ("owner", "admin"):
            identity = account or nick
            if self._check_rate_limit(irc, command, identity, nick, channel, text, tier=tier):
                return PreflightResult(blocked=True, nick=nick, channel=channel, account=account)

        return PreflightResult(blocked=False, nick=nick, channel=channel, account=account)

    def _is_rate_limited(self, command: str, account: str, now: float, *, tier: str) -> bool:
        """Check if a user exceeds the per-command rate limit.

        Evicts timestamps outside the configured window before checking.

        Args:
            command: Command name (ask, code, or draw).
            account: NickServ account name or nick-based identity.
            now: Current time (seconds since epoch).
            tier: User tier (trusted, registered, unregistered).

        Returns:
            True if the user has exceeded the rate limit.
        """
        max_count, window = self._get_tier_limits(command, tier)

        # count=0 means rate limiting is disabled for this tier
        if max_count == 0:
            return False

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
        *,
        tier: str,
    ) -> bool:
        """Check rate limit and send error if exceeded.

        Always records the request timestamp. When enforceRateLimits is True
        and the user is over the limit, sends an error reply and logs
        ``status="rate_limited"``.

        Args:
            irc: IRC connection.
            command: Command name.
            account: NickServ account name or nick-based identity.
            nick: Resolved identity for logging.
            channel: Channel name.
            text: Prompt text for logging.
            tier: User tier (trusted, registered, unregistered).

        Returns:
            True if the request should be blocked.
        """
        now = time.time()
        over_limit = self._is_rate_limited(command, account, now, tier=tier)

        # Always record the hit (so the window tracks correctly)
        self._record_rate_limit_hit(command, account, now)

        if over_limit:
            enforce = self.registryValue("enforceRateLimits")
            max_count, window = self._get_tier_limits(command, tier)
            key = f"{command}:{account}"
            count = len(self._rate_buckets.get(key, ()))
            if enforce:
                self.log.info(
                    "rate_limited command=%s account=%s tier=%s count=%d limit=%d window=%ss",
                    command,
                    account,
                    tier,
                    count,
                    max_count,
                    window,
                )
                irc.error(_("Rate limit exceeded for %s. Try again in %ds.") % (command, window))
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
                "rate_limit_shadow command=%s account=%s tier=%s count=%d limit=%d window=%ss",
                command,
                account,
                tier,
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

    # Tier config key prefixes: tier -> config infix
    _TIER_CONFIG_PREFIX = {
        "trusted": "Trusted",
        "unregistered": "Unreg",
        "registered": "",  # base config (no prefix)
    }

    def _resolve_tier(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
        """Classify a user into a rate-limit tier based on Limnoria capabilities.

        Checks capabilities from most to least privileged.

        Args:
            irc: IRC connection (for account lookup).
            msg: IRC message (uses msg.prefix for capability check).

        Returns:
            One of: "owner", "admin", "trusted", "registered", "unregistered".
        """
        prefix = msg.prefix
        if ircdb.checkCapability(prefix, "owner"):
            return "owner"
        if ircdb.checkCapability(prefix, "admin"):
            return "admin"
        if ircdb.checkCapability(prefix, "trusted"):
            return "trusted"
        nick = ircutils.nickFromHostmask(prefix)
        try:
            account = irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            account = None
        return "registered" if account else "unregistered"

    def _get_tier_limits(self, command: str, tier: str) -> tuple[int, int]:
        """Look up rate limit count and window for a command+tier.

        Args:
            command: Command name (ask, code, draw).
            tier: User tier (trusted, registered, unregistered).

        Returns:
            (max_count, window_seconds). max_count=0 means disabled.
        """
        infix = self._TIER_CONFIG_PREFIX.get(tier, "")
        count_key = f"{command}{infix}RateLimitCount"
        window_key = f"{command}{infix}RateLimitWindow"
        return self.registryValue(count_key), self.registryValue(window_key)

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

    def _get_user_memories(self, nick: str) -> list[str]:
        """Get memory facts for a user as a list of strings."""
        if self.db is None:
            return []
        rows = self.db.get_memories(nick)
        return [row.fact for row in rows]

    def _schedule_memory_extraction(
        self, nick: str, channel: str, user_text: str, assistant_response: str
    ) -> None:
        """Schedule background memory extraction for a user interaction.

        Args:
            nick: User's resolved identity
            channel: Channel where the interaction happened
            user_text: What the user said
            assistant_response: What the bot replied
        """
        try:
            if not self.registryValue("memoryEnabled", channel):
                return

            existing_rows = self.db.get_memories(nick)
            existing_facts = [r.fact for r in existing_rows]
            max_memories = self.registryValue("memoryMaxPerUser")

            if len(existing_rows) >= max_memories:
                return

            snapshot_count = len(existing_rows)

            def _extract_memories_bg() -> None:
                try:
                    extraction = self.llm_service.extract_memories(
                        nick, channel, user_text, assistant_response, existing_facts
                    )
                    if not extraction.add:
                        return

                    # Race protection: abort if memories changed during LLM call
                    current = self.db.get_memories(nick)
                    if len(current) != snapshot_count:
                        log.info(
                            "Memory extraction for %s aborted: count changed (%d -> %d)",
                            nick,
                            snapshot_count,
                            len(current),
                        )
                        return

                    # Add new facts (respecting cap)
                    saved: list[str] = []
                    current_count = len(current)
                    for fact in extraction.add:
                        if current_count >= max_memories:
                            break
                        self.db.save_memory(nick, fact, channel)
                        saved.append(fact)
                        current_count += 1

                    if not saved:
                        return

                    # Trigger cleanup if counter reaches interval
                    cleanup_interval = self.registryValue("memoryCleanupInterval")
                    if cleanup_interval > 0:
                        count = self.db.increment_memory_saves(nick)
                        if count >= cleanup_interval:
                            self.db.reset_memory_saves(nick)
                            self._run_memory_cleanup(nick, channel)

                except Exception:
                    log.exception("Memory extraction failed for %s", nick)

            event_name = f"llm_memory_{uuid.uuid4().hex[:8]}"
            schedule.addEvent(_extract_memories_bg, time.time() + 0.1, name=event_name)

        except Exception:
            log.exception("Memory extraction scheduling failed for %s", nick)

    def _run_memory_cleanup(self, nick: str, channel: str) -> str:
        """Run memory cleanup for a user. Returns a summary string."""
        snapshot = self.db.get_memories(nick)
        if len(snapshot) < 2:
            return "Not enough memories to clean up."

        before_count = len(snapshot)
        result = self.llm_service.cleanup_memories(nick, channel, snapshot)

        if result.error:
            log.warning("Memory cleanup failed for %s: %s", nick, result.error)
            short = result.error.split(":")[0] if ":" in result.error else result.error
            return f"Cleanup failed ({short}). Try again later."

        # Abort if memory count changed during LLM call (race protection)
        current = self.db.get_memories(nick)
        if len(current) != len(snapshot):
            return "Cleanup skipped — memories changed while processing."

        # Apply drops
        dropped = 0
        for idx in result.drop:
            if 0 <= idx < len(snapshot):
                self.db.delete_memory(nick, snapshot[idx].id)
                dropped += 1

        # Apply merges: delete sources, insert merged fact
        merged = 0
        merged_sources = 0
        for entry in result.merge:
            sources = [snapshot[i] for i in entry.indices if 0 <= i < len(snapshot)]
            if not sources:
                continue
            oldest = min(sources, key=lambda s: s.created_at)
            for source in sources:
                self.db.delete_memory(nick, source.id)
            self.db.save_memory(nick, entry.text, oldest.source_channel)
            merged += 1
            merged_sources += len(sources)
        after_count = before_count - dropped - merged_sources + merged

        parts = [f"Before: {before_count}"]
        if dropped:
            parts.append(f"dropped: {dropped}")
        if merged:
            parts.append(f"merged: {merged_sources} → {merged}")
        parts.append(f"after: {after_count}")
        return " | ".join(parts)

    def _store_context_and_log_usage(
        self,
        nick: str,
        channel: str,
        command: str,
        text: str,
        response: str,
        result: CompletionResult | ImageResult,
        irc: callbacks.Irc,
        msg: IrcMsg,
    ) -> None:
        """Store conversation context and log API usage for a command.

        Shared between all commands (ask, code, draw).

        Args:
            nick: User's nick
            channel: Channel name
            command: Command name ("ask", "code", or "draw")
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

        # Schedule background memory extraction for eligible commands
        if command in _MEMORY_COMMANDS and result.error is None:
            self._schedule_memory_extraction(nick, channel, text, result.content)

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

            memories = self._get_user_memories(nick)
            user_instruction = self.db.get_instruction(nick)

            # Build system prompt with optional user instruction
            ask_prompt = self.registryValue("askSystemPrompt", channel)
            effective_prompt = f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None

            with self._allow_concurrent():
                if images:
                    # Clean prompt by removing image URLs
                    clean_prompt = text
                    for img in images:
                        clean_prompt = clean_prompt.replace(img, "").strip()

                    result = self.llm_service.completion(
                        clean_prompt,
                        command="ask",
                        images=images,
                        history=history,
                        channel_history=channel_history,
                        irc=irc,
                        msg=msg,
                        memories=memories,
                        system_prompt=effective_prompt,
                    )
                else:
                    result = self.llm_service.completion(
                        text,
                        command="ask",
                        history=history,
                        channel_history=channel_history,
                        irc=irc,
                        msg=msg,
                        memories=memories,
                        system_prompt=effective_prompt,
                    )

                # Format response with grounding icon if search was used
                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                action_text = self._extract_action(irc, response)
                if action_text:
                    if result.grounding_used:
                        action_text = f"{GROUNDING_ICON} {action_text}"
                    self.log.info("sending action to %s/%s", channel, nick)
                    target = msg.args[0]
                    irc.queueMsg(ircmsgs.action(target, action_text))
                    # Store context as "* BotNick action_text" so follow-ups
                    # understand the bot emoted rather than said something
                    response = f"* {irc.nick} {action_text}"
                else:
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

            memories = self._get_user_memories(nick)

            with self._allow_concurrent():
                result = self.llm_service.completion(
                    text,
                    command="code",
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=memories,
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

    def forget(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str | None,
    ) -> None:
        """[<channel>]

        Clear your volatile memory (conversation context) for the current or specified
        channel. Use this to start fresh. Volatile memory expires automatically after a
        timeout.
        """
        nick = self._get_identity(irc, msg)
        # Default to current channel if not specified
        if channel is None:
            channel = self._get_channel(msg)
        self.context.clear(nick, channel)
        irc.reply(_("Context cleared."), prefixNick=False)

    forget = wrap(forget, [optional("channel")])

    def memories(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<nick> | del(ete) <id> [<id>...] | edit <id> <text> | clear | cleanup [nick]]

        Manage your non-volatile memory (stored facts the bot remembers about you
        across conversations). Use 'delete <id> [<id>...]' to remove one or more
        memories, 'edit <id> <text>' to update one, 'clear' to remove all, or
        'cleanup' to trigger a cleanup pass. Bot owners can use 'memories <nick>'
        or 'memories cleanup <nick>' for other users.
        """
        nick = self._get_identity(irc, msg)

        if not text:
            # List own memories (newest first)
            self._memories_list(irc, nick, nick)
            return

        parts = text.split(None, 2)
        subcommand = parts[0].lower()

        if subcommand == "clear":
            count = self.db.delete_all_memories(nick)
            label = "memory" if count == 1 else "memories"
            irc.reply(f"Cleared {count} {label}.", prefixNick=False)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            try:
                memory_ids = [int(x) for x in raw_ids]
            except ValueError:
                irc.reply("Usage: memories delete <id> [<id> ...]", prefixNick=False)
                return
            deleted = sum(1 for mid in memory_ids if self.db.delete_memory(nick, mid))
            if deleted == 0:
                irc.error("No matching memories found.")
            elif deleted == 1:
                irc.reply("Memory deleted.", prefixNick=False)
            else:
                irc.reply(f"Deleted {deleted} memories.", prefixNick=False)

        elif subcommand == "edit" and len(parts) == 3:
            try:
                memory_id = int(parts[1])
            except ValueError:
                irc.reply("Usage: memories edit <id> <new text>", prefixNick=False)
                return
            new_text = parts[2].strip()
            if not new_text:
                irc.reply("Usage: memories edit <id> <new text>", prefixNick=False)
                return
            if self.db.update_memory(nick, memory_id, new_text):
                irc.reply("Memory updated.", prefixNick=False)
            else:
                irc.error("Memory not found or doesn't belong to you.")

        elif subcommand == "cleanup":
            # cleanup [nick] — nick requires owner
            if len(parts) >= 2:
                if not ircdb.checkCapability(msg.prefix, "owner"):
                    irc.error("Only bot owners can clean up other users' memories.")
                    return
                target = parts[1]
            else:
                target = nick
            channel = msg.channel or msg.args[0] if msg.args else "#unknown"
            summary = self._run_memory_cleanup(target, channel)
            irc.reply(summary, prefixNick=False)

        elif len(parts) == 1:
            # Owner viewing another user's memories
            if not ircdb.checkCapability(msg.prefix, "owner"):
                irc.reply(
                    "Usage: memories [del <id> | edit <id> <text> | clear | cleanup]",
                    prefixNick=False,
                )
                return
            target = parts[0]
            self._memories_list(irc, target, target)

        else:
            irc.reply(
                "Usage: memories [del <id> | edit <id> <text> | clear | cleanup]",
                prefixNick=False,
            )

    def _memories_list(self, irc: callbacks.Irc, nick: str, display_name: str) -> None:
        """List memories for a user using Limnoria's built-in pagination."""
        rows = self.db.get_memories(nick)
        if not rows:
            irc.reply(f"No memories stored for {display_name}.", prefixNick=False)
            return
        items = [f"[{r.id}] {r.fact}" for r in rows]
        irc.replies(items, joiner=" | ", prefixNick=False)

    memories = wrap(memories, [optional("text")])

    def instruct(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<instruction> | clear]

        Set persistent instructions that shape how %ask responds to you.
        Your instruction is prepended to the system prompt for every %ask call.

        Examples:
          %instruct You are Captain Picard. Respond in character.
          %instruct Respond only in haiku
          %instruct clear
          %instruct          (show current instruction)
        """
        nick = self._get_identity(irc, msg)

        if not text:
            current = self.db.get_instruction(nick)
            if current:
                irc.reply(f"Current instruction: {current}", prefixNick=False)
            else:
                irc.reply("No instruction set. Use %instruct <text> to set one.", prefixNick=False)
            return

        if text.strip().lower() == "clear":
            if self.db.delete_instruction(nick):
                irc.reply("Instruction cleared.", prefixNick=False)
            else:
                irc.reply("No instruction to clear.", prefixNick=False)
            return

        self.db.save_instruction(nick, text)
        irc.reply("Instruction set.", prefixNick=False)

    instruct = wrap(instruct, [optional("text")])

    @wrap(["text"])
    def meta(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """Manage settings via natural language.

        Uses tool calling to interpret your request and perform the
        appropriate action (set instructions, manage memories, etc.).
        """
        channel = self._get_channel(msg)

        if not self.registryValue("metaEnabled", channel):
            irc.reply(_("The meta command is not enabled in this channel."))
            return

        # Use "ask" for rate limiting — meta shares the ask tier
        preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if preflight.blocked:
            return

        result: MetaResult = self.llm_service.meta_completion(
            prompt=text,
            nick=preflight.nick,
            channel=preflight.channel,
            db=self.db,
            context=self.context,
            bot_nick=irc.nick,
        )

        if result.error:
            self.log.warning("meta command error: %s", result.error)

        if not result.is_meta:
            # Explicit @meta for a non-config request — helpful message
            irc.reply(
                _(
                    "I can manage your instructions, memories, "
                    "and conversation context. Try: @meta list my "
                    "memories"
                ),
                prefixNick=False,
            )
        elif result.content:
            irc.reply(result.content, prefixNick=False)

        # Log usage as "meta" command type
        self.db.log_usage(
            preflight.nick,
            preflight.channel,
            "meta",
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost,
        )

    def usage(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<nick or #channel>]

        Show API usage statistics.

        No argument in a channel: shows channel stats and your personal stats.
        No argument via PM: shows global overview (admin only).
        <nick>: shows that user's stats (scoped to current channel if in one).
        <#channel>: shows that channel's stats.
        """
        target = text.strip() if text else None

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
            if not ircdb.checkCapability(msg.prefix, "admin"):
                irc.error(_("You need the 'admin' capability to view global usage stats."))
                return
            self._usage_global(irc, msg)

    usage = wrap(usage, [optional("text")])

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

        # Global context stats
        ctx_global = self.context.get_stats()
        parts.append(
            f"Context: {ctx_global['active_conversations']} conversations,"
            f" {ctx_global['total_messages']} messages"
        )

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

        # Format context part
        ctx_cfg = self._get_context_config(channel)
        ctx_stats = self.context.get_user_stats(nick, channel, config=ctx_cfg)
        if not ctx_stats["enabled"]:
            ctx_part = "Context: disabled"
        elif ctx_stats["message_count"] == 0:
            ctx_part = "Context: empty"
        else:
            remaining = ctx_stats["seconds_until_expiry"]
            minutes = remaining // 60
            ctx_part = f"Context: {ctx_stats['message_count']}/{ctx_stats['max_messages']} msgs"
            ctx_part += f", expires in {minutes}m" if minutes > 0 else ", expiring soon"

        irc.reply(f"{chan_part} | {nick_part} | {ctx_part}", prefixNick=False)

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

    def _remind_list(self, irc: callbacks.Irc, nick: str) -> None:
        """List pending reminders for a user."""
        user_reminders = self._get_user_reminders(nick)
        if not user_reminders:
            irc.reply(_("You have no pending reminders."))
            return
        irc.reply(self._format_reminders(user_reminders))

    def _remind_set(self, irc: callbacks.Irc, msg: IrcMsg, nick: str, text: str) -> None:
        """Parse and schedule a natural language reminder."""
        channel = self._get_channel(msg)

        with self._trace_request("remind", nick, channel):
            with self._allow_concurrent():
                result = self.llm_service.parse_reminder(text, channel)

            if result.action == "clarify":
                irc.reply(result.confirmation)
                return

            if result.seconds is None:
                irc.reply(_("I couldn't determine when to remind you. Please try again."))
                return

            if result.seconds < 10:
                irc.error(_("Reminder must be at least 10 seconds from now."))
                return

            if result.seconds > 604800:  # 7 days
                irc.error(_("Reminder can't be more than 7 days out."))
                return

            reminder_message = result.message or text
            event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
            deliver = self._make_reminder_delivery_closure(
                nick, channel, reminder_message, event_name
            )

            try:
                schedule.addEvent(deliver, time.time() + result.seconds, name=event_name)
                with self._reminders_lock:
                    self._reminders[event_name] = (nick, channel, reminder_message)

                self.db.save_reminder(
                    event_name,
                    nick,
                    channel,
                    reminder_message,
                    time.time() + result.seconds,
                )

                reply = self.llm_service.sanitize_output(result.confirmation)
                if result.note:
                    reply = f"{reply} ({self.llm_service.sanitize_output(result.note)})"
                irc.reply(reply)
            except Exception as e:
                self.log.error("Failed to schedule reminder: %s", e)
                irc.error(_("Failed to set reminder."))

    def remind(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<reminder text> | list | del(ete) <id> [<id>...] | clear]

        Set and manage reminders using natural language.

        Examples:
          %remind in 30 minutes check the build
          %remind list
          %remind delete abc1
          %remind clear
        """
        nick = self._get_identity(irc, msg)

        if not text:
            self._remind_list(irc, nick)
            return

        parts = text.split(None, 1)
        subcommand = parts[0].lower()

        if subcommand == "list":
            self._remind_list(irc, nick)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            deleted = 0
            for rid in raw_ids:
                target = self._find_user_reminder(nick, rid)
                if target:
                    with contextlib.suppress(KeyError):
                        schedule.removeEvent(target)
                    with self._reminders_lock:
                        self._reminders.pop(target, None)
                    self.db.delete_reminder(target)
                    deleted += 1
            if deleted == 0:
                irc.error(_("No matching reminders found."))
            elif deleted == 1:
                irc.reply(_("Reminder cancelled."), prefixNick=False)
            else:
                irc.reply(f"Cancelled {deleted} reminders.", prefixNick=False)

        elif subcommand == "clear":
            user_reminders = self._get_user_reminders(nick)
            if not user_reminders:
                irc.reply(_("No reminders to clear."), prefixNick=False)
                return
            for name, _data in user_reminders:
                with contextlib.suppress(KeyError):
                    schedule.removeEvent(name)
                with self._reminders_lock:
                    self._reminders.pop(name, None)
                self.db.delete_reminder(name)
            label = "reminder" if len(user_reminders) == 1 else "reminders"
            irc.reply(f"Cleared {len(user_reminders)} {label}.", prefixNick=False)

        else:
            self._remind_set(irc, msg, nick, text)

    remind = wrap(remind, [optional("text")])


Class = LLM
