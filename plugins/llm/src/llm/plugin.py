"""LLM plugin implementation."""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import mimetypes
import random
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
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

from . import limnoria_bridge
from .config import resolve_setting
from .context import ContextConfig, ConversationContext, Role
from .persistence import LLMDatabase, ReminderRow
from .service import (
    AssistantRequestContext,
    AssistantResult,
    CompletionResult,
    ImageResult,
    LLMService,
    account_from_server_tags,
    irc_has_caps,
    truncate_to_word_boundary,
)
from .tracing import TraceFilter, generate_request_id, request_id

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

    from .assistant import ToolResult
    from .service import PendingTaskResult

_ = PluginInternationalization("LLM")

# Icon shown when Google grounding/search was used in the response
GROUNDING_ICON = "\U0001f310"  # 🌐 (globe with meridians)

# Commands that support long-term memory extraction
_MEMORY_COMMANDS = frozenset({"ask", "code"})

# C0 control characters except TAB (\x09), LF (\x0a), CR (\x0d).
# Includes ESC (\x1b) which starts ANSI sequences like \x1b[6n whose
# brackets crash Limnoria's nested-command tokenizer.
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_REQUEST_CONTEXT_CAPABILITIES = frozenset(
    {"llm.ask", "llm.code", "llm.draw", "owner", "admin", "trusted"}
)

# Reminder-mutation tools whose successful execution already produced a
# user-visible emoji reaction. When the assistant loop ends with one of
# these as the last successful tool AND no follow-up text, the chat
# reply is suppressed to avoid a duplicate ack. See Task B5 of the
# 2026-04-30 reminder simplification plan.
_REMINDER_MUTATION_TOOLS = frozenset({"set_reminder", "delete_reminder", "cancel_all_reminders"})

_FULL_ANSWER_LABEL = "Full answer"


@dataclass(frozen=True)
class Identity:
    """A user's stable storage handle paired with their live IRC nick.

    Three uses, one type:

    * ``raw_nick`` — what the user is presenting as on IRC right now.
      Use this for replies, displays, and IRC-protocol operations.
    * ``account`` — the NickServ account, or ``None`` if unidentified.
      Use this for "must be identified" gates and ownership checks
      that need to survive a nick change.
    * ``key`` — ``account or raw_nick``.  Use this for storage keys,
      rate-limit buckets, conversation context, and memory lookup.

    Two ``Identity`` values refer to the same user when their accounts
    match (case-insensitive) — or, lacking accounts on one or both
    sides, when their raw nicks match (case-insensitive).
    """

    raw_nick: str
    account: str | None

    @property
    def key(self) -> str:
        """Stable storage key — account when identified, raw nick otherwise."""
        return self.account or self.raw_nick

    def matches(self, other: Identity) -> bool:
        """True when both identities refer to the same user.

        Account-to-account match wins when both sides have one; falls
        back to raw-nick comparison when either side is unidentified.
        Both comparisons are case-insensitive (IRC nicks and NickServ
        account names are case-insensitive on AfterNet).
        """
        if self.account and other.account:
            return ircutils.toLower(self.account) == ircutils.toLower(other.account)
        return ircutils.toLower(self.raw_nick) == ircutils.toLower(other.raw_nick)


class ReminderScheduleResult(NamedTuple):
    """Result of scheduling a reminder."""

    ok: bool
    message: str


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
        name="g",
        args="<question>",
        description=(
            "Ask Grok with full chat-profile tool access (search, fetch, code, "
            "draw, reminders, memory). Tool gating respects IRC capabilities. "
            "Shares the ask rate-limit bucket."
        ),
        examples=(
            "%g what's the deal with airline food",
            "%g search the web for today's CVEs and summarize",
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


def _patch_irc_dojoin(plugin: LLM) -> None:
    """Replace supybot.irclib.Irc.doJoin to skip slow auto-queries on JOIN.

    Why: Limnoria's stock doJoin queues MODE +b (ban-list) and a WHO sync on
    every channel join. Nothing in this codebase reads ban state, and on
    servers with account-tag + extended-join the WHO is redundant — both
    queries serialize behind connection registration and meaningfully delay
    startup notification on rejoin.

    The patch always drops MODE +b and conditionally drops the WHO (gated by
    :meth:`LLM._will_skip_auto_who`). The plain MODE <channel> query is kept
    because Limnoria reads channel-mode state in many places.

    Re-patches on every plugin __init__ so the closure tracks the current
    LLM instance after a reload. Cheap; the patch is global to all Irc
    instances, so a multi-instance LLM plugin would have the last-init
    instance win — not a real concern for this single-plugin deployment.
    """
    from supybot import irclib, ircmsgs

    def doJoin(self, msg):  # noqa: N802
        if msg.nick != self.nick:
            return
        channel = msg.args[0]
        skip_who = plugin._will_skip_auto_who(self)
        if not skip_who:
            self.queueMsg(ircmsgs.who(channel, args=("%tuhnairf,1",)))
            # Track start of WHO sync so do315 can compute elapsed time.
            self.startedSync[channel] = time.time()
        self.queueMsg(ircmsgs.mode(channel))  # plain channel modes; ends with 329
        # Always skip MODE +b — nothing in the codebase reads ban-list state.
        # If WHO is skipped, do NOT touch startedSync — do315 will never arrive
        # and the dict would leak across rejoins.

    irclib.Irc.doJoin = doJoin  # ty: ignore[invalid-assignment]


def _patch_irc_docapnew() -> None:
    """Make doCapNew also request experimental caps (draft/multiline et al).

    Why: Limnoria's stock doCapNew filters CAP NEW announcements through
    REQUEST_CAPABILITIES only, ignoring REQUEST_EXPERIMENTAL_CAPABILITIES
    even when experimentalExtensions is enabled. AfterNET's bouncer
    advertises draft/multiline via CAP NEW post-SASL, so without this
    patch Limnoria never requests it and long replies fall back to
    @more pagination.
    """
    from supybot import conf, irclib

    def doCapNew(self, msg):  # noqa: N802
        if len(msg.args) != 3:
            log.warning("Bad CAP NEW from server: %r", msg)
            return
        caps = msg.args[2].split()
        assert caps, "Empty list of capabilities"
        self._addCapabilities(msg.args[2], msg)
        if self.state.fsm.state == irclib.IrcStateFsm.States.SHUTTING_DOWN:
            return
        want = irclib.Irc.REQUEST_CAPABILITIES
        if conf.supybot.protocols.irc.experimentalExtensions():
            want = want | irclib.Irc.REQUEST_EXPERIMENTAL_CAPABILITIES
        new = set(self.state.capabilities_ls) & want - self.state.capabilities_ack
        if new:
            self.requestCapabilities(new)

    irclib.Irc.doCapNew = doCapNew  # ty: ignore[invalid-assignment]


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

        _patch_irc_dojoin(self)
        _patch_irc_docapnew()

        # Initialize conversation context (loads persisted conversations from DB)
        self._init_context()

        # Track nicks already migrated to account-based identity this session
        self._migrated_nicks: set[str] = set()

        # In-memory per-command rate-limit buckets: "{command}:{account}" -> deque of timestamps
        self._rate_buckets: dict[str, collections.deque[float]] = {}

        self._reminders: dict[str, ReminderRow] = {}
        self._reminders_lock = threading.Lock()

        # Spontaneous participation cooldown tracking: channel -> last_fire_timestamp
        self._spontaneous_cooldowns: dict[str, float] = {}

        # Pending spontaneous schedule events (cancelled on unload)
        self._spontaneous_events: set[str] = set()

        # Reload persisted reminders from database
        self._reload_reminders(irc)

        # Re-register persisted scheduled LLM tasks (Phase 2 Task 3 / B3).
        self.llm_service.restore_scheduled_llm_tasks()

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
            # Close the main-thread DB connection. Worker-thread thread-local
            # connections are released as those threads exit; we don't track
            # them centrally, so sqlite ResourceWarnings can still appear
            # under reload-heavy/test-heavy workloads (see pyproject.toml).
            self.db.close()

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

        # Pending-task delivery bypasses _send_long_reply, so collapse multi-line
        # content. A raw \n on PRIVMSG triggers Excess Flood disconnects.
        text = self._collapse_for_irc(text) or text

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
        if r.status == "completed" and delivered:
            self._log_pending_delivery_usage(r, nick, target)

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
        caller = self._resolve_identity(irc, msg)
        message_text = msg.args[1] if len(msg.args) > 1 else ""

        # Store in conversation context for richer follow-up questions
        # Use display nick for channel context (what the LLM sees) so it
        # addresses people by their visible IRC name, not their account name.
        ctx_cfg = self._get_context_config(channel)
        self.context.add_message(
            caller.key, channel, Role.USER, message_text, config=ctx_cfg, persist=False
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
                    self._schedule_spontaneous(irc, channel, caller.key, message_text)

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

                api_key = resolve_setting(
                    self,
                    "assistantApiKey",
                    channel,
                    fallbacks=("spontaneousApiKey", "askApiKey"),
                )
                if not api_key:
                    return

                model = resolve_setting(
                    self,
                    "assistantModel",
                    channel,
                    fallbacks=("spontaneousModel", "askModel"),
                )
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

                response = self.llm_service.sanitize_output(result.content)
                action_text = self._extract_action(irc, response)
                if action_text:
                    irc.queueMsg(ircmsgs.action(channel, action_text))
                else:
                    # Spontaneous replies bypass _send_long_reply, so collapse
                    # multi-line content here — raw \n on PRIVMSG triggers
                    # Excess Flood disconnects on AfterNET.
                    irc.queueMsg(
                        ircmsgs.privmsg(channel, self._collapse_for_irc(response) or response)
                    )

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

    def _will_skip_auto_who(self, irc: callbacks.Irc) -> bool:
        """Return True iff the auto-WHO on channel join should be suppressed.

        Gate: both 'account-tag' AND 'extended-join' IRCv3 caps must be ACK'd
        (account-tag rides on PRIVMSG-class messages; extended-join rides on
        JOIN itself — together they obviate the auto-WHO scan), AND the
        operator-controlled ``skipAutoWhoOnJoin`` config must be True.
        """
        if not irc_has_caps(irc, "account-tag", "extended-join"):
            return False
        return bool(self.registryValue("skipAutoWhoOnJoin"))

    def doJoin(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Track channels the bot is joining for startup notification.

        When the bot joins a channel, we add it to _pending_channels.
        The channel is removed when we receive do315 (end of WHO).

        If the auto-WHO on join is being suppressed (account-tag + extended-join
        + skipAutoWhoOnJoin), do315 will never fire — so we must NOT add to
        _pending_channels here. The do376 2-second fallback (line 828) is then
        responsible for firing the startup notification.
        """
        if not ircutils.strEqual(irc.nick, msg.nick):
            return
        if self._will_skip_auto_who(irc):
            return
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
        self,
        nick: str,
        channel: str,
        message: str,
        event_name: str,
        *,
        action_prompt: str = "",
        account: str | None = None,
        chain_position: int = 1,
        recurrence_seconds: int | None = None,
        recurrence_rrule: str | None = None,
        watch_mode: bool = False,
    ):
        """Create a reminder delivery closure with error handling.

        Wraps delivery in try/finally so cleanup (removing from _reminders
        and database) always happens even if queueMsg raises.

        Args:
            nick: User's nick
            channel: Channel to deliver to (or nick for PM delivery)
            message: Reminder message
            event_name: Scheduler event name for cleanup
            chain_position: 1-based position of this reminder within its chain.
            recurrence_seconds: Numeric cadence (seconds) for the chain, or None.
            recurrence_rrule: RFC 5545 RRULE string for the chain, or None.
            watch_mode: True if the chain may emit ``[silent]`` per fire.

        Returns:
            Callable for use with schedule.addEvent
        """
        lock = self._reminders_lock
        parent_chain: int = chain_position or 1
        is_structured = self._is_structured_recurring(
            recurrence_seconds=recurrence_seconds,
            recurrence_rrule=recurrence_rrule,
        )
        # If the command was sent via PM, channel is the bot's own nick.
        # Deliver to the user's nick instead.
        target = channel if ircutils.isChannel(channel) else nick

        def _deliver() -> None:
            try:
                for active_irc in world.ircs:

                    def _send(text: str, *, _irc=active_irc) -> None:
                        safe_text = self.llm_service.sanitize_output(text)
                        # Reminders go straight to PRIVMSG without the
                        # _send_long_reply pagination, so collapse newlines —
                        # raw \n in a PRIVMSG body causes Excess Flood.
                        safe_text = self._collapse_for_irc(safe_text) or safe_text
                        _irc.queueMsg(ircmsgs.privmsg(target, f"{nick}: {safe_text}"))

                    # Legacy reminder path: plain echo delivery.
                    if not action_prompt:
                        _send(f"Reminder: {message}")
                        break

                    bot_nick = getattr(active_irc, "nick", None)
                    if not bot_nick:
                        self.log.warning(
                            "reminder_action_missing_bot_nick nick=%s channel=%s event=%s",
                            nick,
                            channel,
                            event_name,
                        )
                        _send(f"Reminder: {message}")
                        break

                    # Wrap the entire action delivery body so any exception
                    # (history gathering, msg construction, registry lookup,
                    # rate-limit check, assistant request, etc.) yields a
                    # generic user-visible fallback rather than silently
                    # losing the reminder. Never include exception text in
                    # user-visible output — full traceback to logs only.
                    try:
                        now = time.time()
                        rl_account = account if account else nick
                        rl_tier = "registered" if account else "unregistered"
                        if self._check_rate_limit(
                            None,
                            "ask",
                            rl_account,
                            "",
                            "",
                            "",
                            tier=rl_tier,
                            silent=True,
                            now=now,
                        ):
                            _send(f"Reminder: {message} (action skipped — daily ask limit reached)")
                            break

                        msg_target = channel if ircutils.isChannel(channel) else nick
                        msg_kwargs: dict[str, object] = {
                            "prefix": f"{nick}!~remind@scheduled",
                            "command": "PRIVMSG",
                            "args": (msg_target, ""),
                        }
                        if account:
                            msg_kwargs["server_tags"] = {"account": account}
                        synthetic_msg = ircmsgs.IrcMsg(**msg_kwargs)

                        request_context = AssistantRequestContext(
                            entry_route="remind_action",
                            profile="remind_action",
                            nick=nick,
                            raw_nick=nick,
                            account=account,
                            channel=channel,
                            is_private=not ircutils.isChannel(channel),
                            is_owner=False,
                            # Same per-feature caps as @ask/@draw/@code; owner/admin excluded.
                            capabilities=frozenset({"llm.ask", "llm.draw", "llm.code"}),
                        )

                        history, channel_history = self._gather_history(nick, channel)
                        memories = self._get_user_memories(nick)
                        user_instruction = self.db.get_instruction(nick)
                        ask_prompt = resolve_setting(
                            self,
                            "assistantSystemPrompt",
                            channel,
                            fallbacks=("askSystemPrompt",),
                        )
                        effective_prompt = (
                            f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None
                        )

                        caller = Identity(raw_nick=nick, account=account)

                        # Structured rows reschedule via _mechanical_reschedule
                        # below. Drop set_reminder from the fire-time tool
                        # surface so the action LLM can't double-schedule.
                        exclude_tools = (
                            frozenset({"set_reminder"}) if is_structured else frozenset()
                        )

                        result = self.llm_service.assistant_request(
                            prompt=action_prompt,
                            request_context=request_context,
                            db=self.db,
                            context=self.context,
                            bot_nick=bot_nick,
                            history=history,
                            channel_history=channel_history,
                            irc=active_irc,
                            msg=synthetic_msg,
                            memories=memories,
                            system_prompt=effective_prompt,
                            search_fn=lambda q: self.llm_service.search_completion(
                                q, channel=channel
                            ),
                            fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                            code_fn=lambda p: self._code_for_assistant(p, channel),
                            draw_fn=lambda p, _irc=active_irc, _msg=synthetic_msg: (
                                self._draw_for_assistant(_irc, _msg, p)
                            ),
                            cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                            exclude_tools=exclude_tools,
                            **self._reminder_fns(
                                caller=caller,
                                irc=active_irc,
                                msg=synthetic_msg,
                                pass_irc_msg_to_callbacks=False,
                            ),
                        )
                        response = result.content.strip() if result.content else ""
                        # Watch-mode sentinel: action LLM signals "no news to
                        # share, just stay scheduled." Usage is still logged
                        # so silent watches show up in the user's stats.
                        is_silent = response == "[silent]"
                        if is_silent:
                            pass  # No user-visible output this fire.
                        elif not response:
                            _send(f"Reminder: {message} (action returned empty response)")
                        elif message:
                            _send(f"Reminder ({message}): {response}")
                        else:
                            _send(f"Reminder: {response}")
                        # Attribute the action fire's LLM cost to the chain
                        # owner (account when present, raw nick as fallback)
                        # so runaway watches are visible in @usage and
                        # weigh against the same identity as rate limits.
                        try:
                            owner_key = account or nick
                            self.db.log_usage(
                                owner_key,
                                channel,
                                "remind_action",
                                result.model,
                                result.prompt_tokens,
                                result.completion_tokens,
                                result.cost,
                                prompt=action_prompt,
                                status=("silent" if is_silent else "success"),
                                error_detail=(result.error or "")[:200],
                            )
                        except Exception:
                            self.log.exception(
                                "reminder_action_usage_log_failed event=%s", event_name
                            )
                        # Structured rows reschedule mechanically (set_reminder
                        # was filtered out of the tool surface so the LLM
                        # cannot have done it). One-shot rows do nothing.
                        # Watch-mode + structured: still reschedule even if
                        # response was [silent]; the watch must keep watching.
                        if is_structured:
                            self._mechanical_reschedule(
                                nick=nick,
                                channel=channel,
                                message=message,
                                event_name=event_name,
                                action_prompt=action_prompt,
                                account=account,
                                chain_position=parent_chain,
                                recurrence_seconds=recurrence_seconds,
                                recurrence_rrule=recurrence_rrule,
                                watch_mode=watch_mode,
                                now=now,
                            )
                    except Exception:
                        self.log.exception(
                            "reminder_action_delivery_failed nick=%s channel=%s event=%s",
                            nick,
                            channel,
                            event_name,
                        )
                        try:
                            _send(
                                f"Reminder action '{message}' failed. "
                                "(Set this reminder again to retry.)"
                            )
                        except Exception:
                            self.log.exception(
                                "reminder_action_fallback_failed nick=%s channel=%s event=%s",
                                nick,
                                channel,
                                event_name,
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

            deliver = self._make_reminder_delivery_closure(
                nick,
                channel,
                message,
                event_name,
                action_prompt=reminder.action_prompt,
                account=reminder.account,
                chain_position=reminder.chain_position or 1,
                recurrence_seconds=reminder.recurrence_seconds,
                recurrence_rrule=reminder.recurrence_rrule,
                watch_mode=reminder.watch_mode,
            )

            if reminder.fire_at <= now:
                # Overdue — deliver immediately
                deliver()
            else:
                # Future — reschedule
                try:
                    schedule.addEvent(deliver, reminder.fire_at, name=event_name)
                    with self._reminders_lock:
                        self._reminders[event_name] = ReminderRow(
                            id=reminder.id,
                            event_name=event_name,
                            nick=nick,
                            channel=channel,
                            message=message,
                            action_prompt=reminder.action_prompt,
                            account=reminder.account,
                            fire_at=reminder.fire_at,
                            created_at=reminder.created_at,
                            chain_position=reminder.chain_position or 1,
                            recurrence_seconds=reminder.recurrence_seconds,
                            recurrence_rrule=reminder.recurrence_rrule,
                            watch_mode=reminder.watch_mode,
                        )
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
        """Route unrecognized addressed text through the chat profile.

        When someone says "vibebot draw a cat" or "vibebot what time is it"
        without a command prefix, the chat profile handles everything — general
        questions AND tool-based operations — via ``_ask_impl`` with the
        ``assistant_request`` facade.
        """
        if not tokens:
            return

        # Check if user has ask capability
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return

        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        text = " ".join(tokens)
        preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if preflight.blocked:
            return

        self._ask_impl(irc, msg, text, preflight, entry_route="invalid_command")

    def _account_from_msg(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Resolve the requesting user's account name from an incoming message.

        Two layers, in order:
        1. ``msg.server_tags['account']`` via :func:`account_from_server_tags`
           — the IRCv3 ``account-tag`` capability. Rides on every
           PRIVMSG/NOTICE/TAGMSG from an identified user, so it's valid even
           for users idling in-channel since before bot start.
        2. ``irc.state.nickToAccount(nick)`` — Limnoria's session cache.
           Populated by account-tag ingest, account-notify, extended-join,
           and WHO replies.

        Returns ``None`` when the user is not identified or unknown.

        Note: this resolver does NOT consult ``ircdb`` hostmask matching. That
        path would silently promote unidentified users to the ``registered``
        tier; owner/admin/trusted gating uses ``ircdb.checkCapability(prefix, …)``
        separately and is unaffected.
        """
        tag_account = account_from_server_tags(msg)
        if tag_account:
            return tag_account
        nick = ircutils.nickFromHostmask(msg.prefix)
        try:
            return irc.state.nickToAccount(nick)
        except (KeyError, AttributeError):
            return None

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
        """Migrate old nick-based rows to the account, once per session.

        Covers both ``usage`` and ``conversations`` tables so historical
        cost data and persisted conversation history follow the user
        once they identify.  In-memory ``ConversationContext`` is also
        rekeyed so the next turn resumes the same thread.

        Skips entirely when the nick and account are the same
        (case-insensitive) or when we've already attempted migration
        for this nick this session.

        Args:
            old_nick: The user's current IRC nick.
            account: The resolved NickServ account name.
        """
        if ircutils.toLower(old_nick) == ircutils.toLower(account):
            return
        key = ircutils.toLower(old_nick)
        if key in self._migrated_nicks:
            return
        self._migrated_nicks.add(key)
        usage_count = self.db.migrate_nick(old_nick, account)
        if usage_count > 0:
            self.log.info("Migrated %d usage row(s) from %s to %s", usage_count, old_nick, account)
        convo_count = self.db.migrate_conversations(old_nick, account)
        if convo_count > 0:
            self.log.info(
                "Migrated %d conversation row(s) from %s to %s",
                convo_count,
                old_nick,
                account,
            )
        # Also rekey in-memory conversation context so the live thread
        # carries over without waiting for DB reload.
        self.context.migrate_user(old_nick, account)

    def _log_pending_delivery_usage(
        self, result: PendingTaskResult, nick: str, target: str
    ) -> None:
        """Log usage for a delivered pending task.

        Prefers the account captured at submission time; falls back to live
        resolution by nick when the captured account is NULL (e.g., user was
        unidentified at request time).
        """
        if result.cost <= 0 and result.prompt_tokens <= 0:
            return
        for irc_conn in world.ircs:
            identity = result.account or self._resolve_nick_to_identity(irc_conn, nick)
            self.db.log_usage(
                identity,
                target,
                result.task_type,
                result.model,
                result.prompt_tokens,
                result.completion_tokens,
                result.cost,
            )
            break

    def _extract_action(self, irc: callbacks.Irc, response: str) -> str | None:
        """Return action text if *response* looks like an IRC action, else ``None``.

        Recognises both ``/me does something`` and ``* BotNick does something``.
        Embedded newlines are collapsed because IRC ACTION payloads are
        single-line; sending a multi-line action would put raw ``\\n`` bytes on
        the wire, which the server parses as separate commands and treats as
        Excess Flood.
        """
        if response.startswith("/me ") and len(response) > 4:
            return self._collapse_for_irc(response[4:]) or None
        star_prefix = f"* {irc.nick} "
        if response.startswith(star_prefix) and len(response) > len(star_prefix):
            return self._collapse_for_irc(response[len(star_prefix) :]) or None
        return None

    @staticmethod
    def _collapse_for_irc(text: str) -> str:
        """Collapse multi-line text into a single IRC-safe line."""
        return " | ".join(line for line in text.splitlines() if line.strip())

    def _build_bridge_tool(self, irc, msg, channel: str, trace: list | None = None):
        """Build the per-request Limnoria bridge tool schema + handler.

        Returns ``(None, None)`` when the bridge is disabled, the allowlist is
        empty, or no allowed command is currently exposable. Otherwise returns
        ``(schema_dict, {"run_limnoria_command": handler})`` for injection
        into ``assistant_completion`` via ``extra_tools`` / ``extra_handlers``.

        When ``trace`` is provided, each successful or failed dispatch appends
        a ``(plugin, command, args, status)`` tuple — used by the optional
        ``bridgeDebugInChannel`` reply footer.
        """
        if not self.registryValue("bridgeEnabled", channel):
            return None, None
        allowed = frozenset(self.registryValue("bridgeAllowedPlugins", channel) or [])
        if not allowed:
            # Empty registry value → fall back to the curated default set.
            # Limnoria persists every registered value to disk so a code
            # default change wouldn't reach existing operators on upgrade
            # (see DEFAULT_ALLOWED_PLUGINS docstring in limnoria_bridge.py).
            allowed = limnoria_bridge.DEFAULT_ALLOWED_PLUGINS
        allow_mutating = bool(self.registryValue("bridgeAllowMutating", channel))

        commands = list(
            limnoria_bridge.enumerate_commands(irc, msg, allowed, allow_mutating=allow_mutating)
        )
        if not commands:
            return None, None

        table = "\n".join(
            f"- {c.plugin}.{c.command}"
            + (f" — {c.arg_syntax}" if c.arg_syntax else "")
            + (f" — {c.description}" if c.description else "")
            for c in commands
        )
        # Footer: if the gate is closed AND any allowlisted plugin has at least
        # one mutating leaf, hint that more commands exist behind the gate.
        # Skips the hint for pure-read allowlists (Time, Math, etc.) where no
        # writes would be hidden.
        mutating_plugins = {p for (p, _leaf) in limnoria_bridge.MUTATING_COMMANDS}
        allowed_canonical = {p.lower() for p in allowed}
        hidden_writes_present = not allow_mutating and bool(allowed_canonical & mutating_plugins)
        footer = (
            "\n\n(write commands hidden — set bridgeAllowMutating True to expose)"
            if hidden_writes_present
            else ""
        )
        schema = {
            "type": "function",
            "function": {
                "name": "run_limnoria_command",
                "description": (
                    "Run a Limnoria plugin command on the user's behalf. "
                    "Available commands:\n" + table + footer
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin": {
                            "type": "string",
                            "description": "Plugin name (e.g. Misc).",
                        },
                        "command": {
                            "type": "string",
                            "description": "Leaf command name (e.g. ping).",
                        },
                        "args": {
                            "type": "string",
                            "description": (
                                "Argument string passed to the plugin command. "
                                "Empty string for commands taking no arguments."
                            ),
                        },
                    },
                    "required": ["plugin", "command", "args"],
                },
            },
        }

        from .assistant import ToolResult

        def handler(arguments):
            plugin_name = str(arguments.get("plugin", ""))
            command_name = str(arguments.get("command", ""))
            arg_string = str(arguments.get("args", ""))
            envelope = limnoria_bridge.dispatch(
                irc,
                msg,
                plugin=plugin_name,
                command=command_name,
                arg_string=arg_string,
                allow_mutating=allow_mutating,
            )
            if trace is not None:
                status = (
                    "ok" if envelope.get("status") == "ok" else f"err:{envelope.get('error', '?')}"
                )
                trace.append((plugin_name, command_name, arg_string, status))
            return ToolResult(content=json.dumps(envelope))

        return schema, {"run_limnoria_command": handler}

    @staticmethod
    def _format_bridge_debug_footer(trace: list) -> str:
        """Render a one-line debug footer for the optional bridgeDebugInChannel mode."""
        if not trace:
            return ""
        parts = []
        for plugin_name, command_name, arg_string, status in trace:
            call = f"{plugin_name}.{command_name}"
            if arg_string:
                call += f" {arg_string}"
            parts.append(f"{call} [{status}]")
        return "[bridge: " + " ; ".join(parts) + "]"

    @staticmethod
    def _trim_long_reply_teaser(teaser: str, max_chars: int) -> str:
        """Collapse and trim a teaser so the link reply stays on one line."""
        teaser = " ".join(teaser.split()) or _FULL_ANSWER_LABEL
        truncated = truncate_to_word_boundary(teaser, max_chars)
        if truncated != teaser:
            truncated = truncated.rstrip(" ,;:-")
        return truncated or _FULL_ANSWER_LABEL

    @staticmethod
    def _fallback_long_reply_teaser(text: str, max_chars: int) -> str:
        """Return a deterministic one-line teaser when LLM summarization fails."""
        teaser = next(
            (line.strip() for line in text.splitlines() if line.strip()), _FULL_ANSWER_LABEL
        )
        teaser = teaser.lstrip("#").strip()
        teaser = teaser.lstrip("-* ").strip()
        return LLM._trim_long_reply_teaser(teaser, max_chars)

    def _send_long_reply(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> None:
        """Reply with ``text``, using draft/multiline batches when supported.

        When the connection has negotiated ``draft/multiline`` (and
        experimentalExtensions is enabled), a long or multi-line reply is
        delivered as a single multiline batch — clients render it as one
        logical message instead of forcing the user through ``@more``.
        Falls back to ``irc.reply`` (Limnoria's pagination) otherwise, or
        when the reply is short enough to fit one IRC line.
        """
        target = msg.channel if msg.channel else msg.nick
        allowed = (
            conf.get(conf.supybot.reply.mores.length, channel=target, network=irc.network) or 400
        )

        # Build chunks: respect explicit \n boundaries, then byte-wrap each
        # line so individual messages fit IRC's per-line limit.
        raw_lines = text.split("\n") if "\n" in text else [text]
        chunks: list[str] = []
        for line in raw_lines:
            if not line:
                chunks.append("")
                continue
            wrapped = ircutils.wrap(line, allowed)
            chunks.extend(wrapped if wrapped else [line])

        line_threshold = int(self.registryValue("longReplyLineThreshold", target) or 0)
        logical_lines = [line for line in raw_lines if line.strip()]
        if line_threshold > 0 and (
            len(logical_lines) > line_threshold or len(chunks) > line_threshold
        ):
            url = self.llm_service.save_markdown_to_http(text)
            if url:
                suffix = f" - {_FULL_ANSWER_LABEL}: {url}"
                configured_max_chars = int(
                    self.registryValue("longReplyTeaserMaxChars", target) or 220
                )
                max_chars = min(configured_max_chars, max(0, allowed - len(suffix)))
                if max_chars <= 0:
                    irc.reply(f"{_FULL_ANSWER_LABEL}: {url}", prefixNick=prefixNick)
                    return
                teaser = self.llm_service.summarize_for_irc(
                    text, channel=target, max_chars=max_chars
                ) or self._fallback_long_reply_teaser(text, max_chars)
                teaser = self._trim_long_reply_teaser(teaser, max_chars)
                irc.reply(f"{teaser}{suffix}", prefixNick=prefixNick)
                return

        if len(chunks) <= 1:
            irc.reply(text, prefixNick=prefixNick)
            return

        multiline_supported = (
            conf.supybot.protocols.irc.experimentalExtensions()
            and "draft/multiline" in irc.state.capabilities_ack
        )
        if not multiline_supported:
            irc.reply(text, prefixNick=prefixNick)
            return

        msgs = [ircmsgs.privmsg(target, ircutils.safeArgument(chunk)) for chunk in chunks]
        irc.queueMultilineBatches(msgs, target, msg.nick, concat=False)

    def _build_request_context(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        preflight: PreflightResult,
        *,
        entry_route: str,
        profile: str,
    ) -> AssistantRequestContext:
        """Normalize route metadata into a shared assistant request context."""
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        capabilities = frozenset(
            capability
            for capability in _REQUEST_CONTEXT_CAPABILITIES
            if ircdb.checkCapability(msg.prefix, capability)
        )
        return AssistantRequestContext(
            entry_route=entry_route,
            profile=profile,
            nick=preflight.nick,
            raw_nick=raw_nick,
            account=preflight.account,
            channel=preflight.channel,
            is_private=not ircutils.isChannel(preflight.channel),
            is_owner="owner" in capabilities,
            capabilities=capabilities,
        )

    def _draw_for_assistant(self, irc: callbacks.Irc, msg: IrcMsg, prompt: str) -> str:
        """Generate an image for the generate_image tool.

        Usage logging is handled by the outer command wrapper via
        ``_store_context_and_log_usage``; leaf tool handlers do not log
        independently.
        """
        result = self.llm_service.image_generation(prompt, irc=irc, msg=msg)
        return result.content

    def _code_for_assistant(self, prompt: str, channel: str) -> ToolResult:
        """Generate code and save to HTTP for the generate_code tool."""
        from .assistant import ToolResult

        try:
            result = self.llm_service.completion(
                prompt,
                command="code",
                system_prompt=self.registryValue("codeSystemPrompt", channel),
            )
            if result.error:
                return ToolResult(content=json.dumps({"error": result.error}))
            url = self.llm_service.save_code_to_http(result.content)
            return ToolResult(
                content=json.dumps({"url": url or "", "code": result.content}),
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost=result.cost,
            )
        except Exception:
            self.log.exception("_code_for_assistant failed")
            return ToolResult(content=json.dumps({"error": "Code generation failed."}))

    def _resolve_identity(self, irc: callbacks.Irc, msg: IrcMsg) -> Identity:
        """Resolve a message sender to a structured :class:`Identity`.

        Reads the IRCv3 account-tag (or layer-2 session cache) via
        :meth:`_account_from_msg`. Triggers a one-time DB migration of
        nick→account rows on first successful resolution per session,
        covering both ``usage`` and ``conversations`` tables.
        """
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        account = self._account_from_msg(irc, msg)
        if account:
            self._maybe_migrate_nick(raw_nick, account)
        return Identity(raw_nick=raw_nick, account=account)

    def _require_account(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Require account identification. Returns account name or None.

        Uses the IRCv3 account-tag-aware resolver. When the user is not
        identified, sends an error reply and returns None. Callers should
        ``return`` immediately when None is returned.
        """
        account = self._account_from_msg(irc, msg)
        if not account:
            irc.error(_("You must be identified to use this command."))
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
            # _require_account returned the account; trigger nick→account migration.
            raw_nick = ircutils.nickFromHostmask(msg.prefix)
            self._maybe_migrate_nick(raw_nick, account)
            nick = account
        else:
            account = self._account_from_msg(irc, msg)
            if account:
                raw_nick = ircutils.nickFromHostmask(msg.prefix)
                self._maybe_migrate_nick(raw_nick, account)
                nick = account
            else:
                nick = ircutils.nickFromHostmask(msg.prefix)

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
        irc: callbacks.Irc | None,
        command: str,
        account: str,
        nick: str,
        channel: str,
        text: str,
        *,
        tier: str,
        silent: bool = False,
        now: float | None = None,
    ) -> bool:
        """Check rate limit; optionally suppress user-facing error and usage row.

        When ``silent=True``:
          - ``irc.error(...)`` is NOT called on overage.
          - ``db.log_usage(..., status="rate_limited")`` is NOT written.
          - ``irc`` may be None (action-fire path has no caller IRC connection).
          - ``nick``/``channel``/``text`` are still accepted but unused in the
            silent branch — kept in the signature for caller-site uniformity.

        ``now`` defaults to ``time.time()`` when not supplied — keeps the
        original non-silent signature working without forcing every caller
        to thread a timestamp.

        Args:
            irc: IRC connection (may be None when ``silent=True``).
            command: Command name.
            account: NickServ account name or nick-based identity.
            nick: Resolved identity for logging.
            channel: Channel name.
            text: Prompt text for logging.
            tier: User tier (trusted, registered, unregistered).
            silent: When True, suppress ``irc.error`` and ``db.log_usage``.
            now: Optional pre-computed timestamp; defaults to ``time.time()``.

        Returns:
            True if the request should be blocked.
        """
        if now is None:
            now = time.time()
        over_limit = self._is_rate_limited(command, account, now, tier=tier)

        # Always record the hit (so the window tracks correctly)
        self._record_rate_limit_hit(command, account, now)

        if not over_limit:
            return False

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
            if not silent and irc is not None:
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
        account = self._account_from_msg(irc, msg)
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

    def _gather_history(
        self,
        nick: str,
        channel: str,
        *,
        max_age_seconds: int | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Return (personal_history, channel_history) for the given nick/channel.

        Returns ([], []) when context is disabled for the channel or when
        ``max_age_seconds`` is 0 (callers use 0 to opt out of context while
        keeping a positive freshness window configurable). A positive value
        filters stale conversations at the context layer.
        """
        if not self._get_context_enabled(channel) or max_age_seconds == 0:
            return [], []
        ctx_cfg = self._get_context_config(channel)
        history = self.context.get_messages(
            nick, channel, config=ctx_cfg, max_age_seconds=max_age_seconds
        )
        channel_history = self.context.get_channel_messages(
            channel,
            exclude_nick=nick,
            config=ctx_cfg,
            max_age_seconds=max_age_seconds,
        )
        return history, channel_history

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

            snapshot_ids = tuple(r.id for r in existing_rows)

            def _extract_memories_bg() -> None:
                try:
                    extraction = self.llm_service.extract_memories(
                        nick, channel, user_text, assistant_response, existing_facts
                    )
                    if not extraction.add:
                        return

                    # Race protection: abort if memory rows changed during LLM
                    # call. Compare row IDs (not just count) so a delete+insert
                    # that preserves the count still triggers an abort.
                    current = self.db.get_memories(nick)
                    current_ids = tuple(r.id for r in current)
                    if current_ids != snapshot_ids:
                        log.info(
                            "Memory extraction for %s aborted: rows changed",
                            nick,
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

        # Abort if memory rows changed during LLM call (race protection).
        # Compare row IDs — a delete+insert preserving count would otherwise
        # let cleanup mis-target indices.
        current = self.db.get_memories(nick)
        snapshot_ids = tuple(r.id for r in snapshot)
        current_ids = tuple(r.id for r in current)
        if current_ids != snapshot_ids:
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
        result: CompletionResult | ImageResult | AssistantResult,
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

        self._ask_impl(irc, msg, text, pf, entry_route="ask")

    ask = wrap(ask, [("checkCapability", "llm.ask"), "text"])

    def _ask_impl(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        pf: PreflightResult,
        *,
        entry_route: str = "ask",
    ) -> None:
        """Core ask logic, separated so invalidCommand can reuse without double-preflight."""
        nick, channel = pf.nick, pf.channel
        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route=entry_route,
            profile="chat",
        )

        caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

        with self._trace_request("ask", nick, channel):
            # Detect images for vision
            images = self.llm_service.detect_images(text)

            history, channel_history = self._gather_history(nick, channel)

            memories = self._get_user_memories(nick)
            user_instruction = self.db.get_instruction(nick)

            # Build system prompt with optional user instruction
            ask_prompt = resolve_setting(
                self,
                "assistantSystemPrompt",
                channel,
                fallbacks=("askSystemPrompt",),
            )
            effective_prompt = f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None

            with self._allow_concurrent():
                request_text = text
                if images:
                    # Clean prompt by removing image URLs
                    for img in images:
                        request_text = request_text.replace(img, "").strip()

                bridge_trace: list = []
                bridge_schema, bridge_handlers = self._build_bridge_tool(
                    irc, msg, channel, trace=bridge_trace
                )
                extra_tools = [bridge_schema] if bridge_schema else None
                bridge_debug = bool(
                    bridge_schema and self.registryValue("bridgeDebugInChannel", channel)
                )

                result = self.llm_service.assistant_request(
                    request_text,
                    request_context=request_context,
                    db=self.db,
                    context=self.context,
                    bot_nick=irc.nick,
                    images=images,
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=memories,
                    system_prompt=effective_prompt,
                    search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                    fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                    code_fn=lambda p: self._code_for_assistant(p, channel),
                    draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                    cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                    extra_tools=extra_tools,
                    extra_handlers=bridge_handlers,
                    **self._reminder_fns(caller=caller, irc=irc, msg=msg),
                )

                # Format response with grounding icon if search was used
                response = result.content

                # Optional in-channel debug footer listing bridge tool calls.
                if bridge_debug and bridge_trace:
                    footer = self._format_bridge_debug_footer(bridge_trace)
                    if footer:
                        response = f"{response}\n{footer}" if response else footer

                # Structured suppression: when the assistant just performed
                # a successful reminder mutation and produced no follow-up
                # text, the user already saw the emoji reaction (clock /
                # thumbs-up). Skip the reply to avoid a duplicate ack.
                # Usage + context still get recorded below.
                if (
                    result.last_successful_tool in _REMINDER_MUTATION_TOOLS
                    and not result.final_text_after_tools.strip()
                ):
                    self.log.info(
                        "suppressing empty post-reminder-mutation reply tool=%s %s/%s",
                        result.last_successful_tool,
                        channel,
                        nick,
                    )
                elif not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return
                else:
                    action_text = self._extract_action(irc, response)
                    if action_text:
                        if result.grounding_used:
                            action_text = f"{GROUNDING_ICON} {action_text}"
                        self.log.info("sending action to %s/%s", channel, nick)
                        target = channel if ircutils.isChannel(channel) else nick
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
                        self._send_long_reply(irc, msg, display_response, prefixNick=False)

            self._store_context_and_log_usage(
                nick, channel, "ask", text, response, result, irc, msg
            )

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

        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route="code",
            profile="code",
        )

        caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

        with self._trace_request("code", nick, channel):
            history, channel_history = self._gather_history(nick, channel)

            memories = self._get_user_memories(nick)
            user_instruction = self.db.get_instruction(nick)
            # Layer user instruction onto CODE_SYSTEM_PROMPT (the facade
            # prompt that tells the planner to call generate_code) — not
            # the registry codeSystemPrompt, which is the inner-call
            # prompt used by _code_for_assistant.
            from .assistant import CODE_SYSTEM_PROMPT

            effective_prompt = (
                f"{user_instruction}\n\n{CODE_SYSTEM_PROMPT}" if user_instruction else None
            )

            with self._allow_concurrent():
                result = self.llm_service.assistant_request(
                    text,
                    request_context=request_context,
                    db=self.db,
                    context=self.context,
                    bot_nick=irc.nick,
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=memories,
                    system_prompt=effective_prompt,
                    search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                    fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                    code_fn=lambda p: self._code_for_assistant(p, channel),
                    draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                    cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                    **self._reminder_fns(caller=caller, irc=irc, msg=msg),
                )

                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                action_text = self._extract_action(irc, response)
                if action_text:
                    if result.grounding_used:
                        action_text = f"{GROUNDING_ICON} {action_text}"
                    self.log.info("sending action to %s/%s", channel, nick)
                    target = channel if ircutils.isChannel(channel) else nick
                    irc.queueMsg(ircmsgs.action(target, action_text))
                    response = f"* {irc.nick} {action_text}"
                else:
                    display_response = (
                        f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                    )
                    self.log.info("replying to %s/%s", channel, nick)
                    self._send_long_reply(irc, msg, display_response, prefixNick=False)

            self._store_context_and_log_usage(
                nick, channel, "code", text, response, result, irc, msg
            )

    code = wrap(code, [("checkCapability", "llm.code"), "text"])

    def g(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<question>

        Ask Grok directly with full chat-profile tool access (search, fetch,
        code, draw, reminders, memory). Tool gating still respects your IRC
        capabilities. Shares the ask rate-limit bucket.

        Examples:
          %g what's the deal with airline food
          %g search the web for the latest CVE on openssl
        """
        if self._is_old_message(msg):
            return

        # Share ask's rate-limit bucket and capability.
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

        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route="g",
            profile="chat",
        )
        caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

        with self._trace_request("g", nick, channel):
            images = self.llm_service.detect_images(text)
            history, channel_history = self._gather_history(nick, channel)
            memories = self._get_user_memories(nick)
            user_instruction = self.db.get_instruction(nick)

            grok_personality = self.registryValue("grokSystemPrompt", channel)
            prefix_parts = [p for p in (user_instruction, grok_personality) if p]
            effective_prompt = "\n\n".join(prefix_parts) if prefix_parts else None

            grok_api_key = self.registryValue("grokApiKey", channel)
            grok_model = self.registryValue("grokModel", channel)

            with self._allow_concurrent():
                request_text = text
                if images:
                    for img in images:
                        request_text = request_text.replace(img, "").strip()

                result = self.llm_service.assistant_request(
                    request_text,
                    request_context=request_context,
                    db=self.db,
                    context=self.context,
                    bot_nick=irc.nick,
                    images=images,
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=memories,
                    system_prompt=effective_prompt,
                    api_key=grok_api_key or None,
                    model_override=grok_model or None,
                    search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                    fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                    code_fn=lambda p: self._code_for_assistant(p, channel),
                    draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                    cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                    **self._reminder_fns(caller=caller, irc=irc, msg=msg),
                )

                response = result.content

                if (
                    result.last_successful_tool in _REMINDER_MUTATION_TOOLS
                    and not result.final_text_after_tools.strip()
                ):
                    self.log.info(
                        "suppressing empty post-reminder-mutation reply tool=%s %s/%s",
                        result.last_successful_tool,
                        channel,
                        nick,
                    )
                elif not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return
                else:
                    action_text = self._extract_action(irc, response)
                    if action_text:
                        if result.grounding_used:
                            action_text = f"{GROUNDING_ICON} {action_text}"
                        self.log.info("sending action to %s/%s", channel, nick)
                        target = channel if ircutils.isChannel(channel) else nick
                        irc.queueMsg(ircmsgs.action(target, action_text))
                        response = f"* {irc.nick} {action_text}"
                    else:
                        display_response = (
                            f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                        )
                        self.log.info("replying to %s/%s", channel, nick)
                        self._send_long_reply(irc, msg, display_response, prefixNick=False)

            self._store_context_and_log_usage(nick, channel, "g", text, response, result, irc, msg)

    g = wrap(g, [("checkCapability", "llm.ask"), "text"])

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

        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route="draw",
            profile="draw",
        )

        caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

        with self._trace_request("draw", nick, channel):
            history, channel_history = self._gather_history(
                nick,
                channel,
                max_age_seconds=self.registryValue("drawContextMaxAgeSeconds", channel),
            )

            with self._allow_concurrent():
                result = self.llm_service.assistant_request(
                    text,
                    request_context=request_context,
                    db=self.db,
                    context=self.context,
                    bot_nick=irc.nick,
                    history=history,
                    channel_history=channel_history,
                    irc=irc,
                    msg=msg,
                    memories=[],
                    draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                    **self._reminder_fns(caller=caller, irc=irc, msg=msg),
                )

                response = result.content
                if not response or not response.strip():
                    irc.error(_("The model returned an empty response. Please try again."))
                    return

                action_text = self._extract_action(irc, response)
                if action_text:
                    if result.grounding_used:
                        action_text = f"{GROUNDING_ICON} {action_text}"
                    self.log.info("sending action to %s/%s", channel, nick)
                    target = channel if ircutils.isChannel(channel) else nick
                    irc.queueMsg(ircmsgs.action(target, action_text))
                    response = f"* {irc.nick} {action_text}"
                else:
                    display_response = (
                        f"{GROUNDING_ICON} {response}" if result.grounding_used else response
                    )
                    self.log.info("replying to %s/%s", channel, nick)
                    self._send_long_reply(irc, msg, display_response, prefixNick=False)

            self._store_context_and_log_usage(
                nick, channel, "draw", text, response, result, irc, msg
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
        caller = self._resolve_identity(irc, msg)
        # Default to current channel if not specified
        if channel is None:
            channel = self._get_channel(msg)
        self.context.clear(caller.key, channel)
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
        caller = self._resolve_identity(irc, msg)

        if not text:
            # List own memories (newest first)
            self._memories_list(irc, caller.key, caller.key)
            return

        parts = text.split(None, 2)
        subcommand = parts[0].lower()

        if subcommand == "clear":
            count = self.db.delete_all_memories(caller.key)
            label = "memory" if count == 1 else "memories"
            irc.reply(f"Cleared {count} {label}.", prefixNick=False)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            try:
                memory_ids = [int(x) for x in raw_ids]
            except ValueError:
                irc.reply("Usage: memories delete <id> [<id> ...]", prefixNick=False)
                return
            deleted = sum(1 for mid in memory_ids if self.db.delete_memory(caller.key, mid))
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
            if self.db.update_memory(caller.key, memory_id, new_text):
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
                target = caller.key
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
        caller = self._resolve_identity(irc, msg)

        if not text:
            current = self.db.get_instruction(caller.key)
            if current:
                irc.reply(f"Current instruction: {current}", prefixNick=False)
            else:
                irc.reply("No instruction set. Use %instruct <text> to set one.", prefixNick=False)
            return

        if text.strip().lower() == "clear":
            if self.db.delete_instruction(caller.key):
                irc.reply("Instruction cleared.", prefixNick=False)
            else:
                irc.reply("No instruction to clear.", prefixNick=False)
            return

        self.db.save_instruction(caller.key, text)
        irc.reply("Instruction set.", prefixNick=False)

    instruct = wrap(instruct, [optional("text")])

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
        caller = self._resolve_identity(irc, msg)

        # This month: first of month midnight UTC
        month_start = self._month_start_ts()

        chan_summary = self.db.get_usage_summary_for_channel(channel, since=month_start)
        nick_summary = self.db.get_usage_summary_for_nick(
            caller.key, since=month_start, channel=channel
        )
        chan_rank = self.db.get_channel_rank(channel, since=month_start)
        nick_rank = self.db.get_nick_rank(caller.key, since=month_start, channel=channel)

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
        ctx_stats = self.context.get_user_stats(caller.key, channel, config=ctx_cfg)
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

    def _reminder_fns(
        self,
        *,
        caller: Identity,
        irc: callbacks.Irc,
        msg: IrcMsg,
        pass_irc_msg_to_callbacks: bool = True,
    ) -> dict[str, Callable[..., object]]:
        """Build the four-lambda reminder-tool dict for assistant calls.

        ``pass_irc_msg_to_callbacks`` is False on the action-fire path: its
        ``synthetic_msg`` has no msgid, so passing ``irc``/``msg`` through would
        just invoke ``_react`` only to have its msgid check fail.
        """
        if pass_irc_msg_to_callbacks:

            def delete_fn(r: str) -> str:
                return self._remind_delete_for_assistant(
                    caller,
                    r,
                    irc=irc,
                    msg=msg,
                )

            def clear_fn() -> str:
                return self._remind_clear_for_assistant(
                    caller,
                    irc=irc,
                    msg=msg,
                )
        else:

            def delete_fn(r: str) -> str:
                return self._remind_delete_for_assistant(caller, r)

            def clear_fn() -> str:
                return self._remind_clear_for_assistant(caller)

        def set_fn(t: str) -> str:
            return self._remind_set_for_assistant(irc, msg, caller, t)

        return {
            "list_reminders_fn": lambda: self._get_user_reminders(caller),
            "set_reminder_fn": set_fn,
            "delete_reminder_fn": delete_fn,
            "cancel_all_reminders_fn": clear_fn,
        }

    def _get_user_reminders(self, caller: Identity) -> list[tuple[str, ReminderRow]]:
        """Get reminders belonging to a specific user.

        Match policy: account-to-account when both the caller and the
        stored row have an account; raw-nick comparison otherwise (see
        :meth:`Identity.matches`).  This lets a user who scheduled a
        reminder while identified still see it after a nick change, and
        keeps unidentified users' reminders scoped to their nick.

        Args:
            caller: The requesting user's :class:`Identity`.

        Returns:
            List of ``(event_name, ReminderRow)`` pairs owned by ``caller``.
        """
        with self._reminders_lock:
            return [
                (name, data)
                for name, data in self._reminders.items()
                if Identity(raw_nick=data.nick, account=data.account).matches(caller)
            ]

    def _format_reminders(
        self,
        reminders: list[tuple[str, ReminderRow]],
    ) -> str:
        """Format reminders list for display.

        Args:
            reminders: List of ``(event_name, ReminderRow)`` pairs.

        Returns:
            Formatted string for IRC display
        """
        parts = []
        for name, data in reminders:
            message = data.message
            action_prompt = data.action_prompt
            # Truncate long messages
            preview = message[:40] + "..." if len(message) > 40 else message
            # Extract ID from event name
            reminder_id = name.split("_")[-1]
            marker = " [auto]" if action_prompt else ""
            parts.append(f"#{reminder_id}: {preview}{marker}")
        return " | ".join(parts)

    def _find_user_reminder(self, caller: Identity, reminder_id: str) -> str | None:
        """Find a reminder event name by ID, scoped to the caller's identity.

        Args:
            caller: The requesting user's :class:`Identity`.
            reminder_id: Reminder ID (last part of event name).

        Returns:
            Event name if found and owned by the caller, ``None`` otherwise.
        """
        with self._reminders_lock:
            for name, data in self._reminders.items():
                if not name.endswith(f"_{reminder_id}"):
                    continue
                if Identity(raw_nick=data.nick, account=data.account).matches(caller):
                    return name
            return None

    def _remind_list(self, irc: callbacks.Irc, caller: Identity) -> None:
        """List pending reminders for the calling user."""
        user_reminders = self._get_user_reminders(caller)
        if not user_reminders:
            irc.reply(_("You have no pending reminders."))
            return
        irc.reply(self._format_reminders(user_reminders))

    _REMINDER_MAX_SECONDS = 604800  # 7 days
    _REMINDER_MAX_CHAIN_POSITION = 50  # cap recurring fires before user re-arms
    _REMINDER_MAX_PENDING_PER_USER = 25  # cap one-shot accumulation per user

    @staticmethod
    def _is_structured_recurring(
        *, recurrence_seconds: int | None, recurrence_rrule: str | None
    ) -> bool:
        """True when the row carries a structured recurrence column (B1+)."""
        return recurrence_seconds is not None or recurrence_rrule is not None

    @staticmethod
    def _next_rrule_fire(rule_str: str, now: float) -> float | None:
        """Compute the next fire time after ``now`` for an RRULE string.

        Uses dateutil with timezone-aware UTC so DST transitions don't
        produce duplicate or skipped fires. Returns None when the rule
        is malformed or has no future occurrence.
        """
        from dateutil.rrule import rrulestr

        try:
            now_utc = datetime.fromtimestamp(now, tz=UTC)
            rule = rrulestr(rule_str, dtstart=now_utc)
            next_dt = rule.after(now_utc)
        except (ValueError, TypeError):
            return None
        if next_dt is None:
            return None
        return next_dt.timestamp()

    def _mechanical_reschedule(
        self,
        *,
        nick: str,
        channel: str,
        message: str,
        event_name: str,
        action_prompt: str,
        account: str | None,
        chain_position: int,
        recurrence_seconds: int | None,
        recurrence_rrule: str | None,
        watch_mode: bool,
        now: float,
    ) -> None:
        """Schedule the next fire of a structured recurring reminder.

        Computes ``next_fire`` from ``recurrence_seconds`` (numeric path)
        or ``recurrence_rrule`` (RFC 5545 parsed timezone-aware UTC),
        enforces the chain_position cap, and registers a fresh schedule
        event + ReminderRow + DB row.

        No-ops when (a) chain_position has hit the cap, (b) the rrule is
        malformed or exhausted, or (c) the original event has been
        cancelled mid-fire (clear-wins-over-mid-fire).
        """
        next_position = chain_position + 1
        if next_position > self._REMINDER_MAX_CHAIN_POSITION:
            self.log.info(
                "reminder_reschedule_skipped reason=cap event=%s position=%d/%d",
                event_name,
                next_position,
                self._REMINDER_MAX_CHAIN_POSITION,
            )
            return

        next_fire: float | None = None
        if recurrence_seconds is not None:
            next_fire = now + recurrence_seconds
        elif recurrence_rrule is not None:
            next_fire = self._next_rrule_fire(recurrence_rrule, now)
            if next_fire is None:
                self.log.warning(
                    "reminder_reschedule_skipped reason=rrule_invalid_or_exhausted "
                    "event=%s rule=%r",
                    event_name,
                    recurrence_rrule,
                )
                return

        if next_fire is None:
            return  # Neither recurrence kind populated — caller's mistake.

        # Clear-wins-over-mid-fire: if cancel_all_reminders or a single
        # delete fired during the action, the original event_name is gone
        # from _reminders. Don't reschedule a cancelled chain.
        with self._reminders_lock:
            if event_name not in self._reminders:
                self.log.info(
                    "reminder_reschedule_skipped reason=cancelled_mid_fire event=%s",
                    event_name,
                )
                return

        new_event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
        new_deliver = self._make_reminder_delivery_closure(
            nick,
            channel,
            message,
            new_event_name,
            action_prompt=action_prompt,
            account=account,
            chain_position=next_position,
            recurrence_seconds=recurrence_seconds,
            recurrence_rrule=recurrence_rrule,
            watch_mode=watch_mode,
        )
        try:
            schedule.addEvent(new_deliver, next_fire, name=new_event_name)
            with self._reminders_lock:
                self._reminders[new_event_name] = ReminderRow(
                    id=0,
                    event_name=new_event_name,
                    nick=nick,
                    channel=channel,
                    message=message,
                    action_prompt=action_prompt,
                    account=account,
                    fire_at=next_fire,
                    created_at=now,
                    chain_position=next_position,
                    recurrence_seconds=recurrence_seconds,
                    recurrence_rrule=recurrence_rrule,
                    watch_mode=watch_mode,
                )
            self.db.save_reminder(
                new_event_name,
                nick,
                channel,
                message,
                next_fire,
                action_prompt=action_prompt,
                account=account,
                chain_position=next_position,
                recurrence_seconds=recurrence_seconds,
                recurrence_rrule=recurrence_rrule,
                watch_mode=watch_mode,
            )
            self.log.info(
                "reminder_reschedule path=mechanical kind=%s event=%s "
                "position=%d/%d next_fire_at=%.3f",
                "seconds" if recurrence_seconds is not None else "rrule",
                new_event_name,
                next_position,
                self._REMINDER_MAX_CHAIN_POSITION,
                next_fire,
            )
        except Exception:
            self.log.exception(
                "reminder_reschedule_failed event=%s reason=schedule_or_persist",
                new_event_name,
            )

    def _react(self, irc: callbacks.Irc, msg: IrcMsg, emoji: str) -> bool:
        """Send a +draft/react reaction to ``msg``. Returns True if queued.

        No-ops cleanly when the server lacks message-tags or the incoming
        message has no msgid — caller should fall back to a text reply.
        """
        server_tags = getattr(msg, "server_tags", None) or {}
        msgid = server_tags.get("msgid")
        target = msg.args[0] if msg.args else ""
        if not msgid or not target:
            self.log.info(
                "react_skipped emoji=%s reason=%s server_tag_keys=%s target=%r",
                emoji,
                "no_msgid" if not msgid else "no_target",
                sorted(server_tags.keys()) if server_tags else [],
                target,
            )
            return False
        return self.llm_service.send_reaction(irc, target, msgid, emoji)

    def _ack(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        emoji: str,
        fallback_text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> None:
        """React with `emoji`; fall back to text if the server can't carry it.

        `prefixNick` mirrors the kwarg on `irc.reply` — pass True when the call
        site previously called `irc.reply(text)` with the default prefix, False
        when it explicitly disabled prefixing.
        """
        if not self._react(irc, msg, emoji):
            irc.reply(fallback_text, prefixNick=prefixNick)

    def _cancel_reminder(self, event_name: str) -> None:
        """Remove a single reminder from scheduler, in-memory dict, and database."""
        with contextlib.suppress(KeyError):
            schedule.removeEvent(event_name)
        with self._reminders_lock:
            self._reminders.pop(event_name, None)
        self.db.delete_reminder(event_name)

    def _schedule_reminder(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Identity,
        text: str,
        *,
        parent_chain: int | None = None,
    ) -> ReminderScheduleResult:
        """Parse, validate, and schedule a reminder.

        ``caller.raw_nick`` is stored as the reminder's owning nick (used
        for the synthetic message prefix at fire time and as the fallback
        match when the caller has no account).  ``caller.account`` is
        captured separately and is the preferred match key on lookup.

        ``parent_chain`` is supplied when an action-fire LLM is rescheduling
        the next occurrence of a recurring reminder. It carries the parent's
        ``chain_position`` so we can enforce the per-chain cap.
        """
        channel = self._get_channel(msg)

        with self._trace_request("remind", caller.key, channel), self._allow_concurrent():
            result = self.llm_service.parse_reminder(text, channel)

        if result.action == "clarify":
            return ReminderScheduleResult(ok=True, message=result.confirmation)

        if result.seconds is None:
            return ReminderScheduleResult(
                ok=False, message="Could not determine when to set the reminder."
            )

        if result.seconds < 10:
            return ReminderScheduleResult(
                ok=False, message="Reminder must be at least 10 seconds from now."
            )

        if result.seconds > self._REMINDER_MAX_SECONDS:
            return ReminderScheduleResult(
                ok=False, message="Reminder can't be more than 7 days out."
            )

        now = time.time()
        if parent_chain is not None:
            parent_position = parent_chain
            chain_position = parent_position + 1
            if chain_position > self._REMINDER_MAX_CHAIN_POSITION:
                return ReminderScheduleResult(
                    ok=False,
                    message=(
                        f"Recurring reminder reached its cap of "
                        f"{self._REMINDER_MAX_CHAIN_POSITION} runs. "
                        "Set it again to continue."
                    ),
                )
        else:
            chain_position = 1
            pending = len(self._get_user_reminders(caller))
            if pending >= self._REMINDER_MAX_PENDING_PER_USER:
                return ReminderScheduleResult(
                    ok=False,
                    message=(
                        f"You already have {pending} pending reminders "
                        f"(cap {self._REMINDER_MAX_PENDING_PER_USER}). "
                        "Cancel some first."
                    ),
                )

        reminder_message = result.message or text
        action_prompt = result.action_prompt
        event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
        recurrence_seconds = result.recurrence_seconds
        recurrence_rrule = result.recurrence_rrule
        watch_mode = result.watch_mode
        deliver = self._make_reminder_delivery_closure(
            caller.raw_nick,
            channel,
            reminder_message,
            event_name,
            action_prompt=action_prompt,
            account=caller.account,
            chain_position=chain_position,
            recurrence_seconds=recurrence_seconds,
            recurrence_rrule=recurrence_rrule,
            watch_mode=watch_mode,
        )

        try:
            schedule.addEvent(deliver, now + result.seconds, name=event_name)
            with self._reminders_lock:
                self._reminders[event_name] = ReminderRow(
                    id=0,
                    event_name=event_name,
                    nick=caller.raw_nick,
                    channel=channel,
                    message=reminder_message,
                    action_prompt=action_prompt,
                    account=caller.account,
                    fire_at=now + result.seconds,
                    created_at=now,
                    chain_position=chain_position,
                    recurrence_seconds=recurrence_seconds,
                    recurrence_rrule=recurrence_rrule,
                    watch_mode=watch_mode,
                )

            self.db.save_reminder(
                event_name,
                caller.raw_nick,
                channel,
                reminder_message,
                now + result.seconds,
                action_prompt=action_prompt,
                account=caller.account,
                chain_position=chain_position,
                recurrence_seconds=recurrence_seconds,
                recurrence_rrule=recurrence_rrule,
                watch_mode=watch_mode,
            )

            reply = self.llm_service.sanitize_output(result.confirmation)
            if result.note:
                reply = f"{reply} ({self.llm_service.sanitize_output(result.note)})"
            if chain_position > 1:
                reply = f"{reply} ({chain_position}/{self._REMINDER_MAX_CHAIN_POSITION})"
            return ReminderScheduleResult(ok=True, message=reply)
        except Exception as e:
            self.log.error("Failed to schedule reminder: %s", e)
            return ReminderScheduleResult(ok=False, message="Failed to set reminder.")

    def _remind_set(self, irc: callbacks.Irc, msg: IrcMsg, caller: Identity, text: str) -> None:
        """Parse and schedule a natural language reminder via IRC."""
        result = self._schedule_reminder(irc, msg, caller, text)
        if result.ok:
            self._ack(irc, msg, "⏰", result.message, prefixNick=True)
        else:
            self._react(irc, msg, "❌")
            irc.error(_(result.message))

    def _remind_set_for_assistant(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Identity,
        text: str,
        *,
        parent_chain: int | None = None,
    ) -> str:
        """Parse and schedule a reminder, returning a result string for meta.

        ``parent_chain`` is provided when the call originates from an
        action-fire LLM rescheduling a recurring chain — see
        :meth:`_schedule_reminder` for cap enforcement.

        Reacts ⏰ to ``msg`` on success and ❌ on cap/parse failure so the
        user gets a visual acknowledgment regardless of whether the model
        speaks. The chat reply path suppresses an empty post-tool reply
        via the structured ``last_successful_tool`` signal.
        """
        result = self._schedule_reminder(irc, msg, caller, text, parent_chain=parent_chain)
        self._react(irc, msg, "⏰" if result.ok else "❌")
        return result.message

    def _remind_delete_for_assistant(
        self,
        caller: Identity,
        reminder_id: str,
        *,
        irc: callbacks.Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Delete a reminder by ID, scoped to the caller's identity.

        When ``irc``/``msg`` are provided, reacts 👍 on success and ❌
        when no matching reminder exists (so the chat path can suppress
        the empty post-tool reply without leaving the user wondering).
        """
        target = self._find_user_reminder(caller, reminder_id)
        if target is None:
            if irc is not None and msg is not None:
                self._react(irc, msg, "❌")
            return f"Reminder {reminder_id} not found."

        self._cancel_reminder(target)
        if irc is not None and msg is not None:
            self._react(irc, msg, "👍")
        return f"Deleted reminder {reminder_id}."

    def _remind_clear_for_assistant(
        self,
        caller: Identity,
        *,
        irc: callbacks.Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> str:
        """Cancel all pending reminders for the caller in one shot.

        Snapshots the user's reminders then removes each — single atomic
        operation from the LLM's perspective so a recurring reminder can't
        slip in a fire between the model's tool calls.
        """
        user_reminders = self._get_user_reminders(caller)
        for name, _data in user_reminders:
            self._cancel_reminder(name)
        count = len(user_reminders)
        if irc is not None and msg is not None:
            self._react(irc, msg, "👍" if count else "👌")
        if count == 0:
            return "No pending reminders to cancel."
        if count == 1:
            return "Cancelled 1 reminder."
        return f"Cancelled {count} reminders."

    def remind(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<reminder text> | list | del(ete) <id> [<id>...] | clear]

        Set and manage reminders using natural language. If your reminder
        asks the bot to *do* something (look something up, check a status,
        fetch a URL), it will run that as an LLM query at fire time;
        otherwise it just echoes your text. Reminders marked [auto] in
        `list` are LLM actions.

        Examples:
          %remind in 30 minutes check the build
          %remind in 2 hours check status of CVE-2026-31431 in Debian
          %remind list
          %remind delete abc1
          %remind clear
        """
        caller = self._resolve_identity(irc, msg)

        if not text:
            self._remind_list(irc, caller)
            return

        parts = text.split(None, 1)
        subcommand = parts[0].lower()

        if subcommand == "list":
            self._remind_list(irc, caller)

        elif subcommand in ("delete", "del") and len(parts) >= 2:
            raw_ids = text.split()[1:]
            deleted = 0
            for rid in raw_ids:
                target = self._find_user_reminder(caller, rid)
                if target:
                    self._cancel_reminder(target)
                    deleted += 1
            if deleted == 0:
                self._react(irc, msg, "❌")
                irc.error(_("No matching reminders found."))
            else:
                label = "reminder" if deleted == 1 else "reminders"
                self._ack(irc, msg, "👍", f"Cancelled {deleted} {label}.")

        elif subcommand == "clear":
            user_reminders = self._get_user_reminders(caller)
            if not user_reminders:
                self._ack(irc, msg, "👌", _("No reminders to clear."))
                return
            for name, _data in user_reminders:
                self._cancel_reminder(name)
            label = "reminder" if len(user_reminders) == 1 else "reminders"
            self._ack(irc, msg, "👍", f"Cleared {len(user_reminders)} {label}.")

        else:
            self._remind_set(irc, msg, caller, text)

    remind = wrap(remind, [optional("text")])


Class = LLM
