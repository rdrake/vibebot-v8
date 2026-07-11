"""LLM plugin implementation."""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import mimetypes
import re
import secrets
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
from .context import ContextConfig, ConversationContext, Role
from .executor import LLMExecutor, RecursiveSubmitError
from .persistence import LLMDatabase, ReminderRow
from .profile import (
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
    PROFILES,
)
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
from .verse import reactions
from .verse.aging import AgingOutcome
from .verse.avatar import (
    _VerseToolResult,
    build_verse_system_prompt,
    dispatch_verse_edit,
    is_ooc,
    make_verse_denial_handlers,
    make_verse_extra_handlers,
    make_verse_tool_specs,
    strip_ooc,
)
from .verse.compaction import CompactionOutcome
from .verse.store import VerseStore

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg

    from .assistant import ToolCallbackResult, ToolResult
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
    {
        "llm.ask",
        "llm.code",
        "llm.draw",
        "llm.verse",
        "llm.verse.gm",
        "llm.verse.edit",
        "owner",
        "admin",
        "trusted",
    }
)

# Pending-task mutation tools whose successful execution already produced
# a user-visible emoji reaction. When the assistant loop ends with one of
# these as the last successful tool AND no follow-up text, the chat
# reply is suppressed to avoid a duplicate ack. See Task B5 of the
# 2026-04-30 reminder simplification plan.
_REMINDER_MUTATION_TOOLS = frozenset(
    {"set_reminder", "cancel_pending_task", "cancel_all_pending_tasks"}
)

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


class VerseRoute(NamedTuple):
    """Result of _verse_route_for.  Populated when the message should be
    handled by the verse engine (verseEnabled, llm.verse capability, avatar
    exists, not OOC)."""

    avatar_id: int
    system_prompt: str
    tools: list[dict]
    store: VerseStore


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
        name="story",
        args="<brief>",
        description=(
            "Generate an illustrated page (prose plus a few AI illustrations) and "
            "post a link when it's ready. Tells an illustrated tale OR explains a "
            "concept with diagrams as a learning aid. No verse mode needed."
        ),
        examples=(
            "%story an illustrated tale of stinky lads winning the pub quiz",
            "%story explain how photosynthesis works, with diagrams",
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
        name="avatar",
        args="[<persona> | clear]",
        description=(
            "Set the persona that shapes your avatar in verse-enabled channels. "
            "Independent of %instruct — this only affects the verse, not %ask."
        ),
        examples=(
            "%avatar A moss-covered tree spirit who speaks in riddles.",
            "%avatar clear",
            "%avatar",
        ),
        category="memory",
    ),
    CommandInfo(
        name="remind",
        args="[<text> | list | del <id> | clear | admin <list|del|clear> <nick> [<id>...]]",
        description="Set and manage reminders using natural language.",
        examples=(
            "%remind in 30 minutes check the build",
            "%remind list",
            "%remind delete abc1",
            "%remind clear",
            "%remind admin list someone",
            "%remind admin del someone abc1",
            "%remind admin clear someone",
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
        name="verseopt",
        args="<in|out>",
        description=(
            "Opt your avatar in or out of the verse for this channel. "
            "Requires the llm.verse capability."
        ),
        examples=("%verseopt in", "%verseopt out"),
        category="utility",
    ),
    CommandInfo(
        name="verse",
        args="",
        description=(
            "Show your current scene one-liner in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("%verse",),
        category="utility",
    ),
    CommandInfo(
        name="look",
        args="[<target>]",
        description=(
            "Show your current scene, or describe a named entity in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("%look", "%look The Clearing", "%look alice"),
        category="utility",
    ),
    CommandInfo(
        name="who",
        args="",
        description=(
            "List active avatars and their locations in the verse. "
            "Requires the llm.verse capability and a verse-enabled channel."
        ),
        examples=("%who",),
        category="utility",
    ),
    CommandInfo(
        name="versedump",
        args="[#channel] [--format=json]",
        description=(
            "Dump the full verse state for a channel as JSON. Requires the llm.verse.gm capability."
        ),
        examples=("%versedump", "%versedump #afnet", "%versedump #afnet --format=json"),
        category="utility",
    ),
    CommandInfo(
        name="versepurge",
        args="[#channel] [token]",
        description=(
            "Wipe the verse store for a channel. "
            "First call issues a confirmation token; second call with the token performs the wipe. "
            "Requires the llm.verse.gm capability."
        ),
        examples=("%versepurge #afnet", "%versepurge #afnet a1b2c3"),
        category="utility",
    ),
    CommandInfo(
        name="versecompact",
        args="<channel>",
        description=(
            "Manually run retention compaction for a channel: summarise old "
            "events into a single digest entry. Requires the llm.verse.gm capability."
        ),
        examples=("%versecompact #afnet",),
        category="utility",
    ),
    CommandInfo(
        name="versedit",
        args="<verb> <args...> [#channel]",
        description=(
            "Edit the verse universe: add/pin/unpin/set/name/desc/retire/restore/"
            "relate/unrelate/event/editevent/delevent/show. "
            "Requires the llm.verse.edit capability."
        ),
        examples=(
            "%versedit add npc Assgas Archie :: Y11 windbag",
            "%versedit pin Assgas Archie",
            "%versedit retire #42",
        ),
        category="utility",
    ),
    CommandInfo(
        name="canon",
        args="<lock|unlock|forget> <name>",
        description=(
            "Lock or release a character as durable canon (always remembered, "
            "aging-exempt). 'forget' is an alias for 'unlock'. "
            "Requires the llm.verse.edit capability and a verse-enabled channel."
        ),
        examples=("%canon lock Harry", "%canon unlock Harry", "%canon forget Harry"),
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


def _safe_database_path(configured: str, default: str) -> str:
    """Return the operator-configured DB path, or ``default`` when it is empty
    or contains a ``..`` traversal component.

    ``databasePath`` is operator-set (capability-gated), so this is
    defense-in-depth: it refuses an obvious traversal like
    ``../../../etc/passwd`` rather than handing it straight to SQLite. Plain
    absolute paths are the normal case and are kept as-is.
    """
    if not configured:
        return default
    if ".." in Path(configured).parts:
        return default
    return configured


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


def _format_compaction_outcome(
    co: CompactionOutcome,
    ao: AgingOutcome | None,
    *,
    min_keep_events: int,
) -> str:
    """Render a single human-readable line covering both compaction and
    (optional) aging counts.

    The two halves used to be separate INFO log records — operators had
    to correlate them by channel name. Folding them into one message
    keeps a daily-pass channel summary on a single line.

    ``ao=None`` is for callers (``@versecompact``) that compact one
    channel without running aging; the aged clause is omitted.
    """
    if co.state == "compacted":
        head = f"compacted {co.total_events} events"
    elif co.state == "skipped_below_floor":
        head = f"skipped (only {co.total_events} events; floor is {min_keep_events})"
    elif co.state == "skipped_no_events":
        head = f"skipped (no events past retention; total {co.total_events})"
    elif co.state == "skipped_disabled":
        head = "skipped (retention disabled)"
    else:  # forward-compat: unknown future state strings
        head = co.state
    if ao is None:
        return head
    aged_kept = ao.scanned - ao.retired
    return f"{head}; aged {ao.retired} entities (kept {aged_kept})"


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

        # Global concurrency cap for all LLM I/O. See
        # docs/plans/2026-05-06-async-llm-concurrency.md
        self._llm_executor = LLMExecutor(
            max_concurrency=self.registryValue("maxConcurrentLLMCalls"),
            log=self.log,
        )

        self.startup_time = time.time()  # Track startup for ZNC playback filtering
        self.build_info = self._get_build_info()

        # Initialize database for persistence (before context, which loads from DB)
        configured_db_path = self.registryValue("databasePath")
        default_db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
        db_path = _safe_database_path(configured_db_path, default_db_path)
        if configured_db_path and db_path != configured_db_path:
            self.log.warning(
                "databasePath %r rejected (path traversal); using default %r",
                configured_db_path,
                db_path,
            )
        self.db = LLMDatabase(db_path)

        _patch_irc_dojoin(self)
        _patch_irc_docapnew()

        # Initialize conversation context (loads persisted conversations from DB)
        self._init_context()

        # Track nicks already migrated to account-based identity this session
        self._migrated_nicks: set[str] = set()
        self._migrated_nicks_lock = threading.Lock()

        # In-memory per-command rate-limit buckets: "{command}:{account}" -> deque of timestamps
        self._rate_buckets: dict[str, collections.deque[float]] = {}
        self._rate_buckets_lock = threading.Lock()

        # Channels already warned about an empty verseModel (warn once per channel).
        self._verse_model_warned: set[str] = set()

        self._reminders: dict[str, ReminderRow] = {}
        self._reminders_lock = threading.Lock()

        # Serializes worker-thread irc.queueMsg calls (see _safe_queue).
        self._irc_send_lock = threading.Lock()

        # Recency-attributed verse reaction signal (see verse/reactions.py).
        # Last verse line the bot said per (network, channel); read by doTagmsg.
        self._last_bot_line: dict[tuple[str, str], dict] = {}
        self._reaction_log_lock = threading.Lock()

        # Per-channel VerseStore cache (keyed by channel name).
        self._verse_stores: dict[str, VerseStore] = {}
        self._verse_stores_lock = threading.Lock()

        # In-memory versepurge confirmation tokens: channel -> (token, expires_at).
        # Resets on plugin reload/bot restart (by design; documented in operator guide).
        self._versepurge_tokens: dict[str, tuple[str, float]] = {}
        self._versepurge_tokens_lock = threading.Lock()

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

        # Safety poll for pending tasks (5-minute fallback for event-driven wakeups).
        # Initialize the in-flight gate BEFORE scheduling — a synchronous fire
        # during construction would NameError without the flag.
        self._safety_poll_inflight = threading.Event()
        with contextlib.suppress(KeyError):
            schedule.removeEvent("llm_pending_tasks")

        schedule.addPeriodicEvent(
            self._enqueue_safety_poll,
            self._SAFETY_POLL_INTERVAL,
            name="llm_pending_tasks",
            now=False,
        )

        # Event-driven queue wakeup state
        self._next_wakeup_time: float | None = None
        self._schedule_queue_wakeup()  # rebuild from DB on startup

        # Register callback for live log level changes
        conf.supybot.plugins.LLM.logLevel.addCallback(self._on_log_level_change)

        # Daily verse compaction timer (PR 3 / E3). Registry keys
        # ``verseCompactionDailyAt`` and ``verseCompactionMinKeepEvents``
        # are added by F1; until then ``_register_compaction_timer``'s
        # try/except falls back to documented defaults so the timer
        # still arms on un-configured installs.
        self._compaction_timer_name = "llm_verse_compact"
        self._register_compaction_timer()

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
        # Shutdown the executor before mutating shared state so workers
        # see closing=True at their commit points. Brief drain gives
        # already-running workers a chance to flush final
        # db.log_usage / queueMsg writes before we close the DB.
        if hasattr(self, "_llm_executor"):
            self._llm_executor.shutdown()
            self._llm_executor.drain(timeout=2.0)

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
        # Daily compaction timer teardown (PR 3 / E3).
        if hasattr(self, "_compaction_timer_name"):
            self._cancel_compaction_timer()

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

    def _enqueue_safety_poll(self) -> None:
        """Submit a single safety-poll worker, deduped by ``_safety_poll_inflight``.

        The flag prevents overlapping polls when a long LLM dispatch holds
        the worker; we never queue more than one inflight at a time.
        """
        if self._llm_executor.closing:
            return
        if self._safety_poll_inflight.is_set():
            return
        self._safety_poll_inflight.set()
        try:
            fut = self._llm_executor.submit("safety_poll", self._check_pending_tasks)
        except Exception:
            # Synchronous submit failure must not leave the flag stuck set.
            self._safety_poll_inflight.clear()
            raise
        fut.add_done_callback(lambda _f: self._safety_poll_inflight.clear())

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
            self._enqueue_safety_poll,
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
            if not self._llm_executor.closing:
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
                # Try to save code to HTTP URL; the prompt makes a better page
                # <title> than the constant "Code" for URL-title bots.
                url = self.llm_service.save_code_to_http(r.content, title=prompt_preview)
                if url:
                    text = f'{nick}: your code is ready! "{prompt_preview}" \u2192 {url}'
                else:
                    text = f"{nick}: {content}"
            elif r.task_type == "draw":
                text = f'{nick}: your image is ready! "{prompt_preview}" \u2192 {content}'
            else:
                # ask or fallback (incl. recovered verse, which is unbounded —
                # verse timeouts recover under the "ask" task_type). Long content
                # is saved to the HTTP server and replaced with a teaser + URL,
                # mirroring the live _send_long_reply path; without this a
                # multi-paragraph scene becomes one oversized PRIVMSG the server
                # silently truncates. Short content stays inline (collapsed below).
                text = self._format_pending_completed_reply(nick, content)
        else:
            return

        # Pending-task delivery bypasses _send_long_reply, so collapse multi-line
        # content. A raw \n on PRIVMSG triggers Excess Flood disconnects.
        text = self._collapse_for_irc(text) or text

        # Try to deliver via IRC. _safe_queue serializes workers and short-
        # circuits cleanly when shutting down — a False return must not
        # advance durable delivery state.
        delivered = False
        try:
            for irc_conn in world.ircs:
                if r.is_channel:
                    if target in irc_conn.state.channels:
                        if self._safe_queue(irc_conn, ircmsgs.privmsg(target, text)):
                            delivered = True
                        break
                else:
                    # PM delivery — use first available connection
                    if self._safe_queue(irc_conn, ircmsgs.privmsg(target, text)):
                        delivered = True
                    break
        except Exception as e:
            self.log.warning(
                "queueMsg failed for task_id=%s: %s",
                r.task_id,
                self.llm_service.sanitize_output(str(e)),
            )
            delivered = False

        # Acknowledge or retry delivery for durable results.
        #
        # A successful send MUST be acked even if shutdown began after the
        # message went out — otherwise the row survives and is re-delivered
        # next process lifetime (duplicate IRC send). Only the not-delivered
        # branch (retry-state writes + wakeup) and usage logging are skipped
        # while closing, since nothing was sent and leaving the row for the
        # next lifetime is the correct, mutation-free behavior.
        if r.task_id is not None:
            if delivered:
                # Best-effort ack: a transient delete failure (e.g. DB lock)
                # after a successful send must not bubble up as a misleading
                # "delivery failed". The row simply re-delivers next tick
                # (at-least-once), which beats losing a delivered result.
                try:
                    self.db.delete_pending_task(r.task_id)
                except Exception as e:
                    self.log.warning(
                        "Delivered task_id=%s but ack(delete) failed; may re-deliver: %s",
                        r.task_id,
                        self.llm_service.sanitize_output(str(e)),
                    )
            elif not self._llm_executor.closing:
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

        # Log usage for completed tasks (a durable-state write — skip while closing).
        if r.status == "completed" and delivered and not self._llm_executor.closing:
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
            msg = ircmsgs.IrcMsg(msg=msg, args=(msg.args[0], cleaned))
            text = cleaned

        # Gate Limnoria's command dispatcher: only messages prefixed with
        # the configured command character reach Owner.doPrivmsg's
        # tokenizer. Nick-addressed channel messages and unprefixed PMs are
        # routed through the assistant in doPrivmsg instead — otherwise
        # plain English verbs like "remove", "later", or "search" collide
        # with built-in plugins and surface ambiguity errors. We tag
        # msg.addressed=''; callbacks.addressed() reads the cached tag,
        # returns '', and Owner.doPrivmsg skips dispatch.
        if msg.prefix and not ircutils.strEqual(irc.nick, msg.nick):
            prefix_chars = conf.supybot.reply.whenAddressedBy.chars()
            if not text or text[0] not in prefix_chars:
                msg.tag("addressed", "")

        return msg

    def doPrivmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Route addressed text to the assistant; observe channel chatter for context.

        Two paths:

        - Addressed text (channel message starting with our nick, or any PM
          without the configured command-prefix character) goes to
          ``_ask_impl``. The Limnoria command dispatcher is suppressed for
          these by ``inFilter``.
        - Other channel messages flow through the existing
          ``contextTrackAllMessages`` capture logic.

        Explicit prefix-char commands (e.g. ``@search`` or ``@later``) are
        handled entirely by Limnoria's dispatcher and short-circuit before
        the assistant routing.
        """
        # Universal early-outs before any branching.
        if self._is_old_message(msg):
            return
        if ircmsgs.isCtcp(msg) and not ircmsgs.isAction(msg):
            return
        if not msg.prefix or ircutils.strEqual(irc.nick, msg.nick):
            return
        # Server-originated PRIVMSGs (e.g. some services responses) carry
        # a server prefix instead of nick!user@host. Downstream code
        # (preflight, account resolution, rate-limit identity) calls
        # ``nickFromHostmask`` which asserts user-hostmask form, so drop
        # these here rather than letting every call site guard.
        if not ircutils.isUserHostmask(msg.prefix):
            return

        text = msg.args[1] if len(msg.args) > 1 else ""
        if not text:
            return

        prefix_chars = conf.supybot.reply.whenAddressedBy.chars()
        if text[0] in prefix_chars:
            # Explicit command — Limnoria's dispatcher handles it.
            return

        target = msg.args[0]
        is_pm = ircutils.nickEqual(target, irc.nick)
        addressed_text = text.strip() if is_pm else self._strip_nick_prefix(irc.nick, text)

        if addressed_text:
            self._route_addressed_to_assistant(irc, msg, addressed_text)
            return

        # Not addressed — channel chatter only.
        channel = msg.channel
        if not channel:
            return

        if not self.registryValue("contextEnabled", channel):
            return
        if not self.registryValue("contextTrackAllMessages", channel):
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

    def doTagmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Capture inbound IRCv3 emoji reactions (+draft/react) to verse lines.

        Recency-attributed, measurement only (no reply). Fully exception-isolated
        so a capture bug can never disturb the IRC event loop. See verse/reactions.py.
        """
        try:
            server_tags = getattr(msg, "server_tags", None) or {}
            react_emoji = server_tags.get("+draft/react")
            if not react_emoji:
                return
            if ircutils.strEqual(msg.nick, irc.nick):  # never count the bot's own reactions
                return
            channel = msg.channel or (msg.args[0] if msg.args else "")
            if not channel or not channel.startswith(("#", "&")):
                return
            if not self.registryValue("verseReactionCaptureEnabled", channel):
                return
            with self._irc_send_lock:
                last = self._last_bot_line.get((irc.network, ircutils.toLower(channel)))
            event = reactions.process_reaction(
                react_emoji=react_emoji,
                reactor=msg.nick,
                channel=channel,
                network=irc.network,
                target_msgid=server_tags.get("+draft/reply"),
                last_bot_line=last,
                now=time.time(),
                capture_enabled=True,
            )
            if event is not None:
                self._append_reaction_event(event)
        except Exception:
            self.log.exception("doTagmsg reaction capture failed")

    def _append_reaction_event(self, event: dict) -> None:
        """Append one reaction event to <data>/verse/reactions.jsonl (thread-safe)."""
        base = Path(conf.supybot.directories.data()) / "verse"
        path = base / "reactions.jsonl"
        with self._reaction_log_lock:
            base.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(reactions.event_to_jsonl(event) + "\n")

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

    def _send_reminder_text(
        self,
        irc: callbacks.Irc,
        target: str,
        nick: str,
        text: str,
    ) -> None:
        """Match the inner _send closure semantics for reminder output.

        Reminders bypass _send_long_reply, so collapse multi-line content
        here — raw \\n in a PRIVMSG body causes Excess Flood disconnects
        on AfterNET.
        """
        safe_text = self.llm_service.sanitize_output(text)
        safe_text = self._collapse_for_irc(safe_text) or safe_text
        prefixed = f"{nick}: {safe_text}" if nick else safe_text
        self._safe_queue(irc, ircmsgs.privmsg(target, prefixed))

    def _finalize_reminder_fire(
        self,
        *,
        event_name: str,
        is_structured: bool,
        reschedule_kwargs: dict | None,
    ) -> None:
        """Cleanup + (optionally) reschedule for a fired reminder.

        Used by every fire branch — main-thread early returns (no IRC,
        rate-limited) AND the worker's finally — so a reminder is never
        left half-cleaned. Closing-gated so a shutting-down plugin does
        not mutate ``_reminders`` or DB.
        """
        if self._llm_executor.closing:
            return
        if is_structured and reschedule_kwargs is not None:
            self._mechanical_reschedule(**reschedule_kwargs)
        with self._reminders_lock:
            self._reminders.pop(event_name, None)
        self.db.delete_reminder(event_name)

    def _fire_reminder_action(
        self,
        *,
        active_irc: callbacks.Irc,
        target: str,
        nick: str,
        channel: str,
        message: str,
        event_name: str,
        action_prompt: str,
        account: str | None,
        bot_nick: str,
        parent_chain: int,
        is_structured: bool,
        recurrence_seconds: int | None,
        recurrence_rrule: str | None,
        watch_mode: bool,
        now: float,
    ) -> None:
        """Worker-thread reminder action body.

        Resolves history, performs the LLM call, dispatches the response,
        logs usage, and finalizes (which optionally reschedules). The
        finalize step always runs in the worker's finally regardless of
        inner-try outcome — a single failed fire no longer kills a
        recurring watch chain (intentional behavior change).
        """
        reschedule_kwargs = (
            {
                "nick": nick,
                "channel": channel,
                "message": message,
                "event_name": event_name,
                "action_prompt": action_prompt,
                "account": account,
                "chain_position": parent_chain,
                "recurrence_seconds": recurrence_seconds,
                "recurrence_rrule": recurrence_rrule,
                "watch_mode": watch_mode,
                "now": now,
            }
            if is_structured
            else None
        )
        try:
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
                entry_route=PROFILE_REMIND_ACTION,
                profile=PROFILE_REMIND_ACTION,
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
            ask_prompt = self.registryValue(
                PROFILES[PROFILE_REMIND_ACTION].overlay_setting, channel
            )
            # The user's @instruct rides as user-role data (see assistant_request
            # user_instruction), not prepended to the system overlay.
            effective_prompt = ask_prompt

            caller = Identity(raw_nick=nick, account=account)
            exclude_tools = frozenset({"set_reminder"}) if is_structured else frozenset()

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
                user_instruction=user_instruction,
                system_prompt=effective_prompt,
                search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                code_fn=lambda p: self._code_for_assistant(p, channel),
                draw_fn=lambda p, _irc=active_irc, _msg=synthetic_msg: self._draw_for_assistant(
                    _irc, _msg, p
                ),
                cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                exclude_tools=exclude_tools,
                **self._pending_task_fns(
                    caller=caller,
                    irc=active_irc,
                    msg=synthetic_msg,
                    channel=channel,
                    pass_irc_msg_to_callbacks=False,
                ),
            )
            response = result.content.strip() if result.content else ""
            is_silent = response == "[silent]"
            if is_silent:
                pass  # No user-visible output this fire.
            elif not response:
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    f"Reminder: {message} (action returned empty response)",
                )
            elif message:
                self._send_reminder_text(
                    active_irc, target, nick, f"Reminder ({message}): {response}"
                )
            else:
                self._send_reminder_text(active_irc, target, nick, f"Reminder: {response}")
            # Attribute the action fire's LLM cost to the chain owner.
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
                self.log.exception("reminder_action_usage_log_failed event=%s", event_name)
        except Exception:
            self.log.exception(
                "reminder_action_delivery_failed nick=%s channel=%s event=%s",
                nick,
                channel,
                event_name,
            )
            try:
                self._send_reminder_text(
                    active_irc,
                    target,
                    nick,
                    f"Reminder action '{message}' failed. (Set this reminder again to retry.)",
                )
            except Exception:
                self.log.exception(
                    "reminder_action_fallback_failed nick=%s channel=%s event=%s",
                    nick,
                    channel,
                    event_name,
                )
        finally:
            # Single source of truth for cleanup. Runs even if the inner
            # try raised — recurring chains survive a single failed fire.
            self._finalize_reminder_fire(
                event_name=event_name,
                is_structured=is_structured,
                reschedule_kwargs=reschedule_kwargs,
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

        The returned callable runs on the main IRC scheduler thread:
        resolves the active IRC connection, runs the rate-limit precheck,
        then either submits the heavy LLM-call body to the executor (for
        action-prompt reminders) or sends the legacy plain-echo reminder
        directly. Cleanup + reschedule run in `_finalize_reminder_fire`.

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
        parent_chain: int = chain_position or 1
        is_structured = self._is_structured_recurring(
            recurrence_seconds=recurrence_seconds,
            recurrence_rrule=recurrence_rrule,
        )
        # If the command was sent via PM, channel is the bot's own nick.
        # Deliver to the user's nick instead.
        target = channel if ircutils.isChannel(channel) else nick

        def _build_reschedule_kwargs(now_t: float) -> dict | None:
            return (
                {
                    "nick": nick,
                    "channel": channel,
                    "message": message,
                    "event_name": event_name,
                    "action_prompt": action_prompt,
                    "account": account,
                    "chain_position": parent_chain,
                    "recurrence_seconds": recurrence_seconds,
                    "recurrence_rrule": recurrence_rrule,
                    "watch_mode": watch_mode,
                    "now": now_t,
                }
                if is_structured
                else None
            )

        def _deliver() -> None:
            now = time.time()
            submitted = False
            try:
                active_irc = next(iter(world.ircs), None)
                if active_irc is None:
                    return

                # Legacy reminder path: plain echo delivery.
                if not action_prompt:
                    self._send_reminder_text(active_irc, target, nick, f"Reminder: {message}")
                    return

                bot_nick = getattr(active_irc, "nick", None)
                if not bot_nick:
                    self.log.warning(
                        "reminder_action_missing_bot_nick nick=%s channel=%s event=%s",
                        nick,
                        channel,
                        event_name,
                    )
                    self._send_reminder_text(active_irc, target, nick, f"Reminder: {message}")
                    return

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
                    self._send_reminder_text(
                        active_irc,
                        target,
                        nick,
                        f"Reminder: {message} (action skipped — daily ask limit reached)",
                    )
                    return

                if self._llm_executor.closing:
                    return

                self._llm_executor.submit(
                    f"reminder:{event_name}",
                    self._fire_reminder_action,
                    active_irc=active_irc,
                    target=target,
                    nick=nick,
                    channel=channel,
                    message=message,
                    event_name=event_name,
                    action_prompt=action_prompt,
                    account=account,
                    bot_nick=bot_nick,
                    parent_chain=parent_chain,
                    is_structured=is_structured,
                    recurrence_seconds=recurrence_seconds,
                    recurrence_rrule=recurrence_rrule,
                    watch_mode=watch_mode,
                    now=now,
                )
                submitted = True
            finally:
                # Finalize on the main thread for every fire branch that
                # did NOT submit work to a worker. The worker's own
                # finally handles cleanup for submitted paths.
                if not submitted:
                    self._finalize_reminder_fire(
                        event_name=event_name,
                        is_structured=is_structured,
                        reschedule_kwargs=_build_reschedule_kwargs(now),
                    )

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

        self._dispatch_addressed_async(irc, msg, text, preflight, entry_route="invalid_command")

    @staticmethod
    def _strip_nick_prefix(bot_nick: str, text: str) -> str | None:
        """Return ``text`` with a leading bot-nick mention stripped, or None.

        Matches Limnoria's nick-addressing rules: optional leading whitespace,
        the bot's nick (case-insensitive), then a separator (whitespace, ``:``,
        ``,``, or ``;``). The nick must terminate at a non-alnum boundary so
        ``vibebotter`` is not treated as ``vibebot``.
        """
        stripped = text.lstrip()
        if not stripped:
            return None
        nick_len = len(bot_nick)
        if not ircutils.strEqual(stripped[:nick_len], bot_nick):
            return None
        rest = stripped[nick_len:]
        if not rest:
            return None
        if rest[0] not in " \t,:;":
            return None
        return rest.lstrip(" \t,:;").strip() or None

    def _route_addressed_to_assistant(self, irc: callbacks.Irc, msg: IrcMsg, text: str) -> None:
        """Run preflight and dispatch addressed text through ``_ask_impl``."""
        if not ircdb.checkCapability(msg.prefix, "llm.ask"):
            return
        preflight = self._run_preflight(irc, msg, text, "ask", require_account=False)
        if preflight.blocked:
            return
        self._dispatch_addressed_async(irc, msg, text, preflight, entry_route="addressed")

    def _dispatch_addressed_async(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        preflight: PreflightResult,
        *,
        entry_route: str,
    ) -> None:
        """Run addressed-text dispatch in a daemon thread so the IRC driver
        thread is freed to flush the typing indicator immediately.

        ``doPrivmsg`` (nick-addressed text and PMs) and ``invalidCommand``
        (bare ``vibebot foo`` that isn't a known command) are ``do*`` /
        invalid-command callbacks that Limnoria runs **synchronously** inside
        ``Irc.feedMsg`` on the socket driver's thread. That driver only
        flushes its outbound queue *after* the callback returns:
        ``supybot.drivers.Socket._read`` loops ``feedMsg`` over every received
        line, then calls ``_sendIfMsgs`` once at the end. Running the
        multi-second LLM generation inline therefore pins the driver — the
        ``+typing=active`` TAGMSG that ``_begin_typing`` enqueues cannot leave
        the socket until generation finishes, so "is composing" only shows up
        at the same instant as the reply. (Moving ``_begin_typing`` earlier in
        the call chain — already done — can't fix this; the bottleneck is the
        blocked flush, not call ordering.)

        Explicit ``@``-prefixed commands don't lag because Limnoria already
        runs them in a ``CommandThread`` (see ``callbacks.py``), which frees
        the driver to flush right away — that's why ``@ask`` shows typing
        promptly but nick-addressing doesn't. Offloading here mirrors that: a
        daemon ``SupyThread`` runs the dispatch, the callback returns at once,
        and the driver flushes the typing TAGMSG on its next loop iteration.
        LLM concurrency stays bounded by the ``_llm_executor.permit()`` inside
        ``_ask_impl`` (identical to the command path), so this unblocks the
        driver without adding new concurrency.

        Deliberately NOT routed through ``_llm_executor.submit``: that worker
        acquires the concurrency semaphore itself, and ``_ask_impl`` acquires
        it again via ``permit()``, so a submitted dispatch would double-acquire
        and deadlock once the pool fills — ``submit`` guards against exactly
        this nesting with ``RecursiveSubmitError``.
        """

        def _work() -> None:
            # Shutdown may have begun between spawn and execution. Bail
            # before any DB/LLM work so we don't read a database that die()
            # is tearing down: die() flips the executor's closing flag first
            # thing, so it doubles as the "shutdown in progress" signal for
            # these daemon threads, which are untracked and not awaited by
            # _llm_executor.drain().
            if self._llm_executor.closing:
                return
            try:
                self._dispatch_with_verse_routing(
                    irc, msg, text, preflight, entry_route=entry_route
                )
            except Exception:
                self.log.exception("addressed dispatch failed in worker thread")

        world.SupyThread(
            target=_work,
            name=f"llm-addressed-{entry_route}",
            daemon=True,
        ).start()

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
        # Server-originated messages (e.g. some operator NOTICEs) carry a
        # bare server prefix instead of nick!user@host. ``nickFromHostmask``
        # asserts user-hostmask form, so guard before calling.
        if not msg.prefix or not ircutils.isUserHostmask(msg.prefix):
            return None
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
        # Atomically claim the migration so concurrent request threads (the
        # @-command CommandThread, the addressed-dispatch SupyThread, and
        # executor workers all reach this) can't both run the body below.
        # The DB work stays outside the lock — only the claim needs to be
        # serialized.
        with self._migrated_nicks_lock:
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
        """Build the per-request Limnoria bridge tool schemas + handlers.

        Returns ``(None, None)`` when the bridge is disabled, the allowlist is
        empty, or no allowed command is currently exposable. Otherwise returns
        ``([run_schema, search_schema], {"run_limnoria_command": ...,
        "search_bridge_commands": ...})`` for injection into
        ``assistant_completion`` via ``extra_tools`` / ``extra_handlers``.

        When ``trace`` is provided, each successful or failed dispatch appends
        a ``(plugin, command, args, status)`` tuple — used by the optional
        ``bridgeDebugInChannel`` reply footer. ``search_bridge_commands`` calls
        append ``("bridge", "search", query, status)``.
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
        search_schema = {
            "type": "function",
            "function": {
                "name": "search_bridge_commands",
                "description": (
                    "Substring-search the available Limnoria bridge commands "
                    "by plugin name, command name, argument syntax, and "
                    "description text. Use this when Misc.apropos returns "
                    "nothing — apropos only matches command NAMES, this also "
                    "scans docstrings. Returns up to `limit` matches as "
                    "Plugin.command — argsyntax — description rows."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Whitespace-separated keywords to match "
                                "against command name, syntax, and description."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max matches to return (1-25).",
                            "minimum": 1,
                            "maximum": 25,
                        },
                    },
                    "required": ["query"],
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

        def search_handler(arguments):
            query = str(arguments.get("query", "")).strip()
            try:
                raw_limit = int(arguments.get("limit", 10))
            except (TypeError, ValueError):
                raw_limit = 10
            limit = max(1, min(25, raw_limit))
            if not query:
                envelope = {"error": "query is required"}
            else:
                matches = limnoria_bridge.search_commands(commands, query, limit=limit)
                envelope = {
                    "status": "ok",
                    "matches": [
                        {
                            "plugin": c.plugin,
                            "command": c.command,
                            "arg_syntax": c.arg_syntax,
                            "description": c.description,
                        }
                        for c in matches
                    ],
                }
            if trace is not None:
                status = (
                    "ok" if envelope.get("status") == "ok" else f"err:{envelope.get('error', '?')}"
                )
                trace.append(("bridge", "search", query, status))
            return ToolResult(content=json.dumps(envelope))

        return [schema, search_schema], {
            "run_limnoria_command": handler,
            "search_bridge_commands": search_handler,
        }

    @staticmethod
    def _format_bridge_debug_footer(trace: list) -> str:
        """Render a one-line debug footer for the optional bridgeDebugInChannel mode.

        This footer is sent to a public channel. Bridge args are LLM-generated
        and may carry secrets (e.g. an API key in a URL), so the raw arg_string
        is NEVER echoed — only its length, as a leak-free "args were passed"
        signal. Operators who need the content can read it from the DEBUG logs.
        """
        if not trace:
            return ""
        parts = []
        for plugin_name, command_name, arg_string, status in trace:
            call = f"{plugin_name}.{command_name}"
            if arg_string:
                call += f" ({len(arg_string)} chars)"
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

    def _format_pending_completed_reply(self, nick: str, content: str) -> str:
        """One-line delivery text for a completed ask/fallback pending result.

        Mirrors ``_send_long_reply``: content that collapses to a single IRC
        wire-line is sent inline; content that would overflow one wire-line is
        saved to the HTTP server and replaced with a teaser + URL. Pending
        delivery runs on the poll thread with no ``irc``/``msg`` and is already
        a deferred-recovery fallback, so a deterministic (non-LLM) teaser is
        used. Falls back to the inline/collapsed line when the save fails, so we
        never emit a raw multi-line body (the outer ``_collapse_for_irc`` still
        guards flood); the only residual risk is server-side truncation when the
        HTTP save itself is unavailable.
        """
        inline = f"{nick}: {content}"
        collapsed = self._collapse_for_irc(inline) or inline
        allowed = conf.supybot.reply.mores.length() or 400
        if len(ircutils.wrap(collapsed, allowed) or [collapsed]) <= 1:
            return collapsed
        url = self.llm_service.save_markdown_to_http(content)
        if not url:
            return collapsed
        suffix = f" - {_FULL_ANSWER_LABEL}: {url}"
        max_chars = max(0, allowed - len(suffix) - len(nick) - 2)
        teaser = self._fallback_long_reply_teaser(content, max_chars)
        return f"{nick}: {teaser}{suffix}"

    def _safe_queue(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """Thread-safe wrapper around irc.queueMsg for worker-thread sends.

        Limnoria's IrcMsgQueue.enqueue (irclib.py:245) mutates internal
        state without an explicit lock and the dedup check is TOCTOU.
        With the new executor pool, multiple workers may call queueMsg
        concurrently — serialize them on the plugin side and short-circuit
        cleanly when the plugin is shutting down.

        Returns True if the message was queued, False if dropped due
        to the plugin closing. Callers that mutate durable state on
        successful send must check the return.
        """
        if self._llm_executor.closing:
            self.log.debug(
                "safe_queue dropped (closing) command=%s",
                getattr(msg, "command", "?"),
            )
            return False
        with self._irc_send_lock:
            irc.queueMsg(msg)
        return True

    @staticmethod
    def _safe_privmsg(target: str, text: str) -> IrcMsg:
        """Build a PRIVMSG whose body is neutralized against IRC injection.

        Routes the body through Limnoria's ``ircutils.safeArgument`` (which
        repr()s any string containing CR, LF, or NUL) so model- or
        user-derived text cannot smuggle a second IRC command onto the wire.
        This is the raw-queue counterpart to the ``safeArgument`` that
        ``irc.reply`` applies on the chat-loop path: ``_safe_queue`` callers
        construct ``ircmsgs.privmsg`` directly and would otherwise rely solely
        on an ``IrcMsg.__init__`` assertion that disappears under ``python -O``.
        Callers should still ``_collapse_for_irc`` multi-line bodies first so a
        legitimate answer is split into a readable line rather than repr()'d.
        """
        return ircmsgs.privmsg(target, ircutils.safeArgument(text))

    def _safe_reply(
        self,
        irc: callbacks.Irc,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> bool:
        """Thread-safe wrapper around ``irc.reply`` for worker-thread sends.

        ``irc.reply`` ultimately reaches the same ``IrcMsgQueue.enqueue`` as
        ``irc.queueMsg``, so it must serialize on the same ``_irc_send_lock``:
        ``ask``/``code``/``draw`` release Limnoria's command RLock via
        ``_allow_concurrent`` and reply from concurrent worker threads.

        Returns True if the reply was sent, False if dropped due to the
        plugin closing.
        """
        if self._llm_executor.closing:
            self.log.debug("safe_reply dropped (closing)")
            return False
        with self._irc_send_lock:
            irc.reply(text, prefixNick=prefixNick)
        return True

    def _safe_error(
        self,
        irc: callbacks.Irc,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.error kwarg)
        Raise: bool = False,  # noqa: N803  (mirrors irc.error kwarg)
    ) -> None:
        """Thread-safe wrapper around ``irc.error`` for worker-thread sends.

        ``irc.error`` ultimately reaches the same ``IrcMsgQueue.enqueue`` as
        ``irc.queueMsg``, so it must serialize on the same ``_irc_send_lock``:
        ``ask``/``code``/``draw`` release Limnoria's command RLock via
        ``_allow_concurrent`` and error from concurrent worker threads.

        The ``Raise`` kwarg preserves Limnoria flow-control semantics: if the
        caller passes ``Raise=True`` the exception is re-raised after the lock
        is released so Limnoria's command dispatcher still sees it.
        """
        if self._llm_executor.closing:
            self.log.debug("safe_error dropped (closing)")
            return
        with self._irc_send_lock:
            irc.error(text, prefixNick=prefixNick, Raise=Raise)

    def _send_long_reply(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        *,
        prefixNick: bool = False,  # noqa: N803  (mirrors irc.reply kwarg)
    ) -> None:
        """Reply with ``text`` as one IRC line, pastebinning anything longer.

        Replies are NEVER paginated across multiple IRC messages: if ``text``
        fits in a single wire-line it is sent as-is; otherwise the whole
        answer is saved to the bot's HTTP server and the channel receives a
        one-line teaser plus the URL. When the save fails, we collapse to a
        single line ("teaser only") so we never trip Excess Flood with a raw
        multi-line body.
        """
        target = msg.channel if msg.channel else msg.nick
        allowed = (
            conf.get(conf.supybot.reply.mores.length, channel=target, network=irc.network) or 400
        )

        # Count rendered wire-lines: respect explicit \n, then byte-wrap each
        # line to allowed. Blank lines are noise on IRC and never count.
        raw_lines = text.split("\n") if "\n" in text else [text]
        logical_lines = [line for line in raw_lines if line.strip()]
        wire_line_count = 0
        single_line: str | None = logical_lines[0] if len(logical_lines) == 1 else None
        for line in logical_lines:
            wrapped = ircutils.wrap(line, allowed) or [line]
            wire_line_count += len(wrapped)
            if wire_line_count > 1:
                single_line = None
                break

        if single_line is not None:
            self._safe_reply(irc, single_line, prefixNick=prefixNick)
            return

        # One summary serves double duty: the page <title> (echoed by URL-title
        # bots in-channel) and the inline IRC teaser. Generated once, before the
        # save, so the title costs nothing extra over the teaser we already make.
        configured_max_chars = int(self.registryValue("longReplyTeaserMaxChars", target) or 220)
        summary = self.llm_service.summarize_for_irc(
            text, channel=target, max_chars=configured_max_chars
        )
        url = self.llm_service.save_markdown_to_http(text, title=summary)
        suffix = f" - {_FULL_ANSWER_LABEL}: {url}" if url else ""
        max_chars = min(configured_max_chars, max(0, allowed - len(suffix)))
        if max_chars <= 0 and url:
            self._safe_reply(irc, f"{_FULL_ANSWER_LABEL}: {url}", prefixNick=prefixNick)
            return
        teaser = summary or self._fallback_long_reply_teaser(text, max_chars)
        teaser = self._trim_long_reply_teaser(teaser, max_chars)
        self._safe_reply(irc, f"{teaser}{suffix}", prefixNick=prefixNick)

    def _record_last_verse_line(self, irc: callbacks.Irc, channel: str, text: str, result) -> None:
        """Remember the bot's last VERSE line per (network, channel) for reaction
        attribution. No-op for non-verse turns. Never disturbs the reply path."""
        if not getattr(result, "was_verse", False):
            return
        try:
            with self._irc_send_lock:
                self._last_bot_line[(irc.network, ircutils.toLower(channel))] = {
                    "text": text,
                    "ts": time.time(),
                }
        except Exception:
            self.log.exception("last_bot_line store failed")

    def _dispatch_assistant_reply(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        result: AssistantResult,
        *,
        nick: str,
        channel: str,
        response: str,
        suppress_reminder_mutations: bool = False,
    ) -> tuple[str, bool]:
        """Send the reply for an assistant result.

        Returns ``(response_for_context, should_log_and_store)``. When the
        response is empty (and not suppressed by a successful reminder
        mutation), the helper emits ``irc.error`` and returns
        ``(response, False)`` so the caller skips
        ``_store_context_and_log_usage``.

        Reminder-mutation suppression (``ask`` only) is checked BEFORE the
        empty-response error branch; that is the existing behaviour and must
        be preserved.
        """
        if suppress_reminder_mutations and (
            result.last_successful_tool in _REMINDER_MUTATION_TOOLS
            and not result.final_text_after_tools.strip()
        ):
            self.log.info(
                "suppressing empty post-reminder-mutation reply tool=%s %s/%s",
                result.last_successful_tool,
                channel,
                nick,
            )
            return response, True

        # verse_storybook delivers its illustrated page link from a background
        # job, so the assistant's post-tool reply is intentionally empty (see
        # the short-circuit in assistant_completion). Suppress it silently —
        # no message, no empty-response error; the async link is the reply.
        if (
            result.last_successful_tool == "verse_storybook"
            and not (result.final_text_after_tools or "").strip()
        ):
            self.log.info("suppressing verse_storybook interim reply %s/%s", channel, nick)
            return response, True

        if not response or not response.strip():
            self._safe_error(irc, _("The model returned an empty response. Please try again."))
            return response, False

        action_text = self._extract_action(irc, response)
        if action_text:
            if result.grounding_used:
                action_text = f"{GROUNDING_ICON} {action_text}"
            self.log.info("sending action to %s/%s", channel, nick)
            target = channel if ircutils.isChannel(channel) else nick
            if not self._safe_queue(irc, ircmsgs.action(target, action_text)):
                return response, False
            self._record_last_verse_line(irc, channel, action_text, result)
            return f"* {irc.nick} {action_text}", True

        display_response = f"{GROUNDING_ICON} {response}" if result.grounding_used else response
        self.log.info("replying to %s/%s", channel, nick)
        self._send_long_reply(irc, msg, display_response, prefixNick=False)
        self._record_last_verse_line(irc, channel, display_response, result)
        return response, True

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

    def _verse_triggered(
        self,
        channel: str,
        store: VerseStore,
        avatar_id: int,
        message_text: str,
    ) -> bool:
        """True when ``message_text`` should drop into verse mode.

        Two signals, either sufficient:
        1. The message names a known active in-universe entity (character,
           place, or item) other than the speaker's own avatar. Reuses
           ``store.match_entities_in_text`` (whole-word, alias-aware,
           capitalized-stoplist rules).
        2. The channel's ``verseTriggerRegex`` (case-insensitive ``re.search``)
           matches. Empty regex disables this signal; a malformed regex is
           ignored rather than raised into the message path.
        """
        if any(e.id != avatar_id for e in store.match_entities_in_text(message_text)):
            return True
        pattern = self.registryValue("verseTriggerRegex", channel)
        if pattern:
            try:
                if re.search(pattern, message_text, re.IGNORECASE):
                    return True
            except re.error:
                pass
        return False

    def _verse_route_for(
        self,
        channel: str,
        nick: str,
        account: str | None,
        message_text: str,
    ) -> VerseRoute | None:
        """Return a VerseRoute when the message should be handled by the verse
        engine, or None to fall through to the normal chat path.

        Gates (applied in order):
        1. verseEnabled must be True for the channel.
        2. User must hold the llm.verse capability.
        3. OOC messages (wrapped in ((...)) or with a leading //) bypass the
           verse engine.

        Body (avatar lookup + route construction) lands in C7c.
        """
        if not self.registryValue("verseEnabled", channel):
            return None
        # Build a synthetic hostmask for ircdb; nick!*@* is sufficient
        # because the test harness patches checkCapability on the prefix string.
        hostmask = f"{nick}!*@*"
        if not ircdb.checkCapability(hostmask, "llm.verse"):
            return None  # capability fallthrough — quiet
        if is_ooc(message_text):
            return None
        # Avatar lookup: account takes priority, then nick.
        store = self._get_or_create_verse_store(channel)
        avatar_id = (
            store.find_avatar_by_account(account) if account else None
        ) or store.find_avatar_by_nick(nick)
        if avatar_id is None:
            return None  # User opted into the channel but isn't in the verse → chat path.
        # Trigger gate: don't route every avatar-user message into verse. Only
        # route when the message carries a verse signal — a configured keyword
        # or a reference to a known in-universe entity (not the speaker's own
        # avatar). Otherwise fall through to the normal chat path.
        if not self._verse_triggered(channel, store, avatar_id, message_text):
            return None
        persona = self.db.get_avatar_persona(nick) or ""
        system_prompt = build_verse_system_prompt(
            store,
            avatar_id,
            persona,
            roster_max_chars=self.registryValue("verseRosterMaxChars", channel),
            message_text=message_text,
            style_exemplars=self.registryValue("verseStyleExemplars", channel),
        )
        max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
        tools = make_verse_tool_specs(
            max_actors=max_actors,
            storybook=bool(self.registryValue("verseStorybookEnabled", channel)),
        )
        return VerseRoute(avatar_id, system_prompt, tools, store)

    def _build_verse_handlers_for(self, channel: str) -> dict | None:
        """Build the verse extra-handlers dict for ``channel``, plumbing
        the per-channel ``verseAutoEntityMaxNamesPerCall`` into the
        dispatch closure.

        Looks up the channel's verse store + most-recent active avatar
        and binds handlers to that pair. Returns None if there is no
        verse store or no active avatar in the channel.

        The dispatch call site (which already has a full ``VerseRoute``
        from ``_verse_route_for``) prefers
        ``_build_verse_handlers_for_route`` so its avatar binding stays
        caller-specific; this helper is the seam that the registry-
        plumbing test relies on.
        """
        store = self._get_or_create_verse_store(channel)
        if store is None:
            return None
        avatars = store.list_entities_by_kind("avatar", status="active")
        if not avatars:
            return None
        max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
        return make_verse_extra_handlers(store, avatars[0].id, max_actors=max_actors)

    def _build_verse_handlers_for_route(self, channel: str, route: VerseRoute) -> dict:
        """Build verse handlers for an existing ``VerseRoute``.

        Reads ``verseAutoEntityMaxNamesPerCall`` (channel-scoped) and
        threads it into ``make_verse_extra_handlers`` so the dispatch
        closure carries the per-channel cap.
        """
        max_actors = self.registryValue("verseAutoEntityMaxNamesPerCall", channel)
        return make_verse_extra_handlers(route.store, route.avatar_id, max_actors=max_actors)

    def _storybook_cooldown_active(self, account: str | None, cooldown: int) -> bool:
        """Single-slot per-account storybook cooldown (reserve-at-start).

        Shared by the verse_storybook tool and the @story command via the
        ``verse_storybook:<account>`` rate-bucket key, so both paths draw from
        ONE cooldown window and can't be stacked to double image spend. Reuses
        ``_rate_buckets``/``_rate_buckets_lock`` (the machinery behind
        ``_is_rate_limited``). Records the hit immediately when not limited; no
        account or non-positive cooldown disables limiting.
        """
        if cooldown <= 0 or not account:
            return False
        key = f"verse_storybook:{account}"
        now = time.time()
        with self._rate_buckets_lock:
            bucket = self._rate_buckets.get(key)
            if bucket and (now - bucket[-1]) < cooldown:
                return True
            if bucket is None:
                bucket = collections.deque(maxlen=1)
                self._rate_buckets[key] = bucket
            bucket.append(now)
            return False

    def _submit_storybook_job(
        self, *, channel: str, nick: str, persona: str, brief: str, account: str | None = None
    ) -> None:
        """Fire a background job that renders a storybook for ``brief`` and
        posts the resulting link to ``channel``. Fire-and-return — never raises
        to the caller. Shared by the verse_storybook tool and the @story command.

        Delivery looks up the live IRC connection via ``world.ircs`` (not a
        captured, possibly-stale ``irc``), mirroring ``_deliver_pending_result``.
        """
        # Record a canon event so an illustrated turn isn't invisible to verse
        # retention (the tool/@story paths short-circuit before any verse record
        # step). Only for verse channels — @story works anywhere, and a bare
        # @story in a non-verse channel or PM must not lazily create a verse DB.
        # Best-effort: never break delivery if the store/avatar lookup fails.
        try:
            if ircutils.isChannel(channel) and self.registryValue("verseEnabled", channel):
                store = self._get_or_create_verse_store(channel)
                avatar_id = (
                    store.find_avatar_by_account(account) if account else None
                ) or store.find_avatar_by_nick(nick)
                if avatar_id is not None:
                    store.record_user_event(
                        actor_id=avatar_id,
                        summary=(brief.strip()[:200] or "told an illustrated tale"),
                        actor_names=[],
                    )
        except Exception:
            self.log.exception("storybook canon-record failed (non-fatal) channel=%s", channel)

        def _deliver(text: str) -> None:
            collapsed = self._collapse_for_irc(text) or text
            if ircutils.isChannel(channel):
                for irc_conn in world.ircs:
                    if channel in irc_conn.state.channels:
                        self._safe_queue(irc_conn, self._safe_privmsg(channel, collapsed))
                        return
                return
            # PM: the "channel" is the bot's own nick, never in state.channels —
            # deliver to the requesting nick on the first available connection.
            for irc_conn in world.ircs:
                if self._safe_queue(irc_conn, self._safe_privmsg(nick, collapsed)):
                    return

        def _job(b: str) -> None:
            try:
                res = self.llm_service.generate_storybook(
                    b, channel=channel, persona=persona, conversation=[]
                )
                if res is not None:
                    # @story's command wrapper returns before generation, so the
                    # shared _store_context_and_log_usage path never sees the
                    # spend — log the (image-dominated) usage here instead.
                    try:
                        self.db.log_usage(
                            nick=nick,
                            channel=channel,
                            command="story",
                            model=res.model,
                            prompt_tokens=res.prompt_tokens,
                            completion_tokens=res.completion_tokens,
                            cost=res.cost,
                            prompt=b[:200],
                        )
                    except Exception:
                        self.log.exception("storybook usage logging failed nick=%s", nick)
                    _deliver(f"the tale is told — {res.title}: {res.url}")
                else:
                    _deliver("the tale slipped away before it could be illustrated.")
            except Exception:
                self.log.exception("storybook job failed channel=%s nick=%s", channel, nick)

        try:
            self._llm_executor.submit("verse_storybook", _job, brief)
        except RecursiveSubmitError:
            _job(brief)

    def _storybook_handler(
        self,
        *,
        irc: callbacks.Irc,
        msg: IrcMsg,
        channel: str,
        account: str | None,
        nick: str,
        persona: str,
    ) -> Callable[[dict], _VerseToolResult]:
        """Build the ``verse_storybook`` tool handler for one completion.

        Returns a synchronous ``Callable[[dict], _VerseToolResult]`` matching
        the verse handler contract. The returned closure captures the caller
        context plus a mutable per-turn counter (scoped to this one handler
        instance → one completion).

        Gating: this factory is only invoked when ``verseStorybookEnabled``
        is True for the channel (see ``_ask_impl`` merge site), so when the
        flag is off the handler is never built and the tool is never even
        advertised. The flag-off verse path is byte-identical to before.

        On invocation the handler enforces, in order: per-turn cap → account
        gate → ``llm.draw`` capability → per-account cooldown. On success it
        fires a background job (which generates the storybook and posts the
        URL to the channel) and immediately returns a "generating" ack so the
        verse model can acknowledge in character without blocking the loop.
        """
        # Per-turn counter, captured by reference via a single-element list so
        # the inner closure can mutate it. Scoped to this handler instance.
        turn_count = [0]
        max_per_turn = int(self.registryValue("verseStorybookMaxPerTurn", channel) or 1)
        cooldown = int(self.registryValue("verseStorybookCooldownSeconds", channel) or 0)

        def _err(message: str) -> _VerseToolResult:
            return _VerseToolResult(content=json.dumps({"status": "error", "error": message}))

        def _call(args: dict) -> _VerseToolResult:
            # 1. Per-turn cap.
            if turn_count[0] >= max_per_turn:
                return _err("already told a tale this turn")
            # 2. Account gate.
            if not account:
                return _err("you must be authenticated to weave a storybook")
            # 3. Capability gate (image spend).
            if not ircdb.checkCapability(msg.prefix, "llm.draw"):
                return _err("you lack the standing to summon illustrations")
            # 4. Per-account cooldown (reserve-at-start).
            if self._storybook_cooldown_active(account, cooldown):
                return _err("the muse needs a moment")

            # TODO(storybook): daily image cap (verseStorybookDailyImageCap) —
            # no straightforward per-account daily image-count query exists on
            # the usage table; cooldown + per-turn cap already bound spend.

            # Reserve the per-turn slot only once we're actually firing.
            turn_count[0] += 1
            brief = ""
            if isinstance(args, dict):
                raw = args.get("brief")
                if isinstance(raw, str):
                    brief = raw
            self._submit_storybook_job(
                channel=channel, nick=nick, persona=persona, brief=brief, account=account
            )
            return _VerseToolResult(
                content=json.dumps(
                    {
                        "status": "ok",
                        "note": (
                            "The illustrated page is rendering in the background and "
                            "its link will be posted automatically the moment it is "
                            "ready. Do NOT announce that a story is coming, do NOT "
                            "mention drawing/rendering/waiting, and do NOT promise a "
                            "link. Just stay in character — reply with a brief beat or "
                            "say nothing further."
                        ),
                    }
                )
            )

        _call.__name__ = "_verse_handler_verse_storybook"
        return _call

    def _verse_edit_handler(
        self,
        *,
        msg: IrcMsg,
        store: VerseStore,
        account: str | None,
    ) -> Callable[[dict], _VerseToolResult]:
        """Build the ``verse_edit`` tool handler for one completion.

        The handler is bound to the TRIGGERING user's IRC prefix (``msg.prefix``)
        so the ``llm.verse.edit`` capability check reflects who actually drove
        this verse turn — NOT the avatar that happens to be active in the
        channel. This is the security-load-bearing gate: an unauthorized caller
        gets a refusal and nothing is written to the store.

        ``dispatch_verse_edit`` (the pure handler) enforces the
        constructive-ops whitelist and payload validation; this wrapper only
        resolves authorization and adapts the result dict to the
        ``_VerseToolResult`` JSON the assistant loop expects. A refused/error
        result is re-emitted with an ``error`` key so the loop's success
        detector (service.py: "error" not in parsed) counts it as a failure.
        """
        authorized = ircdb.checkCapability(msg.prefix, "llm.verse.edit")

        def _call(args: dict) -> _VerseToolResult:
            op = args.get("op") if isinstance(args, dict) else None
            payload = args.get("payload") if isinstance(args, dict) else None
            if not isinstance(payload, dict):
                payload = {}
            result = dispatch_verse_edit(
                store,
                op=op,
                payload=payload,
                authorized=authorized,
                account=account or "anon",
            )
            if result.get("status") != "ok":
                return _VerseToolResult(
                    content=json.dumps(
                        {
                            "status": result.get("status", "error"),
                            "error": result.get("detail", "verse_edit failed"),
                        }
                    )
                )
            return _VerseToolResult(content=json.dumps(result))

        _call.__name__ = "_verse_handler_verse_edit"
        return _call

    def _draw_for_assistant(
        self, irc: callbacks.Irc, msg: IrcMsg, prompt: str
    ) -> ToolCallbackResult:
        """Generate an image for the generate_image tool.

        Usage logging is handled by the outer command wrapper via
        ``_store_context_and_log_usage``; leaf tool handlers do not log
        independently.
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        result = self.llm_service.image_generation(prompt, irc=irc, msg=msg)
        return _ToolCallbackResult(not bool(result.error), result.content)

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
            url = self.llm_service.save_code_to_http(result.content, title=prompt)
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
            self._safe_error(irc, _("You must be identified to use this command."))
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
        with self._rate_buckets_lock:
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
        with self._rate_buckets_lock:
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
        with self._rate_buckets_lock:
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
                self._safe_error(
                    irc, _("Rate limit exceeded for %s. Try again in %ds.") % (command, window)
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

        Two-stage flow:

        1. The extractor proposes new candidate facts and/or reinforces
           existing ones. New entries land in ``memory_candidates`` with
           ``mentions=1``; reinforcements bump that counter.
        2. Once a candidate's ``mentions`` reaches
           ``memoryPromotionThreshold`` it is promoted to ``memories`` and
           removed from the candidate table. Untouched candidates older
           than ``memoryCandidateTTLDays`` are pruned on each pass.

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

            existing_candidates = self.db.get_memory_candidates(nick)
            candidate_facts = [c.fact for c in existing_candidates]
            snapshot_memory_ids = tuple(r.id for r in existing_rows)
            snapshot_candidate_ids = tuple(c.id for c in existing_candidates)

            def _extract_memories_bg() -> None:
                # Short-circuit at the top — extraction may have been
                # queued before die() but not yet started.
                if self._llm_executor.closing:
                    return
                try:
                    extraction = self.llm_service.extract_memories(
                        nick,
                        channel,
                        user_text,
                        assistant_response,
                        existing_facts,
                        candidate_facts,
                    )
                    if not extraction.add and not extraction.reinforce:
                        self._prune_stale_memory_candidates(nick)
                        return

                    # Race protection: abort if memories or candidates
                    # changed during the LLM call. The reinforce indices
                    # reference our candidate snapshot, so a stale list
                    # would mis-target rows.
                    current = self.db.get_memories(nick)
                    current_candidates = self.db.get_memory_candidates(nick)
                    if (
                        tuple(r.id for r in current) != snapshot_memory_ids
                        or tuple(c.id for c in current_candidates) != snapshot_candidate_ids
                    ):
                        log.info(
                            "Memory extraction for %s aborted: rows changed",
                            nick,
                        )
                        return

                    threshold = self.registryValue("memoryPromotionThreshold")
                    promoted: list[str] = []
                    current_count = len(current)

                    # Reinforce — bump existing candidates, promote any
                    # that cross the threshold.
                    for idx in extraction.reinforce:
                        if not 0 <= idx < len(existing_candidates):
                            continue
                        cand = existing_candidates[idx]
                        new_mentions = self.db.reinforce_memory_candidate(cand.id, nick)
                        if new_mentions == 0:
                            continue
                        if new_mentions >= threshold and current_count < max_memories:
                            self.db.save_memory(nick, cand.fact, cand.source_channel)
                            self.db.delete_memory_candidate(cand.id, nick)
                            promoted.append(cand.fact)
                            current_count += 1

                    # Add — new candidates start at mentions=1 (or promote
                    # immediately if threshold == 1).
                    for fact in extraction.add:
                        if threshold <= 1 and current_count < max_memories:
                            self.db.save_memory(nick, fact, channel)
                            promoted.append(fact)
                            current_count += 1
                        else:
                            self.db.add_memory_candidate(nick, fact, channel)

                    self._prune_stale_memory_candidates(nick)

                    if not promoted:
                        return

                    # Trigger cleanup if counter reaches interval — only
                    # promotions count toward the cleanup cadence, since
                    # they're what changes the durable memory set.
                    cleanup_interval = self.registryValue("memoryCleanupInterval")
                    if cleanup_interval > 0:
                        count = self.db.increment_memory_saves(nick)
                        if count >= cleanup_interval:
                            # Re-check closing before the cleanup pass:
                            # extraction itself can take seconds.
                            if self._llm_executor.closing:
                                return
                            self.db.reset_memory_saves(nick)
                            self._run_memory_cleanup(nick, channel)

                except Exception:
                    log.exception("Memory extraction failed for %s", nick)

            event_name = f"llm_memory_{uuid.uuid4().hex[:8]}"

            def _enqueue() -> None:
                if self._llm_executor.closing:
                    return
                self._llm_executor.submit(f"memory_extract:{nick}", _extract_memories_bg)

            schedule.addEvent(_enqueue, time.time() + 0.1, name=event_name)

        except Exception:
            log.exception("Memory extraction scheduling failed for %s", nick)

    def _prune_stale_memory_candidates(self, nick: str) -> None:
        """Drop candidates whose last_seen is older than the configured TTL.

        TTL of 0 disables decay (candidates linger until promoted or the
        admin clears them).
        """
        ttl_days = self.registryValue("memoryCandidateTTLDays")
        if ttl_days <= 0:
            return
        cutoff = time.time() - (ttl_days * 86400)
        try:
            self.db.prune_memory_candidates(nick, cutoff)
        except Exception:
            log.exception("Memory candidate pruning failed for %s", nick)

    def _run_memory_cleanup(self, nick: str, channel: str) -> ToolCallbackResult:
        """Run memory cleanup for a user. Returns a ToolCallbackResult."""
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        snapshot = self.db.get_memories(nick)
        if len(snapshot) < 2:
            return _ToolCallbackResult(True, "Not enough memories to clean up.")

        before_count = len(snapshot)
        result = self.llm_service.cleanup_memories(nick, channel, snapshot)

        if result.error:
            log.warning("Memory cleanup failed for %s: %s", nick, result.error)
            short = result.error.split(":")[0] if ":" in result.error else result.error
            return _ToolCallbackResult(False, f"Cleanup failed ({short}). Try again later.")

        # Abort if memory rows changed during LLM call (race protection).
        # Compare row IDs — a delete+insert preserving count would otherwise
        # let cleanup mis-target indices.
        current = self.db.get_memories(nick)
        snapshot_ids = tuple(r.id for r in snapshot)
        current_ids = tuple(r.id for r in current)
        if current_ids != snapshot_ids:
            return _ToolCallbackResult(True, "Cleanup skipped — memories changed while processing.")

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
        return _ToolCallbackResult(True, " | ".join(parts))

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

        self._dispatch_with_verse_routing(irc, msg, text, pf, entry_route="ask")

    ask = wrap(ask, [("checkCapability", "llm.ask"), "text"])

    def _dispatch_with_verse_routing(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        preflight: PreflightResult,
        *,
        entry_route: str,
    ) -> None:
        """Look up a VerseRoute for the (channel, nick, account, text) and
        dispatch to ``_ask_impl`` with the verse overrides applied when one
        matches; otherwise fall through to the chat path. Used from every
        call site that turns user-addressed text into an assistant request
        (the explicit ``@ask`` command, ``invalidCommand`` for bare
        ``vibebot foo`` text, and ``_route_addressed_to_assistant`` for
        nick-comma-prefixed text). Without this on every entry point a
        verse-enabled channel still falls back to the chat profile for any
        message that isn't routed through ``@ask``, so verse_record never
        fires and the canon goes unrecorded."""
        # Fire the typing indicator the moment we know preflight passed —
        # before any DB work (verse route lookup, history fetch, memory
        # fetch, executor permit acquisition). Otherwise the user waits
        # several seconds before "is composing" appears.
        stop_typing = self.llm_service._begin_typing(irc, msg)
        try:
            route = self._verse_route_for(
                preflight.channel, preflight.nick, preflight.account, text
            )
            if route is None:
                verse_enabled = self.registryValue("verseEnabled", preflight.channel)
                # OOC opt-out (((like this)) or a leading //): _verse_route_for
                # returns None for an avatar-holder who marked a message to skip
                # the verse engine. Strip the marker here so the chat model
                # receives a clean prompt instead of the literal ((...)) / //.
                # Gated on verseEnabled because these markers only carry OOC
                # meaning on a verse channel — elsewhere they are ordinary text.
                if verse_enabled and is_ooc(text):
                    text = strip_ooc(text)
                # Verse-enabled channel + non-opted-in speaker: advertise the
                # verse tool *schemas* anyway so the channel's tool surface is
                # byte-identical across all speakers. Invocations land on
                # denial handlers (see ``_ask_impl``). Without this branch the
                # tools list flips between roughly 15 and 23 entries depending
                # on opt-in state, which fragments xAI's automatic prompt
                # cache per-user instead of per-channel.
                if verse_enabled:
                    max_actors = self.registryValue(
                        "verseAutoEntityMaxNamesPerCall", preflight.channel
                    )
                    verse_specs = make_verse_tool_specs(
                        max_actors=max_actors,
                        storybook=bool(
                            self.registryValue("verseStorybookEnabled", preflight.channel)
                        ),
                    )
                    self._ask_impl(
                        irc,
                        msg,
                        text,
                        preflight,
                        entry_route=entry_route,
                        extra_tools_override=verse_specs,
                    )
                    return
                self._ask_impl(irc, msg, text, preflight, entry_route=entry_route)
                return
            # Per-channel verse model override. Empty string falls back to
            # assistantModel inside ``_ask_impl``. Useful when the channel's
            # assistantModel is a reasoning model that hard-caps verse output
            # at ~120 visible tokens regardless of prompt — point ``verseModel``
            # at a non-reasoning model (e.g. gemini-flash-latest) for richer
            # long-form scenes without affecting chat-mode behavior.
            verse_model = self.registryValue("verseModel", preflight.channel) or None
            if verse_model is None and preflight.channel not in self._verse_model_warned:
                self._verse_model_warned.add(preflight.channel)
                self.log.warning(
                    "verse turn on channel=%s has empty verseModel; falling back to "
                    "assistantModel — set a non-reasoning verseModel or verse prose may be "
                    "cratered by a reasoning model",
                    preflight.channel,
                )
            self._ask_impl(
                irc,
                msg,
                text,
                preflight,
                entry_route=entry_route,
                system_prompt_override=route.system_prompt,
                extra_tools_override=route.tools,
                profile_override=PROFILE_VERSE,
                verse_route=route,
                model_override=verse_model,
            )
        finally:
            stop_typing()

    def _ask_impl(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        text: str,
        pf: PreflightResult,
        *,
        entry_route: str = "ask",
        system_prompt_override: str | None = None,
        extra_tools_override: list[dict] | None = None,
        profile_override: str | None = None,
        verse_route: VerseRoute | None = None,
        model_override: str | None = None,
    ) -> None:
        """Core ask logic, separated so invalidCommand can reuse without double-preflight.

        Optional keyword-only overrides for verse routing (C7c/C7d):
        - ``system_prompt_override``: when set, replaces the normal
          ``assistantSystemPrompt`` personality overlay entirely.
        - ``extra_tools_override``: tool specs appended to the profile tool list.
        - ``profile_override``: when set, overrides the profile selection
          (e.g. PROFILE_VERSE to bypass the token cap).
        - ``verse_route``: when set, verse tool handlers are built and merged
          into extra_handlers so the assistant loop can dispatch them (C7d).
        """
        nick, channel = pf.nick, pf.channel
        effective_profile = profile_override or PROFILE_CHAT
        request_context = self._build_request_context(
            irc,
            msg,
            pf,
            entry_route=entry_route,
            profile=effective_profile,
        )

        caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

        with self._trace_request("ask", nick, channel):
            # Detect images for vision
            images = self.llm_service.detect_images(text)

            history, channel_history = self._gather_history(nick, channel)

            memories = self._get_user_memories(nick)
            user_instruction = self.db.get_instruction(nick)

            # Personality overlay assembly.
            #
            # Default (chat): channel ``assistantSystemPrompt``, optionally
            # prefixed with the user's persistent @instruct. The structural
            # framework (plain-text rules, tool-behavior rules) is layered in
            # by ``assistant_completion``.
            #
            # Verse: the override is just avatar identity + scene + recent
            # canon. Empirically dropping the channel's ``assistantSystemPrompt``
            # ("tell the tallest tales", "go mental") here cratered output
            # length on #afternet — grok happily produced 600+ tokens in chat
            # under the same model with that overlay, but only ~150 tokens in
            # verse without it. The energy/length pump comes from the
            # personality overlay, not the framework. So we PREPEND the
            # channel overlay (and @instruct) to the verse scene context;
            # the framework's verse-mode rules still apply on top.
            # The user's @instruct rides as user-role data (passed as
            # user_instruction below), NOT prepended to the system overlay —
            # a user request must not pose as system/developer authority. The
            # channel overlay (ask_prompt) and verse override are unchanged, so
            # the verse energy/length pump is preserved.
            ask_prompt = self.registryValue(PROFILES[effective_profile].overlay_setting, channel)
            if system_prompt_override is not None:
                parts: list[str] = []
                if ask_prompt:
                    parts.append(ask_prompt)
                parts.append(system_prompt_override)
                effective_prompt = "\n\n".join(parts)
            else:
                effective_prompt = ask_prompt

            with self._allow_concurrent(), self._llm_executor.permit():
                request_text = text
                if images:
                    # Clean prompt by removing image URLs
                    for img in images:
                        request_text = request_text.replace(img, "").strip()

                bridge_trace: list = []
                bridge_schemas, bridge_handlers = self._build_bridge_tool(
                    irc, msg, channel, trace=bridge_trace
                )
                # Combine bridge tools with any verse tools from the route.
                bridge_list = list(bridge_schemas) if bridge_schemas else []
                verse_list = list(extra_tools_override) if extra_tools_override else []
                extra_tools = (bridge_list + verse_list) or None
                bridge_debug = bool(
                    bridge_schemas and self.registryValue("bridgeDebugInChannel", channel)
                )

                # C7d: merge verse handlers into extra_handlers so the
                # assistant_request loop can dispatch verse tool calls
                # in-flight rather than passing them to the generic executor.
                #
                # When verse tool *schemas* are advertised on a verse-enabled
                # channel but the caller hasn't opted in (verse_route is None
                # while extra_tools_override carries verse specs), wire denial
                # handlers so the model gets a clean rejection. Advertising
                # the same tool surface to every speaker keeps the channel's
                # cacheable prefix byte-stable across opted-in/non-opted-in
                # users — the cohort split that was costing prompt-cache hits.
                if verse_route is not None:
                    verse_handlers = self._build_verse_handlers_for_route(channel, verse_route)
                    combined_handlers: dict | None = {
                        **(bridge_handlers or {}),
                        **verse_handlers,
                    }
                    # verse_edit (gated): overlay a live handler bound to the
                    # TRIGGERING user's prefix so the llm.verse.edit capability
                    # check reflects who drove this turn, not the channel's
                    # active avatar. The tool spec is always advertised on a
                    # verse route (make_verse_tool_specs), so the handler must
                    # always be present here — an unauthorized caller is refused
                    # inside dispatch_verse_edit, not by withholding the tool.
                    combined_handlers["verse_edit"] = self._verse_edit_handler(
                        msg=msg,
                        store=verse_route.store,
                        account=pf.account,
                    )
                    # Storybook tool (gated): overlay a live handler that can
                    # capture irc/msg/channel/account/nick/persona. Only wired
                    # when the per-channel flag is on; otherwise the spec isn't
                    # advertised and this key is never reached. The persona is
                    # the clean character voice (NOT the system_prompt overlay).
                    if self.registryValue("verseStorybookEnabled", channel):
                        persona = self.db.get_avatar_persona(nick) or ""
                        combined_handlers["verse_storybook"] = self._storybook_handler(
                            irc=irc,
                            msg=msg,
                            channel=channel,
                            account=pf.account,
                            nick=nick,
                            persona=persona,
                        )
                elif verse_list:
                    denial_handlers = make_verse_denial_handlers(verse_list)
                    combined_handlers = {
                        **(bridge_handlers or {}),
                        **denial_handlers,
                    }
                else:
                    combined_handlers = bridge_handlers

                # Typing keepalive is started in
                # ``_dispatch_with_verse_routing`` so it covers the verse
                # route lookup and history/memory fetches that run before
                # this point. ``manage_typing=False`` keeps the
                # service-level layer from clobbering it.
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
                    user_instruction=user_instruction,
                    system_prompt=effective_prompt,
                    model_override=model_override,
                    search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                    fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                    code_fn=lambda p: self._code_for_assistant(p, channel),
                    draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                    cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                    extra_tools=extra_tools,
                    extra_handlers=combined_handlers,
                    manage_typing=False,
                    **self._pending_task_fns(caller=caller, irc=irc, msg=msg, channel=channel),
                )

                response = result.content

                # Optional in-channel debug footer listing bridge tool calls.
                if bridge_debug and bridge_trace:
                    footer = self._format_bridge_debug_footer(bridge_trace)
                    if footer:
                        response = f"{response}\n{footer}" if response else footer

                response, should_log = self._dispatch_assistant_reply(
                    irc,
                    msg,
                    result,
                    nick=nick,
                    channel=channel,
                    response=response,
                    suppress_reminder_mutations=True,
                )

            if should_log:
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

        # Typing fires before any DB work (history, memory, executor
        # permit) so the user sees "is composing" within ~50ms instead
        # of after several seconds of synchronous setup.
        stop_typing = self.llm_service._begin_typing(irc, msg)
        try:
            request_context = self._build_request_context(
                irc,
                msg,
                pf,
                entry_route=PROFILE_CODE,
                profile=PROFILE_CODE,
            )

            caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

            with self._trace_request("code", nick, channel):
                history, channel_history = self._gather_history(nick, channel)

                memories = self._get_user_memories(nick)
                user_instruction = self.db.get_instruction(nick)
                # CODE_SYSTEM_PROMPT is the facade prompt that tells the planner
                # to call generate_code (not the registry codeSystemPrompt, the
                # inner-call prompt used by _code_for_assistant). The user's
                # @instruct now rides as user-role data (user_instruction below)
                # instead of being prepended here, so it cannot pose as system
                # authority. The system_prompt selection is unchanged: facade
                # when an instruction exists, else None (profile supplies it).
                from .prompts import CODE_SYSTEM_PROMPT

                effective_prompt = CODE_SYSTEM_PROMPT if user_instruction else None

                with self._allow_concurrent(), self._llm_executor.permit():
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
                        user_instruction=user_instruction,
                        system_prompt=effective_prompt,
                        search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                        fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                        code_fn=lambda p: self._code_for_assistant(p, channel),
                        draw_fn=lambda p: self._draw_for_assistant(irc, msg, p),
                        cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                        manage_typing=False,
                        **self._pending_task_fns(caller=caller, irc=irc, msg=msg, channel=channel),
                    )

                    response, should_log = self._dispatch_assistant_reply(
                        irc,
                        msg,
                        result,
                        nick=nick,
                        channel=channel,
                        response=result.content,
                    )

                if should_log:
                    self._store_context_and_log_usage(
                        nick, channel, "code", text, response, result, irc, msg
                    )
        finally:
            stop_typing()

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

        # Typing fires immediately after preflight so users see "is
        # composing" before history fetch / executor permit / image
        # generation latency stack up.
        stop_typing = self.llm_service._begin_typing(irc, msg)
        try:
            request_context = self._build_request_context(
                irc,
                msg,
                pf,
                entry_route=PROFILE_DRAW,
                profile=PROFILE_DRAW,
            )

            caller = Identity(raw_nick=request_context.raw_nick, account=pf.account)

            with self._trace_request("draw", nick, channel):
                history, channel_history = self._gather_history(
                    nick,
                    channel,
                    max_age_seconds=self.registryValue("drawContextMaxAgeSeconds", channel),
                )

                with self._allow_concurrent(), self._llm_executor.permit():
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
                        manage_typing=False,
                        **self._pending_task_fns(caller=caller, irc=irc, msg=msg, channel=channel),
                    )

                    response, should_log = self._dispatch_assistant_reply(
                        irc,
                        msg,
                        result,
                        nick=nick,
                        channel=channel,
                        response=result.content,
                    )

                if should_log:
                    self._store_context_and_log_usage(
                        nick, channel, "draw", text, response, result, irc, msg
                    )
        finally:
            stop_typing()

    draw = wrap(draw, [("checkCapability", "llm.draw"), "text"])

    def story(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str,
    ) -> None:
        """<brief>

        Generate an illustrated page from your description and post a link when
        it's ready. Either tells an illustrated tale OR explains a concept with
        diagrams as a learning aid — picks the mode from your brief. Draws a few
        AI illustrations, no verse mode required. Same image-spend gate as @draw
        (authenticated + llm.draw).

        Examples:
          @story an illustrated tale of stinky lads winning the pub quiz
          @story explain how photosynthesis works, with diagrams
        """
        # Skip ZNC playback messages
        if self._is_old_message(msg):
            return

        pf = self._run_preflight(irc, msg, text, "story", require_account=True)
        if pf.blocked:
            return
        nick, channel = pf.nick, pf.channel

        brief = (text or "").strip()
        if not brief:
            self._safe_error(irc, _("Tell me what the story should be about."))
            return

        # Per-account cooldown (shared with verse_storybook), reserve-at-start.
        cooldown = int(self.registryValue("verseStorybookCooldownSeconds", channel) or 0)
        if self._storybook_cooldown_active(pf.account, cooldown):
            self._safe_error(irc, _("Storybook cooldown is active — try again in a bit."))
            return

        # Fire-and-return: the page is rendered + posted asynchronously, so the
        # user gets the link when it's ready with no interim chatter (mirrors
        # the verse_storybook UX). Persona is the caller's avatar voice if set.
        persona = self.db.get_avatar_persona(nick) or ""
        self._submit_storybook_job(
            channel=channel, nick=nick, persona=persona, brief=brief, account=pf.account
        )

    story = wrap(story, [("checkCapability", "llm.draw"), "text"])

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
        self.context.clear_channel(channel)
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
                irc.error("Usage: memories delete <id> [<id> ...]")
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
                irc.error("Usage: memories edit <id> <new text>")
                return
            new_text = parts[2].strip()
            if not new_text:
                irc.error("Usage: memories edit <id> <new text>")
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
            irc.reply(summary.message, prefixNick=False)

        elif len(parts) == 1:
            # Owner viewing another user's memories
            if not ircdb.checkCapability(msg.prefix, "owner"):
                irc.error("Usage: memories [del <id> | edit <id> <text> | clear | cleanup]")
                return
            target = parts[0]
            self._memories_list(irc, target, target)

        else:
            irc.error("Usage: memories [del <id> | edit <id> <text> | clear | cleanup]")

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

        is_clear = text.strip().lower() == "clear"

        if is_clear:
            if self.db.delete_instruction(caller.key):
                irc.reply("Instruction cleared.", prefixNick=False)
            else:
                irc.reply("No instruction to clear.", prefixNick=False)
            return

        self.db.save_instruction(caller.key, text)
        irc.reply("Instruction set.", prefixNick=False)

    instruct = wrap(instruct, [optional("text")])

    def avatar(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None,
    ) -> None:
        """[<persona> | clear]

        Set the persona that shapes your avatar in verse-enabled channels.
        Independent of %instruct — this only affects the verse, not %ask.

        Examples:
          %avatar A moss-covered tree spirit who speaks in riddles.
          %avatar clear
          %avatar          (show current persona)
        """
        caller = self._resolve_identity(irc, msg)

        if not text:
            current = self.db.get_avatar_persona(caller.key)
            if current:
                irc.reply(f"Current persona: {current}", prefixNick=False)
            else:
                irc.reply("No persona set. Use %avatar <text> to set one.", prefixNick=False)
            return

        channel = msg.args[0] if msg.args else None
        is_clear = text.strip().lower() == "clear"
        new_summary = "" if is_clear else text

        # If in a verse-enabled channel with an active avatar, mirror to the
        # avatar's summary so /look and other display paths reflect the change
        # immediately. The persona registry remains the source of truth for
        # the verse system prompt.
        if channel and self.registryValue("verseEnabled", channel):
            store = self._get_or_create_verse_store(channel)
            avatar_id = (
                store.find_avatar_by_account(caller.account) if caller.account else None
            ) or store.find_avatar_by_nick(caller.raw_nick)
            if avatar_id is not None:
                entity = store.get_entity(avatar_id)
                if entity is not None and entity.status == "active":
                    with store.write_transaction() as conn:
                        conn.execute(
                            "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
                            (new_summary, time.time(), avatar_id),
                        )

        if is_clear:
            if self.db.delete_avatar_persona(caller.key):
                irc.reply("Persona cleared.", prefixNick=False)
            else:
                irc.reply("No persona to clear.", prefixNick=False)
            return

        self.db.save_avatar_persona(caller.key, text)
        irc.reply("Persona set.", prefixNick=False)

    avatar = wrap(avatar, [optional("text")])

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

        # LLM executor utilization (running/queued/max).
        ex = self._llm_executor
        parts.append(f"executor: {ex.running()}/{ex.queued()}/{ex.max_concurrency}")

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

    def _pending_task_fns(
        self,
        *,
        caller: Identity,
        irc: callbacks.Irc,
        msg: IrcMsg,
        channel: str,
        pass_irc_msg_to_callbacks: bool = True,
    ) -> dict[str, Callable[..., object]]:
        """Build the unified pending-task tool dict for assistant calls.

        Returns callables for the consolidated ``list_pending_tasks`` /
        ``cancel_pending_task`` / ``cancel_all_pending_tasks`` /
        ``set_reminder`` / ``schedule_llm_task`` tool surface. The list and
        cancel paths span both reminders (``set_reminder``) and scheduled
        LLM tasks (``schedule_llm_task``) so the model can never look at
        only one kind when the user asks "what do I have scheduled?" or
        "cancel my X".

        ``pass_irc_msg_to_callbacks`` is False on the action-fire path: its
        ``synthetic_msg`` has no msgid, so passing ``irc``/``msg`` through
        would just invoke ``_react`` only to have its msgid check fail.
        """
        react_irc = irc if pass_irc_msg_to_callbacks else None
        react_msg = msg if pass_irc_msg_to_callbacks else None

        def set_reminder_fn(text: str) -> ToolCallbackResult:
            return self._remind_set_for_assistant(irc, msg, caller, text)

        def schedule_fn(
            *,
            when_natural: str,
            prompt: str,
            reply_target: str | None = None,
        ) -> dict[str, object]:
            result = self.llm_service.schedule_llm_task(
                irc=irc,
                msg=msg,
                creator_nick=caller.raw_nick,
                account=caller.account,
                channel=channel,
                when_natural=when_natural,
                prompt=prompt,
                reply_target=reply_target,
            )
            return {
                "status": result.status,
                "event_name": result.event_name,
                "fire_at": result.fire_at,
                "message": result.message,
                "note": result.note,
            }

        def list_pending_tasks_fn() -> list[dict[str, object]]:
            tasks: list[dict[str, object]] = []
            for name, data in self._get_user_reminders(caller):
                # event-name format: "llm_reminder_<owner>_<id>"; the LLM
                # only needs the trailing id (matches existing UX).
                rid = name.split("_")[-1]
                tasks.append(
                    {
                        "kind": "reminder",
                        "id": rid,
                        "channel": data[1],
                        "description": data[2],
                    }
                )
            for row in self.llm_service.list_scheduled_llm_tasks(
                creator_nick=caller.raw_nick, account=caller.account
            ):
                tasks.append(
                    {
                        "kind": "scheduled_task",
                        "id": row.event_name,
                        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row.fire_at)),
                        "channel": row.channel,
                        "description": row.prompt[:80],
                        "recurrence": (
                            f"every {row.recurrence_seconds}s"
                            if row.recurrence_seconds is not None
                            else row.recurrence_rrule
                        ),
                    }
                )
            return tasks

        def cancel_pending_task_fn(task_id: str) -> dict[str, object]:
            if task_id.startswith("llm_task_"):
                result = self.llm_service.cancel_scheduled_llm_task(
                    event_name=task_id,
                    creator_nick=caller.raw_nick,
                    account=caller.account,
                )
                return {
                    "status": result.status,
                    "kind": "scheduled_task",
                    "id": result.event_name,
                    "message": result.message,
                }
            result = self._remind_delete_for_assistant(
                caller, task_id, irc=react_irc, msg=react_msg
            )
            return {
                "status": "ok" if result.ok else "error",
                "kind": "reminder",
                "id": task_id,
                "message": result.message,
            }

        def cancel_all_pending_tasks_fn() -> dict[str, object]:
            reminder_msg = self._remind_clear_for_assistant(caller, irc=react_irc, msg=react_msg)
            scheduled_rows = self.llm_service.list_scheduled_llm_tasks(
                creator_nick=caller.raw_nick, account=caller.account
            )
            scheduled_cancelled = 0
            for row in scheduled_rows:
                result = self.llm_service.cancel_scheduled_llm_task(
                    event_name=row.event_name,
                    creator_nick=caller.raw_nick,
                    account=caller.account,
                )
                if result.status == "ok":
                    scheduled_cancelled += 1
            return {
                "reminders_message": reminder_msg,
                "scheduled_tasks_cancelled": scheduled_cancelled,
            }

        return {
            "set_reminder_fn": set_reminder_fn,
            "schedule_llm_task_fn": schedule_fn,
            "list_pending_tasks_fn": list_pending_tasks_fn,
            "cancel_pending_task_fn": cancel_pending_task_fn,
            "cancel_all_pending_tasks_fn": cancel_all_pending_tasks_fn,
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

    def _get_reminders_for_target(self, target: str) -> list[tuple[str, ReminderRow]]:
        """Return reminders whose stored nick or account matches ``target``.

        Owner-only callers identify another user by either the nick that
        scheduled the reminder or the NickServ account it was stored
        under, case-insensitively.
        """
        target_lower = ircutils.toLower(target)
        with self._reminders_lock:
            return [
                (name, data)
                for name, data in self._reminders.items()
                if ircutils.toLower(data.nick) == target_lower
                or (data.account and ircutils.toLower(data.account) == target_lower)
            ]

    def _find_reminder_for_target(self, target: str, reminder_id: str) -> str | None:
        """Find a single reminder owned by ``target`` by reminder ID."""
        target_lower = ircutils.toLower(target)
        with self._reminders_lock:
            for name, data in self._reminders.items():
                if not name.endswith(f"_{reminder_id}"):
                    continue
                if ircutils.toLower(data.nick) == target_lower or (
                    data.account and ircutils.toLower(data.account) == target_lower
                ):
                    return name
            return None

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

        with (
            self._trace_request("remind", caller.key, channel),
            self._allow_concurrent(),
            self._llm_executor.permit(),
        ):
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
            return
        log.info(
            "remind_set blocked nick=%s reason=%s",
            caller.key,
            result.message,
        )
        self._react(irc, msg, "❌")
        self._safe_error(irc, _(result.message))

    def _remind_set_for_assistant(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        caller: Identity,
        text: str,
        *,
        parent_chain: int | None = None,
    ) -> ToolCallbackResult:
        """Parse and schedule a reminder, returning a ToolCallbackResult for meta.

        ``parent_chain`` is provided when the call originates from an
        action-fire LLM rescheduling a recurring chain — see
        :meth:`_schedule_reminder` for cap enforcement.

        Reacts ⏰ to ``msg`` on success and ❌ on cap/parse failure so the
        user gets a visual acknowledgment regardless of whether the model
        speaks. The chat reply path suppresses an empty post-tool reply
        via the structured ``last_successful_tool`` signal.
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        result = self._schedule_reminder(irc, msg, caller, text, parent_chain=parent_chain)
        self._react(irc, msg, "⏰" if result.ok else "❌")
        return _ToolCallbackResult(result.ok, result.message)

    def _remind_delete_for_assistant(
        self,
        caller: Identity,
        reminder_id: str,
        *,
        irc: callbacks.Irc | None = None,
        msg: IrcMsg | None = None,
    ) -> ToolCallbackResult:
        """Delete a reminder by ID, scoped to the caller's identity.

        When ``irc``/``msg`` are provided, reacts 👍 on success and ❌
        when no matching reminder exists (so the chat path can suppress
        the empty post-tool reply without leaving the user wondering).
        """
        from .assistant import ToolCallbackResult as _ToolCallbackResult

        target = self._find_user_reminder(caller, reminder_id)
        if target is None:
            if irc is not None and msg is not None:
                self._react(irc, msg, "❌")
            return _ToolCallbackResult(False, f"Reminder {reminder_id} not found.")

        self._cancel_reminder(target)
        if irc is not None and msg is not None:
            self._react(irc, msg, "👍")
        return _ToolCallbackResult(True, f"Deleted reminder {reminder_id}.")

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
          %remind admin list <nick>      (owner only)
          %remind admin del <nick> <id>  (owner only)
          %remind admin clear <nick>     (owner only)
        """
        caller = self._resolve_identity(irc, msg)

        if not text:
            self._remind_list(irc, caller)
            return

        parts = text.split(None, 1)
        subcommand = parts[0].lower()

        if subcommand == "admin":
            if not ircdb.checkCapability(msg.prefix, "owner"):
                irc.error(_("Only bot owners can manage other users' reminders."))
                return
            self._remind_admin(irc, msg, parts[1] if len(parts) >= 2 else "")
            return

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

    def _cancel_scheduled_llm_task_admin(self, event_name: str) -> None:
        """Owner-only cancel: skip ownership check, remove schedule + DB row."""
        with contextlib.suppress(KeyError):
            schedule.removeEvent(event_name)
        self.db.delete_scheduled_llm_task(event_name)

    def _remind_admin(self, irc: callbacks.Irc, msg: IrcMsg, rest: str) -> None:
        """Owner-only dispatcher for cross-user reminder management.

        Subcommands: ``list <target>``, ``del <target> <id> [<id>...]``,
        ``clear <target>``.  ``target`` matches stored nick or account
        case-insensitively.  ``clear`` cancels both ``reminders`` and
        ``scheduled_llm_tasks`` rows because users speak of both as
        "reminders" and the difference is invisible to them.
        """
        tokens = rest.split()
        if len(tokens) < 2:
            irc.error(_("Usage: remind admin <list|del|clear> <nick> [<id>...]"))
            return

        action = tokens[0].lower()
        target = tokens[1]

        if action == "list":
            reminder_rows = self._get_reminders_for_target(target)
            task_rows = self.db.load_scheduled_llm_tasks_for_target(target)
            if not reminder_rows and not task_rows:
                irc.reply(f"No pending reminders or scheduled tasks for {target}.")
                return
            parts = []
            if reminder_rows:
                parts.append(self._format_reminders(reminder_rows))
            for row in task_rows:
                preview = (row.prompt[:40] + "...") if len(row.prompt) > 40 else row.prompt
                parts.append(f"task:{row.event_name}: {preview}")
            irc.reply(f"{target}: " + " | ".join(parts))

        elif action in ("delete", "del"):
            if len(tokens) < 3:
                irc.error(_("Usage: remind admin del <nick> <id> [<id>...]"))
                return
            deleted = 0
            for rid in tokens[2:]:
                reminder_event = self._find_reminder_for_target(target, rid)
                if reminder_event:
                    self._cancel_reminder(reminder_event)
                    deleted += 1
                    continue
                task_row = self.db.get_scheduled_llm_task(rid)
                if task_row is not None and (
                    ircutils.toLower(task_row.creator_nick) == ircutils.toLower(target)
                    or (
                        task_row.account
                        and ircutils.toLower(task_row.account) == ircutils.toLower(target)
                    )
                ):
                    self._cancel_scheduled_llm_task_admin(rid)
                    deleted += 1
            if deleted == 0:
                self._react(irc, msg, "❌")
                irc.error(_("No matching reminders or tasks found."))
            else:
                label = "entry" if deleted == 1 else "entries"
                self._ack(irc, msg, "👍", f"Cancelled {deleted} {label} for {target}.")

        elif action == "clear":
            reminder_rows = self._get_reminders_for_target(target)
            task_rows = self.db.load_scheduled_llm_tasks_for_target(target)
            total = len(reminder_rows) + len(task_rows)
            if total == 0:
                self._ack(irc, msg, "👌", f"Nothing to clear for {target}.")
                return
            for name, _data in reminder_rows:
                self._cancel_reminder(name)
            for row in task_rows:
                self._cancel_scheduled_llm_task_admin(row.event_name)
            label = "entry" if total == 1 else "entries"
            self._ack(irc, msg, "👍", f"Cleared {total} {label} for {target}.")

        else:
            irc.error(_("Usage: remind admin <list|del|clear> <nick> [<id>...]"))

    remind = wrap(remind, [optional("text")])

    # =========================================================================
    # Verse subsystem
    # =========================================================================

    # -------------------------------------------------------------------------
    # Verse subsystem channel helpers
    # -------------------------------------------------------------------------

    def _all_known_channels(self) -> set[str]:
        """Channels seen on any connected Irc. Used by ``_verse_enabled_channels``."""
        seen: set[str] = set()
        for irc_conn in world.ircs:
            seen.update(irc_conn.state.channels.keys())
        return seen

    def _verse_enabled_channels(self) -> list[str]:
        """All channels with verseEnabled=True. Read from registry every call
        (callers are not in hot paths)."""
        out: list[str] = []
        for ch in self._all_known_channels():
            if self.registryValue("verseEnabled", ch):
                out.append(ch)
        return out

    # -------------------------------------------------------------------------
    # Daily verse compaction timer (PR 3 / E3)
    # -------------------------------------------------------------------------

    def _register_compaction_timer(self) -> None:
        """Arm the next single-shot compaction firing.

        Idempotent: ``register_daily_timer`` cancels any existing event
        with the same name first, so callers can re-arm without bothering
        to cancel.
        """
        from llm.verse.compaction import register_daily_timer

        try:
            fire_at = self.registryValue("verseCompactionDailyAt") or "03:00"
        except Exception:
            # Registry key not yet defined (F1 adds it). Fall back so
            # the timer still arms on un-configured installs.
            fire_at = "03:00"
        try:
            register_daily_timer(
                schedule_module=schedule,
                fire_at_local=fire_at,
                callback=self._compaction_tick,
                name=self._compaction_timer_name,
            )
        except Exception:
            # On addEvent failure the timer is gone until next plugin
            # reload — daily cadence makes this acceptable.
            self.log.exception("verse: failed to register compaction timer")

    def _cancel_compaction_timer(self) -> None:
        from llm.verse.compaction import cancel_daily_timer

        cancel_daily_timer(schedule_module=schedule, name=self._compaction_timer_name)

    def _compaction_tick(self) -> None:
        """One firing of the daily timer: offload the pass, then re-arm.

        ``_run_compaction_pass`` makes one blocking LLM call per
        verse-enabled channel. This callback fires on Limnoria's scheduler
        thread — the IRC driver's main loop — which cannot flush the
        outbound queue or answer PINGs while a callback runs. So the work is
        handed to ``_llm_executor`` instead of running inline, and
        ``_dispatch_addressed_async``
        carries the full rationale for why blocking the driver thread is the
        bug to avoid. The re-arm runs in ``finally`` so a failed submit never
        kills the timer — the next day still gets a shot.
        """
        try:
            if not self._llm_executor.closing:
                self._llm_executor.submit("verse_compaction", self._run_compaction_pass)
        except Exception:
            self.log.exception("verse compaction submit failed")
        finally:
            self._register_compaction_timer()

    def _run_compaction_pass(self) -> None:
        """Walk every verse-enabled channel and compact it.

        Per-channel failures are logged and swallowed so one bad verse
        doesn't abort the rest of the pass.
        """
        from llm.verse import compaction as _compaction
        from llm.verse.aging import age_auto_created_entities

        retention_days_default = 30
        # NB: explicit ``int(value)`` rather than ``int(value or 20)``
        # — 0 is a legitimate registry value (NonNegativeInteger
        # accepts it) and the ``or`` form would coerce it to 20.
        try:
            _raw_min_keep = self.registryValue("verseCompactionMinKeepEvents")
        except Exception:
            # Registry key not yet defined (F1 adds it) or load error.
            _raw_min_keep = 20
        try:
            min_keep = int(_raw_min_keep) if _raw_min_keep is not None else 20
        except (TypeError, ValueError):
            min_keep = 20
        model = self.registryValue("verseCompactionModel") or "gemini/gemini-flash-lite-latest"
        api_key = self.registryValue("assistantApiKey") or None
        client = _compaction.LiteLLMVerseClient(api_key=api_key)

        def _log_usage(*, op: str, model: str, usage, channel: str) -> None:
            self.db.log_usage(
                nick="loom",
                channel=channel,
                command=f"loom:{op}",
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=usage.cost,
            )

        for channel in self._verse_enabled_channels():
            # Defensive re-check — _verse_enabled_channels already filters,
            # but the registry could flip mid-pass.
            try:
                if not self.registryValue("verseEnabled", channel):
                    continue
            except Exception:
                continue
            store = self._get_or_create_verse_store(channel)
            # Honour zero — operators set retention=0 to disable.
            try:
                _raw_retention = self.registryValue("verseEventRetentionDays", channel)
            except Exception:
                _raw_retention = retention_days_default
            try:
                retention_days = (
                    int(_raw_retention) if _raw_retention is not None else retention_days_default
                )
            except (TypeError, ValueError):
                retention_days = retention_days_default
            # Aging runs BEFORE compaction. compact_verse's digest heartbeat
            # bumps last_seen_ts=now() on every entity in the new digest, so
            # running it first would refresh long-silent NPCs and defeat
            # verseAutoEntityRetireDays. Aging-first reads the true last_seen_ts
            # before the heartbeat can resurrect a stale NPC. Independent of the
            # compaction outcome and wrapped in its own try/except so one
            # channel's failure doesn't abort the rest of the pass.
            aging_outcome: AgingOutcome | None = None
            try:
                retire_days = self.registryValue("verseAutoEntityRetireDays", channel)
                aging_outcome = age_auto_created_entities(
                    store,
                    retire_after_days=retire_days,
                    now=time.time,
                )
            except Exception:
                self.log.exception(
                    "verse aging failed for %s; continuing with next channel",
                    channel,
                )

            outcome: CompactionOutcome | None = None
            try:
                outcome = _compaction.compact_verse(
                    store,
                    retention_days=retention_days,
                    min_keep_events=min_keep,
                    model=model,
                    client=client,
                    # Default-arg captures the loop variable.
                    log_usage=lambda *, op, model, usage, channel=channel: _log_usage(
                        op=op, model=model, usage=usage, channel=channel
                    ),
                    now=time.time,
                )
            except Exception:
                self.log.exception("verse compaction failed for %s; continuing", channel)

            # Render the per-channel summary as one line, but only if
            # compaction itself succeeded — a raised compact_verse already
            # got its own .exception log above.
            if outcome is not None:
                msg = _format_compaction_outcome(outcome, aging_outcome, min_keep_events=min_keep)
                self.log.info("compaction outcome for %s: %s", channel, msg)

    def _get_or_create_verse_store(self, channel: str) -> VerseStore:
        """Return the VerseStore for *channel*, creating it lazily on first access.

        Thread-safe: guarded by ``_verse_stores_lock``.  The store itself is
        safe to use outside the lock — it carries its own write lock.
        """
        with self._verse_stores_lock:
            store = self._verse_stores.get(channel)
            if store is None:
                base = Path(conf.supybot.directories.data()) / "verse"
                store = VerseStore(base, channel)
                self._verse_stores[channel] = store
            return store

    def verseopt(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        mode: str,
    ) -> None:
        """<in|out>

        Opt your avatar in or out of the verse for this channel.

          @verseopt in   — create or reactivate your avatar; shows the opening scene.
          @verseopt out  — retire your avatar until you opt back in.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = msg.args[0] if msg.args else None
        if not channel or not ircutils.isChannel(channel):
            irc.error(_("This command must be used in a channel."), prefixNick=False)
            return

        if not self.registryValue("verseEnabled", channel):
            irc.reply(
                "This channel doesn't have a verse. Ask the operator to set verseEnabled.",
                prefixNick=False,
            )
            return

        caller = self._resolve_identity(irc, msg)
        nick = caller.raw_nick
        account = caller.account

        if mode == "in":
            persona_text = self.db.get_avatar_persona(caller.key) or ""
            store = self._get_or_create_verse_store(channel)
            result = store.opt_in_avatar(nick, account, persona_text)
            reply = result.scene_text
            if result.was_already_opted_in:
                reply = "You are already opted in. " + reply
            irc.reply(reply, prefixNick=False)

        else:  # mode == "out"
            store = self._get_or_create_verse_store(channel)
            entity_id = store.find_avatar_by_account(account) if account else None
            if entity_id is None:
                entity_id = store.find_avatar_by_nick(nick)
            if entity_id is None:
                irc.reply("You don't have an avatar in this channel.", prefixNick=False)
                return
            store.unlink_avatar(entity_id)
            irc.reply(
                "Avatar retired. Use @verseopt in to rejoin.",
                prefixNick=False,
            )

    verseopt = wrap(verseopt, [("checkCapability", "llm.verse"), ("literal", ("in", "out"))])

    # ------------------------------------------------------------------
    # Helpers shared by @verse / @look / @who
    # ------------------------------------------------------------------

    _NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    _NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    def _check_verse_channel(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
        """Return the channel name if verse is enabled, else reply and return None."""
        channel = msg.args[0] if msg.args else None
        if not channel or not ircutils.isChannel(channel):
            irc.error(_("This command must be used in a channel."), prefixNick=False)
            return None
        if not self.registryValue("verseEnabled", channel):
            irc.reply(self._NO_VERSE_REPLY, prefixNick=False)
            return None
        return channel

    def _avatar_scene_oneliner(self, store: VerseStore, entity_id: int) -> str:
        """Build an IRC-friendly one-liner for the avatar's current location."""
        location_id_str = store.get_attribute(entity_id, "location")
        if location_id_str is None:
            return "You are nowhere in particular."
        try:
            place = store.get_entity(int(location_id_str))
        except (ValueError, TypeError):
            place = None
        if place is None:
            return "You are nowhere in particular."
        return f"You are at {place.name}. {place.summary}".rstrip()

    # ------------------------------------------------------------------
    # @verse — show caller's current scene one-liner
    # ------------------------------------------------------------------

    def verse(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        Show your current scene one-liner in the verse.

          @verse — display where your avatar currently is.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        caller = self._resolve_identity(irc, msg)
        store = self._get_or_create_verse_store(channel)
        account = caller.account
        entity_id = store.find_avatar_by_account(account) if account else None
        if entity_id is None:
            entity_id = store.find_avatar_by_nick(caller.raw_nick)
        if entity_id is None:
            irc.reply(self._NO_AVATAR_REPLY, prefixNick=False)
            return

        irc.reply(self._avatar_scene_oneliner(store, entity_id), prefixNick=False)

    verse = wrap(verse, [("checkCapability", "llm.verse")])

    # ------------------------------------------------------------------
    # @look [target] — scene or entity description
    # ------------------------------------------------------------------

    def look(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        target: str | None = None,
    ) -> None:
        """[<target>]

        Show your current scene, or describe a named entity in the verse.

          @look          — show where you are (same as @verse).
          @look <name>   — describe an entity by name.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        store = self._get_or_create_verse_store(channel)

        if target is None:
            # No target: show caller's scene (avatar required).
            caller = self._resolve_identity(irc, msg)
            account = caller.account
            entity_id = store.find_avatar_by_account(account) if account else None
            if entity_id is None:
                entity_id = store.find_avatar_by_nick(caller.raw_nick)
            if entity_id is None:
                irc.reply(self._NO_AVATAR_REPLY, prefixNick=False)
                return
            irc.reply(self._avatar_scene_oneliner(store, entity_id), prefixNick=False)
        else:
            # Target given: look up entity by name.
            entity = store.find_entity_by_name(target)
            if entity is None:
                irc.reply("Nothing matches.", prefixNick=False)
                return
            irc.reply(f"{entity.name}: {entity.summary}", prefixNick=False)

    look = wrap(look, [("checkCapability", "llm.verse"), optional("text")])

    # ------------------------------------------------------------------
    # @who — list active avatars and their locations
    # ------------------------------------------------------------------

    def who(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        List active avatars and their current locations in the verse.

          @who — roster of opted-in avatars.

        Requires the llm.verse capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return

        store = self._get_or_create_verse_store(channel)
        avatars = store.list_entities_by_kind("avatar", status="active")
        if not avatars:
            irc.reply("Nobody is opted in here yet.", prefixNick=False)
            return

        parts: list[str] = []
        for avatar in avatars:
            location_id_str = store.get_attribute(avatar.id, "location")
            if location_id_str is not None:
                try:
                    place = store.get_entity(int(location_id_str))
                except (ValueError, TypeError):
                    place = None
                if place is not None:
                    parts.append(f"{avatar.name} (at {place.name})")
                    continue
            parts.append(avatar.name)

        irc.reply(", ".join(parts), prefixNick=False)

    who = wrap(who, [("checkCapability", "llm.verse")])

    # ------------------------------------------------------------------
    # @canon lock|unlock|forget <name> — author-gated durable canon
    # ------------------------------------------------------------------

    def canon(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        action: str,
        name: str,
    ) -> None:
        """<lock|unlock|forget> <name>

        Lock or release a character as durable canon (always remembered,
        aging-exempt). 'forget' is an alias for 'unlock'.

          @canon lock <name>   — mark a character as durable canon.
          @canon unlock <name> — release it.
          @canon forget <name> — same as unlock.

        Requires the llm.verse.edit capability and a verse-enabled channel.
        """
        channel = self._check_verse_channel(irc, msg)
        if channel is None:
            return
        store = self._get_or_create_verse_store(channel)
        ent = store.find_entity_by_name_or_alias(name)
        if ent is None:
            irc.error(f"No such character: {name}", prefixNick=False)
            return
        store.set_author_locked(ent.id, action == "lock")
        irc.replySuccess()

    canon = wrap(
        canon,
        [
            ("checkCapability", "llm.verse.edit"),
            ("literal", ("lock", "unlock", "forget")),
            "text",
        ],
    )

    # ------------------------------------------------------------------
    # @versedump #chan [--format=json]  — owner dump of full verse state
    # ------------------------------------------------------------------

    def versedump(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None = None,
    ) -> None:
        """[#channel] [--format=json]

        Dump the full verse state for the given channel as JSON.

          @versedump #chan           — dump in JSON (default).
          @versedump #chan --format=json  — explicit JSON.

        YAML is not supported (pyyaml is not a project dependency).
        Requires the llm.verse.gm capability.
        """
        if not ircdb.checkCapability(msg.prefix, "llm.verse.gm"):
            irc.error("You don't have the llm.verse.gm capability.", prefixNick=False)
            return

        # Parse channel and optional --format from the free-text args.
        raw = (text or "").split()
        channel: str | None = None
        fmt = "json"
        for token in raw:
            if token.startswith("--format="):
                fmt = token[len("--format=") :]
            elif ircutils.isChannel(token):
                channel = token

        if channel is None:
            # Fall back to the channel the command arrived in.
            ch = msg.args[0] if msg.args else None
            if ch and ircutils.isChannel(ch):
                channel = ch
            else:
                irc.error("Specify a channel: @versedump #chan", prefixNick=False)
                return

        if fmt not in ("json",):
            irc.reply(
                "Unsupported format. Only --format=json is supported.",
                prefixNick=False,
            )
            return

        store = self._get_or_create_verse_store(channel)

        # Collect all entities (all kinds, all statuses) with their attributes.
        with store.read_connection() as conn:
            entity_rows = conn.execute(
                "SELECT id, kind, name, summary, status, created_at, updated_at"
                " FROM entities ORDER BY id ASC"
            ).fetchall()
            avatar_link_rows = conn.execute(
                "SELECT entity_id, nick, account FROM avatar_link ORDER BY entity_id ASC"
            ).fetchall()

        entities_out = []
        for row in entity_rows:
            eid = row[0]
            entities_out.append(
                {
                    "id": eid,
                    "kind": row[1],
                    "name": row[2],
                    "summary": row[3],
                    "status": row[4],
                    "created_at": row[5],
                    "updated_at": row[6],
                    "attributes": store.list_attributes(eid),
                }
            )

        # Relations (all, no filter).
        relations_raw = store.list_relations()
        relations_out = [
            {"id": r.id, "from_id": r.from_id, "to_id": r.to_id, "kind": r.kind, "note": r.note}
            for r in relations_raw
        ]

        # Recent events (capped at 200 to keep IRC PMs manageable).
        events_raw = store.recent_events(limit=200)
        events_out = [
            {
                "id": ev.id,
                "ts": ev.ts,
                "summary": ev.summary,
                "entity_ids": list(ev.entity_ids),
                "source": ev.source,
            }
            for ev in events_raw
        ]

        avatar_links_out = [
            {"entity_id": row[0], "nick": row[1], "account": row[2]} for row in avatar_link_rows
        ]

        # Proposals (the @versedit audit trail) are not included in the dump;
        # versedump focuses on entities, relations, events, and avatar links.
        dump = {
            "schema_version": 1,
            "channel": channel,
            "entities": entities_out,
            "relations": relations_out,
            "events": events_out,
            "avatar_links": avatar_links_out,
            "proposals": [],
        }

        # The dump is a single fat JSON line — for an active verse it can
        # easily exceed Limnoria's pagination. Publish to the bot's HTTP
        # pastebin and reply with just the URL; fall back to inline only
        # if the pastebin write fails (e.g. http server not configured).
        body = json.dumps(dump, indent=2)
        markdown_body = f"# versedump {channel}\n\n```json\n{body}\n```\n"
        url = self.llm_service.save_markdown_to_http(markdown_body)
        if url:
            irc.reply(
                f"versedump {channel}: {url}",
                prefixNick=False,
            )
        else:
            irc.reply(json.dumps(dump, separators=(",", ":")), prefixNick=False)

    versedump = wrap(versedump, [optional("text")])

    # ------------------------------------------------------------------
    # @versedit <verb> <args...> [#channel]  — operator universe editing
    # ------------------------------------------------------------------

    def versedit(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        rest: str,
        channel_arg: str | None = None,
    ) -> None:
        """[#channel] <verb> <args...>

        Edit the verse universe: add, pin, unpin, set, name, desc, retire,
        restore, relate, unrelate, event, editevent, delevent, show.
        Requires the llm.verse.edit capability. A leading #channel targets
        that channel (useful in a PM, to avoid flooding the channel itself);
        otherwise the channel the command was run in is used.
        """
        # A leading "#channel" token lets an operator target a channel from a
        # private message — where msg carries no channel of its own (msg.channel
        # is None and msg.args[0] is the bot's nick) — so a batch of edits can be
        # pasted into a DM without flooding the channel. Explicit beats implicit.
        parts = rest.split(None, 1)
        if parts and ircutils.isChannel(parts[0]):
            channel_arg = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
        channel = channel_arg or (msg.args[0] if msg.args else None)
        if not channel or not ircutils.isChannel(channel):
            irc.error("Specify a channel.", prefixNick=False)
            return
        store = self._get_or_create_verse_store(channel)
        tokens = rest.split(None, 1)
        verb = tokens[0].lower() if tokens else ""
        body = tokens[1] if len(tokens) > 1 else ""
        try:
            reply = self._versedit_dispatch(store, verb, body)
        except (LookupError, ValueError, PermissionError) as exc:
            irc.error(str(exc), prefixNick=False)
            return
        irc.reply(reply, prefixNick=False)

    versedit = wrap(
        versedit,
        [
            # NB: ("checkCapability", ...) checks the cap GLOBALLY against
            # msg.prefix — it does NOT channel-scope to the optional("channel")
            # arg. llm.verse.edit is
            # a global cap everywhere in this plugin; a per-channel grant would
            # fail closed (deny), never escalate. Do not assume per-channel
            # enforcement exists here.
            ("checkCapability", "llm.verse.edit"),
            "text",
            optional("channel"),
        ],
    )

    def _versedit_dispatch(self, store: VerseStore, verb: str, body: str) -> str:
        if verb == "add":
            kind_rest = body.split(None, 1)
            if len(kind_rest) < 2:
                raise ValueError("usage: versedit add <kind> <name> [:: summary]")
            kind, name_part = kind_rest[0], kind_rest[1]
            name, summary = (name_part.split("::", 1) + [""])[:2]
            name, summary = name.strip(), summary.strip()
            if kind not in ("avatar", "npc", "place", "faction", "item"):
                raise ValueError("kind must be avatar|npc|place|faction|item")
            if store.active_name_exists(name):
                raise ValueError(f"an active entity named {name!r} already exists")
            new_id = store.apply_direct(
                op="add_entity",
                payload={"kind": kind, "name": name, "summary": summary},
                source="operator",
                provenance="@versedit add",
            )
            return f"added {kind} #{new_id}: {name}"
        if verb in ("pin", "unpin"):
            eid = store.resolve_ref(body.strip())
            store.apply_direct(
                op="set_pinned",
                payload={"entity_id": eid, "pinned": verb == "pin"},
                source="operator",
                provenance=f"@versedit {verb}",
            )
            return f"{verb}ned #{eid}"
        if verb == "set":
            parts = body.split(None, 2)
            if len(parts) < 3:
                raise ValueError("usage: versedit set <ref> <key> <value>")
            eid = store.resolve_ref(parts[0])
            store.apply_direct(
                op="set_attribute",
                payload={"entity_id": eid, "key": parts[1], "value": parts[2]},
                source="operator",
                provenance="@versedit set",
            )
            return f"set {parts[1]} on #{eid}"
        if verb == "name":
            ref, _, newname = body.partition(" ")
            newname = newname.strip()
            if not newname:
                raise ValueError("usage: versedit name <ref> <new-name>")
            eid = store.resolve_ref(ref)
            if store.active_name_exists(newname):
                raise ValueError(f"an active entity named {newname!r} already exists")
            store.apply_direct(
                op="update_entity",
                payload={"entity_id": eid, "name": newname},
                source="operator",
                provenance="@versedit name",
            )
            return f"renamed #{eid} -> {newname}"
        if verb == "desc":
            ref, _, summary = body.partition("::")
            eid = store.resolve_ref(ref.strip())
            store.apply_direct(
                op="update_entity",
                payload={"entity_id": eid, "summary": summary.strip()},
                source="operator",
                provenance="@versedit desc",
            )
            return f"updated summary of #{eid}"
        if verb in ("retire", "restore"):
            eid = store.resolve_ref(body.strip())
            status = "retired" if verb == "retire" else "active"
            store.apply_direct(
                op="set_status",
                payload={"entity_id": eid, "status": status},
                source="operator",
                provenance=f"@versedit {verb}",
            )
            return f"{verb}d #{eid}"
        if verb == "relate":
            head, _, note = body.partition("::")
            parts = head.split()
            if len(parts) != 3:
                raise ValueError("usage: versedit relate <ref> <kind> <ref> [:: note]")
            from_id = store.resolve_ref(parts[0])
            to_id = store.resolve_ref(parts[2])
            rid = store.apply_direct(
                op="add_relation",
                payload={
                    "from_id": from_id,
                    "to_id": to_id,
                    "kind": parts[1],
                    "note": note.strip(),
                },
                source="operator",
                provenance="@versedit relate",
            )
            return f"related #{from_id} -{parts[1]}-> #{to_id} (relation #{rid})"
        if verb == "unrelate":
            rid = int(body.strip())
            store.apply_direct(
                op="delete_relation",
                payload={"relation_id": rid},
                source="operator",
                provenance="@versedit unrelate",
            )
            return f"deleted relation #{rid}"
        if verb == "event":
            summary, _, ids_part = body.partition("@")
            entity_ids = (
                [int(x) for x in ids_part.split(",") if x.strip().isdigit()] if ids_part else []
            )
            new_id = store.apply_direct(
                op="add_event",
                payload={"summary": summary.strip(), "entity_ids": entity_ids},
                source="operator",
                provenance="@versedit event",
            )
            return f"added event #{new_id}"
        if verb == "editevent":
            id_part, _, summary = body.partition("::")
            ev_id = int(id_part.strip())
            store.apply_direct(
                op="edit_event",
                payload={"event_id": ev_id, "summary": summary.strip()},
                source="operator",
                provenance="@versedit editevent",
            )
            return f"edited event #{ev_id}"
        if verb == "delevent":
            ev_id = int(body.strip())
            store.apply_direct(
                op="delete_event",
                payload={"event_id": ev_id},
                source="operator",
                provenance="@versedit delevent",
            )
            return f"deleted event #{ev_id}"
        if verb == "show":
            eid = store.resolve_ref(body.strip())
            ent = store.get_entity(eid)
            if ent is None:
                raise LookupError(f"entity #{eid} does not exist")
            attrs = store.list_attributes(eid)
            return f"#{eid} {ent.kind} {ent.name} [{ent.status}] — {ent.summary} | attrs={attrs}"
        raise ValueError(f"unknown verb: {verb!r}")

    # ------------------------------------------------------------------
    # @versepurge #chan [token]  — wipe a channel's verse with 2-step confirm
    # ------------------------------------------------------------------

    def _versepurge_check_token(
        self,
        channel: str,
        presented: str,
        now_func: Callable[[], float] = time.time,
    ) -> bool:
        """Return True and clear token if presented token is valid and unexpired.

        Clears expired tokens on False return.  Uses secrets.compare_digest
        to avoid timing side-channels.
        """
        # Lock the check-and-clear so two confirmations can't both consume
        # the same token (the read, compare, and delete must be atomic).
        with self._versepurge_tokens_lock:
            entry = self._versepurge_tokens.get(channel)
            if entry is None:
                return False
            stored_token, expires_at = entry
            if now_func() >= expires_at:
                # Expired — clear stale entry.
                del self._versepurge_tokens[channel]
                return False
            if secrets.compare_digest(stored_token, presented):
                del self._versepurge_tokens[channel]
                return True
            return False

    def versepurge(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        text: str | None = None,
    ) -> None:
        """[#channel] [<token>]

        Wipe the verse store for a channel. Two-step confirmation required.

          Step 1: @versepurge #chan
                  — issues a one-time 6-character token valid for 60 seconds.

          Step 2: @versepurge #chan <token>
                  — confirms; purges all verse data for the channel.

        Tokens are per-channel and in-memory only; they reset on plugin
        reload or bot restart. All times are approximate (IRC scheduler thread).
        Requires the llm.verse.gm capability.
        """
        if not ircdb.checkCapability(msg.prefix, "llm.verse.gm"):
            irc.error("You don't have the llm.verse.gm capability.", prefixNick=False)
            return

        # Parse channel and optional token from free-text args.
        raw = (text or "").split()
        channel: str | None = None
        token_presented: str | None = None

        for token in raw:
            if ircutils.isChannel(token):
                channel = token
            elif channel is not None and token_presented is None:
                token_presented = token

        if channel is None:
            ch = msg.args[0] if msg.args else None
            if ch and ircutils.isChannel(ch):
                channel = ch
            else:
                irc.error("Specify a channel: @versepurge #chan", prefixNick=False)
                return

        if token_presented is not None:
            # Step 2: confirm purge.
            if self._versepurge_check_token(channel, token_presented):
                # Pop the store AND unlink its files under the SAME lock. The
                # plugin is threaded=True, so verse @ask (a SupyThread) and
                # executor workers hold their own thread-local conns — the
                # scheduler thread is NOT the sole holder. Both this block and
                # _get_or_create_verse_store take _verse_stores_lock, so holding
                # it across the unlink prevents a concurrent cache-miss from
                # reconstructing a store on the files mid-purge and leaving a
                # half-written DB behind. (A worker that captured the store
                # before the purge still holds its own conn; its in-flight write
                # is best-effort and may be lost — acceptable for this admin-only,
                # token-gated destructive command.)
                with self._verse_stores_lock:
                    store = self._verse_stores.pop(channel, None)
                    if store is not None:
                        db_path = Path(store.path)
                        db_path.unlink(missing_ok=True)
                        db_path.with_suffix(".db-wal").unlink(missing_ok=True)
                        db_path.with_suffix(".db-shm").unlink(missing_ok=True)
                irc.reply(f"Verse for {channel} purged.", prefixNick=False)
            else:
                irc.reply(
                    f"Token expired or invalid. Run @versepurge {channel} again to start over.",
                    prefixNick=False,
                )
        else:
            # Step 1: issue (or reissue) token. Lock the read-modify-write so
            # two concurrent issuances can't clobber each other's token.
            new_token = secrets.token_hex(3)
            with self._versepurge_tokens_lock:
                existing = self._versepurge_tokens.get(channel)
                reissued = False
                if existing is not None:
                    _, exp = existing
                    if time.time() < exp:
                        # Reissue while old token still valid — invalidate old one.
                        reissued = True
                    # Old token expired — fall through to fresh issue.
                self._versepurge_tokens[channel] = (new_token, time.time() + 60.0)
            if reissued:
                irc.reply(
                    f"Previous token invalidated. Confirm with @versepurge {channel}"
                    f" {new_token} within 60s.",
                    prefixNick=False,
                )
            else:
                irc.reply(
                    f"Confirm with @versepurge {channel} {new_token} within 60s.",
                    prefixNick=False,
                )

    versepurge = wrap(versepurge, [optional("text")])

    # ------------------------------------------------------------------
    # @versecompact — manual retention compaction (E4)
    # ------------------------------------------------------------------

    def versecompact(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
        channel: str,
    ) -> None:
        """<channel>

        Manually run retention compaction for <channel>. Mirrors what the
        daily timer does (``_run_compaction_pass``) but for one channel only.
        Requires capability llm.verse.gm.
        """
        if not self.registryValue("verseEnabled", channel):
            irc.reply(
                f"verseEnabled is False for {channel}; nothing to compact.",
                prefixNick=False,
            )
            return

        from llm.verse import compaction as _compaction

        store = self._get_or_create_verse_store(channel)
        # Honour zero — see ``_run_compaction_pass`` note.
        try:
            _raw_retention = self.registryValue("verseEventRetentionDays", channel)
        except Exception:
            _raw_retention = 30
        try:
            retention_days = int(_raw_retention) if _raw_retention is not None else 30
        except (TypeError, ValueError):
            retention_days = 30
        try:
            _raw_min_keep = self.registryValue("verseCompactionMinKeepEvents")
        except Exception:
            # Registry key not yet defined (F1 adds it).
            _raw_min_keep = 20
        try:
            min_keep = int(_raw_min_keep) if _raw_min_keep is not None else 20
        except (TypeError, ValueError):
            min_keep = 20
        model = self.registryValue("verseCompactionModel") or "gemini/gemini-flash-lite-latest"
        api_key = self.registryValue("assistantApiKey") or None
        client = _compaction.LiteLLMVerseClient(api_key=api_key)

        def _log_usage(*, op: str, model: str, usage, channel: str = channel) -> None:
            self.db.log_usage(
                nick="loom",
                channel=channel,
                command=f"loom:{op}",
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=usage.cost,
            )

        try:
            outcome = _compaction.compact_verse(
                store,
                retention_days=retention_days,
                min_keep_events=min_keep,
                model=model,
                client=client,
                log_usage=_log_usage,
                now=time.time,
            )
        except Exception as exc:
            self.log.exception("@versecompact failed for %s", channel)
            irc.error(
                f"compaction failed for {channel}: {type(exc).__name__}",
                prefixNick=False,
            )
            return

        outcome_msg = _format_compaction_outcome(outcome, None, min_keep_events=min_keep)
        irc.reply(
            f"compaction outcome for {channel}: {outcome_msg}",
            prefixNick=False,
        )

    versecompact = wrap(
        versecompact,
        [
            ("checkCapability", "llm.verse.gm"),
            "channel",
        ],
    )


Class = LLM
