"""Pytest configuration and shared fixtures for LLM plugin tests."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from llm.persistence import LLMDatabase, ReminderRow
from llm.plugin import LLM
from llm.service import LLMService

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture

# =============================================================================
# Test constants
# =============================================================================

TEST_MODEL = "gpt-4"
TEST_URL_BASE = "https://example.com/llm"

# Fake values, each comfortably over apikeys.MIN_REDACTABLE_LEN.
FAKE_PROVIDER_KEYS = {
    "XAI_API_KEY": "xai-fake-key-for-tests-0000",
    "GEMINI_API_KEY": "AIza-fake-key-for-tests-0000",
    "OPENAI_API_KEY": "sk-fake-key-for-tests-0000",
    "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-tests-0000",
}

# Duplicated from llm.apikeys (Task 2) rather than imported: that module
# does not exist yet and this fixture must not depend on it. Task 3 adds a
# test asserting the two copies stay in sync.
_SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# =============================================================================
# Autouse fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known, fake set of provider credentials.

    LiteLLM calls load_dotenv() at import, so a developer's real .env is in
    os.environ before collection. Without this, tests can pass locally against
    real keys, fail in CI, print real keys into a failure diff, or reach the
    network.
    """
    for name in list(os.environ):
        if name.upper().endswith(_SECRET_SUFFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture(autouse=True)
def _restore_global_logging_filters() -> Generator[None]:
    """Undo any ``SecretFilter`` installation left on process-global logging state.

    ``apikeys.install_secret_filter()`` (installed from ``LLM.__init__`` as of
    Task 4) attaches to handlers already sitting on root/``supybot``/``llm``/
    ``LiteLLM``/``LiteLLM Proxy``/``LiteLLM Router``, plus
    ``logging.lastResort`` — real, process-global state, not anything scoped
    to a single test. Any test that constructs a real plugin (``LLM(mock_irc)``,
    not just ``make_service()``) triggers this via ``__init__``. pytest reuses
    one session-scoped ``LogCaptureHandler`` for the whole run, so a filter
    installed by one test survives into every later test's ``caplog`` —
    silently turning any of ``FAKE_PROVIDER_KEYS``'s values into
    ``[REDACTED]`` in unrelated assertions, well after the test that
    installed the filter has finished. Snapshotting and restoring
    ``handler.filters`` here, for every test in the suite, confines
    installation to the test that triggered it. The three ``LiteLLM*`` loggers
    are covered too — ``import litellm`` gives each its own stderr handler
    that install_secret_filter() reaches directly; missing them here would
    leak a filter onto those handlers across the whole remaining test run.

    Generalized from the equivalent class-local fixture this module used to
    carry only for ``TestInstallSecretFilter`` in ``test_apikeys.py`` — every
    test that builds a real plugin needs the same protection, not just the
    tests that exercise ``install_secret_filter`` directly.
    """
    loggers = [
        logging.getLogger(name)
        for name in ("", "supybot", "llm", "LiteLLM", "LiteLLM Proxy", "LiteLLM Router")
    ]
    handler_snapshot = {
        handler: list(handler.filters) for logger in loggers for handler in logger.handlers
    }
    last_resort_snapshot = (
        list(logging.lastResort.filters) if logging.lastResort is not None else None
    )
    yield
    for handler, filters in handler_snapshot.items():
        handler.filters = filters
    if logging.lastResort is not None and last_resort_snapshot is not None:
        logging.lastResort.filters = last_resort_snapshot


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse outbound network, but permit loopback.

    The guard exists to stop tests reaching real providers. A test-local
    HTTP server on 127.0.0.1 is not a provider call, and blocking it would
    make the redirect-refusal tests in test_statuspage_fetch.py impossible
    to write without duplicating this fixture.
    """
    loopback = {"127.0.0.1", "::1", "localhost"}
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def _guard(real):
        def _inner(self, addr, *args, **kwargs):
            host = addr[0] if isinstance(addr, tuple) else None
            if host in loopback:
                return real(self, addr, *args, **kwargs)
            raise RuntimeError("test attempted a real network connection — mock the provider call")

        return _inner

    monkeypatch.setattr(socket.socket, "connect", _guard(real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", _guard(real_connect_ex))


@pytest.fixture(autouse=True)
def _run_dispatch_threads_inline(mocker: MockerFixture) -> None:
    """Run ``_dispatch_addressed_async`` inline instead of on a daemon thread.

    Production offloads addressed-message dispatch (doPrivmsg / invalidCommand)
    to a ``world.SupyThread`` so the IRC driver thread is freed to flush the
    ``+typing`` indicator immediately instead of blocking on LLM generation
    (otherwise "is composing" only appears at the same instant as the reply).
    Tests assert on dispatch side effects right after the call, so replacing
    ``SupyThread`` with an inline runner avoids a worker-thread race. Tests
    that need to inspect the offload itself re-patch ``SupyThread`` locally.
    """

    class _InlineThread:
        def __init__(self, *args: object, target: object = None, **kwargs: object) -> None:
            self._target = target

        def start(self) -> None:
            if callable(self._target):
                self._target()

    mocker.patch("llm.plugin.world.SupyThread", _InlineThread)


# =============================================================================
# Session-scoped fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def cleanup_limnoria_logging() -> Generator[None]:
    """Clean up Limnoria's logging handlers to prevent errors on shutdown.

    Limnoria registers atexit handlers that try to log shutdown messages.
    When pytest closes stdout before these handlers run, it causes
    'I/O operation on closed file' errors. This fixture removes
    Limnoria's stream handlers before pytest cleanup.
    """
    yield

    # Remove all stream handlers from supybot logger to prevent
    # logging to closed stdout/stderr after pytest finishes
    try:
        import supybot.log

        supybot_logger = logging.getLogger("supybot")
        for handler in supybot_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                supybot_logger.removeHandler(handler)

        # Also clean the root supybot module logger if it exists
        if hasattr(supybot.log, "log"):
            for handler in supybot.log.log.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    supybot.log.log.removeHandler(handler)
    except (ImportError, AttributeError):
        pass

    # Unregister supybot's atexit handlers that cause the logging errors
    try:
        import supybot.world

        # Clear the atexit callbacks that supybot registers
        if hasattr(supybot.world, "dying"):
            supybot.world.dying = True  # Prevent further shutdown logging
    except (ImportError, AttributeError):
        pass


# =============================================================================
# Database fixture
# =============================================================================


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[LLMDatabase, None, None]:
    """Create an LLMDatabase backed by a temporary file with automatic cleanup."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    yield db
    db.close()


# =============================================================================
# Reminder fixture helpers
# =============================================================================


def make_reminder_row(
    *,
    event_name: str = "evt",
    nick: str = "testnick",
    channel: str = "#test",
    message: str = "",
    action_prompt: str = "",
    account: str | None = None,
    fire_at: float = 0.0,
    chain_position: int = 1,
    recurrence_seconds: int | None = None,
    recurrence_rrule: str | None = None,
    watch_mode: bool = False,
    id: int = 0,  # noqa: A002 — keyword-only builder, builtin shadow is fine.
    created_at: float = 0.0,
) -> ReminderRow:
    """Build a ReminderRow with sensible defaults for tests.

    Localizes test exposure to the ReminderRow shape so future column churn
    only touches this helper. Keyword-only on purpose: positional args invite
    silent breakage when the row layout changes.
    """
    return ReminderRow(
        id=id,
        event_name=event_name,
        nick=nick,
        channel=channel,
        message=message,
        action_prompt=action_prompt,
        account=account,
        fire_at=fire_at,
        created_at=created_at,
        chain_position=chain_position,
        recurrence_seconds=recurrence_seconds,
        recurrence_rrule=recurrence_rrule,
        watch_mode=watch_mode,
    )


# =============================================================================
# litellm completion-response builders
# =============================================================================


def make_tool_call(
    name: str,
    arguments: dict[str, Any] | str | None = None,
    *,
    call_id: str = "call_0",
) -> MagicMock:
    """Build a mock litellm tool_call.

    Produces the ``.id`` / ``.type`` / ``.function.name`` / ``.function.arguments``
    shape that the assistant tool loop reads (service.py:assistant_completion).
    ``arguments`` may be a dict (JSON-encoded for you) or a raw string (passed
    through verbatim — useful for exercising malformed-JSON handling). Centralizes
    the litellm object layout that the tool-loop tests otherwise hand-roll, so a
    litellm shape change is absorbed in one place.
    """
    if arguments is None:
        arguments = {}
    arguments_str = arguments if isinstance(arguments, str) else json.dumps(arguments)

    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.type = "function"
    tool_call.function.name = name
    tool_call.function.arguments = arguments_str
    return tool_call


def make_completion_response(
    content: str | None = "response text",
    *,
    tool_calls: list[Any] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = TEST_MODEL,
    grounding: bool = False,
) -> MagicMock:
    """Build a mock that mimics a litellm chat-completion response.

    Mirrors the shape hand-rolled across test_service / test_assistant /
    test_reminders: ``r.choices[0].message.content`` / ``.tool_calls`` and
    ``r.usage.prompt_tokens`` / ``.completion_tokens``. Pass ``tool_calls`` built
    with :func:`make_tool_call` for tool-loop tests; ``grounding=True`` attaches
    the vertex grounding marker in ``_hidden_params``. Keeping this in one place
    confines litellm-response coupling so a shape change touches one builder
    rather than hundreds of call sites.
    """
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.role = "assistant"

    choice = MagicMock()
    choice.message = message
    choice.grounding_metadata = None

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.model_extra = {}
    response._hidden_params = (
        {"vertex_ai_grounding_metadata": {"search_queries": ["q"]}} if grounding else {}
    )

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response.usage = usage

    return response


# =============================================================================
# Shared fixtures for plugin initialization
# =============================================================================


@pytest.fixture
def mock_irc(mocker: MockerFixture) -> MagicMock:
    """Create a mock IRC object suitable for plugin initialization.

    Provides the minimum attributes required by LLM.__init__:
    nick, state.channels, and state.capabilities_ack.
    """
    irc = mocker.MagicMock()
    irc.nick = "testbot"
    irc.state = mocker.MagicMock()
    irc.state.channels = {}
    irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    irc.state.nickToAccount = mocker.MagicMock(return_value=None)
    return irc


@pytest.fixture
def plugin_env(mocker: MockerFixture):
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    registry = make_registry_side_effect()

    mock_irc = mocker.MagicMock()
    mock_irc.nick = "testbot"
    mock_irc.state = mocker.MagicMock()
    mock_irc.state.channels = {"#test": mocker.MagicMock(topic="Test topic")}
    mock_irc.state.capabilities_ack = set()
    # Default: no NickServ account (nick fallback)
    mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

    mock_msg = mocker.MagicMock()
    mock_msg.prefix = "testnick!user@host"
    mock_msg.args = ("#test", "test message")
    mock_msg.time = time.time() + 100  # future time -- not ZNC playback
    mock_msg.channel = "#test"
    mock_msg.nick = "testnick"
    mock_msg.server_tags = {}  # default: no IRCv3 account-tag

    mocker.patch.object(LLM, "registryValue", side_effect=registry)
    mocker.patch("llm.plugin.LLMService")
    mocker.patch("llm.plugin.LLMDatabase")
    mocker.patch("llm.plugin.log")
    mocker.patch("llm.plugin.httpserver")
    mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    mocker.patch("llm.plugin.schedule.removeEvent")
    mocker.patch("llm.plugin.schedule.addEvent")
    # Default: registered user (grant llm.* command caps but not owner/admin/trusted)
    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: cap.startswith("llm."),
    )

    plugin = LLM(mock_irc)
    # After __init__, swap registryValue to a plain MagicMock so
    # each test can override specific keys while keeping defaults.
    plugin.registryValue = mocker.MagicMock(side_effect=registry)

    # Provide the MetaSynchronized RLock that _allow_concurrent expects.
    plugin._MetaSynchronized_rlock = threading.RLock()

    # sanitize_output is a passthrough in tests (the mock would return MagicMock).
    plugin.llm_service.sanitize_output.side_effect = lambda x: x

    # During the unified-assistant transition, ask-style command tests still
    # exercise the existing completion bridge through assistant_request().
    # The bridge wraps CompletionResult into AssistantResult so the chat
    # impl's structured-signal suppression check (last_successful_tool /
    # final_text_after_tools) sees defaults rather than AttributeError.
    def _assistant_request_bridge(
        prompt: str,
        *,
        request_context,
        db=None,
        context=None,
        bot_nick=None,
        images=None,
        history=None,
        channel_history=None,
        irc=None,
        msg=None,
        system_prompt=None,
        memories=None,
        user_instruction=None,
        search_fn=None,
        fetch_fn=None,
        code_fn=None,
        draw_fn=None,
        cleanup_fn=None,
        set_reminder_fn=None,
        list_pending_tasks_fn=None,
        cancel_pending_task_fn=None,
        cancel_all_pending_tasks_fn=None,
        schedule_llm_task_fn=None,
        extra_tools=None,
        extra_handlers=None,
        model_override=None,
        manage_typing=True,
        exclude_tools=frozenset(),
    ):
        from llm.service import AssistantResult as _AssistantResult

        completion_result = plugin.llm_service.completion(
            prompt,
            command="ask",
            images=images,
            history=history,
            channel_history=channel_history,
            irc=irc,
            msg=msg,
            system_prompt=system_prompt,
            memories=memories,
            user_instruction=user_instruction,
        )
        return _AssistantResult(
            content=completion_result.content,
            prompt_tokens=completion_result.prompt_tokens,
            completion_tokens=completion_result.completion_tokens,
            cost=completion_result.cost,
            model=completion_result.model,
            grounding_used=completion_result.grounding_used,
            error=completion_result.error,
        )

    plugin.llm_service.assistant_request.side_effect = _assistant_request_bridge

    # migrate_nick / migrate_conversations / migrate_user_data return an int
    # (0 = nothing to migrate) by default.
    plugin.db.migrate_nick.return_value = 0
    plugin.db.migrate_conversations.return_value = 0
    plugin.db.migrate_user_data.return_value = 0

    try:
        yield plugin, mock_irc, mock_msg
    finally:
        # die() must be idempotent under the executor wiring (see Task 3
        # — shutdown is idempotent, db.close is sqlite-idempotent). Don't
        # suppress exceptions here: if die() raises, that's a real
        # lifecycle bug we want the test to surface.
        plugin.die()


def make_registry_side_effect(overrides: dict[str, Any] | None = None):
    """Create a registryValue side_effect function with standard defaults.

    The base defaults provide the superset of config keys used across all test
    files. Tests can override specific keys via the ``overrides`` parameter.

    Args:
        overrides: Optional dict of config keys to override or extend.

    Returns:
        A callable suitable for ``registryValue.side_effect``.
    """
    defaults: dict[str, Any] = {
        # Capability-based settings (Phase 2 Task 5a/5b — sole surface).
        "assistantModel": TEST_MODEL,
        "assistantSystemPrompt": "You are helpful.",
        "imageModel": "dall-e-3",
        # Plugin-level init config
        "httpRoot": "",
        "databasePath": "",
        "contextMaxMessages": 20,
        "contextTimeoutMinutes": 30,
        "contextEnabled": True,
        "channelContextMaxMessages": 10,
        "contextTrackAllMessages": False,
        # Code command
        "codeModel": TEST_MODEL,
        "codeSystemPrompt": "You write code.",
        # Draw command
        "drawTimeout": 60,
        "drawAutoRewriteMax": 2,
        "drawContextMaxAgeSeconds": 60,
        # Expiry (pending task retry)
        "askExpiry": 60,
        "codeExpiry": 60,
        "drawExpiry": 60,
        # Memory extraction
        "memoryEnabled": True,
        "memoryMaxPerUser": 50,
        "memoryCleanupInterval": 3,
        "memoryPromotionThreshold": 2,
        "memoryCandidateTTLDays": 14,
        # Shared
        "timeout": 30,
        "maxPromptLength": 10000,
        "commandPrefixes": ["."],
        "httpUrlBase": TEST_URL_BASE,
        "longReplyTeaserMaxChars": 220,
        "fileCleanupAge": 24,
        "fileCleanupMax": 100,
        "logLevel": "WARNING",
        # Rate limiting
        "enforceRateLimits": True,
        # ask
        "askRateLimitCount": 15,
        "askRateLimitWindow": 60,
        "askTrustedRateLimitCount": 15,
        "askTrustedRateLimitWindow": 60,
        "askUnregRateLimitCount": 15,
        "askUnregRateLimitWindow": 60,
        # code
        "codeRateLimitCount": 10,
        "codeRateLimitWindow": 60,
        "codeTrustedRateLimitCount": 0,
        "codeTrustedRateLimitWindow": 60,
        "codeUnregRateLimitCount": 2,
        "codeUnregRateLimitWindow": 60,
        # draw
        "drawRateLimitCount": 2,
        "drawRateLimitWindow": 300,
        "drawTrustedRateLimitCount": 5,
        "drawTrustedRateLimitWindow": 60,
        "drawUnregRateLimitCount": 0,
        "drawUnregRateLimitWindow": 60,
        # story (mirrors draw — expensive image command)
        "storyRateLimitCount": 2,
        "storyRateLimitWindow": 300,
        "storyTrustedRateLimitCount": 5,
        "storyTrustedRateLimitWindow": 60,
        "storyUnregRateLimitCount": 0,
        "storyUnregRateLimitWindow": 60,
        # Search/fetch tools
        "searchModel": "",
        # Assistant tool-calling backend
        "metaMaxSteps": 7,
        # IRCv3 join optimization
        "skipAutoWhoOnJoin": True,
        # Pending-task tool gate (default off, mirrors config.py)
        "pendingTasksEnabled": False,
        # Limnoria bridge
        "bridgeEnabled": False,
        "bridgeAllowedPlugins": [],
        "bridgeAllowMutating": False,
        "bridgeDebugInChannel": False,
        # Async LLM concurrency cap
        "maxConcurrentLLMCalls": 16,
        # Verse subsystem
        "verseEnabled": False,
        "verseCompactionModel": "gemini/gemini-flash-lite-latest",
        "verseStyleExemplars": [],
    }
    if overrides:
        defaults.update(overrides)

    def side_effect(key: str, *args: object) -> object:
        return defaults.get(key, "")

    return side_effect


# =============================================================================
# Service factory fixture
# =============================================================================


@pytest.fixture
def make_service(mocker: MockerFixture) -> Callable[..., tuple[LLMService, Mock]]:
    """Factory fixture that creates an LLMService with standard config defaults.

    Usage::

        def test_something(make_service):
            service, plugin = make_service()
            # or with overrides:
            service, plugin = make_service(assistantModel="gemini/gemini-2.0-flash")
    """

    def _make(**overrides: Any) -> tuple[LLMService, Mock]:
        plugin = mocker.Mock()
        plugin.log = mocker.Mock()
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect(overrides or None))

        # Service tests dispatch scheduled-task fires synchronously to
        # assert downstream effects; with the executor migration the
        # real fire() submits the worker via plugin._llm_executor.submit.
        # Run the worker inline here so existing tests behave as before.
        plugin._llm_executor.closing = False

        def _sync_submit(_label, fn, *args, **kwargs):
            fn(*args, **kwargs)
            return mocker.Mock()

        plugin._llm_executor.submit.side_effect = _sync_submit

        # _safe_queue is the worker-side wrapper around irc.queueMsg.
        # Existing tests assert directly on irc.queueMsg, so make
        # _safe_queue a passthrough that calls queueMsg.
        def _passthrough_safe_queue(irc, msg):
            irc.queueMsg(msg)
            return True

        plugin._safe_queue.side_effect = _passthrough_safe_queue

        # Every raw-queue send builds its IrcMsg with _safe_privmsg (the
        # safeArgument counterpart to irc.reply's). Use the real staticmethod so
        # tests inspecting queueMsg see an actual PRIVMSG, not a bare Mock.
        from llm.plugin import LLM

        plugin._safe_privmsg.side_effect = LLM._safe_privmsg

        return LLMService(plugin), plugin

    return _make


# =============================================================================
# Plugin initialization patches
# =============================================================================


def plugin_init_patches(
    mocker: MockerFixture, *, mock_database: bool = True
) -> dict[str, MagicMock]:
    """Patch all external dependencies for LLM.__init__.

    This patches LLMService, log, httpserver.hook, schedule.addPeriodicEvent,
    and schedule.removeEvent. Optionally patches LLMDatabase.  All patches are
    automatically reverted at the end of the test by pytest-mock.

    Also stubs LLMExecutor with a synchronous, deterministic test double so
    background-work tests (reminder action, scheduled task)
    can assert on side effects immediately after the schedule callback
    fires — no thread-pool race.

    Args:
        mocker: The pytest-mock fixture.
        mock_database: If True (default), also patches LLMDatabase.

    Returns:
        Dict of patch names to their MagicMock objects.
    """
    patches: dict[str, Any] = {}
    patches["LLMService"] = mocker.patch("llm.plugin.LLMService")
    if mock_database:
        patches["LLMDatabase"] = mocker.patch("llm.plugin.LLMDatabase")
    patches["log"] = mocker.patch("llm.plugin.log")
    patches["httpserver_hook"] = mocker.patch("llm.plugin.httpserver.hook")
    patches["addPeriodicEvent"] = mocker.patch("llm.plugin.schedule.addPeriodicEvent")
    patches["removeEvent"] = mocker.patch("llm.plugin.schedule.removeEvent")
    # Also patch addEvent so the daily compaction timer (PR 3 / E3)
    # doesn't hit the real supybot schedule between tests — repeated
    # registrations would otherwise collide on the unique-name check.
    patches["addEvent"] = mocker.patch("llm.plugin.schedule.addEvent")

    # LLMExecutor stub: submit runs the function synchronously so tests
    # that callback() and immediately assert on queueMsg / db writes
    # don't race a worker thread.
    sync_stub = _SyncLLMExecutor()
    patches["LLMExecutor"] = mocker.patch("llm.plugin.LLMExecutor", return_value=sync_stub)
    return patches


class _SyncLLMExecutor:
    """Synchronous test double for LLMExecutor used by plugin_init_patches.

    `submit(label, fn, ...)` runs `fn` inline and returns a completed
    `Future` carrying its result. Counters track in-flight tasks so
    `running()` and `queued()` look correct mid-call. Drain/shutdown
    are no-ops.
    """

    def __init__(self, max_concurrency: int = 16) -> None:
        from concurrent.futures import Future

        self._max = max_concurrency
        self._running = 0
        self._queued = 0
        self.closing = False
        self._Future = Future

    @property
    def max_concurrency(self) -> int:
        return self._max

    def running(self) -> int:
        return self._running

    def queued(self) -> int:
        return self._queued

    def permit(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            self._running += 1
            try:
                yield
            finally:
                self._running -= 1

        return _cm()

    def submit(self, _label, fn, *args, **kwargs):
        fut = self._Future()
        self._running += 1
        try:
            result = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)
        else:
            fut.set_result(result)
        finally:
            self._running -= 1
        return fut

    def drain(self, timeout: float) -> bool:  # noqa: ARG002
        return True

    def shutdown(self) -> None:
        self.closing = True


@pytest.fixture
def status_plugin():
    """A minimal stand-in exercising the status poller logic in isolation.

    Builds the real methods onto a bare object rather than constructing the
    whole LLM plugin, which needs an IRC connection, a database, and an
    executor pool. The methods under test only touch attributes defined here.
    """
    from unittest.mock import MagicMock

    from llm import statuspage
    from llm.plugin import LLM

    obj = MagicMock()
    obj._registry = {"statusPageUrl": "https://status.claude.com"}
    obj.registryValue = lambda key, *a, **k: obj._registry.get(key)
    obj._STATUS_POLL_INTERVAL = LLM._STATUS_POLL_INTERVAL
    obj._STATUS_MAX_ANNOUNCE_PER_POLL = LLM._STATUS_MAX_ANNOUNCE_PER_POLL
    obj._STATUS_FETCH_FLOOR = LLM._STATUS_FETCH_FLOOR
    obj._status_state = statuspage.StatusState()
    obj._status_read_cache = None
    obj._status_last_fetch = 0.0
    obj._fetch_calls = 0
    obj._fake_snapshot = None
    obj._fake_error = None
    obj._now = 1000.0

    def fake_fetch():
        obj._fetch_calls += 1
        if obj._fake_error:
            err, obj._fake_error = obj._fake_error, None
            raise err
        snap = obj._fake_snapshot
        if snap is None:
            snap = statuspage.Snapshot(
                page_name="Claude",
                page_url="https://status.claude.com",
                indicator="none",
                description="All Systems Operational",
                components={},
                incidents={},
                fetched_at=obj._now,
            )
        return snap

    obj._status_fetch_snapshot = fake_fetch
    obj._status_now = lambda: obj._now
    obj._announce_status = MagicMock()
    obj._run_status_poll = LLM._run_status_poll.__get__(obj)
    obj._status_fetch_now = LLM._status_fetch_now.__get__(obj)
    return obj
