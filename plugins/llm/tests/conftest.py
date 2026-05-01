"""Pytest configuration and shared fixtures for LLM plugin tests."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from llm.persistence import LLMDatabase, ReminderRow
from llm.plugin import LLM
from llm.service import LLMService

if TYPE_CHECKING:
    from unittest.mock import MagicMock, Mock

    from pytest_mock import MockerFixture

# =============================================================================
# Test constants
# =============================================================================

TEST_MODEL = "gpt-4"
TEST_API_KEY = "test-key"
TEST_URL_BASE = "https://example.com/llm"

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
        search_fn=None,
        fetch_fn=None,
        code_fn=None,
        draw_fn=None,
        cleanup_fn=None,
        list_reminders_fn=None,
        set_reminder_fn=None,
        delete_reminder_fn=None,
        cancel_all_reminders_fn=None,
    ):
        return plugin.llm_service.completion(
            prompt,
            command="ask",
            images=images,
            history=history,
            channel_history=channel_history,
            irc=irc,
            msg=msg,
            system_prompt=system_prompt,
            memories=memories,
        )

    plugin.llm_service.assistant_request.side_effect = _assistant_request_bridge

    # migrate_nick / migrate_conversations return an int (0 = nothing to migrate) by default.
    plugin.db.migrate_nick.return_value = 0
    plugin.db.migrate_conversations.return_value = 0

    return plugin, mock_irc, mock_msg


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
        # Plugin-level init config
        "httpRoot": "",
        "databasePath": "",
        "contextMaxMessages": 20,
        "contextTimeoutMinutes": 30,
        "contextEnabled": True,
        "channelContextMaxMessages": 10,
        "contextTrackAllMessages": False,
        # Ask command
        "askApiKey": TEST_API_KEY,
        "askModel": TEST_MODEL,
        "askSystemPrompt": "You are helpful.",
        # Code command
        "codeApiKey": TEST_API_KEY,
        "codeModel": TEST_MODEL,
        "codeSystemPrompt": "You write code.",
        # Draw command
        "drawApiKey": TEST_API_KEY,
        "drawModel": "dall-e-3",
        "drawTimeout": 60,
        "drawAutoRewriteMax": 2,
        "drawContextMaxAgeSeconds": 60,
        # Expiry (pending task retry)
        "askExpiry": 60,
        "codeExpiry": 60,
        "drawExpiry": 60,
        # Memory extraction
        "memoryEnabled": True,
        "memoryExtractionModel": "gemini/gemini-2.0-flash-lite",
        "memoryCleanupModel": "gemini/gemini-3.1-flash-lite-preview",
        "memoryApiKey": "",
        "memoryMaxPerUser": 50,
        "memoryCleanupInterval": 3,
        # Spontaneous participation
        "spontaneousEnabled": False,
        "spontaneousChance": 15,
        "spontaneousCooldown": 2,
        "spontaneousModel": "gemini/gemini-2.0-flash-lite",
        "spontaneousApiKey": "",
        "spontaneousSystemPrompt": "You are a regular in this IRC channel.",
        # Shared
        "timeout": 30,
        "maxPromptLength": 10000,
        "commandPrefixes": ["."],
        "httpUrlBase": TEST_URL_BASE,
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
        # Search/fetch tools
        "searchApiKey": "",
        "searchModel": "",
        # Assistant tool-calling backend
        "metaModel": "",
        "metaApiKey": "",
        "metaMaxSteps": 7,
        # IRCv3 join optimization
        "skipAutoWhoOnJoin": True,
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
            service, plugin = make_service(askModel="gemini/gemini-2.0-flash")
    """

    def _make(**overrides: Any) -> tuple[LLMService, Mock]:
        plugin = mocker.Mock()
        plugin.log = mocker.Mock()
        plugin.registryValue = mocker.Mock(side_effect=make_registry_side_effect(overrides or None))
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
    return patches
