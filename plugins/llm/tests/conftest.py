"""Pytest configuration and shared fixtures for LLM plugin tests."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import pytest
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
        # Picard command (reuses ask model/key)
        "picardSystemPrompt": "You are Captain Picard.",
        # Code command
        "codeApiKey": TEST_API_KEY,
        "codeModel": TEST_MODEL,
        "codeSystemPrompt": "You write code.",
        # Draw command
        "drawApiKey": TEST_API_KEY,
        "drawModel": "dall-e-3",
        "drawTimeout": 60,
        "drawAutoRewriteMax": 2,
        # Animate command
        "animateApiKey": TEST_API_KEY,
        "animateModel": "grok-imagine-video",
        "animateTimeout": 600,
        # Expiry (pending task retry)
        "askExpiry": 60,
        "codeExpiry": 60,
        "drawExpiry": 60,
        "animateExpiry": 3600,
        # Memory extraction
        "memoryEnabled": True,
        "memoryExtractionModel": "gemini/gemini-2.0-flash-lite",
        "memoryMaxPerUser": 50,
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
        # animate
        "animateRateLimitCount": 2,
        "animateRateLimitWindow": 600,
        "animateTrustedRateLimitCount": 5,
        "animateTrustedRateLimitWindow": 600,
        "animateUnregRateLimitCount": 0,
        "animateUnregRateLimitWindow": 600,
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
