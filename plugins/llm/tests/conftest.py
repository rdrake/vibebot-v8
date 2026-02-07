"""Pytest configuration and shared fixtures for LLM plugin tests."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

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
def mock_irc() -> MagicMock:
    """Create a mock IRC object suitable for plugin initialization.

    Provides the minimum attributes required by LLM.__init__:
    nick, state.channels, and state.capabilities_ack.
    """
    irc = MagicMock()
    irc.nick = "testbot"
    irc.state = MagicMock()
    irc.state.channels = {}
    irc.state.capabilities_ack = set()
    return irc


def make_registry_side_effect(overrides: dict[str, Any] | None = None):
    """Create a registryValue side_effect function with standard defaults.

    The base defaults provide the minimum config needed to initialize the LLM
    plugin. Tests can override specific keys via the ``overrides`` parameter.

    Args:
        overrides: Optional dict of config keys to override or extend.

    Returns:
        A callable suitable for ``registryValue.side_effect``.
    """
    defaults: dict[str, Any] = {
        "httpRoot": "",
        "databasePath": "",
        "contextMaxMessages": 20,
        "contextTimeoutMinutes": 30,
        "contextEnabled": True,
        "channelContextMaxMessages": 10,
    }
    if overrides:
        defaults.update(overrides)

    def side_effect(key: str, *args: object) -> object:
        return defaults.get(key, "")

    return side_effect


@contextlib.contextmanager
def plugin_init_patches(*, mock_database: bool = True) -> Generator[dict[str, MagicMock]]:
    """Context manager that patches all external dependencies for LLM.__init__.

    This patches LLMService, log, httpserver.hook, schedule.addPeriodicEvent,
    and schedule.removeEvent. Optionally patches LLMDatabase.

    Args:
        mock_database: If True (default), also patches LLMDatabase.

    Yields:
        Dict of patch names to their MagicMock objects.
    """
    patches: dict[str, Any] = {}

    with contextlib.ExitStack() as stack:
        patches["LLMService"] = stack.enter_context(patch("llm.plugin.LLMService"))
        if mock_database:
            patches["LLMDatabase"] = stack.enter_context(patch("llm.plugin.LLMDatabase"))
        patches["log"] = stack.enter_context(patch("llm.plugin.log"))
        patches["httpserver_hook"] = stack.enter_context(patch("llm.plugin.httpserver.hook"))
        patches["addPeriodicEvent"] = stack.enter_context(
            patch("llm.plugin.schedule.addPeriodicEvent")
        )
        patches["removeEvent"] = stack.enter_context(patch("llm.plugin.schedule.removeEvent"))
        yield patches
