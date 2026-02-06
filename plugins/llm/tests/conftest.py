"""Pytest configuration for LLM plugin tests."""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_limnoria_logging() -> None:
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
