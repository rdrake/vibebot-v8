"""Pytest configuration and shared fixtures for RPG plugin tests."""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_limnoria_logging() -> Generator[None]:
    """Clean up Limnoria's logging handlers to prevent errors on shutdown."""
    yield

    try:
        import supybot.log

        supybot_logger = logging.getLogger("supybot")
        for handler in supybot_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler):
                supybot_logger.removeHandler(handler)

        if hasattr(supybot.log, "log"):
            for handler in supybot.log.log.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    supybot.log.log.removeHandler(handler)
    except (ImportError, AttributeError):
        pass

    try:
        import supybot.world

        if hasattr(supybot.world, "dying"):
            supybot.world.dying = True
    except (ImportError, AttributeError):
        pass
