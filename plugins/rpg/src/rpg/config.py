"""Configuration for RPG plugin."""

from __future__ import annotations

import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("RPG")


def configure(advanced: bool) -> None:
    """Plugin configuration wizard."""
    conf.registerPlugin("RPG", True)


RPG = conf.registerPlugin("RPG")

conf.registerGlobalValue(
    RPG,
    "narratorApiKey",
    registry.String("", _("""API key for narrator LLM calls."""), private=True),
)

conf.registerChannelValue(
    RPG,
    "narratorModel",
    registry.String(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for narrator flavor text (cheap flash-tier recommended)."""),
    ),
)

conf.registerGlobalValue(
    RPG,
    "narratorTimeout",
    registry.PositiveInteger(
        2,
        _(
            """Timeout in seconds for narrator LLM calls. Falls back to deterministic text on timeout."""
        ),
    ),
)

conf.registerGlobalValue(
    RPG,
    "databasePath",
    registry.String(
        "",
        _("""Path to SQLite database. If empty, uses Limnoria's data directory (data/RPG.db)."""),
    ),
)

conf.registerChannelValue(
    RPG,
    "enabled",
    registry.Boolean(False, _("""Enable RPG in this channel.""")),
)

conf.registerGlobalValue(
    RPG,
    "combatRoundSeconds",
    registry.PositiveInteger(
        20,
        _("""Seconds per combat round before AFK auto-action."""),
    ),
)

conf.registerGlobalValue(
    RPG,
    "spawnCooldownMinutes",
    registry.PositiveInteger(
        30,
        _("""Minutes before enemies respawn in a cleared room."""),
    ),
)
