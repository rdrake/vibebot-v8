"""Configuration for NickInMiddle plugin."""

from __future__ import annotations

import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("NickInMiddle")


def configure(advanced: bool) -> None:  # noqa: ARG001
    conf.registerPlugin("NickInMiddle", True)


NickInMiddle = conf.registerPlugin("NickInMiddle")

conf.registerChannelValue(
    NickInMiddle,
    "enabled",
    registry.Boolean(
        True,
        _(
            """Whether to rewrite messages that contain the bot's nick in the
            middle so that the bot treats them as addressed. When enabled,
            'do this, vibebot, please' is rewritten to 'vibebot do this, please'
            before the normal addressing logic runs."""
        ),
    ),
)
