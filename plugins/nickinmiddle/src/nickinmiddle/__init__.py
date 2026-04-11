"""NickInMiddle: Recognize the bot's nick in the middle of a message."""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, plugin

Class = plugin.NickInMiddle
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
