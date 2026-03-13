"""RPG: Linux filesystem dungeon master for IRC."""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, plugin

Class = plugin.RPG
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
