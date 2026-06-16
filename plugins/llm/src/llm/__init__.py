"""LLM: AI-powered commands using LiteLLM."""

from __future__ import annotations

import sys

__version__ = "0.1.0"

# Make Limnoria's ``reload`` a true deep reload. ``loadPluginModule`` only
# re-execs this package's ``__init__``; the ``from . import`` below would then
# rebind to already-cached submodules (config, plugin, service, verse.*) — so
# edits to those files (e.g. prompt constants in service.py) would silently run
# stale. Dropping any cached submodules first forces the imports below to
# re-execute the whole tree. On first import at startup nothing is cached, so
# this is a no-op then; it only bites on ``@reload``.
for _name in [m for m in list(sys.modules) if m.startswith(__name__ + ".")]:
    del sys.modules[_name]

from . import config, plugin  # noqa: E402  (must follow the submodule purge)

Class = plugin.LLM
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
