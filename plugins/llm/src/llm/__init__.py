"""LLM: AI-powered commands using LiteLLM."""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, plugin

Class = plugin.LLM
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
