"""Configuration for LLM plugin."""

from __future__ import annotations

import difflib
import logging
import threading

import litellm
import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("LLM")

_log = logging.getLogger("supybot.plugins.LLM.config")


class ValidatedModelName(registry.String):
    """A model name validated against litellm's known models.

    - Empty strings are accepted (means "not configured").
    - Models that litellm can parse (provider recognized) are accepted.
      If the specific model isn't in litellm.model_list, a warning is logged once.
    - Models that litellm cannot parse at all are rejected with suggestions.
    """

    _warned: set[str] = set()
    _warned_lock: threading.Lock = threading.Lock()

    def setValue(self, v: str) -> None:  # noqa: N802
        v = v.strip()
        if v:
            self._validate_model(v)
        super().setValue(v)

    def _validate_model(self, model: str) -> None:
        """Validate a model name against litellm's known models."""
        try:
            litellm.get_llm_provider(model)
        except litellm.exceptions.BadRequestError:
            suggestions = self._suggest_models(model)
            msg = f"Unknown model: {model!r}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            else:
                msg += " See https://docs.litellm.ai/docs/providers for supported models."
            raise registry.InvalidRegistryValue(msg)  # noqa: B904

        # Provider recognized — warn once if model is not in the known list
        if model not in litellm.model_list:
            with ValidatedModelName._warned_lock:
                if model not in ValidatedModelName._warned:
                    ValidatedModelName._warned.add(model)
                    suggestions = self._suggest_models(model)
                    hint = ""
                    if suggestions:
                        hint = f" Similar known models: {', '.join(suggestions)}."
                    _log.warning(
                        "Model %r not in litellm's known model list (may be a custom "
                        "deployment or newer model).%s",
                        model,
                        hint,
                    )

    @staticmethod
    def _suggest_models(model: str, n: int = 3, cutoff: float = 0.6) -> list[str]:
        """Suggest similar model names using fuzzy matching."""
        # Try matching against full model list first
        matches = difflib.get_close_matches(model, litellm.model_list, n=n, cutoff=cutoff)
        if matches:
            return matches

        # If provider-prefixed, also try matching within that provider's models
        if "/" in model:
            provider, model_name = model.split("/", 1)
            provider_models = litellm.models_by_provider.get(provider, set())
            if provider_models:
                sub_matches = difflib.get_close_matches(
                    model_name, list(provider_models), n=n, cutoff=cutoff
                )
                return [f"{provider}/{m}" for m in sub_matches]

        return []


def configure(advanced: bool) -> None:
    """Plugin configuration wizard."""
    from supybot.questions import yn  # noqa: F401

    conf.registerPlugin("LLM", True)

    print("=" * 60)
    print("LLM Plugin Configuration")
    print("=" * 60)
    print("\nThis plugin provides AI-powered commands using LiteLLM.")
    print("You'll need API keys for the models you want to use.")
    print("\nYou can configure API keys now or later using:")
    print("  config plugins.LLM.askApiKey YOUR_KEY")
    print("\nFor more info, see the README.md")
    print("=" * 60)


LLM = conf.registerPlugin("LLM")

# ============================================================================
# API Keys (private - never logged)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "askApiKey",
    registry.String("", _("""API key for ask command"""), private=True),
)

conf.registerGlobalValue(
    LLM,
    "codeApiKey",
    registry.String("", _("""API key for code command"""), private=True),
)

conf.registerGlobalValue(
    LLM,
    "drawApiKey",
    registry.String("", _("""API key for draw command"""), private=True),
)

# ============================================================================
# System Prompts (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "askSystemPrompt",
    registry.String(
        "You are a helpful IRC assistant. Keep responses concise and suitable for IRC chat. "
        "Avoid markdown formatting. Be direct and informative.",
        _("""System prompt for ask command - defines bot personality and constraints"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeSystemPrompt",
    registry.String(
        "You are a helpful code assistant. Explain your code and provide context. "
        "Use markdown formatting for code blocks. "
        "For math equations, use $...$ for inline math and $$...$$ for display math.",
        _("""System prompt for code command"""),
    ),
)

# ============================================================================
# Model Configuration (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "askModel",
    ValidatedModelName(
        "gemini/gemini-flash-latest",
        _("""Model for ask command (supports vision)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeModel",
    ValidatedModelName(
        "gemini/gemini-1.5-flash",
        _("""Model for code generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "drawModel",
    ValidatedModelName(
        "vertex_ai/imagen-4.0-generate-001",
        _("""Model for image generation"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawTimeout",
    registry.NonNegativeInteger(
        120,
        _("""Timeout for image generation API calls in seconds. Image generation
        is slower than text completion, especially with auto-rewrite retries.
        Set to 0 to use the global timeout setting instead."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "drawAutoRewriteMax",
    registry.NonNegativeInteger(
        3,
        _("""Maximum number of automatic prompt rewrites when image generation
        is blocked by content safety filters. Set to 0 to disable. Each retry
        uses the ask model to rewrite the prompt."""),
    ),
)

# ============================================================================
# Video Generation (animate command)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "animateApiKey",
    registry.String("", _("""API key for animate command"""), private=True),
)

conf.registerChannelValue(
    LLM,
    "animateModel",
    registry.String(
        "grok-imagine-video",
        _("""Model for video generation"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateTimeout",
    registry.NonNegativeInteger(
        600,
        _("""Timeout for video generation API calls in seconds. Video generation
        is slower than image generation due to polling for completion.
        Set to 0 to use the global timeout setting instead."""),
    ),
)

# ============================================================================
# HTTP Server Settings (for code/image output)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "httpRoot",
    registry.String(
        "",
        _("""Filesystem path to save code/image files. If empty, uses
        Limnoria's web directory (data/web/llm/) and built-in HTTP server.
        If set, files are saved there and Limnoria's HTTP server is NOT
        used (external server like nginx expected)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "httpUrlBase",
    registry.String(
        "",
        _("""Base URL for accessing saved files. If empty, uses Limnoria's
        HTTP server publicUrl + /llm/. Example: https://example.com/llm"""),
    ),
)

# ============================================================================
# Conversation Context (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "contextEnabled",
    registry.Boolean(
        True,
        _("""Enable conversation context (memory between messages)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "contextMaxMessages",
    registry.PositiveInteger(
        20,
        _("""Maximum messages to keep in conversation history"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "contextTimeoutMinutes",
    registry.PositiveInteger(
        5,
        _("""Clear context after this many minutes of inactivity"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "contextTrackAllMessages",
    registry.Boolean(
        False,
        _("""Track all channel messages for richer context (privacy: disabled by default
        since messages are sent to third-party LLM providers)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "channelContextMaxMessages",
    registry.PositiveInteger(
        10,
        _("""Maximum messages in shared channel context. This allows the bot to
        follow group conversations - when Alice asks something, Bob can continue
        the thread because the bot remembers Alice's exchange."""),
    ),
)

# ============================================================================
# Advanced Settings
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "timeout",
    registry.PositiveInteger(
        30,
        _("""Timeout for LLM API calls in seconds"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "maxPromptLength",
    registry.PositiveInteger(
        10000,
        _("""Maximum length of user prompts in characters"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "fileCleanupAge",
    registry.PositiveInteger(
        720,
        _("""Delete HTTP files older than this many hours (default: 720 = 30 days)"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "fileCleanupMax",
    registry.PositiveInteger(
        1000,
        _("""Maximum number of files to keep in HTTP directory"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "commandPrefixes",
    registry.SpaceSeparatedListOfStrings(
        [".", "/"],
        _("""Command prefixes to sanitize in output. Lines starting with these
        are prefixed with a space to prevent IRC command injection. Default: . /"""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "databasePath",
    registry.String(
        "",
        _("""Path to SQLite database file for persistence (reminders, usage tracking).
        If empty, uses Limnoria's data directory (data/LLM.db)."""),
    ),
)
