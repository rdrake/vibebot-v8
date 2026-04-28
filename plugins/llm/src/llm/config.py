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


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ValidatedLogLevel(registry.String):
    """A log level name validated against Python's standard levels."""

    def setValue(self, v: str) -> None:  # noqa: N802
        v = v.strip().upper()
        if v not in _VALID_LOG_LEVELS:
            raise registry.InvalidRegistryValue(
                f"Invalid log level: {v!r}. Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        super().setValue(v)


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

conf.registerGlobalValue(
    LLM,
    "searchApiKey",
    registry.String(
        "",
        _("""API key for search/fetch tools. Falls back to askApiKey if empty."""),
        private=True,
    ),
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

conf.registerChannelValue(
    LLM,
    "searchModel",
    ValidatedModelName(
        "",
        _("""Model for search/fetch tools. Falls back to askModel if empty."""),
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

conf.registerChannelValue(
    LLM,
    "drawContextMaxAgeSeconds",
    registry.NonNegativeInteger(
        60,
        _("""Only pass conversation context to draw requests when the
        conversation's last activity is within this many seconds. Keeps
        draw calls grounded in very recent discussion without overwhelming
        the image model with stale history. Set to 0 to disable (always
        start fresh)."""),
    ),
)

# ============================================================================
# Memory Extraction
# ============================================================================

conf.registerChannelValue(
    LLM,
    "memoryEnabled",
    registry.Boolean(True, _("""Enable automatic memory extraction from command interactions.""")),
)

conf.registerChannelValue(
    LLM,
    "memoryExtractionModel",
    ValidatedModelName(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for memory extraction (cheap flash-tier recommended)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memoryMaxPerUser",
    registry.PositiveInteger(50, _("""Maximum number of memories stored per user.""")),
)
conf.registerGlobalValue(
    LLM,
    "memoryApiKey",
    registry.String(
        "",
        _("""API key for memory extraction model. Falls back to askApiKey if empty."""),
        private=True,
    ),
)

conf.registerChannelValue(
    LLM,
    "memoryCleanupModel",
    ValidatedModelName(
        "gemini/gemini-3.1-flash-lite-preview",
        _("""Model for memory cleanup (flash-tier recommended)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memoryCleanupInterval",
    registry.NonNegativeInteger(
        3,
        _("""Number of memory extraction passes (that save at least one fact)
        between automatic cleanup runs. Set to 0 to disable periodic cleanup."""),
    ),
)

# ============================================================================
# Spontaneous Participation
# ============================================================================

conf.registerChannelValue(
    LLM,
    "spontaneousEnabled",
    registry.Boolean(
        False,
        _("""Enable spontaneous channel participation. Requires contextTrackAllMessages."""),
    ),
)
conf.registerChannelValue(
    LLM,
    "spontaneousChance",
    registry.PositiveInteger(
        15, _("""Percent chance (1-100) of evaluating a spontaneous reply per message.""")
    ),
)
conf.registerChannelValue(
    LLM,
    "spontaneousCooldown",
    registry.PositiveInteger(2, _("""Minimum minutes between spontaneous replies per channel.""")),
)
conf.registerChannelValue(
    LLM,
    "spontaneousModel",
    ValidatedModelName(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for spontaneous participation (cheap flash-tier recommended)."""),
    ),
)
conf.registerGlobalValue(
    LLM,
    "spontaneousApiKey",
    registry.String(
        "", _("""API key for spontaneous model. Falls back to askApiKey if empty."""), private=True
    ),
)
conf.registerChannelValue(
    LLM,
    "spontaneousSystemPrompt",
    registry.String(
        "You are a regular in this IRC channel. You see the recent conversation "
        "and can jump in if you have something useful, funny, or relevant to add. "
        "Keep it brief — one or two sentences max. Match the tone of the channel. "
        "If the conversation is dead or you have nothing to add, respond with exactly PASS. "
        "You're a channel regular, not an assistant — be natural, have opinions, be yourself.",
        _("""System prompt for spontaneous channel participation."""),
    ),
)

# ============================================================================
# Pending Task Retry (per-command expiry)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "askExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out ask requests.
        Set to 0 to disable background retry for ask."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out code requests.
        Set to 0 to disable background retry for code."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawExpiry",
    registry.NonNegativeInteger(
        60,
        _("""Maximum seconds to keep retrying timed-out draw requests.
        Set to 0 to disable background retry for draw."""),
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

conf.registerGlobalValue(
    LLM,
    "helpUrl",
    registry.String(
        "https://rdrake.github.io/vibebot-v8/",
        _("""URL to the help documentation page. Shown in plugin help output."""),
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
        ["."],
        _("""Command prefixes to sanitize in output. Lines starting with these
        are prefixed with a space to prevent IRC command injection. Default: ."""),
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

conf.registerGlobalValue(
    LLM,
    "logLevel",
    ValidatedLogLevel(
        "WARNING",
        _("""Log level for LLM plugin (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Set to DEBUG for verbose tracing including server response headers."""),
    ),
)

# ============================================================================
# Rate Limiting (per-command, per-tier)
#
# Tiers (checked from most to least privileged):
#   owner/admin  — always exempt (no config needed)
#   trusted      — relaxed limits (Trusted prefix)
#   registered   — standard limits (no prefix, backwards-compatible)
#   unregistered — strictest limits (Unreg prefix)
#
# Setting any count to 0 disables rate limiting for that command+tier.
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "enforceRateLimits",
    registry.Boolean(
        True,
        _("""Enable per-user rate limiting for commands.
        When False, limits are tracked and logged but not enforced (monitor mode).
        Set to True to actively block requests that exceed the limit."""),
    ),
)

# --- ask (cheapest) ---

conf.registerGlobalValue(
    LLM,
    "askRateLimitCount",
    registry.NonNegativeInteger(
        15,
        _("""Max ask requests per registered user within askRateLimitWindow seconds.
        Set to 0 to disable rate limiting for this tier."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askTrustedRateLimitCount",
    registry.NonNegativeInteger(
        15,
        _("""Max ask requests per trusted user within askTrustedRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askUnregRateLimitCount",
    registry.NonNegativeInteger(
        15,
        _("""Max ask requests per unregistered user within askUnregRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (unregistered tier)."""),
    ),
)

# --- code ---

conf.registerGlobalValue(
    LLM,
    "codeRateLimitCount",
    registry.NonNegativeInteger(
        10,
        _("""Max code requests per registered user within codeRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeTrustedRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max code requests per trusted user within codeTrustedRateLimitWindow seconds.
        Set to 0 to disable (trusted users unlimited for code)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeUnregRateLimitCount",
    registry.NonNegativeInteger(
        2,
        _("""Max code requests per unregistered user within codeUnregRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (unregistered tier)."""),
    ),
)

# --- draw (expensive) ---

conf.registerGlobalValue(
    LLM,
    "drawRateLimitCount",
    registry.NonNegativeInteger(
        2,
        _("""Max draw requests per registered user within drawRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawRateLimitWindow",
    registry.PositiveInteger(
        300,
        _("""Time window in seconds for counting draw requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawTrustedRateLimitCount",
    registry.NonNegativeInteger(
        5,
        _("""Max draw requests per trusted user within drawTrustedRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting draw requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawUnregRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max draw requests per unregistered user within drawUnregRateLimitWindow seconds.
        Set to 0 to disable. Note: draw already requires NickServ, so unreg users
        are blocked before this check."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting draw requests (unregistered tier)."""),
    ),
)

# ============================================================================
# Assistant Tool-Calling Backend (shared by @ask, @code, @draw, invalidCommand)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "metaModel",
    ValidatedModelName(
        "",
        _("""Model for the shared assistant tool-calling backend.
        Must support function/tool calling. If empty, falls back to askModel."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "metaApiKey",
    registry.String(
        "",
        _("""API key for the shared assistant backend. Falls back to askApiKey if empty."""),
        private=True,
    ),
)

conf.registerGlobalValue(
    LLM,
    "metaMaxSteps",
    registry.PositiveInteger(
        12,
        _("""Maximum tool-call round trips per assistant invocation.
        Prevents runaway tool loops."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "skipAutoWhoOnJoin",
    registry.Boolean(
        True,
        _(
            """If True (default), suppress Limnoria's automatic WHO query on channel join
            when both 'account-tag' and 'extended-join' IRCv3 capabilities are ACK'd.
            Set False to restore the legacy WHO query (emergency disable for servers
            where account-tag/extended-join misbehave). The MODE +b ban-list query is
            always suppressed regardless of this flag — nothing reads ban state."""
        ),
    ),
)
