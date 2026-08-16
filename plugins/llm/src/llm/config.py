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
        except Exception as exc:  # noqa: BLE001
            # BadRequestError is litellm's documented "unknown model" signal,
            # but get_llm_provider can also raise other types internally (e.g.
            # a KeyError/ValueError on a malformed provider prefix). Any of
            # those escaping here would abort registry construction / bot.conf
            # loading at startup instead of rejecting one bad value, so map
            # every failure to a clean InvalidRegistryValue.
            suggestions = self._suggest_models(model)
            msg = f"Unknown model: {model!r}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            else:
                msg += " See https://docs.litellm.ai/docs/providers for supported models."
            if not isinstance(exc, litellm.exceptions.BadRequestError):
                msg += f" ({type(exc).__name__})"
            raise registry.InvalidRegistryValue(msg) from exc

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
    conf.registerPlugin("LLM", True)

    print("=" * 60)
    print("LLM Plugin Configuration")
    print("=" * 60)
    print("\nThis plugin provides AI-powered commands using LiteLLM.")
    print("You'll need API keys for the models you want to use.")
    print("\nAPI keys come from the environment, one per provider:")
    print("  XAI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
    print("\nFor more info, see the README.md")
    print("=" * 60)


LLM = conf.registerPlugin("LLM")

# ============================================================================
# System Prompts (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "assistantSystemPrompt",
    registry.String(
        "You are a helpful IRC assistant. This is IRC chat — replies are read "
        "in a terminal client, not rendered as markdown. Keep answers tight: "
        "lead with the answer, aim for one line, never exceed three. Plain "
        "text only — no bold, italics, headings, backticks, code fences, "
        "bullet lists, or [label](url) links. Write URLs bare.",
        _("""System prompt for all assistant work - defines bot personality and constraints."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeSystemPrompt",
    registry.String(
        "You are a helpful code assistant. Explain your code and provide context. "
        "Replies go to IRC: keep prose to one or two short lines, plain text, no "
        "markdown. Code itself is delivered via the generate_code tool's URL — "
        "do not paste the code body into the chat reply.",
        _("""System prompt for code command"""),
    ),
)

# ============================================================================
# Model Configuration (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "assistantModel",
    ValidatedModelName(
        "gemini/gemini-flash-latest",
        _("""Model used for all assistant text+tool work (chat, planner loop,
        memory, reminder parsing, scheduled tasks). Must support
        vision if image URLs in chat should work."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeModel",
    ValidatedModelName(
        "gemini/gemini-flash-latest",
        _("""Model for code generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "imageModel",
    ValidatedModelName(
        "gemini/imagen-4.0-fast-generate-001",
        _("""Model for image generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "searchModel",
    ValidatedModelName(
        "",
        _("""Model for search/fetch tools. Falls back to assistantModel if empty."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseModel",
    ValidatedModelName(
        "",
        _("""Model for verse-mode (in-world roleplay) replies. Falls back to
        assistantModel if empty. Useful when the channel's assistantModel is a
        reasoning model that empirically produces terse output for long-form
        scenes — point this at a non-reasoning model (e.g. gemini-flash-latest)
        to get richer prose for verse turns without changing chat-mode
        behavior."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStyleExemplars",
    registry.Json(
        [],
        _("""Curated "taste" style exemplars — a JSON list of strings injected
        into the verse system prompt to bias prose toward what the channel's
        critics like. Default empty = verse prompt unchanged. Populated offline
        by mining channel logs (plugins/llm/src/llm/verse/taste_mine.py) and
        curated by hand; deploy by editing this value in bot.conf while the bot
        is stopped (a JSON array survives the round-trip cleanly)."""),
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
        # 1, not the 3 this defaulted to while the loop was unreachable on the
        # prod image model. Every rewrite spends a BILLED image call on top of
        # the one already refused, and the recoveries observed on 2026-08-15 all
        # landed on the first changed prompt. A cap of 1 buys the bulk of them
        # for one extra call; 3 mostly buys three refusals of a prompt the
        # provider was never going to pass.
        1,
        _("""Maximum number of automatic prompt rewrites when image generation
        is blocked by content safety filters. Set to 0 to disable. Each retry
        uses the ask model to rewrite the prompt, then spends another billed
        image call."""),
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

conf.registerGlobalValue(
    LLM,
    "memoryMaxPerUser",
    registry.PositiveInteger(50, _("""Maximum number of memories stored per user.""")),
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

conf.registerGlobalValue(
    LLM,
    "memoryPromotionThreshold",
    registry.PositiveInteger(
        2,
        _("""Number of times a candidate fact must be reinforced before it
        becomes a durable memory. 1 disables the candidate stage (every
        extraction is saved immediately, restoring legacy behavior)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "memoryCandidateTTLDays",
    registry.NonNegativeInteger(
        14,
        _("""Days a candidate fact may sit unreinforced before it is pruned.
        Set to 0 to disable TTL pruning."""),
    ),
)

# ============================================================================
# Verse (forest-verse avatar/event subsystem)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "verseEnabled",
    registry.Boolean(
        False,
        _("""Enable the verse avatar/event subsystem in this channel.
        When False, @verse commands are disabled and no verse events are
        recorded regardless of other verse settings."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseTriggerRegex",
    registry.String(
        r"\bverse\b",
        _("""Regex (case-insensitive, re.search) that force-routes a message
        into verse mode when it matches. This is in ADDITION to the always-on
        signal that a message naming a known in-universe entity (character,
        place, or item, other than the speaker's own avatar) triggers verse.
        Empty string disables the keyword trigger, leaving only entity
        references. A malformed regex is ignored (treated as no match).
        Non-triggering messages fall through to the normal chat path."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseEventRetentionDays",
    registry.NonNegativeInteger(
        30,
        _("""Number of days to retain verse events before pruning.
        Older events are removed during housekeeping runs. Must be
        non-negative: a negative value would push the prune cutoff into
        the future and delete every event."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseAutoEntityRetireDays",
    registry.NonNegativeInteger(
        14,
        _("""Days of no reference before auto-created NPCs retire. 0 disables sweep."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseAutoEntityMaxNamesPerCall",
    registry.PositiveInteger(
        8,
        _("""Hard cap on verse_record `actors` array length. The advertised
        tool spec's maxItems is set from this; dispatch enforces. Increase
        past 16 only if your verse routinely cites large casts."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseRosterMaxChars",
    registry.PositiveInteger(
        4000,
        _("""Max characters of the canon-roster block (pinned or author-locked
        entities) injected into every verse system prompt. Canon entities
        beyond the cap are dropped with a (roster truncated) marker."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "verseCompactionModel",
    registry.String(
        "gemini/gemini-flash-lite-latest",
        _("""Model used by the verse compaction job to summarise old events
        into a lore digest. Mirrors the default of the old
        ``loomModel`` key; split out when the loom was removed so compaction
        can be configured independently."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "verseCompactionDailyAt",
    registry.String(
        "03:00",
        _("""Local-time HH:MM at which the daily verse-event-retention
        compaction job fires. Empty or malformed values defer the next
        run by one hour."""),
    ),
)

# ============================================================================
# Storybook (illustrated story tool) — despite the verse* prefix these keys
# also govern the standalone @story command, which needs no verse. The
# prefix is kept because renaming registry keys silently orphans prod
# bot.conf values.
# ============================================================================

conf.registerChannelValue(
    LLM,
    "verseStorybookEnabled",
    registry.Boolean(
        False,
        _("""Expose the verse_storybook tool in this channel. (@story is
        independent of this flag — it is gated like @draw.)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseRoleplayStickyTtlSeconds",
    registry.NonNegativeInteger(
        900,
        _("""Sliding inactivity timeout (seconds) for sticky @rp roleplay mode
        (`@rp on`). Each in-character turn refreshes it; after this much silence
        you drop back to normal chat. 0 = never auto-expire (until `@rp off` or
        a bot restart)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseChatRecordEnabled",
    registry.Boolean(
        False,
        _("""Let canon accrue from ORDINARY chat, not just @rp roleplay turns.
        When on, an opted-in avatar's normal (non-roleplay) chat turn can call
        verse_record to save a genuinely new, durable canon fact; the model is
        nudged to do so sparingly. Off (default) keeps chat-path verse_record
        denied, so canon only grows during explicit roleplay. Guards
        canon-pollution risk behind an opt-in per the cautious verse-write
        rollout pattern."""),
    ),
)

# (Verse engagement measurement, not storybook — filed here historically.)
conf.registerChannelValue(
    LLM,
    "verseReactionCaptureEnabled",
    registry.Boolean(
        True,
        _("""Capture inbound IRCv3 emoji reactions (+draft/react) to the bot's
        verse lines as an offline approval signal (reactions.jsonl). Recency-
        attributed; measurement only — no behaviour change. Kill-switch."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookMaxImages",
    registry.NonNegativeInteger(
        5,
        _("""Max illustrations drawn per story (verse_storybook and @story)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStoryAmbientMaxImages",
    registry.NonNegativeInteger(
        1,
        _("""Image cap for an AMBIENT verse story — a plain narrative mention or a
        recount question ("what have the lads done today"). These default to
        prose-first tales; the odd hero image is fine but they don't want the
        full illustrated storybook. An explicit "illustrate" cue, @story, or
        the verse_storybook tool use the larger verseStorybookMaxImages instead.
        0 = text only."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookMaxPerTurn",
    registry.NonNegativeInteger(
        1,
        _("""Max verse_storybook calls honored per completion."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookCooldownSeconds",
    registry.NonNegativeInteger(
        300,
        _("""Per-account cooldown between stories (shared by verse_storybook
        and @story)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookDailyImageCap",
    registry.NonNegativeInteger(
        30,
        _("""Per-account daily ceiling on storybook images (enforcement
        deferred; cooldown + per-turn cap bound spend today)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookMaxChars",
    registry.NonNegativeInteger(
        6000,
        _("""Story length cap before image embedding (verse_storybook and
        @story)."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "verseStorybookImageTimeout",
    registry.NonNegativeInteger(
        45,
        _("""Per-image timeout (seconds) for storybook draws (verse_storybook
        and @story)."""),
    ),
)

# (Verse retention compaction, not storybook — filed here historically.)
conf.registerGlobalValue(
    LLM,
    "verseCompactionMinKeepEvents",
    registry.NonNegativeInteger(
        20,
        _("""Floor on total event count below which a verse is left
        alone by compaction. Prevents thrashing small verses."""),
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
# HTTP Server Settings (for code/image/answer output)
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
    "imageUploadUrl",
    registry.String(
        "",
        _("""Endpoint of an external image host for generated images. Expects a
        multipart POST on the "images[]" field and a JSON reply shaped like
        {"results": [{"success": true, "filePath": "/img/x.png"}]} — for example
        https://paste.boxlabs.uk/img/. If empty, images are stored locally under
        httpRoot and served from httpUrlBase. Any upload failure falls back to
        local storage, so this is safe to leave pointed at a flaky host."""),
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

conf.registerChannelValue(
    LLM,
    "longReplyTeaserMaxChars",
    registry.PositiveInteger(
        220,
        _("""Maximum characters for the one-line teaser shown in IRC when a
        long reply is linked to the full HTML answer."""),
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
        [".", "@"],
        _("""Command prefixes to sanitize in output. Lines starting with these
        are prefixed with a space to prevent IRC command injection.
        Default: . @"""),
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


def _register_rate_limit_block(
    command: str,
    *,
    counts: tuple[int, int, int],  # (registered, trusted, unreg)
    windows: tuple[int, int, int],  # (registered, trusted, unreg)
) -> None:
    tiers: tuple[tuple[str, str], ...] = (
        ("", "registered tier"),
        ("Trusted", "trusted tier"),
        ("Unreg", "unregistered tier"),
    )
    for (tier, label), count, window in zip(tiers, counts, windows, strict=True):
        conf.registerGlobalValue(
            LLM,
            f"{command}{tier}RateLimitCount",
            registry.NonNegativeInteger(
                count,
                _(
                    f"Max {command} requests per {label} within "
                    f"{command}{tier}RateLimitWindow seconds. "
                    "Set to 0 to disable rate limiting for this tier."
                ),
            ),
        )
        conf.registerGlobalValue(
            LLM,
            f"{command}{tier}RateLimitWindow",
            registry.PositiveInteger(
                window,
                _(f"Time window in seconds for counting {command} requests ({label})."),
            ),
        )


# --- ask (cheapest) ---
_register_rate_limit_block(
    "ask",
    counts=(15, 15, 15),
    windows=(60, 60, 60),
)

# --- code ---
_register_rate_limit_block(
    "code",
    counts=(10, 0, 2),
    windows=(60, 60, 60),
)

# --- draw (expensive) ---
# Unregistered is the STRICTEST tier: 1 image/hour, not 0. A count of 0 means
# "rate limiting disabled" (see _is_rate_limited), so 0 would give accountless
# users UNLIMITED access to the most expensive command — the opposite of the
# intended tier ordering. Keep it a small positive cap.
_register_rate_limit_block(
    "draw",
    counts=(2, 5, 1),
    windows=(300, 60, 3600),
)

# --- story (expensive, illustrated storybook) ---
_register_rate_limit_block(
    "story",
    counts=(2, 5, 1),
    windows=(300, 60, 3600),
)

# ============================================================================
# Assistant Tool-Calling Backend (shared by @ask, @code, @draw, invalidCommand)
# ============================================================================

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

conf.registerGlobalValue(
    LLM,
    "maxConcurrentLLMCalls",
    registry.PositiveInteger(
        16,
        _("""Maximum number of simultaneous outbound LLM calls (across the
        command path and background work — memory extraction, watch-mode
        reminders, scheduled tasks). Lower this on small hosts or when the
        provider rate-limits aggressively."""),
    ),
)

# ============================================================================
# Limnoria tool bridge (Phase 1)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "bridgeEnabled",
    registry.Boolean(
        False,
        _("""When True, expose loaded Limnoria plugin commands to the LLM
        as a tool, restricted by bridgeAllowedPlugins and Limnoria's
        capability system. Default off."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeAllowedPlugins",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Space-separated list of Limnoria plugin names whose commands
        the LLM may call when bridgeEnabled is True. When empty (the
        default) the bridge falls back to a curated read-safe set: Misc
        Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG
        — write commands inside those plugins stay hidden until
        bridgeAllowMutating is True. Set this to a non-empty list to
        override the curated set with your own selection. Set
        bridgeEnabled False to disable the bridge entirely."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeAllowMutating",
    registry.Boolean(
        False,
        _("""When True, the Limnoria bridge exposes commands that modify
        persistent state (sending notes, registering feeds, mutating karma,
        etc.). When False (the default), only read-only commands are exposed
        — write commands are hidden from the LLM's tool description and any
        attempt to dispatch one returns an error envelope.

        Per-command classification lives in MUTATING_COMMANDS in
        plugins/llm/src/llm/limnoria_bridge.py."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "pendingTasksEnabled",
    registry.Boolean(
        False,
        _("""When True, chat completions in this channel advertise the
        reminder/scheduled-task tools (set_reminder, schedule_llm_task,
        list_pending_tasks, cancel_pending_task, cancel_all_pending_tasks)
        and their operating rules, so natural-language scheduling via @ask
        works ('remind me in 10 minutes', 'every hour post X').

        Off by default: the five tool schemas plus the prompt rules cost
        roughly 1,100 prompt tokens on EVERY completion, whether or not
        anyone schedules anything. The @remind command and the firing of
        already-created reminders/tasks work regardless of this setting —
        it only gates whether the model can create/list/cancel them."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeScheduledTaskLimit",
    registry.NonNegativeInteger(
        5,
        _("""Maximum number of active LLM-scheduled tasks per creator in this
        channel. Enforced at create time by the schedule_llm_task tool. Set to
        0 to disable scheduling entirely.

        Each fire is checked against the user's ask rate limit but consumes no
        slot (plugin._unattended_ask_rate_limited peeks with record=False): a
        fire is skipped when the user has already maxed their own interactive
        use. This value caps the *number* of pending schedules, not their
        cumulative cost. The bridge* prefix is a leftover from the Phase 2
        bridge work that introduced scheduling; a fire runs through
        _run_unattended_assistant, which wires no bridge tools, so a scheduled
        task cannot execute Limnoria commands."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeDebugInChannel",
    registry.Boolean(
        False,
        _("""When True, append a one-line debug footer to LLM replies listing
        every Limnoria bridge tool call made during the turn (plugin, command,
        args, ok/err). Off by default; useful to confirm the LLM is actually
        using bridged commands."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "statusPageUrl",
    registry.String(
        "https://status.claude.com",
        _("""Base URL of an Atlassian Statuspage-hosted service status page
        (no trailing path). The bot polls {url}/api/v2/summary.json to answer
        status questions and to announce new incidents. Set to the empty
        string to disable status awareness entirely."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "statusAnnounce",
    registry.Boolean(
        False,
        _("""Announce status-page incidents in this channel as they open and
        again as they resolve. Off by default. When enabling this, remove the
        equivalent RSS feed announcement for the channel first, or every
        incident is reported twice."""),
    ),
)
