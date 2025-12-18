"""Configuration for LLM plugin."""

from __future__ import annotations

import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("LLM")


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
        "Use markdown formatting for code blocks.",
        _("""System prompt for code command"""),
    ),
)

# ============================================================================
# Model Configuration (channel-specific with global defaults)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "askModel",
    registry.String(
        "gemini/gemini-flash-latest",
        _("""Model for ask command (supports vision)"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "codeModel",
    registry.String(
        "gemini/gemini-1.5-flash",
        _("""Model for code generation"""),
    ),
)

conf.registerChannelValue(
    LLM,
    "drawModel",
    registry.String(
        "vertex_ai/imagen-4.0-generate-001",
        _("""Model for image generation"""),
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

conf.registerChannelValue(
    LLM,
    "codeThreshold",
    registry.PositiveInteger(
        20,
        _("""Line count threshold - use HTTP link if code exceeds this"""),
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
        30,
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
