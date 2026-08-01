"""Provider-scoped API key resolution.

A key is a property of which provider is being paid, not of which channel or
which role asked. This module maps a model identifier to its provider and then
to a single environment variable, so the key is always anchored to the model
actually being called.

Pure and dependency-light on purpose: no plugin, no service, no Limnoria
registry. That keeps it unit-testable and keeps key resolution out of
``service.py``.
"""

from __future__ import annotations

import os

import litellm

# The providers this deployment pays for directly. Anything else LiteLLM
# recognises — vertex_ai, openrouter, azure, bedrock — resolves to None so
# LiteLLM uses its own native credential mechanism (ADC, IAM, its own env vars).
# Narrowing that to an allowlist would turn a multi-provider plugin into a
# four-provider one.
PROVIDER_ENV_VARS: dict[str, str] = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def provider_of(model: str) -> str:
    """Return the LiteLLM provider for ``model``, or "" if unresolvable.

    Uses ``litellm.get_llm_provider`` rather than splitting on "/" because
    unprefixed model names are legal — LiteLLM resolves ``gpt-4`` and
    ``dall-e-3`` to openai — and this plugin's config validator accepts anything
    LiteLLM accepts.

    Never raises. LiteLLM raises ``BadRequestError`` for names it cannot place,
    and callers sit inside failure handlers where a new exception type would
    surface as an unhandled crash rather than a configuration error.
    """
    if not model or not model.strip():
        return ""
    try:
        return str(litellm.get_llm_provider(model)[1])
    except Exception:  # noqa: BLE001 — an unplaceable model is a config error, not a crash
        return ""


def env_var_for(model: str) -> str | None:
    """Environment variable name ``model`` needs, or None if unmanaged.

    Returns the *name*, never the value, so callers can tell an operator which
    variable to set.
    """
    return PROVIDER_ENV_VARS.get(provider_of(model))


def is_managed(model: str) -> bool:
    """True if this deployment supplies ``model``'s credential directly.

    False means "pass None and let LiteLLM resolve" — the path that keeps
    vertex_ai (ADC), openrouter, azure and bedrock working.
    """
    return provider_of(model) in PROVIDER_ENV_VARS


def api_key_for(model: str) -> str | None:
    """Configured API key for ``model``'s provider, or None.

    Read from the environment on every call rather than cached, so the value
    redaction scrubs can never diverge from the value actually sent.
    """
    name = env_var_for(model)
    if not name:
        return None
    return os.environ.get(name, "").strip() or None
