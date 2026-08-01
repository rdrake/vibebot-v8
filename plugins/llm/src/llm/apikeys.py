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

import logging
import os
from collections.abc import Mapping

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
    try:
        if not model or not model.strip():
            return ""
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


# Credentials do not all end in _API_KEY. A provider whose secret is named
# differently would otherwise sit outside redaction entirely.
SECRET_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# Real keys are far longer (Gemini 39, xAI ~84). The floor stops a short junk
# value such as FOO_API_KEY=disabled turning redaction into find-and-replace on
# a common word across every log line.
MIN_REDACTABLE_LEN = 16

REDACTED = "[REDACTED]"


def _env_snapshot() -> dict[str, str]:
    """Best-effort atomic-ish snapshot of ``os.environ``.

    ``os.environ`` is a ``MutableMapping``, not a real dict: enumerating it
    snapshots the keys (``list(self._data)``, genuinely atomic) but then looks
    each one up in a *separate* step. A concurrent ``setenv``/``delenv``
    between those two steps raises ``KeyError`` — ``list(os.environ.items())``
    does not protect against this, because the race is in the per-key lookup,
    not in the outer iteration. Production never mutates the environment, but
    the test suite does constantly, and this filter runs on every log record,
    so it must not raise. The race window is a handful of bytecodes; a few
    retries resolve it in practice, and falling back to ``{}`` after that
    keeps the "never raise" guarantee even under pathological contention.
    """
    for _ in range(5):
        try:
            return dict(os.environ)
        except KeyError:
            continue
    return {}


def _secret_items() -> list[tuple[str, str]]:
    """(name, stripped value) for every environment secret worth redacting."""
    items = []
    for name, raw in _env_snapshot().items():
        if not name.upper().endswith(SECRET_SUFFIXES):
            continue
        value = raw.strip()
        if len(value) >= MIN_REDACTABLE_LEN:
            items.append((name, value))
    return items


def known_secret_values() -> set[str]:
    """Every environment value that must never appear in output."""
    return {value for _name, value in _secret_items()}


def secret_var_names() -> list[str]:
    """Sorted names of the variables redaction covers. Names only, never values."""
    return sorted(name for name, _value in _secret_items())


def scrub(text: str | None) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``."""
    if not text:
        return ""
    result = str(text)
    for secret in known_secret_values():
        result = result.replace(secret, REDACTED)
    return result


class SecretFilter(logging.Filter):
    """Strip API keys from log records before they are formatted.

    Covers message, arguments, traceback and stack info, because supybot's
    ``Logger.exception`` uses all of them: it writes the raw traceback and calls
    ``collect_extra_debug_data()``, a repr of every frame local and every
    attribute of ``self``.

    Arguments are scrubbed by their ``str()`` value rather than by type: an
    exception object carries the provider's error body, which routinely echoes
    the submitted key.

    Never drops a record, and never breaks formatting — redaction must not cost
    observability. Value replacement is defence in depth, not a boundary: it
    cannot catch an encoded or truncated credential.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not known_secret_values():
            return True
        record.msg = scrub(str(record.msg)) if record.msg is not None else record.msg
        record.args = self._scrub_args(record.args)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True

    @staticmethod
    def _scrub_args(
        args: tuple[object, ...] | Mapping[str, object] | None,
    ) -> tuple[object, ...] | Mapping[str, object] | None:
        """Scrub arguments without changing their shape.

        A lone Mapping must stay a Mapping: logging unwraps it into ``args``,
        and turning it into a tuple of keys makes ``getMessage()`` raise
        "format requires a mapping".
        """
        if not args:
            return args
        if isinstance(args, Mapping):
            return {str(key): scrub(str(value)) for key, value in args.items()}
        return tuple(scrub(str(arg)) if not isinstance(arg, (int, float)) else arg for arg in args)


def install_secret_filter() -> int:
    """Attach ``SecretFilter`` to every output handler, idempotently.

    Handlers, not loggers: a logger's filters run only for records that
    originate on it, and propagation to an ancestor runs the ancestor's
    *handlers*, not its *filters*. This plugin logs through at least ten loggers
    across two hierarchies (``supybot.plugins.LLM.*`` and ``llm.verse.*``), so
    per-logger installation would cover whichever two we happened to name.

    Handlers created later — supybot adds per-plugin file handlers when
    ``individualLogfiles`` is true; prod has it false — are not covered. Calling
    this again picks them up.

    Returns the number of handlers newly filtered, for the startup log line.
    """
    installed = 0
    targets = [logging.getLogger(), logging.getLogger("supybot"), logging.getLogger("llm")]
    for logger in targets:
        for handler in logger.handlers:
            if not any(isinstance(existing, SecretFilter) for existing in handler.filters):
                handler.addFilter(SecretFilter())
                installed += 1
    return installed
