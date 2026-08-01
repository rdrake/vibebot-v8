# Provider-scoped API keys implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace twelve per-channel, per-role API-key registry entries with one environment variable per provider, resolved from the model being called.

**Architecture:** A new pure module `llm/apikeys.py` maps a model to its provider via `litellm.get_llm_provider()` and then to a single environment variable. Every key lookup in the plugin resolves through it, which anchors the key to the model and eliminates three live cross-provider mismatches. A `logging.Filter` scrubs secrets from log records, replacing hand-applied redaction that the exception path defeats. The four `*ApiKey` registry settings are then deleted outright.

**Tech stack:** Python 3.12-3.14, Limnoria (supybot), LiteLLM, pytest, uv, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-08-01-provider-scoped-api-keys-design.md`

## Global constraints

- Target repo root: `/Users/rdrake/workspace/afternet/vibebot-v8`. Plugin source: `plugins/llm/src/llm/`. Tests: `plugins/llm/tests/`.
- Run tests with `make test` (or `uv run pytest plugins/llm/tests/...` for single files). Full gate: `make check`.
- Coverage gate is `fail_under = 93` (`pyproject.toml:75`), currently at 94%. `apikeys.py` must land near 100%.
- Lint and typecheck must pass: `make lint && make typecheck`. Run from the repo root — a hook runs these after every edit and will fail from any other directory.
- Never log, print, or assert on a real API key value. Tests use fake values only.
- Commit after each task. Direct commits to `main` are fine in this repo; do not open a PR.
- Provider environment variables, exact spelling: `XAI_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.
- `MIN_REDACTABLE_LEN = 16`. `SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")`.
- Do NOT add `vertex_ai` support. It authenticates by service account; it is deliberately out of scope.
- Every task's steps assume the previous tasks are committed.

---

### Task 1: `apikeys.py` — model-to-key resolution

**Files:**
- Create: `plugins/llm/src/llm/apikeys.py`
- Test: `plugins/llm/tests/test_apikeys.py`

**Interfaces:**
- Consumes: nothing (leaf module — imports only `os` and `litellm`).
- Produces:
  - `PROVIDER_ENV_VARS: dict[str, str]`
  - `provider_of(model: str) -> str` — `""` when unresolvable
  - `env_var_for(model: str) -> str | None` — variable *name*, not value
  - `api_key_for(model: str) -> str | None` — variable *value*

- [ ] **Step 1: Write the failing test**

Create `plugins/llm/tests/test_apikeys.py`:

```python
"""Tests for provider-scoped API key resolution."""

from __future__ import annotations

import pytest

from llm import apikeys


class TestProviderOf:
    """provider_of maps a model identifier to a LiteLLM provider name."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("xai/grok-4.3", "xai"),
            ("gemini/gemini-3-flash-preview", "gemini"),
            ("openai/gpt-5.2", "openai"),
            ("anthropic/claude-3-opus", "anthropic"),
            # Unprefixed names are legal and widespread in this test suite.
            ("gpt-4", "openai"),
            ("dall-e-3", "openai"),
        ],
    )
    def test_known_models_resolve(self, model: str, expected: str) -> None:
        """GIVEN a model LiteLLM recognises WHEN provider_of THEN its provider."""
        assert apikeys.provider_of(model) == expected

    @pytest.mark.parametrize("model", ["", "   ", "not-a-real-model-xyz", "claude-3-opus"])
    def test_unresolvable_models_return_empty(self, model: str) -> None:
        """GIVEN a model LiteLLM rejects WHEN provider_of THEN "" and no raise.

        LiteLLM raises BadRequestError for these; key resolution runs on paths
        with their own error handling and must never add a new exception type.
        """
        assert apikeys.provider_of(model) == ""

    def test_provider_is_lowercased(self) -> None:
        """GIVEN a mixed-case prefix WHEN provider_of THEN lowercase."""
        assert apikeys.provider_of("XAI/grok-4.3") == "xai"


class TestEnvVarFor:
    """env_var_for reports which variable a model needs, for error messages."""

    def test_mapped_provider(self) -> None:
        """GIVEN an xai model WHEN env_var_for THEN XAI_API_KEY."""
        assert apikeys.env_var_for("xai/grok-4.3") == "XAI_API_KEY"

    def test_unmapped_provider(self) -> None:
        """GIVEN a vertex_ai model WHEN env_var_for THEN None (unsupported)."""
        assert apikeys.env_var_for("vertex_ai/imagen-4.0-generate-001") is None

    def test_unresolvable_model(self) -> None:
        """GIVEN an unresolvable model WHEN env_var_for THEN None."""
        assert apikeys.env_var_for("not-a-real-model-xyz") is None


class TestApiKeyFor:
    """api_key_for reads the provider's variable from the environment."""

    def test_returns_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY set WHEN api_key_for an xai model THEN that value."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"

    def test_missing_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY unset WHEN api_key_for THEN None."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_var_returns_none(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """GIVEN a blank XAI_API_KEY WHEN api_key_for THEN None, not "".

        An empty string would read as "configured" at the guards and then fail
        at the provider instead of at the config error.
        """
        monkeypatch.setenv("XAI_API_KEY", value)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    def test_value_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a value with a trailing newline WHEN api_key_for THEN stripped.

        docker --env-file and hand-edited files both produce these.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "  AIza-fake-value-for-tests\n")
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "AIza-fake-value-for-tests"

    def test_unmapped_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a vertex_ai model WHEN api_key_for THEN None (unsupported)."""
        monkeypatch.setenv("VERTEX_AI_API_KEY", "should-not-be-used-anywhere")
        assert apikeys.api_key_for("vertex_ai/imagen-4.0-generate-001") is None

    def test_provider_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN both vars set WHEN resolving each THEN no cross-provider bleed."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "gemini-fake-value-for-tests"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/rdrake/workspace/afternet/vibebot-v8
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'llm.apikeys'`.

- [ ] **Step 3: Write the implementation**

Create `plugins/llm/src/llm/apikeys.py`:

```python
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

# One variable per provider. LiteLLM uses these same names natively, so a value
# set here also satisfies any code path that bypasses this module.
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
    ``dall-e-3`` to openai — and this plugin's own config validator
    (``config.ValidatedModelName``) accepts anything LiteLLM accepts.

    Never raises. LiteLLM raises ``BadRequestError`` for names it cannot place,
    and every caller sits on a path with its own error handling; a new exception
    type here would surface as an unhandled failure rather than a config error.
    """
    if not model or not model.strip():
        return ""
    try:
        return str(litellm.get_llm_provider(model)[1]).lower()
    except Exception:  # noqa: BLE001 — unresolvable model is a config error, not a crash
        return ""


def env_var_for(model: str) -> str | None:
    """Return the environment variable name ``model`` needs, or None.

    Returns the *name*, never the value. Callers use it to tell an operator
    which variable to set.
    """
    return PROVIDER_ENV_VARS.get(provider_of(model))


def api_key_for(model: str) -> str | None:
    """Return the configured API key for ``model``'s provider, or None.

    Read from the environment on every call rather than cached at import, so the
    value redaction scrubs can never diverge from the value actually sent.
    """
    name = env_var_for(model)
    if not name:
        return None
    return os.environ.get(name, "").strip() or None
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: all PASS. If `anthropic/claude-3-opus` fails, check the installed LiteLLM's provider map and adjust the parametrized case to a model it recognises — do not change `provider_of`.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/apikeys.py plugins/llm/tests/test_apikeys.py
git commit -m "feat(apikeys): resolve API keys from the model's provider"
```

---

### Task 2: `apikeys.py` — secret discovery and `SecretFilter`

**Files:**
- Modify: `plugins/llm/src/llm/apikeys.py`
- Test: `plugins/llm/tests/test_apikeys.py`

**Interfaces:**
- Consumes: `PROVIDER_ENV_VARS` from Task 1.
- Produces:
  - `SECRET_SUFFIXES: tuple[str, ...]`, `MIN_REDACTABLE_LEN: int`
  - `known_secret_values() -> set[str]`
  - `scrub(text: str) -> str`
  - `SecretFilter` (a `logging.Filter` subclass)

**Why a filter and not more `_sanitize` calls:** supybot's `Logger.exception` (`supybot/log.py:76-88`) writes the raw traceback *and* calls `utils.python.collect_extra_debug_data()`, which `repr()`s every local in every frame plus every attribute of `self`. Hand-applied redaction cannot cover that. The debug dump arrives as a log record *argument*, and the traceback arrives as `exc_info` — so the filter must handle `record.msg`, `record.args`, and `record.exc_info`, not just the message.

- [ ] **Step 1: Write the failing test**

Append to `plugins/llm/tests/test_apikeys.py`:

```python
import logging


class TestKnownSecretValues:
    """known_secret_values collects what redaction must scrub."""

    def test_collects_provider_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN provider vars set WHEN known_secret_values THEN both included."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-value-long-enough")
        values = apikeys.known_secret_values()
        assert "xai-fake-value-long-enough" in values
        assert "gemini-fake-value-long-enough" in values

    @pytest.mark.parametrize(
        "name",
        ["SOME_API_KEY", "HF_TOKEN", "CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS"],
    )
    def test_collects_by_suffix(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        """GIVEN a var with a secret suffix WHEN collected THEN included.

        Adjacent credentials do not all end in _API_KEY; missing them would put
        them outside redaction entirely.
        """
        monkeypatch.setenv(name, "some-credential-value-long-enough")
        assert "some-credential-value-long-enough" in apikeys.known_secret_values()

    def test_ignores_unrelated_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a non-secret var WHEN collected THEN excluded."""
        monkeypatch.setenv("EDITOR", "a-value-that-is-long-enough")
        assert "a-value-that-is-long-enough" not in apikeys.known_secret_values()

    def test_ignores_short_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a short secret-suffixed var WHEN collected THEN excluded.

        Redacting a short common word would replace it everywhere in every log
        line, destroying the operator's ability to read logs at all.
        """
        monkeypatch.setenv("FOO_API_KEY", "disabled")
        assert "disabled" not in apikeys.known_secret_values()

    def test_ignores_blank_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an empty secret var WHEN collected THEN no empty string.

        An empty entry would make scrub() insert [REDACTED] between every
        character.
        """
        monkeypatch.setenv("BAR_API_KEY", "")
        assert "" not in apikeys.known_secret_values()


class TestScrub:
    """scrub replaces known secret values with a marker."""

    def test_replaces_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN text containing the key WHEN scrub THEN [REDACTED]."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        text = "AuthenticationError: bad key xai-fake-value-long-enough for model"
        assert apikeys.scrub(text) == "AuthenticationError: bad key [REDACTED] for model"

    def test_passthrough_without_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no secret vars WHEN scrub THEN text unchanged."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert apikeys.scrub("nothing to hide here") == "nothing to hide here"


class TestSecretFilter:
    """SecretFilter scrubs message, args, and formatted tracebacks."""

    @staticmethod
    def _record(msg: str, args: object = None, exc_info: object = None) -> logging.LogRecord:
        return logging.LogRecord(
            name="LLM.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=args,
            exc_info=exc_info,
        )

    def test_scrubs_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a key in record.msg WHEN filtered THEN redacted."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        record = self._record("call failed with xai-fake-value-long-enough")
        assert apikeys.SecretFilter().filter(record) is True
        assert "xai-fake-value-long-enough" not in record.getMessage()

    def test_scrubs_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a key in record.args WHEN filtered THEN redacted.

        supybot's Logger.exception emits collect_extra_debug_data() — a repr of
        every frame local — as a log argument, so args is the leak path that
        matters most.
        """
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        record = self._record("%s", ("api_key='xai-fake-value-long-enough'",))
        apikeys.SecretFilter().filter(record)
        assert "xai-fake-value-long-enough" not in record.getMessage()

    def test_scrubs_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a key in the exception text WHEN filtered THEN redacted."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        try:
            raise ValueError("auth failed for xai-fake-value-long-enough")
        except ValueError:
            import sys

            record = self._record("boom", exc_info=sys.exc_info())
        apikeys.SecretFilter().filter(record)
        assert record.exc_text is not None
        assert "xai-fake-value-long-enough" not in record.exc_text

    def test_never_drops_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no secrets configured WHEN filtered THEN record still emitted."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert apikeys.SecretFilter().filter(self._record("plain message")) is True

    def test_non_string_args_survive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN numeric args WHEN filtered THEN formatting still works."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-long-enough")
        record = self._record("took %d ms", (42,))
        apikeys.SecretFilter().filter(record)
        assert record.getMessage() == "took 42 ms"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -k "Secret or Scrub" -v
```

Expected: FAIL with `AttributeError: module 'llm.apikeys' has no attribute 'known_secret_values'`.

- [ ] **Step 3: Write the implementation**

Append to `plugins/llm/src/llm/apikeys.py`:

```python
import logging

# Credentials do not all end in _API_KEY. A provider added later whose secret is
# named differently would otherwise sit outside redaction entirely.
SECRET_SUFFIXES: tuple[str, ...] = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")

# Real keys are far longer (Gemini 39, xAI ~84). The floor stops a short junk
# value such as FOO_API_KEY=disabled turning redaction into find-and-replace on
# a common word across every log line.
MIN_REDACTABLE_LEN = 16

REDACTED = "[REDACTED]"


def known_secret_values() -> set[str]:
    """Every environment value that must never appear in output."""
    return {
        value
        for name, raw in os.environ.items()
        if name.upper().endswith(SECRET_SUFFIXES)
        for value in (raw.strip(),)
        if len(value) >= MIN_REDACTABLE_LEN
    }


def secret_var_names() -> list[str]:
    """Sorted names of the variables redaction covers. Names only, never values."""
    return sorted(
        name
        for name, raw in os.environ.items()
        if name.upper().endswith(SECRET_SUFFIXES) and len(raw.strip()) >= MIN_REDACTABLE_LEN
    )


def scrub(text: str) -> str:
    """Replace every known secret value in ``text`` with ``[REDACTED]``."""
    if not text:
        return ""
    result = str(text)
    for secret in known_secret_values():
        result = result.replace(secret, REDACTED)
    return result


class SecretFilter(logging.Filter):
    """Strip API keys from log records before they are formatted.

    Covers three surfaces, because supybot's ``Logger.exception`` uses all
    three: the message itself, the arguments (``collect_extra_debug_data``
    emits a repr of every frame local as an argument), and the exception
    traceback. The traceback is materialised here so the scrubbed text is what
    the handler formats later.

    Never drops a record — redaction must not cost observability.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = known_secret_values()
        if not secrets:
            return True
        if isinstance(record.msg, str):
            record.msg = scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: scrub(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    scrub(arg) if isinstance(arg, str) else arg for arg in record.args
                )
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_text = scrub(record.exc_text)
        return True
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/apikeys.py plugins/llm/tests/test_apikeys.py
git commit -m "feat(apikeys): scrub secrets from log records via a logging filter"
```

---

### Task 3: Test-harness environment isolation

**Files:**
- Modify: `plugins/llm/tests/conftest.py:28` (`TEST_MODEL`), `:403-474` (key stubs, `imageModel`)
- Test: the whole suite is the test here.

**Interfaces:**
- Consumes: `apikeys.SECRET_SUFFIXES`.
- Produces: an autouse fixture guaranteeing a known environment for every test, and provider-prefixed fixture models.

**Why this comes before any wiring:** `litellm/__init__.py:27` calls `load_dotenv()` at import time and `.env` is gitignored, so importing `llm.service` injects a developer's **real** keys into `os.environ` before any test runs. Without a scrubber, tests would pass locally for the wrong reason, and a redaction test could never assert an exact set. This task is a no-op behaviourally — it only makes the later tasks provable.

- [ ] **Step 1: Add the autouse environment fixture**

In `plugins/llm/tests/conftest.py`, after the existing imports:

```python
import os

import pytest

from llm import apikeys

# Fake values, long enough to clear apikeys.MIN_REDACTABLE_LEN.
FAKE_PROVIDER_KEYS = {
    "XAI_API_KEY": "xai-fake-key-for-tests-0000",
    "GEMINI_API_KEY": "AIza-fake-key-for-tests-0000",
    "OPENAI_API_KEY": "sk-fake-key-for-tests-0000",
    "ANTHROPIC_API_KEY": "sk-ant-fake-key-for-tests-0000",
}


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test a known, fake set of provider credentials.

    LiteLLM calls load_dotenv() at import, so a developer's real .env leaks into
    os.environ before collection. Without this, tests can pass locally against
    real keys and fail in CI — or worse, reach the network.
    """
    for name in list(os.environ):
        if name.upper().endswith(apikeys.SECRET_SUFFIXES):
            monkeypatch.delenv(name, raising=False)
    for name, value in FAKE_PROVIDER_KEYS.items():
        monkeypatch.setenv(name, value)
```

- [ ] **Step 2: Give the fixture models a resolvable provider**

In `plugins/llm/tests/conftest.py`:
- `:28` — leave `TEST_MODEL = "gpt-4"` as is. It resolves to openai through LiteLLM, which is correct and exercises the unprefixed path that ~101 test occurrences rely on. Changing it would be churn with no benefit.
- `:405` — leave `"imageModel": "dall-e-3"` as is, for the same reason (resolves to openai).
- Verify both assumptions explicitly rather than trusting this note:

```bash
uv run python -c "
from llm import apikeys
for m in ('gpt-4', 'dall-e-3'):
    print(m, '->', apikeys.provider_of(m), apikeys.env_var_for(m))
"
```

Expected: `gpt-4 -> openai OPENAI_API_KEY` and `dall-e-3 -> openai OPENAI_API_KEY`. If either prints an empty provider, prefix the fixture value with `openai/` and re-run the suite.

- [ ] **Step 3: Run the full suite to confirm nothing regressed**

```bash
make test
```

Expected: 2549 passed. This task changes no production code, so any failure is a fixture bug — most likely a test that asserted on an environment variable the scrubber now deletes. Fix by setting the variable inside that test.

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/tests/conftest.py
git commit -m "test: isolate provider credentials from the developer environment"
```

---

### Task 4: Resolve keys from the model in `service.py`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — sites `2637`, `2718`, `3182-3184`, `3602`, `3672`, `3787`, `4090`, `4175`, `4325`, `5117`, `6124`, `6195`
- Test: `plugins/llm/tests/test_service_completion.py`, `test_service_memory.py`, `test_provider_edge_cases.py`

**Interfaces:**
- Consumes: `apikeys.api_key_for`, `apikeys.env_var_for`.
- Produces: `LLMService._api_key_for(self, model: str) -> str | None` and `LLMService._missing_key_error(self, model: str) -> str`.

**Line numbers shift as you edit.** Work bottom-up (highest line number first) or re-grep between edits: `grep -n "ApiKey" plugins/llm/src/llm/service.py`.

- [ ] **Step 1: Write the failing test**

Add to `plugins/llm/tests/test_provider_edge_cases.py`:

```python
class TestModelAnchoredKeyResolution:
    """The key sent to a provider is the one that provider's variable holds."""

    def test_xai_model_gets_xai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an xai model WHEN completing THEN XAI_API_KEY is sent."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service(assistantModel="xai/grok-4.3")
        assert service._api_key_for("xai/grok-4.3") == "xai-fake-key-for-tests-0000"

    def test_gemini_model_gets_gemini_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a gemini model WHEN resolving THEN GEMINI_API_KEY, not the xai one."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        service, _ = make_service()
        assert service._api_key_for("gemini/gemini-3-flash-preview") == "AIza-fake-key-for-tests-0000"

    def test_missing_key_error_names_the_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no key WHEN building the error THEN it names provider and variable.

        A bare "API key not configured" sends an operator hunting a key that is
        set, when the real problem is the model's provider.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service()
        message = service._missing_key_error("xai/grok-4.3")
        assert "xai" in message
        assert "XAI_API_KEY" in message

    def test_unsupported_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a vertex_ai model WHEN building the error THEN it says unsupported."""
        service, _ = make_service()
        message = service._missing_key_error("vertex_ai/imagen-4.0-generate-001")
        assert "vertex_ai" in message

    def test_error_never_contains_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a configured key WHEN building any error THEN the value is absent."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service()
        assert "xai-fake-key-for-tests-0000" not in service._missing_key_error("xai/grok-4.3")
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest plugins/llm/tests/test_provider_edge_cases.py -k ModelAnchored -v
```

Expected: FAIL, `AttributeError: 'LLMService' object has no attribute '_api_key_for'`.

- [ ] **Step 3: Add the helpers**

In `service.py`, add these next to `_API_KEY_NAMES` (`1186-1191`). Do **not** delete `_API_KEY_NAMES` here — `_configured_api_keys` still uses it until Task 6 removes both together:

```python
    def _api_key_for(self, model: str) -> str | None:
        """API key for ``model``'s provider, or None if unset or unsupported."""
        return apikeys.api_key_for(model)

    def _missing_key_error(self, model: str) -> str:
        """Operator-facing message naming the provider and the variable wanted."""
        provider = apikeys.provider_of(model) or "unknown"
        var = apikeys.env_var_for(model)
        if var:
            return _("Error: no API key configured for provider '%s' (set %s)") % (provider, var)
        return _("Error: no API key configured for provider '%s' (unsupported)") % provider
```

Add `from llm import apikeys` to the imports. Leave `_configured_api_keys` and `_sanitize` alone for now — Task 6 replaces them, and deleting them here would break redaction between commits.

- [ ] **Step 4: Convert the eight guarded sites**

For each, fetch the model first, then resolve. The pattern:

```python
model = self.plugin.registryValue("assistantModel", target)
api_key = self._api_key_for(model)
if not api_key:
    # ... existing failure return, with reason/content replaced by
    #     self._missing_key_error(model)
```

Sites and the model each one must use:

| Site | Model source |
|---|---|
| `2637` (pending-task retry) | `task.model` |
| `2718` (pending draw retry) | `task.model` |
| `3029` (`assistant_completion` command path) | the `model_name` registry value already selected above it |
| `3602` (reminder parse) | `registryValue("assistantModel", target)` — hoist above the guard |
| `3788` (ask helper) | `registryValue("assistantModel", target)` — hoist above the guard |
| `4091` (image prompt rewrite) | `registryValue("assistantModel", target)` — hoist above the guard |
| `4328` (`assistant_completion` profile path) | `model` from `4324`, already in scope |
| `5117` (draw command) | `registryValue("imageModel", channel)` — hoist above the guard |

`2637` and `2718` set `reason=`; the rest set `content=`/`error=`. Keep each site's existing return shape and only replace the message text.

- [ ] **Step 5: Convert the four unguarded sites**

- `3672` — `api_key=self._api_key_for(model)` (`model` is in scope; the caller already guarded at `3602`).
- `4175` — `api_key=self._api_key_for(model)` (the `model` parameter).
- `6124`, `6195` — `api_key = self._api_key_for(model)`, `model` already fetched one line above.

These previously sent `""` and now send `None`, which is what enables LiteLLM's implicit environment lookup — a lookup that finds the same variable `api_key_for` just read. Behaviour is unchanged; the equivalence is deliberate, not accidental.

- [ ] **Step 6: Collapse the search key chain**

At `3179-3184`, replace the `searchApiKey or assistantApiKey` chain with a single resolve on the search model:

```python
model = self.plugin.registryValue("searchModel", target) or self.plugin.registryValue(
    "assistantModel", target
)
api_key = self._api_key_for(model)
```

This is a bug fix, not just a simplification: the two chains resolved independently, so grounded search on a grok channel sent a Gemini key to an xAI model.

- [ ] **Step 7: Run the affected tests**

```bash
uv run pytest plugins/llm/tests/test_provider_edge_cases.py \
  plugins/llm/tests/test_service_completion.py \
  plugins/llm/tests/test_service_memory.py \
  plugins/llm/tests/test_reminders.py -v
```

Expected: PASS. Tests asserting the exact `api_key` kwarg (`test_service_memory.py:105,725,757,806`, `test_service_completion.py:117`, `test_reminders.py:721`) now see the fake provider value from Task 3's fixture instead of the registry stub — update the expected value to `FAKE_PROVIDER_KEYS["OPENAI_API_KEY"]`, matching `TEST_MODEL`'s provider.

- [ ] **Step 8: Run the full suite, lint, typecheck, commit**

```bash
make test && make lint && make typecheck
git add plugins/llm/src/llm/service.py plugins/llm/tests/
git commit -m "refactor(service): resolve API keys from the model's provider

Anchoring the key to the model fixes two live mismatches: the pending-task
retry path paired codeApiKey with assistantModel, and grounded search paired
searchApiKey with a model chain resolved independently."
```

---

### Task 5: Resolve keys from the model in the verse compaction path

**Files:**
- Modify: `plugins/llm/src/llm/verse/compaction.py:147,273-286`, `plugins/llm/src/llm/plugin.py:6347-6349`
- Test: `plugins/llm/tests/test_plugin_verse.py`

**Interfaces:**
- Consumes: `apikeys.api_key_for`.
- Produces: `LiteLLMVerseClient` no longer takes `api_key`; it resolves per call from its own `model` argument.

**Why not just fix the call site:** `plugin.py:6347` hardcodes a `gemini/` model while `6348` reads `assistantApiKey` — an xAI key in prod. Resolving at construction would fix today's mismatch but leaves a model-independent seam that re-diverges the moment compaction calls a second model.

- [ ] **Step 1: Write the failing test**

Add to `plugins/llm/tests/test_plugin_verse.py`:

```python
class TestCompactionKeyResolution:
    """The compaction client resolves its key from the model it is calling."""

    def test_key_matches_the_call_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a gemini compaction model WHEN calling THEN GEMINI_API_KEY is sent."""
        from llm.verse import compaction

        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        seen: dict[str, object] = {}

        def fake_completion(**kwargs: object) -> object:
            seen.update(kwargs)
            raise RuntimeError("stop here — we only care about the kwargs")

        monkeypatch.setattr(compaction.litellm, "completion", fake_completion)
        client = compaction.LiteLLMVerseClient()
        with pytest.raises(RuntimeError):
            client.call(op="compact", model="gemini/gemini-flash-lite-latest", messages=[])
        assert seen["api_key"] == "AIza-fake-key-for-tests-0000"

    def test_client_takes_no_api_key_argument(self) -> None:
        """GIVEN the constructor WHEN inspected THEN no api_key parameter remains."""
        import inspect

        from llm.verse import compaction

        assert "api_key" not in inspect.signature(compaction.LiteLLMVerseClient).parameters
```

Adjust the `client.call(...)` keyword names to match the real signature at `compaction.py:147` if they differ — read it before writing the test.

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest plugins/llm/tests/test_plugin_verse.py -k CompactionKeyResolution -v
```

Expected: FAIL — the constructor still accepts `api_key`.

- [ ] **Step 3: Implement**

In `verse/compaction.py`:
- Delete the `api_key` parameter and `self._api_key` (`273-286`). Storing a key on `self` also fed supybot's attribute walk on exceptions.
- In `call()`, set `kwargs["api_key"] = apikeys.api_key_for(model)`.
- Import `from llm import apikeys`.

In `plugin.py:6347-6349`, delete the `api_key = self.registryValue("assistantApiKey")` line and construct `LiteLLMVerseClient()` with no arguments.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_plugin_verse.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/verse/compaction.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "fix(verse): resolve the compaction key from the compaction model

The client paired a hardcoded gemini/ model with assistantApiKey, an xAI key
in production."
```

---

### Task 6: Redaction cutover

**Files:**
- Modify: `plugins/llm/src/llm/service.py:1193-1240` (`_configured_api_keys`, `_sanitize`), `:1164` (filter install), `plugins/llm/src/llm/plugin.py:730` (filter install)
- Test: `plugins/llm/tests/test_service_core.py:218-300`

**Interfaces:**
- Consumes: `apikeys.scrub`, `apikeys.SecretFilter`, `apikeys.secret_var_names`.
- Produces: `_sanitize` unchanged in signature, backed by the environment.

**This task must land before Task 8.** Deleting the registry settings first would leave `_configured_api_keys` walking settings that no longer exist.

- [ ] **Step 1: Write the failing test**

Replace the five `_sanitize` tests at `test_service_core.py:218-300`. Delete `test_api_key_sanitization_channel_specific_key` outright — it calls `conf.supybot.plugins.LLM.get("assistantApiKey").get("#forest").setValue(...)`, which will raise `NonExistentRegistryEntry` after Task 8, and the per-channel key it tests no longer exists as a concept. Replace with:

```python
    def test_sanitize_redacts_environment_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a provider key in the environment WHEN sanitizing THEN redacted."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service()
        text = "AuthenticationError: key xai-fake-key-for-tests-0000 rejected"
        assert "xai-fake-key-for-tests-0000" not in service._sanitize(text)
        assert "[REDACTED]" in service._sanitize(text)

    def test_sanitize_handles_none(self) -> None:
        """GIVEN None WHEN sanitizing THEN empty string, as before."""
        service, _ = make_service()
        assert service._sanitize(None) == ""

    def test_secret_filter_installed_on_service_logger(self) -> None:
        """GIVEN a constructed service WHEN inspecting its logger THEN SecretFilter present."""
        from llm import apikeys

        service, _ = make_service()
        assert any(isinstance(f, apikeys.SecretFilter) for f in service.log.filters)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest plugins/llm/tests/test_service_core.py -k "sanitize or secret_filter" -v
```

Expected: FAIL on the filter test; the redaction test may pass by accident if a registry stub happens to hold the same value — do not take that as success.

- [ ] **Step 3: Implement**

- Delete `_configured_api_keys` (`1193-1220`) entirely.
- Rewrite `_sanitize` (`1222-1240`) as `return apikeys.scrub(text)`, keeping the docstring's explanation of *why* it is value-replacement rather than regex.
- At `service.py:1164` and `plugin.py:730`, add `self.log.addFilter(apikeys.SecretFilter())` beside the existing `TraceFilter`.
- Grep for any other logger the plugin owns and attach there too:

```bash
grep -rn "getPluginLogger" plugins/llm/src/llm/ | grep -v __pycache__
```

- In `plugin.py.__init__`, after the filters are installed, log the coverage summary — names only:

```python
        self.log.info(
            "secret redaction active for %d variable(s): %s",
            len(apikeys.secret_var_names()),
            ", ".join(apikeys.secret_var_names()) or "none",
        )
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_service_core.py -v && make test
```

Expected: PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_service_core.py
git commit -m "feat(logging): redact secrets at the logging filter, sourced from env

Hand-applied _sanitize never covered supybot's Logger.exception, which writes
the raw traceback plus a repr of every frame local."
```

---

### Task 7: Delete `profile.api_key_setting`

**Files:**
- Modify: `plugins/llm/src/llm/profile.py:60,80,93,106,118,132,157`, `service.py:4325-4327`
- Test: `plugins/llm/tests/test_profile.py:62-65,127,182-183`, `test_assistant.py:4378,4399,4451,4515,4596,4658`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Profile` without the `api_key_setting` field.

- [ ] **Step 1: Update the tests first**

- `test_profile.py:62-65` — delete `test_api_key_setting_is_a_known_registry_key` (both parametrized methods, 10 cases) and the `EXPECTED_API_KEY` table at `:127`.
- `test_assistant.py:4378` `test_model_setting_is_read_from_profile` — it asserts `"SENTINEL_API_KEY" in registry_calls` and `"assistantApiKey" not in registry_calls` (`:4439,4441`). Delete the two key assertions; keep the model assertions, which are the point of the test.
- `test_assistant.py:4399,4451,4515,4596,4658` — remove the `api_key_setting=` keyword from five `Profile(...)` constructions. Four are overlay/model tests and need no other change.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest plugins/llm/tests/test_profile.py plugins/llm/tests/test_assistant.py -v
```

Expected: FAIL — `Profile.__init__` still requires `api_key_setting`.

- [ ] **Step 3: Implement**

- `profile.py` — delete the `api_key_setting: str` field (`:80`), its docstring line (`:60`), and the five `api_key_setting="assistantApiKey",` entries (`93`, `106`, `118`, `132`, `157`).
- `service.py:4325-4327` — already converted in Task 4 to resolve from `model`; confirm no `profile.api_key_setting` reference remains: `grep -rn api_key_setting plugins/llm/`.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest plugins/llm/tests/test_profile.py plugins/llm/tests/test_assistant.py -v
```

Expected: PASS.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/profile.py plugins/llm/tests/test_profile.py plugins/llm/tests/test_assistant.py
git commit -m "refactor(profile): drop api_key_setting, identical on every profile"
```

---

### Task 8: Delete the registry key settings

**Files:**
- Modify: `plugins/llm/src/llm/config.py:120,131-171,228-233`
- Test: `plugins/llm/tests/test_config.py:137,159,196-209`, `conftest.py:403,406,417,474`, `test_plugin_verse.py:2806,2855`, `test_service_memory.py:24,53,67,113,148,199`, `test_stress.py:291`

**Interfaces:**
- Consumes: nothing.
- Produces: a plugin with no `*ApiKey` registry settings. Any surviving `registryValue("...ApiKey")` call raises `NonExistentRegistryEntry`.

- [ ] **Step 1: Confirm nothing still reads them**

```bash
grep -rn "ApiKey" plugins/llm/src/ | grep -v __pycache__
```

Expected: no hits. If any remain, they were missed in Task 4 — fix them there before continuing.

- [ ] **Step 2: Update the tests**

- `conftest.py:403,406,417,474` — delete the four key stubs.
- `test_config.py:137` — the wizard string assertion; update to the new text from Step 3.
- `test_config.py:159` — `codeApiKey._private`; delete.
- `test_config.py:196-209` — defaults and `_private` flags; delete the four key entries, and update the `imageModel` default assertion at `:203` to `gemini/imagen-4.0-fast-generate-001`.
- `test_plugin_verse.py:2806,2855` — delete the `if key == "assistantApiKey"` branches from the bare-LLM stubs.
- `test_service_memory.py:24,53,67,113,148,199` — these stub `assistantApiKey` to drive the missing-key guard. Repoint at the environment: `monkeypatch.delenv("OPENAI_API_KEY", raising=False)` (the provider of `TEST_MODEL`).

- [ ] **Step 3: Implement**

In `config.py`:
- Delete the four `registerChannelValue` blocks (`131-171`) and the `# API Keys` header comment above them.
- Delete the wizard's key line (`:120`) and reword the surrounding text to point at environment variables:

```python
    print("\nAPI keys come from the environment, one per provider:")
    print("  XAI_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
```

- Change the `imageModel` default (`228-233`) from `vertex_ai/imagen-4.0-generate-001` to `gemini/imagen-4.0-fast-generate-001`. `vertex_ai` authenticates by service account and has no single key variable; the new default matches `assistantModel`'s `gemini/` provider, so a fresh install needs exactly one variable.

- [ ] **Step 4: Run the full suite**

```bash
make test
```

Expected: PASS. A `NonExistentRegistryEntry` here means a test still stubs or reads a deleted setting — grep the traceback's key name across `plugins/llm/tests/`.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
make lint && make typecheck
git add plugins/llm/src/llm/config.py plugins/llm/tests/
git commit -m "feat(config)!: delete the four API-key registry settings

Keys now come from one environment variable per provider, which makes
per-channel key drift structurally impossible."
```

---

### Task 9: Delete the dead key-plumbing parameters

**Files:**
- Modify: `plugins/llm/src/llm/service.py:2969,3028,3500,3544,4242,4325,2631-2636,3021,3025,4155-4175`

**Interfaces:**
- Consumes: nothing.
- Produces: `assistant_completion`, `_completion_with_tool_fallback`'s facade, and the profile path without `api_key` parameters; `_generate_image_once` without `channel`.

**Verified dead:** no caller in `plugins/llm` passes `api_key` to these three; the `**` spreads at their call sites expand `_pending_task_fns`, which returns only callables; there is no `functools.partial` or dynamic dispatch reaching them. The 10 `api_key=` occurrences in tests all target `_xai_responses_call` (`3286`) or `_completion_with_tool_fallback` (`1870`), where the parameter is required and stays.

- [ ] **Step 1: Re-verify before deleting**

```bash
grep -rn "api_key=" plugins/llm/src plugins/llm/tests | grep -v __pycache__ | grep -v "api_key=api_key"
```

Expected: only `_xai_responses_call` and `_completion_with_tool_fallback` call sites. If anything else appears, stop and re-check — the deletion premise is wrong.

- [ ] **Step 2: Delete**

- `service.py:2969` and `3028` — the `api_key` parameter of `assistant_completion` and its `effective_api_key = api_key or ...` line, which Task 4 already reduced to a resolve.
- `service.py:3500` and `3544` — the facade's `api_key` parameter and the forward.
- `service.py:4242` and `4325` — the profile path's `api_key` parameter and its `or` line.
- `service.py:2631-2636` — the whole `api_key_name` ladder.
- `service.py:3021` and `3025` — the two `api_key_name = ...` lines **only**. Do not delete the block: the same `if/else` selects `codeModel`/`codeSystemPrompt` versus `assistantModel`/`assistantSystemPrompt`.
- `service.py:4155-4175` — the `channel` parameter of `_generate_image_once` and its docstring line. It existed solely for the per-channel `imageApiKey` lookup, and it passed a raw `msg.args[0]` — a nick in a PM — as a registry scope, unlike every other site.

Update each caller to stop passing the deleted arguments.

- [ ] **Step 3: Run the full suite, lint, typecheck**

```bash
make test && make lint && make typecheck
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/
git commit -m "refactor(service): delete key-plumbing that reconciled per-role keys"
```

---

### Task 10: Redesign the concurrency isolation test

**Files:**
- Modify: `plugins/llm/tests/test_stress.py:264-313`

**Interfaces:**
- Consumes: `FAKE_PROVIDER_KEYS` from `conftest.py`.
- Produces: nothing consumed downstream.

`test_completion_api_key_isolation` spawns 20 threads with 20 distinct `assistantApiKey` values and asserts `len(unique_keys) == 20`. One key per provider makes that unrepresentable — but the property it protects (a key must not bleed between concurrent requests) still matters. Re-express it across providers.

- [ ] **Step 1: Rewrite the test**

```python
    def test_completion_key_isolation_across_providers(self) -> None:
        """GIVEN concurrent requests on different providers THEN no key bleeds.

        Each thread's call must carry its own provider's key. Re-expressed from
        the pre-migration version, which gave each thread a distinct registry
        key — impossible once there is one key per provider, but the
        cross-request bleed it guarded against is still worth catching.
        """
        models = ["xai/grok-4.3", "gemini/gemini-3-flash-preview", "openai/gpt-5.2"]
        expected = {
            "xai/grok-4.3": FAKE_PROVIDER_KEYS["XAI_API_KEY"],
            "gemini/gemini-3-flash-preview": FAKE_PROVIDER_KEYS["GEMINI_API_KEY"],
            "openai/gpt-5.2": FAKE_PROVIDER_KEYS["OPENAI_API_KEY"],
        }
        observed: list[tuple[str, str | None]] = []
        lock = threading.Lock()
        service, _ = make_service()

        def worker(index: int) -> None:
            model = models[index % len(models)]
            resolved = service._api_key_for(model)
            with lock:
                observed.append((model, resolved))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(observed) == 20
        for model, resolved in observed:
            assert resolved == expected[model], f"key bleed: {model} got the wrong provider's key"
```

- [ ] **Step 2: Run it**

```bash
uv run pytest plugins/llm/tests/test_stress.py -k isolation -v
```

Expected: PASS. Import `FAKE_PROVIDER_KEYS` from `conftest` if it is not already available.

- [ ] **Step 3: Commit**

```bash
make lint && make typecheck
git add plugins/llm/tests/test_stress.py
git commit -m "test(stress): re-express key isolation across providers"
```

---

### Task 11: Documentation and operator runbook

**Files:**
- Modify: `.env.example`, `README.md:34`, `docs/guide/operator/configuration.md:25-28,33`, `docs/guide/operator/tuning-monitoring.md:22,25,166`, `docs/guide/operator/memory-promotion.md:88`, `docs/guide/operator/operations.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Rewrite `.env.example`**

It currently says "these env vars are optional — API keys can be configured via bot.conf instead", which becomes false, and lists `VERTEX_AI_API_KEY`/`VERTEX_PROJECT`/`VERTEX_LOCATION`, which this design does not support. `Makefile:244-258` copies this file to the live env path, so it is not merely illustrative.

State: one variable per provider, required, no bot.conf alternative. Include the `docker run --env-file` format rules — no quotes, no trailing comments, no spaces around `=`, LF endings — because that parser is stricter than a shell and a malformed file crashloops the container.

- [ ] **Step 2: Update the operator guides**

- `configuration.md:25-28,33` — replace the four-row key table and the `@config` example with the environment variables.
- `tuning-monitoring.md:22,25,166` — remove the `searchApiKey` → `assistantApiKey` fallback prose. **`:166` matters most**: it tells an operator to run `@config plugins.LLM.assistantApiKey` to confirm a key is set, which is precisely the diagnostic someone reaches for during this migration and precisely what stops working. Replace with the container-side check:

```
docker exec vibebot python3 -c "import os;k=os.environ.get('XAI_API_KEY','');print(len(k), k[:4], k[-4:])"
```

- `memory-promotion.md:88` — drop the key reference.
- `README.md:34` — replace `@config plugins.LLM.assistantApiKey YOUR_KEY` with the environment variable.

- [ ] **Step 3: Add a rollback section to `operations.md`**

There is none today, and the migration needs one. A revert push takes 10-20 minutes and can be blocked by a single flaky test, since `docker.yml` gates on CI across a three-version matrix with `--cov-fail-under=93`. The fast path is an image pin — every build is tagged `type=sha` (`docker.yml:41-46`):

```bash
# Stop the updater first: vibebot-updater.service hardcodes :latest and would
# otherwise bounce the pinned container every 15 minutes.
systemctl --user stop vibebot-updater.timer
mkdir -p ~/.config/systemd/user/vibebot.service.d
printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot-v8:sha-<PREV>\n' \
  > ~/.config/systemd/user/vibebot.service.d/override.conf
systemctl --user daemon-reload && systemctl --user restart vibebot
```

Document how to find `<PREV>`: `docker inspect vibebot --format '{{.Config.Image}}'` before deploying, or the GitHub Actions run summary.

- [ ] **Step 4: Verify the docs build and commit**

```bash
uv run mkdocs build --strict 2>&1 | tail -5
git add .env.example README.md docs/guide/operator/
git commit -m "docs: describe provider environment variables and add a rollback runbook"
```

If `mkdocs build --strict` is not wired up in this repo, skip it and rely on the pre-commit hooks.

---

### Task 12: Final verification

**Files:** none modified.

- [ ] **Step 1: Full gate**

```bash
make check
```

Expected: lint, format-check, typecheck, syntax-check, and 2549 tests all pass, with coverage at or above 93%.

- [ ] **Step 2: Confirm the deletions actually happened**

```bash
grep -rn "ApiKey\|api_key_setting" plugins/llm/src/ docs/ README.md .env.example | grep -v __pycache__
```

Expected: no hits outside `CHANGELOG.md` (historical) and the spec/plan documents.

- [ ] **Step 3: Confirm coverage of the new module**

```bash
uv run pytest plugins/llm/tests/test_apikeys.py --cov=llm.apikeys --cov-report=term-missing
```

Expected: at or near 100%. Anything uncovered in `apikeys.py` is a missing test, not an acceptable gap — the module is pure.

- [ ] **Step 4: Prove redaction end to end**

```bash
uv run pytest plugins/llm/tests/ -k "secret or sanitize or redact" -v
```

Expected: PASS, including the traceback and args cases.

- [ ] **Step 5: Commit any stragglers and push**

```bash
git status --short
git push origin main
```

Then wait for **both** CI and the Docker build — they are separate workflows, and the auto-deploy fires on the Docker build's success, not CI's.

---

## Migration checklist (operator, after the code is deployed)

Not part of the code change. Execute in order; step 2 is the highest outage risk in the whole exercise.

- [ ] **Pre-flight, read-only.** Confirm `supybot.flush` and `supybot.upkeepInterval` are not overridden in prod `bot.conf`. If `flush` is false, the automatic cleanup below does not happen and step 5 becomes mandatory. Record the current `searchModel` and `imageModel` values, and the deployed image tag: `docker inspect vibebot --format '{{.Config.Image}}'`.
- [ ] **Back up.** Copy `bot.conf` off the host and record KEY-A through KEY-D in a password manager. Limnoria's rolling `bot.conf.backup.<timestamp>` files roll forward and will all be post-deletion within a couple of flush cycles.
- [ ] **Populate the env file** (`/home/vibebot/.config/vibebot/env`): `GEMINI_API_KEY` = KEY-A, `XAI_API_KEY` = **KEY-B** (the key powering `#afternet`, confirmed canonical). Give `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` real values or delete them — a wrong-length placeholder is worse than an absent variable. No quotes, no trailing comments, no spaces around `=`, LF endings, and no stray `IMAGE=` line (it would override the deployed image). Restart, then verify **from inside the container**, not by reading the file:
  `docker exec vibebot python3 -c "import os;k=os.environ.get('XAI_API_KEY','');print(len(k),k[:4],k[-4:])"`
- [ ] **Deploy the code** and watch for the `secret redaction active for N variable(s)` line at startup.
- [ ] **Verify within the first hour**, before the upkeep flush rewrites `bot.conf`: `@ask` on a grok channel, `@ask` on the global Gemini path, `@draw`, and one grounded-search request.
- [ ] **Sweep `bot.conf` by prefix, with the bot stopped** — every `supybot.plugins.LLM.*ApiKey*` line, including the stale `askApiKey`/`drawApiKey` entries from settings renamed long ago. Expect the flush to have removed most already.
- [ ] **Revoke** KEY-C and KEY-D, plus whatever credentials the stale `askApiKey`/`drawApiKey` entries held.
- [ ] **Decide on `supybot.commands.allowShell`** (currently `True`, `bot.conf:153`). The `Debug` plugin's `environ` command replies `repr(os.environ)` **to the channel** and is one `@load` away for an owner. Before this change the worst owner fat-finger was one key delivered to a PM; afterwards it is all four keys in a channel. `allowShell: False` closes it.
