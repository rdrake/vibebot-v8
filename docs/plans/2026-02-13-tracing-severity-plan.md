# Tracing: Server Headers & Log Severity — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable log severity and server-identifying HTTP header extraction to LLM plugin tracing.

**Architecture:** A new `logLevel` Limnoria config key controls plugin log verbosity at runtime. A new `extract_server_headers()` helper in `tracing.py` pulls server-identifying headers from LiteLLM responses and exceptions. Error handlers and success paths in `service.py` call this helper and log results at DEBUG level, visible only when `logLevel` is set to `DEBUG`.

**Tech Stack:** Python logging, Limnoria registry callbacks, LiteLLM response/exception objects, httpx.Headers

---

### Task 1: Add `logLevel` to test fixtures

**Files:**
- Modify: `plugins/llm/tests/conftest.py:104-142` (defaults dict)

**Step 1: Add `logLevel` default**

In the `defaults` dict inside `make_registry_side_effect`, add `logLevel` alongside the other shared config keys:

```python
        # Shared
        "timeout": 30,
        "maxPromptLength": 10000,
        "commandPrefixes": [".", "/"],
        "httpUrlBase": TEST_URL_BASE,
        "fileCleanupAge": 24,
        "fileCleanupMax": 100,
        "logLevel": "WARNING",
```

**Step 2: Run existing tests to verify nothing breaks**

Run: `make test`
Expected: All tests pass (the new key is inert since nothing reads it yet)

**Step 3: Commit**

```bash
git add plugins/llm/tests/conftest.py
git commit -m "test: add logLevel to shared test fixtures"
```

---

### Task 2: `ValidatedLogLevel` class and `logLevel` config key

**Files:**
- Create: `plugins/llm/tests/test_tracing.py`
- Modify: `plugins/llm/src/llm/config.py`

**Step 1: Write failing tests for `ValidatedLogLevel`**

Create `plugins/llm/tests/test_tracing.py`:

```python
"""Tests for tracing utilities and config."""

from __future__ import annotations

import pytest
import supybot.registry as registry


class TestValidatedLogLevel:
    """Tests for ValidatedLogLevel registry type."""

    def test_accepts_warning(self) -> None:
        """GIVEN 'WARNING' WHEN set THEN accepted."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("WARNING")
        assert v() == "WARNING"

    def test_accepts_debug(self) -> None:
        """GIVEN 'DEBUG' WHEN set THEN accepted."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("DEBUG")
        assert v() == "DEBUG"

    def test_accepts_lowercase(self) -> None:
        """GIVEN 'debug' WHEN set THEN normalized to 'DEBUG'."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        v.setValue("debug")
        assert v() == "DEBUG"

    @pytest.mark.parametrize("value", ["VERBOSE", "3", "TRACE", ""])
    def test_rejects_invalid(self, value: str) -> None:
        """GIVEN invalid level WHEN set THEN raises InvalidRegistryValue."""
        from llm.config import ValidatedLogLevel

        v = ValidatedLogLevel("WARNING", "test")
        with pytest.raises(registry.InvalidRegistryValue):
            v.setValue(value)
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_tracing.py::TestValidatedLogLevel -v`
Expected: FAIL — `ValidatedLogLevel` does not exist yet

**Step 3: Implement `ValidatedLogLevel` and config key**

In `plugins/llm/src/llm/config.py`, add the class after `ValidatedModelName` (around line 84):

```python
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
```

Then add the config registration at the end of the file (after the `fileCleanupMax` block, in the Advanced Settings section):

```python
conf.registerGlobalValue(
    LLM,
    "logLevel",
    ValidatedLogLevel(
        "WARNING",
        _("""Log level for LLM plugin (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Set to DEBUG for verbose tracing including server response headers."""),
    ),
)
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_tracing.py::TestValidatedLogLevel -v`
Expected: All 5 tests PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_tracing.py
git commit -m "feat: add ValidatedLogLevel config type and logLevel setting"
```

---

### Task 3: `extract_server_headers` helper

**Files:**
- Modify: `plugins/llm/tests/test_tracing.py`
- Modify: `plugins/llm/src/llm/tracing.py`

**Step 1: Write failing tests for `extract_server_headers`**

Append to `plugins/llm/tests/test_tracing.py`:

```python
import httpx
from llm.tracing import extract_server_headers


class TestExtractServerHeaders:
    """Tests for extract_server_headers."""

    def test_extracts_from_response_headers(self) -> None:
        """GIVEN response with _response_headers WHEN extracted THEN returns matching headers."""

        class FakeResponse:
            _response_headers = {"x-request-id": "abc123", "content-type": "application/json"}

        result = extract_server_headers(FakeResponse())
        assert result == {"x-request-id": "abc123"}

    def test_extracts_from_exception_response(self) -> None:
        """GIVEN exception with response.headers WHEN extracted THEN returns matching headers."""

        class FakeException(Exception):
            response = httpx.Response(
                400,
                headers={"cf-ray": "def456-YYZ", "x-request-id": "req-789"},
            )

        result = extract_server_headers(FakeException())
        assert result == {"cf-ray": "def456-YYZ", "x-request-id": "req-789"}

    def test_extracts_from_direct_headers(self) -> None:
        """GIVEN object with .headers dict WHEN extracted THEN returns matching headers."""

        class FakeObj:
            headers = {"server": "nginx/1.25", "x-served-by": "node-3"}

        result = extract_server_headers(FakeObj())
        assert result == {"server": "nginx/1.25", "x-served-by": "node-3"}

    def test_returns_empty_when_no_headers(self) -> None:
        """GIVEN object with no header attributes WHEN extracted THEN returns empty dict."""
        result = extract_server_headers(object())
        assert result == {}

    def test_returns_empty_for_none(self) -> None:
        """GIVEN None WHEN extracted THEN returns empty dict."""
        result = extract_server_headers(None)
        assert result == {}

    def test_ignores_non_server_headers(self) -> None:
        """GIVEN headers with only non-server headers WHEN extracted THEN returns empty."""

        class FakeResponse:
            _response_headers = {
                "content-type": "application/json",
                "content-length": "42",
            }

        result = extract_server_headers(FakeResponse())
        assert result == {}

    def test_case_insensitive_header_names(self) -> None:
        """GIVEN headers with mixed case WHEN extracted THEN matches case-insensitively."""

        class FakeResponse:
            _response_headers = httpx.Headers(
                {"X-Request-ID": "abc", "CF-Ray": "def"}
            )

        result = extract_server_headers(FakeResponse())
        assert result == {"x-request-id": "abc", "cf-ray": "def"}
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_tracing.py::TestExtractServerHeaders -v`
Expected: FAIL — `extract_server_headers` does not exist

**Step 3: Implement `extract_server_headers`**

In `plugins/llm/src/llm/tracing.py`, add after the existing code:

```python
# Headers that identify the backend server handling a request.
SERVER_ID_HEADERS = frozenset(
    ("x-request-id", "cf-ray", "server", "x-server-id", "x-served-by")
)


def extract_server_headers(source: object | None) -> dict[str, str]:
    """Extract server-identifying HTTP headers from a LiteLLM response or exception.

    Checks (in order):
    1. source._response_headers  (LiteLLM successful completions)
    2. source.response.headers   (LiteLLM exceptions with httpx.Response)
    3. source.headers            (fallback)

    Args:
        source: A LiteLLM response object, exception, or None.

    Returns:
        Dict of header-name -> value for recognised server headers.
        Empty dict when no headers are available.
    """
    if source is None:
        return {}

    raw: object | None = None

    # 1. LiteLLM response objects (_response_headers attribute)
    raw = getattr(source, "_response_headers", None)

    # 2. LiteLLM exceptions wrap httpx.Response on .response
    if raw is None:
        resp = getattr(source, "response", None)
        if resp is not None:
            raw = getattr(resp, "headers", None)

    # 3. Direct .headers fallback
    if raw is None:
        raw = getattr(source, "headers", None)

    if raw is None:
        return {}

    # raw may be dict, httpx.Headers, or similar mapping
    try:
        items = raw.items() if hasattr(raw, "items") else []
    except Exception:
        return {}

    return {
        k.lower(): v
        for k, v in items
        if k.lower() in SERVER_ID_HEADERS
    }
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_tracing.py::TestExtractServerHeaders -v`
Expected: All 7 tests PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/tracing.py plugins/llm/tests/test_tracing.py
git commit -m "feat: add extract_server_headers for tracing server identity"
```

---

### Task 4: Wire up log level in plugin.py

**Files:**
- Modify: `plugins/llm/tests/test_plugin.py`
- Modify: `plugins/llm/src/llm/plugin.py`

**Step 1: Write failing test for log level init**

Find the existing plugin init test class in `test_plugin.py`. Add a test that verifies the plugin logger level is set from config. The test should use the existing `plugin_init_patches` helper and `make_registry_side_effect`:

```python
def test_init_applies_log_level(self) -> None:
    """GIVEN logLevel=DEBUG WHEN plugin initialized THEN logger level is DEBUG."""
    import logging

    assert self.plugin.log.level == logging.WARNING  # default from fixture
```

Note: This test will initially pass if the logger defaults to WARNING. The real verification is that changing the config actually changes the level. A second test:

```python
def test_init_applies_custom_log_level(self, mocker: MockerFixture, mock_irc) -> None:
    """GIVEN logLevel=DEBUG in config WHEN plugin initialized THEN logger level is DEBUG."""
    import logging

    patches = plugin_init_patches(mocker)
    side_effect = make_registry_side_effect({"logLevel": "DEBUG"})
    LLM_cls = self._get_llm_class()
    with mocker.patch.object(LLM_cls, "registryValue", side_effect=side_effect):
        plugin = LLM_cls(mock_irc)
    assert plugin.log.level == logging.DEBUG
```

The exact test structure depends on how plugin tests are organized. The key assertion: `plugin.log.level` matches the configured `logLevel`.

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_plugin.py -k "log_level" -v`
Expected: FAIL — plugin doesn't set log level yet

**Step 3: Implement log level wiring**

In `plugins/llm/src/llm/plugin.py`, add to the imports:

```python
import logging
```

In `__init__`, after `self.log.addFilter(TraceFilter())` (line 286), add:

```python
        # Apply configured log level to plugin and service loggers
        self._apply_log_level()
```

Add the helper method to the `LLM` class:

```python
    def _apply_log_level(self) -> None:
        """Set plugin logger levels from the logLevel config value."""
        level_name = self.registryValue("logLevel")
        level = getattr(logging, level_name, logging.WARNING)
        self.log.setLevel(level)
        self.llm_service.log.setLevel(level)
```

In `__init__`, after the periodic event scheduling (after the `schedule.addPeriodicEvent` calls), register the callback for live updates:

```python
        # Register callback for live log level changes
        conf.supybot.plugins.LLM.logLevel.addCallback(self._on_log_level_change)
```

And the callback method:

```python
    def _on_log_level_change(self, *args: object) -> None:
        """Called when logLevel config changes at runtime."""
        self._apply_log_level()
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_plugin.py -k "log_level" -v`
Expected: PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: wire logLevel config to plugin and service loggers"
```

---

### Task 5: Header logging on error paths

**Files:**
- Modify: `plugins/llm/tests/test_service.py`
- Modify: `plugins/llm/src/llm/service.py`

**Step 1: Write failing tests for error header logging**

Add to `test_service.py` in an appropriate test class (e.g., `TestImageGeneration` or a new class):

```python
class TestServerHeaderLogging:
    """Tests for server header extraction in error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_completion_error_logs_server_headers(self) -> None:
        """GIVEN completion raises with response headers WHEN error caught THEN headers logged at DEBUG."""
        import httpx
        import litellm

        exc = litellm.APIError(
            status_code=500,
            message="server error",
            llm_provider="xai",
            model="xai/grok-2",
        )
        exc.response = httpx.Response(
            500, headers={"x-request-id": "srv-abc", "cf-ray": "ray-123"}
        )

        self.mocker.patch("llm.service.litellm.completion", side_effect=exc)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is not None

        # Verify debug log was called with header info
        debug_calls = [
            str(c) for c in self.service.log.debug.call_args_list
        ]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"

    def test_image_generation_error_logs_server_headers(self) -> None:
        """GIVEN image_generation raises with response headers WHEN error caught THEN headers logged."""
        import httpx
        import litellm

        exc = litellm.BadRequestError(
            message="moderation_blocked",
            model="xai/grok-imagine-image",
            llm_provider="xai",
            response=httpx.Response(
                400, headers={"x-request-id": "img-xyz", "cf-ray": "ray-456"}
            ),
        )

        self.mocker.patch("llm.service.litellm.image_generation", side_effect=exc)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%draw cat")

        result = self.service.image_generation("a cat", msg=mock_msg)
        assert result.error is not None

        debug_calls = [
            str(c) for c in self.service.log.debug.call_args_list
        ]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"

    def test_no_crash_when_exception_has_no_headers(self) -> None:
        """GIVEN exception without response headers WHEN error caught THEN no crash."""
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("generic error"),
        )

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is not None  # completed without crash
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_service.py::TestServerHeaderLogging -v`
Expected: FAIL — no header logging exists yet

**Step 3: Implement header logging in error paths**

In `plugins/llm/src/llm/service.py`, add to imports:

```python
from .tracing import TraceFilter, extract_server_headers, request_id
```

Add a helper method to `LLMService`:

```python
    def _log_server_headers(self, source: object | None) -> None:
        """Log server-identifying headers from a response or exception at DEBUG level."""
        headers = extract_server_headers(source)
        if headers:
            self.log.debug("server headers: %s", headers)
```

Then add `self._log_server_headers(e)` calls in these error handlers:

- **`_completion_with_tool_fallback`** (line ~681): After `except litellm.BadRequestError as e:` and before the `if "tools"` check
- **`completion`** (line ~1477): After `except litellm.Timeout as e:`
- **`completion`** (line ~1508): After `except Exception as e:`
- **`_attempt_image_generation`**: No change needed — exceptions propagate up to `image_generation`
- **`image_generation`** (line ~1930): After `except litellm.Timeout as e:`
- **`image_generation`** (line ~1957): After `except litellm.ContentPolicyViolationError as e:`
- **`image_generation`** (line ~1960): After `except Exception as e:`
- **`image_generation`** (line ~2016): After `except litellm.ContentPolicyViolationError as e:` (rewrite loop)
- **`image_generation`** (line ~2019): After `except Exception as e:` (rewrite loop)
- **`image_generation`** (line ~2045): After `except Exception as e:` (outer)
- **`video_generation`** (line ~2237): After `except requests.HTTPError as e:` — use `self._log_server_headers(e.response)` (requests exception, not litellm)
- **`video_generation`** (line ~2263): After `except Exception as e:`

Each insertion is a single line: `self._log_server_headers(e)` (or `self._log_server_headers(e.response)` for requests exceptions).

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_service.py::TestServerHeaderLogging -v`
Expected: All 3 tests PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: log server headers from LiteLLM errors at DEBUG level"
```

---

### Task 6: Header logging on success paths

**Files:**
- Modify: `plugins/llm/tests/test_service.py`
- Modify: `plugins/llm/src/llm/service.py`

**Step 1: Write failing test for success header logging**

Add to `TestServerHeaderLogging` in `test_service.py`:

```python
    def test_completion_success_logs_server_headers(self) -> None:
        """GIVEN successful completion with _response_headers WHEN returned THEN headers logged."""
        mock_response = self.mocker.Mock()
        mock_response.choices = [self.mocker.Mock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response._response_headers = {
            "x-request-id": "success-abc",
            "content-type": "application/json",
        }
        mock_response.usage = self.mocker.Mock(
            prompt_tokens=10, completion_tokens=5
        )
        mock_response.id = "test-id"

        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is None

        debug_calls = [
            str(c) for c in self.service.log.debug.call_args_list
        ]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_service.py::TestServerHeaderLogging::test_completion_success_logs_server_headers -v`
Expected: FAIL — no success header logging yet

**Step 3: Implement success header logging**

In `service.py`, in the `completion` method, after the existing info log at line ~1461:

```python
            self.log.info("completion response: id=%s", getattr(response, "id", "n/a"))
            self._log_server_headers(response)
```

And in `_attempt_image_generation`, after the existing info log at line ~1827:

```python
        self.log.info("image_generation response: id=%s", getattr(response, "id", "n/a"))
        self._log_server_headers(response)
```

(For image gen success, `_response_headers` won't be populated by LiteLLM, so this will be a no-op — but it's future-proof.)

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_service.py::TestServerHeaderLogging -v`
Expected: All 4 tests PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: log server headers from successful completions at DEBUG level"
```

---

### Task 7: Final verification

**Step 1: Run full preflight**

Run: `make preflight`
Expected: All checks pass — format, lint, typecheck, test with 80%+ coverage

**Step 2: Verify the feature end-to-end**

Mentally trace the flow:
1. User sets `%config plugins.LLM.logLevel DEBUG` — callback fires, loggers updated
2. User runs `%draw a cat` — xAI returns error with `x-request-id` in response headers
3. Error handler calls `self._log_server_headers(e)` which calls `extract_server_headers(e)`
4. Function finds `e.response.headers`, extracts `x-request-id`
5. `self.log.debug("server headers: {'x-request-id': 'abc'}")` — visible because level is DEBUG
6. TraceFilter prepends `[trace_id]` so output is: `[a1b2c3d4] server headers: {'x-request-id': 'abc'}`

**Step 3: No additional commit needed**
