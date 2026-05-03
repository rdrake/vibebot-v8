# Tracing: Server Headers & Log Severity

**Date:** 2026-02-13
**Status:** Approved

## Problem

When xAI image generation fails, we cannot determine which backend server refused the request. Response headers (e.g. `x-request-id`, `cf-ray`) identify the specific server, but they are not captured or logged. Additionally, the plugin has no configurable log verbosity — all log lines are emitted at fixed levels with no way to increase detail for live debugging.

## Design

### Feature 1: Configurable log level

Add a `logLevel` Limnoria config key (global, default `"WARNING"`). A `ValidatedLogLevel` registry subclass validates against Python log level names. On plugin init, both the plugin and service loggers are set to this level. A registry callback updates both loggers immediately when the config changes at runtime.

### Feature 2: Server header extraction

Add a helper `extract_server_headers()` in `tracing.py` that extracts server-identifying HTTP headers from LiteLLM responses or exceptions. It checks, in order:

1. `source._response_headers` (populated by LiteLLM on successful chat completions)
2. `source.response.headers` (populated by LiteLLM on exceptions)
3. `source.headers` (fallback)

Target headers: `x-request-id`, `cf-ray`, `server`, `x-server-id`, `x-served-by`.

### Feature 3: Logging integration

**On errors (all commands):** After catching any LiteLLM exception, call `extract_server_headers(exception)` and log the result at DEBUG level. The existing `[trace_id]` prefix from `TraceFilter` correlates these with the request.

**On success (chat completions — ask, code):** After `litellm.completion()` returns, extract headers from the response and log at DEBUG.

**On success (image generation — draw):** No change. LiteLLM does not populate `_response_headers` on `ImageResponse` in its current OpenAI-compat code path. If LiteLLM adds this later, `extract_server_headers` will pick it up automatically.

At the default WARNING level, all debug header lines are silent. Setting `logLevel` to `DEBUG` enables them.

## Files changed

| File | Change |
|------|--------|
| `config.py` | `ValidatedLogLevel` class, `logLevel` config key |
| `tracing.py` | `SERVER_ID_HEADERS` constant, `extract_server_headers()` function |
| `plugin.py` | Apply log level on init, register callback for live updates |
| `service.py` | Call `extract_server_headers` in error handlers and after completions, log at DEBUG |
| `conftest.py` | Add `logLevel` to default registry values |
| `test_service.py` | Tests for header logging on errors and successes |
| `test_provider_edge_cases.py` | Add `logLevel` to fixtures if needed |
| `tests/test_tracing.py` (new) | Tests for `extract_server_headers` and `ValidatedLogLevel` |

## Approach trade-offs

**Chosen:** Direct extraction from exceptions and response objects. Covers failures (the primary debugging need) fully and gets success headers for free on chat completions.

**Rejected — LiteLLM CustomLogger callback:** Would capture headers for all calls including successful image gen, but adds complexity, relies on callback internals, and requires correlating callbacks back to request IDs via metadata.

**Rejected — Wrapping the OpenAI client:** Would get image gen success headers but is fragile and breaks on LiteLLM updates.
