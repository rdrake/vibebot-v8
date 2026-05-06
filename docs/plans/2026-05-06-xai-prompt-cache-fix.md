# xAI Prompt Cache Fix

Date: 2026-05-06

## Problem

xAI prompt caching should be automatic, but our current xAI requests are not
showing useful cache hits. The local request path already tries to support
sticky routing for xAI Chat Completions by adding `x-grok-conv-id`, but the
Responses API path uses the wrong cache control and our logging cannot show
Responses cache hits even if they occur.

xAI documents three relevant caveats:

- Chat Completions should use the `x-grok-conv-id` HTTP header for sticky
  routing.
- Responses API should use the `prompt_cache_key` request body field instead.
- Reasoning models need previous `reasoning_content` preserved, or stateful
  Responses chaining with `previous_response_id`, to maintain multi-turn cache
  hits.

Sources:

- <https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits>
- <https://docs.x.ai/developers/advanced-api-usage/prompt-caching/multi-turn>
- <https://docs.x.ai/developers/advanced-api-usage/prompt-caching/usage-and-pricing>
- <https://docs.x.ai/developers/model-capabilities/text/reasoning>
- <https://docs.litellm.ai/>

## Current State

Chat and assistant calls use `litellm.completion()` and inject
`extra_headers={"x-grok-conv-id": "chan:<channel>"}` for `xai/...` models.
That is the right API surface for Chat Completions.

Search and URL calls for xAI use `litellm.responses()` because xAI Live Search
is now exposed through the Responses API. That path currently passes the same
`x-grok-conv-id` header, but xAI documents `prompt_cache_key` as a body field
for Responses API requests.

The Responses path also logs `cached_tokens=0` unconditionally. xAI reports
Responses cache reads at `usage.input_tokens_details.cached_tokens`, so the
current logs cannot distinguish misses from unobserved hits.

Conversation history stores only visible `role` and `content`. That means we
do not currently preserve LiteLLM's standardized `message.reasoning_content`
for reasoning models such as newer Grok models.

## Options

### Option 1: Minimal LiteLLM Fix

Keep the current LiteLLM split:

- `litellm.completion()` for chat and assistant flows.
- `litellm.responses()` for xAI search and URL flows.

Make the xAI cache key explicit per API surface:

- Chat Completions: continue sending `x-grok-conv-id` through `extra_headers`.
- Responses API: send `prompt_cache_key` through `extra_body`.
- Responses logging: extract and log
  `usage.input_tokens_details.cached_tokens`.

This is the lowest-risk fix and should be done first.

### Option 2: Preserve Reasoning Content

Extend conversation history to retain provider reasoning state where LiteLLM
exposes it, especially `response.choices[0].message.reasoning_content`.

This targets xAI's reasoning-model caveat directly, but it touches context
storage and persistence. It needs explicit handling for privacy, retention,
and compatibility with non-xAI providers.

### Option 3: Move xAI Chat to Stateful Responses

Route xAI assistant/chat flows through `litellm.responses()` and use
`previous_response_id` for stateful continuation.

This aligns with xAI's recommended stateful path for reasoning models, but it
is a larger architecture change. The current assistant loop is shaped around
Chat Completions messages, function-call tool turns, timeout stashing, and
conversation history.

## Recommendation

Implement Option 1 first.

Add a helper that returns a stable xAI cache key:

```python
def _xai_cache_key(model: str, channel: str | None) -> str | None:
    if not channel or not LLMService._is_xai_model(model):
        return None
    return f"chan:{channel}"
```

Use that helper in both API paths:

```python
# Chat Completions
cache_key = self._xai_cache_key(model, channel)
if cache_key:
    existing = kwargs.get("extra_headers") or {}
    kwargs["extra_headers"] = {
        **existing,
        "x-grok-conv-id": cache_key,
    }

# Responses API
cache_key = self._xai_cache_key(model, channel)
if cache_key:
    existing = response_kwargs.get("extra_body") or {}
    response_kwargs["extra_body"] = {
        **existing,
        "prompt_cache_key": cache_key,
    }
```

Update Responses usage extraction to include cached tokens:

```python
usage = getattr(response, "usage", None)
details = getattr(usage, "input_tokens_details", None) if usage else None
cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
```

Then log that value instead of hardcoding `cached_tokens=0`.

## Tests

Add focused service tests:

- Chat Completions xAI calls still pass `x-grok-conv-id` via
  `extra_headers`.
- xAI Responses calls pass `prompt_cache_key` via `extra_body`.
- xAI Responses calls do not pass `x-grok-conv-id` via `extra_headers`.
- Responses cached-token extraction reads
  `usage.input_tokens_details.cached_tokens`.
- Existing Responses usage extraction still handles missing usage details.

## Follow-Up

After Option 1 ships, inspect production `completion_timing` logs for xAI:

- If Responses cache hits appear, the transport-level bug is fixed.
- If Chat Completions still miss with stable `prefix_hash`, design Option 2:
  preserve LiteLLM `reasoning_content` in assistant history for xAI reasoning
  models.
- If preserving reasoning state becomes too invasive, revisit Option 3 and
  move xAI assistant turns to stateful LiteLLM Responses.
