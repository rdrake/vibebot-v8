# Unified Assistant Implementation Plan

**Goal:** Replace the special-case `meta` path with a shared assistant
backend that serves PMs, bot-nick mentions, unknown addressed text, and
existing commands through one policy-aware tool substrate.

**Architecture:** A shared `ToolSpec` registry and assistant executor sit
under all entry routes. Explicit commands remain as thin wrappers, while
PMs and nick mentions use planner mode. Grounding is exposed as a leaf
server tool rather than mixed into the main planner call.

**Tech Stack:** Limnoria plugin hooks, LiteLLM, existing persistence and
context layers, existing reminder scheduling helpers

**Design Doc:** `docs/plans/2026-04-11-unified-assistant-design.md`

## Phase 1: Shared Tool Registry and Policy Metadata

### Task 1: Introduce server-side tool policy metadata

**Files:**
- Modify: `plugins/llm/src/llm/meta.py`
- Modify: `plugins/llm/src/llm/service.py`
- Add or modify tests in `plugins/llm/tests/test_meta.py`

**Steps:**

1. Add a `ToolSpec` structure that holds:
   - model-visible schema
   - handler name or callable
   - required capability
   - authenticated-account requirement
   - rate bucket
   - destructive flag
   - route-profile visibility
2. Convert existing assistant tools to registry entries.
3. Update executor dispatch to resolve tools through the registry.
4. Enforce access checks in executor dispatch, not just in prompts.

**Verification:**

```bash
make test
make lint
make typecheck
```

## Phase 2: Unified Facade and First Caller Conversion

### Task 2: Add request context and `assistant_request()`, then convert `@ask` and `invalidCommand`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/service.py`
- Modify tests in `plugins/llm/tests/test_plugin.py`
- Modify tests in `plugins/llm/tests/test_meta.py`
- Modify tests in `plugins/llm/tests/test_commands.py`

**Steps:**

1. Add a request-context structure in `plugin.py` or `service.py`.
2. Build that context from current preflight results plus route metadata.
3. Add `assistant_request()` in `service.py` as the shared entry point.
4. Route `@ask` through the new shared facade.
5. Route `invalidCommand` through the same facade instead of the special
   `meta` path.
6. Preserve no-double-preflight behavior.
7. Keep action formatting, context storage, and usage logging unchanged.
8. Keep the current grounded `completion()` path available behind the
   facade until grounding leaf tools land, so `@ask` does not lose current
   grounding behavior during the transition.

**Verification:**

```bash
make test
make lint
make typecheck
```

## Phase 3: PM and Bot-Nick Mention Routing

### Task 3: Add mention-triggered assistant requests with a config toggle

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/config.py`
- Modify tests in `plugins/llm/tests/test_plugin.py`
- Add or modify tests in `plugins/llm/tests/test_commands.py`

**Steps:**

1. Add exact bot-nick mention detection.
2. Treat PMs as always addressed.
3. Reuse the shared assistant facade with the `chat` profile.
4. Add a `mentionEnabled` config toggle so the behavior can be disabled
   without reverting code.
5. Add dedupe protection so explicit commands and `invalidCommand` do not
   also answer the same line.
6. Preserve the existing all-message context tracking behavior.
7. Extract helper methods from `doPrivmsg` so the new logic does not further
   expand one large method body.

**Verification:**

```bash
make test
make lint
make typecheck
```

## Phase 4: Grounding as a Leaf Tool

### Task 4: Add grounded search and URL fetch assistant tools

**Files:**
- Modify: `plugins/llm/src/llm/service.py`
- Modify shared tool definitions
- Modify tests in `plugins/llm/tests/test_service.py`
- Add focused assistant tool tests

**Steps:**

1. Add a `search_web` tool that internally uses the current grounded
   Gemini path.
2. Add a `fetch_url` tool or equivalent URL-context tool.
3. Return structured JSON with summary, sources, and `grounding_used`.
4. Bubble the grounding flag up to final IRC rendering.
5. Keep provider-native grounding out of the main planner loop.
6. Remove transitional ask-path grounding exceptions only after the leaf
   tools are working end to end.

**Verification:**

```bash
make test
make lint
make typecheck
```

## Phase 5: Command Wrapper Conversion

### Task 5: Convert `@code` and `@draw` into wrapper profiles

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/service.py`
- Modify: `plugins/llm/tests/test_commands.py`
- Modify: `plugins/llm/tests/test_service.py`

**Steps:**

1. Add `generate_code` and `generate_image` tools or route adapters.
2. Enforce existing `@code` and `@draw` policy requirements at the tool
   level as well as the route level.
3. Keep current rendering behavior:
   - code summary plus HTTP link
   - image URL reply
4. Preserve draw authenticated-account requirements and draw-specific rate
   limits.
5. Remove the dedicated code summarization side path where the planner
   already has a final response turn; let the assistant summarize large
   tool outputs from structured tool results instead.

**Verification:**

```bash
make test
make lint
make typecheck
make preflight
```

### Task 6: Keep deterministic wrappers for state commands

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify shared executor helpers if needed
- Modify tests in `plugins/llm/tests/test_commands.py`

**Steps:**

1. Keep `@memories`, `@instruct`, `@forget`, `@remind`, and `@usage`
   as deterministic wrappers.
2. Point them at the shared executor instead of bespoke command-local logic
   where practical.
3. Preserve exact user-facing syntax and error messages unless the new
   behavior is intentionally improved.

**Verification:**

```bash
make test
make lint
make typecheck
```

## Phase 6: Remove the Special `meta` Concept

### Task 7: Demote `@meta` and clean up naming

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/meta.py`
- Modify: `plugins/llm/src/llm/service.py`
- Modify: `docs/guide/reference/commands.md`
- Modify: `docs/guide/user/ai-commands.md`
- Modify any help-generation sources
- Modify tests that mention `meta`

**Steps:**

1. Remove `@meta` from generated help and docs.
2. Keep it as an alias temporarily if backward compatibility is still
   desired.
3. Rename shared internals away from `meta` terminology only after the
   shared assistant facade is stable and already in use.

**Verification:**

```bash
make test
make lint
make typecheck
make docs
```

## Phase 7: Observability and Hardening

### Task 8: Add per-tool logging and denial visibility

**Files:**
- Modify: `plugins/llm/src/llm/service.py`
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify tracing or persistence as needed
- Add tests where practical

**Steps:**

1. Log outer route and profile for every assistant request.
2. Log per-tool allow or deny decisions.
3. Include tool-level rate bucket and policy-denial reason in logs.
4. Decide whether structured logs are sufficient or whether a new
   persistence table is justified.

**Verification:**

```bash
make test
make lint
make typecheck
make preflight
```

## Recommended Order

1. Tool registry with policy metadata
2. Unified facade plus first caller conversion
3. Mention and PM routing with `mentionEnabled`
4. Grounding leaf tools
5. `@code` and `@draw` wrapper conversion
6. Deterministic state-wrapper integration
7. `@meta` demotion plus naming cleanup
8. Observability hardening

## Notes

- Keep route-level preflight in `plugin.py`; do not move capability or auth
  logic into prompts.
- Keep tool-level checks even when the route already passed preflight.
- Avoid silent fallback that strips tools on stateful assistant paths.
- Keep the current grounded ask path working until grounding leaf tools are
  ready, then remove the temporary bridge.
- Prefer incremental migration over a single large rewrite so the current
  command surface stays usable throughout the refactor.
