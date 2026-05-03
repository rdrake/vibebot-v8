---
status: review
date: 2026-05-02
reviewed_plan: docs/plans/2026-05-02-limnoria-tool-bridge-plan.md
---

# Limnoria Tool Bridge Plan Review

## Findings

### High: Capability checks are specified incorrectly

The plan calls `callbacks.checkCommandCapability(msg, cb, [cmd])`, but
Limnoria asserts that list-form command names start with the plugin name.
Use `cmd` for the leaf check and mirror `_callCommand`'s plugin/prefix
checks.

Relevant plan section:

- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:73`

Relevant Limnoria behavior:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:440`
- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1575`

### High: The integration target is off

The plan assumes tool assembly and dispatch live in `service.py`, but the
actual registry and executor are centralized in `assistant.py`.
`service.py` calls `get_tools_for_profile()` and `executor.execute()`.

The plan should add dynamic `extra_tools` and `extra_handlers` plumbing
instead of adding a branch to a nonexistent `_handle_tool_call`.

Relevant plan section:

- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:94`

Relevant implementation:

- `plugins/llm/src/llm/assistant.py:605`
- `plugins/llm/src/llm/service.py:2592`
- `plugins/llm/src/llm/service.py:2698`

### High: Raw bridge errors will be misclassified as successful tools

The plan returns plain strings like `"not permitted"` and `"unknown plugin"`.
The assistant loop treats non-JSON tool results as success unless it finds an
`"error"` key.

Bridge results should be JSON shaped:

- Success: `{"status": "ok", "reply": "..."}`
- Failure: `{"error": "..."}`

Relevant plan section:

- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:77`

### Medium: Dispatch should use Limnoria's native positional args path

`dispatch()` should call `_callCommand([command], proxy, msg, tokens)`, not
`args=tokens`. The standard Limnoria command path passes the token list as the
fourth positional argument. Keyword `args=` may work for some wrapped commands,
but it is not the native command path and is riskier for plugin callbacks and
pre-command hooks.

Relevant Limnoria path:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1213`

### Medium: `enumerate_commands()` needs explicit allowed plugin input

The plan says `enumerate_commands(irc, msg)` filters by `bridgeAllowedPlugins`,
but that function has no config or channel input. Either pass
`allowed_plugins` explicitly from `service.py`, or make the bridge module
depend on the LLM plugin config.

Recommendation: pass a normalized set in. It keeps `limnoria_bridge.py`
testable.

Relevant plan section:

- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:68`

### Medium: The proposed capability test can pass for the wrong reason

The test case uses `Channel.kick`, but the plan denies the `Channel` plugin
entirely. That means the test could pass because the plugin was denied, not
because capability filtering works.

Use a stub plugin or a non-denied plugin command with an anti-capability
configured.

Relevant plan sections:

- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:64`
- `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md:111`

### Medium: Plugin-level allowlisting is too coarse for autonomous LLM use

`Karma` mutates state, and `Web.fetch` is broader than `Web.title`.
Capability gating answers whether the invoking user may run a command; it
does not answer whether the command is appropriate for an LLM to choose
autonomously.

Phase 1 should either be read-only by default or include a per-command
deny/allow classification for mutating and external-fetch commands.

## Recommended Plan Changes

The overall direction is sound: a generic dispatcher plus Limnoria capability
checks is the right first move.

Before implementation, revise the plan to:

- Integrate through `assistant.py`'s `ToolSpec` and `AssistantToolExecutor`
  design, with dynamic bridge tool injection from `service.py`.
- Return JSON-shaped tool results so denial and execution errors are not
  treated as successful tool calls.
- Mirror Limnoria's real capability checks and `_callCommand` argument path.
- Pass `bridgeAllowedPlugins` into command enumeration explicitly.
- Replace the `Channel.kick` capability test with a test that isolates
  capability filtering from plugin deny-list filtering.
- Narrow Phase 1's default command exposure to read-only commands, or add
  command-level safety classification.
