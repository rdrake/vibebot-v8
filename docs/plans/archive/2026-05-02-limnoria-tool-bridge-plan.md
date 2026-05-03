---
status: revised-after-codex-and-claude-review
date: 2026-05-02
revisions:
  - 2026-05-02: codex review (`...-review.md`)
  - 2026-05-02: claude code-reviewer review (`...-claude-review.md`)
---

> **Revision note (2026-05-02):** First-draft plan reviewed by Codex (7 findings)
> and Claude code-reviewer (concurrence + 6 additional findings). Three blockers
> required redesign before implementation:
>
> 1. **Integration target** — there is no `_handle_tool_call` in `service.py`;
>    the actual tool path goes through `AssistantToolExecutor.execute()` in
>    `assistant.py` (registry-driven). Plan now specifies `extra_tools` /
>    `extra_handlers` injection plumbing into `run_meta_completion()` and a new
>    side-channel branch in the executor loop in `service.py:2698`.
> 2. **Capability check signature** — list-form arg must start with the plugin
>    name (`callbacks.py:443-445` asserts), so we must pass the **string** form
>    of the command name to `checkCommandCapability` for the leaf check, exactly
>    like `_callCommand` does at `callbacks.py:1591`. The plan's old
>    `[cmd]`-list form would `AssertionError` on first iteration.
> 3. **`Web.fetch` SSRF** — capability gating answers "may this user run it,"
>    not "is this URL safe for the bot's network context." `Web.fetch` is now
>    in the hard-coded `DENY_COMMANDS` regardless of operator allowlist.
>
> Smaller fixes folded in: dispatch uses positional `tokens` (not kwarg
> `args=`); error/success JSON shape matches `assistant.py:676-683`;
> `enumerate_commands` takes `allowed_plugins` as an explicit parameter;
> `tokenize` is called with `channel=` / `network=`; capability test no
> longer uses a deny-listed plugin; Karma drops out of the default starter
> allowlist (mutation risk); `Utilities.apply` is denied to close a
> bypass-via-`apply` escalation path; the `ReplyIrcProxy.__init__` `msg.channel`
> mutation is documented and Phase 1 reuses the original `msg` (no synthesis);
> threading semantics (synchronous bridge call on the LLM `CommandThread`,
> slow stock-plugin commands stall the LLM response, respect the existing
> `timeout` registry value) are now stated explicitly.

# Limnoria Tool Bridge — Phase 1 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose Limnoria's loaded plugin commands to the LLM as natural-language tools, gated by Limnoria's existing capability system, so the bot can answer "@vibebot last seen alice" or "title that URL" by dispatching to a stock plugin command instead of duplicating the feature inside `LLM`.

**Guiding principle (the user's words):** *"Defer as much functionality to built-in Limnoria plugins and only implement what's missing and a natural language shim between them."*

**Architecture:** One PR. No new abstractions beyond what's needed to enumerate, dispatch, and capture replies. The plumbing changes to `assistant.py` / `service.py` are the *only* structural additions; the bridge module itself is leaf code.

## Background

### Loaded plugins today (`bot.conf`)
`Channel`, `Config`, `LLM`, `Owner`, `User`. Nothing else.

### Limnoria's introspection surface
- `irc.callbacks` → `list[Plugin]` of all loaded plugins.
- `cb.name()` / `cb.canonicalName()` — plugin name (canonical is lowercased).
- `cb.listCommands()` → `list[str]` of leaf command names.
- `cb.getCommandMethod([name])` → bound method.
- `cb.isCommandMethod(name)` → bool.
- Method `__doc__` follows a strict convention: **line 1 is argument syntax**, rest is description. Good enough as a natural-language schema for the LLM; no need to parse `wrap()` specs.

### Capability gating (`callbacks.checkCommandCapability`, `callbacks.py:440`)
Limnoria already enforces `-owner`, `-admin`, `-trusted`, `-aka.*`, `-alias.*`, `-scheduler.add/remove/repeat` as default-deny anti-capabilities (`supybot/ircdb.py:1351`). Per-channel overrides set by the operator are honored automatically.

**Important detail (review finding):** `checkCommandCapability` asserts that list-form `commandName[0] == plugin.canonicalName()`. Because `cb.listCommands()` returns leaf names like `"ping"`, calling it as `checkCommandCapability(msg, cb, ["ping"])` triggers `AssertionError`. We follow `_callCommand`'s own pattern at `callbacks.py:1591`: pass the **string** form of the leaf command name. Return value is `False` (allowed), `True` (default-deny), or a non-empty string (specific anti-cap blocked it). Filter on truthiness.

### Reply capture
Stock commands call `irc.reply()` directly. `callbacks.ReplyIrcProxy` (`supybot/callbacks.py:642`) is the right base for a buffering proxy: subclass and override `reply()` / `error()` to append to a list.

**Important detail (review finding):** `ReplyIrcProxy.__init__` calls `self.getRealIrc()._setMsgChannel(self.msg)` (`callbacks.py:655`), which mutates `msg.channel` based on `msg.args[0]`. For Phase 1 we always reuse the **original** `msg` (the one the user sent to the LLM plugin), so `msg.channel` is already correct and the mutation is a no-op. **Do not** fabricate a synthetic `msg` for the proxy in Phase 1; if a future phase needs to, `msg.args[0]` must be set to the channel name first.

### Threading semantics
The LLM plugin runs `_callCommand` on a `CommandThread` (`plugin.py:424` sets `threaded = True`). Our bridge calls `target_cb._callCommand(...)` synchronously inside that same thread. `_callCommand` is in `__firewalled__` (not `__synchronized__`); `callCommand` is `__synchronized__` on a per-instance `RLock`, so cross-thread contention with the same target plugin would block (not deadlock — `RLock` is re-entrant for the same thread). The bridge bypasses `finalEval()`'s `world.isMainThread()` check, so target-plugin `threaded=True` does *not* spawn a fresh thread. Result: a slow `Web.title` HTTP fetch stalls the LLM's `CommandThread` until completion. Mitigation: the existing `supybot.plugins.LLM.timeout` registry value already bounds the litellm call; we additionally rely on the target plugin's own timeouts (e.g. `supybot.plugins.Web.timeout`).

### Existing tool plumbing — verified
- `assistant.py:582-602`: `ASSISTANT_TOOL_SPECS` is a static tuple built at import time, keyed into `ASSISTANT_TOOL_REGISTRY: dict[str, ToolSpec]`.
- `assistant.py:605-620`: `get_tools_for_profile(route_profile, *, exclude)` returns schemas for the model.
- `assistant.py:685-703`: `AssistantToolExecutor.execute(tool_name, arguments)` looks up `ASSISTANT_TOOL_REGISTRY.get(tool_name)` — returns `_err("Unknown tool: ...")` if missing.
- `assistant.py:676-683`: `_ok(message)` / `_err(message)` — JSON envelope helpers.
- `service.py:2592`: `profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)`.
- `service.py:2698`: `tool_result = executor.execute(tc.function.name, args)`.
- `service.py:2705-2710`: parses `tool_result.content` as JSON and updates `last_successful_tool` only if the parsed dict has no `"error"` key.

There is no existing dynamic-tool injection hook. Phase 1 adds one.

## Scope (Phase 1)

**In scope:**
1. Operator enables a small allowlist of safe stock plugins in `bot.conf`. Recommended starter set (read-only, no-mutation): `Misc`, `Time`, `Math`, `Utilities`, `Seen`. The user confirms before we ship. (`Web` is also reasonable but requires the `Web.fetch` deny rule to be in place — see DENY_COMMANDS below.)
2. New module `plugins/llm/src/llm/limnoria_bridge.py` with command enumeration, dispatch, and `BufferingIrcProxy`.
3. New plumbing in `assistant.py` and `service.py`:
   - `run_meta_completion(...)` accepts `extra_tools: list[dict] | None = None` and `extra_handlers: dict[str, Callable[[dict], ToolResult]] | None = None`.
   - `profile_tools` at `service.py:2592` becomes `get_tools_for_profile(...) + (extra_tools or [])`.
   - The dispatch branch at `service.py:2698` becomes:
     ```python
     if extra_handlers and tc.function.name in extra_handlers:
         tool_result = extra_handlers[tc.function.name](args)
     else:
         tool_result = executor.execute(tc.function.name, args)
     ```
4. Single LLM tool exposed: `run_limnoria_command(plugin: str, command: str, args: str)`. The LLM picks `plugin` + `command` + a single freeform `args` string — Limnoria's own tokenizer and `wrap()` specs validate the args. Tool result is a JSON envelope (see "Result shape").
5. Operator-facing per-channel registry values in `plugins/llm/src/llm/config.py`:
   - `bridgeEnabled: Boolean(False)`
   - `bridgeAllowedPlugins: SpaceSeparatedListOfStrings([])`
6. Hard-coded deny lists in `limnoria_bridge.py` (defense in depth on top of operator allowlist + Limnoria capability checks). See DENY_PLUGINS / DENY_COMMANDS below.
7. Tests: capability denial, plugin deny, command deny, reply capture, error path, no-such-plugin, no-such-command, malformed arg path, JSON envelope correctness.

**Out of scope (deferred to Phase 2 — see end of doc):**
- Replacing the LLM plugin's reminder/scheduler with `Scheduler` / `Later`.
- Auto-generating one LLM tool per Limnoria command (richer schema, more tokens).
- Mapping the LLM's existing `generate_image`, web fetch, etc. to Limnoria equivalents.
- Exposing mutating commands like `Karma.increment` / `Karma.decrement`.

## Why one generic dispatcher (not one tool per command) for Phase 1

A typical Limnoria deploy has 30+ commands across the allowlisted plugins. One tool per command balloons the system prompt by thousands of tokens on every request. The single-dispatcher form costs ~1 tool definition; the LLM picks plugin + command from a list we render once into the description. We can revisit per-command tools later if we observe the LLM picking poorly with the generic form.

## Result shape (JSON envelope)

All bridge results are JSON, matching the existing `_ok` / `_err` contract used by `AssistantToolExecutor` (`assistant.py:676-683`):

| Outcome | JSON |
| --- | --- |
| Success with reply text | `{"status": "ok", "reply": "<captured reply>"}` |
| Success with no reply (e.g. silent ack) | `{"status": "ok", "reply": ""}` |
| Unknown plugin | `{"error": "unknown plugin: <name>"}` |
| Unknown command | `{"error": "unknown command: <plugin>.<cmd>"}` |
| Capability denied | `{"error": "not permitted: <plugin>.<cmd>"}` |
| Deny-listed | `{"error": "denied: <plugin>.<cmd>"}` |
| Uncaught exception in target | `{"error": "<exception message>"}` |

This ensures `service.py:2705-2710`'s `last_successful_tool` guard correctly distinguishes errors (no update) from successes (update), matching how every other tool in the codebase behaves.

## Hard-coded safety lists

In `limnoria_bridge.py` module scope:

```python
DENY_PLUGINS: frozenset[str] = frozenset({
    # Auth / management — capability checks already block non-owners, but
    # we deny at the bridge layer too so the LLM never sees these as options.
    "LLM", "Owner", "Admin", "Config", "Channel", "User",
})

DENY_COMMANDS: frozenset[tuple[str, str]] = frozenset({
    # Pastebin/scrollback — interactive only, no value via LLM.
    ("misc", "more"),
    ("misc", "clearmores"),
    # SSRF vector: arbitrary URL fetch with bot's network privileges.
    # Capability gating does NOT prevent this; deny unconditionally.
    ("web", "fetch"),
    # `apply <command> <args>` re-dispatches through Limnoria's command
    # engine, which would bypass our per-command deny entries.
    ("utilities", "apply"),
})
```

Both lists use `cb.canonicalName()` (lowercase) for matching.

## Tasks

### A — Foundation

**A1. New file: `plugins/llm/src/llm/limnoria_bridge.py`**

Module exports:

- `DENY_PLUGINS: frozenset[str]` — as above.
- `DENY_COMMANDS: frozenset[tuple[str, str]]` — as above.
- `class BufferingIrcProxy(callbacks.ReplyIrcProxy)` — overrides `reply(s, msg=None, **kwargs)` and `error(s, msg=None, **kwargs)` to append to `self.buffer: list[str]` and return without queueing IRC traffic. Other rich-reply methods inherit and route through these.
- `@dataclass class BridgeCommand` — `plugin: str`, `command: str`, `arg_syntax: str`, `description: str`.
- `enumerate_commands(irc, msg, allowed_plugins: frozenset[str]) -> list[BridgeCommand]` — see A2.
- `dispatch(irc, msg, plugin: str, command: str, arg_string: str) -> dict[str, Any]` — see A3. Returns the parsed envelope dict (caller `json.dumps` it into the `ToolResult`).

**A2. `enumerate_commands(irc, msg, allowed_plugins)`**

```
for cb in irc.callbacks:
    if cb.name() in DENY_PLUGINS: continue
    if cb.name() not in allowed_plugins: continue
    for cmd in cb.listCommands():
        if (cb.canonicalName(), cmd) in DENY_COMMANDS: continue
        # Capability check: pass the *string* form of the leaf name.
        # Limnoria's _callCommand uses this exact pattern at line 1591.
        denial = callbacks.checkCommandCapability(msg, cb, cmd)
        if denial:  # True or non-empty string → blocked
            continue
        method = cb.getCommandMethod([cmd])
        doc = (method.__doc__ or "").strip().splitlines()
        arg_syntax = doc[0].strip() if doc else ""
        description = " ".join(line.strip() for line in doc[1:]).strip()
        yield BridgeCommand(cb.name(), cmd, arg_syntax, description)
```

If `allowed_plugins` is empty → returns empty list (the tool is then not registered with the LLM at all; see B2).

**A3. `dispatch(irc, msg, plugin, command, arg_string)`**

```
cb = irc.getCallback(plugin)
if cb is None: return {"error": f"unknown plugin: {plugin}"}
if cb.name() in DENY_PLUGINS: return {"error": f"denied: {plugin}.{command}"}
if (cb.canonicalName(), command) in DENY_COMMANDS:
    return {"error": f"denied: {plugin}.{command}"}
if not cb.isCommandMethod(command):
    return {"error": f"unknown command: {plugin}.{command}"}
denial = callbacks.checkCommandCapability(msg, cb, command)
if denial: return {"error": f"not permitted: {plugin}.{command}"}

proxy = BufferingIrcProxy(irc, msg)
tokens = callbacks.tokenize(arg_string, channel=msg.channel, network=irc.network)
try:
    cb._callCommand([command], proxy, msg, tokens)  # POSITIONAL — not args=tokens
except Exception as exc:
    return {"error": str(exc) or exc.__class__.__name__}
return {"status": "ok", "reply": "\n".join(proxy.buffer)}
```

Notes:
- `[command]` is a single-element list of the leaf name (e.g. `["ping"]`). `_callCommand` at `callbacks.py:1583` will prepend the plugin's canonical name itself if needed.
- `tokens` passed positionally as the fourth argument matches the normal call path at `callbacks.py:1213` (`cb._callCommand(command, self, self.msg, args)` where `args` is positional).
- `callbacks.tokenize(arg_string, channel=..., network=...)` honors per-channel bracket and pipe config (`callbacks.py:420-426`).

### B — Plugin integration

**B1. Registry values in `plugins/llm/src/llm/config.py`**

Near the existing per-channel LLM values:

```python
conf.registerChannelValue(
    LLM, "bridgeEnabled",
    registry.Boolean(False, _("""When True, expose loaded Limnoria plugin
    commands to the LLM as a tool, restricted by bridgeAllowedPlugins and
    Limnoria's capability system. Default off.""")),
)
conf.registerChannelValue(
    LLM, "bridgeAllowedPlugins",
    registry.SpaceSeparatedListOfStrings([], _("""Space-separated list of
    Limnoria plugin names whose commands the LLM may call when bridgeEnabled
    is True. Empty = no commands exposed (the bridge tool is not registered
    with the LLM at all). Recommended starter set: Misc Time Math Utilities
    Seen.""")),
)
```

**B2. Bridge-tool builder in `plugins/llm/src/llm/service.py`**

New helper method `_build_bridge_tool(irc, msg, channel) -> tuple[dict | None, dict[str, Callable] | None]`:

- If `not self.registryValue("bridgeEnabled", channel)` → return `(None, None)`.
- `allowed = frozenset(self.registryValue("bridgeAllowedPlugins", channel) or [])`.
- If `not allowed` → return `(None, None)`.
- `commands = limnoria_bridge.enumerate_commands(irc, msg, allowed)`.
- If `not commands` → return `(None, None)`.
- Build the OpenAI/Anthropic tool schema:
  ```python
  schema = {
      "type": "function",
      "function": {
          "name": "run_limnoria_command",
          "description": "Run a Limnoria plugin command on the user's behalf. "
                         "Available commands:\n" + _render_command_table(commands),
          "parameters": {
              "type": "object",
              "properties": {
                  "plugin": {"type": "string", "description": "Plugin name."},
                  "command": {"type": "string", "description": "Command name."},
                  "args": {"type": "string", "description": "Argument string."},
              },
              "required": ["plugin", "command", "args"],
          },
      },
  }
  ```
- Build the handler closure:
  ```python
  def handler(arguments: dict[str, Any]) -> ToolResult:
      envelope = limnoria_bridge.dispatch(
          irc, msg,
          plugin=arguments.get("plugin", ""),
          command=arguments.get("command", ""),
          arg_string=arguments.get("args", ""),
      )
      return ToolResult(content=json.dumps(envelope))
  ```
- Return `(schema, {"run_limnoria_command": handler})`.

**B3. Wire into `run_meta_completion` (`service.py`)**

- Add parameters: `extra_tools: list[dict] | None = None`, `extra_handlers: dict[str, Callable[[dict], ToolResult]] | None = None`.
- At line 2592:
  ```python
  profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)
  if extra_tools:
      profile_tools = profile_tools + extra_tools
  ```
- At the dispatch branch around line 2698:
  ```python
  if extra_handlers and tc.function.name in extra_handlers:
      tool_result = extra_handlers[tc.function.name](args)
  else:
      tool_result = executor.execute(tc.function.name, args)
  ```
- The `last_successful_tool` parsing at lines 2705-2710 is unchanged — the JSON envelope from `dispatch()` already conforms to the `{"status": "ok"}` / `{"error": ...}` contract.

**B4. Caller in the chat profile**

Identify the existing call site that invokes `run_meta_completion` for the chat route profile (the one that today builds the `executor` with `set_reminder_fn` etc.). Before calling, build the bridge tool:

```python
bridge_schema, bridge_handlers = self._build_bridge_tool(irc, msg, channel)
extra_tools = [bridge_schema] if bridge_schema else None
result = self.run_meta_completion(
    ...,  # existing args
    extra_tools=extra_tools,
    extra_handlers=bridge_handlers,
)
```

For Phase 1, only wire this into the chat profile. Other profiles (`code`, `remind_action`) do not get the bridge.

### C — Tests

**C1. `plugins/llm/tests/test_limnoria_bridge.py`** — new file.

Use the existing test harness (`conftest.py` builds a fake `irc` / `msg`). Test cases:

- `test_enumerate_skips_deny_plugins` — Owner is loaded but is in `DENY_PLUGINS`; not enumerated even when explicitly added to `allowed_plugins`.
- `test_enumerate_skips_deny_commands` — `Web.fetch` not enumerated even when `Web` is in `allowed_plugins`.
- `test_enumerate_skips_lacking_capability` — register a stub plugin `StubPlugin` (allowed, not deny-listed) with a command `restricted` for which the test `msg.prefix` lacks the required capability. Configure an anti-capability against the test user and assert `enumerate_commands` does not yield the `restricted` command. **Do not use `Channel.kick`** — `Channel` is in `DENY_PLUGINS` and the test would pass for the wrong reason.
- `test_enumerate_yields_command_when_authorized` — same stub plugin without anti-cap; command appears.
- `test_enumerate_empty_allowed_plugins_returns_empty` — explicit guard.
- `test_dispatch_captures_reply` — call `Misc.ping` with a real `Misc` plugin instance; assert envelope is `{"status": "ok", "reply": "pong"}` (or whatever Misc actually returns).
- `test_dispatch_unknown_plugin` — envelope is `{"error": "unknown plugin: ..."}`.
- `test_dispatch_unknown_command` — envelope is `{"error": "unknown command: ..."}`.
- `test_dispatch_capability_denied` — envelope is `{"error": "not permitted: ..."}`.
- `test_dispatch_command_deny_listed` — `dispatch(..., plugin="web", command="fetch", ...)` returns `{"error": "denied: web.fetch"}` even when Web is in `allowed_plugins` (defense-in-depth at dispatch time).
- `test_dispatch_argument_error` — passing wrong args → target plugin's `error()` captured into the buffer; envelope is `{"status": "ok", "reply": "<help text>"}` because Limnoria's argument-error path calls `irc.reply(help)` not `irc.error()` (`callbacks.py:1612-1621`). The buffer captures it as a reply. This is acceptable behavior — the LLM gets the help string back and can re-call.
- `test_dispatch_uncaught_exception_returns_error` — patch the target method to raise; envelope is `{"error": "..."}`, no exception propagates.
- `test_buffering_proxy_does_not_queue_irc_messages` — assert `irc.queueMsg` is NOT called during dispatch.
- `test_buffering_proxy_does_not_clobber_msg_channel` — pre-set `msg.channel`, dispatch, assert `msg.channel` unchanged after.

**C2. Service integration tests** — add to `plugins/llm/tests/test_service.py`:

- `test_extra_tools_appended_to_profile_tools` — call `run_meta_completion` with `extra_tools=[fake_schema]`; assert litellm.completion received it in the `tools` kwarg.
- `test_extra_handlers_dispatched_before_executor` — fake LLM response chooses `run_limnoria_command`; assert `extra_handlers["run_limnoria_command"]` was called and the executor was NOT.
- `test_extra_handlers_error_envelope_does_not_set_last_successful_tool` — handler returns `{"error": ...}` envelope; assert `last_successful_tool` remains None.
- `test_extra_handlers_ok_envelope_sets_last_successful_tool` — handler returns `{"status": "ok", "reply": "x"}`; assert `last_successful_tool == "run_limnoria_command"`.

**C3. Existing tests** — `uv run pytest plugins/llm/tests` must remain green.

### D — Docs

**D1. Operator doc** — append a section to the appropriate file under `docs/guide/operator/` (`tuning-monitoring.md` is a possible host) explaining `bridgeEnabled` / `bridgeAllowedPlugins`, the recommended starter set, and the deny lists. Note: the operator must also load the target plugins via Limnoria's normal `load <plugin>` flow; the bridge can only enumerate plugins that are actually loaded.

**D2. AGENTS.md** — one-line addition pointing at `limnoria_bridge.py` if there is a feature/module catalog.

## Validation (before merge)

1. `uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v` — all green.
2. `uv run pytest plugins/llm/tests` — no regressions.
3. Manual smoke against a dev bot:
   - Load `Misc` and `Time` via `load Misc` / `load Time`.
   - In `#test`: `config channel #test plugins.LLM.bridgeEnabled True`.
   - `config channel #test plugins.LLM.bridgeAllowedPlugins Misc Time`.
   - Ask: "@vibebot ping the bot" → should pick `Misc.ping`, return `"pong"`.
   - Ask: "@vibebot what time is it in UTC" → should pick `Time.time` or similar.
   - As non-op (test from a different account), ask anything that hits a default-denied command (`Owner.load Foo`); confirm the LLM sees no such tool / dispatch returns `not permitted`.
4. Confirm `msg.channel` is preserved through dispatch (assert in C1 covers this; manually verify by checking logs).
5. Confirm no IRC traffic for buffered replies — no double-reply in `#test` when the LLM uses the bridge.
6. Check `supybot.plugins.LLM.timeout` is honored — set it low and run a Web command that's slow; LLM responds with its own timeout error before the stalled Web call returns.

## Open questions for code review

1. **Default `bridgeAllowedPlugins`** — ship empty (operator must opt-in plugin by plugin) or with a "safe" default like `Misc Time`? Current plan: empty.
2. **Truncation of `reply` field** — should `dispatch()` truncate the buffered reply to a sane length before returning? Long buffered replies waste LLM tokens. Current plan: no truncation; the LLM plugin's existing reply-shortening (long-reply linker) handles output sizing on its own side. Worth confirming this is sufficient.
3. **`_render_command_table()` format** — Markdown table? Plain `plugin.command — arg_syntax` lines? Plain lines are likely cheapest for the LLM to parse.
4. **Multiple bridge tool calls in one turn** — the loop already handles this by iterating `message.tool_calls`. No additional work required, but worth a regression test.
5. **Error string sanitization** — `dispatch()` returns `str(exc)` in the uncaught-exception path. If a target plugin's exception message contains user input, this is fine (it's already inside a JSON envelope), but worth a quick check that we're not leaking internal paths or tracebacks into the LLM context.

---

## Phase 2 (deferred — separate plan)

**Goal:** Replace LLM plugin features with Limnoria-native ones where the Limnoria version is at least as good.

**Candidate replacements:**
- `Later.tell` for offline messaging — LLM plugin currently has nothing here, this is pure addition.
- `Note.send` for registered-user notes — same, addition not replacement.
- `Seen` for last-seen — replaces nothing in LLM plugin currently (Phase 1 already covers via the bridge).
- `Karma` — mutating; deferred until we have a per-command "mutating: bool" classification beyond plugin-level allowlists.
- `Web.title` — safe to expose in Phase 1 if `Web.fetch` is in `DENY_COMMANDS` (which it is). May graduate from "operator opt-in" to a Phase-1.5 default.
- `Scheduler` — does **not** replace the LLM reminder system. The LLM scheduler is richer (NL parsing, RRULE, watch mode, structured DB rows). Phase 2 should not delete it. Possibly: expose `Scheduler.add` for the niche of "run an exact bot command in N seconds" while keeping the LLM scheduler for everything natural-language.

**Phase 2 design questions to defer:**
- One LLM tool per Limnoria command (richer schema, more tokens) vs. the generic dispatcher? Decision deferred until we observe how the LLM picks with the generic form in Phase 1.
- Per-command mutation classification, so the bridge can default-deny mutating commands even when their plugin is allowlisted.
- Should the LLM's existing tools (`generate_image`, web fetch) be retired in favor of stock plugins where coverage overlaps?
- Migration path for users with reminders in the LLM DB if any reminder logic moves to `Scheduler` — probably none, keep both; LLM owns natural-language reminders, Scheduler owns operator-set crons.

Phase 2 is a separate PR after Phase 1 has soaked in production.
