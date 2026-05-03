---
status: review
date: 2026-05-02
reviewed_plan: docs/plans/2026-05-02-limnoria-tool-bridge-plan.md
reviewer: claude-code-reviewer
---

# Limnoria Tool Bridge — Second Opinion Review

## Codex Review Concurrence

### Finding 1 (High): Capability check signature wrong

**Verdict: Confirmed, but the explanation needs sharpening.**

Codex is correct that the plan's `callbacks.checkCommandCapability(msg, cb, [cmd])` will
assert-fail at runtime, but the precise reason is worth stating exactly.

From `.venv/lib/python3.14/site-packages/supybot/callbacks.py:442-445`:

```python
if not isinstance(commandName, minisix.string_types):
    assert commandName[0] == plugin, (
        'checkCommandCapability no longer accepts command names '
        'that do not start with the callback\'s name ...')
```

`cb.name().lower()` for `Misc` is `"misc"`. The plan iterates
`cb.listCommands()`, which yields `"ping"`, then passes `["ping"]` as the
list-form argument. The assert fires immediately: `"ping" == "misc"` is `False`.

The fix is to either pass the string form `cmd` (works for the leaf check
and is the pattern `_callCommand` uses at line 1591) or pass
`[cb.canonicalName(), cmd]` for the prefixed-list form. Either works;
string form is simpler and mirrors what `_callCommand` does for the
leaf-capability pass before it loops over the full prefix.

One additional precision issue: the plan says "if non-empty → user lacks
capability → skip." The return value of `checkCommandCapability` is one of
three things: `False` (allowed), `True` (blocked by default-deny), or a
non-empty string (specific anti-capability blocked it). "Non-empty" as a
filter would pass `False` (correct) but the phrasing is misleading — `True`
is not "empty." "If truthy" is the correct characterization. This is
language imprecision only; truthiness-based dispatch works.

### Finding 2 (High): Integration target is off

**Verdict: Confirmed and critical.**

The plan's B2/B3 speak of `_build_limnoria_bridge_tool` and a
`_handle_tool_call` dispatcher in `service.py`. Neither exists. The actual
tool loop is in `service.py:2698` which calls
`executor.execute(tc.function.name, args)`, where `executor` is an
`AssistantToolExecutor`. That executor dispatches via
`ASSISTANT_TOOL_REGISTRY` (built from `ASSISTANT_TOOL_SPECS` at
`assistant.py:601-602`), which is a module-level constant tuple populated at
import time from `ASSISTANT_TOOLS` at `assistant.py:582-598`.

There is no existing hook for injecting per-request dynamic tools into either
the tool list or the executor. The plan needs to specify where this
injection happens:

1. `get_tools_for_profile()` at `assistant.py:605` returns from
   `ASSISTANT_TOOL_SPECS`, a static tuple. Dynamic bridge tools cannot be
   added to this without structural changes. The plan must either extend
   `run_meta_completion()` in `service.py` to accept `extra_tools` to append
   to `profile_tools` (line 2592), or add bridge-tool injection into the
   static registry at load time (undesirable — it would need the `irc`/`msg`
   context at module import time, which is impossible).

2. The executor lookup at `assistant.py:695` does
   `ASSISTANT_TOOL_REGISTRY.get(tool_name)`. A bridge tool call would return
   `None` there, resulting in an "Unknown tool" error. The plan must extend
   `executor.execute()` or the loop at `service.py:2698` to handle bridge
   calls as a separate code path.

Codex's recommendation to add `extra_tools` / `extra_handlers` plumbing is
the right architectural answer. Concur.

### Finding 3 (High): Error result shape wrong

**Verdict: Confirmed.**

`service.py:2699-2710` inspects the JSON content for the `"error"` key to
decide whether to set `last_successful_tool`. Plain strings like
`"not permitted"` are not JSON and fall into the `except (json.JSONDecodeError,
TypeError)` branch at line 2706-2708, where `parsed = None`. The `if
isinstance(parsed, dict) and "error" not in parsed` condition is then `False`
(because `None` is not a dict), so `last_successful_tool` is NOT updated — the
error case accidentally avoids the "success" path. However this is narrowly
correct only because of the `isinstance(..., dict)` guard; if the bridge returns
a non-JSON plain string that happens to be treated as a tool result, the model
still sees it as content and will treat it as a successful factual reply.

The real problem is that the model receives the raw string `"not permitted"` or
`"unknown plugin"` as the tool result content. The model has no structural signal
that this is an error versus a legitimate reply. It will typically incorporate
the string into its response text, which is semantically wrong (the bot would
say something like "not permitted" to the user as if that is an answer).

Using `{"error": "not permitted"}` correctly signals the loop's
`last_successful_tool` guard and gives the model a structured error to surface.
Codex's proposed shape `{"status": "ok", "reply": "..."}` / `{"error": "..."}` is
consistent with the existing pattern at `assistant.py:676-683` and must be adopted.

### Finding 4 (Medium): Dispatch arg path — positional not keyword

**Verdict: Confirmed.**

`_callCommand` at `.venv/.../callbacks.py:1575` is declared:

```python
def _callCommand(self, command, irc, msg, *args, **kwargs):
```

The normal call path (line 1213) passes `args` positionally:

```python
cb._callCommand(command, self, self.msg, args)
```

The plan specifies `cb._callCommand([command], proxy, msg, args=tokens)` — a
keyword argument. Since `*args` in the signature captures positional variadics,
passing `args=tokens` via keyword sets a key in `**kwargs`, not `*args`. The
`callCommand` delegation at line 1607 is `self.callCommand(command, irc, msg,
*args, **kwargs)`, which passes the kwargs dictionary through. Some `wrap()`-
based commands inspect `args` from the positional spread — they would receive an
empty positional args list, breaking argument parsing entirely. Pass the token
list positionally: `cb._callCommand([command], proxy, msg, tokens)`.

### Finding 5 (Medium): `enumerate_commands` needs config input

**Verdict: Confirmed, with one refinement.**

The plan's task A3 says "skip if ... not in operator's `bridgeAllowedPlugins`"
without giving `enumerate_commands(irc, msg)` any way to access that config.
Codex is correct that this is an incomplete signature.

The refinement: the plan's B2 notes that `bridgeAllowedPlugins` is a
channel-scoped registry value. The caller in `service.py` already knows the
channel and has access to `self.plugin.registryValue(...)`. The cleanest fix
is to pass the resolved set as a parameter — `enumerate_commands(irc, msg,
allowed_plugins: frozenset[str])` — which also makes the function testable
without depending on the live config. Concur with Codex.

### Finding 6 (Medium): Faulty test case for capability filtering

**Verdict: Confirmed.**

The plan at line 111 proposes `test_enumerate_skips_lacking_capability` with
`Channel.kick`. Since `Channel` is in `DENY_PLUGINS` (plan line 64), the
plugin is filtered before the capability check is ever reached. The test would
pass even if the capability check code were entirely absent. A correct test
needs a non-denied plugin command that has an anti-capability configured on
the test msg. Concur with Codex.

### Finding 7 (Medium): Plugin-level allowlist too coarse

**Verdict: Confirmed, but the severity assessment should be elevated to High
for the specific `Web.fetch` case.**

`Karma.increment` / `Karma.decrement` do mutate state, but they require the
user to type `thing++` / `thing--` — the LLM itself calling those autonomously
is the real risk, and the plan already acknowledges Karma mutates state. The
concern is valid as Medium.

`Web.fetch`, however, is more serious. It fetches arbitrary URLs and returns
raw content, up to `supybot.plugins.Web.fetch.maximum` bytes. An LLM with
autonomous access to `Web.fetch` can be prompted via user input to fetch
internal URLs (e.g. `http://localhost:11434` on a self-hosted Ollama box, or
`http://169.254.169.254/` on a cloud VM). This is a server-side request forgery
(SSRF) vector that capability gating does not prevent — capability gating
answers whether the invoking user may call the command, not whether the URL
is safe for the bot's network context. This should be treated as High severity
and `Web.fetch` should be excluded from Phase 1's default exposure, or require
an explicit opt-in deny-list entry in `DENY_COMMANDS`.

`Web.title` is safe by comparison — it only returns the HTML `<title>` tag.

---

## Additional Findings

### High: `ReplyIrcProxy.__init__` mutates real `irc` object state

The plan says to subclass `callbacks.ReplyIrcProxy` for `BufferingIrcProxy`.
`ReplyIrcProxy.__init__` at `.venv/.../callbacks.py:655` unconditionally calls:

```python
self.getRealIrc()._setMsgChannel(self.msg)
```

`_setMsgChannel` (`.venv/.../irclib.py:1633`) sets `msg.channel` on the
message object:

```python
msg.channel = channel
```

This is not a side-effect on the real `irc` object's internal queue; it
mutates the `msg` attribute of the `IrcMsg` being passed in. Since the
bridge reuses the *original* `msg` (the one the user sent to the LLM plugin),
this will overwrite `msg.channel` in place. If the original `msg` is still
held by the LLM plugin's processing path (and it is — it's on the call stack),
the mutation is benign here, since the channel is already set correctly. But
it is an implicit shared-state mutation that is easy to get wrong if the proxy
constructor is ever called with a different or synthetic msg. The implementation
must document this and, if a synthetic msg is fabricated for bridge use, ensure
`msg.args[0]` is set to the channel name before constructing the proxy, or the
`_setMsgChannel` call will silently set `msg.channel = None`.

Severity: High (silent breakage is possible with a synthetic msg).

### High: `dynamic` scope — enumeration path is safe, but `getCommandHelp` is fragile

`dynamic` at `.venv/.../dynamicScope.py` is a frame-walking scope object, not
a `threading.local`. It resolves attribute access by walking `sys._getframe()`
backwards looking for a local variable named `irc` or `msg` in any enclosing
frame. This means `dynamic.irc` and `dynamic.msg` resolve to whatever `irc`
and `msg` locals exist in the call stack at the time of the call.

The plan calls `cb.getCommandHelp([cmd])` in `enumerate_commands()` to extract
the docstring. `getCommandHelp` at `.venv/.../callbacks.py:1639-1642` reads
`dynamic.msg` and `dynamic.irc` to determine channel and network for a config
lookup (whether to use simple syntax). These `dynamic` attributes fall back to
`None` if the names are not on the stack, and `getCommandHelp` guards them with
`if dynamic.msg is not None`. So `getCommandHelp` degrades gracefully to the
global default when called outside a normal message-dispatch frame — it will
not crash, but it may not return network/channel-specific help text.

However, since `enumerate_commands` is called from inside `service.py`'s
`run_meta_completion()` which has `irc` and `msg` as local variables, those
names will be on the frame stack. The `dynamic` frame-walker will find them.
The help text rendered would therefore pick up the correct channel/network
config. This is actually safe — the plan does not need to explicitly set
`dynamic.irc`/`dynamic.msg`.

Revised severity: Low for the enumeration path. Flag as an awareness note,
not a blocker.

### High: Threading and lock re-entrancy with `__synchronized__` commands

`Commands.__synchronized__` at `.venv/.../callbacks.py:1443-1447` includes
`callCommand` (not `_callCommand` — `_callCommand` is in `__firewalled__`,
not `__synchronized__`). The `MetaSynchronized` wrapper at
`.venv/.../utils/python.py:86-94` uses a per-instance `threading.RLock`.

The key sequence is:
1. User message arrives on the main thread.
2. LLM plugin has `threaded = True` (`plugin.py:424`), so Limnoria spawns a
   `CommandThread` to run `LLM._callCommand` off the main thread.
3. Inside that thread, the bridge calls `other_cb._callCommand(...)`, which
   delegates to `other_cb.callCommand(...)`.
4. `callCommand` is synchronized on `other_cb`'s RLock.

The critical question: is `other_cb`'s `callCommand` already locked from
another thread at that moment? Only if another IRC message already triggered
the same plugin concurrently. With a single IRC connection and Limnoria's
single-main-thread dispatch, concurrent activation of the same plugin is
unlikely in normal operation, but not impossible if the main thread is also
dispatching a message to, say, `Misc` at the same time the bridge calls `Misc`.
The lock is an `RLock`, so if the *same thread* already holds it (re-entrancy
from the same CommandThread), it would succeed. Cross-thread contention would
block, not deadlock. This is a latency concern, not a correctness bug, for the
Phase 1 allowlisted plugins.

More importantly: `finalEval()` at line 1207-1213 checks
`world.isMainThread()` to decide whether to spawn a new `CommandThread`. Since
the bridge calls `_callCommand` directly (bypassing `finalEval`), the
`CommandThread` spawning logic is never triggered. The bridge call runs
synchronously in the current thread. If the target plugin is itself `threaded`
(e.g. `Web`), this means the Web HTTP fetch blocks the LLM's CommandThread
until the fetch completes. That is the correct behavior for the bridge — we
want the result synchronously — but the plan should state it explicitly, and
the timeout handling in `dispatch()` should account for slow network calls.

Severity: Medium (no deadlock, but documentation gap and potential for
slow Web commands to stall the LLM response).

### Medium: `callbacks.tokenize` signature mismatch

The plan says "tokenize `arg_string` via `callbacks.tokenize(arg_string)`."
The actual signature at `.venv/.../callbacks.py:420`:

```python
def tokenize(s, channel=None, network=None):
```

Calling it without `channel` and `network` is fine — it falls back to the
global config for nested-command brackets and pipe syntax. However, bracket
and pipe settings can be channel-specific (line 426:
`nested.brackets.getSpecific(network, channel)()`). For a bot serving a
single channel, this probably does not matter. But if the channel has custom
bracket config, passing `channel` and `network` would produce the correct
tokenization for that channel. The plan should specify:

```python
callbacks.tokenize(arg_string, channel=msg.channel, network=irc.network)
```

Severity: Medium (correctness gap in the presence of non-default nested
command config).

### Medium: `_callCommand` first positional arg must be a list

The plan notes `cb._callCommand([command], proxy, msg, tokens)`. The
`[command]` notation is correct — the `command` argument is expected to be
a list of strings (the command name path), as used throughout the codebase
(line 1210: `args=(command, self, self.msg, args)` where `command` is already
a list). The plan should state this clearly and ensure the list contains only
the leaf name, e.g. `["ping"]` not `["misc", "ping"]`, since `_callCommand`
at line 1583 will prepend the canonical name if needed:

```python
if len(command) == 1 or command[0] != self.canonicalName():
    fullCommandName = [self.canonicalName()] + command
```

So passing `["ping"]` is correct; passing `["misc", "ping"]` is also correct
but redundant. The plan is fine here — just be explicit that `command` is a
single-element list of the leaf name from `listCommands()`.

Severity: Low (clarification only).

### Medium: `bridgeAllowedPlugins` empty-vs-deny interaction is correct but under-specified

The plan says `bridgeAllowedPlugins` defaults to empty, and `DENY_PLUGINS` is
a hard-coded frozenset. The enumeration logic at plan line 71 is:

> Skip if `cb.name()` in `DENY_PLUGINS` or not in operator's
> `bridgeAllowedPlugins`.

If `bridgeAllowedPlugins` is empty (the default), then every plugin fails the
"not in bridgeAllowedPlugins" check — nothing is exposed. This is the safe
default the plan intends ("disabled = bridge tool not registered at all" when
`bridgeEnabled` is False; empty allowlist = nothing exposed when
`bridgeEnabled` is True). The interaction is correct.

The under-specification: the plan does not explicitly say whether an empty
`bridgeAllowedPlugins` with `bridgeEnabled True` yields a no-op or an error.
Based on B2 ("if empty list → return `None`"), it yields a no-op — the bridge
tool is simply not registered with the LLM. That is good behavior and should
be documented explicitly to prevent an implementor from treating the empty-list
case as "expose all non-denied plugins."

Severity: Low (documentation gap only; logic is correct as written).

### Low: `Utilities.apply` and `Utilities.echo` are SSRF-adjacent

`Utilities.echo` simply echoes its input back. `Utilities.apply` takes a
command name and arguments as text and re-dispatches them through Limnoria's
command engine. If `Utilities` is on the allowlist and `Web.fetch` is also
reachable (even via another mechanism), an LLM with access to `apply` can
call `apply fetch <url>` and bypass any `Web.fetch` deny entry. If `Web` is
not loaded at all, this risk is absent. This is an escalation path to be
aware of if both `Utilities` and `Web` are enabled simultaneously.

Severity: Low (multi-step, requires `Web` to be loaded and exploitable by
the LLM via `apply`; worth a comment in the deny list).

---

## Verdict

The plan's architectural direction is sound: a single generic dispatcher
backed by Limnoria's existing capability checks is the right Phase 1
approach. The NL-shim-over-stock-plugins principle is well-reasoned.

However, the plan is **not approvable as written**. It has three blockers
that will cause runtime failures or security gaps without revision:

1. **Integration target** (Finding 2): There is no `_handle_tool_call` hook
   in `service.py`. The plan must specify how dynamic bridge tools are
   injected into `get_tools_for_profile()` and how the executor dispatch loop
   handles them. This requires structural additions to `assistant.py` or
   `service.py` that the plan does not describe.

2. **`checkCommandCapability` signature** (Finding 1): Passing `[cmd]` where
   `cmd` is a leaf name (e.g. `"ping"`) triggers an immediate `AssertionError`.
   Both enumerate and dispatch paths must use the string form or the correctly
   prefixed list form.

3. **`_callCommand` arg passing** (Finding 4) and **`Web.fetch` SSRF**
   (Finding 7 + Additional): The keyword `args=` path is incorrect; and
   `Web.fetch` must be explicitly denied in `DENY_COMMANDS` before Phase 1
   ships, not deferred to "operator judgment."

The plan should be revised to address Findings 1-4 from Codex plus the
`ReplyIrcProxy` mutation note and the `Web.fetch` SSRF elevation before
implementation begins. The remaining findings are non-blocking and can be
addressed in code review during implementation.
