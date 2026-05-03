---
status: ready-to-execute
date: 2026-05-02
revisions:
  - 2026-05-02: codex implementation-plan review folded in (BufferingIrcProxy.error Raise=True, tokenize inside try, real-registry config test, concrete B2/C2 test bodies, make preflight in validation)
design_plan: docs/plans/2026-05-02-limnoria-tool-bridge-plan.md
codex_design_review: docs/plans/2026-05-02-limnoria-tool-bridge-review.md
claude_design_review: docs/plans/2026-05-02-limnoria-tool-bridge-claude-review.md
codex_implementation_review: docs/plans/2026-05-02-limnoria-tool-bridge-implementation-review.md
---

# Limnoria Tool Bridge — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose loaded Limnoria plugin commands to the LLM as a single natural-language tool, gated by Limnoria's own capability system plus operator allowlist plus hard-coded deny lists, so the bot can defer to stock plugins (`Misc.ping`, `Time.time`, `Seen.seen`, etc.) instead of reimplementing them inside the LLM plugin.

**Guiding principle (the user's words):** *"Defer as much functionality to built-in Limnoria plugins and only implement what's missing and a natural language shim between them."*

**Architecture:** One PR. New leaf module `plugins/llm/src/llm/limnoria_bridge.py` does enumeration, dispatch, and reply buffering. Two narrow plumbing changes thread `extra_tools` and `extra_handlers` through `assistant_request` → `assistant_completion` so the bridge can inject a per-request tool without touching the static `ASSISTANT_TOOL_SPECS` registry. Two operator-facing per-channel registry values (`bridgeEnabled`, `bridgeAllowedPlugins`) gate the whole thing; both default off / empty.

**Tech stack:** Python 3.14, Limnoria (`supybot.callbacks`), litellm tool-calling JSON schema, pytest. All build/test commands run via `uv run`.

**Naming note:** The design plan refers to `run_meta_completion`. The actual method in this codebase is `LLMService.assistant_completion` (called from the `assistant_request` facade at `service.py:1932`). Tasks below use the real names.

---

## Pre-flight (do first, do not skip)

### Task 0: Verify codebase facts before touching anything

**Step 0.1: Confirm the integration points are still where the design plan says.**

Run these and read the matched lines:

```bash
grep -n "ASSISTANT_TOOL_SPECS\|ASSISTANT_TOOL_REGISTRY\|def get_tools_for_profile\|class AssistantToolExecutor\|def execute" plugins/llm/src/llm/assistant.py
grep -n "def assistant_completion\|def assistant_request\|profile_tools = get_tools_for_profile\|tool_result = executor.execute" plugins/llm/src/llm/service.py
```

Expected (from the design plan, verified 2026-05-02):
- `assistant.py`: `ASSISTANT_TOOL_SPECS` ~line 601, `ASSISTANT_TOOL_REGISTRY` ~line 602, `get_tools_for_profile` ~line 605, `AssistantToolExecutor.execute` ~line 685, `_ok` / `_err` ~lines 676-683.
- `service.py`: `assistant_request` ~line 1932, `assistant_completion` ~line 2459, `profile_tools = get_tools_for_profile(...)` ~line 2592, `tool_result = executor.execute(...)` ~line 2698, JSON-error guard updating `last_successful_tool` ~lines 2705-2710.

If the line numbers have drifted (e.g. another commit landed first), update the task references below before implementing. The *symbols* are the truth, not the line numbers.

**Step 0.2: Confirm baseline tests are green.**

```bash
uv run pytest plugins/llm/tests -q
```

Expected: all green. If anything fails, stop and report — fixing pre-existing failures is not in scope.

**Step 0.3: Commit nothing. This task is read-only.**

---

## A — Foundation: the bridge module

### Task A1: Stub `limnoria_bridge.py` with imports, deny lists, and dataclass

**Files:**
- Create: `plugins/llm/src/llm/limnoria_bridge.py`

**Step 1: Write the failing test.**

Create `plugins/llm/tests/test_limnoria_bridge.py`:

```python
"""Tests for the Limnoria tool bridge."""

from __future__ import annotations

import pytest


def test_module_exposes_deny_lists_and_dataclass():
    from llm import limnoria_bridge as lb

    assert isinstance(lb.DENY_PLUGINS, frozenset)
    assert "LLM" in lb.DENY_PLUGINS
    assert "Owner" in lb.DENY_PLUGINS
    assert "Admin" in lb.DENY_PLUGINS
    assert "Config" in lb.DENY_PLUGINS
    assert "Channel" in lb.DENY_PLUGINS
    assert "User" in lb.DENY_PLUGINS

    assert isinstance(lb.DENY_COMMANDS, frozenset)
    assert ("misc", "more") in lb.DENY_COMMANDS
    assert ("misc", "clearmores") in lb.DENY_COMMANDS
    assert ("web", "fetch") in lb.DENY_COMMANDS
    assert ("utilities", "apply") in lb.DENY_COMMANDS

    cmd = lb.BridgeCommand(plugin="Misc", command="ping", arg_syntax="", description="takes no arguments")
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"
```

**Step 2: Run it; verify it fails.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py::test_module_exposes_deny_lists_and_dataclass -v
```

Expected: `ModuleNotFoundError: No module named 'llm.limnoria_bridge'`.

**Step 3: Create the module.**

```python
"""Limnoria → LLM tool bridge.

Exposes loaded Limnoria plugin commands to the LLM as a single
``run_limnoria_command`` tool. Enforces a layered denial model:

1. Hard-coded ``DENY_PLUGINS`` / ``DENY_COMMANDS`` (this module).
2. Operator-set ``bridgeAllowedPlugins`` (per-channel registry).
3. Limnoria's own capability system via ``checkCommandCapability``.

See docs/plans/2026-05-02-limnoria-tool-bridge-plan.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass

# Plugin names matched against ``cb.name()`` (the user-facing CamelCase form).
DENY_PLUGINS: frozenset[str] = frozenset({
    # Auth / management — capability checks already gate non-owners, but we
    # deny at the bridge layer too so the LLM never sees these as options.
    "LLM", "Owner", "Admin", "Config", "Channel", "User",
})

# (canonical_plugin_name, leaf_command) tuples. Both lowercase — matched
# against ``cb.canonicalName()`` (already lowercase) and the leaf name from
# ``cb.listCommands()``.
DENY_COMMANDS: frozenset[tuple[str, str]] = frozenset({
    # Pastebin/scrollback — interactive only, no value via LLM.
    ("misc", "more"),
    ("misc", "clearmores"),
    # SSRF vector: arbitrary URL fetch with the bot's network privileges.
    # Capability gating answers "may this user run it," not "is this URL
    # safe for the bot's network context." Deny unconditionally.
    ("web", "fetch"),
    # ``apply <command> <args>`` re-dispatches through Limnoria's command
    # engine, which would bypass our per-command deny entries.
    ("utilities", "apply"),
})


@dataclass(frozen=True)
class BridgeCommand:
    """One enumerated, callable Limnoria command."""

    plugin: str           # cb.name() — CamelCase, used in operator config
    command: str          # leaf command name from cb.listCommands()
    arg_syntax: str       # first line of method.__doc__
    description: str      # remaining lines of method.__doc__, joined
```

**Step 4: Run the test; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py::test_module_exposes_deny_lists_and_dataclass -v
```

Expected: PASS.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): scaffold Limnoria tool bridge module"
```

---

### Task A2: `BufferingIrcProxy` — capture replies without queueing IRC traffic

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py` (add class)
- Modify: `plugins/llm/tests/test_limnoria_bridge.py` (add tests)

**Background:** Stock commands call `irc.reply(s)` / `irc.error(s)`. `callbacks.ReplyIrcProxy` (`.venv/lib/python3.14/site-packages/supybot/callbacks.py:642`) is the right base class — its `__init__` calls `self.getRealIrc()._setMsgChannel(self.msg)` (line 655) which mutates `msg.channel` from `msg.args[0]`. Since Phase 1 always reuses the *original* `msg` (whose `channel` is already correct), the mutation is a no-op. **Do not** synthesise a new `msg` for the proxy.

**Important behavior to preserve:** `ReplyIrcProxy.error(..., Raise=True)` raises `callbacks.Error` (`.venv/.../callbacks.py:675`). Plugins use `Raise=True` for early-exit flow control; if our override silently appends and returns, the calling command continues past what should have been a hard stop, producing false success or unintended mutation. The override must capture the text **and** still raise `callbacks.Error` when `Raise=True`. The `dispatch()` `try/except Exception` in Task A4 already swallows the raise into the `{"error": ...}` envelope.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_limnoria_bridge.py`:

```python
def test_buffering_proxy_captures_reply_text(mocker):
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    proxy.reply("hello world")
    proxy.reply("second line")

    assert proxy.buffer == ["hello world", "second line"]


def test_buffering_proxy_captures_error_text(mocker):
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    proxy.error("nope")

    assert proxy.buffer == ["nope"]


def test_buffering_proxy_does_not_queue_irc_traffic(mocker):
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    proxy.reply("captured")

    real_irc.queueMsg.assert_not_called()
    real_irc.sendMsg.assert_not_called()


def test_buffering_proxy_preserves_msg_channel(mocker):
    """ReplyIrcProxy.__init__ sets msg.channel from msg.args[0]; reusing
    the original msg means the channel value is unchanged."""
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    BufferingIrcProxy(real_irc, msg)

    assert msg.channel == "#test"


def test_buffering_proxy_error_raise_true_still_raises(mocker):
    """error(Raise=True) must raise callbacks.Error so command flow stops.

    Many plugins use error(..., Raise=True) for early-exit; swallowing it
    silently lets the command continue past what should be a hard stop.
    """
    from supybot import callbacks
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    with pytest.raises(callbacks.Error):
        proxy.error("nope", Raise=True)
    # Buffer still captured the text before raising.
    assert proxy.buffer == ["nope"]


def test_buffering_proxy_error_default_does_not_raise(mocker):
    """error() without Raise=True must NOT raise — only buffer the text."""
    from llm.limnoria_bridge import BufferingIrcProxy

    real_irc = mocker.MagicMock()
    real_irc.network = "testnet"
    msg = mocker.MagicMock()
    msg.args = ("#test", "trigger")
    msg.channel = "#test"

    proxy = BufferingIrcProxy(real_irc, msg)
    proxy.error("nope")  # default Raise=False
    assert proxy.buffer == ["nope"]
```

**Step 2: Run; verify they fail with `ImportError: cannot import name 'BufferingIrcProxy'`.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k buffering_proxy
```

**Step 3: Implement `BufferingIrcProxy`.**

Add to `plugins/llm/src/llm/limnoria_bridge.py`:

```python
from supybot import callbacks


class BufferingIrcProxy(callbacks.ReplyIrcProxy):
    """An ``IrcProxy`` that captures replies into a list instead of
    queueing them onto the IRC connection.

    All the rich-reply machinery (``reply``, ``error``, ``replies``,
    ``replySuccess``, etc.) flows through ``reply()`` and ``error()``
    in the base class, so overriding those two is sufficient.
    """

    def __init__(self, irc, msg):
        super().__init__(irc, msg)
        self.buffer: list[str] = []

    def reply(self, s, msg=None, **kwargs):  # noqa: ARG002 (signature compat)
        self.buffer.append(s)
        return None

    def error(self, s, msg=None, **kwargs):  # noqa: ARG002 (signature compat)
        # Buffer the text first so the dispatch error envelope can include
        # it on the Raise=True path (the exception is caught in dispatch()).
        self.buffer.append(s)
        if kwargs.get("Raise"):
            # Preserve ReplyIrcProxy.error()'s control-flow contract — some
            # commands use Raise=True for early-exit. See callbacks.py:675.
            raise callbacks.Error(s)
        return None
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k buffering_proxy
```

Expected: 4 PASS.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): add BufferingIrcProxy for bridge reply capture"
```

---

### Task A3: `enumerate_commands` — list callable bridged commands

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py`
- Modify: `plugins/llm/tests/test_limnoria_bridge.py`

**Background:**
- `cb.listCommands()` returns leaf names (`"ping"`, `"time"`).
- `checkCommandCapability(msg, cb, name)` accepts the **string** form for the leaf check (mirrors `_callCommand` at `callbacks.py:1591`). List form `[name]` triggers `AssertionError` because `name != cb.canonicalName()`.
- Return value: `False` (allowed), `True` (default-deny), non-empty string (specific anti-cap blocked it). Filter on **truthiness**.
- Method `__doc__`: line 1 is argument syntax (Limnoria convention); rest is description.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_limnoria_bridge.py`:

```python
def _stub_callback(mocker, name, canonical=None, commands=None, docstrings=None):
    """Build a fake Limnoria plugin callback with controllable commands."""
    cb = mocker.MagicMock()
    cb.name.return_value = name
    cb.canonicalName.return_value = canonical or name.lower()
    cb.listCommands.return_value = list(commands or [])

    docs = docstrings or {}
    def _get_method(path):
        leaf = path[-1] if isinstance(path, list) else path
        method = mocker.MagicMock()
        method.__doc__ = docs.get(leaf, "")
        return method
    cb.getCommandMethod.side_effect = _get_method
    cb.isCommandMethod.side_effect = lambda c: c in (commands or [])
    return cb


def _fake_irc_with_callbacks(mocker, callbacks_list, network="testnet"):
    irc = mocker.MagicMock()
    irc.callbacks = list(callbacks_list)
    irc.network = network
    return irc


def _fake_msg(mocker, channel="#test", prefix="testnick!user@host"):
    msg = mocker.MagicMock()
    msg.prefix = prefix
    msg.channel = channel
    msg.args = (channel, "trigger")
    return msg


def test_enumerate_yields_command_when_authorized(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker, "Misc", commands=["ping"],
        docstrings={"ping": "takes no arguments\n\nReplies with pong."},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Misc"})))

    assert len(result) == 1
    assert result[0].plugin == "Misc"
    assert result[0].command == "ping"
    assert result[0].arg_syntax == "takes no arguments"
    assert "Replies with pong." in result[0].description


def test_enumerate_skips_deny_plugin_even_if_allowed(mocker):
    """Owner is hard-deny; explicitly adding it to the allowlist must not expose it."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Owner", commands=["load"])
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Owner"})))

    assert result == []


def test_enumerate_skips_plugin_not_in_allowlist(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"], docstrings={"ping": "x"})
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset()))  # empty allowlist

    assert result == []


def test_enumerate_skips_deny_command(mocker):
    """Web is allowed by operator, but Web.fetch is in DENY_COMMANDS."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker, "Web", canonical="web", commands=["fetch", "title"],
        docstrings={"fetch": "<url>", "title": "<url>"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Web"})))

    leaves = {c.command for c in result}
    assert leaves == {"title"}  # fetch is denied


def test_enumerate_skips_lacking_capability(mocker):
    """Stub plugin (NOT in DENY_PLUGINS) whose command is anti-capability blocked."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker, "StubPlugin", canonical="stubplugin",
        commands=["restricted", "open"],
        docstrings={"restricted": "x", "open": "y"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    # Capability check returns truthy ("anti-cap-name") for `restricted`,
    # False (allowed) for `open`.
    def _check(_msg, _cb, name):
        return "stubplugin.restricted" if name == "restricted" else False
    mocker.patch.object(lb.callbacks, "checkCommandCapability", side_effect=_check)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"StubPlugin"})))

    leaves = {c.command for c in result}
    assert leaves == {"open"}


def test_enumerate_passes_string_form_to_capability_check(mocker):
    """Regression: list form [cmd] triggers AssertionError in Limnoria.

    See callbacks.py:443-445 — checkCommandCapability asserts that
    list-form names start with the plugin's canonical name. We pass the
    string form to mirror _callCommand's leaf-check pattern at line 1591.
    """
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"], docstrings={"ping": "x"})
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)

    seen = []
    def _check(_msg, _cb, name):
        seen.append(name)
        return False
    mocker.patch.object(lb.callbacks, "checkCommandCapability", side_effect=_check)

    list(lb.enumerate_commands(irc, msg, frozenset({"Misc"})))

    assert seen == ["ping"]
    assert all(isinstance(n, str) for n in seen)
```

**Step 2: Run; verify all six fail.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k enumerate
```

Expected: 6 FAIL with `AttributeError: ... has no attribute 'enumerate_commands'`.

**Step 3: Implement `enumerate_commands`.**

Append to `plugins/llm/src/llm/limnoria_bridge.py`:

```python
from collections.abc import Iterator
from typing import Any  # noqa: F401  (used in later tasks)


def enumerate_commands(
    irc: Any,
    msg: Any,
    allowed_plugins: frozenset[str],
) -> Iterator[BridgeCommand]:
    """Yield every loaded command the LLM is allowed to call.

    A command is yielded when ALL of:
    - Its plugin is in ``allowed_plugins`` (operator allowlist).
    - Its plugin is NOT in ``DENY_PLUGINS`` (hard deny).
    - Its (canonical_plugin, leaf) tuple is NOT in ``DENY_COMMANDS``.
    - ``checkCommandCapability(msg, cb, leaf)`` returns falsy
      (i.e. allowed for the calling user).

    The capability check uses the string form of the leaf name to
    mirror ``_callCommand``'s pattern at supybot/callbacks.py:1591;
    list form ``[leaf]`` triggers an AssertionError because the leaf
    is not the plugin's canonical name.
    """
    for cb in irc.callbacks:
        plugin_name = cb.name()
        if plugin_name in DENY_PLUGINS:
            continue
        if plugin_name not in allowed_plugins:
            continue
        canonical = cb.canonicalName()
        for leaf in cb.listCommands():
            if (canonical, leaf) in DENY_COMMANDS:
                continue
            denial = callbacks.checkCommandCapability(msg, cb, leaf)
            if denial:
                continue
            method = cb.getCommandMethod([leaf])
            doc_lines = (method.__doc__ or "").strip().splitlines()
            arg_syntax = doc_lines[0].strip() if doc_lines else ""
            description = " ".join(line.strip() for line in doc_lines[1:]).strip()
            yield BridgeCommand(
                plugin=plugin_name,
                command=leaf,
                arg_syntax=arg_syntax,
                description=description,
            )
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v
```

Expected: all green so far (A1 + A2 + A3 tests).

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): bridge enumerate_commands with deny lists and capability gate"
```

---

### Task A4: `dispatch` — call a command and return a JSON envelope

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py`
- Modify: `plugins/llm/tests/test_limnoria_bridge.py`

**Background:**
- `irc.getCallback(name)` resolves a plugin by name.
- `cb._callCommand(command, irc, msg, *args, **kwargs)` is the dispatch entry. The first arg is a **list** (`[leaf]`); `_callCommand` prepends the canonical name itself if needed (`callbacks.py:1583`).
- `args` (the token list) is positional, not keyword (`callbacks.py:1213`).
- `callbacks.tokenize(s, channel=None, network=None)` honors per-channel bracket / pipe config.
- Argument errors (wrong arity, bad type) inside `wrap()` call `irc.reply(help)` (not `irc.error`); the help text is captured in `proxy.buffer` and surfaces as `{"status": "ok", "reply": "<help>"}`. The LLM gets the help text and can re-call.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_limnoria_bridge.py`:

```python
def test_dispatch_unknown_plugin(mocker):
    from llm import limnoria_bridge as lb

    irc = mocker.MagicMock()
    irc.getCallback.return_value = None
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Nope", command="x", arg_string="")
    assert out == {"error": "unknown plugin: Nope"}


def test_dispatch_deny_plugin_blocks_call(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Owner", commands=["load"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Owner", command="load", arg_string="Foo")
    assert out == {"error": "denied: Owner.load"}


def test_dispatch_deny_command_blocks_call(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Web", canonical="web", commands=["fetch"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Web", command="fetch", arg_string="http://x")
    assert out == {"error": "denied: Web.fetch"}


def test_dispatch_unknown_command(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(irc, msg, plugin="Misc", command="bogus", arg_string="")
    assert out == {"error": "unknown command: Misc.bogus"}


def test_dispatch_capability_denied(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value="anti.cap")

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"error": "not permitted: Misc.ping"}


def test_dispatch_captures_reply(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])

    def _fake_call(command, proxy, _msg, _tokens):
        proxy.reply("pong")
    cb._callCommand.side_effect = _fake_call

    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"status": "ok", "reply": "pong"}


def test_dispatch_passes_command_as_list_and_tokens_positionally(mocker):
    """Regression: _callCommand requires a list-form command and positional tokens.

    Keyword `args=tokens` ends up in **kwargs (the wrap() spec receives an
    empty positional args list), breaking argument parsing.
    """
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.return_value = None
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=["arg1", "arg2"])

    lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="arg1 arg2")

    args, kwargs = cb._callCommand.call_args
    assert args[0] == ["ping"]
    # args = (command_list, irc, msg, tokens)
    assert args[3] == ["arg1", "arg2"]
    assert "args" not in kwargs


def test_dispatch_uncaught_exception_returns_error(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.side_effect = RuntimeError("boom")
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="")
    assert out == {"error": "boom"}


def test_dispatch_argument_error_returned_as_reply(mocker):
    """wrap() argument errors come through irc.reply(help_text), not irc.error."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])

    def _fake_call(_command, proxy, _msg, _tokens):
        proxy.reply("(ping takes no arguments)")
    cb._callCommand.side_effect = _fake_call

    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=["unexpected"])

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="unexpected")
    assert out == {"status": "ok", "reply": "(ping takes no arguments)"}


def test_dispatch_malformed_args_returns_error_envelope(mocker):
    """tokenize() raises SyntaxError on malformed brackets/pipes — the
    bridge must catch it and return an error envelope, not propagate."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(
        lb.callbacks, "tokenize", side_effect=SyntaxError("unmatched bracket")
    )

    out = lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="[oops")
    assert out == {"error": "unmatched bracket"}
    cb._callCommand.assert_not_called()


def test_dispatch_tokenize_called_with_channel_and_network(mocker):
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", commands=["ping"])
    cb._callCommand.return_value = None
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker, channel="#test")
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    tok = mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    lb.dispatch(irc, msg, plugin="Misc", command="ping", arg_string="hi")

    tok.assert_called_once_with("hi", channel="#test", network="testnet")
```

**Step 2: Run; verify all fail.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k dispatch
```

Expected: 10 FAIL (no `dispatch` symbol).

**Step 3: Implement `dispatch`.**

Append to `plugins/llm/src/llm/limnoria_bridge.py`:

```python
def dispatch(
    irc: Any,
    msg: Any,
    *,
    plugin: str,
    command: str,
    arg_string: str,
) -> dict[str, Any]:
    """Run ``plugin.command arg_string`` through Limnoria's command path.

    Layered checks before dispatch:
    1. Plugin must resolve via ``irc.getCallback(plugin)``.
    2. Plugin must not be in ``DENY_PLUGINS``.
    3. (canonical_plugin, command) must not be in ``DENY_COMMANDS``.
    4. ``cb.isCommandMethod(command)`` must be True.
    5. ``checkCommandCapability(msg, cb, command)`` must be falsy.

    On success, returns ``{"status": "ok", "reply": "<captured text>"}``.
    On any check failure or uncaught exception, returns
    ``{"error": "<reason>"}``. The shape matches ``AssistantToolExecutor._ok``
    / ``_err`` (see assistant.py:676-683) so the assistant loop's
    ``last_successful_tool`` guard at service.py:2705-2710 fires correctly.
    """
    cb = irc.getCallback(plugin)
    if cb is None:
        return {"error": f"unknown plugin: {plugin}"}
    if cb.name() in DENY_PLUGINS:
        return {"error": f"denied: {plugin}.{command}"}
    if (cb.canonicalName(), command) in DENY_COMMANDS:
        return {"error": f"denied: {plugin}.{command}"}
    if not cb.isCommandMethod(command):
        return {"error": f"unknown command: {plugin}.{command}"}
    denial = callbacks.checkCommandCapability(msg, cb, command)
    if denial:
        return {"error": f"not permitted: {plugin}.{command}"}

    proxy = BufferingIrcProxy(irc, msg)
    try:
        # tokenize() raises SyntaxError on malformed bracket/pipe/quote
        # syntax (callbacks.py:431) — keep it inside the try so the
        # error-envelope contract holds for malformed args too.
        tokens = callbacks.tokenize(
            arg_string, channel=msg.channel, network=irc.network
        )
        # Positional args; keyword `args=tokens` would land in **kwargs and
        # break wrap()-based commands. See callbacks.py:1213.
        cb._callCommand([command], proxy, msg, tokens)
    except Exception as exc:  # noqa: BLE001 — translating to JSON envelope
        return {"error": str(exc) or exc.__class__.__name__}
    return {"status": "ok", "reply": "\n".join(proxy.buffer)}
```

**Step 4: Run; verify all pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v
```

Expected: every test in the file passes.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): bridge dispatch with JSON envelope and positional tokens"
```

---

## B — Plumbing: thread `extra_tools` / `extra_handlers` through assistant_completion

### Task B1: Register `bridgeEnabled` and `bridgeAllowedPlugins` registry values

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (append two `registerChannelValue` blocks near the other channel-scoped feature flags around line ~280)
- Modify: `plugins/llm/tests/test_config.py` (assert defaults against the real registry, following the existing `TestConfigValues` pattern)
- Modify: `plugins/llm/tests/conftest.py` — add `bridgeEnabled` and `bridgeAllowedPlugins` to `make_registry_side_effect`'s defaults dict at `conftest.py:276` so plugin-level tests in tasks C1/C2 can override them. Add the two keys with values `False` and `[]` respectively.

**Step 1: Write failing test.**

Append to `plugins/llm/tests/test_config.py`. The fixture-mocked `plugin.registryValue` returns from a local dict (not Limnoria's real registry), so it cannot tell whether the registry value is actually registered. Assert directly against the registered defaults:

```python
def test_bridge_registry_values_registered_with_safe_defaults():
    """B1: bridgeEnabled defaults to False and bridgeAllowedPlugins to []."""
    import supybot.conf as conf
    import llm.config  # noqa: F401 — import side effect registers the values

    assert conf.supybot.plugins.LLM.bridgeEnabled() is False
    # SpaceSeparatedListOfStrings() returns a list-like; coerce for comparison.
    assert list(conf.supybot.plugins.LLM.bridgeAllowedPlugins()) == []
```

(Match the pattern any other registry-default tests in `test_config.py` already use; if there's a `TestConfigValues` class, place it there.)

**Step 2: Run; verify it fails.**

```bash
uv run pytest plugins/llm/tests/test_config.py -v -k bridge
```

Expected: FAIL — registry value not defined.

**Step 3: Add registry values to `plugins/llm/src/llm/config.py`.**

Insert near the existing per-channel feature flags (a sensible spot is right after the memory cleanup block ends, around `config.py:330` — but match the file's existing section comments). Add a `# Limnoria bridge` section comment for discoverability:

```python
# ============================================================================
# Limnoria tool bridge (Phase 1)
# ============================================================================

conf.registerChannelValue(
    LLM,
    "bridgeEnabled",
    registry.Boolean(
        False,
        _("""When True, expose loaded Limnoria plugin commands to the LLM
        as a tool, restricted by bridgeAllowedPlugins and Limnoria's
        capability system. Default off."""),
    ),
)

conf.registerChannelValue(
    LLM,
    "bridgeAllowedPlugins",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""Space-separated list of Limnoria plugin names whose commands
        the LLM may call when bridgeEnabled is True. Empty (the default)
        means no commands are exposed — the bridge tool is not registered
        with the LLM at all. Recommended starter set: Misc Time Math
        Utilities Seen."""),
    ),
)
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_config.py -v -k bridge
```

Expected: PASS.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat(llm): register bridgeEnabled and bridgeAllowedPlugins"
```

---

### Task B2: Thread `extra_tools` and `extra_handlers` through `assistant_completion`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (`assistant_completion`, currently `service.py:2459`; injection points at `2592` and `2698`)
- Modify: `plugins/llm/tests/test_service.py` (or `test_assistant.py` — wherever `assistant_completion` is currently exercised; pick the existing file)

**Background:** the dispatch loop currently does `tool_result = executor.execute(tc.function.name, args)` unconditionally (`service.py:2698`). Bridge tool calls would land in the executor and get rejected as `Unknown tool: run_limnoria_command`. We need a side-channel that runs *before* the executor.

**Step 1: Write failing tests.**

Existing `assistant_completion` tests live in `plugins/llm/tests/test_assistant.py`. Add the four tests **there** (not in a new file) so they can reuse whatever litellm.completion patching pattern that file already establishes. Run `grep -n "def test_assistant\|litellm.completion\|assistant_completion" plugins/llm/tests/test_assistant.py` first to find the canonical mocking helper, then mirror it.

The four tests are below. Where they say "wire to existing harness," that means: build the same fake-response sequence the file already uses, just substitute the new kwargs (`extra_tools=`, `extra_handlers=`).

```python
"""Tests for the extra_tools / extra_handlers plumbing on assistant_completion."""

from __future__ import annotations

import json
from typing import Any

from llm.assistant import ToolResult


def _stub_litellm_response(mocker, *, content=None, tool_calls=None):
    """Build a stub litellm response with given choice content / tool_calls."""
    response = mocker.MagicMock()
    choice = mocker.MagicMock()
    msg = mocker.MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or None
    choice.message = msg
    response.choices = [choice]
    response.usage = mocker.MagicMock(prompt_tokens=1, completion_tokens=1)
    return response


def _tool_call(mocker, name, arguments_json):
    tc = mocker.MagicMock()
    tc.id = "call-1"
    tc.function = mocker.MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments_json
    return tc


def test_extra_tools_appended_to_profile_tools(plugin_env, mocker):
    """assistant_completion must include extra_tools in the litellm tools= kwarg."""
    fake_extra = {
        "type": "function",
        "function": {"name": "run_limnoria_command", "parameters": {}},
    }
    response = _stub_litellm_response(mocker, content="done")
    completion_mock = mocker.patch(
        "llm.service.litellm.completion", return_value=response
    )

    # Use the same setup the existing assistant_completion tests in this
    # file use to invoke the method (svc, db, context, etc.). Pass
    # extra_tools=[fake_extra]. Assertion:
    tools_kwarg = completion_mock.call_args.kwargs["tools"]
    assert fake_extra in tools_kwarg


def test_extra_handlers_dispatched_before_executor(plugin_env, mocker):
    """When the model picks a tool name in extra_handlers, the handler runs
    and AssistantToolExecutor.execute is NOT called for that name."""
    handler = mocker.MagicMock(
        return_value=mocker.MagicMock(content='{"status": "ok", "reply": "ok"}')
    )
    tc = _tool_call(mocker, "run_limnoria_command",
                    '{"plugin": "Misc", "command": "ping", "args": ""}')
    first = _stub_litellm_response(mocker, content=None, tool_calls=[tc])
    second = _stub_litellm_response(mocker, content="final")
    mocker.patch("llm.service.litellm.completion", side_effect=[first, second])
    exec_mock = mocker.patch(
        "llm.service.AssistantToolExecutor.execute",
        return_value=mocker.MagicMock(content='{"status": "ok"}'),
    )

    # Invoke assistant_completion with extra_handlers={"run_limnoria_command": handler}.
    handler.assert_called_once_with(
        {"plugin": "Misc", "command": "ping", "args": ""}
    )
    # Executor must not have been called for the bridge tool name.
    for call in exec_mock.call_args_list:
        assert call.args[0] != "run_limnoria_command"


def test_extra_handlers_error_envelope_does_not_set_last_successful_tool(plugin_env, mocker):
    """{"error": ...} envelope must not update last_successful_tool."""
    handler = mocker.MagicMock(
        return_value=mocker.MagicMock(content='{"error": "denied: Misc.ping"}')
    )
    tc = _tool_call(mocker, "run_limnoria_command", '{"plugin": "Misc", "command": "ping", "args": ""}')
    first = _stub_litellm_response(mocker, content=None, tool_calls=[tc])
    second = _stub_litellm_response(mocker, content="final")
    mocker.patch("llm.service.litellm.completion", side_effect=[first, second])

    # result = svc.assistant_completion(..., extra_handlers={"run_limnoria_command": handler})
    # assert result.last_successful_tool is None


def test_extra_handlers_ok_envelope_sets_last_successful_tool(plugin_env, mocker):
    """{"status": "ok", "reply": "x"} envelope must set last_successful_tool."""
    handler = mocker.MagicMock(
        return_value=mocker.MagicMock(content='{"status": "ok", "reply": "pong"}')
    )
    tc = _tool_call(mocker, "run_limnoria_command", '{"plugin": "Misc", "command": "ping", "args": ""}')
    first = _stub_litellm_response(mocker, content=None, tool_calls=[tc])
    second = _stub_litellm_response(mocker, content="final")
    mocker.patch("llm.service.litellm.completion", side_effect=[first, second])

    # result = svc.assistant_completion(..., extra_handlers={"run_limnoria_command": handler})
    # assert result.last_successful_tool == "run_limnoria_command"
```

> **Implementer note:** the actual `svc.assistant_completion(...)` invocation in each test must mirror the existing `test_assistant.py` invocation — same `db`, `context`, `nick`, `channel`, `bot_nick` arguments. Don't reinvent the harness; copy the canonical call from the file. The assertions at the bottom of each test (against `completion_mock.call_args`, against `handler.call_args`, against `result.last_successful_tool`) are the load-bearing parts.

**Step 2: Run; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_assistant_completion_extras.py -v
```

Expected: 4 FAIL.

**Step 3: Add the parameters and routing logic to `assistant_completion`.**

In `plugins/llm/src/llm/service.py` modify `assistant_completion` (currently `service.py:2459`):

3a. **Imports** — add `Callable` to the typing import block at the top of the file if it's not already there.

3b. **Signature** — add two parameters at the end of the keyword-only block (just before `exclude_tools`):

```python
extra_tools: list[dict[str, Any]] | None = None,
extra_handlers: dict[str, Callable[[dict[str, Any]], "ToolResult"]] | None = None,
```

(Use the string forward reference for `ToolResult` if its import would create a cycle; `from .assistant import ToolResult` is already imported inline at line 2517.)

3c. **profile_tools assembly** — at `service.py:2592`, replace:

```python
profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)
```

with:

```python
profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)
if extra_tools:
    profile_tools = profile_tools + list(extra_tools)
```

3d. **Tool dispatch branch** — at `service.py:2698`, replace:

```python
tool_result = executor.execute(tc.function.name, args)
```

with:

```python
if extra_handlers and tc.function.name in extra_handlers:
    tool_result = extra_handlers[tc.function.name](args)
else:
    tool_result = executor.execute(tc.function.name, args)
```

3e. **`assistant_request` facade** (`service.py:1932`) — add the same two parameters and forward them to `assistant_completion`. The dispatcher at line 1992 already forwards by keyword; add `extra_tools=extra_tools` and `extra_handlers=extra_handlers` to the kwargs.

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_assistant_completion_extras.py -v
uv run pytest plugins/llm/tests -q  # nothing else regressed
```

Expected: 4 new tests PASS, full suite green.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_assistant_completion_extras.py
git commit -m "feat(llm): thread extra_tools and extra_handlers through assistant loop"
```

---

## C — Wire the bridge into the chat profile

### Task C1: Bridge-tool builder helper on `LLMService` (or `LLM` plugin)

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (add `_build_bridge_tool` near the other private helpers — co-locate with `_get_provider_kwargs` or similar; you can also place it on `LLM` in `plugin.py` and pass through the schema/handlers — pick whichever owner already holds `registryValue("bridgeEnabled", channel)` calls in nearby code, to keep helper density consistent. Recommendation: `LLM` plugin because that's where other `registryValue` lookups for `assistant_request` callers live.)
- Modify: `plugins/llm/tests/test_plugin.py` (or wherever bridge-builder ends up living)

**Step 1: Write failing tests.**

```python
def test_build_bridge_tool_returns_none_when_disabled(plugin_env):
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        False if k == "bridgeEnabled" else
        [] if k == "bridgeAllowedPlugins" else
        None
    )
    schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")
    assert schema is None
    assert handlers is None


def test_build_bridge_tool_returns_none_when_allowlist_empty(plugin_env):
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        [] if k == "bridgeAllowedPlugins" else
        None
    )
    schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")
    assert schema is None
    assert handlers is None


def test_build_bridge_tool_returns_schema_and_handler_when_commands_present(plugin_env, mocker):
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Misc"] if k == "bridgeAllowedPlugins" else
        None
    )
    fake_cmds = [mocker.MagicMock(plugin="Misc", command="ping",
                                  arg_syntax="takes no arguments",
                                  description="Replies with pong.")]
    mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=fake_cmds)

    schema, handlers = plugin._build_bridge_tool(irc, msg, "#test")
    assert schema["function"]["name"] == "run_limnoria_command"
    assert "run_limnoria_command" in handlers
    # Description should mention the available command somewhere.
    assert "Misc.ping" in schema["function"]["description"] or "Misc ping" in schema["function"]["description"]


def test_build_bridge_tool_handler_returns_tool_result_with_json(plugin_env, mocker):
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Misc"] if k == "bridgeAllowedPlugins" else
        None
    )
    mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=[
        mocker.MagicMock(plugin="Misc", command="ping",
                         arg_syntax="", description=""),
    ])
    mocker.patch("llm.limnoria_bridge.dispatch", return_value={"status": "ok", "reply": "pong"})

    _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
    result = handlers["run_limnoria_command"]({"plugin": "Misc", "command": "ping", "args": ""})
    import json
    assert json.loads(result.content) == {"status": "ok", "reply": "pong"}
```

**Step 2: Run; verify all fail.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge_tool
```

**Step 3: Implement `_build_bridge_tool` on `LLM` (in `plugins/llm/src/llm/plugin.py`).**

Top-of-file imports:

```python
from . import limnoria_bridge
```

Inside class `LLM`, add the helper:

```python
def _build_bridge_tool(self, irc, msg, channel: str):
    """Build the per-request Limnoria bridge tool schema + handler.

    Returns ``(None, None)`` when the bridge is disabled, the allowlist is
    empty, or no allowed command is currently exposable. Otherwise returns
    ``(schema_dict, {"run_limnoria_command": handler})`` for injection
    into ``assistant_completion`` via ``extra_tools`` / ``extra_handlers``.
    """
    if not self.registryValue("bridgeEnabled", channel):
        return None, None
    allowed = frozenset(
        self.registryValue("bridgeAllowedPlugins", channel) or []
    )
    if not allowed:
        return None, None

    commands = list(limnoria_bridge.enumerate_commands(irc, msg, allowed))
    if not commands:
        return None, None

    table = "\n".join(
        f"- {c.plugin}.{c.command}"
        + (f" — {c.arg_syntax}" if c.arg_syntax else "")
        + (f" — {c.description}" if c.description else "")
        for c in commands
    )
    schema = {
        "type": "function",
        "function": {
            "name": "run_limnoria_command",
            "description": (
                "Run a Limnoria plugin command on the user's behalf. "
                "Available commands:\n" + table
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin": {
                        "type": "string",
                        "description": "Plugin name (e.g. Misc).",
                    },
                    "command": {
                        "type": "string",
                        "description": "Leaf command name (e.g. ping).",
                    },
                    "args": {
                        "type": "string",
                        "description": (
                            "Argument string passed to the plugin command. "
                            "Empty string for commands taking no arguments."
                        ),
                    },
                },
                "required": ["plugin", "command", "args"],
            },
        },
    }

    from .assistant import ToolResult

    def handler(arguments):
        envelope = limnoria_bridge.dispatch(
            irc, msg,
            plugin=str(arguments.get("plugin", "")),
            command=str(arguments.get("command", "")),
            arg_string=str(arguments.get("args", "")),
        )
        return ToolResult(content=json.dumps(envelope))

    return schema, {"run_limnoria_command": handler}
```

(`json` is already imported in `plugin.py`; if not, add it.)

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge_tool
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): add LLM._build_bridge_tool helper"
```

---

### Task C2: Inject bridge tool at the chat profile call site

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` at `plugin.py:2366` (the chat / `@ask` call site that currently passes `search_fn`, `fetch_fn`, etc. to `assistant_request`).

**Step 1: Write failing test.**

In the same test file used for chat / @ask flow (likely `plugins/llm/tests/test_plugin.py` or `test_commands.py`):

```python
def test_chat_call_site_passes_bridge_extras_when_enabled(plugin_env, mocker):
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Misc"] if k == "bridgeAllowedPlugins" else
        # … fall back to existing make_registry_side_effect for everything else
        make_registry_side_effect()(k, ch)
    )

    # Patch enumerate_commands to return one fake command so _build_bridge_tool
    # actually returns a schema (not None).
    mocker.patch("llm.limnoria_bridge.enumerate_commands", return_value=[
        mocker.MagicMock(plugin="Misc", command="ping", arg_syntax="", description=""),
    ])

    # Trigger the @ask path (or whatever the chat profile entry is).
    # … existing test pattern …

    # Final assertion: assistant_request was called with extra_tools / extra_handlers
    # that include run_limnoria_command.
    call_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
    assert call_kwargs["extra_tools"] is not None
    assert any(t["function"]["name"] == "run_limnoria_command" for t in call_kwargs["extra_tools"])
    assert "run_limnoria_command" in (call_kwargs["extra_handlers"] or {})


def test_chat_call_site_omits_bridge_extras_when_disabled(plugin_env, mocker):
    """When bridgeEnabled defaults to False, extra_tools/extra_handlers
    are not passed (or pass through as None) to assistant_request."""
    plugin, irc, msg = plugin_env
    # plugin_env's make_registry_side_effect already returns False for
    # bridgeEnabled (added to defaults dict in B1 conftest update).
    # Trigger the @ask flow using whatever harness test_commands.py /
    # test_plugin.py already use to invoke the chat profile — copy the
    # canonical pattern.
    call_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
    assert call_kwargs.get("extra_tools") is None
    assert call_kwargs.get("extra_handlers") is None
```

> **Implementer note:** the existing `@ask` happy-path test in `plugins/llm/tests/test_commands.py` is the harness to copy. Run `grep -n "def test.*ask\|assistant_request" plugins/llm/tests/test_commands.py` to find it. Use that test's setup verbatim; the only addition is the registryValue overrides for `bridgeEnabled` / `bridgeAllowedPlugins` and the assertion against `plugin.llm_service.assistant_request.call_args.kwargs`.

**Step 2: Run; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge_extras
```

**Step 3: Modify the chat call site.**

At `plugins/llm/src/llm/plugin.py:2366`, just **before** the `result = self.llm_service.assistant_request(...)` call, build the bridge:

```python
bridge_schema, bridge_handlers = self._build_bridge_tool(irc, msg, channel)
extra_tools = [bridge_schema] if bridge_schema else None
```

Then add to the kwargs of the `assistant_request(...)` call:

```python
extra_tools=extra_tools,
extra_handlers=bridge_handlers,
```

**Phase 1 scope:** wire only the chat profile call site (`plugin.py:2366`). Do NOT wire the other three `assistant_request` call sites at `plugin.py:1181`, `2489`, `2595`, `2701` — those are reminder-action / code / draw / spontaneous flows and the design plan defers them. (If they all share a helper, fine; otherwise leave them alone.)

**Step 4: Run.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge_extras
uv run pytest plugins/llm/tests -q
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): inject Limnoria bridge tool into @ask chat profile"
```

---

## D — Documentation

### Task D1: Operator documentation

**Files:**
- Modify: a file under `docs/guide/operator/` — the design plan suggests `tuning-monitoring.md`. Verify with `ls docs/guide/operator/` and pick the closest existing host (or `configuration.md` if that exists). If none fit, create `docs/guide/operator/limnoria-bridge.md` and link it from the relevant index.

**Step 1: Add an operator-facing section.**

Topics:
- What the bridge does (one paragraph).
- How to enable per channel:
  ```
  config channel #yourchan plugins.LLM.bridgeEnabled True
  config channel #yourchan plugins.LLM.bridgeAllowedPlugins Misc Time
  ```
- Reminder that the operator must `load Misc` / `load Time` etc. through Limnoria first; the bridge can only enumerate plugins that are actually loaded.
- Mention the recommended starter set (Misc, Time, Math, Utilities, Seen) and that it ships disabled.
- Document the hard-coded deny lists (DENY_PLUGINS, DENY_COMMANDS) and call out specifically that `Web.fetch` and `Utilities.apply` are denied unconditionally even if the operator allowlists `Web` or `Utilities`.
- Point at `plugins/llm/src/llm/limnoria_bridge.py` for the source of truth.

**Step 2: Build the docs locally if there's a build step (`mkdocs serve` or similar) and skim. Otherwise inspect the rendered Markdown.**

**Step 3: Commit.**

```bash
git add docs/guide/operator/<file>.md
git commit -m "docs(llm): document Limnoria tool bridge configuration"
```

### Task D2: AGENTS.md mention (only if there is a module catalog)

**Files:** `AGENTS.md`.

If `AGENTS.md` has a "modules" or "key files" section, add one line: `plugins/llm/src/llm/limnoria_bridge.py` — Limnoria → LLM tool bridge (Phase 1; see docs/plans/2026-05-02-limnoria-tool-bridge-plan.md).

If there is no such section, **skip this task entirely** — do not invent one.

---

## Validation

### Automated

```bash
# Bridge module unit tests.
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v

# Plumbing tests (added in B2 to plugins/llm/tests/test_assistant.py).
uv run pytest plugins/llm/tests/test_assistant.py -v -k "extra_tools or extra_handlers"

# Full LLM suite — must remain green.
uv run pytest plugins/llm/tests -q

# Repository-wide gates from AGENTS.md before declaring done.
make lint
make typecheck
make preflight
```

All commands must be green before declaring Phase 1 done. Per `AGENTS.md`, `make preflight` is the canonical sign-off; the narrower `uv run pytest` invocations above are for task-local TDD only.

### Manual smoke test (operator runs against a dev bot)

1. Load the stock plugins: in IRC, as owner — `@load Misc` and `@load Time`.
2. Enable the bridge for one channel:
   - `@config channel #test plugins.LLM.bridgeEnabled True`
   - `@config channel #test plugins.LLM.bridgeAllowedPlugins Misc Time`
3. Smoke prompts in `#test`:
   - `@vibebot ping the bot` → bridge picks `Misc.ping` → reply contains `pong`.
   - `@vibebot what time is it in UTC` → bridge picks `Time.time` (or similar).
4. Capability negative: as a non-op (different account), ask the bot to do something default-denied (`@vibebot please load the foo plugin`). The bridge must not surface `Owner.load` (it's in `DENY_PLUGINS`); even if the LLM tries, dispatch returns `{"error": "denied: Owner.load"}`.
5. Channel preservation: confirm `#test` is still present in subsequent log lines for that user (i.e. `msg.channel` was not clobbered by the proxy — should be a no-op since the original `msg` is reused).
6. No double-reply: after a successful bridged call, IRC should see only the LLM's natural-language wrap-around reply, not the raw `pong` line that the stock plugin would have queued.
7. Timeout behavior: lower `supybot.plugins.LLM.timeout` to 5 seconds, attempt a bridged call to a slow command (if `Web` is loaded with the SSRF deny in place, `Web.title` against a slow URL is the test). The LLM call should time out cleanly without hanging the bot.

---

## Open questions

These are the five items from the design plan's "Open questions for code review" section — none are blockers but the implementer should flag any that come up while writing the code:

1. **Default `bridgeAllowedPlugins`** — current plan: empty (operator must opt-in plugin by plugin). Confirm with the user before merge if a "safe default" of `Misc Time` is preferable.
2. **Reply truncation** — `dispatch()` returns the full buffered reply unmodified. The LLM plugin's existing long-reply linker handles output sizing on its own side. If the smoke test surfaces token-budget issues for verbose stock-plugin replies, add a length cap inside `dispatch()` and re-test.
3. **`_render_command_table()` format** — Task C1 uses plain `- Plugin.command — arg_syntax — description` lines. If the LLM picks wrongly with this form during smoke testing, a Markdown table or one-tool-per-command form may help. Defer until evidence.
4. **Multiple bridge tool calls in one turn** — the existing loop iterates `message.tool_calls` so this should "just work." Add a regression test if any smoke-test prompt happens to trigger it.
5. **Error string sanitization** — `dispatch()` returns `str(exc)` on uncaught exceptions. Quick audit for sensitive paths/tracebacks: if the implementer sees stack frames or filesystem paths leaking into the JSON envelope, wrap with the existing `_sanitize` helper from `LLMService` (see `service.py` for the helper).

Phase 2 work (replacing reminders with `Scheduler`/`Later`, per-command tools, retiring `generate_image` in favor of stock plugins) is **out of scope** for this plan — see the design plan's "Phase 2" section.

---

## Execution order summary

| Order | Task | Output |
| --- | --- | --- |
| 0 | Pre-flight | (no commit) |
| 1 | A1: scaffold module + dataclass + deny lists | commit |
| 2 | A2: BufferingIrcProxy | commit |
| 3 | A3: enumerate_commands | commit |
| 4 | A4: dispatch | commit |
| 5 | B1: registry values | commit |
| 6 | B2: extra_tools / extra_handlers plumbing | commit |
| 7 | C1: _build_bridge_tool helper | commit |
| 8 | C2: chat call site injection | commit |
| 9 | D1: operator docs | commit |
| 10 | D2: AGENTS.md (if applicable) | commit |
| 11 | Validation: full suite + smoke | (no commit) |

Each task is independently verifiable: tests pass after the commit, and reverting one commit cleanly leaves the codebase in a working state.
