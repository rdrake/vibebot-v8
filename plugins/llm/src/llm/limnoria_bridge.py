"""Limnoria → LLM tool bridge.

Exposes loaded Limnoria plugin commands to the LLM as a single
``run_limnoria_command`` tool. Enforces a layered denial model:

1. Hard-coded ``DENY_PLUGINS`` / ``DENY_COMMANDS`` (this module).
2. Operator-set ``bridgeAllowedPlugins`` (per-channel registry).
3. Limnoria's own capability system via ``checkCommandCapability``.

See docs/plans/2026-05-02-limnoria-tool-bridge-plan.md for the full design.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any  # noqa: F401  (used in later tasks)

from supybot import callbacks

# Plugin names matched against ``cb.name()`` (the user-facing CamelCase form).
DENY_PLUGINS: frozenset[str] = frozenset(
    {
        # Auth / management — capability checks already gate non-owners, but we
        # deny at the bridge layer too so the LLM never sees these as options.
        "LLM",
        "Owner",
        "Admin",
        "Config",
        "Channel",
        "User",
    }
)

# (canonical_plugin_name, leaf_command) tuples. Both lowercase — matched
# against ``cb.canonicalName()`` (already lowercase) and the leaf name from
# ``cb.listCommands()``.
DENY_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
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
    }
)


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


@dataclass(frozen=True)
class BridgeCommand:
    """One enumerated, callable Limnoria command."""

    plugin: str  # cb.name() — CamelCase, used in operator config
    command: str  # leaf command name from cb.listCommands()
    arg_syntax: str  # first line of method.__doc__
    description: str  # remaining lines of method.__doc__, joined


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
