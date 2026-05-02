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


@dataclass(frozen=True)
class BridgeCommand:
    """One enumerated, callable Limnoria command."""

    plugin: str  # cb.name() — CamelCase, used in operator config
    command: str  # leaf command name from cb.listCommands()
    arg_syntax: str  # first line of method.__doc__
    description: str  # remaining lines of method.__doc__, joined
