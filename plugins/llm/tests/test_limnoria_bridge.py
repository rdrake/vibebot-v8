"""Tests for the Limnoria tool bridge."""

from __future__ import annotations


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

    cmd = lb.BridgeCommand(
        plugin="Misc", command="ping", arg_syntax="", description="takes no arguments"
    )
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"
