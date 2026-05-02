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

    cmd = lb.BridgeCommand(
        plugin="Misc", command="ping", arg_syntax="", description="takes no arguments"
    )
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"


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
    from llm.limnoria_bridge import BufferingIrcProxy
    from supybot import callbacks

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
