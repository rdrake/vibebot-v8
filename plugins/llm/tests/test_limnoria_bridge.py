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
        mocker,
        "Misc",
        commands=["ping"],
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
        mocker,
        "Web",
        canonical="web",
        commands=["fetch", "title"],
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
        mocker,
        "StubPlugin",
        canonical="stubplugin",
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
