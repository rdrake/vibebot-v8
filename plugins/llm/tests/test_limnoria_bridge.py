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
    assert ("utilities", "let") in lb.DENY_COMMANDS

    cmd = lb.BridgeCommand(
        plugin="Misc", command="ping", arg_syntax="", description="takes no arguments"
    )
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"


def test_mutating_commands_covers_default_allowlist_writes():
    """Every mutating command in the Phase 2 Task 2 default allowlist must be
    in MUTATING_COMMANDS. Reads must NOT be in it. Tuples are
    (canonical_plugin_lowercase, leaf_lowercase) — same shape as DENY_COMMANDS."""
    from llm import limnoria_bridge as lb

    assert isinstance(lb.MUTATING_COMMANDS, frozenset)

    expected_mutating = {
        ("misc", "tell"),
        ("misc", "noticetell"),
        ("later", "tell"),
        ("later", "remove"),
        ("later", "undo"),
        ("note", "send"),
        ("note", "reply"),
        ("note", "unsend"),
        ("karma", "clear"),
        ("karma", "dump"),
        ("karma", "load"),
        ("quotegrabs", "grab"),
        ("quotegrabs", "ungrab"),
        ("rss", "add"),
        ("rss", "remove"),
        ("rss", "rss"),  # update_feed_if_needed → announce_feed → IRC writes
        ("rss", "info"),  # same update_feed_if_needed side effect
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    # Reads in the same plugins must NOT be classified mutating.
    expected_read_only = {
        ("misc", "ping"),
        ("misc", "last"),
        ("misc", "version"),
        ("time", "time"),
        ("math", "calc"),
        ("utilities", "echo"),
        ("seen", "seen"),
        ("seen", "last"),
        ("web", "title"),
        ("later", "notes"),
        ("note", "search"),
        ("note", "list"),
        ("note", "note"),
        ("note", "next"),
        ("karma", "karma"),
        ("karma", "most"),
        ("quotegrabs", "quote"),
        ("quotegrabs", "random"),
        ("ddg", "search"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_mutating_commands_covers_forward_look_writes():
    """Quote/Todo/Factoids/Scheduler are not yet in the default allowlist but
    we classify them now so the gate is correct when they're added later."""
    from llm import limnoria_bridge as lb

    expected_mutating = {
        ("quote", "add"),
        ("quote", "remove"),
        ("quote", "change"),
        ("quote", "replace"),
        ("todo", "add"),
        ("todo", "remove"),
        ("todo", "setpriority"),
        ("todo", "change"),
        ("factoids", "learn"),
        ("factoids", "alias"),
        ("factoids", "lock"),
        ("factoids", "unlock"),
        ("factoids", "forget"),
        ("factoids", "change"),
        ("factoids", "whatis"),  # _updateRank writes when keepRankInfo=True (default)
        ("scheduler", "add"),
        ("scheduler", "remind"),
        ("scheduler", "remove"),
        ("scheduler", "repeat"),
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    expected_read_only = {
        ("quote", "search"),
        ("quote", "get"),
        ("quote", "stats"),
        ("quote", "random"),
        ("todo", "todo"),
        ("todo", "search"),
        ("factoids", "random"),
        ("factoids", "info"),
        ("factoids", "rank"),
        ("factoids", "search"),
        ("scheduler", "list"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_mutating_commands_lowercase_invariant():
    """Match the DENY_COMMANDS shape — both elements lowercase."""
    from llm import limnoria_bridge as lb

    for plugin, leaf in lb.MUTATING_COMMANDS:
        assert plugin == plugin.lower(), plugin
        assert leaf == leaf.lower(), leaf


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


def test_enumerate_skips_mutating_commands_when_gate_closed(mocker):
    """With allow_mutating=False (the default), MUTATING_COMMANDS leaves
    are filtered out even if their plugin is allowlisted."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes", "remove", "undo"],
        docstrings={
            "tell": "<nick> <text>",
            "notes": "takes no arguments",
            "remove": "<id>",
            "undo": "takes no arguments",
        },
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Later"}), allow_mutating=False))

    leaves = {c.command for c in result}
    assert leaves == {"notes"}  # tell, remove, undo all in MUTATING_COMMANDS


def test_enumerate_yields_mutating_commands_when_gate_open(mocker):
    """With allow_mutating=True, MUTATING_COMMANDS is not consulted —
    the existing capability + DENY filters still apply, but writes pass."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes", "remove", "undo"],
        docstrings={
            "tell": "<nick> <text>",
            "notes": "takes no arguments",
            "remove": "<id>",
            "undo": "takes no arguments",
        },
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Later"}), allow_mutating=True))

    leaves = {c.command for c in result}
    assert leaves == {"tell", "notes", "remove", "undo"}


def test_enumerate_default_keyword_is_gate_closed(mocker):
    """Calling enumerate_commands without allow_mutating= defaults to the
    closed gate — backwards-compat safety: an old caller that forgets to
    pass the kwarg still gets safe behavior."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes"],
        docstrings={"tell": "<nick> <text>", "notes": "x"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Later"})))
    leaves = {c.command for c in result}
    assert leaves == {"notes"}


def test_enumerate_gate_does_not_affect_pure_read_only_plugins(mocker):
    """A plugin with no entries in MUTATING_COMMANDS (e.g. Time) yields
    the same set whether the gate is open or closed."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Time",
        canonical="time",
        commands=["time", "at", "until"],
        docstrings={"time": "x", "at": "y", "until": "z"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    closed = {
        c.command
        for c in lb.enumerate_commands(irc, msg, frozenset({"Time"}), allow_mutating=False)
    }
    open_ = {
        c.command for c in lb.enumerate_commands(irc, msg, frozenset({"Time"}), allow_mutating=True)
    }
    assert closed == open_ == {"time", "at", "until"}


def test_enumerate_gate_preserves_deny_commands_filtering(mocker):
    """DENY_COMMANDS still bites even when allow_mutating=True. Web.fetch
    is denied unconditionally (SSRF), independent of the mutation gate."""
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

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Web"}), allow_mutating=True))
    leaves = {c.command for c in result}
    assert leaves == {"title"}  # fetch is in DENY_COMMANDS


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
    mocker.patch.object(lb.callbacks, "tokenize", side_effect=SyntaxError("unmatched bracket"))

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
