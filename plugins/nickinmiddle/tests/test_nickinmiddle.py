"""Tests for NickInMiddle plugin."""

from __future__ import annotations

import supybot.ircmsgs as ircmsgs
from nickinmiddle.config import configure
from nickinmiddle.plugin import NickInMiddle


def test_configure_runs() -> None:
    """`configure()` should register the plugin without raising."""
    configure(advanced=False)
    configure(advanced=True)


class FakeIrc:
    """Minimal stand-in for the Irc object."""

    def __init__(self, nick: str = "vibebot", network: str = "test") -> None:
        self.nick = nick
        self.network = network


class FakePlugin(NickInMiddle):
    """Subclass that avoids full Plugin.__init__ and stubs registryValue."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        addressing_nicks: tuple[str, ...] | None = None,
    ) -> None:  # noqa: D107
        self._enabled = enabled
        self._addressing_nicks_override = addressing_nicks or ()

    def registryValue(self, name, channel=None, network=None, *, value=True):  # noqa: N802
        if name == "enabled":
            return self._enabled
        raise KeyError(name)

    def _addressing_nicks(self, irc, channel):  # noqa: ANN001
        return (irc.nick, *self._addressing_nicks_override)


def _chan_msg(text: str, channel: str = "#test") -> ircmsgs.IrcMsg:
    """Build a channel PRIVMSG with .channel set (as the Irc object would)."""
    msg = ircmsgs.IrcMsg(
        prefix="user!ident@host",
        command="PRIVMSG",
        args=(channel, text),
    )
    msg.channel = channel
    return msg


def _pm_msg(text: str, target: str = "vibebot") -> ircmsgs.IrcMsg:
    """Build a private PRIVMSG."""
    return ircmsgs.IrcMsg(
        prefix="user!ident@host",
        command="PRIVMSG",
        args=(target, text),
    )


class TestNickInMiddle:
    """Test the inFilter rewriting logic."""

    def test_nick_in_middle_comma(self) -> None:
        """'do this, vibebot, please' should be rewritten."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("do this, vibebot, please")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "vibebot do this, please"

    def test_nick_in_middle_spaces(self) -> None:
        """'can you vibebot tell me' should be rewritten."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("can you vibebot tell me")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "vibebot can you tell me"

    def test_nick_at_start_not_rewritten(self) -> None:
        """'vibebot do this' should pass through untouched."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("vibebot do this")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "vibebot do this"

    def test_nick_at_end_not_rewritten(self) -> None:
        """'do this vibebot' should pass through (handled by atEnd)."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("do this vibebot")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "do this vibebot"

    def test_nick_at_end_with_comma_not_rewritten(self) -> None:
        """'do this, vibebot' should pass through (handled by atEnd)."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("do this, vibebot")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "do this, vibebot"

    def test_private_message_not_rewritten(self) -> None:
        """PMs should never be rewritten."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _pm_msg("hey vibebot do something")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "hey vibebot do something"

    def test_disabled_not_rewritten(self) -> None:
        """When disabled, messages pass through."""
        plugin = FakePlugin(enabled=False)
        irc = FakeIrc()
        msg = _chan_msg("can you, vibebot, help")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "can you, vibebot, help"

    def test_case_insensitive(self) -> None:
        """Nick matching should be case-insensitive."""
        plugin = FakePlugin()
        irc = FakeIrc(nick="VibeBot")
        msg = _chan_msg("can you, vibebot, help")
        result = plugin.inFilter(irc, msg)
        assert result.args[1].lower().startswith("vibebot")
        # The command portion should follow the nick.
        assert "help" in result.args[1]

    def test_no_nick_present(self) -> None:
        """Messages without the nick pass through."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("just a normal message")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "just a normal message"

    def test_nick_embedded_in_word_not_matched(self) -> None:
        """'myvibebotfriend' should not trigger a rewrite."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("hey myvibebotfriend do stuff")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "hey myvibebotfriend do stuff"

    def test_notice_ignored(self) -> None:
        """NOTICE messages should pass through."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = ircmsgs.IrcMsg(
            prefix="user!ident@host",
            command="NOTICE",
            args=("#test", "hey vibebot do stuff"),
        )
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "hey vibebot do stuff"

    def test_preserves_prefix(self) -> None:
        """Rewritten message should preserve the original prefix."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("please, vibebot, help me")
        result = plugin.inFilter(irc, msg)
        assert result.prefix == "user!ident@host"
        assert result.args[0] == "#test"

    def test_preserves_channel_metadata(self) -> None:
        """Rewritten messages should stay channel messages downstream."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("please, vibebot, help me")
        result = plugin.inFilter(irc, msg)
        assert result.channel == "#test"

    def test_nick_with_colon_separator(self) -> None:
        """'hey vibebot: do this' has nick at start-ish — but let's ensure
        the boundary check works for middle occurrences with colons."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("ok so vibebot: what do you think")
        result = plugin.inFilter(irc, msg)
        # Nick was in the middle, should rewrite.
        assert result.args[1].startswith("vibebot")

    def test_ctcp_action_not_rewritten(self) -> None:
        """CTCP ACTION payloads should never be treated as addressed text."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("\x01ACTION asks vibebot for help\x01")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "\x01ACTION asks vibebot for help\x01"

    def test_rfc1459_equivalent_nick_is_rewritten(self) -> None:
        """RFC1459-equivalent nick spellings should match in the middle."""
        plugin = FakePlugin()
        irc = FakeIrc(nick="Vibe[bot]")
        msg = _chan_msg("can you, vibe{bot}, help")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "Vibe[bot] can you, help"

    def test_nick_with_trailing_question_mark(self) -> None:
        """'Why is X, vibebot?  What's up' should be rewritten."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("Why is Eric Adams now Albanian, vibebot?  What's up with that?")
        result = plugin.inFilter(irc, msg)
        assert result.args[1].startswith("vibebot")
        assert "Eric Adams" in result.args[1]

    def test_nick_with_trailing_exclamation(self) -> None:
        """'help me, vibebot! I need you' should be rewritten."""
        plugin = FakePlugin()
        irc = FakeIrc()
        msg = _chan_msg("help me, vibebot! I need you")
        result = plugin.inFilter(irc, msg)
        assert result.args[1].startswith("vibebot")

    def test_configured_addressing_alias_is_rewritten(self) -> None:
        """Configured address aliases should work in middle-position rewrites."""
        plugin = FakePlugin(addressing_nicks=("assistant",))
        irc = FakeIrc()
        msg = _chan_msg("can you, assistant, help")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "vibebot can you, help"
