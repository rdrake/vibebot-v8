"""Bot-to-bot loop guard: cap replies to nicks the network flags with +B.

Two bots that answer each other never get bored. AfterNET advertises
``BOT=B`` and six nicks on it carry the flag (LarryBot, Ender, LXIX, X69,
reborn, and vibebot itself), so the network already knows who is a robot —
it just does not say so on PRIVMSG. It says so in the WHOX status field,
which Limnoria parses and throws away (irclib.py:981).

Humans are never counted and never capped. An unflagged nick is treated as
a person, so the guard fails open: the worst case is the loop we have
today, never a silenced user.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _whox(bot_nick: str, nick: str, status: str) -> tuple[str, ...]:
    """A WHOX 354 reply in Limnoria's arg shape: querytype 1, 9 fields."""
    return (bot_nick, "1", "user", "1.2.3.4", "host", nick, status, "account", "gecos")


class TestBotFlagFromWhox:
    """The B in the WHOX status field is the whole detector."""

    def test_status_with_b_flags_the_nick(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN a 354 whose status carries B THEN the nick is known to be a bot."""
        plugin, mock_irc, _ = plugin_env
        msg = mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB"))

        plugin.do354(mock_irc, msg)

        assert plugin._known_bot(mock_irc, "LarryBot") is True

    def test_status_without_b_flags_a_human(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN a 354 with no B THEN the nick is known NOT to be a bot.

        Recorded as a definite False, not left unknown: that is what stops the
        probe firing again on every line the person types.
        """
        plugin, mock_irc, _ = plugin_env
        msg = mocker.MagicMock(args=_whox("testbot", "forestchav", "Hxz"))

        plugin.do354(mock_irc, msg)

        assert plugin._known_bot(mock_irc, "forestchav") is False

    def test_unseen_nick_is_unknown(self, plugin_env) -> None:
        """No WHO reply yet → None, which the guard treats as a person."""
        plugin, mock_irc, _ = plugin_env

        assert plugin._known_bot(mock_irc, "stranger") is None

    def test_flag_lookup_is_case_insensitive(self, plugin_env, mocker: MockerFixture) -> None:
        """IRC nicks are case-insensitive; LarryBot and larrybot are one bot."""
        plugin, mock_irc, _ = plugin_env
        msg = mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB"))

        plugin.do354(mock_irc, msg)

        assert plugin._known_bot(mock_irc, "larrybot") is True

    def test_malformed_354_is_ignored(self, plugin_env, mocker: MockerFixture) -> None:
        """A short or foreign-querytype reply must not raise or record."""
        plugin, mock_irc, _ = plugin_env

        plugin.do354(mock_irc, mocker.MagicMock(args=("testbot", "2", "too", "few")))

        assert plugin._known_bot(mock_irc, "too") is None


class TestBotFlagProbe:
    """Prod skips the WHO on join, so the flag has to be asked for.

    ``skipAutoWhoOnJoin`` is True in production and the last WHO reply the
    bot saw was 2026-02-13. Waiting for a channel sync that never comes would
    make the whole guard inert.
    """

    def test_unknown_nick_is_probed(self, plugin_env) -> None:
        """GIVEN an unknown speaker WHEN it addresses us THEN a WHO is queued."""
        plugin, mock_irc, _ = plugin_env

        plugin._probe_bot_flag(mock_irc, "stranger")

        assert mock_irc.queueMsg.called
        queued = mock_irc.queueMsg.call_args.args[0]
        assert "stranger" in queued.args

    def test_probe_is_not_repeated(self, plugin_env) -> None:
        """One WHO per nick per window — a chatty stranger is not a WHO flood."""
        plugin, mock_irc, _ = plugin_env

        plugin._probe_bot_flag(mock_irc, "stranger")
        plugin._probe_bot_flag(mock_irc, "stranger")
        plugin._probe_bot_flag(mock_irc, "stranger")

        assert mock_irc.queueMsg.call_count == 1

    def test_known_nick_is_not_probed(self, plugin_env, mocker: MockerFixture) -> None:
        """An answer already on file needs no question."""
        plugin, mock_irc, _ = plugin_env
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))

        plugin._probe_bot_flag(mock_irc, "LarryBot")

        mock_irc.queueMsg.assert_not_called()


class TestBotLoopCap:
    """The cap itself: N consecutive replies to one flagged bot, then quiet."""

    @pytest.fixture
    def wired(self, plugin_env, mocker: MockerFixture):
        """Plugin whose dispatch is observable and whose preflight always passes."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        preflight = mocker.MagicMock(blocked=False)
        mocker.patch.object(plugin, "_run_preflight", return_value=preflight)
        dispatch = mocker.patch.object(plugin, "_dispatch_addressed_async")
        return plugin, mock_irc, mock_msg, dispatch

    def _as(self, msg, nick: str, target: str = "#test") -> None:
        msg.nick = nick
        msg.prefix = f"{nick}!user@host"
        msg.args = (target, f"testbot: {nick} says hello")
        msg.channel = target if target.startswith("#") else None

    def test_bot_is_answered_up_to_the_limit(self, wired, mocker: MockerFixture) -> None:
        """GIVEN a flagged bot WHEN it addresses us 3 times THEN all 3 are answered."""
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        self._as(mock_msg, "LarryBot")

        for _ in range(3):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 3

    def test_the_reply_after_the_limit_is_silent(self, wired, mocker: MockerFixture) -> None:
        """The 4th consecutive turn is where the loop would have run forever."""
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        self._as(mock_msg, "LarryBot")

        for _ in range(6):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 3

    def test_a_human_turn_frees_the_bot(self, wired, mocker: MockerFixture) -> None:
        """A person joining the thread clears the count for that channel.

        The guard exists to stop two robots talking to themselves, not to end
        a conversation someone is actually part of.
        """
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "rdrake", "Hxz")))
        self._as(mock_msg, "LarryBot")
        for _ in range(5):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")
        assert dispatch.call_count == 3

        human = mocker.MagicMock(nick="rdrake", args=("#test", "oi"), channel="#test")
        plugin._note_channel_speaker(mock_irc, human)

        self._as(mock_msg, "LarryBot")
        plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 4

    def test_an_unflagged_nick_is_never_capped(self, wired) -> None:
        """Nobody who has not been proven a bot is ever silenced."""
        plugin, mock_irc, mock_msg, dispatch = wired
        self._as(mock_msg, "forestchav")

        for _ in range(10):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 10

    def test_the_count_expires(self, wired, mocker: MockerFixture) -> None:
        """A bot that pings us once an hour is not a loop and is not capped."""
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        self._as(mock_msg, "LarryBot")
        for _ in range(5):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")
        assert dispatch.call_count == 3

        # Age every count past botLoopWindow (300s in the test registry).
        with plugin._bot_loop_lock:
            plugin._bot_reply_counts = {
                key: (count, seen - 301) for key, (count, seen) in plugin._bot_reply_counts.items()
            }

        plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 4

    def test_channels_are_counted_separately(self, wired, mocker: MockerFixture) -> None:
        """One bot looping in #a must not silence the same bot in #b."""
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))

        self._as(mock_msg, "LarryBot", target="#test")
        for _ in range(5):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")
        assert dispatch.call_count == 3

        self._as(mock_msg, "LarryBot", target="#other")
        plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 4

    def test_a_pm_from_a_bot_is_capped_too(self, wired, mocker: MockerFixture) -> None:
        """A loop in query is still a loop."""
        plugin, mock_irc, mock_msg, dispatch = wired
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        self._as(mock_msg, "LarryBot", target="testbot")

        for _ in range(6):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 3

    def test_a_zero_limit_disables_the_guard(self, wired, mocker: MockerFixture) -> None:
        """botLoopReplyLimit=0 is the operator's off switch."""
        plugin, mock_irc, mock_msg, dispatch = wired
        base = plugin.registryValue.side_effect

        def registry(name, *args, **kwargs):
            if name == "botLoopReplyLimit":
                return 0
            return base(name, *args, **kwargs)

        plugin.registryValue.side_effect = registry
        plugin.do354(mock_irc, mocker.MagicMock(args=_whox("testbot", "LarryBot", "HxzB")))
        self._as(mock_msg, "LarryBot")

        for _ in range(8):
            plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 8


class TestBotLoopGuardIsSafe:
    """The guard runs on every addressed line, so it must never be the thing
    that breaks one."""

    def test_a_broken_flag_store_does_not_stop_a_reply(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """GIVEN the guard raises WHEN a line arrives THEN the reply still happens."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin, "_run_preflight", return_value=mocker.MagicMock(blocked=False))
        dispatch = mocker.patch.object(plugin, "_dispatch_addressed_async")
        mocker.patch.object(plugin, "_bot_loop_blocked", side_effect=RuntimeError("boom"))

        plugin._route_addressed_to_assistant(mock_irc, mock_msg, "hello")

        assert dispatch.call_count == 1
