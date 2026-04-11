"""NickInMiddle plugin implementation.

Rewrites incoming PRIVMSG messages so that the bot's nick, when it appears
in the *middle* of a channel message, is moved to the front.  This lets the
normal Limnoria addressing logic (``callbacks._addressed``) recognise the
message as addressed without any core changes.

Example (bot nick = ``vibebot``):

    can you, vibebot, tell me the weather
    →  vibebot tell me the weather

The rewrite only fires when:

- The message is a channel PRIVMSG (not a PM).
- The nick is *not* already the first or last word (those cases are handled
  natively by ``supybot.reply.whenAddressedBy.nick`` and ``.nick.atEnd``).
- There is a word-boundary separator (space, comma, colon, semicolon) on
  both sides of the nick.
"""

from __future__ import annotations

import re
from typing import Any

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.ircmsgs as ircmsgs
import supybot.ircutils as ircutils

_MIDDLE_TOKEN_RE = re.compile(r"[^\s,;:]+")
_STRIP_SEPARATORS = " \t,;:"
_SEPARATOR_CHARS = frozenset(_STRIP_SEPARATORS)


class NickInMiddle(callbacks.Plugin):
    """Recognise the bot's nick when it appears in the middle of a message."""

    def _addressing_nicks(self, irc: Any, channel: str) -> tuple[str, ...]:
        """Return the nick spellings Limnoria treats as addressing the bot."""
        configured = conf.supybot.reply.whenAddressedBy.nicks.getSpecific(
            network=irc.network,
            channel=channel,
        )()
        return (irc.nick, *configured)

    def _find_middle_address(self, irc: Any, channel: str, text: str) -> tuple[str, str] | None:
        """Return the payload parts around a middle-position addressing nick."""
        addressing_nicks = self._addressing_nicks(irc, channel)
        for match in _MIDDLE_TOKEN_RE.finditer(text):
            start, end = match.span()
            if start == 0 or end == len(text):
                continue
            if text[start - 1] not in _SEPARATOR_CHARS or text[end] not in _SEPARATOR_CHARS:
                continue

            candidate = match.group(0).rstrip("?.!")
            if not any(ircutils.nickEqual(candidate, nick) for nick in addressing_nicks):
                continue

            before = text[:start].rstrip(" \t")
            after = text[end:].lstrip(_STRIP_SEPARATORS)
            if before.rstrip(",;:") and after:
                return before, after

        return None

    def inFilter(self, irc: Any, msg: ircmsgs.IrcMsg) -> ircmsgs.IrcMsg:  # noqa: N802
        if msg.command != "PRIVMSG":
            return msg
        # Only rewrite in channels, not PMs.
        if not msg.channel:
            return msg
        if not self.registryValue("enabled", msg.channel, irc.network):
            return msg

        text = msg.args[1]
        if not text:
            return msg
        if text.startswith("\x01") and text.endswith("\x01"):
            return msg

        payload_parts = self._find_middle_address(irc, msg.channel, text)
        if not payload_parts:
            return msg

        before, after = payload_parts
        rewritten = f"{irc.nick} {before} {after}"

        rewritten_msg = ircmsgs.IrcMsg(msg=msg, args=(msg.args[0], rewritten))
        rewritten_msg.channel = msg.channel
        return rewritten_msg


Class = NickInMiddle
