"""RPG plugin implementation — IRC command layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import supybot.callbacks as callbacks
import supybot.log as log
from supybot.commands import optional, wrap

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg


class RPG(callbacks.Plugin):
    """Linux filesystem RPG — explore dungeons with shell commands."""

    threaded = True

    def __init__(self, irc: callbacks.Irc) -> None:
        super().__init__(irc)
        self.log = log.getPluginLogger("RPG")

    def die(self) -> None:
        super().die()

    class Rpg(callbacks.Commands):  # noqa: N801 — Limnoria maps class name to IRC command
        """RPG game commands — use shell commands to explore a Linux filesystem world."""

        def cd(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], destination: str) -> None:
            """<path> — Move to a location."""
            irc.reply(f"cd: not yet implemented — destination: {destination}")

        cd = wrap(cd, ["text"])

        def ls(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], flags: str) -> None:
            """[flags] — Look around. Use -a to reveal hidden things."""
            irc.reply("ls: not yet implemented")

        ls = wrap(ls, [optional("text")])

        def cat(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], target: str) -> None:
            """<thing> — Examine an item, NPC, or object."""
            irc.reply(f"cat: not yet implemented — target: {target}")

        cat = wrap(cat, ["text"])

        def pwd(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Show current location."""
            irc.reply("pwd: not yet implemented")

        pwd = wrap(pwd, [])

        def whoami(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Show character stats."""
            irc.reply("whoami: not yet implemented")

        whoami = wrap(whoami, [])

    rpg = Rpg
