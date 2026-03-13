"""RPG plugin implementation — IRC command layer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import supybot.callbacks as callbacks
import supybot.conf as conf
import supybot.log as log
from supybot.commands import optional, wrap

from .combat import CombatManager
from .engine import GameEngine
from .narrator import Narrator
from .persistence import RPGDatabase
from .world import WorldMap

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg


class RPG(callbacks.Plugin):
    """Linux filesystem RPG — explore dungeons with shell commands."""

    threaded = True

    def __init__(self, irc: callbacks.Irc) -> None:
        super().__init__(irc)
        self.log = log.getPluginLogger("RPG")

        # Database
        db_path = self.registryValue("databasePath")
        if not db_path:
            db_path = str(Path(conf.supybot.directories.data()) / "RPG.db")
        self._db = RPGDatabase(db_path)

        # World
        self._world = WorldMap.starter()

        # Engine and combat
        cooldown = self.registryValue("spawnCooldownMinutes")
        self._engine = GameEngine(db=self._db, world=self._world, spawn_cooldown_minutes=cooldown)
        self._combat = CombatManager(db=self._db, world=self._world)

        # Narrator
        self._narrator = Narrator(
            model=self.registryValue("narratorModel"),
            api_key=self.registryValue("narratorApiKey"),
            timeout=self.registryValue("narratorTimeout"),
        )

    def die(self) -> None:
        super().die()

    def _get_nick(self, msg: IrcMsg) -> str:
        """Extract nick from IRC message."""
        return msg.nick or "unknown"

    def _get_channel(self, msg: IrcMsg) -> str:
        """Extract channel from IRC message."""
        return msg.channel or msg.args[0]

    def _check_enabled(self, irc: callbacks.Irc, msg: IrcMsg) -> bool:
        """Check if RPG is enabled in this channel."""
        channel = self._get_channel(msg)
        if not self.registryValue("enabled", channel):
            irc.reply(
                "RPG is not enabled in this channel. Ask an admin to set plugins.RPG.enabled True."
            )
            return False
        return True

    class Rpg(callbacks.Commands):  # noqa: N801 — Limnoria maps class name to IRC command
        """RPG game commands — explore a Linux filesystem world."""

        def cd(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], destination: str) -> None:
            """<path> — Move to a location.

            Supports relative paths (tavern), absolute (/forest/cave), and parent (..)."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            event = plugin._engine.move(nick, channel, destination)
            if event.error:
                irc.reply(event.error)
                return

            text = plugin._narrator.narrate_room(
                room_path=event.location,
                description_hint=event.description_hint,
                enemies=event.enemies,
                items=event.items,
                exits=event.exits,
            )
            irc.reply(text)

        cd = wrap(cd, ["text"])

        def ls(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], flags: str) -> None:
            """[flags] — Look around. Use -a to reveal hidden things."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            show_hidden = flags is not None and "-a" in flags
            event = plugin._engine.look(nick, channel, show_hidden=show_hidden)
            if event.error:
                irc.reply(event.error)
                return

            parts = [event.location]
            if event.enemies:
                parts.append(f"Enemies: {', '.join(event.enemies)}")
            if event.items:
                parts.append(f"Items: {', '.join(event.items)}")
            parts.append(f"Exits: {', '.join(event.exits)}")
            irc.reply(" | ".join(parts))

        ls = wrap(ls, [optional("text")])

        def cat(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], target: str) -> None:
            """<thing> — Examine an item, NPC, or object."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            event = plugin._engine.examine(nick, channel, target)
            if event.error:
                irc.reply(event.error)
                return
            irc.reply(f"{event.target}: {event.description}")

        cat = wrap(cat, ["text"])

        def rm(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], enemy: str) -> None:
            """<enemy> — Attack an enemy."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            result = plugin._combat.attack(nick, channel, enemy)
            if result.error:
                irc.reply(result.error)
                return

            text = plugin._narrator.narrate_combat(
                attacker=nick,
                target=enemy,
                hit=result.hit,
                damage=result.damage,
                enemy_killed=result.enemy_killed,
            )
            parts = [text]

            if result.enemy_killed:
                parts.append(f"+{result.xp_gained} XP, +{result.gold_gained} gold")
            if result.counterattack_damage > 0:
                parts.append(f"{enemy} hits back for {result.counterattack_damage} dmg")
            if result.leveled_up:
                parts.append(f"LEVEL UP! You are now level {result.new_level}")
            if result.player_died:
                parts.append("You died! Respawning at /town/tavern...")

            irc.reply(" | ".join(parts))

        rm = wrap(rm, ["text"])

        def mv(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], text: str) -> None:
            """<item> ~/inventory — Pick up an item."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            # Parse: "item_name ~/inventory" or just "item_name"
            item_name = text.split()[0] if text else ""
            if not item_name:
                irc.reply("mv: missing operand")
                return

            result = plugin._combat.pickup_item(nick, channel, item_name)
            if result.error:
                irc.reply(result.error)
                return

            bonuses = []
            if result.attack_bonus:
                bonuses.append(f"ATK +{result.attack_bonus}")
            if result.defense_bonus:
                bonuses.append(f"DEF +{result.defense_bonus}")
            bonus_str = f" [{', '.join(bonuses)}]" if bonuses else ""
            irc.reply(f"Picked up {result.item_name}{bonus_str}")

        mv = wrap(mv, ["text"])

        def pwd(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Show current location."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            loc = plugin._engine.current_location(nick, channel)
            irc.reply(loc)

        pwd = wrap(pwd, [])

        def whoami(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Show character stats."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            info = plugin._engine.character_info(nick, channel)
            items = plugin._db.get_inventory(nick, channel)
            equipped = [i.name for i in items if i.equipped]

            line = (
                f"{info.nick} | HP:{info.hp}/{info.max_hp} ATK:{info.attack} "
                f"DEF:{info.defense} XP:{info.xp} LVL:{info.level} GOLD:{info.gold} "
                f"| {info.location}"
            )
            if equipped:
                line += f" | Equipped: {', '.join(equipped)}"
            irc.reply(line)

        whoami = wrap(whoami, [])

        def man(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str], topic: str) -> None:
            """<thing> — Get lore or help about something."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            # Check if it's an examinable thing in the room
            event = plugin._engine.examine(nick, channel, topic)
            if event.description:
                irc.reply(f"{event.target}: {event.description}")
                return

            # Generic help for commands
            help_text = {
                "cd": "cd <path> -- Move. Supports relative, absolute, and .. paths.",
                "ls": "ls [-a] -- Look around. -a reveals hidden dotfile rooms.",
                "cat": "cat <thing> -- Examine an item, enemy, or object.",
                "rm": "rm <enemy> -- Attack an enemy. d20 + ATK vs DEF + 10.",
                "mv": "mv <item> ~/inventory -- Pick up an item from the room.",
                "pwd": "pwd -- Show your current location.",
                "whoami": "whoami -- Show your character stats.",
                "man": "man <topic> -- Get help or lore about something.",
                "sleep": "sleep -- Rest and recover HP (outside combat).",
                "history": "history -- Show your recent actions.",
            }
            if topic.lower() in help_text:
                irc.reply(help_text[topic.lower()])
            else:
                irc.reply(f"man: no manual entry for {topic}")

        man = wrap(man, ["text"])

        def sleep(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Rest and recover HP (outside combat only)."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            event = plugin._engine.rest(nick, channel)
            if event.hp_restored == 0:
                irc.reply("You rest but you're already at full HP.")
            else:
                irc.reply(
                    f"You rest... HP restored: +{event.hp_restored}"
                    f" (now {event.hp_after}/{event.hp_after})"
                )

        sleep = wrap(sleep, [])

        def history(self, irc: callbacks.Irc, msg: IrcMsg, args: list[str]) -> None:
            """— Show recent actions (placeholder for v1)."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            irc.reply("history: not yet implemented")

        history = wrap(history, [])

    rpg = Rpg
