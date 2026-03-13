"""Game engine — movement, inspection, character management, rest."""

from __future__ import annotations

import time
from typing import NamedTuple

from .persistence import CharacterRow, RPGDatabase
from .world import WorldMap


class MoveEvent(NamedTuple):
    """Result of a cd command."""

    location: str
    description_hint: str
    enemies: list[str]
    items: list[str]
    exits: list[str]
    error: str | None = None


class LookEvent(NamedTuple):
    """Result of an ls command."""

    location: str
    description_hint: str
    enemies: list[str]
    items: list[str]
    exits: list[str]
    error: str | None = None


class ExamineEvent(NamedTuple):
    """Result of a cat command."""

    target: str
    description: str | None = None
    error: str | None = None


class RestEvent(NamedTuple):
    """Result of a sleep command."""

    hp_before: int
    hp_after: int
    hp_restored: int


class CharacterInfo(NamedTuple):
    """Character stats for whoami."""

    nick: str
    hp: int
    max_hp: int
    attack: int
    defense: int
    xp: int
    level: int
    gold: int
    location: str


class GameEngine:
    """Core game logic — movement, inspection, character lifecycle."""

    def __init__(
        self,
        *,
        db: RPGDatabase,
        world: WorldMap,
        spawn_cooldown_minutes: int = 30,
    ) -> None:
        self.db = db
        self.world = world
        self._spawn_cooldown_seconds = spawn_cooldown_minutes * 60

    def ensure_character(self, nick: str, channel: str) -> CharacterRow:
        """Get or create a character, returning the current state."""
        self.db.create_character(nick, channel)
        char = self.db.get_character(nick, channel)
        assert char is not None  # Just created, must exist
        return char

    def _get_room_enemies(self, channel: str, room_path: str) -> list[str]:
        """Get active enemies in a room (respects spawn cooldown)."""
        room = self.world.get_room(room_path)
        if room is None or not room.spawns:
            return []

        cleared_at = self.db.get_room_cleared_at(channel, room_path)
        if cleared_at is not None:
            elapsed = time.time() - cleared_at
            if elapsed < self._spawn_cooldown_seconds:
                return []
            # Cooldown expired — reset so enemies respawn
            self.db.reset_room(channel, room_path)

        enemies: list[str] = []
        for spawn in room.spawns:
            enemies.extend([spawn.name] * spawn.count)
        return enemies

    def _get_room_items(self, room_path: str) -> list[str]:
        """Get items available in a room."""
        room = self.world.get_room(room_path)
        if room is None:
            return []
        return [item.name for item in room.items]

    def move(self, nick: str, channel: str, destination: str) -> MoveEvent:
        """Move player to a new location (cd command)."""
        char = self.ensure_character(nick, channel)
        resolved = self.world.resolve_path(char.location, destination)

        if resolved is None:
            return MoveEvent(
                location=char.location,
                description_hint="",
                enemies=[],
                items=[],
                exits=[],
                error=f"cd: no such file or directory: {destination}",
            )

        self.db.update_character(nick, channel, location=resolved)
        room = self.world.get_room(resolved)
        assert room is not None

        enemies = self._get_room_enemies(channel, resolved)
        items = self._get_room_items(resolved)
        exits = self.world.get_exits(resolved, include_hidden=False)

        return MoveEvent(
            location=resolved,
            description_hint=room.description_hint,
            enemies=enemies,
            items=items,
            exits=exits,
        )

    def look(self, nick: str, channel: str, *, show_hidden: bool = False) -> LookEvent:
        """Look around current room (ls command)."""
        char = self.ensure_character(nick, channel)
        room = self.world.get_room(char.location)

        if room is None:
            return LookEvent(
                location=char.location,
                description_hint="",
                enemies=[],
                items=[],
                exits=[],
                error="ls: cannot access: room not found",
            )

        enemies = self._get_room_enemies(channel, char.location)
        items = self._get_room_items(char.location)
        exits = self.world.get_exits(char.location, include_hidden=show_hidden)

        return LookEvent(
            location=char.location,
            description_hint=room.description_hint,
            enemies=enemies,
            items=items,
            exits=exits,
        )

    def examine(self, nick: str, channel: str, target: str) -> ExamineEvent:
        """Examine an item or object (cat command)."""
        char = self.ensure_character(nick, channel)
        room = self.world.get_room(char.location)

        if room is None:
            return ExamineEvent(target=target, error="cat: room not found")

        # Check room items
        for item in room.items:
            if item.name == target:
                desc = item.description or f"A mysterious item: {item.name}"
                if item.attack_bonus:
                    desc += f" [ATK +{item.attack_bonus}]"
                if item.defense_bonus:
                    desc += f" [DEF +{item.defense_bonus}]"
                return ExamineEvent(target=target, description=desc)

        # Check enemies
        for spawn in room.spawns:
            if spawn.name == target:
                return ExamineEvent(
                    target=target,
                    description=(
                        f"{spawn.name} — HP:{spawn.hp} ATK:{spawn.attack} DEF:{spawn.defense}"
                    ),
                )

        return ExamineEvent(target=target, error=f"cat: {target}: No such file or directory")

    def character_info(self, nick: str, channel: str) -> CharacterInfo:
        """Get character stats (whoami command)."""
        char = self.ensure_character(nick, channel)
        return CharacterInfo(
            nick=char.nick,
            hp=char.hp,
            max_hp=char.max_hp,
            attack=char.attack,
            defense=char.defense,
            xp=char.xp,
            level=char.level,
            gold=char.gold,
            location=char.location,
        )

    def current_location(self, nick: str, channel: str) -> str:
        """Get current location (pwd command)."""
        char = self.ensure_character(nick, channel)
        return char.location

    def rest(self, nick: str, channel: str) -> RestEvent:
        """Rest and recover HP (sleep command)."""
        char = self.ensure_character(nick, channel)
        hp_before = char.hp

        if char.hp >= char.max_hp:
            return RestEvent(hp_before=hp_before, hp_after=char.hp, hp_restored=0)

        # Restore 50% of missing HP (minimum 1)
        missing = char.max_hp - char.hp
        restored = max(1, missing // 2)
        new_hp = min(char.max_hp, char.hp + restored)

        self.db.update_character(nick, channel, hp=new_hp)
        return RestEvent(hp_before=hp_before, hp_after=new_hp, hp_restored=new_hp - hp_before)
