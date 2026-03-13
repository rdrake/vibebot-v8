"""World map — room graph, path resolution, and starter map data."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field


@dataclass
class EnemySpawn:
    """An enemy that can appear in a room."""

    name: str
    hp: int
    attack: int
    defense: int
    xp_reward: int
    gold_reward: int
    count: int = 1


@dataclass
class ItemDrop:
    """An item that can be found in a room."""

    name: str
    attack_bonus: int = 0
    defense_bonus: int = 0
    description: str = ""


@dataclass
class Room:
    """A single room in the world."""

    path: str
    description_hint: str
    hidden: bool = False
    spawns: list[EnemySpawn] = field(default_factory=list)
    items: list[ItemDrop] = field(default_factory=list)


class WorldMap:
    """Graph of rooms with path-based navigation."""

    def __init__(self, rooms: dict[str, Room]) -> None:
        self._rooms = rooms

    @property
    def rooms(self) -> dict[str, Room]:
        return self._rooms

    def get_room(self, path: str) -> Room | None:
        """Get a room by its absolute path."""
        return self._rooms.get(path)

    def resolve_path(self, current: str, destination: str) -> str | None:
        """Resolve a destination path relative to current location.

        Supports absolute paths (/forest/cave), relative paths (cave),
        parent navigation (..), and root (/).

        Returns:
            Resolved absolute path if room exists, None otherwise.
        """
        if destination.startswith("/"):
            resolved = posixpath.normpath(destination)
        else:
            resolved = posixpath.normpath(posixpath.join(current, destination))

        # normpath may produce paths without leading /
        if not resolved.startswith("/"):
            resolved = "/" + resolved

        # Clamp to root
        if resolved == "/.":
            resolved = "/"

        if resolved in self._rooms:
            return resolved
        return None

    def get_exits(self, path: str, *, include_hidden: bool = True) -> list[str]:
        """List exits from a room.

        Returns child room basenames plus '..' if not at root.
        """
        exits: list[str] = []

        # Add parent exit if not at root
        if path != "/":
            exits.append("..")

        # Find child rooms (direct children only)
        prefix = path.rstrip("/") + "/"
        for room_path, room in self._rooms.items():
            if not room_path.startswith(prefix):
                continue
            # Only direct children (no further / after prefix)
            remainder = room_path[len(prefix) :]
            if "/" in remainder:
                continue
            if not include_hidden and room.hidden:
                continue
            exits.append(remainder)

        return sorted(exits)

    @classmethod
    def starter(cls) -> WorldMap:
        """Create the v1 starter world map (~12 rooms)."""
        rooms: dict[str, Room] = {}

        def add(
            path: str,
            hint: str,
            *,
            hidden: bool = False,
            spawns: list[EnemySpawn] | None = None,
            items: list[ItemDrop] | None = None,
        ) -> None:
            rooms[path] = Room(
                path=path,
                description_hint=hint,
                hidden=hidden,
                spawns=spawns or [],
                items=items or [],
            )

        # Root
        add("/", "The world stretches before you. Paths lead to town, forest, and dungeon.")

        # Town
        add("/town", "A quiet frontier town. Smoke rises from the blacksmith's forge.")
        add("/town/tavern", "The Rusty Pipe tavern. A fire crackles. The barkeep nods.")
        add(
            "/town/blacksmith",
            "Weapons and armor line the walls. The smith hammers at an anvil.",
        )
        add(
            "/town/.armory",
            "A hidden cache behind the blacksmith. Rare gear glints in the dark.",
            hidden=True,
            items=[
                ItemDrop(
                    "enchanted_shield.dat",
                    defense_bonus=3,
                    description="A shield that hums with energy.",
                )
            ],
        )

        # Forest
        add("/forest", "Ancient trees tower overhead. Something rustles in the underbrush.")
        add(
            "/forest/clearing",
            "A sun-dappled clearing. Small creatures scurry about.",
            spawns=[
                EnemySpawn("rat", hp=8, attack=2, defense=1, xp_reward=5, gold_reward=2, count=2)
            ],
        )
        add(
            "/forest/cave",
            "A dark cave mouth yawns open. Webs hang from the ceiling.",
            spawns=[EnemySpawn("spider", hp=15, attack=4, defense=2, xp_reward=12, gold_reward=5)],
            items=[
                ItemDrop(
                    "rusty_sword.txt",
                    attack_bonus=2,
                    description="A dull blade, better than fists.",
                )
            ],
        )
        add(
            "/forest/.fairy_grove",
            "A hidden glade shimmering with light. A healing spring bubbles.",
            hidden=True,
        )

        # Dungeon
        add("/dungeon", "Stone steps descend into darkness. Cold air rises from below.")
        add(
            "/dungeon/level1",
            "A damp corridor. Torchlight flickers on rough-hewn walls.",
            spawns=[
                EnemySpawn(
                    "goblin", hp=12, attack=3, defense=2, xp_reward=10, gold_reward=4, count=2
                )
            ],
        )
        add(
            "/dungeon/level2",
            "Bones crunch underfoot. The walls are carved with warnings.",
            spawns=[
                EnemySpawn("skeleton", hp=20, attack=5, defense=4, xp_reward=20, gold_reward=8)
            ],
            items=[
                ItemDrop("iron_armor.bin", defense_bonus=3, description="Dented but solid plate.")
            ],
        )
        add(
            "/dungeon/level3",
            "Pressure plates line the floor. The air smells of sulfur.",
            spawns=[
                EnemySpawn("dark_knight", hp=30, attack=7, defense=5, xp_reward=35, gold_reward=15)
            ],
        )
        add(
            "/dungeon/boss_chamber",
            "A vast cavern. A dragon coils on a mountain of gold.",
            spawns=[
                EnemySpawn("dragon", hp=100, attack=12, defense=8, xp_reward=200, gold_reward=100)
            ],
            items=[
                ItemDrop(
                    "dragonbane.exe",
                    attack_bonus=8,
                    description="A legendary blade forged to slay dragons.",
                )
            ],
        )

        return cls(rooms)
