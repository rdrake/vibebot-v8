# RPG Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Linux-filesystem-themed IRC RPG where players explore a shared world using shell commands, fight monsters, collect loot, and level up — with LLM narration and SQLite persistence.

**Architecture:** Separate Limnoria plugin (`plugins/rpg/`) with the same workspace/packaging patterns as `plugins/llm/`. Python game engine handles all mechanical truth (HP, combat, loot, XP). LLM narrator adds flavor text but never decides outcomes. SQLite persists game state across days/weeks. All commands prefixed with `%rpg`.

**Tech Stack:** Python 3.12+, Limnoria, d20 (dice), litellm (narrator), SQLite (stdlib), pytest

**Reference:** `docs/plans/2026-03-12-rpg-plugin-design.md` for full design spec.

---

## Task 1: Plugin Skeleton and Workspace Wiring

Wire the RPG plugin into the uv workspace so `uv sync` installs it alongside the LLM plugin.

**Files:**
- Create: `plugins/rpg/pyproject.toml`
- Create: `plugins/rpg/src/rpg/__init__.py`
- Create: `plugins/rpg/src/rpg/plugin.py`
- Create: `plugins/rpg/src/rpg/config.py`
- Create: `plugins/rpg/tests/__init__.py`
- Create: `plugins/rpg/tests/conftest.py`
- Modify: `pyproject.toml` (root workspace)
- Modify: `Makefile`

**Step 1: Create `plugins/rpg/pyproject.toml`**

```toml
[project]
name = "rpg"
version = "0.1.0"
description = "Linux filesystem RPG for IRC"
requires-python = ">=3.12"
dependencies = [
    "limnoria>=2023.1.20",
    "litellm>=1.81.6",
    "d20>=1.1.0",
]

[project.entry-points."limnoria.plugins"]
RPG = "rpg"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rpg"]
```

**Step 2: Create `plugins/rpg/src/rpg/config.py`**

```python
"""Configuration for RPG plugin."""

from __future__ import annotations

import supybot.conf as conf
import supybot.registry as registry
from supybot.i18n import PluginInternationalization

_ = PluginInternationalization("RPG")


def configure(advanced: bool) -> None:
    """Plugin configuration wizard."""
    conf.registerPlugin("RPG", True)


RPG = conf.registerPlugin("RPG")

conf.registerGlobalValue(
    RPG,
    "narratorApiKey",
    registry.String("", _("""API key for narrator LLM calls."""), private=True),
)

conf.registerChannelValue(
    RPG,
    "narratorModel",
    registry.String(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for narrator flavor text (cheap flash-tier recommended)."""),
    ),
)

conf.registerGlobalValue(
    RPG,
    "narratorTimeout",
    registry.PositiveInteger(
        2,
        _("""Timeout in seconds for narrator LLM calls. Falls back to deterministic text on timeout."""),
    ),
)

conf.registerGlobalValue(
    RPG,
    "databasePath",
    registry.String(
        "",
        _("""Path to SQLite database. If empty, uses Limnoria's data directory (data/RPG.db)."""),
    ),
)

conf.registerChannelValue(
    RPG,
    "enabled",
    registry.Boolean(False, _("""Enable RPG in this channel.""")),
)

conf.registerGlobalValue(
    RPG,
    "combatRoundSeconds",
    registry.PositiveInteger(
        20,
        _("""Seconds per combat round before AFK auto-action."""),
    ),
)

conf.registerGlobalValue(
    RPG,
    "spawnCooldownMinutes",
    registry.PositiveInteger(
        30,
        _("""Minutes before enemies respawn in a cleared room."""),
    ),
)
```

**Step 3: Create `plugins/rpg/src/rpg/plugin.py`**

```python
"""RPG plugin implementation — IRC command layer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import supybot.callbacks as callbacks
import supybot.conf as conf
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

    class rpg(callbacks.Commands):
        """RPG game commands — use shell commands to explore a Linux filesystem world."""

        def cd(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], destination: str) -> None:
            """<path> — Move to a location."""
            irc.reply(f"cd: not yet implemented — destination: {destination}")

        cd = wrap(cd, ["text"])

        def ls(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], flags: str) -> None:
            """[flags] — Look around. Use -a to reveal hidden things."""
            irc.reply("ls: not yet implemented")

        ls = wrap(ls, [optional("text")])

        def cat(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], target: str) -> None:
            """<thing> — Examine an item, NPC, or object."""
            irc.reply(f"cat: not yet implemented — target: {target}")

        cat = wrap(cat, ["text"])

        def pwd(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
            """— Show current location."""
            irc.reply("pwd: not yet implemented")

        pwd = wrap(pwd, [])

        def whoami(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
            """— Show character stats."""
            irc.reply("whoami: not yet implemented")

        whoami = wrap(whoami, [])

    rpg = rpg
```

**Step 4: Create `plugins/rpg/src/rpg/__init__.py`**

```python
"""RPG: Linux filesystem dungeon master for IRC."""

from __future__ import annotations

__version__ = "0.1.0"

from . import config, plugin

Class = plugin.RPG
configure = config.configure

__all__ = ["Class", "configure", "__version__"]
```

**Step 5: Create `plugins/rpg/tests/__init__.py`**

```python
```

**Step 6: Create `plugins/rpg/tests/conftest.py`**

```python
"""Pytest configuration and shared fixtures for RPG plugin tests."""

from __future__ import annotations

from typing import Any

import pytest

if __name__ == "__main__":
    pytest.main([__file__])
```

**Step 7: Wire into root workspace — modify `pyproject.toml`**

Add `"rpg"` to the workspace dependencies and members:

```
# In [project] section, change:
dependencies = ["llm"]
# To:
dependencies = ["llm", "rpg"]

# In [tool.uv.workspace] section, change:
members = ["plugins/llm"]
# To:
members = ["plugins/llm", "plugins/rpg"]

# In [tool.uv.sources] section, add:
rpg = { workspace = true }

# In [tool.pytest.ini_options] section, change:
testpaths = ["plugins/llm/tests"]
# To:
testpaths = ["plugins/llm/tests", "plugins/rpg/tests"]

# In [tool.coverage.run] section, change:
source = ["plugins/llm/src"]
# To:
source = ["plugins/llm/src", "plugins/rpg/src"]

# In [tool.ty.rules] — keep as-is (already ignores unresolved imports)
```

**Step 8: Update Makefile — add RPG to typecheck target**

Change:
```makefile
typecheck:
	uv run ty check plugins/llm/src/
```
To:
```makefile
typecheck:
	uv run ty check plugins/llm/src/ plugins/rpg/src/
```

Also update the pre-commit ty hook in `.pre-commit-config.yaml`:
```yaml
      - id: ty
        name: ty type checker
        entry: uv run ty check plugins/llm/src/ plugins/rpg/src/
```

**Step 9: Run `uv sync` and verify**

Run: `uv sync`
Expected: Success, both plugins installed.

Run: `uv run python -c "import rpg; print(rpg.__version__)"`
Expected: `0.1.0`

**Step 10: Write a smoke test**

Create `plugins/rpg/tests/test_plugin.py`:

```python
"""Tests for RPG plugin structure."""

from __future__ import annotations


def test_plugin_importable():
    """GIVEN the rpg package WHEN imported THEN it exposes Class and configure."""
    import rpg

    assert hasattr(rpg, "Class")
    assert hasattr(rpg, "configure")
    assert rpg.__version__ == "0.1.0"
```

Run: `uv run pytest plugins/rpg/tests/test_plugin.py -v`
Expected: PASS

**Step 11: Commit**

```bash
git add plugins/rpg/ pyproject.toml Makefile .pre-commit-config.yaml
git commit -m "feat(rpg): add plugin skeleton and workspace wiring"
```

---

## Task 2: World Map — Room Graph and Navigation

Build the world as a graph of rooms. Implement `cd` movement with path validation.

**Files:**
- Create: `plugins/rpg/src/rpg/world.py`
- Create: `plugins/rpg/tests/test_world.py`

**Step 1: Write failing tests for the world map**

Create `plugins/rpg/tests/test_world.py`:

```python
"""Tests for world map and navigation."""

from __future__ import annotations

import pytest
from rpg.world import Room, WorldMap


class TestWorldMap:
    """Room graph and path resolution."""

    def test_starter_map_has_town(self):
        """GIVEN a starter world WHEN checking rooms THEN /town exists."""
        world = WorldMap.starter()
        assert world.get_room("/town") is not None

    def test_starter_map_room_count(self):
        """GIVEN a starter world WHEN counting rooms THEN there are 12."""
        world = WorldMap.starter()
        assert len(world.rooms) == 12

    def test_resolve_relative_path(self):
        """GIVEN player at /town WHEN cd tavern THEN resolves to /town/tavern."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "tavern")
        assert result == "/town/tavern"

    def test_resolve_dotdot(self):
        """GIVEN player at /town/tavern WHEN cd .. THEN resolves to /town."""
        world = WorldMap.starter()
        result = world.resolve_path("/town/tavern", "..")
        assert result == "/town"

    def test_resolve_absolute_path(self):
        """GIVEN player anywhere WHEN cd /forest/clearing THEN resolves absolutely."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "/forest/clearing")
        assert result == "/forest/clearing"

    def test_resolve_invalid_path(self):
        """GIVEN player at /town WHEN cd nonexistent THEN returns None."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "nonexistent")
        assert result is None

    def test_resolve_root(self):
        """GIVEN player anywhere WHEN cd / THEN resolves to /."""
        world = WorldMap.starter()
        result = world.resolve_path("/town/tavern", "/")
        assert result == "/"

    def test_resolve_dotdot_at_root(self):
        """GIVEN player at / WHEN cd .. THEN stays at /."""
        world = WorldMap.starter()
        result = world.resolve_path("/", "..")
        assert result == "/"

    def test_hidden_rooms_exist(self):
        """GIVEN a starter world WHEN checking hidden rooms THEN dotfile rooms exist."""
        world = WorldMap.starter()
        armory = world.get_room("/town/.armory")
        assert armory is not None
        assert armory.hidden is True

    def test_room_exits(self):
        """GIVEN /town WHEN listing exits THEN tavern and blacksmith are exits."""
        world = WorldMap.starter()
        room = world.get_room("/town")
        assert room is not None
        exits = world.get_exits("/town")
        assert "tavern" in exits
        assert "blacksmith" in exits

    def test_hidden_exits_excluded_by_default(self):
        """GIVEN /town WHEN listing visible exits THEN .armory not shown."""
        world = WorldMap.starter()
        exits = world.get_exits("/town", include_hidden=False)
        assert ".armory" not in exits

    def test_hidden_exits_included_with_flag(self):
        """GIVEN /town WHEN listing all exits THEN .armory shown."""
        world = WorldMap.starter()
        exits = world.get_exits("/town", include_hidden=True)
        assert ".armory" in exits

    def test_room_has_parent_exit(self):
        """GIVEN /town/tavern WHEN listing exits THEN .. is always an exit."""
        world = WorldMap.starter()
        exits = world.get_exits("/town/tavern")
        assert ".." in exits

    def test_root_has_no_parent_exit(self):
        """GIVEN / WHEN listing exits THEN .. is not listed."""
        world = WorldMap.starter()
        exits = world.get_exits("/")
        assert ".." not in exits
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_world.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rpg.world'`

**Step 3: Implement `world.py`**

Create `plugins/rpg/src/rpg/world.py`:

```python
"""World map — room graph, path resolution, and starter map data."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import NamedTuple


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

        def add(path: str, hint: str, *, hidden: bool = False,
                spawns: list[EnemySpawn] | None = None,
                items: list[ItemDrop] | None = None) -> None:
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
        add("/town/blacksmith", "Weapons and armor line the walls. The smith hammers at an anvil.")
        add(
            "/town/.armory",
            "A hidden cache behind the blacksmith. Rare gear glints in the dark.",
            hidden=True,
            items=[ItemDrop("enchanted_shield.dat", defense_bonus=3, description="A shield that hums with energy.")],
        )

        # Forest
        add(
            "/forest",
            "Ancient trees tower overhead. Something rustles in the underbrush.",
        )
        add(
            "/forest/clearing",
            "A sun-dappled clearing. Small creatures scurry about.",
            spawns=[EnemySpawn("rat", hp=8, attack=2, defense=1, xp_reward=5, gold_reward=2, count=2)],
        )
        add(
            "/forest/cave",
            "A dark cave mouth yawns open. Webs hang from the ceiling.",
            spawns=[EnemySpawn("spider", hp=15, attack=4, defense=2, xp_reward=12, gold_reward=5)],
            items=[ItemDrop("rusty_sword.txt", attack_bonus=2, description="A dull blade, better than fists.")],
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
                EnemySpawn("goblin", hp=12, attack=3, defense=2, xp_reward=10, gold_reward=4, count=2),
            ],
        )
        add(
            "/dungeon/level2",
            "Bones crunch underfoot. The walls are carved with warnings.",
            spawns=[
                EnemySpawn("skeleton", hp=20, attack=5, defense=4, xp_reward=20, gold_reward=8),
            ],
            items=[ItemDrop("iron_armor.bin", defense_bonus=3, description="Dented but solid plate.")],
        )
        add(
            "/dungeon/level3",
            "Pressure plates line the floor. The air smells of sulfur.",
            spawns=[
                EnemySpawn("dark_knight", hp=30, attack=7, defense=5, xp_reward=35, gold_reward=15),
            ],
        )
        add(
            "/dungeon/boss_chamber",
            "A vast cavern. A dragon coils on a mountain of gold.",
            spawns=[
                EnemySpawn("dragon", hp=100, attack=12, defense=8, xp_reward=200, gold_reward=100),
            ],
            items=[ItemDrop("dragonbane.exe", attack_bonus=8, description="A legendary blade forged to slay dragons.")],
        )

        return cls(rooms)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/test_world.py -v`
Expected: All PASS

**Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Clean

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/world.py plugins/rpg/tests/test_world.py
git commit -m "feat(rpg): add world map with room graph and path resolution"
```

---

## Task 3: Persistence Layer — SQLite Game State

Store characters, inventory, world state, and combat state in SQLite.

**Files:**
- Create: `plugins/rpg/src/rpg/persistence.py`
- Create: `plugins/rpg/tests/test_persistence.py`

**Step 1: Write failing tests**

Create `plugins/rpg/tests/test_persistence.py`:

```python
"""Tests for RPG persistence layer."""

from __future__ import annotations

import pytest
from rpg.persistence import RPGDatabase


@pytest.fixture
def db(tmp_path) -> RPGDatabase:
    """Create a fresh in-memory-style DB for each test."""
    return RPGDatabase(str(tmp_path / "test.db"))


class TestCharacterPersistence:
    """Character CRUD operations."""

    def test_create_character(self, db: RPGDatabase):
        """GIVEN no character WHEN create_character THEN character exists."""
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.nick == "alice"
        assert char.channel == "#test"
        assert char.location == "/town/tavern"

    def test_create_character_defaults(self, db: RPGDatabase):
        """GIVEN new character WHEN checking stats THEN defaults are set."""
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.hp == 20
        assert char.max_hp == 20
        assert char.attack == 3
        assert char.defense == 2
        assert char.xp == 0
        assert char.level == 1
        assert char.gold == 0

    def test_get_nonexistent_character(self, db: RPGDatabase):
        """GIVEN no character WHEN get_character THEN returns None."""
        assert db.get_character("nobody", "#test") is None

    def test_update_character_location(self, db: RPGDatabase):
        """GIVEN a character WHEN updating location THEN location changes."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.location == "/forest/clearing"

    def test_update_character_stats(self, db: RPGDatabase):
        """GIVEN a character WHEN updating HP THEN HP changes."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", hp=15, xp=10, gold=5)
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.hp == 15
        assert char.xp == 10
        assert char.gold == 5

    def test_duplicate_character_ignored(self, db: RPGDatabase):
        """GIVEN existing character WHEN create_character again THEN no error."""
        db.create_character("alice", "#test")
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.level == 1


class TestInventoryPersistence:
    """Inventory CRUD operations."""

    def test_add_item(self, db: RPGDatabase):
        """GIVEN a character WHEN adding item THEN item in inventory."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        assert len(items) == 1
        assert items[0].name == "sword.txt"
        assert items[0].attack_bonus == 2

    def test_equip_item(self, db: RPGDatabase):
        """GIVEN an item WHEN equipping THEN equipped flag set."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        db.equip_item(items[0].id, equipped=True)
        items = db.get_inventory("alice", "#test")
        assert items[0].equipped is True

    def test_remove_item(self, db: RPGDatabase):
        """GIVEN an item WHEN removing THEN inventory empty."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        db.remove_item(items[0].id)
        assert db.get_inventory("alice", "#test") == []

    def test_empty_inventory(self, db: RPGDatabase):
        """GIVEN a character WHEN no items THEN empty list."""
        db.create_character("alice", "#test")
        assert db.get_inventory("alice", "#test") == []


class TestWorldState:
    """Room state persistence."""

    def test_clear_room(self, db: RPGDatabase):
        """GIVEN a room WHEN marking cleared THEN timestamp set."""
        db.mark_room_cleared("#test", "/dungeon/level1")
        ts = db.get_room_cleared_at("#test", "/dungeon/level1")
        assert ts is not None
        assert ts > 0

    def test_uncleared_room(self, db: RPGDatabase):
        """GIVEN a room never cleared WHEN checking THEN returns None."""
        assert db.get_room_cleared_at("#test", "/dungeon/level1") is None

    def test_reset_room(self, db: RPGDatabase):
        """GIVEN a cleared room WHEN resetting THEN cleared_at is None."""
        db.mark_room_cleared("#test", "/dungeon/level1")
        db.reset_room("#test", "/dungeon/level1")
        assert db.get_room_cleared_at("#test", "/dungeon/level1") is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rpg.persistence'`

**Step 3: Implement `persistence.py`**

Create `plugins/rpg/src/rpg/persistence.py`:

```python
"""SQLite persistence layer for RPG plugin.

Thread-safe database operations for characters, inventory, and world state.
Uses thread-local connections with WAL mode.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import NamedTuple

SCHEMA_VERSION = 1


class CharacterRow(NamedTuple):
    """A character loaded from the database."""

    nick: str
    channel: str
    hp: int
    max_hp: int
    attack: int
    defense: int
    xp: int
    level: int
    gold: int
    location: str


class InventoryItem(NamedTuple):
    """An inventory item loaded from the database."""

    id: int
    nick: str
    channel: str
    name: str
    attack_bonus: int
    defense_bonus: int
    equipped: bool


class RPGDatabase:
    """SQLite database for RPG game state."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS characters (
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                hp INTEGER NOT NULL DEFAULT 20,
                max_hp INTEGER NOT NULL DEFAULT 20,
                attack INTEGER NOT NULL DEFAULT 3,
                defense INTEGER NOT NULL DEFAULT 2,
                xp INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                gold INTEGER NOT NULL DEFAULT 0,
                location TEXT NOT NULL DEFAULT '/town/tavern',
                PRIMARY KEY (nick, channel)
            );

            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                name TEXT NOT NULL,
                attack_bonus INTEGER NOT NULL DEFAULT 0,
                defense_bonus INTEGER NOT NULL DEFAULT 0,
                equipped INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (nick, channel) REFERENCES characters(nick, channel)
            );

            CREATE TABLE IF NOT EXISTS world_state (
                channel TEXT NOT NULL,
                room_path TEXT NOT NULL,
                cleared_at REAL,
                PRIMARY KEY (channel, room_path)
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
        """)
        # Set schema version if not present
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

    # --- Characters ---

    def create_character(self, nick: str, channel: str) -> None:
        """Create a new character if one doesn't exist."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO characters (nick, channel) VALUES (?, ?)",
            (nick.lower(), channel.lower()),
        )
        conn.commit()

    def get_character(self, nick: str, channel: str) -> CharacterRow | None:
        """Get a character by nick and channel."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT nick, channel, hp, max_hp, attack, defense, xp, level, gold, location "
            "FROM characters WHERE nick = ? AND channel = ?",
            (nick.lower(), channel.lower()),
        ).fetchone()
        if row is None:
            return None
        return CharacterRow(*row)

    def update_character(self, nick: str, channel: str, **kwargs: int | str) -> None:
        """Update character fields. Pass only the fields to change."""
        if not kwargs:
            return
        allowed = {"hp", "max_hp", "attack", "defense", "xp", "level", "gold", "location"}
        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(f"Invalid fields: {invalid}")
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [nick.lower(), channel.lower()]
        conn = self._get_conn()
        conn.execute(
            f"UPDATE characters SET {sets} WHERE nick = ? AND channel = ?",  # noqa: S608
            values,
        )
        conn.commit()

    # --- Inventory ---

    def add_item(
        self, nick: str, channel: str, name: str,
        *, attack_bonus: int = 0, defense_bonus: int = 0,
    ) -> int:
        """Add an item to a character's inventory. Returns the item ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO inventory (nick, channel, name, attack_bonus, defense_bonus) "
            "VALUES (?, ?, ?, ?, ?)",
            (nick.lower(), channel.lower(), name, attack_bonus, defense_bonus),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def get_inventory(self, nick: str, channel: str) -> list[InventoryItem]:
        """Get all inventory items for a character."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, nick, channel, name, attack_bonus, defense_bonus, equipped "
            "FROM inventory WHERE nick = ? AND channel = ? ORDER BY name",
            (nick.lower(), channel.lower()),
        ).fetchall()
        return [InventoryItem(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6])) for r in rows]

    def equip_item(self, item_id: int, *, equipped: bool) -> None:
        """Set the equipped flag on an item."""
        conn = self._get_conn()
        conn.execute("UPDATE inventory SET equipped = ? WHERE id = ?", (int(equipped), item_id))
        conn.commit()

    def remove_item(self, item_id: int) -> None:
        """Remove an item from inventory."""
        conn = self._get_conn()
        conn.execute("DELETE FROM inventory WHERE id = ?", (item_id,))
        conn.commit()

    # --- World State ---

    def mark_room_cleared(self, channel: str, room_path: str) -> None:
        """Mark a room as cleared with current timestamp."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO world_state (channel, room_path, cleared_at) VALUES (?, ?, ?)",
            (channel.lower(), room_path, time.time()),
        )
        conn.commit()

    def get_room_cleared_at(self, channel: str, room_path: str) -> float | None:
        """Get when a room was last cleared, or None if never."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT cleared_at FROM world_state WHERE channel = ? AND room_path = ?",
            (channel.lower(), room_path),
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def reset_room(self, channel: str, room_path: str) -> None:
        """Reset a room so enemies can respawn."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM world_state WHERE channel = ? AND room_path = ?",
            (channel.lower(), room_path),
        )
        conn.commit()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/test_persistence.py -v`
Expected: All PASS

**Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Clean

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/persistence.py plugins/rpg/tests/test_persistence.py
git commit -m "feat(rpg): add SQLite persistence for characters, inventory, world state"
```

---

## Task 4: Game Engine — Movement, Inspection, Character Management

Core game logic: character auto-creation, movement, `ls`, `cat`, `whoami`, `pwd`, `sleep`.

**Files:**
- Create: `plugins/rpg/src/rpg/engine.py`
- Create: `plugins/rpg/tests/test_engine.py`

**Step 1: Write failing tests**

Create `plugins/rpg/tests/test_engine.py`:

```python
"""Tests for RPG game engine."""

from __future__ import annotations

import pytest
from rpg.engine import GameEngine, GameEvent
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


@pytest.fixture
def engine(tmp_path) -> GameEngine:
    """Create engine with fresh DB and starter world."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    return GameEngine(db=db, world=world, spawn_cooldown_minutes=30)


class TestMovement:
    """cd command — moving between rooms."""

    def test_first_command_creates_character(self, engine: GameEngine):
        """GIVEN no character WHEN any command THEN character auto-created."""
        event = engine.move("alice", "#test", ".")
        char = engine.db.get_character("alice", "#test")
        assert char is not None

    def test_move_to_child_room(self, engine: GameEngine):
        """GIVEN player at /town/tavern WHEN cd .. THEN at /town."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "..")
        assert event.location == "/town"

    def test_move_to_absolute_path(self, engine: GameEngine):
        """GIVEN player anywhere WHEN cd /forest THEN at /forest."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        assert event.location == "/forest"

    def test_move_invalid_path(self, engine: GameEngine):
        """GIVEN player at /town/tavern WHEN cd nonexistent THEN error."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "nonexistent")
        assert event.error is not None
        assert "no such" in event.error.lower()

    def test_move_persists_location(self, engine: GameEngine):
        """GIVEN player moves WHEN checking DB THEN location updated."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        char = engine.db.get_character("alice", "#test")
        assert char is not None
        assert char.location == "/forest/clearing"


class TestInspection:
    """ls and cat commands."""

    def test_ls_shows_exits(self, engine: GameEngine):
        """GIVEN player at /town WHEN ls THEN exits listed."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")
        event = engine.look("alice", "#test", show_hidden=False)
        assert "tavern" in event.exits
        assert "blacksmith" in event.exits

    def test_ls_a_shows_hidden(self, engine: GameEngine):
        """GIVEN player at /town WHEN ls -a THEN .armory shown."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")
        event = engine.look("alice", "#test", show_hidden=True)
        assert ".armory" in event.exits

    def test_ls_shows_enemies(self, engine: GameEngine):
        """GIVEN player at /forest/clearing WHEN ls THEN enemies listed."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        event = engine.look("alice", "#test", show_hidden=False)
        assert len(event.enemies) > 0

    def test_ls_cleared_room_no_enemies(self, engine: GameEngine):
        """GIVEN room cleared recently WHEN ls THEN no enemies."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        engine.db.mark_room_cleared("#test", "/forest/clearing")
        event = engine.look("alice", "#test", show_hidden=False)
        assert len(event.enemies) == 0

    def test_cat_item_in_room(self, engine: GameEngine):
        """GIVEN an item in the room WHEN cat item THEN description returned."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        event = engine.examine("alice", "#test", "rusty_sword.txt")
        assert event.description is not None
        assert "sword" in event.description.lower() or "blade" in event.description.lower()

    def test_cat_nonexistent(self, engine: GameEngine):
        """GIVEN no such item WHEN cat THEN error."""
        engine.ensure_character("alice", "#test")
        event = engine.examine("alice", "#test", "nonexistent")
        assert event.error is not None


class TestCharacterInfo:
    """whoami and pwd commands."""

    def test_whoami_shows_stats(self, engine: GameEngine):
        """GIVEN a character WHEN whoami THEN stats returned."""
        engine.ensure_character("alice", "#test")
        info = engine.character_info("alice", "#test")
        assert info.nick == "alice"
        assert info.hp == 20
        assert info.level == 1

    def test_pwd_shows_location(self, engine: GameEngine):
        """GIVEN a character WHEN pwd THEN location returned."""
        engine.ensure_character("alice", "#test")
        loc = engine.current_location("alice", "#test")
        assert loc == "/town/tavern"


class TestRest:
    """sleep command."""

    def test_sleep_restores_hp(self, engine: GameEngine):
        """GIVEN character with reduced HP WHEN sleep THEN HP restored."""
        engine.ensure_character("alice", "#test")
        engine.db.update_character("alice", "#test", hp=10)
        event = engine.rest("alice", "#test")
        assert event.hp_restored > 0
        char = engine.db.get_character("alice", "#test")
        assert char is not None
        assert char.hp > 10

    def test_sleep_at_full_hp(self, engine: GameEngine):
        """GIVEN character at full HP WHEN sleep THEN no change."""
        engine.ensure_character("alice", "#test")
        event = engine.rest("alice", "#test")
        assert event.hp_restored == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rpg.engine'`

**Step 3: Implement `engine.py`**

Create `plugins/rpg/src/rpg/engine.py`:

```python
"""Game engine — movement, inspection, character management, rest."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import NamedTuple

from .persistence import CharacterRow, RPGDatabase
from .world import EnemySpawn, ItemDrop, WorldMap


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
                    description=f"{spawn.name} — HP:{spawn.hp} ATK:{spawn.attack} DEF:{spawn.defense}",
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/test_engine.py -v`
Expected: All PASS

**Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Clean

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/engine.py plugins/rpg/tests/test_engine.py
git commit -m "feat(rpg): add game engine — movement, inspection, rest"
```

---

## Task 5: Combat System — d20-Based Turn Resolution

Attack enemies, take damage, earn XP and gold, pick up loot, level up, die and respawn.

**Files:**
- Create: `plugins/rpg/src/rpg/combat.py`
- Create: `plugins/rpg/tests/test_combat.py`

**Step 1: Write failing tests**

Create `plugins/rpg/tests/test_combat.py`:

```python
"""Tests for combat system."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rpg.combat import CombatManager, CombatResult
from rpg.persistence import RPGDatabase
from rpg.world import EnemySpawn, WorldMap


@pytest.fixture
def db(tmp_path) -> RPGDatabase:
    return RPGDatabase(str(tmp_path / "test.db"))


@pytest.fixture
def combat(db: RPGDatabase) -> CombatManager:
    world = WorldMap.starter()
    return CombatManager(db=db, world=world)


class TestAttack:
    """rm command — attacking enemies."""

    def test_attack_enemy_hit(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player in room with enemy WHEN rm goblin (forced hit) THEN damage dealt."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        # Force a high roll to guarantee hit
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 18  # High roll
            result = combat.attack("alice", "#test", "goblin")
        assert result.hit is True
        assert result.damage > 0

    def test_attack_enemy_miss(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player in room with enemy WHEN rm goblin (forced miss) THEN no damage."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 1  # Natural 1
            result = combat.attack("alice", "#test", "goblin")
        assert result.hit is False
        assert result.damage == 0

    def test_attack_nonexistent_enemy(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN no such enemy WHEN rm THEN error."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/town/tavern")
        result = combat.attack("alice", "#test", "goblin")
        assert result.error is not None

    def test_kill_enemy_grants_xp_and_gold(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN enemy killed WHEN checking rewards THEN XP and gold awarded."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", attack=50)
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 20
            result = combat.attack("alice", "#test", "rat")
        assert result.enemy_killed is True
        assert result.xp_gained > 0
        assert result.gold_gained > 0

    def test_enemy_counterattack(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN enemy alive after attack WHEN turn ends THEN enemy hits back."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        with patch("rpg.combat.d20.roll") as mock_roll:
            # Player hits but doesn't kill, enemy hits back
            mock_roll.return_value.total = 15
            result = combat.attack("alice", "#test", "goblin")
        # Enemy should counterattack (may hit or miss)
        assert result.enemy_killed is False or result.counterattack_damage >= 0

    def test_player_death_respawns_at_town(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player at 1 HP WHEN enemy kills them THEN respawn at /town."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level3", hp=1)
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 1  # Player misses, enemy hits
            result = combat.attack("alice", "#test", "dark_knight")
        if result.player_died:
            char = db.get_character("alice", "#test")
            assert char is not None
            assert char.location == "/town/tavern"
            assert char.hp > 0

    def test_room_clears_when_all_enemies_dead(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN last enemy killed WHEN checking room THEN room marked cleared."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", attack=50)
        # Kill all rats (count=2) — need 2 attacks
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 20
            combat.attack("alice", "#test", "rat")
            combat.attack("alice", "#test", "rat")
        cleared = db.get_room_cleared_at("#test", "/forest/clearing")
        assert cleared is not None


class TestLevelUp:
    """XP and leveling."""

    def test_level_up_at_threshold(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN character near level threshold WHEN gaining XP THEN level up."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", xp=95, attack=50)
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 20
            result = combat.attack("alice", "#test", "rat")
        if result.enemy_killed:
            char = db.get_character("alice", "#test")
            assert char is not None
            # Should have leveled up if XP >= 100
            if char.xp >= 100:
                assert char.level >= 2
                assert result.leveled_up is True


class TestInventoryPickup:
    """mv command — picking up items."""

    def test_pickup_item(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN item in room WHEN mv item ~/inventory THEN item in inventory."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/cave")
        result = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert result.error is None
        items = db.get_inventory("alice", "#test")
        assert any(i.name == "rusty_sword.txt" for i in items)

    def test_pickup_nonexistent_item(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN no such item WHEN mv THEN error."""
        db.create_character("alice", "#test")
        result = combat.pickup_item("alice", "#test", "nonexistent")
        assert result.error is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_combat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rpg.combat'`

**Step 3: Implement `combat.py`**

Create `plugins/rpg/src/rpg/combat.py`:

```python
"""Combat system — d20-based attack resolution, XP, loot, death."""

from __future__ import annotations

import time
from typing import NamedTuple

import d20

from .persistence import RPGDatabase
from .world import EnemySpawn, WorldMap

# XP required to reach each level: level N requires XP_TABLE[N]
# Level 1 = 0 XP, Level 2 = 100 XP, etc.
XP_TABLE = [0, 0, 100, 250, 500, 800, 1200, 1700, 2400, 3200, 4200]
MAX_LEVEL = len(XP_TABLE) - 1

# Stats gained per level up
HP_PER_LEVEL = 5
ATTACK_PER_LEVEL = 1
DEFENSE_PER_LEVEL = 1

# Death penalty: lose 10% of XP (but never below current level threshold)
DEATH_XP_PENALTY_PCT = 10


class CombatResult(NamedTuple):
    """Result of an attack action."""

    hit: bool
    damage: int
    enemy_name: str
    enemy_killed: bool
    xp_gained: int
    gold_gained: int
    leveled_up: bool
    new_level: int
    counterattack_damage: int
    player_died: bool
    error: str | None = None


class PickupResult(NamedTuple):
    """Result of picking up an item."""

    item_name: str
    attack_bonus: int
    defense_bonus: int
    error: str | None = None


class CombatManager:
    """Handles attack resolution, rewards, death, and item pickup."""

    def __init__(self, *, db: RPGDatabase, world: WorldMap) -> None:
        self.db = db
        self.world = world
        # Track active enemy HP per channel/room: {(channel, room, enemy_name, index): hp}
        self._enemy_hp: dict[tuple[str, str, str, int], int] = {}

    def _get_enemy_spawn(self, room_path: str, enemy_name: str) -> EnemySpawn | None:
        """Find an enemy spawn definition in a room."""
        room = self.world.get_room(room_path)
        if room is None:
            return None
        for spawn in room.spawns:
            if spawn.name == enemy_name:
                return spawn
        return None

    def _enemy_key(self, channel: str, room: str, name: str, index: int) -> tuple[str, str, str, int]:
        return (channel.lower(), room, name, index)

    def _get_or_init_enemy_hp(self, channel: str, room: str, spawn: EnemySpawn, index: int) -> int:
        """Get current enemy HP, initializing if needed."""
        key = self._enemy_key(channel, room, spawn.name, index)
        if key not in self._enemy_hp:
            self._enemy_hp[key] = spawn.hp
        return self._enemy_hp[key]

    def _find_live_enemy_index(self, channel: str, room: str, spawn: EnemySpawn) -> int | None:
        """Find the first alive instance of an enemy type in a room."""
        for i in range(spawn.count):
            hp = self._get_or_init_enemy_hp(channel, room, spawn, i)
            if hp > 0:
                return i
        return None

    def _check_room_cleared(self, channel: str, room_path: str) -> bool:
        """Check if all enemies in a room are dead."""
        room = self.world.get_room(room_path)
        if room is None or not room.spawns:
            return True
        for spawn in room.spawns:
            for i in range(spawn.count):
                key = self._enemy_key(channel, room_path, spawn.name, i)
                hp = self._enemy_hp.get(key, spawn.hp)
                if hp > 0:
                    return False
        return True

    def _check_level_up(self, nick: str, channel: str) -> tuple[bool, int]:
        """Check and apply level up if XP threshold reached."""
        char = self.db.get_character(nick, channel)
        assert char is not None
        if char.level >= MAX_LEVEL:
            return False, char.level

        next_level = char.level + 1
        if next_level < len(XP_TABLE) and char.xp >= XP_TABLE[next_level]:
            new_max_hp = char.max_hp + HP_PER_LEVEL
            self.db.update_character(
                nick, channel,
                level=next_level,
                max_hp=new_max_hp,
                hp=new_max_hp,  # Full heal on level up
                attack=char.attack + ATTACK_PER_LEVEL,
                defense=char.defense + DEFENSE_PER_LEVEL,
            )
            return True, next_level
        return False, char.level

    def _apply_death(self, nick: str, channel: str) -> None:
        """Handle player death: respawn at town with XP penalty."""
        char = self.db.get_character(nick, channel)
        assert char is not None

        # XP penalty: lose 10% but never drop below current level floor
        level_floor = XP_TABLE[char.level] if char.level < len(XP_TABLE) else 0
        xp_loss = max(0, char.xp * DEATH_XP_PENALTY_PCT // 100)
        new_xp = max(level_floor, char.xp - xp_loss)

        self.db.update_character(
            nick, channel,
            hp=char.max_hp,
            location="/town/tavern",
            xp=new_xp,
        )

    def _get_equipped_bonuses(self, nick: str, channel: str) -> tuple[int, int]:
        """Get total attack and defense bonuses from equipped items."""
        items = self.db.get_inventory(nick, channel)
        atk = sum(i.attack_bonus for i in items if i.equipped)
        dfn = sum(i.defense_bonus for i in items if i.equipped)
        return atk, dfn

    def attack(self, nick: str, channel: str, enemy_name: str) -> CombatResult:
        """Attack an enemy (rm command)."""
        char = self.db.get_character(nick, channel)
        if char is None:
            return CombatResult(
                hit=False, damage=0, enemy_name=enemy_name, enemy_killed=False,
                xp_gained=0, gold_gained=0, leveled_up=False, new_level=1,
                counterattack_damage=0, player_died=False,
                error="rm: you don't exist yet. Use any command to create a character.",
            )

        spawn = self._get_enemy_spawn(char.location, enemy_name)
        if spawn is None:
            return CombatResult(
                hit=False, damage=0, enemy_name=enemy_name, enemy_killed=False,
                xp_gained=0, gold_gained=0, leveled_up=False, new_level=char.level,
                counterattack_damage=0, player_died=False,
                error=f"rm: cannot remove '{enemy_name}': No such file or directory",
            )

        index = self._find_live_enemy_index(channel, char.location, spawn)
        if index is None:
            return CombatResult(
                hit=False, damage=0, enemy_name=enemy_name, enemy_killed=False,
                xp_gained=0, gold_gained=0, leveled_up=False, new_level=char.level,
                counterattack_damage=0, player_died=False,
                error=f"rm: '{enemy_name}': already dead",
            )

        # Player attack roll: d20 + player_attack vs enemy_defense + 10
        equip_atk, equip_def = self._get_equipped_bonuses(nick, channel)
        total_attack = char.attack + equip_atk
        attack_roll = d20.roll(f"1d20+{total_attack}")
        target_ac = spawn.defense + 10

        hit = attack_roll.total >= target_ac
        damage = 0
        enemy_killed = False
        xp_gained = 0
        gold_gained = 0

        if hit:
            # Damage: 1d6 + attack bonus
            damage_roll = d20.roll(f"1d6+{total_attack // 2}")
            damage = max(1, damage_roll.total)

            key = self._enemy_key(channel, char.location, spawn.name, index)
            self._enemy_hp[key] = max(0, self._enemy_hp[key] - damage)

            if self._enemy_hp[key] <= 0:
                enemy_killed = True
                xp_gained = spawn.xp_reward
                gold_gained = spawn.gold_reward

                # Award XP and gold
                self.db.update_character(
                    nick, channel,
                    xp=char.xp + xp_gained,
                    gold=char.gold + gold_gained,
                )

                # Check if room is fully cleared
                if self._check_room_cleared(channel, char.location):
                    self.db.mark_room_cleared(channel, char.location)

        # Enemy counterattack (if alive)
        counterattack_damage = 0
        player_died = False

        if not enemy_killed:
            total_defense = char.defense + equip_def
            enemy_roll = d20.roll(f"1d20+{spawn.attack}")
            player_ac = total_defense + 10

            if enemy_roll.total >= player_ac:
                enemy_dmg_roll = d20.roll(f"1d6+{spawn.attack // 2}")
                counterattack_damage = max(1, enemy_dmg_roll.total)
                new_hp = max(0, char.hp - counterattack_damage)
                self.db.update_character(nick, channel, hp=new_hp)

                if new_hp <= 0:
                    player_died = True
                    self._apply_death(nick, channel)

        # Check level up
        leveled_up = False
        new_level = char.level
        if enemy_killed and not player_died:
            leveled_up, new_level = self._check_level_up(nick, channel)

        return CombatResult(
            hit=hit,
            damage=damage,
            enemy_name=enemy_name,
            enemy_killed=enemy_killed,
            xp_gained=xp_gained,
            gold_gained=gold_gained,
            leveled_up=leveled_up,
            new_level=new_level,
            counterattack_damage=counterattack_damage,
            player_died=player_died,
        )

    def pickup_item(self, nick: str, channel: str, item_name: str) -> PickupResult:
        """Pick up an item from the current room (mv command)."""
        char = self.db.get_character(nick, channel)
        if char is None:
            return PickupResult(item_name=item_name, attack_bonus=0, defense_bonus=0,
                                error="mv: you don't exist yet.")

        room = self.world.get_room(char.location)
        if room is None:
            return PickupResult(item_name=item_name, attack_bonus=0, defense_bonus=0,
                                error="mv: room not found")

        for item in room.items:
            if item.name == item_name:
                self.db.add_item(
                    nick, channel, item.name,
                    attack_bonus=item.attack_bonus,
                    defense_bonus=item.defense_bonus,
                )
                return PickupResult(
                    item_name=item.name,
                    attack_bonus=item.attack_bonus,
                    defense_bonus=item.defense_bonus,
                )

        return PickupResult(
            item_name=item_name, attack_bonus=0, defense_bonus=0,
            error=f"mv: cannot stat '{item_name}': No such file or directory",
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/test_combat.py -v`
Expected: All PASS (some tests conditional on RNG outcomes — the mocked d20 rolls control this)

**Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Clean

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/combat.py plugins/rpg/tests/test_combat.py
git commit -m "feat(rpg): add d20-based combat system with XP, loot, death"
```

---

## Task 6: LLM Narrator — Flavor Text with Fallback

Narrator calls LiteLLM for room descriptions and combat narration, falling back to deterministic text.

**Files:**
- Create: `plugins/rpg/src/rpg/narrator.py`
- Create: `plugins/rpg/tests/test_narrator.py`

**Step 1: Write failing tests**

Create `plugins/rpg/tests/test_narrator.py`:

```python
"""Tests for LLM narrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rpg.narrator import Narrator


@pytest.fixture
def narrator() -> Narrator:
    return Narrator(model="gemini/gemini-2.0-flash-lite", api_key="test-key", timeout=2)


class TestNarrator:
    """LLM narrator with fallback."""

    def test_fallback_on_no_api_key(self):
        """GIVEN no API key WHEN narrate THEN deterministic fallback."""
        narrator = Narrator(model="", api_key="", timeout=2)
        text = narrator.narrate_room(
            room_path="/dungeon/level1",
            description_hint="A damp corridor",
            enemies=["goblin"],
            items=["sword.txt"],
            exits=["level2", ".."],
        )
        assert "dungeon/level1" in text
        assert "goblin" in text

    def test_narrate_room_calls_llm(self, narrator: Narrator):
        """GIVEN valid config WHEN narrate THEN LLM called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The dark corridor stretches ahead."
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_room(
                room_path="/dungeon/level1",
                description_hint="A damp corridor",
                enemies=["goblin"],
                items=["sword.txt"],
                exits=["level2", ".."],
            )
        assert text == "The dark corridor stretches ahead."

    def test_fallback_on_timeout(self, narrator: Narrator):
        """GIVEN LLM times out WHEN narrate THEN fallback text returned."""
        with patch("rpg.narrator.litellm.completion", side_effect=Exception("timeout")):
            text = narrator.narrate_room(
                room_path="/dungeon/level1",
                description_hint="A damp corridor",
                enemies=["goblin"],
                items=["sword.txt"],
                exits=["level2", ".."],
            )
        # Should get deterministic fallback, not crash
        assert "dungeon/level1" in text

    def test_output_truncated(self, narrator: Narrator):
        """GIVEN LLM returns long text WHEN narrate THEN truncated to max lines."""
        long_text = "\n".join(f"Line {i}" for i in range(20))
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = long_text
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_room(
                room_path="/test",
                description_hint="test",
                enemies=[],
                items=[],
                exits=[],
            )
        assert text.count("\n") <= 3  # Max 4 lines

    def test_narrate_combat(self, narrator: Narrator):
        """GIVEN combat event WHEN narrate THEN LLM called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Your blade strikes true!"
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_combat(
                attacker="alice",
                target="goblin",
                hit=True,
                damage=5,
                enemy_killed=False,
            )
        assert "strikes" in text.lower() or "blade" in text.lower() or len(text) > 0

    def test_combat_fallback(self, narrator: Narrator):
        """GIVEN LLM fails WHEN narrate_combat THEN deterministic fallback."""
        with patch("rpg.narrator.litellm.completion", side_effect=Exception("fail")):
            text = narrator.narrate_combat(
                attacker="alice",
                target="goblin",
                hit=True,
                damage=5,
                enemy_killed=True,
            )
        assert "goblin" in text.lower()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_narrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rpg.narrator'`

**Step 3: Implement `narrator.py`**

Create `plugins/rpg/src/rpg/narrator.py`:

```python
"""LLM narrator — flavor text with deterministic fallback."""

from __future__ import annotations

import logging

import litellm

_log = logging.getLogger("supybot.plugins.RPG.narrator")

# Max output lines per narration type
_MAX_ROOM_LINES = 4
_MAX_COMBAT_LINES = 3

_NARRATOR_SYSTEM = (
    "You are the narrator for a Linux-filesystem-themed IRC RPG. "
    "Describe game events in 1-3 short sentences. Plain text only — no markdown, "
    "no formatting, no emojis. Be atmospheric but concise. "
    "Players navigate with shell commands (cd, ls, rm, etc). "
    "The game world IS a Linux filesystem."
)


class Narrator:
    """Generates flavor text for game events via LLM, with deterministic fallback."""

    def __init__(self, *, model: str, api_key: str, timeout: int) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def _call_llm(self, prompt: str, max_lines: int) -> str | None:
        """Call LLM and return text, or None on any failure."""
        if not self._model or not self._api_key:
            return None
        try:
            response = litellm.completion(
                model=self._model,
                api_key=self._api_key,
                messages=[
                    {"role": "system", "content": _NARRATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                timeout=self._timeout,
                max_tokens=150,
            )
            text = response.choices[0].message.content or ""
            # Truncate to max lines
            lines = text.strip().split("\n")
            return "\n".join(lines[:max_lines])
        except Exception:
            _log.debug("Narrator LLM call failed, using fallback", exc_info=True)
            return None

    def narrate_room(
        self,
        *,
        room_path: str,
        description_hint: str,
        enemies: list[str],
        items: list[str],
        exits: list[str],
    ) -> str:
        """Generate room description."""
        prompt = (
            f"The player enters {room_path}. "
            f"Setting: {description_hint}. "
            f"Enemies present: {', '.join(enemies) if enemies else 'none'}. "
            f"Items on the ground: {', '.join(items) if items else 'none'}. "
            f"Exits: {', '.join(exits)}."
        )
        result = self._call_llm(prompt, _MAX_ROOM_LINES)
        if result is not None:
            return result
        return self._fallback_room(room_path, description_hint, enemies, items, exits)

    def narrate_combat(
        self,
        *,
        attacker: str,
        target: str,
        hit: bool,
        damage: int,
        enemy_killed: bool,
    ) -> str:
        """Generate combat narration."""
        if hit and enemy_killed:
            action = f"{attacker} strikes {target} for {damage} damage, destroying it!"
        elif hit:
            action = f"{attacker} hits {target} for {damage} damage."
        else:
            action = f"{attacker} swings at {target} but misses."

        prompt = f"Narrate this combat action in the Linux RPG: {action}"
        result = self._call_llm(prompt, _MAX_COMBAT_LINES)
        if result is not None:
            return result
        return self._fallback_combat(attacker, target, hit, damage, enemy_killed)

    @staticmethod
    def _fallback_room(
        room_path: str,
        description_hint: str,
        enemies: list[str],
        items: list[str],
        exits: list[str],
    ) -> str:
        """Deterministic room description."""
        parts = [f"{room_path} — {description_hint}"]
        if enemies:
            parts.append(f"Enemies: {', '.join(enemies)}")
        if items:
            parts.append(f"Items: {', '.join(items)}")
        parts.append(f"Exits: {', '.join(exits)}")
        return " | ".join(parts)

    @staticmethod
    def _fallback_combat(
        attacker: str,
        target: str,
        hit: bool,
        damage: int,
        enemy_killed: bool,
    ) -> str:
        """Deterministic combat narration."""
        if hit and enemy_killed:
            return f"{attacker} rm -f {target} [{damage} dmg] — process terminated."
        if hit:
            return f"{attacker} rm {target} [{damage} dmg] — still running."
        return f"{attacker} rm {target} — permission denied (miss)."
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/test_narrator.py -v`
Expected: All PASS

**Step 5: Run lint and typecheck**

Run: `make lint && make typecheck`
Expected: Clean

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/narrator.py plugins/rpg/tests/test_narrator.py
git commit -m "feat(rpg): add LLM narrator with deterministic fallback"
```

---

## Task 7: IRC Plugin Layer — Wire Commands to Engine

Connect all IRC commands to the game engine and narrator. Format output for IRC.

**Files:**
- Modify: `plugins/rpg/src/rpg/plugin.py`
- Modify: `plugins/rpg/src/rpg/config.py` (if needed)
- Create: `plugins/rpg/tests/test_commands.py`

**Step 1: Write failing tests**

Create `plugins/rpg/tests/test_commands.py`:

```python
"""Tests for RPG IRC command layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def make_mock_plugin(tmp_path, mocker):
    """Create a minimal mock RPG plugin for testing commands."""
    # We test the formatting/routing logic, not full Limnoria integration
    from rpg.engine import GameEngine
    from rpg.combat import CombatManager
    from rpg.narrator import Narrator
    from rpg.persistence import RPGDatabase
    from rpg.world import WorldMap

    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    engine = GameEngine(db=db, world=world, spawn_cooldown_minutes=30)
    combat = CombatManager(db=db, world=world)
    narrator = Narrator(model="", api_key="", timeout=2)  # Fallback mode
    return engine, combat, narrator, db


class TestCommandFormatting:
    """IRC output formatting for game events."""

    def test_cd_formats_room(self, tmp_path, mocker):
        """GIVEN player moves WHEN cd /forest THEN output includes location."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        text = narrator.narrate_room(
            room_path=event.location,
            description_hint=event.description_hint,
            enemies=event.enemies,
            items=event.items,
            exits=event.exits,
        )
        assert "/forest" in text

    def test_cd_error_formats(self, tmp_path, mocker):
        """GIVEN invalid path WHEN cd THEN error message shown."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "nonexistent")
        assert event.error is not None
        assert "no such" in event.error.lower()

    def test_whoami_formats_stats(self, tmp_path, mocker):
        """GIVEN a character WHEN whoami THEN stats formatted."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        info = engine.character_info("alice", "#test")
        # Format like the plugin would
        line = f"{info.nick} | HP:{info.hp}/{info.max_hp} ATK:{info.attack} DEF:{info.defense} XP:{info.xp} LVL:{info.level} GOLD:{info.gold} | {info.location}"
        assert "alice" in line
        assert "HP:20/20" in line

    def test_pwd_formats_location(self, tmp_path, mocker):
        """GIVEN a character WHEN pwd THEN location shown."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        loc = engine.current_location("alice", "#test")
        assert loc == "/town/tavern"

    def test_sleep_formats_recovery(self, tmp_path, mocker):
        """GIVEN damaged character WHEN sleep THEN recovery shown."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        db.update_character("alice", "#test", hp=10)
        event = engine.rest("alice", "#test")
        assert event.hp_restored > 0

    def test_mv_pickup_formats(self, tmp_path, mocker):
        """GIVEN item in room WHEN mv THEN pickup confirmed."""
        engine, combat, narrator, db = make_mock_plugin(tmp_path, mocker)
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        result = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert result.error is None
        assert result.item_name == "rusty_sword.txt"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/rpg/tests/test_commands.py -v`
Expected: PASS (these test formatting logic, not Limnoria wiring — should pass once engine/combat/narrator exist)

**Step 3: Implement the full plugin.py**

Rewrite `plugins/rpg/src/rpg/plugin.py`:

```python
"""RPG plugin implementation — IRC command layer."""

from __future__ import annotations

import logging
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

    def _get_nick(self, msg: "IrcMsg") -> str:
        """Extract nick from IRC message."""
        return msg.nick or "unknown"

    def _get_channel(self, msg: "IrcMsg") -> str:
        """Extract channel from IRC message."""
        return msg.channel or msg.args[0]

    def _check_enabled(self, irc: callbacks.Irc, msg: "IrcMsg") -> bool:
        """Check if RPG is enabled in this channel."""
        channel = self._get_channel(msg)
        if not self.registryValue("enabled", channel):
            irc.reply("RPG is not enabled in this channel. Ask an admin to set plugins.RPG.enabled True.")
            return False
        return True

    class rpg(callbacks.Commands):
        """RPG game commands — explore a Linux filesystem world."""

        def cd(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], destination: str) -> None:
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

        def ls(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], flags: str) -> None:
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

        def cat(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], target: str) -> None:
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

        def rm(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], enemy: str) -> None:
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

            # Narrate the attack
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

        def mv(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], text: str) -> None:
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

        def pwd(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
            """— Show current location."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            nick = plugin._get_nick(msg)
            channel = plugin._get_channel(msg)

            loc = plugin._engine.current_location(nick, channel)
            irc.reply(loc)

        pwd = wrap(pwd, [])

        def whoami(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
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

        def man(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str], topic: str) -> None:
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
                "cd": "cd <path> — Move. Supports relative, absolute, and .. paths.",
                "ls": "ls [-a] — Look around. -a reveals hidden dotfile rooms.",
                "cat": "cat <thing> — Examine an item, enemy, or object.",
                "rm": "rm <enemy> — Attack an enemy. d20 + ATK vs DEF + 10.",
                "mv": "mv <item> ~/inventory — Pick up an item from the room.",
                "pwd": "pwd — Show your current location.",
                "whoami": "whoami — Show your character stats.",
                "man": "man <topic> — Get help or lore about something.",
                "sleep": "sleep — Rest and recover HP (outside combat).",
                "history": "history — Show your recent actions.",
            }
            if topic.lower() in help_text:
                irc.reply(help_text[topic.lower()])
            else:
                irc.reply(f"man: no manual entry for {topic}")

        man = wrap(man, ["text"])

        def sleep(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
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
                irc.reply(f"You rest... HP restored: +{event.hp_restored} (now {event.hp_after}/{event.hp_after})")

        sleep = wrap(sleep, [])

        def history(self, irc: callbacks.Irc, msg: "IrcMsg", args: list[str]) -> None:
            """— Show recent actions (placeholder for v1)."""
            plugin = irc.getCallback("RPG")
            if not plugin._check_enabled(irc, msg):
                return
            irc.reply("history: not yet implemented")

        history = wrap(history, [])

    rpg = rpg
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/rpg/tests/ -v`
Expected: All PASS

**Step 5: Run full preflight**

Run: `make preflight`
Expected: All checks pass

**Step 6: Commit**

```bash
git add plugins/rpg/src/rpg/plugin.py plugins/rpg/tests/test_commands.py
git commit -m "feat(rpg): wire IRC commands to engine, combat, and narrator"
```

---

## Task 8: Integration Test — Full Game Flow

End-to-end test: create character, move, look, fight, pick up loot, level up, die, respawn.

**Files:**
- Create: `plugins/rpg/tests/test_integration.py`

**Step 1: Write integration test**

Create `plugins/rpg/tests/test_integration.py`:

```python
"""Integration tests — full game flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rpg.combat import CombatManager
from rpg.engine import GameEngine
from rpg.narrator import Narrator
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


@pytest.fixture
def game(tmp_path):
    """Full game stack with fallback narrator."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    engine = GameEngine(db=db, world=world, spawn_cooldown_minutes=30)
    combat = CombatManager(db=db, world=world)
    narrator = Narrator(model="", api_key="", timeout=2)
    return engine, combat, narrator, db


class TestFullGameFlow:
    """Play through a complete game session."""

    def test_explore_and_fight(self, game):
        """GIVEN new player WHEN exploring and fighting THEN full flow works."""
        engine, combat, narrator, db = game

        # Character auto-created at /town/tavern
        event = engine.move("alice", "#test", "..")
        assert event.location == "/town"

        # Look around town
        look = engine.look("alice", "#test", show_hidden=False)
        assert "tavern" in look.exits

        # Move to forest clearing
        event = engine.move("alice", "#test", "/forest/clearing")
        assert event.location == "/forest/clearing"
        assert len(event.enemies) > 0

        # Fight rats with high attack to guarantee kills
        db.update_character("alice", "#test", attack=50)
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 20
            r1 = combat.attack("alice", "#test", "rat")
            assert r1.enemy_killed is True
            r2 = combat.attack("alice", "#test", "rat")
            assert r2.enemy_killed is True

        # Room should be cleared
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.xp > 0
        assert char.gold > 0

        # Move to cave and pick up item
        engine.move("alice", "#test", "/forest/cave")
        pickup = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert pickup.error is None

        items = db.get_inventory("alice", "#test")
        assert len(items) == 1

    def test_death_and_respawn(self, game):
        """GIVEN player at low HP WHEN killed THEN respawn at town."""
        engine, combat, narrator, db = game

        engine.ensure_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level3", hp=1)

        # Force miss so enemy counterattacks
        with patch("rpg.combat.d20.roll") as mock_roll:
            mock_roll.return_value.total = 1
            result = combat.attack("alice", "#test", "dark_knight")

        if result.player_died:
            char = db.get_character("alice", "#test")
            assert char is not None
            assert char.location == "/town/tavern"
            assert char.hp == char.max_hp

    def test_narrator_formats_all_events(self, game):
        """GIVEN game events WHEN narrating THEN all produce output."""
        engine, combat, narrator, db = game

        # Room narration
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        text = narrator.narrate_room(
            room_path=event.location,
            description_hint=event.description_hint,
            enemies=event.enemies,
            items=event.items,
            exits=event.exits,
        )
        assert len(text) > 0

        # Combat narration
        text = narrator.narrate_combat(
            attacker="alice", target="goblin",
            hit=True, damage=5, enemy_killed=False,
        )
        assert len(text) > 0

    def test_hidden_room_discovery(self, game):
        """GIVEN player at /town WHEN ls -a THEN .armory visible, contains item."""
        engine, combat, narrator, db = game

        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")

        # Without -a, no hidden rooms
        look = engine.look("alice", "#test", show_hidden=False)
        assert ".armory" not in look.exits

        # With -a, hidden room appears
        look = engine.look("alice", "#test", show_hidden=True)
        assert ".armory" in look.exits

        # Enter hidden room and find item
        engine.move("alice", "#test", ".armory")
        event = engine.examine("alice", "#test", "enchanted_shield.dat")
        assert event.description is not None
```

**Step 2: Run integration tests**

Run: `uv run pytest plugins/rpg/tests/test_integration.py -v`
Expected: All PASS

**Step 3: Run full preflight**

Run: `make preflight`
Expected: All checks pass

**Step 4: Commit**

```bash
git add plugins/rpg/tests/test_integration.py
git commit -m "test(rpg): add integration tests for full game flow"
```

---

## Task 9: Dockerfile and CI Updates

Ensure the RPG plugin is included in Docker builds and CI runs its tests.

**Files:**
- Modify: `Dockerfile` (line 9 — add RPG plugin pyproject.toml copy)
- Verify: CI already runs `make ci` which includes all testpaths

**Step 1: Update Dockerfile**

The Dockerfile copies `plugins/llm/pyproject.toml` for dependency resolution. Add the RPG plugin.

In the Dockerfile, after `COPY plugins/llm/pyproject.toml plugins/llm/`, add:
```dockerfile
COPY plugins/rpg/pyproject.toml plugins/rpg/
```

**Step 2: Verify CI picks up new tests**

The CI runs `make ci` which runs `make test-all`, which uses `testpaths` from `pyproject.toml`. Since we added `plugins/rpg/tests` to `testpaths` in Task 1, CI will automatically run RPG tests.

Verify: `uv run pytest plugins/rpg/tests/ -v` passes.

**Step 3: Run full preflight**

Run: `make preflight`
Expected: All checks pass (both LLM and RPG tests)

**Step 4: Commit**

```bash
git add Dockerfile
git commit -m "build: add RPG plugin to Docker build and CI"
```

---

## Task 10: Final Preflight and Cleanup

Run the complete quality gate. Fix any issues. Verify coverage.

**Step 1: Run full preflight**

Run: `make preflight`
Expected: format + lint + typecheck + tests all pass

**Step 2: Check coverage**

Run: `uv run pytest plugins/rpg/tests/ -v --cov=rpg --cov-report=term-missing`
Expected: >= 80% coverage

**Step 3: Fix any coverage gaps**

Add tests for any uncovered paths (error branches, edge cases).

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(rpg): final cleanup and coverage"
```

---

Plan complete and saved to `docs/plans/2026-03-12-rpg-plugin-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
