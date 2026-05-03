# RPG Plugin Design — Linux Filesystem Dungeon Master

**Status:** Approved — v1 scope locked

## Concept

An IRC RPG where the game world is a Linux filesystem. Players explore dungeons, fight monsters, and collect loot using shell commands. An LLM narrates events but Python enforces all game mechanics. Games persist across days and weeks. One shared campaign per channel.

## v1 Scope

Minimal viable game: movement, exploration, basic combat, one dungeon arc, persistence. No classes, quests, crafting, trading, or procedural generation.

## Commands

All commands are prefixed with `%rpg` to avoid collisions (e.g., `%rpg cd dungeon/level1`).

| Command | Action |
|---------|--------|
| `cd <path>` | Move to a location |
| `ls` / `ls -a` | Look around (hidden things are dotfiles) |
| `cat <thing>` | Examine item, NPC, or object |
| `mv <item> ~/inventory` | Pick up item |
| `rm <enemy>` | Attack (starts/continues combat) |
| `pwd` | Where am I? |
| `whoami` | Character sheet (HP, XP, level, inventory) |
| `man <thing>` | Lore/help about anything |
| `history` | Recent actions recap |
| `sleep` | Rest and recover HP (out of combat only) |

## Character Model

Minimal stats: HP, max_hp, attack, defense, XP, level, gold. No classes, no stat arrays. Leveling increases HP, attack, and defense by fixed amounts. Auto-created on first command.

## Combat System

- `rm <enemy>` initiates combat. All players in the room are snapshotted into the encounter.
- **20-second round timer.** Players act with `rm <enemy>` (attack) or other valid commands.
- AFK handling: auto-defend (round 1 missed), auto-attack (round 2), auto-withdraw to last safe room (round 3).
- Resolution: d20 + attack vs defense. Damage is a dice roll based on level and weapon.
- Trash mobs auto-resolve in one summary message.
- Boss fights are multi-round with LLM narration.
- **Death:** Respawn at `/town` with XP penalty. Not permadeath.
- **Anti-grief:** Per-user action cooldown, per-room combat lock (one encounter at a time), NickServ-identified users only.

## IRC Output Caps

- Standard action: max 2 lines
- Combat round summary: max 3 lines
- Boss/milestone: max 4 lines
- Lists truncate with "+N more"

## World Map (v1, ~12 rooms)

```
/
├── town/
│   ├── tavern/          # Rest, info, starting point
│   ├── blacksmith/      # Buy gear with gold
│   └── .armory/         # Hidden — reward for ls -a
├── forest/
│   ├── clearing/        # Easy mobs (rats, wolves)
│   ├── cave/            # Medium mobs (bats, spiders)
│   └── .fairy_grove/    # Hidden — healing spring
└── dungeon/
    ├── level1/          # Goblins, loot
    ├── level2/          # Skeletons, harder
    ├── level3/          # Traps, tough mobs
    └── boss_chamber/    # Boss fight, campaign climax
```

- Rooms have: description hint, enemy spawns, item drops, exits, hidden flag.
- Enemy spawns reset on a timer (~30 min after cleared). Boss does not respawn.
- Hidden rooms require `ls -a` to discover.

## Persistence (SQLite)

| Table | Stores |
|-------|--------|
| `characters` | Nick, HP, max_hp, attack, defense, XP, level, location, gold, channel |
| `inventory` | Character FK, item name, item stats, equipped flag |
| `world_state` | Channel, room path, cleared timestamp, loot taken flags |
| `combat_state` | Channel, room, participants, turn order, enemy HP, round number |

Characters are per-channel. Combat state survives disconnects — encounter pauses and resumes when a player returns to the room.

## LLM Narrator

The LLM adds flavor text to engine events. It never decides game outcomes.

**Contract:** Engine sends structured JSON state, narrator returns 1-4 lines of descriptive text.

```python
# Engine sends:
{"event": "enter_room", "room": "dungeon/level1",
 "enemies": ["goblin", "goblin"], "items": ["rusty_sword.txt"]}

# Narrator returns:
"The stairway opens into a damp corridor. Two goblins snarl
from behind a collapsed pillar. Something metallic glints on the floor."
```

- **2-second timeout.** If narrator is slow or unavailable, deterministic fallback text is used.
- Narrator token cap set low to enforce brevity.
- RPG plugin calls LiteLLM directly — no dependency on the LLM plugin.

## Architecture

```
plugins/rpg/
├── src/rpg/
│   ├── __init__.py        # Plugin exports
│   ├── plugin.py          # IRC command parsing, output formatting
│   ├── engine.py          # Game state machine, movement, loot, XP
│   ├── combat.py          # Turn-based combat, d20 resolution
│   ├── world.py           # Room graph, spawn tables, map data
│   ├── narrator.py        # LiteLLM wrapper, fallback text, output caps
│   ├── persistence.py     # SQLite read/write
│   └── config.py          # Limnoria registry config
├── tests/
└── pyproject.toml
```

### Separation of Concerns

- **engine.py / combat.py / world.py:** All mechanical truth — HP, damage, loot, movement validation, XP, leveling.
- **narrator.py:** LLM integration. Generates flavor text given structured game state. Falls back to deterministic text.
- **plugin.py:** IRC layer. Parses commands, calls engine, formats output for IRC.
- **persistence.py:** SQLite read/write. Saves and loads game state.
- **config.py:** Limnoria registry. Model, API key, narrator timeout, game balance knobs.

## Dependencies

- **d20** — dice rolling (PyPI, stable)
- **litellm** — narrator LLM calls (already in workspace)
- **sqlite3** — stdlib

## Future (not v1)

- Classes (Warrior, Mage, Rogue, Cleric) with Linux-flavored subclass names and lore
- `sudo` as earned charges from milestones (not class-locked)
- `kill -SIGNAL` abilities (max 4: TERM, HUP, INT, USR1)
- Quest board system
- Crafting (`touch`), trading (`chown`)
- `ln -s` fast travel, `mkdir` camps, `du` inventory weight
- Seeded world expansion via `mount`
- Procedural dungeon generation (engine-driven, not LLM)
