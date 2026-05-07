# RPG plugin

A lightweight, filesystem-themed roleplay game for IRC. Players explore
"directories" as rooms, fight processes, roll dice, and (optionally)
get LLM-narrated outcomes.

## Loading

```
@load RPG
```

Capabilities and registry options live under `supybot.plugins.RPG`.

## Internal layout

```
plugins/rpg/src/rpg/
├── plugin.py        # IRC command surface
├── engine.py        # Core game loop and state mutations
├── combat.py        # Turn-based combat resolution
├── dice.py          # Dice roller (deterministic when seeded for tests)
├── world.py         # Starter world map (rooms / exits / spawns)
├── narrator.py      # Optional LLM-driven prose narration
├── persistence.py   # SQLite store for player state
└── config.py        # Limnoria registry options
```

## Configuration

The narrator is opt-in. Configure a model and API key independently of
the LLM plugin so the two systems can use different providers:

```
@config plugins.RPG.narratorModel openai/gpt-4o-mini
@config plugins.RPG.narratorApiKey sk-...
@config plugins.RPG.narratorTimeout 20
```

The database path defaults to `<supybot.directories.data>/RPG.db`, set
explicitly via `plugins.RPG.databasePath` if you want it elsewhere.

Spawn cooldown (in minutes between hostile spawns per channel) is
controlled by `plugins.RPG.spawnCooldownMinutes`.
