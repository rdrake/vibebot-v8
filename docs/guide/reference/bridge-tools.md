# Limnoria Bridge Tools

The LLM has tools that wrap Limnoria's stock plugin commands. When a user asks the assistant in natural language ("what time is it in Tokyo?", "have you seen alice?"), the model can call the corresponding Limnoria command directly instead of having a custom implementation.

This page describes the bridge's gating rules and how operators control what's exposed.

## How it works

`enumerate_commands` in `plugins/llm/src/llm/limnoria_bridge.py` walks `irc.callbacks` on every assistant turn, so the tool surface tracks whatever Limnoria has loaded at that moment. Newly loaded plugins are visible without a bot restart.

A command is exposed when **all** of these hold:

1. `bridgeEnabled` is `True` (default).
2. The owning plugin is in the operator allowlist (`bridgeAllowedPlugins`, falling back to the curated `DEFAULT_ALLOWED_PLUGINS` set when empty).
3. The command is **not** in `DENY_COMMANDS` (e.g. `web fetch`, `utilities apply`/`let`, `misc more`/`clearmores`).
4. Either the command is read-only, **or** `bridgeAllowMutating` is `True` for the channel.
5. The caller has the Limnoria capability for the command (same check Limnoria itself runs for IRC dispatch).

## Default allowlist

Curated set used when `bridgeAllowedPlugins` is empty. Every plugin in the set is either read-only by nature or has its writes gated behind `bridgeAllowMutating`.

- `Misc`
- `Time`
- `Math`
- `Utilities`
- `Seen`
- `Web`
- `Later`
- `Note`
- `Karma`
- `QuoteGrabs`
- `RSS`
- `DDG`

For the canonical, up-to-date list see `DEFAULT_ALLOWED_PLUGINS` in `plugins/llm/src/llm/limnoria_bridge.py`.

## Loading the plugins

`bridgeAllowedPlugins` only matters for plugins Limnoria has actually loaded. Out of the curated set, only `Misc` and `Utilities` ship under `alwaysLoadImportant`. To get the rest, run these as bot owner:

```
@load Time
@load Math
@load Seen
@load Web
@load Later
@load Note
@load Karma
@load QuoteGrabs
@load RSS
@load DDG
```

Limnoria persists each `load` to `bot.conf` automatically. The bridge picks them up on the next assistant turn, with no bot restart needed.

To see the read-only commands a plugin actually exports, ask the bot:

```
@list Time
@help time
```

Whatever shows up under `@list <Plugin>` is what the LLM can call, minus anything held back by `DENY_COMMANDS` or `MUTATING_COMMANDS`.

## Mutating commands

When `bridgeAllowMutating` is `False` (the default), the bridge filters out commands that change persistent state, send messages on behalf of a different user, or have read-with-side-effects. Filtering happens at enumeration *and* at dispatch as in-depth defense. The full set lives at the top of `limnoria_bridge.py` under `MUTATING_COMMANDS`.

To allow mutations on a specific channel:

```
@config channel #yourchan plugins.LLM.bridgeAllowMutating True
```

Operator caveats:

- `RSS` mutations can announce feed entries to every channel subscribed to the feed. Enabling mutations in one channel can cause writes elsewhere.
- `Karma` `clear`/`dump`/`load` are destructive against the karma DB.
- `Factoids` `whatis` is classified mutating because the default `keepRankInfo` writes a usage counter on every read.

## Operator config

| Setting | Default | Description |
|---------|---------|-------------|
| `bridgeEnabled` | `True` | Primary switch. `False` removes every bridge tool. |
| `bridgeAllowedPlugins` | (empty, falls back to curated default) | Space-separated plugin names. A non-empty value replaces the curated set entirely. Names use camel case to match `cb.name()`. |
| `bridgeAllowMutating` | `False` | Per-channel. Allows commands listed in `MUTATING_COMMANDS`. |
| `bridgeDebugInChannel` | `False` | Per-channel. When `True`, dispatch failures echo a debug line into the channel. |
| `bridgeScheduledTaskLimit` | (see `config.py`) | Per-user cap on outstanding `schedule_llm_task` entries. |

To add a plugin not in the curated default, list every plugin you want exposed (the registry value replaces the default rather than adding to it). For example, to extend the default set with `Factoids` and `Dict`:

```
@config plugins.LLM.bridgeAllowedPlugins Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG Factoids Dict
```

Once a plugin is in `bridgeAllowedPlugins` and loaded into Limnoria, every read-only command it exports becomes available to the LLM on the next turn.
