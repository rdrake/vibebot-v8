# Bridge tools

The assistant carries tools that wrap Limnoria's stock plugin commands.
When a user asks in natural language ("what time is it in Tokyo?",
"have you seen alice?"), the model calls the corresponding Limnoria
command directly instead of relying on a custom implementation.

This page describes the bridge's gating rules and the knobs operators
use to control what the model sees.

## How it works

`enumerate_commands` in `plugins/llm/src/llm/limnoria_bridge.py` walks
`irc.callbacks` on every assistant turn, so the tool surface tracks
whatever Limnoria has loaded at that moment. Newly loaded plugins
become visible without a bot restart.

The bridge exposes a command when **all** of these hold:

1. `bridgeEnabled` is `True` for the channel (default `False`).
2. The owning plugin is in the operator allowlist
   (`bridgeAllowedPlugins`), or in the curated
   `DEFAULT_ALLOWED_PLUGINS` set when the allowlist is empty.
3. The command is **not** in `DENY_COMMANDS`. Examples: `misc more`
   (interactive pagination) and the `web` fetch-style commands, which
   the bridge denies as server-side request forgery risks.
4. Either the command is read-only, or `bridgeAllowMutating` is `True`
   for the channel.
5. The caller holds the Limnoria capability for the command, the same
   check Limnoria runs for IRC dispatch.

## Default allowlist

The curated set applies when `bridgeAllowedPlugins` is empty. Every
plugin in the set is either read-only by nature or has its writes gated
behind `bridgeAllowMutating`.

`Misc`, `Time`, `Math`, `Utilities`, `Seen`, `Web`, `Later`, `Note`,
`Karma`, `QuoteGrabs`, `RSS`, `DDG`

For the canonical, current list see `DEFAULT_ALLOWED_PLUGINS` in
`plugins/llm/src/llm/limnoria_bridge.py`.

## Loading the plugins

`bridgeAllowedPlugins` only matters for plugins Limnoria has loaded.
Out of the curated set, only `Misc` and `Utilities` ship under
`alwaysLoadImportant`. To get the rest, run these as bot owner:

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

Limnoria persists each `load` to `bot.conf` automatically. The bridge
picks them up on the next assistant turn, with no bot restart needed.

To see the commands a plugin exports, ask the bot:

```
@list Time
@help time
```

Whatever shows up under `@list <Plugin>` is what the model can call,
minus anything held back by `DENY_COMMANDS` or `MUTATING_COMMANDS`.

## Mutating commands

When `bridgeAllowMutating` is `False`, the default, the bridge filters
out commands that change persistent state, send messages on behalf of
another user, or read with side effects. Filtering happens at
enumeration *and* at dispatch, as a defence-in-depth measure. The full
set lives
at the top of `limnoria_bridge.py` under `MUTATING_COMMANDS`.

To allow mutations on a specific channel:

```
@config channel #yourchan plugins.LLM.bridgeAllowMutating True
```

Operator caveats:

- `RSS` mutations can announce feed entries to every channel subscribed
  to the feed. Enabling mutations in one channel can cause writes
  elsewhere.
- `Karma` `clear`, `dump`, and `load` are destructive against the karma
  database.
- `Factoids` `whatis` counts as mutating because the default
  `keepRankInfo` writes a usage counter on every read.

## Native tools

Beyond the bridge, the assistant carries its own tools. Users never
call these directly; the model picks them from conversation.

| Group | Tools |
|-------|-------|
| Instructions | `get_instruction`, `set_instruction`, `clear_instruction` |
| Memories | `list_memories`, `save_memory`, `delete_memory`, `update_memory`, `clear_memories`, `cleanup_memories` |
| Context | `forget_context` |
| Usage | `get_usage`, `get_channel_usage` |
| Reminders and tasks | `set_reminder`, `schedule_llm_task`, `list_pending_tasks`, `cancel_pending_task`, `cancel_all_pending_tasks` |
| Generation | `generate_image`, `generate_code` |
| Web | `search_web`, `fetch_url` |

On verse-routed turns the model also receives the verse tool set:
`verse_act`, `verse_move`, `verse_look`, `verse_recall`,
`verse_record`, `verse_edit` (only usable when the speaker holds
`llm.verse.edit`), and `verse_storybook` (only present when
`verseStorybookEnabled` is on). See
[verse operations](../operator/forest-verse.md).

## Scheduled LLM tasks

`schedule_llm_task` lets the assistant create recurring agentic work
("every weekday at 09:00, summarize the news"). The
`bridgeScheduledTaskLimit` key (default 5, 0 disables scheduling) caps
active schedules per creator in a channel. Each fire still counts
against the user's normal `@ask` rate-limit bucket; the cap limits the
number of pending schedules, not their cumulative cost. Owners can
list and cancel any user's tasks with `@remind admin`.

## Operator config

| Setting | Default | Description |
|---------|---------|-------------|
| `bridgeEnabled` | `False` | Primary switch, per channel. `False` removes every bridge tool. |
| `bridgeAllowedPlugins` | empty (curated default applies) | Space-separated plugin names. A non-empty value replaces the curated set entirely. Names use camel case to match `cb.name()`. |
| `bridgeAllowMutating` | `False` | Per channel. Exposes commands listed in `MUTATING_COMMANDS`. |
| `bridgeScheduledTaskLimit` | `5` | Per-creator cap on active `schedule_llm_task` entries. `0` disables scheduling. |
| `bridgeDebugInChannel` | `False` | Per channel. Appends a one-line debug footer listing every bridge call made during the turn. |

To add a plugin outside the curated default, list every plugin you want
exposed, because the registry value replaces the default rather than
extending it. For example, to add `Factoids` and `Dict`:

```
@config plugins.LLM.bridgeAllowedPlugins Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG Factoids Dict
```

Once a plugin is in `bridgeAllowedPlugins` and loaded into Limnoria,
every read-only command it exports becomes available to the model on
the next turn.
