# Bridge tools

The assistant carries tools that wrap Limnoria's stock plugin commands.
When a user asks in natural language ("what time is it in Tokyo?",
"have you seen alice?"), the model calls the corresponding Limnoria
command directly instead of relying on a custom implementation.

This page describes the bridge's gating rules and the knobs operators
use to control what the model sees.

## How it works

The bridge injects two tools: `run_limnoria_command` and
`search_bridge_commands`. `enumerate_commands` in
`plugins/llm/src/llm/limnoria_bridge.py` walks `irc.callbacks` each time
the pair is built, so the tool surface tracks whatever Limnoria has
loaded at that moment. Newly loaded plugins become visible without a bot
restart.

`run_limnoria_command`'s description carries the whole exposed command
list, one line per command with its argument syntax and docstring. When
the mutation gate is closed and an allowlisted plugin has writes to hide,
the list ends with `(write commands hidden — set bridgeAllowMutating True
to expose)`. `search_bridge_commands` substring-searches that same list
(`limit` 1 to 25, default 10); it exists because Limnoria's
`Misc.apropos` matches command names only, not descriptions.

Both tools ride `@ask`, nick-addressed mentions, PMs, and `@rp`/verse
turns — everything that reaches `_ask_impl`, which builds the pair
unconditionally. `@code`, `@draw`, reminder fires, and scheduled tasks
never carry them.

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

`bridgeAllowedPlugins` only matters for plugins Limnoria has loaded. Out
of the curated set, only `Misc` loads on its own: `alwaysLoadImportant`
covers Admin, Channel, Config, Misc, Owner and User, and the bridge
hard-denies every one of those but `Misc`. Everything else, `Utilities`
included, needs an explicit load as bot owner:

```
@load Utilities
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

## Refused calls

A refusal is not silent. Dispatch hands the model a JSON envelope, so it
can correct itself or tell the user why. A call that runs returns
`{"status": "ok", "reply": "<captured text>"}`; a refusal returns one of:

- `{"error": "denied: write commands disabled"}` — the command is in
  `MUTATING_COMMANDS` and `bridgeAllowMutating` is `False`.
- `{"error": "not permitted: Plugin.command"}` — the caller lacks the
  Limnoria capability for that command.
- `{"error": "denied: Plugin.command"}` — a `DENY_COMMANDS` entry, or a
  plugin in `DENY_PLUGINS`.
- `{"error": "unknown plugin: X"}` or
  `{"error": "unknown command: Plugin.command"}` — the model invented a
  name.
- `{"error": "command failed"}` — the plugin raised. The exception text
  goes to the log, not to the model; anything the plugin printed before
  raising comes back under `partial_output`.
- `{"error": "<tokeniser message>"}` — the argument string failed
  Limnoria's tokeniser: an unbalanced quote, bracket, or pipe. The
  message passes through verbatim so the model can correct its own call.

## Native tools

Beyond the bridge, the assistant carries its own tools. Users never
call these directly; the model picks them from conversation. Which ones
it is offered depends on the route that handled the message:

| Route | Entry points | Tools offered |
|-------|--------------|---------------|
| chat | `@ask`, nick-addressed mentions, PMs | `check_service_status`, `fetch_url`, `generate_code`, `generate_image`, `save_memory`, `search_web`; plus `set_reminder` and `schedule_llm_task` where `pendingTasksEnabled` is on. Six with the shipped defaults; eight when `pendingTasksEnabled` is on; one fewer in each case when both `statusPageUrls` and `statusQueryablePages` are empty. |
| verse | `@rp`, a live sticky `@rp` session, the one-shot ambient-prose promotion | `fetch_url`, `generate_code`, `generate_image`, `save_memory`, `search_web` (5) |
| code | `@code` | `fetch_url`, `generate_code`, `search_web` (3) |
| draw | `@draw` | `generate_image` (1) |
| remind_action | reminder and scheduled-task fires | all 21 |

`assistant.py` defines 21 tool schemas, but defining one is not
advertising it. Thirteen are hidden from chat and verse because each
duplicates a command the user can already type:

- `list_memories`, `update_memory`, `delete_memory`, `clear_memories`,
  `cleanup_memories` → `@memories`, `@memories edit <id> <text>`,
  `@memories delete <id>`, `@memories clear`, `@memories cleanup`
- `set_instruction`, `clear_instruction` → `@instruct <instruction>`,
  `@instruct clear`
- `get_usage`, `get_channel_usage` → `@usage [<nick or #channel>]`
- `forget_context` → `@forget [<channel>]`
- `list_pending_tasks`, `cancel_pending_task`,
  `cancel_all_pending_tasks` → `@remind list`, `@remind delete <id>`,
  `@remind clear`

Keeping the advertised surface small is a correctness measure, not
tidiness. `xai/grok-4-1-fast-reasoning` starts returning empty
completions once the tool count climbs past roughly 25 — four
empty-response incidents on 2026-05-10, more than any day in the
preceding 30 — and a non-reasoning model asked to pick one tool out of
twenty will sometimes pick none and answer from its own invention, which
is how a draw request came back with a fabricated image URL on
2026-08-01. Hiding
these costs no capability: the handlers still exist, the commands above
still work, and memories still arrive on their own through the
background extraction pass. `save_memory` stays on the chat surface
because it is the one write extraction cannot replace — an explicit
"remember this" should stick at once rather than wait for the
candidate-reinforcement threshold.

Reminder fires keep all 21. A scheduled task runs as its creator with no
user present to type a command, so it may legitimately need to tidy up
after itself.

`set_reminder` and `schedule_llm_task` appear on the chat surface only
when `pendingTasksEnabled` is on for the channel, and it defaults to
off. Their schemas and prompt rules cost roughly 1,100 prompt tokens on
every completion, so channels that never schedule anything should not
pay for them. The `@remind` command and already-scheduled fires ignore
the gate. The matching list and cancel tools are off the chat surface
either way: users list and cancel both kinds with `@remind list`,
`@remind del` and `@remind clear`.

`check_service_status` covers two tiers of page. `statusPageUrls` entries back
the poller's cached snapshot: one entry per page when the model omits
`service`, or a single named lookup. `statusQueryablePages` entries are
answered only by name, fetched lazily and cached for 5 minutes — they never
appear in the omitted-`service` answer. Only `statusPageUrls` pages are named
in the tool's description text, in a "Monitored services" sentence, since
that sentence describes what an omitted `service` returns; the `service`
argument's enum is wider and lists every configured name across both keys,
which is how a queryable page gets asked for by name at all. The tool leaves
the request only when both keys are empty, rather than shipping a tool that
could only answer "not configured".

Verse is a strict subset of chat: everything chat hides, plus
`set_reminder`, `schedule_llm_task` and `check_service_status`, none of
which has an in-character use. The subset relation is asserted by
`test_verse_profile_is_strict_subset_of_chat`.

On any turn in a verse-enabled channel the model also receives the verse
tool set, advertised to every speaker whether or not they have opted in,
so the channel's cacheable prompt prefix stays byte-identical across
users; calls from a speaker with no avatar land on denial handlers. The
set is `verse_act`, `verse_move`, `verse_look`, `verse_recall`,
`verse_record`, `verse_edit` (refused unless the caller holds
`llm.verse.edit`), and `verse_storybook` (present only when
`verseStorybookEnabled` is on). See
[verse operations](../operator/verse.md).

## Scheduled LLM tasks

`schedule_llm_task` lets the assistant create recurring agentic work
("every weekday at 09:00, summarise the news"). The
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
| `pendingTasksEnabled` | `False` | Per channel. Advertises `set_reminder` and `schedule_llm_task` on the chat surface, so the model can create reminders and schedules from plain language. Listing and cancelling stay command-only. |
| `bridgeDebugInChannel` | `False` | Per channel. Appends a one-line footer listing every bridge call made during the turn, as `Plugin.command (N chars) [ok]`, or `[err:<reason>]` when the call was refused. Argument text is never echoed, only its length: bridge arguments are model-generated and can carry secrets. Full arguments go to the DEBUG log. |

To add a plugin outside the curated default, list every plugin you want
exposed, because the registry value replaces the default rather than
extending it. For example, to add `Factoids` and `Dict`:

```
@config plugins.LLM.bridgeAllowedPlugins Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG Factoids Dict
```

Once a plugin is in `bridgeAllowedPlugins` and loaded into Limnoria,
every read-only command it exports becomes available to the model on
the next turn.
