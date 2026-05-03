# Tuning & Monitoring

## Models and API keys (capability-based)

The model/key surface is capability-based: one assistant loop, an image generator, and a couple of specialized tool models.

| Setting | Scope | Used for |
|---------|-------|----------|
| `assistantModel` | Channel | Chat, planner loop, bridge tool selection, reminder parsing, memory extraction, memory cleanup, spontaneous participation, image-prompt rewrite, scheduled tasks |
| `assistantApiKey` | Channel | Same as above (private) |
| `assistantSystemPrompt` | Channel | Personality and constraints for assistant work |
| `imageModel` | Channel | Image generation |
| `imageApiKey` | Channel | Image generation (private) — does not auto-fall-back to `assistantApiKey` |
| `codeModel` / `codeApiKey` / `codeSystemPrompt` | Channel | `@code` (separate so cost/quality can differ from chat) |
| `searchModel` / `searchApiKey` | Channel | Web search/URL fetch tools (provider-specific grounding requirements). `searchApiKey` falls back to `assistantApiKey`; `searchModel` falls back to `assistantModel` if empty |

```
@config channel #yourchan plugins.LLM.assistantApiKey <your-key>
@config channel #yourchan plugins.LLM.assistantModel gemini/gemini-flash-latest
```

## Context tuning

Conversation context allows the bot to remember recent exchanges within a channel. These settings control how context behaves.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `contextEnabled` | `True` | Channel | Enable conversation context |
| `contextMaxMessages` | `20` | Channel | Max messages kept in per-user conversation history |
| `contextTimeoutMinutes` | `5` | Channel | Clear context after this many minutes of inactivity |
| `contextTrackAllMessages` | `False` | Channel | Track all channel messages, not just bot interactions |
| `channelContextMaxMessages` | `10` | Channel | Max messages in the shared channel context |

### Privacy note on contextTrackAllMessages

When `contextTrackAllMessages` is enabled, **all messages in the channel are sent to third-party LLM providers** as part of the conversation context. This is disabled by default for privacy reasons. Enable it only in channels where users are aware and consent to this. It is required for spontaneous participation to work.

### Channel context

The `channelContextMaxMessages` setting controls the shared context that lets the bot follow group conversations. When Alice asks something and Bob follows up, the bot can connect the two because both exchanges are in the shared channel context.

## Memory tuning

Non-volatile memory stores facts about users that persist across sessions and context resets.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `memoryEnabled` | `True` | Channel | Enable automatic memory extraction |
| `memoryMaxPerUser` | `50` | Global | Maximum memories stored per user |
| `memoryCleanupInterval` | `3` | Global | Extraction passes (with saves) between cleanup runs. `0` to disable |

Memory extraction and cleanup both run on the configured `assistantModel`/`assistantApiKey`. The extraction runs in the background after `@ask` interactions; the cleanup process periodically consolidates duplicate or outdated memories.

## Spontaneous participation

When enabled, the bot occasionally joins channel conversations without being explicitly invoked.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `spontaneousEnabled` | `False` | Channel | Enable spontaneous replies |
| `spontaneousChance` | `15` | Channel | Percent chance (1-100) of evaluating a reply per message |
| `spontaneousCooldown` | `2` | Channel | Minimum minutes between spontaneous replies per channel |
| `spontaneousSystemPrompt` | *(see below)* | Channel | Personality prompt for spontaneous participation |

Spontaneous participation runs on the configured `assistantModel`/`assistantApiKey`. It requires `contextTrackAllMessages` to be enabled in the channel -- the bot needs to see the conversation to decide when to join in.

The default system prompt instructs the model to act as a channel regular who can respond naturally or reply with `PASS` to stay silent.

## Limnoria tool bridge

The bridge exposes loaded Limnoria plugin commands to the LLM as a single dispatch tool. When the model needs to answer a factual question that a stock plugin already handles — a ping, the current time, a seen lookup — it can defer to that plugin rather than guessing or duplicating the feature inside the LLM plugin. Phase 1 is opt-in per channel; no plugin is bridged globally.

### Enabling the bridge

One per-channel setting opts the channel in:

```
@config channel #yourchan plugins.LLM.bridgeEnabled True
```

That activates the bridge with the curated default plugin set (see below). To override the default with a custom list, also set `bridgeAllowedPlugins`:

```
@config channel #yourchan plugins.LLM.bridgeAllowedPlugins Misc Time
```

A non-empty list replaces the default — you take full control of what the LLM sees. To disable the bridge entirely, set `bridgeEnabled False`.

### Curated default plugin set

When `bridgeAllowedPlugins` is empty (the registry default), the bridge falls back to:

`Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG`

Each of those is either pure-read or has its write commands gated separately by `bridgeAllowMutating` (see below). The set is curated so a fresh `bridgeEnabled True` gives operators useful coverage without forcing per-channel allowlisting.

The list is defined as `DEFAULT_ALLOWED_PLUGINS` in `plugins/llm/src/llm/limnoria_bridge.py`.

### Plugin loading prerequisite

The bridge can only expose plugins that are already loaded in Limnoria. Load them through the normal flow first:

```
@load Misc
@load Time
@load Later
@load Note
```

If a plugin in the (effective) allowlist is not loaded, its commands are silently absent from the tool.

### Security model

Three independent layers limit what the bridge can reach:

**Hard-coded denied plugins** — never bridged, regardless of operator config:

| Plugin | Reason |
|--------|--------|
| `LLM` | Recursion |
| `Owner` | Bot management |
| `Admin` | Bot management |
| `Config` | Bot management |
| `Channel` | Channel management |
| `User` | Account management |

**Hard-coded denied commands** — blocked even when their host plugin is allowlisted:

| Command | Reason |
|---------|--------|
| `Web.fetch` | SSRF vector |
| `Utilities.apply` | Re-dispatch bypass |
| `Utilities.let` | Re-dispatch bypass (same shape as `apply`) |
| `Misc.more` | Interactive scrollback only |
| `Misc.clearmores` | Interactive scrollback only |

**Limnoria's capability system** — applied per command at dispatch time. Commands guarded by default-deny anti-capabilities (`-owner`, `-admin`, `-trusted`, `-aka.*`, `-alias.*`, `-scheduler.add`, `-scheduler.remove`, `-scheduler.repeat`) will be refused unless the bot user has been explicitly granted the capability.

### Plugin name format

Names in `bridgeAllowedPlugins` must match Limnoria's camel-case form exactly — `Misc`, `Time`, `Math`. Lowercase or mismatched names are ignored silently.

### Write-command gate (`bridgeAllowMutating`)

Commands that modify persistent state — sending offline notes, registering RSS feeds, mutating karma, queueing PMs to other users — are hidden from the LLM by default, even when their host plugin is allowlisted. The bridge tool description omits them, and any hallucinated dispatch returns `denied: write commands disabled` as a layered defense.

To expose write commands per channel:

```
@config channel #yourchan plugins.LLM.bridgeAllowMutating True
```

Default is `False`. The classification list — what counts as a write — lives in `MUTATING_COMMANDS` in `plugins/llm/src/llm/limnoria_bridge.py`.

When the gate is closed and an allowlisted plugin has at least one hidden write, the bridge tool description appends a footer telling the LLM the gate exists. Pure-read allowlists (e.g. `Time Math Utilities Seen`) get no footer because nothing was hidden.

**Phase 1 → Phase 2 migration:** an operator who allowlisted `Later`, `Note`, `Karma`, `QuoteGrabs`, or `RSS` in Phase 1 will see those plugins' write commands disappear after upgrading. `Misc` is also affected — `Misc.tell` and `Misc.noticetell` send PMs/notices to other users on the caller's behalf and are classified mutating, so the recommended starter set's `Misc` will also lose those two leaves. Setting `bridgeAllowMutating True` per channel restores the prior behavior. `Time`, `Math`, `Utilities`, `Seen`, `Web`, and `DDG` are pure-read in the bridge and are unaffected.

### Scheduled LLM tasks (`schedule_llm_task`)

The `schedule_llm_task` tool schedules a future `@ask` invocation — at fire time the bot replays the user's IRC identity and runs the prompt through the assistant loop with full tool access (search, fetch, draw, code, the bridge). It is distinct from `set_reminder`:

- `set_reminder` delivers fixed text at fire time. Use it for "remind me to switch laundry over at 6 p.m.".
- `schedule_llm_task` runs an LLM turn at fire time. Use it for "every Monday at 9 a.m. check my open PRs and tell me which are stale" — anything that needs *tools* when it fires.

Both tools coexist; the LLM is taught to pick between them based on whether tool use is required at fire time.

**Per-channel budget:**

```
@config channel #yourchan plugins.LLM.bridgeScheduledTaskLimit 10
```

Default is `5`. Each creator's active scheduled tasks (in this channel) are capped here. Setting the value to `0` disables scheduling entirely for the channel. Each fire still counts against the user's normal `askRateLimit` bucket — this knob caps the *number* of pending schedules, not their cumulative cost.

**Identity at fire time:** schedules are bound to the creator's account at create time (the tool refuses unauthenticated callers). The persisted IRC prefix is replayed as a synthetic message at fire time, so account-identified users keep their capabilities across disconnects. Identity-only refusals at create time are a hard no — point users at NickServ-style auth before they can schedule.

**Loop guard:** a fired task cannot recursively call `schedule_llm_task` (depth cap of 1). It can still call `set_reminder`, the bridge tools, and any other LLM-native tool — only nested *scheduling* is blocked.

**Inspection:**

- The LLM lists/cancels schedules through the unified `list_pending_tasks` and `cancel_pending_task` tools (the same surface that handles reminders) — drive these by asking the bot in chat. `@scheduler list` will *not* show these tasks because they live in the LLM's own DB-backed schedule, not in Limnoria's Scheduler plugin events.
- Enable `bridgeDebugInChannel` to see fire-time bridge calls inline.
- Phase 2 Task 1's read/write gate (`bridgeAllowMutating` + `MUTATING_COMMANDS`) applies to bridge calls made *during* a fired task's @ask turn — write commands are hidden unless the channel has the gate open.

### Known limitation: nested subcommand groups

Plugins that group sub-leaves under a nested `Commands` class (notably `RSS`'s `announce add` / `announce remove` / `announce list` / `announce channels`) are not surfaced by the bridge today — Limnoria returns those leaves as multi-word strings and the bridge's enumerate/dispatch path expects single-token leaves. They are unreachable through the LLM whether the gate is open or closed. Operators who need to manage RSS announce subscriptions should keep using the native `@rss announce …` IRC command.

### Source

`plugins/llm/src/llm/limnoria_bridge.py`

## HTTP output

The `@code` and `@draw` commands save output as files served over HTTP. Two modes exist:

### Built-in HTTP server (default)

When `httpRoot` is empty, files are saved to Limnoria's web directory (`data/web/llm/`) and served by Limnoria's built-in HTTP server. See the [Limnoria HTTP server documentation](https://docs.limnoria.net/use/httpserver.html) for how to configure the server's port and public URL.

### External web server

Set `httpRoot` to a filesystem path (e.g., `/var/www/llm`) and `httpUrlBase` to the corresponding public URL (e.g., `https://example.com/llm`). Files are saved to the directory and Limnoria's HTTP server is not used -- you are responsible for serving the directory with nginx, caddy, or similar.

| Setting | Default | Description |
|---------|---------|-------------|
| `httpRoot` | (empty) | Filesystem path for output files. Empty = use Limnoria's built-in server |
| `httpUrlBase` | (empty) | Base URL for accessing files. Empty = use Limnoria's public URL + `/llm/` |
| `longReplyLineThreshold` | `6` | Generated IRC reply lines after which chat replies are saved as HTML and replaced with a one-line teaser plus link. Set `0` to disable |
| `longReplyTeaserMaxChars` | `220` | Maximum characters in the one-line teaser for linked long replies |

### File cleanup

Output files are cleaned up automatically based on age and count:

| Setting | Default | Description |
|---------|---------|-------------|
| `fileCleanupAge` | `720` | Delete files older than this many hours (default: 30 days) |
| `fileCleanupMax` | `1000` | Maximum number of files to keep in the output directory |

## Monitoring

### Log locations

For systemd deployments:

```bash
journalctl --user -u vibebot         # service logs
journalctl --user -u vibebot -f      # follow logs in real time
```

The bot also writes to its own log file:

```bash
tail -f logs/messages.log            # from the bot's working directory
```

### Log level

The `logLevel` setting controls how verbose the plugin's logging is:

| Level | Use case |
|-------|----------|
| `WARNING` (default) | Normal operation -- only warnings and errors |
| `INFO` | See request flow and timing |
| `DEBUG` | Verbose tracing including server response headers |

Set it from IRC:

```
@config plugins.LLM.logLevel DEBUG
```

### Storage location

The bot stores reminders, usage statistics, and memories in a SQLite database. The location depends on `databasePath`:

- If empty (default): uses `data/LLM.db` relative to the bot's working directory
- If set: uses the specified path

### Common issues

**API key not working**

- Verify the key is set: `@config plugins.LLM.assistantApiKey` (it will show as masked)
- Check that the key matches the model's provider. A Google API key will not work with an `anthropic/` model.
- Set `logLevel` to `DEBUG` and retry to see the full error in logs.

**Context not working (bot does not remember previous messages)**

- Check that context is enabled: `@config channel #yourchannel plugins.LLM.contextEnabled`
- Check the timeout: if `contextTimeoutMinutes` is too short, context may expire between messages.
- Users can manually clear context with `@forget`. It resets automatically after the timeout.

**HTTP output not saving (no link from @code/@draw)**

- If using the built-in server: ensure Limnoria's HTTP server is enabled and its `publicUrl` is set. See the [Limnoria HTTP server documentation](https://docs.limnoria.net/use/httpserver.html).
- If using an external server: verify `httpRoot` is writable by the bot process, and `httpUrlBase` matches the directory served by your web server.
- Check `logLevel DEBUG` output for file save errors.

**Spontaneous replies not triggering**

- `spontaneousEnabled` must be `True` for the channel.
- `contextTrackAllMessages` must also be `True` for the same channel.
- The `spontaneousChance` is a percent chance per message, so low values may take many messages before triggering.
- Check `spontaneousCooldown` -- the bot will not reply more than once per cooldown period.
