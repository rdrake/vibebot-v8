# Tuning and monitoring

## Choosing models

The model surface splits by workload so cost and quality can differ per task. All model and key settings are channel-overridable.

| Setting | Used for |
|---------|----------|
| `assistantModel` | Chat, the planner loop, bridge tool selection, reminder parsing, memory extraction and cleanup, image-prompt rewrites, scheduled tasks |
| `codeModel` | `@code`, kept separate so code generation can run a stronger or cheaper model than chat |
| `imageModel` | Image generation |
| `searchModel` | Web search and URL fetch tools. Falls back to `assistantModel` when empty |
| `verseModel` | Verse-mode narration. Falls back to `assistantModel` when empty |
| `verseCompactionModel` | The daily verse compaction digest (global) |

Guidance from operating this surface:

- `assistantModel` needs vision support if users paste image URLs into chat.
- Reasoning models make terse, flat narrators. If the assistant model is a reasoning model, point `verseModel` at a non-reasoning model for verse prose.
- Compaction is a cheap summarization job; leave `verseCompactionModel` on a flash-lite class model.

Keys are not part of this per-command surface: one environment variable per *provider* covers every model on that provider, regardless of which setting names the model. Point a channel at a different model and, if that model's provider is already configured, no key change is needed. See [Configuration → API keys](configuration.md#api-keys) for the variable names.

```
@config channel #yourchan plugins.LLM.assistantModel gemini/gemini-flash-latest
```

## Context tuning

Conversation context lets the bot follow recent exchanges in a channel.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `contextEnabled` | `True` | channel | Enable conversation context |
| `contextMaxMessages` | `20` | channel | Messages kept in per-user history |
| `contextTimeoutMinutes` | `5` | channel | Clear context after this much inactivity |
| `contextTrackAllMessages` | `False` | channel | Track every channel message, not just bot interactions |
| `channelContextMaxMessages` | `10` | channel | Messages in the shared channel context |

The shared channel context is what lets the bot connect a question from one user with a follow-up from another.

**Privacy note.** With `contextTrackAllMessages` enabled, every message in the channel is sent to third-party LLM providers as context. The setting is off by default. Enable it only where users know and consent.

## Limnoria tool bridge

The bridge exposes loaded Limnoria plugin commands to the LLM as one dispatch tool. When the model needs something a stock plugin already does, a ping, the time, a seen lookup, it defers to that plugin instead of guessing. The bridge is opt-in per channel.

### Enabling

```
@config channel #yourchan plugins.LLM.bridgeEnabled True
```

That activates the curated default plugin set. To take full control, set an explicit list; a non-empty value replaces the default:

```
@config channel #yourchan plugins.LLM.bridgeAllowedPlugins Misc Time
```

Names must match Limnoria's exact form (`Misc`, `Time`, `Math`); mismatched names are ignored silently. The bridge can only expose plugins that are already loaded, so `@load` them first.

### Curated default set

With `bridgeAllowedPlugins` empty, the bridge exposes:

`Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG`

Each is pure-read or has its write commands gated by `bridgeAllowMutating`. The list lives in `DEFAULT_ALLOWED_PLUGINS` in `plugins/llm/src/llm/limnoria_bridge.py`.

### Security model

Three independent layers limit what the bridge reaches:

1. **Hard-denied plugins**, never bridged regardless of config: `LLM` (recursion), `Owner`, `Admin`, `Config`, `Channel`, `User` (bot, channel, and account management).
2. **Hard-denied commands**, blocked even inside allowlisted plugins: `Web.fetch` (request-forgery vector), `Utilities.apply` and `Utilities.let` (re-dispatch bypass), `Misc.more` and `Misc.clearmores` (interactive scrollback only).
3. **Limnoria capabilities**, applied per command at dispatch time. Commands guarded by default-deny anti-capabilities are refused unless explicitly granted.

### Write-command gate

Commands that change persistent state (offline notes, RSS registrations, karma changes, queued PMs) stay hidden from the LLM by default, even when their plugin is allowlisted. Any hallucinated dispatch of a hidden write returns a denial. Open the gate per channel:

```
@config channel #yourchan plugins.LLM.bridgeAllowMutating True
```

The classification lives in `MUTATING_COMMANDS` in `plugins/llm/src/llm/limnoria_bridge.py`. When the gate is closed and an allowlisted plugin has hidden writes, the tool description tells the model the gate exists.

### Scheduled LLM tasks

The `schedule_llm_task` tool schedules a future assistant turn. At fire time the bot replays the creator's IRC identity and runs the prompt with full tool access. It differs from `set_reminder`:

- `set_reminder` delivers fixed text at fire time. "Remind me to switch the laundry at 6."
- `schedule_llm_task` runs an LLM turn at fire time. "Every Monday at 9, check my open PRs and tell me which are stale." Use it when the task needs tools when it fires.

Operational properties:

- `bridgeScheduledTaskLimit` (default 5, channel) caps each creator's active schedules; `0` disables scheduling. Each fire still counts against the creator's `ask` rate-limit bucket.
- Schedules bind to the creator's account at create time; the tool refuses unauthenticated callers.
- A fired task cannot schedule another task (depth cap of 1), which blocks recursion.
- Users list and cancel schedules by asking the bot in chat; the model uses its unified pending-task tools. Owners can also use `@remind admin`. Limnoria's own `@scheduler list` does not show these tasks; they live in the plugin's database.

### Known limitation

Plugins that nest sub-leaves under a `Commands` group (notably `RSS`'s `announce add` and friends) are not surfaced by the bridge. Manage those with the native IRC commands.

## HTTP output

`@code`, `@draw`, and `@story` save output as files served over HTTP.

- **Built-in server (default).** With `httpRoot` empty, files land in Limnoria's web directory (`data/web/llm/`) and its built-in HTTP server serves them. See the [Limnoria HTTP server documentation](https://docs.limnoria.net/use/httpserver.html) for port and public URL setup.
- **External server.** Set `httpRoot` to a filesystem path (for example `/var/www/llm`) and `httpUrlBase` to the matching public URL. You serve the directory with nginx, caddy, or similar.

| Setting | Default | Description |
|---------|---------|-------------|
| `httpRoot` | empty | Filesystem path for output files. Empty uses the built-in server |
| `httpUrlBase` | empty | Public base URL. Empty uses Limnoria's public URL plus `/llm/` |
| `longReplyTeaserMaxChars` | `220` | Characters in the one-line teaser when a long reply is saved as HTML and linked |
| `fileCleanupAge` | `720` | Delete output files older than this many hours (30 days) |
| `fileCleanupMax` | `1000` | Cap on files kept in the output directory |

## Concurrency

Every outbound LLM call, foreground or background, runs through one bounded executor so a slow provider never backs up the IRC event loop.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `maxConcurrentLLMCalls` | `16` | global | Cap on simultaneous outbound LLM calls |

Lower it on small hosts or when a provider rate-limits aggressively. The global `@usage` report (admin, by PM) appends an `executor: running/queued/max` field; sustained queueing with `running` at the cap means the executor is the bottleneck.

## Monitoring

### Logs

```bash
journalctl --user -u vibebot         # service logs
journalctl --user -u vibebot -f      # follow in real time
tail -f logs/messages.log            # bot message log, from the working directory
```

At startup, an `INFO`-level line reports secret redaction coverage: `secret redaction: %d handler(s) filtered, %d variable(s) covered: %s`, naming which environment variable *names* it covers (never their values). That line is the only positive confirmation that redaction is actually installed and live — worth checking after any change that touches logging handlers, given that credentials also sit directly in the container's environment (see [Operations → Credentials in the environment](operations.md#credentials-in-the-environment)).

### Log level

| Level | Use case |
|-------|----------|
| `WARNING` (default) | Normal operation: warnings and errors only |
| `INFO` | Request flow and timing |
| `DEBUG` | Verbose tracing, including provider response detail |

```
@config plugins.LLM.logLevel DEBUG
```

### Usage statistics

`@usage` shows API usage for yourself and the channel; the global overview by PM is admin-only. One caveat: verse turns are recorded under the `ask` label, so verse traffic is displayed as `ask` rows in the report.

### Storage

Reminders, usage statistics, and memories live in one SQLite database: `data/LLM.db` by default, or the path in `databasePath`. Each verse-enabled channel adds its own database under `data/verse/`.

### Common issues

**API key not working.**

- The error names the missing variable: `no API key configured for provider 'xai' (set XAI_API_KEY)`. Grep logs for `no API key configured` rather than the old "API key not configured" wording, which no longer appears anywhere.
- Confirm the key is set in the container, without printing it, by comparing a hash rather than a prefix:

  ```bash
  docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"
  ```

  A length of `0` means the variable is unset or empty. Compare the hash against a known-good value computed the same way, rather than eyeballing the key itself.
- Check the key matches the model's provider; a Gemini key does not work with an `anthropic/` model.
- Set `logLevel` to `DEBUG` and retry to see the full provider error.

**Bot does not remember previous messages.**

- Confirm context is on: `@config channel #yourchannel plugins.LLM.contextEnabled`.
- Check `contextTimeoutMinutes`; a short timeout expires context between messages.
- Users clear their own context with `@forget`.

**No link from `@code` or `@draw`.**

- Built-in server: confirm Limnoria's HTTP server is enabled and its public URL is set.
- External server: confirm the bot can write to `httpRoot` and that `httpUrlBase` matches what your web server serves.
- Check `DEBUG` logs for file-save errors.
