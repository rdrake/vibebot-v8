# Tuning & Monitoring

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
| `memoryExtractionModel` | `gemini/gemini-2.0-flash-lite` | Channel | Model for extracting facts (cheap flash-tier recommended) |
| `memoryCleanupModel` | `gemini/gemini-3.1-flash-lite-preview` | Channel | Model for deduplicating and cleaning memories |
| `memoryCleanupInterval` | `3` | Global | Extraction passes (with saves) between cleanup runs. `0` to disable |
| `memoryApiKey` | (empty) | Global | API key for memory models. Falls back to `askApiKey` if empty |

Memory extraction runs in the background after `@ask` interactions. The cleanup process periodically consolidates duplicate or outdated memories.

## Spontaneous participation

When enabled, the bot occasionally joins channel conversations without being explicitly invoked.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `spontaneousEnabled` | `False` | Channel | Enable spontaneous replies |
| `spontaneousChance` | `15` | Channel | Percent chance (1-100) of evaluating a reply per message |
| `spontaneousCooldown` | `2` | Channel | Minimum minutes between spontaneous replies per channel |
| `spontaneousModel` | `gemini/gemini-2.0-flash-lite` | Channel | Model for spontaneous replies (cheap flash-tier recommended) |
| `spontaneousSystemPrompt` | *(see below)* | Channel | Personality prompt for spontaneous participation |
| `spontaneousApiKey` | (empty) | Global | API key. Falls back to `askApiKey` if empty |

Spontaneous participation requires `contextTrackAllMessages` to be enabled in the channel -- the bot needs to see the conversation to decide when to join in.

The default system prompt instructs the model to act as a channel regular who can respond naturally or reply with `PASS` to stay silent.

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
| `longReplyLineThreshold` | `6` | Lines after which chat replies are saved as HTML and replaced with a one-line teaser plus link. Set `0` to disable |
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

- Verify the key is set: `@config plugins.LLM.askApiKey` (it will show as masked)
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
