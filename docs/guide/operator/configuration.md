# Configuration

VibeBot uses Limnoria's registry for all configuration. See the [Limnoria configuration documentation](https://docs.limnoria.net/use/configuration.html) for how the registry works in general.

Set values from IRC with:

```
@config plugins.LLM.<settingName> <value>
```

View a current value by omitting the value. Channel-overridable settings also accept a per-channel form:

```
@config channel #channel plugins.LLM.<settingName> <value>
```

**Scope** in the tables below means: *channel* settings accept per-channel overrides; *global* settings apply everywhere.

## API keys

API keys are **not** part of the Limnoria registry. They come from environment variables, one per provider, read at the point each provider call is made and selected by the provider of the model being called — not by which command or channel triggered the call. There is no `@config` setting for a key anymore; the old `assistantApiKey` / `codeApiKey` / `imageApiKey` / `searchApiKey` registry settings were removed, not deprecated, and `@config` on them now errors.

| Variable | Provider |
|----------|----------|
| `XAI_API_KEY` | xAI (Grok) |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |

Any other provider LiteLLM recognises — `vertex_ai`, `openrouter`, `azure`, `bedrock`, and so on — is not one of the four managed providers above. It resolves to no key from this plugin, and LiteLLM falls back to that provider's own native credentials (Application Default Credentials, IAM, or its own environment variables). The default `imageModel`, `vertex_ai/imagen-4.0-generate-001`, works this way via ADC and needs nothing from the table above.

Set the variables in `.env.example`, copied to the env file the container loads (see [Operations](operations.md)). A model whose provider has no key configured fails with an error naming the missing variable, for example `no API key configured for provider 'xai' (set XAI_API_KEY)`.

## Model selection

Models follow [LiteLLM's provider/model format](https://docs.litellm.ai/docs/providers). Model names are validated against LiteLLM's known providers; the bot rejects unrecognized names and suggests alternatives.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `assistantModel` | channel | `gemini/gemini-flash-latest` | All assistant text and tool work: chat, planner, memory, reminders, scheduled tasks. Needs vision support if users paste image URLs |
| `codeModel` | channel | `gemini/gemini-1.5-flash` | Model for `@code` |
| `imageModel` | channel | `vertex_ai/imagen-4.0-generate-001` | Model for `@draw` |
| `searchModel` | channel | empty | Model for web search and URL fetch. Falls back to `assistantModel` |
| `verseModel` | channel | empty | Model for verse-mode replies. Falls back to `assistantModel`. Set this when the assistant model is a terse reasoning model that writes poor prose |
| `verseCompactionModel` | global | `gemini/gemini-flash-lite-latest` | Cheap model for the daily verse compaction job |

Channel overrides let different channels run different models:

```
@config channel #dev plugins.LLM.assistantModel anthropic/claude-sonnet-4-20250514
@config channel #casual plugins.LLM.assistantModel gemini/gemini-flash-latest
```

## System prompts

System prompts define the bot's personality and constraints. Both are channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantSystemPrompt` | Personality and constraints for all assistant work. Verse mode layers scene context on top of this prompt rather than replacing it |
| `codeSystemPrompt` | Instructions for `@code` output format |

Example: give a channel a specialized personality:

```
@config channel #python plugins.LLM.assistantSystemPrompt You are a Python expert. Answer with Python examples. Be concise.
```

## Verse

The verse is the per-channel world model. See [The verse](forest-verse.md) for concepts and operation; this table is the key reference.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseEnabled` | channel | `False` | Main switch. When off, verse commands are unavailable and no events are recorded |
| `verseTriggerRegex` | channel | `\bverse\b` | Case-insensitive pattern that marks a message as a canon reference, alongside entity mentions. A match grounds the reply in canon and, for an avatar holder, turns it into a prose tale. It does not enter roleplay mode; that needs `@rp`. Empty disables the keyword signal |
| `verseRoleplayStickyTtlSeconds` | channel | `900` | Sliding inactivity timeout for sticky `@rp on` roleplay. Each in-character turn refreshes it. `0` never expires |
| `verseChatRecordEnabled` | channel | `False` | Let an opted-in avatar's ordinary chat turn call `verse_record`. Off keeps canon growing only during roleplay |
| `verseStoryAmbientMaxImages` | channel | `1` | Image budget for an ambient story that reaches the storybook without an explicit illustrate cue. `0` is text only |
| `verseEventRetentionDays` | channel | `30` | Events older than this are eligible for compaction |
| `verseAutoEntityRetireDays` | channel | `14` | Days without a mention before auto-created non-player characters (NPCs) retire. `0` disables the sweep. Pinned entities are exempt |
| `verseAutoEntityMaxNamesPerCall` | channel | `8` | Cap on the `actors` array in one `verse_record` call |
| `verseRosterMaxChars` | channel | `4000` | Character cap on the canon-roster block in the verse system prompt |
| `verseStyleExemplars` | channel | `[]` | Curated style exemplars injected into the verse prompt. Populate offline with the taste miner |
| `verseReactionCaptureEnabled` | channel | `True` | Capture IRCv3 emoji reactions to verse lines as an offline approval signal |
| `verseCompactionDailyAt` | global | `03:00` | Local `HH:MM` time for the daily compaction sweep |
| `verseCompactionMinKeepEvents` | global | `20` | Verses with fewer total events are skipped by compaction |

### Storybook

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseStorybookEnabled` | channel | `False` | Expose the `verse_storybook` tool on verse turns |
| `verseStorybookMaxImages` | channel | `5` | Cap on illustrations per storybook |
| `verseStorybookMaxPerTurn` | channel | `1` | Cap on storybooks per verse turn |
| `verseStorybookCooldownSeconds` | channel | `300` | Per-account cooldown between storybooks |
| `verseStorybookDailyImageCap` | channel | `30` | Daily ceiling on storybook illustrations |
| `verseStorybookMaxChars` | channel | `6000` | Character cap on the storybook body |
| `verseStorybookImageTimeout` | channel | `45` | Seconds to wait for each illustration |

## Memory

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `memoryEnabled` | channel | `True` | Enable automatic memory extraction |
| `memoryMaxPerUser` | global | `50` | Cap on durable memories per user |
| `memoryCleanupInterval` | global | `3` | Extraction passes between cleanup runs. `0` disables |
| `memoryPromotionThreshold` | global | `2` | Mentions a candidate fact needs before promotion. `1` restores single-stage saving |
| `memoryCandidateTTLDays` | global | `14` | Days an unreinforced candidate survives. `0` disables pruning |

See [Memory promotion](memory-promotion.md) for how the two-stage pipeline works.

## Conversation context

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `contextEnabled` | channel | `True` | Enable conversation context |
| `contextMaxMessages` | channel | `20` | Messages kept in per-user history |
| `contextTimeoutMinutes` | channel | `5` | Clear context after this much inactivity |
| `contextTrackAllMessages` | channel | `False` | Track every channel message, not just bot interactions. Off by default for privacy |
| `channelContextMaxMessages` | channel | `10` | Messages in the shared channel context |

## Rate limiting

Rate limits follow the pattern `{command}{tier}RateLimitCount` and `{command}{tier}RateLimitWindow` for the commands `ask`, `code`, `draw`, and `story`. All rate-limit settings are global. See [Rate limiting and security](rate-limiting-security.md) for tiers, defaults, and the observe-only switch `enforceRateLimits`.

## Web and HTTP output

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `httpRoot` | global | empty | Filesystem path for output files. Empty uses Limnoria's built-in HTTP server |
| `httpUrlBase` | global | empty | Public base URL for output files. Empty uses Limnoria's public URL plus `/llm/` |
| `imageUploadUrl` | global | empty | External image host for generated images. Empty stores them locally |
| `helpUrl` | global | `https://rdrake.github.io/vibebot-v8/` | URL shown in help output |
| `longReplyTeaserMaxChars` | channel | `220` | Characters in the one-line teaser for long replies saved as HTML |
| `fileCleanupAge` | global | `720` | Delete output files older than this many hours (30 days) |
| `fileCleanupMax` | global | `1000` | Cap on files kept in the output directory |

### Offloading images to an external host

Set `imageUploadUrl` to an uploader that takes a multipart `POST` on the `images[]` field and answers with JSON:

```json
{"results": [{"success": true, "filePath": "/img/img_abc123.png"}]}
```

Generated images then live on that host instead of under `httpRoot`, and storybook pages embed the absolute URL. Pages themselves (answers, code, stories) still come from your own HTTP server, so this offloads bandwidth rather than replacing it.

Every failure falls back to local storage: endpoint unreachable, upload rejected, image over 10 MB, or a reply naming a URL that is not an image on the configured host. An outage costs a slower draw, nothing more.

[paste.boxlabs.uk/img/](https://paste.boxlabs.uk/img/) is Eck's uploader, the host this contract follows, and it needs no key. Ask the operator before you point the bot at somebody else's host.

## Limnoria tool bridge

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `bridgeEnabled` | channel | `False` | Expose loaded Limnoria plugin commands to the LLM as a tool |
| `bridgeAllowedPlugins` | channel | empty | Plugin allowlist. Empty uses the curated read-safe default set |
| `bridgeAllowMutating` | channel | `False` | Expose state-changing commands through the bridge |
| `bridgeScheduledTaskLimit` | channel | `5` | Active scheduled LLM tasks per creator per channel. `0` disables scheduling |
| `bridgeDebugInChannel` | channel | `False` | Append a bridge-call debug footer to replies |

See [Tuning and monitoring](tuning-monitoring.md) for the bridge's security model and the scheduled-task tool.

## Reminders and scheduled tasks

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `pendingTasksEnabled` | channel | `False` | Advertise the five reminder and scheduled-task tools to the model, so natural-language scheduling works. Off by default: the schemas and their rules cost roughly 1,100 prompt tokens on every completion in the channel |

The `@remind` command and the firing of reminders and tasks already created work regardless. This key only controls whether the model can create, list, and cancel them from plain language.

## Timeouts and retries

When an API call times out, the bot retries in the background for a bounded window.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `timeout` | global | `30` | Timeout for LLM API calls, in seconds |
| `drawTimeout` | global | `120` | Timeout for image generation. `0` uses the global timeout |
| `askExpiry` | global | `60` | Seconds to keep retrying a timed-out `@ask`. `0` disables |
| `codeExpiry` | global | `60` | Seconds to keep retrying a timed-out `@code`. `0` disables |
| `drawExpiry` | global | `60` | Seconds to keep retrying a timed-out `@draw`. `0` disables |

## Draw behaviour

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `drawAutoRewriteMax` | channel | `3` | Automatic prompt rewrites when a safety filter blocks a request. `0` disables |
| `drawContextMaxAgeSeconds` | channel | `60` | Pass conversation context to draw requests only when the last activity is this recent. `0` always starts fresh |

## Other settings

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `maxPromptLength` | global | `10000` | Longest accepted user prompt, in characters |
| `commandPrefixes` | global | `.` | Space-separated prefixes sanitized in output to prevent IRC command injection |
| `databasePath` | global | empty | Path to the SQLite database. Empty uses `data/LLM.db` |
| `logLevel` | global | `WARNING` | Plugin log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `metaMaxSteps` | global | `12` | Cap on tool-call round trips per assistant turn. Stops runaway tool loops |
| `maxConcurrentLLMCalls` | global | `16` | Cap on simultaneous outbound LLM calls across all surfaces |
| `skipAutoWhoOnJoin` | global | `True` | Skip Limnoria's automatic `WHO` on join when the `account-tag` and `extended-join` capabilities are active |
| `enforceRateLimits` | global | `True` | Enforce rate limits. `False` tracks and logs without blocking |
