# Configuration

VibeBot uses Limnoria's registry system for all configuration. See the [Limnoria configuration documentation](https://docs.limnoria.net/use/configuration.html) for general information on how the registry works.

Set values from IRC with:

```
@config plugins.LLM.<settingName> <value>
```

Or view current values:

```
@config plugins.LLM.<settingName>
```

## API keys

API keys are stored as private values -- they are never displayed or logged by Limnoria. Capability-based, channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantApiKey` | API key for all assistant work: `@ask`, planner loop, memory, spontaneous, reminder parsing, scheduled tasks |
| `codeApiKey` | API key for `@code` |
| `imageApiKey` | API key for `@draw`. Does not auto-fall-back to `assistantApiKey` (image providers usually use a separate account) |
| `searchApiKey` | API key for web search/URL fetch tools. Falls back to `assistantApiKey` if empty |

Set a key from IRC (as bot owner):

```
@config plugins.LLM.assistantApiKey sk-your-key-here
```

## Model selection

Models follow [LiteLLM's provider/model format](https://docs.litellm.ai/docs/providers):

| Setting | Default | Description |
|---------|---------|-------------|
| `assistantModel` | `gemini/gemini-flash-latest` | All assistant text+tool work (chat, planner, memory, spontaneous, reminders, scheduled tasks). Must support vision if image URLs in chat should work. |
| `codeModel` | `gemini/gemini-1.5-flash` | Model for `@code` |
| `imageModel` | `vertex_ai/imagen-4.0-generate-001` | Model for `@draw` |
| `searchModel` | (empty, falls back to `assistantModel`) | Model for web search and URL fetching |

Model names are validated against LiteLLM's known providers. If you set an unrecognized model, the bot will reject it and suggest alternatives.

Models are channel-overridable, so you can run different models in different channels:

```
@config channel #dev plugins.LLM.assistantModel anthropic/claude-sonnet-4-20250514
@config channel #casual plugins.LLM.assistantModel gemini/gemini-flash-latest
```

## System prompts

System prompts define the bot's personality and behavior. They are channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantSystemPrompt` | Personality and constraints for all assistant work |
| `codeSystemPrompt` | Instructions for `@code` output format |
| `spontaneousSystemPrompt` | Personality for spontaneous channel participation |

Example -- give a channel a specialized personality:

```
@config channel #python plugins.LLM.assistantSystemPrompt You are a Python expert. Answer questions with Python examples. Be concise.
```

## Per-channel overrides

Most settings support per-channel values. Channel values override the global default for that channel only. The general pattern is:

```
@config channel #channel plugins.LLM.<settingName> <value>
```

Most settings support per-channel overrides: API keys, model selection, system prompts, context settings, memory, and spontaneous participation. Rate limits are global only.

## Other settings

| Setting | Default | Description |
|---------|---------|-------------|
| `helpUrl` | `https://rdrake.github.io/vibebot-v8/` | URL shown in help output |
| `timeout` | `30` | Timeout for LLM API calls (seconds) |
| `drawTimeout` | `120` | Timeout for image generation (seconds). Set to `0` to use the global timeout |
| `drawAutoRewriteMax` | `3` | Max automatic prompt rewrites when blocked by safety filters. Set to `0` to disable |
| `drawContextMaxAgeSeconds` | `60` | Pass conversation context to draw requests only when the last activity is within this many seconds. `0` to always start fresh |
| `maxPromptLength` | `10000` | Maximum user prompt length (characters) |
| `commandPrefixes` | `.` | Space-separated prefixes to sanitize in output (prevents IRC command injection) |
| `databasePath` | (empty) | Path to SQLite database. If empty, uses `data/LLM.db` |
| `logLevel` | `WARNING` | Plugin log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `metaMaxSteps` | `12` | Cap on tool-call round trips per assistant invocation. Stops runaway tool loops |
| `memoryPromotionThreshold` | `2` | Times a candidate fact must be reinforced before it becomes a durable memory. `1` disables the candidate stage (every extraction saved immediately) |
| `memoryCandidateTTLDays` | `14` | Days a candidate fact can sit unreinforced before pruning. `0` disables TTL pruning |
| `skipAutoWhoOnJoin` | `True` | Suppress Limnoria's automatic `WHO` query on channel join when both the `account-tag` and `extended-join` IRCv3 capabilities have been acknowledged. Set `False` only if those capabilities misbehave on your server |

## Retry settings

When an API call times out, the bot can retry in the background. These settings control how long retries continue:

| Setting | Default | Description |
|---------|---------|-------------|
| `askExpiry` | `60` | Max seconds to retry timed-out `@ask` requests. `0` to disable |
| `codeExpiry` | `60` | Max seconds to retry timed-out `@code` requests. `0` to disable |
| `drawExpiry` | `60` | Max seconds to retry timed-out `@draw` requests. `0` to disable |
