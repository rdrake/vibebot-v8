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

Each command has its own API key setting. Keys are stored as private values -- they are never displayed or logged by Limnoria.

| Setting | Description |
|---------|-------------|
| `askApiKey` | API key for the `@ask` command |
| `codeApiKey` | API key for the `@code` command |
| `drawApiKey` | API key for the `@draw` command |
| `memoryApiKey` | API key for memory extraction. Falls back to `askApiKey` if empty |
| `spontaneousApiKey` | API key for spontaneous participation. Falls back to `askApiKey` if empty |

Set a key from IRC (as bot owner):

```
@config plugins.LLM.askApiKey sk-your-key-here
```

API keys are global settings -- they cannot be overridden per channel.

## Model selection

Each command uses a separate model setting. Models follow [LiteLLM's provider/model format](https://docs.litellm.ai/docs/providers):

| Setting | Default | Description |
|---------|---------|-------------|
| `askModel` | `gemini/gemini-flash-latest` | Model for `@ask` (supports vision) |
| `codeModel` | `gemini/gemini-1.5-flash` | Model for `@code` |
| `drawModel` | `vertex_ai/imagen-4.0-generate-001` | Model for `@draw` |
| `memoryExtractionModel` | `gemini/gemini-2.0-flash-lite` | Model for memory extraction |
| `memoryCleanupModel` | `gemini/gemini-3.1-flash-lite-preview` | Model for memory cleanup |
| `spontaneousModel` | `gemini/gemini-2.0-flash-lite` | Model for spontaneous replies |

Model names are validated against LiteLLM's known providers. If you set an unrecognized model, the bot will reject it and suggest alternatives.

Models are channel-overridable, so you can run different models in different channels:

```
@config channel #dev plugins.LLM.askModel anthropic/claude-sonnet-4-20250514
@config channel #casual plugins.LLM.askModel gemini/gemini-flash-latest
```

## System prompts

System prompts define the bot's personality and behavior. They are channel-overridable.

| Setting | Description |
|---------|-------------|
| `askSystemPrompt` | Personality and constraints for `@ask` |
| `codeSystemPrompt` | Instructions for `@code` output format |
| `spontaneousSystemPrompt` | Personality for spontaneous channel participation |

Example -- give a channel a specialized personality:

```
@config channel #python plugins.LLM.askSystemPrompt You are a Python expert. Answer questions with Python examples. Be concise.
```

## Per-channel overrides

Most settings support per-channel values. Channel values override the global default for that channel only. The general pattern is:

```
@config channel #channel plugins.LLM.<settingName> <value>
```

Settings that support per-channel overrides include: model selection, system prompts, context settings, memory, and spontaneous participation. API keys and rate limits are global only.

## Other settings

| Setting | Default | Description |
|---------|---------|-------------|
| `helpUrl` | `https://rdrake.github.io/vibebot-v8/` | URL shown in help output |
| `timeout` | `30` | Timeout for LLM API calls (seconds) |
| `drawTimeout` | `120` | Timeout for image generation (seconds). Set to `0` to use the global timeout |
| `drawAutoRewriteMax` | `3` | Max automatic prompt rewrites when blocked by safety filters. Set to `0` to disable |
| `maxPromptLength` | `10000` | Maximum user prompt length (characters) |
| `commandPrefixes` | `.` | Space-separated prefixes to sanitize in output (prevents IRC command injection) |
| `databasePath` | (empty) | Path to SQLite database. If empty, uses `data/LLM.db` |
| `logLevel` | `WARNING` | Plugin log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

## Retry settings

When an API call times out, the bot can retry in the background. These settings control how long retries continue:

| Setting | Default | Description |
|---------|---------|-------------|
| `askExpiry` | `60` | Max seconds to retry timed-out `@ask` requests. `0` to disable |
| `codeExpiry` | `60` | Max seconds to retry timed-out `@code` requests. `0` to disable |
| `drawExpiry` | `60` | Max seconds to retry timed-out `@draw` requests. `0` to disable |
