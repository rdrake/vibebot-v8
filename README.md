# VibeBot v8

Modern IRC bot with AI capabilities powered by LiteLLM.

## Features

- **Multi-provider AI**: OpenAI, Anthropic, Google Gemini, xAI Grok, and more via LiteLLM
- **Volatile memory**: Conversation context for natural follow-up questions (expires after timeout)
- **Vision support**: Automatically detects image URLs in prompts
- **Code generation**: Smart HTTP link generation for long code
- **Image generation**: Text-to-image via Vertex AI Imagen and xAI grok-imagine
- **Non-volatile memory**: Automatically extracts and remembers facts about users across conversations
- **Reminders & scheduled tasks**: Natural-language reminders plus recurring `schedule_llm_task` agentic flows
- **Abuse controls**: Capability checks, account gating, tiered rate limiting, bounded LLM concurrency
- **NickInMiddle plugin**: Companion plugin that injects the speaker's nick into the middle of bot replies for AfterNet readability
- **Modern Python**: Python 3.12–3.14 with full type hints
- **Quality tools**: Ruff for linting/formatting, ty for type checking, Hypothesis property tests, prek pre-commit hooks

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

make install
make run
```

Configure API keys via bot commands:
```
@config plugins.LLM.assistantApiKey YOUR_KEY
```

## Docker

Build and run locally:

```bash
make docker-build
make docker-run
```

Or pull from GHCR:

```bash
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

## Production Deployment

Install as a systemd user service:

```bash
make install-service
```

Then follow the printed instructions to copy your `bot.conf` and enable the service.

### Auto-Updates

Install the auto-update timer to automatically pull new images from GHCR:

```bash
make install-timer
```

This checks for updates every 15 minutes and restarts the bot if a new version is found.

```bash
# Check timer status
systemctl --user status vibebot-updater.timer

# View update logs
journalctl --user -u vibebot-updater.service -f

# Disable auto-updates
make uninstall-timer
```

## Static Assets (Reverse Proxy)

When serving generated code snippets and images via Nginx or Apache, set the
public URL the bot should advertise:

```
@config supybot.servers.http.publicUrl https://example.com
```

The bot will generate URLs like `https://example.com/llm/filename.py`.

### Nginx example

If you let Limnoria's built-in HTTP server handle requests, proxy `/llm`
through Nginx so users hit a clean public hostname:

```nginx
location /llm/ {
    proxy_pass         http://127.0.0.1:8080/llm/;
    proxy_set_header   Host $host;
    proxy_set_header   X-Forwarded-For $remote_addr;
    proxy_read_timeout 30s;

    # Generated artifacts are immutable (filenames are content-hashed).
    add_header Cache-Control "public, max-age=31536000, immutable";
}
```

If you instead point `httpRoot` at a directory served directly by Nginx,
serve the directory and skip the proxy:

```nginx
location /llm/ {
    alias       /var/www/llm/;
    autoindex   off;
    add_header  Cache-Control "public, max-age=31536000, immutable";
}
```

### Caddy example

```caddyfile
example.com {
    handle /llm/* {
        reverse_proxy 127.0.0.1:8080
        header Cache-Control "public, max-age=31536000, immutable"
    }
}
```

## Commands

### User Commands

| Command | Description |
|---------|-------------|
| `@ask <question>` | Ask with context, vision, and optional instructions |
| `@code <request>` | Generate code with HTTP link output |
| `@draw <prompt>` | Generate image (account required) |
| `@forget [channel]` | Clear volatile memory (conversation context) |
| `@memories [subcommand]` | Manage non-volatile memory (stored facts) |
| `@instruct [text \| clear]` | Set persistent instructions for ask |
| `@remind [text \| list \| del \| clear]` | Set and manage reminders |
| `@usage [nick \| #channel]` | View API usage statistics |

## Configuration

### Models

Configure models in `bot.conf`:

```
# Free tier (Gemini Flash)
supybot.plugins.LLM.assistantModel: gemini/gemini-1.5-flash
supybot.plugins.LLM.codeModel: gemini/gemini-1.5-flash

# Paid tier (Vertex Imagen)
supybot.plugins.LLM.imageModel: vertex_ai/imagen-4.0-generate-001
```

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for supported models.

### Volatile Memory (Conversation Context)

```
supybot.plugins.LLM.contextEnabled: True
supybot.plugins.LLM.contextMaxMessages: 20
supybot.plugins.LLM.contextTimeoutMinutes: 5
```

Volatile memory is per-user per-channel. Cleared by `@forget`, after the configured idle timeout (default 5 minutes), or when max messages exceeded.

### Non-volatile Memory (Stored Facts)

```
supybot.plugins.LLM.memoryEnabled: True
supybot.plugins.LLM.memoryMaxPerUser: 50
```

Memory extraction and cleanup share the configured `assistantModel` / `assistantApiKey`.

Facts are automatically extracted from `@ask` and `@code` conversations. Users manage non-volatile memory with `@memories`.

### Abuse Controls

The plugin layers several protections:

- Capability checks on command wrappers
- Authenticated-account requirement for expensive commands (`draw`)
- Tiered per-account rate limiting (registered, trusted, unregistered) for all commands

Protection matrix:

| Command | Capability | Authenticated Required | Rate Limited |
|---------|------------|-------------------|--------------|
| `@ask` | `llm.ask` | No | Yes (optional) |
| `@code` | `llm.code` | No | Yes (optional) |
| `@draw` | `llm.draw` | Yes | Yes (optional) |

Rate-limit config (per-command, per-tier):

```
supybot.plugins.LLM.enforceRateLimits: True
# ask (count/window per tier)
supybot.plugins.LLM.askRateLimitCount: 15
supybot.plugins.LLM.askRateLimitWindow: 60
supybot.plugins.LLM.askTrustedRateLimitCount: 15
supybot.plugins.LLM.askUnregRateLimitCount: 15
# code
supybot.plugins.LLM.codeRateLimitCount: 10
supybot.plugins.LLM.codeRateLimitWindow: 60
supybot.plugins.LLM.codeTrustedRateLimitCount: 0   # unlimited
supybot.plugins.LLM.codeUnregRateLimitCount: 2
# draw
supybot.plugins.LLM.drawRateLimitCount: 2
supybot.plugins.LLM.drawRateLimitWindow: 300
supybot.plugins.LLM.drawTrustedRateLimitCount: 5
supybot.plugins.LLM.drawTrustedRateLimitWindow: 60
supybot.plugins.LLM.drawUnregRateLimitCount: 0
```

See `docs/guide/operator/rate-limiting-security.md` for the full per-tier matrix and authoritative defaults.

### IRC Staging Smoke Checklist

Run this sequence on a staging bot connected to IRC:

1. Monitor mode (`enforceRateLimits=False`):
   - Send `draw` prompts above configured threshold.
   - Verify requests still execute (not blocked).
   - Verify logs include `rate_limit_shadow` entries.
2. Enforced mode (`enforceRateLimits=True`):
   - Repeat prompts above threshold.
   - Verify bot replies with rate-limit error and provider call is not executed.
   - Verify usage rows include `status=rate_limited`.

### HTTP Output

```
supybot.plugins.LLM.httpRoot: /var/www/llm
supybot.plugins.LLM.httpUrlBase: https://example.com/llm
```

If `httpRoot` is empty (default), uses Limnoria's built-in HTTP server at `data/web/llm/`.

## Development

### Run Tests

```bash
make test
```

### Lint and Format

```bash
make lint        # Check code
make format      # Format code
make typecheck   # Check types
make check       # Run all checks
```

### Code Quality

This project uses:
- **uv**: Fast Python package manager
- **prek**: Fast Rust-based pre-commit hooks
- **Ruff**: Fast Python linter and formatter
- **deptry**: Dependency issue detection
- **ty**: Astral's static type checker
- **pytest**: Testing framework with 93% coverage floor
- **Hypothesis**: Property-based tests for invariants (`test_*_properties.py`)
- **Dependabot**: Automated dependency updates (weekly)

All code must pass linting, formatting, type checking, and tests with ≥93% coverage.

## Architecture

```
vibebot-v8/
├── plugins/llm/             # Main AI plugin (LiteLLM + assistant tooling)
│   ├── src/llm/
│   │   ├── plugin.py        # IRC command handlers
│   │   ├── service.py       # LiteLLM business logic
│   │   ├── assistant.py     # Tool-using chat profile
│   │   ├── executor.py      # Bounded LLM concurrency executor
│   │   ├── persistence.py   # SQLite store (memories, reminders, schedules)
│   │   ├── limnoria_bridge.py  # Allowlisted Limnoria-as-tool surface
│   │   ├── tracing.py       # Structured trace severity helpers
│   │   ├── config.py        # Registry options
│   │   └── context.py       # Conversation history
│   └── tests/               # Unit + Hypothesis property tests
├── plugins/rpg/             # Lightweight RPG plugin
├── plugins/nickinmiddle/    # Inserts speaker's nick mid-reply (AfterNet UX)
├── bot.conf                 # Bot configuration
└── pyproject.toml           # Workspace + dependencies
```

### Design Principles

1. **Security First**
   - API keys never logged (sanitized in all error paths)
   - Malicious URLs blocked (javascript:, data:, file:, path traversal)
   - Thread-safe API key handling (passed directly, never env vars)

2. **Separation of Concerns**
   - `plugin.py`: IRC protocol and command routing
   - `service.py`: AI API calls and business logic
   - `context.py`: Conversation history management

3. **Modern Python**
   - Python 3.12+ type hints throughout
   - Type checking with ty
   - Modern patterns (dataclasses, context managers)

## Troubleshooting

### API Key Not Working

Check configuration via `@config`:
```
@config plugins.LLM.assistantApiKey
```

Should show the key is set (value is private and not displayed in full).

### Volatile Memory Not Working

Clear and retry:
```
@forget
@ask Your new question here
```

### Code Not Saving to HTTP

1. Check directory exists and is writable:
   ```bash
   ls -la /var/www/llm
   ```

2. Check web server is serving the directory

3. Check logs:
   ```bash
   tail -f logs/messages.log
   ```

## License

See LICENSE file for details.

## Credits

- Built with [Limnoria](https://github.com/ProgVal/Limnoria)
- Powered by [LiteLLM](https://github.com/BerriAI/litellm)
- Developed for AfterNET IRC (irc.afternet.org)
