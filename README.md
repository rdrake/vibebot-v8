# VibeBot v8

Modern IRC bot with AI capabilities powered by LiteLLM.

## Features

- **Multi-provider AI**: OpenAI, Anthropic, Google Gemini, and more via LiteLLM
- **Conversation context**: Follow-up questions remember previous messages
- **Vision support**: Automatically detects image URLs in prompts
- **Code generation**: Smart HTTP link generation for long code
- **Image generation**: Text-to-image via Vertex AI Imagen
- **Rate limiting**: Per-user rate limits to prevent abuse
- **Modern Python**: Python 3.14 with full type hints
- **Quality tools**: Ruff for linting/formatting, ty for type checking

## Quick Start

```bash
make install
make run
```

Configure API keys via bot commands:
```
%config plugins.LLM.askApiKey YOUR_KEY
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

## Static Assets (Reverse Proxy)

When serving code/images via Nginx or Apache, set the public URL:

```
%config supybot.servers.http.publicUrl https://example.com
```

The bot will generate URLs like `https://example.com/llm/filename.py`.

## Commands

### User Commands

| Command | Description |
|---------|-------------|
| `%ask <question>` | Ask AI a question (supports vision with image URLs, remembers context) |
| `%code <request>` | Generate code (no context, each request independent) |
| `%draw <prompt>` | Generate an image |
| `%forget` | Clear your conversation context |

### Admin Commands

| Command | Description |
|---------|-------------|
| `%llmkeys` | Check API key status (shows first 3 chars only) |
| `%llmreset [nick]` | Reset rate limits for a user or all users |

## Configuration

### Models

Configure models in `bot.conf`:

```
# Free tier (Gemini Flash)
supybot.plugins.LLM.askModel: gemini/gemini-1.5-flash
supybot.plugins.LLM.codeModel: gemini/gemini-1.5-flash

# Paid tier (Vertex Imagen)
supybot.plugins.LLM.drawModel: vertex_ai/imagen-4.0-generate-001
```

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for supported models.

### Rate Limiting

```
supybot.plugins.LLM.rateLimitEnabled: True
supybot.plugins.LLM.rateLimitRequests: 10
supybot.plugins.LLM.rateLimitWindow: 60
```

This allows 10 requests per user per 60 seconds.

### Conversation Context

```
supybot.plugins.LLM.contextEnabled: True
supybot.plugins.LLM.contextMaxMessages: 20
supybot.plugins.LLM.contextTimeoutMinutes: 30
```

Context is per-user per-channel. Cleared after 30 minutes of inactivity or when max messages exceeded.

### HTTP Output

```
supybot.plugins.LLM.httpRoot: /var/www/llm
supybot.plugins.LLM.codeUrlBase: https://example.com/llm
supybot.plugins.LLM.codeThreshold: 20
```

Code longer than `codeThreshold` lines is saved to HTTP instead of pasted to IRC.

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
- **Ruff**: Fast Python linter and formatter
- **ty**: Astral's static type checker
- **pytest**: Testing framework

All code must pass linting, formatting, type checking, and tests.

## Architecture

```
vibebot-v8/
├── plugins/llm/
│   ├── src/llm/
│   │   ├── plugin.py       # IRC command handlers
│   │   ├── service.py      # LiteLLM business logic
│   │   ├── config.py       # Configuration definitions
│   │   ├── rate_limiter.py # Rate limiting
│   │   └── context.py      # Conversation history
│   └── tests/              # Unit tests
├── bot.conf                # Bot configuration
└── pyproject.toml          # Dependencies and tools
```

### Design Principles

1. **Security First**
   - API keys never logged (sanitized in all error paths)
   - Malicious URLs blocked (javascript:, data:, file:, path traversal)
   - Thread-safe API key handling (passed directly, never env vars)

2. **Separation of Concerns**
   - `plugin.py`: IRC protocol and command routing
   - `service.py`: AI API calls and business logic
   - `rate_limiter.py`: Rate limiting logic
   - `context.py`: Conversation history management

3. **Modern Python**
   - Python 3.14 type hints throughout
   - Type checking with ty
   - Modern patterns (dataclasses, context managers)

## Troubleshooting

### API Key Not Working

Check configuration:
```
%llmkeys
```

Should show `AIz...(36 chars hidden)` or similar.

### Rate Limited

Admin can reset:
```
%llmreset         # Reset all
%llmreset nick    # Reset specific user
```

### Context Not Working

Clear and retry:
```
%forget
%ask Your new question here
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
