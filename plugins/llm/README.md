# LLM Plugin for Limnoria

AI-powered IRC commands using LiteLLM.

## Features

- Multi-provider support (OpenAI, Anthropic, Google, etc.)
- Vision support with automatic image URL detection
- Conversation context (memory between messages)
- Abuse controls with explicit manual moderation
- Thread-safe API key handling
- Comprehensive error handling

## Installation

This plugin is part of the VibeBot v8 workspace. Install dependencies:

```bash
cd ../..
make install
```

## Testing

```bash
make test
```

## Security

### Critical Patterns

1. **Thread-safe API keys**: API keys are passed directly to `litellm.completion()`, never mutating environment variables. This prevents race conditions in multi-threaded environments.

2. **API key sanitization**: All error messages are sanitized using `_sanitize()` method to remove API keys before logging.

3. **Malicious URL blocking**: `validate_image_url()` blocks:
   - Non-HTTP schemes (javascript:, data:, file:, ftp:)
   - Path traversal attempts (../)
   - Invalid image extensions

4. **Safe key display**: `llmkeys` command shows only first 3 characters of API keys.

5. **Private configuration**: All API key config values marked `private=True` in Limnoria.

## API Reference

### LLMService

Main service class for AI interactions.

#### Methods

- `completion(prompt, command, images, history)` - Generate text completion
- `image_generation(prompt)` - Generate image
- `save_code_to_http(code, language)` - Save code to HTTP server
- `validate_image_url(url)` - Validate image URL for security
- `safe_key_display(api_key)` - Safely display API key

### ConversationContext

Thread-safe conversation history manager.

#### Methods

- `add_message(nick, channel, role, content)` - Add message to history
- `get_messages(nick, channel)` - Get conversation history
- `clear(nick, channel)` - Clear specific user's context
- `clear_all()` - Clear all contexts
- `get_stats()` - Get context statistics

## Configuration

See main README for full configuration options.

### Command Protection Matrix

| Command | Capability | NickServ Required | Rate Limited |
|---------|------------|-------------------|--------------|
| `%ask` | `llm.ask` | No | No |
| `%code` | `llm.code` | No | No |
| `%draw` | `llm.draw` | Yes | Yes (optional) |
| `%animate` / `%video` | `llm.animate` | Yes | Yes (optional) |

### Moderation Model

- Manual account moderation only: `%flag`, `%unflag`, `%flagged`
- Flagged accounts are blocked across user-facing commands
- No automatic flagging side effects
- Optional draw/animate limiter controlled by:
  - `enforceRateLimits`
  - `drawRateLimitCount`, `drawRateLimitWindow`
  - `animateRateLimitCount`, `animateRateLimitWindow`

### Staging Smoke Test

1. Set `enforceRateLimits=False`, exceed draw/animate limits, confirm requests still run and logs emit `rate_limit_shadow`.
2. Set `enforceRateLimits=True`, exceed limits again, confirm requests are blocked and usage status is `rate_limited`.
3. Validate `%flag`, `%flagged`, `%unflag` flow blocks then restores command access.
4. Confirm `%animate`/`%video` require `llm.animate` capability.

## Development

### Adding New Commands

1. Add command method to `plugin.py`:
```python
def mycommand(self, irc, msg, args, text):
    """<args>

    Help text here.
    """
    # Skip ZNC playback messages
    if self._is_old_message(msg):
        return

    # Your logic here
    irc.reply("Response")

mycommand = wrap(mycommand, ["text"])
```

2. Add configuration to `config.py` if needed

3. Add tests to `tests/`

### Code Style

- Use Ruff for linting and formatting
- Use ty for type checking
- All functions must have type hints
- All public methods must have docstrings

## License

See LICENSE file for details.
