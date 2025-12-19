# LLM Plugin for Limnoria

AI-powered IRC commands using LiteLLM.

## Features

- Multi-provider support (OpenAI, Anthropic, Google, etc.)
- Vision support with automatic image URL detection
- Conversation context (memory between messages)
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
