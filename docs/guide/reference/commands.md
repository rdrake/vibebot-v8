# Command Reference

Quick reference for all VibeBot commands. For detailed usage and examples, see the [User Guide](../user/getting-started.md).

## Natural language

Most bot features are available through natural language. Mention the bot by name in a channel or send it a PM:

```
VibeBot, what's the weather like on Mars?
VibeBot, remind me in 2 hours to deploy
VibeBot, what do you remember about me?
VibeBot, how much have I used this month?
VibeBot, set my instruction to respond in haiku
```

The bot uses tools internally to handle your request -- managing memories, setting reminders, checking usage, and more. Natural language is especially useful for combining multiple actions in a single message.

Use the following commands when you want direct, predictable behavior.

## Commands at a glance

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@ask` | `<question>` | Ask the AI a question |
| `@code` | `<request>` | Generate code |
| `@draw` | `<prompt>` | Generate an image |
| `@forget` | `[channel]` | Clear conversation context |
| `@memories` | `[del <id> \| edit <id> <text> \| clear \| cleanup]` | Manage stored facts |
| `@instruct` | `[<instruction> \| clear]` | Set persistent instructions |
| `@remind` | `[<text> \| list \| del <id> \| clear]` | Set and manage reminders |
| `@usage` | `[nick \| #channel]` | Show API usage stats |

## Command details

### ask

Ask the AI a question. Supports conversation context (follow-up questions) and vision (include image URLs).

```
@ask What is the capital of France?
@ask Describe this: https://example.com/image.jpg
@ask And what about Germany?
```

See [AI Commands -- ask](../user/ai-commands.md#ask) for full details.

### code

Generate code based on your request. Code is saved to an HTTP link with syntax highlighting.

```
@code Python function to calculate fibonacci numbers
@code Now add memoization to that
```

See [AI Commands -- code](../user/ai-commands.md#code) for full details.

### draw

Generate an image from a text description. Requires an authenticated account.

```
@draw A sunset over mountains in watercolor style
@draw A cyberpunk cityscape at night
```

See [AI Commands -- draw](../user/ai-commands.md#draw) for full details.

### forget

Clear your volatile memory (conversation context) for the current or specified channel.

```
@forget
@forget #channel
```

See [Memory -- Volatile context](../user/memory.md#volatile-context) for full details.

### memories

Manage your non-volatile memory (stored facts the bot remembers about you across conversations).

```
@memories
@memories delete 3
@memories edit 5 corrected fact
@memories clear
```

See [Memory -- Non-volatile memory](../user/memory.md#non-volatile-memory) for full details.

### instruct

Set persistent instructions that shape how `@ask` responds to you. Your instruction is prepended to the system prompt.

```
@instruct You are Captain Picard. Respond in character.
@instruct Respond only in haiku
@instruct clear
@instruct
```

See [Memory -- Custom instructions](../user/memory.md#custom-instructions) for full details.

### remind

Set and manage reminders using natural language.

```
@remind in 30 minutes check the build
@remind list
@remind delete abc1
@remind clear
```

See [Reminders & Usage -- remind](../user/reminders-usage.md#remind) for full details.

### usage

Show API usage statistics for yourself, another user, or a channel.

```
@usage
@usage someone
@usage #channel
```

See [Reminders & Usage -- usage](../user/reminders-usage.md#usage) for full details.

## Features

- **Natural language interaction** -- Mention the bot by name or send a PM to ask questions, manage memories, set reminders, and more without commands.
- **Volatile memory** -- The bot remembers your recent conversation for follow-up questions. Context is per-user, per-channel, and expires after a period of inactivity.
- **Non-volatile memory** -- Store facts about yourself that persist across conversations and sessions.
- **Vision** -- Include image URLs in your `@ask` messages and the bot will describe or reason about them.
- **Syntax-highlighted code** -- `@code` responses are served as HTTP links with syntax highlighting, keeping IRC clean.
- **Spontaneous participation** -- When enabled by the operator, the bot can join channel conversations naturally without being addressed directly.
- **Multi-provider AI** -- Powered by LiteLLM, supporting OpenAI, Anthropic, Google Gemini, and Vertex AI models behind a unified interface.
