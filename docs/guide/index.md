# VibeBot v8

AI on IRC. Ask questions, generate code, create images, and hold conversations with large language models -- all without leaving your IRC client.

```
<rdrake> @ask What causes the northern lights?
<VibeBot> The northern lights (aurora borealis) happen when charged particles
          from the sun collide with gases in Earth's atmosphere. The particles
          follow magnetic field lines toward the poles, exciting nitrogen and
          oxygen molecules that release photons as they return to their ground
          state. Green comes from oxygen, purple and blue from nitrogen.
<rdrake> @ask How far south can you see them?
<VibeBot> During strong solar storms, auroras have been visible as far south
          as Texas and Florida (around 30°N). Typically you need to be above
          50°N for regular sightings -- places like northern Canada, Iceland,
          Scandinavia, and Alaska.
```

## Features

- **Natural language interaction** -- Just mention the bot by name or send it a PM. No commands needed -- ask questions, manage reminders, check usage, and more through plain conversation.
- **Conversation memory** -- Follow up on previous questions. The bot tracks context per user and channel.
- **Vision** -- Drop an image URL into `@ask` and the bot will describe or reason about it.
- **Code generation** -- `@code` produces syntax-highlighted output served as an HTTP link, keeping IRC clean.
- **Image generation** -- `@draw` creates images from text descriptions via Vertex AI Imagen.
- **[Scheduled tasks](user/scheduled-tasks.md)** -- Recurring or one-shot LLM runs with full tool access at run time, distinct from plain reminders.
- **Stored facts** -- Save things about yourself with `@memories` that persist across sessions, with [two-stage promotion](operator/memory-promotion.md) so casual remarks don't become memories.
- **[Spontaneous participation](operator/spontaneous.md)** -- Per-channel opt-in for the bot to chime in unprompted, with a tunable cadence.
- **[Forest mode](operator/forest-mode.md)** -- Per-user opt-in for long-form replies that bypass the channel reply cap and channel persona.
- **[Limnoria bridge](reference/bridge-tools.md)** -- The LLM can call stock Limnoria plugin commands (Time, Math, Seen, Web, Karma, RSS, DDG, and so on) as tools, with read-only by default and mutations gated.
- **Custom instructions** -- Shape how the bot responds to you with `@instruct`.
- **Multi-provider AI** -- Supports OpenAI, Anthropic, Google Gemini, and xAI through [LiteLLM](https://github.com/BerriAI/litellm).

## Documentation

**[User Guide](user/getting-started.md)** -- Start here. Learn the commands and how to get the most out of the bot.

**[Operator Guide](operator/installation.md)** -- Deploy and configure VibeBot for your network.

**[Command Reference](reference/commands.md)** -- Every command, its syntax, and examples.

## Links

- [GitHub](https://github.com/rdrake/vibebot-v8)
- [Limnoria](https://github.com/ProgVal/Limnoria)
- [LiteLLM](https://github.com/BerriAI/litellm)
