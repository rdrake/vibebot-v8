# Plugins

VibeBot v8 is a [Limnoria](https://docs.limnoria.net/) workspace that
ships two first-party plugins. Each is independently loadable via
`@load <name>` from IRC.

| Plugin | What it does |
|--------|--------------|
| [LLM](llm.md) | The main AI surface: `@ask`, `@code`, `@draw`, memory, reminders, scheduled tasks, the assistant tool surface, and the Limnoria-as-tool bridge. |
| [NickInMiddle](nickinmiddle.md) | Tiny `inFilter` plugin that recognises the bot's nick when it appears mid-sentence so Limnoria's normal addressing logic fires. |

Both live in the same uv workspace (`plugins/<name>/`). They share
linting, type-checking, and the 93% test coverage floor.
