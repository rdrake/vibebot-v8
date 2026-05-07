# Plugins

VibeBot v8 is a [Limnoria](https://docs.limnoria.net/) workspace that
ships three first-party plugins. Each is independently loadable via
`@load <name>` from IRC.

| Plugin | What it does |
|--------|--------------|
| [LLM](llm.md) | The main AI surface: `@ask`, `@code`, `@draw`, memory, reminders, scheduled tasks, the assistant tool surface, and the Limnoria-as-tool bridge. |
| [RPG](rpg.md) | A lightweight filesystem-themed roleplay game with rooms, combat, dice, and an optional LLM narrator. |
| [NickInMiddle](nickinmiddle.md) | Tiny `inFilter` plugin that recognises the bot's nick when it appears mid-sentence so Limnoria's normal addressing logic fires. |

All three live in the same uv workspace (`plugins/<name>/`). They share
linting, type-checking, and the 93% test coverage floor.
