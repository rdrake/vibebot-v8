# LLM plugin

The LLM plugin is VibeBot's main AI surface. It provides
natural-language chat, code and image generation, illustrated stories,
persistent memory, and reminders. It also runs scheduled agentic
tasks, hosts a persistent roleplay world (the verse), and carries a
curated bridge that lets the model call other Limnoria plugins as
tools.

## Command families

| Family | Commands | Notes |
|--------|----------|-------|
| Core AI | `@ask`, `@code`, `@draw`, `@story` | Chat with vision, code with HTTP links, images, illustrated pages. |
| Memory | `@forget`, `@memories`, `@instruct` | Volatile context, durable facts, persistent instructions. |
| Reminders | `@remind` | Natural-language reminders with recurring support. |
| Accounting | `@usage` | Per-account and per-channel API usage. |
| Verse (user) | `@verseopt`, `@verse`, `@look`, `@who`, `@avatar` | Opt in, inspect the scene, set a persona. |
| Verse (editor) | `@canon`, `@versedit` | Curate durable canon. Requires `llm.verse.edit`. |
| Verse (GM) | `@versedump`, `@versepurge`, `@versecompact` | Inspect, wipe, and compact a channel's verse. Requires `llm.verse.gm`. |

See the [command reference](../reference/commands.md) for full syntax
and capability requirements, and the
[configuration guide](../operator/configuration.md) for every registry
key.

Most features also work through plain language: mention the bot by
name or send it a PM, and the assistant picks the right tool itself.

## Capabilities

Limnoria capabilities gate each command family: `llm.ask`, `llm.code`,
`llm.draw` (also covers `@story`), `llm.verse`, `llm.verse.edit`, and
`llm.verse.gm`. Owner and admin accounts bypass rate limits.

## Internal layout

```
plugins/llm/src/llm/
├── plugin.py          # IRC command surface + Limnoria glue
├── service.py         # LiteLLM calls, sanitization, output shaping
├── assistant.py       # Tool-using chat profile (function calling)
├── executor.py        # LLMExecutor: global concurrency cap
├── persistence.py     # SQLite store (memories, reminders, schedules, usage)
├── limnoria_bridge.py # Allowlisted "Limnoria as tools" surface
├── context.py         # Conversation history with TTL
├── config.py          # Limnoria registry options
├── tracing.py         # Structured trace severity helpers
└── verse/             # Verse store, avatar engine, compaction, taste tools
```

## Concurrency model

Every blocking LLM call goes through `LLMExecutor` (in `executor.py`),
which combines a `BoundedSemaphore` and a `ThreadPoolExecutor`. This
design:

- caps total in-flight LLM I/O with
  `supybot.plugins.LLM.maxConcurrentLLMCalls` (default 16),
- keeps the IRC main thread responsive, because no `litellm.*` call can
  block it directly, and
- shares one shutdown gate, so plugin reloads drain in-flight work
  cleanly.

`@usage` exposes the live counts (`running/queued/max`) for tuning.

## Storage

Persistent state lives in a single SQLite database managed by
`persistence.py`. All writes go through the `_write_txn` context
manager, so a failure mid-update rolls back rather than leaving partial
state. Tables cover memories, reminders, scheduled tasks, conversation
snapshots, and per-account usage. Each verse-enabled channel gets its
own SQLite store under `data/verse/`.

## See also

- [Memory and instructions](../user/memory.md)
- [Reminders and usage](../user/reminders-usage.md)
- [Scheduled tasks](../user/scheduled-tasks.md)
- [Bridge tools](../reference/bridge-tools.md)
- [Verse operations](../operator/forest-verse.md)
- [Tuning and monitoring](../operator/tuning-monitoring.md)
