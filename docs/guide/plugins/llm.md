# LLM plugin

The LLM plugin is VibeBot's main AI surface. It provides natural-language
chat, code generation, image generation, persistent memory, reminders,
scheduled agentic tasks, and a curated bridge that lets the model invoke
other Limnoria plugins as tools.

## Surface

User-facing commands:

- `@ask <prompt>` — main chat surface, vision-capable, with conversation history
- `@code <prompt>` — code generation with HTTP link output for long snippets
- `@draw <prompt>` — image generation (account required)
- `@forget [channel]` — clear volatile conversation memory
- `@memories` — manage non-volatile per-user facts
- `@instruct` — set persistent per-user instructions for `@ask`
- `@remind` — natural-language reminders
- `@usage` — per-account usage statistics

See the [Command Reference](../reference/commands.md) for full syntax.

## Internal layout

```
plugins/llm/src/llm/
├── plugin.py          # IRC command surface + Limnoria glue
├── service.py         # LiteLLM calls, sanitization, output shaping
├── assistant.py       # Tool-using chat profile (function calling)
├── executor.py        # LLMExecutor — global concurrency cap
├── persistence.py     # SQLite store (memories, reminders, schedules, usage)
├── limnoria_bridge.py # Allowlisted "Limnoria as tools" surface
├── context.py         # Conversation history with TTL
├── config.py          # Limnoria registry options
└── tracing.py         # Structured trace severity helpers
```

## Concurrency model

Every blocking LLM call goes through `LLMExecutor` (in `executor.py`),
which combines a `BoundedSemaphore` and a `ThreadPoolExecutor`. This:

- Caps total in-flight LLM I/O via
  `supybot.plugins.LLM.maxConcurrentLLMCalls` (default: see registry).
- Keeps the IRC main thread responsive — no `litellm.*` is allowed to
  block it directly.
- Shares one shutdown gate so plugin reloads cleanly drain in-flight work.

`@usage` exposes the live counts (`running/queued/max`) for tuning.

## Storage

Persistent state lives in a single SQLite database managed by
`persistence.py`. All writes go through the `_write_txn` context manager
so a failure mid-update rolls back rather than leaving partial state.
Tables cover memories, reminders, scheduled tasks, conversation
snapshots (for migrations), and per-account usage.

## See also

- [Memory & Instructions](../user/memory.md)
- [Reminders & Usage](../user/reminders-usage.md)
- [Scheduled Tasks](../user/scheduled-tasks.md)
- [Bridge Tools](../reference/bridge-tools.md)
- [Tuning & Monitoring](../operator/tuning-monitoring.md)
