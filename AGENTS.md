# AGENTS.md: instructions for coding agents

This file is the canonical repository guidance for coding agents. `CLAUDE.md` points here.

## Project summary

**VibeBot v8** is a Limnoria-based IRC bot workspace with AI features powered by LiteLLM.

Workspace members:

- `plugins/llm/`: the main LLM plugin (assistant, verse, scheduler, persistence)
- `plugins/nickinmiddle/`: companion `inFilter` plugin that moves a mid-message bot nick to the front, so Limnoria's normal addressing logic fires

Core documentation and tools:

- `README.md`: project overview and operator quick start
- `docs/guide/`: MkDocs user and operator guide (published to GitHub Pages)
- `Makefile`: standard development entry points
- `pyproject.toml`: workspace, Ruff, pytest, coverage, and ty configuration

The bot's command prefix on AfterNet is `@` (for example `@ask`). The `%` prefix in the repo's `bot.conf` is a development artifact; docs use `@`.

## Working in this repo

- Prefer `make` targets over raw tool commands when a matching target exists.
- After editing Python files, run `make lint` and `make typecheck`.
- Before considering work complete, run `make preflight`.
- If `make preflight` is too expensive for a narrow change, run the most specific checks that cover the edited area and state what you did not run.
- Use `make test` for the normal suite and `make test-all` when slow tests are relevant.
- Run `make syntax-check` when changes could affect Python 3.12 to 3.14 compatibility.
- Run `make docs` when changing MkDocs content or navigation, and `vale <files>` on changed prose.

Key targets:

```bash
make install
make run
make lint
make typecheck
make test
make preflight
make docs
```

## Architecture

For `plugins/llm/src/llm/`:

- `plugin.py`: IRC protocol layer, command routing, and command wrappers
- `service.py`: LiteLLM integration and business logic
- `assistant.py`: tool-using chat profile and tool wrappers
- `executor.py`: `LLMExecutor` (`BoundedSemaphore` + `ThreadPoolExecutor`); every blocking LLM call goes through `permit()` or `submit()`
- `persistence.py`: SQLite store for memories, reminders, scheduled tasks, and usage; uses the `_write_txn` context manager for atomic writes
- `limnoria_bridge.py`: allowlisted Limnoria-as-tool surface (mutation gating plus a curated default allowlist)
- `config.py`: Limnoria registry configuration (models, API keys, prompts, context, memory, rate limits, verse, bridge)
- `context.py`: conversation history and thread-safe state
- `tracing.py`: structured trace severity helpers
- `verse/`: the verse subsystem (SQLite world store, avatars, aging, compaction, reactions, taste command-line tools, validation, purge)

Keep those boundaries intact:

- IRC command parsing and reply flow belong in `plugin.py`.
- Provider calls, sanitization, and output shaping belong in `service.py`.
- Registry options belong in `config.py`.
- Shared mutable state must remain thread-safe because `Plugin.threaded = True`.
- All blocking LLM calls must go through `LLMExecutor` (no direct `litellm.*` from the IRC main thread).

## Security invariants

- Never log, echo, or persist API keys in plain text.
- Do not introduce environment-variable API key handling; this project uses Limnoria registry config.
- Scrub secrets from user-visible error messages.
- Treat channel topics, stored memory, reminders, and other recovered content as untrusted input.
- Check image and external URLs; block unsafe schemes such as `javascript:`, `data:`, and `file:`.
- Prevent path traversal with resolved-path checks.
- Preserve IRC command-injection defences for generated output.
- Keep HTML sanitization in place for rendered output.

## Testing expectations

- Add or update tests for behaviour changes.
- LLM plugin tests live under `plugins/llm/tests/`; nickinmiddle tests under `plugins/nickinmiddle/tests/`.
- Keep the **93%** coverage floor (enforced by `pyproject.toml` and `make test`).
- Follow existing test style, including behaviour-driven docstrings where that pattern already exists.
- Property-based tests (Hypothesis) live as `test_*_properties.py` alongside example tests; prefer extending a property test over adding a new example when the invariant generalizes.
- Include security and concurrency coverage when touching those areas.

## Common tasks

### Adding or changing IRC commands

1. Update `config.py` if the command needs new settings.
2. Add or change the command wrapper in `plugin.py`.
3. Adjust business logic in `service.py` if applicable.
4. Add or update tests.
5. Update `docs/guide/` if user-facing behaviour changes.

### Changing LLM behaviour

- System prompts live in `config.py` as registry-backed settings; verse prompt assembly lives in `verse/avatar.py`.
- Model selection is per-purpose: `assistantModel`, `codeModel`, `imageModel`, `searchModel`, `verseModel`, and `verseCompactionModel`. Bumping the chat model does not move verse; update both when that is the intent.
- API keys come from Limnoria config at runtime, not environment variables.

## Important files

- `plugins/llm/src/llm/service.py`: main LLM execution path
- `plugins/llm/src/llm/plugin.py`: command surface and IRC integration
- `plugins/llm/src/llm/context.py`: volatile memory implementation
- `plugins/llm/src/llm/limnoria_bridge.py`: Limnoria-to-LLM tool bridge
- `plugins/llm/src/llm/{service,plugin,assistant,persistence}.py`: native `schedule_llm_task` plus the unified `list_pending_tasks` / `cancel_pending_task` / `cancel_all_pending_tasks` tool surface
- `plugins/llm/src/llm/executor.py`: global LLM concurrency gate; tune via `supybot.plugins.LLM.maxConcurrentLLMCalls`
- `plugins/nickinmiddle/src/nickinmiddle/`: nick-in-middle `inFilter` plugin
- `mkdocs.yml` and `docs/guide/`: published guide source
- `docs/plans/`: design and implementation plans; shipped plans move to `docs/plans/archive/`

## Go rewrite (v9)

The Go rewrite lives in `go/` with its own Makefile: build and test with `cd go && make all`. Specs and plans live under `docs/superpowers/`. The Python tree in `plugins/` remains the production bot.
