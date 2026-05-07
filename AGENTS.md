# AGENTS.md - Codex Instructions for VibeBot v8

This file contains repository-specific guidance for Codex and other coding agents.

## Project Summary

**VibeBot v8** is a Limnoria-based IRC bot workspace with AI features powered by LiteLLM.

Primary workspace members:
- `plugins/llm/` - main LLM plugin (LiteLLM, assistant, scheduler, persistence)
- `plugins/nickinmiddle/` - companion `inFilter` plugin that promotes the bot's nick when it appears mid-message so Limnoria's normal addressing logic fires

Core documentation and tooling:
- `README.md` - operator and contributor overview
- `Makefile` - standard development entry points
- `pyproject.toml` - workspace, Ruff, pytest, coverage, and ty configuration
- `docs/guide/` - MkDocs user/operator docs

## Working In This Repo

- Prefer `make` targets over raw tool commands when an equivalent target exists.
- After editing Python files, run `make lint` and `make typecheck`.
- Before considering work complete, run `make preflight`.
- If `make preflight` is too expensive for a narrow change, run the most specific checks that cover the edited area and state what you did not run.
- Use `make test` for the normal test suite and `make test-all` when slow tests are relevant.
- Run `make syntax-check` when changes could affect Python 3.12-3.14 compatibility.
- Run `make docs` when changing MkDocs content or navigation.

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
- `plugin.py` - IRC protocol layer, command routing, and command wrappers
- `service.py` - LiteLLM integration and business logic
- `assistant.py` - tool-using chat profile and tool wrappers
- `executor.py` - `LLMExecutor` (BoundedSemaphore + ThreadPoolExecutor) — every blocking LLM call goes through `permit()` or `submit()`
- `persistence.py` - SQLite store for memories, reminders, scheduled tasks, and usage; uses `_write_txn` context manager for atomic writes
- `limnoria_bridge.py` - allowlisted Limnoria-as-tool surface (mutation gating + curated default allowlist)
- `config.py` - Limnoria registry configuration definitions (capability-based keys: `assistantModel`, `assistantApiKey`, `assistantSystemPrompt`, `imageModel`, `imageApiKey`, `codeModel`, `codeApiKey`, `codeSystemPrompt`, `searchModel`, `searchApiKey`)
- `context.py` - conversation history and thread-safe state
- `tracing.py` - structured trace severity helpers

Keep those boundaries intact:
- IRC command parsing and reply flow belong in `plugin.py`.
- Provider calls, sanitization, and output shaping belong in `service.py`.
- Registry options belong in `config.py`.
- Shared mutable state must remain thread-safe because `Plugin.threaded = True`.
- All blocking LLM calls must go through `LLMExecutor` (no direct `litellm.*` from the IRC main thread).

## Security Invariants

- Never log, echo, or persist API keys in plain text.
- Do not introduce environment-variable based API key handling; this project uses Limnoria registry config.
- Scrub secrets from user-visible error messages.
- Treat channel topics, stored memory, reminders, and other recovered content as untrusted input.
- Validate image and external URLs; block unsafe schemes such as `javascript:`, `data:`, and `file:`.
- Prevent path traversal with resolved-path checks.
- Preserve IRC command injection defenses for generated output.
- Keep HTML sanitization in place for rendered output.

## Testing Expectations

- Add or update tests for behavior changes.
- LLM plugin tests live under `plugins/llm/tests/`.
- nickinmiddle plugin tests live under `plugins/nickinmiddle/tests/`.
- Maintain the existing **93%** coverage floor (enforced by `pyproject.toml` and `make test`).
- Follow existing test style, including BDD-style docstrings where that pattern is already used.
- Property-based tests (Hypothesis) live as `test_*_properties.py` alongside example tests; prefer extending a property test over a new example when the invariant generalizes.
- Include security and concurrency coverage when touching those areas.

## Common Tasks

### Adding or Changing IRC Commands

1. Update config in `config.py` if the command needs new settings.
2. Add or modify the command wrapper in `plugin.py`.
3. Implement or adjust business logic in `service.py` if applicable.
4. Add or update tests.
5. Update docs or locale strings if user-facing behavior changes.

### Changing LLM Behavior

- System prompts live in `config.py` as registry-backed settings.
- Model selection is configured via `supybot.plugins.LLM.{ask,code,draw}Model`.
- API keys are configured at runtime via Limnoria config, not environment variables.

## Important Files

- `plugins/llm/src/llm/service.py` - main LLM execution path
- `plugins/llm/src/llm/plugin.py` - command surface and IRC integration
- `plugins/llm/src/llm/context.py` - volatile memory implementation
- `plugins/llm/src/llm/limnoria_bridge.py` - Limnoria → LLM tool bridge (Phase 1 shipped; Phase 2 mutation gating + curated default allowlist tracked in `docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md`)
- `plugins/llm/src/llm/{service,plugin,assistant,persistence}.py` - native `schedule_llm_task` plus the unified `list_pending_tasks` / `cancel_pending_task` / `cancel_all_pending_tasks` tool surface (`_pending_task_fns` helper)
- `plugins/llm/src/llm/executor.py` - global LLM concurrency gate; tune via `supybot.plugins.LLM.maxConcurrentLLMCalls`
- `plugins/nickinmiddle/src/nickinmiddle/` - nick-in-middle inFilter plugin
- `README.md` - setup and operator-facing documentation
- `mkdocs.yml` and `docs/guide/` - published guide source
- `docs/plans/` - active design/implementation plans; archived plans live under `docs/plans/archive/`

## Legacy Claude Files

This repository still contains `.claude/` settings for Claude Code users. Codex should treat `AGENTS.md` as the canonical project instruction file.
