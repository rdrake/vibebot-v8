# AGENTS.md - Codex Instructions for VibeBot v8

This file contains repository-specific guidance for Codex and other coding agents.

## Project Summary

**VibeBot v8** is a Limnoria-based IRC bot workspace with AI features powered by LiteLLM.

Primary workspace members:
- `plugins/llm/` - main LLM plugin
- `plugins/rpg/` - RPG plugin

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
- `config.py` - Limnoria registry configuration definitions
- `context.py` - conversation history and thread-safe state

Keep those boundaries intact:
- IRC command parsing and reply flow belong in `plugin.py`.
- Provider calls, sanitization, and output shaping belong in `service.py`.
- Registry options belong in `config.py`.
- Shared mutable state must remain thread-safe because `Plugin.threaded = True`.

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
- RPG plugin tests live under `plugins/rpg/tests/`.
- Maintain the existing 80% coverage floor.
- Follow existing test style, including BDD-style docstrings where that pattern is already used.
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
- `plugins/llm/src/llm/limnoria_bridge.py` - Limnoria → LLM tool bridge (Phase 1; see docs/plans/2026-05-02-limnoria-tool-bridge-plan.md)
- `plugins/rpg/src/rpg/` - RPG plugin implementation
- `README.md` - setup and operator-facing documentation
- `mkdocs.yml` and `docs/guide/` - published guide source

## Legacy Claude Files

This repository still contains `.claude/` settings for Claude Code users. Codex should treat `AGENTS.md` as the canonical project instruction file.
