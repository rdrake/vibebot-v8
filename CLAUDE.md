# CLAUDE.md - AI Assistant Guidelines for VibeBot v8

This document provides comprehensive guidance for AI assistants working with the VibeBot v8 codebase.

## Project Overview

**VibeBot v8** is a modern IRC bot built on [Limnoria](https://github.com/ProgVal/Limnoria) with AI capabilities powered by [LiteLLM](https://github.com/BerriAI/litellm). It enables IRC users to interact with various AI models (OpenAI, Anthropic, Google Gemini, Vertex AI) directly from chat.

### Key Features
- Multi-provider AI through LiteLLM abstraction
- Volatile memory (conversation context) and non-volatile memory (stored facts)
- Vision support with automatic image URL detection
- Code generation with HTTP link support for long outputs
- Image generation via Vertex AI Imagen
- Internationalization (DE, FR, IT, FI, RU)

## Repository Structure

```
vibebot-v8/
├── plugins/llm/                    # Main LLM plugin (workspace member)
│   ├── src/llm/
│   │   ├── __init__.py             # Plugin exports: Class, configure
│   │   ├── plugin.py               # IRC protocol & command handlers
│   │   ├── service.py              # LiteLLM business logic
│   │   ├── config.py               # Registry configuration definitions
│   │   └── context.py              # Thread-safe conversation history
│   ├── locales/                    # i18n translations
│   ├── tests/                      # Comprehensive test suite (9 files)
│   └── pyproject.toml              # Plugin package config
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                  # CI pipeline (lint, typecheck, test)
│   │   └── docker.yml              # Docker build & push
│   └── dependabot.yml              # Automated dependency updates
│
├── docs/
│   ├── guide/                      # MkDocs source pages (user guide, operator guide, reference)
│   ├── reviews/                    # Code review documents
│   └── plans/                      # Design documents
│
├── mkdocs.yml                      # MkDocs Material configuration
├── .pre-commit-config.yaml         # Git hooks config
├── Dockerfile                      # Multi-stage Docker build
├── Makefile                        # Development commands
├── pyproject.toml                  # Root workspace config
└── vibebot.service                 # systemd user service
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ (targets 3.14) |
| Package Manager | [uv](https://github.com/astral-sh/uv) |
| IRC Framework | Limnoria >=2023.1.20 |
| LLM Provider | LiteLLM >=1.55.0 |
| Linter/Formatter | Ruff >=0.14.9 |
| Type Checker | ty (Astral) |
| Testing | pytest with pytest-cov |
| HTML Sanitization | nh3 |
| Documentation | MkDocs Material |

## Quality Gates

- After editing Python files, run `make lint` and `make typecheck` to catch issues immediately (automated via Claude Code hooks in `.claude/settings.json`)
- Before claiming any task is complete, run `make preflight` (auto-format + lint + typecheck + tests)
- Fix type checker warnings properly rather than adding inline suppression comments, unless the warning is a confirmed false positive from Limnoria's dynamic attributes (e.g., `supybot.ircdb.users`)
- Complete the full cycle for every task: implement → verify (`make preflight`) → commit
- If a task is too large for one session, commit progress at logical boundaries rather than leaving work half-done

## Development Commands

All development is done through the Makefile:

```bash
# Setup
make install              # Install dependencies with uv
make install-hooks        # Install pre-commit hooks

# Development
make run                  # Start bot: uv run limnoria bot.conf
make test                 # Run pytest with coverage (80% threshold)
make lint                 # Check code with ruff
make format               # Auto-format with ruff
make typecheck            # Check types with ty
make check                # Run lint + format-check + typecheck + test
make preflight            # Auto-format then run all checks (use this before committing)

# Documentation
make docs                 # Build MkDocs site (strict mode, zero warnings)
make docs-serve           # Serve docs locally with live reload

# CI/Quality
make ci                   # Full CI pipeline (sync --locked, pre-commit, test)
make pre-commit           # Run all pre-commit hooks

# Worktree workflow
make worktree-create BRANCH=fix/my-fix   # Create isolated worktree with deps installed
make worktree-remove BRANCH=fix/my-fix   # Remove worktree and delete branch

# GitHub helpers
make wait-ci              # Watch current GitHub Actions run until completion
make rebase-pr PR=42      # Ask dependabot to rebase a PR

# Cleanup
make clean                # Remove cache files
make deep-clean           # Remove venv and uv cache
```

## Code Quality Requirements

### Git Hooks (enforced)

**Pre-commit (on every commit):**
1. **gitleaks** - Blocks commits containing secrets/API keys
2. **ruff** - Linting with auto-fix
3. **ruff-format** - Code formatting
4. **ty** - Type checking on `plugins/llm/src/`
5. **Standard hooks** - Merge conflicts, large files, trailing whitespace

**Pre-push (on every push):**
6. **full-check** - Runs `make check` (lint + format-check + typecheck + test)

### Ruff Configuration
- Target: Python 3.14
- Line length: 100 characters
- Quote style: Double quotes
- Rules enabled: E, W, F, I, N, UP, B, C4, SIM

### Test Coverage
- Minimum threshold: **80%**
- Run with: `make test`
- Tests located in: `plugins/llm/tests/`

## Architecture

### Separation of Concerns

```
plugin.py  → IRC protocol layer (command routing, message handling)
service.py → Business logic layer (LiteLLM calls, sanitization, file handling)
context.py → Data layer (conversation history storage, thread-safety)
config.py  → Configuration registry (Limnoria config definitions)
```

### Key Design Patterns

1. **Thread Safety**: `Plugin.threaded = True` runs commands in worker threads; `ConversationContext` uses `threading.Lock()` for safe access

2. **Type Hints**: Full PEP 484 typing with `TYPE_CHECKING` imports for runtime-heavy types:
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg
```

3. **NamedTuples for Results**:
```python
class ValidationResult(NamedTuple):
    is_valid: bool
    error: str = ""
```

## Security Patterns (Critical)

### API Key Handling
- **Never** store API keys in environment variables
- **Never** log API keys in plain text
- Pass keys directly to `litellm.completion()` via `api_key=` parameter
- API keys are scrubbed from error messages before display to users

### Input Validation
- Image URLs must use HTTP/HTTPS (block `javascript:`, `data:`, `file:`)
- Path traversal blocked with `Path.resolve()` + `is_relative_to()`
- IRC command injection prevented (lines starting with `.` or `/` are prefixed with space)

### Output Sanitization
- HTML output sanitized with `nh3` (allowlist-based)
- API keys scrubbed from error messages before display
- LLM responses checked for IRC command injection

### Prompt Injection Defense
- System prompts use clear `INSTRUCTIONS` vs `CONTEXT` sections
- Channel topics treated as untrusted user data (moved to user message)
- Anti-injection preamble in system prompt

## IRC Commands

| Command | Description |
|---------|-------------|
| `%ask <question>` | Ask with context, vision, and optional instructions |
| `%code <request>` | Generate code with HTTP link output |
| `%draw <prompt>` | Generate image (account required) |
| `%forget [channel]` | Clear volatile memory (conversation context) |
| `%memories [subcommand]` | Manage non-volatile memory (stored facts) |
| `%instruct [text \| clear]` | Set persistent instructions for ask |
| `%remind [text \| list \| del \| clear]` | Set and manage reminders |
| `%usage [nick \| #channel]` | View API usage statistics |

## Testing Guidelines

### Test File Organization
- `test_plugin.py` - Plugin structure, commands, HTTP callback
- `test_service.py` - LiteLLM integration, sanitization
- `test_context.py` - Conversation context, thread-safety
- `test_integration.py` - Full command flows
- `test_etiquette.py` - Rate limiting, safety
- `test_html_output.py` - XSS prevention
- `test_stress.py` - Load testing, concurrency

### Test Conventions
- BDD-style docstrings: `"""GIVEN ... WHEN ... THEN ..."""`
- Use `unittest.mock` for Limnoria IRC objects
- Include thread-safety tests for shared state
- Security tests for injection attacks

## Configuration

Bot config lives in `bot.conf`. Models are set via `supybot.plugins.LLM.{ask,code,draw}Model` and API keys via `supybot.plugins.LLM.{ask,code,draw}ApiKey` (set at runtime with `%config`).

## Common Tasks

### Adding a New Command
1. Add config options in `config.py` (model, API key, system prompt)
2. Add command method in `plugin.py` with `@wrap` decorator
3. Add business logic in `service.py`
4. Add tests in appropriate test file
5. Update locales if adding user-facing strings

### Modifying LLM Behavior
- System prompts: `config.py` → `{command}SystemPrompt`
- Model selection: `config.py` → `{command}Model`
- API key: `config.py` → `{command}ApiKey`

### Running Locally
```bash
make install
make install-hooks
# Create bot.conf with Limnoria configuration
make run
```

## Standard Workflows

### Pre-commit Quality Check
Always use `make preflight` instead of running `make format` and `make check` separately. It auto-formats first, then runs all checks in one pass:
```bash
make preflight            # format + lint + format-check + typecheck + test
```

### Branch PR Workflow (with worktrees)
Use worktrees to isolate branch work from your main checkout:
```bash
make worktree-create BRANCH=fix/my-fix
cd .worktrees/fix/my-fix
# ... make changes, run make preflight, commit, push ...
gh pr create --title "Fix: description" --body "..."
# After merge:
cd ../..
make worktree-remove BRANCH=fix/my-fix
git pull
```

### Waiting for CI
Instead of polling with `sleep` and `gh pr view`, use:
```bash
make wait-ci              # blocks until the current run completes or fails
```

### Managing Dependabot PRs
To trigger a rebase on a dependabot PR:
```bash
make rebase-pr PR=42
make wait-ci              # wait for CI to pass after rebase
gh pr merge 42 --merge    # merge once green
```

## Important Files to Know

| File | Purpose |
|------|---------|
| `plugins/llm/src/llm/service.py` | Core LLM logic, start here for AI-related changes |
| `plugins/llm/src/llm/plugin.py` | IRC command handlers |
| `plugins/llm/src/llm/context.py` | Volatile memory (conversation context) management |
| `pyproject.toml` | Root config, ruff/pytest settings |
| `Makefile` | All development commands |

## Do's and Don'ts

### Do
- Run `make preflight` before committing (auto-formats, then runs all checks)
- Add tests for new functionality (maintain 80%+ coverage)
- Use type hints for all function signatures
- Follow existing patterns in `service.py` and `plugin.py`
- Use `NamedTuple` for structured return values
- Sanitize all user input before processing
- Use `threading.Lock()` for shared mutable state

### Don't
- Store API keys in environment variables
- Log sensitive information (API keys, user messages)
- Use `os.environ` for configuration (use Limnoria's registry)
- Skip pre-commit hooks (`--no-verify`)
- Add dependencies without updating `plugins/llm/pyproject.toml`
- Introduce breaking changes to IRC command syntax

## Deployment

See `Makefile` targets: `make docker-build`, `make docker-run`, `make install-service`, `make install-timer`.

## Production Debugging

```bash
ssh vibebot@rdrake.org "tail -100 ~/vibebot-v8/logs/messages.log"   # recent logs
ssh vibebot@rdrake.org "tail -f ~/vibebot-v8/logs/messages.log"     # follow logs
ssh vibebot@rdrake.org "cd ~/vibebot-v8 && git pull && systemctl --user restart vibebot"  # restart
```
