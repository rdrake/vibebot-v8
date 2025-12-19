# Contributing to VibeBot v8

## Getting Started

1. Clone the repository
2. Install dependencies: `make install`
3. Install git hooks: `make install-hooks`

## Pre-commit Hooks

This project uses pre-commit hooks to catch issues before they hit the repo. Install them with:

```bash
make install-hooks
```

The hooks check:

- **Secrets** (gitleaks) - blocks API keys, passwords, tokens
- **Lint** (ruff) - catches code issues and auto-fixes where possible
- **Format** (ruff) - ensures consistent style
- **Types** (ty) - catches type errors
- **Merge conflicts** - blocks unresolved conflict markers
- **Large files** - blocks files over 500KB
- **Whitespace** - fixes trailing whitespace and end-of-file issues

If a check fails, fix the issue and try committing again. Run `make pre-commit` to manually run all hooks on all files.

## Code Quality

All code must pass before merging:

```bash
make check  # Runs lint, format-check, typecheck, and tests
```

Individual checks:

```bash
make lint        # Check for code issues
make format      # Auto-fix formatting
make typecheck   # Check types
make test        # Run tests
```

### Coverage

Tests must maintain ≥80% code coverage. Run with coverage report:

```bash
uv run pytest plugins/llm/tests/ --cov --cov-report=term-missing
```

## Pull Request Guidelines

1. Create a feature branch from `main`
2. Make your changes
3. Ensure `make check` passes
4. Submit a PR with a clear description of changes

## Dependencies

- Python 3.14+ required
- Uses [uv](https://github.com/astral-sh/uv) for dependency management
- Uses [pre-commit](https://pre-commit.com/) for git hooks (installed automatically)
