# Contributing to VibeBot v8

## Getting Started

### Prerequisites

Install [uv](https://github.com/astral-sh/uv) (fast Python package manager):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository
2. Install dependencies: `make install`
3. Install git hooks: `make install-hooks`

## Git Hooks

This project uses [prek](https://github.com/j178/prek) (a fast Rust-based pre-commit replacement) to catch issues before they hit the repo. Install them with:

```bash
make install-hooks
```

The hooks check:

- **Secrets** (gitleaks) - blocks API keys, passwords, tokens
- **Lint** (ruff) - catches code issues and auto-fixes where possible
- **Format** (ruff) - ensures consistent style
- **Dependencies** (deptry) - detects unused/missing dependencies
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

### Creating a PR

1. Create a feature branch from `main`
2. Make your changes
3. Ensure `make check` passes
4. Submit a PR with a clear description of changes
5. Label your PR appropriately (see below)

### Required CI Checks

All checks must pass before merge:

- `check (3.12)` - lint, typecheck, test on Python 3.12
- `check (3.13)` - lint, typecheck, test on Python 3.13
- `check (3.14)` - lint, typecheck, test on Python 3.14
- `secrets` - gitleaks secret scanning

### PR Labels

Label PRs for automatic changelog generation:

| Label | Changelog Category |
|-------|-------------------|
| `enhancement`, `feature` | 🚀 Features |
| `bug`, `fix` | 🐛 Bug Fixes |
| `chore`, `dependencies`, `documentation` | 🧹 Maintenance |

### Branch Rules

- All changes require a PR (no direct pushes to `main`)
- Branch must be up-to-date before merge
- CI must pass before merge

## Releases

Releases are created via GitHub and trigger Docker image builds.

### Creating a Release

```bash
gh release create v1.0.0 --generate-notes
```

GitHub auto-generates the changelog grouped by PR labels. Publishing a release triggers CI, which pushes Docker images with version tags (e.g., `:v1.0.0`, `:1.0`).

## Dependencies

- Python 3.12+ required
- Uses [uv](https://github.com/astral-sh/uv) for dependency management
- Uses [prek](https://github.com/j178/prek) for git hooks (fast Rust-based pre-commit replacement)
