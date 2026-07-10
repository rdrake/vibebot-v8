# Contributing to VibeBot v8

## Getting started

### Prerequisites

Install [uv](https://github.com/astral-sh/uv), the Python package manager this project uses:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. Clone the repository.
2. Install dependencies: `make install`
3. Install git hooks: `make install-hooks`

## Git hooks

This project uses [prek](https://github.com/j178/prek), a fast Rust-based pre-commit runner, to catch issues before they reach the repo. `make install-hooks` installs them.

The hooks check:

- **Secrets** (gitleaks): blocks API keys, passwords, and tokens
- **Lint** (ruff): catches code issues and auto-fixes where possible
- **Format** (ruff): keeps style consistent
- **Types** (ty): catches type errors
- **Hygiene**: merge-conflict markers, oversize files, trailing whitespace

If a check fails, fix the issue and commit again. Run `make pre-commit` to run all hooks on all files.

## Code quality

Run the full gate before calling any change done:

```bash
make preflight  # format, then lint + format-check + typecheck + syntax-check + test
```

Individual checks:

```bash
make lint         # ruff check
make format       # ruff format
make typecheck    # ty
make syntax-check # Python 3.12 to 3.14 compatibility
make test         # pytest, skipping slow tests
make test-all     # full suite, including slow tests
```

### Coverage

Tests must keep at least **93%** branch coverage; `make test` enforces the floor. For a detailed report:

```bash
uv run pytest plugins/llm/tests/ --cov --cov-report=term-missing
```

### Docs

The published guide builds from `docs/guide/` with MkDocs (`make docs`). Prose lints with [Vale](https://vale.sh/): run `vale docs/guide README.md` and keep new content free of errors and warnings. Project terms live in `styles/config/vocabularies/VibeBot/accept.txt`.

## Commit messages

Write [conventional commits](https://www.conventionalcommits.org/): `type(scope): summary`. git-cliff generates the changelog from them, grouping by type:

| Prefix | Changelog section |
|--------|-------------------|
| `feat` | Features |
| `fix` | Bug fixes |
| `perf` | Performance |
| `refactor` | Refactor |
| `docs` | Documentation |
| `test` | Tests |
| `ci`, `build`, `deps`, `chore` | Maintenance |

## Branches and pull requests

The maintainer commits to `main` directly. Outside contributions arrive as pull requests from feature branches, and CI must pass before merge:

- `lint`: prek hooks plus the syntax check
- `check (3.12 / 3.13 / 3.14)`: the full test suite on each supported Python
- `secrets`: gitleaks scanning

Pushes to `main` that pass CI build and publish the Docker image, then restart production automatically. Treat every push to `main` as a deploy.

## Releases

Push a semver tag to cut a release:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The release workflow generates notes with git-cliff, creates the GitHub release, and opens a PR that refreshes `CHANGELOG.md` on `main`. Run `make changelog` to preview the changelog locally.

## Dependencies

- Python 3.12 or newer (CI tests 3.12, 3.13, and 3.14)
- [uv](https://github.com/astral-sh/uv) manages the workspace and lockfile
- Dependabot proposes weekly updates; fold lockfile bumps into a single `uv lock` rather than merging per-package PRs
