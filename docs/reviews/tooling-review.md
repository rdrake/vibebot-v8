# Tooling Review

## Scope
- Pre-commit hooks (`.pre-commit-config.yaml`)
- Ruff lint/format (`pyproject.toml`)
- Type checking with ty (`pyproject.toml`)
- Tests/coverage (`pyproject.toml`, `Makefile`)
- GitHub Actions (`.github/workflows/ci.yml`, `.github/workflows/docker.yml`)
- Dependabot (`.github/dependabot.yml`)

## What is working well
- Good local hygiene via pre-commit (secrets, lint, format, types, whitespace, merge markers).
- CI mirrors local checks and uses `uv sync --locked`, which keeps builds reproducible.
- Coverage tracking is configured and tests are scoped to the plugin.
- Dependabot covers Actions and uv dependencies.
- Docker build/push pipeline is solid for releases.

## Gaps and risks
- CI does not run the pre-commit-only checks (whitespace, merge markers, large files).
- Coverage enforcement may be implicit; `fail_under` is set but not explicitly enforced in CI.
- CI and Makefile duplicate command logic (risk of drift as tooling evolves).
- Python 3.14 is pinned directly in CI and config, which is fine today but not set up for future 3.15+
  expansion.

## Recommended changes
- Consolidate CI to call a single Makefile target (e.g., `make check` or a new `make ci`)
  so command logic lives in one place and drift risk is minimized.
- Consider making that target run `pre-commit` plus tests only, and keep `lint/format/typecheck`
  as convenience targets for local use. This avoids CI running the same checks twice.
- Add a CI matrix for `python-version` with a single entry (3.14) today, so extending to 3.15+
  later is a one-line change. Keep `tool.ruff.target-version` and `tool.ty.environment` at the
  minimum supported version to preserve compatibility.
- Make coverage enforcement explicit in CI with `--cov-fail-under=80` so it is deterministic.
- Add a PR-only Docker build job (no push) to catch Dockerfile breakage before merges.

## Notes on Python versioning
- You do not need to support pre-3.14. Keep `requires-python = ">=3.14"`.
- When you decide to add 3.15, add it to the CI matrix first; only raise the minimum version
  when you are ready to drop 3.14.
