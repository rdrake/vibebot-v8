# Branch Protection (optional)

VibeBot v8's CI/CD pipeline does not require GitHub branch protection to
function — the upstream repository (`rdrake/vibebot-v8`) is operated
single-maintainer and pushes to `main` are normal. This page is included
for **forks and team deployments** that do want a hard PR gate.

## Recommended ruleset

In **Settings → Branches**, add a ruleset (or classic rule) for the `main`
branch:

| Setting | Value |
|---------|-------|
| Require a pull request before merging | ✅ Enabled |
| Required approvals | 0 (solo) or 1+ (team) |
| Require status checks to pass before merging | ✅ Enabled |
| Require branches to be up to date before merging | ✅ Enabled |
| Do not allow bypassing the above settings | ✅ Enabled |

## Required status checks

Add these status checks after the first CI run on a PR (otherwise the
checks will not yet appear in the picker):

- `check (3.12)`
- `check (3.13)`
- `check (3.14)`
- `lint`
- `secrets`

> **Note:** The `docker` job (Docker build smoke test in `ci.yml`) is
> intentionally **not** required. It is a PR-only smoke build (`push:
> false`) and serves as an advisory canary. The real production image
> build lives in `docker.yml` and is gated on CI success separately. See
> [Why `docker` is advisory](#why-docker-is-advisory) below.

## Optional settings

| Setting | When to use |
|---------|-------------|
| Require conversation resolution | Team workflows where review comments must be addressed before merge. |
| Require signed commits | Hardened deployments; needs GPG keys for every contributor. |
| Require linear history | If you prefer rebase/squash merges over merge commits. |

## Why `docker` is advisory

The CI workflow defines four jobs for PRs: `lint`, `check` (matrix
3.12/3.13/3.14), `secrets`, and `docker`. The `docker` job is gated with
`if: github.event_name == 'pull_request'`, builds the image, but never
pushes it. The real production build/push is a separate workflow
(`docker.yml`) triggered by `workflow_run` after CI succeeds on main.

**Risk accepted:** a PR that breaks the Dockerfile can be merged. CI
passes (lint/test/secrets are fine), `docker.yml` then fails on main,
and the repo is on main with an unpushable image until a follow-up fix
lands. The previous image keeps serving production until a new one is
pushed.

**Why accept it:** Docker builds are slow and the Dockerfile changes
rarely. Making it a required check would add latency to every PR merge
for a rare, quickly recoverable failure mode.

## Verifying the rule

After configuring:

1. Try pushing a commit directly to `main` from a non-bypass account →
   should be rejected.
2. Open a PR with a deliberately failing test → merge button should be
   disabled.
3. Fix the test → merge button should re-enable once checks pass.
