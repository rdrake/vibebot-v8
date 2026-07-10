# Branch protection (optional)

VibeBot's pipeline does not require GitHub branch protection. The upstream repository (`rdrake/vibebot-v8`) is single-maintainer; direct pushes to `main` are normal, and the CI, Docker build, and auto-deploy pipeline is the safety net. This page is for forks and team deployments that want a hard PR gate.

## Recommended ruleset

In **Settings → Branches**, add a ruleset for `main`:

| Setting | Value |
|---------|-------|
| Require a pull request before merging | Enabled |
| Required approvals | 0 (solo) or 1 or more (team) |
| Require status checks to pass before merging | Enabled |
| Require branches to be up to date before merging | Enabled |
| Do not allow bypassing the above settings | Enabled |

## Required status checks

Add these checks after the first CI run on a PR; they do not appear in the picker before then:

- `check (3.12)`
- `check (3.13)`
- `check (3.14)`
- `lint`
- `secrets`

The `docker` job in `ci.yml` is deliberately not required. That job is a PR-only smoke build that never pushes; the production image build lives in `docker.yml`, gated separately on CI success.

## Optional settings

| Setting | When to use |
|---------|-------------|
| Require conversation resolution | Team workflows where review comments must be addressed before merge |
| Require signed commits | Hardened deployments; every contributor needs a signing key |
| Require linear history | If you prefer rebase or squash merges over merge commits |

## Why the docker check is advisory

For PRs, CI runs four jobs: `lint`, `check` (a 3.12/3.13/3.14 matrix), `secrets`, and `docker`. The `docker` job builds the image but never pushes. The production build and push is a separate workflow, `docker.yml`, triggered after CI succeeds on `main`.

**Risk accepted:** a PR that breaks the Dockerfile can merge. CI passes, `docker.yml` then fails on `main`, and the previous image keeps serving production until a fix lands.

**Why accept it:** Docker builds are slow and the Dockerfile changes rarely. Requiring the check would add latency to every merge to guard against a rare, quickly recoverable failure.

## Verifying the rule

1. Push a commit directly to `main` from a non-bypass account; it should be rejected.
2. Open a PR with a deliberately failing test; the merge button should be unavailable.
3. Fix the test; the merge button should re-enable once checks pass.
