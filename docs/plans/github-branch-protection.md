# GitHub Branch Protection Configuration

Configure branch protection rules to enforce the CI/CD workflow.

## Steps

1. Go to **Settings → Branches** in your GitHub repository
2. Click **Add branch ruleset** (or "Add rule" for classic protection)
3. Set branch name pattern: `main`

## Required Settings

| Setting | Value |
|---------|-------|
| Require a pull request before merging | ✅ Enabled |
| Required approvals | 0 (solo project) |
| Require status checks to pass before merging | ✅ Enabled |
| Require branches to be up to date before merging | ✅ Enabled |
| Do not allow bypassing the above settings | ✅ Enabled |

## Required Status Checks

Add these status checks (they appear after the first CI run on a PR):

- `check (3.12)`
- `check (3.13)`
- `check (3.14)`
- `secrets`

## Optional Settings

| Setting | Recommendation |
|---------|----------------|
| Require conversation resolution | Optional - useful if you want to ensure all review comments are addressed |
| Require signed commits | Optional - adds verification but requires GPG setup |
| Require linear history | Optional - enforces rebase/squash merges |

## Verification

After configuring:

1. Try pushing directly to main → Should be rejected
2. Create a PR with a failing test → Should block merge
3. Fix the test → Should allow merge

## Direct Link

```
https://github.com/YOUR_USERNAME/vibebot-v8/settings/branches
```
