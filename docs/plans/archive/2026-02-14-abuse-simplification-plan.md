# Abuse Simplification Implementation Plan

**Date:** 2026-02-14
**Status:** Draft
**Owner:** LLM plugin maintainers

## Goal

Reduce abuse-mitigation complexity while keeping the highest-value protections:

- Keep command authorization and NickServ gating where it matters.
- Keep manual moderation (`%flag`, `%unflag`, `%flagged`).
- Keep usage auditing with explicit status values.
- Add a focused, low-complexity per-command limiter for expensive commands.
- Remove or defer auto-flag and owner broadcast complexity.

## Scope Decisions

### Keep

- Capability wrappers on command entrypoints.
- NickServ account requirement for `%draw` and `%animate`.
- Flagged-user blocking on all user-facing commands.
- Usage logging for success and blocked/error outcomes.
- Manual flag admin commands and DB table.

### Simplify

- Replace distributed precheck logic with one shared preflight path.
- Limit rate-limiting scope to `%draw` and `%animate` only.
- Keep limiter in memory (no schema changes for rate limiting).

### Defer (remove for now)

- Auto-flag threshold logic (`flagThreshold`, `flagWindow`, `_maybe_auto_flag`).
- Owner notification fanout (`_notify_owners`) for abuse events.
- Any queued/offline abuse-alert delivery mechanism.

## Target State

1. Every command runs the same preflight sequence:
   `capability -> account requirement (if needed) -> flagged check -> rate limit (if enabled)`.
2. `%draw` and `%animate` are the only commands with explicit per-command rate limits.
3. Abuse moderation remains explicit and manual (admin actions), with clear usage rows (`status`).
4. Config surface is smaller and easier for operators to reason about.

## File-Level Change Plan

## 1) `plugins/llm/src/llm/plugin.py`

### 1.1 Introduce a shared preflight helper

- [ ] Add a small preflight result object (dataclass or NamedTuple) near helper methods.
- [ ] Add `_run_preflight(...)` with inputs:
  - `command: str`
  - `irc, msg`
  - `text: str`
  - `require_account: bool`
  - `apply_rate_limit: bool`
- [ ] Ensure `_run_preflight(...)` handles:
  - identity resolve through existing account-aware path (preserve nick->account migration side effects)
  - account resolve / NickServ requirement
  - flagged block check
  - optional rate-limit check
  - standardized blocked logging
- [ ] Return enough data for handlers to continue without duplicate lookups:
  - identity used for logging (`nick_or_account`)
  - channel
  - account (or `None`)

### 1.2 Switch commands to shared preflight

- [ ] Refactor `ask` to call `_run_preflight(... require_account=False, apply_rate_limit=False)`.
- [ ] Refactor `code` to call `_run_preflight(... require_account=False, apply_rate_limit=False)`.
- [ ] Refactor `draw` to call `_run_preflight(... require_account=True, apply_rate_limit=True)`.
- [ ] Refactor `animate` to call `_run_preflight(... require_account=True, apply_rate_limit=True)`.
- [ ] Remove duplicated inline auth/flagged checks in each command body.

### 1.3 Remove auto-flag and owner-notify path

- [ ] Remove `_maybe_auto_flag(...)`.
- [ ] Remove `_notify_owners(...)`.
- [ ] Remove calls to `_maybe_auto_flag(...)` after `content_blocked`.
- [ ] Remove calls to `_notify_owners(...)` from `%flag` and `%unflag`.
- [ ] Keep `%flag`, `%unflag`, `%flagged` functionality intact.

### 1.4 Add in-memory per-command limiter (draw/animate only)

- [ ] Add private in-memory store:
  - `self._rate_buckets: dict[str, collections.deque[float]]`
  - key format: `"{command}:{account_or_prefix}"`
- [ ] Add helper `_is_rate_limited(command, key, now) -> bool`.
- [ ] Add helper `_check_rate_limit(...) -> bool` that:
  - always evaluates command-specific window/count
  - checks enabled switch
  - when enforcement is enabled and blocked:
    - logs `status="rate_limited"` on block
    - sends short user-facing error
  - when enforcement is disabled and threshold is exceeded:
    - allows command execution
    - emits a structured monitor log entry (`rate_limit_shadow`) for rollout tuning
- [ ] Ensure limiter bucket lifecycle is bounded:
  - evict expired timestamps before each decision
  - delete bucket keys when deque becomes empty
- [ ] Apply only for `draw` and `animate`.

### 1.5 Command capability consistency

- [ ] Update `animate` wrapper to require capability, matching other commands:
  - from `animate = wrap(animate, ["text"])`
  - to `animate = wrap(animate, [("checkCapability", "llm.animate"), "text"])`
- [ ] Keep `video = animate` alias behavior unchanged.

## 2) `plugins/llm/src/llm/config.py`

### 2.1 Remove deferred auto-flag config keys

- [ ] Remove `flagThreshold`.
- [ ] Remove `flagWindow`.

### 2.2 Add minimal rate-limit config keys

- [ ] Add `enforceRateLimits` (global bool, default `False` for monitor-first rollout).
- [ ] Add `drawRateLimitCount` (global int, default `3`).
- [ ] Add `drawRateLimitWindow` (global int seconds, default `60`).
- [ ] Add `animateRateLimitCount` (global int, default `2`).
- [ ] Add `animateRateLimitWindow` (global int seconds, default `600`).
- [ ] Add help text explaining these apply to expensive commands only.

### 2.3 Operator control boundaries

- [ ] Verify channel-safe knobs remain op-settable:
  - `askSystemPrompt`, `codeSystemPrompt`, `askModel`, `codeModel`, `drawModel`, `animateModel`.
- [ ] Keep sensitive/global knobs non-channel-op-editable by design:
  - API keys
  - database path
  - global limiter enforcement and thresholds

## 3) `plugins/llm/src/llm/persistence.py`

### 3.1 No schema changes required

- [ ] Keep existing `flagged_users` table and methods.
- [ ] Keep existing usage `status` column for auditability.
- [ ] Do not add rate-limit persistence in this simplification pass.

### 3.2 Status taxonomy update

- [ ] Ensure `rate_limited` is documented in method docstrings/comments where statuses are listed.

## 4) Tests

## 4.1 `plugins/llm/tests/test_commands.py`

- [ ] Keep and adjust auth tests:
  - `test_draw_requires_nickserv_auth` remains valid.
- [ ] Add tests:
  - `animate` requires `llm.animate` capability.
  - draw rate-limited when over threshold.
  - animate rate-limited when over threshold.
  - ask/code are not rate-limited by new limiter.
- [ ] Remove tests asserting auto-flag callback invocation from draw/animate command paths.
- [ ] Remove/update tests that assert `%flag`/`%unflag` owner notifications.

## 4.2 `plugins/llm/tests/test_plugin.py`

- [ ] Remove `_maybe_auto_flag` test class.
- [ ] Remove `_notify_owners` helper tests.
- [ ] Add focused tests for new helpers:
  - `_is_rate_limited` window eviction behavior.
  - `_check_rate_limit` blocked vs allowed paths.
  - `_run_preflight` blocked logging behavior by reason.

## 4.3 `plugins/llm/tests/test_integration.py`

- [ ] Remove or rewrite full auto-flag integration scenario.
- [ ] Add integration scenario:
  - repeated draw prompts trigger `rate_limited`,
  - `%unflag` flow still works independently,
  - normal draw succeeds after window expiration.
- [ ] Remove assertions that `%flag`/`%unflag` trigger owner notifications.

## 4.4 `plugins/llm/tests/conftest.py`

- [ ] Replace default `flagThreshold`/`flagWindow` fixture keys with new limiter keys.
- [ ] Set `enforceRateLimits=False` default for compatibility unless individual tests opt in.

## 4.5 `plugins/llm/tests/test_animate.py`

- [ ] Update animate command tests for new `llm.animate` capability wrapper.
- [ ] Add/adjust tests for animate limiter behavior in monitor mode vs enforced mode.

## 5) Docs

## 5.1 `README.md`

- [ ] Update abuse protection section to reflect:
  - manual moderation + flagged-user blocking
  - explicit per-command limiter for draw/animate
  - no automatic flagging in simplified mode

## 5.2 `plugins/llm/README.md`

- [ ] Document command-level protection matrix:
  - capability required
  - NickServ required
  - rate-limited

## Execution Checklist (Ordered)

1. [ ] Add config keys and remove auto-flag config in `config.py`.
2. [ ] Implement in-memory limiter + shared preflight in `plugin.py`.
3. [ ] Remove auto-flag and owner-notify helpers and callsites.
4. [ ] Add `llm.animate` capability wrapper in `plugin.py`.
5. [ ] Update tests (`test_commands.py`, `test_plugin.py`, `test_integration.py`, `test_animate.py`, `conftest.py`).
6. [ ] Update README docs.
7. [ ] Run full plugin test suite.
8. [ ] Run lint/type checks.
9. [ ] Manual smoke test on IRC staging.

## Verification Commands

```bash
# Focused command/auth/rate tests
uv run pytest plugins/llm/tests/test_commands.py -q

# Plugin helper/unit tests
uv run pytest plugins/llm/tests/test_plugin.py -q

# Integration flow tests
uv run pytest plugins/llm/tests/test_integration.py -q

# Animate command/capability tests
uv run pytest plugins/llm/tests/test_animate.py -q

# Full plugin suite
uv run pytest plugins/llm/tests -q

# Lint/type checks
uv run ruff check plugins/llm/src/llm plugins/llm/tests
uv run ty check plugins/llm/src/llm
```

## Rollout Plan

### Phase A: Merge with monitor-first defaults

- [ ] Merge with `enforceRateLimits=False`.
- [ ] Observe `rate_limit_shadow` monitor logs in production-like traffic for 3-7 days.
- [ ] Tune count/window thresholds from observed patterns.

### Phase B: Enforce expensive-command limits

- [ ] Set `enforceRateLimits=True`.
- [ ] Enable only draw/animate limiting (already scoped).
- [ ] Monitor `rate_limited` usage rows, error rates, and user complaints for 48 hours.

### Phase C: Stabilize

- [ ] Freeze limiter defaults.
- [ ] Archive old auto-flag design docs as superseded by simplified model.

## Acceptance Criteria

- [ ] `draw` and `animate` require both capability and NickServ account.
- [ ] `ask` and `code` still function for unauthed users (subject to existing capability policy).
- [ ] Repeated draw/animate requests above threshold return `rate_limited` and do not hit provider calls.
- [ ] With `enforceRateLimits=False`, over-threshold draw/animate requests still execute but produce `rate_limit_shadow` monitor logs.
- [ ] Flagged users remain blocked consistently across all commands.
- [ ] No auto-flagging or owner abuse-notice side effects remain in runtime code.
- [ ] Test suite passes with updated expectations.

## Risks and Mitigations

- Risk: removing auto-flag increases moderator load.
  - Mitigation: keep fast manual `%flag` workflow and clear `rate_limited`/`content_blocked` audit rows.
- Risk: in-memory limiter resets on restart.
  - Mitigation: acceptable for simplification pass; revisit only if abuse recurs after restarts.
- Risk: capability behavior change for `animate`.
  - Mitigation: communicate new `llm.animate` requirement and add explicit tests.
