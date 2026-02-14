# Abuse Mitigation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add auth gating on draw, full usage auditing with prompts/refusals, user flagging with auto-flag, and owner alerts.

**Architecture:** Extends the existing SQLite persistence layer with new columns on `usage` and a new `flagged_users` table. Plugin gets shared helpers for NickServ auth and flag checks. All commands log every request (success or failure) with prompt text and status. Auto-flag threshold triggers on content safety refusals and sends IRC NOTICE to owners.

**Tech Stack:** Python 3.12+, SQLite (ALTER TABLE migrations), Limnoria IRC framework, pytest

**Status:** v1 implemented in commits d9faffc..c7f15fb. This revision (v2) fixes bugs and gaps found during post-implementation review.

---

## Post-Implementation Review: Bugs & Gaps

| # | Severity | Issue | Root Cause |
|---|----------|-------|------------|
| A | **Bug** | Auto-flag never triggers — `count_recent_refusals` queries `nick` column but `_maybe_auto_flag` passes NickServ `account` | usage table stores IRC nick, not account; the two differ (e.g., nick `alice` → account `alice_acct`) |
| B | **Gap** | `ask`/`code` content safety refusals don't trigger auto-flag | Error handling classifies all failures as generic `"error"`, never `"content_blocked"` |
| C | **Gap** | Flagged users bypass check on `ask`/`code` by not identifying with NickServ | `_check_flagged` returns False when account is None |
| D | **Cleanup** | Missing fixture updates in 4 test files | Custom registry mocks in test_etiquette, test_reminders, test_stress, test_service don't include `flagThreshold`/`flagWindow` |
| E | **Design** | No flag history — re-flagging overwrites previous flag reason/type | `UNIQUE` constraint on `account` means one row per user, UPDATE on re-flag |
| F | **Plan** | Task dependency ordering was backwards | Task 10 calls `_notify_owners` which doesn't exist until Task 11; Tasks 1-2 split a single migration across two commits |

---

## v2 Fix Tasks

### Task 1: Fix nick-vs-account Mismatch in Refusal Counting (Bug A)

**Problem:** `_maybe_auto_flag` passes `account` (NickServ account name) to `count_recent_refusals`, which queries `WHERE nick = ?`. But the `usage` table stores the IRC nick, not the account. When nick ≠ account, the count is always 0 and auto-flag never fires.

**Fix approach:** Store the resolved identity (which already maps nick→account when available via `_resolve_nick_to_identity` / `_get_identity`) in the `nick` column. This is already partially done — `_store_context_and_log_usage` uses `nick` which comes from `_get_identity` in most call paths. The problem is the early-exit paths (auth_failure, flagged_blocked, content_blocked in draw/animate) that use `ircutils.nickFromHostmask` directly.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (draw, animate early-exit log_usage calls)
- Test: `plugins/llm/tests/test_integration.py`

**Step 1: Write the failing test**

In `test_integration.py`, add or modify the auto-flag flow test to use a nick that differs from the account:

```python
def test_auto_flag_triggers_when_nick_differs_from_account(self, ...):
    """GIVEN nick 'alice' with account 'alice_acct' WHEN content blocked 3x THEN auto-flagged."""
    # Setup: nickToAccount("alice") returns "alice_acct"
    # Mock _get_identity / _resolve_nick_to_identity to return "alice_acct"
    # Fire 3 draw requests that get content_blocked
    # Assert: db.is_user_flagged("alice_acct") is True
    # Assert: count_recent_refusals("alice_acct", since) == 3
```

**Step 2: Run test to verify it fails**

Run: `make test -- -k test_auto_flag_triggers_when_nick_differs -v`
Expected: FAIL — refusal count is 0 because usage rows store "alice" but count queries "alice_acct".

**Step 3: Fix the draw/animate early-exit paths**

In draw, after the content_blocked log_usage call, ensure the nick used is the resolved identity (account), not the raw IRC nick:

```python
# After image generation returns with error
nick = self._get_identity(irc, msg)  # resolves to account when identified
# ... existing status determination ...
self.db.log_usage(
    nick, channel, "draw", result.model, ...,
    prompt=text, status=status, error_detail=error_detail,
)
if status == "content_blocked" and account:
    self._maybe_auto_flag(irc, account, channel)
```

Verify that ALL log_usage calls in draw and animate use the resolved identity, not the raw nick from `ircutils.nickFromHostmask`.

**Step 4: Run all tests**

Run: `make preflight`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "fix: use resolved identity in usage logging so auto-flag counts match"
```

---

### Task 2: Detect Content Blocks in `ask`/`code` (Gap B)

**Problem:** `ask` and `code` commands catch all LiteLLM errors uniformly as `"error"` status. When a content safety violation occurs (e.g., `ContentPolicyViolationError`), it's logged as a generic error, so it doesn't contribute to the auto-flag refusal count.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_store_context_and_log_usage` or the ask/code error paths)
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Write the failing test**

```python
def test_ask_content_blocked_logs_content_blocked_status(self, ...):
    """GIVEN ask that hits content safety WHEN completed THEN usage logged as content_blocked."""
    # Mock service.chat_completion to return result with error containing
    # "content policy" or matching _is_content_blocked_error
    result = ChatResult(content="", error="content policy violation", ...)
    service.chat_completion.return_value = result
    plugin.ask(irc, msg, [], "bad prompt")
    # Check log_usage was called with status="content_blocked"
```

**Step 2: Run test to verify it fails**

Run: `make test -- -k test_ask_content_blocked_logs -v`
Expected: FAIL — status is "error", not "content_blocked".

**Step 3: Update `_store_context_and_log_usage`**

This method handles logging for ask/code. Add content block detection:

```python
def _store_context_and_log_usage(
    self, nick, channel, command, text, response, result, irc, msg,
) -> None:
    # Store context (only on success, unchanged)
    if result.error is None and self._get_context_enabled(channel):
        ...

    # Determine status
    if result.error is None:
        status = "success"
    elif self._is_content_blocked_error(result.error):
        status = "content_blocked"
    else:
        status = "error"

    error_detail = (result.error or "")[:200]
    self.db.log_usage(
        nick, channel, command, result.model,
        result.prompt_tokens, result.completion_tokens, result.cost,
        prompt=text, status=status, error_detail=error_detail,
    )

    # Trigger auto-flag check on content blocks
    if status == "content_blocked":
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        try:
            account = irc.state.nickToAccount(raw_nick)
        except (KeyError, AttributeError):
            account = None
        if account:
            self._maybe_auto_flag(irc, account, channel)
```

**Step 4: Run all tests**

Run: `make preflight`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "fix: detect content blocks in ask/code and trigger auto-flag"
```

---

### Task 3: Close Flag Evasion via NickServ Opt-Out (Gap C)

**Problem:** A flagged user can bypass the flag check on `ask`/`code` by simply not identifying with NickServ. The `_check_flagged` helper returns False when account is None.

**Design decision needed:** We have two options:
1. **Require NickServ for all commands** — blocks anonymous usage entirely (breaking change)
2. **Check by nick as fallback** — maintain a nick→account cache from previous identifications, check both

**Recommended: Option 2 — nick fallback with known-accounts lookup**

Add a helper that checks both: if the user is identified, check their account; if not, check if their nick matches any flagged account's known nicks (from previous usage rows).

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (add `get_nicks_for_account`)
- Modify: `plugins/llm/src/llm/plugin.py` (`_check_flagged`)
- Test: `plugins/llm/tests/test_plugin.py`, `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing test**

```python
def test_check_flagged_blocks_unidentified_user_with_known_nick(self, ...):
    """GIVEN flagged account 'alice_acct' with known nick 'alice' WHEN unidentified 'alice' uses ask THEN blocked."""
    plugin.db.is_user_flagged.return_value = False  # account=None path
    plugin.db.is_nick_flagged.return_value = True  # nick lookup hits
    result = plugin._check_flagged(irc, msg, None)
    assert result is True
```

**Step 2: Implement `is_nick_flagged` in persistence.py**

```python
def is_nick_flagged(self, nick: str) -> bool:
    """Check if a nick belongs to any actively flagged account.

    Looks up usage rows to find the most recent account associated with
    this nick, then checks if that account is flagged.
    """
    conn = self._connect()
    try:
        # Find the most recent account for this nick from usage history
        # This relies on usage rows storing the resolved identity
        row = conn.execute(
            "SELECT fu.account FROM flagged_users fu "
            "WHERE fu.resolved_at IS NULL "
            "AND fu.account IN ("
            "  SELECT DISTINCT nick FROM usage WHERE nick = ?"
            ")",
            (nick,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()
```

**Step 3: Update `_check_flagged`**

```python
def _check_flagged(self, irc, msg, account):
    if account and self.db.is_user_flagged(account):
        irc.error(_("Your account has been suspended. Contact a bot admin."))
        return True
    if account is None:
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        nick = self._resolve_nick_to_identity(irc, raw_nick)
        if self.db.is_nick_flagged(nick):
            irc.error(_("Your account has been suspended. Contact a bot admin."))
            return True
    return False
```

**Step 4: Run all tests**

Run: `make preflight`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "fix: check flagged status by nick when user is not identified"
```

---

### Task 4: Add Missing Fixture Defaults (Cleanup D)

**Problem:** Four test files have custom `registryValue` mocks that don't include `flagThreshold` and `flagWindow`. If any code path reads these during those tests, it gets an empty string instead of an integer.

**Files to update:**
- `plugins/llm/tests/test_etiquette.py` (line ~259)
- `plugins/llm/tests/test_reminders.py` (line ~209)
- `plugins/llm/tests/test_stress.py` (line ~288)
- `plugins/llm/tests/test_service.py` (all custom lambda mocks — ~23 locations)

**Step 1: Write a test that catches this**

No new test needed — this is a preventive cleanup. But verify:

Run: `make test -v`

If all pass currently, the missing keys aren't being hit. But they should be there for correctness.

**Step 2: Add the defaults**

For each custom registry mock dict, add:
```python
"flagThreshold": 5,
"flagWindow": 3600,
```

For test_service.py, the cleanest approach is to define a shared base dict at module level and spread it into each custom mock:

```python
_BASE_REGISTRY = {
    "flagThreshold": 5,
    "flagWindow": 3600,
}
```

Then in each custom mock: `{**_BASE_REGISTRY, "askApiKey": "sk-test", ...}`

**Step 3: Run all tests**

Run: `make preflight`
Expected: PASS

**Step 4: Commit**

```bash
git add plugins/llm/tests/
git commit -m "fix: add missing flagThreshold/flagWindow to all test fixtures"
```

---

### Task 5: (Optional) Add Flag History Table

**Problem:** The `flagged_users` table uses `UNIQUE` on `account`, meaning each user can only have one flag row. Re-flagging after unflag does an UPDATE, overwriting the previous reason and who resolved it. There's no audit trail.

**This task is optional** — skip if the simpler model is sufficient for now.

**Approach:** Add a `flag_history` table that records every flag/unflag event. Keep the existing `flagged_users` table as-is for fast lookups (it becomes the "current state" table), and insert into `flag_history` on every state change.

```sql
CREATE TABLE IF NOT EXISTS flag_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL,
    action TEXT NOT NULL,  -- 'flagged' or 'unflagged'
    reason TEXT NOT NULL DEFAULT '',
    auto_flagged INTEGER NOT NULL DEFAULT 0,
    performed_by TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL
);
```

Wire `flag_user` and `unflag_user` to also insert into `flag_history`.

---

## Original Plan Summary (v1 — already implemented)

| # | Commit | Files | Status |
|---|--------|-------|--------|
| 1 | `feat: add prompt, status, error_detail columns to usage table` | persistence.py, test_persistence.py | Done |
| 2 | `feat: add flagged_users table for abuse tracking` | persistence.py, test_persistence.py | Done |
| 3 | `feat: add flagged user CRUD methods and refusal counting` | persistence.py, test_persistence.py | Done |
| 4 | `feat: extend log_usage to store prompt, status, error_detail` | persistence.py, test_persistence.py | Done |
| 5 | `feat: add flagThreshold and flagWindow config values` | config.py, conftest.py | Done |
| 6 | `feat: add shared _require_account NickServ helper` | plugin.py, test_plugin.py | Done |
| 7 | `feat: require NickServ identification for draw command` | plugin.py, test_commands.py, test_animate.py | Done |
| 8 | `feat: add pre-command flag check to block suspended users` | plugin.py, test_plugin.py | Done |
| 9 | `feat: log all command outcomes with prompt text and status` | plugin.py, test_commands.py | Done |
| 10 | `feat: auto-flag users after content safety refusal threshold` | plugin.py, test_plugin.py | Done |
| 11 | `feat: add _notify_owners IRC NOTICE helper` | plugin.py, test_plugin.py | Done |
| 12 | `feat: add %flag, %unflag, %flagged admin commands` | plugin.py, test_commands.py | Done |
| 13 | `test: add integration test for auto-flag abuse flow` | test_integration.py | Done |

## v2 Fix Summary

| # | Commit | Severity | Issue |
|---|--------|----------|-------|
| 1 | `fix: use resolved identity in usage logging so auto-flag counts match` | Bug | nick≠account breaks refusal counting |
| 2 | `fix: detect content blocks in ask/code and trigger auto-flag` | Gap | ask/code abuse doesn't trigger auto-flag |
| 3 | `fix: check flagged status by nick when user is not identified` | Gap | Flagged users evade check by not identifying |
| 4 | `fix: add missing flagThreshold/flagWindow to all test fixtures` | Cleanup | Incomplete test mocks |
| 5 | _(optional)_ `feat: add flag_history audit table` | Design | No audit trail of previous flags |

## Plan v1 Structural Issues (for future plans)

- **Tasks 1-2 should be one commit.** Both modify the same v2 migration block. Splitting them means an intermediate commit has `PRAGMA user_version = 2` set but `flagged_users` doesn't exist.
- **Tasks 10-11 dependency ordering was backwards.** Task 10 calls `_notify_owners` which is defined in Task 11. Should be: 11 → 10, or combined.
- **Test fixture updates (Task 5) were incomplete.** Only updated conftest.py, but 4 other test files have independent registry mocks that also needed the new keys.
