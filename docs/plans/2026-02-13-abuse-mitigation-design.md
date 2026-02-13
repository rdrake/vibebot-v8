# Abuse Mitigation: Auth Gating, Usage Auditing, and User Flagging

**Date:** 2026-02-13
**Status:** Approved

## Problem

Users are abusing the bot's image generation and other commands. There is no way to:

- Restrict expensive commands (draw) to identified users
- Track what prompts are being sent or which requests get refused
- Flag abusive users for review or block them from the bot
- Alert bot owners when abuse is detected

## Overview

Five changes, layered on top of each other:

1. **Schema changes** — extend `usage` table, add `flagged_users` table
2. **NickServ gate on draw** — require identification, matching animate
3. **Enhanced usage logging** — log every request (success and failure) with prompt text
4. **User flagging and restriction** — auto-flag on refusal threshold, admin commands, pre-command block
5. **Owner alerts** — IRC NOTICE to owners on flag events

## 1. Schema Changes

### `usage` table — new columns

Added via `ALTER TABLE ADD COLUMN` in `_migrate()`. Safe in SQLite, preserves all existing data. Existing rows get defaults.

| Column | Type | Default | Purpose |
|---|---|---|---|
| `prompt` | TEXT | `''` | Full prompt text for every request |
| `status` | TEXT | `'success'` | Outcome category (see section 3) |
| `error_detail` | TEXT | `''` | Truncated error message, max 200 chars |

New index: `idx_usage_nick_status` on `(nick, status)` for abuse queries.

### New `flagged_users` table

```sql
CREATE TABLE IF NOT EXISTS flagged_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT UNIQUE NOT NULL,
    flagged_at REAL NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    auto_flagged INTEGER NOT NULL DEFAULT 0,
    resolved_at REAL,
    resolved_by TEXT
);
```

`SCHEMA_VERSION` bumps from 1 to 2.

## 2. NickServ Gate on Draw

### Shared helper

Extract NickServ check from `animate` into a reusable plugin method:

```python
def _require_account(self, irc, msg) -> str | None:
    """Resolve NickServ account or send error. Returns account or None."""
    raw_nick = ircutils.nickFromHostmask(msg.prefix)
    try:
        account = irc.state.nickToAccount(raw_nick)
    except (KeyError, AttributeError):
        account = None
    if not account:
        irc.error(_("You must be identified with NickServ to use this command."))
        return None
    return account
```

### Draw command

Keep existing `llm.draw` capability check via `@wrap`. Add `_require_account()` call at top of method body. Both gates must pass.

### Animate command

Replace inline NickServ check with `_require_account()`. Behavior unchanged, just DRY.

### Auth failure logging

When `_require_account()` returns None, log a usage row with `status='auth_failure'` and the prompt text before returning.

## 3. Enhanced Usage Logging

### Extended `log_usage` signature

```python
def log_usage(
    self, nick, channel, command, model,
    prompt_tokens, completion_tokens, cost,
    prompt="", status="success", error_detail="",
) -> None:
```

### Log on ALL outcomes

Current state: only successful requests are logged. After: every request gets a usage row.

| Status value | When |
|---|---|
| `success` | Normal completion |
| `content_blocked` | Provider content safety refusal |
| `auth_failure` | NickServ check failed |
| `validation_error` | Empty prompt, too long, bad input |
| `timeout` | Request timed out |
| `error` | Any other error (API, network, etc.) |

### Call sites to update

1. **`_store_context_and_log_usage`** (ask/code) — always log, pass prompt and status
2. **`draw` command** — log on success and on content_blocked/error paths
3. **`animate` command** — same pattern as draw
4. **`_require_account` failures** — log `status='auth_failure'` with prompt
5. **`_deliver_pending_task`** — add failure logging

### Existing `%usage` command

Unaffected. Summary queries (`SUM(cost)`, `COUNT(*)`) still work. Failed requests with `cost=0` add to count but not cost totals. Filtering the display to exclude failures is a future enhancement if needed.

## 4. User Flagging and Restriction

### Config values (global)

| Config key | Type | Default | Purpose |
|---|---|---|---|
| `flagThreshold` | Integer | `5` | Content blocks to trigger auto-flag |
| `flagWindow` | Integer | `3600` | Time window in seconds (default 1 hour) |

### Auto-flag logic

After every `content_blocked` usage log, query recent refusals:

```sql
SELECT COUNT(*) FROM usage
WHERE nick = ? AND status = 'content_blocked' AND timestamp >= ?
```

If count >= `flagThreshold`, insert into `flagged_users` with `auto_flagged=1`.

### Pre-command check

Called at the top of `ask`, `code`, `draw`, `animate`:

```python
def _check_flagged(self, irc, msg, account) -> bool:
    """Return True if user is flagged and should be blocked."""
    if self.db.is_user_flagged(account):
        irc.error(_("Your account has been suspended. Contact a bot admin."))
        return True
    return False
```

For `draw`/`animate`: reuse account from `_require_account()`.

For `ask`/`code`: attempt account resolution but don't require it. Unflagged unidentified users can still use ask/code. Only block if the resolved account is in the flagged table.

### Admin commands

| Command | Capability | Description |
|---|---|---|
| `%flag <nick> [reason]` | `admin` | Manually flag user (resolves nick to account) |
| `%unflag <nick>` | `admin` | Clear flag (sets `resolved_at`, `resolved_by`) |
| `%flagged` | `admin` | List all currently flagged (unresolved) users |

`%unflag` sets `resolved_at` and `resolved_by` for audit trail — does not delete the row.

### Database methods

- `flag_user(account, reason, auto_flagged)` — INSERT OR IGNORE (idempotent)
- `unflag_user(account, resolved_by)` — UPDATE SET resolved_at, resolved_by
- `is_user_flagged(account) -> bool` — check `resolved_at IS NULL`
- `get_flagged_users() -> list[FlaggedUserRow]` — all unresolved flags
- `count_recent_refusals(nick, since) -> int` — for auto-flag threshold

## 5. Owner Alerts

### Mechanism

IRC NOTICE to all online users with `owner` capability on flag events.

### Events

| Event | Message |
|---|---|
| Auto-flag | `"[LLM] Auto-flagged user {account}: {count} content blocks in {window}. Use %flagged to review."` |
| Manual flag | `"[LLM] {admin} flagged user {account}: {reason}"` |
| Manual unflag | `"[LLM] {admin} unflagged user {account}."` |

### Finding online owners

```python
def _notify_owners(self, irc, message):
    """Send NOTICE to all online users with owner capability."""
    for u in ircdb.users():
        user = ircdb.users.getUser(u)
        if user.checkCapability("owner"):
            for nick in irc.state.nicksToHostmasks:
                hostmask = irc.state.nicksToHostmasks[nick]
                if user.checkHostmask(hostmask):
                    irc.queueMsg(ircmsgs.notice(nick, message))
                    break
```

### No offline queuing

If no owner is online, the alert is not queued. The `flagged_users` table is the durable record. Owners can run `%flagged` at any time. Queued alerts are a future enhancement.
