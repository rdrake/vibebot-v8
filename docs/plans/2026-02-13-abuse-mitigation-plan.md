# Abuse Mitigation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add auth gating on draw, full usage auditing with prompts/refusals, user flagging with auto-flag, and owner alerts.

**Architecture:** Extends the existing SQLite persistence layer with new columns on `usage` and a new `flagged_users` table. Plugin gets shared helpers for NickServ auth and flag checks. All commands log every request (success or failure) with prompt text and status. Auto-flag threshold triggers on content safety refusals and sends IRC NOTICE to owners.

**Tech Stack:** Python 3.12+, SQLite (ALTER TABLE migrations), Limnoria IRC framework, pytest

---

### Task 1: Schema Migration — Add Columns to `usage` Table

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:14` (SCHEMA_VERSION), `plugins/llm/src/llm/persistence.py:107-167` (_migrate)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing test**

In `plugins/llm/tests/test_persistence.py`, add to `TestDatabaseInit`:

```python
def test_usage_table_has_prompt_column(self, tmp_path: Path) -> None:
    """GIVEN a new database WHEN initialized THEN usage table has prompt column."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    try:
        cursor = conn.execute("PRAGMA table_info(usage)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "prompt" in columns
    finally:
        conn.close()

def test_usage_table_has_status_column(self, tmp_path: Path) -> None:
    """GIVEN a new database WHEN initialized THEN usage table has status column."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    try:
        cursor = conn.execute("PRAGMA table_info(usage)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "status" in columns
    finally:
        conn.close()

def test_usage_table_has_error_detail_column(self, tmp_path: Path) -> None:
    """GIVEN a new database WHEN initialized THEN usage table has error_detail column."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    try:
        cursor = conn.execute("PRAGMA table_info(usage)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "error_detail" in columns
    finally:
        conn.close()
```

**Step 2: Run tests to verify they fail**

Run: `make test -- -k "test_usage_table_has_prompt_column or test_usage_table_has_status_column or test_usage_table_has_error_detail_column" -v`
Expected: FAIL — columns don't exist yet.

**Step 3: Implement the migration**

In `plugins/llm/src/llm/persistence.py`:

1. Change `SCHEMA_VERSION = 1` to `SCHEMA_VERSION = 2`.

2. In `_migrate()`, after the existing `CREATE TABLE` executescript block (after `conn.commit()` on line ~165), add migration logic:

```python
# Check current schema version and apply migrations
user_version = conn.execute("PRAGMA user_version").fetchone()[0]

if user_version < 2:
    # Add new columns to usage table for audit tracking
    conn.executescript("""
        ALTER TABLE usage ADD COLUMN prompt TEXT NOT NULL DEFAULT '';
        ALTER TABLE usage ADD COLUMN status TEXT NOT NULL DEFAULT 'success';
        ALTER TABLE usage ADD COLUMN error_detail TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_usage_nick_status
            ON usage(nick, status);
    """)
    conn.commit()

conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
conn.commit()
```

Also set `PRAGMA user_version = 1` after the initial CREATE TABLE block for new databases (so the version tracking starts clean).

**Step 4: Run tests to verify they pass**

Run: `make test -- -k "test_usage_table_has" -v`
Expected: PASS

**Step 5: Write migration-from-v1 test**

```python
def test_migration_preserves_existing_usage_data(self, tmp_path: Path) -> None:
    """GIVEN a v1 database with usage data WHEN migrated THEN data preserved with defaults."""
    db_path = str(tmp_path / "test.db")
    # Create a v1-style database manually
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            nick TEXT NOT NULL,
            channel TEXT NOT NULL,
            command TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            cost REAL NOT NULL DEFAULT 0.0
        );
    """)
    conn.execute(
        "INSERT INTO usage (timestamp, nick, channel, command, model, prompt_tokens, "
        "completion_tokens, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (1000.0, "alice", "#test", "ask", "gpt-4", 100, 50, 0.01),
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()

    # Open with LLMDatabase — should migrate
    db = LLMDatabase(db_path)
    summary = db.get_usage_summary()
    assert summary.total_requests == 1
    assert summary.total_cost == pytest.approx(0.01)

    # Check new columns have defaults
    conn = db._connect()
    try:
        row = conn.execute("SELECT prompt, status, error_detail FROM usage").fetchone()
        assert row == ("", "success", "")
    finally:
        conn.close()
```

**Step 6: Run all persistence tests**

Run: `make test -- -k test_persistence -v`
Expected: PASS

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add prompt, status, error_detail columns to usage table"
```

---

### Task 2: Schema Migration — Create `flagged_users` Table

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:107-167` (_migrate)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing test**

```python
def test_creates_flagged_users_table(self, tmp_path: Path) -> None:
    """GIVEN a new database WHEN initialized THEN flagged_users table exists."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='flagged_users'"
        )
        assert cursor.fetchone() is not None
    finally:
        conn.close()
```

**Step 2: Run test to verify it fails**

Run: `make test -- -k test_creates_flagged_users_table -v`
Expected: FAIL

**Step 3: Add table creation to _migrate()**

In the v2 migration block (from Task 1), add:

```python
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

**Step 4: Run test to verify it passes**

Run: `make test -- -k test_creates_flagged_users_table -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add flagged_users table for abuse tracking"
```

---

### Task 3: Add `FlaggedUserRow` NamedTuple and Database Methods

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (new NamedTuple + 5 methods)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Add `FlaggedUserRow` NamedTuple**

After `PendingTaskRow` in `persistence.py`:

```python
class FlaggedUserRow(NamedTuple):
    """A flagged user loaded from the database."""

    id: int
    account: str
    flagged_at: float
    reason: str
    auto_flagged: int  # 1 = auto-flagged, 0 = manual
    resolved_at: float | None
    resolved_by: str | None
```

**Step 2: Write failing tests for all 5 methods**

Add a new test class `TestFlaggedUsers` in `test_persistence.py`:

```python
class TestFlaggedUsers:
    """Test user flagging CRUD operations."""

    def test_flag_user_creates_record(self, tmp_path: Path) -> None:
        """GIVEN a database WHEN flag_user called THEN record created."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "repeated content violations", auto_flagged=True)
        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].account == "alice"
        assert flagged[0].reason == "repeated content violations"
        assert flagged[0].auto_flagged == 1
        assert flagged[0].resolved_at is None

    def test_flag_user_idempotent(self, tmp_path: Path) -> None:
        """GIVEN already-flagged user WHEN flagged again THEN no duplicate."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "first reason", auto_flagged=True)
        db.flag_user("alice", "second reason", auto_flagged=False)
        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].reason == "first reason"  # original preserved

    def test_is_user_flagged_returns_true(self, tmp_path: Path) -> None:
        """GIVEN a flagged user WHEN checking THEN returns True."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "bad behavior", auto_flagged=False)
        assert db.is_user_flagged("alice") is True

    def test_is_user_flagged_returns_false_when_not_flagged(self, tmp_path: Path) -> None:
        """GIVEN no flagged users WHEN checking THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.is_user_flagged("alice") is False

    def test_is_user_flagged_returns_false_after_unflag(self, tmp_path: Path) -> None:
        """GIVEN a flagged then unflagged user WHEN checking THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "reason", auto_flagged=True)
        db.unflag_user("alice", "admin_bob")
        assert db.is_user_flagged("alice") is False

    def test_unflag_sets_resolved_fields(self, tmp_path: Path) -> None:
        """GIVEN a flagged user WHEN unflagged THEN resolved_at and resolved_by set."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "reason", auto_flagged=True)
        result = db.unflag_user("alice", "admin_bob")
        assert result is True

        # Check the row directly
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT resolved_at, resolved_by FROM flagged_users WHERE account = ?",
                ("alice",),
            ).fetchone()
            assert row[0] is not None  # resolved_at is set
            assert row[1] == "admin_bob"
        finally:
            conn.close()

    def test_unflag_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """GIVEN no flagged users WHEN unflagging THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.unflag_user("nobody", "admin") is False

    def test_get_flagged_users_excludes_resolved(self, tmp_path: Path) -> None:
        """GIVEN flagged and unflagged users WHEN listing THEN only active flags shown."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "reason1", auto_flagged=True)
        db.flag_user("bob", "reason2", auto_flagged=False)
        db.unflag_user("alice", "admin")
        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].account == "bob"

    def test_count_recent_refusals(self, tmp_path: Path) -> None:
        """GIVEN content_blocked usage records WHEN counting THEN correct count."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        # Log some content_blocked records
        db.log_usage("alice", "#test", "draw", "dall-e-3", 0, 0, 0.0,
                      prompt="bad prompt 1", status="content_blocked")
        db.log_usage("alice", "#test", "draw", "dall-e-3", 0, 0, 0.0,
                      prompt="bad prompt 2", status="content_blocked")
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01,
                      prompt="good prompt", status="success")
        db.log_usage("bob", "#test", "draw", "dall-e-3", 0, 0, 0.0,
                      prompt="bob prompt", status="content_blocked")

        count = db.count_recent_refusals("alice", since=now - 3600)
        assert count == 2  # only alice's content_blocked, not bob's or success

    def test_count_recent_refusals_respects_time_window(self, tmp_path: Path) -> None:
        """GIVEN old and new refusals WHEN counting with since THEN only recent."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # Insert an old refusal directly
        conn = db._connect()
        try:
            conn.execute(
                "INSERT INTO usage (timestamp, nick, channel, command, model, "
                "prompt_tokens, completion_tokens, cost, prompt, status, error_detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now - 7200, "alice", "#test", "draw", "dall-e-3", 0, 0, 0.0,
                 "old prompt", "content_blocked", ""),
            )
            conn.commit()
        finally:
            conn.close()

        # Insert a recent refusal
        db.log_usage("alice", "#test", "draw", "dall-e-3", 0, 0, 0.0,
                      prompt="new prompt", status="content_blocked")

        count = db.count_recent_refusals("alice", since=now - 3600)
        assert count == 1  # only the recent one

    def test_reflag_after_unflag_creates_new_flag(self, tmp_path: Path) -> None:
        """GIVEN unflagged user WHEN flagged again THEN new active flag exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "first offense", auto_flagged=True)
        db.unflag_user("alice", "admin")
        assert db.is_user_flagged("alice") is False

        # Re-flag: since the row exists (resolved), we need to re-activate it
        db.flag_user("alice", "second offense", auto_flagged=False)
        assert db.is_user_flagged("alice") is True
```

**Step 3: Run tests to verify they fail**

Run: `make test -- -k TestFlaggedUsers -v`
Expected: FAIL — methods don't exist yet.

**Step 4: Implement the 5 database methods**

In `persistence.py`, add a new section after the usage operations:

```python
# ------------------------------------------------------------------
# Flagged user operations
# ------------------------------------------------------------------

def flag_user(
    self,
    account: str,
    reason: str,
    auto_flagged: bool,
) -> bool:
    """Flag a user account for abuse review.

    If the account has a resolved (cleared) flag, it is re-activated.
    If the account has an active flag, this is a no-op.

    Args:
        account: NickServ account name.
        reason: Human-readable reason for the flag.
        auto_flagged: True if triggered automatically by threshold.

    Returns:
        True if a new flag was created or re-activated, False if already active.
    """
    conn = self._connect()
    try:
        # Check if there's an existing row
        row = conn.execute(
            "SELECT id, resolved_at FROM flagged_users WHERE account = ?",
            (account,),
        ).fetchone()

        if row is None:
            # New flag
            conn.execute(
                "INSERT INTO flagged_users (account, flagged_at, reason, auto_flagged) "
                "VALUES (?, ?, ?, ?)",
                (account, time.time(), reason, 1 if auto_flagged else 0),
            )
            conn.commit()
            return True
        elif row[1] is not None:
            # Resolved flag — re-activate
            conn.execute(
                "UPDATE flagged_users SET flagged_at = ?, reason = ?, "
                "auto_flagged = ?, resolved_at = NULL, resolved_by = NULL "
                "WHERE account = ?",
                (time.time(), reason, 1 if auto_flagged else 0, account),
            )
            conn.commit()
            return True
        else:
            # Already actively flagged — no-op
            return False
    finally:
        conn.close()

def unflag_user(self, account: str, resolved_by: str) -> bool:
    """Clear the flag on a user account.

    Sets resolved_at and resolved_by. Does not delete the row (audit trail).

    Args:
        account: NickServ account name.
        resolved_by: Account of the admin who cleared the flag.

    Returns:
        True if a flag was resolved, False if not found or already resolved.
    """
    conn = self._connect()
    try:
        cursor = conn.execute(
            "UPDATE flagged_users SET resolved_at = ?, resolved_by = ? "
            "WHERE account = ? AND resolved_at IS NULL",
            (time.time(), resolved_by, account),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def is_user_flagged(self, account: str) -> bool:
    """Check if a user account has an active (unresolved) flag.

    Args:
        account: NickServ account name.

    Returns:
        True if the account has an active flag.
    """
    conn = self._connect()
    try:
        row = conn.execute(
            "SELECT 1 FROM flagged_users WHERE account = ? AND resolved_at IS NULL",
            (account,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()

def get_flagged_users(self) -> list[FlaggedUserRow]:
    """List all users with active (unresolved) flags.

    Returns:
        List of FlaggedUserRow ordered by flagged_at descending.
    """
    conn = self._connect()
    try:
        rows = conn.execute(
            "SELECT id, account, flagged_at, reason, auto_flagged, resolved_at, resolved_by "
            "FROM flagged_users WHERE resolved_at IS NULL ORDER BY flagged_at DESC",
        ).fetchall()
        return [FlaggedUserRow(*row) for row in rows]
    finally:
        conn.close()

def count_recent_refusals(self, nick: str, since: float) -> int:
    """Count content_blocked usage records for a nick since a timestamp.

    Args:
        nick: IRC nick or account name.
        since: Unix timestamp (only count records after this time).

    Returns:
        Number of content_blocked records.
    """
    conn = self._connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM usage "
            "WHERE nick = ? AND status = 'content_blocked' AND timestamp >= ?",
            (nick, since),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()
```

**Step 5: Run tests to verify they pass**

Run: `make test -- -k TestFlaggedUsers -v`
Expected: PASS

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add flagged user CRUD methods and refusal counting"
```

---

### Task 4: Extend `log_usage` to Accept prompt, status, error_detail

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:536-576` (log_usage method)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing test**

Add to `TestUsageTracking`:

```python
def test_log_usage_stores_prompt_and_status(self, tmp_path: Path) -> None:
    """GIVEN a database WHEN logging usage with prompt/status THEN stored."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    db.log_usage(
        "alice", "#test", "draw", "dall-e-3", 0, 0, 0.0,
        prompt="a cat in space", status="content_blocked",
        error_detail="content policy violation",
    )

    conn = db._connect()
    try:
        row = conn.execute(
            "SELECT prompt, status, error_detail FROM usage WHERE nick = ?",
            ("alice",),
        ).fetchone()
        assert row == ("a cat in space", "content_blocked", "content policy violation")
    finally:
        conn.close()

def test_log_usage_defaults_to_success(self, tmp_path: Path) -> None:
    """GIVEN a database WHEN logging usage without status THEN defaults to success."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01)

    conn = db._connect()
    try:
        row = conn.execute(
            "SELECT prompt, status, error_detail FROM usage WHERE nick = ?",
            ("alice",),
        ).fetchone()
        assert row == ("", "success", "")
    finally:
        conn.close()
```

**Step 2: Run tests to verify they fail**

Run: `make test -- -k "test_log_usage_stores_prompt or test_log_usage_defaults" -v`
Expected: FAIL — log_usage doesn't accept those parameters yet.

**Step 3: Update `log_usage` signature and SQL**

```python
def log_usage(
    self,
    nick: str,
    channel: str,
    command: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    prompt: str = "",
    status: str = "success",
    error_detail: str = "",
) -> None:
    conn = self._connect()
    try:
        conn.execute(
            "INSERT INTO usage "
            "(timestamp, nick, channel, command, model, prompt_tokens, "
            "completion_tokens, cost, prompt, status, error_detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                nick,
                channel,
                command,
                model,
                prompt_tokens,
                completion_tokens,
                cost,
                prompt,
                status,
                error_detail,
            ),
        )
        conn.commit()
    finally:
        conn.close()
```

**Step 4: Run all persistence tests to verify nothing broke**

Run: `make test -- -k test_persistence -v`
Expected: ALL PASS (existing tests use positional args and still work with defaults).

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: extend log_usage to store prompt, status, error_detail"
```

---

### Task 5: Config Values for Flag Threshold and Window

**Files:**
- Modify: `plugins/llm/src/llm/config.py`
- Modify: `plugins/llm/tests/conftest.py` (add to defaults)
- Test: `plugins/llm/tests/test_config.py`

**Step 1: Add config registrations**

At the end of `config.py`, before the closing section, add a new section:

```python
# ============================================================================
# Abuse Flagging
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "flagThreshold",
    registry.PositiveInteger(
        5,
        _("""Number of content safety refusals within flagWindow seconds that
        triggers automatic flagging of a user account. Set high to reduce
        false positives."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "flagWindow",
    registry.PositiveInteger(
        3600,
        _("""Time window in seconds for counting content safety refusals
        toward the auto-flag threshold. Default: 3600 (1 hour)."""),
    ),
)
```

**Step 2: Update test fixture defaults**

In `plugins/llm/tests/conftest.py`, add to the `defaults` dict in `make_registry_side_effect`:

```python
# Abuse flagging
"flagThreshold": 5,
"flagWindow": 3600,
```

**Step 3: Run make preflight to verify nothing broke**

Run: `make preflight`
Expected: PASS

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat: add flagThreshold and flagWindow config values"
```

---

### Task 6: Shared `_require_account` Helper on Plugin

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing tests**

Add a new test class in `test_plugin.py`:

```python
class TestRequireAccount:
    """Test _require_account NickServ gate helper."""

    def test_returns_account_when_identified(self, mocker, mock_irc):
        """GIVEN identified user WHEN _require_account THEN returns account name."""
        # ... setup plugin with patches ...
        mock_irc.state.nickToAccount.return_value = "alice_account"
        msg = mocker.MagicMock()
        msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, msg)
        assert result == "alice_account"

    def test_returns_none_and_errors_when_unidentified(self, mocker, mock_irc):
        """GIVEN unidentified user WHEN _require_account THEN returns None and sends error."""
        mock_irc.state.nickToAccount.return_value = None
        msg = mocker.MagicMock()
        msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, msg)
        assert result is None
        mock_irc.error.assert_called_once()

    def test_returns_none_on_key_error(self, mocker, mock_irc):
        """GIVEN nickToAccount raises KeyError WHEN _require_account THEN returns None."""
        mock_irc.state.nickToAccount.side_effect = KeyError("no such nick")
        msg = mocker.MagicMock()
        msg.prefix = "alice!user@host"

        result = plugin._require_account(mock_irc, msg)
        assert result is None
        mock_irc.error.assert_called_once()
```

Note: the exact test setup will follow the patterns in the existing `test_plugin.py` (use `plugin_init_patches` context manager from conftest to initialize the plugin).

**Step 2: Run tests to verify they fail**

Run: `make test -- -k TestRequireAccount -v`
Expected: FAIL — `_require_account` doesn't exist yet.

**Step 3: Implement `_require_account`**

Add to the plugin class, near other helper methods (around line 900, near `_get_identity`):

```python
def _require_account(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
    """Require NickServ identification. Returns account name or None.

    If the user is not identified, sends an error reply and returns None.
    Callers should ``return`` immediately when None is returned.
    """
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

**Step 4: Run tests to verify they pass**

Run: `make test -- -k TestRequireAccount -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: add shared _require_account NickServ helper"
```

---

### Task 7: Wire `_require_account` into `draw` and `animate`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1133-1278` (draw and animate methods)
- Test: `plugins/llm/tests/test_commands.py` and/or `plugins/llm/tests/test_animate.py`

**Step 1: Write the failing test for draw auth**

Add a test that verifies draw requires NickServ identification:

```python
def test_draw_requires_nickserv_identification(self, mocker, ...):
    """GIVEN unidentified user WHEN draw called THEN error about NickServ."""
    # Setup: nickToAccount returns None
    mock_irc.state.nickToAccount.return_value = None
    # Call draw
    plugin.draw(mock_irc, msg, [], "a cat")
    mock_irc.error.assert_called_once()
    assert "NickServ" in str(mock_irc.error.call_args)
```

**Step 2: Run test to verify it fails**

Run: `make test -- -k test_draw_requires_nickserv -v`
Expected: FAIL — draw currently doesn't check NickServ.

**Step 3: Add `_require_account` to draw**

At the top of the `draw` method body, after `_is_old_message` check:

```python
# Require NickServ identification
account = self._require_account(irc, msg)
if account is None:
    return
```

**Step 4: Refactor animate to use `_require_account`**

Replace the inline NickServ check in `animate` (lines 1226-1234) with:

```python
# Require NickServ identification
account = self._require_account(irc, msg)
if account is None:
    return

raw_nick = ircutils.nickFromHostmask(msg.prefix)
nick = self._resolve_nick_to_identity(irc, raw_nick)
```

**Step 5: Run all tests**

Run: `make test -v`
Expected: ALL PASS. Some existing tests may need `nickToAccount` mocked to return an account for draw tests that expect success.

**Step 6: Fix any broken tests**

If existing draw tests fail because they don't mock `nickToAccount`, update the fixtures to set `mock_irc.state.nickToAccount.return_value = "test_account"` for draw-related tests.

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat: require NickServ identification for draw command"
```

---

### Task 8: Add `_check_flagged` Helper and Wire into Commands

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing tests**

```python
class TestCheckFlagged:
    """Test _check_flagged pre-command gate."""

    def test_returns_true_for_flagged_user(self, ...):
        """GIVEN flagged account WHEN _check_flagged THEN returns True and sends error."""
        plugin.db.is_user_flagged.return_value = True
        result = plugin._check_flagged(mock_irc, msg, "alice")
        assert result is True
        mock_irc.error.assert_called_once()

    def test_returns_false_for_unflagged_user(self, ...):
        """GIVEN unflagged account WHEN _check_flagged THEN returns False."""
        plugin.db.is_user_flagged.return_value = False
        result = plugin._check_flagged(mock_irc, msg, "alice")
        assert result is False
        mock_irc.error.assert_not_called()

    def test_returns_false_for_none_account(self, ...):
        """GIVEN None account WHEN _check_flagged THEN returns False (unidentified users pass)."""
        result = plugin._check_flagged(mock_irc, msg, None)
        assert result is False
```

**Step 2: Run tests to verify they fail**

Run: `make test -- -k TestCheckFlagged -v`

**Step 3: Implement `_check_flagged`**

```python
def _check_flagged(
    self, irc: callbacks.Irc, msg: IrcMsg, account: str | None
) -> bool:
    """Check if a user account is flagged for abuse.

    Returns True (and sends error) if the user should be blocked.
    Returns False if the user is clear to proceed.
    Unidentified users (account=None) are not checked.
    """
    if account is None:
        return False
    if self.db.is_user_flagged(account):
        irc.error(_("Your account has been suspended. Contact a bot admin."))
        return True
    return False
```

**Step 4: Wire into commands**

- **draw/animate**: After `_require_account` returns account, call `_check_flagged(irc, msg, account)`. Return if True.
- **ask/code**: Attempt account resolution (try `irc.state.nickToAccount`, catch exceptions, default None). Call `_check_flagged`. Return if True. This does NOT require NickServ — unidentified users pass through.

**Step 5: Run all tests**

Run: `make test -v`
Expected: PASS

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat: add pre-command flag check to block suspended users"
```

---

### Task 9: Log All Outcomes with Prompt and Status

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (all command methods + `_store_context_and_log_usage`)
- Test: `plugins/llm/tests/test_commands.py`

This is the largest task. Every code path that returns to the user must log a usage row.

**Step 1: Write failing tests for failure logging**

Tests should verify that when a command errors, a usage row is still created with the appropriate status.

```python
def test_draw_content_blocked_logs_usage_with_status(self, ...):
    """GIVEN draw that hits content safety WHEN completed THEN usage logged as content_blocked."""
    # Mock image_generation to return result with error
    service.image_generation.return_value = ImageResult(
        content="blocked", error="content policy", ...
    )
    plugin.draw(irc, msg, [], "bad prompt")
    plugin.db.log_usage.assert_called_once()
    call_kwargs = plugin.db.log_usage.call_args
    assert call_kwargs[1]["status"] == "content_blocked"  # or positional
    assert "bad prompt" in call_kwargs[1]["prompt"]
```

**Step 2: Update `_store_context_and_log_usage`**

This method currently only logs on success. Change it to always log:

```python
def _store_context_and_log_usage(
    self, nick, channel, command, text, response, result, irc, msg,
) -> None:
    # Store context (only on success, unchanged)
    if result.error is None and self._get_context_enabled(channel):
        ...

    # Log usage — ALWAYS, with prompt and status
    status = "success" if result.error is None else "error"
    error_detail = (result.error or "")[:200]
    self.db.log_usage(
        nick, channel, command, result.model,
        result.prompt_tokens, result.completion_tokens, result.cost,
        prompt=text, status=status, error_detail=error_detail,
    )
```

**Step 3: Update draw command logging**

Replace the conditional `if result.error is None:` with unconditional logging. Determine status from result:

```python
status = "success" if result.error is None else "content_blocked"
# Distinguish content blocks from other errors based on result
self.db.log_usage(
    nick, channel, "draw", result.model,
    result.prompt_tokens, result.completion_tokens, result.cost,
    prompt=text, status=status,
    error_detail=(result.error or "")[:200],
)
```

**Step 4: Update animate command logging**

Same pattern as draw.

**Step 5: Log auth failures**

In draw/animate, when `_require_account` returns None:

```python
account = self._require_account(irc, msg)
if account is None:
    nick = ircutils.nickFromHostmask(msg.prefix)
    channel = self._get_channel(msg)
    self.db.log_usage(
        nick, channel, "draw", "", 0, 0, 0.0,
        prompt=text, status="auth_failure",
    )
    return
```

**Step 6: Log flag-blocked requests**

When `_check_flagged` returns True:

```python
if self._check_flagged(irc, msg, account):
    self.db.log_usage(
        nick, channel, command, "", 0, 0, 0.0,
        prompt=text, status="flagged_blocked",
    )
    return
```

**Step 7: Run all tests**

Run: `make test -v`
Expected: PASS. Some tests may need adjustment for the new `log_usage` calls (verify mock call counts, new arguments).

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat: log all command outcomes with prompt text and status"
```

---

### Task 10: Auto-Flag Logic

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

```python
class TestAutoFlag:
    """Test automatic flagging after content safety threshold."""

    def test_auto_flags_after_threshold(self, ...):
        """GIVEN 5 content blocks in window WHEN check runs THEN user auto-flagged."""
        plugin.db.count_recent_refusals.return_value = 5
        plugin.registryValue.side_effect = make_registry_side_effect({
            "flagThreshold": 5,
            "flagWindow": 3600,
        })
        plugin._maybe_auto_flag(irc, "alice", "#test")
        plugin.db.flag_user.assert_called_once_with("alice", mocker.ANY, auto_flagged=True)

    def test_no_flag_below_threshold(self, ...):
        """GIVEN 3 content blocks (below threshold 5) WHEN check runs THEN no flag."""
        plugin.db.count_recent_refusals.return_value = 3
        plugin._maybe_auto_flag(irc, "alice", "#test")
        plugin.db.flag_user.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `make test -- -k TestAutoFlag -v`

**Step 3: Implement `_maybe_auto_flag`**

```python
def _maybe_auto_flag(
    self, irc: callbacks.Irc, account: str, channel: str,
) -> None:
    """Check if a user should be auto-flagged based on recent refusals."""
    threshold = self.registryValue("flagThreshold")
    window = self.registryValue("flagWindow")
    since = time.time() - window
    count = self.db.count_recent_refusals(account, since)
    if count >= threshold:
        created = self.db.flag_user(
            account,
            f"{count} content blocks in {window}s",
            auto_flagged=True,
        )
        if created:
            self._notify_owners(
                irc,
                f"[LLM] Auto-flagged user {account}: "
                f"{count} content blocks in {window // 60}min. "
                f"Use %flagged to review.",
            )
```

**Step 4: Wire into usage logging**

After every `log_usage` call with `status="content_blocked"`, call `_maybe_auto_flag` if the user has an account:

```python
if status == "content_blocked" and account:
    self._maybe_auto_flag(irc, account, channel)
```

**Step 5: Run all tests**

Run: `make test -v`
Expected: PASS

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat: auto-flag users after content safety refusal threshold"
```

---

### Task 11: `_notify_owners` Helper

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

```python
class TestNotifyOwners:
    """Test IRC NOTICE to bot owners."""

    def test_sends_notice_to_online_owner(self, mocker, ...):
        """GIVEN online owner WHEN _notify_owners THEN NOTICE sent."""
        # Mock ircdb.users() to return one owner user
        # Mock irc.state.nicksToHostmasks to map a nick to that owner
        # Call _notify_owners
        # Assert irc.queueMsg called with ircmsgs.notice
        ...

    def test_no_notice_when_no_owner_online(self, mocker, ...):
        """GIVEN no online owners WHEN _notify_owners THEN no message sent."""
        ...
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `_notify_owners`**

```python
def _notify_owners(self, irc: callbacks.Irc, message: str) -> None:
    """Send IRC NOTICE to all online users with owner capability."""
    try:
        for user_id in ircdb.users():
            user = ircdb.users.getUser(user_id)
            if not user.checkCapability("owner"):
                continue
            for nick in irc.state.nicksToHostmasks:
                hostmask = irc.state.nicksToHostmasks[nick]
                if user.checkHostmask(hostmask):
                    irc.queueMsg(ircmsgs.notice(nick, message))
                    break
    except Exception:
        self.log.exception("Failed to notify owners")
```

Add import at top of plugin.py if not already present:

```python
from supybot import ircdb, ircmsgs
```

**Step 4: Run tests to verify they pass**

Run: `make test -- -k TestNotifyOwners -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: add _notify_owners IRC NOTICE helper"
```

---

### Task 12: Admin Commands — `%flag`, `%unflag`, `%flagged`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Write failing tests for each command**

```python
class TestFlagCommands:
    """Test %flag, %unflag, %flagged admin commands."""

    def test_flag_command_flags_user(self, ...):
        """GIVEN admin WHEN %flag alice spam THEN user flagged and confirmed."""
        mock_irc.state.nickToAccount.return_value = "alice_account"
        plugin.flag(mock_irc, msg, [], "alice", "spam")
        plugin.db.flag_user.assert_called_once()
        mock_irc.reply.assert_called_once()

    def test_flag_requires_identified_target(self, ...):
        """GIVEN unidentified target WHEN %flag THEN error."""
        mock_irc.state.nickToAccount.return_value = None
        plugin.flag(mock_irc, msg, [], "unknown_nick", "reason")
        mock_irc.error.assert_called_once()

    def test_unflag_command_unflags_user(self, ...):
        """GIVEN admin WHEN %unflag alice THEN user unflagged."""
        mock_irc.state.nickToAccount.return_value = "alice_account"
        plugin.db.unflag_user.return_value = True
        plugin.unflag(mock_irc, msg, [], "alice")
        mock_irc.reply.assert_called_once()

    def test_flagged_command_lists_flagged_users(self, ...):
        """GIVEN flagged users WHEN %flagged THEN list shown."""
        plugin.db.get_flagged_users.return_value = [
            FlaggedUserRow(1, "alice", 1000.0, "spam", 1, None, None),
        ]
        plugin.flagged(mock_irc, msg, [])
        mock_irc.reply.assert_called_once()

    def test_flagged_empty_shows_message(self, ...):
        """GIVEN no flagged users WHEN %flagged THEN shows 'none'."""
        plugin.db.get_flagged_users.return_value = []
        plugin.flagged(mock_irc, msg, [])
        mock_irc.reply.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `make test -- -k TestFlagCommands -v`

**Step 3: Implement the commands**

```python
def flag(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    nick: str,
    reason: str,
) -> None:
    """<nick> <reason>

    Flag a user account for abuse. Resolves nick to NickServ account.
    Flagged users are blocked from using bot commands.
    """
    raw_nick = nick
    try:
        account = irc.state.nickToAccount(raw_nick)
    except (KeyError, AttributeError):
        account = None
    if not account:
        irc.error(_("Cannot resolve %s to a NickServ account. "
                     "User must be online and identified.") % raw_nick)
        return

    created = self.db.flag_user(account, reason, auto_flagged=False)
    if created:
        admin_account = self._get_identity(irc, msg)
        self._notify_owners(
            irc,
            f"[LLM] {admin_account} flagged user {account}: {reason}",
        )
        irc.reply(_("Flagged %s (%s).") % (raw_nick, account), private=True)
    else:
        irc.reply(_("%s is already flagged.") % account, private=True)

flag = wrap(flag, ["admin", "nick", "text"])

def unflag(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    nick: str,
) -> None:
    """<nick>

    Remove the abuse flag from a user account.
    """
    raw_nick = nick
    try:
        account = irc.state.nickToAccount(raw_nick)
    except (KeyError, AttributeError):
        account = None
    if not account:
        irc.error(_("Cannot resolve %s to a NickServ account.") % raw_nick)
        return

    admin_account = self._get_identity(irc, msg)
    result = self.db.unflag_user(account, admin_account)
    if result:
        self._notify_owners(
            irc,
            f"[LLM] {admin_account} unflagged user {account}.",
        )
        irc.reply(_("Unflagged %s (%s).") % (raw_nick, account), private=True)
    else:
        irc.reply(_("%s is not currently flagged.") % account, private=True)

unflag = wrap(unflag, ["admin", "nick"])

def flagged(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
) -> None:
    """(takes no arguments)

    List all currently flagged user accounts.
    """
    users = self.db.get_flagged_users()
    if not users:
        irc.reply(_("No flagged users."), private=True)
        return

    lines = []
    for u in users:
        flag_type = "auto" if u.auto_flagged else "manual"
        lines.append(f"{u.account} ({flag_type}): {u.reason}")
    irc.reply(" | ".join(lines), private=True)

flagged = wrap(flagged, ["admin"])
```

**Step 4: Run all tests**

Run: `make test -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "feat: add %flag, %unflag, %flagged admin commands"
```

---

### Task 13: Final Integration Test and Preflight

**Files:**
- Test: `plugins/llm/tests/test_integration.py` (optional — add an integration test)
- All files touched

**Step 1: Write an integration test for the full flow**

```python
def test_auto_flag_full_flow(self, ...):
    """GIVEN user hitting content blocks WHEN threshold reached THEN flagged and blocked."""
    # 1. User makes 5 draw requests that all get content_blocked
    # 2. After 5th, auto-flag triggers
    # 3. 6th request is blocked by _check_flagged
    # 4. Admin runs %unflag
    # 5. User can make requests again
```

**Step 2: Run `make preflight`**

Run: `make preflight`
Expected: ALL GREEN — format, lint, typecheck, tests all pass.

**Step 3: Commit integration test**

```bash
git add plugins/llm/tests/test_integration.py
git commit -m "test: add integration test for auto-flag abuse flow"
```

**Step 4: Final verification**

Run: `make preflight`
Expected: ALL GREEN.

---

## Summary of Commits

| # | Commit | Files |
|---|--------|-------|
| 1 | `feat: add prompt, status, error_detail columns to usage table` | persistence.py, test_persistence.py |
| 2 | `feat: add flagged_users table for abuse tracking` | persistence.py, test_persistence.py |
| 3 | `feat: add flagged user CRUD methods and refusal counting` | persistence.py, test_persistence.py |
| 4 | `feat: extend log_usage to store prompt, status, error_detail` | persistence.py, test_persistence.py |
| 5 | `feat: add flagThreshold and flagWindow config values` | config.py, conftest.py |
| 6 | `feat: add shared _require_account NickServ helper` | plugin.py, test_plugin.py |
| 7 | `feat: require NickServ identification for draw command` | plugin.py, test_commands.py, test_animate.py |
| 8 | `feat: add pre-command flag check to block suspended users` | plugin.py, test_plugin.py |
| 9 | `feat: log all command outcomes with prompt text and status` | plugin.py, test_commands.py |
| 10 | `feat: auto-flag users after content safety refusal threshold` | plugin.py, test_plugin.py |
| 11 | `feat: add _notify_owners IRC NOTICE helper` | plugin.py, test_plugin.py |
| 12 | `feat: add %flag, %unflag, %flagged admin commands` | plugin.py, test_commands.py |
| 13 | `test: add integration test for auto-flag abuse flow` | test_integration.py |
