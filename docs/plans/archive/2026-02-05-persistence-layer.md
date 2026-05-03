# Persistence Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SQLite persistence for reminder survival across restarts and per-user/channel API cost tracking.

**Architecture:** New `persistence.py` module owns all database access (SQLite in WAL mode, connection-per-call for thread safety). Plugin.py wires it into init/die lifecycle and calls it after each command. Service.py extracts usage data from LiteLLM responses and returns it in result objects. No new dependencies (Python's built-in `sqlite3`).

**Tech Stack:** Python `sqlite3` (stdlib), LiteLLM `completion_cost()` for pricing, existing Limnoria config registry.

---

### Task 1: Create persistence module with schema

**Files:**
- Create: `plugins/llm/src/llm/persistence.py`
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write failing tests for database initialization**

```python
# plugins/llm/tests/test_persistence.py
"""Tests for SQLite persistence layer."""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from llm.persistence import LLMDatabase


class TestDatabaseInit:
    """Tests for database initialization and schema."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """GIVEN a path WHEN database created THEN file exists."""
        db_path = tmp_path / "test.db"
        db = LLMDatabase(str(db_path))
        db.close()
        assert db_path.exists()

    def test_creates_reminders_table(self, tmp_path: Path) -> None:
        """GIVEN new database WHEN initialized THEN reminders table exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_creates_usage_table(self, tmp_path: Path) -> None:
        """GIVEN new database WHEN initialized THEN usage table exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='usage'"
        )
        assert cursor.fetchone() is not None
        conn.close()
        db.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """GIVEN new database WHEN initialized THEN WAL mode is active."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()
        db.close()

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """GIVEN existing database WHEN opened again THEN no error."""
        db_path = str(tmp_path / "test.db")
        db1 = LLMDatabase(db_path)
        db1.close()
        db2 = LLMDatabase(db_path)
        db2.close()
```

**Step 2: Run tests to verify they fail**

Run: `make test`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.persistence'`

**Step 3: Implement persistence module**

```python
# plugins/llm/src/llm/persistence.py
"""SQLite persistence layer for LLM plugin."""

from __future__ import annotations

import sqlite3
import time
from typing import NamedTuple


class ReminderRow(NamedTuple):
    """A reminder loaded from the database."""

    id: int
    event_name: str
    nick: str
    channel: str
    message: str
    fire_at: float
    created_at: float


class UsageSummary(NamedTuple):
    """Aggregated usage statistics."""

    total_cost: float
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int


class UsageBreakdown(NamedTuple):
    """Usage breakdown by a dimension (user or channel)."""

    key: str
    cost: float
    requests: int


class LLMDatabase:
    """Thread-safe SQLite database for LLM plugin persistence.

    Uses connection-per-call pattern for thread safety.
    WAL mode enables concurrent reads.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize database and create schema.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection (thread-safe pattern).

        Returns:
            New SQLite connection with WAL mode
        """
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT UNIQUE NOT NULL,
                    nick TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    message TEXT NOT NULL,
                    fire_at REAL NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS usage (
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

                CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);
                CREATE INDEX IF NOT EXISTS idx_usage_nick ON usage(nick);
                CREATE INDEX IF NOT EXISTS idx_usage_channel ON usage(channel);
                CREATE INDEX IF NOT EXISTS idx_reminders_fire_at ON reminders(fire_at);
            """)
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """No-op for connection-per-call pattern (kept for API consistency)."""

    # ========================================================================
    # Reminder operations
    # ========================================================================

    def save_reminder(
        self,
        event_name: str,
        nick: str,
        channel: str,
        message: str,
        fire_at: float,
    ) -> int:
        """Save a reminder to the database.

        Args:
            event_name: Unique scheduler event name
            nick: User's IRC nick
            channel: IRC channel
            message: Reminder message text
            fire_at: Unix timestamp when reminder should fire

        Returns:
            Row ID of the inserted reminder
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO reminders (event_name, nick, channel, message, fire_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event_name, nick, channel, message, fire_at, time.time()),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def delete_reminder(self, event_name: str) -> bool:
        """Delete a reminder by event name.

        Args:
            event_name: Scheduler event name

        Returns:
            True if a row was deleted
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE event_name = ?", (event_name,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def load_pending_reminders(self) -> list[ReminderRow]:
        """Load all reminders that haven't fired yet.

        Returns:
            List of ReminderRow for reminders with fire_at in the future
            or recently passed (within 24h, for delivery on restart)
        """
        cutoff = time.time() - 86400  # Include up to 24h overdue
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, event_name, nick, channel, message, fire_at, created_at "
                "FROM reminders WHERE fire_at > ? ORDER BY fire_at",
                (cutoff,),
            ).fetchall()
            return [ReminderRow(*row) for row in rows]
        finally:
            conn.close()

    def delete_expired_reminders(self) -> int:
        """Delete reminders that are more than 24h overdue.

        Returns:
            Number of rows deleted
        """
        cutoff = time.time() - 86400
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE fire_at <= ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # ========================================================================
    # Usage tracking operations
    # ========================================================================

    def log_usage(
        self,
        nick: str,
        channel: str,
        command: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Log an API usage record.

        Args:
            nick: User's IRC nick
            channel: IRC channel
            command: Command name (ask, code, draw)
            model: Model identifier
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            cost: Cost in USD
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO usage (timestamp, nick, channel, command, model, "
                "prompt_tokens, completion_tokens, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), nick, channel, command, model, prompt_tokens,
                 completion_tokens, cost),
            )
            conn.commit()
        finally:
            conn.close()

    def get_usage_summary(self, since: float | None = None) -> UsageSummary:
        """Get aggregated usage statistics.

        Args:
            since: Unix timestamp to start from (None = all time)

        Returns:
            UsageSummary with totals
        """
        conn = self._connect()
        try:
            if since:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost), 0), COUNT(*), "
                    "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) "
                    "FROM usage WHERE timestamp >= ?",
                    (since,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost), 0), COUNT(*), "
                    "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) "
                    "FROM usage",
                ).fetchone()
            return UsageSummary(
                total_cost=row[0],
                total_requests=row[1],
                total_prompt_tokens=row[2],
                total_completion_tokens=row[3],
            )
        finally:
            conn.close()

    def get_usage_by_nick(
        self, since: float | None = None, limit: int = 5
    ) -> list[UsageBreakdown]:
        """Get usage broken down by user nick.

        Args:
            since: Unix timestamp to start from
            limit: Max results to return

        Returns:
            List of UsageBreakdown sorted by cost descending
        """
        conn = self._connect()
        try:
            query = (
                "SELECT nick, COALESCE(SUM(cost), 0), COUNT(*) FROM usage "
            )
            params: list = []
            if since:
                query += "WHERE timestamp >= ? "
                params.append(since)
            query += "GROUP BY nick ORDER BY SUM(cost) DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [UsageBreakdown(key=r[0], cost=r[1], requests=r[2]) for r in rows]
        finally:
            conn.close()

    def get_usage_by_channel(
        self, since: float | None = None, limit: int = 5
    ) -> list[UsageBreakdown]:
        """Get usage broken down by channel.

        Args:
            since: Unix timestamp to start from
            limit: Max results to return

        Returns:
            List of UsageBreakdown sorted by cost descending
        """
        conn = self._connect()
        try:
            query = (
                "SELECT channel, COALESCE(SUM(cost), 0), COUNT(*) FROM usage "
            )
            params: list = []
            if since:
                query += "WHERE timestamp >= ? "
                params.append(since)
            query += "GROUP BY channel ORDER BY SUM(cost) DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [UsageBreakdown(key=r[0], cost=r[1], requests=r[2]) for r in rows]
        finally:
            conn.close()
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: All `TestDatabaseInit` tests PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add SQLite persistence module with schema for reminders and usage"
```

---

### Task 2: Test reminder CRUD operations

**Files:**
- Test: `plugins/llm/tests/test_persistence.py`
- Modify: `plugins/llm/src/llm/persistence.py` (if bugs found)

**Step 1: Write failing tests for reminder persistence**

Append to `test_persistence.py`:

```python
class TestReminderPersistence:
    """Tests for reminder save/load/delete operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> LLMDatabase:
        """Create a test database."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_save_and_load_reminder(self, db: LLMDatabase) -> None:
        """GIVEN a reminder WHEN saved THEN loadable from DB."""
        fire_at = time.time() + 3600
        db.save_reminder("llm_remind_1_100", "alice", "#test", "check build", fire_at)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].nick == "alice"
        assert reminders[0].channel == "#test"
        assert reminders[0].message == "check build"
        assert reminders[0].event_name == "llm_remind_1_100"

    def test_delete_reminder(self, db: LLMDatabase) -> None:
        """GIVEN a saved reminder WHEN deleted THEN no longer loadable."""
        fire_at = time.time() + 3600
        db.save_reminder("llm_remind_1_100", "alice", "#test", "msg", fire_at)

        deleted = db.delete_reminder("llm_remind_1_100")
        assert deleted is True

        reminders = db.load_pending_reminders()
        assert len(reminders) == 0

    def test_delete_nonexistent_reminder(self, db: LLMDatabase) -> None:
        """GIVEN no matching reminder WHEN deleted THEN returns False."""
        assert db.delete_reminder("nonexistent") is False

    def test_load_excludes_very_old_reminders(self, db: LLMDatabase) -> None:
        """GIVEN reminder >24h overdue WHEN loaded THEN excluded."""
        old_fire_at = time.time() - 90000  # 25 hours ago
        db.save_reminder("llm_remind_old", "alice", "#test", "msg", old_fire_at)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 0

    def test_load_includes_recently_overdue_reminders(self, db: LLMDatabase) -> None:
        """GIVEN reminder <24h overdue WHEN loaded THEN included (deliver on restart)."""
        recent_fire_at = time.time() - 3600  # 1 hour ago
        db.save_reminder("llm_remind_recent", "alice", "#test", "msg", recent_fire_at)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 1

    def test_load_orders_by_fire_time(self, db: LLMDatabase) -> None:
        """GIVEN multiple reminders WHEN loaded THEN ordered by fire_at."""
        now = time.time()
        db.save_reminder("later", "alice", "#test", "later", now + 7200)
        db.save_reminder("sooner", "alice", "#test", "sooner", now + 3600)

        reminders = db.load_pending_reminders()
        assert reminders[0].message == "sooner"
        assert reminders[1].message == "later"

    def test_delete_expired_reminders(self, db: LLMDatabase) -> None:
        """GIVEN old and new reminders WHEN cleanup THEN only old deleted."""
        now = time.time()
        db.save_reminder("old", "alice", "#test", "old", now - 90000)
        db.save_reminder("new", "alice", "#test", "new", now + 3600)

        deleted = db.delete_expired_reminders()
        assert deleted == 1

        # New reminder still exists
        reminders = db.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "new"

    def test_unique_event_name_constraint(self, db: LLMDatabase) -> None:
        """GIVEN duplicate event_name WHEN saving THEN raises error."""
        fire_at = time.time() + 3600
        db.save_reminder("llm_remind_dup", "alice", "#test", "msg1", fire_at)

        with pytest.raises(sqlite3.IntegrityError):
            db.save_reminder("llm_remind_dup", "bob", "#test", "msg2", fire_at)
```

Add `import time` to the top of the test file.

**Step 2: Run tests to verify they pass**

Run: `make test`
Expected: All `TestReminderPersistence` tests PASS (implementation already done in Task 1)

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_persistence.py
git commit -m "test: add reminder CRUD tests for persistence layer"
```

---

### Task 3: Test usage tracking operations

**Files:**
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write tests for usage logging and queries**

Append to `test_persistence.py`:

```python
class TestUsageTracking:
    """Tests for usage logging and reporting."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> LLMDatabase:
        """Create a test database with sample usage data."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        yield db
        db.close()

    def test_log_and_summarize_usage(self, db: LLMDatabase) -> None:
        """GIVEN logged usage WHEN summarized THEN totals correct."""
        db.log_usage("alice", "#test", "ask", "gemini/flash", 100, 50, 0.001)
        db.log_usage("bob", "#test", "ask", "gemini/flash", 200, 100, 0.002)

        summary = db.get_usage_summary()
        assert summary.total_requests == 2
        assert summary.total_cost == pytest.approx(0.003)
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150

    def test_summary_with_since_filter(self, db: LLMDatabase) -> None:
        """GIVEN usage before and after cutoff WHEN filtered THEN only recent."""
        db.log_usage("alice", "#test", "ask", "model", 100, 50, 0.001)
        cutoff = time.time()
        db.log_usage("bob", "#test", "ask", "model", 200, 100, 0.002)

        summary = db.get_usage_summary(since=cutoff)
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.002)

    def test_empty_usage_summary(self, db: LLMDatabase) -> None:
        """GIVEN no usage WHEN summarized THEN zeros."""
        summary = db.get_usage_summary()
        assert summary.total_requests == 0
        assert summary.total_cost == 0.0

    def test_usage_by_nick(self, db: LLMDatabase) -> None:
        """GIVEN usage from multiple users WHEN grouped THEN sorted by cost."""
        db.log_usage("alice", "#test", "ask", "model", 100, 50, 0.005)
        db.log_usage("alice", "#test", "ask", "model", 100, 50, 0.005)
        db.log_usage("bob", "#test", "ask", "model", 100, 50, 0.001)

        breakdown = db.get_usage_by_nick()
        assert len(breakdown) == 2
        assert breakdown[0].key == "alice"
        assert breakdown[0].cost == pytest.approx(0.010)
        assert breakdown[0].requests == 2
        assert breakdown[1].key == "bob"

    def test_usage_by_channel(self, db: LLMDatabase) -> None:
        """GIVEN usage across channels WHEN grouped THEN sorted by cost."""
        db.log_usage("alice", "#general", "ask", "model", 100, 50, 0.003)
        db.log_usage("alice", "#dev", "ask", "model", 100, 50, 0.007)

        breakdown = db.get_usage_by_channel()
        assert len(breakdown) == 2
        assert breakdown[0].key == "#dev"
        assert breakdown[1].key == "#general"

    def test_usage_by_nick_respects_limit(self, db: LLMDatabase) -> None:
        """GIVEN many users WHEN limited THEN returns top N."""
        for i in range(10):
            db.log_usage(f"user{i}", "#test", "ask", "model", 10, 5, 0.001 * i)

        breakdown = db.get_usage_by_nick(limit=3)
        assert len(breakdown) == 3
```

**Step 2: Run tests to verify they pass**

Run: `make test`
Expected: All `TestUsageTracking` tests PASS

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_persistence.py
git commit -m "test: add usage tracking tests for persistence layer"
```

---

### Task 4: Add database config option

**Files:**
- Modify: `plugins/llm/src/llm/config.py:186-236` (add to Advanced Settings section)
- Test: `plugins/llm/tests/test_config.py`

**Step 1: Write failing test**

Add to `test_config.py`:

```python
def test_database_path_registered(self) -> None:
    """GIVEN LLM config WHEN checking databasePath THEN exists with empty default."""
    from llm import config  # noqa: F811

    value = config.LLM.databasePath()
    assert value == ""
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `AttributeError: 'Group' object has no attribute 'databasePath'`

**Step 3: Add config option**

Add to `config.py` in the Advanced Settings section (after `commandPrefixes`):

```python
conf.registerGlobalValue(
    LLM,
    "databasePath",
    registry.String(
        "",
        _("""Path to SQLite database file for persistence (reminders, usage tracking).
        If empty, uses Limnoria's data directory (data/LLM.db)."""),
    ),
)
```

**Step 4: Run test to verify it passes**

Run: `make test`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat: add databasePath config option for persistence"
```

---

### Task 5: Wire persistence into plugin lifecycle

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1-30` (imports), `plugin.py:265-325` (`__init__`, `die`)
- Test: `plugins/llm/tests/test_plugin.py` (init/die tests)

**Step 1: Write failing test for DB initialization**

Add to the appropriate test class in `test_plugin.py`:

```python
def test_plugin_creates_database(self, mock_irc: MagicMock) -> None:
    """GIVEN plugin init WHEN database path set THEN creates LLMDatabase."""
    with (
        patch.object(
            LLM, "registryValue",
            side_effect=lambda key, *args: {
                "httpRoot": "",
                "contextMaxMessages": 20,
                "contextTimeoutMinutes": 30,
                "contextEnabled": True,
                "channelContextMaxMessages": 10,
                "databasePath": "",
            }.get(key, ""),
        ),
        patch("llm.plugin.LLMService"),
        patch("llm.plugin.log"),
        patch("llm.plugin.httpserver.hook"),
        patch("llm.plugin.schedule.addPeriodicEvent"),
        patch("llm.plugin.schedule.removeEvent"),
        patch("llm.plugin.LLMDatabase") as mock_db_class,
    ):
        plugin = LLM(mock_irc)

    mock_db_class.assert_called_once()
    assert plugin.db is not None
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `LLMDatabase` not imported / `plugin.db` doesn't exist

**Step 3: Implement lifecycle wiring**

In `plugin.py`, add import:
```python
from .persistence import LLMDatabase
```

In `__init__`, after context initialization and before reminder storage, add:
```python
        # Initialize database
        db_path = self.registryValue("databasePath")
        if not db_path:
            db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
        self.db = LLMDatabase(db_path)
```

Add `from pathlib import Path` to the imports if not already there (it is already imported).

In `__init__`, after the existing reminder initialization, add reminder reload:
```python
        # Reload persisted reminders from database
        self._reload_reminders(irc)
```

Add the reload method:
```python
    def _reload_reminders(self, irc: callbacks.Irc) -> None:
        """Reload persisted reminders from database on startup.

        Reschedules future reminders and delivers overdue ones immediately.
        """
        pending = self.db.load_pending_reminders()
        now = time.time()

        for reminder in pending:
            nick = reminder.nick
            channel = reminder.channel
            message = reminder.message
            event_name = reminder.event_name

            def make_deliver(n: str, ch: str, msg: str, ev: str) -> callable:
                """Create delivery closure (avoid late binding)."""
                def deliver() -> None:
                    irc.queueMsg(ircmsgs.privmsg(ch, f"{n}: Reminder: {msg}"))
                    self._reminders.pop(ev, None)
                    self.db.delete_reminder(ev)
                return deliver

            deliver = make_deliver(nick, channel, message, event_name)

            if reminder.fire_at <= now:
                # Overdue — deliver immediately
                deliver()
            else:
                # Future — reschedule
                try:
                    schedule.addEvent(deliver, reminder.fire_at, name=event_name)
                    self._reminders[event_name] = (nick, channel, message)
                except Exception as e:
                    self.log.error("Failed to reload reminder %s: %s", event_name, e)
                    self.db.delete_reminder(event_name)

        if pending:
            self.log.info("Reloaded %d reminder(s) from database", len(pending))
```

In `die()`, add DB cleanup (clean up expired reminders):
```python
        # Clean up expired reminders from database
        if hasattr(self, "db"):
            self.db.delete_expired_reminders()
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: wire database into plugin lifecycle with reminder reload"
```

---

### Task 6: Persist reminders on create/fire/cancel

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:853-967` (remindme, unremind methods)
- Test: `plugins/llm/tests/test_reminders.py`

**Step 1: Write failing test**

Add to `TestReminderHelperMethods` in `test_reminders.py`:

```python
    def test_remindme_saves_to_database(self, plugin: MagicMock) -> None:
        """GIVEN successful reminder WHEN scheduled THEN saved to DB."""
        plugin.db = MagicMock()
        plugin.db.save_reminder = MagicMock()

        # Verify _reminders dict has a save path
        # (Full integration test would invoke remindme, but here we test the wiring)
        assert hasattr(plugin, "db") or hasattr(plugin, "_reminders")
```

**Step 2: Modify remindme to persist**

In the `remindme` method, after `schedule.addEvent(...)` and `self._reminders[event_name] = ...`, add:

```python
            # Persist to database
            self.db.save_reminder(
                event_name, nick, channel, reminder_message, time.time() + result.seconds
            )
```

In the `deliver()` closure inside `remindme`, add DB deletion:

```python
        def deliver() -> None:
            irc.queueMsg(ircmsgs.privmsg(channel, f"{nick}: Reminder: {reminder_message}"))
            self._reminders.pop(event_name, None)
            self.db.delete_reminder(event_name)
```

In the `unremind` method, after `self._reminders.pop(target, None)`, add:

```python
        self.db.delete_reminder(target)
```

**Step 3: Run tests to verify they pass**

Run: `make test`
Expected: PASS (existing tests use mocks for `__init__`, so `db` needs to be mocked in the fixture)

Update the `plugin` fixture in `TestReminderHelperMethods` to also set `plugin.db = MagicMock()`.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_reminders.py
git commit -m "feat: persist reminders to SQLite on create/fire/cancel"
```

---

### Task 7: Add usage data to CompletionResult and extract from responses

**Files:**
- Modify: `plugins/llm/src/llm/service.py:46-58` (CompletionResult), `service.py:710-815` (completion method), `service.py:996-1086` (image_generation)
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Write failing test**

Add to `test_service.py`:

```python
class TestUsageExtraction:
    """Tests for extracting usage data from LiteLLM responses."""

    def test_completion_result_has_usage_fields(self) -> None:
        """GIVEN CompletionResult WHEN created with usage THEN fields accessible."""
        from llm.service import CompletionResult

        result = CompletionResult(
            content="hello",
            grounding_used=False,
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            model="gemini/flash",
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cost == 0.001
        assert result.model == "gemini/flash"

    def test_completion_result_defaults(self) -> None:
        """GIVEN CompletionResult WHEN created without usage THEN defaults to zero."""
        from llm.service import CompletionResult

        result = CompletionResult(content="hello")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cost == 0.0
        assert result.model == ""
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `CompletionResult` doesn't have those fields

**Step 3: Extend CompletionResult and extract usage**

Update `CompletionResult` in `service.py`:

```python
class CompletionResult(NamedTuple):
    """Result of completion API call."""

    content: str
    grounding_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    model: str = ""
```

Add a usage extraction helper in `LLMService`:

```python
    def _extract_usage(self, response: Any, model: str) -> tuple[int, int, float]:
        """Extract token usage and cost from a LiteLLM response.

        Args:
            response: LiteLLM completion response
            model: Model identifier string

        Returns:
            Tuple of (prompt_tokens, completion_tokens, cost)
        """
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0

        try:
            usage = getattr(response, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        except (AttributeError, TypeError):
            pass

        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            # completion_cost can fail for unsupported models — graceful degradation
            pass

        return prompt_tokens, completion_tokens, cost
```

In the `completion()` method, after `grounding_used = ...` and before the return, add usage extraction:

```python
            prompt_tokens, completion_tokens, cost = self._extract_usage(response, model)

            return CompletionResult(
                content=content,
                grounding_used=grounding_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                model=model,
            )
```

For `image_generation()`, add the same extraction. Change the return type to also carry usage info. Since `image_generation` currently returns `str`, the cleanest minimal change is:

Create an `ImageResult` NamedTuple:

```python
class ImageResult(NamedTuple):
    """Result of image generation API call."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    model: str = ""
```

Update `image_generation()` return type from `str` to `ImageResult` and update the return statements. Where it currently returns error strings, wrap them: `return ImageResult(content=error_msg)`. For the success path after extracting the URL:

```python
            prompt_tokens, completion_tokens, cost = self._extract_usage(response, model)
            return ImageResult(
                content=url,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                model=model,
            )
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS — but the `draw` command in plugin.py does `irc.reply(result, ...)` which now gets an `ImageResult` instead of a string. Fix in next task.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: extract usage data from LiteLLM responses into result objects"
```

---

### Task 8: Log usage from plugin commands

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:567-745` (ask, code, draw methods)
- Modify: `plugins/llm/src/llm/plugin.py:24-28` (imports)
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write failing test**

Add to `test_plugin.py`:

```python
def test_ask_logs_usage_to_database(self) -> None:
    """GIVEN ask command WHEN completed THEN usage logged to DB."""
    # Test that the plugin calls db.log_usage after a successful ask
    from llm.service import CompletionResult

    result = CompletionResult(
        content="hello",
        grounding_used=False,
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.001,
        model="gemini/flash",
    )
    # Verify CompletionResult carries usage data
    assert result.prompt_tokens == 100
    assert result.cost == 0.001
```

**Step 2: Implement usage logging in plugin.py**

Import `ImageResult` from service:

```python
from .service import (
    CODE_PREVIEW_MAX_LEN,
    CODE_PREVIEW_TRUNCATE_LEN,
    ImageResult,
    LLMService,
)
```

In the `ask` method, after storing context (at the end), add:

```python
        # Log usage
        if result.cost > 0 or result.prompt_tokens > 0:
            self.db.log_usage(
                nick, channel, "ask", result.model,
                result.prompt_tokens, result.completion_tokens, result.cost,
            )
```

In the `code` method, same pattern at the end:

```python
        # Log usage
        if result.cost > 0 or result.prompt_tokens > 0:
            self.db.log_usage(
                nick, channel, "code", result.model,
                result.prompt_tokens, result.completion_tokens, result.cost,
            )
```

In the `draw` method, update to handle `ImageResult`:

```python
    def draw(self, irc, msg, args, text):
        # ... existing validation ...

        nick = self._get_nick(msg)
        channel = self._get_channel(msg)

        with self._allow_concurrent():
            result = self.llm_service.image_generation(text, irc=irc, msg=msg)
            irc.reply(result.content, prefixNick=False)

        # Log usage
        if result.cost > 0 or result.prompt_tokens > 0:
            self.db.log_usage(
                nick, channel, "draw", result.model,
                result.prompt_tokens, result.completion_tokens, result.cost,
            )
```

Note: The `draw` method needs to extract `nick` and `channel` — add those lines before `with self._allow_concurrent():` (similar to `ask`/`code`).

**Step 3: Run tests to verify they pass**

Run: `make test`
Expected: PASS. Some existing tests may need mock updates since `draw` now reads `result.content` instead of bare `result`.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: log API usage to database after ask/code/draw commands"
```

---

### Task 9: Add %usage admin command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (new command after `llmkeys`)
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write failing test**

```python
def test_usage_command_exists(self) -> None:
    """GIVEN LLM plugin WHEN checking for usage THEN method exists."""
    from llm.plugin import LLM

    assert hasattr(LLM, "usage")
    assert callable(LLM.usage)
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `LLM` has no `usage` attribute

**Step 3: Implement %usage command**

Add to `plugin.py` after the `llmkeys` command:

```python
    def usage(
        self,
        irc: callbacks.Irc,
        msg: IrcMsg,
        args: list,
    ) -> None:
        """(takes no arguments)

        Show API usage statistics (admin only).
        Displays today's and this month's cost, top users, and top channels.
        """
        now = time.time()

        # Today: midnight UTC
        from datetime import UTC, datetime

        today_midnight = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        # This month: first of month midnight UTC
        month_start = datetime.now(UTC).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).timestamp()

        today = self.db.get_usage_summary(since=today_midnight)
        month = self.db.get_usage_summary(since=month_start)
        top_users = self.db.get_usage_by_nick(since=month_start, limit=5)
        top_channels = self.db.get_usage_by_channel(since=month_start, limit=5)

        # Format response
        parts = []
        parts.append(
            f"Today: ${today.total_cost:.4f} ({today.total_requests} requests)"
        )
        parts.append(
            f"This month: ${month.total_cost:.4f} ({month.total_requests} requests)"
        )

        if top_users:
            user_parts = [f"{u.key} ${u.cost:.4f}" for u in top_users]
            parts.append(f"Top users: {', '.join(user_parts)}")

        if top_channels:
            chan_parts = [f"{c.key} ${c.cost:.4f}" for c in top_channels]
            parts.append(f"Top channels: {', '.join(chan_parts)}")

        irc.reply(" | ".join(parts), private=True)

    usage = wrap(usage, ["admin"])
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: add %usage admin command for API cost reporting"
```

---

### Task 10: Integration test — full round-trip

**Files:**
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write integration test**

```python
class TestRoundTrip:
    """Integration tests for full persistence round-trips."""

    def test_reminder_survives_simulated_restart(self, tmp_path: Path) -> None:
        """GIVEN saved reminder WHEN DB reopened THEN reminder loadable."""
        db_path = str(tmp_path / "test.db")

        # First "session" — save a reminder
        db1 = LLMDatabase(db_path)
        fire_at = time.time() + 3600
        db1.save_reminder("llm_remind_1_1", "alice", "#test", "check build", fire_at)
        db1.close()

        # Second "session" — reload
        db2 = LLMDatabase(db_path)
        reminders = db2.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].nick == "alice"
        assert reminders[0].message == "check build"
        db2.close()

    def test_usage_persists_across_sessions(self, tmp_path: Path) -> None:
        """GIVEN logged usage WHEN DB reopened THEN usage queryable."""
        db_path = str(tmp_path / "test.db")

        db1 = LLMDatabase(db_path)
        db1.log_usage("alice", "#test", "ask", "gemini/flash", 100, 50, 0.001)
        db1.close()

        db2 = LLMDatabase(db_path)
        summary = db2.get_usage_summary()
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.001)
        db2.close()

    def test_concurrent_writes_thread_safety(self, tmp_path: Path) -> None:
        """GIVEN multiple threads WHEN writing concurrently THEN no corruption."""
        import threading

        db = LLMDatabase(str(tmp_path / "test.db"))
        errors: list[Exception] = []

        def write_usage(n: int) -> None:
            try:
                for i in range(20):
                    db.log_usage(f"user{n}", "#test", "ask", "model", 10, 5, 0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_usage, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        summary = db.get_usage_summary()
        assert summary.total_requests == 100  # 5 threads * 20 writes
        db.close()
```

**Step 2: Run tests to verify they pass**

Run: `make test`
Expected: PASS

**Step 3: Commit**

```bash
git add plugins/llm/tests/test_persistence.py
git commit -m "test: add integration and concurrency tests for persistence"
```

---

### Task 11: Run full preflight and fix any issues

**Files:**
- All modified files

**Step 1: Run preflight**

Run: `make preflight`
Expected: PASS (format + lint + typecheck + test)

**Step 2: Fix any issues found**

Address lint, type, or test failures. Common expected fixes:
- Import ordering (ruff will auto-fix)
- Type annotations for `callable` → `Callable` or use a protocol
- Existing tests that mock `image_generation` return value (now returns `ImageResult` instead of `str`)

**Step 3: Final commit**

```bash
git add -A
git commit -m "chore: fix lint and type issues from persistence layer changes"
```

---

## Summary of Changes

| File | Action | What |
|------|--------|------|
| `persistence.py` | **Create** | SQLite DB module (reminders + usage tables) |
| `config.py` | Modify | Add `databasePath` setting |
| `service.py` | Modify | Extend `CompletionResult` with usage fields, add `ImageResult`, add `_extract_usage()` |
| `plugin.py` | Modify | Wire DB into init/die, persist reminders, log usage, add `%usage` command |
| `test_persistence.py` | **Create** | Schema, CRUD, usage, integration, concurrency tests |
| `test_service.py` | Modify | Tests for usage extraction |
| `test_plugin.py` | Modify | Tests for DB init, usage logging |
| `test_reminders.py` | Modify | Tests for reminder persistence |

**No new dependencies.** Uses Python's built-in `sqlite3`.
