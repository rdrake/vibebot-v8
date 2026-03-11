"""Tests for SQLite persistence layer."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from llm.persistence import FlaggedUserRow, LLMDatabase, PendingTaskRow, ReminderRow, UsageRank


class TestDatabaseInit:
    """Test database initialization and schema creation."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        """GIVEN a path WHEN LLMDatabase is created THEN database file exists."""
        db_path = str(tmp_path / "test.db")
        LLMDatabase(db_path)
        assert Path(db_path).exists()

    def test_creates_reminders_table(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN reminders table exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='reminders'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_creates_usage_table(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN usage table exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='usage'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """GIVEN a database WHEN connected THEN WAL journal mode is active."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row is not None
            assert row[0] == "wal"
        finally:
            conn.close()

    def test_usage_table_has_prompt_column(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN usage table has prompt column."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            columns = conn.execute("PRAGMA table_info(usage)").fetchall()
            column_names = [col[1] for col in columns]
            assert "prompt" in column_names
        finally:
            conn.close()

    def test_usage_table_has_status_column(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN usage table has status column."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            columns = conn.execute("PRAGMA table_info(usage)").fetchall()
            column_names = [col[1] for col in columns]
            assert "status" in column_names
        finally:
            conn.close()

    def test_usage_table_has_error_detail_column(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN usage table has error_detail column."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            columns = conn.execute("PRAGMA table_info(usage)").fetchall()
            column_names = [col[1] for col in columns]
            assert "error_detail" in column_names
        finally:
            conn.close()

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

    def test_creates_conversations_table(self, tmp_path: Path) -> None:
        """GIVEN a new database WHEN initialized THEN conversations table exists."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """GIVEN an existing database WHEN opened again THEN no error."""
        db_path = str(tmp_path / "test.db")
        db1 = LLMDatabase(db_path)
        db1.save_reminder("evt1", "nick", "#chan", "msg", time.time() + 60)

        # Open the same database again — should not raise or lose data
        db2 = LLMDatabase(db_path)
        reminders = db2.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "evt1"


class TestSchemaMigration:
    """Test schema version migration from v1 to v2."""

    def test_migration_preserves_existing_usage_data(self, tmp_path: Path) -> None:
        """GIVEN a v1 database with usage data WHEN opened with v2 code THEN data preserved with defaults."""
        db_path = str(tmp_path / "test.db")

        # Manually create a v1 database (no prompt/status/error_detail columns)
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
            "INSERT INTO usage "
            "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1000000.0, "alice", "#general", "ask", "gpt-4", 100, 50, 0.01),
        )
        conn.commit()
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        # Open with LLMDatabase — should run v2 migration
        db = LLMDatabase(db_path)

        # Verify original data is preserved
        summary = db.get_usage_summary()
        assert summary.total_requests == 1
        assert summary.total_prompt_tokens == 100
        assert summary.total_completion_tokens == 50
        assert summary.total_cost == pytest.approx(0.01)

        # Verify new columns exist with defaults
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == ""  # prompt default
            assert row[1] == "success"  # status default
            assert row[2] == ""  # error_detail default
        finally:
            conn.close()

        # Verify flagged_users table was also created
        conn = db._connect()
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='flagged_users'"
            )
            assert cursor.fetchone() is not None
        finally:
            conn.close()


class TestReminderPersistence:
    """Test reminder CRUD operations."""

    def test_save_and_load_reminder(self, tmp_path: Path) -> None:
        """GIVEN a database WHEN a reminder is saved THEN it can be loaded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        fire_at = time.time() + 300
        row_id = db.save_reminder("test_event", "alice", "#general", "Check build", fire_at)

        assert isinstance(row_id, int)
        assert row_id > 0

        reminders = db.load_pending_reminders()
        assert len(reminders) == 1

        r = reminders[0]
        assert isinstance(r, ReminderRow)
        assert r.event_name == "test_event"
        assert r.nick == "alice"
        assert r.channel == "#general"
        assert r.message == "Check build"
        assert r.fire_at == fire_at

    def test_delete_reminder_returns_true(self, tmp_path: Path) -> None:
        """GIVEN a saved reminder WHEN deleted THEN returns True."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_reminder("evt1", "alice", "#chan", "msg", time.time() + 60)

        assert db.delete_reminder("evt1") is True

    def test_delete_nonexistent_reminder_returns_false(self, tmp_path: Path) -> None:
        """GIVEN no reminders WHEN deleting nonexistent event THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.delete_reminder("no_such_event") is False

    def test_load_excludes_reminders_older_than_24h(self, tmp_path: Path) -> None:
        """GIVEN a reminder >24h overdue WHEN loading pending THEN it is excluded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # fire_at was 25 hours ago — should be excluded
        old_fire_at = time.time() - (25 * 3600)
        db.save_reminder("old_event", "alice", "#chan", "old msg", old_fire_at)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 0

    def test_load_includes_reminders_less_than_24h_overdue(self, tmp_path: Path) -> None:
        """GIVEN a reminder <24h overdue WHEN loading pending THEN it is included."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # fire_at was 23 hours ago — should still be included for delivery on restart
        recent_fire_at = time.time() - (23 * 3600)
        db.save_reminder("recent_event", "alice", "#chan", "recent msg", recent_fire_at)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "recent_event"

    def test_load_orders_by_fire_at_ascending(self, tmp_path: Path) -> None:
        """GIVEN multiple reminders WHEN loading THEN ordered by fire_at ascending."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        db.save_reminder("later", "alice", "#chan", "later", now + 600)
        db.save_reminder("sooner", "alice", "#chan", "sooner", now + 60)
        db.save_reminder("middle", "alice", "#chan", "middle", now + 300)

        reminders = db.load_pending_reminders()
        assert len(reminders) == 3
        assert reminders[0].event_name == "sooner"
        assert reminders[1].event_name == "middle"
        assert reminders[2].event_name == "later"

    def test_delete_expired_reminders_only_removes_old(self, tmp_path: Path) -> None:
        """GIVEN old and new reminders WHEN deleting expired THEN only old ones removed."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        # Old reminder: fire_at was 25 hours ago
        db.save_reminder("old", "alice", "#chan", "old", now - (25 * 3600))
        # Recent reminder: fire_at is in the future
        db.save_reminder("new", "alice", "#chan", "new", now + 300)

        deleted = db.delete_expired_reminders()
        assert deleted == 1

        # Only the new reminder should remain
        reminders = db.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "new"

    def test_unique_event_name_constraint(self, tmp_path: Path) -> None:
        """GIVEN a saved reminder WHEN saving with same event_name THEN IntegrityError."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_reminder("dup_event", "alice", "#chan", "first", time.time() + 60)

        with pytest.raises(sqlite3.IntegrityError):
            db.save_reminder("dup_event", "bob", "#other", "second", time.time() + 120)


class TestUsageTracking:
    """Test usage logging and aggregation."""

    def test_log_usage_and_summarize(self, tmp_path: Path) -> None:
        """GIVEN logged usage WHEN summarizing THEN totals are correct."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#chan", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.02)

        summary = db.get_usage_summary()
        assert summary.total_requests == 2
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150
        assert summary.total_cost == pytest.approx(0.03)

    def test_summary_with_since_filter(self, tmp_path: Path) -> None:
        """GIVEN old and new usage WHEN summarizing with since THEN only counts recent."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # Insert an "old" record by manipulating time
        conn = db._connect()
        try:
            old_time = time.time() - 7200  # 2 hours ago
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#chan", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()

        # Insert a recent record
        db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.02)

        # Filter to only the last hour
        since = time.time() - 3600
        summary = db.get_usage_summary(since=since)
        assert summary.total_requests == 1
        assert summary.total_prompt_tokens == 200
        assert summary.total_completion_tokens == 100
        assert summary.total_cost == pytest.approx(0.02)

    def test_empty_summary_returns_zeros(self, tmp_path: Path) -> None:
        """GIVEN no usage records WHEN summarizing THEN returns zeros."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        summary = db.get_usage_summary()
        assert summary.total_requests == 0
        assert summary.total_prompt_tokens == 0
        assert summary.total_completion_tokens == 0
        assert summary.total_cost == pytest.approx(0.0)

    def test_usage_by_nick_sorted_by_cost(self, tmp_path: Path) -> None:
        """GIVEN usage from multiple nicks WHEN querying by nick THEN sorted by cost desc."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#chan", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.05)
        db.log_usage("charlie", "#chan", "ask", "gpt-4", 50, 25, 0.03)

        breakdown = db.get_usage_by_nick()
        assert len(breakdown) == 3
        assert breakdown[0].name == "bob"
        assert breakdown[0].total_cost == pytest.approx(0.05)
        assert breakdown[1].name == "charlie"
        assert breakdown[1].total_cost == pytest.approx(0.03)
        assert breakdown[2].name == "alice"
        assert breakdown[2].total_cost == pytest.approx(0.01)

    def test_usage_by_channel_sorted_by_cost(self, tmp_path: Path) -> None:
        """GIVEN usage in multiple channels WHEN querying by channel THEN sorted by cost desc."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("alice", "#dev", "ask", "gpt-4", 200, 100, 0.05)
        db.log_usage("bob", "#general", "code", "gpt-4", 50, 25, 0.02)

        breakdown = db.get_usage_by_channel()
        assert len(breakdown) == 2
        assert breakdown[0].name == "#dev"
        assert breakdown[0].total_cost == pytest.approx(0.05)
        assert breakdown[1].name == "#general"
        assert breakdown[1].total_cost == pytest.approx(0.03)

    def test_usage_by_nick_respects_limit(self, tmp_path: Path) -> None:
        """GIVEN many nicks WHEN querying with limit THEN only top N returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        for i in range(10):
            db.log_usage(f"user{i}", "#chan", "ask", "gpt-4", 100, 50, 0.01 * (i + 1))

        breakdown = db.get_usage_by_nick(limit=3)
        assert len(breakdown) == 3
        # Should be the top 3 by cost (user9, user8, user7)
        assert breakdown[0].name == "user9"
        assert breakdown[1].name == "user8"
        assert breakdown[2].name == "user7"


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
        db = LLMDatabase(str(tmp_path / "test.db"))
        errors: list[Exception] = []

        def write_usage(n: int) -> None:
            try:
                for _i in range(20):
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


class TestFilteredUsageSummary:
    """Test channel- and nick-filtered usage summaries."""

    def test_channel_summary_filters_to_channel(self, tmp_path: Path) -> None:
        """GIVEN usage in two channels WHEN querying one THEN only that channel counted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#general", "ask", "gpt-4", 200, 100, 0.02)
        db.log_usage("alice", "#dev", "ask", "gpt-4", 300, 150, 0.05)

        summary = db.get_usage_summary_for_channel("#general")
        assert summary.total_requests == 2
        assert summary.total_cost == pytest.approx(0.03)

    def test_channel_summary_with_since_filter(self, tmp_path: Path) -> None:
        """GIVEN old and new usage WHEN filtering by since THEN only recent counted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#test", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()
        db.log_usage("bob", "#test", "ask", "gpt-4", 200, 100, 0.02)

        since = time.time() - 3600
        summary = db.get_usage_summary_for_channel("#test", since=since)
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.02)

    def test_channel_summary_empty(self, tmp_path: Path) -> None:
        """GIVEN no usage WHEN querying channel THEN returns zeros."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        summary = db.get_usage_summary_for_channel("#empty")
        assert summary.total_requests == 0
        assert summary.total_cost == pytest.approx(0.0)

    def test_nick_summary_filters_to_nick(self, tmp_path: Path) -> None:
        """GIVEN usage from two nicks WHEN querying one THEN only that nick counted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#test", "ask", "gpt-4", 200, 100, 0.05)

        summary = db.get_usage_summary_for_nick("alice")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.01)

    def test_nick_summary_scoped_to_channel(self, tmp_path: Path) -> None:
        """GIVEN usage in two channels WHEN querying nick with channel THEN scoped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("alice", "#dev", "ask", "gpt-4", 200, 100, 0.05)

        summary = db.get_usage_summary_for_nick("alice", channel="#general")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.01)

    def test_nick_summary_with_since_and_channel(self, tmp_path: Path) -> None:
        """GIVEN old and new usage WHEN filtering by since and channel THEN both applied."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#test", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()
        db.log_usage("alice", "#test", "ask", "gpt-4", 200, 100, 0.02)
        db.log_usage("alice", "#other", "ask", "gpt-4", 300, 150, 0.03)

        since = time.time() - 3600
        summary = db.get_usage_summary_for_nick("alice", since=since, channel="#test")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.02)


class TestUsageRanking:
    """Test rank computation for channels and nicks."""

    def test_channel_rank_top(self, tmp_path: Path) -> None:
        """GIVEN channel with highest cost WHEN ranking THEN rank is 1."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#top", "ask", "gpt-4", 100, 50, 0.10)
        db.log_usage("bob", "#middle", "ask", "gpt-4", 100, 50, 0.05)
        db.log_usage("charlie", "#bottom", "ask", "gpt-4", 100, 50, 0.01)

        rank = db.get_channel_rank("#top")
        assert rank == UsageRank(rank=1, total=3)

    def test_channel_rank_middle(self, tmp_path: Path) -> None:
        """GIVEN channel with middle cost WHEN ranking THEN rank is 2."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#top", "ask", "gpt-4", 100, 50, 0.10)
        db.log_usage("bob", "#middle", "ask", "gpt-4", 100, 50, 0.05)
        db.log_usage("charlie", "#bottom", "ask", "gpt-4", 100, 50, 0.01)

        rank = db.get_channel_rank("#middle")
        assert rank == UsageRank(rank=2, total=3)

    def test_channel_rank_bottom(self, tmp_path: Path) -> None:
        """GIVEN channel with lowest cost WHEN ranking THEN rank is 3."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#top", "ask", "gpt-4", 100, 50, 0.10)
        db.log_usage("bob", "#middle", "ask", "gpt-4", 100, 50, 0.05)
        db.log_usage("charlie", "#bottom", "ask", "gpt-4", 100, 50, 0.01)

        rank = db.get_channel_rank("#bottom")
        assert rank == UsageRank(rank=3, total=3)

    def test_channel_rank_unknown(self, tmp_path: Path) -> None:
        """GIVEN channel with no usage WHEN ranking THEN rank is 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#known", "ask", "gpt-4", 100, 50, 0.10)

        rank = db.get_channel_rank("#unknown")
        assert rank == UsageRank(rank=0, total=1)

    def test_channel_rank_empty_db(self, tmp_path: Path) -> None:
        """GIVEN no usage data WHEN ranking THEN rank is 0 total is 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        rank = db.get_channel_rank("#any")
        assert rank == UsageRank(rank=0, total=0)

    def test_channel_rank_with_since(self, tmp_path: Path) -> None:
        """GIVEN old and new usage WHEN ranking with since THEN only recent counted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#old_top", "ask", "gpt-4", 100, 50, 1.00),
            )
            conn.commit()
        finally:
            conn.close()
        db.log_usage("bob", "#recent", "ask", "gpt-4", 100, 50, 0.01)

        since = time.time() - 3600
        rank = db.get_channel_rank("#recent", since=since)
        assert rank == UsageRank(rank=1, total=1)

    def test_nick_rank_top(self, tmp_path: Path) -> None:
        """GIVEN nick with highest cost WHEN ranking THEN rank is 1."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.10)
        db.log_usage("bob", "#test", "ask", "gpt-4", 100, 50, 0.05)
        db.log_usage("charlie", "#test", "ask", "gpt-4", 100, 50, 0.01)

        rank = db.get_nick_rank("alice")
        assert rank == UsageRank(rank=1, total=3)

    def test_nick_rank_unknown(self, tmp_path: Path) -> None:
        """GIVEN nick with no usage WHEN ranking THEN rank is 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.10)

        rank = db.get_nick_rank("unknown_user")
        assert rank == UsageRank(rank=0, total=1)

    def test_nick_rank_scoped_to_channel(self, tmp_path: Path) -> None:
        """GIVEN usage in multiple channels WHEN ranking with channel THEN scoped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # alice is top globally but not in #dev
        db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.50)
        db.log_usage("alice", "#dev", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#dev", "ask", "gpt-4", 100, 50, 0.10)

        rank = db.get_nick_rank("alice", channel="#dev")
        assert rank == UsageRank(rank=2, total=2)

    def test_nick_rank_empty_db(self, tmp_path: Path) -> None:
        """GIVEN no usage data WHEN ranking nick THEN rank is 0 total is 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        rank = db.get_nick_rank("anyone")
        assert rank == UsageRank(rank=0, total=0)


class TestMigrateNick:
    """Test nick-to-account migration for usage rows."""

    def test_migrate_updates_matching_rows(self, tmp_path: Path) -> None:
        """GIVEN usage under old nick WHEN migrating THEN rows updated to account."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("Rubin[F]", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("Rubin[F]", "#test", "code", "gpt-4", 200, 100, 0.02)

        count = db.migrate_nick("Rubin[F]", "Rubin")
        assert count == 2

        # Old nick should have no rows
        summary_old = db.get_usage_summary_for_nick("Rubin[F]")
        assert summary_old.total_requests == 0

        # Account should have all rows
        summary_new = db.get_usage_summary_for_nick("Rubin")
        assert summary_new.total_requests == 2
        assert summary_new.total_cost == pytest.approx(0.03)

    def test_migrate_case_insensitive(self, tmp_path: Path) -> None:
        """GIVEN usage under different casings WHEN migrating THEN all casings updated."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("rubin", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("RUBIN", "#test", "code", "gpt-4", 200, 100, 0.02)

        count = db.migrate_nick("Rubin", "RubinAccount")
        assert count == 2

        summary = db.get_usage_summary_for_nick("RubinAccount")
        assert summary.total_requests == 2

    def test_migrate_skips_already_migrated(self, tmp_path: Path) -> None:
        """GIVEN rows already under account WHEN migrating THEN those rows untouched."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("Rubin", "#test", "ask", "gpt-4", 100, 50, 0.01)  # already correct
        db.log_usage("Rubin[F]", "#test", "code", "gpt-4", 200, 100, 0.02)  # needs migration

        count = db.migrate_nick("Rubin[F]", "Rubin")
        assert count == 1

        summary = db.get_usage_summary_for_nick("Rubin")
        assert summary.total_requests == 2

    def test_migrate_returns_zero_when_nothing_to_migrate(self, tmp_path: Path) -> None:
        """GIVEN no matching rows WHEN migrating THEN returns zero."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01)

        count = db.migrate_nick("nonexistent", "SomeAccount")
        assert count == 0

    def test_migrate_idempotent(self, tmp_path: Path) -> None:
        """GIVEN already-migrated rows WHEN migrating again THEN zero updates."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("OldNick", "#test", "ask", "gpt-4", 100, 50, 0.01)

        first = db.migrate_nick("OldNick", "Account")
        assert first == 1

        second = db.migrate_nick("OldNick", "Account")
        assert second == 0

    def test_migrate_preserves_other_nicks(self, tmp_path: Path) -> None:
        """GIVEN usage from multiple nicks WHEN migrating one THEN others untouched."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01)
        db.log_usage("bob", "#test", "ask", "gpt-4", 200, 100, 0.02)

        db.migrate_nick("alice", "AliceAccount")

        # alice's rows migrated
        assert db.get_usage_summary_for_nick("AliceAccount").total_requests == 1
        # bob's rows untouched
        assert db.get_usage_summary_for_nick("bob").total_requests == 1


class TestPendingTasks:
    """Test pending task CRUD operations and claim/release semantics."""

    def test_save_and_load_pending_task(self, tmp_path: Path) -> None:
        """GIVEN a database WHEN a pending task is saved THEN it can be loaded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#general",
            is_channel=True,
            prompt_preview="What is the weather?",
            model="gpt-4",
            request_data='{"messages": []}',
            submitted_at=now,
            expires_at=now + 60,
            next_attempt_at=now,
        )

        assert isinstance(task_id, int)
        assert task_id > 0

        tasks = db.load_pending_tasks()
        assert len(tasks) == 1

        t = tasks[0]
        assert isinstance(t, PendingTaskRow)
        assert t.id == task_id
        assert t.task_type == "ask"
        assert t.nick == "alice"
        assert t.reply_target == "#general"
        assert t.is_channel == 1
        assert t.prompt_preview == "What is the weather?"
        assert t.model == "gpt-4"
        assert t.request_data == '{"messages": []}'
        assert t.attempt_count == 0
        assert t.last_error == ""

    def test_claim_due_tasks_sets_lease(self, tmp_path: Path) -> None:
        """GIVEN a due task WHEN claimed THEN claimed_until is set."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        db.save_pending_task(
            task_type="code",
            nick="bob",
            reply_target="#dev",
            is_channel=True,
            prompt_preview="Write a sort",
            model="gpt-4",
            request_data="{}",
            submitted_at=now - 10,
            expires_at=now + 50,
            next_attempt_at=now - 5,
        )

        claimed = db.claim_due_pending_tasks(now, limit=10, lease_seconds=120)
        assert len(claimed) == 1
        assert claimed[0].task_type == "code"

        # After claiming, the task should not be claimable again
        claimed_again = db.claim_due_pending_tasks(now, limit=10, lease_seconds=120)
        assert len(claimed_again) == 0

    def test_claim_skips_not_due_and_claimed(self, tmp_path: Path) -> None:
        """GIVEN tasks not yet due or already claimed WHEN claiming THEN skipped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # Not yet due (next_attempt_at in future)
        db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="future task",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now + 60,
        )

        claimed = db.claim_due_pending_tasks(now, limit=10, lease_seconds=120)
        assert len(claimed) == 0

    def test_release_increments_attempt_and_sets_backoff(self, tmp_path: Path) -> None:
        """GIVEN a claimed task WHEN released with increment THEN attempt_count bumped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = db.save_pending_task(
            task_type="draw",
            nick="charlie",
            reply_target="#art",
            is_channel=True,
            prompt_preview="a cat",
            model="dall-e-3",
            request_data='{"prompt": "a cat"}',
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        # Claim then release
        db.claim_due_pending_tasks(now, limit=1, lease_seconds=120)
        next_at = now + 30
        result = db.release_pending_task(task_id, next_at, "timeout", increment_attempt=True)
        assert result is True

        tasks = db.load_pending_tasks()
        assert len(tasks) == 1
        assert tasks[0].attempt_count == 1
        assert tasks[0].next_attempt_at == next_at
        assert tasks[0].last_error == "timeout"
        assert tasks[0].claimed_until == 0

    def test_release_without_increment_for_undeliverable_channel(self, tmp_path: Path) -> None:
        """GIVEN a claimed task WHEN released without increment THEN attempt_count unchanged."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = db.save_pending_task(
            task_type="ask",
            nick="dave",
            reply_target="#offline",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        db.claim_due_pending_tasks(now, limit=1, lease_seconds=120)
        db.release_pending_task(task_id, now + 30, "Channel not available", increment_attempt=False)

        tasks = db.load_pending_tasks()
        assert tasks[0].attempt_count == 0

    def test_delete_expired_returns_rows(self, tmp_path: Path) -> None:
        """GIVEN expired tasks WHEN deleting THEN returns expired rows and removes them."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # Expired task
        db.save_pending_task(
            task_type="ask",
            nick="eve",
            reply_target="#test",
            is_channel=True,
            prompt_preview="old question",
            model="gpt-4",
            request_data="{}",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )

        # Still valid task
        db.save_pending_task(
            task_type="code",
            nick="frank",
            reply_target="#test",
            is_channel=True,
            prompt_preview="new code",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        expired = db.delete_expired_pending_tasks(now)
        assert len(expired) == 1
        assert expired[0].nick == "eve"

        # Only the valid task remains
        remaining = db.load_pending_tasks()
        assert len(remaining) == 1
        assert remaining[0].nick == "frank"

    def test_survives_reopen(self, tmp_path: Path) -> None:
        """GIVEN saved pending task WHEN DB reopened THEN task loadable."""
        db_path = str(tmp_path / "test.db")
        now = time.time()

        db1 = LLMDatabase(db_path)
        db1.save_pending_task(
            task_type="animate",
            nick="grace",
            reply_target="grace",
            is_channel=False,
            prompt_preview="dancing cat",
            model="grok-imagine-video",
            request_data='{"request_id": "req-999"}',
            submitted_at=now,
            expires_at=now + 3600,
            next_attempt_at=now,
        )
        db1.close()

        db2 = LLMDatabase(db_path)
        tasks = db2.load_pending_tasks()
        assert len(tasks) == 1
        assert tasks[0].nick == "grace"
        assert tasks[0].task_type == "animate"
        assert tasks[0].is_channel == 0
        db2.close()


class TestDeliveryStatePersistence:
    """Test delivery state transitions and filtered queries for Phase 1b."""

    def _save_task(self, db, now, **overrides):
        """Helper to save a pending task with sensible defaults."""
        defaults = {
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello",
            "model": "gpt-4",
            "request_data": "{}",
            "submitted_at": now,
            "expires_at": now + 120,
            "next_attempt_at": now,
        }
        defaults.update(overrides)
        return db.save_pending_task(**defaults)

    def test_update_task_for_delivery_sets_ready(self, tmp_path: Path) -> None:
        """GIVEN a pending task WHEN update_task_for_delivery called THEN delivery_state='ready'."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = self._save_task(db, now)

        db.update_task_for_delivery(
            task_id,
            delivery_state="ready",
            result_payload='{"content": "hello world"}',
        )

        tasks = db.load_pending_tasks()
        assert len(tasks) == 1
        assert tasks[0].delivery_state == "ready"
        assert tasks[0].result_payload == '{"content": "hello world"}'

    def test_update_task_for_delivery_sets_failed_terminal(self, tmp_path: Path) -> None:
        """GIVEN a pending task WHEN provider fails terminally THEN delivery_state='ready' with failure reason."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = self._save_task(db, now)

        db.update_task_for_delivery(
            task_id,
            delivery_state="ready",
            result_payload='{"status": "failed_terminal", "reason": "auth error"}',
        )

        tasks = db.load_pending_tasks()
        assert tasks[0].delivery_state == "ready"
        assert "failed_terminal" in tasks[0].result_payload

    def test_claim_filters_by_delivery_state(self, tmp_path: Path) -> None:
        """GIVEN tasks with different delivery_states WHEN claiming with filter THEN only matching tasks returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # pending task (provider phase)
        self._save_task(db, now, nick="pending_nick", next_attempt_at=now - 5)
        # ready task (delivery phase)
        ready_id = self._save_task(db, now, nick="ready_nick", next_attempt_at=now - 5)
        db.update_task_for_delivery(ready_id, "ready", '{"content": "result"}')

        # Provider claim: should only get pending
        provider_claimed = db.claim_due_pending_tasks(
            now, limit=10, lease_seconds=120, delivery_state_filter="pending"
        )
        assert len(provider_claimed) == 1
        assert provider_claimed[0].nick == "pending_nick"

    def test_claim_delivery_ready_and_retrying(self, tmp_path: Path) -> None:
        """GIVEN ready and retrying tasks WHEN delivery claim THEN both returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        ready_id = self._save_task(db, now, nick="ready_nick", next_attempt_at=now - 5)
        db.update_task_for_delivery(ready_id, "ready", '{"content": "r1"}')

        retrying_id = self._save_task(db, now, nick="retrying_nick", next_attempt_at=now - 5)
        db.update_task_for_delivery(retrying_id, "retrying", '{"content": "r2"}')

        delivery_claimed = db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
        )
        assert len(delivery_claimed) == 2

    def test_update_delivery_attempt(self, tmp_path: Path) -> None:
        """GIVEN a ready task WHEN delivery fails THEN retrying with incremented attempt count."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = self._save_task(db, now)
        db.update_task_for_delivery(task_id, "ready", '{"content": "result"}')

        db.update_delivery_attempt(
            task_id,
            delivery_state="retrying",
            last_delivery_error="queueMsg failed",
            delivery_attempt_count=1,
            next_attempt_at=now + 15,
        )

        tasks = db.load_pending_tasks()
        assert tasks[0].delivery_state == "retrying"
        assert tasks[0].last_delivery_error == "queueMsg failed"
        assert tasks[0].delivery_attempt_count == 1
        assert tasks[0].next_attempt_at == now + 15

    def test_delivery_failed_not_auto_claimed(self, tmp_path: Path) -> None:
        """GIVEN a delivery_failed task WHEN claiming for delivery THEN not returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = self._save_task(db, now, next_attempt_at=now - 5)
        db.update_task_for_delivery(task_id, "delivery_failed", '{"content": "result"}')

        claimed = db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
        )
        assert len(claimed) == 0

    def test_delivery_claim_skips_exhausted_attempts(self, tmp_path: Path) -> None:
        """GIVEN retrying rows at/under cap WHEN claiming THEN exhausted rows are excluded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        retriable_id = self._save_task(db, now, nick="retriable", next_attempt_at=now - 5)
        db.update_task_for_delivery(retriable_id, "ready", '{"content": "r1"}')
        db.update_delivery_attempt(
            retriable_id,
            delivery_state="retrying",
            last_delivery_error="net",
            delivery_attempt_count=9,
            next_attempt_at=now - 5,
        )

        exhausted_id = self._save_task(db, now, nick="exhausted", next_attempt_at=now - 5)
        db.update_task_for_delivery(exhausted_id, "ready", '{"content": "r2"}')
        db.update_delivery_attempt(
            exhausted_id,
            delivery_state="retrying",
            last_delivery_error="net",
            delivery_attempt_count=10,
            next_attempt_at=now - 5,
        )

        claimed = db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
            max_delivery_attempts=10,
        )
        assert len(claimed) == 1
        assert claimed[0].nick == "retriable"

    def test_expired_only_deletes_pending_delivery_state(self, tmp_path: Path) -> None:
        """GIVEN expired tasks with delivery_state='ready' WHEN expiry sweep THEN NOT deleted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # Expired pending task — should be deleted
        self._save_task(
            db,
            now,
            nick="expired_pending",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )

        # Expired but delivery_state='ready' — should NOT be deleted
        ready_id = self._save_task(
            db,
            now,
            nick="expired_ready",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )
        db.update_task_for_delivery(ready_id, "ready", '{"content": "result"}')

        expired = db.delete_expired_pending_tasks(now)
        assert len(expired) == 1
        assert expired[0].nick == "expired_pending"

        remaining = db.load_pending_tasks()
        assert len(remaining) == 1
        assert remaining[0].nick == "expired_ready"


class TestSchemaV3Migration:
    """Test schema v3 migration adds delivery columns to pending_tasks."""

    def test_migration_from_v2_adds_delivery_columns(self, tmp_path: Path) -> None:
        """GIVEN a v2 database with pending_tasks WHEN opened with v3 code THEN new columns exist with defaults."""
        db_path = str(tmp_path / "test.db")
        now = time.time()

        # Create a v2-schema database manually
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT UNIQUE NOT NULL,
                nick TEXT NOT NULL, channel TEXT NOT NULL,
                message TEXT NOT NULL, fire_at REAL NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL, nick TEXT NOT NULL, channel TEXT NOT NULL,
                command TEXT NOT NULL, model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0.0,
                prompt TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'success',
                error_detail TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS pending_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL, nick TEXT NOT NULL,
                reply_target TEXT NOT NULL, is_channel INTEGER NOT NULL,
                prompt_preview TEXT NOT NULL, model TEXT NOT NULL,
                request_data TEXT NOT NULL DEFAULT '{}',
                submitted_at REAL NOT NULL, expires_at REAL NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                claimed_until REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS flagged_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account TEXT UNIQUE NOT NULL,
                flagged_at REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
                auto_flagged INTEGER NOT NULL DEFAULT 0,
                resolved_at REAL, resolved_by TEXT
            );
        """)
        # Insert a pending task with v2 schema
        conn.execute(
            "INSERT INTO pending_tasks "
            "(task_type, nick, reply_target, is_channel, prompt_preview, model, "
            "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
            "claimed_until, last_error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, '')",
            (
                "animate",
                "alice",
                "#test",
                1,
                "dancing cat",
                "grok-imagine-video",
                '{"request_id": "req-1"}',
                now,
                now + 3600,
                now,
            ),
        )
        conn.commit()
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        # Open with v3-capable LLMDatabase
        db = LLMDatabase(db_path)

        # Verify new columns exist with correct defaults
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT delivery_state, result_payload, last_delivery_error, "
                "delivery_attempt_count, origin_request_id FROM pending_tasks WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == "pending"  # delivery_state default
            assert row[1] == ""  # result_payload default
            assert row[2] == ""  # last_delivery_error default
            assert row[3] == 0  # delivery_attempt_count default
            assert row[4] == ""  # origin_request_id default
        finally:
            conn.close()

    def test_pending_task_row_includes_delivery_fields(self, tmp_path: Path) -> None:
        """GIVEN a v3 database WHEN loading pending tasks THEN PendingTaskRow has delivery fields."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        db.save_pending_task(
            task_type="ask",
            nick="bob",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 60,
            next_attempt_at=now,
        )

        tasks = db.load_pending_tasks()
        assert len(tasks) == 1
        t = tasks[0]
        assert t.delivery_state == "pending"
        assert t.result_payload == ""
        assert t.last_delivery_error == ""
        assert t.delivery_attempt_count == 0
        assert t.origin_request_id == ""

    def test_schema_version_is_4(self, tmp_path: Path) -> None:
        """GIVEN a fresh database WHEN opened THEN schema version is 4."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        conn = db._connect()
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            assert row is not None
            assert row[0] == 4
        finally:
            conn.close()


class TestGetNextDueTime:
    """Test get_next_due_time() for event-driven queue wakeups (Phase 2)."""

    def _save_task(self, db, now, **overrides):
        """Helper to save a pending task with sensible defaults."""
        defaults = {
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello",
            "model": "gpt-4",
            "request_data": "{}",
            "submitted_at": now,
            "expires_at": now + 120,
            "next_attempt_at": now,
        }
        defaults.update(overrides)
        return db.save_pending_task(**defaults)

    def test_empty_queue_returns_none(self, tmp_path: Path) -> None:
        """GIVEN no pending tasks WHEN get_next_due_time called THEN returns None."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.get_next_due_time() is None

    def test_returns_earliest_next_attempt_at(self, tmp_path: Path) -> None:
        """GIVEN multiple pending tasks WHEN get_next_due_time called THEN returns the earliest."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        self._save_task(db, now, nick="later", next_attempt_at=now + 60)
        self._save_task(db, now, nick="sooner", next_attempt_at=now + 10)
        self._save_task(db, now, nick="middle", next_attempt_at=now + 30)

        result = db.get_next_due_time()
        assert result == now + 10

    def test_excludes_claimed_tasks(self, tmp_path: Path) -> None:
        """GIVEN a claimed task WHEN get_next_due_time called THEN it is excluded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        self._save_task(db, now, nick="claimed", next_attempt_at=now + 5)
        db.claim_due_pending_tasks(now + 10, limit=10, lease_seconds=120)
        # The only task is now claimed — should return None
        assert db.get_next_due_time() is None

    def test_includes_lease_expired_claimed_tasks(self, tmp_path: Path) -> None:
        """GIVEN claim lease already expired WHEN get_next_due_time THEN task is considered."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()
        task_id = self._save_task(db, now, nick="leased", next_attempt_at=now + 5)

        # Claim the row first, then force claimed_until into the past.
        db.claim_due_pending_tasks(now + 10, limit=10, lease_seconds=120)
        conn = db._connect()
        try:
            conn.execute(
                "UPDATE pending_tasks SET claimed_until = ? WHERE id = ?",
                (time.time() - 1, task_id),
            )
            conn.commit()
        finally:
            conn.close()

        assert db.get_next_due_time() == pytest.approx(now + 5)

    def test_includes_pending_and_delivery_states(self, tmp_path: Path) -> None:
        """GIVEN tasks in pending, ready, and retrying states WHEN get_next_due_time THEN all considered."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        # pending task due later
        self._save_task(db, now, nick="pending", next_attempt_at=now + 60)
        # ready task due sooner
        ready_id = self._save_task(db, now, nick="ready", next_attempt_at=now + 20)
        db.update_task_for_delivery(ready_id, "ready", '{"content": "r1"}')
        # retrying task due soonest
        retrying_id = self._save_task(db, now, nick="retrying", next_attempt_at=now + 10)
        db.update_task_for_delivery(retrying_id, "retrying", '{"content": "r2"}')

        assert db.get_next_due_time() == now + 10

    def test_excludes_terminal_delivery_states(self, tmp_path: Path) -> None:
        """GIVEN tasks in delivery_failed state WHEN get_next_due_time THEN excluded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        now = time.time()

        failed_id = self._save_task(db, now, nick="failed", next_attempt_at=now + 5)
        db.update_task_for_delivery(failed_id, "delivery_failed", '{"content": "r1"}')

        assert db.get_next_due_time() is None


class TestLogUsageExtended:
    """Test the extended log_usage parameters (prompt, status, error_detail)."""

    def test_log_usage_stores_prompt_and_status(self, tmp_path: Path) -> None:
        """GIVEN new log_usage params WHEN logging with prompt/status/error_detail THEN stored."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage(
            "alice",
            "#test",
            "ask",
            "gpt-4",
            100,
            50,
            0.01,
            prompt="tell me a joke",
            status="content_blocked",
            error_detail="moderation filter triggered",
        )

        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == "tell me a joke"
            assert row[1] == "content_blocked"
            assert row[2] == "moderation filter triggered"
        finally:
            conn.close()

    def test_log_usage_defaults_to_success(self, tmp_path: Path) -> None:
        """GIVEN no new params WHEN logging usage THEN defaults applied."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.log_usage("bob", "#test", "code", "gpt-4", 200, 100, 0.02)

        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'bob'"
            ).fetchone()
            assert row is not None
            assert row[0] == ""
            assert row[1] == "success"
            assert row[2] == ""
        finally:
            conn.close()


class TestFlaggedUsers:
    """Test flagged user CRUD operations and refusal counting."""

    def test_flag_user_creates_record(self, tmp_path: Path) -> None:
        """GIVEN no flags WHEN flagging a user THEN record appears in get_flagged_users."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        result = db.flag_user("alice", "repeated abuse", auto_flagged=True)
        assert result is True

        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert isinstance(flagged[0], FlaggedUserRow)
        assert flagged[0].account == "alice"
        assert flagged[0].reason == "repeated abuse"
        assert flagged[0].auto_flagged == 1
        assert flagged[0].resolved_at is None
        assert flagged[0].resolved_by is None

    def test_flag_user_idempotent(self, tmp_path: Path) -> None:
        """GIVEN an already-flagged user WHEN flagging again THEN no-op, original reason preserved."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "first reason", auto_flagged=False)
        result = db.flag_user("alice", "second reason", auto_flagged=True)
        assert result is False

        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].reason == "first reason"
        assert flagged[0].auto_flagged == 0

    def test_is_user_flagged_returns_true(self, tmp_path: Path) -> None:
        """GIVEN a flagged user WHEN checking is_user_flagged THEN returns True."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "test", auto_flagged=False)
        assert db.is_user_flagged("alice") is True

    def test_is_user_flagged_returns_false_when_not_flagged(self, tmp_path: Path) -> None:
        """GIVEN no flags WHEN checking is_user_flagged THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.is_user_flagged("nobody") is False

    def test_is_user_flagged_returns_false_after_unflag(self, tmp_path: Path) -> None:
        """GIVEN a flagged then unflagged user WHEN checking THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "test", auto_flagged=False)
        db.unflag_user("alice", resolved_by="admin")
        assert db.is_user_flagged("alice") is False

    def test_unflag_sets_resolved_fields(self, tmp_path: Path) -> None:
        """GIVEN a flagged user WHEN unflagging THEN resolved_at and resolved_by are set."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "test", auto_flagged=True)
        before = time.time()
        result = db.unflag_user("alice", resolved_by="admin")
        after = time.time()
        assert result is True

        # Read the raw row to check resolved fields
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT resolved_at, resolved_by FROM flagged_users WHERE account = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] is not None
            assert before <= row[0] <= after
            assert row[1] == "admin"
        finally:
            conn.close()

    def test_unflag_nonexistent_returns_false(self, tmp_path: Path) -> None:
        """GIVEN no flags WHEN unflagging a user THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        result = db.unflag_user("nobody", resolved_by="admin")
        assert result is False

    def test_get_flagged_users_excludes_resolved(self, tmp_path: Path) -> None:
        """GIVEN two flagged users, one resolved WHEN listing THEN only active returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "reason a", auto_flagged=False)
        db.flag_user("bob", "reason b", auto_flagged=True)
        db.unflag_user("alice", resolved_by="admin")

        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].account == "bob"

    def test_reflag_after_unflag_creates_new_flag(self, tmp_path: Path) -> None:
        """GIVEN a flagged-then-unflagged user WHEN re-flagging THEN active again."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.flag_user("alice", "first offense", auto_flagged=False)
        db.unflag_user("alice", resolved_by="admin")
        assert db.is_user_flagged("alice") is False

        result = db.flag_user("alice", "second offense", auto_flagged=True)
        assert result is True
        assert db.is_user_flagged("alice") is True

        flagged = db.get_flagged_users()
        assert len(flagged) == 1
        assert flagged[0].account == "alice"
        assert flagged[0].reason == "second offense"
        assert flagged[0].auto_flagged == 1
        assert flagged[0].resolved_at is None


class TestConversationPersistence:
    """Test conversation persistence methods."""

    def test_save_and_load_conversation(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN load_conversations THEN it is returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        db.save_conversation("User1", "#Channel", messages, 1000.0)

        loaded = db.load_conversations()
        assert len(loaded) == 1
        nick, channel, msgs, last_activity = loaded[0]
        assert nick == "user1"
        assert channel == "#channel"
        assert msgs == messages
        assert last_activity == 1000.0

    def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """GIVEN an existing conversation WHEN saved again THEN it is replaced."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "first"}], 1000.0)
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "second"}], 2000.0)

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][2] == [{"role": "user", "content": "second"}]
        assert loaded[0][3] == 2000.0

    def test_save_lowercases_nick_and_channel(self, tmp_path: Path) -> None:
        """GIVEN mixed-case nick/channel WHEN saved THEN stored lowercased."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("UserName", "#MyChannel", [], 1000.0)

        loaded = db.load_conversations()
        assert loaded[0][0] == "username"
        assert loaded[0][1] == "#mychannel"

    def test_delete_conversation(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN deleted THEN load returns empty."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "hi"}], 1000.0)
        db.delete_conversation("user1", "#chan")

        loaded = db.load_conversations()
        assert len(loaded) == 0

    def test_delete_conversation_lowercases(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN deleted with different case THEN still deleted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [], 1000.0)
        db.delete_conversation("User1", "#Chan")

        assert len(db.load_conversations()) == 0

    def test_delete_all_conversations(self, tmp_path: Path) -> None:
        """GIVEN multiple conversations WHEN delete_all THEN all are removed."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [], 1000.0)
        db.save_conversation("user2", "#chan", [], 1000.0)
        db.delete_all_conversations()

        assert len(db.load_conversations()) == 0

    def test_load_skips_corrupt_json(self, tmp_path: Path) -> None:
        """GIVEN a row with invalid JSON WHEN load THEN it is skipped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # Insert valid row
        db.save_conversation("good", "#chan", [{"role": "user", "content": "ok"}], 1000.0)
        # Manually insert corrupt row
        conn = db._connect()
        conn.execute(
            "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
            "VALUES (?, ?, ?, ?)",
            ("bad", "#chan", "NOT VALID JSON{{{", 1000.0),
        )
        conn.commit()

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][0] == "good"
