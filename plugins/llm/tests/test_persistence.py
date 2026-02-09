"""Tests for SQLite persistence layer."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from llm.persistence import LLMDatabase, ReminderRow, UsageRank


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
