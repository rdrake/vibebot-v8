"""SQLite persistence layer for LLM plugin.

Provides thread-safe database operations for reminders and usage tracking
using a connection-per-call pattern with WAL mode.
"""

from __future__ import annotations

import sqlite3
import time
from typing import NamedTuple

# Schema version for future migrations
SCHEMA_VERSION = 1

# Reminders older than 24 hours past their fire_at are considered expired
EXPIRY_THRESHOLD_SECONDS = 86400  # 24 hours


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

    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float


class UsageBreakdown(NamedTuple):
    """Usage statistics grouped by a dimension (nick or channel)."""

    name: str
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float


class UsageRank(NamedTuple):
    """Rank position within a leaderboard.

    rank=0 means the entry has no usage data; rank=1 is the top spender.
    """

    rank: int  # 1-based position, 0 = no data
    total: int  # total entries in the leaderboard


class LLMDatabase:
    """SQLite database for LLM plugin persistence.

    Uses a connection-per-call pattern for thread safety: each public method
    opens its own connection, executes the query, and closes it. WAL mode is
    enabled on every connection for concurrent read performance.
    """

    def __init__(self, db_path: str) -> None:
        """Create or open the database and run schema migration.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with WAL mode enabled.

        Returns:
            A new sqlite3.Connection with WAL journal mode.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate(self) -> None:
        """Run schema migration to create tables and indexes."""
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
                CREATE INDEX IF NOT EXISTS idx_reminders_fire_at
                    ON reminders(fire_at);

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
                CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                    ON usage(timestamp);
                CREATE INDEX IF NOT EXISTS idx_usage_nick
                    ON usage(nick);
                CREATE INDEX IF NOT EXISTS idx_usage_channel
                    ON usage(channel);
            """)
            conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        """Close the database (no-op for connection-per-call pattern).

        Kept for API consistency so callers can treat this like a resource
        that needs cleanup.
        """

    # ------------------------------------------------------------------
    # Reminder operations
    # ------------------------------------------------------------------

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
            event_name: Unique identifier for the reminder event.
            nick: IRC nick that created the reminder.
            channel: IRC channel the reminder was created in.
            message: Reminder message text.
            fire_at: Unix timestamp when the reminder should fire.

        Returns:
            The row ID of the inserted reminder.

        Raises:
            sqlite3.IntegrityError: If event_name already exists.
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
            event_name: Unique identifier for the reminder event.

        Returns:
            True if a reminder was deleted, False if not found.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE event_name = ?",
                (event_name,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def load_pending_reminders(self) -> list[ReminderRow]:
        """Load reminders that are still pending delivery.

        Returns reminders whose fire_at is within the last 24 hours or in the
        future. This allows delivery of slightly overdue reminders (e.g., after
        a bot restart) while excluding very old ones.

        Returns:
            List of ReminderRow ordered by fire_at ascending.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
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
        """Delete reminders that are more than 24 hours overdue.

        Returns:
            Number of reminders deleted.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE fire_at <= ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Usage operations
    # ------------------------------------------------------------------

    def log_usage(
        self,
        nick: str,
        channel: str,
        command: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        """Log a usage event.

        Args:
            nick: IRC nick that triggered the command.
            channel: IRC channel or PM target.
            command: Command name (ask, code, draw).
            model: Model identifier used.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
            cost: Estimated cost in USD.
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    nick,
                    channel,
                    command,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_usage_summary(self, since: float | None = None) -> UsageSummary:
        """Get aggregated usage statistics.

        Args:
            since: Optional Unix timestamp to filter results (only include
                   records after this time). If None, includes all records.

        Returns:
            UsageSummary with totals for requests, tokens, and cost.
        """
        conn = self._connect()
        try:
            if since is not None:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    "FROM usage WHERE timestamp >= ?",
                    (since,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    "FROM usage",
                ).fetchone()
            if row is None:
                return UsageSummary(
                    total_requests=0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    total_cost=0.0,
                )
            return UsageSummary(
                total_requests=row[0],
                total_prompt_tokens=row[1],
                total_completion_tokens=row[2],
                total_cost=row[3],
            )
        finally:
            conn.close()

    def get_usage_by_nick(self, since: float | None = None, limit: int = 5) -> list[UsageBreakdown]:
        """Get usage statistics grouped by nick, sorted by cost descending.

        Args:
            since: Optional Unix timestamp filter.
            limit: Maximum number of results to return.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        return self._get_usage_by_dimension("nick", since, limit)

    def get_usage_by_channel(
        self, since: float | None = None, limit: int = 5
    ) -> list[UsageBreakdown]:
        """Get usage statistics grouped by channel, sorted by cost descending.

        Args:
            since: Optional Unix timestamp filter.
            limit: Maximum number of results to return.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        return self._get_usage_by_dimension("channel", since, limit)

    def _get_usage_by_dimension(
        self, dimension: str, since: float | None, limit: int
    ) -> list[UsageBreakdown]:
        """Get usage grouped by a dimension (nick or channel).

        Args:
            dimension: Column name to group by ("nick" or "channel").
            since: Optional Unix timestamp filter.
            limit: Maximum number of results.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        assert dimension in ("nick", "channel"), f"Invalid dimension: {dimension}"
        # dimension is always a hardcoded column name from our own code,
        # never user input, so string interpolation is safe here.
        conn = self._connect()
        try:
            if since is not None:
                rows = conn.execute(
                    f"SELECT {dimension}, COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    f"COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    f"FROM usage WHERE timestamp >= ? "
                    f"GROUP BY {dimension} ORDER BY SUM(cost) DESC LIMIT ?",
                    (since, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {dimension}, COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    f"COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    f"FROM usage "
                    f"GROUP BY {dimension} ORDER BY SUM(cost) DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                UsageBreakdown(
                    name=row[0],
                    total_requests=row[1],
                    total_prompt_tokens=row[2],
                    total_completion_tokens=row[3],
                    total_cost=row[4],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_usage_summary_for_channel(
        self, channel: str, since: float | None = None
    ) -> UsageSummary:
        """Get aggregated usage statistics for a specific channel.

        Args:
            channel: IRC channel name to filter by.
            since: Optional Unix timestamp filter.

        Returns:
            UsageSummary with totals for the given channel.
        """
        conn = self._connect()
        try:
            if since is not None:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    "FROM usage WHERE channel = ? AND timestamp >= ?",
                    (channel, since),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                    "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                    "FROM usage WHERE channel = ?",
                    (channel,),
                ).fetchone()
            if row is None:
                return UsageSummary(0, 0, 0, 0.0)
            return UsageSummary(
                total_requests=row[0],
                total_prompt_tokens=row[1],
                total_completion_tokens=row[2],
                total_cost=row[3],
            )
        finally:
            conn.close()

    def get_usage_summary_for_nick(
        self, nick: str, since: float | None = None, channel: str | None = None
    ) -> UsageSummary:
        """Get aggregated usage statistics for a specific nick.

        Args:
            nick: IRC nick to filter by.
            since: Optional Unix timestamp filter.
            channel: Optional channel to further scope the query.

        Returns:
            UsageSummary with totals for the given nick (optionally in a channel).
        """
        conn = self._connect()
        try:
            conditions = ["nick = ?"]
            params: list[object] = [nick]
            if channel is not None:
                conditions.append("channel = ?")
                params.append(channel)
            if since is not None:
                conditions.append("timestamp >= ?")
                params.append(since)
            where = " AND ".join(conditions)
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                f"FROM usage WHERE {where}",
                tuple(params),
            ).fetchone()
            if row is None:
                return UsageSummary(0, 0, 0, 0.0)
            return UsageSummary(
                total_requests=row[0],
                total_prompt_tokens=row[1],
                total_completion_tokens=row[2],
                total_cost=row[3],
            )
        finally:
            conn.close()

    def get_channel_rank(self, channel: str, since: float | None = None) -> UsageRank:
        """Get the cost rank of a channel among all channels.

        Args:
            channel: IRC channel name.
            since: Optional Unix timestamp filter.

        Returns:
            UsageRank with 1-based rank (0 if channel has no usage).
        """
        return self._get_rank("channel", channel, since)

    def get_nick_rank(
        self, nick: str, since: float | None = None, channel: str | None = None
    ) -> UsageRank:
        """Get the cost rank of a nick among all nicks.

        Args:
            nick: IRC nick.
            since: Optional Unix timestamp filter.
            channel: Optional channel to scope the ranking.

        Returns:
            UsageRank with 1-based rank (0 if nick has no usage).
        """
        return self._get_rank("nick", nick, since, scope_channel=channel)

    def _get_rank(
        self,
        dimension: str,
        value: str,
        since: float | None,
        scope_channel: str | None = None,
    ) -> UsageRank:
        """Compute the cost rank of a value within a dimension.

        Uses a count-of-higher approach for SQLite compatibility (no window
        functions required): rank = number of entries with strictly higher
        total cost + 1.

        Args:
            dimension: Column to rank by ("nick" or "channel").
            value: The specific nick or channel to find the rank of.
            since: Optional Unix timestamp filter.
            scope_channel: Optional channel to scope the query (only for nick ranking).

        Returns:
            UsageRank with 1-based rank, or rank=0 if the value has no usage.
        """
        assert dimension in ("nick", "channel"), f"Invalid dimension: {dimension}"
        conn = self._connect()
        try:
            # Build WHERE clause fragments
            time_filter = "timestamp >= ?" if since is not None else None
            channel_filter = "channel = ?" if scope_channel is not None else None

            base_conditions = [c for c in (time_filter, channel_filter) if c]
            base_where = (" WHERE " + " AND ".join(base_conditions)) if base_conditions else ""
            base_params: list[object] = []
            if since is not None:
                base_params.append(since)
            if scope_channel is not None:
                base_params.append(scope_channel)

            # Total distinct entries
            total_row = conn.execute(
                f"SELECT COUNT(DISTINCT {dimension}) FROM usage{base_where}",
                tuple(base_params),
            ).fetchone()
            total = total_row[0] if total_row else 0

            # Get the value's total cost
            value_conditions = [f"{dimension} = ?"] + list(base_conditions)
            value_where = " AND ".join(value_conditions)
            value_params: list[object] = [value, *base_params]

            cost_row = conn.execute(
                f"SELECT COALESCE(SUM(cost), 0.0) FROM usage WHERE {value_where}",
                tuple(value_params),
            ).fetchone()
            value_cost = cost_row[0] if cost_row else 0.0

            if value_cost == 0.0:
                # Check if there's actually any usage for this value
                count_row = conn.execute(
                    f"SELECT COUNT(*) FROM usage WHERE {value_where}",
                    tuple(value_params),
                ).fetchone()
                if count_row is None or count_row[0] == 0:
                    return UsageRank(rank=0, total=total)

            # Count entries with strictly higher cost
            rank_sql = (
                f"SELECT COUNT(*) FROM "
                f"(SELECT {dimension}, SUM(cost) AS total_cost "
                f"FROM usage{base_where} "
                f"GROUP BY {dimension}) sub "
                f"WHERE sub.total_cost > ?"
            )
            rank_params: list[object] = [*base_params, value_cost]
            rank_row = conn.execute(rank_sql, tuple(rank_params)).fetchone()
            rank = (rank_row[0] + 1) if rank_row else 1

            return UsageRank(rank=rank, total=total)
        finally:
            conn.close()
