"""SQLite persistence layer for LLM plugin.

Provides thread-safe database operations for reminders, usage tracking,
and pending task queue.  Uses thread-local connections with WAL mode for
concurrent read performance without the overhead of reconnecting on every call.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import NamedTuple

# Schema version for future migrations
SCHEMA_VERSION = 13

# Reminders older than 24 hours past their fire_at are considered expired
EXPIRY_THRESHOLD_SECONDS = 86400  # 24 hours


class ReminderRow(NamedTuple):
    """A reminder loaded from the database."""

    id: int
    event_name: str
    nick: str
    channel: str
    message: str
    action_prompt: str
    account: str | None
    fire_at: float
    created_at: float
    chain_position: int
    recurrence_seconds: int | None
    recurrence_rrule: str | None
    watch_mode: bool


class ScheduledLlmTaskRow(NamedTuple):
    """A scheduled LLM task loaded from the database."""

    id: int
    event_name: str
    creator_nick: str
    account: str | None
    channel: str
    network: str
    wire_msg: str
    prompt: str
    fire_at: float
    created_at: float
    recurrence_seconds: int | None
    recurrence_rrule: str | None
    chain_position: int
    watch_mode: bool

    def rehydrate_msg(self):
        """Build a fresh ``IrcMsg`` from the persisted wire string."""
        from supybot.ircmsgs import IrcMsg

        return IrcMsg(s=self.wire_msg)


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


class PendingTaskRow(NamedTuple):
    """A pending task loaded from the database."""

    id: int
    task_type: str  # ask|code|draw
    nick: str
    reply_target: str  # channel name or PM nick
    is_channel: int  # 1 channel, 0 PM
    prompt_preview: str
    model: str
    request_data: str  # JSON blob
    submitted_at: float
    expires_at: float
    attempt_count: int
    next_attempt_at: float
    claimed_until: float
    last_error: str
    delivery_state: str  # pending|ready|retrying|delivered|delivery_failed|expired|failed_terminal
    result_payload: str  # JSON blob of delivery content
    last_delivery_error: str
    delivery_attempt_count: int
    origin_request_id: str
    account: str | None


class MemoryRow(NamedTuple):
    """A long-term memory (fact) about a user."""

    id: int
    nick: str
    fact: str
    source_channel: str
    created_at: float


class LLMDatabase:
    """SQLite database for LLM plugin persistence.

    Uses thread-local connections for thread safety: each thread gets its
    own long-lived connection (created lazily on first use).  WAL mode is
    set once per connection for concurrent read performance.
    """

    def __init__(self, db_path: str) -> None:
        """Create or open the database and run schema migration.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed.

        Reuses connections within the same thread to avoid the overhead of
        opening a new connection (and re-setting WAL mode) on every call.
        WAL mode is persistent on the database file so it only needs to be
        set once per connection.

        If the cached connection was closed externally (e.g. by a caller
        that obtained it via ``_connect()`` and called ``conn.close()``),
        a fresh connection is created transparently.

        Returns:
            A sqlite3.Connection with WAL journal mode.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                # Cheapest possible liveness probe — never hits disk.
                conn.execute("SELECT 1")
                return conn
            except sqlite3.ProgrammingError:
                self._local.conn = None
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        self._local.conn = conn
        return conn

    def _migrate(self) -> None:
        """Run schema migration to create tables and indexes.

        Uses SQLite's ``PRAGMA user_version`` to track which migrations have
        been applied.  New tables are created with ``CREATE TABLE IF NOT EXISTS``
        so the initial DDL is always safe to re-run, and incremental column /
        table additions are guarded by version checks.
        """
        conn = self._connect()
        try:
            # --- v1 baseline: create core tables ---------------------------
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

                CREATE TABLE IF NOT EXISTS pending_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_type TEXT NOT NULL,
                    nick TEXT NOT NULL,
                    reply_target TEXT NOT NULL,
                    is_channel INTEGER NOT NULL,
                    prompt_preview TEXT NOT NULL,
                    model TEXT NOT NULL,
                    request_data TEXT NOT NULL DEFAULT '{}',
                    submitted_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL,
                    claimed_until REAL NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_pending_tasks_expires_at
                    ON pending_tasks(expires_at);
                CREATE INDEX IF NOT EXISTS idx_pending_tasks_due
                    ON pending_tasks(next_attempt_at, claimed_until);
                CREATE INDEX IF NOT EXISTS idx_pending_tasks_type
                    ON pending_tasks(task_type);
            """)
            conn.commit()

            # --- version-gated migrations ----------------------------------
            row = conn.execute("PRAGMA user_version").fetchone()
            current_version = row[0] if row else 0

            if current_version < 2:
                conn.executescript("""
                    ALTER TABLE usage ADD COLUMN prompt TEXT NOT NULL DEFAULT '';
                    ALTER TABLE usage ADD COLUMN status TEXT NOT NULL DEFAULT 'success';
                    ALTER TABLE usage ADD COLUMN error_detail TEXT NOT NULL DEFAULT '';

                    CREATE INDEX IF NOT EXISTS idx_usage_nick_status
                        ON usage(nick, status);

                """)
                conn.commit()

            if current_version < 3:
                conn.executescript("""
                    ALTER TABLE pending_tasks
                        ADD COLUMN delivery_state TEXT NOT NULL DEFAULT 'pending';
                    ALTER TABLE pending_tasks
                        ADD COLUMN result_payload TEXT NOT NULL DEFAULT '';
                    ALTER TABLE pending_tasks
                        ADD COLUMN last_delivery_error TEXT NOT NULL DEFAULT '';
                    ALTER TABLE pending_tasks
                        ADD COLUMN delivery_attempt_count INTEGER NOT NULL DEFAULT 0;
                    ALTER TABLE pending_tasks
                        ADD COLUMN origin_request_id TEXT NOT NULL DEFAULT '';

                    CREATE INDEX IF NOT EXISTS idx_pending_tasks_delivery_state
                        ON pending_tasks(delivery_state);
                """)
                conn.commit()

            if current_version < 4:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        nick TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        messages TEXT NOT NULL,
                        last_activity REAL NOT NULL,
                        PRIMARY KEY (nick, channel)
                    );
                """)
                conn.commit()

            if current_version < 5:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        nick TEXT NOT NULL,
                        fact TEXT NOT NULL,
                        source_channel TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_memories_nick
                        ON memories(nick);
                """)
                conn.commit()

            if current_version < 6:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memory_cleanup_state (
                        nick TEXT PRIMARY KEY,
                        saves_since_cleanup INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.commit()

            if current_version < 7:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS user_instructions (
                        nick TEXT PRIMARY KEY,
                        instruction TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                """)
                conn.commit()

            if current_version < 8:
                conn.executescript("""
                    ALTER TABLE pending_tasks
                        ADD COLUMN account TEXT;
                """)
                conn.commit()

            if current_version < 9:
                conn.executescript("""
                    ALTER TABLE reminders
                        ADD COLUMN action_prompt TEXT NOT NULL DEFAULT '';
                    ALTER TABLE reminders
                        ADD COLUMN account TEXT;
                """)
                conn.commit()

            if current_version < 10:
                # Per-chain caps: each reminder belongs to a chain (started by
                # a chat-level set, extended by action-fire reschedules). Old
                # rows backfill chain_id = event_name (single-fire chain),
                # chain_position = 1, chain_started_at = created_at.
                conn.executescript("""
                    ALTER TABLE reminders
                        ADD COLUMN chain_id TEXT NOT NULL DEFAULT '';
                    ALTER TABLE reminders
                        ADD COLUMN chain_position INTEGER NOT NULL DEFAULT 1;
                    ALTER TABLE reminders
                        ADD COLUMN chain_started_at REAL NOT NULL DEFAULT 0;
                """)
                conn.execute("UPDATE reminders SET chain_id = event_name WHERE chain_id = ''")
                conn.execute(
                    "UPDATE reminders SET chain_started_at = created_at WHERE chain_started_at = 0"
                )
                conn.commit()

            if current_version < 11:
                # chain_id was stored on every row but never used as a lookup
                # key — chain_position and chain_started_at carry the cap and
                # TTL semantics. Drop the unused column.
                conn.executescript("""
                    ALTER TABLE reminders DROP COLUMN chain_id;
                """)
                conn.commit()

            if current_version < 12:
                # B0.5 strategy: graceful degradation. Existing rows have NULL
                # recurrence_seconds/recurrence_rrule and watch_mode=0; the
                # legacy LLM-tool reschedule path keeps them firing via
                # parenthetical parsing in action_prompt until they exhaust
                # naturally. New rows populate the structured columns directly
                # (B2 parser, B4 mechanical reschedule). The 30-day chain TTL
                # is also retired here — the 50-fire chain_position cap remains
                # the sole runaway guard.
                conn.executescript("""
                    ALTER TABLE reminders DROP COLUMN chain_started_at;
                    ALTER TABLE reminders ADD COLUMN recurrence_seconds INTEGER;
                    ALTER TABLE reminders ADD COLUMN recurrence_rrule TEXT;
                    ALTER TABLE reminders ADD COLUMN watch_mode INTEGER NOT NULL DEFAULT 0;
                """)
                conn.commit()

            if current_version < 13:
                # Task 3 (Limnoria bridge Phase 2): native LLM tool
                # ``schedule_llm_task`` and friends. One row per active
                # schedule. Persists wire-format msg so the fire closure can
                # rebuild a fresh IrcMsg without relying on pickle (msg.tags
                # would be lost over IrcMsg.__reduce__).
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS scheduled_llm_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_name TEXT UNIQUE NOT NULL,
                        creator_nick TEXT NOT NULL,
                        account TEXT,
                        channel TEXT NOT NULL,
                        network TEXT NOT NULL,
                        wire_msg TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        fire_at REAL NOT NULL,
                        created_at REAL NOT NULL,
                        recurrence_seconds INTEGER,
                        recurrence_rrule TEXT,
                        chain_position INTEGER NOT NULL DEFAULT 1,
                        watch_mode INTEGER NOT NULL DEFAULT 0
                    );
                    CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_fire_at
                        ON scheduled_llm_tasks(fire_at);
                    CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_account
                        ON scheduled_llm_tasks(account);
                    CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_creator_nick
                        ON scheduled_llm_tasks(creator_nick);
                    CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_owner_channel
                        ON scheduled_llm_tasks(account, creator_nick, channel);
                """)
                conn.commit()

            # Stamp the schema version so future opens skip completed migrations.
            # PRAGMA statements cannot be part of executescript, so use execute.
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
        finally:
            pass

    def close(self) -> None:
        """Close the current thread's connection if open."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        """Best-effort cleanup of the current thread's connection."""
        self.close()

    # ------------------------------------------------------------------
    # Conversation persistence
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        nick: str,
        channel: str,
        messages: list[dict[str, str]],
        last_activity: float,
    ) -> None:
        """Persist a conversation's messages to the database.

        Args:
            nick: User's IRC nick (lowercased before storage).
            channel: IRC channel (lowercased before storage).
            messages: List of message dicts (role + content).
            last_activity: Timestamp of last activity.
        """
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
            "VALUES (?, ?, ?, ?)",
            (nick.lower(), channel.lower(), json.dumps(messages), last_activity),
        )
        conn.commit()

    def delete_conversation(self, nick: str, channel: str) -> None:
        """Delete a conversation from the database.

        Args:
            nick: User's IRC nick.
            channel: IRC channel.
        """
        conn = self._connect()
        conn.execute(
            "DELETE FROM conversations WHERE nick = ? AND channel = ?",
            (nick.lower(), channel.lower()),
        )
        conn.commit()

    def delete_all_conversations(self) -> None:
        """Delete all conversations from the database."""
        conn = self._connect()
        conn.execute("DELETE FROM conversations")
        conn.commit()

    def load_conversations(self) -> list[tuple[str, str, list[dict[str, str]], float]]:
        """Load all conversations from the database.

        Returns:
            List of (nick, channel, messages, last_activity) tuples.
            Rows with corrupt JSON are logged and skipped.
        """
        log = logging.getLogger("supybot.plugins.LLM")
        conn = self._connect()
        rows = conn.execute(
            "SELECT nick, channel, messages, last_activity FROM conversations"
        ).fetchall()

        result: list[tuple[str, str, list[dict[str, str]], float]] = []
        for nick, channel, messages_json, last_activity in rows:
            try:
                messages = json.loads(messages_json)
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping corrupt conversation for %s/%s", nick, channel)
                conn.execute(
                    "DELETE FROM conversations WHERE nick = ? AND channel = ?",
                    (nick, channel),
                )
                conn.commit()
                continue
            result.append((nick, channel, messages, last_activity))
        return result

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
        *,
        action_prompt: str = "",
        account: str | None = None,
        chain_position: int = 1,
        recurrence_seconds: int | None = None,
        recurrence_rrule: str | None = None,
        watch_mode: bool = False,
    ) -> int:
        """Save a reminder to the database.

        Args:
            event_name: Unique identifier for the reminder event.
            nick: IRC nick that created the reminder.
            channel: IRC channel the reminder was created in.
            message: Reminder message text.
            fire_at: Unix timestamp when the reminder should fire.
            action_prompt: Optional follow-up LLM prompt to run when reminder fires.
            account: Requester's resolved account name, or None if unknown.
            chain_position: 1-based position within the chain.
            recurrence_seconds: Numeric cadence (seconds) for recurring fires,
                or None for one-shot / RRULE-driven rows.
            recurrence_rrule: RFC 5545 RRULE string for calendar-driven cadences,
                or None for one-shot / numeric rows.
            watch_mode: True when the action LLM may emit ``[silent]`` to skip
                user-visible delivery for a fire (long-running watch).

        Returns:
            The row ID of the inserted reminder.

        Raises:
            ValueError: If both ``recurrence_seconds`` and ``recurrence_rrule``
                are non-null (they are mutually exclusive).
            sqlite3.IntegrityError: If event_name already exists.
        """
        if recurrence_seconds is not None and recurrence_rrule is not None:
            raise ValueError("recurrence_seconds and recurrence_rrule are mutually exclusive")
        now = time.time()
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO reminders "
                "(event_name, nick, channel, message, action_prompt, account, "
                "fire_at, created_at, chain_position, "
                "recurrence_seconds, recurrence_rrule, watch_mode) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_name,
                    nick,
                    channel,
                    message,
                    action_prompt,
                    account,
                    fire_at,
                    now,
                    chain_position,
                    recurrence_seconds,
                    recurrence_rrule,
                    int(watch_mode),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            pass

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
            pass

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
                "SELECT id, event_name, nick, channel, message, action_prompt, account, "
                "fire_at, created_at, chain_position, "
                "recurrence_seconds, recurrence_rrule, watch_mode "
                "FROM reminders WHERE fire_at > ? ORDER BY fire_at",
                (cutoff,),
            ).fetchall()
            # watch_mode is stored as INTEGER 0/1; expose as bool on the row.
            return [
                ReminderRow(
                    id=row[0],
                    event_name=row[1],
                    nick=row[2],
                    channel=row[3],
                    message=row[4],
                    action_prompt=row[5],
                    account=row[6],
                    fire_at=row[7],
                    created_at=row[8],
                    chain_position=row[9],
                    recurrence_seconds=row[10],
                    recurrence_rrule=row[11],
                    watch_mode=bool(row[12]),
                )
                for row in rows
            ]
        finally:
            pass

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
            pass

    # ------------------------------------------------------------------
    # Scheduled LLM task operations
    # ------------------------------------------------------------------

    _SCHEDULED_LLM_TASK_COLUMNS = (
        "id, event_name, creator_nick, account, channel, network, wire_msg, "
        "prompt, fire_at, created_at, recurrence_seconds, recurrence_rrule, "
        "chain_position, watch_mode"
    )

    @staticmethod
    def _row_to_scheduled_llm_task(row: tuple) -> ScheduledLlmTaskRow:
        return ScheduledLlmTaskRow(
            id=row[0],
            event_name=row[1],
            creator_nick=row[2],
            account=row[3],
            channel=row[4],
            network=row[5],
            wire_msg=row[6],
            prompt=row[7],
            fire_at=row[8],
            created_at=row[9],
            recurrence_seconds=row[10],
            recurrence_rrule=row[11],
            chain_position=row[12],
            watch_mode=bool(row[13]),
        )

    def save_scheduled_llm_task(
        self,
        event_name: str,
        creator_nick: str,
        account: str | None,
        channel: str,
        network: str,
        wire_msg: str,
        prompt: str,
        fire_at: float,
        *,
        recurrence_seconds: int | None = None,
        recurrence_rrule: str | None = None,
        chain_position: int = 1,
        watch_mode: bool = False,
    ) -> int:
        """Save a scheduled LLM task row.

        Raises:
            ValueError: if both recurrence kinds are non-null.
            sqlite3.IntegrityError: if event_name already exists.
        """
        if recurrence_seconds is not None and recurrence_rrule is not None:
            raise ValueError("recurrence_seconds and recurrence_rrule are mutually exclusive")
        now = time.time()
        conn = self._connect()
        cursor = conn.execute(
            "INSERT INTO scheduled_llm_tasks "
            "(event_name, creator_nick, account, channel, network, wire_msg, "
            "prompt, fire_at, created_at, recurrence_seconds, recurrence_rrule, "
            "chain_position, watch_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_name,
                creator_nick,
                account,
                channel,
                network,
                wire_msg,
                prompt,
                fire_at,
                now,
                recurrence_seconds,
                recurrence_rrule,
                chain_position,
                int(watch_mode),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def update_scheduled_llm_task_fire_at(
        self,
        event_name: str,
        fire_at: float,
        *,
        chain_position: int | None = None,
    ) -> None:
        """Update fire_at (and optionally chain_position) for a row."""
        conn = self._connect()
        if chain_position is None:
            conn.execute(
                "UPDATE scheduled_llm_tasks SET fire_at = ? WHERE event_name = ?",
                (fire_at, event_name),
            )
        else:
            conn.execute(
                "UPDATE scheduled_llm_tasks SET fire_at = ?, chain_position = ? "
                "WHERE event_name = ?",
                (fire_at, chain_position, event_name),
            )
        conn.commit()

    def delete_scheduled_llm_task(self, event_name: str) -> bool:
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM scheduled_llm_tasks WHERE event_name = ?",
            (event_name,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def load_active_scheduled_llm_tasks(self) -> list[ScheduledLlmTaskRow]:
        """Load rows whose fire_at is within the last 24 hours or in the future."""
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks WHERE fire_at > ? ORDER BY fire_at",
            (cutoff,),
        ).fetchall()
        return [self._row_to_scheduled_llm_task(r) for r in rows]

    def get_scheduled_llm_task(self, event_name: str) -> ScheduledLlmTaskRow | None:
        """Indexed point-lookup by event_name."""
        conn = self._connect()
        row = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks WHERE event_name = ?",
            (event_name,),
        ).fetchone()
        return self._row_to_scheduled_llm_task(row) if row else None

    def load_scheduled_llm_tasks_for(
        self, *, account: str | None, nick: str
    ) -> list[ScheduledLlmTaskRow]:
        """Active rows owned by the caller. Case-insensitive Identity semantics."""
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        if account is not None:
            rows = conn.execute(
                f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
                "FROM scheduled_llm_tasks "
                "WHERE lower(account) = lower(?) AND fire_at > ? "
                "ORDER BY fire_at",
                (account, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
                "FROM scheduled_llm_tasks "
                "WHERE account IS NULL AND lower(creator_nick) = lower(?) "
                "AND fire_at > ? ORDER BY fire_at",
                (nick, cutoff),
            ).fetchall()
        return [self._row_to_scheduled_llm_task(r) for r in rows]

    def count_scheduled_llm_tasks_for(self, *, account: str | None, nick: str, channel: str) -> int:
        """Count active rows owned by the caller in this channel.

        When ``account`` is non-None, count rows with that account regardless of
        nick. Otherwise count by raw nick. Comparisons are case-insensitive to
        match ``Identity.matches``.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        if account is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM scheduled_llm_tasks "
                "WHERE lower(account) = lower(?) AND channel = ? AND fire_at > ?",
                (account, channel, cutoff),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM scheduled_llm_tasks "
                "WHERE account IS NULL AND lower(creator_nick) = lower(?) "
                "AND channel = ? AND fire_at > ?",
                (nick, channel, cutoff),
            ).fetchone()
        return int(row[0] if row else 0)

    # ------------------------------------------------------------------
    # Pending task operations
    # ------------------------------------------------------------------

    _PENDING_TASK_COLUMNS = (
        "id, task_type, nick, reply_target, is_channel, prompt_preview, model, "
        "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
        "claimed_until, last_error, delivery_state, result_payload, "
        "last_delivery_error, delivery_attempt_count, origin_request_id, account"
    )

    def save_pending_task(
        self,
        task_type: str,
        nick: str,
        reply_target: str,
        is_channel: bool,
        prompt_preview: str,
        model: str,
        request_data: str,
        submitted_at: float,
        expires_at: float,
        next_attempt_at: float,
        origin_request_id: str = "",
        account: str | None = None,
    ) -> int:
        """Save a pending task to the database.

        Args:
            task_type: Command type (ask, code, draw).
            nick: IRC nick that initiated the command.
            reply_target: Channel or PM nick for delivery.
            is_channel: True if reply_target is a channel.
            prompt_preview: Truncated prompt for display.
            model: Model identifier.
            request_data: JSON-serialized request payload.
            submitted_at: Unix timestamp when originally submitted.
            expires_at: Unix timestamp after which to stop retrying.
            next_attempt_at: Unix timestamp for first retry.
            origin_request_id: Stable trace/request ID captured at request acceptance.
            account: Resolved account name at submission time, or None if the
                requester was not identified. Delivery-time logging reads this
                directly instead of doing a late nick→account lookup.

        Returns:
            The row ID of the inserted task.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO pending_tasks "
                "(task_type, nick, reply_target, is_channel, prompt_preview, model, "
                "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
                "claimed_until, last_error, origin_request_id, account) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, '', ?, ?)",
                (
                    task_type,
                    nick,
                    reply_target,
                    1 if is_channel else 0,
                    prompt_preview,
                    model,
                    request_data,
                    submitted_at,
                    expires_at,
                    next_attempt_at,
                    origin_request_id,
                    account,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            pass

    def claim_due_pending_tasks(
        self,
        now: float,
        limit: int,
        lease_seconds: int,
        delivery_state_filter: str | tuple[str, ...] | None = None,
        max_delivery_attempts: int | None = None,
    ) -> list[PendingTaskRow]:
        """Atomically claim pending tasks that are due for retry.

        Uses BEGIN IMMEDIATE for exclusive write access so concurrent
        callers cannot claim the same rows.

        Args:
            now: Current Unix timestamp.
            limit: Maximum number of tasks to claim.
            lease_seconds: How long to hold the claim (seconds).
            delivery_state_filter: Optional filter on delivery_state. Can be a
                single state string or a tuple of states to match.
            max_delivery_attempts: Optional cap used by delivery-phase callers
                to skip rows that already exhausted retries.

        Returns:
            List of claimed PendingTaskRow objects.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            if delivery_state_filter is not None:
                if isinstance(delivery_state_filter, str):
                    state_clause = "AND delivery_state = ?"
                    state_params: tuple[object, ...] = (delivery_state_filter,)
                else:
                    placeholders = ",".join("?" for _ in delivery_state_filter)
                    state_clause = f"AND delivery_state IN ({placeholders})"
                    state_params = tuple(delivery_state_filter)
            else:
                state_clause = ""
                state_params = ()

            if max_delivery_attempts is not None:
                attempt_clause = "AND delivery_attempt_count < ?"
                attempt_params: tuple[object, ...] = (max_delivery_attempts,)
            else:
                attempt_clause = ""
                attempt_params = ()

            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks "
                f"WHERE next_attempt_at <= ? AND claimed_until <= ? {state_clause} {attempt_clause} "
                "ORDER BY next_attempt_at LIMIT ?",
                (now, now, *state_params, *attempt_params, limit),
            ).fetchall()

            if not rows:
                conn.commit()
                return []

            claimed_until = now + lease_seconds
            ids = [row[0] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE pending_tasks SET claimed_until = ? WHERE id IN ({placeholders})",
                [claimed_until, *ids],
            )
            conn.commit()
            return [PendingTaskRow(*row) for row in rows]
        except Exception:
            conn.rollback()
            raise
        finally:
            pass

    def release_pending_task(
        self,
        task_id: int,
        next_attempt_at: float,
        last_error: str,
        increment_attempt: bool = True,
    ) -> bool:
        """Release a claimed task back to the queue for later retry.

        Args:
            task_id: ID of the task to release.
            next_attempt_at: When to retry next.
            last_error: Error message from this attempt.
            increment_attempt: Whether to bump attempt_count.

        Returns:
            True if the task was updated, False if not found.
        """
        conn = self._connect()
        try:
            if increment_attempt:
                cursor = conn.execute(
                    "UPDATE pending_tasks SET "
                    "next_attempt_at = ?, claimed_until = 0, "
                    "last_error = ?, attempt_count = attempt_count + 1 "
                    "WHERE id = ?",
                    (next_attempt_at, last_error, task_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE pending_tasks SET "
                    "next_attempt_at = ?, claimed_until = 0, last_error = ? "
                    "WHERE id = ?",
                    (next_attempt_at, last_error, task_id),
                )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def delete_pending_task(self, task_id: int) -> bool:
        """Delete a pending task by ID.

        Args:
            task_id: ID of the task to delete.

        Returns:
            True if a task was deleted, False if not found.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM pending_tasks WHERE id = ?",
                (task_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def update_task_for_delivery(
        self,
        task_id: int,
        delivery_state: str,
        result_payload: str,
    ) -> bool:
        """Transition a task to a delivery state with its result payload.

        Args:
            task_id: ID of the task to update.
            delivery_state: New delivery state (ready, failed_terminal, etc.).
            result_payload: JSON-serialized result for delivery.

        Returns:
            True if the task was updated, False if not found.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE pending_tasks SET "
                "delivery_state = ?, result_payload = ?, claimed_until = 0 "
                "WHERE id = ?",
                (delivery_state, result_payload, task_id),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def update_delivery_attempt(
        self,
        task_id: int,
        delivery_state: str,
        last_delivery_error: str,
        delivery_attempt_count: int,
        next_attempt_at: float,
    ) -> bool:
        """Record a delivery attempt outcome (success or failure).

        Args:
            task_id: ID of the task to update.
            delivery_state: New delivery state (retrying, delivery_failed).
            last_delivery_error: Error message from the delivery attempt.
            delivery_attempt_count: Updated delivery attempt count.
            next_attempt_at: When to retry delivery next.

        Returns:
            True if the task was updated, False if not found.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE pending_tasks SET "
                "delivery_state = ?, last_delivery_error = ?, "
                "delivery_attempt_count = ?, next_attempt_at = ?, claimed_until = 0 "
                "WHERE id = ?",
                (
                    delivery_state,
                    last_delivery_error,
                    delivery_attempt_count,
                    next_attempt_at,
                    task_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def delete_expired_pending_tasks(self, now: float) -> list[PendingTaskRow]:
        """Delete pending tasks whose expires_at has passed.

        Only deletes tasks still in the provider phase (delivery_state='pending').
        Tasks that already have results (ready/retrying) are preserved for
        delivery retry.

        Args:
            now: Current Unix timestamp.

        Returns:
            List of expired PendingTaskRow objects (before deletion).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks "
                "WHERE expires_at <= ? AND delivery_state = 'pending'",
                (now,),
            ).fetchall()
            if rows:
                ids = [row[0] for row in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM pending_tasks WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
            return [PendingTaskRow(*row) for row in rows]
        finally:
            pass

    def get_next_due_time(self) -> float | None:
        """Return the earliest next_attempt_at for actionable unclaimed tasks.

        Only considers tasks that could be processed by the scheduler:
        unclaimed (or lease-expired) rows with delivery_state in
        (pending, ready, retrying).

        Returns:
            Earliest next_attempt_at timestamp, or None if the queue is empty.
        """
        conn = self._connect()
        try:
            now = time.time()
            row = conn.execute(
                "SELECT MIN(next_attempt_at) FROM pending_tasks "
                "WHERE claimed_until <= ? "
                "AND delivery_state IN ('pending', 'ready', 'retrying')",
                (now,),
            ).fetchone()
            if row is None or row[0] is None:
                return None
            return row[0]
        finally:
            pass

    def load_pending_tasks(self, task_type: str | None = None) -> list[PendingTaskRow]:
        """Load pending tasks, optionally filtered by type.

        Intended for debugging and tests.

        Args:
            task_type: Optional filter (ask, code, draw).

        Returns:
            List of PendingTaskRow ordered by submitted_at ascending.
        """
        conn = self._connect()
        try:
            if task_type is not None:
                rows = conn.execute(
                    f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks "
                    "WHERE task_type = ? ORDER BY submitted_at",
                    (task_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks ORDER BY submitted_at",
                ).fetchall()
            return [PendingTaskRow(*row) for row in rows]
        finally:
            pass

    # ------------------------------------------------------------------
    # Usage migration
    # ------------------------------------------------------------------

    def migrate_nick(self, old_nick: str, new_nick: str) -> int:
        """Migrate usage rows from an old nick to a new identity.

        Used when switching from nick-based to account-based tracking:
        rows logged under the raw IRC nick are re-attributed to the
        NickServ account name so ``%usage`` queries return complete data.

        The match is case-insensitive (IRC nicks are case-insensitive).
        Rows that already carry *new_nick* are left untouched.

        Args:
            old_nick: Previous nick value (e.g. ``"Rubin[F]"``).
            new_nick: New identity value (e.g. ``"Rubin"``).

        Returns:
            Number of rows updated.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE usage SET nick = ? WHERE LOWER(nick) = LOWER(?) AND nick != ?",
                (new_nick, old_nick, new_nick),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            pass

    def migrate_conversations(self, old_nick: str, new_nick: str) -> int:
        """Re-attribute conversation rows from an old nick to a new identity.

        Companion to :meth:`migrate_nick`. When a user identifies for the
        first time in a session, persisted conversation history logged
        under the raw nick is moved to the account so follow-up turns
        resume the same context.

        ``conversations`` is keyed on ``(nick, channel)``. If a row already
        exists at the destination key for some channel, the source row is
        deleted (the destination is the canonical, identified-user copy).
        Conflict-free rows are simply renamed. Stored values are
        lowercased; the match is case-insensitive.

        Args:
            old_nick: Previous nick value.
            new_nick: New identity value (typically a NickServ account).

        Returns:
            Number of rows updated (renamed only; conflicts dropped don't count).
        """
        old = old_nick.lower()
        new = new_nick.lower()
        if old == new:
            return 0
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM conversations WHERE nick = ? AND channel IN ("
                "  SELECT channel FROM conversations WHERE nick = ?"
                ")",
                (old, new),
            )
            cursor = conn.execute(
                "UPDATE conversations SET nick = ? WHERE nick = ?",
                (new, old),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            pass

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
        prompt: str = "",
        status: str = "success",
        error_detail: str = "",
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
            prompt: The user's prompt text (for audit/flagging).
            status: Outcome status.  Known values: ``"success"``,
                ``"error"``, ``"content_blocked"``,
                ``"auth_failure"``, ``"rate_limited"``.
            error_detail: Additional error context when status is not success.
        """
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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

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
            pass

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def save_memory(self, nick: str, fact: str, source_channel: str) -> int:
        """Save a memory fact for a user.

        Args:
            nick: IRC nick (stored lowercased).
            fact: The fact to remember about the user.
            source_channel: Channel where the fact was learned.

        Returns:
            The row ID of the inserted memory.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO memories (nick, fact, source_channel, created_at) VALUES (?, ?, ?, ?)",
                (nick.lower(), fact, source_channel.lower(), time.time()),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            pass

    def get_memories(self, nick: str) -> list[MemoryRow]:
        """Get all memories for a user, most recent first.

        Args:
            nick: IRC nick (matched case-insensitively).

        Returns:
            List of MemoryRow ordered by created_at descending (newest first).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, nick, fact, source_channel, created_at FROM memories "
                "WHERE nick = ? ORDER BY created_at DESC",
                (nick.lower(),),
            ).fetchall()
            return [MemoryRow(*row) for row in rows]
        finally:
            pass

    def delete_memory(self, nick: str, memory_id: int) -> bool:
        """Delete a specific memory by ID and nick.

        The nick check prevents users from deleting other users' memories.

        Args:
            nick: IRC nick (must match the memory's owner).
            memory_id: Row ID of the memory to delete.

        Returns:
            True if a memory was deleted, False if not found or wrong owner.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ? AND nick = ?",
                (memory_id, nick.lower()),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def update_memory(self, nick: str, memory_id: int, new_fact: str) -> bool:
        """Update the fact text of a specific memory.

        The nick check prevents users from editing other users' memories.

        Args:
            nick: IRC nick (must match the memory's owner).
            memory_id: Row ID of the memory to update.
            new_fact: The new fact text.

        Returns:
            True if a memory was updated, False if not found or wrong owner.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE memories SET fact = ? WHERE id = ? AND nick = ?",
                (new_fact, memory_id, nick.lower()),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            pass

    def delete_all_memories(self, nick: str) -> int:
        """Delete all memories for a user.

        Args:
            nick: IRC nick whose memories should be deleted.

        Returns:
            Number of memories deleted.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM memories WHERE nick = ?",
                (nick.lower(),),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            pass

    def increment_memory_saves(self, nick: str) -> int:
        """Increment the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).

        Returns:
            The new counter value after incrementing.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT INTO memory_cleanup_state (nick, saves_since_cleanup) "
                "VALUES (?, 1) "
                "ON CONFLICT(nick) DO UPDATE SET saves_since_cleanup = saves_since_cleanup + 1 "
                "RETURNING saves_since_cleanup",
                (nick.lower(),),
            )
            row = cursor.fetchone()
            conn.commit()
            return row[0] if row else 0
        finally:
            pass

    def reset_memory_saves(self, nick: str) -> None:
        """Reset the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).
        """
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memory_cleanup_state SET saves_since_cleanup = 0 WHERE nick = ?",
                (nick.lower(),),
            )
            conn.commit()
        finally:
            pass

    def get_memory_saves(self, nick: str) -> int:
        """Get the current memory-saves-since-cleanup count for a user.

        Args:
            nick: IRC nick (matched case-insensitively).

        Returns:
            Current counter value, or 0 if no record exists.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT saves_since_cleanup FROM memory_cleanup_state WHERE nick = ?",
                (nick.lower(),),
            ).fetchone()
            return row[0] if row else 0
        finally:
            pass

    # ------------------------------------------------------------------
    # User instruction operations
    # ------------------------------------------------------------------

    def get_instruction(self, nick: str) -> str | None:
        """Get the user's persistent instruction, or None if not set."""
        conn = self._connect()
        row = conn.execute(
            "SELECT instruction FROM user_instructions WHERE nick = ?",
            (nick,),
        ).fetchone()
        return row[0] if row else None

    def save_instruction(self, nick: str, instruction: str) -> None:
        """Save or overwrite the user's persistent instruction."""
        conn = self._connect()
        conn.execute(
            "INSERT INTO user_instructions (nick, instruction, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(nick) DO UPDATE SET instruction = excluded.instruction, "
            "updated_at = excluded.updated_at",
            (nick, instruction, time.time()),
        )
        conn.commit()

    def delete_instruction(self, nick: str) -> bool:
        """Delete the user's instruction. Returns True if one was deleted."""
        conn = self._connect()
        cursor = conn.execute(
            "DELETE FROM user_instructions WHERE nick = ?",
            (nick,),
        )
        conn.commit()
        return cursor.rowcount > 0
