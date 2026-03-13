"""SQLite persistence layer for RPG plugin.

Thread-safe database operations for characters, inventory, and world state.
Uses thread-local connections with WAL mode.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import NamedTuple

SCHEMA_VERSION = 1


class CharacterRow(NamedTuple):
    """A character loaded from the database."""

    nick: str
    channel: str
    hp: int
    max_hp: int
    attack: int
    defense: int
    xp: int
    level: int
    gold: int
    location: str


class InventoryItem(NamedTuple):
    """An inventory item loaded from the database."""

    id: int
    nick: str
    channel: str
    name: str
    attack_bonus: int
    defense_bonus: int
    equipped: bool


class RPGDatabase:
    """SQLite database for RPG game state."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS characters (
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                hp INTEGER NOT NULL DEFAULT 20,
                max_hp INTEGER NOT NULL DEFAULT 20,
                attack INTEGER NOT NULL DEFAULT 3,
                defense INTEGER NOT NULL DEFAULT 2,
                xp INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 1,
                gold INTEGER NOT NULL DEFAULT 0,
                location TEXT NOT NULL DEFAULT '/town/tavern',
                PRIMARY KEY (nick, channel)
            );

            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                name TEXT NOT NULL,
                attack_bonus INTEGER NOT NULL DEFAULT 0,
                defense_bonus INTEGER NOT NULL DEFAULT 0,
                equipped INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (nick, channel) REFERENCES characters(nick, channel)
            );

            CREATE TABLE IF NOT EXISTS world_state (
                channel TEXT NOT NULL,
                room_path TEXT NOT NULL,
                cleared_at REAL,
                PRIMARY KEY (channel, room_path)
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );
        """)
        # Set schema version if not present
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()

    # --- Characters ---

    def create_character(self, nick: str, channel: str) -> None:
        """Create a new character if one doesn't exist."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO characters (nick, channel) VALUES (?, ?)",
            (nick.lower(), channel.lower()),
        )
        conn.commit()

    def get_character(self, nick: str, channel: str) -> CharacterRow | None:
        """Get a character by nick and channel."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT nick, channel, hp, max_hp, attack, defense, xp, level, gold, location "
            "FROM characters WHERE nick = ? AND channel = ?",
            (nick.lower(), channel.lower()),
        ).fetchone()
        if row is None:
            return None
        return CharacterRow(*row)

    def update_character(self, nick: str, channel: str, **kwargs: int | str) -> None:
        """Update character fields. Pass only the fields to change."""
        if not kwargs:
            return
        allowed = {"hp", "max_hp", "attack", "defense", "xp", "level", "gold", "location"}
        invalid = set(kwargs) - allowed
        if invalid:
            msg = f"Invalid fields: {invalid}"
            raise ValueError(msg)
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [nick.lower(), channel.lower()]
        conn = self._get_conn()
        conn.execute(
            f"UPDATE characters SET {sets} WHERE nick = ? AND channel = ?",  # noqa: S608
            values,
        )
        conn.commit()

    # --- Inventory ---

    def add_item(
        self,
        nick: str,
        channel: str,
        name: str,
        *,
        attack_bonus: int = 0,
        defense_bonus: int = 0,
    ) -> int:
        """Add an item to a character's inventory. Returns the item ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO inventory (nick, channel, name, attack_bonus, defense_bonus) "
            "VALUES (?, ?, ?, ?, ?)",
            (nick.lower(), channel.lower(), name, attack_bonus, defense_bonus),
        )
        conn.commit()
        return cursor.lastrowid or 0

    def get_inventory(self, nick: str, channel: str) -> list[InventoryItem]:
        """Get all inventory items for a character."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, nick, channel, name, attack_bonus, defense_bonus, equipped "
            "FROM inventory WHERE nick = ? AND channel = ? ORDER BY name",
            (nick.lower(), channel.lower()),
        ).fetchall()
        return [InventoryItem(r[0], r[1], r[2], r[3], r[4], r[5], bool(r[6])) for r in rows]

    def equip_item(self, item_id: int, *, equipped: bool) -> None:
        """Set the equipped flag on an item."""
        conn = self._get_conn()
        conn.execute("UPDATE inventory SET equipped = ? WHERE id = ?", (int(equipped), item_id))
        conn.commit()

    def remove_item(self, item_id: int) -> None:
        """Remove an item from inventory."""
        conn = self._get_conn()
        conn.execute("DELETE FROM inventory WHERE id = ?", (item_id,))
        conn.commit()

    # --- World State ---

    def mark_room_cleared(self, channel: str, room_path: str) -> None:
        """Mark a room as cleared with current timestamp."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO world_state (channel, room_path, cleared_at) VALUES (?, ?, ?)",
            (channel.lower(), room_path, time.time()),
        )
        conn.commit()

    def get_room_cleared_at(self, channel: str, room_path: str) -> float | None:
        """Get when a room was last cleared, or None if never."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT cleared_at FROM world_state WHERE channel = ? AND room_path = ?",
            (channel.lower(), room_path),
        ).fetchone()
        if row is None:
            return None
        return row[0]

    def reset_room(self, channel: str, room_path: str) -> None:
        """Reset a room so enemies can respawn."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM world_state WHERE channel = ? AND room_path = ?",
            (channel.lower(), room_path),
        )
        conn.commit()
