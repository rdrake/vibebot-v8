"""Tests for RPG persistence layer."""

from __future__ import annotations

import pytest
from rpg.persistence import RPGDatabase


@pytest.fixture
def db(tmp_path) -> RPGDatabase:
    """Create a fresh in-memory-style DB for each test."""
    return RPGDatabase(str(tmp_path / "test.db"))


class TestCharacterPersistence:
    """Character CRUD operations."""

    def test_create_character(self, db: RPGDatabase):
        """GIVEN no character WHEN create_character THEN character exists."""
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.nick == "alice"
        assert char.channel == "#test"
        assert char.location == "/town/tavern"

    def test_create_character_defaults(self, db: RPGDatabase):
        """GIVEN new character WHEN checking stats THEN defaults are set."""
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.hp == 20
        assert char.max_hp == 20
        assert char.attack == 3
        assert char.defense == 2
        assert char.xp == 0
        assert char.level == 1
        assert char.gold == 0

    def test_get_nonexistent_character(self, db: RPGDatabase):
        """GIVEN no character WHEN get_character THEN returns None."""
        assert db.get_character("nobody", "#test") is None

    def test_update_character_location(self, db: RPGDatabase):
        """GIVEN a character WHEN updating location THEN location changes."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.location == "/forest/clearing"

    def test_update_character_stats(self, db: RPGDatabase):
        """GIVEN a character WHEN updating HP THEN HP changes."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", hp=15, xp=10, gold=5)
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.hp == 15
        assert char.xp == 10
        assert char.gold == 5

    def test_duplicate_character_ignored(self, db: RPGDatabase):
        """GIVEN existing character WHEN create_character again THEN no error."""
        db.create_character("alice", "#test")
        db.create_character("alice", "#test")
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.level == 1


class TestInventoryPersistence:
    """Inventory CRUD operations."""

    def test_add_item(self, db: RPGDatabase):
        """GIVEN a character WHEN adding item THEN item in inventory."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        assert len(items) == 1
        assert items[0].name == "sword.txt"
        assert items[0].attack_bonus == 2

    def test_equip_item(self, db: RPGDatabase):
        """GIVEN an item WHEN equipping THEN equipped flag set."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        db.equip_item(items[0].id, equipped=True)
        items = db.get_inventory("alice", "#test")
        assert items[0].equipped is True

    def test_remove_item(self, db: RPGDatabase):
        """GIVEN an item WHEN removing THEN inventory empty."""
        db.create_character("alice", "#test")
        db.add_item("alice", "#test", "sword.txt", attack_bonus=2)
        items = db.get_inventory("alice", "#test")
        db.remove_item(items[0].id)
        assert db.get_inventory("alice", "#test") == []

    def test_empty_inventory(self, db: RPGDatabase):
        """GIVEN a character WHEN no items THEN empty list."""
        db.create_character("alice", "#test")
        assert db.get_inventory("alice", "#test") == []


class TestWorldState:
    """Room state persistence."""

    def test_clear_room(self, db: RPGDatabase):
        """GIVEN a room WHEN marking cleared THEN timestamp set."""
        db.mark_room_cleared("#test", "/dungeon/level1")
        ts = db.get_room_cleared_at("#test", "/dungeon/level1")
        assert ts is not None
        assert ts > 0

    def test_uncleared_room(self, db: RPGDatabase):
        """GIVEN a room never cleared WHEN checking THEN returns None."""
        assert db.get_room_cleared_at("#test", "/dungeon/level1") is None

    def test_reset_room(self, db: RPGDatabase):
        """GIVEN a cleared room WHEN resetting THEN cleared_at is None."""
        db.mark_room_cleared("#test", "/dungeon/level1")
        db.reset_room("#test", "/dungeon/level1")
        assert db.get_room_cleared_at("#test", "/dungeon/level1") is None
