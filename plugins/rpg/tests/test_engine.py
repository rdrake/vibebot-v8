"""Tests for RPG game engine."""

from __future__ import annotations

import pytest
from rpg.engine import GameEngine
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


@pytest.fixture
def engine(tmp_path) -> GameEngine:
    """Create engine with fresh DB and starter world."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    return GameEngine(db=db, world=world, spawn_cooldown_minutes=30)


class TestMovement:
    """cd command — moving between rooms."""

    def test_first_command_creates_character(self, engine: GameEngine):
        """GIVEN no character WHEN any command THEN character auto-created."""
        engine.move("alice", "#test", ".")
        char = engine.db.get_character("alice", "#test")
        assert char is not None

    def test_move_to_child_room(self, engine: GameEngine):
        """GIVEN player at /town/tavern WHEN cd .. THEN at /town."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "..")
        assert event.location == "/town"

    def test_move_to_absolute_path(self, engine: GameEngine):
        """GIVEN player anywhere WHEN cd /forest THEN at /forest."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        assert event.location == "/forest"

    def test_move_invalid_path(self, engine: GameEngine):
        """GIVEN player at /town/tavern WHEN cd nonexistent THEN error."""
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "nonexistent")
        assert event.error is not None
        assert "no such" in event.error.lower()

    def test_move_persists_location(self, engine: GameEngine):
        """GIVEN player moves WHEN checking DB THEN location updated."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        char = engine.db.get_character("alice", "#test")
        assert char is not None
        assert char.location == "/forest/clearing"


class TestInspection:
    """ls and cat commands."""

    def test_ls_shows_exits(self, engine: GameEngine):
        """GIVEN player at /town WHEN ls THEN exits listed."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")
        event = engine.look("alice", "#test", show_hidden=False)
        assert "tavern" in event.exits
        assert "blacksmith" in event.exits

    def test_ls_a_shows_hidden(self, engine: GameEngine):
        """GIVEN player at /town WHEN ls -a THEN .armory shown."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")
        event = engine.look("alice", "#test", show_hidden=True)
        assert ".armory" in event.exits

    def test_ls_shows_enemies(self, engine: GameEngine):
        """GIVEN player at /forest/clearing WHEN ls THEN enemies listed."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        event = engine.look("alice", "#test", show_hidden=False)
        assert len(event.enemies) > 0

    def test_ls_cleared_room_no_enemies(self, engine: GameEngine):
        """GIVEN room cleared recently WHEN ls THEN no enemies."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/clearing")
        engine.db.mark_room_cleared("#test", "/forest/clearing")
        event = engine.look("alice", "#test", show_hidden=False)
        assert len(event.enemies) == 0

    def test_cat_item_in_room(self, engine: GameEngine):
        """GIVEN an item in the room WHEN cat item THEN description returned."""
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        event = engine.examine("alice", "#test", "rusty_sword.txt")
        assert event.description is not None
        assert "sword" in event.description.lower() or "blade" in event.description.lower()

    def test_cat_nonexistent(self, engine: GameEngine):
        """GIVEN no such item WHEN cat THEN error."""
        engine.ensure_character("alice", "#test")
        event = engine.examine("alice", "#test", "nonexistent")
        assert event.error is not None


class TestCharacterInfo:
    """whoami and pwd commands."""

    def test_whoami_shows_stats(self, engine: GameEngine):
        """GIVEN a character WHEN whoami THEN stats returned."""
        engine.ensure_character("alice", "#test")
        info = engine.character_info("alice", "#test")
        assert info.nick == "alice"
        assert info.hp == 20
        assert info.level == 1

    def test_pwd_shows_location(self, engine: GameEngine):
        """GIVEN a character WHEN pwd THEN location returned."""
        engine.ensure_character("alice", "#test")
        loc = engine.current_location("alice", "#test")
        assert loc == "/town/tavern"


class TestRest:
    """sleep command."""

    def test_sleep_restores_hp(self, engine: GameEngine):
        """GIVEN character with reduced HP WHEN sleep THEN HP restored."""
        engine.ensure_character("alice", "#test")
        engine.db.update_character("alice", "#test", hp=10)
        event = engine.rest("alice", "#test")
        assert event.hp_restored > 0
        char = engine.db.get_character("alice", "#test")
        assert char is not None
        assert char.hp > 10

    def test_sleep_at_full_hp(self, engine: GameEngine):
        """GIVEN character at full HP WHEN sleep THEN no change."""
        engine.ensure_character("alice", "#test")
        event = engine.rest("alice", "#test")
        assert event.hp_restored == 0
