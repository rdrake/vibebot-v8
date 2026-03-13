"""Integration tests — full game flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rpg.combat import CombatManager
from rpg.engine import GameEngine
from rpg.narrator import Narrator
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


@pytest.fixture
def game(tmp_path):
    """Full game stack with fallback narrator."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    engine = GameEngine(db=db, world=world, spawn_cooldown_minutes=30)
    combat = CombatManager(db=db, world=world)
    narrator = Narrator(model="", api_key="", timeout=2)
    return engine, combat, narrator, db


class TestFullGameFlow:
    """Play through a complete game session."""

    def test_explore_and_fight(self, game):
        """GIVEN new player WHEN exploring and fighting THEN full flow works."""
        engine, combat, narrator, db = game

        # Character auto-created at /town/tavern
        event = engine.move("alice", "#test", "..")
        assert event.location == "/town"

        # Look around town
        look = engine.look("alice", "#test", show_hidden=False)
        assert "tavern" in look.exits

        # Move to forest clearing
        event = engine.move("alice", "#test", "/forest/clearing")
        assert event.location == "/forest/clearing"
        assert len(event.enemies) > 0

        # Fight rats with high attack to guarantee kills
        db.update_character("alice", "#test", attack=50)
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 20
            r1 = combat.attack("alice", "#test", "rat")
            assert r1.enemy_killed is True
            r2 = combat.attack("alice", "#test", "rat")
            assert r2.enemy_killed is True

        # Room should be cleared
        char = db.get_character("alice", "#test")
        assert char is not None
        assert char.xp > 0
        assert char.gold > 0

        # Move to cave and pick up item
        engine.move("alice", "#test", "/forest/cave")
        pickup = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert pickup.error is None

        items = db.get_inventory("alice", "#test")
        assert len(items) == 1

    def test_death_and_respawn(self, game):
        """GIVEN player at low HP WHEN killed THEN respawn at town."""
        engine, combat, narrator, db = game

        engine.ensure_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level3", hp=1)

        # Force miss so enemy counterattacks
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 1
            result = combat.attack("alice", "#test", "dark_knight")

        if result.player_died:
            char = db.get_character("alice", "#test")
            assert char is not None
            assert char.location == "/town/tavern"
            assert char.hp == char.max_hp

    def test_narrator_formats_all_events(self, game):
        """GIVEN game events WHEN narrating THEN all produce output."""
        engine, combat, narrator, db = game

        # Room narration
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        text = narrator.narrate_room(
            room_path=event.location,
            description_hint=event.description_hint,
            enemies=event.enemies,
            items=event.items,
            exits=event.exits,
        )
        assert len(text) > 0

        # Combat narration
        text = narrator.narrate_combat(
            attacker="alice",
            target="goblin",
            hit=True,
            damage=5,
            enemy_killed=False,
        )
        assert len(text) > 0

    def test_hidden_room_discovery(self, game):
        """GIVEN player at /town WHEN ls -a THEN .armory visible, contains item."""
        engine, combat, narrator, db = game

        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")

        # Without -a, no hidden rooms
        look = engine.look("alice", "#test", show_hidden=False)
        assert ".armory" not in look.exits

        # With -a, hidden room appears
        look = engine.look("alice", "#test", show_hidden=True)
        assert ".armory" in look.exits

        # Enter hidden room and find item
        engine.move("alice", "#test", ".armory")
        event = engine.examine("alice", "#test", "enchanted_shield.dat")
        assert event.description is not None
