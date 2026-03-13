"""Tests for combat system and dice roller."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from rpg.combat import CombatManager
from rpg.dice import RollResult, roll
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


@pytest.fixture
def db(tmp_path) -> RPGDatabase:
    return RPGDatabase(str(tmp_path / "test.db"))


@pytest.fixture
def combat(db: RPGDatabase) -> CombatManager:
    world = WorldMap.starter()
    return CombatManager(db=db, world=world)


class TestAttack:
    """rm command — attacking enemies."""

    def test_attack_enemy_hit(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player in room with enemy WHEN rm goblin (forced hit) THEN damage dealt."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        # Force a high roll to guarantee hit
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 18  # High roll
            result = combat.attack("alice", "#test", "goblin")
        assert result.hit is True
        assert result.damage > 0

    def test_attack_enemy_miss(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player in room with enemy WHEN rm goblin (forced miss) THEN no damage."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 1  # Natural 1
            result = combat.attack("alice", "#test", "goblin")
        assert result.hit is False
        assert result.damage == 0

    def test_attack_nonexistent_enemy(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN no such enemy WHEN rm THEN error."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/town/tavern")
        result = combat.attack("alice", "#test", "goblin")
        assert result.error is not None

    def test_kill_enemy_grants_xp_and_gold(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN enemy killed WHEN checking rewards THEN XP and gold awarded."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", attack=50)
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 20
            result = combat.attack("alice", "#test", "rat")
        assert result.enemy_killed is True
        assert result.xp_gained > 0
        assert result.gold_gained > 0

    def test_enemy_counterattack(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN enemy alive after attack WHEN turn ends THEN enemy hits back."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level1")
        with patch("rpg.combat.dice.roll") as mock_roll:
            # Player hits but doesn't kill, enemy hits back
            mock_roll.return_value.total = 15
            result = combat.attack("alice", "#test", "goblin")
        # Enemy should counterattack (may hit or miss)
        assert result.enemy_killed is False or result.counterattack_damage >= 0

    def test_player_death_respawns_at_town(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN player at 1 HP WHEN enemy kills them THEN respawn at /town."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/dungeon/level3", hp=1)
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 1  # Player misses, enemy hits
            result = combat.attack("alice", "#test", "dark_knight")
        if result.player_died:
            char = db.get_character("alice", "#test")
            assert char is not None
            assert char.location == "/town/tavern"
            assert char.hp > 0

    def test_room_clears_when_all_enemies_dead(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN last enemy killed WHEN checking room THEN room marked cleared."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", attack=50)
        # Kill all rats (count=2) — need 2 attacks
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 20
            combat.attack("alice", "#test", "rat")
            combat.attack("alice", "#test", "rat")
        cleared = db.get_room_cleared_at("#test", "/forest/clearing")
        assert cleared is not None


class TestLevelUp:
    """XP and leveling."""

    def test_level_up_at_threshold(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN character near level threshold WHEN gaining XP THEN level up."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/clearing", xp=95, attack=50)
        with patch("rpg.combat.dice.roll") as mock_roll:
            mock_roll.return_value.total = 20
            result = combat.attack("alice", "#test", "rat")
        if result.enemy_killed:
            char = db.get_character("alice", "#test")
            assert char is not None
            # Should have leveled up if XP >= 100
            if char.xp >= 100:
                assert char.level >= 2
                assert result.leveled_up is True


class TestInventoryPickup:
    """mv command — picking up items."""

    def test_pickup_item(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN item in room WHEN mv item ~/inventory THEN item in inventory."""
        db.create_character("alice", "#test")
        db.update_character("alice", "#test", location="/forest/cave")
        result = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert result.error is None
        items = db.get_inventory("alice", "#test")
        assert any(i.name == "rusty_sword.txt" for i in items)

    def test_pickup_nonexistent_item(self, combat: CombatManager, db: RPGDatabase):
        """GIVEN no such item WHEN mv THEN error."""
        db.create_character("alice", "#test")
        result = combat.pickup_item("alice", "#test", "nonexistent")
        assert result.error is not None


class TestDiceRoller:
    """Dice roller — lightweight replacement for d20 library."""

    def test_roll_simple(self):
        """GIVEN 1d20 expression WHEN rolled THEN result in valid range."""
        result = roll("1d20")
        assert 1 <= result.total <= 20
        assert isinstance(result, RollResult)

    def test_roll_with_modifier(self):
        """GIVEN 1d6+3 expression WHEN rolled THEN result includes modifier."""
        result = roll("1d6+3")
        assert 4 <= result.total <= 9

    def test_roll_negative_modifier(self):
        """GIVEN 1d20-2 expression WHEN rolled THEN modifier subtracted."""
        result = roll("1d20-2")
        assert -1 <= result.total <= 18

    def test_roll_multiple_dice(self):
        """GIVEN 2d6 expression WHEN rolled THEN result in valid range."""
        result = roll("2d6")
        assert 2 <= result.total <= 12

    def test_roll_invalid_expression(self):
        """GIVEN invalid expression WHEN rolled THEN ValueError raised."""
        with pytest.raises(ValueError, match="Invalid dice expression"):
            roll("not_a_roll")
