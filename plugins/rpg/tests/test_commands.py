"""Tests for RPG IRC command layer."""

from __future__ import annotations

from rpg.combat import CombatManager
from rpg.engine import GameEngine
from rpg.narrator import Narrator
from rpg.persistence import RPGDatabase
from rpg.world import WorldMap


def make_game(tmp_path):
    """Create a minimal game stack for testing commands."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    engine = GameEngine(db=db, world=world, spawn_cooldown_minutes=30)
    combat = CombatManager(db=db, world=world)
    narrator = Narrator(model="", api_key="", timeout=2)  # Fallback mode
    return engine, combat, narrator, db


class TestCommandFormatting:
    """IRC output formatting for game events."""

    def test_cd_formats_room(self, tmp_path):
        """GIVEN player moves WHEN cd /forest THEN output includes location."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "/forest")
        text = narrator.narrate_room(
            room_path=event.location,
            description_hint=event.description_hint,
            enemies=event.enemies,
            items=event.items,
            exits=event.exits,
        )
        assert "/forest" in text

    def test_cd_error_formats(self, tmp_path):
        """GIVEN invalid path WHEN cd THEN error message shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        event = engine.move("alice", "#test", "nonexistent")
        assert event.error is not None
        assert "no such" in event.error.lower()

    def test_whoami_formats_stats(self, tmp_path):
        """GIVEN a character WHEN whoami THEN stats formatted."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        info = engine.character_info("alice", "#test")
        line = (
            f"{info.nick} | HP:{info.hp}/{info.max_hp} ATK:{info.attack} "
            f"DEF:{info.defense} XP:{info.xp} LVL:{info.level} GOLD:{info.gold} "
            f"| {info.location}"
        )
        assert "alice" in line
        assert "HP:20/20" in line

    def test_pwd_formats_location(self, tmp_path):
        """GIVEN a character WHEN pwd THEN location shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        loc = engine.current_location("alice", "#test")
        assert loc == "/town/tavern"

    def test_sleep_formats_recovery(self, tmp_path):
        """GIVEN damaged character WHEN sleep THEN recovery shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        db.update_character("alice", "#test", hp=10)
        event = engine.rest("alice", "#test")
        assert event.hp_restored > 0

    def test_mv_pickup_formats(self, tmp_path):
        """GIVEN item in room WHEN mv THEN pickup confirmed."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        result = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert result.error is None
        assert result.item_name == "rusty_sword.txt"

    def test_ls_formats_room_contents(self, tmp_path):
        """GIVEN player in room WHEN ls THEN room contents formatted."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        event = engine.look("alice", "#test")
        assert event.location == "/town/tavern"
        assert event.error is None

    def test_ls_hidden_shows_dotfiles(self, tmp_path):
        """GIVEN player near hidden room WHEN ls -a THEN hidden exits appear."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/town")
        event_normal = engine.look("alice", "#test", show_hidden=False)
        event_hidden = engine.look("alice", "#test", show_hidden=True)
        # Hidden room .armory should only show with -a
        assert ".armory" not in event_normal.exits
        assert ".armory" in event_hidden.exits

    def test_cat_examine_item(self, tmp_path):
        """GIVEN item in room WHEN cat THEN description shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        event = engine.examine("alice", "#test", "rusty_sword.txt")
        assert event.description is not None
        assert "blade" in event.description.lower() or "ATK" in event.description

    def test_cat_examine_missing(self, tmp_path):
        """GIVEN no such target WHEN cat THEN error shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        event = engine.examine("alice", "#test", "nonexistent")
        assert event.error is not None
        assert "No such file" in event.error

    def test_man_help_text(self):
        """GIVEN man command WHEN known topic THEN help returned."""
        help_text = {
            "cd": "cd <path>",
            "ls": "ls [-a]",
            "cat": "cat <thing>",
            "rm": "rm <enemy>",
            "mv": "mv <item>",
            "pwd": "pwd",
            "whoami": "whoami",
            "man": "man <topic>",
            "sleep": "sleep",
            "history": "history",
        }
        for topic, expected_prefix in help_text.items():
            assert help_text[topic].startswith(expected_prefix)

    def test_sleep_full_hp_no_restore(self, tmp_path):
        """GIVEN full HP character WHEN sleep THEN no HP restored."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        event = engine.rest("alice", "#test")
        assert event.hp_restored == 0

    def test_mv_missing_item_error(self, tmp_path):
        """GIVEN no item in room WHEN mv THEN error shown."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        result = combat.pickup_item("alice", "#test", "nonexistent.txt")
        assert result.error is not None
        assert "No such file" in result.error

    def test_mv_pickup_with_bonuses(self, tmp_path):
        """GIVEN weapon in room WHEN mv THEN bonuses included."""
        engine, combat, narrator, db = make_game(tmp_path)
        engine.ensure_character("alice", "#test")
        engine.move("alice", "#test", "/forest/cave")
        result = combat.pickup_item("alice", "#test", "rusty_sword.txt")
        assert result.attack_bonus == 2
        assert result.defense_bonus == 0
