"""Tests for RPG plugin.py IRC command methods.

These tests exercise the actual Rpg nested-class command methods in plugin.py
by extracting the original (pre-wrap) functions from Limnoria's wrap() closure
and calling them directly with mocked IRC objects and real game logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from rpg.combat import CombatManager, CombatResult, PickupResult
from rpg.engine import GameEngine, LookEvent
from rpg.narrator import Narrator
from rpg.persistence import InventoryItem, RPGDatabase
from rpg.plugin import RPG
from rpg.world import WorldMap

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — extract original unwrapped command functions
# ---------------------------------------------------------------------------


def _unwrap(wrapped_func):
    """Extract the original function from Limnoria's wrap() closure.

    wrap() creates a closure ``newf(self, irc, msg, args)`` that stores the
    original function in ``__closure__[0]``.  We pull it out so we can call
    the original signature directly (self, irc, msg, args, <parsed_args>).
    """
    for cell in wrapped_func.__closure__ or ():
        obj = cell.cell_contents
        if callable(obj) and obj is not wrapped_func:
            return obj
    return wrapped_func  # fallback: already unwrapped


# Pre-extract all original command functions once at import time.
_cd = _unwrap(RPG.Rpg.cd)
_ls = _unwrap(RPG.Rpg.ls)
_cat = _unwrap(RPG.Rpg.cat)
_rm = _unwrap(RPG.Rpg.rm)
_mv = _unwrap(RPG.Rpg.mv)
_pwd = _unwrap(RPG.Rpg.pwd)
_whoami = _unwrap(RPG.Rpg.whoami)
_man = _unwrap(RPG.Rpg.man)
_sleep = _unwrap(RPG.Rpg.sleep)
_history = _unwrap(RPG.Rpg.history)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def game_stack(tmp_path: Path):
    """Create real game components for testing."""
    db = RPGDatabase(str(tmp_path / "test.db"))
    world = WorldMap.starter()
    engine = GameEngine(db=db, world=world, spawn_cooldown_minutes=30)
    combat = CombatManager(db=db, world=world)
    narrator = Narrator(model="", api_key="", timeout=2)  # Fallback mode
    return engine, combat, narrator, db


@pytest.fixture
def mock_plugin(game_stack):
    """Create a mock RPG plugin with real game logic wired in."""
    engine, combat, narrator, db = game_stack
    plugin = MagicMock(spec=RPG)
    plugin._engine = engine
    plugin._combat = combat
    plugin._narrator = narrator
    plugin._db = db

    # Wire through real helper methods
    plugin._get_nick = RPG._get_nick.__get__(plugin, RPG)
    plugin._get_channel = RPG._get_channel.__get__(plugin, RPG)
    plugin._check_enabled = RPG._check_enabled.__get__(plugin, RPG)

    # Default: enabled
    plugin.registryValue.return_value = True

    return plugin


@pytest.fixture
def irc(mock_plugin):
    """Create mock IRC with getCallback returning our mock plugin."""
    mock_irc = MagicMock()
    mock_irc.getCallback.return_value = mock_plugin
    return mock_irc


@pytest.fixture
def msg():
    """Create a mock IRC message."""
    m = MagicMock()
    m.nick = "alice"
    m.channel = "#test"
    m.args = ("#test", "some text")
    return m


@pytest.fixture
def rpg_self():
    """Create a minimal stand-in for ``self`` (the Rpg Commands instance).

    The unwrapped functions receive ``self`` but only use it to call
    ``irc.getCallback("RPG")`` — they never touch ``self`` directly.
    """
    return MagicMock()


@pytest.fixture
def setup_character(mock_plugin):
    """Ensure character exists in game before commands run."""
    mock_plugin._engine.ensure_character("alice", "#test")


# ---------------------------------------------------------------------------
# Plugin __init__ and die
# ---------------------------------------------------------------------------


class TestPluginLifecycle:
    """Test RPG plugin __init__ and die."""

    @patch("rpg.plugin.RPGDatabase")
    @patch("rpg.plugin.WorldMap")
    @patch("rpg.plugin.GameEngine")
    @patch("rpg.plugin.CombatManager")
    @patch("rpg.plugin.Narrator")
    @patch("rpg.plugin.conf")
    @patch("rpg.plugin.log")
    @patch("rpg.plugin.callbacks.Plugin.__init__", return_value=None)
    def test_init_sets_up_components(
        self,
        mock_super,
        mock_log,
        mock_conf,
        mock_narrator,
        mock_combat,
        mock_engine,
        mock_world,
        mock_db,
    ):
        """GIVEN RPG plugin WHEN __init__ called THEN sets up db, world, engine, combat, narrator."""
        mock_irc = MagicMock()

        def registry_side_effect(key):
            return {
                "databasePath": "/tmp/test.db",
                "spawnCooldownMinutes": 30,
                "narratorModel": "gpt-4",
                "narratorApiKey": "test-key",
                "narratorTimeout": 5,
            }.get(key, "")

        with patch.object(RPG, "registryValue", side_effect=registry_side_effect):
            RPG(mock_irc)

        mock_db.assert_called_once_with("/tmp/test.db")
        mock_world.starter.assert_called_once()
        mock_engine.assert_called_once()
        mock_combat.assert_called_once()
        mock_narrator.assert_called_once()

    @patch("rpg.plugin.RPGDatabase")
    @patch("rpg.plugin.WorldMap")
    @patch("rpg.plugin.GameEngine")
    @patch("rpg.plugin.CombatManager")
    @patch("rpg.plugin.Narrator")
    @patch("rpg.plugin.conf")
    @patch("rpg.plugin.log")
    @patch("rpg.plugin.callbacks.Plugin.__init__", return_value=None)
    def test_init_uses_default_db_path_when_empty(
        self,
        mock_super,
        mock_log,
        mock_conf,
        mock_narrator,
        mock_combat,
        mock_engine,
        mock_world,
        mock_db,
    ):
        """GIVEN empty databasePath WHEN __init__ THEN uses data dir default."""
        mock_irc = MagicMock()
        mock_conf.supybot.directories.data.return_value = "/data"

        def registry_side_effect(key):
            return {
                "databasePath": "",
                "spawnCooldownMinutes": 30,
                "narratorModel": "",
                "narratorApiKey": "",
                "narratorTimeout": 2,
            }.get(key, "")

        with patch.object(RPG, "registryValue", side_effect=registry_side_effect):
            RPG(mock_irc)

        mock_db.assert_called_once_with("/data/RPG.db")

    @patch("rpg.plugin.callbacks.Plugin.die")
    @patch("rpg.plugin.callbacks.Plugin.__init__", return_value=None)
    def test_die_calls_super(self, mock_init, mock_die):
        """GIVEN RPG plugin WHEN die called THEN calls super().die()."""
        plugin = RPG.__new__(RPG)
        plugin.die()
        mock_die.assert_called_once()


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Test _get_nick, _get_channel, _check_enabled."""

    def test_get_nick_from_msg(self, mock_plugin, msg):
        """GIVEN msg with nick WHEN _get_nick THEN returns nick."""
        assert mock_plugin._get_nick(msg) == "alice"

    def test_get_nick_fallback(self, mock_plugin):
        """GIVEN msg with no nick WHEN _get_nick THEN returns 'unknown'."""
        m = MagicMock()
        m.nick = ""
        assert mock_plugin._get_nick(m) == "unknown"

    def test_get_channel_from_msg(self, mock_plugin, msg):
        """GIVEN msg with channel WHEN _get_channel THEN returns channel."""
        assert mock_plugin._get_channel(msg) == "#test"

    def test_get_channel_fallback_to_args(self, mock_plugin):
        """GIVEN msg with no channel WHEN _get_channel THEN uses args[0]."""
        m = MagicMock()
        m.channel = ""
        m.args = ("#fallback", "text")
        assert mock_plugin._get_channel(m) == "#fallback"

    def test_check_enabled_true(self, mock_plugin, irc, msg):
        """GIVEN enabled channel WHEN _check_enabled THEN returns True."""
        mock_plugin.registryValue.return_value = True
        assert mock_plugin._check_enabled(irc, msg) is True
        irc.reply.assert_not_called()

    def test_check_enabled_false(self, mock_plugin, irc, msg):
        """GIVEN disabled channel WHEN _check_enabled THEN returns False and replies."""
        mock_plugin.registryValue.return_value = False
        assert mock_plugin._check_enabled(irc, msg) is False
        irc.reply.assert_called_once()
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# cd command
# ---------------------------------------------------------------------------


class TestCdCommand:
    """Test the cd command method."""

    def test_cd_success(self, rpg_self, irc, msg, setup_character):
        """GIVEN player WHEN cd /forest THEN narrates room."""
        _cd(rpg_self, irc, msg, [], "/forest")
        irc.reply.assert_called_once()
        assert "/forest" in irc.reply.call_args[0][0]

    def test_cd_error(self, rpg_self, irc, msg, setup_character):
        """GIVEN player WHEN cd nonexistent THEN replies with error."""
        _cd(rpg_self, irc, msg, [], "nonexistent")
        irc.reply.assert_called_once()
        assert "no such" in irc.reply.call_args[0][0].lower()

    def test_cd_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled channel WHEN cd THEN replies not enabled."""
        mock_plugin.registryValue.return_value = False
        _cd(rpg_self, irc, msg, [], "/forest")
        irc.reply.assert_called_once()
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# ls command
# ---------------------------------------------------------------------------


class TestLsCommand:
    """Test the ls command method."""

    def test_ls_success(self, rpg_self, irc, msg, setup_character):
        """GIVEN player WHEN ls THEN shows room info."""
        _ls(rpg_self, irc, msg, [], None)
        irc.reply.assert_called_once()
        assert "Exits:" in irc.reply.call_args[0][0]

    def test_ls_with_hidden_flag(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN player in /town WHEN ls -a THEN shows hidden rooms."""
        mock_plugin._engine.move("alice", "#test", "/town")
        _ls(rpg_self, irc, msg, [], "-a")
        assert ".armory" in irc.reply.call_args[0][0]

    def test_ls_error(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN engine returns error WHEN ls THEN shows error."""
        mock_plugin._engine = MagicMock()
        mock_plugin._engine.look.return_value = LookEvent(
            location="",
            description_hint="",
            enemies=[],
            items=[],
            exits=[],
            error="not found",
        )
        _ls(rpg_self, irc, msg, [], None)
        irc.reply.assert_called_once_with("not found")

    def test_ls_with_enemies_and_items(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN room with enemies and items WHEN ls THEN all shown."""
        mock_plugin._engine = MagicMock()
        mock_plugin._engine.look.return_value = LookEvent(
            location="/cave",
            description_hint="dark cave",
            enemies=["goblin"],
            items=["sword.txt"],
            exits=["north", "south"],
        )
        _ls(rpg_self, irc, msg, [], None)
        reply = irc.reply.call_args[0][0]
        assert "Enemies: goblin" in reply
        assert "Items: sword.txt" in reply
        assert "Exits: north, south" in reply

    def test_ls_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN ls THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _ls(rpg_self, irc, msg, [], None)
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# cat command
# ---------------------------------------------------------------------------


class TestCatCommand:
    """Test the cat command method."""

    def test_cat_success(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN item in room WHEN cat THEN description shown."""
        mock_plugin._engine.move("alice", "#test", "/forest/cave")
        _cat(rpg_self, irc, msg, [], "rusty_sword.txt")
        assert "rusty_sword.txt:" in irc.reply.call_args[0][0]

    def test_cat_error(self, rpg_self, irc, msg, setup_character):
        """GIVEN no such target WHEN cat THEN error shown."""
        _cat(rpg_self, irc, msg, [], "nonexistent")
        assert "No such file" in irc.reply.call_args[0][0]

    def test_cat_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN cat THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _cat(rpg_self, irc, msg, [], "something")
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# rm command
# ---------------------------------------------------------------------------


class TestRmCommand:
    """Test the rm (attack) command method."""

    def test_rm_error_no_enemy(self, rpg_self, irc, msg, setup_character):
        """GIVEN no enemy WHEN rm THEN error shown."""
        _rm(rpg_self, irc, msg, [], "ghost")
        assert irc.reply.call_args[0][0]  # some error message

    def test_rm_success_with_kill(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN enemy killed WHEN rm THEN shows XP and gold."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.attack.return_value = CombatResult(
            hit=True,
            damage=10,
            enemy_name="goblin",
            enemy_killed=True,
            xp_gained=50,
            gold_gained=10,
            leveled_up=False,
            new_level=1,
            counterattack_damage=0,
            player_died=False,
        )
        mock_plugin._narrator = MagicMock()
        mock_plugin._narrator.narrate_combat.return_value = "You slash the goblin!"

        _rm(rpg_self, irc, msg, [], "goblin")
        reply = irc.reply.call_args[0][0]
        assert "+50 XP" in reply
        assert "+10 gold" in reply

    def test_rm_with_counterattack(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN enemy hits back WHEN rm THEN shows counterattack damage."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.attack.return_value = CombatResult(
            hit=True,
            damage=5,
            enemy_name="orc",
            enemy_killed=False,
            xp_gained=0,
            gold_gained=0,
            leveled_up=False,
            new_level=1,
            counterattack_damage=3,
            player_died=False,
        )
        mock_plugin._narrator = MagicMock()
        mock_plugin._narrator.narrate_combat.return_value = "You swing!"

        _rm(rpg_self, irc, msg, [], "orc")
        assert "hits back for 3 dmg" in irc.reply.call_args[0][0]

    def test_rm_with_level_up(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN enough XP WHEN rm kills THEN shows level up."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.attack.return_value = CombatResult(
            hit=True,
            damage=20,
            enemy_name="dragon",
            enemy_killed=True,
            xp_gained=500,
            gold_gained=100,
            leveled_up=True,
            new_level=3,
            counterattack_damage=0,
            player_died=False,
        )
        mock_plugin._narrator = MagicMock()
        mock_plugin._narrator.narrate_combat.return_value = "You vanquish the dragon!"

        _rm(rpg_self, irc, msg, [], "dragon")
        reply = irc.reply.call_args[0][0]
        assert "LEVEL UP" in reply
        assert "level 3" in reply

    def test_rm_player_died(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN fatal counterattack WHEN rm THEN shows death message."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.attack.return_value = CombatResult(
            hit=False,
            damage=0,
            enemy_name="boss",
            enemy_killed=False,
            xp_gained=0,
            gold_gained=0,
            leveled_up=False,
            new_level=1,
            counterattack_damage=100,
            player_died=True,
        )
        mock_plugin._narrator = MagicMock()
        mock_plugin._narrator.narrate_combat.return_value = "You miss!"

        _rm(rpg_self, irc, msg, [], "boss")
        reply = irc.reply.call_args[0][0]
        assert "You died" in reply
        assert "Respawning" in reply

    def test_rm_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN rm THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _rm(rpg_self, irc, msg, [], "enemy")
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# mv command
# ---------------------------------------------------------------------------


class TestMvCommand:
    """Test the mv (pickup item) command method."""

    def test_mv_success(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN item in room WHEN mv THEN picks up item."""
        mock_plugin._engine.move("alice", "#test", "/forest/cave")
        _mv(rpg_self, irc, msg, [], "rusty_sword.txt ~/inventory")
        assert "Picked up rusty_sword.txt" in irc.reply.call_args[0][0]

    def test_mv_with_bonuses(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN weapon with bonuses WHEN mv THEN shows bonus info."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.pickup_item.return_value = PickupResult(
            item_name="magic_sword.txt",
            attack_bonus=5,
            defense_bonus=2,
        )
        _mv(rpg_self, irc, msg, [], "magic_sword.txt ~/inventory")
        reply = irc.reply.call_args[0][0]
        assert "ATK +5" in reply
        assert "DEF +2" in reply

    def test_mv_no_bonus(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN item with no bonuses WHEN mv THEN no brackets shown."""
        mock_plugin._combat = MagicMock()
        mock_plugin._combat.pickup_item.return_value = PickupResult(
            item_name="scroll.txt",
            attack_bonus=0,
            defense_bonus=0,
        )
        _mv(rpg_self, irc, msg, [], "scroll.txt ~/inventory")
        reply = irc.reply.call_args[0][0]
        assert reply == "Picked up scroll.txt"
        assert "[" not in reply

    def test_mv_error(self, rpg_self, irc, msg, setup_character):
        """GIVEN no such item WHEN mv THEN error shown."""
        _mv(rpg_self, irc, msg, [], "nonexistent ~/inventory")
        assert irc.reply.call_args[0][0]  # some error

    def test_mv_empty_item_name(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN empty text WHEN mv THEN missing operand error."""
        _mv(rpg_self, irc, msg, [], "")
        irc.reply.assert_called_once_with("mv: missing operand")

    def test_mv_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN mv THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _mv(rpg_self, irc, msg, [], "item")
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# pwd command
# ---------------------------------------------------------------------------


class TestPwdCommand:
    """Test the pwd command method."""

    def test_pwd_shows_location(self, rpg_self, irc, msg, setup_character):
        """GIVEN player WHEN pwd THEN shows current location."""
        _pwd(rpg_self, irc, msg, [])
        irc.reply.assert_called_once_with("/town/tavern")

    def test_pwd_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN pwd THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _pwd(rpg_self, irc, msg, [])
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# whoami command
# ---------------------------------------------------------------------------


class TestWhoamiCommand:
    """Test the whoami command method."""

    def test_whoami_basic_stats(self, rpg_self, irc, msg, setup_character):
        """GIVEN character WHEN whoami THEN shows stats."""
        _whoami(rpg_self, irc, msg, [])
        reply = irc.reply.call_args[0][0]
        assert "alice" in reply
        assert "HP:" in reply
        assert "ATK:" in reply
        assert "DEF:" in reply
        assert "LVL:" in reply

    def test_whoami_with_equipped(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN character with equipped items WHEN whoami THEN shows equipped."""
        mock_plugin._db.get_inventory = MagicMock(
            return_value=[
                InventoryItem(
                    id=1,
                    nick="alice",
                    channel="#test",
                    name="magic_sword.txt",
                    attack_bonus=3,
                    defense_bonus=0,
                    equipped=True,
                ),
                InventoryItem(
                    id=2,
                    nick="alice",
                    channel="#test",
                    name="shield.txt",
                    attack_bonus=0,
                    defense_bonus=2,
                    equipped=True,
                ),
                InventoryItem(
                    id=3,
                    nick="alice",
                    channel="#test",
                    name="potion.txt",
                    attack_bonus=0,
                    defense_bonus=0,
                    equipped=False,
                ),
            ]
        )
        _whoami(rpg_self, irc, msg, [])
        reply = irc.reply.call_args[0][0]
        assert "Equipped: magic_sword.txt, shield.txt" in reply

    def test_whoami_no_equipped(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN character with no equipped items WHEN whoami THEN no Equipped shown."""
        mock_plugin._db.get_inventory = MagicMock(return_value=[])
        _whoami(rpg_self, irc, msg, [])
        assert "Equipped" not in irc.reply.call_args[0][0]

    def test_whoami_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN whoami THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _whoami(rpg_self, irc, msg, [])
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# man command
# ---------------------------------------------------------------------------


class TestManCommand:
    """Test the man command method."""

    def test_man_known_command(self, rpg_self, irc, msg, setup_character):
        """GIVEN known topic WHEN man cd THEN shows help."""
        _man(rpg_self, irc, msg, [], "cd")
        assert "cd <path>" in irc.reply.call_args[0][0]

    def test_man_all_commands(self, rpg_self, irc, msg, setup_character):
        """GIVEN all help topics WHEN man each THEN help returned."""
        for topic in ("cd", "ls", "cat", "rm", "mv", "pwd", "whoami", "man", "sleep", "history"):
            irc.reset_mock()
            _man(rpg_self, irc, msg, [], topic)
            assert topic in irc.reply.call_args[0][0]

    def test_man_unknown_topic(self, rpg_self, irc, msg, setup_character):
        """GIVEN unknown topic WHEN man THEN no manual entry."""
        _man(rpg_self, irc, msg, [], "unknown_thing")
        assert "no manual entry" in irc.reply.call_args[0][0]

    def test_man_examines_room_first(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN examinable thing WHEN man THEN shows description."""
        mock_plugin._engine.move("alice", "#test", "/forest/cave")
        _man(rpg_self, irc, msg, [], "rusty_sword.txt")
        assert "rusty_sword.txt:" in irc.reply.call_args[0][0]

    def test_man_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN man THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _man(rpg_self, irc, msg, [], "cd")
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# sleep command
# ---------------------------------------------------------------------------


class TestSleepCommand:
    """Test the sleep command method."""

    def test_sleep_restores_hp(self, rpg_self, irc, msg, mock_plugin, setup_character):
        """GIVEN damaged character WHEN sleep THEN HP restored."""
        mock_plugin._db.update_character("alice", "#test", hp=10)
        _sleep(rpg_self, irc, msg, [])
        assert "HP restored" in irc.reply.call_args[0][0]

    def test_sleep_full_hp(self, rpg_self, irc, msg, setup_character):
        """GIVEN full HP WHEN sleep THEN already full message."""
        _sleep(rpg_self, irc, msg, [])
        assert "already at full HP" in irc.reply.call_args[0][0]

    def test_sleep_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN sleep THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _sleep(rpg_self, irc, msg, [])
        assert "not enabled" in irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------


class TestHistoryCommand:
    """Test the history command method."""

    def test_history_placeholder(self, rpg_self, irc, msg, setup_character):
        """GIVEN player WHEN history THEN shows not implemented."""
        _history(rpg_self, irc, msg, [])
        irc.reply.assert_called_once_with("history: not yet implemented")

    def test_history_disabled(self, rpg_self, irc, msg, mock_plugin):
        """GIVEN disabled WHEN history THEN not enabled."""
        mock_plugin.registryValue.return_value = False
        _history(rpg_self, irc, msg, [])
        assert "not enabled" in irc.reply.call_args[0][0]
