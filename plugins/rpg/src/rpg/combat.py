"""Combat system — d20-based attack resolution, XP, loot, death."""

from __future__ import annotations

from typing import NamedTuple

from . import dice
from .persistence import RPGDatabase
from .world import EnemySpawn, WorldMap

# XP required to reach each level: level N requires XP_TABLE[N]
# Level 1 = 0 XP, Level 2 = 100 XP, etc.
XP_TABLE = [0, 0, 100, 250, 500, 800, 1200, 1700, 2400, 3200, 4200]
MAX_LEVEL = len(XP_TABLE) - 1

# Stats gained per level up
HP_PER_LEVEL = 5
ATTACK_PER_LEVEL = 1
DEFENSE_PER_LEVEL = 1

# Death penalty: lose 10% of XP (but never below current level threshold)
DEATH_XP_PENALTY_PCT = 10


class CombatResult(NamedTuple):
    """Result of an attack action."""

    hit: bool
    damage: int
    enemy_name: str
    enemy_killed: bool
    xp_gained: int
    gold_gained: int
    leveled_up: bool
    new_level: int
    counterattack_damage: int
    player_died: bool
    error: str | None = None


class PickupResult(NamedTuple):
    """Result of picking up an item."""

    item_name: str
    attack_bonus: int
    defense_bonus: int
    error: str | None = None


class CombatManager:
    """Handles attack resolution, rewards, death, and item pickup."""

    def __init__(self, *, db: RPGDatabase, world: WorldMap) -> None:
        self.db = db
        self.world = world
        # Track active enemy HP per channel/room: {(channel, room, enemy_name, index): hp}
        self._enemy_hp: dict[tuple[str, str, str, int], int] = {}

    def _get_enemy_spawn(self, room_path: str, enemy_name: str) -> EnemySpawn | None:
        """Find an enemy spawn definition in a room."""
        room = self.world.get_room(room_path)
        if room is None:
            return None
        for spawn in room.spawns:
            if spawn.name == enemy_name:
                return spawn
        return None

    def _enemy_key(
        self, channel: str, room: str, name: str, index: int
    ) -> tuple[str, str, str, int]:
        return (channel.lower(), room, name, index)

    def _get_or_init_enemy_hp(self, channel: str, room: str, spawn: EnemySpawn, index: int) -> int:
        """Get current enemy HP, initializing if needed."""
        key = self._enemy_key(channel, room, spawn.name, index)
        if key not in self._enemy_hp:
            self._enemy_hp[key] = spawn.hp
        return self._enemy_hp[key]

    def _find_live_enemy_index(self, channel: str, room: str, spawn: EnemySpawn) -> int | None:
        """Find the first alive instance of an enemy type in a room."""
        for i in range(spawn.count):
            hp = self._get_or_init_enemy_hp(channel, room, spawn, i)
            if hp > 0:
                return i
        return None

    def _check_room_cleared(self, channel: str, room_path: str) -> bool:
        """Check if all enemies in a room are dead."""
        room = self.world.get_room(room_path)
        if room is None or not room.spawns:
            return True
        for spawn in room.spawns:
            for i in range(spawn.count):
                key = self._enemy_key(channel, room_path, spawn.name, i)
                hp = self._enemy_hp.get(key, spawn.hp)
                if hp > 0:
                    return False
        return True

    def _check_level_up(self, nick: str, channel: str) -> tuple[bool, int]:
        """Check and apply level up if XP threshold reached."""
        char = self.db.get_character(nick, channel)
        assert char is not None
        if char.level >= MAX_LEVEL:
            return False, char.level

        next_level = char.level + 1
        if next_level < len(XP_TABLE) and char.xp >= XP_TABLE[next_level]:
            new_max_hp = char.max_hp + HP_PER_LEVEL
            self.db.update_character(
                nick,
                channel,
                level=next_level,
                max_hp=new_max_hp,
                hp=new_max_hp,  # Full heal on level up
                attack=char.attack + ATTACK_PER_LEVEL,
                defense=char.defense + DEFENSE_PER_LEVEL,
            )
            return True, next_level
        return False, char.level

    def _apply_death(self, nick: str, channel: str) -> None:
        """Handle player death: respawn at town with XP penalty."""
        char = self.db.get_character(nick, channel)
        assert char is not None

        # XP penalty: lose 10% but never drop below current level floor
        level_floor = XP_TABLE[char.level] if char.level < len(XP_TABLE) else 0
        xp_loss = max(0, char.xp * DEATH_XP_PENALTY_PCT // 100)
        new_xp = max(level_floor, char.xp - xp_loss)

        self.db.update_character(
            nick,
            channel,
            hp=char.max_hp,
            location="/town/tavern",
            xp=new_xp,
        )

    def _get_equipped_bonuses(self, nick: str, channel: str) -> tuple[int, int]:
        """Get total attack and defense bonuses from equipped items."""
        items = self.db.get_inventory(nick, channel)
        atk = sum(i.attack_bonus for i in items if i.equipped)
        dfn = sum(i.defense_bonus for i in items if i.equipped)
        return atk, dfn

    def attack(self, nick: str, channel: str, enemy_name: str) -> CombatResult:
        """Attack an enemy (rm command)."""
        char = self.db.get_character(nick, channel)
        if char is None:
            return CombatResult(
                hit=False,
                damage=0,
                enemy_name=enemy_name,
                enemy_killed=False,
                xp_gained=0,
                gold_gained=0,
                leveled_up=False,
                new_level=1,
                counterattack_damage=0,
                player_died=False,
                error="rm: you don't exist yet. Use any command to create a character.",
            )

        spawn = self._get_enemy_spawn(char.location, enemy_name)
        if spawn is None:
            return CombatResult(
                hit=False,
                damage=0,
                enemy_name=enemy_name,
                enemy_killed=False,
                xp_gained=0,
                gold_gained=0,
                leveled_up=False,
                new_level=char.level,
                counterattack_damage=0,
                player_died=False,
                error=f"rm: cannot remove '{enemy_name}': No such file or directory",
            )

        index = self._find_live_enemy_index(channel, char.location, spawn)
        if index is None:
            return CombatResult(
                hit=False,
                damage=0,
                enemy_name=enemy_name,
                enemy_killed=False,
                xp_gained=0,
                gold_gained=0,
                leveled_up=False,
                new_level=char.level,
                counterattack_damage=0,
                player_died=False,
                error=f"rm: '{enemy_name}': already dead",
            )

        # Player attack roll: d20 + player_attack vs enemy_defense + 10
        equip_atk, equip_def = self._get_equipped_bonuses(nick, channel)
        total_attack = char.attack + equip_atk
        attack_roll = dice.roll(f"1d20+{total_attack}")
        target_ac = spawn.defense + 10

        hit = attack_roll.total >= target_ac
        damage = 0
        enemy_killed = False
        xp_gained = 0
        gold_gained = 0

        if hit:
            # Damage: 1d6 + attack bonus
            damage_roll = dice.roll(f"1d6+{total_attack // 2}")
            damage = max(1, damage_roll.total)

            key = self._enemy_key(channel, char.location, spawn.name, index)
            self._enemy_hp[key] = max(0, self._enemy_hp[key] - damage)

            if self._enemy_hp[key] <= 0:
                enemy_killed = True
                xp_gained = spawn.xp_reward
                gold_gained = spawn.gold_reward

                # Award XP and gold
                self.db.update_character(
                    nick,
                    channel,
                    xp=char.xp + xp_gained,
                    gold=char.gold + gold_gained,
                )

                # Check if room is fully cleared
                if self._check_room_cleared(channel, char.location):
                    self.db.mark_room_cleared(channel, char.location)

        # Enemy counterattack (if alive)
        counterattack_damage = 0
        player_died = False

        if not enemy_killed:
            total_defense = char.defense + equip_def
            enemy_roll = dice.roll(f"1d20+{spawn.attack}")
            player_ac = total_defense + 10

            if enemy_roll.total >= player_ac:
                enemy_dmg_roll = dice.roll(f"1d6+{spawn.attack // 2}")
                counterattack_damage = max(1, enemy_dmg_roll.total)
                new_hp = max(0, char.hp - counterattack_damage)
                self.db.update_character(nick, channel, hp=new_hp)

                if new_hp <= 0:
                    player_died = True
                    self._apply_death(nick, channel)

        # Check level up
        leveled_up = False
        new_level = char.level
        if enemy_killed and not player_died:
            leveled_up, new_level = self._check_level_up(nick, channel)

        return CombatResult(
            hit=hit,
            damage=damage,
            enemy_name=enemy_name,
            enemy_killed=enemy_killed,
            xp_gained=xp_gained,
            gold_gained=gold_gained,
            leveled_up=leveled_up,
            new_level=new_level,
            counterattack_damage=counterattack_damage,
            player_died=player_died,
        )

    def pickup_item(self, nick: str, channel: str, item_name: str) -> PickupResult:
        """Pick up an item from the current room (mv command)."""
        char = self.db.get_character(nick, channel)
        if char is None:
            return PickupResult(
                item_name=item_name,
                attack_bonus=0,
                defense_bonus=0,
                error="mv: you don't exist yet.",
            )

        room = self.world.get_room(char.location)
        if room is None:
            return PickupResult(
                item_name=item_name,
                attack_bonus=0,
                defense_bonus=0,
                error="mv: room not found",
            )

        for item in room.items:
            if item.name == item_name:
                self.db.add_item(
                    nick,
                    channel,
                    item.name,
                    attack_bonus=item.attack_bonus,
                    defense_bonus=item.defense_bonus,
                )
                return PickupResult(
                    item_name=item.name,
                    attack_bonus=item.attack_bonus,
                    defense_bonus=item.defense_bonus,
                )

        return PickupResult(
            item_name=item_name,
            attack_bonus=0,
            defense_bonus=0,
            error=f"mv: cannot stat '{item_name}': No such file or directory",
        )
