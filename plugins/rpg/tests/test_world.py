"""Tests for world map and navigation."""

from __future__ import annotations

from rpg.world import WorldMap


class TestWorldMap:
    """Room graph and path resolution."""

    def test_starter_map_has_town(self):
        """GIVEN a starter world WHEN checking rooms THEN /town exists."""
        world = WorldMap.starter()
        assert world.get_room("/town") is not None

    def test_starter_map_room_count(self):
        """GIVEN a starter world WHEN counting rooms THEN there are 14."""
        world = WorldMap.starter()
        assert len(world.rooms) == 14

    def test_resolve_relative_path(self):
        """GIVEN player at /town WHEN cd tavern THEN resolves to /town/tavern."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "tavern")
        assert result == "/town/tavern"

    def test_resolve_dotdot(self):
        """GIVEN player at /town/tavern WHEN cd .. THEN resolves to /town."""
        world = WorldMap.starter()
        result = world.resolve_path("/town/tavern", "..")
        assert result == "/town"

    def test_resolve_absolute_path(self):
        """GIVEN player anywhere WHEN cd /forest/clearing THEN resolves absolutely."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "/forest/clearing")
        assert result == "/forest/clearing"

    def test_resolve_invalid_path(self):
        """GIVEN player at /town WHEN cd nonexistent THEN returns None."""
        world = WorldMap.starter()
        result = world.resolve_path("/town", "nonexistent")
        assert result is None

    def test_resolve_root(self):
        """GIVEN player anywhere WHEN cd / THEN resolves to /."""
        world = WorldMap.starter()
        result = world.resolve_path("/town/tavern", "/")
        assert result == "/"

    def test_resolve_dotdot_at_root(self):
        """GIVEN player at / WHEN cd .. THEN stays at /."""
        world = WorldMap.starter()
        result = world.resolve_path("/", "..")
        assert result == "/"

    def test_hidden_rooms_exist(self):
        """GIVEN a starter world WHEN checking hidden rooms THEN dotfile rooms exist."""
        world = WorldMap.starter()
        armory = world.get_room("/town/.armory")
        assert armory is not None
        assert armory.hidden is True

    def test_room_exits(self):
        """GIVEN /town WHEN listing exits THEN tavern and blacksmith are exits."""
        world = WorldMap.starter()
        room = world.get_room("/town")
        assert room is not None
        exits = world.get_exits("/town")
        assert "tavern" in exits
        assert "blacksmith" in exits

    def test_hidden_exits_excluded_by_default(self):
        """GIVEN /town WHEN listing visible exits THEN .armory not shown."""
        world = WorldMap.starter()
        exits = world.get_exits("/town", include_hidden=False)
        assert ".armory" not in exits

    def test_hidden_exits_included_with_flag(self):
        """GIVEN /town WHEN listing all exits THEN .armory shown."""
        world = WorldMap.starter()
        exits = world.get_exits("/town", include_hidden=True)
        assert ".armory" in exits

    def test_room_has_parent_exit(self):
        """GIVEN /town/tavern WHEN listing exits THEN .. is always an exit."""
        world = WorldMap.starter()
        exits = world.get_exits("/town/tavern")
        assert ".." in exits

    def test_root_has_no_parent_exit(self):
        """GIVEN / WHEN listing exits THEN .. is not listed."""
        world = WorldMap.starter()
        exits = world.get_exits("/")
        assert ".." not in exits
