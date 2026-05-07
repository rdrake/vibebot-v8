"""Tests for the verse store."""

from __future__ import annotations

import time
from pathlib import Path

from llm.verse.store import db_path_for_channel


class TestDbPathForChannel:
    def test_lowercases_and_sanitizes(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#Foo")
        assert result.parent == verse_db_dir
        assert result.name.startswith("_foo_")
        assert result.suffix == ".db"

    def test_distinguishes_case_variants(self, verse_db_dir: Path) -> None:
        upper = db_path_for_channel(verse_db_dir, "#Foo")
        lower = db_path_for_channel(verse_db_dir, "#foo")
        assert upper != lower

    def test_strips_funky_characters(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#foo!bar/baz")
        assert "!" not in result.name
        assert "/" not in result.name

    def test_idempotent(self, verse_db_dir: Path) -> None:
        a = db_path_for_channel(verse_db_dir, "#afnet")
        b = db_path_for_channel(verse_db_dir, "#afnet")
        assert a == b


class TestVerseStoreInit:
    def test_creates_db_with_schema(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row is not None
            assert row[0] >= 1

    def test_idempotent_init(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#afnet")
        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            assert count == 1


class TestEntityCrud:
    def test_add_returns_id_and_persists(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "A mysterious stranger")
        assert isinstance(eid, int)
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.id == eid
        assert entity.kind == "npc"
        assert entity.name == "Alice"
        assert entity.summary == "A mysterious stranger"
        assert entity.status == "active"
        assert isinstance(entity.created_at, float)
        assert isinstance(entity.updated_at, float)
        assert abs(entity.created_at - entity.updated_at) < 1.0

    def test_get_unknown_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.get_entity(99999) is None

    def test_find_by_name_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test entity")
        for variant in ("alice", "ALICE", "Alice"):
            result = store.find_entity_by_name(variant)
            assert result is not None, f"Should find entity with name variant {variant!r}"
            assert result.id == eid

    def test_find_by_name_with_kind_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        place_id = store.add_entity("place", "Crossroads", "A dusty crossroads")
        _npc_id = store.add_entity("npc", "Crossroads", "A person named Crossroads")
        result = store.find_entity_by_name("Crossroads", kind="place")
        assert result is not None
        assert result.id == place_id
        assert result.kind == "place"

    def test_set_status_updates_updated_at(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Bob", "A test npc")
        original = store.get_entity(eid)
        assert original is not None
        time.sleep(0.01)
        store.set_status(eid, "retired")
        updated = store.get_entity(eid)
        assert updated is not None
        assert updated.status == "retired"
        assert updated.updated_at > original.created_at

    def test_set_status_unknown_id_silent(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.set_status(99999, "retired")  # must not raise

    def test_list_entities_by_kind_filters_by_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("place", "Town Square", "Central plaza")
        store.add_entity("place", "Harbor", "The docks")
        retired_id = store.add_entity("place", "Old Mill", "Abandoned mill")
        store.set_status(retired_id, "retired")

        active = store.list_entities_by_kind("place")
        assert len(active) == 2

        all_places = store.list_entities_by_kind("place", status=None)
        assert len(all_places) == 3

        retired = store.list_entities_by_kind("place", status="retired")
        assert len(retired) == 1
        assert retired[0].name == "Old Mill"

    def test_list_entities_by_kind_orders_by_updated_at_desc(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        id_a = store.add_entity("place", "Alpha", "First added")
        time.sleep(0.01)
        id_b = store.add_entity("place", "Beta", "Second added")

        results = store.list_entities_by_kind("place")
        assert len(results) == 2
        assert results[0].id == id_b
        assert results[1].id == id_a


class TestAttributesCrud:
    def test_set_and_get(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        assert store.get_attribute(eid, "color") == "red"

    def test_get_missing_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        assert store.get_attribute(eid, "nonexistent") is None

    def test_set_overwrites_existing(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "color", "blue")
        assert store.get_attribute(eid, "color") == "blue"
        attrs = store.list_attributes(eid)
        assert list(attrs.keys()).count("color") == 1

    def test_list_attributes_returns_dict(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "mood", "cheerful")
        store.set_attribute(eid, "weapon", "staff")
        attrs = store.list_attributes(eid)
        assert attrs == {"color": "red", "mood": "cheerful", "weapon": "staff"}

    def test_list_attributes_empty(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        assert store.list_attributes(eid) == {}

    def test_attributes_cascade_on_entity_delete(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "mood", "happy")
        # manually delete the entity to trigger ON DELETE CASCADE
        with store.write_transaction() as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
        assert store.list_attributes(eid) == {}


class TestRelationsCrud:
    def test_add_returns_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        rid = store.add_relation(a, b, "allied_with")
        assert isinstance(rid, int)
        assert rid > 0

    def test_list_no_filter_returns_all(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations()
        assert len(rels) == 3
        # ordered by id ASC
        assert rels[0].id < rels[1].id < rels[2].id

    def test_list_filtered_by_from_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(from_id=a)
        assert len(rels) == 2
        assert all(r.from_id == a for r in rels)

    def test_list_filtered_by_to_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(to_id=c)
        assert len(rels) == 2
        assert all(r.to_id == c for r in rels)

    def test_list_filtered_by_kind(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        allied = store.list_relations(kind="allied_with")
        assert len(allied) == 2
        hates = store.list_relations(kind="hates")
        assert len(hates) == 1

    def test_list_filtered_by_combo(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(from_id=a, kind="allied_with")
        assert len(rels) == 1
        assert rels[0].from_id == a
        assert rels[0].to_id == b
        assert rels[0].kind == "allied_with"

    def test_list_relations_empty_when_no_match(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        store.add_relation(a, b, "allied_with")
        assert store.list_relations(kind="nonexistent_kind") == []
