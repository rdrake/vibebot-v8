"""Tests for the verse store."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

import pytest
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


class TestEventsCrud:
    def test_add_returns_id_and_persists(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_event("A dragon appeared", [1, 2], "avatar")
        assert isinstance(eid, int)
        assert eid > 0
        events = store.recent_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.id == eid
        assert ev.summary == "A dragon appeared"
        assert ev.source == "avatar"
        assert ev.entity_ids == (1, 2)
        assert isinstance(ev.ts, float)

    def test_entity_ids_round_trip(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Test event", [3, 5, 11], "avatar")
        events = store.recent_events()
        assert events[0].entity_ids == (3, 5, 11)

    def test_entity_ids_empty_default(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Empty entity event", [], "loom")
        events = store.recent_events()
        assert events[0].entity_ids == ()

    def test_recent_events_newest_first(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        id1 = store.add_event("First", [], "avatar")
        time.sleep(0.01)
        id2 = store.add_event("Second", [], "avatar")
        time.sleep(0.01)
        id3 = store.add_event("Third", [], "avatar")
        events = store.recent_events(limit=10)
        assert [ev.id for ev in events] == [id3, id2, id1]

    def test_recent_events_limit(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(5):
            store.add_event(f"Event {i}", [], "loom")
        events = store.recent_events(limit=2)
        assert len(events) == 2

    def test_recent_events_exclude_sources_single(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Avatar event", [], "avatar")
        store.add_event("Loom event", [], "loom")
        store.add_event("Crosspoll event", [], "crosspoll")
        events = store.recent_events(exclude_sources=("crosspoll",))
        assert len(events) == 2
        sources = {ev.source for ev in events}
        assert "crosspoll" not in sources

    def test_recent_events_exclude_sources_multiple(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Avatar event", [], "avatar")
        store.add_event("Loom event", [], "loom")
        store.add_event("Crosspoll event", [], "crosspoll")
        events = store.recent_events(exclude_sources=("crosspoll", "loom"))
        assert len(events) == 1
        assert events[0].source == "avatar"

    def test_recent_events_exclude_sources_empty_default_includes_all(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Avatar event", [], "avatar")
        store.add_event("Loom event", [], "loom")
        store.add_event("Crosspoll event", [], "crosspoll")
        events = store.recent_events()
        assert len(events) == 3

    def test_recent_events_empty_db_returns_empty_list(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.recent_events() == []


class TestAvatarLinkCrud:
    def test_link_inserts_row(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        store.link_avatar(eid, nick="Alice", account=None)
        assert store.find_avatar_by_nick("alice") == eid

    def test_link_with_account(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        store.link_avatar(eid, nick="Alice", account="alice@network")
        assert store.find_avatar_by_account("alice@network") == eid
        assert store.find_avatar_by_account("Alice@Network") is None  # case-sensitive

    def test_link_upserts_existing_row(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        store.link_avatar(eid, nick="Alice", account=None)
        store.link_avatar(eid, nick="Bob", account=None)
        assert store.find_avatar_by_nick("alice") is None
        assert store.find_avatar_by_nick("bob") == eid

    def test_find_unknown_nick_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.find_avatar_by_nick("ghost") is None

    def test_find_unknown_account_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.find_avatar_by_account("nope") is None

    def test_find_by_nick_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        store.link_avatar(eid, nick="Alice")
        for variant in ("ALICE", "alice", "aLiCe"):
            assert store.find_avatar_by_nick(variant) == eid

    def test_unlink_removes_link_and_retires_entity(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        store.link_avatar(eid, nick="Alice")
        store.unlink_avatar(eid)
        assert store.find_avatar_by_nick("Alice") is None
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.status == "retired"

    def test_unlink_atomicity(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Bob", "")
        store.link_avatar(eid, nick="Bob", account="bob@net")
        store.unlink_avatar(eid)
        # Both effects visible after single call
        assert store.find_avatar_by_nick("Bob") is None
        assert store.find_avatar_by_account("bob@net") is None
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.status == "retired"

    def test_unlink_unknown_id_silent(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.unlink_avatar(99999)  # must not raise

    def test_two_avatars_distinct_links(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid_a = store.add_entity("avatar", "Alice", "")
        eid_b = store.add_entity("avatar", "Bob", "")
        store.link_avatar(eid_a, nick="Alice")
        store.link_avatar(eid_b, nick="Bob")
        assert store.find_avatar_by_nick("alice") == eid_a
        assert store.find_avatar_by_nick("bob") == eid_b
        store.unlink_avatar(eid_a)
        assert store.find_avatar_by_nick("alice") is None
        assert store.find_avatar_by_nick("bob") == eid_b
        entity_b = store.get_entity(eid_b)
        assert entity_b is not None
        assert entity_b.status == "active"


class TestOptInAvatar:
    def test_new_user_creates_avatar_and_default_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import AvatarOptInResult, VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        result = store.opt_in_avatar("alice", None, "curious traveller")

        assert isinstance(result, AvatarOptInResult)
        assert result.was_already_opted_in is False
        assert result.place_name == "The Clearing"
        assert result.scene_text.startswith("You step into The Clearing.")
        assert "A quiet woodland clearing where new stories begin." in result.scene_text

        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.kind == "avatar"
        assert entity.status == "active"
        assert entity.name == "alice"
        assert entity.summary == "curious traveller"

        places = store.list_entities_by_kind("place")
        assert len(places) == 1

        assert store.find_avatar_by_nick("alice") == result.entity_id
        assert store.get_attribute(result.entity_id, "location") == "The Clearing"

    def test_second_user_lands_at_existing_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.opt_in_avatar("alice", None, "first in")
        bob_result = store.opt_in_avatar("bob", None, "second in")

        places = store.list_entities_by_kind("place")
        assert len(places) == 1

        assert bob_result.place_name == "The Clearing"
        assert store.get_attribute(bob_result.entity_id, "location") == "The Clearing"

    def test_already_opted_in_active_returns_existing(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        second = store.opt_in_avatar("alice", None, "different text")

        assert second.was_already_opted_in is True
        assert second.entity_id == first.entity_id
        assert second.place_name == "The Clearing"
        assert "The Clearing" in second.scene_text

        # Only one avatar entity
        avatars = store.list_entities_by_kind("avatar")
        assert len(avatars) == 1

    def test_retired_avatar_reactivates(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        eid = first.entity_id

        # Soft pause: retire entity but leave link
        store.set_status(eid, "retired")

        result = store.opt_in_avatar("alice", None, "coming back")

        assert result.was_already_opted_in is False
        assert result.entity_id == eid

        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.status == "active"

    def test_retired_avatar_via_unlink_then_reopt_creates_new(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        old_eid = first.entity_id

        # Hard goodbye: removes link AND retires
        store.unlink_avatar(old_eid)

        result = store.opt_in_avatar("alice", None, "starting fresh")

        assert result.was_already_opted_in is False
        assert result.entity_id != old_eid

    def test_account_lookup_prefers_account_over_nick(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", "alice@net", "curious traveller")
        eid = first.entity_id

        # Nick changed on IRC side, account stable
        result = store.opt_in_avatar("aliceX", "alice@net", "new session")

        assert result.was_already_opted_in is True
        assert result.entity_id == eid

        # Link nick should be updated
        assert store.find_avatar_by_nick("aliceX") == eid

    def test_concurrent_opt_in_distinct_nicks_one_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent-optin")

        def do_opt_in(nick: str):  # type: ignore[return]
            return store.opt_in_avatar(nick, None, f"{nick} instruct")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(do_opt_in, ["alice", "bob"]))

        # Both succeed
        assert all(r is not None for r in results)
        # Only one place
        assert len(store.list_entities_by_kind("place")) == 1
        # Both land at same place
        assert results[0].place_name == results[1].place_name
        # Distinct entity ids
        assert results[0].entity_id != results[1].entity_id

    def test_scene_text_does_not_mention_verse_act(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        result = store.opt_in_avatar("alice", None, "curious traveller")
        assert "verse_act" not in result.scene_text

    def test_place_selection_uses_most_recent_event(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        # Pre-create two places
        store.add_entity("place", "Place A", "Summary of A.")
        place_b_id = store.add_entity("place", "Place B", "Summary of B.")

        # Add event referencing place B — makes it the most recently referenced
        store.add_event("Something happened at B", [place_b_id], "loom")

        result = store.opt_in_avatar("alice", None, "curious traveller")
        assert result.place_name == "Place B"
        assert store.get_attribute(result.entity_id, "location") == "Place B"


class TestProposalsCRUD:
    def test_add_proposal_pending_default(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "the bell rang", "entity_ids": []},
            confidence=0.9,
            provenance="line-3",
        )
        assert isinstance(pid, str) and len(pid) > 0
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT id, op, status, confidence, reviewer, reviewed_at "
                "FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
            assert row[0] == pid
            assert row[1] == "add_event"
            assert row[2] == "pending"
            assert row[3] == 0.9
            assert row[4] is None
            assert row[5] is None

    def test_add_proposal_with_preset_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "auto", "entity_ids": []},
            confidence=0.95,
            provenance="line-1",
            status="approved",
            reviewer="loom",
        )
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT status, reviewer, reviewed_at FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
            assert row[0] == "approved"
            assert row[1] == "loom"
            assert row[2] is not None and row[2] > 0

    def test_add_proposal_rejects_invalid_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(ValueError):
            store.add_proposal(
                cycle_id="c-1",
                op="add_event",
                payload={"summary": "x", "entity_ids": []},
                confidence=0.9,
                provenance="x",
                status="weird",
            )


class TestWriteLockConcurrency:
    def test_concurrent_add_entity_yields_unique_ids(self, verse_db_dir: Path) -> None:
        """50 concurrent add_entity calls across 8 threads — all rows persist
        with unique IDs, none are lost or duplicated."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent")

        def worker(i: int) -> int:
            return store.add_entity(kind="npc", name=f"worker_{i}", summary="")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(worker, range(50)))

        assert len(set(ids)) == 50, "duplicate ids returned"
        assert all(i > 0 for i in ids), "non-positive id returned"

        names = [e.name for e in store.list_entities_by_kind("npc")]
        assert sorted(names) == sorted(f"worker_{i}" for i in range(50))

    def test_concurrent_set_attribute_no_lost_writes(self, verse_db_dir: Path) -> None:
        """Concurrent upserts to the same entity-key pair — last writer wins;
        no transaction is dropped (count of distinct values seen during the
        test <= thread count, no SQL errors)."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent2")
        eid = store.add_entity(kind="place", name="arena", summary="")
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                store.set_attribute(eid, "round", str(i))
            except BaseException as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(50)))

        assert errors == []
        # The final value is one of {0,...,49}; we don't care which.
        final = store.get_attribute(eid, "round")
        assert final is not None
        assert int(final) in range(50)
