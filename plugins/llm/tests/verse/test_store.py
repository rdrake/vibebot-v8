"""Tests for the verse store."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
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

    def test_list_proposals_filters_and_decodes(self, verse_db_dir: Path) -> None:
        from llm.verse.store import Proposal, VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        p1 = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "first"},
            confidence=0.9,
        )
        time.sleep(0.005)
        p2 = store.add_proposal(
            cycle_id="c-2",
            op="add_event",
            payload={"summary": "second"},
            confidence=0.5,
        )
        rows = store.list_proposals()
        assert [r.id for r in rows] == [p2, p1]
        assert isinstance(rows[0], Proposal)
        assert rows[0].payload == {"summary": "second"}
        assert [r.id for r in store.list_proposals(status="pending")] == [p2, p1]
        assert [r.id for r in store.list_proposals(cycle_id="c-1")] == [p1]

    def test_get_proposal_unknown_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.get_proposal("nope") is None

    def test_get_proposal_known_returns_proposal(self, verse_db_dir: Path) -> None:
        from llm.verse.store import Proposal, VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": [1]},
            confidence=0.4,
            provenance="line-2",
        )
        p = store.get_proposal(pid)
        assert isinstance(p, Proposal)
        assert p.id == pid
        assert p.payload == {"summary": "x", "entity_ids": [1]}
        assert p.confidence == 0.4
        assert p.status == "pending"

    def test_update_proposal_status_records_reviewer(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x"},
            confidence=0.9,
        )
        store.update_proposal_status(pid, status="approved", reviewer="alice")
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"
        assert p.reviewer == "alice"
        assert p.reviewed_at is not None and p.reviewed_at > 0

    def test_update_proposal_status_rejects_invalid(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x"},
            confidence=0.9,
        )
        with pytest.raises(ValueError):
            store.update_proposal_status(pid, status="weird", reviewer="alice")

    def test_update_proposal_status_unknown_id_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(LookupError):
            store.update_proposal_status("nope", status="approved", reviewer="alice")

    def test_list_proposals_status_approved_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "p"},
            confidence=0.9,
        )
        approved = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "a"},
            confidence=0.9,
            status="approved",
            reviewer="loom",
        )
        rows = store.list_proposals(status="approved")
        assert [r.id for r in rows] == [approved]


class TestApplyProposal:
    def test_apply_add_event_inserts_event(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        store.apply_proposal(
            op="add_event",
            payload={"summary": "Forest enters the clearing", "entity_ids": [eid]},
            source="loom",
        )
        events = store.recent_events()
        assert len(events) == 1
        assert events[0].summary == "Forest enters the clearing"
        assert events[0].source == "loom"
        assert events[0].entity_ids == (eid,)

    def test_apply_set_attribute_writes_kv(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        store.apply_proposal(
            op="set_attribute",
            payload={"entity_id": eid, "key": "mood", "value": "wary"},
            source="loom",
        )
        assert store.get_attribute(eid, "mood") == "wary"

    def test_apply_add_relation(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("avatar", "Forest")
        b = store.add_entity("npc", "Owl")
        store.apply_proposal(
            op="add_relation",
            payload={"from_id": a, "to_id": b, "kind": "allied_with", "note": ""},
            source="loom",
        )
        rels = store.list_relations(from_id=a)
        assert len(rels) == 1
        assert rels[0].kind == "allied_with"
        assert rels[0].to_id == b

    def test_apply_add_entity_creates_with_summary(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.apply_proposal(
            op="add_entity",
            payload={
                "kind": "place",
                "name": "Hollow Oak",
                "summary": "A leaning trunk on the path.",
            },
            source="loom",
        )
        assert isinstance(new_id, int)
        e = store.get_entity(new_id)
        assert e is not None
        assert e.kind == "place"
        assert e.name == "Hollow Oak"

    def test_apply_unknown_op_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(ValueError):
            store.apply_proposal(op="nuke", payload={}, source="loom")

    def test_apply_missing_payload_field_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(KeyError):
            store.apply_proposal(op="add_event", payload={}, source="loom")


class TestApplyAndRecordProposal:
    def test_one_transaction_event_plus_audit(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Forest")
        pid = store.apply_and_record_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": [eid]},
            confidence=0.95,
            provenance="line-1",
            reviewer="loom",
        )
        assert isinstance(pid, str) and len(pid) > 0
        events = store.recent_events()
        assert len(events) == 1
        assert events[0].summary == "x"
        rows = store.list_proposals()
        assert len(rows) == 1
        assert rows[0].status == "approved"
        assert rows[0].reviewer == "loom"

    def test_failure_inside_op_rolls_back_audit(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(Exception):  # noqa: B017,PT011
            store.apply_and_record_proposal(
                cycle_id="c-1",
                op="set_attribute",
                payload={"entity_id": 9999, "key": "k", "value": "v"},
                confidence=0.95,
                provenance="x",
                reviewer="loom",
            )
        assert store.list_proposals() == []


class TestApplyProposalAndMark:
    def test_pending_to_approved_atomically(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="line-1",
        )
        store.apply_proposal_and_mark(pid, reviewer="alice")
        events = store.recent_events()
        assert len(events) == 1
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"
        assert p.reviewer == "alice"

    def test_unknown_id_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with pytest.raises(LookupError):
            store.apply_proposal_and_mark("nope", reviewer="alice")

    def test_already_terminal_status_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
            status="approved",
            reviewer="bob",
        )
        with pytest.raises(ValueError):
            store.apply_proposal_and_mark(pid, reviewer="alice")


class TestListActiveVerses:
    def test_returns_paths_for_existing_dbs(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore, list_active_verses

        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#forest")
        result = list_active_verses(verse_db_dir)
        assert len(result) == 2
        for path in result:
            assert path.suffix == ".db"
            assert path.exists()

    def test_empty_dir_returns_empty_list(self, verse_db_dir: Path) -> None:
        from llm.verse.store import list_active_verses

        assert list_active_verses(verse_db_dir) == []

    def test_missing_dir_returns_empty_list(self, tmp_path: Path) -> None:
        from llm.verse.store import list_active_verses

        assert list_active_verses(tmp_path / "nope") == []


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


class TestEventsOlderThan:
    def test_returns_oldest_first(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        ids: list[int] = []
        for ts in (10.0, 20.0, 30.0):
            ids.append(
                insert_event_at(
                    store,
                    summary=f"e{ts}",
                    entity_ids=[],
                    source="loom",
                    ts=ts,
                )
            )
        rows = store.events_older_than(cutoff_ts=25.0)
        assert [r.id for r in rows] == [ids[0], ids[1]]
        assert [r.ts for r in rows] == [10.0, 20.0]

    def test_empty_when_no_events_below_cutoff(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=100.0)
        assert store.events_older_than(cutoff_ts=50.0) == []

    def test_includes_all_sources(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="a", entity_ids=[], source="avatar", ts=5.0)
        insert_event_at(store, summary="b", entity_ids=[], source="loom", ts=6.0)
        insert_event_at(store, summary="c", entity_ids=[], source="crosspoll", ts=7.0)
        rows = store.events_older_than(cutoff_ts=10.0)
        assert {r.source for r in rows} == {"avatar", "loom", "crosspoll"}


class TestReplaceEventsWithLoreDigest:
    def test_replaces_atomically_and_returns_new_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        old_ids = [
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=[],
                source="avatar",
                ts=float(i),
            )
            for i in range(5)
        ]
        new_id = store.replace_events_with_lore_digest(
            delete_ids=old_ids,
            summary="A digest of five small events.",
            entity_ids=(),
            ts=100.0,
        )
        assert new_id > 0
        # surviving rows: only the new digest event
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id, summary, source FROM events ORDER BY id ASC").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == new_id
        assert rows[0][1] == "A digest of five small events."
        assert rows[0][2] == "loom"

    def test_rolls_back_on_invalid_source(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        oid = insert_event_at(store, summary="e", entity_ids=[], source="avatar", ts=1.0)
        # Force a CHECK violation by exercising the inner helper with a
        # source not in the events.source CHECK list.
        with pytest.raises(Exception):  # noqa: B017,PT011
            store._replace_events_with_source(  # type: ignore[attr-defined]
                delete_ids=[oid],
                summary="x",
                entity_ids=(),
                ts=2.0,
                source="not_a_real_source",
            )
        # original event still present, no digest row created
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id FROM events").fetchall()
        assert [r[0] for r in rows] == [oid]

    def test_no_delete_ids_still_inserts_digest(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="empty digest",
            entity_ids=(),
            ts=42.0,
        )
        assert new_id > 0

    def test_entity_ids_are_json_encoded(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="d",
            entity_ids=(1, 2, 3),
            ts=10.0,
        )
        with store.read_connection() as conn:
            row = conn.execute("SELECT entity_ids FROM events WHERE id=?", (new_id,)).fetchone()
        assert json.loads(row[0]) == [1, 2, 3]


class TestAddProposalAcceptsId:
    def test_default_generates_uuid_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0,
            provenance="t",
        )
        # 32-char lowercase hex (uuid4 .hex)
        assert isinstance(pid, str) and len(pid) == 32
        assert all(c in "0123456789abcdef" for c in pid)

    def test_caller_supplied_id_is_used_verbatim(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid_in = "deadbeef" * 4  # 32 chars
        pid_out = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0,
            provenance="t",
            proposal_id=pid_in,
        )
        assert pid_out == pid_in
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT id FROM proposals WHERE id=?",
                (pid_in,),
            ).fetchone()
        assert row is not None and row[0] == pid_in

    def test_caller_supplied_duplicate_id_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = "abcd" * 8
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0,
            provenance="t",
            proposal_id=pid,
        )
        with pytest.raises(sqlite3.IntegrityError):
            store.add_proposal(
                cycle_id="c-1",
                op="add_event",
                payload={"summary": "y", "entity_ids": []},
                confidence=0.0,
                provenance="t",
                proposal_id=pid,
            )


class TestInlineHelpers:
    def test_add_entity_inline_runs_on_caller_conn(self, verse_db_dir: Path) -> None:
        """Caller opens its own write_transaction, calls _add_entity_inline,
        and a sibling INSERT in the same tx — all without lock reentry."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        with store.write_transaction() as conn:
            eid = store._add_entity_inline(  # type: ignore[attr-defined]
                conn, "npc", "ghost", "a wisp of vapour"
            )
            # Sibling INSERT in the same tx proves we hold the same conn.
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) VALUES (?, 'inline_marker', '1')",
                (eid,),
            )
        assert store.find_entity_by_name("ghost", kind="npc") is not None
        assert store.get_attribute(eid, "inline_marker") == "1"

    def test_set_attribute_inline_upserts_on_caller_conn(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        eid = store.add_entity("npc", "moss", "")
        with store.write_transaction() as conn:
            store._set_attribute_inline(  # type: ignore[attr-defined]
                conn, eid, "k", "v1"
            )
            store._set_attribute_inline(  # type: ignore[attr-defined]
                conn, eid, "k", "v2"
            )
        assert store.get_attribute(eid, "k") == "v2"


class TestApplyProposalAndMarkEventSource:
    def test_default_source_is_loom_and_proposal_marked_approved(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.0,
            provenance="t",
        )
        store.apply_proposal_and_mark(pid, reviewer="op")
        with store.read_connection() as conn:
            ev_row = conn.execute("SELECT source FROM events WHERE summary='x'").fetchone()
            pr_row = conn.execute(
                "SELECT status, reviewer, reviewed_at FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
        assert ev_row[0] == "loom"
        # apply_proposal_and_mark contract: status flipped, reviewer
        # recorded, reviewed_at populated.
        assert pr_row[0] == "approved"
        assert pr_row[1] == "op"
        assert pr_row[2] is not None and pr_row[2] > 0

    def test_event_source_crosspoll_and_proposal_marked_approved(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": "incoming", "entity_ids": []},
            confidence=0.0,
            provenance="crosspoll from #other",
        )
        store.apply_proposal_and_mark(pid, reviewer="op", event_source="crosspoll")
        with store.read_connection() as conn:
            ev_row = conn.execute("SELECT source FROM events WHERE summary='incoming'").fetchone()
            pr_row = conn.execute(
                "SELECT status, reviewer FROM proposals WHERE id=?",
                (pid,),
            ).fetchone()
        assert ev_row[0] == "crosspoll"
        assert pr_row[0] == "approved"
        assert pr_row[1] == "op"

    def test_already_approved_raises_and_does_not_double_apply(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "once", "entity_ids": []},
            confidence=0.0,
            provenance="t",
        )
        store.apply_proposal_and_mark(pid, reviewer="op")
        with pytest.raises(ValueError):
            store.apply_proposal_and_mark(pid, reviewer="op", event_source="crosspoll")
        # Only one event row; no double-apply. The 'crosspoll' source
        # was rejected because the proposal was already terminal.
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events WHERE summary='once'").fetchone()[0]
        assert count == 1
