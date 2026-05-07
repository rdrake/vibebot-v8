"""Tests for verb whitelist (VERB_TABLE) in avatar module."""

from __future__ import annotations

import pytest
from llm.verse.avatar import VERB_TABLE, ActResult, VerbEffect, verse_act
from llm.verse.store import VerseStore


class TestVerbWhitelist:
    def test_event_only_verbs(self) -> None:
        event_only = [
            "whisper",
            "speak",
            "listen",
            "examine",
            "wait",
            "signal",
            "gesture",
            "search",
        ]
        for verb in event_only:
            assert VERB_TABLE[verb] is VerbEffect.EVENT_ONLY, f"{verb!r} should be EVENT_ONLY"

    def test_move_verbs(self) -> None:
        for verb in ("move", "flee", "follow"):
            assert VERB_TABLE[verb] is VerbEffect.MOVE, f"{verb!r} should be MOVE"

    def test_item_verbs(self) -> None:
        for verb in ("take", "drop", "give"):
            assert VERB_TABLE[verb] is VerbEffect.ITEM, f"{verb!r} should be ITEM"

    @pytest.mark.parametrize("verb", ["teleport", "attack", "cast", "eat"])
    def test_off_list_verbs_absent(self, verb: str) -> None:
        assert verb not in VERB_TABLE

    def test_table_size(self) -> None:
        assert len(VERB_TABLE) == 14


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: pytest.TempPathFactory) -> VerseStore:
    return VerseStore(tmp_path, "#test")


def _opt_in(store: VerseStore, nick: str = "alice") -> int:
    """Opt in a user and return their avatar entity_id."""
    result = store.opt_in_avatar(nick=nick, account=None, instruct_text="A wanderer.")
    return result.entity_id


# ---------------------------------------------------------------------------
# TestVerseAct
# ---------------------------------------------------------------------------


class TestVerseAct:
    def test_event_only_verb_records_event(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "speak")

        assert isinstance(result, ActResult)
        events = store.recent_events(limit=10)
        assert len(events) == 1
        ev = events[0]
        assert ev.source == "avatar"
        assert "speak" in ev.summary
        assert alice_id in ev.entity_ids
        assert result.scene_shift_text.startswith("You speak")

    def test_event_only_verb_with_target(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "whisper", target="bob")

        events = store.recent_events(limit=10)
        assert len(events) == 1
        assert "bob" in events[0].summary
        assert result.scene_shift_text == "You whisper bob."

    def test_move_verb_success_updates_location(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        riverside_id = store.add_entity("place", "Riverside", "A bend in the river.")
        result = verse_act(store, alice_id, "move", target="Riverside")

        # Location attribute updated
        assert store.get_attribute(alice_id, "location") == "Riverside"
        # Event written with both ids
        events = store.recent_events(limit=10)
        move_events = [e for e in events if "Riverside" in e.summary]
        assert len(move_events) == 1
        assert alice_id in move_events[0].entity_ids
        assert riverside_id in move_events[0].entity_ids
        # scene_shift
        assert result.scene_shift_text == "You move to Riverside."

    def test_move_verb_target_not_found_writes_event_no_side_effect(
        self, store: VerseStore
    ) -> None:
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")

        result = verse_act(store, alice_id, "move", target="Nowhere")

        # Event IS written
        events = store.recent_events(limit=10)
        assert any("Nowhere" in e.summary for e in events)
        # Location UNCHANGED
        assert store.get_attribute(alice_id, "location") == original_location
        # Exact scene_shift text
        assert result.scene_shift_text == "You can't find that place."

    def test_move_verb_via_avatar_target_resolves_to_their_location(
        self, store: VerseStore
    ) -> None:
        alice_id = _opt_in(store, nick="alice")
        bob_id = _opt_in(store, nick="bob")

        # Move alice to Cliff
        cliff_id = store.add_entity("place", "Cliff", "A high ledge.")
        verse_act(store, alice_id, "move", target="Cliff")
        assert store.get_attribute(alice_id, "location") == "Cliff"

        # Bob follows alice (avatar target resolves to alice's location)
        result = verse_act(store, bob_id, "move", target="alice")
        assert store.get_attribute(bob_id, "location") == "Cliff"
        events = store.recent_events(limit=20)
        bob_move = [e for e in events if bob_id in e.entity_ids and cliff_id in e.entity_ids]
        assert len(bob_move) == 1
        assert result.scene_shift_text == "You move to Cliff."

    def test_item_verb_target_not_found_writes_event_no_relation(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "give", target="Phantom Sword")

        # Event written
        events = store.recent_events(limit=10)
        assert any("Phantom Sword" in e.summary for e in events)
        # No relation added
        assert store.list_relations(from_id=alice_id) == []
        # scene_shift acknowledges absence
        assert "Phantom Sword" in result.scene_shift_text or "find" in result.scene_shift_text

    def test_item_verb_success_writes_event(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        sword_id = store.add_entity("item", "sword", "A sharp blade.")

        result = verse_act(store, alice_id, "take", target="sword")

        events = store.recent_events(limit=10)
        take_events = [e for e in events if "sword" in e.summary]
        assert len(take_events) == 1
        assert alice_id in take_events[0].entity_ids
        assert sword_id in take_events[0].entity_ids
        # scene_shift mentions verb and item
        assert "take" in result.scene_shift_text or "sword" in result.scene_shift_text
        # No relation added (v1 design)
        assert store.list_relations(from_id=alice_id) == []

    def test_off_list_verb_writes_event_no_side_effect(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")

        result = verse_act(store, alice_id, "teleport", target="moon")

        events = store.recent_events(limit=10)
        assert len(events) == 1
        assert "teleport" in result.scene_shift_text
        # Location unchanged
        assert store.get_attribute(alice_id, "location") == original_location

    def test_retired_avatar_raises_before_write(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        store.set_status(alice_id, "retired")

        before_count = len(store.recent_events(limit=100))

        with pytest.raises(ValueError, match="avatar retired"):
            verse_act(store, alice_id, "speak")

        # No event written
        assert len(store.recent_events(limit=100)) == before_count

    def test_unknown_avatar_id_raises(self, store: VerseStore) -> None:
        with pytest.raises(ValueError, match="avatar retired"):
            verse_act(store, 99999, "speak")
