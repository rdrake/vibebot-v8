"""Tests for VerseStore.record_user_event (verse_record's atomic DB path)."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#record")


def _opt_in(store: VerseStore, nick: str = "alice") -> int:
    """Convenience: opt a nick into the verse and return the avatar entity id."""
    result = store.opt_in_avatar(nick, account=None, instruct_text=f"{nick} instruct")
    return result.entity_id


class TestRecordUserEvent:
    def test_all_new_names_create_npcs_and_link_event(self, store: VerseStore) -> None:
        """GIVEN unknown actor names WHEN record_user_event called THEN
        each name becomes an active npc with auto_created='1', and the
        event row's entity_ids list starts with the caller's avatar."""
        alice_id = _opt_in(store)
        event_id = store.record_user_event(
            actor_id=alice_id,
            summary="stinky dan threw a guff grenade at Andrew",
            actor_names=["stinky dan", "Andrew"],
            now=lambda: 100.0,
        )
        # NPC rows exist
        dan = store.find_active_entity_by_name("stinky dan")
        andrew = store.find_active_entity_by_name("Andrew")
        assert dan is not None and dan.kind == "npc"
        assert andrew is not None and andrew.kind == "npc"
        # auto_created marker on each
        assert store.get_attribute(dan.id, "auto_created") == "1"
        assert store.get_attribute(andrew.id, "auto_created") == "1"
        # last_seen_ts = now
        assert store.get_attribute(dan.id, "last_seen_ts") == "100.0"
        assert store.get_attribute(andrew.id, "last_seen_ts") == "100.0"
        # Event row links caller first, then actors in order
        events = store.recent_events(limit=10)
        ev = next(e for e in events if e.id == event_id)
        assert list(ev.entity_ids) == [alice_id, dan.id, andrew.id]
        assert ev.source == "avatar"
        assert ev.summary == "stinky dan threw a guff grenade at Andrew"

    def test_avatar_actor_not_tagged_or_bumped(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, "alice")
        andrew_id = _opt_in(store, "andrew")
        store.record_user_event(
            actor_id=alice_id,
            summary="alice greeted Andrew",
            actor_names=["andrew"],
            now=lambda: 100.0,
        )
        assert store.get_attribute(andrew_id, "auto_created") is None
        assert store.get_attribute(andrew_id, "last_seen_ts") is None
        events = store.recent_events(limit=10)
        assert any(list(e.entity_ids) == [alice_id, andrew_id] for e in events)

    def test_existing_npc_reused_and_heartbeat_updated(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        dan_id = store.add_entity("npc", "dan", "")
        store.set_attribute(dan_id, "auto_created", "1")
        store.set_attribute(dan_id, "last_seen_ts", "50.0")

        store.record_user_event(
            actor_id=alice_id,
            summary="dan returned",
            actor_names=["dan"],
            now=lambda: 200.0,
        )
        rows = [e for e in store.list_entities_by_kind("npc") if e.name == "dan"]
        assert len(rows) == 1
        assert store.get_attribute(dan_id, "last_seen_ts") == "200.0"
