"""Tests for verse/aging.age_auto_created_entities."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#aging")


SECONDS_PER_DAY = 86400.0


class TestAgeAutoCreatedEntities:
    def test_retire_after_days_zero_disables(self, store: VerseStore) -> None:
        from llm.verse.aging import AgingOutcome, age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=0, now=lambda: 1e9)
        assert outcome == AgingOutcome(scanned=0, retired=0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_retires_past_cutoff(self, store: VerseStore) -> None:
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "100.0")  # very stale
        now = 100.0 + 30 * SECONDS_PER_DAY  # 30 days later

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: now)
        assert outcome.scanned == 1
        assert outcome.retired == 1
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "retired"

    def test_keeps_recent(self, store: VerseStore) -> None:
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "moss", "")
        store.set_attribute(eid, "auto_created", "1")
        last_seen = 1000.0
        store.set_attribute(eid, "last_seen_ts", str(last_seen))
        now = last_seen + 5 * SECONDS_PER_DAY  # 5 days < 14-day cutoff

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: now)
        assert outcome == (1, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_skips_manually_created(self, store: VerseStore) -> None:
        """An NPC without auto_created='1' must never be touched, even if
        last_seen_ts is past cutoff."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "manual", "")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_skips_avatar_kind_defensively(self, store: VerseStore) -> None:
        """Even if a bug somewhere stamps auto_created='1' on an avatar,
        aging must not retire it. The kind!='avatar' guard is defensive."""
        from llm.verse.aging import age_auto_created_entities

        avatar_id = store.add_entity("avatar", "alice", "")
        store.set_attribute(avatar_id, "auto_created", "1")
        store.set_attribute(avatar_id, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (avatar_id,)).fetchone()
        assert row[0] == "active"

    def test_digest_insert_bumps_last_seen(self, store: VerseStore) -> None:
        """When _replace_events_with_source inserts a digest, every entity
        in entity_ids has last_seen_ts bumped to ts. The bump is on the
        same conn as the INSERT — atomic with the digest write."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")  # stale

        digest_ts = 1000.0
        store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="ghost remained",
            entity_ids=(eid,),
            ts=digest_ts,
        )
        # Heartbeat fired
        assert store.get_attribute(eid, "last_seen_ts") == str(digest_ts)
        # Aging now sees a fresh entity → keeps it
        outcome = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: digest_ts + SECONDS_PER_DAY
        )
        assert outcome == (1, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"
