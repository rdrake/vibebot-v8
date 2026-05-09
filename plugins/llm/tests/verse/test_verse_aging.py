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

    def test_digest_truncated_entities_correctly_age(self, store: VerseStore) -> None:
        """Setup: _MAX_DIGEST_ENTITY_IDS + 8 auto-created NPCs all with
        last_seen_ts=0. Insert a digest with all of them in entity_ids; the
        digest layer truncates to _MAX_DIGEST_ENTITY_IDS. Aging then runs
        and the 8 truncated-out NPCs are correctly retired."""
        from llm.verse.aging import age_auto_created_entities
        from llm.verse.compaction import _MAX_DIGEST_ENTITY_IDS

        n_total = _MAX_DIGEST_ENTITY_IDS + 8
        ids: list[int] = []
        for i in range(n_total):
            eid = store.add_entity("npc", f"ghost{i}", "")
            store.set_attribute(eid, "auto_created", "1")
            store.set_attribute(eid, "last_seen_ts", "0.0")
            ids.append(eid)

        # digest_ts must be > retire_after_days so truncated-out entities
        # (last_seen_ts=0) age past cutoff while bumped survivors stay
        # fresh.
        digest_ts = 30 * SECONDS_PER_DAY
        union_ids_truncated = ids[:_MAX_DIGEST_ENTITY_IDS]
        store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="many ghosts",
            entity_ids=union_ids_truncated,
            ts=digest_ts,
        )

        outcome = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: digest_ts + 5 * SECONDS_PER_DAY
        )
        assert outcome.retired == 8
        survivors_status = []
        truncated_status = []
        with store.read_connection() as conn:
            for i, eid in enumerate(ids):
                row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
                if i < _MAX_DIGEST_ENTITY_IDS:
                    survivors_status.append(row[0])
                else:
                    truncated_status.append(row[0])
        assert set(survivors_status) == {"active"}
        assert set(truncated_status) == {"retired"}

    def test_loom_applied_proposal_bumps_last_seen(self, store: VerseStore) -> None:
        """An apply_or_queue call landing as 'applied' bumps last_seen_ts
        on every entity_id referenced by the proposal payload."""
        from llm.verse.aging import age_auto_created_entities
        from llm.verse.loom import ParsedProposal, apply_or_queue

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "ghost lurked", "entity_ids": [eid]},
            confidence=0.95,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(
            store,
            prop,
            cycle_id="cyc-1",
            threshold=0.7,
        )
        assert outcome.outcome == "applied"
        last_seen = float(store.get_attribute(eid, "last_seen_ts") or "0")
        assert last_seen > 0.0
        keep = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: last_seen + SECONDS_PER_DAY
        )
        assert keep.retired == 0

    def test_loom_queued_proposal_does_not_bump(self, store: VerseStore) -> None:
        """Below-threshold proposals queue rather than apply; no bump."""
        from llm.verse.loom import ParsedProposal, apply_or_queue

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "maybe ghost", "entity_ids": [eid]},
            confidence=0.10,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(store, prop, cycle_id="cyc-q", threshold=0.7)
        assert outcome.outcome == "queued"
        assert store.get_attribute(eid, "last_seen_ts") == "0.0"

    def test_loom_rejected_invalid_refs_does_not_bump(self, store: VerseStore) -> None:
        """Proposals referencing nonexistent entity ids auto-reject; no bump."""
        from llm.verse.loom import ParsedProposal, apply_or_queue

        real_eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(real_eid, "auto_created", "1")
        store.set_attribute(real_eid, "last_seen_ts", "0.0")
        nonexistent_id = real_eid + 999_999
        prop = ParsedProposal(
            op="add_event",
            payload={
                "summary": "phantom event",
                "entity_ids": [real_eid, nonexistent_id],
            },
            confidence=0.95,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(store, prop, cycle_id="cyc-r", threshold=0.7)
        assert outcome.outcome != "applied"
        assert store.get_attribute(real_eid, "last_seen_ts") == "0.0"
