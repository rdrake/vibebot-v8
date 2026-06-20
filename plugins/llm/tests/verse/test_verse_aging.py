"""Tests for verse/aging.age_auto_created_entities."""

from __future__ import annotations

import logging
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

    def test_missing_last_seen_ts_is_left_untouched(self, store: VerseStore) -> None:
        """An auto_created entity with NO last_seen_ts attribute at all is
        scanned but never retired (no heartbeat -> no decision). Recovered
        state may lack the key entirely; aging must not crash or retire on
        absence."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        # Deliberately do NOT set last_seen_ts.
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome.scanned == 1
        assert outcome.retired == 0
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_malformed_last_seen_ts_warns_and_skips(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-numeric last_seen_ts (corrupt/recovered state) is logged at
        WARNING and the entity is skipped: scanned but not retired, and never
        crashes on float() of garbage."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="llm.verse.aging"):
            outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome.scanned == 1
        assert outcome.retired == 0
        assert any("malformed last_seen_ts" in r.message for r in caplog.records)
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

    def test_skips_pinned_entities(self, store: VerseStore) -> None:
        """A pinned auto_created NPC is explicit operator canon and must
        never be auto-retired, even when silent well past the cutoff.
        Pinned entities are protected, not aging candidates: scanned stays
        0 and the entity remains active."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "assgas archie", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")  # ancient
        store.set_attribute(eid, "pinned", "1")

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
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

    def test_loom_applied_set_attribute_bumps_last_seen(self, store: VerseStore) -> None:
        """An applied set_attribute proposal bumps last_seen_ts on the
        target entity. Per design v2.3 §4.2: heartbeat scope dispatches
        on op."""
        from llm.verse.loom import ParsedProposal, apply_or_queue

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        prop = ParsedProposal(
            op="set_attribute",
            payload={"entity_id": eid, "key": "mood", "value": "wary"},
            confidence=0.95,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(store, prop, cycle_id="cyc-sa", threshold=0.7)
        assert outcome.outcome == "applied"
        last_seen = float(store.get_attribute(eid, "last_seen_ts") or "0")
        assert last_seen > 0.0

    def test_loom_applied_add_relation_bumps_both_endpoints(self, store: VerseStore) -> None:
        """An applied add_relation proposal bumps last_seen_ts on both
        from_id and to_id endpoints."""
        from llm.verse.loom import ParsedProposal, apply_or_queue

        a = store.add_entity("npc", "alpha", "")
        b = store.add_entity("npc", "beta", "")
        for eid in (a, b):
            store.set_attribute(eid, "auto_created", "1")
            store.set_attribute(eid, "last_seen_ts", "0.0")

        prop = ParsedProposal(
            op="add_relation",
            payload={"from_id": a, "to_id": b, "kind": "ally"},
            confidence=0.95,
            provenance="test",
            rationale="r",
        )
        outcome = apply_or_queue(store, prop, cycle_id="cyc-ar", threshold=0.7)
        assert outcome.outcome == "applied"
        assert float(store.get_attribute(a, "last_seen_ts") or "0") > 0.0
        assert float(store.get_attribute(b, "last_seen_ts") or "0") > 0.0

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


class TestAgingExemptsAuthorLocked:
    def test_author_locked_npc_not_retired(self, store):
        import time as _t

        h = store.add_entity("npc", "Harry")
        store.set_attribute(h, "auto_created", "1")
        store.set_attribute(h, "last_seen_ts", "0.0")  # ancient
        store.set_author_locked(h, True)
        from llm.verse.aging import age_auto_created_entities

        age_auto_created_entities(store, retire_after_days=1, now=lambda: _t.time())
        assert store.get_entity(h).status == "active"


class TestAgingExemptsOccupiedPlace:
    def test_occupied_place_survives_aging(self, store: VerseStore) -> None:
        """A stale auto_created place that is still an active avatar's
        location must never be auto-retired: retiring it would strand the
        avatar at a ghost location. The place is exempt (not scanned)."""
        from llm.verse.aging import age_auto_created_entities

        place_id = store.add_entity("place", "The Clearing", "a quiet glade")
        store.set_attribute(place_id, "auto_created", "1")
        store.set_attribute(place_id, "last_seen_ts", "0")  # ancient

        avatar_id = store.add_entity("avatar", "wanderer", "")
        store.set_attribute(avatar_id, "location", "The Clearing")

        outcome = age_auto_created_entities(store, retire_after_days=30, now=lambda: 1e9)
        assert store.get_entity(place_id).status == "active"
        assert outcome.retired == 0
        # Exemption must sit BEFORE `scanned += 1` (same as pinned/author_locked),
        # so an occupied place is not even counted as scanned.
        assert outcome.scanned == 0

    def test_unoccupied_stale_place_is_retired(self, store: VerseStore) -> None:
        """Same ancient auto_created place, but with NO avatar standing in
        it, is retired normally. Guards against the occupancy exemption
        being too broad."""
        from llm.verse.aging import age_auto_created_entities

        place_id = store.add_entity("place", "The Clearing", "a quiet glade")
        store.set_attribute(place_id, "auto_created", "1")
        store.set_attribute(place_id, "last_seen_ts", "0")  # ancient

        outcome = age_auto_created_entities(store, retire_after_days=30, now=lambda: 1e9)
        assert store.get_entity(place_id).status == "retired"
        assert outcome.retired == 1
        assert outcome.scanned == 1  # not exempt → counted and retired
