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
