"""Tests for the daily verse retention compaction helper."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verse"
    d.mkdir()
    return d


class _FakeClient:
    def __init__(self, content: str = "A digest of past events.") -> None:
        self.content = content
        self.calls: list[dict] = []

    def call(self, *, op: str, model: str, messages: list[dict[str, str]]):
        from llm.verse.loom import LoomCallUsage

        self.calls.append({"op": op, "model": model, "messages": messages})
        return self.content, LoomCallUsage(prompt_tokens=10, completion_tokens=20, cost=0.0)


class TestCompactVerse:
    def test_skips_when_retention_zero(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store,
            retention_days=0,
            min_keep_events=20,
            model="gemini/gemini-flash-lite-latest",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out == "skipped_disabled"

    def test_skips_when_below_min_keep(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 100_000_000.0,
        )
        assert out == "skipped_below_floor"

    def test_skips_when_no_old_events(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=[],
                source="loom",
                ts=1_000_000.0 - i,
            )
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out == "skipped_no_events"

    def test_compacts_old_events_into_single_digest(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        for i in range(25):
            insert_event_at(
                store,
                summary=f"new{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 1.0,
            )
        usage_calls: list[dict] = []
        client = _FakeClient(content="Past events: a wood, a brook, a whisper.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="gemini/gemini-flash-lite-latest",
            client=client,
            log_usage=lambda **kw: usage_calls.append(kw),
            now=lambda: now,
        )
        assert out == "compacted"
        with store.read_connection() as conn:
            rows = conn.execute("SELECT summary, source FROM events ORDER BY ts ASC").fetchall()
        assert len(rows) == 26
        assert rows[0][1] == "loom"
        assert "Past events" in rows[0][0]
        assert client.calls and client.calls[0]["op"] == "compact"
        assert len(usage_calls) == 1
        assert usage_calls[0]["op"] == "compact"

    def test_long_backlog_only_deletes_what_was_summarised(self, verse_db_dir: Path) -> None:
        """If there are 500 old events and the per-pass cap is 200,
        exactly 200 originals are deleted; 300 survive for the next pass.
        Regression test for the v1 plan bug where ALL olds were deleted
        but only the last 200 were shown to the model."""
        from llm.verse.compaction import _MAX_EVENTS_PER_PASS, compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(500):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day - i,
            )
        client = _FakeClient(content="A long-ago digest.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert out == "compacted"
        assert _MAX_EVENTS_PER_PASS == 200
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 500 - _MAX_EVENTS_PER_PASS + 1

    def test_bullet_trim_only_deletes_events_in_prompt(self, verse_db_dir: Path) -> None:
        """When the bullet block exceeds _MAX_BULLET_BLOCK_CHARS, oldest
        bullets are dropped from the prompt. Regression: previously the
        delete_ids list still referenced the FULL batch, so events the
        LLM never saw got deleted. Now: delete_ids matches exactly the
        events whose bullets remain in the prompt; events with trimmed
        bullets survive for the next pass.
        """
        from llm.verse.compaction import (
            _MAX_BULLET_BLOCK_CHARS,
            _MAX_SUMMARY_CHARS_PER_EVENT,
            compact_verse,
        )
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")

        # Long summaries cap at _MAX_SUMMARY_CHARS_PER_EVENT (240).
        # Bullet line is "- " + summary -> ~242 chars per line.
        # 70 events * (242 + 1 newline) ~= 17000 chars > 16000 cap.
        n_events = 70
        long_summary = "x" * (_MAX_SUMMARY_CHARS_PER_EVENT * 2)
        ids: list[int] = []
        for i in range(n_events):
            eid = insert_event_at(
                store,
                summary=f"e{i:03d}-{long_summary}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day - (n_events - i),
            )
            ids.append(eid)

        client = _FakeClient(content="A digest.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert out == "compacted"

        # The bullet block sent to the LLM should be capped.
        bullets = client.calls[0]["messages"][1]["content"]
        assert len(bullets) <= _MAX_BULLET_BLOCK_CHARS
        kept_bullet_lines = bullets.count("\n") + 1 if bullets else 0
        assert kept_bullet_lines < n_events  # i.e. some were trimmed

        # After compaction, surviving events = (n_events - kept) originals
        # + 1 digest. Events whose bullets were trimmed off the front
        # MUST survive; their ids are the OLDEST ones (lowest ts).
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id, source FROM events ORDER BY ts ASC, id ASC").fetchall()
        # 1 digest row (source='loom') + (n_events - kept_bullet_lines) survivors.
        survivor_ids = [r[0] for r in rows if r[1] != "loom"]
        digest_rows = [r for r in rows if r[1] == "loom"]
        assert len(digest_rows) == 1
        # Number of deletions equals number of bullet lines actually shown.
        assert len(survivor_ids) == n_events - kept_bullet_lines
        # Survivors are the OLDEST events (front-trimmed off the prompt).
        assert survivor_ids == ids[: n_events - kept_bullet_lines]

    def test_per_event_summary_cap_truncates_long_summaries(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import (
            _MAX_SUMMARY_CHARS_PER_EVENT,
            compact_verse,
        )
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        long_summary = "x" * 5000
        for _ in range(25):
            insert_event_at(
                store,
                summary=long_summary,
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        client = _FakeClient()
        compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert client.calls
        bullets = client.calls[0]["messages"][1]["content"]
        for line in bullets.splitlines():
            assert len(line) <= _MAX_SUMMARY_CHARS_PER_EVENT + 4

    def test_entity_ids_truncation_logs_when_capped(self, verse_db_dir: Path, caplog) -> None:
        import json as _json
        import logging

        from llm.verse.compaction import _MAX_DIGEST_ENTITY_IDS, compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=list(range(i * 4, i * 4 + 4)),
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        with caplog.at_level(logging.INFO, logger="llm.verse.compaction"):
            compact_verse(
                store,
                retention_days=30,
                min_keep_events=20,
                model="m",
                client=_FakeClient(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        assert any("entity_ids truncated" in r.message for r in caplog.records)
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_ids FROM events WHERE source='loom' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert len(_json.loads(row[0])) == _MAX_DIGEST_ENTITY_IDS

    def test_client_failure_aborts_without_data_loss(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )

        class Bomb:
            def call(self, **kw):
                raise RuntimeError("model down")

        with pytest.raises(RuntimeError):
            compact_verse(
                store,
                retention_days=30,
                min_keep_events=20,
                model="m",
                client=Bomb(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 25


class TestNextLocalTime:
    def test_returns_today_when_hhmm_in_future(self) -> None:
        import time as _t

        from llm.verse.compaction import _next_local_time

        # construct a "now" at local 03:00, ask for 10:00
        struct = _t.struct_time((2026, 5, 8, 3, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert out > now_ts
        assert (out - now_ts) < 86400  # under one day away

    def test_returns_tomorrow_when_hhmm_already_passed(self) -> None:
        import time as _t

        from llm.verse.compaction import _next_local_time

        struct = _t.struct_time((2026, 5, 8, 14, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert (out - now_ts) > 0
        assert (out - now_ts) < 86400  # under one day away

    def test_malformed_hhmm_falls_back_to_one_hour(self) -> None:
        from llm.verse.compaction import _next_local_time

        out = _next_local_time("not-a-time", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0

    def test_out_of_range_hhmm_falls_back(self) -> None:
        from llm.verse.compaction import _next_local_time

        out = _next_local_time("25:99", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0
