"""Tests for the forest-verse loom orchestrator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


class _StubMsg:
    content = "ok"


class _StubChoice:
    message = _StubMsg()


class _StubUsage:
    prompt_tokens = 7
    completion_tokens = 3


class _StubResp:
    choices = [_StubChoice()]
    usage = _StubUsage()


class TestApplyOrQueue:
    def test_high_confidence_event_auto_applies_and_records_audit_row(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "applied"
        assert len(store.recent_events()) == 1
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1
        assert rows[0].status == "approved"
        assert rows[0].reviewer == "loom"

    def test_low_confidence_queues_pending(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "queued"
        assert store.recent_events() == []
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1
        assert rows[0].status == "pending"

    def test_add_entity_always_queues_regardless_of_confidence(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(
            op="add_entity",
            payload={"kind": "place", "name": "Hollow Oak"},
            confidence=0.99,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result == "queued"
        assert store.list_entities_by_kind("place") == []
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1
        assert rows[0].status == "pending"


class TestLoomCycle:
    def test_append_grows_transcript_in_order(self) -> None:
        from llm.verse.loom import LoomCycle

        c = LoomCycle(
            cycle_id="c1",
            channel="#afnet",
            started_at=0.0,
            verse_stable_block="block",
        )
        c.append_transcript("botA", "hi")
        c.append_transcript("botB", "yo")
        assert c.transcript == [("botA", "hi"), ("botB", "yo")]

    def test_snapshot_transcript_returns_a_copy(self) -> None:
        from llm.verse.loom import LoomCycle

        c = LoomCycle(
            cycle_id="c1",
            channel="#afnet",
            started_at=0.0,
            verse_stable_block="block",
        )
        c.append_transcript("botA", "hi")
        snap = c.snapshot_transcript()
        c.append_transcript("botB", "yo")
        assert snap == [("botA", "hi")]


class TestLiteLLMLoomClient:
    def test_returns_content_and_usage(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        import litellm
        from llm.verse.loom import LiteLLMLoomClient

        monkeypatch.setattr(litellm, "completion", lambda **_: _StubResp())
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
        caplog.set_level(logging.WARNING, logger="llm.verse.loom")
        client = LiteLLMLoomClient()
        content, usage = client.call(op="seed", model="gemini/x", messages=[])
        assert content == "ok"
        assert usage.prompt_tokens == 7
        assert usage.completion_tokens == 3
        assert any("op=loom:seed" in rec.message for rec in caplog.records)


class TestTruncateTranscript:
    def test_caps_lines(self) -> None:
        from llm.verse.loom import truncate_transcript

        lines = [("a", f"x{i}") for i in range(100)]
        out = truncate_transcript(lines, max_lines=10, max_chars=10_000)
        assert len(out) == 10
        assert out[-1] == ("a", "x99")

    def test_caps_chars_after_lines(self) -> None:
        from llm.verse.loom import truncate_transcript

        lines = [("a", "x" * 100) for _ in range(50)]
        out = truncate_transcript(lines, max_lines=40, max_chars=500)
        total = sum(len(t) for _, t in out)
        assert total <= 500
        assert len(out) <= 5

    def test_dedupes_consecutive_identical_tuples(self) -> None:
        from llm.verse.loom import truncate_transcript

        lines = [("a", "ping"), ("a", "ping"), ("b", "ping"), ("a", "ping")]
        out = truncate_transcript(lines, max_lines=40, max_chars=10_000)
        assert out == [("a", "ping"), ("b", "ping"), ("a", "ping")]

    def test_empty_input_empty_output(self) -> None:
        from llm.verse.loom import truncate_transcript

        assert truncate_transcript([], max_lines=40, max_chars=8000) == []


class TestParseDigest:
    def test_parses_valid_array(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"add_event",'
            '"payload":{"summary":"x","entity_ids":[]},'
            '"confidence":0.9,"provenance":"l-1","rationale":"y"}]'
        )
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].op == "add_event"
        assert out[0].confidence == 0.9

    def test_strips_json_code_fence(self) -> None:
        from llm.verse.loom import parse_digest

        text = "```json\n[]\n```"
        assert parse_digest(text) == []

    def test_drops_proposals_missing_required_fields(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            "["
            '{"op":"add_event","payload":{},"confidence":0.9,'
            '"provenance":"x","rationale":"y"},'
            '{"op":"BOGUS","payload":{},"confidence":0.5,'
            '"provenance":"x","rationale":"y"},'
            '{"op":"add_event","payload":{"summary":"k","entity_ids":[]},'
            '"confidence":0.7,"provenance":"x","rationale":"y"}'
            "]"
        )
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].payload["summary"] == "k"

    def test_clamps_confidence_to_unit_interval(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"add_event",'
            '"payload":{"summary":"x","entity_ids":[]},'
            '"confidence":2.5,"provenance":"x","rationale":"y"}]'
        )
        out = parse_digest(text)
        assert out[0].confidence == 1.0

    def test_returns_empty_on_hard_parse_error(self) -> None:
        from llm.verse.loom import parse_digest

        assert parse_digest("not json at all") == []

    def test_drops_when_required_payload_value_wrong_type(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"add_event",'
            '"payload":{"summary":"x","entity_ids":"not-a-list"},'
            '"confidence":0.9,"provenance":"x","rationale":"y"}]'
        )
        assert parse_digest(text) == []

    def test_drops_when_entity_ids_element_not_int(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"add_event",'
            '"payload":{"summary":"x","entity_ids":["bad"]},'
            '"confidence":0.9,"provenance":"x","rationale":"y"}]'
        )
        assert parse_digest(text) == []

    def test_rejects_bool_as_int_for_entity_id(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"set_attribute",'
            '"payload":{"entity_id":true,"key":"k","value":"v"},'
            '"confidence":0.9,"provenance":"x","rationale":"y"}]'
        )
        assert parse_digest(text) == []


class TestPromptBuilders:
    def test_static_prefix_is_constant(self) -> None:
        from llm.verse.loom import LOOM_STATIC_PREFIX

        assert isinstance(LOOM_STATIC_PREFIX, str)
        assert "proposal" in LOOM_STATIC_PREFIX.lower()
        assert "json" in LOOM_STATIC_PREFIX.lower()
        assert '"id"' not in LOOM_STATIC_PREFIX

    def test_verse_stable_block_deterministic(self) -> None:
        from llm.verse.loom import VerseSnapshot, build_verse_stable_block

        snap = VerseSnapshot(
            channel="#afnet",
            summary="Three avatars wander a moonlit grove.",
            top_entities=[("avatar", "Forest"), ("place", "Hollow Oak")],
            recent_events=["Forest entered the grove.", "Owl hooted thrice."],
        )
        a = build_verse_stable_block(snap)
        b = build_verse_stable_block(snap)
        assert a == b
        assert "Forest" in a
        assert "Hollow Oak" in a
        assert "Owl hooted" in a

    def test_seed_tail_includes_emit_instruction(self) -> None:
        from llm.verse.loom import build_seed_tail

        out = build_seed_tail()
        assert "one line" in out.lower() or "1 line" in out.lower()

    def test_beat_tail_includes_transcript(self) -> None:
        from llm.verse.loom import build_beat_tail

        out = build_beat_tail(loom_transcript_so_far=[("botB", "the bell echoes")])
        assert "botB" in out
        assert "bell" in out

    def test_digest_tail_demands_json_array(self) -> None:
        from llm.verse.loom import build_digest_tail

        out = build_digest_tail(
            loom_transcript_so_far=[("botB", "the bell echoes")],
        )
        assert "json" in out.lower()
        assert "array" in out.lower() or "list" in out.lower()


class TestPickFocusVerse:
    def test_returns_none_if_all_in_cooldown(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse

        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=10, last_cycle_at=now - 5.0),
            VerseCandidate(channel="#b", weight=10, last_cycle_at=now - 5.0),
        ]
        assert pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0) is None

    def test_picks_highest_weight_outside_cooldown(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse

        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=2, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#b", weight=8, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#c", weight=5, last_cycle_at=now - 5.0),
        ]
        result = pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0)
        assert result is not None
        assert result.channel == "#b"

    def test_round_robin_with_three_tied_candidates(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse

        now = 1000.0
        candidates = [
            VerseCandidate(channel="#a", weight=5, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#b", weight=5, last_cycle_at=now - 60.0),
            VerseCandidate(channel="#c", weight=5, last_cycle_at=now - 60.0),
        ]
        picks = []
        for p in range(6):
            choice = pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=p)
            assert choice is not None
            picks.append(choice.channel)
        assert picks == ["#a", "#b", "#c", "#a", "#b", "#c"]

    def test_never_cycled_treated_as_eligible(self) -> None:
        from llm.verse.loom import VerseCandidate, pick_focus_verse

        now = 1000.0
        candidates = [VerseCandidate(channel="#a", weight=1, last_cycle_at=None)]
        result = pick_focus_verse(candidates, now=now, cooldown_s=20, pointer=0)
        assert result is not None
        assert result.channel == "#a"


def test_loomconfig_holds_all_settings() -> None:
    from llm.verse.loom import LoomConfig

    cfg = LoomConfig(
        network="afternet",
        loom_channel="#forest",
        bot_nicks=("botA", "botB"),
        model="gemini/gemini-flash-lite-latest",
        cycle_interval_s=300,
        verse_cooldown_s=1200,
        beat_window_s=90,
        transcript_max_lines=40,
        transcript_max_chars=8000,
        auto_apply_threshold=0.85,
    )
    assert cfg.loom_channel == "#forest"
    assert cfg.network == "afternet"
    assert cfg.bot_nicks == ("botA", "botB")
    with pytest.raises(FrozenInstanceError):
        cfg.cycle_interval_s = 1  # type: ignore[misc]
