"""Tests for the forest-verse loom orchestrator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest


def _minimal_cfg():
    from ._fakes import make_loom_config

    return make_loom_config(
        network="afternet",
        model="gemini/x",
        verse_cooldown_s=20,
    )


def _make_reactive(
    verse_db_dir,
    *,
    chimein="the bell still hums",
    digest="[]",
    post_returns=True,
    channels=("#forest",),
):
    """Build (loom, bridge, client, store) for reactive-trigger tests.

    submit() runs inline in FakeBridge, so observe_transcript drives the
    whole open+chime synchronously; fire the scheduled callback for digest.
    """
    from llm.verse.loom import Loom
    from llm.verse.store import VerseStore

    from ._fakes import FakeBridge, StubClient, make_snapshot

    store = VerseStore(verse_db_dir, "#forest")
    # Always seed a "#forest" snapshot so a no-candidate test can restore the
    # channel and re-trigger without rebuilding the snapshot map.
    snaps = {c: make_snapshot(c, summary="g") for c in (*channels, "#forest")}
    bridge = FakeBridge(
        channels=list(channels),
        weights=dict.fromkeys(channels, 5),
        store=store,
        snapshots=snaps,
        post_returns=post_returns,
    )
    client = StubClient({"chimein": chimein, "digest": digest})
    loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
    return loom, bridge, client, store


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


class TestParseDigestExtra:
    def test_top_level_not_list_returns_empty(self) -> None:
        from llm.verse.loom import parse_digest

        assert parse_digest('{"not": "a list"}') == []

    def test_non_dict_item_dropped(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            "[42, "
            '{"op":"add_event",'
            '"payload":{"summary":"k","entity_ids":[]},'
            '"confidence":0.7,"provenance":"x","rationale":"y"}]'
        )
        out = parse_digest(text)
        assert len(out) == 1

    def test_payload_not_dict_dropped(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            "["
            '{"op":"add_event","payload":"oops",'
            '"confidence":0.7,"provenance":"x","rationale":"y"}'
            "]"
        )
        assert parse_digest(text) == []

    def test_unparseable_confidence_falls_back_to_zero(self) -> None:
        from llm.verse.loom import parse_digest

        text = (
            '[{"op":"add_event",'
            '"payload":{"summary":"x","entity_ids":[]},'
            '"confidence":"NaaN-ish","provenance":"x","rationale":"y"}]'
        )
        out = parse_digest(text)
        assert out and out[0].confidence == 0.0


class TestLoomFailureBranches:
    def test_digest_call_exception_finalizes_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_snapshot

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": make_snapshot("#afnet", summary="g")},
        )

        class _DigestBoom:
            def call(
                self, *, op: str, model: str, messages: list[dict[str, str]]
            ) -> tuple[str, LoomCallUsage]:
                if op == "digest":
                    raise RuntimeError("dig boom")
                return f"reply-{op}", LoomCallUsage(1, 1, 0.0)

        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=_DigestBoom())
        loom.observe_transcript("botB", "yes")  # chime-in posts, schedules after_chime
        loom.observe_transcript("botC", "no")  # appended to the cycle
        bridge.scheduled[-1][1]()  # after_chime -> digest (boom)
        assert loom._active is None

    def test_digest_with_empty_transcript_finalizes(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient, make_snapshot

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": make_snapshot("#afnet", summary="g")},
        )
        loom = Loom(
            cfg=_minimal_cfg(),
            bridge=bridge,
            client=StubClient({"chimein": "ring", "digest": "[]"}),
        )
        loom.observe_transcript("botB", "yes")  # chime-in posts, schedules after_chime
        # Clear the cycle's transcript to simulate the digest having no
        # transcript (e.g. truncated to empty by tight caps).
        loom._active.transcript.clear()
        bridge.scheduled[-1][1]()  # after_chime -> digest -> finalize
        assert loom._active is None


class TestLoomDigestPhase:
    def test_full_cycle_applies_high_confidence_event(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient, make_snapshot

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("avatar", "Forest")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={
                "#afnet": make_snapshot(
                    "#afnet", summary="grove", top_entities=[("avatar", "Forest", 1)]
                )
            },
        )
        client = StubClient(
            {
                "chimein": "shadows lengthen",
                "digest": (
                    '[{"op":"add_event",'
                    '"payload":{"summary":"a chime","entity_ids":[]},'
                    '"confidence":0.95,"provenance":"l-1","rationale":"r"}]'
                ),
            }
        )
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)

        loom.observe_transcript("botB", "I hear it too")  # chime-in (inline)
        loom.observe_transcript("botC", "the wind takes it")  # appended to cycle
        bridge.scheduled[-1][1]()  # after_chime -> digest

        events = store.recent_events()
        assert any(e.summary == "a chime" for e in events)
        rows = store.list_proposals(status="approved")
        assert len(rows) == 1
        assert rows[0].reviewer == "loom"
        assert client.calls == ["chimein", "digest"]
        assert loom._active is None
        assert [u[1] for u in bridge.usage_log] == ["chimein", "digest"]

    def test_uses_snapshotted_stable_block_across_phases(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_snapshot

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={
                "#afnet": make_snapshot(
                    "#afnet", summary="grove", top_entities=[("avatar", "Forest", 1)]
                )
            },
        )
        captured: list[str] = []

        class CapturingClient:
            def __init__(self, replies: dict[str, str]) -> None:
                self.replies = dict(replies)
                self.calls: list[str] = []

            def call(
                self,
                *,
                op: str,
                model: str,
                messages: list[dict[str, str]],
            ) -> tuple[str, LoomCallUsage]:
                captured.append(messages[1]["content"])
                self.calls.append(op)
                return self.replies[op], LoomCallUsage(10, 5, 0.0)

        client = CapturingClient(
            {"chimein": "ring", "digest": "[]"},
        )
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.observe_transcript("botB", "I hear it")  # opens cycle; chime-in captures block
        # Change the snapshot AFTER the cycle opened — the digest must still
        # use the block snapshotted at open, not this newer one.
        bridge.snapshots["#afnet"] = make_snapshot(
            "#afnet",
            summary="different summary",
            top_entities=[("avatar", "Different", 2)],
            recent_events=["a new event"],
        )
        bridge.scheduled[-1][1]()  # after_chime -> digest captures block
        assert captured[0] == captured[1]
        assert "different summary" not in captured[0]


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
        assert result.outcome == "applied"
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
        assert result.outcome == "queued"
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
        assert result.outcome == "queued"
        assert store.list_entities_by_kind("place") == []
        rows = store.list_proposals(cycle_id="c1")
        assert len(rows) == 1
        assert rows[0].status == "pending"

    def test_add_relation_with_bogus_ids_auto_rejected(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(
            op="add_relation",
            payload={"from_id": 0, "to_id": 3, "kind": "previously_owned"},
            confidence=0.95,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result.outcome == "rejected_invalid_refs"
        # No relation was applied (0 and 3 don't exist).
        assert store.list_relations() == []
        # Proposal row is rejected with the auto-validator reviewer.
        rows = store.list_proposals(cycle_id="c1", status="rejected")
        assert len(rows) == 1
        assert rows[0].reviewer == "auto-validator"

    def test_add_event_with_orphan_entity_id_auto_rejected(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("avatar", "Forest")  # id=1
        prop = ParsedProposal(
            op="add_event",
            payload={"summary": "x", "entity_ids": [1, 99]},
            confidence=0.95,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result.outcome == "rejected_invalid_refs"
        assert store.recent_events() == []
        rows = store.list_proposals(cycle_id="c1", status="rejected")
        assert len(rows) == 1

    def test_set_attribute_with_orphan_entity_id_auto_rejected(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        prop = ParsedProposal(
            op="set_attribute",
            payload={"entity_id": 99, "key": "k", "value": "v"},
            confidence=0.95,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result.outcome == "rejected_invalid_refs"

    def test_relation_with_existing_ids_still_works(self, verse_db_dir) -> None:
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("avatar", "Forest")
        b = store.add_entity("npc", "Owl")
        prop = ParsedProposal(
            op="add_relation",
            payload={"from_id": a, "to_id": b, "kind": "allied_with"},
            confidence=0.95,
            provenance="l-1",
            rationale="r",
        )
        result = apply_or_queue(store, prop, cycle_id="c1", threshold=0.85)
        assert result.outcome == "applied"
        assert len(store.list_relations(from_id=a)) == 1


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

    def test_threads_api_key_to_litellm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm
        from llm.verse.loom import LiteLLMLoomClient

        captured: dict[str, object] = {}

        def _fake_completion(**kwargs):
            captured.update(kwargs)
            return _StubResp()

        monkeypatch.setattr(litellm, "completion", _fake_completion)
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
        client = LiteLLMLoomClient(api_key="sk-test-123")
        client.call(op="seed", model="gemini/x", messages=[])
        assert captured.get("api_key") == "sk-test-123"

    def test_omits_api_key_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm
        from llm.verse.loom import LiteLLMLoomClient

        captured: dict[str, object] = {}

        def _fake_completion(**kwargs):
            captured.update(kwargs)
            return _StubResp()

        monkeypatch.setattr(litellm, "completion", _fake_completion)
        monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.0)
        # Empty string treated as "no key" — falls back to env-var auth.
        client = LiteLLMLoomClient(api_key="")
        client.call(op="seed", model="gemini/x", messages=[])
        assert "api_key" not in captured


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
            top_entities=[("avatar", "Forest", 1), ("place", "Hollow Oak", 2)],
            recent_events=["Forest entered the grove.", "Owl hooted thrice."],
        )
        a = build_verse_stable_block(snap)
        b = build_verse_stable_block(snap)
        assert a == b
        assert "Forest" in a
        assert "Hollow Oak" in a
        assert "Owl hooted" in a

    def test_digest_tail_demands_json_array(self) -> None:
        from llm.verse.loom import build_digest_tail

        out = build_digest_tail(
            loom_transcript_so_far=[("botB", "the bell echoes")],
        )
        assert "json" in out.lower()
        assert "array" in out.lower() or "list" in out.lower()


class TestBuildVerseStableBlock:
    def test_lists_entities_with_ids(self) -> None:
        from llm.verse.loom import VerseSnapshot, build_verse_stable_block

        snap = VerseSnapshot(
            channel="#afnet",
            summary="A wood at the edge of town.",
            top_entities=[
                ("place", "the brook", 4),
                ("avatar", "rin", 7),
            ],
            recent_events=["someone whispered"],
        )
        out = build_verse_stable_block(snap)
        assert "- place: the brook (id=4)" in out
        assert "- avatar: rin (id=7)" in out

    def test_recent_events_preserved(self) -> None:
        from llm.verse.loom import VerseSnapshot, build_verse_stable_block

        snap = VerseSnapshot(
            channel="#afnet",
            summary="x",
            top_entities=[],
            recent_events=["A", "B"],
        )
        out = build_verse_stable_block(snap)
        assert "- A" in out and "- B" in out


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


class TestStaticPrefixMentionsEntityIds:
    def test_prefix_documents_id_inclusion(self) -> None:
        from llm.verse.loom import LOOM_STATIC_PREFIX

        # Two assertions — neither is fragile to whitespace.
        assert "(id=" in LOOM_STATIC_PREFIX
        assert "reuse" in LOOM_STATIC_PREFIX.lower() and "id" in LOOM_STATIC_PREFIX.lower()


class TestParseDigestCrosspollSeed:
    def test_accepts_crosspoll_seed_op(self) -> None:
        from llm.verse.loom import parse_digest

        text = """
        [
          {
            "op": "crosspoll_seed",
            "payload": {"summary": "rumour from the brook", "entity_ids": [4]},
            "confidence": 0.6,
            "provenance": "transcript-line-2",
            "rationale": "ambient riffing"
          }
        ]
        """
        out = parse_digest(text)
        assert len(out) == 1
        assert out[0].op == "crosspoll_seed"
        assert out[0].payload["summary"] == "rumour from the brook"
        assert out[0].payload["entity_ids"] == [4]

    def test_rejects_crosspoll_seed_with_bad_payload(self) -> None:
        from llm.verse.loom import parse_digest

        text = '[{"op":"crosspoll_seed","payload":{"summary":"ok"},"confidence":0.5,"provenance":"p","rationale":"r"}]'
        # missing entity_ids
        out = parse_digest(text)
        assert out == []


class TestProposalEntityRefsResolveCrosspoll:
    def test_seed_refs_validate_against_source_store(self) -> None:
        from llm.verse.loom import ParsedProposal, _proposal_entity_refs_resolve

        class FakeStore:
            def __init__(self, known: set[int]) -> None:
                self.known = known

            def entity_exists(self, eid: int) -> bool:
                return eid in self.known

        store = FakeStore({4, 7})
        ok = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "x", "entity_ids": [4, 7]},
            confidence=0.5,
            provenance="p",
            rationale="r",
        )
        bad = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "x", "entity_ids": [99]},
            confidence=0.5,
            provenance="p",
            rationale="r",
        )
        assert _proposal_entity_refs_resolve(store, ok) is True
        assert _proposal_entity_refs_resolve(store, bad) is False


class TestApplyOrQueueCrosspollSeed:
    def _seed(self, **over: Any) -> Any:
        from llm.verse.loom import ParsedProposal

        base: dict[str, Any] = {
            "op": "crosspoll_seed",
            "payload": {"summary": "rumour", "entity_ids": []},
            "confidence": 0.6,
            "provenance": "p",
            "rationale": "r",
        }
        base.update(over)
        return ParsedProposal(**base)

    def test_disabled_send_returns_skipped(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.enqueued: list[tuple[str, str]] = []

            def enqueue_seed(self, *, source_channel, summary, payload):
                self.enqueued.append((source_channel, summary))
                return 1

        cx = FakeCross()
        result = apply_or_queue(
            store,
            self._seed(),
            cycle_id="c-1",
            threshold=0.85,
            crosspoll_store=cx,
            source_channel="#afnet",
            allow_send=False,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert result.outcome == "crosspoll_skipped_disabled"
        assert cx.enqueued == []

    def test_at_limit_returns_skipped(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def enqueue_seed(self, **kw):
                return 0

        result = apply_or_queue(
            store,
            self._seed(),
            cycle_id="c-1",
            threshold=0.85,
            crosspoll_store=FakeCross(),
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=1,
        )
        assert result.outcome == "crosspoll_skipped_limit"

    def test_emits_seed_writes_audit_event_and_increments(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def enqueue_seed(self, *, source_channel, summary, payload):
                self.calls.append(
                    {
                        "source_channel": source_channel,
                        "summary": summary,
                        "payload": payload,
                    }
                )
                return 42

        cx = FakeCross()
        result = apply_or_queue(
            store,
            self._seed(),
            cycle_id="c-1",
            threshold=0.85,
            crosspoll_store=cx,
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=2,
            already_emitted=0,
        )
        assert result.outcome == "crosspoll_emitted"
        assert result.seed_id == 42
        assert cx.calls == [
            {
                "source_channel": "#afnet",
                "summary": "rumour",
                "payload": {"summary": "rumour", "entity_ids": []},
            }
        ]
        # one audit event present, source='loom'
        with store.read_connection() as conn:
            rows = conn.execute("SELECT summary, source FROM events ORDER BY id ASC").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "loom"
        assert "crosspoll" in rows[0][0].lower()

    def test_invalid_refs_rejected_before_send_check(self, verse_db_dir: Path) -> None:
        # entity_ids=[99] doesn't resolve in this verse; we must hit the
        # rejected_invalid_refs branch even when allow_send=True. The
        # existing proposals.op CHECK rejects 'crosspoll_seed', so this
        # path must NOT call store.add_proposal.
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def enqueue_seed(self, **kw):
                raise AssertionError("must not be called")

        result = apply_or_queue(
            store,
            self._seed(payload={"summary": "x", "entity_ids": [99]}),
            cycle_id="c-1",
            threshold=0.85,
            crosspoll_store=FakeCross(),
            source_channel="#afnet",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert result.outcome == "rejected_invalid_refs"
        # No proposals row was written (CHECK would have rejected it).
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_no_outcome_writes_crosspoll_seed_to_proposals(self, verse_db_dir: Path) -> None:
        """Schema-invariant regression test: across every apply_or_queue
        outcome for op='crosspoll_seed', no proposals row with
        op='crosspoll_seed' is ever written."""
        from llm.verse.loom import apply_or_queue
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def __init__(self) -> None:
                self.calls = 0

            def enqueue_seed(self, *, source_channel, summary, payload):
                self.calls += 1
                return self.calls

        cx = FakeCross()
        # Drive every outcome we care about: emit, skip-disabled,
        # skip-limit, rejected-invalid-refs.
        outcomes: list[str] = []
        outcomes.append(
            apply_or_queue(
                store,
                self._seed(),
                cycle_id="c-1",
                threshold=0.85,
                crosspoll_store=cx,
                source_channel="#afnet",
                allow_send=True,
                per_cycle_limit=1,
                already_emitted=0,
            ).outcome
        )
        outcomes.append(
            apply_or_queue(
                store,
                self._seed(),
                cycle_id="c-1",
                threshold=0.85,
                crosspoll_store=cx,
                source_channel="#afnet",
                allow_send=False,
                per_cycle_limit=1,
                already_emitted=0,
            ).outcome
        )
        outcomes.append(
            apply_or_queue(
                store,
                self._seed(),
                cycle_id="c-1",
                threshold=0.85,
                crosspoll_store=cx,
                source_channel="#afnet",
                allow_send=True,
                per_cycle_limit=1,
                already_emitted=1,
            ).outcome
        )
        outcomes.append(
            apply_or_queue(
                store,
                self._seed(payload={"summary": "x", "entity_ids": [99]}),
                cycle_id="c-1",
                threshold=0.85,
                crosspoll_store=cx,
                source_channel="#afnet",
                allow_send=True,
                per_cycle_limit=1,
                already_emitted=0,
            ).outcome
        )
        # Sanity: we drove the four distinct branches.
        assert set(outcomes) == {
            "crosspoll_emitted",
            "crosspoll_skipped_disabled",
            "crosspoll_skipped_limit",
            "rejected_invalid_refs",
        }
        # The schema-invariant assertion:
        with store.read_connection() as conn:
            bad = conn.execute(
                "SELECT COUNT(*) FROM proposals WHERE op='crosspoll_seed'"
            ).fetchone()[0]
        assert bad == 0


class TestLoomConfigCrosspollDefault:
    def test_per_cycle_limit_defaults_present_in_dataclass(self) -> None:
        from llm.verse.loom import LoomConfig

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="gemini/gemini-flash-lite-latest",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        assert cfg.crosspoll_per_cycle_limit == 1


class TestLoomBridgeProtocolHasCrosspoll:
    def test_protocol_documents_three_new_methods(self) -> None:
        import inspect

        from llm.verse.loom import LoomBridge

        members = {n for n, _ in inspect.getmembers(LoomBridge)}
        for name in ("crosspoll_store", "verse_allow_send", "verse_allow_receive"):
            assert name in members, f"LoomBridge missing {name}"


class TestDigestPhaseRoutesCrosspoll:
    def test_emit_caps_at_per_cycle_limit(self, verse_db_dir: Path) -> None:
        """Two seeds in one digest, limit=1 -> only first enqueued."""
        from llm.verse.loom import (
            LoomCycle,
            apply_or_queue,
            parse_digest,
        )
        from llm.verse.store import VerseStore

        from .conftest import fixture_text

        store = VerseStore(verse_db_dir, "#afnet")
        digest = fixture_text("digests/two_seeds.json")
        proposals = parse_digest(digest)
        assert len(proposals) == 2

        enqueued: list[str] = []

        class FakeCross:
            def enqueue_seed(self, *, source_channel, summary, payload):
                enqueued.append(summary)
                return len(enqueued)

        cx = FakeCross()
        cycle = LoomCycle(
            cycle_id="c1",
            channel="#afnet",
            started_at=0.0,
            verse_stable_block="block",
        )
        for p in proposals:
            r = apply_or_queue(
                store,
                p,
                cycle_id=cycle.cycle_id,
                threshold=0.85,
                crosspoll_store=cx,
                source_channel=cycle.channel,
                allow_send=True,
                per_cycle_limit=1,
                already_emitted=cycle.emitted_seeds,
            )
            if r.outcome == "crosspoll_emitted":
                cycle.emitted_seeds += 1

        assert cycle.emitted_seeds == 1
        assert enqueued == ["first whisper"]
        # second seed was skipped, NOT silently re-enqueued
        assert "second whisper" not in enqueued


class TestDigestPhaseIsolatesCrosspollFailure:
    """Regression: previously a failure in crosspoll_store() during
    digest dropped EVERY proposal, not just crosspoll seeds. Now: the
    crosspoll-store lookup is deferred until a crosspoll_seed proposal
    needs it, and any failure isolates to that branch."""

    def test_non_crosspoll_proposals_proceed_when_cx_store_raises(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config, make_snapshot

        store = VerseStore(verse_db_dir, "#afnet")
        # Need an entity so the add_event proposal's entity_ids resolve.
        eid = store.add_entity(kind="place", name="Clearing", summary="quiet")

        digest_payload = (
            "["
            '{"op":"crosspoll_seed","payload":{"summary":"seed","entity_ids":[]},'
            '"confidence":0.6,"provenance":"x","rationale":""},'
            '{"op":"add_event","payload":{"summary":"a quiet step","entity_ids":'
            f"[{eid}]"
            '},"confidence":0.5,"provenance":"x","rationale":""}'
            "]"
        )

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def call(self, *, op, model, messages):
                self.calls.append(op)
                if op == "digest":
                    return digest_payload, LoomCallUsage(0, 0, 0.0)
                return "x", LoomCallUsage(0, 0, 0.0)

        # crosspoll_store() raises: the receive hook calls this too, so this
        # also exercises the IMPORTANT-7 swallow. allow_receive is False so
        # the receive hook bails before it would blow up.
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll_raises=RuntimeError("crosspoll DB unavailable"),
            allow_send=True,
            allow_receive=False,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())

        # Drive the digest phase directly. tick() would also work but
        # this avoids replaying the seed/beat path.
        from llm.verse.loom import LoomCycle, build_verse_stable_block

        snap = make_snapshot(
            "#afnet",
            top_entities=[("place", "Clearing", eid)],
        )
        cycle = LoomCycle(
            cycle_id="c-isolate",
            channel="#afnet",
            started_at=0.0,
            verse_stable_block=build_verse_stable_block(snap),
        )
        # Seed the transcript so digest_phase doesn't bail on empty.
        cycle.append_transcript("alice", "hello")
        loom._active = cycle

        # Must not raise even though crosspoll_store() blows up.
        loom._digest_phase(cycle)

        # Non-crosspoll proposal landed in the proposals table (queued
        # because confidence < threshold of 0.85).
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT op, status FROM proposals WHERE cycle_id='c-isolate'"
            ).fetchall()
        assert rows == [("add_event", "pending")]


class TestReactiveConsumesSeed:
    def test_receiver_pulls_one_seed_inserts_proposal(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeSeed:
            id = 7
            source_channel = "#other"
            summary = "incoming whisper"
            payload: dict[str, Any] = {}
            created_at = 0.0

        class FakeCross:
            def __init__(self) -> None:
                self._available: list[Any] = [FakeSeed()]
                self.claims: list[tuple[int, str, str]] = []

            def claim_seed_for(self, ch: str, *, proposal_id: str) -> Any | None:
                if not self._available:
                    return None
                seed = self._available.pop(0)
                self.claims.append((seed.id, ch, proposal_id))
                return seed

        cx = FakeCross()

        class FakeClient:
            def call(self, *, op, model, messages):
                # Empty content — seed phase short-circuits and the cycle
                # finalises before any beats get scheduled.
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll=cx,
            allow_receive=True,
        )
        cfg = make_loom_config(model="gemini/gemini-flash-lite-latest")
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        loom.observe_transcript("botB", "ping")

        assert cx.claims and cx.claims[0][1] == "#afnet"
        assert cx.claims[0][0] == 7
        proposal_id_claimed = cx.claims[0][2]
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id, op, status, payload FROM proposals").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == proposal_id_claimed
        assert rows[0][1] == "add_event"
        assert rows[0][2] == "pending"
        import json

        assert json.loads(rows[0][3])["summary"] == "incoming whisper"

    def test_no_pull_when_receive_disabled(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def claim_seed_for(self, ch, *, proposal_id):
                raise AssertionError("must not be called when receive disabled")

        cx = FakeCross()

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll=cx,
            allow_receive=False,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        loom.observe_transcript("botB", "ping")

        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_claim_raises_is_swallowed(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class BoomCross:
            def claim_seed_for(self, ch, *, proposal_id):
                raise RuntimeError("db gone")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll=BoomCross(),
            allow_receive=True,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        # Must not raise — the loom-cycle continues despite the claim error.
        loom.observe_transcript("botB", "ping")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_no_seed_available_is_noop(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class EmptyCross:
            def claim_seed_for(self, ch, *, proposal_id):
                return None

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll=EmptyCross(),
            allow_receive=True,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        loom.observe_transcript("botB", "ping")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_proposal_insert_failure_logs_dangling_row(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeSeed:
            id = 99
            source_channel = "#other"
            summary = "doomed"
            payload: dict[str, Any] = {}
            created_at = 0.0

        class FakeCross:
            def __init__(self) -> None:
                self.released: list[tuple[int, str]] = []

            def claim_seed_for(self, ch, *, proposal_id):
                return FakeSeed()

            def release_claim(self, seed_id: int, dest_channel: str) -> bool:
                self.released.append((seed_id, dest_channel))
                return True

        class BoomStore:
            """Wraps real store but raises on add_proposal."""

            def add_proposal(self, **kw):
                raise RuntimeError("disk full")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        cx_singleton = FakeCross()

        # Return the boom-store so add_proposal blows up.
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=BoomStore(),
            snapshots={},
            weight=1,
            crosspoll=cx_singleton,
            allow_receive=True,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        # Must not raise — proposal insert failure is logged + swallowed.
        loom.observe_transcript("botB", "ping")
        # Real store is untouched (we used BoomStore for the consume hook).
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0
        # Regression: insert failure must release the consumption row so
        # the seed isn't lost. Without this the row is permanent.
        assert cx_singleton.released == [(99, "#afnet")]

    def test_proposal_insert_failure_re_pends_via_real_store(self, tmp_path: Path) -> None:
        """End-to-end: with a real CrosspollStore, an add_proposal failure
        must leave the seed pending again (release_claim ran)."""
        from llm.verse.crosspoll_store import CrosspollStore
        from llm.verse.loom import Loom, LoomCallUsage

        from ._fakes import FakeBridge, make_loom_config

        cx = CrosspollStore(tmp_path / "verse")
        cx.enqueue_seed(source_channel="#other", summary="doomed", payload={})

        class BoomStore:
            def add_proposal(self, **kw):
                raise RuntimeError("disk full")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=BoomStore(),
            snapshots={},
            weight=1,
            crosspoll=cx,
            allow_receive=True,
        )
        cfg = make_loom_config()
        # Pending count starts at 1.
        assert cx.pending_count_for("#afnet") == 1
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        loom.observe_transcript("botB", "ping")
        # After the failed insert + release, the seed is pending again.
        assert cx.pending_count_for("#afnet") == 1

    def test_release_claim_retries_transient_failure(self, tmp_path: Path) -> None:
        """A transient release_claim failure (e.g. a momentary SQLite lock)
        after a proposal-insert failure must be retried so the seed is NOT
        orphaned: the consumption row is eventually removed and the seed
        becomes pending again."""
        import sqlite3

        from llm.verse.crosspoll_store import CrosspollStore
        from llm.verse.loom import Loom, LoomCallUsage

        from ._fakes import FakeBridge, make_loom_config

        cx = CrosspollStore(tmp_path / "verse")
        cx.enqueue_seed(source_channel="#other", summary="doomed", payload={})

        # release_claim raises on its first call, then delegates to the real
        # implementation — simulating a transient lock that clears on retry.
        real_release = cx.release_claim
        calls = {"n": 0}

        def flaky_release(seed_id: int, dest_channel: str) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_release(seed_id, dest_channel)

        cx.release_claim = flaky_release  # type: ignore[method-assign]

        class BoomStore:
            def add_proposal(self, **kw):
                raise RuntimeError("disk full")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=BoomStore(),
            snapshots={},
            weight=1,
            crosspoll=cx,
            allow_receive=True,
        )
        cfg = make_loom_config()
        assert cx.pending_count_for("#afnet") == 1
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        loom.observe_transcript("botB", "ping")

        # The release was retried past the first transient failure...
        assert calls["n"] >= 2
        # ...so the consumption row was removed and the seed is pending again.
        assert cx.pending_count_for("#afnet") == 1

    def test_release_claim_exhausts_retries_orphans_seed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If EVERY release_claim attempt fails after a proposal-insert
        failure, _release_claim_with_retry hits its exhaustion branch:
        it logs at ERROR (seed lost for this dest until manual cleanup)
        and returns without re-pending. The consumption row is never
        removed, so the seed stays orphaned (pending_count stays 0) and
        the loom tick must not raise."""
        import logging
        import sqlite3

        from llm.verse.crosspoll_store import CrosspollStore
        from llm.verse.loom import Loom, LoomCallUsage

        from ._fakes import FakeBridge, make_loom_config

        cx = CrosspollStore(tmp_path / "verse")
        cx.enqueue_seed(source_channel="#other", summary="doomed", payload={})

        # release_claim raises on EVERY call: simulates a lock that never
        # clears, driving the retry loop to exhaustion.
        calls = {"n": 0}

        def always_fail_release(seed_id: int, dest_channel: str) -> bool:
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        cx.release_claim = always_fail_release  # type: ignore[method-assign]

        class BoomStore:
            def add_proposal(self, **kw):
                raise RuntimeError("disk full")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=BoomStore(),
            snapshots={},
            weight=1,
            crosspoll=cx,
            allow_receive=True,
        )
        cfg = make_loom_config()
        assert cx.pending_count_for("#afnet") == 1
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        caplog.set_level(logging.ERROR, logger="llm.verse.loom")
        # Must not raise even though release_claim never succeeds.
        loom.observe_transcript("botB", "ping")

        # All three attempts were made (default attempts=3) and every one
        # raised, so the retry loop ran to exhaustion.
        assert calls["n"] == 3
        # Exhaustion branch logged at ERROR with the orphan-cleanup wording.
        assert any(
            rec.levelno == logging.ERROR
            and "release_claim failed" in rec.message
            and rec.name == "llm.verse.loom"
            for rec in caplog.records
        )
        # Release never succeeded, so the consumption row stays: the seed is
        # orphaned (NOT re-pended) -> pending count stays 0, unlike the
        # retry-then-success path which re-pends it back to 1.
        assert cx.pending_count_for("#afnet") == 0

    def test_consume_swallows_bridge_construction_failure(self, verse_db_dir: Path) -> None:
        """Regression: ``crosspoll_store()`` raising during the receive
        hook must NOT abort the loom tick — the seed/beat/digest path
        doesn't depend on receive working."""
        from llm.verse.loom import Loom, LoomCallUsage
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, make_loom_config

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeClient:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def call(self, *, op, model, messages):
                self.calls.append(op)
                return "", LoomCallUsage(0, 0, 0.0)

        bridge = FakeBridge(
            channels=["#afnet"],
            weights={},
            store=store,
            snapshots={},
            weight=1,
            crosspoll_raises=RuntimeError("DB unavailable"),
            allow_receive=True,
        )
        cfg = make_loom_config()
        loom = Loom(cfg=cfg, bridge=bridge, client=FakeClient())
        # Must not raise — bridge failure is logged + swallowed.
        loom.observe_transcript("botB", "ping")
        # Seed phase still ran (called once, even though content is empty).
        # No assertion on seed run — both behaviours are acceptable
        # because the seed phase runs regardless.


class TestCrosspollEndToEnd:
    def test_seed_emitted_then_consumed_then_approved(self, tmp_path: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore
        from llm.verse.loom import ParsedProposal, apply_or_queue
        from llm.verse.store import VerseStore

        verse_dir = tmp_path / "verse"
        verse_dir.mkdir()
        cx = CrosspollStore(verse_dir)
        src_store = VerseStore(verse_dir, "#alpha")
        rcv_store = VerseStore(verse_dir, "#beta")

        # Source emits one crosspoll_seed via apply_or_queue.
        seed_prop = ParsedProposal(
            op="crosspoll_seed",
            payload={"summary": "a rumour from alpha", "entity_ids": []},
            confidence=0.7,
            provenance="t-1",
            rationale="ambient",
        )
        out = apply_or_queue(
            src_store,
            seed_prop,
            cycle_id="c-src",
            threshold=0.85,
            crosspoll_store=cx,
            source_channel="#alpha",
            allow_send=True,
            per_cycle_limit=1,
            already_emitted=0,
        )
        assert out.outcome == "crosspoll_emitted"

        # Receiver atomically claims the seed (consumption row + read in
        # one TX), then inserts the local pending proposal with the same id.
        import uuid as _uuid

        proposal_id = _uuid.uuid4().hex
        seed = cx.claim_seed_for("#beta", proposal_id=proposal_id)
        assert seed is not None and seed.source_channel == "#alpha"
        rcv_store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": seed.summary, "entity_ids": []},
            confidence=0.0,
            provenance=f"crosspoll from #alpha (seed-id={seed.id})",
            proposal_id=proposal_id,
        )

        # Operator approves; receiver event row gets source='crosspoll'.
        rcv_store.apply_proposal_and_mark(proposal_id, reviewer="op", event_source="crosspoll")
        with rcv_store.read_connection() as conn:
            rows = conn.execute("SELECT summary, source FROM events").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "a rumour from alpha"
        assert rows[0][1] == "crosspoll"

        # Second claim returns None — already consumed for this dest.
        assert cx.claim_seed_for("#beta", proposal_id="p-x") is None


class TestBuildChimeinTail:
    def test_frames_lines_as_spontaneous_and_forbids_json(self) -> None:
        from llm.verse.loom import build_chimein_tail

        tail = build_chimein_tail(
            loom_transcript_so_far=[("botB", "the bell rings"), ("botC", "i hear it")]
        )
        assert "botB: the bell rings" in tail
        assert "botC: i hear it" in tail
        # Framed as the others speaking unprompted, not replying to us.
        assert "unprompted" in tail
        assert "replied" not in tail
        # Same guardrails as seed/beat: one line, no JSON.
        assert "Do NOT emit JSON" in tail


class TestReactiveTrigger:
    def test_first_line_opens_cycle_and_posts_single_chimein(self, verse_db_dir) -> None:
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="the bell still hums")

        loom.observe_transcript("botB", "the bell rings")

        # Exactly one post (the chime-in); a digest is scheduled, not posted.
        assert bridge.posts == ["the bell still hums"]
        assert bridge.scheduled[-1][2] == "llm_loom_after_chime"
        assert client.calls == ["chimein"]

    def test_chimein_transcript_includes_triggering_line(self, verse_db_dir) -> None:
        loom, _, client, _ = _make_reactive(verse_db_dir, chimein="ok")

        loom.observe_transcript("botB", "the bell rings")
        # The chime-in user message must contain the spontaneous first line.
        # FakeBridge does not capture messages, so assert via a capturing client.
        assert "the bell rings" in client.last_user_content

    def test_second_line_within_interval_is_ignored(self, verse_db_dir) -> None:
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok")

        loom.observe_transcript("botB", "first")  # opens + chimes (inline)
        bridge.scheduled[-1][1]()  # after_chime -> digest -> _active=None
        bridge.t += 10  # still < cycle_interval_s (300)
        loom.observe_transcript("botC", "second")  # within interval -> ignored

        assert client.calls == ["chimein", "digest"]  # no second chime-in
        assert bridge.posts == ["ok"]

    def test_line_after_interval_opens_new_cycle(self, verse_db_dir) -> None:
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok")

        loom.observe_transcript("botB", "first")
        bridge.scheduled[-1][1]()  # finalize cycle 1
        # Advance past cycle_interval_s (300). verse_cooldown_s is 20, well
        # below this gap, so #forest is eligible to be re-picked.
        bridge.t += _minimal_cfg().cycle_interval_s + 1
        loom.observe_transcript("botC", "second")  # now due again

        assert bridge.posts == ["ok", "ok"]
        assert client.calls == ["chimein", "digest", "chimein"]

    def test_lines_during_active_cycle_append_not_retrigger(self, verse_db_dir) -> None:
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok")

        loom.observe_transcript("botB", "first")  # opens cycle, posts chimein
        loom.observe_transcript("botC", "second")  # active cycle -> append only
        # No new chime-in posted; second line waits for digest.
        assert bridge.posts == ["ok"]
        bridge.scheduled[-1][1]()  # digest sees both lines
        assert "second" in client.last_user_content  # digest user content

    def test_no_eligible_verse_rolls_back_and_stays_due(self, verse_db_dir) -> None:
        # No candidate channels -> worker finds no verse -> rollback.
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok", channels=())

        loom.observe_transcript("botB", "first")  # forms, worker finds no verse
        assert bridge.posts == []
        assert client.calls == []
        assert loom._active is None
        assert loom._pointer == 0  # idle rollback must NOT advance round-robin
        # Still due: restoring a channel and firing a new line opens a cycle.
        # (_make_reactive always seeds a "#forest" snapshot, so it's present.)
        bridge.channels = ["#forest"]
        bridge.weights = {"#forest": 5}
        loom.observe_transcript("botC", "second")
        assert bridge.posts == ["ok"]
        assert loom._pointer == 1  # advances only on a successful pick

    def test_post_failure_rolls_back_and_stays_due(self, verse_db_dir) -> None:
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok", post_returns=False)

        loom.observe_transcript("botB", "first")  # chime-in post fails
        assert bridge.scheduled == []  # no digest scheduled
        # Cooldown rolled back: a new line (post now works) opens a cycle.
        bridge.post_returns = True
        loom.observe_transcript("botC", "second")
        # FakeBridge.post_to_loom_channel appends *then* returns its status,
        # so the failed first attempt is recorded too: two appends total.
        assert bridge.posts == ["ok", "ok"]
        assert bridge.scheduled[-1][2] == "llm_loom_after_chime"  # second armed digest

    def test_empty_chimein_rolls_back_and_stays_due(self, verse_db_dir) -> None:
        # An empty/whitespace model response must NOT burn the interval gate
        # or cool down the verse — it is a no-op attempt, identical in spirit
        # to a post failure.
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="   ")

        loom.observe_transcript("botB", "first")  # empty chime-in -> rollback
        assert bridge.posts == []
        assert bridge.scheduled == []
        assert loom._active is None
        assert loom._last_chime_at is None  # interval gate NOT consumed
        # Still due: a non-empty reply on the next line opens a cycle.
        client.replies["chimein"] = "now i speak"
        loom.observe_transcript("botC", "second")
        assert bridge.posts == ["now i speak"]

    def test_chimein_strips_leaked_control_tokens(self, verse_db_dir) -> None:
        # A model that leaks its end-of-sequence sentinel as literal text must
        # not post it to the loom channel (sibling of sanitize_output's strip).
        loom, bridge, _, _ = _make_reactive(verse_db_dir, chimein="the bell still hums<|eos|>")

        loom.observe_transcript("botB", "the bell rings")

        assert bridge.posts == ["the bell still hums"]

    def test_chimein_only_control_token_rolls_back(self, verse_db_dir) -> None:
        # A line that is nothing but a leaked sentinel collapses to empty once
        # stripped; it must roll back like any empty response, never post blank.
        loom, bridge, _, _ = _make_reactive(verse_db_dir, chimein="<|eos|>")

        loom.observe_transcript("botB", "first")

        assert bridge.posts == []
        assert bridge.scheduled == []
        assert loom._active is None
        assert loom._last_chime_at is None  # interval gate NOT consumed

    def test_chimein_call_exception_finalizes_cycle(self, verse_db_dir) -> None:
        from ._fakes import StubClient

        class BoomClient(StubClient):
            def call(self, *, op, model, messages):
                if op == "chimein":
                    self.calls.append(op)  # record before raising
                    raise RuntimeError("boom")
                return super().call(op=op, model=model, messages=messages)

        loom, bridge, _, _ = _make_reactive(verse_db_dir, chimein="x")
        # Swap in the exploding client (same replies dict shape).
        client = BoomClient({"chimein": "x", "digest": "[]"})
        loom._client = client

        loom.observe_transcript("botB", "first")
        assert bridge.posts == []
        assert bridge.scheduled == []
        # Rolled back -> still due.
        loom.observe_transcript("botC", "second")
        assert client.calls.count("chimein") == 2

    def test_trigger_path_does_not_snapshot_on_driver_thread(self, verse_db_dir) -> None:
        # The cheap trigger path forms the cycle and offloads; snapshot must
        # happen inside the submitted worker, not before submit() is called.
        loom, bridge, client, _ = _make_reactive(verse_db_dir, chimein="ok")

        order = []
        orig_snapshot = bridge.snapshot
        orig_submit = bridge.submit

        def tracking_snapshot(channel):
            order.append("snapshot")
            return orig_snapshot(channel)

        def tracking_submit(label, fn):
            order.append(f"submit:{label}")
            return orig_submit(label, fn)

        bridge.snapshot = tracking_snapshot
        bridge.submit = tracking_submit

        loom.observe_transcript("botB", "first")

        # submit:loom:open is recorded before the first snapshot call.
        assert order[0] == "submit:loom:open"
        assert "snapshot" in order
        assert order.index("submit:loom:open") < order.index("snapshot")
