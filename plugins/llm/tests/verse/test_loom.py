"""Tests for the forest-verse loom orchestrator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest


def _minimal_cfg():
    from llm.verse.loom import LoomConfig

    return LoomConfig(
        network="afternet",
        loom_channel="#forest",
        bot_nicks=(),
        model="gemini/x",
        cycle_interval_s=300,
        verse_cooldown_s=20,
        beat_window_s=90,
        transcript_max_lines=40,
        transcript_max_chars=8000,
        auto_apply_threshold=0.85,
    )


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
    def test_seed_call_exception_aborts_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )

        class _BoomClient:
            def call(
                self, *, op: str, model: str, messages: list[dict[str, str]]
            ) -> tuple[str, LoomCallUsage]:
                raise RuntimeError("boom")

        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=_BoomClient())
        loom.tick()
        assert loom._active is None
        assert bridge.scheduled == []

    def test_empty_seed_content_aborts_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=StubClient({"seed": "   "}))
        loom.tick()
        assert loom._active is None
        assert bridge.posts == []

    def test_after_beat1_with_no_active_cycle_is_noop(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(channels=[], weights={}, store=store, snapshots={})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=StubClient({}))
        loom.after_beat1()
        loom.after_beat2()
        assert bridge.submitted_labels == []

    def test_beat_call_exception_finalizes_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )

        class _PartialBoom:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def call(
                self, *, op: str, model: str, messages: list[dict[str, str]]
            ) -> tuple[str, LoomCallUsage]:
                self.calls.append(op)
                if op == "seed":
                    return "ring", LoomCallUsage(1, 1, 0.0)
                raise RuntimeError("beat boom")

        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=_PartialBoom())
        loom.tick()
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()
        assert loom._active is None

    def test_empty_beat_content_still_schedules_digest(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )
        loom = Loom(
            cfg=_minimal_cfg(),
            bridge=bridge,
            client=StubClient({"seed": "ring", "beat": "   "}),
        )
        loom.tick()
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()
        # No second post (empty beat content), but digest still scheduled.
        assert bridge.posts == ["ring"]
        assert bridge.scheduled[-1][2] == "llm_loom_after_beat2"

    def test_digest_call_exception_finalizes_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )

        class _DigestBoom:
            def call(
                self, *, op: str, model: str, messages: list[dict[str, str]]
            ) -> tuple[str, LoomCallUsage]:
                if op == "digest":
                    raise RuntimeError("dig boom")
                return f"reply-{op}", LoomCallUsage(1, 1, 0.0)

        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=_DigestBoom())
        loom.tick()
        loom.observe_transcript("botB", "yes")
        bridge.scheduled[0][1]()
        loom.observe_transcript("botC", "no")
        bridge.scheduled[-1][1]()
        assert loom._active is None

    def test_digest_with_empty_transcript_finalizes(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "g", [], [])},
        )
        loom = Loom(
            cfg=_minimal_cfg(),
            bridge=bridge,
            client=StubClient({"seed": "ring", "beat": "echo"}),
        )
        loom.tick()
        loom.observe_transcript("botB", "yes")
        # Beat scheduled. Fire the beat — it has transcript so it posts.
        bridge.scheduled[0][1]()
        # Now manually clear the cycle's transcript to simulate the digest
        # having no transcript (e.g. truncated to empty by tight caps).
        loom._active.transcript.clear()
        bridge.scheduled[-1][1]()
        assert loom._active is None


class TestLoomTick:
    def test_tick_with_no_candidates_does_nothing(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(channels=[], weights={}, store=store, snapshots={})
        client = StubClient({})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        assert client.calls == []
        assert bridge.posts == []
        assert bridge.scheduled == []
        assert loom._active is None

    def test_idle_tick_does_not_advance_pointer(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(channels=[], weights={}, store=store, snapshots={})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=StubClient({}))
        loom.tick()
        loom.tick()
        loom.tick()
        assert loom._pointer == 0

    def test_tick_records_last_cycle_at_for_picked_channel(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [("avatar", "Forest", 1)], [])},
        )
        client = StubClient({"seed": "the bell rings"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        assert loom._last_cycle_by_channel["#afnet"] == bridge.now()
        assert bridge.posts == ["the bell rings"]
        assert bridge.scheduled
        assert bridge.scheduled[0][2] == "llm_loom_after_beat1"

    def test_tick_aborts_if_post_to_channel_fails(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
            post_returns=False,
        )
        client = StubClient({"seed": "the bell rings"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        assert bridge.scheduled == []
        assert "#afnet" not in loom._last_cycle_by_channel
        assert loom._active is None


class TestLoomAfterBeat1:
    def test_idle_short_circuit_finalizes_cycle(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
        )
        client = StubClient({"seed": "a faint hum"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        bridge.scheduled[0][1]()
        assert client.calls == ["seed"]
        assert len(bridge.posts) == 1
        assert loom._active is None
        assert [s[2] for s in bridge.scheduled] == ["llm_loom_after_beat1"]

    def test_with_transcript_posts_beat_and_schedules_digest(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [], [])},
        )
        client = StubClient({"seed": "ring", "beat": "shadows lengthen"})
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()
        assert bridge.posts[-1] == "shadows lengthen"
        assert bridge.scheduled[-1][2] == "llm_loom_after_beat2"
        assert client.calls == ["seed", "beat"]


class TestLoomDigestPhase:
    def test_full_cycle_applies_high_confidence_event(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge, StubClient

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("avatar", "Forest")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [("avatar", "Forest", 1)], [])},
        )
        client = StubClient(
            {
                "seed": "the bell rings",
                "beat": "shadows lengthen",
                "digest": (
                    '[{"op":"add_event",'
                    '"payload":{"summary":"a chime","entity_ids":[]},'
                    '"confidence":0.95,"provenance":"l-1","rationale":"r"}]'
                ),
            }
        )
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)

        loom.tick()
        loom.observe_transcript("botB", "I hear it too")
        bridge.scheduled[0][1]()
        loom.observe_transcript("botC", "the wind takes it")
        bridge.scheduled[-1][1]()

        events = store.recent_events()
        assert any(e.summary == "a chime" for e in events)
        rows = store.list_proposals(status="approved")
        assert len(rows) == 1
        assert rows[0].reviewer == "loom"
        assert client.calls == ["seed", "beat", "digest"]
        assert loom._active is None
        assert [u[1] for u in bridge.usage_log] == ["seed", "beat", "digest"]

    def test_uses_snapshotted_stable_block_across_phases(self, verse_db_dir) -> None:
        from llm.verse.loom import Loom, LoomCallUsage, VerseSnapshot
        from llm.verse.store import VerseStore

        from ._fakes import FakeBridge

        store = VerseStore(verse_db_dir, "#afnet")
        bridge = FakeBridge(
            channels=["#afnet"],
            weights={"#afnet": 5},
            store=store,
            snapshots={"#afnet": VerseSnapshot("#afnet", "grove", [("avatar", "Forest", 1)], [])},
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
            {"seed": "ring", "beat": "echo", "digest": "[]"},
        )
        loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)
        loom.tick()
        bridge.snapshots["#afnet"] = VerseSnapshot(
            "#afnet",
            "different summary",
            [("avatar", "Different", 2)],
            ["a new event"],
        )
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[0][1]()
        loom.observe_transcript("botB", "I hear it")
        bridge.scheduled[-1][1]()
        assert captured[0] == captured[1] == captured[2]
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


class TestLoomTickConsumesSeed:
    def test_receiver_pulls_one_seed_inserts_proposal(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import (
            Loom,
            LoomCallUsage,
            LoomConfig,
            VerseSnapshot,
        )
        from llm.verse.store import VerseStore

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

        class FakeBridge:
            def list_candidate_channels(self) -> list[str]:
                return ["#afnet"]

            def candidate_weight(self, channel: str) -> int:
                return 1

            def snapshot(self, channel: str) -> VerseSnapshot:
                return VerseSnapshot(
                    channel=channel,
                    summary="x",
                    top_entities=[],
                    recent_events=[],
                )

            def post_to_loom_channel(self, text: str) -> bool:
                return True

            def schedule_after(self, delay_s, fn, name):
                pass

            def submit(self, label, fn):
                fn()

            def now(self) -> float:
                return 1000.0

            def store_for(self, channel: str) -> Any:
                return store

            def log_usage(self, *, channel, op, model, usage):
                pass

            def crosspoll_store(self) -> Any | None:
                return cx

            def verse_allow_send(self, channel: str) -> bool:
                return False

            def verse_allow_receive(self, channel: str) -> bool:
                return True

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
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        loom.tick()

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
        from llm.verse.loom import (
            Loom,
            LoomCallUsage,
            LoomConfig,
            VerseSnapshot,
        )
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeCross:
            def claim_seed_for(self, ch, *, proposal_id):
                raise AssertionError("must not be called when receive disabled")

        cx = FakeCross()

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        class FakeBridge:
            def list_candidate_channels(self):
                return ["#afnet"]

            def candidate_weight(self, channel):
                return 1

            def snapshot(self, channel):
                return VerseSnapshot(
                    channel=channel,
                    summary="x",
                    top_entities=[],
                    recent_events=[],
                )

            def post_to_loom_channel(self, text):
                return True

            def schedule_after(self, *a, **kw):
                pass

            def submit(self, label, fn):
                fn()

            def now(self):
                return 1000.0

            def store_for(self, channel):
                return store

            def log_usage(self, **kw):
                pass

            def crosspoll_store(self):
                return cx

            def verse_allow_send(self, channel):
                return False

            def verse_allow_receive(self, channel):
                return False

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="m",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        loom.tick()

        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_claim_raises_is_swallowed(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import (
            Loom,
            LoomCallUsage,
            LoomConfig,
            VerseSnapshot,
        )
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class BoomCross:
            def claim_seed_for(self, ch, *, proposal_id):
                raise RuntimeError("db gone")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        class FakeBridge:
            def list_candidate_channels(self):
                return ["#afnet"]

            def candidate_weight(self, channel):
                return 1

            def snapshot(self, channel):
                return VerseSnapshot(
                    channel=channel, summary="x", top_entities=[], recent_events=[]
                )

            def post_to_loom_channel(self, text):
                return True

            def schedule_after(self, *a, **kw):
                pass

            def submit(self, label, fn):
                fn()

            def now(self):
                return 1000.0

            def store_for(self, channel):
                return store

            def log_usage(self, **kw):
                pass

            def crosspoll_store(self):
                return BoomCross()

            def verse_allow_send(self, channel):
                return False

            def verse_allow_receive(self, channel):
                return True

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="m",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        # Must not raise — the loom-cycle continues despite the claim error.
        loom.tick()
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_no_seed_available_is_noop(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import (
            Loom,
            LoomCallUsage,
            LoomConfig,
            VerseSnapshot,
        )
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class EmptyCross:
            def claim_seed_for(self, ch, *, proposal_id):
                return None

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        class FakeBridge:
            def list_candidate_channels(self):
                return ["#afnet"]

            def candidate_weight(self, channel):
                return 1

            def snapshot(self, channel):
                return VerseSnapshot(
                    channel=channel, summary="x", top_entities=[], recent_events=[]
                )

            def post_to_loom_channel(self, text):
                return True

            def schedule_after(self, *a, **kw):
                pass

            def submit(self, label, fn):
                fn()

            def now(self):
                return 1000.0

            def store_for(self, channel):
                return store

            def log_usage(self, **kw):
                pass

            def crosspoll_store(self):
                return EmptyCross()

            def verse_allow_send(self, channel):
                return False

            def verse_allow_receive(self, channel):
                return True

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="m",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        loom.tick()
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0

    def test_proposal_insert_failure_logs_dangling_row(self, verse_db_dir: Path) -> None:
        from llm.verse.loom import (
            Loom,
            LoomCallUsage,
            LoomConfig,
            VerseSnapshot,
        )
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")

        class FakeSeed:
            id = 99
            source_channel = "#other"
            summary = "doomed"
            payload: dict[str, Any] = {}
            created_at = 0.0

        class FakeCross:
            def claim_seed_for(self, ch, *, proposal_id):
                return FakeSeed()

        class BoomStore:
            """Wraps real store but raises on add_proposal."""

            def add_proposal(self, **kw):
                raise RuntimeError("disk full")

        class FakeClient:
            def call(self, **kw):
                return "", LoomCallUsage(0, 0, 0.0)

        class FakeBridge:
            def list_candidate_channels(self):
                return ["#afnet"]

            def candidate_weight(self, channel):
                return 1

            def snapshot(self, channel):
                return VerseSnapshot(
                    channel=channel, summary="x", top_entities=[], recent_events=[]
                )

            def post_to_loom_channel(self, text):
                return True

            def schedule_after(self, *a, **kw):
                pass

            def submit(self, label, fn):
                fn()

            def now(self):
                return 1000.0

            def store_for(self, channel):
                # Return the boom-store so add_proposal blows up.
                return BoomStore()

            def log_usage(self, **kw):
                pass

            def crosspoll_store(self):
                return FakeCross()

            def verse_allow_send(self, channel):
                return False

            def verse_allow_receive(self, channel):
                return True

        cfg = LoomConfig(
            network="afnet",
            loom_channel="#forest",
            bot_nicks=(),
            model="m",
            cycle_interval_s=300,
            verse_cooldown_s=1200,
            beat_window_s=90,
            transcript_max_lines=40,
            transcript_max_chars=8000,
            auto_apply_threshold=0.85,
            crosspoll_per_cycle_limit=1,
        )
        loom = Loom(cfg=cfg, bridge=FakeBridge(), client=FakeClient())
        # Must not raise — proposal insert failure is logged + swallowed.
        loom.tick()
        # Real store is untouched (we used BoomStore for the consume hook).
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        assert count == 0
