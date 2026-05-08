"""End-to-end integration test for the forest-verse loom orchestrator.

Drives a real ``VerseStore`` through ``Loom.tick → after_beat1 → after_beat2``
with a ``FakeBridge`` and ``StubClient`` so the test stays deterministic
without touching IRC or the LLM. Verifies:

- High-confidence non-entity proposals auto-apply with an audit row
  (``status='approved' reviewer='loom'``).
- Low-confidence proposals queue ``status='pending'``.
- ``add_entity`` proposals always queue regardless of confidence.
- All three loom phases log usage via the bridge.
- An operator approves a queued proposal and the mutation lands.
- ``_load_proposal`` short-id prefix lookup works against the integration
  store.
"""

from __future__ import annotations


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


def _digest_payload() -> str:
    """Three proposals: high-conf event (auto-apply), low-conf
    set_attribute (queue), high-conf add_entity (queue)."""
    return (
        "["
        '{"op":"add_event",'
        '"payload":{"summary":"a chime echoes","entity_ids":[]},'
        '"confidence":0.95,"provenance":"l-1","rationale":"r1"},'
        '{"op":"set_attribute",'
        '"payload":{"entity_id":1,"key":"mood","value":"wary"},'
        '"confidence":0.4,"provenance":"l-2","rationale":"r2"},'
        '{"op":"add_entity",'
        '"payload":{"kind":"place","name":"Hollow Oak",'
        '           "summary":"A leaning trunk."},'
        '"confidence":0.99,"provenance":"l-3","rationale":"r3"}'
        "]"
    )


def test_full_cycle_then_operator_approval(verse_db_dir, tmp_path) -> None:
    from llm.verse.loom import Loom, VerseSnapshot
    from llm.verse.store import VerseStore

    from ._fakes import FakeBridge, StubClient

    store = VerseStore(verse_db_dir, "#forest")
    forest_id = store.add_entity("avatar", "Forest")

    bridge = FakeBridge(
        channels=["#forest"],
        weights={"#forest": 7},
        store=store,
        snapshots={
            "#forest": VerseSnapshot(
                "#forest",
                "1 active avatar",
                [("avatar", "Forest", forest_id)],
                [],
            )
        },
    )
    client = StubClient(
        {
            "seed": "the bell rings in the grove",
            "beat": "shadows lengthen across the path",
            "digest": _digest_payload(),
        }
    )
    loom = Loom(cfg=_minimal_cfg(), bridge=bridge, client=client)

    # --- Drive the full cycle ---
    loom.tick()
    assert bridge.posts == ["the bell rings in the grove"]
    loom.observe_transcript("botB", "I hear it too")
    bridge.scheduled[0][1]()  # after_beat1
    assert bridge.posts[-1] == "shadows lengthen across the path"
    loom.observe_transcript("botC", "the wind takes it")
    bridge.scheduled[-1][1]()  # after_beat2

    # --- Auto-applied event landed with audit row ---
    events = store.recent_events()
    assert any(e.summary == "a chime echoes" and e.source == "loom" for e in events)
    approved = store.list_proposals(status="approved")
    assert len(approved) == 1
    assert approved[0].reviewer == "loom"
    assert approved[0].op == "add_event"

    # --- Low-conf and add_entity both queued pending ---
    pending = store.list_proposals(status="pending")
    assert {p.op for p in pending} == {"set_attribute", "add_entity"}

    # --- All three calls logged via the bridge ---
    assert [u[1] for u in bridge.usage_log] == ["seed", "beat", "digest"]

    # --- Cycle finalized; pointer rotated exactly once ---
    assert loom._active is None
    assert loom._pointer == 1

    # --- Operator approves the queued add_entity by short-id prefix ---
    add_entity_proposal = next(p for p in pending if p.op == "add_entity")
    short_id = add_entity_proposal.id[:6]

    # _load_proposal lives on the plugin in production; we exercise the
    # store-only equivalent here.
    rows = [x for x in store.list_proposals(limit=200) if x.id.startswith(short_id)]
    assert len(rows) == 1
    assert rows[0].id == add_entity_proposal.id

    store.apply_proposal_and_mark(add_entity_proposal.id, reviewer="alice")
    final = store.get_proposal(add_entity_proposal.id)
    assert final is not None
    assert final.status == "approved"
    assert final.reviewer == "alice"
    places = store.list_entities_by_kind("place")
    assert any(p.name == "Hollow Oak" for p in places)

    # --- Forest avatar still present (not perturbed by loom mutations) ---
    e = store.get_entity(forest_id)
    assert e is not None and e.name == "Forest"

    # tmp_path used implicitly; assertion to keep the lint happy.
    assert tmp_path.exists()
