import json

from llm.verse.purge import list_loom_digest_candidates, purge_loom_data

_MIN_CHARS = 300


def _add_event(store, summary, entity_ids, source):
    """Seed one event with an event_actor link, returning its id."""
    return store.apply_direct(
        op="add_event",
        payload={"summary": summary, "entity_ids": list(entity_ids)},
        source=source,
        provenance="test",
    )


def test_apply_direct_writes_event_actor(store):
    """Sanity: the seeding path actually creates event_actor rows.

    The purge's orphan logic reads event_actor; if add_event didn't write it,
    every other assertion here would be meaningless.
    """
    eid = store.add_entity("npc", "probe", "")
    _add_event(store, "probe acted", [eid], "avatar")
    with store.read_connection() as conn:
        n = conn.execute("SELECT COUNT(*) FROM event_actor WHERE entity_id=?", (eid,)).fetchone()[0]
    assert n >= 1


def test_purge_removes_idlerpg_junk_keeps_canon(store):
    """End-to-end: loom/crosspoll junk deleted, reviewed digest re-stamped to 'llm', pinned canon and non-orphan auto-NPCs preserved."""
    # --- canon: a pinned roster entity with mixed-source events ---
    freddie = store.add_entity("npc", "Farty Freddie", "")
    store.set_attribute(freddie, "pinned", "1")
    keep_ev = _add_event(store, "Freddie's real deed", [freddie], "avatar")
    _add_event(store, "freddie defeats jspiros in combat", [freddie], "loom")

    # --- an authored operator event (no entity link) ---
    _add_event(store, "The Cathedral Siege", [], "operator")

    # --- a compaction lore-digest mis-stamped 'loom' (>300 chars) ---
    chronicler = store.add_entity("npc", "Stinky Sebastian", "")
    store.set_attribute(chronicler, "pinned", "1")
    digest_summary = "Chronicler fc42 recounts the anarchic reign of the Stinky Lads. " + (
        "Poo Pete and Assripping Alex schemed through the long winter, while " * 6
    )
    assert len(digest_summary) > _MIN_CHARS
    digest_ev = _add_event(store, digest_summary, [chronicler], "loom")

    # --- orphan auto-NPC: only loom events ---
    blaat = store.add_entity("npc", "blaat", "")
    store.set_attribute(blaat, "auto_created", "1")
    _add_event(store, "blaat defeats jspiros in combat", [blaat], "loom")

    # --- NON-orphan auto-NPC: has a surviving (avatar) event ---
    survivor = store.add_entity("npc", "survivor-npc", "")
    store.set_attribute(survivor, "auto_created", "1")
    _add_event(store, "survivor did a real thing", [survivor], "avatar")
    _add_event(store, "survivor combat junk", [survivor], "loom")

    # --- digest review: operator confirms the digest id(s) ---
    candidates = list_loom_digest_candidates(store, min_chars=_MIN_CHARS)
    cand_ids = [cid for cid, _ in candidates]
    assert digest_ev in cand_ids

    result = purge_loom_data(store, digest_ids=[digest_ev])

    # digest re-stamped + survives
    assert result.digests_restamped == 1
    with store.read_connection() as conn:
        src = conn.execute("SELECT source FROM events WHERE id=?", (digest_ev,)).fetchone()
        assert src is not None and src[0] == "llm"

    # canon intact
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (freddie,)).fetchone()
        assert conn.execute("SELECT 1 FROM events WHERE id=?", (keep_ev,)).fetchone()
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (chronicler,)).fetchone()
        assert conn.execute("SELECT 1 FROM events WHERE summary='The Cathedral Siege'").fetchone()

    # orphan deleted, non-orphan survivor kept
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (blaat,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (survivor,)).fetchone()
    assert result.entities_deleted == 1


def test_purge_dual_linkage_guard_spares_json_only_reference(store):
    """An auto-NPC referenced only via a SURVIVING event's entity_ids JSON
    (no event_actor row) must NOT be deleted, even if event_actor says
    loom-only."""
    npc = store.add_entity("npc", "json-ghost", "")
    store.set_attribute(npc, "auto_created", "1")
    # A loom event links it via event_actor (would mark it orphan)...
    _add_event(store, "json-ghost combat", [npc], "loom")
    # ...but a surviving operator event references it ONLY via entity_ids JSON,
    # with no event_actor row (legacy linkage). Insert raw.
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?,?,?,?)",
            (1.0, "legacy json-only mention", json.dumps([npc]), "operator"),
        )

    purge_loom_data(store, digest_ids=[])

    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (npc,)).fetchone()


def test_purge_no_digests_is_safe(store):
    """digest_ids=() re-stamps nothing and still purges junk."""
    n = store.add_entity("npc", "blaat", "")
    store.set_attribute(n, "auto_created", "1")
    _add_event(store, "blaat combat", [n], "loom")
    result = purge_loom_data(store, digest_ids=[])
    assert result.digests_restamped == 0
    assert result.events_deleted == 1


def test_purge_spares_autocreated_entity_whose_only_event_is_a_restamped_digest(store):
    """The re-stamp (step 0) runs BEFORE orphan computation (step 1): an
    UNPINNED auto-NPC whose sole event is a reviewed digest keeps a surviving
    ('llm') link after the re-stamp and is therefore NOT deleted. This is the
    case the big test misses (it uses a pinned chronicler, which survives for a
    different reason) — it proves the ordering matters."""
    npc = store.add_entity("npc", "lonely-chronicled-npc", "")
    store.set_attribute(npc, "auto_created", "1")
    long_summary = "A chronicle of the lonely npc's deeds. " + (
        "It wandered far and its doings were recorded at length. " * 8
    )
    assert len(long_summary) > 300
    digest = _add_event(store, long_summary, [npc], "loom")

    result = purge_loom_data(store, digest_ids=[digest])

    assert result.digests_restamped == 1
    with store.read_connection() as conn:
        assert conn.execute("SELECT 1 FROM entities WHERE id=?", (npc,)).fetchone()
        src = conn.execute("SELECT source FROM events WHERE id=?", (digest,)).fetchone()
        assert src is not None and src[0] == "llm"
