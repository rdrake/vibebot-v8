from llm.verse.avatar import (
    VERSE_SCENE_MARKER,
    build_story_world_context,
    build_verse_context_block,
    build_verse_system_prompt,
)
from llm.verse.store import VerseStore


def test_verse_context_block_facts_without_roleplay_framing(tmp_path):
    """The chat-path canon block carries facts (roster + referenced cast) but
    NONE of the roleplay framing — no identity line, persona, or scene marker."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    block = build_verse_context_block(store, "what would Assgas Archie say?", avatar_id=me)
    assert "Assgas Archie: Y11 windbag" in block
    # Facts only — no persona takeover.
    assert "You are" not in block
    assert VERSE_SCENE_MARKER not in block
    assert "Persona" not in block
    # Speaker isn't described back to themselves.
    assert "Hero" not in block


def test_verse_context_block_empty_when_no_canon_and_no_match(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("avatar", "Hero")
    store.add_entity("npc", "Ghost", "unpinned, unmentioned")
    assert build_verse_context_block(store, "hello there") == ""


def test_story_world_context_lists_canon(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.add_entity("npc", "Nobody", "not pinned")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    ctx = build_story_world_context(store)
    assert "- Assgas Archie: Y11 windbag" in ctx
    assert "Nobody" not in ctx  # unpinned → not canon


def test_story_world_context_empty_when_no_canon(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Ghost", "unpinned")
    assert build_story_world_context(store) == ""


def test_pinned_entities_appear_in_prompt(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" in prompt
    assert "Assgas Archie: Y11 windbag" in prompt


def test_roster_omitted_when_none_pinned(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" not in prompt


def test_roster_respects_char_cap(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    for i in range(30):
        e = store.add_entity("npc", f"Lad{i:02d}", "x" * 40)
        store.apply_direct(
            op="set_pinned",
            payload={"entity_id": e, "pinned": True},
            source="operator",
            provenance="t",
        )
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=200)
    roster = prompt.split("Established characters in this world:")[1].split(VERSE_SCENE_MARKER)[0]
    assert len(roster) <= 260  # cap + the truncation marker line
    assert "(roster truncated)" in roster


def test_canon_first_scene_after(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    t = store.add_entity("npc", "Toby", "year 9")
    store.add_relation(h, t, "rival_of")
    out = build_verse_system_prompt(
        store,
        avatar_id,
        "be a year 8 boy",
        roster_max_chars=4000,
        message_text="did Harry and Toby fight?",
    )
    assert out.index("Established characters") < out.index(VERSE_SCENE_MARKER)
    assert "Harry" in out and "Toby" in out and "rival of" in out


def test_prefix_byte_identical_when_only_message_changes(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    a = build_verse_system_prompt(
        store, avatar_id, "p", roster_max_chars=4000, message_text="hi Harry"
    )
    b = build_verse_system_prompt(
        store, avatar_id, "p", roster_max_chars=4000, message_text="yo Toby"
    )
    assert a.split(VERSE_SCENE_MARKER)[0] == b.split(VERSE_SCENE_MARKER)[0]


def _pin(store, eid):
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": eid, "pinned": True},
        source="operator",
        provenance="t",
    )


def test_roster_lines_carry_entity_ids(tmp_path):
    """Every roster/cast line starts with '#<id>'. Without it verse_edit's
    set_attribute/update_entity (which take a numeric entity_id, and have no
    name lookup tool) cannot address anything that already exists — the model
    falls back to add_entity and duplicates the roster."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)

    block = build_verse_context_block(store, "what about Assgas Archie?", avatar_id=me)
    assert f"- #{archie} Assgas Archie: Y11 windbag" in block

    prompt = build_verse_system_prompt(store, me, "persona", message_text="Assgas Archie")
    assert f"- #{archie} Assgas Archie: Y11 windbag" in prompt


def test_roster_line_without_summary_still_carries_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    bare = store.add_entity("npc", "Nameless")
    _pin(store, bare)
    block = build_verse_context_block(store, "tell me about Nameless", avatar_id=me)
    assert f"- #{bare} Nameless" in block


def test_story_world_context_has_no_ids(tmp_path):
    """The storybook generator only needs names to stay true to canon; it never
    edits, so ids would be prose noise."""
    store = VerseStore(tmp_path, "#chan")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)
    out = build_story_world_context(store)
    assert "- Assgas Archie: Y11 windbag" in out
    assert f"#{archie}" not in out
