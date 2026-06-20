from llm.verse.avatar import VERSE_SCENE_MARKER, build_verse_system_prompt
from llm.verse.store import VerseStore


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
