from llm.verse.avatar import build_verse_system_prompt
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
    roster = prompt.split("Established characters in this world:")[1]
    assert len(roster) <= 260  # cap + the truncation marker line
    assert "(roster truncated)" in roster
