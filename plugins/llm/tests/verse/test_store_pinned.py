from llm.verse.store import VerseStore


def test_list_pinned_returns_only_active_pinned(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    a = store.add_entity("npc", "Archie", "stinky")
    store.add_entity("npc", "Bob", "plain")
    store.apply_direct(
        op="set_pinned", payload={"entity_id": a, "pinned": True}, source="operator", provenance="t"
    )
    pinned = store.list_pinned_entities()
    assert [e.name for e in pinned] == ["Archie"]
    store.apply_direct(
        op="set_status",
        payload={"entity_id": a, "status": "retired"},
        source="operator",
        provenance="t",
    )
    assert store.list_pinned_entities() == []


def test_active_name_exists(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Archie")
    assert store.active_name_exists("archie") is True
    assert store.active_name_exists("nobody") is False
