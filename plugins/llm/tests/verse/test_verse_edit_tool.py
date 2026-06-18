from llm.verse.avatar import dispatch_verse_edit
from llm.verse.store import VerseStore


def test_verse_edit_unauthorized_is_noop(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "X"},
        authorized=False,
        account="nobody",
    )
    assert result["status"] == "refused"
    assert store.list_entities_by_kind("npc") == []


def test_verse_edit_authorized_applies(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "Archie"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "ok"
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Archie"]


def test_verse_edit_rejects_destructive_op(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="delete_event",
        payload={"event_id": 1},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "op" in result["detail"]
