import pytest
from llm.verse.store import VerseStore

VALID_SOURCES = {"operator", "loom", "llm", "crosspoll", "avatar"}


def _store(tmp_path):
    return VerseStore(tmp_path, "#chan")


def test_invalid_source_rejected(tmp_path):
    store = _store(tmp_path)
    with store.write_transaction() as conn, pytest.raises(ValueError, match="source"):
        store._apply_op_inline(
            conn, op="add_event", payload={"summary": "x", "entity_ids": []}, source="bogus"
        )


def test_destructive_op_blocked_for_non_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="loom")
    with store.write_transaction() as conn, pytest.raises(PermissionError):
        store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="llm")


def test_destructive_op_allowed_for_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="operator")
    with store.write_transaction() as conn:
        store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="operator")
    assert store.recent_events(limit=10) == []
