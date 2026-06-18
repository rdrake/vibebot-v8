import pytest
from llm.verse.store import VerseStore


def test_resolve_ref_by_hash_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Bob")
    assert store.resolve_ref("#%d" % eid) == eid  # noqa: UP031


def test_resolve_ref_by_name(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Bob")
    assert store.resolve_ref("Bob") == eid


def test_resolve_ref_numeric_name_is_not_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "7")  # literally named "7"
    assert store.resolve_ref("7") == eid  # name, not id  # noqa: UP031
    assert store.resolve_ref("#%d" % eid) == eid  # explicit id form  # noqa: UP031


def test_resolve_ref_unknown_raises(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    with pytest.raises(LookupError):
        store.resolve_ref("ghost")
