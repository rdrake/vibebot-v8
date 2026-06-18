from llm.verse.loom import validate_payload


def test_validate_payload_ok():
    assert validate_payload("add_entity", {"kind": "npc", "name": "Bob"}) is None


def test_validate_payload_missing_field():
    assert "name" in (validate_payload("add_entity", {"kind": "npc"}) or "")


def test_validate_payload_update_entity():
    assert validate_payload("update_entity", {"entity_id": 3, "summary": "x"}) is None
    assert validate_payload("update_entity", {"entity_id": "x", "summary": "y"}) is not None


def test_validate_payload_unknown_op():
    assert validate_payload("nope", {}) is not None
