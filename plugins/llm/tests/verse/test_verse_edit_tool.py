import json

import pytest
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


def test_verse_edit_apply_error_is_reported(tmp_path):
    """A constructive op that raises in apply_direct (e.g. relating to a
    nonexistent entity) returns status=error with the exception message,
    not an unhandled traceback."""
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_relation",
        payload={"from_id": 999, "to_id": 998, "kind": "knows"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "does not exist" in result["detail"]


def test_verse_edit_invalid_payload_rejected(tmp_path):
    """validate_payload rejects a malformed payload before apply_direct."""
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc"},  # missing name
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"


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


# ---------------------------------------------------------------------------
# Integration: the plugin-layer _verse_edit_handler binds the gate to the
# TRIGGERING user's IRC prefix. These tests assert the capability decides
# whether the store is mutated.
# ---------------------------------------------------------------------------


def _make_plugin(mocker):
    from llm.plugin import LLM

    mocker.patch.object(LLM, "__init__", lambda self, irc: None)
    return LLM.__new__(LLM)


def test_handler_unauthorized_user_mutates_nothing(tmp_path, mocker):
    """An invoking user WITHOUT llm.verse.edit gets a refusal and writes nothing."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
    msg = mocker.MagicMock()
    msg.prefix = "nobody!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="nobody")
    result = handler({"op": "add_entity", "payload": {"kind": "npc", "name": "Ghost"}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "refused"
    assert "error" in parsed  # loop counts this as a failure
    assert store.list_entities_by_kind("npc") == []


def test_handler_authorized_user_applies(tmp_path, mocker):
    """An invoking user WITH llm.verse.edit applies the edit to the store."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler({"op": "add_entity", "payload": {"kind": "npc", "name": "Archie"}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "ok"
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Archie"]


def test_handler_gate_checks_triggering_prefix(tmp_path, mocker):
    """The capability check is computed against the invoking msg.prefix."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    check = mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    plugin._verse_edit_handler(msg=msg, store=store, account="gm")

    check.assert_called_once_with("gm!user@host", "llm.verse.edit")


def test_handler_authorized_but_destructive_op_refused(tmp_path, mocker):
    """Even an authorized user cannot delete via verse_edit; store is untouched."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Doomed")
    ev_id = store.add_event("a thing happened", [eid], source="llm")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler({"op": "delete_event", "payload": {"event_id": ev_id}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "error"
    assert store.recent_events()  # event survives


def test_verse_edit_tool_spec_advertised():
    """make_verse_tool_specs always advertises verse_edit so the channel tool
    surface stays byte-stable; the gate lives in the handler, not the schema."""
    from llm.verse.avatar import make_verse_tool_specs

    names = {s["function"]["name"] for s in make_verse_tool_specs()}
    assert "verse_edit" in names


def test_verse_edit_covered_by_denial_handlers():
    """A non-opted-in speaker's verse_edit call lands on a denial handler."""
    from llm.verse.avatar import make_verse_denial_handlers, make_verse_tool_specs

    specs = make_verse_tool_specs()
    denials = make_verse_denial_handlers(specs)
    assert "verse_edit" in denials


@pytest.mark.parametrize("bad_args", [None, [], "scalar", {"op": "add_entity"}])
def test_handler_tolerates_malformed_args(tmp_path, mocker, bad_args):
    """Non-dict args / missing payload must not raise out of the turn."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler(bad_args)
    parsed = json.loads(result.content)
    assert parsed["status"] in {"ok", "error", "refused"}
