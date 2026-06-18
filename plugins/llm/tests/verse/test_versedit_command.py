import pytest
from llm.verse.store import VerseStore


def test_versedit_add_then_pin(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    new_id = store.apply_direct(
        op="add_entity",
        payload={"kind": "npc", "name": "Archie", "summary": "stinky"},
        source="operator",
        provenance="@versedit add",
    )
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": new_id, "pinned": True},
        source="operator",
        provenance="@versedit pin",
    )
    assert [e.name for e in store.list_pinned_entities()] == ["Archie"]


def test_versedit_event_and_delete(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    a = store.add_entity("npc", "Archie")
    ev = store.apply_direct(
        op="add_event",
        payload={"summary": "Archie parps", "entity_ids": [a]},
        source="operator",
        provenance="t",
    )
    assert store.recent_events(limit=5)[0].summary == "Archie parps"
    store.apply_direct(
        op="delete_event",
        payload={"event_id": ev},
        source="operator",
        provenance="t",
    )
    assert store.recent_events(limit=5) == []


def test_versedit_retire_clears_avatar_link(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    res = store.opt_in_avatar(nick="bob", account="bob!acct", instruct_text="")
    store.apply_direct(
        op="set_status",
        payload={"entity_id": res.entity_id, "status": "retired"},
        source="operator",
        provenance="t",
    )
    assert store.find_avatar_by_nick("bob") is None


class TestVerseditCommand:
    """Integration: drive @versedit through the wrapped Limnoria command."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        msg.args = ("#afnet", "")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_add_npc_replies_and_persists(self, verse_env):
        plugin, irc, msg, store = verse_env
        plugin.versedit(irc, msg, ["add", "npc", "Archie", "::", "stinky"])
        reply = irc.reply.call_args[0][0]
        assert "added npc" in reply
        assert "Archie" in reply
        assert store.active_name_exists("Archie")
        ent_id = store.resolve_ref("Archie")
        assert store.get_entity(ent_id).summary == "stinky"

    def test_unknown_verb_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["frobnicate", "stuff"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("unknown verb" in e for e in errors)
