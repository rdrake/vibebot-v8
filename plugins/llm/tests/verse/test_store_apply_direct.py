from llm.verse.store import VerseStore


def test_apply_direct_applies_and_audits(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    new_id = store.apply_direct(
        op="add_entity",
        payload={"kind": "npc", "name": "Archie", "summary": "stinky"},
        source="operator",
        provenance="@versedit",
    )
    assert store.get_entity(new_id).name == "Archie"
    with store.read_connection() as conn:
        row = conn.execute(
            "SELECT op, status, provenance FROM proposals ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    assert row == ("add_entity", "approved", "@versedit")
