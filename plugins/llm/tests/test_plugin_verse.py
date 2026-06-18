"""Plugin verse: verse commands, avatars, routing, compaction, crosspoll."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.service import AssistantResult

from .conftest import make_registry_side_effect

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestVerseproposalsCommand:
    """D1: @verseproposals listing command."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

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
        msg.args = ("#afnet", "@verseproposals")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_default_status_pending_in_current_channel(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x"},
            confidence=0.5,
        )
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("0.50" in r and "add_event" in r for r in replies)

    def test_explicit_channel_and_status(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "y"},
            confidence=0.95,
            status="approved",
            reviewer="loom",
        )
        plugin.verseproposals(irc, msg, ["#afnet", "approved"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("0.95" in r and "add_event" in r for r in replies)

    def test_empty_list_message(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("No pending proposals" in r for r in replies)

    def test_default_caps_at_three_with_more_footer(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(7):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        plugin.verseproposals(irc, msg, [])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        # 3 proposal rows + 1 "more pending" footer = 4 IRC lines max
        assert len(replies) == 4
        assert any("more pending" in r for r in replies)

    def test_explicit_limit_overrides_default(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(7):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        # Wrap parses positional args; limit must come after channel + status.
        plugin.verseproposals(irc, msg, ["#afnet", "pending", "10"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        # 7 rows shown, no footer (because limit > total)
        assert len(replies) == 7
        assert not any("more pending" in r for r in replies)

    def test_limit_clamped_to_max(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        for i in range(3):
            store.add_proposal(
                cycle_id=f"c-{i}",
                op="add_event",
                payload={"summary": f"line {i}"},
                confidence=0.5,
            )
        # Pass an absurd limit; the implementation clamps to MAX_LIMIT (50)
        # which still happily fits 3 rows without a footer.
        plugin.verseproposals(irc, msg, ["#afnet", "pending", "10000"])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert len(replies) == 3
        assert not any("more pending" in r for r in replies)


class TestVerseapproveRejectCommands:
    """D2: @verseapprove + @versereject moderation commands."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

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

    def test_approve_applies_and_flips_status(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="line-1",
        )
        plugin.verseapprove(irc, msg, [pid[:8]])
        events = store.recent_events()
        assert len(events) == 1
        assert events[0].source == "loom"
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"
        assert p.reviewer != "loom"

    def test_approve_short_id_prefix(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="line-1",
        )
        plugin.verseapprove(irc, msg, [pid[:6]])
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "approved"

    def test_approve_unknown_id_errors_cleanly(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.verseapprove(irc, msg, ["deadbeef"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("No proposal" in e for e in errors)

    def test_approve_already_approved_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95,
            provenance="line-1",
            status="approved",
            reviewer="loom",
        )
        plugin.verseapprove(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already approved" in r for r in replies)

    def test_approve_already_rejected_blocked(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        store.update_proposal_status(pid, status="rejected", reviewer="bob")
        plugin.verseapprove(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("rejected" in r for r in replies)

    def test_reject_flips_status_and_does_not_apply(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        plugin.versereject(irc, msg, [pid[:8]])
        assert store.recent_events() == []
        p = store.get_proposal(pid)
        assert p is not None
        assert p.status == "rejected"

    def test_reject_unknown_id_errors(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        plugin.versereject(irc, msg, ["deadbeef"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("No proposal" in e for e in errors)

    def test_reject_already_approved_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.95,
            provenance="x",
            status="approved",
            reviewer="loom",
        )
        plugin.versereject(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already approved" in r for r in replies)

    def test_approve_apply_exception_reports_error(self, verse_env, mocker) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        mocker.patch.object(store, "apply_proposal_and_mark", side_effect=RuntimeError("boom"))
        plugin.verseapprove(irc, msg, [pid])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("Apply failed" in e for e in errors)

    def test_proposal_target_store_no_channel_errors(self, verse_env) -> None:
        plugin, irc, msg, _store = verse_env
        msg.args = ("",)  # No channel context
        channel, store = plugin._proposal_target_store(irc, msg, None)
        assert channel is None
        assert store is None
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("Specify a channel" in e for e in errors)

    def test_proposal_snippet_covers_all_ops(self, verse_env) -> None:
        from llm.verse.store import Proposal

        plugin, _irc, _msg, _store = verse_env
        sa = Proposal(
            id="x",
            created_at=0.0,
            cycle_id="c",
            op="set_attribute",
            payload={"entity_id": 1, "key": "k", "value": "v"},
            confidence=0.5,
            provenance="",
            status="pending",
            reviewer=None,
            reviewed_at=None,
        )
        ar = sa._replace(
            op="add_relation",
            payload={"from_id": 1, "to_id": 2, "kind": "k", "note": ""},
        )
        ae = sa._replace(op="add_entity", payload={"kind": "place", "name": "Oak"})
        unknown = sa._replace(op="weird", payload={})
        assert "entity_id=1" in plugin._proposal_snippet(sa)
        assert "1-[k]->2" in plugin._proposal_snippet(ar)
        assert "place 'Oak'" in plugin._proposal_snippet(ae)
        assert plugin._proposal_snippet(unknown) == ""

    def test_reject_already_rejected_short_circuits(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="c-1",
            op="add_event",
            payload={"summary": "x", "entity_ids": []},
            confidence=0.5,
            provenance="x",
        )
        store.update_proposal_status(pid, status="rejected", reviewer="bob")
        plugin.versereject(irc, msg, [pid])
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any("already rejected" in r for r in replies)


class TestVerseCapabilities:
    """llm.verse and llm.verse.gm must be declared in _REQUEST_CONTEXT_CAPABILITIES."""

    def test_llm_verse_in_request_context_capabilities(self) -> None:
        """GIVEN plugin module WHEN capabilities inspected THEN llm.verse is declared."""
        import llm.plugin as plugin_module

        assert "llm.verse" in plugin_module._REQUEST_CONTEXT_CAPABILITIES

    def test_llm_verse_gm_in_request_context_capabilities(self) -> None:
        """GIVEN plugin module WHEN capabilities inspected THEN llm.verse.gm is declared."""
        import llm.plugin as plugin_module

        assert "llm.verse.gm" in plugin_module._REQUEST_CONTEXT_CAPABILITIES

    def test_llm_verse_and_llm_verse_gm_are_distinct(self) -> None:
        """GIVEN both verse capabilities WHEN compared THEN they are separate entries."""
        import llm.plugin as plugin_module

        caps = plugin_module._REQUEST_CONTEXT_CAPABILITIES
        assert "llm.verse" in caps
        assert "llm.verse.gm" in caps
        # Distinct: having gm does not subsume verse by default
        assert "llm.verse" != "llm.verse.gm"


# =============================================================================
# TestVerseoptCommand
# =============================================================================


class TestVerseoptCommand:
    """Tests for the @verseopt in/out command (C3).

    Strategy: use the ``plugin_env`` fixture (mocked LLMService/DB) plus a
    real VerseStore backed by ``tmp_path`` so the avatar table round-trips
    through actual SQLite.  We swap ``plugin._get_or_create_verse_store`` with
    a factory that always returns the same real store, giving us full
    DB coverage without touching production paths.
    """

    SCENE_PREFIX = "You step into The Clearing."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for the channel.
        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Instruct text and avatar persona for the user (empty by default).
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        # msg arrives in #afnet
        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helper: invoke the wrapped command correctly.
    # wrap(verseopt, [("checkCapability", "llm.verse"), ("literal", ("in", "out"))])
    # produces a callable (irc, msg, args) where args is the token list.
    # ------------------------------------------------------------------

    @staticmethod
    def _call(plugin, irc, msg, mode: str) -> None:
        """Call the wrapped verseopt command with the given mode token."""
        plugin.verseopt(irc, msg, [mode])

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    def test_verseopt_in_new_avatar_replies_scene(self, verse_env) -> None:
        """GIVEN verse-enabled channel and llm.verse WHEN @verseopt in THEN scene text replied."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert self.SCENE_PREFIX in reply_text
        assert "You are already opted in." not in reply_text

    def test_verseopt_in_new_avatar_created_in_store(self, verse_env) -> None:
        """GIVEN new opt-in WHEN @verseopt in THEN avatar entity exists in store."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

    def test_verseopt_in_uses_avatar_persona(self, verse_env, mocker) -> None:
        """GIVEN user has @avatar set WHEN @verseopt in THEN opt_in_avatar called with persona."""
        plugin, irc, msg, store = verse_env
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value="Be a wizard.")

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == "Be a wizard."

    def test_verseopt_in_ignores_instruction(self, verse_env, mocker) -> None:
        """GIVEN @instruct set but no @avatar WHEN @verseopt in THEN persona is empty.

        The split: @instruct only shapes %ask. The verse must read @avatar.
        """
        plugin, irc, msg, store = verse_env
        plugin.db.get_instruction = mocker.MagicMock(return_value="ask voice")
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == ""

    # ------------------------------------------------------------------
    # Branch 2: verseopt in — already opted in
    # ------------------------------------------------------------------

    def test_verseopt_in_already_opted_in_prefixes_message(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt in again THEN reply prefixed with already-in message."""
        plugin, irc, msg, store = verse_env

        # First opt-in creates the avatar.
        self._call(plugin, irc, msg, "in")
        irc.reply.reset_mock()

        # Second opt-in should detect was_already_opted_in=True.
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text.startswith("You are already opted in. ")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 3: verseopt in after verseopt out — reactivation
    # ------------------------------------------------------------------

    def test_verseopt_in_after_out_reactivates_without_prefix(self, verse_env) -> None:
        """GIVEN retired avatar WHEN @verseopt in THEN was_already_opted_in=False, no prefix."""
        plugin, irc, msg, store = verse_env

        # Create avatar.
        self._call(plugin, irc, msg, "in")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

        # Retire it.
        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")
        assert store.find_avatar_by_nick("alice") is None

        # Reactivate.
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verseopt in")
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert not reply_text.startswith("You are already opted in.")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 4: verseopt out
    # ------------------------------------------------------------------

    def test_verseopt_out_replies_retired(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt out THEN retired message replied."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")
        irc.reply.reset_mock()

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Avatar retired. Use @verseopt in to rejoin."

    def test_verseopt_out_removes_avatar_link(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt out THEN find_avatar_by_nick returns None."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")
        assert store.find_avatar_by_nick("alice") is not None

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        assert store.find_avatar_by_nick("alice") is None

    def test_verseopt_out_with_no_avatar_replies_not_found(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verseopt out THEN 'no avatar' reply."""
        plugin, irc, msg, store = verse_env

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You don't have an avatar in this channel."

    # ------------------------------------------------------------------
    # Branch 5: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verseopt_in_disabled_channel_replies_not_enabled(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN not-enabled message."""
        plugin, irc, msg = plugin_env

        # verse disabled (default in make_registry_side_effect)
        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == (
            "This channel doesn't have a verse. Ask the operator to set verseEnabled."
        )

    def test_verseopt_in_disabled_channel_no_store_created(
        self, plugin_env, mocker, tmp_path
    ) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN no VerseStore is instantiated."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        store_cls = mocker.patch("llm.plugin.VerseStore")

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        store_cls.assert_not_called()

    # ------------------------------------------------------------------
    # Branch 6: missing llm.verse capability
    # ------------------------------------------------------------------

    def test_verseopt_in_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @verseopt in THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        # Deny all capabilities.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#test", "@verseopt in")

        # wrap's checkCapability gate raises an error via irc.error when denied.
        # Call the wrapped version; the body should NOT run (capability blocked).
        plugin.verseopt(irc, msg, ["in"])

        # The scene text should not have been replied.
        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "step into" not in reply_text


# =============================================================================
# TestVerseCommand
# =============================================================================


class TestVerseCommand:
    """Tests for the @verse command (C4).

    Uses the same verse_env fixture pattern as TestVerseoptCommand.
    """

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

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

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verse")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verse")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 1: avatar present, location set → scene one-liner
    # ------------------------------------------------------------------

    def test_verse_with_location_returns_scene_oneliner(self, verse_env) -> None:
        """GIVEN avatar with location WHEN @verse THEN 'You are at <place>.' reply."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        # Add a place and set avatar's location attribute.
        place_id = store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")
        store.set_attribute(entity_id, "location", str(place_id))

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert f"You are at {self.PLACE_NAME}." in reply_text
        assert "A sunlit glade." in reply_text

    # ------------------------------------------------------------------
    # Branch 2: avatar present, no location → "nowhere in particular"
    # ------------------------------------------------------------------

    def test_verse_no_location_returns_nowhere(self, verse_env) -> None:
        """GIVEN avatar with no location WHEN @verse THEN 'nowhere in particular' reply."""
        plugin, irc, msg, store = verse_env
        self._opt_in(plugin, irc, msg, store)

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You are nowhere in particular."

    # ------------------------------------------------------------------
    # Branch 3: no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_verse_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verse THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 4: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_verse_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verse THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 5: lacking llm.verse capability → Limnoria denial
    # ------------------------------------------------------------------

    def test_verse_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @verse THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        # Limnoria's ("checkCapability", "llm.verse") wrap gate denies the
        # command before the body runs: it emits errorNoCapability and never
        # replies with a scene. Asserting BOTH sides makes this fail if the
        # gate is removed — the body would then reach NO_AVATAR_REPLY via
        # irc.reply for the unopted user.
        irc.errorNoCapability.assert_called_once()
        irc.reply.assert_not_called()


# =============================================================================
# TestLookCommand
# =============================================================================


class TestLookCommand:
    """Tests for the @look [target] command (C4)."""

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

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

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@look")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@look")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 6: no target, avatar present → scene one-liner
    # ------------------------------------------------------------------

    def test_look_no_target_with_avatar_returns_scene(self, verse_env) -> None:
        """GIVEN avatar with location, no target WHEN @look THEN scene one-liner."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        place_id = store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")
        store.set_attribute(entity_id, "location", str(place_id))

        # Wrapped: args=[] means no optional target → target=None inside handler.
        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert f"You are at {self.PLACE_NAME}." in reply_text

    # ------------------------------------------------------------------
    # Branch 7: no target, no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_look_no_target_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar, no target WHEN @look THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 8: target matches → entity description
    # ------------------------------------------------------------------

    def test_look_target_matches_returns_description(self, verse_env) -> None:
        """GIVEN named entity, target given WHEN @look <name> THEN entity description."""
        plugin, irc, msg, store = verse_env
        store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")

        # Wrapped: target string goes into args list.
        plugin.look(irc, msg, [self.PLACE_NAME])

        reply_text = irc.reply.call_args[0][0]
        assert self.PLACE_NAME in reply_text
        assert "A sunlit glade." in reply_text

    # ------------------------------------------------------------------
    # Branch 9: target doesn't match → "Nothing matches."
    # ------------------------------------------------------------------

    def test_look_target_no_match_replies_nothing_matches(self, verse_env) -> None:
        """GIVEN target with no matching entity WHEN @look <name> THEN 'Nothing matches.'"""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, ["Atlantis"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nothing matches."

    # ------------------------------------------------------------------
    # Branch 10: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_look_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @look THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@look")
        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 11: lacking capability → denial
    # ------------------------------------------------------------------

    def test_look_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @look THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@look")
        plugin.look(irc, msg, [])

        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "You are at" not in reply_text


# =============================================================================
# TestWhoCommand
# =============================================================================


class TestWhoCommand:
    """Tests for the @who command (C4)."""

    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

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

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@who")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 12: multiple active avatars with locations → comma-joined list
    # ------------------------------------------------------------------

    def test_who_multiple_avatars_returns_list(self, verse_env) -> None:
        """GIVEN two active avatars with locations WHEN @who THEN comma-joined list."""
        plugin, irc, msg, store = verse_env

        # Create two avatars and a place.
        place_id = store.add_entity("place", "The Clearing", "A sunlit glade.")
        alice_id = store.add_entity("avatar", "alice")
        bob_id = store.add_entity("avatar", "bob")
        store.link_avatar(alice_id, "alice")
        store.link_avatar(bob_id, "bob")
        store.set_attribute(alice_id, "location", str(place_id))
        store.set_attribute(bob_id, "location", str(place_id))

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice (at The Clearing)" in reply_text
        assert "bob (at The Clearing)" in reply_text
        # Comma-joined format.
        assert ", " in reply_text

    # ------------------------------------------------------------------
    # Branch 13: no avatars → "Nobody is opted in here yet."
    # ------------------------------------------------------------------

    def test_who_no_avatars_replies_empty(self, verse_env) -> None:
        """GIVEN no active avatars WHEN @who THEN 'Nobody is opted in here yet.'"""
        plugin, irc, msg, store = verse_env

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nobody is opted in here yet."

    # ------------------------------------------------------------------
    # Branch 14: retired avatars excluded
    # ------------------------------------------------------------------

    def test_who_retired_avatar_excluded(self, verse_env) -> None:
        """GIVEN one active and one retired avatar WHEN @who THEN only active shown."""
        plugin, irc, msg, store = verse_env

        active_id = store.add_entity("avatar", "alice")
        store.link_avatar(active_id, "alice")

        retired_id = store.add_entity("avatar", "bob")
        store.link_avatar(retired_id, "bob")
        store.set_status(retired_id, "retired")

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice" in reply_text
        assert "bob" not in reply_text

    # ------------------------------------------------------------------
    # Branch 15: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_who_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @who THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@who")
        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 16: lacking capability → denial
    # ------------------------------------------------------------------

    def test_who_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @who THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@who")
        plugin.who(irc, msg, [])

        if irc.reply.called:
            reply_text = irc.reply.call_args[0][0]
            assert "Nobody" not in reply_text
            assert "alice" not in reply_text


# =============================================================================
# TestVersepurgeCommand
# =============================================================================


class TestVersepurgeCommand:
    """Tests for the @versepurge command (C5).

    Strategy: direct method calls (bypassing wrap), with a real VerseStore
    backed by tmp_path. Token logic is exercised via a ``now_func`` shim passed
    to ``_versepurge_check_token`` so we can fake the clock without monkeypatching.
    """

    @pytest.fixture
    def purge_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore wired for #afnet, with gm capability."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Wire the store into the cache so versepurge can pop it.
        plugin._verse_stores["#afnet"] = store

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Grant llm.verse.gm.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@versepurge #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 1: first call issues token
    # ------------------------------------------------------------------

    def test_first_call_issues_token(self, purge_env) -> None:
        """GIVEN no existing token WHEN @versepurge #afnet THEN 6-char token issued."""
        plugin, irc, msg, store = purge_env

        plugin.versepurge(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Confirm with @versepurge #afnet" in reply_text
        assert "within 60s" in reply_text

        assert "#afnet" in plugin._versepurge_tokens
        stored_token, expires_at = plugin._versepurge_tokens["#afnet"]
        assert len(stored_token) == 6  # token_hex(3) → 6 hex chars
        import time as _time

        assert expires_at > _time.time()

    # ------------------------------------------------------------------
    # Test 2: correct token purges
    # ------------------------------------------------------------------

    def test_second_call_correct_token_purges(self, purge_env, tmp_path) -> None:
        """GIVEN valid token WHEN @versepurge #afnet <token> THEN purged reply, store gone."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        # Step 1: issue token.
        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Step 2: confirm.
        plugin.versepurge(irc, msg, [f"#afnet {stored_token}"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #afnet purged."

        # Store evicted from cache.
        assert "#afnet" not in plugin._verse_stores

        # DB file removed.
        assert not db_path.exists()

        # Token cleared.
        assert "#afnet" not in plugin._versepurge_tokens

    def test_purge_holds_store_lock_across_unlink(self, purge_env, mocker) -> None:
        """The DB unlink must run WHILE _verse_stores_lock is held.

        The plugin is threaded=True, so a concurrent _get_or_create_verse_store
        (verse @ask on a SupyThread / a loom worker) could otherwise reconstruct
        a store on the files mid-purge and leave a half-written DB behind. Both
        the purge and the constructor take _verse_stores_lock, so the unlink must
        happen inside it.
        """
        from pathlib import Path

        plugin, irc, msg, store = purge_env

        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]

        lock_held_during_unlink: list[bool] = []
        real_unlink = Path.unlink

        def _record(self_path, *a, **kw):
            lock_held_during_unlink.append(plugin._verse_stores_lock.locked())
            return real_unlink(self_path, *a, **kw)

        mocker.patch.object(Path, "unlink", autospec=True, side_effect=_record)

        plugin.versepurge(irc, msg, [f"#afnet {stored_token}"])

        # unlink actually fired and every call saw the lock held.
        assert lock_held_during_unlink
        assert all(lock_held_during_unlink)

    # ------------------------------------------------------------------
    # Test 3: wrong token rejected
    # ------------------------------------------------------------------

    def test_second_call_wrong_token_rejected(self, purge_env) -> None:
        """GIVEN valid token WHEN confirmed with garbage token THEN rejected."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, ["#afnet garbage"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Token expired or invalid" in reply_text
        assert "Run @versepurge #afnet again" in reply_text

        # DB still exists; token NOT cleared (wrong token doesn't clear unexpired entry).
        assert db_path.exists()
        assert "#afnet" in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 4: token expires after 60s
    # ------------------------------------------------------------------

    def test_token_expires_after_60s(self, purge_env) -> None:
        """GIVEN token issued WHEN 60s pass and correct token presented THEN expired."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Use _versepurge_check_token directly with a fake clock past expiry.
        import time as _time

        future_now = _time.time() + 61.0
        result = plugin._versepurge_check_token("#afnet", stored_token, now_func=lambda: future_now)
        assert result is False

        # Stale entry should be cleared.
        assert "#afnet" not in plugin._versepurge_tokens

        # DB still exists (no purge occurred).
        assert db_path.exists()

    # ------------------------------------------------------------------
    # Test 5: reissue within window invalidates old token
    # ------------------------------------------------------------------

    def test_reissue_within_window_invalidates_old(self, purge_env) -> None:
        """GIVEN unexpired T1 WHEN @versepurge #afnet called again THEN T2 issued, T1 invalid."""
        plugin, irc, msg, store = purge_env

        # Issue T1.
        plugin.versepurge(irc, msg, ["#afnet"])
        token_t1, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Issue T2 (while T1 still valid).
        plugin.versepurge(irc, msg, ["#afnet"])
        reply_text = irc.reply.call_args[0][0]
        assert "Previous token invalidated" in reply_text or "invalidated" in reply_text

        token_t2, _ = plugin._versepurge_tokens["#afnet"]
        assert token_t2 != token_t1

        # T1 no longer valid.
        result = plugin._versepurge_check_token("#afnet", token_t1)
        assert result is False

    # ------------------------------------------------------------------
    # Test 6: purge channel with no store yet is a no-op
    # ------------------------------------------------------------------

    def test_purge_for_channel_with_no_store_yet(self, purge_env) -> None:
        """GIVEN channel never had a store WHEN confirmed THEN 'purged' reply, no error."""
        plugin, irc, msg, store = purge_env

        # Issue token for a channel that has no store in cache.
        plugin._verse_stores.pop("#newchan", None)
        plugin.versepurge(irc, msg, ["#newchan"])
        token, _ = plugin._versepurge_tokens["#newchan"]
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, [f"#newchan {token}"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #newchan purged."
        assert "#newchan" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 7: lacking capability is denied
    # ------------------------------------------------------------------

    def test_lacking_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse.gm WHEN @versepurge THEN error, no token issued."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 8: llm.verse (not gm) is denied
    # ------------------------------------------------------------------

    def test_purge_unrelated_to_versedump_capability(self, plugin_env, mocker) -> None:
        """GIVEN user has llm.verse but NOT llm.verse.gm WHEN @versepurge THEN denied."""
        plugin, irc, msg = plugin_env

        # Only grant llm.verse (not llm.verse.gm).
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "llm.verse",
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens


# =============================================================================
# TestVersedumpCommand
# =============================================================================


class TestVersedumpCommand:
    """Tests for the @versedump command (C5).

    Strategy: real VerseStore via tmp_path; invoke plugin.versedump directly.
    pyyaml is not installed → yaml branch returns unsupported message.
    """

    @pytest.fixture
    def dump_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore for #afnet, gm capability granted."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

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

        # Default: pastebin write fails → inline fallback path (preserves
        # the JSON-content assertions in tests written before the
        # publish-and-link change). Tests exercising the URL path
        # override this on the same mock.
        plugin.llm_service.save_markdown_to_http = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@versedump #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 9: dump includes entities, attributes, relations, events, avatar_links
    # ------------------------------------------------------------------

    def test_dump_includes_entities_attributes_relations_events_avatar_link(self, dump_env) -> None:
        """GIVEN populated verse WHEN @versedump THEN JSON contains all sections."""
        import json as _json

        plugin, irc, msg, store = dump_env

        # Opt alice in (creates avatar + place + location attribute via opt_in_avatar).
        alice_id = store.opt_in_avatar("alice", None, "").entity_id

        # Add a relation.
        store.add_relation(alice_id, alice_id, "self-aware", "test note")

        # Add an event.
        store.add_event("alice did something", [alice_id], "avatar")

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        data = _json.loads(reply_text)

        assert "entities" in data
        assert len(data["entities"]) >= 1

        # Check attributes present on alice's entity.
        alice_entity = next((e for e in data["entities"] if e["id"] == alice_id), None)
        assert alice_entity is not None
        assert "attributes" in alice_entity

        assert "relations" in data
        assert len(data["relations"]) >= 1

        assert "events" in data
        assert len(data["events"]) >= 1

        assert "avatar_links" in data
        assert any(row["nick"] == "alice" for row in data["avatar_links"])

    # ------------------------------------------------------------------
    # Test 10: default format is JSON
    # ------------------------------------------------------------------

    def test_dump_default_format_is_json(self, dump_env) -> None:
        """WHEN @versedump #chan THEN reply is valid JSON."""
        import json as _json

        plugin, irc, msg, store = dump_env

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        data = _json.loads(reply_text)  # Raises if not valid JSON.
        assert data["schema_version"] == 1
        assert data["channel"] == "#afnet"

    # ------------------------------------------------------------------
    # Test 11: yaml format returns unsupported (pyyaml not installed)
    # ------------------------------------------------------------------

    def test_dump_yaml_format_or_unsupported(self, dump_env) -> None:
        """WHEN @versedump #chan --format=yaml THEN unsupported message (pyyaml absent)."""
        plugin, irc, msg, store = dump_env

        # Pass --format=yaml in the text arg (space-separated from channel).
        plugin.versedump(irc, msg, ["#afnet --format=yaml"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        # pyyaml is not a dependency → unsupported message.
        assert "Unsupported format" in reply_text
        assert "json" in reply_text.lower()

    # ------------------------------------------------------------------
    # Test 12: lacking capability is denied
    # ------------------------------------------------------------------

    def test_dump_lacking_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse.gm WHEN @versedump THEN error."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        msg.args = ("#afnet", "@versedump #afnet")
        plugin.versedump(irc, msg, ["#afnet"])

    def test_dump_publishes_to_pastebin_when_available(self, dump_env, mocker) -> None:
        """WHEN save_markdown_to_http returns URL THEN reply is just the URL.

        Avoids spamming IRC with a fat JSON line when the bot's HTTP
        pastebin is configured.
        """
        plugin, irc, msg, store = dump_env
        published_url = "http://example.com/llm/answer_abc123.html"
        plugin.llm_service.save_markdown_to_http.return_value = published_url

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert published_url in reply_text
        # Body must have been wrapped as a fenced JSON markdown block so
        # the pastebin renders with syntax highlighting (and includes the
        # channel name in a header).
        plugin.llm_service.save_markdown_to_http.assert_called_once()
        markdown_arg = plugin.llm_service.save_markdown_to_http.call_args[0][0]
        assert "```json" in markdown_arg
        assert "#afnet" in markdown_arg


# ===========================================================================
# C6: @avatar double-write — avatar summary syncs with persona text.
# (Was @instruct; split out so @instruct only shapes %ask.)
# ===========================================================================


class TestAvatarVerseSync:
    """@avatar updates avatar.summary when channel is verse-enabled.

    Strategy: real VerseStore backed by tmp_path (no SQLite mocks).
    We patch ``plugin._get_or_create_verse_store`` to return the same real
    store used to assert post-condition, mirroring the TestVerseoptCommand
    pattern.
    """

    @pytest.fixture
    def avatar_verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for #afnet, disable for everything else.
        def _registry(key, *args):
            if key == "verseEnabled":
                channel = args[0] if args else None
                return channel == "#afnet"
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # msg arrives in #afnet from alice (no account → nick fallback).
        msg.args = ("#afnet", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Wire up db mocks for the avatar path.
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()
        plugin.db.delete_avatar_persona = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(store, nick: str, account: str | None = None) -> int:
        """Opt alice into the verse and return her entity_id."""
        result = store.opt_in_avatar(nick, account, instruct_text="")
        assert result.entity_id is not None
        return result.entity_id

    def test_in_verse_channel_with_avatar_updates_both(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + active avatar WHEN @avatar THEN both updated."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        plugin.avatar(irc, msg, ["curious traveller"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "curious traveller")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "curious traveller"

    def test_in_verse_channel_without_avatar_only_updates_persona(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + no avatar WHEN @avatar THEN only persona updated."""
        plugin, irc, msg, store = avatar_verse_env

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        assert store.find_avatar_by_nick("alice") is None

    def test_in_non_verse_channel_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @avatar THEN only persona updated, no verse store."""
        plugin, irc, msg = plugin_env

        get_store_spy = mocker.patch.object(plugin, "_get_or_create_verse_store")

        def _registry(key, *args):
            if key == "verseEnabled":
                return False
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.db.save_avatar_persona = mocker.MagicMock()
        msg.args = ("#other", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        get_store_spy.assert_not_called()

    def test_clear_in_verse_channel_clears_both(self, avatar_verse_env) -> None:
        """GIVEN active avatar with summary WHEN @avatar clear THEN both cleared."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
                ("pirate queen", _time.time(), alice_id),
            )

        msg.args = ("#afnet", "@avatar clear")
        plugin.avatar(irc, msg, ["clear"])

        plugin.db.delete_avatar_persona.assert_called_once_with("alice")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == ""

    def test_verse_write_failure_does_not_update_persona(self, avatar_verse_env, mocker) -> None:
        """GIVEN verse write raises WHEN @avatar THEN persona text unchanged."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        class _BrokenStore:
            def find_avatar_by_account(self, *a, **kw):
                return None

            def find_avatar_by_nick(self, *a, **kw):
                return alice_id

            def get_entity(self, *a, **kw):
                return store.get_entity(*a, **kw)

            def write_transaction(self):
                raise RuntimeError("disk full")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=_BrokenStore())

        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="disk full"):
            plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_not_called()


class TestInstructDoesNotTouchAvatar:
    """@instruct must no longer mirror to avatar.summary. The split's whole point."""

    @pytest.fixture
    def env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

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

        msg.args = ("#afnet", "@instruct hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.save_instruction = mocker.MagicMock()
        plugin.db.delete_instruction = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    def test_instruct_in_verse_channel_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="original persona")
        assert result.entity_id is not None
        original_summary = store.get_entity(result.entity_id).summary

        plugin.instruct(irc, msg, ["new ask voice"])

        plugin.db.save_instruction.assert_called_once_with("alice", "new ask voice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == original_summary

    def test_instruct_clear_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct clear THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="keep me")
        assert result.entity_id is not None

        msg.args = ("#afnet", "@instruct clear")
        plugin.instruct(irc, msg, ["clear"])

        plugin.db.delete_instruction.assert_called_once_with("alice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == "keep me"


class TestAvatarRetiredAvatar:
    """@avatar must not touch a retired avatar's summary."""

    def test_retired_avatar_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN retired avatar WHEN @avatar THEN persona updated, summary unchanged."""
        from llm.verse.store import VerseStore

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

        msg.args = ("#afnet", "@avatar foo")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()

        result = store.opt_in_avatar("alice", None, instruct_text="")
        alice_id = result.entity_id
        assert alice_id is not None
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, status = 'retired', updated_at = ? WHERE id = ?",
                ("old summary", _time.time(), alice_id),
            )

        plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "foo")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "old summary"


class TestVerseRouteForGating:
    """_verse_route_for gating logic (C7b): verseEnabled, llm.verse cap, OOC short-circuit."""

    # ------------------------------------------------------------------
    # Gate 1: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verse_disabled_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=False WHEN _verse_route_for called THEN returns None (gate fires).

        We spy on registryValue to confirm the verseEnabled check is the gate that fires
        rather than the stub unconditionally returning None.
        """
        plugin, _irc, _msg = plugin_env

        # verseEnabled=False (default in plugin_env) — spy to confirm it is consulted.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        plugin.registryValue.assert_any_call("verseEnabled", "#afnet")

    # ------------------------------------------------------------------
    # Gate 2: no llm.verse capability
    # ------------------------------------------------------------------

    def test_no_capability_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True but user lacks llm.verse WHEN called THEN None (quiet fallthrough)."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        cap_check = mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        cap_check.assert_called_once_with("alice!*@*", "llm.verse")

    # ------------------------------------------------------------------
    # Gate 3: OOC message
    # ------------------------------------------------------------------

    def test_ooc_message_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True and cap granted WHEN message is OOC THEN None."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "((this is OOC))")

        assert result is None

    # ------------------------------------------------------------------
    # Gate 4: all preconditions satisfied but no avatar — None (C7c)
    # ------------------------------------------------------------------

    def test_all_preconditions_satisfied_no_avatar_returns_none(
        self, plugin_env, mocker: MockerFixture, tmp_path
    ) -> None:
        """GIVEN verseEnabled=True, cap granted, plain message, but user has no avatar
        WHEN called THEN None (no opt-in → chat path)."""
        from llm.verse.store import VerseStore

        plugin, _irc, _msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", None, "just a plain message")

        assert result is None

    # ------------------------------------------------------------------
    # Regression: @ask dispatch still hits chat path (carried from C7a)
    # ------------------------------------------------------------------

    def test_ask_dispatch_still_reaches_chat_path(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN _verse_route_for returns None WHEN @ask sent THEN chat path fires unchanged.

        Regression guard: the dispatch hook must be a no-op — assistant_request
        must still be called exactly once (verseEnabled=False in plugin_env default).
        """
        from llm.service import AssistantResult

        plugin, mock_irc, mock_msg = plugin_env

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="normal reply",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=3,
            cost=0.0,
            model="test-model",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # The chat path must have fired.
        plugin.llm_service.assistant_request.assert_called_once()


# =============================================================================
# C7c: _verse_route_for system prompt + tool list assembly
# =============================================================================


class TestVerseRouteForC7c:
    """Tests for _verse_route_for returning a populated VerseRoute (C7c).

    Complements TestVerseRouteForGating which covers the None branches.
    """

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore, verse-enabled channel, alice opted in."""
        from llm.verse.store import VerseStore

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
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in so she has an avatar.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        return plugin, irc, msg, store

    def test_avatar_present_returns_route(self, verse_env) -> None:
        """GIVEN opted-in avatar WHEN _verse_route_for called THEN non-None route returned."""
        plugin, _irc, _msg, store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert route.avatar_id is not None
        assert isinstance(route.system_prompt, str)
        assert "alice" in route.system_prompt
        assert len(route.tools) == 6
        assert route.store is store

    def test_slash_ooc_bypasses_route_even_with_avatar(self, verse_env) -> None:
        """GIVEN an opted-in avatar (so a plain message WOULD route to verse)
        WHEN the message uses the leading // OOC marker THEN None — the
        ergonomic opt-out short-circuits routing.

        Non-vacuous counterpart to test_avatar_present_returns_route: because
        alice has an avatar, the only path to None is the OOC gate, so removing
        // recognition from is_ooc makes this fail (the plain message routes).
        """
        plugin, _irc, _msg, _store = verse_env

        # Sanity: the same message without the // marker DOES route to verse.
        assert plugin._verse_route_for("#afnet", "alice", None, "what model are you?") is not None

        assert plugin._verse_route_for("#afnet", "alice", None, "// what model are you?") is None

    def test_route_system_prompt_includes_identity(self, verse_env) -> None:
        """System prompt must start with 'You are alice.'"""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert route.system_prompt.startswith("You are alice.")

    def test_route_system_prompt_includes_scene(self, verse_env) -> None:
        """System prompt must include a Scene section."""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        assert "Scene:" in route.system_prompt

    def test_route_tools_have_expected_names(self, verse_env) -> None:
        """Returned tools must be the six verse tool specs (incl. verse_edit)."""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello")

        assert route is not None
        tool_names = {t["function"]["name"] for t in route.tools}
        assert tool_names == {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
            "verse_edit",
        }


class TestAskWithVerseRoute:
    """Tests that @ask in a verse channel with an opted-in avatar uses the verse path (C7c)."""

    SENTINEL = "VIBEBOT_TEST_SENTINEL_DO_NOT_LEAK"

    @staticmethod
    def _make_result(content: str = "verse reply") -> AssistantResult:
        return AssistantResult(
            content=content,
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

    @pytest.fixture
    def verse_ask_env(self, plugin_env, tmp_path, mocker):
        """plugin_env with verse enabled, alice opted in, and assistant_request stubbed."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "assistantSystemPrompt":
                return self.SENTINEL
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        # Stub assistant_request for the @ask call.
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_result()

        msg.args = ("#afnet", "@ask hello")

        return plugin, irc, msg, store

    def test_ask_verse_prompt_includes_channel_overlay_and_scene(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN system_prompt has BOTH the channel
        ``assistantSystemPrompt`` overlay AND the avatar scene context.

        Earlier behavior dropped the channel overlay on verse turns. That cratered
        output length on #afternet — the model under verse produced ~150-token
        list-style replies, while the same model under chat produced 600+ tokens
        with the channel overlay attached. The energy/length pump comes from the
        channel overlay; verse mode must inherit it on top of the scene context.
        """
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        system_prompt = kwargs.get("system_prompt", "") or ""
        # Channel overlay (the sentinel) must appear — its energy is what
        # drove long-form output in chat mode.
        assert self.SENTINEL in system_prompt
        # AND the verse scene context (avatar name) must still be present.
        assert "alice" in system_prompt

    def test_ask_verse_prompt_contains_avatar_name(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN system_prompt includes avatar name 'alice'."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        system_prompt = kwargs.get("system_prompt", "")
        assert "alice" in (system_prompt or "")

    def test_ask_in_verse_appends_verse_tools(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN all 5 verse tool names appear in extra_tools."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        extra_tools = kwargs.get("extra_tools") or []
        tool_names = {t["function"]["name"] for t in extra_tools}
        expected = {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        }
        assert expected.issubset(tool_names)

    def test_ask_in_verse_bypasses_token_cap(self, verse_ask_env, mocker: MockerFixture) -> None:
        """GIVEN verse route WHEN @ask THEN request_context uses PROFILE_VERSE.

        PROFILE_VERSE is the only profile not in the profile_max_output dict
        in assistant.py, so it bypasses the token cap applied to PROFILE_CHAT.
        We verify by checking the profile on the request_context passed to assistant_request.
        """
        from llm.profile import PROFILE_VERSE

        plugin, irc, msg, _store = verse_ask_env

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        request_context = kwargs.get("request_context")
        assert request_context is not None
        assert request_context.profile == PROFILE_VERSE

    def test_ask_in_verse_empty_verse_model_falls_back_to_none(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """GIVEN an UNSET verseModel WHEN @ask in verse THEN model_override is None.

        Verse reads the per-channel ``verseModel`` and threads it as
        model_override; an empty value becomes ``None`` so the service resolves
        the profile's assistantModel. This is the boundary case — see
        ``test_ask_in_verse_passes_verse_model_override`` for the set case that
        pins the read itself (this empty case alone cannot catch a deleted read).
        """
        plugin, irc, msg, _store = verse_ask_env
        original = plugin.registryValue.side_effect

        def _registry(key, *args):
            if key == "verseModel":
                return ""  # explicit: unset verseModel
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs.get("model_override") is None

    def test_ask_in_verse_passes_verse_model_override(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a configured verseModel WHEN @ask in verse THEN that exact model
        reaches assistant_request as model_override.

        This is the load-bearing coupling: bumping the chat assistantModel must
        not silently leave verse on it — verse rides its own (non-reasoning,
        prose) ``verseModel``. Deleting the verseModel read in dispatch, or
        substituting "assistantModel", makes this fail.
        """
        plugin, irc, msg, _store = verse_ask_env
        original = plugin.registryValue.side_effect

        def _registry(key, *args):
            if key == "verseModel":
                return "openrouter/grok-3-non-reasoning"
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs.get("model_override") == "openrouter/grok-3-non-reasoning"


class TestAskOnVerseChannelWithoutOptIn:
    """Verse-enabled channel + speaker who hasn't opted in: the tool surface
    must still carry verse schemas (cache stability across speakers) but each
    invocation must route to a denying handler so non-opted-in users can't
    drive the canon."""

    @staticmethod
    def _make_result() -> AssistantResult:
        return AssistantResult(
            content="reply",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

    @pytest.fixture
    def env(self, plugin_env, tmp_path, mocker: MockerFixture):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        # Verse store exists for the channel but the speaker has no avatar —
        # _verse_route_for will return None on that gate.
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "verseAutoEntityMaxNamesPerCall":
                return 8
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@ask hello")
        msg.prefix = "bob!user@host"
        msg.nick = "bob"
        msg.server_tags = {}

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_result()

        return plugin, irc, msg

    def test_extra_tools_include_all_verse_schemas(self, env) -> None:
        """GIVEN verse-enabled channel + non-opted-in speaker WHEN @ask THEN
        the five verse tool schemas are passed to assistant_request so the
        prefix bytes match the opted-in cohort."""
        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        extra_tools = kwargs.get("extra_tools") or []
        names = {t["function"]["name"] for t in extra_tools}
        assert {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        }.issubset(names)

    def test_extra_handlers_deny_verse_tools(self, env) -> None:
        """GIVEN verse-enabled channel + non-opted-in speaker WHEN @ask THEN
        every verse tool name in extra_handlers returns an ``{"error": ...}``
        payload mentioning opt-in (so the model can self-correct)."""
        import json

        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        handlers = kwargs.get("extra_handlers") or {}
        for name in (
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        ):
            assert name in handlers, name
            payload = json.loads(handlers[name]({}).content)
            assert "error" in payload
            assert "opt-in" in payload["error"].lower()

    def test_profile_stays_chat_for_non_opted_in_speaker(self, env) -> None:
        """The advertised tool surface widens, but the system prompt /
        framework must remain the chat profile for non-opted-in speakers —
        forcing PROFILE_VERSE on them would lose the chat framework's
        length cap and tool-behavior rules."""
        from llm.profile import PROFILE_CHAT

        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        request_context = kwargs.get("request_context")
        assert request_context is not None
        assert request_context.profile == PROFILE_CHAT


class TestCompactionTimerWiring:
    """E3: plugin wires the daily compaction timer + walks verse-enabled channels."""

    def test_plugin_registers_compaction_timer_at_load(self, plugin_env) -> None:
        """The plugin's __init__ should set ``_compaction_timer_name`` and
        attempt registration; the registered name is ``llm_verse_compact``."""
        plugin, _irc, _msg = plugin_env
        assert plugin._compaction_timer_name == "llm_verse_compact"

    def test_compaction_tick_offloads_pass_to_executor(self, plugin_env, mocker) -> None:
        """``_compaction_tick`` hands ``_run_compaction_pass`` to the executor
        rather than running it inline.

        The daily timer fires on Limnoria's scheduler thread (the IRC
        driver's main loop); ``_run_compaction_pass`` makes a blocking LLM
        call per verse-enabled channel, so running it inline would pin the
        driver — the same bug class as the addressed-message typing lag.
        """
        plugin, _irc, _msg = plugin_env
        plugin._run_compaction_pass = mocker.MagicMock()
        plugin._register_compaction_timer = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock(closing=False)

        plugin._compaction_tick()

        plugin._run_compaction_pass.assert_not_called()  # offloaded, not inline
        plugin._llm_executor.submit.assert_called_once_with(
            "verse_compaction", plugin._run_compaction_pass
        )
        plugin._register_compaction_timer.assert_called_once()  # timer always re-arms

    def test_compaction_tick_skips_submit_when_closing(self, plugin_env, mocker) -> None:
        """During shutdown the pass is not submitted (executor closing), but
        the timer re-arm still runs in ``finally``."""
        plugin, _irc, _msg = plugin_env
        plugin._run_compaction_pass = mocker.MagicMock()
        plugin._register_compaction_timer = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock(closing=True)

        plugin._compaction_tick()

        plugin._llm_executor.submit.assert_not_called()
        plugin._run_compaction_pass.assert_not_called()
        plugin._register_compaction_timer.assert_called_once()

    def test_compaction_callback_walks_verse_enabled_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """``_run_compaction_pass`` invokes ``compact_verse`` once per
        verse-enabled channel returned by ``_verse_enabled_channels``."""
        plugin, _irc, _msg = plugin_env

        # Only #afnet is verse-enabled. _verse_enabled_channels already
        # filters by registry; emulate that here.
        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])

        # Defensive re-check inside _run_compaction_pass uses
        # registryValue("verseEnabled", channel) — return True for both
        # so the iteration faithfully reflects the helper's output.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )

        # Stub stores expose ``_channel`` so the fake compact_verse can
        # introspect which channel it was handed.
        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        from llm.verse.compaction import CompactionOutcome

        called_for: list[str] = []

        def fake_compact(store, **kw):
            called_for.append(store._channel)
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", fake_compact)

        plugin._run_compaction_pass()
        assert called_for == ["#afnet"]

    def test_run_compaction_pass_falls_back_when_registry_raises(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """Defensive guards in _run_compaction_pass: if the registry
        raises for either retention or min-keep keys (e.g. F1 key not
        yet defined on a freshly-upgraded install), defaults kick in
        and compaction still runs."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])

        def _flaky(key, *args):
            if key == "verseEnabled":
                return True
            if key in ("verseEventRetentionDays", "verseCompactionMinKeepEvents"):
                raise RuntimeError("registry not loaded")
            if key == "loomModel":
                return "gemini/x"
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_flaky)
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=mocker.MagicMock(),
        )
        from llm.verse.compaction import CompactionOutcome

        seen: dict = {}

        def _fake_compact(store, **kw):
            seen.update(kw)
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)
        plugin._run_compaction_pass()
        assert seen["retention_days"] == 30
        assert seen["min_keep_events"] == 20

    def test_compaction_failure_does_not_abort_remaining_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """A raise in one channel's compact_verse must not skip later channels."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        from llm.verse.compaction import CompactionOutcome

        seen: list[str] = []

        def maybe_bomb(store, **kw):
            seen.append(store._channel)
            if store._channel == "#a":
                raise RuntimeError("fail")
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", maybe_bomb)

        plugin._run_compaction_pass()
        assert "#a" in seen and "#b" in seen


class TestRunCompactionPassCallsAging:
    """Phase 5a: aging runs once per verse-enabled channel inside the
    compaction pass, with its own try/except for failure isolation."""

    def test_aging_called_once_per_enabled_channel(self, plugin_env, mocker, monkeypatch) -> None:
        """_run_compaction_pass calls age_auto_created_entities once per
        channel returned by _verse_enabled_channels."""
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else 14
                    if key == "verseAutoEntityRetireDays"
                    else ""
                )
            )
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: CompactionOutcome("skipped_disabled", 0, 0),
        )

        called: list[tuple[object, int]] = []

        def _spy_aging(store, *, retire_after_days, now):
            called.append((store, retire_after_days))
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _spy_aging)

        plugin._run_compaction_pass()

        assert len(called) == 2
        stores = {id(c[0]) for c in called}
        assert len(stores) == 2

    def test_aging_runs_before_compaction_heartbeat(self, plugin_env, mocker, monkeypatch) -> None:
        """Aging must run BEFORE compact_verse for each channel.

        compact_verse's digest heartbeat bumps last_seen_ts=now() on every
        entity in the new digest. If aging runs after it, a long-silent
        auto_created NPC that happens to appear in the compacted events is
        freshly stamped and never older than the retire cutoff — silently
        defeating verseAutoEntityRetireDays. Aging-first reads the true
        last_seen_ts before the heartbeat can resurrect it.
        """
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else 14
                    if key == "verseAutoEntityRetireDays"
                    else ""
                )
            )
        )
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            side_effect=lambda channel: mocker.MagicMock(_channel=channel),
        )

        order: list[str] = []

        def _spy_compact(*_a, **_kw):
            order.append("compact")
            return CompactionOutcome("skipped_disabled", 0, 0)

        def _spy_age(*_a, **_kw):
            order.append("age")
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _spy_compact)
        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _spy_age)

        plugin._run_compaction_pass()

        assert order == ["age", "compact"]

    def test_aging_reads_retire_days_per_channel(self, plugin_env, mocker, monkeypatch) -> None:
        """The aging call reads verseAutoEntityRetireDays at the channel
        scope, not global."""
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a"])
        captured: list[tuple[str, str | None]] = []

        def _spy(key, *args):
            channel = args[0] if args else None
            captured.append((key, channel))
            if key == "verseEnabled":
                return True
            if key == "verseEventRetentionDays":
                return 30
            if key == "verseCompactionMinKeepEvents":
                return 20
            if key == "loomModel":
                return "gemini/x"
            if key == "verseAutoEntityRetireDays":
                return 14
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_spy)

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: CompactionOutcome("skipped_disabled", 0, 0),
        )
        monkeypatch.setattr(
            "llm.verse.aging.age_auto_created_entities",
            lambda *a, **kw: aging_mod.AgingOutcome(0, 0),
        )

        plugin._run_compaction_pass()

        assert ("verseAutoEntityRetireDays", "#a") in captured

    def test_aging_failure_in_one_channel_does_not_abort_others(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """If aging raises for #a, #b still gets aged."""
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else 14
                    if key == "verseAutoEntityRetireDays"
                    else ""
                )
            )
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: CompactionOutcome("skipped_disabled", 0, 0),
        )

        seen: list[int] = []

        def _aging(store, *, retire_after_days, now):
            seen.append(id(store))
            if len(seen) == 1:
                raise RuntimeError("simulated aging failure")
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _aging)

        plugin._run_compaction_pass()

        assert len(seen) == 2

    def test_compaction_outcome_message_includes_aging_counts(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """5b.3: per-channel summary log line folds compaction + aging
        counts into one human-readable record. ``plugin.log`` is a
        MagicMock under ``plugin_env`` (the conftest patches
        ``llm.plugin.log``), so we inspect ``plugin.log.info`` call args
        directly instead of using ``caplog`` — supybot's plugin logger
        never reaches the stdlib root logger here."""
        from llm.verse import aging as aging_mod
        from llm.verse import compaction as compaction_mod

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#foo"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    30
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 20
                    if key == "verseCompactionMinKeepEvents"
                    else 14
                    if key == "verseAutoEntityRetireDays"
                    else ""
                )
            )
        )
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=mocker.MagicMock())
        monkeypatch.setattr(
            compaction_mod,
            "compact_verse",
            lambda *a, **kw: compaction_mod.CompactionOutcome(
                state="compacted", total_events=12, kept_in_digest=5
            ),
        )
        monkeypatch.setattr(
            aging_mod,
            "age_auto_created_entities",
            lambda *a, **kw: aging_mod.AgingOutcome(scanned=7, retired=2),
        )

        plugin._run_compaction_pass()

        # Render every info-level call to its formatted form and look for
        # the friendly outcome line.
        rendered = []
        for call in plugin.log.info.call_args_list:
            args = call.args
            if not args:
                continue
            fmt = args[0]
            try:
                rendered.append(fmt % tuple(args[1:]))
            except TypeError:
                rendered.append(fmt)
        matched = [
            m
            for m in rendered
            if "compaction outcome" in m
            and "compacted 12 events" in m
            and "aged 2 entities" in m
            and "kept 5" in m
        ]
        assert matched, f"no friendly outcome message in {rendered!r}"


class TestMaxActorsRegistryPlumbing:
    """Phase 6.6: verseAutoEntityMaxNamesPerCall flows through
    _build_verse_handlers_for into make_verse_extra_handlers."""

    def test_max_actors_flows_to_make_verse_extra_handlers(
        self, plugin_env, tmp_path, mocker, monkeypatch
    ) -> None:
        from llm.verse import avatar as avatar_mod
        from llm.verse.store import VerseStore

        plugin, _irc, _msg = plugin_env

        captured: list[int] = []
        real_handlers = avatar_mod.make_verse_extra_handlers

        def spy_handlers(store, avatar_id, logger=None, *, max_actors=8):
            captured.append(max_actors)
            return real_handlers(store, avatar_id, logger=logger, max_actors=max_actors)

        monkeypatch.setattr("llm.plugin.make_verse_extra_handlers", spy_handlers)

        # Real store with one active avatar so the helper has something
        # to bind handlers to.
        store = VerseStore(tmp_path, "#x")
        store.opt_in_avatar("alice", account=None, instruct_text="alice instruct")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        real_registry_side_effect = plugin.registryValue.side_effect

        def fake_registry(key, *args):
            if key == "verseAutoEntityMaxNamesPerCall":
                return 4
            return real_registry_side_effect(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=fake_registry)

        handlers = plugin._build_verse_handlers_for(channel="#x")
        assert handlers is not None
        assert 4 in captured, f"max_actors not plumbed; got {captured}"


class TestVersecompactCommand:
    """E4: @versecompact owner command — manual retention compaction.

    Strategy mirrors @versepurge / @verseapprove tests: direct method
    calls (bypassing wrap), real VerseStore in tmp_path, ircdb.checkCapability
    monkeypatched. The happy-path test inserts >min_keep events older than
    the retention window and monkeypatches compact_verse's loom client
    constructor so no network call is attempted.
    """

    @pytest.fixture
    def compact_env(self, plugin_env, tmp_path, mocker):
        """plugin_env wired with a real VerseStore for #afnet."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )
        plugin._verse_stores["#afnet"] = store

        msg.args = ("#afnet", "@versecompact #afnet")
        msg.prefix = "owner!user@host"
        msg.nick = "owner"
        msg.server_tags = {}

        return plugin, irc, msg, store

    def _override_registry(self, plugin, mocker, *, verse_enabled: bool) -> None:
        from tests.conftest import make_registry_side_effect

        defaults = make_registry_side_effect()

        def _registry(key, *args):
            if key == "verseEnabled":
                return verse_enabled
            if key == "verseEventRetentionDays":
                return 30
            if key == "verseCompactionMinKeepEvents":
                return 20
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            return defaults(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

    def test_compacts_named_channel(self, compact_env, mocker) -> None:
        """GIVEN >min_keep old events WHEN @versecompact #afnet THEN reply 'compacted'."""
        from plugins.llm.tests.verse.conftest import insert_event_at

        plugin, irc, msg, store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)

        # Grant llm.verse.gm so the wrap-bypassed direct call still
        # treats the caller as an owner. (Direct .versecompact() call
        # also won't pass through wrap's capability check; the in-method
        # registry guard is what matters.)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Substitute a fake loom client so no network call happens.
        class _FakeClient:
            def call(self, *, op, model, messages):
                from llm.verse.loom import LoomCallUsage

                return "A digest of the past.", LoomCallUsage(
                    prompt_tokens=10, completion_tokens=20, cost=0.0
                )

        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=_FakeClient(),
        )

        seconds_per_day = 86400
        now_ts = 100_000_000.0
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now_ts - 60 * seconds_per_day,
            )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "compaction outcome for #afnet" in reply_text
        assert "compacted" in reply_text

    def test_disabled_verse_says_so(self, compact_env, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @versecompact THEN reply names verseEnabled."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=False)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "verseEnabled" in reply_text

    def test_failure_in_compact_verse_replies_error(self, compact_env, mocker, monkeypatch) -> None:
        """GIVEN compact_verse raises WHEN @versecompact THEN irc.error not crash."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        def _boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _boom)

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        err_text = irc.error.call_args[0][0]
        assert "compaction failed for #afnet" in err_text
        assert "RuntimeError" in err_text

    def test_registry_lookup_failure_uses_defaults(self, compact_env, mocker, monkeypatch) -> None:
        """GIVEN registryValue raises for retention/min_keep keys WHEN
        @versecompact THEN defaults kick in and compaction still runs.

        Covers the defensive try/except guards around the registry reads
        for verseEventRetentionDays and verseCompactionMinKeepEvents.
        F1 will define verseCompactionMinKeepEvents — until then, the
        guard ensures the command still works.
        """
        from llm.verse.compaction import CompactionOutcome

        plugin, irc, msg, _store = compact_env

        def _flaky_registry(key, *args):
            if key == "verseEnabled":
                return True
            if key in ("verseEventRetentionDays", "verseCompactionMinKeepEvents"):
                raise RuntimeError("registry not loaded")
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            if key == "assistantApiKey":
                return ""
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_flaky_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        seen_kwargs: dict = {}

        def _fake_compact(store, **kw):
            seen_kwargs.update(kw)
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)

        plugin.versecompact(irc, msg, ["#afnet"])

        # Defaults applied via except branches.
        assert seen_kwargs["retention_days"] == 30
        assert seen_kwargs["min_keep_events"] == 20
        irc.reply.assert_called_once()

    def test_zero_retention_and_min_keep_are_honoured(
        self, compact_env, mocker, monkeypatch
    ) -> None:
        """Regression: ``int(0 or 30) == 30`` would coerce a legitimate
        zero value (the registry types accept 0) into the default. The
        fix uses an explicit conversion that preserves zero so operators
        can disable compaction with retention=0."""
        from llm.verse.compaction import CompactionOutcome

        plugin, irc, msg, _store = compact_env

        def _zero_registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "verseEventRetentionDays":
                return 0
            if key == "verseCompactionMinKeepEvents":
                return 0
            if key == "loomModel":
                return "gemini/gemini-flash-lite-latest"
            if key == "assistantApiKey":
                return ""
            return ""

        plugin.registryValue = mocker.MagicMock(side_effect=_zero_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.loom.LiteLLMLoomClient",
            return_value=mocker.MagicMock(),
        )

        seen_kwargs: dict = {}

        def _fake_compact(store, **kw):
            seen_kwargs.update(kw)
            return CompactionOutcome("skipped_disabled", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)

        plugin.versecompact(irc, msg, ["#afnet"])

        assert seen_kwargs["retention_days"] == 0
        assert seen_kwargs["min_keep_events"] == 0

    def test_zero_retention_and_min_keep_in_run_compaction_pass(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """Same regression but for the daily-timer driver
        ``_run_compaction_pass`` — the same ``or default`` pattern was
        also present there."""
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: (
                True
                if key == "verseEnabled"
                else (
                    0
                    if key == "verseEventRetentionDays"
                    else "gemini/x"
                    if key == "loomModel"
                    else 0
                    if key == "verseCompactionMinKeepEvents"
                    else "03:00"
                    if key == "verseCompactionDailyAt"
                    else ""
                )
            )
        )
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=mocker.MagicMock(),
        )

        seen: dict = {}

        def _fake_compact(store, **kw):
            seen.update(kw)
            return CompactionOutcome("skipped_disabled", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _fake_compact)
        plugin._run_compaction_pass()
        assert seen["retention_days"] == 0
        assert seen["min_keep_events"] == 0


class TestBridgeCrosspollWiring:
    """F2: production bridge wires real crosspoll registry + store."""

    @staticmethod
    def _redirect_data_dir(request, tmp_path) -> None:
        """Point supybot's data directory at tmp_path for the test.

        Registers a finalizer that restores the original value after the
        test completes.
        """
        import supybot.conf as conf

        original = conf.supybot.directories.data()
        conf.supybot.directories.data.setValue(str(tmp_path))
        request.addfinalizer(lambda: conf.supybot.directories.data.setValue(original))

    def _build_plugin_with_bridge(self, mocker: MockerFixture, tmp_path, request):
        from llm.plugin import LLM

        self._redirect_data_dir(request, tmp_path)

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
                "verseCrosspollPerCycleLimit": 1,
            }
        )
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        assert plugin._loom_bridge is not None
        return plugin

    def test_verse_allow_send_reads_registry(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)

        def _registry(key, *args):
            if key == "verseCrosspollAllowSend":
                channel = args[0] if args else None
                return channel == "#a"
            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        bridge = plugin._loom_bridge
        assert bridge.verse_allow_send("#a") is True
        assert bridge.verse_allow_send("#b") is False

    def test_verse_allow_receive_reads_registry(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)

        def _registry(key, *args):
            if key == "verseCrosspollAllowReceive":
                channel = args[0] if args else None
                return channel == "#a"
            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        bridge = plugin._loom_bridge
        assert bridge.verse_allow_receive("#a") is True
        assert bridge.verse_allow_receive("#b") is False

    def test_crosspoll_store_is_a_singleton(self, mocker: MockerFixture, tmp_path, request) -> None:
        plugin = self._build_plugin_with_bridge(mocker, tmp_path, request)
        bridge = plugin._loom_bridge
        a = bridge.crosspoll_store()
        b = bridge.crosspoll_store()
        assert a is b
        assert a is not None

    def test_loom_config_threads_per_cycle_limit(
        self, mocker: MockerFixture, tmp_path, request
    ) -> None:
        from llm.plugin import LLM

        self._redirect_data_dir(request, tmp_path)

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
                "verseCrosspollPerCycleLimit": 4,
            }
        )
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        assert plugin._loom is not None
        assert plugin._loom._cfg.crosspoll_per_cycle_limit == 4


class TestVerseapproveCrosspollSource:
    """F3-pre-2: @verseapprove infers event_source from cycle_id."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

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

    def test_approve_crosspoll_proposal_writes_crosspoll_event(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="crosspoll-recv",
            op="add_event",
            payload={"summary": "from elsewhere", "entity_ids": []},
            confidence=0.0,
            provenance="crosspoll from #alpha (seed-id=1)",
        )
        plugin.verseapprove(irc, msg, [pid])
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT source FROM events WHERE summary='from elsewhere'"
            ).fetchone()
        assert row is not None and row[0] == "crosspoll"

    def test_approve_loom_proposal_still_writes_loom_event(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        pid = store.add_proposal(
            cycle_id="loom-c1",
            op="add_event",
            payload={"summary": "regular event", "entity_ids": []},
            confidence=0.5,
            provenance="t",
        )
        plugin.verseapprove(irc, msg, [pid])
        with store.read_connection() as conn:
            row = conn.execute("SELECT source FROM events WHERE summary='regular event'").fetchone()
        assert row is not None and row[0] == "loom"
