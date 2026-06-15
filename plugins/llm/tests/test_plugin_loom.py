"""Plugin loom wiring: loom hook, bridge, and wiring."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .conftest import make_registry_side_effect

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class _FakeLoom:
    """Tiny test double for the Loom orchestrator."""

    def __init__(self) -> None:
        self.observed: list[tuple[str, str]] = []

    def observe_transcript(self, nick: str, text: str) -> None:
        self.observed.append((nick, text))


class TestDoPrivmsgLoomHook:
    """C3: doPrivmsg captures loom-channel chatter on the loom network."""

    def _build_loom_plugin(self, mocker: MockerFixture):
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.network = "afternet"
        mock_irc.nick = "vibebot"
        mock_irc.state = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        return plugin

    def _make_msg(
        self,
        mocker: MockerFixture,
        *,
        target: str,
        nick: str,
        text: str,
    ):
        msg = mocker.MagicMock()
        msg.prefix = f"{nick}!u@h"
        msg.nick = nick
        msg.args = (target, text)
        msg.time = time.time() + 100  # not ZNC playback
        msg.channel = target
        msg.server_tags = {}
        return msg

    def _make_irc(self, mocker: MockerFixture, *, network: str):
        irc = mocker.MagicMock()
        irc.network = network
        irc.nick = "vibebot"
        irc.state = mocker.MagicMock()
        irc.state.channels = {}
        return irc

    def test_doprivmsg_appends_loom_transcript(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="the bell")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == [("botB", "the bell")]

    def test_doprivmsg_ignores_other_networks(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="freenode")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="hi")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []

    def test_doprivmsg_filters_by_bot_allowlist_when_set(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ("botB",)
        irc = self._make_irc(mocker, network="afternet")
        plugin.doPrivmsg(irc, self._make_msg(mocker, target="#forest", nick="alice", text="hi"))
        plugin.doPrivmsg(irc, self._make_msg(mocker, target="#forest", nick="botB", text="hi"))
        assert plugin._loom.observed == [("botB", "hi")]

    def test_doprivmsg_does_not_capture_prefix_commands(self, mocker: MockerFixture) -> None:
        # @verseapprove or @versereject in the loom channel must NOT land
        # in the loom transcript — they're commands, not improv. The
        # supybot default for whenAddressedBy.chars() is empty in tests, so
        # we set it to the operator-configured "@" used in production.
        import supybot.conf as conf

        prefix_value = conf.supybot.reply.whenAddressedBy.chars
        original = prefix_value()
        prefix_value.setValue("@")
        try:
            plugin = self._build_loom_plugin(mocker)
            plugin._loom = _FakeLoom()
            plugin._loom_channel_cache = "#forest"
            plugin._loom_network_cache = "afternet"
            plugin._loom_bot_nicks_cache = ()
            irc = self._make_irc(mocker, network="afternet")
            msg = self._make_msg(
                mocker, target="#forest", nick="alice", text="@verseapprove abc123"
            )
            plugin.doPrivmsg(irc, msg)
            assert plugin._loom.observed == []
        finally:
            prefix_value.setValue(original)

    def test_doprivmsg_ignores_other_channels_on_loom_network(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#other", nick="botB", text="hi")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []

    def test_doprivmsg_skips_capture_when_flag_disabled(self, mocker: MockerFixture) -> None:
        plugin = self._build_loom_plugin(mocker)
        plugin._loom = _FakeLoom()
        plugin._loom_channel_cache = "#forest"
        plugin._loom_network_cache = "afternet"
        plugin._loom_bot_nicks_cache = ()
        plugin._loom_capture_transcript_cache = False
        irc = self._make_irc(mocker, network="afternet")
        msg = self._make_msg(mocker, target="#forest", nick="botB", text="hi there")
        plugin.doPrivmsg(irc, msg)
        assert plugin._loom.observed == []


class TestPluginLoomBridge:
    """Cover _PluginLoomBridge methods."""

    def _make_bridge(self, mocker: MockerFixture, tmp_path):
        from llm.plugin import LLM, _PluginLoomBridge

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        bridge = _PluginLoomBridge(plugin, "afternet", "#forest")
        bridge._verse_data_dir = tmp_path
        return plugin, bridge

    def test_list_candidate_channels_no_irc_returns_empty(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        mocker.patch("llm.plugin.world.getIrc", return_value=None)
        assert bridge.list_candidate_channels() == []

    def test_list_candidate_channels_intersects_enabled_joined_ondisk(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        VerseStore(tmp_path, "#forest")
        VerseStore(tmp_path, "#noverse")

        plugin._verse_enabled_channels = lambda: ["#forest", "#unjoined"]

        irc_stub = mocker.MagicMock()
        irc_stub.state.channels = {"#forest": object()}
        mocker.patch("llm.plugin.world.getIrc", return_value=irc_stub)

        out = bridge.list_candidate_channels()
        assert out == ["#forest"]

    def test_candidate_weight_combines_avatars_and_events(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        store.add_entity("avatar", "Forest")
        store.add_entity("avatar", "Owl")
        store.add_event("a chime", [], "loom")
        plugin._get_or_create_verse_store = lambda ch: store
        assert bridge.candidate_weight("#forest") == 2 * 2 + 1

    def test_snapshot_returns_versesnapshot(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.loom import VerseSnapshot
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        forest_id = store.add_entity("avatar", "Forest")
        grove_id = store.add_entity("place", "Grove")
        store.add_event("a chime", [], "loom")
        plugin._get_or_create_verse_store = lambda ch: store

        snap = bridge.snapshot("#forest")
        assert isinstance(snap, VerseSnapshot)
        assert snap.channel == "#forest"
        assert ("avatar", "Forest", forest_id) in snap.top_entities
        assert ("place", "Grove", grove_id) in snap.top_entities
        assert "a chime" in snap.recent_events

    def test_snapshot_excludes_crosspoll_events(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.store import VerseStore

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        verse_dir = tmp_path / "stores"
        verse_dir.mkdir()
        store = VerseStore(verse_dir, "#forest")
        store.add_event("regular", [], "loom")
        store.add_event("from elsewhere", [], "crosspoll")
        plugin._get_or_create_verse_store = lambda ch: store

        snap = bridge.snapshot("#forest")
        joined = "\n".join(snap.recent_events)
        assert "regular" in joined
        assert "from elsewhere" not in joined

    def test_post_to_loom_channel_returns_false_without_irc(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        mocker.patch("llm.plugin.world.getIrc", return_value=None)
        assert bridge.post_to_loom_channel("hello") is False

    def test_post_to_loom_channel_queues_when_irc_present(
        self, mocker: MockerFixture, tmp_path
    ) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        irc_stub = mocker.MagicMock()
        mocker.patch("llm.plugin.world.getIrc", return_value=irc_stub)
        assert bridge.post_to_loom_channel("ring") is True
        irc_stub.queueMsg.assert_called_once()

    def test_post_to_loom_channel_drops_when_closing(self, mocker: MockerFixture, tmp_path) -> None:
        plugin, bridge = self._make_bridge(mocker, tmp_path)
        plugin._llm_executor.shutdown()
        irc_stub = mocker.MagicMock()
        mocker.patch("llm.plugin.world.getIrc", return_value=irc_stub)
        assert bridge.post_to_loom_channel("ring") is False
        irc_stub.queueMsg.assert_not_called()

    def test_schedule_after_replaces_existing_event(self, mocker: MockerFixture, tmp_path) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        rm = mocker.patch("llm.plugin.schedule.removeEvent")
        add = mocker.patch("llm.plugin.schedule.addEvent")
        bridge.schedule_after(5.0, lambda: None, "n")
        rm.assert_called_with("n")
        add.assert_called_once()

    def test_submit_routes_to_executor(self, mocker: MockerFixture, tmp_path) -> None:
        plugin, bridge = self._make_bridge(mocker, tmp_path)
        plugin._llm_executor = mocker.MagicMock()
        bridge.submit("loom:seed", lambda: None)
        plugin._llm_executor.submit.assert_called_once()

    def test_now_returns_float(self, mocker: MockerFixture, tmp_path) -> None:
        _plugin, bridge = self._make_bridge(mocker, tmp_path)
        assert isinstance(bridge.now(), float)

    def test_store_for_delegates(self, mocker: MockerFixture, tmp_path) -> None:
        plugin, bridge = self._make_bridge(mocker, tmp_path)
        sentinel = object()
        plugin._get_or_create_verse_store = lambda ch: sentinel
        assert bridge.store_for("#x") is sentinel

    def test_log_usage_routes_to_db(self, mocker: MockerFixture, tmp_path) -> None:
        from llm.verse.loom import LoomCallUsage

        plugin, bridge = self._make_bridge(mocker, tmp_path)
        plugin.db = mocker.MagicMock()
        bridge.log_usage(
            channel="#forest",
            op="seed",
            model="gemini/x",
            usage=LoomCallUsage(7, 3, 0.0),
        )
        plugin.db.log_usage.assert_called_once()
        kwargs = plugin.db.log_usage.call_args.kwargs
        assert kwargs["nick"] == "loom"
        assert kwargs["command"] == "loom:seed"
        assert kwargs["prompt_tokens"] == 7


class TestLoomWiring:
    """C2: plugin wires Loom + bridge via _wire_loom_if_enabled()."""

    def _build(self, mocker: MockerFixture, overrides: dict[str, object]):
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        registry = make_registry_side_effect(overrides)
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch("llm.plugin.schedule.addEvent")
        return LLM(mock_irc)

    def test_loom_disabled_when_loom_channel_empty(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {"loomNetwork": "afternet", "loomChannel": ""})
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None

    def test_loom_disabled_when_loom_network_empty(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {"loomNetwork": "", "loomChannel": "#forest"})
        assert plugin._loom is None
        assert plugin._loom_network_cache is None

    def test_loom_disabled_when_loom_channel_not_a_channel_name(
        self, mocker: MockerFixture
    ) -> None:
        # Operators sometimes paste smart-quoted text from a doc instead
        # of typing a literal channel name. Without validation the bot
        # PRIVMSGs that string as a nick and the server returns 401.
        plugin = self._build(mocker, {"loomNetwork": "afternet", "loomChannel": "“”"})
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None

    def test_loom_wired_when_both_set(self, mocker: MockerFixture) -> None:
        plugin = self._build(
            mocker,
            {
                "loomNetwork": "afternet",
                "loomChannel": "#forest",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "botA, botB",
                "verseAutoApplyThreshold": 0.85,
            },
        )
        assert plugin._loom is not None
        assert plugin._loom_channel_cache == "#forest"
        assert plugin._loom_network_cache == "afternet"
        assert plugin._loom_bot_nicks_cache == ("botA", "botB")

    @staticmethod
    def _wiring_overrides(bot_nicks: str) -> dict[str, object]:
        return {
            "loomNetwork": "afternet",
            "loomChannel": "#forest",
            "loomModel": "gemini/x",
            "loomCycleInterval": 5,
            "loomVerseCooldown": 20,
            "loomBeatWindow": 90,
            "loomTranscriptMaxLines": 40,
            "loomTranscriptMaxChars": 8000,
            "loomBotNicks": bot_nicks,
            "verseAutoApplyThreshold": 0.85,
        }

    def test_wire_loom_warns_when_bot_nicks_empty(self, mocker: MockerFixture) -> None:
        """An empty loomBotNicks means the loom captures EVERY participant —
        humans included — into transcripts that drive verse canon. That is a
        consent hazard on a mixed channel, so wiring must emit a WARN. (The
        behaviour itself is unchanged: empty still captures all, which is
        correct for a bot-only venue.)
        """
        # Boot with loom disabled so construction itself warns nothing.
        plugin = self._build(mocker, {})
        plugin.log = mocker.MagicMock()
        side = make_registry_side_effect(self._wiring_overrides(""))
        plugin.registryValue = mocker.MagicMock(side_effect=side)

        plugin._wire_loom_if_enabled()

        assert plugin._loom is not None  # still wired — no behaviour change
        warned = " ".join(str(c) for c in plugin.log.warning.call_args_list).lower()
        assert "loombotnicks" in warned
        assert "human" in warned

    def test_wire_loom_does_not_warn_when_bot_nicks_set(self, mocker: MockerFixture) -> None:
        """With an explicit bot allowlist the capture set is bounded, so no
        human-capture WARN should fire."""
        plugin = self._build(mocker, {})
        plugin.log = mocker.MagicMock()
        side = make_registry_side_effect(self._wiring_overrides("botA, botB"))
        plugin.registryValue = mocker.MagicMock(side_effect=side)

        plugin._wire_loom_if_enabled()

        warned = " ".join(str(c) for c in plugin.log.warning.call_args_list).lower()
        assert "human" not in warned

    def test_rewire_after_disabling_clears_caches(self, mocker: MockerFixture) -> None:
        plugin = self._build(
            mocker,
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
            },
        )
        assert plugin._loom is not None
        # Now flip loomChannel to empty and re-wire.
        new_side = make_registry_side_effect({"loomNetwork": "afternet", "loomChannel": ""})
        plugin.registryValue = mocker.MagicMock(side_effect=new_side)
        plugin._wire_loom_if_enabled()
        assert plugin._loom is None
        assert plugin._loom_channel_cache is None
        assert plugin._loom_bot_nicks_cache == ()

    def test_on_loom_config_change_rewires_after_live_config(self, mocker: MockerFixture) -> None:
        # Boot with loom disabled (defaults).
        plugin = self._build(mocker, {})
        assert plugin._loom is None

        # Operator runs @config plugins.LLM.loomNetwork / loomChannel ...
        # — the callback fires _on_loom_config_change which re-runs
        # _wire_loom_if_enabled with the new registry values.
        new_side = make_registry_side_effect(
            {
                "loomNetwork": "afternet",
                "loomChannel": "#cybercafe",
                "loomModel": "gemini/x",
                "loomCycleInterval": 5,
                "loomVerseCooldown": 20,
                "loomBeatWindow": 90,
                "loomTranscriptMaxLines": 40,
                "loomTranscriptMaxChars": 8000,
                "loomBotNicks": "",
                "verseAutoApplyThreshold": 0.85,
            }
        )
        plugin.registryValue = mocker.MagicMock(side_effect=new_side)

        plugin._on_loom_config_change()

        assert plugin._loom is not None
        assert plugin._loom_channel_cache == "#cybercafe"
        assert plugin._loom_network_cache == "afternet"

    def test_on_loom_config_change_swallows_exceptions(self, mocker: MockerFixture) -> None:
        plugin = self._build(mocker, {})
        mocker.patch.object(plugin, "_wire_loom_if_enabled", side_effect=RuntimeError("boom"))
        # Must not raise — non-fatal, logged only.
        plugin._on_loom_config_change()
