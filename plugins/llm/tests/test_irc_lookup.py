"""@channels, @names and the irc_lookup tool: LIST/NAMES round trips.

The server is simulated by giving ``irc.queueMsg`` a side effect that feeds
the matching numerics straight back through the plugin's ``doNNN`` handlers,
which is the shape of a real round trip minus the socket: the command thread
sends, the driver thread answers, the command thread wakes up.
"""

from __future__ import annotations

import json

import pytest
from llm.plugin import AssistantResult
from supybot import ircmsgs

from .conftest import make_registry_side_effect


def numeric(command: str, *args: str) -> ircmsgs.IrcMsg:
    return ircmsgs.IrcMsg(command=command, args=("testbot", *args), prefix="irc.example.net")


def serve_list(plugin, irc, rows: list[tuple[str, int, str]]) -> None:
    """Answer the next LIST with ``rows`` then 323."""

    def on_send(m) -> None:
        if m.command != "LIST":
            return
        for name, users, topic in rows:
            plugin.do322(irc, numeric("322", name, str(users), topic))
        plugin.do323(irc, numeric("323", "End of /LIST"))

    irc.queueMsg.side_effect = on_send


def serve_names(plugin, irc, channel: str, lines: list[str], *, error: str | None = None) -> None:
    """Answer the next NAMES for ``channel`` with 353 lines then 366 (or 403)."""

    def on_send(m) -> None:
        if m.command != "NAMES":
            return
        if error is not None:
            plugin.do403(irc, numeric("403", channel, error))
            return
        for line in lines:
            plugin.do353(irc, numeric("353", "=", channel, line))
        plugin.do366(irc, numeric("366", channel, "End of /NAMES list."))

    irc.queueMsg.side_effect = on_send


@pytest.fixture
def lookup_env(plugin_env):
    plugin, irc, msg = plugin_env
    irc.network = "afternet"
    plugin.registryValue.side_effect = make_registry_side_effect({"ircLookupEnabled": True})
    return plugin, irc, msg


def _sent_commands(irc) -> list[str]:
    return [c.args[0].command for c in irc.queueMsg.call_args_list]


class TestChannelsCommand:
    def test_lists_busiest_channels_first(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_list(plugin, irc, [("#quiet", 2, ""), ("#busy", 30, "Busy place")])

        plugin.channels(irc, msg, [])

        assert _sent_commands(irc) == ["LIST"]
        irc.reply.assert_called_once()
        assert irc.reply.call_args.args[0] == "2 channels: #busy (30) Busy place | #quiet (2)"

    def test_exact_channel_name_answers_that_channel(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_list(plugin, irc, [("#linux", 5, "Kernel talk"), ("#linuxhelp", 3, "")])

        plugin.channels(irc, msg, ["#linux"])

        assert irc.reply.call_args.args[0] == "#linux (5) Kernel talk"

    def test_glob_and_min_users(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_list(plugin, irc, [("#linux", 5, ""), ("#linuxhelp", 1, ""), ("#bsd", 8, "")])

        plugin.channels(irc, msg, ["--min", "3", "#linux*"])

        assert irc.reply.call_args.args[0] == "1 channel: #linux (5)"

    def test_server_silence_is_reported_not_hung(self, lookup_env, mocker) -> None:
        plugin, irc, msg = lookup_env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.01)

        plugin.channels(irc, msg, [])

        irc.reply.assert_not_called()
        irc.error.assert_called_once()
        assert "did not answer" in irc.error.call_args.args[0]

    def test_second_call_within_ttl_does_not_resend_list(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_list(plugin, irc, [("#a", 1, "")])

        plugin.channels(irc, msg, [])
        plugin.channels(irc, msg, ["#a"])

        assert _sent_commands(irc) == ["LIST"]

    def test_znc_playback_is_ignored(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        msg.time = plugin.startup_time - 60

        plugin.channels(irc, msg, [])

        irc.queueMsg.assert_not_called()
        irc.reply.assert_not_called()


class TestNamesCommand:
    def test_lists_members_of_a_named_channel(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_names(plugin, irc, "#elsewhere", ["@alice +bob", "carol"])

        plugin.names(irc, msg, ["#elsewhere"])

        sent = irc.queueMsg.call_args_list[0].args[0]
        assert sent.command == "NAMES" and sent.args == ("#elsewhere",)
        assert irc.reply.call_args.args[0] == "#elsewhere (3): @alice +bob carol"

    def test_defaults_to_the_current_channel(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_names(plugin, irc, "#test", ["testnick"])

        plugin.names(irc, msg, [])

        assert irc.reply.call_args.args[0] == "#test (1): testnick"

    def test_no_such_channel_is_an_error(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_names(plugin, irc, "#nope", [], error="No such channel")

        plugin.names(irc, msg, ["#nope"])

        irc.reply.assert_not_called()
        assert "No such channel" in irc.error.call_args.args[0]

    def test_secret_channel_reads_as_not_visible(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        serve_names(plugin, irc, "#secret", [])

        plugin.names(irc, msg, ["#secret"])

        assert "no visible members" in irc.reply.call_args.args[0]

    def test_non_channel_argument_is_rejected_without_sending(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env

        plugin.names(irc, msg, ["alice"])

        irc.queueMsg.assert_not_called()
        irc.error.assert_called_once()


class TestIrcLookupTool:
    def test_schema_offers_channels_and_names(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env

        schemas, handlers = plugin._build_irc_lookup_tool(irc)

        assert [s["function"]["name"] for s in schemas] == ["irc_lookup"]
        params = schemas[0]["function"]["parameters"]
        assert params["properties"]["kind"]["enum"] == ["channels", "names"]
        assert set(handlers) == {"irc_lookup"}

    def test_channels_returns_structured_rows(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        serve_list(plugin, irc, [("#a", 3, "\x02Alpha\x02"), ("#b", 12, "")])
        _, handlers = plugin._build_irc_lookup_tool(irc)

        result = handlers["irc_lookup"]({"kind": "channels"})

        payload = json.loads(result.content)
        assert payload["status"] == "ok"
        assert payload["total"] == 2
        assert payload["channels"] == [
            {"name": "#b", "users": 12, "topic": ""},
            {"name": "#a", "users": 3, "topic": "Alpha"},
        ]

    def test_channels_filters_by_target_glob(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        serve_list(plugin, irc, [("#linux", 3, ""), ("#bsd", 12, "")])
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(
            handlers["irc_lookup"]({"kind": "channels", "target": "#lin*"}).content
        )

        assert [c["name"] for c in payload["channels"]] == ["#linux"]

    def test_channels_caps_rows_but_reports_total(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        serve_list(plugin, irc, [(f"#c{i}", i, "") for i in range(60)])
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "channels"}).content)

        assert payload["total"] == 60
        assert len(payload["channels"]) == 25
        assert payload["channels"][0]["name"] == "#c59"

    def test_names_returns_nicks(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        serve_names(plugin, irc, "#chan", ["@alice +bob", "carol"])
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "names", "target": "#chan"}).content)

        assert payload == {
            "status": "ok",
            "channel": "#chan",
            "count": 3,
            "nicks": ["@alice", "+bob", "carol"],
        }

    def test_names_requires_a_channel_target(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "names", "target": "alice"}).content)

        assert "error" in payload
        irc.queueMsg.assert_not_called()

    def test_names_server_error_is_passed_through(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        serve_names(plugin, irc, "#nope", [], error="No such channel")
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "names", "target": "#nope"}).content)

        assert payload["error"] == "No such channel"

    def test_timeout_is_an_error_payload(self, lookup_env, mocker) -> None:
        plugin, irc, _msg = lookup_env
        mocker.patch.object(plugin, "_IRC_QUERY_TIMEOUT", 0.01)
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "channels"}).content)

        assert "did not answer" in payload["error"]

    def test_unknown_kind_is_an_error(self, lookup_env) -> None:
        plugin, irc, _msg = lookup_env
        _, handlers = plugin._build_irc_lookup_tool(irc)

        payload = json.loads(handlers["irc_lookup"]({"kind": "whois"}).content)

        assert "error" in payload


class TestChatWiring:
    def _result(self) -> AssistantResult:
        return AssistantResult(
            content="ok",
            grounding_used=False,
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="m",
        )

    def test_ask_advertises_irc_lookup_when_enabled(self, lookup_env) -> None:
        plugin, irc, msg = lookup_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["who is in #linux"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        names = [t["function"]["name"] for t in kwargs["extra_tools"]]
        assert names == ["irc_lookup"]
        assert "irc_lookup" in kwargs["extra_handlers"]

    def test_ask_omits_irc_lookup_when_disabled(self, plugin_env) -> None:
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect({"ircLookupEnabled": False})
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs["extra_tools"] is None
