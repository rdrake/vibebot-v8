"""Per-user time zones: @tz, the CTCP TIME fallback, and reminder parsing.

Regression: on 2026-09-26 "remind me to strike tomorrow morning" fired at
09:00 UTC, which was 05:00 for the Ontario user who asked.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest
from llm.plugin import LLM, Identity
from llm.service import ReminderParseResult
from llm.usertz import SOURCE_CTCP, SOURCE_DEFAULT, SOURCE_SET, UTC_DEFAULT, UserTz

from .conftest import make_completion_response

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

TORONTO = ZoneInfo("America/Toronto")


def _replies(mock_irc: MagicMock) -> list[str]:
    return [c.args[0] for c in mock_irc.reply.call_args_list]


class TestTzCommand:
    def test_sets_valid_zone(self, plugin_env) -> None:
        plugin, mock_irc, mock_msg = plugin_env

        plugin.tz(mock_irc, mock_msg, ["America/Toronto"])

        plugin.db.save_user_timezone.assert_called_once_with("testnick", "America/Toronto")
        assert "America/Toronto (UTC-0" in _replies(mock_irc)[-1]

    def test_rejects_unknown_zone(self, plugin_env) -> None:
        plugin, mock_irc, mock_msg = plugin_env

        plugin.tz(mock_irc, mock_msg, ["EST-ish"])

        plugin.db.save_user_timezone.assert_not_called()
        assert "Unknown time zone" in _replies(mock_irc)[-1]

    def test_shows_current_zone(self, plugin_env) -> None:
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_user_timezone.return_value = "Europe/London"

        plugin.tz(mock_irc, mock_msg, [])

        assert _replies(mock_irc)[-1].startswith("Your time zone: Europe/London")

    def test_show_unset_explains_fallback(self, plugin_env) -> None:
        plugin, mock_irc, mock_msg = plugin_env

        plugin.tz(mock_irc, mock_msg, [])

        assert "@tz" in _replies(mock_irc)[-1]

    def test_clear(self, plugin_env) -> None:
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_user_timezone.return_value = True

        plugin.tz(mock_irc, mock_msg, ["clear"])

        plugin.db.delete_user_timezone.assert_called_once_with("testnick")
        assert _replies(mock_irc)[-1] == "Time zone cleared."


class TestResolveUserTz:
    def test_stored_zone_wins_without_probing(self, plugin_env) -> None:
        plugin, mock_irc, _ = plugin_env
        plugin.db.get_user_timezone.return_value = "America/Toronto"

        tz = plugin._resolve_user_tz(Identity("rdrake", "rdrake"), irc=mock_irc)

        assert tz == UserTz(tz=TORONTO, source=SOURCE_SET)
        plugin._probe_ctcp_tz.assert_not_called()

    def test_account_is_the_storage_key(self, plugin_env) -> None:
        plugin, mock_irc, _ = plugin_env

        plugin._resolve_user_tz(Identity("rd_phone", "rdrake"), irc=mock_irc)

        plugin.db.get_user_timezone.assert_called_once_with("rdrake")

    def test_falls_back_to_ctcp(self, plugin_env) -> None:
        plugin, mock_irc, _ = plugin_env
        offset = timezone(timedelta(hours=-4))
        plugin._probe_ctcp_tz.return_value = offset

        tz = plugin._resolve_user_tz(Identity("rdrake", None), irc=mock_irc)

        assert tz == UserTz(tz=offset, source=SOURCE_CTCP)
        plugin._probe_ctcp_tz.assert_called_once_with(mock_irc, "rdrake")

    def test_no_irc_means_no_probe(self, plugin_env) -> None:
        plugin, _, _ = plugin_env

        assert plugin._resolve_user_tz(Identity("rdrake", None)) == UTC_DEFAULT
        plugin._probe_ctcp_tz.assert_not_called()

    def test_cached_answer_skips_probe(self, plugin_env) -> None:
        plugin, mock_irc, _ = plugin_env
        offset = timezone(timedelta(hours=1))
        plugin._ctcp_tz_cache["rdrake"] = (offset, float("inf"))

        tz = plugin._resolve_user_tz(Identity("RDrake", None), irc=mock_irc)

        assert tz.tz == offset
        plugin._probe_ctcp_tz.assert_not_called()

    def test_cached_silence_is_utc_without_probe(self, plugin_env) -> None:
        plugin, mock_irc, _ = plugin_env
        plugin._ctcp_tz_cache["rdrake"] = (None, float("inf"))

        assert plugin._resolve_user_tz(Identity("rdrake", None), irc=mock_irc) == UTC_DEFAULT
        plugin._probe_ctcp_tz.assert_not_called()


class TestCtcpTimeProbe:
    """The real probe, with a NOTICE delivered from another thread."""

    @pytest.fixture
    def plugin(self, plugin_env, monkeypatch: pytest.MonkeyPatch):
        plugin, mock_irc, _ = plugin_env
        monkeypatch.setattr(LLM, "_CTCP_TIME_TIMEOUT_SECONDS", 2.0)
        return plugin, mock_irc

    @staticmethod
    def _notice(mocker: MockerFixture, sender: str, text: str, target: str = "testbot"):
        msg = mocker.MagicMock()
        msg.command = "NOTICE"
        msg.nick = sender
        msg.prefix = f"{sender}!user@host"
        msg.args = (target, text)
        return msg

    def _answer_when_sent(self, plugin, mock_irc, notice) -> None:
        """Deliver ``notice`` through inFilter once the probe has gone out."""

        def on_queue(_msg):
            threading.Thread(target=plugin.inFilter, args=(mock_irc, notice)).start()

        mock_irc.queueMsg.side_effect = on_queue

    def test_reads_offset_from_reply(self, plugin, mocker: MockerFixture) -> None:
        plugin, mock_irc = plugin
        local = datetime.now(UTC).astimezone(timezone(timedelta(hours=-4)))
        reply = f"\x01TIME {local.strftime('%a %b %d %H:%M:%S %Y')}\x01"
        self._answer_when_sent(plugin, mock_irc, self._notice(mocker, "rdrake", reply))

        offset = LLM._probe_ctcp_tz(plugin, mock_irc, "rdrake")

        assert offset == timezone(timedelta(hours=-4))
        sent = mock_irc.queueMsg.call_args.args[0]
        assert sent.command == "PRIVMSG"
        assert sent.args == ("rdrake", "\x01TIME\x01")
        assert plugin._ctcp_tz_cache["rdrake"][0] == offset
        assert plugin._ctcp_tz_waiters == {}

    def test_silence_is_cached_as_none(self, plugin, monkeypatch: pytest.MonkeyPatch) -> None:
        plugin, mock_irc = plugin
        monkeypatch.setattr(LLM, "_CTCP_TIME_TIMEOUT_SECONDS", 0.0)

        assert LLM._probe_ctcp_tz(plugin, mock_irc, "rdrake") is None
        assert plugin._ctcp_tz_cache["rdrake"][0] is None

    def test_reply_from_another_nick_is_ignored(
        self, plugin, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin, mock_irc = plugin
        forged = self._notice(mocker, "mallory", "\x01TIME Sat, 26 Sep 2026 09:00:00 +1400\x01")
        self._answer_when_sent(plugin, mock_irc, forged)
        monkeypatch.setattr(LLM, "_CTCP_TIME_TIMEOUT_SECONDS", 0.3)

        assert LLM._probe_ctcp_tz(plugin, mock_irc, "rdrake") is None

    def test_unsolicited_reply_is_ignored(self, plugin, mocker: MockerFixture) -> None:
        plugin, mock_irc = plugin

        plugin.inFilter(mock_irc, self._notice(mocker, "rdrake", "\x01TIME 12:00:00\x01"))

        assert plugin._ctcp_tz_waiters == {}
        assert plugin._ctcp_tz_cache == {}


class TestScheduleReminderUsesZone:
    def test_zone_reaches_the_parser(self, plugin_env, mocker: MockerFixture) -> None:
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_user_timezone.return_value = "America/Toronto"
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule", seconds=3600, message="strike", confirmation="Set."
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_reminder(
            mock_irc, mock_msg, Identity("testnick", None), "tomorrow morning strike"
        )

        kwargs = plugin.llm_service.parse_reminder.call_args.kwargs
        assert kwargs["user_tz"] == UserTz(tz=TORONTO, source=SOURCE_SET)

    def test_chain_reschedule_does_not_probe(self, plugin_env, mocker: MockerFixture) -> None:
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule", seconds=3600, message="x", confirmation="Set."
        )
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin._schedule_reminder(
            mock_irc, mock_msg, Identity("testnick", None), "x", parent_chain=1
        )

        plugin._probe_ctcp_tz.assert_not_called()


class TestNextRruleFireInZone:
    # 2026-09-26 05:00 in Toronto (EDT, UTC-4).
    NOW = datetime(2026, 9, 26, 9, 0, tzinfo=UTC).timestamp()

    def test_byhour_is_local_wall_clock(self) -> None:
        nxt = LLM._next_rrule_fire("FREQ=DAILY;BYHOUR=9;BYMINUTE=0", self.NOW, TORONTO)
        assert datetime.fromtimestamp(nxt, UTC) == datetime(2026, 9, 26, 13, 0, tzinfo=UTC)

    def test_survives_dst_change(self) -> None:
        # 2026-11-01 is the fall-back; 9am local is 14:00 UTC after it.
        after = datetime(2026, 11, 1, 12, 0, tzinfo=UTC).timestamp()
        nxt = LLM._next_rrule_fire("FREQ=DAILY;BYHOUR=9;BYMINUTE=0", after, TORONTO)
        assert datetime.fromtimestamp(nxt, UTC) == datetime(2026, 11, 1, 14, 0, tzinfo=UTC)

    def test_default_is_utc(self) -> None:
        nxt = LLM._next_rrule_fire("FREQ=DAILY;BYHOUR=10;BYMINUTE=0", self.NOW)
        assert datetime.fromtimestamp(nxt, UTC) == datetime(2026, 9, 26, 10, 0, tzinfo=UTC)


class TestParseReminderPrompt:
    @pytest.fixture
    def service(self, mocker: MockerFixture):
        from llm.service import LLMService

        plugin = mocker.MagicMock()
        plugin.registryValue.side_effect = lambda key, *args: {
            "assistantModel": "gemini/gemini-2.0-flash",
            "timeout": 30,
        }.get(key, "")
        mocker.patch("llm.service.log")
        return LLMService(plugin)

    def _system_prompt(self, service, mocker: MockerFixture, user_tz: UserTz) -> str:
        completion = mocker.patch("llm.service.litellm.completion")
        completion.return_value = make_completion_response(
            json.dumps({"action": "clarify", "confirmation": "When?"})
        )
        service.parse_reminder("tomorrow morning strike", user_tz=user_tz)
        messages = completion.call_args.kwargs["messages"]
        return next(m["content"] for m in messages if m["role"] == "system")

    def test_local_time_and_zone_in_prompt(self, service, mocker: MockerFixture) -> None:
        prompt = self._system_prompt(service, mocker, UserTz(tz=TORONTO, source=SOURCE_SET))

        assert "America/Toronto (UTC-0" in prompt
        assert "user's local time" in prompt
        assert "set note to null" in prompt

    def test_ctcp_offset_asks_for_tz_in_note(self, service, mocker: MockerFixture) -> None:
        offset = UserTz(tz=timezone(timedelta(hours=-4)), source=SOURCE_CTCP)
        prompt = self._system_prompt(service, mocker, offset)

        assert "UTC-04:00" in prompt
        assert "client's clock" in prompt

    def test_unknown_zone_says_utc_and_points_at_tz(self, service, mocker: MockerFixture) -> None:
        assert UTC_DEFAULT.source == SOURCE_DEFAULT
        prompt = self._system_prompt(service, mocker, UTC_DEFAULT)

        assert "Assuming UTC; set your zone with @tz" in prompt


class TestMechanicalRescheduleUsesOwnerZone:
    def test_rrule_next_fire_uses_owner_zone(self, plugin_env, mocker: MockerFixture) -> None:
        plugin, _, _ = plugin_env
        plugin.db.get_user_timezone.return_value = "America/Toronto"
        mocker.patch("llm.plugin.schedule.addEvent")
        next_fire = mocker.patch.object(LLM, "_next_rrule_fire", return_value=None)

        plugin._mechanical_reschedule(
            nick="rd_phone",
            channel="#t",
            message="m",
            event_name="llm_remind_x",
            action_prompt="",
            account="rdrake",
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule="FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
            watch_mode=False,
            now=0.0,
        )

        plugin.db.get_user_timezone.assert_called_once_with("rdrake")
        assert next_fire.call_args.args[2] == TORONTO
        plugin._probe_ctcp_tz.assert_not_called()
