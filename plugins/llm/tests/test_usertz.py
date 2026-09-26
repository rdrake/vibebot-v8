"""Tests for llm.usertz: zone validation and CTCP TIME offset parsing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from llm.usertz import (
    SOURCE_CTCP,
    SOURCE_SET,
    UTC_DEFAULT,
    UserTz,
    format_offset,
    offset_from_ctcp_time,
    parse_zone,
)

# 09:00 UTC on the morning the "strike" reminder fired at 05:00 Ontario time.
NOW = datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC)


class TestParseZone:
    def test_iana_name(self) -> None:
        assert parse_zone("America/Toronto") == ZoneInfo("America/Toronto")

    def test_strips_whitespace(self) -> None:
        assert parse_zone("  Europe/London ") == ZoneInfo("Europe/London")

    @pytest.mark.parametrize(
        "name", ["", "Mars/Olympus", "EDT-ish", "../../etc/passwd", "/etc/localtime"]
    )
    def test_rejects_non_zones(self, name: str) -> None:
        assert parse_zone(name) is None


class TestFormatOffset:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (timedelta(0), "UTC"),
            (timedelta(hours=-4), "UTC-04:00"),
            (timedelta(hours=5, minutes=30), "UTC+05:30"),
        ],
    )
    def test_format(self, offset: timedelta, expected: str) -> None:
        assert format_offset(offset) == expected


class TestUserTzLabel:
    def test_named_zone_shows_name_and_current_offset(self) -> None:
        tz = UserTz(tz=ZoneInfo("America/Toronto"), source=SOURCE_SET)
        assert tz.label(NOW) == "America/Toronto (UTC-04:00)"

    def test_named_zone_follows_dst(self) -> None:
        tz = UserTz(tz=ZoneInfo("America/Toronto"), source=SOURCE_SET)
        assert tz.label(datetime(2026, 12, 1, tzinfo=UTC)) == "America/Toronto (UTC-05:00)"

    def test_fixed_offset_shows_offset_only(self) -> None:
        tz = UserTz(tz=timezone(timedelta(hours=-4)), source=SOURCE_CTCP)
        assert tz.label(NOW) == "UTC-04:00"

    def test_default_is_utc(self) -> None:
        assert UTC_DEFAULT.label(NOW) == "UTC"


class TestOffsetFromCtcpTime:
    def test_bare_local_time_hexchat_style(self) -> None:
        tz = offset_from_ctcp_time("Sat Sep 26 05:00:02 2026", NOW)
        assert tz == timezone(timedelta(hours=-4))

    def test_explicit_offset_weechat_style(self) -> None:
        tz = offset_from_ctcp_time("Sat, 26 Sep 2026 14:30:00 +0530", NOW)
        assert tz == timezone(timedelta(hours=5, minutes=30))

    def test_rounds_skew_to_quarter_hour(self) -> None:
        # 3 minutes of clock skew on a UTC+1 client.
        tz = offset_from_ctcp_time("Sat Sep 26 10:03:00 2026", NOW)
        assert tz == timezone(timedelta(hours=1))

    def test_time_without_date_across_midnight(self) -> None:
        # 23:30 local while UTC is already 03:30 the next day: UTC-4, not UTC+20.
        now = datetime(2026, 9, 27, 3, 30, tzinfo=UTC)
        tz = offset_from_ctcp_time("23:30:00", now)
        assert tz == timezone(timedelta(hours=-4))

    def test_clock_days_off_is_rejected(self) -> None:
        assert offset_from_ctcp_time("Mon Sep 21 09:00:00 2026", NOW) is None

    @pytest.mark.parametrize("reply", ["", "no idea", "\x01"])
    def test_garbage_is_rejected(self, reply: str) -> None:
        assert offset_from_ctcp_time(reply, NOW) is None
