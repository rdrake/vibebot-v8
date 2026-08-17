"""Poller lifecycle and the read-cache / lifecycle-state ownership split."""

from __future__ import annotations

import pytest
from llm import statuspage
from llm.plugin import LLM


def green_snapshot(fetched_at: float = 1000.0, *, incidents=()) -> statuspage.Snapshot:
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="minor" if incidents else "none",
        description="Partial System Outage" if incidents else "All Systems Operational",
        components={"Claude API (api.anthropic.com)": "operational"},
        incidents={i.id: i for i in incidents},
        fetched_at=fetched_at,
    )


def incident(incident_id="inc1") -> statuspage.IncidentView:
    return statuspage.IncidentView(
        id=incident_id,
        name="Elevated error rates on Claude Opus 4.5",
        status="investigating",
        impact="minor",
        affected_components=("Claude API (api.anthropic.com)",),
        started_at=None,
        created_at=None,
        latest_update_body="We are investigating.",
        latest_update_at=None,
    )


class TestOwnershipSplit:
    def test_inline_fetch_does_not_advance_lifecycle_state(self, status_plugin):
        """The defect two reviewers found independently: a user's query must
        not consume the announcement."""
        plugin = status_plugin
        plugin._run_status_poll()  # cold start, seeds empty
        assert plugin._status_state.seeded is True

        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._now = 2000.0  # clear the 30s fetch floor
        plugin._status_fetch_now()  # the tool's inline path

        assert plugin._status_read_cache.incidents, "read cache refreshed"
        assert plugin._status_state.active == {}, "lifecycle state untouched"

    def test_incident_seen_first_by_the_tool_is_still_announced(self, status_plugin):
        plugin = status_plugin
        plugin._run_status_poll()
        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._now = 2000.0  # clear the 30s fetch floor
        plugin._status_fetch_now()
        plugin._run_status_poll()
        assert plugin._announce_status.call_count == 1
        delta = plugin._announce_status.call_args[0][0]
        assert [i.id for i in delta.opened] == ["inc1"]


class TestDeltaReachesTheAnnouncer:
    """The poll's own gate on calling _announce_status.

    Every other announcer test calls ``_announce_status`` directly with a
    hand-built Delta, so the branch that decides whether to call it at all was
    never covered from this side.
    """

    def test_an_all_clear_alone_still_reaches_the_announcer(self, status_plugin):
        """Observed in prod: incident hdynq1pc0fn8 cleared at 00:27:36 on
        2026-08-15 with the bot live and in the channel, and #clanker heard
        nothing. The resolved branch was added inside ``_announce_status``
        while the caller stayed gated on ``delta.opened``, so an incident that
        cleared in a pass where nothing new opened was parked in
        ``pending_resolved`` and never spoken."""
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()  # cold start seeds inc1 as live
        plugin._announce_status.reset_mock()

        plugin._fake_snapshot = green_snapshot(2000.0)  # inc1 gone from the unresolved set
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 1
        delta = plugin._announce_status.call_args[0][0]
        assert [i.id for i in delta.resolved] == ["inc1"]
        assert delta.opened == ()

    def test_a_quiet_poll_does_not_call_the_announcer(self, status_plugin):
        """The gate still has to gate: nothing opened and nothing cleared must
        not spend a pass in the announcer."""
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()
        plugin._announce_status.reset_mock()

        plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        plugin._run_status_poll()

        assert plugin._announce_status.call_count == 0


class TestFailureHandling:
    def test_fetch_error_retains_last_good_state(self, status_plugin):
        plugin = status_plugin
        plugin._fake_snapshot = green_snapshot(1000.0, incidents=[incident()])
        plugin._run_status_poll()
        before = plugin._status_state

        plugin._fake_error = statuspage.FetchError("boom")
        plugin._run_status_poll()

        assert plugin._status_state is before, "state must not advance on failure"
        assert plugin._status_read_cache is not None

    def test_invalid_payload_does_not_seed(self, status_plugin):
        plugin = status_plugin
        plugin._fake_error = statuspage.InvalidPayload("garbage")
        plugin._run_status_poll()
        assert plugin._status_state.seeded is False, "a bad body is not a cold start"

    def test_poll_swallows_unexpected_errors(self, status_plugin):
        plugin = status_plugin
        plugin._fake_error = RuntimeError("unexpected")
        plugin._run_status_poll()  # must not raise


class TestFetchFloor:
    def test_inline_fetch_respects_the_floor(self, status_plugin):
        plugin = status_plugin
        plugin._status_last_fetch = 999.0
        plugin._now = 1000.0
        before = plugin._fetch_calls
        plugin._status_fetch_now()
        assert plugin._fetch_calls == before, "inside the 30s floor, serve cache"

    def test_inline_fetch_proceeds_past_the_floor(self, status_plugin):
        plugin = status_plugin
        plugin._status_last_fetch = 900.0
        plugin._now = 1000.0
        before = plugin._fetch_calls
        plugin._status_fetch_now()
        assert plugin._fetch_calls == before + 1


class TestDisabled:
    def test_empty_url_disables_polling(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrl"] = ""
        before = plugin._fetch_calls
        plugin._run_status_poll()
        assert plugin._fetch_calls == before


class TestArming:
    def test_arms_even_when_url_is_empty(self, status_plugin, mocker):
        """Re-enabling statusPageUrl must resume polling without a reload."""
        sched = mocker.patch("llm.plugin.schedule")
        status_plugin._registry["statusPageUrl"] = ""
        status_plugin._schedule_status_poll()
        assert sched.addEvent.called, "an empty URL must not disarm the timer forever"

    def test_does_not_arm_when_closing(self, status_plugin, mocker):
        sched = mocker.patch("llm.plugin.schedule")
        status_plugin._llm_executor.closing = True
        status_plugin._schedule_status_poll()
        assert not sched.addEvent.called


class TestInflightDedup:
    def test_second_enqueue_while_inflight_submits_nothing(self, status_plugin, mocker):
        mocker.patch("llm.plugin.schedule")
        status_plugin._status_poll_inflight.set()
        status_plugin._enqueue_status_poll()
        assert not status_plugin._llm_executor.submit.called


class TestNotModified:
    def test_304_with_cache_returns_cached_snapshot_with_new_timestamp(self, status_plugin, mocker):
        cached = green_snapshot(1000.0)
        status_plugin._status_read_cache = cached
        status_plugin._now = 2000.0
        mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                not_modified=True, payload=None, etag='W/"a"', modified=None
            ),
        )
        snap = LLM._status_fetch_snapshot.__get__(status_plugin)()
        assert snap.fetched_at == 2000.0
        assert snap.incidents == cached.incidents
        assert snap.etag == cached.etag

    def test_304_with_no_cache_raises_fetch_error(self, status_plugin, mocker):
        status_plugin._status_read_cache = None
        mocker.patch(
            "llm.plugin.statuspage.fetch_summary",
            return_value=statuspage.FetchResult(
                not_modified=True, payload=None, etag=None, modified=None
            ),
        )
        with pytest.raises(statuspage.FetchError):
            LLM._status_fetch_snapshot.__get__(status_plugin)()


class TestSourceList:
    def test_canonicalizes_dedupes_and_preserves_order(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "https://status.claude.com/",
            "https://www.githubstatus.com",
            "HTTPS://STATUS.CLAUDE.COM",
        ]
        assert plugin._status_sources() == [
            "https://status.claude.com",
            "https://www.githubstatus.com",
        ]

    def test_unusable_entries_are_dropped_not_fatal(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [
            "not a url",
            "https://www.githubstatus.com",
        ]
        assert plugin._status_sources() == ["https://www.githubstatus.com"]

    def test_empty_list_disables(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = []
        assert plugin._status_sources() == []

    def test_caps_at_max_sources(self, status_plugin):
        plugin = status_plugin
        plugin._registry["statusPageUrls"] = [f"https://status{i}.example.com" for i in range(9)]
        assert len(plugin._status_sources()) == plugin._STATUS_MAX_SOURCES


CLAUDE = "https://status.claude.com"
GITHUB = "https://www.githubstatus.com"


class TestPerSourceState:
    def test_pruning_clears_every_keyed_structure(self, status_plugin):
        """Pruning only _status_state would leave the other five growing without
        bound — the 5-source cap bounds the configured set, not the historical
        one."""
        plugin = status_plugin
        plugin._status_state = {GITHUB: statuspage.StatusState(seeded=True)}
        plugin._status_read_cache = {GITHUB: green_snapshot(1000.0)}
        plugin._status_last_fetch = {GITHUB: 1000.0}
        plugin._status_history_cache = {GITHUB: ()}
        plugin._status_history_at = {GITHUB: 1000.0}
        plugin._status_history_failed_at = {GITHUB: 1000.0}

        plugin._status_prune_sources([CLAUDE])

        for name in (
            "_status_state",
            "_status_read_cache",
            "_status_last_fetch",
            "_status_history_cache",
            "_status_history_at",
            "_status_history_failed_at",
        ):
            assert getattr(plugin, name) == {}, f"{name} still holds the removed source"

    def test_pruning_keeps_configured_sources(self, status_plugin):
        plugin = status_plugin
        plugin._status_state = {
            CLAUDE: statuspage.StatusState(seeded=True),
            GITHUB: statuspage.StatusState(seeded=True),
        }
        plugin._status_prune_sources([CLAUDE, GITHUB])
        assert set(plugin._status_state) == {CLAUDE, GITHUB}

    def test_fetch_floor_is_per_source(self, status_plugin):
        """One source's inline fetch must not suppress another's."""
        plugin = status_plugin
        plugin._now = 1000.0
        plugin._status_last_fetch = {CLAUDE: 1000.0}
        plugin._status_fetch_now(GITHUB)
        assert plugin._fetch_calls == 1, "GitHub blocked by Claude's floor"
        plugin._status_fetch_now(CLAUDE)
        assert plugin._fetch_calls == 1, "Claude's own floor did not hold"

    def test_read_cache_is_keyed(self, status_plugin):
        plugin = status_plugin
        plugin._now = 2000.0
        plugin._status_fetch_now(GITHUB)
        assert GITHUB in plugin._status_read_cache
        assert CLAUDE not in plugin._status_read_cache
