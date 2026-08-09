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
