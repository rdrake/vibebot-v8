"""Poller lifecycle and the read-cache / lifecycle-state ownership split."""

from __future__ import annotations

from llm import statuspage


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

    def test_poll_swallows_errors_so_the_schedule_survives(self, status_plugin):
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
