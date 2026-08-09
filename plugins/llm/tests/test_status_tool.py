"""The check_service_status tool: schema shape, visibility, and handler."""

from __future__ import annotations

import json

from llm import assistant, statuspage
from llm.profile import PROFILE_CHAT, PROFILE_REMIND_ACTION, PROFILE_VERSE

from .conftest import make_completion_response


def green_snapshot(fetched_at: float = 1000.0, *, incidents=()) -> statuspage.Snapshot:
    """Mirrors test_status_poller.py's helper of the same name."""
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="minor" if incidents else "none",
        description="Partial System Outage" if incidents else "All Systems Operational",
        components={"Claude API (api.anthropic.com)": "operational"},
        incidents={i.id: i for i in incidents},
        fetched_at=fetched_at,
    )


class TestToolSchema:
    def test_tool_is_registered(self):
        assert "check_service_status" in assistant.ASSISTANT_TOOL_REGISTRY

    def test_tool_parameters_are_exactly_include_history(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert set(spec.schema["parameters"]["properties"]) == {"include_history"}
        assert spec.schema["parameters"].get("required", []) == []

    def test_include_history_description_mentions_past_or_resolved(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        desc = spec.schema["parameters"]["properties"]["include_history"]["description"]
        assert "past" in desc.lower() or "resolved" in desc.lower()

    def test_visible_in_chat_and_remind_action_only(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert spec.visible_in == frozenset({PROFILE_CHAT, PROFILE_REMIND_ACTION})

    def test_not_visible_in_verse(self):
        """Verse must stay a strict subset of chat, and storytelling has no
        use for a status check."""
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert PROFILE_VERSE not in spec.visible_in

    def test_requires_only_llm_ask(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert spec.capability == "llm.ask"
        assert spec.require_account is False

    def test_description_pins_the_recency_threshold(self):
        """Without this the model calls a three-day-old incident 'recent'."""
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert "recently" in spec.schema["description"]
        assert "latest_update_age_sec" in spec.schema["description"]


class TestHandler:
    def _executor(self, status_fn):
        from unittest.mock import MagicMock

        return assistant.AssistantToolExecutor(
            db=MagicMock(),
            context=MagicMock(),
            nick="tester",
            channel="#test",
            status_fn=status_fn,
        )

    def test_returns_the_payload_as_json(self):
        payload = {"indicator": "none", "description": "All Systems Operational"}
        ex = self._executor(lambda **_: payload)
        assert json.loads(ex._tool_check_service_status({})) == payload

    def test_unavailable_when_not_wired(self):
        ex = self._executor(None)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_callback_failure_becomes_an_error_envelope(self):
        def boom(**_):
            raise RuntimeError("no cache")

        ex = self._executor(boom)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_ignores_hallucinated_arguments(self):
        """Only include_history is defined, but a model may still send extras."""
        payload = {"indicator": "none"}
        ex = self._executor(lambda **_: payload)
        assert json.loads(ex._tool_check_service_status({"service": "anthropic"})) == payload

    def test_defaults_include_history_to_false(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({})
        assert captured["include_history"] is False

    def test_passes_include_history_true_through(self):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"include_history": True})
        assert captured["include_history"] is True

    def _captured_include_history(self, raw):
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return {"indicator": "none"}

        ex = self._executor(spy)
        ex._tool_check_service_status({"include_history": raw})
        return captured["include_history"]

    def test_stringified_true_values_parse_as_true(self):
        """Models sometimes emit stringified tool arguments (documented gap);
        bool("false") is truthy in Python, so this must be parsed, not cast."""
        for raw in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert self._captured_include_history(raw) is True, raw

    def test_stringified_false_values_parse_as_false(self):
        for raw in ("false", "False", "0", "", "no"):
            assert self._captured_include_history(raw) is False, raw

    def test_none_and_absent_parse_as_false(self):
        assert self._captured_include_history(None) is False


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


class TestToolWiringGate:
    """service.py ~4996: status_fn must be wired only when statusPageUrl is
    configured — with it empty, config.py says status awareness is fully
    disabled, so the tool must not occupy a chat-surface slot at all."""

    def _run(self, mocker, make_service, *, status_page_url: str):
        from llm.assistant import AssistantToolExecutor
        from llm.profile import PROFILE_CHAT

        service, plugin = make_service(statusPageUrl=status_page_url)
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("hi"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        # AssistantToolExecutor is imported locally inside assistant_completion
        # (service.py ~4744), so the patch target is the defining module, not
        # llm.service's (nonexistent) module-level attribute.
        executor_spy = mocker.patch(
            "llm.assistant.AssistantToolExecutor", wraps=AssistantToolExecutor
        )

        service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        return executor_spy, plugin

    def test_wired_when_status_page_url_is_configured(self, mocker, make_service):
        executor_spy, plugin = self._run(
            mocker, make_service, status_page_url="https://status.claude.com"
        )
        kwargs = executor_spy.call_args.kwargs
        assert kwargs["status_fn"] is plugin._status_tool_payload

    def test_absent_when_status_page_url_is_empty(self, mocker, make_service):
        executor_spy, _plugin = self._run(mocker, make_service, status_page_url="")
        kwargs = executor_spy.call_args.kwargs
        assert kwargs["status_fn"] is None


class TestToolSchemaGateOnConfig:
    """The schema itself, not just status_fn, must not occupy a chat-surface
    slot when the feature is unconfigured — an offered tool that can only
    answer 'not configured' still costs prompt tokens on every completion."""

    def _tool_names(self, mocker, make_service, *, status_page_url: str) -> set[str]:
        from llm.profile import PROFILE_CHAT

        service, plugin = make_service(statusPageUrl=status_page_url)
        plugin._status_tool_payload = mocker.Mock(name="_status_tool_payload")
        completion = mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("hi"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        tools = completion.call_args.kwargs.get("tools") or []
        return {(t.get("function", t) or {}).get("name") for t in tools}

    def test_absent_from_tool_list_when_status_page_url_is_empty(self, mocker, make_service):
        names = self._tool_names(mocker, make_service, status_page_url="")
        assert "check_service_status" not in names

    def test_present_in_tool_list_when_status_page_url_is_set(self, mocker, make_service):
        names = self._tool_names(mocker, make_service, status_page_url="https://status.claude.com")
        assert "check_service_status" in names


class TestToolPayloadOwnership:
    def test_tool_payload_does_not_advance_lifecycle_state(self, status_plugin):
        """The tool path is a NEW writer to shared state; Task 5's tests do
        not cover it."""
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._run_status_poll()  # cold start, seeds empty
        before = status_plugin._status_state

        status_plugin._fake_snapshot = green_snapshot(2000.0, incidents=[incident()])
        status_plugin._now = 2000.0  # clear the 30s fetch floor

        status_plugin._status_tool_payload()

        assert status_plugin._status_state is before, "the tool path must not write lifecycle state"


class TestToolPayloadStaleness:
    """The refresh/staleness branches _status_tool_payload takes around the
    read cache: cold, fresh, and stale-with-failed-refresh."""

    def test_cold_start_with_no_cache_returns_error_envelope(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = None
        status_plugin._fake_error = statuspage.FetchError("unreachable")

        payload = status_plugin._status_tool_payload()

        assert "error" in payload
        assert "indicator" not in payload, "must never read as 'all systems operational'"

    def test_fresh_cache_is_served_without_a_stale_marker(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = green_snapshot(1000.0)
        status_plugin._now = 1000.0

        payload = status_plugin._status_tool_payload()

        assert "stale" not in payload
        assert "error" not in payload
        assert payload["indicator"] == "none"

    def test_stale_cache_with_failing_refresh_is_marked_stale_and_keeps_data(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = green_snapshot(0.0)
        status_plugin._now = 10_000.0  # far beyond 2 * _STATUS_POLL_INTERVAL
        status_plugin._status_last_fetch = 0.0  # clear the 30s floor
        status_plugin._fake_error = statuspage.FetchError("unreachable")

        payload = status_plugin._status_tool_payload()

        assert payload["stale"] is True
        assert "error" in payload
        assert payload["indicator"] == "none", "last-known data must survive alongside the error"


def history_entry(entry_id="inc1") -> statuspage.HistoryEntry:
    from datetime import UTC, datetime

    return statuspage.HistoryEntry(
        id=entry_id,
        name="Elevated error rates on Claude Opus 4.5",
        status="resolved",
        impact="minor",
        started_at=datetime(2026, 8, 5, 13, 55, tzinfo=UTC),
        resolved_at=datetime(2026, 8, 5, 15, 10, tzinfo=UTC),
    )


class TestStatusHistoryPayload:
    """_status_history_payload: lazy fetch, TTL cache, failure-tolerant,
    never touches _status_state or _status_read_cache."""

    def _bind(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_history_payload = LLM._status_history_payload.__get__(status_plugin)
        return status_plugin

    def test_cache_hit_inside_ttl_does_not_refetch(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fetch = mocker.patch("llm.plugin.statuspage.fetch_incidents")
        plugin._status_history_cache = (history_entry(),)
        plugin._status_history_at = 1000.0
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL - 1

        result = plugin._status_history_payload()

        fetch.assert_not_called()
        assert result[0]["name"] == "Elevated error rates on Claude Opus 4.5"

    def test_expiry_refetches(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        fetch = mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = (history_entry(),)
        plugin._status_history_at = 1000.0
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL + 1

        plugin._status_history_payload()

        fetch.assert_called_once()

    def test_fetch_failure_returns_stale_cache_without_raising(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = (history_entry(),)
        plugin._status_history_at = 0.0
        plugin._now = 1000.0 + plugin._STATUS_HISTORY_TTL + 1

        result = plugin._status_history_payload()

        assert result[0]["name"] == "Elevated error rates on Claude Opus 4.5"

    def test_fetch_failure_with_no_cache_returns_empty_list(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        plugin._now = 1000.0

        result = plugin._status_history_payload()

        assert result == []

    def test_invalid_payload_on_fetch_is_swallowed_like_any_other_failure(
        self, status_plugin, mocker
    ):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"not": "a valid history payload"})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        plugin._now = 1000.0

        result = plugin._status_history_payload()

        assert result == []

    def test_does_not_touch_status_state_or_read_cache(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        before_state = plugin._status_state
        before_read_cache = plugin._status_read_cache
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        plugin._now = 1000.0

        plugin._status_history_payload()

        assert plugin._status_state is before_state
        assert plugin._status_read_cache is before_read_cache

    def test_failure_stamps_backoff_and_a_second_immediate_call_does_not_refetch(
        self, status_plugin, mocker
    ):
        """Without this, three 'when did it last go down' questions during an
        outage each pay a 30s-timeout fetch while holding an executor permit."""
        plugin = self._bind(status_plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        plugin._status_history_failed_at = 0.0
        plugin._now = 1000.0

        first = plugin._status_history_payload()
        assert first == []
        assert plugin._status_history_failed_at == 1000.0
        fetch.assert_called_once()

        plugin._now = 1000.0 + plugin._STATUS_HISTORY_RETRY - 1
        second = plugin._status_history_payload()

        # Still just the one call: the backoff window has not elapsed.
        fetch.assert_called_once()
        assert second == []

    def test_retries_after_the_backoff_window_elapses(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fetch = mocker.patch(
            "llm.plugin.statuspage.fetch_incidents", side_effect=statuspage.FetchError("down")
        )
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        plugin._status_history_failed_at = 0.0
        plugin._now = 1000.0

        plugin._status_history_payload()
        fetch.assert_called_once()

        plugin._now = 1000.0 + plugin._STATUS_HISTORY_RETRY + 1
        plugin._status_history_payload()

        assert fetch.call_count == 2

    def test_success_clears_the_failure_stamp(self, status_plugin, mocker):
        plugin = self._bind(status_plugin)
        fake_result = mocker.Mock(payload={"page": {}, "incidents": []})
        mocker.patch("llm.plugin.statuspage.fetch_incidents", return_value=fake_result)
        plugin._status_history_cache = None
        plugin._status_history_at = 0.0
        # Past the backoff window (retry=120), or this call would itself be
        # skipped as still-backing-off and never reach the fetch.
        plugin._status_history_failed_at = 1000.0 - plugin._STATUS_HISTORY_RETRY - 1
        plugin._now = 1000.0

        plugin._status_history_payload()

        assert plugin._status_history_failed_at == 0.0


class TestStatusToolPayloadIncludeHistory:
    def test_include_history_false_has_no_recent_incidents_key(self, status_plugin):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_read_cache = green_snapshot(1000.0)
        status_plugin._now = 1000.0

        payload = status_plugin._status_tool_payload(include_history=False)

        assert "recent_incidents" not in payload
        assert payload["indicator"] == "none"

    def test_include_history_true_includes_recent_incidents(self, status_plugin, mocker):
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._status_history_payload = LLM._status_history_payload.__get__(status_plugin)
        status_plugin._status_read_cache = green_snapshot(1000.0)
        status_plugin._now = 1000.0
        status_plugin._status_history_cache = (history_entry(),)
        status_plugin._status_history_at = 1000.0

        payload = status_plugin._status_tool_payload(include_history=True)

        assert "recent_incidents" in payload
        assert payload["recent_incidents"][0]["name"] == "Elevated error rates on Claude Opus 4.5"
        assert payload["indicator"] == "none", "current status is always present"
