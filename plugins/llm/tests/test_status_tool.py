"""The check_service_status tool: schema shape, visibility, and handler."""

from __future__ import annotations

import json

from llm import assistant
from llm.profile import PROFILE_CHAT, PROFILE_REMIND_ACTION, PROFILE_VERSE


class TestToolSchema:
    def test_tool_is_registered(self):
        assert "check_service_status" in assistant.ASSISTANT_TOOL_REGISTRY

    def test_tool_takes_no_parameters(self):
        spec = assistant.ASSISTANT_TOOL_REGISTRY["check_service_status"]
        assert spec.schema["parameters"]["properties"] == {}
        assert spec.schema["parameters"].get("required", []) == []

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
        ex = self._executor(lambda: payload)
        assert json.loads(ex._tool_check_service_status({})) == payload

    def test_unavailable_when_not_wired(self):
        ex = self._executor(None)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_callback_failure_becomes_an_error_envelope(self):
        def boom():
            raise RuntimeError("no cache")

        ex = self._executor(boom)
        result = json.loads(ex._tool_check_service_status({}))
        assert "error" in result

    def test_ignores_hallucinated_arguments(self):
        """The schema takes none, but a model may still send some."""
        payload = {"indicator": "none"}
        ex = self._executor(lambda: payload)
        assert json.loads(ex._tool_check_service_status({"service": "anthropic"})) == payload


class TestToolPayloadOwnership:
    def test_tool_payload_does_not_advance_lifecycle_state(self, status_plugin):
        """The tool path is a NEW writer to shared state; Task 5's tests do
        not cover it."""
        from llm import statuspage
        from llm.plugin import LLM

        status_plugin._status_tool_payload = LLM._status_tool_payload.__get__(status_plugin)
        status_plugin._run_status_poll()  # cold start, seeds empty
        before = status_plugin._status_state

        incident = statuspage.IncidentView(
            id="inc1",
            name="Elevated error rates on Claude Opus 4.5",
            status="investigating",
            impact="minor",
            affected_components=("Claude API (api.anthropic.com)",),
            started_at=None,
            created_at=None,
            latest_update_body="We are investigating.",
            latest_update_at=None,
        )
        status_plugin._fake_snapshot = statuspage.Snapshot(
            page_name="Claude",
            page_url="https://status.claude.com",
            indicator="minor",
            description="Partial System Outage",
            components={"Claude API (api.anthropic.com)": "operational"},
            incidents={incident.id: incident},
            fetched_at=2000.0,
        )
        status_plugin._now = 2000.0  # clear the 30s fetch floor

        status_plugin._status_tool_payload()

        assert status_plugin._status_state is before, "the tool path must not write lifecycle state"
