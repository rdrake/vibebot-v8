"""Announcer: template-primary, LLM rewrite as a post-checked upgrade."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from llm import statuspage
from llm.service import LLMService

from .conftest import make_completion_response


def incident(*, name: str = "Elevated error rates on Claude Opus 4.5") -> statuspage.IncidentView:
    return statuspage.IncidentView(
        id="inc1",
        name=name,
        status="investigating",
        impact="minor",
        affected_components=("Claude API (api.anthropic.com)",),
        started_at=None,
        created_at=None,
        latest_update_body="We are investigating.",
        latest_update_at=None,
    )


class TestTemplatePath:
    def test_template_sends_when_rewrite_returns_none(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert plugin._safe_queue.call_count == 1
        sent = plugin._sent_text[0]
        assert "Elevated error rates on Claude Opus 4.5" in sent

    def test_rewrite_is_used_when_it_passes(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value="Heads up — Claude's API is throwing errors on Opus 4.5."
        )
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "Heads up" in plugin._sent_text[0]


class TestPostChecks:
    def test_rejects_rewrite_carrying_a_foreign_url(self, announcing_plugin):
        """The highest-value filter on this path: injected page text steering
        unprompted channel speech toward a link."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value="Claude is down, see https://evil.example/fix"
        )
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "evil.example" not in plugin._sent_text[0]
        assert "Elevated error rates" in plugin._sent_text[0], "fell back to template"

    def test_accepts_rewrite_linking_the_known_host(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(
            return_value="Claude API is degraded — https://status.claude.com"
        )
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "status.claude.com" in plugin._sent_text[0]

    def test_rejects_rewrite_that_never_names_the_service(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="Something somewhere is broken.")
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "Elevated error rates" in plugin._sent_text[0]

    def test_rejects_empty_rewrite(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="   ")
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "Elevated error rates" in plugin._sent_text[0]


class TestSendPipeline:
    def test_sanitize_output_runs_on_the_template_path(self, announcing_plugin):
        """safeArgument covers CR/LF/NUL only and explicitly not CTCP
        (plugin.py:2867-2868), and the template path carries third-party text
        nearly verbatim."""
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value=None)
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert plugin.llm_service.sanitize_output.called

    def test_line_is_truncated(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_rewrite = MagicMock(return_value="Claude " + "x" * 900)
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert len(plugin._sent_text[0]) == 400
        assert plugin._sent_text[0].startswith("Claude")


class TestMarkingAndBudget:
    def test_marks_announced_only_on_successful_queue(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = True
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "inc1" in plugin._status_state.announced

    def test_does_not_mark_when_the_queue_drops(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._safe_queue.return_value = False
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert "inc1" not in plugin._status_state.announced, "must retry next poll"

    def test_over_budget_skips_the_rewrite_but_still_announces(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._status_announce_times = [plugin._now] * plugin._STATUS_ANNOUNCE_MAX_PER_HOUR
        plugin._status_rewrite = MagicMock()
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert plugin._status_rewrite.call_count == 0, "no completion when over budget"
        assert plugin._safe_queue.call_count == 1, "template still goes out"


class TestChannelSelection:
    def test_only_opted_in_channels_receive_it(self, announcing_plugin):
        plugin = announcing_plugin
        plugin._announce_channels = {"#yes": True, "#no": False}
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        targets = [c.args[1].args[0] for c in plugin._safe_queue.call_args_list]
        assert targets == ["#yes"]

    def test_channel_collection_is_copied_before_iteration(self, announcing_plugin):
        """Stock RSS copies (RSS/plugin.py:405); this repo iterates live and
        survives only because callers swallow. Here a RuntimeError would drop
        the outage announcement during exactly the churn an outage causes."""
        plugin = announcing_plugin
        plugin._announce_channels = {"#a": True, "#b": True}

        original = plugin._all_known_channels

        def mutating():
            result = original()
            plugin._announce_channels["#c"] = True
            return result

        plugin._all_known_channels = mutating
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        targets = [c.args[1].args[0] for c in plugin._safe_queue.call_args_list]
        assert "#c" not in targets, "iteration must use a snapshot, not live state"
        assert sorted(targets) == ["#a", "#b"]


class TestStatusRewrite:
    def test_completion_failure_returns_none_and_template_still_sends(self, announcing_plugin):
        """The announcer must survive a provider outage — it exists to report one."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin.llm_service.status_announce_completion.side_effect = RuntimeError("provider down")
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert plugin._safe_queue.call_count == 1
        assert "Elevated error rates" in plugin._sent_text[0]

    def test_rewrite_passes_only_sanitised_fields_to_the_completion(self, announcing_plugin):
        """Raw third-party prose must never reach the completion's user block."""
        from llm.plugin import LLM

        plugin = announcing_plugin
        plugin._status_rewrite = LLM._status_rewrite.__get__(plugin)
        plugin._status_rewrite(incident(name="bad\x01name <|tok|>"), "#test")
        facts = plugin.llm_service.status_announce_completion.call_args.kwargs["facts"]
        assert "\x01" not in facts["name"]
        assert "<|" not in facts["name"]


class TestIrcForChannel:
    def test_returns_none_when_no_connection_holds_the_channel(self, announcing_plugin, mocker):
        from llm.plugin import LLM

        mocker.patch("llm.plugin.world").ircs = []
        assert LLM._irc_for_channel.__get__(announcing_plugin)("#nowhere") is None


class TestStatusAnnounceCompletion:
    """LLMService.status_announce_completion: tool-less one-shot rewrite.

    Mirrors TestParseReminderService's mock_plugin/service fixture pair
    (test_reminders.py) — the established shape for unit-testing a one-shot
    completion method with litellm.completion mocked out.
    """

    @pytest.fixture
    def mock_plugin(self, mocker):
        plugin = mocker.MagicMock()
        plugin.registryValue.side_effect = lambda key, *args, **kwargs: {
            "assistantModel": "gemini/gemini-2.0-flash",
            "timeout": 30,
            "assistantSystemPrompt": "",
        }.get(key, "")
        return plugin

    @pytest.fixture
    def service(self, mock_plugin, mocker):
        mocker.patch("llm.service.log")
        return LLMService(mock_plugin)

    def test_no_tools_key_in_optional_kwargs(self, service, mocker):
        """include_tools=False is what makes this tool-less."""
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_completion.return_value = make_completion_response("Claude API is degraded.")
        service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert "tools" not in mock_completion.call_args.kwargs

    def test_facts_arrive_as_the_json_user_block(self, service, mocker):
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_completion.return_value = make_completion_response("Claude API is degraded.")
        facts = {"name": "Elevated errors", "service": "Claude", "url": "https://status.claude.com"}
        service.status_announce_completion(facts=facts, channel="#test")
        messages = mock_completion.call_args.kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert json.loads(user_msg["content"]) == facts

    def test_none_content_returns_none(self, service, mocker):
        mock_completion = mocker.patch("llm.service.litellm.completion")
        mock_completion.return_value = make_completion_response(None)
        result = service.status_announce_completion(facts={"name": "x"}, channel="#test")
        assert result is None
