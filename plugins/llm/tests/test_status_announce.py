"""Announcer: template-primary, LLM rewrite as a post-checked upgrade."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm import statuspage


def incident() -> statuspage.IncidentView:
    return statuspage.IncidentView(
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
        plugin._status_rewrite = MagicMock(return_value="x" * 900)
        plugin._announce_status(statuspage.Delta(opened=(incident(),)))
        assert len(plugin._sent_text[0]) <= 400


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
