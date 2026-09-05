"""The subject-research pre-stage in front of the @draw and @animate planners.

An image or video generator knows nothing by name. The planners are already
told to keep the user's word and put a description beside it, but for a real
person or a real event nothing supplies the description — the verse canon block
covers what the channel invented, not what the world already contains. This
stage supplies it.

Everything worth testing here is the parse. The researcher is a general model
answering in prose by default, and whatever it returns is about to be pasted
into another model's system prompt, so the filter is by SHAPE: only the
dash-prefixed lines the prompt asked for survive. A refusal, a preamble or a
trailing caveat is dropped for having no dash, which is the same failure
test_image_failure_guard.py guards one stage later — a model reading its own
refusal back and parroting it forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .conftest import make_completion_response

if TYPE_CHECKING:
    pass


def _answering(mocker, service, text: str, **usage: Any):
    """Stub the provider call so the researcher returns ``text``."""
    mocker.patch.object(service, "_is_xai_model", return_value=False)
    return mocker.patch.object(
        service,
        "_completion_with_tool_fallback",
        return_value=make_completion_response(
            text,
            prompt_tokens=usage.get("prompt_tokens", 40),
            completion_tokens=usage.get("completion_tokens", 60),
        ),
    )


class TestDossierParsing:
    """Only the lines the prompt asked for reach the planner."""

    def test_keeps_the_list_lines(self, make_service, mocker) -> None:
        """GIVEN a well-formed list WHEN researched THEN every line survives."""
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "- Winston Churchill — heavyset man in his late sixties, bald crown, "
            "jowled face, three-piece pinstripe suit and a cigar\n"
            "- the Blitz — London after dark in 1940, smoke and searchlights",
        )

        result = service.subject_dossier("churchill during the blitz", channel="#test")

        assert result.text.count("\n") == 1
        assert "Winston Churchill" in result.text
        assert "the Blitz" in result.text

    def test_drops_preamble_and_caveats(self, make_service, mocker) -> None:
        """GIVEN prose wrapped around the list WHEN parsed THEN only the list survives.

        The researcher is not a formatter. It opens with "Here's what I found"
        and closes with a note about accuracy unless stopped, and both would
        otherwise be read by the planner as instructions.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "Here are the subjects I could identify:\n"
            "- Amelia Earhart — slim woman in her thirties, cropped wavy hair, "
            "leather flying jacket\n"
            "\nNote that appearance details vary between sources.",
        )

        result = service.subject_dossier("amelia earhart taking off", channel="#test")

        assert result.text == (
            "- Amelia Earhart — slim woman in her thirties, cropped wavy hair, "
            "leather flying jacket"
        )

    def test_a_refusal_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN the researcher declines WHEN parsed THEN nothing is injected.

        A refusal pasted into the planner's system prompt is a model reading a
        "no" and agreeing with it. Dropping it means the request renders with
        less detail, which is the correct failure.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "I'm not able to provide detailed physical descriptions of real "
            "individuals. Let me know if I can help another way.",
        )

        result = service.subject_dossier("a real person", channel="#test")

        assert result.text == ""

    def test_none_answer_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN a request naming nobody real WHEN researched THEN no block."""
        service, _ = make_service()
        _answering(mocker, service, "NONE")

        result = service.subject_dossier(
            "a slow aerial shot over a pine forest at sunrise", channel="#test"
        )

        assert result.text == ""

    def test_bullet_characters_are_normalised(self, make_service, mocker) -> None:
        """GIVEN asterisk or bullet markers WHEN parsed THEN lines become dashes."""
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "* Elvis Presley — pompadour, white jumpsuit\n• Graceland — white-columned Memphis mansion",
        )

        result = service.subject_dossier("elvis at graceland", channel="#test")

        assert result.text.splitlines() == [
            "- Elvis Presley — pompadour, white jumpsuit",
            "- Graceland — white-columned Memphis mansion",
        ]

    def test_line_count_is_capped(self, make_service, mocker) -> None:
        """GIVEN more subjects than the cap WHEN parsed THEN the cap holds.

        The block rides in the planner's system prompt on every request; an
        unbounded one is an unbounded prompt.
        """
        service, _ = make_service()
        _answering(mocker, service, "\n".join(f"- Subject {i} — a description" for i in range(20)))

        result = service.subject_dossier("a crowd scene", channel="#test")

        assert len(result.text.splitlines()) == 8

    def test_character_budget_is_capped(self, make_service, mocker) -> None:
        """GIVEN one enormous line WHEN parsed THEN the budget is not exceeded."""
        service, _ = make_service()
        _answering(mocker, service, f"- Subject — {'x' * 4000}")

        result = service.subject_dossier("a subject", channel="#test")

        assert len(result.text) <= 1500


class TestDossierModelSelection:
    """Three keys, most specific first."""

    def test_prefers_subject_research_model(self, make_service, mocker) -> None:
        """GIVEN all three keys set WHEN researched THEN the research key wins."""
        service, _ = make_service(
            subjectResearchModel="gemini/gemini-2.5-flash",
            searchModel="gemini/gemini-2.0-flash",
            assistantModel="gemini/gemini-flash-latest",
        )
        call = _answering(mocker, service, "- A — b")

        result = service.subject_dossier("x", channel="#test")

        assert call.call_args.kwargs["model"] == "gemini/gemini-2.5-flash"
        assert result.model == "gemini/gemini-2.5-flash"

    def test_falls_back_to_search_model(self, make_service, mocker) -> None:
        """GIVEN no research key WHEN researched THEN searchModel is used."""
        service, _ = make_service(
            subjectResearchModel="",
            searchModel="gemini/gemini-2.0-flash",
            assistantModel="gemini/gemini-flash-latest",
        )
        call = _answering(mocker, service, "- A — b")

        service.subject_dossier("x", channel="#test")

        assert call.call_args.kwargs["model"] == "gemini/gemini-2.0-flash"

    def test_falls_back_to_assistant_model(self, make_service, mocker) -> None:
        """GIVEN neither key WHEN researched THEN assistantModel is used."""
        service, _ = make_service(
            subjectResearchModel="",
            searchModel="",
            assistantModel="gemini/gemini-flash-latest",
        )
        call = _answering(mocker, service, "- A — b")

        service.subject_dossier("x", channel="#test")

        assert call.call_args.kwargs["model"] == "gemini/gemini-flash-latest"


class TestDossierFailureAndSpend:
    """A failure costs detail, never the turn — and a call that happened is booked."""

    def test_provider_failure_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN the provider raises WHEN researched THEN text is empty, no raise.

        _grounded_completion catches and returns a JSON error string, which has
        no dash-prefixed line and so parses to nothing on its own. This asserts
        that arrangement holds rather than trusting it.
        """
        service, _ = make_service()
        mocker.patch.object(service, "_is_xai_model", return_value=False)
        mocker.patch.object(
            service, "_completion_with_tool_fallback", side_effect=RuntimeError("boom")
        )

        result = service.subject_dossier("churchill", channel="#test")

        assert result.text == ""
        assert result.cost == 0.0

    def test_usage_rides_back_for_booking(self, make_service, mocker) -> None:
        """GIVEN a successful call WHEN researched THEN tokens come back with it.

        The caller books its own usage row; it can only do that if the numbers
        survive the return.
        """
        service, _ = make_service()
        _answering(mocker, service, "- A — b", prompt_tokens=123, completion_tokens=45)

        result = service.subject_dossier("x", channel="#test")

        assert result.prompt_tokens == 123
        assert result.completion_tokens == 45

    def test_braces_in_the_request_survive(self, make_service, mocker) -> None:
        """GIVEN braces in the user's words WHEN researched THEN no format error.

        An IRC user can type '{'. Assembling the research prompt with
        str.format would raise KeyError on it and take the whole command down.
        """
        service, _ = make_service()
        call = _answering(mocker, service, "NONE")

        service.subject_dossier("draw {this} and {that}", channel="#test")

        sent = call.call_args.kwargs["messages"][0]["content"]
        assert "{this}" in sent
