"""Falling back to a self-hosted endpoint when the provider refuses a draw.

Measured 2026-08-19: xAI refused 27% of first calls, and the rewriter recovered
6 of 6 — but recovery means the user gets a picture of a SOFTENED prompt, marked
🔁, for ~$0.02 and a second billed call. A self-hosted box needs no persuading,
so the fallback delivers what was actually asked for, free.

The prompt sent to the fallback is therefore the ORIGINAL one. Sending it a
rewrite would spend the fidelity this exists to buy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.service import LLMService

_FALLBACK_BASE = "http://video.example.com:14205/v1"
_FALLBACK_MODEL = "openai//work/models/MiniMax-H3/FL2VA"
_KEY = "not-a-real-token-for-tests"


@pytest.fixture
def fallback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)
    monkeypatch.setenv("XAI_API_KEY", _KEY)


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    overrides.setdefault("imageModel", "xai/grok-imagine")
    overrides.setdefault("drawAutoRewriteMax", 0)
    overrides.setdefault("imageFallbackApiBase", _FALLBACK_BASE)
    overrides.setdefault("imageFallbackModel", _FALLBACK_MODEL)
    return make_service(**overrides)


def _drawn(mocker, url: str = "http://host/img_fb.png", model: str = _FALLBACK_MODEL):
    from llm.service import ImageResult

    return ImageResult(content=url, url=url, model=model, cost=0.0)


class TestRefusalFallsBack:
    """A refused draw goes to the box rather than back to the user as an error."""

    def test_refusal_is_served_by_the_fallback(self, make_service, fallback_env, mocker) -> None:
        """GIVEN the provider refuses WHEN a fallback is set THEN an image still lands."""
        service, _ = _service(make_service)
        attempt = mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, _drawn(mocker)]
        )

        result = service.image_generation("two men boxing")

        assert result.error is None
        assert result.content == "http://host/img_fb.png"
        assert attempt.call_count == 2

    def test_fallback_gets_the_original_prompt(self, make_service, fallback_env, mocker) -> None:
        """GIVEN a refusal WHEN the fallback runs THEN it draws what was asked.

        The whole point of preferring the box over a rewrite: no softening.
        """
        service, _ = _service(make_service)
        attempt = mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, _drawn(mocker)]
        )

        service.image_generation("two men boxing")

        assert attempt.call_args_list[1].args[0] == "two men boxing"

    def test_fallback_uses_its_own_model(self, make_service, fallback_env, mocker) -> None:
        """GIVEN a fallback model WHEN it runs THEN that model is the one called."""
        service, _ = _service(make_service)
        attempt = mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, _drawn(mocker)]
        )

        service.image_generation("two men boxing")

        assert attempt.call_args_list[1].args[1] == _FALLBACK_MODEL
        assert attempt.call_args_list[1].kwargs["fallback"] is True

    def test_delivered_image_is_not_marked_reworded(
        self, make_service, fallback_env, mocker
    ) -> None:
        """GIVEN a fallback draw WHEN it lands THEN no rewrite marker rides along.

        🔁 tells the user their words were changed. The fallback did not change
        them, so claiming otherwise would be a lie about their own prompt.
        """
        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, _drawn(mocker)]
        )

        result = service.image_generation("two men boxing")

        assert not result.rewritten_prompt

    def test_the_refusal_is_still_recorded(self, make_service, fallback_env, mocker) -> None:
        """GIVEN a refusal recovered by the fallback THEN the refusal is still booked.

        content_blocked rows are how the refusal rate is measured at all; a
        recovery that hides them would blind the thing it was built from.
        """
        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, _drawn(mocker)]
        )

        result = service.image_generation("two men boxing")

        assert len(result.blocked_attempts) == 1
        assert result.blocked_attempts[0].prompt == "two men boxing"


class TestFallbackAfterRewrites:
    """With rewrites enabled the fallback is the last resort, not the first."""

    def test_rewrites_run_first_then_the_fallback(self, make_service, fallback_env, mocker) -> None:
        """GIVEN rewrites are enabled WHEN all are refused THEN the fallback still saves it."""
        service, _ = _service(make_service, drawAutoRewriteMax=1)
        mocker.patch.object(
            service,
            "_rewrite_prompt_for_safety",
            return_value=("two athletes sparring", 10, 5, 0.001),
        )
        attempt = mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, None, _drawn(mocker)]
        )

        result = service.image_generation("two men boxing")

        assert result.error is None
        assert attempt.call_count == 3
        # Rewrite tried the softened prompt; the fallback went back to the original.
        assert attempt.call_args_list[1].args[0] == "two athletes sparring"
        assert attempt.call_args_list[2].args[0] == "two men boxing"


class TestFallbackAbsentOrBroken:
    """Nothing configured, or a fallback that also fails, must not make it worse."""

    def test_no_fallback_configured_keeps_the_old_error(
        self, make_service, fallback_env, mocker
    ) -> None:
        """GIVEN no fallback WHEN refused THEN the user gets the refusal message."""
        service, _ = _service(make_service, imageFallbackApiBase="")
        attempt = mocker.patch.object(service, "_attempt_image_generation", return_value=None)

        result = service.image_generation("two men boxing")

        assert result.error is not None
        assert attempt.call_count == 1

    def test_failing_fallback_reports_the_refusal_not_a_second_error(
        self, make_service, fallback_env, mocker
    ) -> None:
        """GIVEN the fallback also fails WHEN it is tried THEN the refusal is what is said.

        The user asked for a picture and did not get one; which of two backends
        disappointed them is not their problem.
        """
        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_attempt_image_generation", side_effect=[None, OSError("box is down")]
        )

        result = service.image_generation("two men boxing")

        assert result.error is not None
        assert "box is down" not in result.content


class TestPrimaryEndpointIsChannelScoped:
    """imageApiBase is a channel value, so the draw path has to pass the channel."""

    def test_first_attempt_receives_the_channel(self, make_service, fallback_env, mocker) -> None:
        """GIVEN a channel message WHEN drawing THEN the endpoint lookup can see it.

        Without this a per-channel imageApiBase silently resolves to the global
        value: registryValue(key, None) is a valid call that returns the wrong
        answer rather than raising.
        """
        service, _ = _service(make_service, imageFallbackApiBase="")
        attempt = mocker.patch.object(
            service, "_attempt_image_generation", return_value=_drawn(mocker)
        )
        msg = mocker.Mock()
        msg.args = ("#chan", "draw a cat")

        service.image_generation("a cat", msg=msg)

        assert attempt.call_args.args[3] == "#chan"
