"""@draw against the self-hosted box instead of a hosted provider.

The video box also serves ``/v1/images/generations``, OpenAI-shaped, behind the
same bearer token. Pointing @draw at it is one config switch plus the plumbing
LiteLLM needs to be aimed somewhere other than the provider it infers from the
model name: an api_base, a key that is not the provider's, and a step count,
which is the difference between a ~30s draw and a ~94s one.

Nothing here changes what happens when ``imageApiBase`` is empty — that is the
hosted path everyone is using today, and it stays byte-for-byte the same call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.service import LLMService

_BASE = "http://video.example.com:14205/v1"
_SELF_MODEL = "openai//work/models/MiniMax-H3/FL2VA"
_KEY = "not-a-real-token-for-tests"


@pytest.fixture
def animate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)


def _image_response(mocker):
    """A LiteLLM-shaped image response carrying b64 data."""
    datum = mocker.Mock(spec=["b64_json"])
    datum.b64_json = "aW1hZ2U="
    resp = mocker.Mock()
    resp.data = [datum]
    resp.id = "img-1"
    return resp


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    return make_service(**overrides)


class TestSelfHostedImageRouting:
    """With imageApiBase set, the call goes to the box, not to the provider."""

    def test_api_base_is_sent(self, make_service, animate_env, mocker) -> None:
        """GIVEN imageApiBase WHEN drawing THEN LiteLLM is aimed at it."""
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", _SELF_MODEL, 120)

        assert call.call_args.kwargs["api_base"] == _BASE

    def test_box_token_is_used_not_the_provider_key(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a self-hosted base WHEN drawing THEN the animate token authenticates it.

        The model is named openai/... so LiteLLM would otherwise demand
        OPENAI_API_KEY, which is a different credential for a different
        service. Provider-scoped keys are deliberate; this must not blur them.
        """
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", _SELF_MODEL, 120)

        assert call.call_args.kwargs["api_key"] == _KEY

    def test_step_count_rides_extra_body(self, make_service, animate_env, mocker) -> None:
        """GIVEN a step count WHEN drawing THEN it reaches the box.

        num_inference_steps is not an OpenAI image parameter, so LiteLLM drops
        it unless it travels in extra_body. Measured on the box: 8 steps ~27s,
        25 steps ~94s, so a silently dropped value is a 3x slower draw.
        """
        service, _ = _service(
            make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL, imageSteps=8
        )
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", _SELF_MODEL, 120)

        assert call.call_args.kwargs["extra_body"]["num_inference_steps"] == 8

    def test_size_is_sent_when_configured(self, make_service, animate_env, mocker) -> None:
        """GIVEN imageSize WHEN drawing THEN the box is told the geometry."""
        service, _ = _service(
            make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL, imageSize="1024x576"
        )
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", _SELF_MODEL, 120)

        assert call.call_args.kwargs["size"] == "1024x576"

    def test_xai_tuning_does_not_leak_to_the_box(self, make_service, animate_env, mocker) -> None:
        """GIVEN a self-hosted model WHEN drawing THEN no xAI-only knobs are sent."""
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", _SELF_MODEL, 120)

        for leaked in ("aspect_ratio", "quality", "resolution"):
            assert leaked not in call.call_args.kwargs


class TestHostedPathUnchanged:
    """The provider path is what everyone is drawing on today. Don't move it."""

    def test_no_api_base_when_unset(self, make_service, mocker, monkeypatch) -> None:
        """GIVEN no imageApiBase WHEN drawing THEN nothing is aimed anywhere new."""
        monkeypatch.setenv("XAI_API_KEY", _KEY)
        service, _ = _service(make_service, imageApiBase="")
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", "xai/grok-imagine", 120)

        assert "api_base" not in call.call_args.kwargs
        assert "extra_body" not in call.call_args.kwargs

    def test_xai_tuning_still_applies(self, make_service, mocker, monkeypatch) -> None:
        """GIVEN an xAI model WHEN drawing THEN its tuning knobs still ride along."""
        monkeypatch.setenv("XAI_API_KEY", _KEY)
        service, _ = _service(make_service, imageApiBase="")
        call = mocker.patch("litellm.image_generation", return_value=_image_response(mocker))
        mocker.patch.object(service, "save_image_to_http", return_value="http://host/img_1.png")

        service._attempt_image_generation("a cat", "xai/grok-imagine", 120)

        assert call.call_args.kwargs["aspect_ratio"] == "9:16"


class TestSelfHostedKeyGuard:
    """A self-hosted base must not trip the managed-provider key check."""

    def test_missing_openai_key_does_not_block_the_box(
        self, make_service, animate_env, monkeypatch
    ) -> None:
        """GIVEN no OPENAI_API_KEY but a self-hosted base THEN the draw is allowed.

        Without this the command refuses before it calls anything, telling the
        user to set a credential for a service it is not talking to.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)

        assert service._missing_image_key_error(_SELF_MODEL, None) is None

    def test_missing_box_token_is_reported(self, make_service, monkeypatch) -> None:
        """GIVEN a self-hosted base and no token THEN it says which one is missing."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)

        error = service._missing_image_key_error(_SELF_MODEL, None)

        assert error and "ANIMATE_API_KEY" in error

    def test_hosted_model_still_checks_its_provider(self, make_service, monkeypatch) -> None:
        """GIVEN no base and a missing provider key THEN the old guard still fires."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = _service(make_service, imageApiBase="")

        error = service._missing_image_key_error("xai/grok-imagine", None)

        assert error and "XAI_API_KEY" in error


class TestSelfHostedIsFreeNotUnpriced:
    """$0.00 for our own hardware is a fact, not a missing price."""

    def test_self_hosted_draw_does_not_warn_about_price(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a self-hosted draw WHEN it is costed THEN no unpriced warning fires.

        The warning exists to catch a paid model whose spend is silently
        vanishing — the bug that made draw cost unreadable for four months. A
        self-hosted box genuinely costs nothing per image, so firing it here
        would train everyone to ignore the one warning that matters. Prod keeps
        WARNING and above, so this would also be permanent log noise.
        """
        service, _ = _service(make_service, imageApiBase=_BASE, imageModel=_SELF_MODEL)
        warn = mocker.patch.object(service.log, "warning")

        assert service._image_price(_SELF_MODEL, self_hosted=True) == 0.0
        warn.assert_not_called()

    def test_unpriced_hosted_model_still_warns(self, make_service, mocker) -> None:
        """GIVEN an unpriced hosted model WHEN costed THEN the warning still fires."""
        service, _ = _service(make_service, imageApiBase="")
        warn = mocker.patch.object(service.log, "warning")

        assert service._image_price("xai/some-new-imagen") == 0.0
        warn.assert_called_once()
