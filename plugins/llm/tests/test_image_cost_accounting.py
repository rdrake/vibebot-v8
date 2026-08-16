"""Tests for image-spend accounting on models LiteLLM cannot price.

Every image model this bot has used is invisible to LiteLLM's cost map —
checked against upstream 1.93.0 and 1.97.0 on 2026-08-16, neither carries a
single grok-imagine entry — so `completion_cost` returns nothing and
IMAGE_COST_PER_IMAGE is the only price there is. Two ways that price was being
lost before it reached the usage table:

* **The tool boundary.** `_draw_for_assistant` knew the cost and returned a
  two-field `ToolCallbackResult` that could not carry it. From 2026-04-11, when
  draws moved off the @draw command onto the generate_image tool, until
  2026-08-16, every image the chat model generated was booked at $0.00. The
  prod usage table shows it plainly: draw rows carry `xai/grok-imagine-image-pro`
  at $0.069 average through 2026-04-11, then switch to *chat* model names at
  ~$0.001 — the text cost of the turn with the image cost missing.

* **Refused generations.** A moderated refusal still bills; xAI says so in the
  error (`'usage': {'cost_in_usd_ticks': 200000000}`). Nothing costed those at
  all, on any path. With draws refused better than half the time, that is the
  larger of the two leaks per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolCallbackResult, ToolResult
from llm.service import IMAGE_COST_PER_IMAGE, LLMService

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

MINTED = "https://paste.boxlabs.uk/img/img_6a81079496b1b.jpg"
IMAGE_MODEL = "xai/grok-imagine-image"
PRICE = IMAGE_COST_PER_IMAGE[IMAGE_MODEL]

XAI_MODERATION_ERROR = (
    "litellm.BadRequestError: XaiException - Error code: 400 - "
    "{'code': 'imagine:content-moderated', 'error': 'Generated image rejected by "
    "content moderation.', 'usage': {'cost_in_usd_ticks': 200000000}}"
)


class TestUnknownModelPricing:
    """`_image_price`: the table is the price, and a miss must be audible."""

    def test_known_model_is_priced_from_the_table(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _ = make_service()
        assert service._image_price(IMAGE_MODEL) == PRICE

    def test_unknown_model_warns_instead_of_silently_costing_zero(
        self,
        make_service,  # type: ignore[no-untyped-def]
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A model swap must not quietly zero the books again.

        Returning 0.0 is the only honest answer without a price, but doing it
        silently is what made four months of draw spend unreadable.
        """
        service, _ = make_service()
        with caplog.at_level("WARNING"):
            assert service._image_price("xai/grok-imagine-image-ultra") == 0.0
        assert "IMAGE_COST_PER_IMAGE" in caplog.text
        assert "xai/grok-imagine-image-ultra" in caplog.text

    def test_unknown_model_warns_once_not_per_call(
        self,
        make_service,  # type: ignore[no-untyped-def]
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Per-call warnings on a busy channel are scrolled past, not read."""
        service, _ = make_service()
        with caplog.at_level("WARNING"):
            for _ in range(5):
                service._image_price("some/unpriced-model")
        assert caplog.text.count("some/unpriced-model") == 1


class TestBilledRefusals:
    """A refusal the provider charged for has to reach the books."""

    def test_moderation_refusal_is_costed(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _ = make_service()
        error = Exception(XAI_MODERATION_ERROR)
        assert service._billed_failure_cost(error, IMAGE_MODEL) == PRICE

    def test_refusal_with_no_usage_block_is_free(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """Not every provider bills for a block, so don't invent a charge.

        The usage block is read as a yes/no signal that the call was charged;
        without one the honest answer is zero.
        """
        service, _ = make_service()
        error = Exception("litellm.BadRequestError: blocked by safety filter")
        assert service._billed_failure_cost(error, IMAGE_MODEL) == 0.0

    def test_refused_generation_bills_through_image_generation(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end: a refused draw returns an error AND a non-zero cost."""
        import litellm as litellm_module

        service, _ = make_service(
            imageModel=IMAGE_MODEL,
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=0,
            httpUrlBase="https://example.com/llm",
        )
        mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm_module.BadRequestError(
                message=XAI_MODERATION_ERROR, model=IMAGE_MODEL, llm_provider="xai"
            ),
        )

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert result.cost == PRICE

    def test_every_attempt_in_the_rewrite_loop_is_billed(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """A refusal plus a refused rewrite is two charges, not one."""
        import litellm as litellm_module

        service, _ = make_service(
            imageModel=IMAGE_MODEL,
            assistantModel="gemini/gemini-flash-latest",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=[
                litellm_module.BadRequestError(
                    message=XAI_MODERATION_ERROR, model=IMAGE_MODEL, llm_provider="xai"
                ),
                litellm_module.BadRequestError(
                    message=XAI_MODERATION_ERROR, model=IMAGE_MODEL, llm_provider="xai"
                ),
            ],
        )
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert result.cost == pytest.approx(PRICE * 2)


class TestCostReachesTheUsageRow:
    """The leak that mattered: cost dropped at the tool boundary."""

    def test_callback_result_carries_usage(self) -> None:
        """Defaults keep every non-spending callback working unchanged."""
        assert ToolCallbackResult(True, "done").cost == 0.0
        spent = ToolCallbackResult(True, MINTED, prompt_tokens=3, completion_tokens=0, cost=0.02)
        assert (spent.prompt_tokens, spent.cost) == (3, 0.02)

    def test_draw_tool_returns_a_costed_result(self, mocker: MockerFixture) -> None:
        """The handler must hand back a ToolResult, not a bare string.

        A bare string is what the executor wraps at cost=0.0, which is exactly
        how the spend disappeared.
        """
        from llm.assistant import AssistantToolExecutor

        executor = object.__new__(AssistantToolExecutor)
        executor._draw_fn = lambda _p: ToolCallbackResult(True, MINTED, cost=PRICE)

        result = AssistantToolExecutor._tool_generate_image(executor, {"prompt": "a party"})

        assert isinstance(result, ToolResult)
        assert result.cost == PRICE
        assert MINTED in result.content

    def test_failed_draw_still_reports_its_cost(self, mocker: MockerFixture) -> None:
        """Refusals are the majority case; they cannot be the free case."""
        from llm.assistant import AssistantToolExecutor

        executor = object.__new__(AssistantToolExecutor)
        executor._draw_fn = lambda _p: ToolCallbackResult(False, "blocked", cost=PRICE)

        result = AssistantToolExecutor._tool_generate_image(executor, {"prompt": "a party"})

        assert isinstance(result, ToolResult)
        assert result.cost == PRICE

    def test_assistant_turn_totals_include_the_image(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end through the chat loop, which is how draws now run.

        This is the number that reaches db.log_usage. Before the fix it came
        back as the text cost alone.
        """
        service, _ = make_service(assistantModel="gpt-4", httpUrlBase="https://irc.rdrake.org/llm")
        mocker.patch(
            "llm.service.litellm.completion",
            return_value=make_completion_response(
                None, tool_calls=[make_tool_call("generate_image", {"prompt": "a party"})]
            ),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.assistant_completion(
            prompt="draw bunga bunga party",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p: ToolCallbackResult(True, MINTED, cost=PRICE),
        )

        assert result.content == MINTED
        assert result.cost == pytest.approx(PRICE)


class TestLiteLLMStillCannotPriceThese:
    """Pins why the local table exists, so nobody deletes it on a version bump."""

    @pytest.mark.parametrize("model", sorted(IMAGE_COST_PER_IMAGE))
    def test_model_is_absent_from_litellms_cost_map(self, model: str) -> None:
        """If this ever fails, LiteLLM gained a price and the entry can go.

        Checked against 1.93.0 (pinned) and 1.97.0 (latest, 2026-08-16): both
        ship 40 xai models, none of them image or video.
        """
        import litellm

        assert model not in litellm.model_cost
        assert model.split("/", 1)[-1] not in litellm.model_cost


def test_service_price_lookup_is_used_for_successful_generations(
    make_service, mocker: MockerFixture
) -> None:
    """A delivered image is priced from the table, not left at LiteLLM's zero."""
    service, _ = make_service(
        imageModel=IMAGE_MODEL,
        drawAutoRewriteMax=0,
        httpUrlBase="https://example.com/llm",
    )
    response = mocker.Mock()
    response.data = [mocker.Mock(url="https://provider.com/image.png", b64_json=None)]
    response.usage = mocker.Mock(prompt_tokens=0, completion_tokens=0)
    mocker.patch("llm.service.litellm.image_generation", return_value=response)
    mocker.patch.object(
        service, "_download_and_save_image", return_value="https://example.com/llm/img_a.png"
    )

    result = service.image_generation("a party")

    assert result.error is None
    assert result.cost == PRICE


class TestLLMServiceHelpersExist:
    """Guards the names the rest of this file leans on."""

    def test_helpers_are_public_enough_to_test(self) -> None:
        assert hasattr(LLMService, "_image_price")
        assert hasattr(LLMService, "_billed_failure_cost")
