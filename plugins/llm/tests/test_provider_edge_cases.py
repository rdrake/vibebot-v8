"""Tests for API provider edge cases.

These tests verify handling of provider-specific behaviors, error conditions,
and edge cases that may occur with different LLM providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
import pytest
from llm.service import LLMService

from .conftest import make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _stub_response():
    """A litellm-shaped chat-completion response, for kwarg-capture stubs."""
    return make_completion_response()


class TestProviderSpecificErrors:
    """Test handling of provider-specific error conditions."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_handles_timeout_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN timeout error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.Timeout(
                message="Request timed out", model="gpt-4", llm_provider="openai"
            ),
        )
        result = service.completion("test", command="ask")

        assert "timed out" in result.content.lower()
        assert result.error is not None

    def test_handles_rate_limit_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN rate limit error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                model="gpt-4",
                llm_provider="openai",
            ),
        )
        result = service.completion("test", command="ask")

        assert "rate limit" in result.content.lower()
        assert "Error" in result.content

    def test_handles_authentication_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN auth error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.AuthenticationError(
                message="Invalid API key",
                model="gpt-4",
                llm_provider="openai",
            ),
        )
        result = service.completion("test", command="ask")

        assert "api key" in result.content.lower() or "invalid" in result.content.lower()
        assert "Error" in result.content

    @pytest.mark.parametrize(
        ("error_class", "error_kwargs", "expected_words"),
        [
            (
                litellm.ContentPolicyViolationError,
                {"message": "Content violates policy", "model": "gpt-4", "llm_provider": "openai"},
                ["safety", "policy"],
            ),
            (
                litellm.BadRequestError,
                {
                    "message": "moderation_blocked: content was flagged",
                    "model": "gpt-4",
                    "llm_provider": "openai",
                },
                ["safety", "policy"],
            ),
            (
                litellm.BadRequestError,
                {
                    "message": "Blocked by safety system filters",
                    "model": "gpt-4",
                    "llm_provider": "openai",
                },
                ["safety", "policy"],
            ),
        ],
        ids=[
            "content_policy_violation",
            "bad_request_moderation_blocked",
            "bad_request_safety_system",
        ],
    )
    def test_handles_content_safety_errors(
        self,
        service: LLMService,
        mocker: MockerFixture,
        error_class,
        error_kwargs,
        expected_words,
    ) -> None:
        """GIVEN content safety error WHEN completing THEN returns user-friendly message."""
        mocker.patch("llm.service.litellm.completion", side_effect=error_class(**error_kwargs))
        result = service.completion("test", command="ask")
        assert any(word in result.content.lower() for word in expected_words)
        assert "Error" in result.content

    def test_handles_generic_api_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN generic API error WHEN completing THEN returns sanitized message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.APIError(
                message="API error with key sk-secret123456789",
                model="gpt-4",
                llm_provider="openai",
                status_code=500,
            ),
        )
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        # API key should be sanitized
        assert "sk-secret123456789" not in result.content

    def test_handles_unknown_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN unknown exception WHEN completing THEN returns generic message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=RuntimeError("Unexpected internal error"),
        )
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "Unexpected internal error" not in result.content  # Don't leak internals


class TestImageGenerationErrors:
    """Test error handling in image generation."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with draw config."""
        service, _ = make_service(
            imageApiKey="test-key", imageModel="dall-e-3", drawAutoRewriteMax=0
        )
        return service

    def test_handles_empty_response_data(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN empty response data WHEN generating image THEN returns content filter message."""
        mock_response = mocker.Mock()
        mock_response.data = []

        mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = service.image_generation("test prompt")

        assert "No image generated" in result.content
        assert "content safety" in result.content.lower()

    def test_handles_none_url_and_b64(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN response with neither URL nor base64 WHEN generating THEN returns error."""
        mock_response = mocker.Mock()
        mock_response.data = [mocker.Mock(url=None, b64_json=None)]

        mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = service.image_generation("test prompt")

        assert "No image generated" in result.content

    def test_handles_timeout_during_generation(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN timeout WHEN generating image THEN returns timeout message."""
        mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm.Timeout(
                message="Generation timed out", model="dall-e-3", llm_provider="openai"
            ),
        )
        result = service.image_generation("test prompt")

        assert "timed out" in result.content.lower()

    def test_sends_typing_done_on_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN error during generation WHEN with irc context THEN sends done indicator."""
        mock_irc = mocker.Mock()
        mock_irc.state = mocker.Mock()
        mock_irc.state.capabilities_ack = {"message-tags"}
        mock_irc.queueMsg = mocker.Mock()

        mock_msg = mocker.Mock()
        mock_msg.args = ("#test",)

        mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=Exception("API error"),
        )
        service.image_generation("test", irc=mock_irc, msg=mock_msg)

        # Should have sent typing indicators (active + done)
        assert mock_irc.queueMsg.call_count == 2
        # Last call should be typing=done
        last_msg = mock_irc.queueMsg.call_args_list[-1][0][0]
        assert last_msg.server_tags == {"+typing": "done"}


class TestPartialResponseHandling:
    """Test handling of partial or malformed responses."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_handles_empty_content_response(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with empty content WHEN completing THEN handles gracefully."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should return empty string, not crash
        assert result.content == ""

    def test_handles_none_content_response(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with None content WHEN completing THEN handles gracefully."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should normalize None to empty string
        assert result.content == ""

    def test_handles_no_choices_response(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN response with no choices WHEN completing THEN handles gracefully."""
        mock_response = mocker.Mock()
        mock_response.choices = []

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        # Should raise IndexError which gets caught
        result = service.completion("test", command="ask")

        assert "Error" in result.content

    def test_handles_whitespace_only_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with only whitespace WHEN completing THEN returns as-is."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "   \n\t  "
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should return the whitespace (sanitized)
        assert result.content.strip() == ""


class TestGeminiSpecificBehaviors:
    """Test Gemini-specific features and behaviors."""

    @pytest.fixture
    def gemini_service(self, make_service) -> LLMService:
        """Create service configured for Gemini."""
        service, _ = make_service(
            assistantApiKey="AIza-test-key", assistantModel="gemini/gemini-2.0-flash"
        )
        return service

    def test_gemini_tools_included_for_2x_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 2.x model WHEN getting tools THEN returns search tools."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-2.0-flash")
        assert tools is not None
        assert len(tools) == 2
        assert {"googleSearch": {}} in tools
        assert {"urlContext": {}} in tools

    def test_gemini_tools_included_for_25_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 2.5 model WHEN getting tools THEN returns search tools."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-2.5-pro")
        assert tools is not None
        assert len(tools) == 2

    def test_gemini_tools_not_included_for_15_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 1.5 model WHEN getting tools THEN returns None."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-1.5-flash")
        assert tools is None

    def test_gemini_tools_not_included_for_imagen(self, gemini_service: LLMService) -> None:
        """GIVEN Imagen model WHEN getting tools THEN returns None."""
        tools = gemini_service._get_gemini_tools("vertex_ai/imagen-4.0-generate-001")
        assert tools is None

    def test_gemini_safety_settings_applied(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini model WHEN getting safety settings THEN all categories set."""
        settings = gemini_service._get_safety_settings()

        expected_categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]

        assert len(settings) == len(expected_categories)
        for setting in settings:
            assert setting["category"] in expected_categories
            assert setting["threshold"] == "BLOCK_NONE"

    def test_completion_passes_gemini_tools(
        self, gemini_service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN Gemini model WHEN completing THEN tools passed to API."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = mocker.Mock()
            response.choices = [mocker.Mock()]
            response.choices[0].message = mocker.Mock()
            response.choices[0].message.content = "Response"
            return response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_call)
        gemini_service.completion("test", command="ask")

        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "safety_settings" in call_kwargs
        assert call_kwargs["safety_settings"] is not None


class TestOpenAISpecificBehaviors:
    """Test OpenAI-specific behaviors."""

    @pytest.fixture
    def openai_service(self, make_service) -> LLMService:
        """Create service configured for OpenAI."""
        service, _ = make_service(assistantApiKey="sk-test-key")
        return service

    def test_no_gemini_tools_for_openai(self, openai_service: LLMService) -> None:
        """GIVEN OpenAI model WHEN getting tools THEN returns None."""
        tools = openai_service._get_gemini_tools("gpt-4")
        assert tools is None

    def test_no_safety_settings_for_openai(
        self, openai_service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN OpenAI model WHEN completing THEN no safety_settings passed."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = mocker.Mock()
            response.choices = [mocker.Mock()]
            response.choices[0].message = mocker.Mock()
            response.choices[0].message.content = "Response"
            return response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_call)
        openai_service.completion("test", command="ask")

        assert call_kwargs.get("safety_settings") is None


class TestAnthropicSpecificBehaviors:
    """Test Anthropic-specific behaviors."""

    @pytest.fixture
    def anthropic_service(self, make_service) -> LLMService:
        """Create service configured for Anthropic."""
        service, _ = make_service(
            assistantApiKey="sk-ant-test-key", assistantModel="anthropic/claude-3-opus"
        )
        return service

    def test_no_gemini_tools_for_anthropic(self, anthropic_service: LLMService) -> None:
        """GIVEN Anthropic model WHEN getting tools THEN returns None."""
        tools = anthropic_service._get_gemini_tools("anthropic/claude-3-opus")
        assert tools is None


class TestSummarizeEdgeCases:
    """Test edge cases in the summarize method."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_summarize_handles_very_long_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN very long content WHEN summarizing THEN still works."""
        long_content = "x" * 100000  # 100KB of content

        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "Summary of large content"

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.summarize(long_content)

        assert result == "Summary of large content"

    def test_summarize_handles_unicode_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN content with unicode WHEN summarizing THEN handles correctly."""
        unicode_content = "Hello 世界 🌍 Привет мир"

        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "Summary with unicode"

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.summarize(unicode_content)

        assert result == "Summary with unicode"

    def test_summarize_handles_code_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN code content WHEN summarizing THEN works."""
        code_content = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "A recursive Fibonacci function"

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.summarize(code_content)

        assert result == "A recursive Fibonacci function"

    def test_summarize_cleans_multiline_response(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN multiline summary response WHEN summarizing THEN collapses to single line."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "Line one.\n\nLine two.\n\nLine three."

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.summarize("content")

        assert "\n" not in result
        assert result == "Line one. Line two. Line three."


class TestAPIKeyHandling:
    """Test API key handling across different scenarios."""

    def test_missing_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the ask model's provider WHEN completing THEN returns error.

        The default ``assistantModel`` is TEST_MODEL ("gpt-4"), whose provider
        is openai, so unsetting OPENAI_API_KEY is what makes the key missing.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service, _ = make_service()
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "OPENAI_API_KEY" in result.content

    def test_empty_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an empty key for the ask model's provider THEN returns error.

        An empty value must be treated exactly like an unset one, otherwise ""
        reaches litellm and the failure surfaces as a provider auth error.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "")
        service, _ = make_service()
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "OPENAI_API_KEY" in result.content

    def test_api_key_sanitized_in_errors(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN API error containing key WHEN handling THEN key sanitized.

        TEST_MODEL is "gpt-4", whose provider is openai, so `_sanitize` (now
        environment-backed) needs the key in OPENAI_API_KEY, not the registry.
        """
        fake_key = "sk-" + "x" * 25  # noqa: S105
        monkeypatch.setenv("OPENAI_API_KEY", fake_key)
        service, _ = make_service()

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception(f"Error with key {fake_key}"),
        )
        result = service.completion("test", command="ask")

        # Key should not appear in result
        assert fake_key not in str(result)


class TestBoundaryKeyResolution:
    """The key litellm receives is the one the model's provider variable holds."""

    def test_xai_model_gets_xai_key(self, make_service, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an xai model WHEN completing THEN XAI_API_KEY reaches litellm."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service(assistantModel="xai/grok-4.3")
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion("ask", model="xai/grok-4.3", messages=[], channel=None)
        assert seen["api_key"] == "xai-fake-key-for-tests-0000"

    def test_gemini_model_gets_gemini_key(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a gemini model WHEN completing THEN the gemini key, not the xai one."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="gemini/gemini-3-flash-preview", messages=[], channel=None
        )
        assert seen["api_key"] == "AIza-fake-key-for-tests-0000"

    def test_unmanaged_provider_passes_none(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a vertex_ai model WHEN completing THEN api_key is None.

        None is what makes LiteLLM fall back to ADC. Sending "" or a mapped
        key here would break service-account auth.
        """
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="vertex_ai/gemini-2.0-flash", messages=[], channel=None
        )
        assert seen["api_key"] is None

    def test_missing_key_error_names_provider_and_variable(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key WHEN building the error THEN it names both."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service()
        message = service._missing_key_error("xai/grok-4.3")
        assert message is not None
        assert "xai" in message
        assert "XAI_API_KEY" in message

    @pytest.mark.parametrize("model", ["vertex_ai/imagen-4.0-generate-001", "", "junk-model"])
    def test_unmanaged_models_have_no_error(self, make_service, model: str) -> None:
        """GIVEN an unmanaged model WHEN building the error THEN None.

        Unmanaged is not a failure — it is delegation to LiteLLM.
        """
        service, _ = make_service()
        assert service._missing_key_error(model) is None

    def test_error_never_contains_a_key(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a configured key WHEN building any error THEN the value is absent."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service()
        assert "xai-fake-key-for-tests-0000" not in (
            service._missing_key_error("xai/grok-4.3") or ""
        )
