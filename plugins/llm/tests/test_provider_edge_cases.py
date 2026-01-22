"""Tests for API provider edge cases.

These tests verify handling of provider-specific behaviors, error conditions,
and edge cases that may occur with different LLM providers.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import litellm
import pytest
from llm.service import LLMService


class TestProviderSpecificErrors:
    """Test handling of provider-specific error conditions."""

    @pytest.fixture
    def service(self) -> LLMService:
        """Create service with mock plugin."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "test-key",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_handles_timeout_error(self, service: LLMService) -> None:
        """GIVEN timeout error WHEN completing THEN returns user-friendly message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=litellm.Timeout(
                message="Request timed out", model="gpt-4", llm_provider="openai"
            ),
        ):
            result = service.completion("test", command="ask")

        assert "timed out" in result.content.lower()
        assert "Error" in result.content

    def test_handles_rate_limit_error(self, service: LLMService) -> None:
        """GIVEN rate limit error WHEN completing THEN returns user-friendly message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                model="gpt-4",
                llm_provider="openai",
            ),
        ):
            result = service.completion("test", command="ask")

        assert "rate limit" in result.content.lower()
        assert "Error" in result.content

    def test_handles_authentication_error(self, service: LLMService) -> None:
        """GIVEN auth error WHEN completing THEN returns user-friendly message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=litellm.AuthenticationError(
                message="Invalid API key",
                model="gpt-4",
                llm_provider="openai",
            ),
        ):
            result = service.completion("test", command="ask")

        assert "api key" in result.content.lower() or "invalid" in result.content.lower()
        assert "Error" in result.content

    def test_handles_content_policy_violation(self, service: LLMService) -> None:
        """GIVEN content policy error WHEN completing THEN returns user-friendly message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=litellm.ContentPolicyViolationError(
                message="Content violates policy",
                model="gpt-4",
                llm_provider="openai",
            ),
        ):
            result = service.completion("test", command="ask")

        assert "safety" in result.content.lower() or "policy" in result.content.lower()
        assert "Error" in result.content

    def test_handles_generic_api_error(self, service: LLMService) -> None:
        """GIVEN generic API error WHEN completing THEN returns sanitized message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=litellm.APIError(
                message="API error with key sk-secret123456789",
                model="gpt-4",
                llm_provider="openai",
                status_code=500,
            ),
        ):
            result = service.completion("test", command="ask")

        assert "Error" in result.content
        # API key should be sanitized
        assert "sk-secret123456789" not in result.content

    def test_handles_unknown_exception(self, service: LLMService) -> None:
        """GIVEN unknown exception WHEN completing THEN returns generic message."""
        with patch(
            "llm.service.litellm.completion",
            side_effect=RuntimeError("Unexpected internal error"),
        ):
            result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "Unexpected internal error" not in result.content  # Don't leak internals


class TestImageGenerationErrors:
    """Test error handling in image generation."""

    @pytest.fixture
    def service(self) -> LLMService:
        """Create service with mock plugin."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "drawApiKey": "test-key",
                "drawModel": "dall-e-3",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_handles_empty_response_data(self, service: LLMService) -> None:
        """GIVEN empty response data WHEN generating image THEN returns content filter message."""
        mock_response = Mock()
        mock_response.data = []

        with patch("llm.service.litellm.image_generation", return_value=mock_response):
            result = service.image_generation("test prompt")

        assert "No image generated" in result
        assert "content safety" in result.lower()

    def test_handles_none_url_and_b64(self, service: LLMService) -> None:
        """GIVEN response with neither URL nor base64 WHEN generating THEN returns error."""
        mock_response = Mock()
        mock_response.data = [Mock(url=None, b64_json=None)]

        with patch("llm.service.litellm.image_generation", return_value=mock_response):
            result = service.image_generation("test prompt")

        assert "No image generated" in result

    def test_handles_timeout_during_generation(self, service: LLMService) -> None:
        """GIVEN timeout WHEN generating image THEN returns timeout message."""
        with patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm.Timeout(
                message="Generation timed out", model="dall-e-3", llm_provider="openai"
            ),
        ):
            result = service.image_generation("test prompt")

        assert "timed out" in result.lower()

    def test_sends_typing_done_on_error(self, service: LLMService) -> None:
        """GIVEN error during generation WHEN with irc context THEN sends done indicator."""
        mock_irc = Mock()
        mock_irc.state = Mock()
        mock_irc.state.capabilities_ack = {"message-tags"}
        mock_irc.queueMsg = Mock()

        mock_msg = Mock()
        mock_msg.args = ("#test",)

        with patch(
            "llm.service.litellm.image_generation",
            side_effect=Exception("API error"),
        ):
            service.image_generation("test", irc=mock_irc, msg=mock_msg)

        # Should have sent typing indicators (active + done)
        assert mock_irc.queueMsg.call_count == 2
        # Last call should be typing=done
        last_msg = mock_irc.queueMsg.call_args_list[-1][0][0]
        assert last_msg.server_tags == {"+typing": "done"}


class TestPartialResponseHandling:
    """Test handling of partial or malformed responses."""

    @pytest.fixture
    def service(self) -> LLMService:
        """Create service with mock plugin."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "test-key",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_handles_empty_content_response(self, service: LLMService) -> None:
        """GIVEN response with empty content WHEN completing THEN handles gracefully."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = None

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.completion("test", command="ask")

        # Should return empty string, not crash
        assert result.content == ""

    def test_handles_none_content_response(self, service: LLMService) -> None:
        """GIVEN response with None content WHEN completing THEN handles gracefully."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.completion("test", command="ask")

        # Should handle None gracefully
        assert result.content is None or result.content == ""

    def test_handles_no_choices_response(self, service: LLMService) -> None:
        """GIVEN response with no choices WHEN completing THEN handles gracefully."""
        mock_response = Mock()
        mock_response.choices = []

        with patch("llm.service.litellm.completion", return_value=mock_response):
            # Should raise IndexError which gets caught
            result = service.completion("test", command="ask")

        assert "Error" in result.content

    def test_handles_whitespace_only_content(self, service: LLMService) -> None:
        """GIVEN response with only whitespace WHEN completing THEN returns as-is."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "   \n\t  "
        mock_response.choices[0].message.tool_calls = None

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.completion("test", command="ask")

        # Should return the whitespace (sanitized)
        assert result.content.strip() == ""


class TestGeminiSpecificBehaviors:
    """Test Gemini-specific features and behaviors."""

    @pytest.fixture
    def gemini_service(self) -> LLMService:
        """Create service configured for Gemini."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "AIza-test-key",
                "askModel": "gemini/gemini-2.0-flash",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

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

    def test_completion_passes_gemini_tools(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini model WHEN completing THEN tools passed to API."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = "Response"
            return response

        with patch("llm.service.litellm.completion", side_effect=capture_call):
            gemini_service.completion("test", command="ask")

        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "safety_settings" in call_kwargs
        assert call_kwargs["safety_settings"] is not None


class TestOpenAISpecificBehaviors:
    """Test OpenAI-specific behaviors."""

    @pytest.fixture
    def openai_service(self) -> LLMService:
        """Create service configured for OpenAI."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "sk-test-key",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_no_gemini_tools_for_openai(self, openai_service: LLMService) -> None:
        """GIVEN OpenAI model WHEN getting tools THEN returns None."""
        tools = openai_service._get_gemini_tools("gpt-4")
        assert tools is None

    def test_no_safety_settings_for_openai(self, openai_service: LLMService) -> None:
        """GIVEN OpenAI model WHEN completing THEN no safety_settings passed."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = "Response"
            return response

        with patch("llm.service.litellm.completion", side_effect=capture_call):
            openai_service.completion("test", command="ask")

        assert call_kwargs.get("safety_settings") is None


class TestAnthropicSpecificBehaviors:
    """Test Anthropic-specific behaviors."""

    @pytest.fixture
    def anthropic_service(self) -> LLMService:
        """Create service configured for Anthropic."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "sk-ant-test-key",
                "askModel": "anthropic/claude-3-opus",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_no_gemini_tools_for_anthropic(self, anthropic_service: LLMService) -> None:
        """GIVEN Anthropic model WHEN getting tools THEN returns None."""
        tools = anthropic_service._get_gemini_tools("anthropic/claude-3-opus")
        assert tools is None


class TestSummarizeEdgeCases:
    """Test edge cases in the summarize method."""

    @pytest.fixture
    def service(self) -> LLMService:
        """Create service with mock plugin."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "askApiKey": "test-key",
                "askModel": "gpt-4",
                "timeout": 30,
            }.get(key)
        )
        return LLMService(mock_plugin)

    def test_summarize_handles_very_long_content(self, service: LLMService) -> None:
        """GIVEN very long content WHEN summarizing THEN still works."""
        long_content = "x" * 100000  # 100KB of content

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Summary of large content"

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.summarize(long_content)

        assert result == "Summary of large content"

    def test_summarize_handles_unicode_content(self, service: LLMService) -> None:
        """GIVEN content with unicode WHEN summarizing THEN handles correctly."""
        unicode_content = "Hello 世界 🌍 Привет мир"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Summary with unicode"

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.summarize(unicode_content)

        assert result == "Summary with unicode"

    def test_summarize_handles_code_content(self, service: LLMService) -> None:
        """GIVEN code content WHEN summarizing THEN works."""
        code_content = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "A recursive Fibonacci function"

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.summarize(code_content)

        assert result == "A recursive Fibonacci function"

    def test_summarize_cleans_multiline_response(self, service: LLMService) -> None:
        """GIVEN multiline summary response WHEN summarizing THEN collapses to single line."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Line one.\n\nLine two.\n\nLine three."

        with patch("llm.service.litellm.completion", return_value=mock_response):
            result = service.summarize("content")

        assert "\n" not in result
        assert result == "Line one. Line two. Line three."


class TestAPIKeyHandling:
    """Test API key handling across different scenarios."""

    def test_missing_ask_key_returns_error(self) -> None:
        """GIVEN no ask API key WHEN completing THEN returns error."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": None,
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )

        service = LLMService(mock_plugin)
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "API key not configured" in result.content

    def test_empty_ask_key_returns_error(self) -> None:
        """GIVEN empty ask API key WHEN completing THEN returns error."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )

        service = LLMService(mock_plugin)
        result = service.completion("test", command="ask")

        assert "Error" in result.content

    def test_api_key_sanitized_in_errors(self) -> None:
        """GIVEN API error containing key WHEN handling THEN key sanitized."""
        mock_plugin = Mock()
        mock_plugin.log = Mock()
        mock_plugin.registryValue = Mock(
            side_effect=lambda key, channel=None: {
                "maxPromptLength": 10000,
                "commandPrefixes": [".", "/"],
                "askApiKey": "sk-secret12345678901234567890",
                "askModel": "gpt-4",
                "askSystemPrompt": "You are helpful.",
                "timeout": 30,
            }.get(key)
        )

        service = LLMService(mock_plugin)

        with patch(
            "llm.service.litellm.completion",
            side_effect=Exception("Error with key sk-secret12345678901234567890"),
        ):
            result = service.completion("test", command="ask")

        # Key should not appear in result
        assert "sk-secret12345678901234567890" not in result
