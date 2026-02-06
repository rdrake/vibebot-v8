"""Tests for LLM plugin configuration."""

from __future__ import annotations

import logging
from io import StringIO
from unittest.mock import patch

import pytest
import supybot.registry as registry
from llm.config import ValidatedModelName


class TestValidatedModelName:
    """Test model name validation with litellm."""

    def setup_method(self) -> None:
        """Clear warned-models cache between tests."""
        ValidatedModelName._warned.clear()

    @staticmethod
    def _make(default: str = "") -> ValidatedModelName:
        """Create a ValidatedModelName instance for testing."""
        return ValidatedModelName(default, "Test model config")

    def test_known_model_accepted(self) -> None:
        """GIVEN a known model WHEN setValue called THEN accepts silently."""
        v = self._make()
        v.setValue("gemini/gemini-2.0-flash")
        assert v() == "gemini/gemini-2.0-flash"

    def test_empty_string_accepted(self) -> None:
        """GIVEN empty string WHEN setValue called THEN accepts (not configured)."""
        v = self._make("gemini/gemini-2.0-flash")
        v.setValue("")
        assert v() == ""

    def test_whitespace_stripped(self) -> None:
        """GIVEN model with surrounding whitespace WHEN setValue THEN strips it."""
        v = self._make()
        v.setValue("  gemini/gemini-2.0-flash  ")
        assert v() == "gemini/gemini-2.0-flash"

    def test_provider_prefixed_unknown_accepted_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN valid provider but unknown model WHEN setValue THEN accepts with warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("openai/my-custom-fine-tune-xyz")
        assert v() == "openai/my-custom-fine-tune-xyz"
        assert "not in litellm" in caplog.text.lower()

    def test_completely_unknown_rejected(self) -> None:
        """GIVEN completely unknown model WHEN setValue THEN raises InvalidRegistryValue."""
        v = self._make()
        with pytest.raises(registry.InvalidRegistryValue, match="Unknown model"):
            v.setValue("totally-bogus-xyz")

    def test_typo_with_valid_provider_warns_with_suggestions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN typo with valid provider WHEN setValue THEN accepts with warning + suggestions."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("gemini/gemni-2.0-flash")
        assert v() == "gemini/gemni-2.0-flash"
        assert "similar known models" in caplog.text.lower()

    def test_typo_without_provider_rejected_with_suggestions(self) -> None:
        """GIVEN typo without provider prefix WHEN setValue THEN rejected with suggestions."""
        v = self._make()
        with pytest.raises(registry.InvalidRegistryValue, match="Did you mean"):
            v.setValue("gpt-4o-mni")

    def test_unknown_no_match_shows_docs_link(self) -> None:
        """GIVEN unknown model with no close matches WHEN setValue THEN shows docs link."""
        v = self._make()
        with pytest.raises(registry.InvalidRegistryValue, match="docs.litellm.ai"):
            v.setValue("zzz-no-match-zzz")

    def test_default_ask_model_passes(self) -> None:
        """GIVEN default askModel value WHEN validated THEN passes."""
        v = self._make()
        v.setValue("gemini/gemini-flash-latest")
        assert v() == "gemini/gemini-flash-latest"

    def test_default_code_model_passes(self) -> None:
        """GIVEN default codeModel value WHEN validated THEN passes."""
        v = self._make()
        v.setValue("gemini/gemini-1.5-flash")
        assert v() == "gemini/gemini-1.5-flash"

    def test_default_draw_model_passes(self) -> None:
        """GIVEN default drawModel value WHEN validated THEN passes (warn OK)."""
        v = self._make()
        # vertex_ai/imagen-4.0-generate-001 may not be in model_list but
        # provider is valid, so it should be accepted (with warning)
        v.setValue("vertex_ai/imagen-4.0-generate-001")
        assert v() == "vertex_ai/imagen-4.0-generate-001"

    def test_suggest_models_returns_close_matches(self) -> None:
        """GIVEN a near-miss model name WHEN _suggest_models THEN returns suggestions."""
        suggestions = ValidatedModelName._suggest_models("gemini/gemni-2.0-flash")
        assert len(suggestions) > 0
        assert any("gemini" in s for s in suggestions)

    def test_warning_only_fires_once_per_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """GIVEN unknown model warned once WHEN setValue again THEN no duplicate warning."""
        v = self._make()
        with caplog.at_level(logging.WARNING, logger="supybot.plugins.LLM.config"):
            v.setValue("openai/my-dedup-test-model")
            caplog.clear()
            v.setValue("openai/my-dedup-test-model")
        assert caplog.text == ""


class TestConfigure:
    """Test plugin configuration wizard."""

    def test_configure_prints_setup_info(self) -> None:
        """GIVEN configure function WHEN called THEN prints setup information."""
        from llm.config import configure

        output = StringIO()
        with (
            patch("sys.stdout", output),
            patch("supybot.conf.registerPlugin"),
        ):
            configure(advanced=False)

        result = output.getvalue()
        assert "LLM Plugin Configuration" in result
        assert "API keys" in result
        assert "config plugins.LLM.askApiKey" in result

    def test_configure_registers_plugin(self) -> None:
        """GIVEN configure function WHEN called THEN registers plugin."""
        from llm.config import configure

        with (
            patch("sys.stdout", StringIO()),
            patch("supybot.conf.registerPlugin") as mock_register,
        ):
            configure(advanced=True)

        mock_register.assert_called_with("LLM", True)


class TestConfigValues:
    """Test configuration value registration."""

    def test_llm_plugin_registered(self) -> None:
        """GIVEN config module WHEN imported THEN LLM plugin is registered."""
        from llm import config

        assert hasattr(config, "LLM")

    def test_ask_api_key_is_private(self) -> None:
        """GIVEN askApiKey config WHEN accessed THEN marked as private."""
        import supybot.conf as conf

        # Access the registry value - private keys should not be logged
        from llm import config  # noqa: F401

        ask_key_value = conf.supybot.plugins.LLM.askApiKey
        # Private registry values have _private attribute
        assert ask_key_value._private is True

    def test_code_api_key_is_private(self) -> None:
        """GIVEN codeApiKey config WHEN accessed THEN marked as private."""
        import supybot.conf as conf
        from llm import config  # noqa: F401

        code_key_value = conf.supybot.plugins.LLM.codeApiKey
        assert code_key_value._private is True

    def test_draw_api_key_is_private(self) -> None:
        """GIVEN drawApiKey config WHEN accessed THEN marked as private."""
        import supybot.conf as conf
        from llm import config  # noqa: F401

        draw_key_value = conf.supybot.plugins.LLM.drawApiKey
        assert draw_key_value._private is True

    def test_database_path_registered(self) -> None:
        """GIVEN LLM config WHEN checking databasePath THEN exists with empty default."""
        from llm import config  # noqa: F811

        value = config.LLM.databasePath()
        assert value == ""
