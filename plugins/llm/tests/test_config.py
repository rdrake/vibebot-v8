"""Tests for LLM plugin configuration."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch


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
