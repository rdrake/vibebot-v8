"""Tests for LLM plugin.

These tests verify the plugin structure, imports, and command registration
without requiring a full Limnoria runtime environment.
"""

from __future__ import annotations


class TestPluginImport:
    """Test plugin module can be imported and has expected structure."""

    def test_plugin_module_imports(self) -> None:
        """GIVEN llm.plugin module WHEN imported THEN no errors."""
        from llm import plugin

        assert plugin is not None

    def test_plugin_class_exists(self) -> None:
        """GIVEN llm.plugin module WHEN accessing Class THEN plugin class found."""
        from llm.plugin import Class

        assert Class is not None
        assert Class.__name__ == "LLM"

    def test_plugin_inherits_from_callbacks(self) -> None:
        """GIVEN LLM class WHEN checking inheritance THEN inherits from Plugin."""
        # Check that LLM inherits from callbacks.Plugin
        import supybot.callbacks as callbacks
        from llm.plugin import LLM

        assert issubclass(LLM, callbacks.Plugin)


class TestCommandExistence:
    """Test that expected commands are defined on the plugin class."""

    def test_ask_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for ask THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "ask")
        assert callable(LLM.ask)

    def test_code_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for code THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "code")
        assert callable(LLM.code)

    def test_draw_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for draw THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "draw")
        assert callable(LLM.draw)

    def test_forget_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for forget THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "forget")
        assert callable(LLM.forget)

    def test_llmkeys_command_exists(self) -> None:
        """GIVEN LLM plugin class WHEN checking for llmkeys THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "llmkeys")
        assert callable(LLM.llmkeys)


class TestPluginConfiguration:
    """Test plugin configuration and service dependencies."""

    def test_plugin_is_threaded(self) -> None:
        """GIVEN LLM plugin class WHEN checking threaded attribute THEN True."""
        from llm.plugin import LLM

        assert LLM.threaded is True

    def test_service_module_imports(self) -> None:
        """GIVEN llm.service module WHEN imported THEN no errors."""
        from llm.service import LLMService

        assert LLMService is not None

    def test_context_module_imports(self) -> None:
        """GIVEN llm.context module WHEN imported THEN no errors."""
        from llm.context import ContextConfig, ConversationContext

        assert ConversationContext is not None
        assert ContextConfig is not None


class TestHTTPCallback:
    """Test HTTP callback class exists and has expected structure."""

    def test_http_callback_class_exists(self) -> None:
        """GIVEN llm.plugin module WHEN accessing LLMHTTPCallback THEN class exists."""
        from llm.plugin import LLMHTTPCallback

        assert LLMHTTPCallback is not None

    def test_http_callback_has_name(self) -> None:
        """GIVEN LLMHTTPCallback WHEN checking name THEN has expected name."""
        from llm.plugin import LLMHTTPCallback

        assert hasattr(LLMHTTPCallback, "name")
        assert LLMHTTPCallback.name == "LLM"

    def test_http_callback_is_public(self) -> None:
        """GIVEN LLMHTTPCallback WHEN checking public THEN is True."""
        from llm.plugin import LLMHTTPCallback

        assert hasattr(LLMHTTPCallback, "public")
        assert LLMHTTPCallback.public is True
