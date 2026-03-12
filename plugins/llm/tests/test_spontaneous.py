"""Tests for spontaneous participation logic.

These tests verify the spontaneous participation feature where the bot
can optionally jump into channel conversations based on a configurable
probability and cooldown system.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestSpontaneousDoPrivmsg:
    """Test spontaneous participation triggering in doPrivmsg."""

    @pytest.fixture
    def plugin_for_spontaneous(self, mock_irc: MagicMock, mocker: MockerFixture) -> tuple:
        """Create plugin configured for spontaneous participation testing."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "contextTrackAllMessages": True,
                "spontaneousEnabled": False,
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc

    @staticmethod
    def _make_msg(mocker: MockerFixture) -> MagicMock:
        """Create a mock IRC channel message."""
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "Hello world")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100  # Future time (not playback)
        return mock_msg

    def test_skips_when_disabled(
        self, plugin_for_spontaneous: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN spontaneousEnabled=False WHEN doPrivmsg fires THEN no spontaneous scheduled."""
        plugin, mock_irc = plugin_for_spontaneous
        mock_msg = self._make_msg(mocker)

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)

        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_not_called()

    def test_fires_on_chance_hit(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN spontaneous enabled and chance hit WHEN doPrivmsg fires THEN schedule called."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "contextTrackAllMessages": True,
                "spontaneousEnabled": True,
                "spontaneousChance": 100,  # guaranteed hit
                "spontaneousCooldown": 0,  # no cooldown
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        mock_msg = self._make_msg(mocker)
        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)

        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")

        # Mock random.randint to return 1 (guaranteed hit for chance <= 100)
        mocker.patch("llm.plugin.random.randint", return_value=1)

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_called_once_with(mock_irc, "#channel")

    def test_respects_cooldown(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN recent spontaneous cooldown WHEN doPrivmsg fires THEN no evaluation fires."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "contextTrackAllMessages": True,
                "spontaneousEnabled": True,
                "spontaneousChance": 100,
                "spontaneousCooldown": 5,  # 5 minute cooldown
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        # Set cooldown to very recent time
        plugin._spontaneous_cooldowns["#channel"] = time.time()

        mock_msg = self._make_msg(mocker)
        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)

        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_not_called()


class TestSpontaneousEvaluate:
    """Test the _evaluate callback invoked by _schedule_spontaneous."""

    @pytest.fixture
    def plugin_with_spontaneous(self, mock_irc: MagicMock, mocker: MockerFixture) -> tuple:
        """Create plugin and trigger _schedule_spontaneous to capture callback."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "spontaneousEnabled": True,
                "spontaneousApiKey": "",
                "askApiKey": "sk-test-ask-key",
                "spontaneousModel": "gemini/gemini-2.0-flash-lite",
                "spontaneousSystemPrompt": "You are a regular in this IRC channel.",
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        # Give the context some channel messages
        plugin.context.add_channel_message(
            "#test", "alice", "user", "I love programming in Python!"
        )
        plugin.context.add_channel_message(
            "#test", "bob", "user", "Me too, especially web development."
        )

        # Trigger _schedule_spontaneous
        plugin._schedule_spontaneous(mock_irc, "#test")

        # Extract the callback from schedule.addEvent
        assert mock_add_event.called, "schedule.addEvent should have been called"
        callback = mock_add_event.call_args[0][0]

        return plugin, mock_irc, callback, mock_add_event

    def test_sends_message_on_non_pass(
        self, plugin_with_spontaneous: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN completion returns content WHEN _evaluate runs THEN irc.queueMsg called."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback, _ = plugin_with_spontaneous

        plugin.llm_service.completion.return_value = CompletionResult(
            content="That's interesting!",
            prompt_tokens=50,
            completion_tokens=10,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        callback()

        mock_irc.queueMsg.assert_called_once()
        sent_msg = mock_irc.queueMsg.call_args[0][0]
        # Verify it's a PRIVMSG to the right channel
        assert sent_msg  # ircmsgs.privmsg returns a mock here

        # Verify completion was called with expected params
        plugin.llm_service.completion.assert_called_once()
        call_kwargs = plugin.llm_service.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-test-ask-key"
        assert call_kwargs.kwargs.get("model_override") == "gemini/gemini-2.0-flash-lite"
        assert call_kwargs.kwargs.get("system_prompt") == "You are a regular in this IRC channel."

    def test_discards_pass_response(
        self, plugin_with_spontaneous: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN completion returns PASS WHEN _evaluate runs THEN no irc.queueMsg."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback, _ = plugin_with_spontaneous

        plugin.llm_service.completion.return_value = CompletionResult(
            content="PASS",
            prompt_tokens=50,
            completion_tokens=1,
            cost=0.0001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        mock_irc.queueMsg.assert_not_called()

    def test_discards_pass_in_sentence(
        self, plugin_with_spontaneous: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN completion returns text containing PASS WHEN _evaluate runs THEN no message."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback, _ = plugin_with_spontaneous

        plugin.llm_service.completion.return_value = CompletionResult(
            content="I'll PASS on this one.",
            prompt_tokens=50,
            completion_tokens=5,
            cost=0.0001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        mock_irc.queueMsg.assert_not_called()

    def test_uses_ask_api_key_as_fallback(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN spontaneousApiKey empty WHEN _evaluate runs THEN askApiKey used."""
        from llm.plugin import LLM
        from llm.service import CompletionResult

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "spontaneousEnabled": True,
                "spontaneousApiKey": "",
                "askApiKey": "sk-fallback-key",
                "spontaneousModel": "gemini/gemini-2.0-flash-lite",
                "spontaneousSystemPrompt": "You are a regular.",
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        plugin.context.add_channel_message("#test", "alice", "user", "Hello!")

        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hi!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        plugin._schedule_spontaneous(mock_irc, "#test")
        callback = mock_add_event.call_args[0][0]
        callback()

        call_kwargs = plugin.llm_service.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-fallback-key"

    def test_uses_dedicated_api_key_when_set(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN spontaneousApiKey set WHEN _evaluate runs THEN dedicated key used."""
        from llm.plugin import LLM
        from llm.service import CompletionResult

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "spontaneousEnabled": True,
                "spontaneousApiKey": "sk-special",
                "askApiKey": "sk-fallback-key",
                "spontaneousModel": "gemini/gemini-2.0-flash-lite",
                "spontaneousSystemPrompt": "You are a regular.",
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        plugin.context.add_channel_message("#test", "alice", "user", "Hello!")

        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hi!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        plugin._schedule_spontaneous(mock_irc, "#test")
        callback = mock_add_event.call_args[0][0]
        callback()

        call_kwargs = plugin.llm_service.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-special"

    def test_no_message_when_no_channel_history(
        self, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN empty channel history WHEN _evaluate runs THEN no completion called."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "spontaneousEnabled": True,
                "spontaneousApiKey": "sk-test",
                "spontaneousModel": "gemini/gemini-2.0-flash-lite",
                "spontaneousSystemPrompt": "You are a regular.",
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        # No channel messages added

        plugin._schedule_spontaneous(mock_irc, "#empty")
        callback = mock_add_event.call_args[0][0]
        callback()

        plugin.llm_service.completion.assert_not_called()
        mock_irc.queueMsg.assert_not_called()

    def test_no_message_when_no_api_key(self, mock_irc: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN no API keys configured WHEN _evaluate runs THEN no completion called."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect(
            {
                "spontaneousEnabled": True,
                "spontaneousApiKey": "",
                "askApiKey": "",
                "spontaneousModel": "gemini/gemini-2.0-flash-lite",
                "spontaneousSystemPrompt": "You are a regular.",
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        plugin.context.add_channel_message("#test", "alice", "user", "Hello!")

        plugin._schedule_spontaneous(mock_irc, "#test")
        callback = mock_add_event.call_args[0][0]
        callback()

        plugin.llm_service.completion.assert_not_called()
        mock_irc.queueMsg.assert_not_called()

    def test_logs_usage_on_success(
        self, plugin_with_spontaneous: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN successful spontaneous response WHEN _evaluate runs THEN usage logged."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback, _ = plugin_with_spontaneous

        plugin.llm_service.completion.return_value = CompletionResult(
            content="That's cool!",
            prompt_tokens=50,
            completion_tokens=10,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        callback()

        plugin.db.log_usage.assert_called_once_with(
            mock_irc.nick,
            "#test",
            "spontaneous",
            "gemini/gemini-2.0-flash-lite",
            50,
            10,
            0.001,
            prompt="[spontaneous]",
            status="success",
        )


class TestCompletionOverrides:
    """Test api_key and model_override parameters in service.completion()."""

    def test_api_key_override(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN api_key override WHEN completion called THEN override used instead of config."""
        service, plugin = make_service()

        # Mock litellm.completion
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mocker.patch("litellm.completion", return_value=mock_response)
        mocker.patch("litellm.completion_cost", return_value=0.001)

        result = service.completion(
            "test prompt",
            command="ask",
            api_key="sk-override-key",
        )

        assert result.error is None
        # Verify litellm.completion was called with the override key
        import litellm

        litellm.completion.assert_called_once()
        call_kwargs = litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-override-key"

    def test_model_override(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN model_override WHEN completion called THEN override used instead of config."""
        service, plugin = make_service()

        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mocker.patch("litellm.completion", return_value=mock_response)
        mocker.patch("litellm.completion_cost", return_value=0.001)

        result = service.completion(
            "test prompt",
            command="ask",
            model_override="custom-model-v2",
        )

        assert result.error is None
        import litellm

        litellm.completion.assert_called_once()
        call_kwargs = litellm.completion.call_args
        assert call_kwargs.kwargs.get("model") == "custom-model-v2"

    def test_api_key_override_bypasses_config_check(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN no config API key but api_key override WHEN completion THEN succeeds."""
        service, plugin = make_service(askApiKey="")

        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = mocker.MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mocker.patch("litellm.completion", return_value=mock_response)
        mocker.patch("litellm.completion_cost", return_value=0.001)

        result = service.completion(
            "test prompt",
            command="ask",
            api_key="sk-override-key",
        )

        # Should succeed, not return "API key not configured" error
        assert result.error is None
        assert result.content == "Hello!"
