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
    from collections.abc import Callable
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


# =============================================================================
# Factory fixtures
# =============================================================================


@pytest.fixture
def spontaneous_env(
    mock_irc: MagicMock, mocker: MockerFixture
) -> Callable[..., tuple[MagicMock, MagicMock, MagicMock]]:
    """Factory: create plugin configured for spontaneous participation testing.

    Returns ``(plugin, mock_irc, mock_add_event)``.  Sensible defaults for all
    spontaneous config; pass ``**overrides`` to customise individual keys.
    """

    def _make(**overrides: object) -> tuple[MagicMock, MagicMock, MagicMock]:
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        defaults: dict[str, object] = {
            "contextTrackAllMessages": True,
            "spontaneousEnabled": True,
            "spontaneousChance": 100,
            "spontaneousCooldown": 0,
            "assistantApiKey": "sk-test-ask-key",
            "assistantModel": "gemini/gemini-2.0-flash-lite",
            "spontaneousSystemPrompt": "You are a regular in this IRC channel.",
        }
        defaults.update(overrides)

        registry_side_effect = make_registry_side_effect(defaults)
        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        patches = plugin_init_patches(mocker)
        patches["LLMService"].return_value.sanitize_output.side_effect = lambda x: x
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc, mock_add_event

    return _make


@pytest.fixture
def evaluate_env(
    spontaneous_env: Callable[..., tuple],
) -> Callable[..., tuple[MagicMock, MagicMock, Callable[[], None]]]:
    """Factory: create plugin, add channel history, extract _evaluate callback.

    Calls ``spontaneous_env`` internally, adds default messages, invokes
    ``_schedule_spontaneous``, and returns ``(plugin, mock_irc, callback)``.
    """

    def _make(
        channel: str = "#test",
        messages: list[tuple[str, str]] | None = None,
        **overrides: object,
    ) -> tuple[MagicMock, MagicMock, Callable[[], None]]:
        plugin, mock_irc, mock_add_event = spontaneous_env(**overrides)

        if messages is None:
            messages = [
                ("alice", "I love programming in Python!"),
                ("bob", "Me too, especially web development."),
            ]
        for nick, text in messages:
            plugin.context.add_channel_message(channel, nick, "user", text)

        last_nick = messages[-1][0] if messages else "unknown"
        last_text = messages[-1][1] if messages else ""
        plugin._schedule_spontaneous(mock_irc, channel, last_nick, last_text)

        assert mock_add_event.called, "schedule.addEvent should have been called"
        callback = mock_add_event.call_args[0][0]

        return plugin, mock_irc, callback

    return _make


# =============================================================================
# doPrivmsg tests
# =============================================================================


class TestSpontaneousDoPrivmsg:
    """Test spontaneous participation triggering in doPrivmsg."""

    @staticmethod
    def _make_msg(mocker: MockerFixture) -> MagicMock:
        """Create a mock IRC channel message."""
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "Hello world")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100  # Future time (not playback)
        mock_msg.server_tags = {}  # default: no IRCv3 account-tag

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)

        return mock_msg

    def test_skips_when_disabled(self, spontaneous_env: Callable, mocker: MockerFixture) -> None:
        """GIVEN spontaneousEnabled=False WHEN doPrivmsg fires THEN no spontaneous scheduled."""
        plugin, mock_irc, _ = spontaneous_env(spontaneousEnabled=False)
        mock_msg = self._make_msg(mocker)
        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_not_called()

    def test_fires_on_chance_hit(self, spontaneous_env: Callable, mocker: MockerFixture) -> None:
        """GIVEN spontaneous enabled and chance hit WHEN doPrivmsg fires THEN schedule called."""
        plugin, mock_irc, _ = spontaneous_env(spontaneousChance=100, spontaneousCooldown=0)
        mock_msg = self._make_msg(mocker)
        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")
        mocker.patch("llm.plugin.random.randint", return_value=1)

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_called_once()
        call_args = mock_schedule.call_args[0]
        assert call_args[0] is mock_irc
        assert call_args[1] == "#channel"

    def test_respects_cooldown(self, spontaneous_env: Callable, mocker: MockerFixture) -> None:
        """GIVEN recent spontaneous cooldown WHEN doPrivmsg fires THEN no evaluation fires."""
        plugin, mock_irc, _ = spontaneous_env(spontaneousChance=100, spontaneousCooldown=5)
        plugin._spontaneous_cooldowns["#channel"] = time.time()
        mock_msg = self._make_msg(mocker)
        mock_schedule = mocker.patch.object(plugin, "_schedule_spontaneous")

        plugin.doPrivmsg(mock_irc, mock_msg)

        mock_schedule.assert_not_called()

    def test_skips_when_context_disabled(
        self, spontaneous_env: Callable, mocker: MockerFixture
    ) -> None:
        """GIVEN contextEnabled=False and spontaneousEnabled=True WHEN doPrivmsg fires THEN no spontaneous scheduled."""
        plugin, mock_irc, mock_add_event = spontaneous_env(
            contextEnabled=False,
            spontaneousEnabled=True,
            spontaneousChance=100,
            spontaneousCooldown=0,
        )
        mock_msg = self._make_msg(mocker)

        plugin.doPrivmsg(mock_irc, mock_msg)

        # No addEvent calls should be spontaneous-related
        for call in mock_add_event.call_args_list:
            if len(call[0]) >= 2:
                event_name = call[0][1] if len(call[0]) > 1 else ""
                assert "spontaneous" not in str(event_name).lower()


# =============================================================================
# _evaluate callback tests
# =============================================================================


class TestSpontaneousEvaluate:
    """Test the _evaluate callback invoked by _schedule_spontaneous."""

    def test_sends_message_on_non_pass(self, evaluate_env: Callable, mocker: MockerFixture) -> None:
        """GIVEN completion returns content WHEN _evaluate runs THEN irc.queueMsg called."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback = evaluate_env()

        plugin.llm_service.completion.return_value = CompletionResult(
            content="That's interesting!",
            prompt_tokens=50,
            completion_tokens=10,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        mock_irc.queueMsg.assert_called_once()
        plugin.llm_service.completion.assert_called_once()
        call_kwargs = plugin.llm_service.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-test-ask-key"
        assert call_kwargs.kwargs.get("model_override") == "gemini/gemini-2.0-flash-lite"
        assert call_kwargs.kwargs.get("system_prompt") == "You are a regular in this IRC channel."

    def test_discards_pass_response(self, evaluate_env: Callable) -> None:
        """GIVEN completion returns PASS WHEN _evaluate runs THEN no irc.queueMsg."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback = evaluate_env()

        plugin.llm_service.completion.return_value = CompletionResult(
            content="PASS",
            prompt_tokens=50,
            completion_tokens=1,
            cost=0.0001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        mock_irc.queueMsg.assert_not_called()

    def test_sends_pass_in_sentence(self, evaluate_env: Callable) -> None:
        """GIVEN completion returns text containing PASS WHEN _evaluate runs THEN message sent."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback = evaluate_env()

        plugin.llm_service.completion.return_value = CompletionResult(
            content="I'll PASS on this one.",
            prompt_tokens=50,
            completion_tokens=5,
            cost=0.0001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        mock_irc.queueMsg.assert_called_once()

    def test_uses_assistant_api_key(self, evaluate_env: Callable) -> None:
        """GIVEN assistantApiKey set WHEN _evaluate runs THEN that key is used."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback = evaluate_env(assistantApiKey="sk-special")

        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hi!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )

        callback()

        call_kwargs = plugin.llm_service.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-special"

    def test_no_message_when_no_channel_history(self, evaluate_env: Callable) -> None:
        """GIVEN empty channel history WHEN _evaluate runs THEN no completion called."""
        plugin, mock_irc, callback = evaluate_env(
            channel="#empty", messages=[], assistantApiKey="sk-test"
        )

        callback()

        plugin.llm_service.completion.assert_not_called()
        mock_irc.queueMsg.assert_not_called()

    def test_no_message_when_no_api_key(self, evaluate_env: Callable) -> None:
        """GIVEN no API keys configured WHEN _evaluate runs THEN no completion called."""
        plugin, mock_irc, callback = evaluate_env(assistantApiKey="")

        callback()

        plugin.llm_service.completion.assert_not_called()
        mock_irc.queueMsg.assert_not_called()

    def test_logs_usage_on_success(self, evaluate_env: Callable, mock_irc: MagicMock) -> None:
        """GIVEN successful spontaneous response WHEN _evaluate runs THEN usage logged."""
        from llm.service import CompletionResult

        plugin, mock_irc, callback = evaluate_env()

        plugin.llm_service.completion.return_value = CompletionResult(
            content="That's cool!",
            prompt_tokens=50,
            completion_tokens=10,
            cost=0.001,
            model="gemini/gemini-2.0-flash-lite",
        )

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


# =============================================================================
# die() cleanup tests
# =============================================================================


class TestSpontaneousDie:
    """Test that die() cancels pending spontaneous events."""

    def test_die_cancels_pending_spontaneous_events(self, mocker: MockerFixture) -> None:
        """GIVEN pending spontaneous events WHEN die() called THEN events cancelled."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._http_callback = None
        plugin._spontaneous_events = {"llm_spontaneous_aaa", "llm_spontaneous_bbb"}
        plugin._spontaneous_cooldowns = {"#test": time.time()}

        mock_remove = mocker.patch("supybot.schedule.removeEvent")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)

        plugin.die()

        removed_names = {call.args[0] for call in mock_remove.call_args_list}
        assert "llm_spontaneous_aaa" in removed_names
        assert "llm_spontaneous_bbb" in removed_names
        assert len(plugin._spontaneous_events) == 0
        assert len(plugin._spontaneous_cooldowns) == 0


# =============================================================================
# Completion override tests (already clean — unchanged)
# =============================================================================


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
        service, plugin = make_service(assistantApiKey="")

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
