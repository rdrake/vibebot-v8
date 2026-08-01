"""Tests for provider-scoped API key resolution."""

from __future__ import annotations

import pytest
from llm import apikeys


class TestProviderOf:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("xai/grok-4.3", "xai"),
            ("gemini/gemini-3-flash-preview", "gemini"),
            ("openai/gpt-5.2", "openai"),
            ("anthropic/claude-3-opus", "anthropic"),
            ("gpt-4", "openai"),  # unprefixed names are legal
            ("dall-e-3", "openai"),
            ("vertex_ai/imagen-4.0-generate-001", "vertex_ai"),  # known, unmanaged
        ],
    )
    def test_known_models_resolve(self, model: str, expected: str) -> None:
        """GIVEN a model LiteLLM recognises WHEN provider_of THEN its provider.

        vertex_ai is included deliberately: "known but unmanaged" and
        "unresolvable" must stay distinguishable, because they take different
        paths at the boundary.
        """
        assert apikeys.provider_of(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "",
            "   ",
            "not-a-real-model-xyz",
            "claude-3-opus",
            "/",
            "//",
            "/gpt-4",
            "a/b/c/d",
            "gpt-4\n",
            "model with spaces",
            "a" * 500,
        ],
    )
    def test_unresolvable_models_return_empty(self, model: str) -> None:
        """GIVEN a model LiteLLM rejects WHEN provider_of THEN "" and no raise.

        LiteLLM raises BadRequestError for these. Key resolution runs inside
        failure handlers, so a new exception type here would surface as an
        unhandled crash instead of a config error.
        """
        assert apikeys.provider_of(model) == ""

    def test_mixed_case_prefix_is_not_resolvable(self) -> None:
        """GIVEN "XAI/grok-4.3" WHEN provider_of THEN "" — LiteLLM is case-sensitive.

        Pinned so the behaviour is deliberate: normalizing the case here would
        only send LiteLLM a model string it rejects a moment later.
        """
        assert apikeys.provider_of("XAI/grok-4.3") == ""

    def test_non_bad_request_exception_is_still_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN litellm.get_llm_provider raises something other than
        BadRequestError WHEN provider_of THEN "" and no raise.

        provider_of runs inside failure handlers, so any exception type escaping
        it — not just the BadRequestError LiteLLM happens to raise today — would
        surface as an unhandled crash instead of a configuration error. This
        pins the ``except Exception`` clause broad: narrowing it to
        ``except litellm.exceptions.BadRequestError`` would still pass every
        other test in this file (LiteLLM 1.93.0 wraps all its internal failures
        into BadRequestError before they escape) while silently reintroducing
        exactly the crash this module exists to prevent.
        """

        def _raise(_model: str) -> tuple[str, str, str, str]:
            raise TypeError("not a BadRequestError")

        monkeypatch.setattr(apikeys.litellm, "get_llm_provider", _raise)
        assert apikeys.provider_of("xai/grok-4.3") == ""


class TestEnvVarFor:
    def test_managed_provider(self) -> None:
        """GIVEN an xai model WHEN env_var_for THEN XAI_API_KEY."""
        assert apikeys.env_var_for("xai/grok-4.3") == "XAI_API_KEY"

    def test_unmanaged_provider(self) -> None:
        """GIVEN a vertex_ai model WHEN env_var_for THEN None."""
        assert apikeys.env_var_for("vertex_ai/imagen-4.0-generate-001") is None


class TestIsManaged:
    def test_managed(self) -> None:
        """GIVEN an xai model WHEN is_managed THEN True."""
        assert apikeys.is_managed("xai/grok-4.3") is True

    @pytest.mark.parametrize(
        "model", ["vertex_ai/imagen-4.0-generate-001", "not-a-real-model-xyz", ""]
    )
    def test_unmanaged(self, model: str) -> None:
        """GIVEN an unmanaged or unresolvable model WHEN is_managed THEN False.

        False means "hand LiteLLM None and let it use its own credentials" —
        which is how vertex_ai keeps working via ADC.
        """
        assert apikeys.is_managed(model) is False


class TestApiKeyFor:
    def test_returns_env_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY set WHEN api_key_for an xai model THEN that value."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"

    def test_missing_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN XAI_API_KEY unset WHEN api_key_for THEN None."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_var_returns_none(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """GIVEN a blank var WHEN api_key_for THEN None, not "".

        "" would read as configured at the guard and then fail at the provider.
        """
        monkeypatch.setenv("XAI_API_KEY", value)
        assert apikeys.api_key_for("xai/grok-4.3") is None

    def test_value_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a trailing newline WHEN api_key_for THEN stripped.

        docker --env-file and hand-edited files both produce these.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "  AIza-fake-value-for-tests\n")
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "AIza-fake-value-for-tests"

    def test_unmanaged_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN a vertex_ai model WHEN api_key_for THEN None even if a var exists.

        Guards against an implementation that derives the variable name as
        f"{provider.upper()}_API_KEY" instead of consulting the map.
        """
        monkeypatch.setenv("VERTEX_AI_API_KEY", "should-not-be-used-anywhere")
        assert apikeys.api_key_for("vertex_ai/imagen-4.0-generate-001") is None

    def test_provider_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN both vars set WHEN resolving each THEN no cross-provider bleed."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-value-for-tests")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-fake-value-for-tests")
        assert apikeys.api_key_for("xai/grok-4.3") == "xai-fake-value-for-tests"
        assert apikeys.api_key_for("gemini/gemini-3-flash-preview") == "gemini-fake-value-for-tests"

    def test_reads_env_on_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN the var changes WHEN resolving again THEN the new value.

        Kills a memoized implementation: redaction reads the same source, so a
        cached key could be sent while an uncached one is scrubbed.
        """
        monkeypatch.setenv("XAI_API_KEY", "first-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "first-fake-value-here"
        monkeypatch.setenv("XAI_API_KEY", "second-fake-value-here")
        assert apikeys.api_key_for("xai/grok-4.3") == "second-fake-value-here"
