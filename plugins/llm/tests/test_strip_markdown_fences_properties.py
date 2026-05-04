"""Property-based tests for ``LLMService._strip_markdown_fences``.

Subsumes most of ``TestCodeFenceEdgeCases`` in
``test_markdown_edge_cases.py``: the round-trip and pass-through
properties below cover the "with language", "without language",
"trailing whitespace", "no fence", "empty body", and "unusual language"
cases that file enumerates.

The implementation regexes (``service.py:139-140``) are
``^```(\\w+)\\n(.*?)\\n?```$`` and ``^```\\n(.*?)\\n?```$`` with
``re.DOTALL``. Two consequences shape the strategies:

* The language token is ``\\w+``: only ``[A-Za-z0-9_]`` characters.
  "cpp-with-features" and similar do **not** parse as a language and
  fall through to the no-fence branch.
* The closing fence is anchored at end-of-string, so ``` `` `` ` `` ``` ``
  appearing inside the body is fine -- only a literal ``\\n``` `` ` ` `` ``
  at the very end could be ambiguous, and even then the regex's
  non-greedy match consumes the smallest body that still terminates at
  ``$``.
"""

from __future__ import annotations

from string import ascii_letters, digits

import pytest
from hypothesis import given
from hypothesis.strategies import none, one_of, text
from llm.service import _FENCE_NO_LANG_RE, _FENCE_WITH_LANG_RE

# Mirror the language-token alphabet from ``_FENCE_WITH_LANG_RE`` exactly.
_LANG_ALPHABET = ascii_letters + digits + "_"
langs = text(alphabet=_LANG_ALPHABET, min_size=1, max_size=10)
bodies = text(max_size=200)


def _is_fenced(s: str) -> bool:
    """Mirror the function's leading ``code.strip()`` and try both regexes."""
    s = s.strip()
    return bool(_FENCE_WITH_LANG_RE.match(s) or _FENCE_NO_LANG_RE.match(s))


class TestStripMarkdownFenceProperties:
    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        self.service, _ = make_service()

    @given(body=bodies, lang=langs)
    def test_round_trip_with_language(self, body: str, lang: str) -> None:
        """``strip(f"```{lang}\\n{body}\\n```") == (body, lang)``.

        Holds for any ``lang`` matching ``\\w+`` and any ``body`` -- the
        regex anchors the closing fence at ``$`` so internal backticks
        in the body cannot terminate the match early.
        """
        wrapped = f"```{lang}\n{body}\n```"
        clean, parsed_lang = self.service._strip_markdown_fences(wrapped)
        assert parsed_lang == lang
        assert clean == body

    @given(body=bodies)
    def test_round_trip_without_language(self, body: str) -> None:
        """Bare fences (no language) round-trip the body and return ``lang=None``."""
        wrapped = f"```\n{body}\n```"
        clean, parsed_lang = self.service._strip_markdown_fences(wrapped)
        assert parsed_lang is None
        assert clean == body

    @given(text_in=text(max_size=200).filter(lambda s: not _is_fenced(s)))
    def test_no_fence_passes_through_stripped(self, text_in: str) -> None:
        """No fence ⇒ output is ``text_in.strip()`` and ``lang is None``."""
        clean, parsed_lang = self.service._strip_markdown_fences(text_in)
        assert parsed_lang is None
        assert clean == text_in.strip()

    @given(body=bodies, lang=one_of(none(), langs))
    def test_clean_text_is_idempotent_under_restrip(self, body: str, lang: str | None) -> None:
        """Stripping clean output again yields no language and the same body
        (modulo a final ``str.strip()``).

        Once the fence is removed, a second call hits the no-fence branch
        and returns ``(clean.strip(), None)``. This catches a regression
        where the regex would re-match across non-fence input or where
        the ``code.strip()`` call were ever moved after the regex check.
        """
        wrapped = f"```{lang}\n{body}\n```" if lang is not None else f"```\n{body}\n```"
        first_clean, _ = self.service._strip_markdown_fences(wrapped)
        second_clean, second_lang = self.service._strip_markdown_fences(first_clean)
        assert second_lang is None
        assert second_clean == first_clean.strip()
