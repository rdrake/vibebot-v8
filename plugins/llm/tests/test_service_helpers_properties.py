"""Property tests for pure helper functions in ``llm.service``.

These functions have no dependencies on plugin/registry/database state,
so they sit in their own module rather than being attached to an
``LLMService`` instance fixture.

Currently covers ``_compute_backoff`` and ``truncate_to_word_boundary``.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers, text
from llm.service import (
    PENDING_INITIAL_BACKOFF_SECONDS,
    PENDING_MAX_BACKOFF_SECONDS,
    LLMService,
    truncate_to_word_boundary,
)

# Cap the attempt count at something well past the saturation point
# (``log2(300/30) ≈ 3.32``) so both the growing and the saturated
# regimes are exercised. ``2**1000`` would be a Python big-int and the
# property still holds, but it slows the test for no extra coverage.
_attempts = integers(min_value=0, max_value=64)


@given(n=_attempts)
def test_backoff_bounded_above(n: int) -> None:
    """Result is never above the configured cap."""
    assert LLMService._compute_backoff(n) <= PENDING_MAX_BACKOFF_SECONDS


@given(n=_attempts)
def test_backoff_bounded_below_by_initial(n: int) -> None:
    """Result is never below the initial backoff (n=0 case is the floor)."""
    assert LLMService._compute_backoff(n) >= PENDING_INITIAL_BACKOFF_SECONDS


@given(n=_attempts)
def test_backoff_monotone_non_decreasing(n: int) -> None:
    """``_compute_backoff(n) <= _compute_backoff(n + 1)``."""
    assert LLMService._compute_backoff(n) <= LLMService._compute_backoff(n + 1)


def test_backoff_at_zero_equals_initial() -> None:
    """At attempt 0, no doubling has happened yet."""
    assert LLMService._compute_backoff(0) == PENDING_INITIAL_BACKOFF_SECONDS


# --- truncate_to_word_boundary -------------------------------------------------
#
# The dangerous edge is the exact-fit boundary: when ``len(text) == max_chars``
# the text already fits and must be returned verbatim. An off-by-one in the
# guard (``<=`` weakened to ``<``) sends that case down the truncation path,
# silently dropping everything after the last interior space.

_lengths = integers(min_value=1, max_value=300)


@given(s=text(min_size=1, max_size=200))
def test_truncate_returns_input_unchanged_when_it_fits(s: str) -> None:
    """Text that already fits is returned verbatim, including at the exact
    ``len(s) == max_chars`` boundary."""
    assert truncate_to_word_boundary(s, len(s)) == s
    assert truncate_to_word_boundary(s, len(s) + 5) == s


@given(s=text(max_size=300), max_chars=_lengths)
def test_truncate_never_exceeds_max_chars(s: str, max_chars: int) -> None:
    """A positive ``max_chars`` is a hard upper bound on the result length."""
    assert len(truncate_to_word_boundary(s, max_chars)) <= max_chars


@given(s=text(max_size=300), max_chars=_lengths)
def test_truncate_never_grows_the_text(s: str, max_chars: int) -> None:
    """Truncation only ever shortens; the result never exceeds the input."""
    assert len(truncate_to_word_boundary(s, max_chars)) <= len(s)


@given(s=text(max_size=200), max_chars=integers(max_value=0))
def test_truncate_nonpositive_max_returns_unchanged(s: str, max_chars: int) -> None:
    """A non-positive ``max_chars`` disables truncation (degenerate guard)."""
    assert truncate_to_word_boundary(s, max_chars) == s
