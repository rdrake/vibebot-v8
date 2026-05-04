"""Property tests for pure helper functions in ``llm.service``.

These functions have no dependencies on plugin/registry/database state,
so they sit in their own module rather than being attached to an
``LLMService`` instance fixture.

Currently covers ``_compute_backoff`` (`service.py:1420-1433`).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers
from llm.service import (
    PENDING_INITIAL_BACKOFF_SECONDS,
    PENDING_MAX_BACKOFF_SECONDS,
    LLMService,
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
