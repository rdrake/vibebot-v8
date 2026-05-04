"""Property-based tests for usage ranking and aggregation.

Augments ``TestUsageRanking`` (``test_persistence.py:595``). The
top/middle/bottom/empty matrix in that class collapses to one
monotone-rank property here; the rest of the file targets the
``cost == 0.0`` short-circuit at ``persistence.py:1571-1578`` and the
top-N truncation in ``get_usage_by_channel``.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import floats, lists, sampled_from, tuples
from llm.persistence import LLMDatabase

CHANNELS = ["#a", "#b", "#c", "#d", "#e"]
NICKS = ["alice", "bob", "charlie", "dave", "eve"]


def _new_db() -> tuple[LLMDatabase, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="usage-prop-"))
    return LLMDatabase(str(work_dir / "u.db")), work_dir


def _populate(db: LLMDatabase, rows: list[tuple[str, float]]) -> None:
    for channel, cost in rows:
        db.log_usage("alice", channel, "ask", "gpt-4", 10, 5, cost)


def _populate_by_nick(db: LLMDatabase, rows: list[tuple[str, float]]) -> None:
    for nick, cost in rows:
        db.log_usage(nick, "#test", "ask", "gpt-4", 10, 5, cost)


def _channels_in(rows: list[tuple[str, float]]) -> Iterator[str]:
    seen: set[str] = set()
    for channel, _ in rows:
        if channel in seen:
            continue
        seen.add(channel)
        yield channel


def _total_cost_per_channel(rows: list[tuple[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for channel, cost in rows:
        totals[channel] = totals.get(channel, 0.0) + cost
    return totals


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_channel_rank_is_monotone_in_cost(rows: list[tuple[str, float]]) -> None:
    """``cost(a) > cost(b)`` ⇒ ``rank(a) < rank(b)`` (ties allowed)."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        totals = _total_cost_per_channel(rows)
        channels = list(_channels_in(rows))
        for i, ch_a in enumerate(channels):
            for ch_b in channels[i + 1 :]:
                ra = db.get_channel_rank(ch_a).rank
                rb = db.get_channel_rank(ch_b).rank
                if totals[ch_a] > totals[ch_b]:
                    assert ra < rb, (
                        f"{ch_a} (cost {totals[ch_a]}, rank {ra}) should rank above {ch_b} (cost {totals[ch_b]}, rank {rb})"
                    )
                elif totals[ch_a] < totals[ch_b]:
                    assert ra > rb
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_channel_rank_in_valid_range(rows: list[tuple[str, float]]) -> None:
    """For any populated DB, ``rank in {0} ∪ [1, total]``."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        for channel in CHANNELS:
            r = db.get_channel_rank(channel)
            assert r.rank == 0 or 1 <= r.rank <= r.total
            assert r.total >= 0
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_unused_channel_rank_is_zero(rows: list[tuple[str, float]]) -> None:
    """A channel that never appears in usage has ``rank == 0``."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        used = {ch for ch, _ in rows}
        unused = [ch for ch in CHANNELS if ch not in used]
        for channel in unused:
            r = db.get_channel_rank(channel)
            assert r.rank == 0
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_channel_summary_matches_inserted_rows(rows: list[tuple[str, float]]) -> None:
    """``get_usage_summary_for_channel(c).total_requests`` equals row count for ``c``."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        counts: dict[str, int] = {}
        for ch, _ in rows:
            counts[ch] = counts.get(ch, 0) + 1
        for channel in CHANNELS:
            summary = db.get_usage_summary_for_channel(channel)
            assert summary.total_requests == counts.get(channel, 0)
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
    limit=sampled_from([1, 2, 3, 5, 10]),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_top_n_truncation_bounds_total_cost(rows: list[tuple[str, float]], limit: int) -> None:
    """``sum(get_usage_by_channel(limit=N).total_cost) <= get_usage_summary().total_cost``."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        breakdown = db.get_usage_by_channel(limit=limit)
        summary = db.get_usage_summary()
        breakdown_sum = sum(b.total_cost for b in breakdown)
        # Floating-point tolerance because SQLite's SUM is double-precision.
        assert breakdown_sum <= summary.total_cost + 1e-9
        # Truncation respects the limit.
        assert len(breakdown) <= limit
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(CHANNELS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_breakdown_is_sorted_by_cost_descending(rows: list[tuple[str, float]]) -> None:
    """``get_usage_by_channel`` is ordered by ``total_cost`` descending."""
    db, work_dir = _new_db()
    try:
        _populate(db, rows)
        breakdown = db.get_usage_by_channel(limit=len(CHANNELS))
        costs = [b.total_cost for b in breakdown]
        assert costs == sorted(costs, reverse=True)
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(NICKS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_nick_rank_is_monotone_in_cost(rows: list[tuple[str, float]]) -> None:
    """Nick variant: ``cost(a) > cost(b)`` ⇒ ``rank(a) < rank(b)``."""
    db, work_dir = _new_db()
    try:
        _populate_by_nick(db, rows)
        totals: dict[str, float] = {}
        for nick, cost in rows:
            totals[nick] = totals.get(nick, 0.0) + cost
        seen: list[str] = []
        for nick, _ in rows:
            if nick not in seen:
                seen.append(nick)
        for i, nick_a in enumerate(seen):
            for nick_b in seen[i + 1 :]:
                ra = db.get_nick_rank(nick_a).rank
                rb = db.get_nick_rank(nick_b).rank
                if totals[nick_a] > totals[nick_b]:
                    assert ra < rb
                elif totals[nick_a] < totals[nick_b]:
                    assert ra > rb
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    rows=lists(
        tuples(sampled_from(NICKS), floats(min_value=0, max_value=1, allow_nan=False)),
        max_size=40,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_unused_nick_rank_is_zero(rows: list[tuple[str, float]]) -> None:
    """A nick that never appears in usage has ``rank == 0``."""
    db, work_dir = _new_db()
    try:
        _populate_by_nick(db, rows)
        used = {n for n, _ in rows}
        for nick in NICKS:
            if nick in used:
                continue
            assert db.get_nick_rank(nick).rank == 0
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.parametrize("populate_other", [True, False])
def test_zero_cost_rows_count_as_used(populate_other: bool) -> None:
    """Rows logged with ``cost=0.0`` still mark the channel as 'used'.

    This locks down the short-circuit at ``persistence.py:1571-1578``:
    ``cost == 0.0`` falls through to the COUNT(*) check, which finds
    rows and returns rank=1 (not rank=0 — that path is reserved for
    truly absent values).
    """
    db, work_dir = _new_db()
    try:
        db.log_usage("alice", "#zero", "ask", "gpt-4", 10, 5, 0.0)
        if populate_other:
            db.log_usage("bob", "#paid", "ask", "gpt-4", 10, 5, 0.10)
        rank = db.get_channel_rank("#zero")
        assert rank.rank != 0  # zero-cost ≠ unused
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)
