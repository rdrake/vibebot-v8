"""Property-based state-machine tests for the pending-task lifecycle.

Locks down lifecycle invariants (claim mutual-exclusion, attempt_count
delta, lease deadline, expiry) over ``LLMDatabase`` 's pending-task API.
``TestDeliveryStatePersistence`` in ``test_persistence.py`` covers the
``delivery_state_filter`` / ``max_delivery_attempts`` paths that this
state machine intentionally does not parameterize.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from hypothesis.strategies import booleans, integers, lists, sampled_from
from llm.persistence import LLMDatabase

TASK_TYPES = ["ask", "code", "draw"]
DELIVERY_STATES = ["ready", "retrying", "delivered", "delivery_failed", "failed_terminal"]


class PendingTaskLifecycleMachine(RuleBasedStateMachine):
    """Drives ``LLMDatabase``'s pending-task API via Hypothesis rules.

    ``time.time`` is patched in ``llm.persistence`` so each rule can set a
    deterministic ``self._now`` before invoking the DB. SQLite + ``BEGIN
    IMMEDIATE`` per rule is genuinely slow, hence ``deadline=None`` and
    ``HealthCheck.too_slow``.
    """

    task_ids: Bundle[int] = Bundle("task_ids")

    def __init__(self) -> None:
        super().__init__()
        self._dir = Path(tempfile.mkdtemp(prefix="pendingtask-prop-"))
        self.db = LLMDatabase(str(self._dir / "p.db"))
        self._t0 = 1_000_000.0
        self._now = self._t0
        self._patcher = patch("llm.persistence.time.time", side_effect=lambda: self._now)
        self._patcher.start()

    def teardown(self) -> None:
        self._patcher.stop()
        self.db.close()
        shutil.rmtree(self._dir, ignore_errors=True)

    def _set_now(self, offset: int) -> float:
        self._now = self._t0 + offset
        return self._now

    @rule(
        target=task_ids,
        task_type=sampled_from(TASK_TYPES),
        expires_offset=integers(min_value=0, max_value=3600),
        next_attempt_offset=integers(min_value=0, max_value=3600),
    )
    def save(self, task_type: str, expires_offset: int, next_attempt_offset: int) -> int:
        now = self._set_now(0)
        return self.db.save_pending_task(
            task_type=task_type,
            nick="alice",
            reply_target="#general",
            is_channel=True,
            prompt_preview="x",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + expires_offset,
            next_attempt_at=now + next_attempt_offset,
        )

    @rule(
        now_offset=integers(min_value=0, max_value=3600),
        limit=integers(min_value=1, max_value=10),
        lease_seconds=integers(min_value=1, max_value=600),
    )
    def claim(self, now_offset: int, limit: int, lease_seconds: int) -> None:
        now = self._set_now(now_offset)
        first = self.db.claim_due_pending_tasks(now, limit, lease_seconds)
        # Mutual exclusion: a second call at the same now returns disjoint IDs.
        second = self.db.claim_due_pending_tasks(now, limit, lease_seconds)
        first_ids = {r.id for r in first}
        second_ids = {r.id for r in second}
        assert first_ids.isdisjoint(second_ids)
        # Every claimed row was actually due and unclaimed at claim time.
        # We cannot inspect the pre-claim row state here, but we can verify the
        # post-state for the first batch: claimed_until == now + lease_seconds.
        if first:
            for row in first:
                # Reload from DB to read post-claim state.
                reloaded = self._row_by_id(row.id)
                assert reloaded is not None
                assert reloaded.claimed_until == pytest.approx(now + lease_seconds)
                # Pre-claim guard: returned rows had next_attempt_at <= now.
                assert row.next_attempt_at <= now
                # And were unclaimed (claimed_until <= now) at claim time.
                assert row.claimed_until <= now

    @rule(
        task_id=task_ids,
        next_attempt_offset=integers(min_value=0, max_value=3600),
        increment=booleans(),
    )
    def release(self, task_id: int, next_attempt_offset: int, increment: bool) -> None:
        before = self._row_by_id(task_id)
        if before is None:
            return  # Task was deleted (e.g. by delete_expired) — skip.
        self._set_now(0)
        self.db.release_pending_task(
            task_id=task_id,
            next_attempt_at=self._t0 + next_attempt_offset,
            last_error="boom",
            increment_attempt=increment,
        )
        after = self._row_by_id(task_id)
        assert after is not None
        if increment:
            assert after.attempt_count == before.attempt_count + 1
        else:
            assert after.attempt_count == before.attempt_count

    @rule(
        task_id=task_ids,
        delivery_state=sampled_from(["ready", "failed_terminal"]),
    )
    def update_for_delivery(self, task_id: int, delivery_state: str) -> None:
        if self._row_by_id(task_id) is None:
            return
        self.db.update_task_for_delivery(task_id, delivery_state, "{}")

    @rule(
        task_id=task_ids,
        delivery_state=sampled_from(["retrying", "delivery_failed", "delivered"]),
        attempt_count=integers(min_value=0, max_value=10),
        next_attempt_offset=integers(min_value=0, max_value=3600),
    )
    def update_delivery_attempt(
        self,
        task_id: int,
        delivery_state: str,
        attempt_count: int,
        next_attempt_offset: int,
    ) -> None:
        if self._row_by_id(task_id) is None:
            return
        self._set_now(0)
        self.db.update_delivery_attempt(
            task_id=task_id,
            delivery_state=delivery_state,
            last_delivery_error="err",
            delivery_attempt_count=attempt_count,
            next_attempt_at=self._t0 + next_attempt_offset,
        )

    @rule(now_offset=integers(min_value=0, max_value=7200))
    def delete_expired(self, now_offset: int) -> None:
        now = self._set_now(now_offset)
        before_all = self.db.load_pending_tasks()
        deleted = self.db.delete_expired_pending_tasks(now)
        after_all = self.db.load_pending_tasks()
        deleted_ids = {r.id for r in deleted}
        before_ids = {r.id for r in before_all}
        after_ids = {r.id for r in after_all}
        # Returned IDs equal the IDs that disappeared.
        assert deleted_ids == before_ids - after_ids
        # Only rows with delivery_state='pending' AND expires_at <= now are removed.
        for row in deleted:
            assert row.delivery_state == "pending"
            assert row.expires_at <= now

    @rule(task_type=sampled_from(TASK_TYPES))
    def load_filtered_is_subset(self, task_type: str) -> None:
        all_rows = self.db.load_pending_tasks()
        filtered = self.db.load_pending_tasks(task_type=task_type)
        all_ids_for_type = {r.id for r in all_rows if r.task_type == task_type}
        filtered_ids = {r.id for r in filtered}
        assert filtered_ids == all_ids_for_type

    def _row_by_id(self, task_id: int):
        for row in self.db.load_pending_tasks():
            if row.id == task_id:
                return row
        return None


PendingTaskLifecycleMachine.TestCase.settings = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
TestPendingTaskLifecycle = PendingTaskLifecycleMachine.TestCase


@given(
    rows=lists(
        # (delivery_state, claimed_until_offset, next_attempt_offset)
        sampled_from(
            [
                ("pending", 0, 0),
                ("pending", 0, 100),
                ("ready", 0, 50),
                ("retrying", 0, 200),
                ("delivered", 0, 0),
                ("delivery_failed", 0, 0),
                ("failed_terminal", 0, 0),
                # Claim still active (claimed_until > now)
                ("pending", 500, 0),
            ]
        ),
        min_size=0,
        max_size=15,
    ),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_get_next_due_time_matches_oracle(rows: list[tuple[str, int, int]]) -> None:
    """``get_next_due_time`` returns ``MIN(next_attempt_at)`` over actionable rows.

    Actionable: ``claimed_until <= now`` AND ``delivery_state IN
    ('pending', 'ready', 'retrying')`` (see ``persistence.py:1166-1186``).

    A fresh database is allocated per example because pytest fixtures
    (``tmp_path``, ``mocker``) cannot be safely reused across
    Hypothesis examples in a function-scoped ``@given`` test.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="next-due-prop-"))
    db_path = work_dir / "next_due.db"
    t0 = 1_000_000.0
    patcher = patch("llm.persistence.time.time", return_value=t0)
    patcher.start()
    db = LLMDatabase(str(db_path))
    try:
        # Insert rows; each row gets a known next_attempt_at.
        next_attempts = []
        for state, claimed_offset, next_offset in rows:
            tid = db.save_pending_task(
                task_type="ask",
                nick="alice",
                reply_target="#g",
                is_channel=True,
                prompt_preview="x",
                model="m",
                request_data="{}",
                submitted_at=t0,
                expires_at=t0 + 3600,
                next_attempt_at=t0 + next_offset,
            )
            next_attempts.append((tid, state, claimed_offset, next_offset))

        # Apply non-default delivery_state and claimed_until via direct SQL.
        # We mirror what the production setters would do but bypass them so the
        # fixture can construct exact combinations that the state machine would
        # take many examples to reach.
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        try:
            for tid, state, claimed_offset, _next_offset in next_attempts:
                conn.execute(
                    "UPDATE pending_tasks SET delivery_state = ?, claimed_until = ? WHERE id = ?",
                    (state, t0 + claimed_offset, tid),
                )
            conn.commit()
        finally:
            conn.close()

        # Compute oracle: MIN(next_attempt_at) over actionable rows.
        actionable = [
            t0 + n
            for (_tid, state, claimed_offset, n) in next_attempts
            if state in ("pending", "ready", "retrying") and t0 + claimed_offset <= t0
        ]
        expected = min(actionable) if actionable else None
        actual = db.get_next_due_time()
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)
    finally:
        db.close()
        patcher.stop()
        shutil.rmtree(work_dir, ignore_errors=True)
