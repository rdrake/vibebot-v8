"""Incident lifecycle transitions.

summary.json carries only UNRESOLVED incidents, so an id disappearing means
"resolved", not "never happened", and an id list that shrinks then grows must
not produce a second announcement.
"""

from __future__ import annotations

from datetime import UTC, datetime

from llm import statuspage


def view(
    incident_id: str, *, status: str = "investigating", minutes: int = 0
) -> statuspage.IncidentView:
    return statuspage.IncidentView(
        id=incident_id,
        name=f"Incident {incident_id}",
        status=status,
        impact="minor",
        affected_components=("Claude API (api.anthropic.com)",),
        started_at=datetime(2026, 8, 9, 12, minutes, tzinfo=UTC),
        created_at=datetime(2026, 8, 9, 12, minutes, tzinfo=UTC),
        latest_update_body="We are investigating.",
        latest_update_at=datetime(2026, 8, 9, 12, minutes, tzinfo=UTC),
    )


def snap(*views: statuspage.IncidentView, fetched_at: float = 1000.0) -> statuspage.Snapshot:
    return statuspage.Snapshot(
        page_name="Claude",
        page_url="https://status.claude.com",
        indicator="minor" if views else "none",
        description="Partial System Outage" if views else "All Systems Operational",
        components={"Claude API (api.anthropic.com)": "operational"},
        incidents={v.id: v for v in views},
        fetched_at=fetched_at,
    )


class TestColdStart:
    def test_first_poll_seeds_silently(self):
        delta, state = statuspage.classify(statuspage.StatusState(), snap(view("A")))
        assert delta.opened == ()
        assert state.seeded is True
        assert "A" in state.announced
        assert "A" in state.active

    def test_failed_first_poll_then_success_still_seeds_silently(self):
        """Seeding keys on a validated parse, not a fetch attempt. A caller
        that never calls classify (because the fetch raised) leaves the state
        unseeded, so the next success is still a cold start."""
        state = statuspage.StatusState()
        assert state.seeded is False
        delta, state = statuspage.classify(state, snap(view("A")))
        assert delta.opened == ()
        assert state.seeded is True


class TestOpened:
    def test_new_incident_after_seeding_is_opened(self):
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        delta, state = statuspage.classify(state, snap(view("A")))
        assert [i.id for i in delta.opened] == ["A"]

    def test_opened_is_not_marked_announced_by_classify(self):
        """The caller marks announced only after a successful queue, so a
        dropped send is retried on the next poll."""
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        delta, state = statuspage.classify(state, snap(view("A")))
        assert delta.opened
        assert "A" not in state.announced

    def test_unannounced_incident_reopens_next_poll(self):
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        _, state = statuspage.classify(state, snap(view("A")))
        delta, _state = statuspage.classify(state, snap(view("A")))
        assert [i.id for i in delta.opened] == ["A"], "not marked announced, so still pending"

    def test_marked_incident_is_not_reannounced(self):
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        _, state = statuspage.classify(state, snap(view("A")))
        state = statuspage.mark_announced(state, "A", now=1000.0)
        delta, _state = statuspage.classify(state, snap(view("A")))
        assert delta.opened == ()

    def test_disappear_then_reappear_announces_once(self):
        """The whole reason for a monotonic announced map: one transient empty
        body must not re-announce a live outage."""
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        _, state = statuspage.classify(state, snap(view("A")))
        state = statuspage.mark_announced(state, "A", now=1000.0)
        _, state = statuspage.classify(state, snap())  # transient empty
        delta, _state = statuspage.classify(state, snap(view("A")))
        assert delta.opened == ()


class TestCap:
    def test_opened_capped_newest_first_with_discard_count(self):
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        incoming = snap(*[view(f"I{n}", minutes=n) for n in range(5)])
        delta, _state = statuspage.classify(state, incoming, max_opened=3)
        assert [i.id for i in delta.opened] == ["I4", "I3", "I2"]
        assert delta.discarded == 2


class TestChangedAndDisappeared:
    def test_status_move_is_changed_not_opened(self):
        _, state = statuspage.classify(statuspage.StatusState(), snap(view("A")))
        delta, _state = statuspage.classify(state, snap(view("A", status="monitoring")))
        assert delta.opened == ()
        assert [i.id for i in delta.changed] == ["A"]

    def test_disappeared_carries_the_previous_view(self):
        """After an incident resolves it vanishes from summary.json along with
        its text, so the retained previous view is the only source for a
        future all-clear line."""
        _, state = statuspage.classify(statuspage.StatusState(), snap(view("A")))
        delta, state = statuspage.classify(state, snap())
        assert [i.id for i in delta.disappeared] == ["A"]
        assert delta.disappeared[0].name == "Incident A"
        assert "A" not in state.active


class TestOpenedAndChangedAreDisjoint:
    def test_unannounced_incident_with_a_status_move_is_only_opened(self):
        """A never-announced incident (delivery failed) whose status also
        moved must not appear in both opened and changed — that would let a
        future all-clear branch double-report it."""
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        # Seed "A" into state.active without marking it announced, mirroring
        # a dropped delivery.
        _, state = statuspage.classify(state, snap(view("A")))
        assert "A" not in state.announced

        delta, _state = statuspage.classify(state, snap(view("A", status="monitoring")))
        assert [i.id for i in delta.opened] == ["A"]
        assert delta.changed == ()

    def test_capped_opened_is_still_excluded_from_changed(self):
        """An incident dropped by max_opened is unannounced too — it must not
        fall through into changed."""
        _, state = statuspage.classify(statuspage.StatusState(), snap())
        _, state = statuspage.classify(state, snap(*[view(f"I{n}", minutes=n) for n in range(5)]))
        for n in range(5):
            assert f"I{n}" not in state.announced

        moved = snap(*[view(f"I{n}", minutes=n, status="monitoring") for n in range(5)])
        delta, _state = statuspage.classify(state, moved, max_opened=3)
        assert delta.discarded == 2
        assert set(delta.changed).isdisjoint(delta.opened)
        assert delta.changed == ()


class TestPruning:
    def test_announced_map_is_bounded(self):
        state = statuspage.StatusState(seeded=True)
        for n in range(300):
            state = statuspage.mark_announced(state, f"I{n}", now=float(n))
        _, state = statuspage.classify(state, snap())
        assert len(state.announced) <= statuspage.MAX_ANNOUNCED_RETAINED

    def test_pruning_never_drops_a_currently_active_id(self):
        state = statuspage.StatusState(seeded=True)
        for n in range(300):
            state = statuspage.mark_announced(state, f"I{n}", now=float(n))
        state = statuspage.mark_announced(state, "OLD", now=-1.0)
        _, state = statuspage.classify(state, snap(view("OLD")))
        assert "OLD" in state.announced, "an active id must survive pruning or it re-announces"
