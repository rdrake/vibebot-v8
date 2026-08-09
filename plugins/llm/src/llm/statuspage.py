"""Atlassian Statuspage v2 API model — pure, no supybot imports.

Fetching, strict parsing, incident-lifecycle classification, sanitisation of
third-party prose, and deterministic line rendering. Kept free of Limnoria
imports so it unit-tests against dicts with no IRC scaffolding, and free of
``llm.service`` imports so there is no import cycle (``service`` and
``plugin`` both consume this module).

The API contract is tenant-agnostic: every Statuspage-hosted service serves
the same ``/api/v2/summary.json``. The genericity lives here, in functions
that take a base URL, rather than in the config or the tool schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

INDICATORS: frozenset[str] = frozenset({"none", "minor", "major", "critical"})

INCIDENT_STATUSES: frozenset[str] = frozenset(
    {"investigating", "identified", "monitoring", "resolved", "postmortem"}
)

COMPONENT_STATUSES: frozenset[str] = frozenset(
    {
        "operational",
        "degraded_performance",
        "partial_outage",
        "major_outage",
        "under_maintenance",
    }
)

# Free-text fields quoted from the status page are capped before they reach
# either the model or the wire.
MAX_FREE_TEXT = 200


class InvalidPayload(ValueError):  # noqa: N818
    """The response was valid JSON but not a valid Statuspage summary.

    Raised rather than degrading to an empty snapshot: an empty snapshot would
    erase the active incident set and make the next poll re-announce a live
    outage.
    """


@dataclass(frozen=True)
class IncidentView:
    """One unresolved incident, field-whitelisted from the API payload."""

    id: str
    name: str
    status: str
    impact: str
    affected_components: tuple[str, ...]
    started_at: datetime | None
    created_at: datetime | None
    latest_update_body: str
    latest_update_at: datetime | None


@dataclass(frozen=True)
class Snapshot:
    """One parsed observation of a status page."""

    page_name: str
    page_url: str
    indicator: str
    description: str
    components: dict[str, str]
    incidents: dict[str, IncidentView]
    fetched_at: float
    etag: str | None = None
    modified: str | None = None


def _require_mapping(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidPayload(f"{field} is not an object")
    return value


def _require_list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        raise InvalidPayload(f"{field} is not a list")
    return value


def _require_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidPayload(f"{field} is not a string")
    return value


def _parse_ts(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, preserving offset. None on anything else.

    Statuspage emits ``...Z``; ``fromisoformat`` accepts that from 3.11 on.
    A missing or unparseable timestamp is tolerated (the field is optional in
    the wild) — it degrades an age calculation, it does not invalidate the
    payload.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _ts_key(value: Any) -> float:
    """Sortable float for a possibly-missing timestamp; undated sorts oldest."""
    stamp = _parse_ts(value)
    return stamp.timestamp() if stamp else float("-inf")


def _parse_incident(raw: Any) -> IncidentView:
    obj = _require_mapping(raw, "incident")

    incident_id = obj.get("id")
    if not isinstance(incident_id, str) or not incident_id:
        raise InvalidPayload("incident has no usable id")

    status = obj.get("status")
    if not isinstance(status, str) or status not in INCIDENT_STATUSES:
        raise InvalidPayload(f"unknown incident status: {status!r}")

    components = obj.get("components")
    affected: tuple[str, ...] = ()
    if isinstance(components, list):
        affected = tuple(
            c["name"] for c in components if isinstance(c, dict) and isinstance(c.get("name"), str)
        )

    updates = obj.get("incident_updates")
    body = ""
    update_at: datetime | None = None
    if isinstance(updates, list) and updates:
        parsed = [u for u in updates if isinstance(u, dict)]
        # Sort on a float, never on datetime: a payload mixing dated and
        # undated updates would otherwise compare aware against naive
        # datetimes and raise TypeError mid-parse.
        parsed.sort(key=lambda u: _ts_key(u.get("display_at")), reverse=True)
        if parsed:
            newest = parsed[0]
            body = newest.get("body") if isinstance(newest.get("body"), str) else ""
            update_at = _parse_ts(newest.get("display_at"))

    incident_name = obj.get("name")
    incident_impact = obj.get("impact")

    return IncidentView(
        id=incident_id,
        name=incident_name if isinstance(incident_name, str) else "",
        status=status,
        impact=incident_impact if isinstance(incident_impact, str) else "",
        affected_components=affected,
        started_at=_parse_ts(obj.get("started_at")),
        created_at=_parse_ts(obj.get("created_at")),
        latest_update_body=body,
        latest_update_at=update_at,
    )


def parse_summary(
    payload: Any,
    *,
    fetched_at: float,
    etag: str | None = None,
    modified: str | None = None,
) -> Snapshot:
    """Strictly parse a ``/api/v2/summary.json`` body into a Snapshot.

    Raises InvalidPayload on anything that is not structurally a summary. The
    caller must treat that as a failed poll: advance neither freshness nor
    lifecycle state.
    """
    root = _require_mapping(payload, "payload")

    status = _require_mapping(root.get("status"), "status")
    indicator = status.get("indicator")
    if not isinstance(indicator, str) or indicator not in INDICATORS:
        raise InvalidPayload(f"unknown indicator: {indicator!r}")
    description = _require_str(status.get("description"), "status.description")

    raw_components = _require_list(root.get("components"), "components")
    _require_list(root.get("incidents"), "incidents")
    _require_list(root.get("scheduled_maintenances"), "scheduled_maintenances")

    components: dict[str, str] = {}
    for item in raw_components:
        if not isinstance(item, dict):
            raise InvalidPayload("component is not an object")
        name = item.get("name")
        comp_status = item.get("status")
        if (
            not isinstance(name, str)
            or not isinstance(comp_status, str)
            or comp_status not in COMPONENT_STATUSES
        ):
            raise InvalidPayload(f"bad component entry: {name!r}/{comp_status!r}")
        components[name] = comp_status

    incidents: dict[str, IncidentView] = {}
    for item in root["incidents"]:
        view = _parse_incident(item)
        incidents[view.id] = view

    page = root.get("page") if isinstance(root.get("page"), dict) else {}

    page_name = page.get("name")
    page_url = page.get("url")

    return Snapshot(
        page_name=page_name if isinstance(page_name, str) else "",
        page_url=page_url if isinstance(page_url, str) else "",
        indicator=indicator,
        description=description,
        components=components,
        incidents=incidents,
        fetched_at=fetched_at,
        etag=etag,
        modified=modified,
    )


# Bound on the announced map. Pruning always retains currently-active ids
# regardless of age — dropping an active id would re-announce a live outage.
MAX_ANNOUNCED_RETAINED = 200


@dataclass(frozen=True)
class StatusState:
    """Lifecycle state for one status page. Advanced by the poller only."""

    active: dict[str, IncidentView] = field(default_factory=dict)
    announced: dict[str, float] = field(default_factory=dict)
    seeded: bool = False


@dataclass(frozen=True)
class Delta:
    """What changed between the retained state and a new snapshot."""

    opened: tuple[IncidentView, ...] = ()
    changed: tuple[IncidentView, ...] = ()
    disappeared: tuple[IncidentView, ...] = ()
    discarded: int = 0


def _sort_key(view: IncidentView) -> float:
    """Newest-first ordering key; undated incidents sort oldest."""
    stamp = view.started_at or view.created_at
    return stamp.timestamp() if stamp else float("-inf")


def _prune(announced: dict[str, float], active_ids: set[str]) -> dict[str, float]:
    if len(announced) <= MAX_ANNOUNCED_RETAINED:
        return announced
    keep = {k: v for k, v in announced.items() if k in active_ids}
    remainder = sorted(
        ((k, v) for k, v in announced.items() if k not in active_ids),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for key, value in remainder[:MAX_ANNOUNCED_RETAINED]:
        keep[key] = value
    return keep


def classify(
    state: StatusState,
    snapshot: Snapshot,
    *,
    max_opened: int = 3,
) -> tuple[Delta, StatusState]:
    """Classify a snapshot against retained state. Pure — mutates nothing.

    On a cold start (``state.seeded`` False) every current incident is
    recorded as already-announced and an empty Delta is returned, so a restart
    during an outage does not re-announce it. This mirrors stock RSS's
    ``initial`` flag.

    ``opened`` is NOT written into ``announced`` — the caller does that via
    ``mark_announced`` after a successful send, so a dropped delivery is
    retried on the next poll.
    """
    current = snapshot.incidents

    if not state.seeded:
        return Delta(), StatusState(
            active=dict(current),
            announced=dict.fromkeys(current, snapshot.fetched_at),
            seeded=True,
        )

    opened = [v for cid, v in current.items() if cid not in state.announced]
    opened.sort(key=_sort_key, reverse=True)
    discarded = max(0, len(opened) - max_opened)

    changed = tuple(
        v
        for cid, v in current.items()
        if cid in state.active and state.active[cid].status != v.status
    )
    disappeared = tuple(v for cid, v in state.active.items() if cid not in current)

    return (
        Delta(
            opened=tuple(opened[:max_opened]),
            changed=changed,
            disappeared=disappeared,
            discarded=discarded,
        ),
        StatusState(
            active=dict(current),
            announced=_prune(dict(state.announced), set(current)),
            seeded=True,
        ),
    )


def mark_announced(state: StatusState, incident_id: str, *, now: float) -> StatusState:
    """Record that ``incident_id`` was successfully announced."""
    announced = dict(state.announced)
    announced[incident_id] = now
    return StatusState(active=state.active, announced=announced, seeded=state.seeded)
