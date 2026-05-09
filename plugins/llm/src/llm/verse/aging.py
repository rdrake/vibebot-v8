"""Soft-retire auto_created NPCs that have been silent past the cutoff.

A pure helper module. Runs in the compaction pass per channel. No
schedule of its own. ``retire_after_days <= 0`` disables — early
return."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

_LOG = logging.getLogger("llm.verse.aging")
_SECONDS_PER_DAY = 86400.0


class AgingOutcome(NamedTuple):
    scanned: int
    retired: int


def age_auto_created_entities(
    store: Any,
    *,
    retire_after_days: int,
    now: Callable[[], float],
) -> AgingOutcome:
    """Soft-retire auto_created='1' entities whose last_seen_ts is
    older than now - retire_after_days*86400. Skips kind='avatar'
    defensively. Returns counts (scanned, retired). retire_after_days<=0
    disables — returns (0, 0)."""
    if retire_after_days <= 0:
        return AgingOutcome(scanned=0, retired=0)

    cutoff = now() - retire_after_days * _SECONDS_PER_DAY
    candidates = store.list_entities_with_attribute(key="auto_created", value="1", status="active")
    scanned = 0
    retired = 0
    for entity in candidates:
        if entity.kind == "avatar":
            continue  # defensive — auto_created on an avatar is a bug, not a target
        scanned += 1
        last_seen_str = store.get_attribute(entity.id, "last_seen_ts")
        if last_seen_str is None:
            continue  # no heartbeat → no decision; leave it
        try:
            last_seen = float(last_seen_str)
        except ValueError:
            _LOG.warning(
                "verse aging: malformed last_seen_ts on entity %s: %r",
                entity.id,
                last_seen_str,
            )
            continue
        if last_seen < cutoff:
            store.set_status(entity.id, "retired")
            retired += 1
    return AgingOutcome(scanned=scanned, retired=retired)
