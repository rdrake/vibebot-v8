"""One-time #idlerpg loom/crosspoll exhaust purge for a single verse store.

NOT a migration: destructive, channel-specific, invoked explicitly ONCE
against prod #afternet after a WAL-safe backup (see the slice-1 design doc
§6). Never auto-runs. Everything happens in ONE ``write_transaction`` with
direct SQL so it respects the store's non-reentrant write lock — never call
public store methods from inside it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

_LOG = logging.getLogger("llm.verse.purge")


class PurgeResult(NamedTuple):
    events_deleted: int
    entities_deleted: int
    digests_restamped: int


def list_loom_digest_candidates(store: Any, *, min_chars: int = 300) -> list[tuple[int, str]]:
    """Return ``(id, summary)`` for ``source='loom'`` events long enough to be
    compaction lore-digests rather than #idlerpg combat lines.

    Read-only. The operator REVIEWS this list and passes the confirmed ids to
    :func:`purge_loom_data` as ``digest_ids`` — we never re-stamp on length
    alone (a long combat brag could clear the threshold).
    """
    with store.read_connection() as conn:
        rows = conn.execute(
            "SELECT id, summary FROM events WHERE source='loom' "
            "AND length(summary) >= ? ORDER BY id",
            (min_chars,),
        ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def purge_loom_data(store: Any, *, digest_ids: Sequence[int] = ()) -> PurgeResult:
    """Delete #idlerpg loom/crosspoll events + their orphaned auto-NPCs.

    ``digest_ids`` are the reviewed compaction lore-digest event ids (from
    :func:`list_loom_digest_candidates`). They are re-stamped ``source='llm'``
    FIRST, so they survive the event delete and so their actors are not
    counted as loom-only orphans.

    One ``write_transaction``: on any error the whole op rolls back.
    """
    digest_id_list = [int(x) for x in digest_ids]
    with store.write_transaction() as conn:
        # 0. Protect reviewed compaction digests (only flip rows still 'loom').
        restamped = 0
        if digest_id_list:
            placeholders = ",".join("?" for _ in digest_id_list)
            cur = conn.execute(
                f"UPDATE events SET source='llm' WHERE id IN ({placeholders}) AND source='loom'",
                digest_id_list,
            )
            restamped = cur.rowcount

        # 1. Compute orphans BEFORE deleting events (event_actor cascades in
        #    step 2). Orphan = auto_created, not pinned/author_locked, has >=1
        #    actor link, and NONE of its actor links point to a surviving
        #    (non-loom/crosspoll) event.
        orphan_rows = conn.execute(
            """
            SELECT e.id FROM entities e
            WHERE EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='auto_created' AND a.value='1'
            )
            AND NOT EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='pinned' AND a.value='1'
            )
            AND NOT EXISTS (
                SELECT 1 FROM attributes a
                WHERE a.entity_id=e.id AND a.key='author_locked' AND a.value='1'
            )
            AND EXISTS (
                SELECT 1 FROM event_actor ea WHERE ea.entity_id=e.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM event_actor ea JOIN events ev ON ev.id=ea.event_id
                WHERE ea.entity_id=e.id AND ev.source NOT IN ('loom','crosspoll')
            )
            """
        ).fetchall()
        orphan_ids = {int(r[0]) for r in orphan_rows}

        # Dual-linkage guard: an entity can also be referenced by a surviving
        # event via the legacy events.entity_ids JSON without an event_actor
        # row. Never delete such an entity even if event_actor says orphan.
        if orphan_ids:
            referenced_json: set[int] = set()
            for (blob,) in conn.execute(
                "SELECT entity_ids FROM events WHERE source NOT IN ('loom','crosspoll')"
            ).fetchall():
                try:
                    for eid in json.loads(blob or "[]"):
                        if isinstance(eid, int):
                            referenced_json.add(eid)
                except (ValueError, TypeError):
                    # A corrupt entity_ids blob on a SURVIVING event must not
                    # silently lose its protective effect in a destructive op.
                    # We skip the row (its ids aren't added to the protected
                    # set) but WARN so the operator reviews before trusting the
                    # counts — an entity that should have been spared could
                    # otherwise be deleted unnoticed.
                    _LOG.warning(
                        "purge: unparseable entity_ids on a surviving event; "
                        "skipping it for dual-linkage protection"
                    )
                    continue
            orphan_ids -= referenced_json

        # 2. Delete loom/crosspoll events (cascades event_actor; the legacy
        #    entity_ids JSON lives on the deleted row, so both linkages go).
        cur = conn.execute("DELETE FROM events WHERE source IN ('loom','crosspoll')")
        events_deleted = cur.rowcount

        # 3. Delete orphaned auto-NPCs (cascades attributes/relations/
        #    entity_alias/event_actor).
        entities_deleted = 0
        if orphan_ids:
            id_list = sorted(orphan_ids)
            placeholders = ",".join("?" for _ in id_list)
            cur = conn.execute(f"DELETE FROM entities WHERE id IN ({placeholders})", id_list)
            entities_deleted = cur.rowcount

    return PurgeResult(events_deleted, entities_deleted, restamped)
