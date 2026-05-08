"""Daily retention compaction for forest-verse channels.

A pure helper (`compact_verse`) and a thin scheduling driver
(`register_daily_timer` / `cancel_daily_timer`). The helper is the unit
of work; the driver picks the fire time, walks verse-enabled channels,
and invokes the helper. Failures abort the helper but never the timer.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

_LOG = logging.getLogger("llm.verse.compaction")
SECONDS_PER_DAY = 86400

# Per-pass tunables. Constants — operators tune retention via registry;
# the rest are safety-net caps.
_MAX_EVENTS_PER_PASS = 200
"""Hard cap on how many old events one compaction pass touches. Long
backlogs drain across multiple daily runs; one pass writes one digest
event covering at most this many originals."""

_MAX_SUMMARY_CHARS_PER_EVENT = 240
"""Per-event truncation in the bullet block; longer summaries are
elided with an ellipsis. Stops a single pathological event from
blowing past the cheap model's context."""

_MAX_BULLET_BLOCK_CHARS = 16000
"""Hard cap on the user-message bullet block, after per-event
truncation. Trim from the front (oldest) so the *newest* of the old
events stay in the prompt."""

_MAX_DIGEST_ENTITY_IDS = 32
"""Cap on the lore-digest event's entity_ids array. Beyond this we log
INFO + drop the rest; entity-heavy verses lose some grounding in their
digest event but the events table remains the canonical truth."""


def compact_verse(
    store: Any,
    *,
    retention_days: int,
    min_keep_events: int,
    model: str,
    client: Any,
    log_usage: Callable[..., None],
    now: Callable[[], float],
) -> str:
    """Compact a single verse. Returns one of:

    - ``'compacted'`` — old events replaced by one digest event
    - ``'skipped_disabled'`` — ``retention_days <= 0``
    - ``'skipped_below_floor'`` — fewer than ``min_keep_events`` total
      events in the store
    - ``'skipped_no_events'`` — no events older than the retention window

    Per-pass behaviour: a single call processes at most
    ``_MAX_EVENTS_PER_PASS`` (200) old events. If the verse has a long
    backlog, additional daily runs drain it incrementally; the digest
    event written by *this* pass covers only the events the LLM
    actually saw.
    """
    if retention_days <= 0:
        return "skipped_disabled"

    with store.read_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if total < min_keep_events:
        return "skipped_below_floor"

    cutoff_ts = now() - retention_days * SECONDS_PER_DAY
    olds = store.events_older_than(cutoff_ts=cutoff_ts)
    if not olds:
        return "skipped_no_events"

    # Process the OLDEST batch first. This guarantees forward progress:
    # even if the verse keeps receiving new events past the retention
    # window, the floor on the events-older-than query keeps shrinking.
    batch = olds[:_MAX_EVENTS_PER_PASS]

    def _truncated(s: str) -> str:
        if len(s) <= _MAX_SUMMARY_CHARS_PER_EVENT:
            return s
        return s[: _MAX_SUMMARY_CHARS_PER_EVENT - 1] + "…"

    # Pair each event with its bullet so the trim drops both together.
    # Without this, oldest bullets could be dropped from the prompt
    # while their event ids remained in delete_ids — deleting events
    # the LLM never saw.
    pairs: list[tuple[Any, str]] = [(e, f"- {_truncated(e.summary)}") for e in batch]
    bullets = "\n".join(b for _, b in pairs)
    if len(bullets) > _MAX_BULLET_BLOCK_CHARS:
        # Trim oldest pairs first; newest of the batch stay in.
        while pairs and len("\n".join(b for _, b in pairs)) > _MAX_BULLET_BLOCK_CHARS:
            pairs.pop(0)
        bullets = "\n".join(b for _, b in pairs)
        _LOG.info(
            "verse compaction: bullet block trimmed to %d chars over %d-event batch",
            len(bullets),
            len(pairs),
        )
    kept_events = [e for e, _ in pairs]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chronicler. Summarise the bullet list of past "
                "events into one paragraph (≤120 words). Do not invent "
                "details; only summarise what is in the list."
            ),
        },
        {"role": "user", "content": bullets},
    ]
    content, usage = client.call(op="compact", model=model, messages=messages)
    summary = (content or "").strip() or "A period of unrecorded events passed."

    # delete_ids and entity_ids union are computed only over the events
    # that actually appeared in the prompt — anything trimmed off the
    # front survives for the next pass.
    delete_ids = [e.id for e in kept_events]
    union_ids: list[int] = []
    seen: set[int] = set()
    for ev in kept_events:
        for eid in ev.entity_ids:
            if eid not in seen:
                seen.add(eid)
                union_ids.append(eid)

    if len(union_ids) > _MAX_DIGEST_ENTITY_IDS:
        _LOG.info(
            "verse compaction: digest entity_ids truncated %d → %d "
            "(union over %d events); rest dropped",
            len(union_ids),
            _MAX_DIGEST_ENTITY_IDS,
            len(kept_events),
        )
        union_ids = union_ids[:_MAX_DIGEST_ENTITY_IDS]

    # Stamp the digest at the most-recent ts of the kept batch so it
    # remains ordered *before* any surviving fresh events (which all
    # have ts >= cutoff). Using ``now()`` would push the digest to the
    # head of the timeline, hiding the fresh events behind it.
    digest_ts = max(e.ts for e in kept_events)
    store.replace_events_with_lore_digest(
        delete_ids=delete_ids,
        summary=summary,
        entity_ids=union_ids,
        ts=digest_ts,
    )
    log_usage(op="compact", model=model, usage=usage)
    return "compacted"


def register_daily_timer(
    *,
    schedule_module: Any,
    fire_at_local: str,
    callback: Callable[[], None],
    name: str = "llm_verse_compact",
    now: Callable[[], float] | None = None,
) -> None:
    """Register a single-shot ``schedule.addEvent`` for the next time the
    local clock reaches ``fire_at_local`` (HH:MM). The callback re-arms
    itself at the end of its run; this function is called once at plugin
    load.

    If a timer with ``name`` is already registered, it is cancelled first
    so duplicate registrations cannot crash ``schedule.addEvent``'s
    name-uniqueness check.
    """
    import time as _time

    now_fn: Callable[[], float] = now if now is not None else _time.time
    cancel_daily_timer(schedule_module=schedule_module, name=name)
    fire_at = _next_local_time(fire_at_local, now=now_fn)
    schedule_module.addEvent(callback, fire_at, name=name)


def cancel_daily_timer(*, schedule_module: Any, name: str = "llm_verse_compact") -> None:
    with contextlib.suppress(KeyError):
        schedule_module.removeEvent(name)


def _next_local_time(hhmm: str, *, now: Callable[[], float]) -> float:
    """Return the next epoch second whose local time is ``hhmm``.

    Falls back to one hour from now if ``hhmm`` is malformed.
    """
    import time

    try:
        h, m = (int(x) for x in hhmm.split(":", 1))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
    except ValueError:
        _LOG.warning("verseCompactionDailyAt malformed (%r); deferring 1h", hhmm)
        return now() + 3600.0
    cur = time.localtime(now())
    candidate = time.mktime(
        (
            cur.tm_year,
            cur.tm_mon,
            cur.tm_mday,
            h,
            m,
            0,
            cur.tm_wday,
            cur.tm_yday,
            cur.tm_isdst,
        )
    )
    if candidate <= now():
        candidate += SECONDS_PER_DAY
    return candidate
