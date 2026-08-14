# Status-announce restart gap

**Status:** Noted, not started (2026-08-14)
**Author:** Richard Drake (with claude)
**Affects:** `statusAnnounce` incident announcer, shipped 2026-08-09
(`8e448c7..c4cf7b7`)
**Priority:** Low. One channel is opted in (`#clanker`), stock RSS is still
announcing the same feed alongside it, and the failure mode is a missed
line rather than a wrong one.

## Summary

`_status_state` lives only in memory. Every restart re-seeds it, and the
seeding path deliberately records every currently-open incident as
already-announced. An incident that opens while the container is
restarting is therefore never announced — not late, not twice, never.

Stock RSS does not have this gap: it persists announced entry ids to disk
and diffs against the persisted set. So a bot restart during an outage
produces exactly the asymmetry that makes the two announcers look
inconsistent when compared side by side.

## Not a repeat-announcement risk

Worth stating plainly, because the two get conflated: the announcer
cannot double-report an incident. Cold-start seeding
(`statuspage.py:385-390`) is what prevents a restart from re-announcing
an ongoing outage, and `mark_announced` is monotonic within a process
lifetime. The missed announcement *is the price of* the no-repeat
guarantee, not a separate bug alongside it.

## Gap 1 — restart amnesia (live)

`plugin.py:869` constructs `StatusState()` fresh on every plugin load,
with no load-from-disk. On the first poll after start, `classify` takes
the unseeded branch (`statuspage.py:385-390`) and writes every incident
in `summary.json` into `announced` with an empty `Delta`.

Exposure window per restart: container downtime, plus up to
`_STATUS_POLL_INTERVAL` (120s) until the first poll lands. Auto-deploy on
a green Docker build restarts the service on every merge to `main`, so
this window opens several times on an active day.

Consequence: an incident opening inside that window is seeded as
already-announced and never fires. It also never fires later, because
`announced` is monotonic and `summary.json` only lists unresolved
incidents.

## Gap 2 — partial delivery marks globally (dormant)

In `_announce_status` (`plugin.py:1411-1462`), `delivered` is a single
flag spanning every overlay group and every channel. Any one successful
`_safe_queue` marks the incident announced for all of them
(`plugin.py:1459-1462`), and channels excluded from `deliverable` because
the bot was not in them at that instant (`plugin.py:1419-1423`) are never
retried.

Dormant today: with one opted-in channel, a failed delivery leaves the
incident unmarked and the next poll retries it, which is the desired
behaviour. This only bites at two or more opted-in channels — for
example, one channel netsplit while another is fine.

## Reference implementation, already in the tree

`supybot/plugins/RSS/plugin.py` solves the same problem and is worth
copying rather than redesigning:

- Announced ids persist to `RSS_announced.flat`, path built at
  `plugin.py:77` via `conf.supybot.directories.data.dirize`.
- Loaded in `__init__` at `plugin.py:243` before any feed is registered.
- Written by `_flush` (`plugin.py:266`), registered on
  `world.flushers` (`plugin.py:259`) and called again from `die`
  (`plugin.py:261`), so a clean shutdown and the periodic flush both
  persist it.
- `get_new_entries` (`plugin.py:421-432`) diffs the fetched entries
  against the persisted set, so a headline published during downtime is
  still new after a restart. `initialAnnounceHeadlines` (`config.py:108`)
  caps the post-restart burst at 5 by default rather than suppressing it.

Note the truncation detail at `plugin.py:431`: RSS keeps `10 * len(entries)`
ids rather than an unbounded set, so a re-appearing old entry does not
re-announce. Any port needs the same bound.

## Sketch of a fix

1. Persist `announced` (id → timestamp) to a small JSON file under the
   data directory. `active` does not need persisting — it is rebuilt from
   the next snapshot.
2. Load it in `__init__` and construct `StatusState(announced=..., seeded=True)`
   when the file exists, so the first post-restart poll classifies against
   real history instead of taking the seed branch.
3. Register the writer on `world.flushers` and call it from the existing
   `die` (`plugin.py:906`), matching RSS.
4. Age entries out — drop ids older than a few days — so the file cannot
   grow without bound. `mark_announced` already stamps a timestamp, which
   is exactly what the reaper needs.
5. Keep the seed branch for the genuine first run, when no file exists.

Cap the post-restart burst the way `initialAnnounceHeadlines` does.
`_STATUS_MAX_ANNOUNCE_PER_POLL` (3) already bounds one poll, but a long
outage with several concurrent incidents would otherwise trickle out
across consecutive polls after a restart.

Gap 2 is a separate, smaller change: track delivery per channel rather
than one boolean, and only mark the incident announced once every opted-in
channel has taken it (or has been unreachable long enough to give up on).
Worth doing only when a second channel opts in.

## Testing notes

- Restart amnesia needs a test that survives a simulated process
  boundary: build a state, persist it, construct a fresh plugin, and
  assert the first `classify` reports the incident as opened rather than
  seeding it away.
- The existing seeding test must stay — first-run behaviour is unchanged
  and is what stops a restart storm.
- `_status_now` is already the clock indirection point, so the age-out
  reaper is testable without sleeping.
