# Queryable status allowlist (phase 2)

**Date:** 2026-08-17
**Status:** approved
**Builds on:** `2026-08-17-multi-source-status-design.md` and
`2026-08-17-incident-io-support-design.md`, both shipped (`c63ce9e..e11db64`)

Realises the "Planned follow-up: a queryable allowlist (phase 2)" section of the
multi-source spec, which was deliberately deferred until multi-source polling was
running.

## Problem

The bot polls three status pages and answers about all of them. Adding a fourth means
polling it forever: a fetch every 120 seconds, a `StatusState`, an entry in six keyed
dicts, and a slot against the 5-source cap — whether or not anyone ever asks.

That cost is right for pages the channel wants *announced*. It is wrong for the long
tail — Cloudflare, AWS, Discord, npm — where the bot only needs to answer a question
when one is asked.

**Polled pages and queryable pages need not be the same set.** Polling costs run
continuously; querying costs one lazy cached fetch, only on demand.

## Decisions

1. **A second config key, `statusQueryablePages`.** A space-separated list of `Name=url`
   pairs, default empty. Names are operator-supplied because the model must be able to
   select one, and a page's own `page_name` is not known until after a fetch — which is
   exactly the fetch we are trying to avoid making.

2. **Selection is an enum-constrained `service` argument.** The provider constrains the
   model to configured values, so there is no fuzzy matcher, no alias table, and no
   unknown-name branch. Rejected: a free-text argument matched server-side, which needs
   all three and lets the model invent a name that quietly matches the wrong page.

3. **`service` omitted keeps today's behaviour exactly** — every polled source, which is
   what makes "are Claude and Codex up?" work in one call. `service` given returns that
   one page only. The common case stays free; the allowlist is opt-in per question.

4. **The enum is built per call from live state.** This is the one thing the phase-2
   sketch did not anticipate. The enum must name polled pages too, but `statusPageUrls`
   holds bare URLs and their display names come from `page_name` in the fetched payload.
   Rather than change a config key that shipped hours ago, the enum is assembled at
   profile-build time from `_status_read_cache[source].page_name`, falling back to the
   host for a page not yet polled. `_with_status_hosts` already rebuilds the schema per
   completion, so this adds no new machinery and self-heals within one 120s poll of
   startup.

5. **Allowlisted pages are never polled, never announced, and never touch
   `_status_state`.** The ownership split that the multi-source work established is
   unchanged: lifecycle state belongs to the poller alone. These pages have no lifecycle
   at all — there is nothing to announce, so there is nothing to consume.

6. **A name resolves to a polled source first.** A URL present in both lists yields the
   polled entry, which is fresher and carries lifecycle state. Dedupe is by canonical
   source id, the same identity the multi-source work established.

## Architecture

### Configuration

```python
conf.registerGlobalValue(
    LLM,
    "statusQueryablePages",
    registry.SpaceSeparatedListOfStrings(
        [],
        _("""..."""),
    ),
)
```

Each entry is `Name=url`. `Name` matches `[A-Za-z0-9._-]{1,32}` and must be unique
case-insensitively; `url` goes through `statuspage.canonical_source`. An entry failing
either check is logged once and dropped, matching how `statusPageUrls` treats a bad
entry — one typo must not disable the rest.

Capped at `_STATUS_MAX_QUERYABLE = 20`. The cap is generous because these cost nothing
until asked for; it exists to bound the enum's token cost and the cache, not the work.

The list is space-separated and names forbid spaces, so `Name=url` parses unambiguously
on the first `=`.

### Name resolution

A new `_status_named_pages()` returns an ordered mapping of display name → canonical
source, built from both keys:

1. Polled sources first, in `statusPageUrls` order. Name is
   `_status_read_cache[source].page_name`, sanitised and URL-stripped, falling back to
   `_status_host(source)` when the page has not been read yet.
2. Then allowlist entries, skipping any whose canonical source is already present from
   step 1 (decision 6).

Case-insensitive collisions resolve to the first occurrence, so a polled page always
wins its name.

### Fetch path

Allowlisted pages reuse `_status_fetch_snapshot` unchanged. State is one new dict:

| Field | Purpose |
|---|---|
| `_status_query_cache: dict[str, Snapshot]` | canonical source → last good reading |
| `_status_query_failed_at: dict[str, float]` | failure backoff, mirroring history |

- TTL `_STATUS_QUERY_TTL = 300`. Shorter than the 600s staleness line because nothing
  else refreshes these — the poller never touches them.
- Failure backoff reuses `_STATUS_HISTORY_RETRY = 120`.
- Bounded at `_STATUS_QUERY_CACHE_MAX = 10` entries, evicting the oldest `fetched_at`
  first. The allowlist may hold 20 pages; the cache holds the 10 most recently read.
  Unbounded growth is the failure mode the multi-source work had to fix by pruning six
  dicts, and this one has no config-driven prune point because entries appear on demand.
- The existing `_STATUS_TOOL_BUDGET = 20` covers it; a targeted query is a single fetch.

### Tool payload

`_status_tool_payload(*, service=None, include_history=False)`:

- `service is None` — unchanged from today: every polled source, aggregate error when
  none could be read.
- `service` names a polled source — a one-entry `services` list built from the poller's
  read cache, refreshed by the same staleness rule.
- `service` names an allowlist page — a one-entry list from `_status_query_cache`,
  fetched on miss or past TTL.
- `service` names nothing (possible if the enum and the config drift within one
  completion) — return the full polled set with a `note` saying the requested name is
  not configured, rather than an error. Failing soft here is right: the model asked a
  reasonable question and the polled answer is still useful.

Entry shape is unchanged — `source`, `service`, `indicator`, `degraded`, `incidents`,
per-entry `error`/`stale`, optional `recent_incidents`.

### Tool schema

`_with_status_hosts` grows into `_with_status_context`, which now injects both the
configured host list into the description and the `service` enum into the parameters. It
keeps copying both dict levels — `ToolSpec.as_tool()` returns a fresh outer dict but
shares the module-level `schema`, so an in-place edit corrupts it process-wide.

When no name is resolvable the `service` property is omitted entirely rather than
emitted with an empty enum, which some providers reject.

## Error handling

| Condition | Behaviour |
|---|---|
| Malformed allowlist entry | Logged once, dropped, rest kept |
| Duplicate name | First wins, later logged and dropped |
| Allowlist URL also polled | Resolves to the polled source |
| Named page unreachable | That entry carries `error`; no fallback to other pages |
| `service` not resolvable | Full polled set plus an explanatory `note` |
| Cache at capacity | Oldest `fetched_at` evicted |
| Budget spent | Cached reading marked `stale`, or an `error` entry if never read |

## Testing

1. `Name=url` parsing: valid pairs, bad name charset, missing `=`, unparseable URL,
   duplicate names, over-cap — each logged and dropped without affecting the rest.
2. Name resolution prefers a polled source over an allowlist entry for the same URL.
3. Enum contents: polled pages appear by `page_name` once read, by host before that;
   allowlist pages by operator name.
4. `service` omitted returns every polled source — the existing behaviour, pinned so it
   cannot regress.
5. `service` naming a polled page returns exactly that one.
6. `service` naming an allowlist page fetches lazily, and a second call inside the TTL
   does not re-fetch.
7. An unresolvable `service` returns the polled set with the note, not an error.
8. Cache eviction at capacity drops the oldest entry.
9. `_status_state` is untouched by every allowlist path — the ownership invariant.
10. The module-level tool schema is not mutated across repeated builds.

The traps that bit the previous two plans still apply: a gate on the caller side of a
seam is proven by nothing on the callee side, and any new method reachable from the
announce path must be bound in `conftest.py`'s `announcing_plugin` fixture. This work
does not touch the announce path — if a change appears to, that is a design error, not a
fixture problem.

## Cutover

`statusQueryablePages` defaults to empty, so deploying changes nothing observable. The
operator opts in by setting it — and because the key will not exist in `bot.conf` until
the bot has run once with it registered, the first set must be `@config` or a
stop-edit-start, per the warning in `docs/guide/operator/configuration.md`.

## Deliberately out of scope

- **Announcing allowlisted pages.** They have no lifecycle state by design. Wanting
  announcements means moving the page to `statusPageUrls`.
- **Aliases.** Decision 3 of the incident.io spec stands: the model maps "Codex" to
  OpenAI itself. An alias list is the fallback if a wrong answer is actually observed.
- **Fully open "any Statuspage URL".** Rejected twice now, for the same reason: a
  hallucinated hostname becomes a fetch whose third-party prose the bot speaks
  unprompted.
- **Raising the 5-source polling cap.** Unrelated; the allowlist exists precisely so the
  polled set can stay small.
