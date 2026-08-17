# Queryable status allowlist (phase 2)

**Date:** 2026-08-17
**Status:** approved, red-teamed, revised
**Builds on:** `2026-08-17-multi-source-status-design.md` and
`2026-08-17-incident-io-support-design.md`, both shipped (`c63ce9e..e11db64`)

Revision history: v1 drafted and approved 2026-08-17; substantially rewritten the same
day after two adversarial passes (self red-team and an independent Codex pass) returned
eleven findings, two of which invalidated v1's central mechanism. Findings are marked
**[RT-S-n]** (self) and **[RT-C-n]** (Codex) inline.

## Problem

The bot polls three status pages. Adding a fourth means polling it forever: a fetch every
120 seconds, a `StatusState`, an entry in six keyed dicts, and a slot against the
5-source cap — whether or not anyone ever asks.

That cost is right for pages the channel wants *announced*. It is wrong for the long tail
— Cloudflare, AWS, Discord, npm — where the bot only needs to answer when asked.

Polled pages and queryable pages need not be the same set. Polling costs run
continuously; querying costs one lazy cached fetch, on demand.

## What v1 got wrong

v1 derived the selector vocabulary from each page's own `page_name`, resolved at call
time from the read cache. Both reviews rejected it independently, for two reasons:

**[RT-S-1, RT-C-2] It reversed a shipped security decision.** The multi-source spec
states that `page_name` is display-only and the operator-configured `source` is identity,
precisely because one page can rename itself "Claude". v1 promoted that same field into
*routing authority*: an operator configures `Cloudflare=https://www.cloudflarestatus.com`,
a compromised polled page returns `page.name = "Cloudflare"`, v1's "polled wins" rule
hands the name to the attacker, and "is Cloudflare down?" is answered from their page.
Sanitising does not turn third-party text into trustworthy identity.

**[RT-S-2, RT-C-5] It churned the paid prompt cache.** Tool schemas are part of the
cached prefix (`service.py:2798` defines the system message plus tool schemas as the bytes
that must be byte-identical for a cache hit). v1's enum changed as the read cache filled,
guaranteeing at least one invalidation per restart — and a hostile page could alternate
its name every poll to invalidate the prefix for every channel, indefinitely.

**The fix for both: enum values come only from operator config.** That makes the
vocabulary a pure function of `bot.conf`, stable across restarts and polls, containing
nothing a third party can influence. `page_name` keeps its existing job — the `service`
display field in the payload, sanitised prose.

## Decisions

1. **Both config keys share one `Name=url` grammar.** `statusPageUrls` gains optional
   names; a bare URL stays valid and falls back to its host. `statusQueryablePages` is
   new and uses the same form. One grammar, one parser, one vocabulary.

2. **Selection is an enum-constrained `service` argument**, built from config alone. The
   provider constrains the model to configured values, so there is no fuzzy matcher, no
   alias table, and no unknown-name branch on the happy path. Rejected: free text matched
   server-side, which needs all three and lets the model invent a name that quietly
   matches the wrong page.

3. **`service` omitted keeps today's behaviour exactly** — every polled source, which is
   what makes "are Claude and Codex up?" work in one call. `service` given returns that
   page only.

4. **[RT-C-3] The tool is gated on polled OR queryable being non-empty.** Today
   `status_fn` and the schema exclusion both key on `_status_sources()` alone
   (`service.py:5197`, `:5222`), so an operator with an empty `statusPageUrls` and twenty
   queryable pages would get no tool at all — defeating the exact separation this feature
   exists to provide.

5. **[RT-C-4] The name→source mapping is resolved once per completion and frozen.**
   `profile_tools` is built once (`service.py:5232`) and reused across every turn of the
   tool loop, and the executor performs no schema validation before dispatch
   (`assistant.py:832`). Recomputing resolution at dispatch time lets a call routed by one
   mapping be executed against another. The mapping built for the enum is captured and
   handed to the executor, so the schema and the dispatcher are one snapshot.

6. **Allowlisted pages are never polled, never announced, and never touch
   `_status_state`.** They have no lifecycle, so there is nothing to announce and nothing
   to consume.

7. **A URL in both keys resolves to the polled entry**, which is fresher and carries
   lifecycle state. Dedupe is by canonical source. Unlike v1 this is now safe, because
   both names are operator-chosen — the collision is a config mistake, not an attack.

## Architecture

### Configuration

```python
conf.registerGlobalValue(
    LLM, "statusPageUrls",
    registry.SpaceSeparatedListOfStrings(
        ["Claude=https://status.claude.com",
         "GitHub=https://www.githubstatus.com",
         "OpenAI=https://status.openai.com"], _("""...""")))

conf.registerGlobalValue(
    LLM, "statusQueryablePages",
    registry.SpaceSeparatedListOfStrings([], _("""...""")))
```

Grammar, shared by both keys and implemented once:

- `Name=url` or bare `url`. Split on the **first** `=`; names forbid `=` and (being in a
  space-separated list) spaces, so this is unambiguous.
- `Name` matches `[A-Za-z0-9._-]{1,32}`. ASCII-only, so plain `.lower()` is sufficient for
  case-insensitive uniqueness.
- A bare URL takes its canonical host as its name, which is what `statusPageUrls` entries
  do today.
- `url` goes through `statuspage.canonical_source`.
- An entry failing any check is logged and dropped; the rest survive. One typo must not
  disable the feature.
- **[RT-C-11]** Two entries whose *names* differ but whose *canonical sources* collide
  (`Foo=https://x` and `Bar=https://x/`) are a config error, not a silent skip: the later
  one is dropped **with a log line naming both**. v1 would have shown `Bar` as valid in
  `@config` while it never appeared in the enum.
- **[RT-C-11]** Warnings fire on the poll path only, reusing the shipped `warn=False`
  convention (`plugin.py:1083`) so a typo does not log once per chat message.

`_STATUS_MAX_QUERYABLE = 20`.

### Name resolution

`_status_named_pages(*, warn=True) -> dict[str, str]` returns an ordered map of name →
canonical source, built purely from config: polled entries first in configured order,
then queryable entries, skipping any canonical source already present (decision 7).
Case-insensitive name collisions resolve to the first occurrence, logged.

This function is the single source of truth for the enum, for `service` resolution, and
for the prune sets below.

### Fetch and caching

Allowlisted pages use `_status_fetch_snapshot`, with one change:

**[RT-C-10]** It currently reads ETag and Last-Modified from `_status_read_cache`
(`plugin.py:1195`). It gains an explicit `cached: Snapshot | None` parameter so the
caller supplies validators from whichever cache it owns; otherwise every allowlist
refresh is an unconditional full GET.

| Field | Purpose |
|---|---|
| `_status_query_cache: dict[str, Snapshot]` | canonical source → last good reading |
| `_status_query_failed_at: dict[str, float]` | failure backoff |

- TTL `_STATUS_QUERY_TTL = 300`, shorter than the 600s staleness line because nothing
  else refreshes these.
- Failure backoff reuses `_STATUS_HISTORY_RETRY = 120`.
- **[RT-C-9] The cache bound equals the allowlist cap (20), not 10.** A 10-entry cache
  under a 20-page allowlist thrashes deterministically: cycling through all twenty inside
  the TTL evicts every entry before it is reused, so every request fetches despite the
  cache. With the bound equal to the cap, eviction only ever fires after config churn.
  Oldest `fetched_at` is evicted first.
- **[RT-C-8] `_status_query_failed_at` is pruned, not merely capped.** v1 bounded only the
  snapshot cache. Rotating twenty failing URLs through config leaves a permanent failure
  timestamp per canonical source ever configured — the exact leak the shipped six-way
  prune exists to prevent.

### Pruning — now eight structures against two sets

**[RT-C-7] This is the subtlest interaction and v1 missed it entirely.** The shipped
`_status_prune_sources` prunes the three history dicts against the **polled** set
(`plugin.py:1115`, called from the poll at `:1426`). An allowlisted history query fetches
up to 4 MB and caches it for an hour — and the next 120-second poll deletes all three of
its entries, along with its failure backoff. The following question refetches 4 MB.

`_status_prune_sources` therefore prunes against **polled ∪ queryable**, from
`_status_named_pages()`, and covers the two new dicts as well as the original six. The
lifecycle dicts (`_status_state`, `_status_read_cache`, `_status_last_fetch`) still prune
against the polled set alone — a queryable page must never acquire lifecycle state.

### Tool payload

`_status_tool_payload(*, service=None, include_history=False, pages=None)` where `pages`
is the frozen mapping from decision 5.

- `service is None` — unchanged: every polled source, aggregate error when none read.
- `service` names a polled source — a one-entry list from the poller's read cache under
  the existing staleness rule.
- `service` names a queryable page — a one-entry list from `_status_query_cache`, fetched
  on miss or past TTL.
- **[RT-S-3, RT-C-6]** `service` resolves to nothing — return the polled set **and** a
  top-level `"error"` naming the unconfigured service. v1 returned a bare `note`, but
  `service.py:5592` records any dict without `"error"` as a *successful* tool call, so
  the model would receive healthy Claude/GitHub/OpenAI entries with nothing forcing it to
  notice the request went unanswered — and could summarise those as Cloudflare's state.
  The polled data still rides along because it is genuinely useful; the `error` is what
  makes the miss legible. This is distinct from the existing top-level `note`, which
  carries the untrusted-content warning and is unchanged.

Entry shape is otherwise unchanged.

### Tool schema injection

`_with_status_hosts` becomes `_with_status_context`, injecting both the host list into the
description and the `service` enum into the parameters.

**[RT-C-1] It must copy four levels, not two.** The shipped helper copies the tool and
`function` dicts (`service.py:141`), which was sufficient when only `description` changed.
Writing into `parameters.properties` reaches objects still shared with the module-level
`ASSISTANT_TOOL_SPECS` (`assistant.py:579`), so a single build would permanently add
`service` to the process-wide schema, and a later build that should omit it would inherit
it. Copy `tool`, `function`, `parameters`, and `properties`.

When no name is resolvable the `service` property is omitted entirely rather than emitted
with an empty enum, which some providers reject.

## Error handling

| Condition | Behaviour |
|---|---|
| Malformed entry in either key | Logged on the poll path, dropped, rest kept |
| Duplicate name | First wins; later dropped and logged |
| Two names, one canonical source | Later dropped and logged naming both |
| URL in both keys | Resolves to the polled source |
| Named page unreachable | That entry carries `error`; no fallback to other pages |
| `service` unresolvable | Polled set plus a top-level `error` naming it |
| Query cache at capacity | Oldest `fetched_at` evicted |
| Budget spent | Cached reading marked `stale`, or an `error` entry if never read |

## Testing

Beyond v1's ten, the reviews named the seams most likely to ship broken:

1. Grammar: `Name=url`, bare url, bad name charset, missing `=`, unparseable URL,
   duplicate name, two names one canonical source, over-cap — each dropped without
   affecting the rest, each logged on the poll path and silent on the request path.
2. **Queryable-only config exposes the tool** — empty `statusPageUrls`, non-empty
   `statusQueryablePages`, assert `status_fn` is wired and the schema present. [RT-C-3]
3. **Deep schema immutability** — build twice and assert `ASSISTANT_TOOL_SPECS`'s
   `parameters.properties` is byte-identical afterward, and that a build without
   queryable pages does not inherit a `service` property from an earlier one. [RT-C-1]
4. **The frozen mapping is what dispatch uses** — resolution changing between build and
   dispatch must not reroute the call. [RT-C-4]
5. **Enum contains no third-party data** — a polled page whose `page_name` collides with
   an operator's queryable name must not capture that name. [RT-S-1, RT-C-2]
6. **Allowlisted history survives a poll** — fetch history for a queryable page, run
   `_run_status_poll`, assert the three history entries and the backoff are still there.
   [RT-C-7]
7. **`_status_query_failed_at` is pruned** — the existing prune test enumerates exactly
   six structures (`test_status_poller.py:266`) and so cannot detect growth in the new
   dicts; extend it to eight. [RT-C-8]
8. **Twenty-service cycle does not thrash** — query all twenty inside the TTL twice and
   assert the second pass performs no fetches. [RT-C-9]
9. Conditional GET: a refresh past TTL sends the query cache's ETag. [RT-C-10]
10. `service` omitted still returns every polled source — pinned so it cannot regress.
11. `service` naming a polled page returns exactly that one; naming a queryable page
    fetches lazily and does not refetch inside the TTL.
12. Unresolvable `service` returns polled data **and** a top-level `error`. [RT-C-6]
13. `_status_state` is untouched by every queryable path.

**[RT-C-12] Existing tests that must change**, so the plan must budget for them:
`test_status_tool.py:34` asserts `include_history` is the schema's only property;
`:96` asserts a `service` argument is ignored; `:667` imports `_with_status_hosts` by
name; `conftest.py:601`/`:637` teach `make_service` only `statusPageUrls` and bind only
`_status_sources`; `status_plugin` (`conftest.py:787`) lacks the new config, caches and
constants.

Three prior reviews on this feature each found assertions that could not fail. Every new
assertion must be shown red against the unfixed code before being accepted.

## Cutover

`statusQueryablePages` defaults to empty, so nothing changes observably until the
operator opts in.

`statusPageUrls`' default gains names. **That default will not reach prod on its own** —
`bot.conf` holds the persisted three-URL line, which overrides it (see
`docs/guide/operator/configuration.md`). Bare URLs remain valid and fall back to their
host, so the deploy is safe with no config change: the enum simply uses hostnames until
the operator rewrites the line. To adopt names, `@config` it or stop-edit-start.

## Deliberately out of scope

- **Announcing allowlisted pages.** They have no lifecycle by design. Wanting
  announcements means moving the page to `statusPageUrls`.
- **Aliases.** The model maps "Codex" to OpenAI itself; revisit only if a wrong answer is
  observed in-channel.
- **Fully open "any Statuspage URL".** Rejected three times now: a hallucinated hostname
  becomes a fetch whose third-party prose the bot speaks unprompted.
- **Raising the 5-source polling cap.** The allowlist exists so the polled set can stay
  small.
