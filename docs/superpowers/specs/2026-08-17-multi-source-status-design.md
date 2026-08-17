# Multi-source service status

**Date:** 2026-08-17
**Status:** approved, red-teamed, revised
**Supersedes parts of:** `2026-08-09-service-status-awareness-design.md`

Revision history: v1 drafted 2026-08-17; revised the same day after two
adversarial passes (self red-team and an independent Codex pass). The material
findings are marked **[RT-S-n]** (self) and **[RT-C-n]** (Codex) inline.

## Problem

The status feature monitors exactly one Atlassian Statuspage instance. Claude is
configured; GitHub is equally relevant to the channel and cannot be added. The
constraint is arbitrary — every piece of the fetch/parse/classify/announce stack
is already per-page — but six pieces of plugin state are singletons, and the
config key, the tool payload and the announcer all assume one page.

Goal: support 0..N sources, where 0 disables the feature exactly as the empty
string does today.

## Data source

GitHub's status page is the same Atlassian Statuspage product, verified live on
2026-08-17:

| Page | `page.name` | Components | Endpoint |
|---|---|---|---|
| `https://status.claude.com` | Claude | 6 | `/api/v2/summary.json` |
| `https://www.githubstatus.com` | GitHub | 12 | `/api/v2/summary.json` |

Identical schema, identical incident-permalink shape
(`{base}/incidents/{id}`). `statuspage.py` needs no changes to parse GitHub.

## Decisions

Three shape questions were settled before design:

1. **Sources are one shared global list.** No per-channel source sets.
2. **Announcing stays an all-or-nothing per-channel opt-in.** A channel with
   `statusAnnounce: True` hears every configured source.
3. **Querying is unconditional.** `check_service_status` is wired on
   `statusPageUrls` being non-empty and is visible in the chat and
   remind_action profiles with no channel gate, so any channel can ask
   "is GitHub down?" without ever receiving an unprompted line. This is
   unchanged behaviour, restated because it is the property that makes the
   all-or-nothing announce opt-in acceptable.

No backwards compatibility. There is one deployment and one operator; the
required config changes are listed under **Cutover** and applied by hand. The
docs carry the migration story for hypothetical future operators, the code does
not.

## Architecture

`statuspage.py` is untouched. Every contract in it stays single-snapshot —
`parse_summary`, `classify`, `to_tool_payload`, `to_history_payload`,
`render_line` — which is what keeps its five pure test files valid. Aggregation
across sources lives entirely in `plugin.py`.

### Configuration

`statusPageUrl` (global String) is replaced by `statusPageUrls` (global
`SpaceSeparatedListOfStrings`):

```python
conf.registerGlobalValue(
    LLM,
    "statusPageUrls",
    registry.SpaceSeparatedListOfStrings(
        ["https://status.claude.com", "https://www.githubstatus.com"],
        _("""..."""),
    ),
)
```

**[RT-C-1] The default must be a Python list, not a space-separated string.**
`registry.Value.__init__` hands the default straight to `setValue`, and
`SeparatedListOf.setValue` does `list(v)` — a string default silently becomes a
list of single characters, and every "source" then fails URL validation. The
space-separated form is only the serialized representation in `bot.conf`.

`statusAnnounce` (channel Bool) is unchanged.

### Canonical source identity **[RT-S-2, RT-C-8]**

The configured string is not safe as a key. `_fetch_json` accepts a trailing
slash (`statuspage.py:826` rstrips the path, `:835` rstrips the base), so
`https://status.claude.com` and `https://status.claude.com/` reach the same API
while hashing differently — two lifecycle states, every incident announced
twice. Scheme/host casing and explicit default ports alias the same way.

A `_status_sources()` helper returns an ordered, deduplicated list of canonical
ids, derived by lowercasing scheme and host, dropping a default port, and
stripping trailing slashes. The canonical id is the dict key everywhere, the
value passed to `fetch_summary`, and the basis of the entry identity below.
Non-canonical or unparseable entries are logged once and dropped rather than
silently keyed. The list is capped at `_STATUS_MAX_SOURCES = 5`; the overflow is
logged, never silently truncated.

### Per-source state

All six singletons become dicts keyed by canonical id:

| Field | Written by |
|---|---|
| `_status_state` | poller only |
| `_status_read_cache` | poller and the tool's inline fetch |
| `_status_last_fetch` | poller and the tool's inline fetch |
| `_status_history_cache` | tool path only |
| `_status_history_at` | tool path only |
| `_status_history_failed_at` | tool path only |

The 2026-08-09 ownership split survives intact and is the reason
`_status_state` stays poller-exclusive: a user asking "is it down?" must not
consume an announcement.

**[RT-C-10] Prune all six together.** Dropping only `_status_state` for a
removed source leaves the other five growing without any bound, since the
5-source cap bounds the *configured* set, not the historical one. Pruning
happens once at the top of each poll against the current canonical set.

### Poll: one worker, a real deadline

The poll stays a single executor job walking sources sequentially — no `submit`
(RecursiveSubmitError from worker context) and no nested `permit()`
(self-deadlock). Each source's fetch, classify and announce is wrapped
individually so one dead page cannot stop the others.

**[RT-C-2] The deadline must be propagated, not merely checked between
sources.** A budget consulted only at the top of each iteration bounds nothing:
a fetch entered at t=44s still runs its full 30s, and the announcement rewrites
run *inline in the same worker*, each costing up to the LLM timeout, once per
incident per overlay group. Three incidents across two overlays is six
completions inside one pass.

The pass therefore carries a deadline computed from `time.monotonic()`, not
`_status_now()` — `_status_now` is wall time (`plugin.py:1061`) and a clock
adjustment would corrupt any deadline built on it. `_status_now` keeps its
existing role as the pinnable wall clock for incident ages and budgets; a
parallel `_status_monotonic()` indirection is added for deadlines so tests can
pin it independently.

- `_STATUS_PASS_BUDGET = 45` seconds, whole pass.
- Each fetch's timeout is `min(registry timeout, 30, remaining)`; a source with
  under ~2s remaining is skipped rather than started.
- Once remaining time drops below `_STATUS_REWRITE_RESERVE = 20`, the pass goes
  template-only. Announcements still go out — the template is the primary path
  and the rewrite has always been an upgrade — so no incident is lost to the
  deadline.
- **[RT-C-11]** `self._llm_executor.closing` is checked before each source and
  again after each blocking fetch, so an unload during a pass stops spending on
  fetches and billed rewrites. IRC delivery is already suppressed by
  `_safe_queue`.

**[RT-C-10] The rotation cursor is a canonical id, not an index.** An index
points at a different source after any reorder or removal.

The cursor advances past every source the pass *attempted*, in a `finally`, so a
source that consistently fails or times out cannot pin itself at the head of the
rotation and starve the rest. It does **not** advance past a source skipped for
lack of remaining budget — those are exactly the sources the next pass must
start with, which is the whole point of the deferral. Attempted-and-failed
advances; never-started does not. If the cursor's id is no longer configured,
the next pass restarts at the head of the list.

### Poll cadence and freshness **[RT-C-5]**

The one-shot re-arms from its own done-callback (`plugin.py:1107`), so the
120s interval is measured from when a pass *ends*. A 45s pass yields a 165s
start-to-start cadence, and with rotation a source at the 5-source cap can go
several passes between polls.

Consequently the tool's staleness rule cannot stay `2 * _STATUS_POLL_INTERVAL`.
Staleness becomes a fixed `_STATUS_STALE_AFTER = 600` seconds measured against
that source's own `snapshot.fetched_at` — the timestamp of its last *successful*
read, not `_status_last_fetch`, which is stamped before an attempt and so
survives a failure. This is independent of source count and rotation depth, so a
healthy-but-rotated source is never reported to the model as unreachable.

`docs/guide/user/service-status.md:20` currently promises a reread every two
minutes and an inline refresh after four. That promise is rewritten to describe
the rotation and the 10-minute staleness line.

### Tool payload

`_status_tool_payload` returns:

```json
{"services": [
  {"source": "status.claude.com", "service": "Claude",
   "indicator": "none", "degraded": [], "incidents": [], "snapshot_age_sec": 31},
  {"source": "www.githubstatus.com", "service": "GitHub",
   "indicator": "major", "incidents": [{"...": "..."}]}
]}
```

Each entry is today's `to_tool_payload` output plus two identity fields.

**[RT-S-5, RT-C-7] Identity is operator-derived; `page_name` is a display label
only.** `service` comes from the page's own `page_name`, which is third-party
data — with one source there was nothing to confuse, with several a page that
renames itself "Claude" makes the model report GitHub's outage as Claude's.
`source` is the canonical host from config and is always present. It is also the
only identity available before a source's first successful fetch, since
`page_name` requires a parsed `Snapshot`: without it, two cold sources produce
two indistinguishable `{"error": ...}` entries.

**[RT-C-7] Aggregate error contract.** `service.py:5557` treats any top-level
dict without an `"error"` key as a successful tool call, so a `services` list
made entirely of failures would be recorded as success. When *every* configured
source fails, the payload also carries a top-level `"error"`. Per-source
`error`/`stale` stay inside their entry, so a partial failure still answers for
the sources that worked.

Zero configured sources keeps today's behaviour: `status_fn` is `None` and the
schema is excluded from the profile.

### Tool-path deadline **[RT-S-1, RT-C-3]**

The tool runs synchronously inside the assistant's tool loop, holding the
enclosing request's permit. Fanning out N stale sources means N sequential
fetches at 30s each, and `include_history` adds a second round against
`incidents.json` (223 KB per source). At the 5-source cap that is nominally
5 minutes for one question. `maxConcurrentLLMCalls` is 16 in the registry
default and on prod, so this does not stall the whole bot — but it does make the
asking user wait minutes and holds a permit throughout.

The tool path gets its own monotonic deadline, `_STATUS_TOOL_BUDGET = 20`
seconds, covering the current-status refreshes and the history fan-out together.
Sources not reached return their last cached snapshot marked `stale: true`, or a
per-entry `error` if never read. History for an unreached source returns `[]`.
The result is always a bounded, partial-but-labelled answer rather than an
unbounded wait.

`_STATUS_FETCH_FLOOR = 30` becomes per source. The check-then-set on
`_status_last_fetch` remains unlocked and therefore racy under concurrent
questions **[RT-C-9]**; the floor is a cost guard, not a correctness guard, and
a duplicate fetch is harmless. Left as-is deliberately.

### Announcer

`_announce_status(source, delta, snapshot)` gains both the canonical source and
the snapshot.

**[RT-C-9] Pass the snapshot the poll classified.** Today `_announce_status`
re-reads `_status_read_cache` (`plugin.py:1471`) after `_run_status_poll`
classified a local `snapshot`. With 16 concurrent permits, a tool-path inline
fetch can replace that cache entry in between, so the label and rewrite facts
would come from a different observation than the delta. Threading the snapshot
through removes the window. Pre-existing bug, cheap to fix while the signature
changes anyway.

`configured`, `configured_host` and `label` are derived per source, so
`_status_rewrite_ok`'s host check validates against the page that actually
raised the incident. `render_line` already prefixes `page_name`, so Claude and
GitHub lines self-distinguish with no format change.

**[RT-S-4, RT-C-12] Global per-pass line cap.** `_STATUS_MAX_ANNOUNCE_PER_POLL`
is per source and caps only openings; `classify`'s `max_resolved` also defaults
to 3 and `_run_status_poll` never passes it. Five sources can therefore emit 15
openings plus 15 all-clears in one pass, against today's ceiling of 3. A global
`_STATUS_MAX_LINES_PER_POLL = 5` bounds the burst across all sources. The
remainder is simply left unmarked, which is already safe: `_announce_status`
marks an incident only after a successful queue, so the next poll retries it.

The 6/hour rewrite budget stays a single global bucket. Over budget falls
through to the template, which still announces, so no source can silence
another — only downgrade its prose.

### Tool description **[RT-C-6]**

The description is rewritten, not suffixed. Appending "(currently:
status.claude.com, www.githubstatus.com)" to a description that still says
"the configured service status page (Claude)" and describes a flat payload
leaves the model expecting one page and receiving a `services` array. The new
text states that results are a per-service list, that errors and staleness are
per service, that history is per service, and that a question about one service
should be answered from that service's entry rather than summarized across all.
The configured hosts are injected at profile-build time so the model knows what
is actually covered.

Injection must copy both levels:

```python
{**tool, "function": {**tool["function"], "description": ...}}
```

`ToolSpec.as_tool()` returns a fresh outer dict but hands back the *shared
module-level* `schema` object (`assistant.py:557`). Mutating in place corrupts
the schema process-wide and re-appends on every call.

## Error handling

| Condition | Behaviour |
|---|---|
| One source unreachable | Its entry carries `error`/`stale`; others answer normally; poll continues |
| All sources unreachable | Per-entry errors plus a top-level `error` so the tool loop records a failure |
| Source never fetched | Entry identified by `source` host alone, `service` absent |
| Non-canonical URL in config | Logged once, dropped from the source list |
| More than 5 sources | Overflow logged, dropped |
| Pass deadline exceeded | Remaining sources deferred to the next pass via the cursor |
| Rewrite reserve exhausted | Template-only for the rest of the pass |
| Plugin unloaded mid-pass | `closing` check ends the pass before the next fetch or rewrite |

## Testing

New coverage, all at plugin level:

1. **Poll isolation** — source A fails, source B still classifies and announces.
2. **State isolation** — an incident on B never touches A's `announced` map.
3. **Canonicalization** — bare and trailing-slash forms of one URL collapse to a
   single source with a single announcement.
4. **Deadline** — a pass that exhausts its budget defers the remaining sources,
   advances the cursor past the slow source, and goes template-only under the
   reserve.
5. **Cursor** — removing or reordering sources does not re-poll or skip; a
   permanently failing source does not starve the rotation.
6. **Pruning** — dropping a source clears all six keyed structures.
7. **Aggregate payload** — partial failure labels the failed entry and keeps the
   healthy one; total failure sets the top-level `error`.
8. **Tool budget** — an unreachable source returns a stale/error entry within
   the budget instead of blocking.
9. **Description injection** — the module-level schema is not mutated across
   two builds.

Two traps carried forward from 2026-08-09, both of which cost real debugging
time before:

- **The gate lives on the caller side.** Every announcer test calls
  `_announce_status` directly with a hand-built `Delta`, which proves nothing
  about whether the path is entered. New branches need a `test_status_poller.py`
  test driving `_run_status_poll` end to end.
- **`conftest.py:893`'s `announcing_plugin` binds real methods onto a
  MagicMock.** Any new method on the announce path — canonicalization, source
  metadata, cursor helpers — must be bound there too, or it returns a truthy
  Mock and every incident is marked announced while nothing sends.

Existing files: the five pure `statuspage.py` test files stay valid by
construction. `test_status_poller.py`, `test_status_announce.py` and
`test_status_tool.py` need rework for keyed state and the new signatures.
`test_config.py` gains a registration/default assertion.

**[RT-C] The shared service fixture needs an explicit decision.**
`conftest.py:501` returns `""` for unknown registry keys, so leaving it alone
runs most service tests with the feature disabled, while giving it the
production two-source default changes the tool surface in unrelated completion
tests. It is set to a single-source default, which keeps the tool present
without doubling payloads in tests that do not care.

## Cutover

Prod (`~/.config/vibebot/bot.conf`) holds `statusPageUrl:
https://status.claude.com` — exactly the registered default, so no operator
value is lost by the rename.

1. Nothing is required *for this deploy specifically*, because `statusPageUrl`
   (singular) is being renamed: the new key has never been persisted, so it
   genuinely takes its default. `statusAnnounce.#clanker: True` and its
   network-scoped duplicate start delivering GitHub incidents after the
   auto-deploy.

   **This exemption does not generalise, and reading it as a rule cost a silent
   failure on 2026-08-17.** Once the bot shuts down with `statusPageUrls`
   registered, Limnoria flushes the registry to `bot.conf` and writes the
   then-current value out as an explicit line — which then overrides any future
   default shipped in `config.py`. Every *later* change to this key's default
   needs an operator edit. See the incident.io spec's cutover section.
2. Optional tidy: delete the now-inert `supybot.plugins.LLM.statusPageUrl` line,
   with the bot stopped — Limnoria rewrites the registry on shutdown and would
   clobber a live edit.
3. To make GitHub opt-in instead, set `supybot.plugins.LLM.statusPageUrls:
   https://status.claude.com` before the deploy.

Expected on first deploy: GitHub's currently-open major incident is seeded as
already-announced, so #clanker hears nothing about it until it resolves, at
which point the all-clear fires. That is the cold-start seeding split working as
designed, not a fault.

#clanker still has the stock RSS announcer armed for Claude as a deliberate
backup. GitHub has no RSS equivalent configured, so it arrives without a double
report.

## Deliberately out of scope

- **Per-channel source selection.** Rejected in favour of the all-or-nothing
  opt-in; querying is already unconditional, which covers the case a channel
  wants information without noise.
- **A `service` argument on the tool.** Not needed here — the tool-path deadline
  bounds the fan-out cost a selector would otherwise avoid. Deferred, not
  rejected: it returns in the phase-2 queryable allowlist below, where the model
  must name which page it wants.
- **A per-channel "status query enabled" bool.** The ~150 prompt tokens the
  schema costs per completion are already being paid today; nothing in current
  traffic justifies the key.
- **The partial-delivery gap.** One channel queuing successfully marks the
  incident announced for all channels, so a netsplit channel misses it. Real,
  pre-existing, documented in
  `docs/plans/2026-08-14-status-announce-restart-gap.md`, and non-blocking while
  one channel is opted in.
- **The restart gap.** An incident that both opens and closes inside a container
  restart window is still silent. Unchanged by this work.
- **Locking the inline-fetch floor.** Racy by design; a duplicate fetch is a
  cost, not a correctness problem.

## Planned follow-up: a queryable allowlist (phase 2)

Polled pages and queryable pages need not be the same set. Polling costs a fetch
every 120s plus lifecycle state per source, which is why this spec caps sources
at 5. Querying costs one lazy fetch, cached, only when someone asks — so a short
polled list can sit inside a much longer allowlist of pages the bot can answer
about but never announces.

Deliberately deferred until multi-source polling is running, because it builds
directly on what this spec establishes: canonical source ids, per-source caches
with their own TTL and failure backoff, and the tool-path deadline. Designing it
against working code beats designing it against a plan.

Sketch, not a commitment:

- `statusQueryablePages`, a global list of `name=url` pairs. Names are needed
  here in a way they are not for the polled list, because the model has to
  select one.
- The `service` argument on `check_service_status`, plus matching from what the
  user actually said ("cloudflare") to an entry.
- A lazy fetch path reusing `_status_fetch_snapshot`, with its own cache, TTL
  and a bounded eviction policy — the allowlist can be long, so cache growth
  needs a ceiling the polled path does not need.
- No polling, no lifecycle state, no announcements for these pages. They are
  read-on-demand only, which is what keeps the cost linear in questions asked
  rather than in pages listed.

A fully open "any Statuspage URL the model names" variant was considered and
rejected for a public channel. The SSRF stack does hold — it was built for
untrusted input — but a hallucinated hostname becomes a fetch whose third-party
prose the bot then speaks unprompted, and nothing bounds how many hosts
accumulate state.
