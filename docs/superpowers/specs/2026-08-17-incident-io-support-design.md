# incident.io status pages

**Date:** 2026-08-17
**Status:** approved
**Builds on:** `2026-08-17-multi-source-status-design.md` (shipped, commits `c63ce9e..b9610cc`)

## Problem

`status.openai.com` is an incident.io page. The multi-source work that just shipped can
poll any number of sources, but our parser rejects this one, so the bot cannot answer
"is OpenAI down?" or "is Codex up?".

The blocker is smaller than "add a second provider". incident.io serves an
**Atlassian-compatible shim** at the same paths, verified live on 2026-08-17:

| Endpoint | Result |
|---|---|
| `/api/v2/summary.json` | 200 `application/json`, keys `page` / `status` / `components` |
| `/api/v2/incidents.json` | 200, 35 KB, 25 incidents |

`page.name` is `OpenAI`, `page.url` is `https://status.openai.com/` (trailing slash —
`canonical_source` already collapses it). The vocabularies match ours: indicator `none`,
component status `operational`, incident status `resolved`, impacts `minor`/`none`.

`parse_incidents` — the history path — already reads it **unchanged**: all 25 incidents
parsed, with real names, impacts and durations.

So this is not a second provider. It is three guards in one function.

## What actually breaks

`parse_summary` rejects the payload at `statuspage.py:215`:

```
InvalidPayload: incidents is not a list
```

incident.io **omits** `incidents` and `scheduled_maintenances` entirely when they are
empty, rather than sending `[]`. Three lines require them:

```python
raw_components = _require_list(root.get("components"), "components")   # :214
_require_list(root.get("incidents"), "incidents")                      # :215
_require_list(root.get("scheduled_maintenances"), ...)                 # :216
```

and `:233` indexes `root["incidents"]` directly, which would `KeyError` even if the
guard passed.

Verified fix, run against the live payload: with `incidents` and `scheduled_maintenances`
defaulted to `[]`, `parse_summary` returns `OpenAI | none | 24 components`.

`_parse_incident` needs **no change** — its `components` handling
(`statuspage.py:156-161`) is already an `isinstance` check defaulting to `()`, so
incident.io's incidents, which carry no `components` key, degrade to an empty
`affected_components` rather than failing.

## Decisions

1. **Absent means empty, for every page.** A missing `incidents`,
   `scheduled_maintenances` or `components` key becomes `[]`. A key that is *present but
   not a list* is still rejected. One rule, no provider sniffing, no branching — the
   strictness that defends against a hostile payload is about structure, and absence of
   an optional collection is not a structural violation.

2. **An unknown component status keeps the component.** This is the substantive change.
   Today `statuspage.py:224-229` rejects the **entire page** if any component carries a
   status outside `COMPONENT_STATUSES`. That failure is worst-case-timed: it would fire
   during an outage, the only time anyone asks.

   The status is now passed through, sanitised, and the component kept. The rejected
   alternative — skipping the unknown component — fails silently in the worst direction:
   the bot would answer "all systems operational" precisely because the one broken
   component was the one discarded.

   `to_tool_payload` already includes any component whose status is not `"operational"`
   in `degraded`, so an unfamiliar value surfaces to the model automatically. The model
   reads prose; it does not need our enum.

   Structural strictness is unchanged: a component that is not an object, or whose
   `name`/`status` is not a string, is still rejected.

2b. **An unrecognised incident status keeps the incident, and is treated as live.**
   Superseded by whole-branch review, 2026-08-17. `_parse_incident` raised out of
   `parse_summary` on any status outside `INCIDENT_STATUSES`, which rejects the
   **entire page**, not just that incident — the asymmetry described below as
   deliberate was actually a bug in how strict a single unfamiliar incident status
   was allowed to be. Failure scenario: incident.io's native lifecycle includes
   `triage`/`fixing` (no live incident.io incident was ever observed at spec time),
   so a real OpenAI outage opening in one of those states would fail every poll for
   the outage's entire duration — no announcement in any `statusAnnounce` channel,
   and the tool answering `stale: True, error: "currently unreachable"` beside
   `description: "All Systems Operational"`. When the incident later resolved and
   dropped out of the payload, nothing would announce that either, because it never
   entered `active`.

   The operator ruled: tolerate it. `_parse_incident` no longer raises on an
   unrecognised `status`; the incident is kept. The status is passed through
   `sanitise_text`, exactly as component status already is — required, not
   optional, since `incident.status` reaches IRC (`render_line`'s
   `f"({incident.status})"`) and the rewrite prompt (`facts["status"]`)
   unsanitised today, safe only because the enum used to bound the value.
   Classification treats an unrecognised status as **not terminal** —
   `TERMINAL_STATUSES` is an explicit allowlist (`{"resolved", "postmortem"}`), so
   anything unfamiliar falls on the live side without a separate branch: it
   announces as opened and resolves normally when it vanishes from the payload.
   Structural strictness is unchanged: an incident with no usable `id`, or a
   non-string `status`, still rejects.

3. **No aliases.** "Are Claude and Codex up?" relies on the model mapping Codex to the
   OpenAI entry. It already receives every configured service in one call, and the tool
   description tells it to answer from the right entry. Zero config, zero tokens.
   Revisit only if a wrong answer is actually observed in-channel.

4. **OpenAI joins the default.** `statusPageUrls` becomes
   `https://status.claude.com https://www.githubstatus.com https://status.openai.com`.
   Three of a five-source cap.

## Residual risk, stated plainly

I verified the healthy path against live data and **inferred the rest**. OpenAI has no
open incident and all 25 historical ones are resolved, so the only values observable
today are `operational` and `resolved`. I have not seen incident.io emit a live incident
status (`investigating` / `identified` / `monitoring`) or a non-operational component.

Decision 2 is what makes that acceptable: if incident.io's live vocabulary diverges,
the page degrades to an unfamiliar status string reaching the model rather than the page
dropping out. Decision 2b (added 2026-08-17, superseding the paragraph this replaced)
extends the same tolerance to incident status: an unknown *incident* status used to
reject the whole page rather than "that incident" as originally believed here — see 2b
for the corrected blast radius and the fix. `parse_incidents` (the `incidents.json`
history parser) carried the identical bug — rejecting the whole batch over one
unfamiliar status in a best-effort path — and got the same fix the same day.
`INCIDENT_STATUSES` no longer gates parsing anywhere; it documents the vocabulary and
backs `TERMINAL_STATUSES`, which is now the single membership test deciding whether an
incident is treated as over.

Second observation from the live payload: 25 components collapse to 24, because
`components` is a dict keyed by name and OpenAI ships two components with the same name.
Pre-existing behaviour for any page, not incident.io-specific, and `to_tool_payload`'s
`degraded` list already exists to preserve non-operational components that collide.

Third, a pre-existing hole that this branch does not close (flagged in whole-branch
review, 2026-08-17, rated Minor deliberately): absent `incidents` is indistinguishable
from a truncated payload. An already-announced active incident plus a payload with
`indicator: "critical"` and `incidents` removed parses, and `classify` puts the incident
in `resolved` — a false all-clear. This is not new: `{"incidents": []}` with
`indicator: "critical"` was already accepted before this branch and yields an identical
`Snapshot`, so a hostile page gains nothing new here; only a buggy *truncated* payload
now takes the same path a page that legitimately omits an empty `incidents` key always
could. Deliberately not cross-checked against `indicator` — closing only the
newly-reachable encoding of a hole that already existed via `[]` would fix nothing and
add branching for no security gain. Revisit only alongside a broader "trust the
indicator over the incident list" design, not as a standalone patch.

## Scope

**In:**
- `parse_summary`: absent collections become empty; unknown component status passes through.
- `config.py`: OpenAI added to the `statusPageUrls` default.
- Tests: the three pure `statuspage` test files that pin the current strictness, plus new
  cases for each relaxation. A fixture built from the real OpenAI payload shape.
- `docs/guide/user/service-status.md`: note that incident.io pages work and that OpenAI
  is monitored by default.

**Out:**
- Provider detection or a second parse path.
- incident.io's *native* API. The compatibility shim is what we consume; if it is ever
  withdrawn, that is a new spec, not a fallback to build now.
- Aliases, per decision 3.
- The phase-2 queryable allowlist from the prior spec, still deferred.

## Testing

1. Absent `incidents` → parses, zero incidents. Present-but-a-string → still `InvalidPayload`.
2. Same pair for `scheduled_maintenances` and `components`.
3. Unknown component status → component present in `to_tool_payload`'s `degraded`, page
   parses, and the status is **`sanitise_text`'d like every other quoted field**. "Passed
   through" in decision 2 means "not mapped onto our enum" — never "not sanitised".
   `COMPONENT_STATUSES` was the only thing bounding this field, so dropping the check
   without substituting the sanitiser lets unbounded third-party text reach the model:
   measured at 529 bytes against a 200-byte cap on every sibling field.
   `sanitise_text("degraded_performance")` is a no-op, so this costs nothing on real
   input. Assert the cap holds for a hostile value.
4. Structurally bad component (not an object; non-string name) → still rejected.
5. Unknown *incident* status → the incident parses and is kept (decision 2b), sanitised
   and capped at `MAX_FREE_TEXT` like every other quoted field, a control-token status is
   stripped, and `classify` treats it as opened rather than resolved — proving it lands on
   the live side of `TERMINAL_STATUSES`. A non-string status, or an incident with no
   usable `id`, still rejects.
6. A real OpenAI-shaped summary fixture parses end to end.
7. `test_config.py` asserts the three-source default.

Every existing strictness test in `test_statuspage_parse.py` that asserts rejection on an
absent key must be re-read rather than deleted: if it pins "absent is rejected", it is now
wrong and should be inverted; if it pins "malformed is rejected", it must stay.

## Cutover

No prod config change. `statusPageUrls` is at its registered default, so the new default
picks up OpenAI on deploy. Cold-start seeding means any OpenAI incident already open at
deploy is recorded as announced and stays silent until it resolves — expected, same as
GitHub on the last deploy.
