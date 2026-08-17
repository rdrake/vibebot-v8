# Handoff: queryable status allowlist (in progress)

**Written:** 2026-08-17
**Status:** Tasks 1-3 of 5 complete and reviewed. Tasks 4-5 remain.
**Delete this file when the work lands.**

## What to read first

1. `docs/superpowers/specs/2026-08-17-queryable-status-allowlist-design.md` — approved and
   red-teamed. **Read its "What v1 got wrong" section before touching anything**: the
   obvious design (deriving selector names from each page's own `page_name`) was rejected
   by two independent reviews, and re-deriving it would reintroduce both defects.
2. `docs/superpowers/plans/2026-08-17-queryable-status-allowlist.md` — the five-task plan.
   Tasks 4 and 5 are unstarted and their code blocks are current.

## Where the work stands

Everything is committed on `main` **and unpushed** — 7 commits ahead of `origin/main` at
the time of writing. Suite: **3162** in `plugins/llm/tests/`, **3185** repo-wide, green.

| Commit | Task |
|---|---|
| `765b25c` | 1 — shared `Name=url` grammar, `_status_named_pages`, two-keep-set pruning |
| `926206c` + `cdba65e` | 2 — lazy query cache (TTL, backoff, bounded eviction, conditional GETs) |
| `07c1a64` | 3 — `service` selection in `_status_tool_payload` |

Prior context, already shipped and deployed, in `e11db64` and earlier: multi-source status
monitoring and incident.io support. Prod runs `e5c26cd`.

## What is left

**Task 4 — enum, gating, frozen mapping.** The one most likely to need a second pass. It
must:
- Replace `_with_status_hosts` with `_with_status_context`, copying **four** dict levels
  (`tool`, `function`, `parameters`, `properties`). Two is what ships today and is
  sufficient only while just `description` changes; writing a property into shared
  `parameters`/`properties` would add `service` to the process-wide schema permanently.
- Gate the tool on **polled OR queryable**. The shipped gate keys on the polled list
  alone, so a queryable-only config would expose no tool at all — defeating the feature.
- Bind the resolved mapping to the callback with `functools.partial(..., pages=...)`, so
  the schema and the dispatcher are one snapshot. `profile_tools` is built once and reused
  across every turn of the tool loop, and the executor does no schema validation.
- Add `service` to the `check_service_status` schema and forward it from the handler.

**Task 5 — docs.** Plus the carried item below.

Then: whole-branch review, then push.

## Carried findings — do not lose these

From Task 1's review, deliberately deferred:

- **CARRY TO TASK 4:** the cross-key name-collision branch (`or name.lower() in lowered`,
  `plugin.py:1161`) has **no test**. Deleting it keeps the suite green, yet it is what
  stops a queryable entry shadowing a polled selector name — the uniqueness Task 4's enum
  depends on.
- **CARRY TO TASK 5:** `docs/guide/operator/tuning-monitoring.md:143` still shows the
  three bare-URL default and the pre-names grammar. It is not in Task 5's file list.
- Minor, deferred: a bare URL whose host exceeds 32 chars or is non-ASCII is now dropped
  by the name regex; two bare URLs differing only by port collapse to one name; an IPv6
  host yields a name containing `:`.
- Minor, deferred: `statusPageUrls` is parsed twice per poll, so a bad entry warns twice.
- Minor, deferred: the cross-key collision log names only the loser; the spec asks for
  both.
- Minor, deferred: no test asserts `warn=False` silences the new parser's diagnostics.

## How to resume

The orchestration used `superpowers:subagent-driven-development`. Its ledger lives at
`.superpowers/sdd/2026-08-17-queryable-status-allowlist/progress.md` — **gitignored**, so
it survives a fresh session but not a `git clean -fdx`. This file is the durable copy.

To continue: run the skill's `scripts/task-brief` for task 4, dispatch an implementer with
the brief plus the constraints above, review, then task 5, then a whole-branch review.

## Rules that bite on this repo

- **Never let a subagent push.** Auto-deploy means any push is a production deploy: CI
  green → Docker build → the 15-minute updater timer restarts the bot. A subagent pushing
  unprompted earlier today put a broken commit on prod for ~14 minutes.
- **Don't `git commit` while a subagent is live** in the worktree — pre-commit stashes and
  restores unstaged changes and can race its edits.
- **A new registry default never reaches prod.** `bot.conf` holds the persisted old value
  and overrides it. `statusPageUrls`' default gains names in this work; prod keeps its
  bare-URL line until someone rewrites it. Bare URLs stay valid, so the deploy is safe.
- Logging uses `%i`, never `%d` — `supybot.utils.str.format` has no `%d` and silently
  shifts args left. There is a test that fails the suite on any new one.
- Seven reviewers on this feature have caught assertions that could not fail. Every new
  assertion gets proven red against the unfixed code before it is accepted.
