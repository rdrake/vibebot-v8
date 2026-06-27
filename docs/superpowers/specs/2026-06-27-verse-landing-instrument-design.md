# Verse Landing-Rate Instrument — Design (Slice 1)

- **Date:** 2026-06-27
- **Status:** Draft (proceeding to implementation under operator authority)
- **Relates to:** builds on `2026-06-21-fc42-taste-exemplars-design.md`. First of a four-slice
  "improve the verse feature" roadmap (1: instrument · 2: engagement · 3: authoring · 4: merge/scrub).

## 1. Problem & goal

The verse feature has shipped many improvements — retrieval into the prompt, 1-hop relations,
three live taste exemplars (since 2026-06-22), storybook, universe editing — but carries **zero
efficacy measurement**. We ship blind: nothing tells us whether the exemplars, or any other
change, moves fc42's experience.

**Goal:** a small, offline, log-only instrument that reports a *landing rate* — how often fc42
reacts positively to verse — over time, with a pre/post-exemplar split, reusing the detector
already shipped in `taste_mine`.

**Non-goals (this slice):** in-bot command, scheduled job, storybook image cap, any
authoring/engagement change, and any bot-runtime behavior change. This slice is **fully
offline** — a read-only CLI over logs and the verse store.

## 2. The signal

### 2.1 Numerator — reactions

Reuse `taste_mine.extract_candidates`: fc42's verbatim **re-pastes** (≥120 chars, naming a known
verse entity) plus explicit **praise** (praise wordlist, attributed to the nearest
non-fc42 / non-URL / non-addressed line). Both paths are entity-gated through
`store.match_entities_in_text`, so they are inherently verse-scoped — this resolves "is this a
verse reaction or an @ask reaction?" for free. It is the same detector that produced the three
live exemplars (113 clean candidates across 69 logs), so the numerator is already validated.

### 2.2 Denominator — fc42 activity

Telemetry cannot help: verse replies are logged to the usage table with a hardcoded
`command="ask"` (the shared @ask/verse reply path, `plugin.py:3846` → `db.log_usage`),
indistinguishable from normal @ask. (An earlier draft mis-cited `service.py:4324`; that
`task_type_map` governs only timed-out-task stashing, not the usage label — see §7.) The verse
`events` table is compaction-lossy (originals are pruned/digested after `verseEventRetentionDays`),
so it has no durable history. The only historically-complete, verse-consistent, log-derivable
denominator is **fc42's own message volume**.

Primary normalized metric: **reactions as a share of fc42's messages** (and per active day).
Interpretation: "of everything fc42 says, what share is him reacting to verse." Confounder, noted
in the output: if fc42's non-verse chatter (football/TV) drops, the share rises without verse
changing — acceptable over short pre/post windows, and flagged in the report.

### 2.3 Dating

`taste_mine` discards the in-line timestamp. ChannelLogger writes one file per day named
`#afternet.YYYY-MM-DD.log`. The report dates each file's reactions and messages by the date
parsed from its **filename**, and buckets by month. Core functions take explicit `(date, lines)`
pairs so they are filename-agnostic and unit-testable; filename→date parsing is thin CLI wiring.

## 3. Output

A markdown report:

- **Headline pre/post-rollout table** (default rollout `2026-06-22`): reactions, fc42 messages,
  active days, reactions-per-100-messages, reactions-per-active-day — for `[start, rollout)` vs
  `[rollout, end]`, with deltas.
- **Monthly trend table** — the same columns, one row per month.
- **Distinct wins** — globally-deduped reaction lines, latest-first, capped (~15), each with date,
  kind, and truncated text, so the operator can eyeball what is landing.
- **Caveats block** (always rendered): small-sample flags; positive-signal-only (silence ≠
  dislike); denominator-is-activity-not-turns; post-window thinness; the §2.2 confounder.

## 4. Architecture / components

New module `plugins/llm/src/llm/verse/taste_report.py` — pure functions plus a thin CLI.

- `DatedLog = tuple[str, list[str]]` — `(iso_date, lines)`.
- `BucketStats` NamedTuple — `(label, fc42_msgs, reactions, active_days)`; rates via small helpers
  (guarding divide-by-zero).
- `Win` NamedTuple — `(date, kind, text)`.
- `build_report(dated_logs, store, *, rollout="2026-06-22") -> Report` where
  `Report(buckets, pre, post, wins, span)`. For each `(date, lines)`:
  `reactions = extract_candidates(lines, store)`;
  `fc42_msgs = sum(_is_fc42(m.nick) for m in iter_messages(lines))`; fold into the month bucket and
  the pre/post split; collect wins (tagged with the file date).
- `render_report(report) -> str` — markdown.
- `_main(argv)` `# pragma: no cover` — argparse (`logs+`, `--verse-dir`, `--channel`,
  `--rollout`, `--out`); glob; parse the date from each filename
  (`\d{4}-\d{2}-\d{2}`; warn-and-skip undated files); read with `errors="replace"`; build; render;
  write + print a one-line summary.

Reuses `taste_mine.{iter_messages, _is_fc42, extract_candidates}` and `store.VerseStore`. No new
dependencies. **No** store writes, config keys, or bot surface.

## 5. Data flow

logs → per-file parse → per-date reactions + fc42 count → month + pre/post aggregation + global
win dedup → render markdown → stdout/file. Entirely offline; read-only on the logs and on the
verse store (only `match_entities_in_text`). Each file is parsed twice (once inside
`extract_candidates`, once via `iter_messages` for the count) — negligible cost for an offline
tool; avoids refactoring `taste_mine`.

## 6. Error handling / edge cases

- **Empty input** → empty report with caveats; no divide-by-zero (rates render `n/a` when the
  denominator is 0).
- **Undated filename** → CLI warns and skips; the core never sees it.
- **Unreadable bytes** → `errors="replace"`, consistent with `taste_mine`.
- **Per-file dedup** is intended: a re-paste on a different day is a new reaction event (correct
  for a rate-over-time). Praise attribution stays within the file (a praise line references
  something said recently — same day — in practice).
- **Distinct-wins dedup is global** (separate from per-bucket counting) so the eyeball list is not
  flooded by a repeated favorite.

## 7. Deferred — per-turn usage denominator (NOT a one-liner)

An earlier draft proposed making verse turns countable in the usage table by relabeling
`task_type_map[PROFILE_VERSE]` from `"ask"` to `"verse"` (`service.py:4324`). A red-team killed it:
that map lives inside the `litellm.Timeout` handler and only labels *stashed, timed-out* tasks for
the retry queue (`pending_tasks.task_type`) — it never touches the usage row. The usage `command`
is a hardcoded `"ask"` on the shared reply path (`plugin.py:3846`). Worse, relabeling the map to
`"verse"` would route verse timeout-recovery into the "unknown task type" branch
(`service.py:2381`/`2392`).

Making verse turns countable in usage is therefore a *larger* change — a route-aware relabel at the
`log_usage` call site plus a newly-recognized task_type — and is **deferred**. Slice 1 needs none
of it: the log-derived fc42-activity denominator (§2.2) is self-sufficient. Recorded here so the
wrong one-liner is not re-attempted.

## 8. Testing

`plugins/llm/tests/verse/test_taste_report.py`, BDD docstrings, the `FakeStore` pattern from
`test_taste_mine.py`. Cover: month bucketing; pre/post split at the rollout boundary (`date <`
vs `>=`); denominator counting; rate computation + divide-by-zero; win dedup + recency cap; render
contains the headline + caveats; empty input. **Branch coverage is on** (`pyproject.toml`
`branch = true`), so cover both sides of the win-replacement branch (`prev is None or
date > prev.date`) and the thin-sample annotation. Maintain the **93%** coverage floor; `_main` is
`# pragma: no cover` (matching `taste_mine`).

## 9. Honest expectations

The baseline pre-rate is solid (~113 reactions back to 2026-04-15). The post window is ~5 days of
thin play, so the immediate pre/post verdict is weak; the instrument's value is establishing the
ruler now and accruing forward signal. Re-runnable anytime via the container recipe.

## 10. Rollout

Offline, operator-run in the prod container (the same recipe as `taste_mine`; the
`__main__.__file__` shim is already present). **No deploy, restart, schema, or config change** —
the slice adds only a new offline module and its tests.
