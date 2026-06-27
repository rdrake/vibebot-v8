# Handoff — Verse Landing-Rate Instrument (Slice 1)

**Status: SHIPPED to `main` 2026-06-27** (FF-merge `cef4b2a..b9ec07b`, pushed). Fully offline —
a new read-only CLI module plus tests; **no bot-runtime, schema, or config change**. The
auto-deploy will rebuild the image with the file present but unused (the bot never imports it).

This is the first of a four-slice "improve the verse feature" roadmap:
**1 instrument (done) · 2 engagement · 3 authoring · 4 merge/scrub.**

## What shipped

- `plugins/llm/src/llm/verse/taste_report.py` — offline landing-rate reporter. Reuses the
  `taste_mine` detector for the numerator (fc42's entity-gated re-pastes + praise) and counts
  fc42's messages for the denominator. Dates each reaction by the per-day log **filename**,
  buckets by month, splits pre/post the exemplar rollout (`2026-06-22`).
- `plugins/llm/tests/verse/test_taste_report.py` — 17 tests, **100% line+branch** on the module.
- Spec: `docs/superpowers/specs/2026-06-27-verse-landing-instrument-design.md`
- Plan: `docs/superpowers/plans/2026-06-27-verse-landing-instrument.md`

Verification at merge: `make preflight` green, full suite **2355 passed / 93.79%**. Two red-teams:
a pre-implementation plan review (caught + killed the bogus usage-label change — see "Deferred")
and a post-implementation mutation-testing review (code correct on every axis; its 3 test-strength
gaps were fixed in commit `b9ec07b`).

## What the report tells you

A markdown report with: a **headline pre/post-rollout table** (reactions, fc42 msgs, active days,
reactions-per-100-msgs, reactions-per-active-day), a **monthly trend**, the **distinct wins**
(actual lines that landed, latest-first), and a **caveats block**. Honest by construction: the
denominator is fc42's *activity* (verse turns aren't isolable in telemetry), reactions are a
positive-only signal (silence ≠ dislike), thin buckets are flagged `thin sample`.

**Expectation:** the pre-rollout baseline is solid (~113 reactions back to 2026-04-15) but the
post window is only days of thin play — this establishes the ruler and accrues forward signal; it
won't hand you a confident "exemplars worked" verdict yet. Re-run it anytime.

## How to run it in prod (operator, in the container)

Robust stdin-script form (Python does the globbing — avoids shell glob/`#`-comment pitfalls; mirrors
the proven `taste_mine` prod recipe). Note: `docker exec` may need explicit authorization (the
command classifier blocks it by default; plain `docker logs` is pre-authorized but `exec` is not).

```bash
ssh -i ~/.ssh/id_rsa vibebot@rdrake.org   # security unlock-keychain first if key auth fails
docker exec -i vibebot /app/.venv/bin/python - <<'PY'
import glob, re
from pathlib import Path
from llm.verse.taste_report import build_report, render_report
from llm.verse.store import VerseStore

logs = sorted(glob.glob('/config/logs/ChannelLogger/afternet/#afternet/#afternet.*.log'))
store = VerseStore(Path('/config/data/verse'), '#afternet')
dated = []
for p in logs:
    m = re.search(r'(\d{4}-\d{2}-\d{2})', Path(p).name)
    if m:
        dated.append((m.group(1), Path(p).read_text(encoding='utf-8', errors='replace').splitlines()))
rep = build_report(dated, store, rollout='2026-06-22')
Path('/tmp/verse_landing_report.md').write_text(render_report(rep))
print('pre', rep.pre, '| post', rep.post, '| wins', len(rep.wins))
PY
docker exec vibebot cat /tmp/verse_landing_report.md
```

Equivalent CLI form (the module is also a console entrypoint; the `__main__.__file__` shim from
`cd973a8` lets the bare `python -m` import succeed). Shell must expand the log glob, so wrap in a
shell and quote `#afternet`:

```bash
docker exec vibebot sh -lc "/app/.venv/bin/python -m llm.verse.taste_report \
  /config/logs/ChannelLogger/afternet/'#afternet'/'#afternet'.*.log \
  --verse-dir /config/data/verse --channel '#afternet' --rollout 2026-06-22 \
  --out /tmp/verse_landing_report.md"
```

Read-only: it never writes the verse store or config. The atexit "Shutdown initiated / Killing
Driver objects" lines are harmless Limnoria teardown noise.

## Deferred — do NOT re-attempt the wrong version

- **Per-turn usage denominator (the "1-liner" trap).** An early draft wanted to relabel
  `task_type_map[PROFILE_VERSE]` from `"ask"` to `"verse"` at `service.py:4324` to make verse turns
  countable in the usage table. **Killed by red-team:** that map is inside the `litellm.Timeout`
  handler and only labels *stashed, timed-out* tasks for the retry queue (`pending_tasks.task_type`)
  — it never touches the usage row. The usage `command` is a hardcoded `"ask"` at `plugin.py:3846`.
  Relabeling the map would not make verse countable **and** would route verse timeout-recovery into
  the "unknown task type" branch (`service.py:2381`/`2392`). A real per-turn denominator is a
  larger route-aware change at the `log_usage` call site + a new recognized task_type — only worth
  it if forward log-volume normalization proves insufficient.
- **Storybook daily image cap micro-slice.** `verseStorybookDailyImageCap` (default 30) is defined
  in `config.py` but **not enforced** (`plugin.py:2741` TODO) — there's no per-account daily
  image-count, only cooldown + per-turn cap. Enforcing it needs a small persistent daily tally
  (~30–50 lines + a counter), not a query. Standalone, low-risk; good next quick win.

## Next slices (own spec → plan → red-team → ship each)

- **Slice 2 — engagement.** The root cause of thin signal: fc42 barely plays. Lower friction to
  enter/continue a scene; optional consent-respecting scene-seeds. Build as prompt/scheduler
  behavior, **not** new model tools (Grok tool-confusion). Highest leverage, riskiest — use this
  instrument as the ruler, and red-team it.
- **Slice 3 — authoring.** Activate the dormant layer (0 aliases, 0 `author_locked`): an alias verb
  so nicknames scene-match + auto-promote-canon-by-talking (deferred Task 8 from retention-v1).
- **Slice 4 — merge/scrub.** Entity dedup (`merge_entity` — the 9-point hardening list is in
  `project_verse_v2_redteam_2026_06_21`) + retroactive mis-tagged-event scrub. Trigger-only.
