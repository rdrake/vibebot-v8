# Verse Landing-Rate Instrument Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline, log-only CLI that reports fc42's verse landing-rate over time (pre/post the 2026-06-22 exemplar rollout), reusing the `taste_mine` detector.

**Architecture:** A new pure-functions-plus-thin-CLI module `verse/taste_report.py`. The numerator is `taste_mine.extract_candidates` (entity-gated reactions); the denominator is fc42's message count from the same logs; reactions are dated by the per-day log filename and bucketed by month, with a pre/post-rollout split. No bot runtime, store, or config changes — fully offline.

**Tech Stack:** Python 3.12+, pytest, the existing `llm.verse` package. Reuses `taste_mine.{iter_messages, extract_candidates, _is_fc42, _dedup_key}` and `store.VerseStore`.

**Spec:** `docs/superpowers/specs/2026-06-27-verse-landing-instrument-design.md`

**Conventions (from AGENTS.md):** run `make lint` + `make typecheck` after editing Python; `make test` enforces the **93%** coverage floor; tests live under `plugins/llm/tests/verse/`; use BDD-style docstrings; `_main` CLI wiring is `# pragma: no cover` (matches `taste_mine`).

**Intentional design note (for the reviewer):** `taste_report` imports two underscore-prefixed helpers (`_is_fc42`, `_dedup_key`) from its sibling `taste_mine`. This is deliberate intra-package reuse to keep the fc42 predicate and the dedup key **identical** across the miner and the reporter (DRY). The alternative — promoting them to public names in the shipped `taste_mine` — is scope creep for this slice. If review prefers, promote in a follow-up.

---

### Task 1: Module skeleton — types, constants, and rate helpers

**Files:**
- Create: `plugins/llm/src/llm/verse/taste_report.py`
- Test: `plugins/llm/tests/verse/test_taste_report.py`

- [ ] **Step 1: Write the failing test**

```python
# plugins/llm/tests/verse/test_taste_report.py
import re as _re

from llm.verse.taste_report import (
    BucketStats,
    Report,
    Win,
    _month,
    build_report,
    per_100,
    per_day,
    render_report,
)


class _Ent:
    def __init__(self, name):
        self.name = name


class FakeStore:
    """Mirrors the real VerseStore.match_entities_in_text contract (see test_taste_mine)."""

    def __init__(self, names):
        self._names = names

    def match_entities_in_text(self, text, limit=12):
        low = text.lower()
        return [
            _Ent(n)
            for n in self._names
            if _re.search(r"(?<!\w)" + _re.escape(n.lower()) + r"(?!\w)", low)
        ][:limit]


def test_month_extracts_year_month():
    """_month() reduces an ISO date to its YYYY-MM bucket key."""
    assert _month("2026-06-22") == "2026-06"


def test_rate_helpers_guard_divide_by_zero():
    """per_100/per_day return None (rendered n/a) when the denominator is zero."""
    assert per_100(0, 0) is None
    assert per_day(5, 0) is None
    assert per_100(3, 200) == 1.5
    assert per_day(6, 3) == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.verse.taste_report'`

- [ ] **Step 3: Write minimal implementation**

```python
# plugins/llm/src/llm/verse/taste_report.py
"""Offline verse landing-rate report: how often fc42 reacts positively to verse, over time.

Read-only. Reuses the taste_mine detector for the numerator (entity-gated reactions) and
counts fc42's messages for the denominator. Dates reactions by the per-day ChannelLogger
filename. See docs/superpowers/specs/2026-06-27-verse-landing-instrument-design.md.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any, NamedTuple

from .taste_mine import _dedup_key, _is_fc42, extract_candidates, iter_messages

DEFAULT_ROLLOUT = "2026-06-22"  # date the curated style exemplars went live for #afternet
_MAX_WINS = 15
_SMALL_SAMPLE_MSGS = 50
_SMALL_SAMPLE_REACTIONS = 5
_WIN_TEXT_CAP = 160


class Win(NamedTuple):
    date: str
    kind: str  # "repaste" | "praise"
    text: str


class BucketStats(NamedTuple):
    label: str
    fc42_msgs: int
    reactions: int
    active_days: int


class Report(NamedTuple):
    buckets: list[BucketStats]  # monthly, sorted by label
    pre: BucketStats  # [start, rollout)
    post: BucketStats  # [rollout, end]
    wins: list[Win]  # globally-deduped, latest-first, capped
    rollout: str


def _month(date: str) -> str:
    return date[:7]


def per_100(reactions: int, msgs: int) -> float | None:
    return None if msgs == 0 else round(reactions * 100.0 / msgs, 2)


def per_day(reactions: int, days: int) -> float | None:
    return None if days == 0 else round(reactions / days, 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/taste_report.py plugins/llm/tests/verse/test_taste_report.py
git commit -m "feat(verse): taste_report skeleton — types + rate helpers"
```

---

### Task 2: `build_report` — buckets, pre/post split, denominator

**Files:**
- Modify: `plugins/llm/src/llm/verse/taste_report.py`
- Test: `plugins/llm/tests/verse/test_taste_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to test_taste_report.py

_REP = (
    "the stinky lads marched into the assembly hall and let off a perfectly timed "
    "duet that turned the leaves yellow indeed mate"
)


def test_build_report_empty_input():
    """No logs -> empty buckets/wins and zeroed pre/post."""
    r = build_report([], FakeStore([]))
    assert r.buckets == []
    assert r.wins == []
    assert r.pre == BucketStats("pre", 0, 0, 0)
    assert r.post == BucketStats("post", 0, 0, 0)
    assert r.rollout == "2026-06-22"


def test_build_report_denominator_counts_only_fc42_messages():
    """fc42 privmsg + action count; non-fc42 lines do not."""
    store = FakeStore(["stinky lads"])
    lines = [
        "2026-06-20T10:00:00  <fc42> good morning all",
        "2026-06-20T10:01:00  <rdrake> hello there",  # not fc42 -> excluded
        "2026-06-20T10:02:00  * fc42 waves at the room",  # fc42 action -> counted
    ]
    r = build_report([("2026-06-20", lines)], store)
    assert r.pre.fc42_msgs == 2
    assert r.pre.reactions == 0  # nothing long enough / praising


def test_build_report_splits_pre_post_at_rollout_boundary():
    """rollout date is inclusive of post: date < rollout -> pre; date >= rollout -> post."""
    store = FakeStore(["stinky lads"])
    r = build_report(
        [
            ("2026-06-21", [f"2026-06-21T10:00:00  <fc42> {_REP}"]),
            ("2026-06-22", [f"2026-06-22T10:00:00  <fc42> {_REP}"]),  # boundary -> post
        ],
        store,
        rollout="2026-06-22",
    )
    assert r.pre.reactions == 1 and r.pre.fc42_msgs == 1
    assert r.post.reactions == 1 and r.post.fc42_msgs == 1


def test_build_report_buckets_by_month_with_active_days():
    """Months aggregate; active_days counts distinct dates with >=1 fc42 message."""
    store = FakeStore(["stinky lads"])
    logs = [
        ("2026-05-30", [f"2026-05-30T10:00:00  <fc42> {_REP}"]),
        ("2026-06-01", [f"2026-06-01T10:00:00  <fc42> {_REP}"]),
        ("2026-06-02", ["2026-06-02T10:00:00  <fc42> just chatting about the football"]),
    ]
    r = build_report(logs, store, rollout="2026-06-22")
    assert [b.label for b in r.buckets] == ["2026-05", "2026-06"]
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.fc42_msgs == 2
    assert june.reactions == 1
    assert june.active_days == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: FAIL — `ImportError`/`AttributeError` is not raised (build_report exists from Task 1? No — it is not yet defined). Expected: FAIL with `ImportError: cannot import name 'build_report'`.

> Note: `build_report`/`render_report`/`Win` were imported in Task 1's test header but not yet defined, so Task 1 actually could not pass. To keep Task 1 green in isolation, `build_report` and `render_report` are stubbed in Task 1 Step 3 ONLY if you ran tasks out of order. If you followed order, define them now.

- [ ] **Step 3: Write the implementation**

```python
# append to taste_report.py (after per_day)

def build_report(
    dated_logs: Iterable[tuple[str, Iterable[str]]],
    store: Any,
    *,
    rollout: str = DEFAULT_ROLLOUT,
) -> Report:
    """Aggregate per-day logs into monthly + pre/post landing stats and a wins list.

    dated_logs: an iterable of (iso_date, lines). Each file is one day; `lines` are raw
    ChannelLogger lines. Numerator = extract_candidates (entity-gated reactions);
    denominator = count of fc42 messages.
    """
    month_msgs: dict[str, int] = defaultdict(int)
    month_reactions: dict[str, int] = defaultdict(int)
    month_days: dict[str, set[str]] = defaultdict(set)
    pre_msgs = pre_reactions = post_msgs = post_reactions = 0
    pre_days: set[str] = set()
    post_days: set[str] = set()
    wins_by_key: dict[str, Win] = {}

    for date, raw_lines in dated_logs:
        lines = list(raw_lines)
        fc42_msgs = sum(1 for m in iter_messages(lines) if _is_fc42(m.nick))
        cands = extract_candidates(lines, store)
        n = len(cands)

        mk = _month(date)
        month_msgs[mk] += fc42_msgs
        month_reactions[mk] += n
        if fc42_msgs:
            month_days[mk].add(date)

        if date < rollout:
            pre_msgs += fc42_msgs
            pre_reactions += n
            if fc42_msgs:
                pre_days.add(date)
        else:
            post_msgs += fc42_msgs
            post_reactions += n
            if fc42_msgs:
                post_days.add(date)

        for c in cands:
            key = _dedup_key(c.text)
            prev = wins_by_key.get(key)
            if prev is None or date > prev.date:
                wins_by_key[key] = Win(date, c.kind, c.text)

    buckets = [
        BucketStats(mk, month_msgs[mk], month_reactions[mk], len(month_days[mk]))
        for mk in sorted(month_msgs)
    ]
    pre = BucketStats("pre", pre_msgs, pre_reactions, len(pre_days))
    post = BucketStats("post", post_msgs, post_reactions, len(post_days))
    wins = sorted(wins_by_key.values(), key=lambda w: w.date, reverse=True)[:_MAX_WINS]
    return Report(buckets, pre, post, wins, rollout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: PASS (all build_report tests green)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/taste_report.py plugins/llm/tests/verse/test_taste_report.py
git commit -m "feat(verse): build_report — monthly buckets + pre/post split + fc42 denominator"
```

---

### Task 3: Distinct wins — global dedup, latest-first, capped

**Files:**
- Test: `plugins/llm/tests/verse/test_taste_report.py`

> `build_report` already populates `wins` (Task 2). This task adds the tests that lock the wins contract — global dedup (decoupled from per-bucket counting), recency ordering, and the cap.

- [ ] **Step 1: Write the failing tests**

```python
# append to test_taste_report.py

def test_wins_dedup_global_but_counts_keep_occurrences():
    """A repeated favorite collapses to one win (latest date) yet still counts per bucket."""
    store = FakeStore(["stinky lads"])
    logs = [
        ("2026-06-01", [f"2026-06-01T10:00:00  <fc42> {_REP}"]),
        ("2026-06-10", [f"2026-06-10T10:00:00  <fc42> {_REP}"]),
    ]
    r = build_report(logs, store)
    assert len(r.wins) == 1
    assert r.wins[0].date == "2026-06-10"  # latest occurrence kept
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.reactions == 2  # both occurrences still counted


def test_wins_latest_first_and_capped_at_15():
    """19 distinct wins -> 15 returned, newest first."""
    store = FakeStore(["lads"])
    logs = []
    for i in range(1, 20):
        d = f"2026-06-{i:02d}"
        txt = (
            f"the lads pulled off distinct caper number {i} in the great assembly hall "
            "with much fanfare and merriment that lasted well into the small hours indeed"
        )
        logs.append((d, [f"{d}T10:00:00  <fc42> {txt}"]))
    r = build_report(logs, store)
    assert len(r.wins) == 15
    assert r.wins[0].date == "2026-06-19"
    assert r.wins[-1].date == "2026-06-05"
```

- [ ] **Step 2: Run tests to verify they pass immediately**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q -k wins`
Expected: PASS — the behavior was implemented in Task 2; these tests pin it. (If either fails, fix `build_report`'s wins logic before continuing.)

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/tests/verse/test_taste_report.py
git commit -m "test(verse): pin taste_report wins dedup/order/cap contract"
```

---

### Task 4: `render_report` — markdown tables, wins, caveats

**Files:**
- Modify: `plugins/llm/src/llm/verse/taste_report.py`
- Test: `plugins/llm/tests/verse/test_taste_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to test_taste_report.py

def test_render_has_headline_monthly_wins_and_caveats():
    """The report renders all four sections, including the silence!=dislike caveat."""
    store = FakeStore(["stinky lads"])
    r = build_report([("2026-06-22", [f"2026-06-22T10:00:00  <fc42> {_REP}"])], store)
    md = render_report(r)
    assert "# Verse landing-rate report" in md
    assert "Headline" in md
    assert "Monthly trend" in md
    assert "Distinct wins" in md
    assert "silence" in md.lower()  # caveat: silence is not dislike


def test_render_shows_na_for_zero_denominator():
    """Empty input -> pre/post rates render as n/a, never a crash."""
    md = render_report(build_report([], FakeStore([])))
    assert "n/a" in md


def test_render_truncates_long_win_text():
    """Win text longer than the cap is truncated with an ellipsis."""
    store = FakeStore(["lads"])
    long = "the lads " + "marched onward through the misty glen and over the hills " * 5
    r = build_report([("2026-06-22", [f"2026-06-22T10:00:00  <fc42> {long}"])], store)
    md = render_report(r)
    assert "..." in md


def test_render_flags_thin_sample():
    """A bucket below the small-sample thresholds is annotated."""
    store = FakeStore(["stinky lads"])
    r = build_report([("2026-06-22", [f"2026-06-22T10:00:00  <fc42> {_REP}"])], store)
    md = render_report(r)
    assert "thin sample" in md
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q -k render`
Expected: FAIL — `AttributeError`/`TypeError` because `render_report` is not yet defined (or returns None).

- [ ] **Step 3: Write the implementation**

```python
# append to taste_report.py (after build_report)

_CAVEATS = [
    "",
    "## How to read this",
    "",
    "- **Positive signal only.** This counts reactions fc42 *volunteered* "
    "(re-pastes + praise). Silence is not dislike; absence of a reaction is not a "
    "negative score.",
    "- **Denominator is activity, not verse turns.** Verse completions are logged as "
    "`command=\"ask\"` and the events table is compaction-lossy, so there is no clean "
    "verse-turn count. We normalise by fc42's own message volume instead.",
    "- **Confounder.** If fc42's non-verse chatter (football/TV) drops, the reaction "
    "share rises without verse changing. Read trends, not absolute levels.",
    "- **Thin post window.** The exemplars went live only days before this report's end; "
    "small buckets are flagged `thin sample` and should not be over-read.",
]


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x}"


def _stat_rows(rows: list[tuple[str, BucketStats]]) -> list[str]:
    out = [
        "| window | fc42 msgs | reactions | per 100 msgs | per active day | active days | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, b in rows:
        note = (
            "thin sample"
            if b.fc42_msgs < _SMALL_SAMPLE_MSGS or b.reactions < _SMALL_SAMPLE_REACTIONS
            else ""
        )
        out.append(
            f"| {label} | {b.fc42_msgs} | {b.reactions} | "
            f"{_fmt(per_100(b.reactions, b.fc42_msgs))} | "
            f"{_fmt(per_day(b.reactions, b.active_days))} | {b.active_days} | {note} |"
        )
    out.append("")
    return out


def render_report(report: Report) -> str:
    lines = [
        "# Verse landing-rate report",
        "",
        f"Rollout boundary: **{report.rollout}** (curated style exemplars went live).",
        "",
        "## Headline — pre vs post rollout",
        "",
    ]
    lines += _stat_rows(
        [
            (f"pre  [.., {report.rollout})", report.pre),
            (f"post [{report.rollout}, ..]", report.post),
        ]
    )
    lines += ["## Monthly trend", ""]
    lines += _stat_rows([(b.label, b) for b in report.buckets])
    lines += ["## Distinct wins (latest first)", ""]
    if not report.wins:
        lines.append("_none detected_")
    for w in report.wins:
        text = w.text if len(w.text) <= _WIN_TEXT_CAP else w.text[: _WIN_TEXT_CAP - 3] + "..."
        lines.append(f"- {w.date} [{w.kind}] {text}")
    lines += _CAVEATS
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/taste_report.py plugins/llm/tests/verse/test_taste_report.py
git commit -m "feat(verse): render_report — markdown tables, wins, caveats"
```

---

### Task 5: CLI entrypoint (`_main`)

**Files:**
- Modify: `plugins/llm/src/llm/verse/taste_report.py`

> No automated test — CLI wiring is `# pragma: no cover`, matching `taste_mine._main`. Verify with a manual smoke run.

- [ ] **Step 1: Add the CLI**

```python
# append to taste_report.py (end of file)

def _main(argv=None):  # pragma: no cover - thin CLI wiring over tested core
    import argparse
    import re
    from pathlib import Path

    from .store import VerseStore

    ap = argparse.ArgumentParser(description="Report fc42's verse landing-rate from logs")
    ap.add_argument("logs", nargs="+", help="ChannelLogger .log files (named ...YYYY-MM-DD.log)")
    ap.add_argument("--verse-dir", required=True, help="verse store base dir")
    ap.add_argument("--channel", default="#afternet")
    ap.add_argument("--rollout", default=DEFAULT_ROLLOUT, help="pre/post boundary date (YYYY-MM-DD)")
    ap.add_argument("--out", default="verse_landing_report.md")
    args = ap.parse_args(argv)

    store = VerseStore(Path(args.verse_dir), args.channel)
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")
    dated: list[tuple[str, list[str]]] = []
    for p in args.logs:
        m = date_re.search(Path(p).name)
        if not m:
            print(f"skip (no YYYY-MM-DD in filename): {p}")
            continue
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        dated.append((m.group(1), text.splitlines()))

    report = build_report(dated, store, rollout=args.rollout)
    Path(args.out).write_text(render_report(report), encoding="utf-8")
    print(
        f"pre={report.pre.reactions}/{report.pre.fc42_msgs} "
        f"post={report.post.reactions}/{report.post.fc42_msgs} "
        f"wins={len(report.wins)} -> {args.out}"
    )


if __name__ == "__main__":  # pragma: no cover
    _main()
```

- [ ] **Step 2: Manual smoke test**

Run:
```bash
cd plugins/llm
printf '2026-06-22T10:00:00  <fc42> the stinky lads marched into the assembly hall and let off a perfectly timed duet that turned the leaves yellow indeed mate\n' > /tmp/'#afternet.2026-06-22.log'
uv run python -m llm.verse.taste_report /tmp/'#afternet.2026-06-22.log' --verse-dir /tmp/verse-smoke --channel '#afternet' --out /tmp/landing.md
```
Expected: prints a `pre=.. post=.. wins=.. -> /tmp/landing.md` line and writes the markdown. (A fresh empty store yields 0 reactions — that's fine; it proves the wiring and the `__main__.__file__` shim path. Reactions need a store whose `match_entities_in_text` knows the entities.)

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/src/llm/verse/taste_report.py
git commit -m "feat(verse): taste_report CLI entrypoint"
```

---

### Task 6: Preflight — lint, typecheck, coverage

**Files:** none (verification only)

- [ ] **Step 1: Lint + typecheck the package**

Run: `make lint && make typecheck`
Expected: clean. Fix any ruff/ty findings in `taste_report.py` (common: unused import, line length). Re-run until clean.

- [ ] **Step 2: Full suite + coverage floor**

Run: `make test`
Expected: all pass, coverage ≥ **93%**. If `taste_report.py` drags coverage, add focused tests for any uncovered branch (e.g. the `prev is None or date > prev.date` win-replacement branch, the `_fmt(None)` path). Do NOT add `# pragma: no cover` to real logic — only `_main` is exempt.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(verse): cover taste_report branches; lint/typecheck clean"
```

---

### Task 7: REMOVED — usage-label forward-proofing was red-teamed and dropped

The contingent usage-label change is **not done in this slice**. A pre-implementation red-team
(see spec §7) found the proposed edit at `service.py:4324` does not control the usage `command`
label at all — that map only labels timed-out tasks stashed for the retry queue
(`pending_tasks.task_type`). The real usage label is a hardcoded `"ask"` at `plugin.py:3846`.
Relabeling the map to `"verse"` would not make verse turns countable **and** would route verse
timeout-recovery into the "unknown task type" branch (`service.py:2381`/`2392`) — actively harmful.

A real per-turn usage denominator is a larger, route-aware change and is deferred. Slice 1 is fully
offline; the log-derived denominator stands alone.

---

## Self-Review

**Spec coverage:**
- §2.1 numerator (reuse extract_candidates) → Task 2 ✓
- §2.2 denominator (fc42 msg count) → Task 2 ✓
- §2.3 dating by filename → Task 5 (CLI) ✓; core stays filename-agnostic → Tasks 2/4 ✓
- §3 output: headline pre/post → Task 4 ✓; monthly trend → Task 4 ✓; distinct wins → Tasks 3/4 ✓; caveats → Task 4 ✓
- §4 components (Win/BucketStats/Report/build/render/_main) → Tasks 1/2/4/5 ✓
- §6 edge cases (empty, divide-by-zero, undated filename, per-file vs global dedup) → Tasks 2/3/4/5 ✓
- §7 usage-label → deferred (red-team); Task 7 removed, not implemented ✓
- §8 testing + 93% floor (branch coverage on) → Task 6 ✓

**Placeholder scan:** no placeholders; every code step shows complete code. (The former Task 7 gate was resolved by the pre-implementation red-team and removed.)

**Type consistency:** `BucketStats(label, fc42_msgs, reactions, active_days)`, `Win(date, kind, text)`, `Report(buckets, pre, post, wins, rollout)`, `build_report(dated_logs, store, *, rollout)`, `render_report(report)`, `per_100(reactions, msgs)`, `per_day(reactions, days)`, `_month(date)` — used identically across Tasks 1–5. ✓

> **Note on Task 1/2 ordering:** Task 1's test header imports `build_report`/`render_report`/`Win`, which are not defined until Tasks 2/4. If running strictly task-by-task, either (a) include thin `def build_report(*a, **k): raise NotImplementedError` / `def render_report(*a, **k): raise NotImplementedError` stubs in Task 1 Step 3 so the module imports, then replace them in Tasks 2/4, or (b) split the Task 1 test file header to import only what Task 1 defines and grow the import line per task. Approach (b) is cleaner; the executor should import per-task. This is called out so the executor isn't surprised by an import error in Task 1.
