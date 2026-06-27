# Verse Reaction Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture inbound IRCv3 👍/👎 reactions to the bot's verse lines, attribute each by recency to the verse turn it reacts to, log it offline, and surface approve/disapprove counts (pre/post the 2026-06-22 exemplar rollout) in the landing report.

**Architecture:** A pure side-effect-free core (`verse/reactions.py`) does classification, recency attribution, JSONL (de)serialisation, and the report section. Thin IRC glue in `plugin.py` records the bot's last verse line per channel (`_last_bot_line`) on send, and a new `doTagmsg` handler matches an inbound reaction to it and appends a JSON line to `reactions.jsonl`. The offline `taste_report --reactions` reads that file. Measurement only — no bot behaviour change, default-on, exception-isolated.

**Tech Stack:** Python 3, Limnoria (`callbacks.Plugin`, IRCv3 `message-tags`), pytest + pytest-mock. Spec: `docs/superpowers/specs/2026-06-27-verse-reaction-signal-design.md`.

---

## File Structure

- **Create** `plugins/llm/src/llm/verse/reactions.py` — pure core: `classify_emoji`, `process_reaction`, `event_to_jsonl`, `parse_reaction_lines`, `ReactionBucket`/`ReactionReport`, `build_reaction_report`, `render_reaction_section`.
- **Create** `plugins/llm/tests/verse/test_reactions.py` — full unit coverage of the pure core.
- **Modify** `plugins/llm/src/llm/verse/taste_report.py` — `_main` gains optional `--reactions`, appends the section.
- **Modify** `plugins/llm/tests/verse/test_taste_report.py` — `_main --reactions` end-to-end test.
- **Modify** `plugins/llm/src/llm/service.py` — add `was_verse` to `AssistantResult`, set it in `assistant_completion`.
- **Modify** `plugins/llm/tests/test_service_completion.py` — `was_verse` field test.
- **Modify** `plugins/llm/src/llm/config.py` — register `verseReactionCaptureEnabled` (default True).
- **Modify** `plugins/llm/src/llm/plugin.py` — `__init__` state, send-hook at `_dispatch_assistant_reply`, new `doTagmsg` + `_append_reaction_event`.
- **Modify** `plugins/llm/tests/test_plugin_verse.py` — send-hook + `doTagmsg` glue tests.

DRY note: the live glue delegates ALL logic to `reactions.py`; `plugin.py` only parses the IRC message and does file IO.

---

### Task 1: `classify_emoji` (pure)

**Files:**
- Create: `plugins/llm/src/llm/verse/reactions.py`
- Test: `plugins/llm/tests/verse/test_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
# plugins/llm/tests/verse/test_reactions.py
from llm.verse.reactions import classify_emoji


def test_classify_thumbs_up():
    assert classify_emoji("\U0001f44d") == "approve"


def test_classify_thumbs_down():
    assert classify_emoji("\U0001f44e") == "disapprove"


def test_classify_skin_tone_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d\U0001f3fd") == "approve"  # 👍🏽


def test_classify_variation_selector_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d️") == "approve"  # 👍️


def test_classify_other_emoji_is_other():
    assert classify_emoji("❤️") == "other"  # ❤️


def test_classify_empty_is_other():
    assert classify_emoji("") == "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.verse.reactions'`

- [ ] **Step 3: Write minimal implementation**

```python
# plugins/llm/src/llm/verse/reactions.py
"""Offline verse reaction signal: capture + report fc42's 👍/👎 to verse lines.

Pure, side-effect-free core (classification, recency attribution, JSONL
(de)serialisation, and the offline report section). The live IRC glue (doTagmsg,
the send-hook) lives in plugin.py and delegates here. See
docs/superpowers/specs/2026-06-27-verse-reaction-signal-design.md.
"""

from __future__ import annotations

_THUMB_UP = "\U0001f44d"
_THUMB_DOWN = "\U0001f44e"
_SKIN_TONES = {"\U0001f3fb", "\U0001f3fc", "\U0001f3fd", "\U0001f3fe", "\U0001f3ff"}
_VARIATION_SELECTOR = "️"


def classify_emoji(emoji: str) -> str:
    """Map a reaction emoji to 'approve' | 'disapprove' | 'other'.

    Strips skin-tone modifiers and the U+FE0F variation selector so 👍🏽 / 👍️ match 👍.
    """
    core = "".join(
        c for c in (emoji or "") if c not in _SKIN_TONES and c != _VARIATION_SELECTOR
    )
    if core == _THUMB_UP:
        return "approve"
    if core == _THUMB_DOWN:
        return "disapprove"
    return "other"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/reactions.py plugins/llm/tests/verse/test_reactions.py
git commit -m "feat(verse): emoji classification for reaction signal"
```

---

### Task 2: `process_reaction` + `event_to_jsonl` + `_iso` (pure capture core)

**Files:**
- Modify: `plugins/llm/src/llm/verse/reactions.py`
- Test: `plugins/llm/tests/verse/test_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reactions.py
import json

from llm.verse.reactions import RECENCY_WINDOW_S, event_to_jsonl, process_reaction


def _line(ts):
    return {"text": "Methane Max hacked the tannoy", "ts": ts}


def test_process_reaction_approve_within_window():
    ev = process_reaction(
        react_emoji="\U0001f44d", reactor="fc42", channel="#afnet", network="afnet",
        target_msgid="m1", last_bot_line=_line(1000.0), now=1007.0, capture_enabled=True,
    )
    assert ev is not None
    assert ev["sentiment"] == "approve"
    assert ev["reactor"] == "fc42"
    assert ev["was_verse"] is True
    assert ev["recency_s"] == 7.0
    assert ev["ts"] == "1970-01-01T00:16:47Z"  # _iso(1007.0)
    assert ev["verse_excerpt"] == "Methane Max hacked the tannoy"


def test_process_reaction_window_boundary_inclusive():
    assert process_reaction(
        react_emoji="\U0001f44e", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=_line(1000.0), now=1000.0 + RECENCY_WINDOW_S,
        capture_enabled=True,
    ) is not None


def test_process_reaction_stale_line_dropped():
    assert process_reaction(
        react_emoji="\U0001f44d", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=_line(1000.0), now=1000.0 + RECENCY_WINDOW_S + 1,
        capture_enabled=True,
    ) is None


def test_process_reaction_disabled_dropped():
    assert process_reaction(
        react_emoji="\U0001f44d", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=_line(1000.0), now=1001.0, capture_enabled=False,
    ) is None


def test_process_reaction_no_emoji_dropped():
    assert process_reaction(
        react_emoji="", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=_line(1000.0), now=1001.0, capture_enabled=True,
    ) is None


def test_process_reaction_no_line_dropped():
    assert process_reaction(
        react_emoji="\U0001f44d", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=None, now=1001.0, capture_enabled=True,
    ) is None


def test_process_reaction_clock_skew_dropped():
    assert process_reaction(
        react_emoji="\U0001f44d", reactor="fc42", channel="#a", network="n",
        target_msgid=None, last_bot_line=_line(2000.0), now=1000.0, capture_enabled=True,
    ) is None


def test_event_to_jsonl_roundtrip_preserves_unicode():
    ev = {"sentiment": "approve", "emoji": "\U0001f44d"}
    assert json.loads(event_to_jsonl(ev)) == ev
    assert "\\u" not in event_to_jsonl(ev)  # ensure_ascii=False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q -k "process_reaction or event_to_jsonl"`
Expected: FAIL — `ImportError: cannot import name 'process_reaction'`

- [ ] **Step 3: Write minimal implementation**

Add to `reactions.py` (after the imports / classify block):

```python
import json
from datetime import datetime, timezone

RECENCY_WINDOW_S = 300.0
_EXCERPT_CAP = 120


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_reaction(
    *,
    react_emoji: str,
    reactor: str,
    channel: str,
    network: str,
    target_msgid: str | None,
    last_bot_line: dict | None,
    now: float,
    capture_enabled: bool,
    window: float = RECENCY_WINDOW_S,
    excerpt_cap: int = _EXCERPT_CAP,
) -> dict | None:
    """Return a JSONL-ready reaction event, or None to skip.

    Skips when capture is disabled, there is no reaction emoji, there is no
    remembered verse line for the channel, or that line is outside the recency
    window (including clock-skew where now < ts).
    """
    if not capture_enabled or not react_emoji or not last_bot_line:
        return None
    ts = last_bot_line.get("ts")
    if ts is None or now < ts or now - ts > window:
        return None
    text = last_bot_line.get("text") or ""
    return {
        "ts": _iso(now),
        "network": network,
        "channel": channel,
        "reactor": reactor,
        "emoji": react_emoji,
        "sentiment": classify_emoji(react_emoji),
        "was_verse": True,
        "target_msgid": target_msgid,
        "recency_s": round(now - ts, 2),
        "verse_excerpt": text[:excerpt_cap],
    }


def event_to_jsonl(event: dict) -> str:
    """Serialise one reaction event to a single JSON line (no trailing newline)."""
    return json.dumps(event, ensure_ascii=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/reactions.py plugins/llm/tests/verse/test_reactions.py
git commit -m "feat(verse): recency-attributed reaction event builder"
```

---

### Task 3: `parse_reaction_lines` (pure, tolerant JSONL reader)

**Files:**
- Modify: `plugins/llm/src/llm/verse/reactions.py`
- Test: `plugins/llm/tests/verse/test_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reactions.py
from llm.verse.reactions import parse_reaction_lines


def test_parse_reaction_lines_tolerant():
    lines = [
        '{"ts": "2026-06-25T10:00:00Z", "sentiment": "approve"}',
        "",                       # blank skipped
        "   ",                    # whitespace skipped
        "not json",               # malformed skipped
        "[1, 2, 3]",              # non-dict skipped
        '{"sentiment": "approve"}',  # no ts skipped
        '{"ts": "2026-06-26T10:00:00Z", "sentiment": "disapprove"}',
    ]
    out = parse_reaction_lines(lines)
    assert len(out) == 2
    assert out[0]["sentiment"] == "approve"
    assert out[1]["sentiment"] == "disapprove"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q -k parse_reaction_lines`
Expected: FAIL — `ImportError: cannot import name 'parse_reaction_lines'`

- [ ] **Step 3: Write minimal implementation**

Add to `reactions.py`:

```python
from collections.abc import Iterable


def parse_reaction_lines(lines: Iterable[str]) -> list[dict]:
    """Parse JSONL reaction events, tolerating blank/malformed/non-dict lines."""
    out: list[dict] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict) and ev.get("ts"):
            out.append(ev)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/reactions.py plugins/llm/tests/verse/test_reactions.py
git commit -m "feat(verse): tolerant reactions JSONL parser"
```

---

### Task 4: `build_reaction_report` + buckets (pure aggregation)

**Files:**
- Modify: `plugins/llm/src/llm/verse/reactions.py`
- Test: `plugins/llm/tests/verse/test_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reactions.py
from llm.verse.reactions import ReactionBucket, build_reaction_report


def _ev(date, sentiment, reactor="fc42", channel="#afnet"):
    return {"ts": f"{date}T10:00:00Z", "sentiment": sentiment, "reactor": reactor,
            "channel": channel, "verse_excerpt": "x"}


def test_build_reaction_report_pre_post_split():
    events = [
        _ev("2026-06-20", "approve"),
        _ev("2026-06-21", "disapprove"),
        _ev("2026-06-23", "approve"),   # post (rollout 2026-06-22)
        _ev("2026-06-24", "approve", reactor="eck"),
    ]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert rep.pre == ReactionBucket("pre", 2, 1, 1, 0, 1)
    assert rep.post == ReactionBucket("post", 2, 2, 0, 0, 2)


def test_build_reaction_report_channel_filter():
    events = [_ev("2026-06-23", "approve", channel="#afnet"),
              _ev("2026-06-23", "approve", channel="#other")]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert rep.post.reactions == 1


def test_build_reaction_report_monthly_and_recent_order():
    events = [_ev("2026-05-10", "approve"), _ev("2026-06-23", "disapprove")]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert [b.label for b in rep.buckets] == ["2026-05", "2026-06"]
    assert rep.recent[0]["ts"].startswith("2026-06-23")  # latest first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q -k build_reaction_report`
Expected: FAIL — `ImportError: cannot import name 'ReactionBucket'`

- [ ] **Step 3: Write minimal implementation**

Add to `reactions.py`:

```python
from collections import defaultdict
from typing import NamedTuple

DEFAULT_ROLLOUT = "2026-06-22"
_RECENT_CAP = 15


class ReactionBucket(NamedTuple):
    label: str
    reactions: int
    approve: int
    disapprove: int
    other: int
    reactors: int


class ReactionReport(NamedTuple):
    buckets: list[ReactionBucket]  # monthly, sorted by label
    pre: ReactionBucket
    post: ReactionBucket
    recent: list[dict]  # latest-first, capped
    rollout: str


def _bucket(label: str, events: list[dict]) -> ReactionBucket:
    approve = sum(1 for e in events if e.get("sentiment") == "approve")
    disapprove = sum(1 for e in events if e.get("sentiment") == "disapprove")
    other = len(events) - approve - disapprove
    reactors = len({e.get("reactor") for e in events})
    return ReactionBucket(label, len(events), approve, disapprove, other, reactors)


def build_reaction_report(
    events: Iterable[dict],
    *,
    rollout: str = DEFAULT_ROLLOUT,
    channel: str | None = None,
) -> ReactionReport:
    """Aggregate reaction events into monthly + pre/post buckets and a recent list.

    Each event is dated by ``ts[:10]``. If ``channel`` is given, only events for
    that channel are counted.
    """
    evs = [
        e
        for e in events
        if (channel is None or e.get("channel") == channel)
        and isinstance(e.get("ts"), str)
        and len(e["ts"]) >= 10
    ]
    by_month: dict[str, list[dict]] = defaultdict(list)
    pre: list[dict] = []
    post: list[dict] = []
    for e in evs:
        date = e["ts"][:10]
        by_month[date[:7]].append(e)
        (pre if date < rollout else post).append(e)
    buckets = [_bucket(m, by_month[m]) for m in sorted(by_month)]
    recent = sorted(evs, key=lambda e: e["ts"], reverse=True)[:_RECENT_CAP]
    return ReactionReport(buckets, _bucket("pre", pre), _bucket("post", post), recent, rollout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/reactions.py plugins/llm/tests/verse/test_reactions.py
git commit -m "feat(verse): reaction report aggregation (pre/post + monthly)"
```

---

### Task 5: `render_reaction_section` (pure markdown)

**Files:**
- Modify: `plugins/llm/src/llm/verse/reactions.py`
- Test: `plugins/llm/tests/verse/test_reactions.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_reactions.py
from llm.verse.reactions import render_reaction_section


def test_render_reaction_section_has_tables_and_caveats():
    events = [_ev("2026-06-23", "approve"), _ev("2026-06-24", "disapprove")]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    out = render_reaction_section(rep)
    assert "## Explicit 👍/👎 reactions" in out
    assert "| window | reactions | 👍 | 👎 | other | net | reactors | note |" in out
    assert "Recency-attributed" in out  # caveat present
    assert "+1" in out or "-1" in out  # net column rendered with sign


def test_render_reaction_section_empty_recent():
    rep = build_reaction_report([], rollout="2026-06-22", channel="#afnet")
    out = render_reaction_section(rep)
    assert "_none captured yet_" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q -k render_reaction_section`
Expected: FAIL — `ImportError: cannot import name 'render_reaction_section'`

- [ ] **Step 3: Write minimal implementation**

Add to `reactions.py`:

```python
_THIN_SAMPLE = 5

_CAVEATS = [
    "",
    "### How to read the reaction signal",
    "",
    "- **Explicit + bidirectional**, but reactions require a reaction-capable "
    "client; absence of a reaction is not disapproval.",
    "- **Recency-attributed.** Each reaction is tied to the bot's most recent verse "
    "line in the channel; a reaction to a non-bot message within the window can be "
    "mis-attributed.",
    "- Thin buckets (`< 5` reactions) are flagged `thin sample`.",
]


def _reaction_rows(rows: list[tuple[str, ReactionBucket]]) -> list[str]:
    out = [
        "| window | reactions | 👍 | 👎 | other | net | reactors | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, b in rows:
        note = "thin sample" if b.reactions < _THIN_SAMPLE else ""
        net = b.approve - b.disapprove
        out.append(
            f"| {label} | {b.reactions} | {b.approve} | {b.disapprove} | "
            f"{b.other} | {net:+d} | {b.reactors} | {note} |"
        )
    out.append("")
    return out


def render_reaction_section(report: ReactionReport) -> str:
    """Render the 'Explicit reactions' markdown section for the landing report."""
    lines = [
        "## Explicit 👍/👎 reactions",
        "",
        f"Rollout boundary: **{report.rollout}**.",
        "",
        "### Headline — pre vs post rollout",
        "",
    ]
    lines += _reaction_rows(
        [
            (f"pre  [.., {report.rollout})", report.pre),
            (f"post [{report.rollout}, ..]", report.post),
        ]
    )
    lines += ["### Monthly trend", ""]
    lines += _reaction_rows([(b.label, b) for b in report.buckets])
    lines += ["### Recent reactions (latest first)", ""]
    if not report.recent:
        lines.append("_none captured yet_")
    for e in report.recent:
        date = e["ts"][:10]
        sentiment = e.get("sentiment", "?")
        reactor = e.get("reactor", "?")
        excerpt = (e.get("verse_excerpt") or "")[:100]
        lines.append(f"- {date} [{sentiment} by {reactor}] {excerpt}")
    lines += _CAVEATS
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes + check coverage**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q --cov=llm.verse.reactions --cov-report=term-missing`
Expected: PASS; `reactions.py` at or near 100% (note any missed lines for follow-up).

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/reactions.py plugins/llm/tests/verse/test_reactions.py
git commit -m "feat(verse): render explicit-reactions report section"
```

---

### Task 6: Wire `--reactions` into `taste_report`

**Files:**
- Modify: `plugins/llm/src/llm/verse/taste_report.py` (the `_main` function, ~line 183-215)
- Test: `plugins/llm/tests/verse/test_taste_report.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_taste_report.py
from pathlib import Path

from llm.verse.taste_report import _main


def test_main_appends_reaction_section(tmp_path: Path):
    # one empty log file named with a date (no entities -> 0 landing reactions)
    log = tmp_path / "#afnet.2026-06-23.log"
    log.write_text("2026-06-23T10:00:00  <fc42> hello\n", encoding="utf-8")
    reactions = tmp_path / "reactions.jsonl"
    reactions.write_text(
        '{"ts": "2026-06-23T10:01:00Z", "sentiment": "approve", "reactor": "fc42",'
        ' "channel": "#afnet", "verse_excerpt": "Methane Max"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    _main([
        str(log), "--verse-dir", str(tmp_path), "--channel", "#afnet",
        "--rollout", "2026-06-22", "--reactions", str(reactions), "--out", str(out),
    ])
    text = out.read_text(encoding="utf-8")
    assert "# Verse landing-rate report" in text       # landing report still present
    assert "## Explicit 👍/👎 reactions" in text        # reaction section appended


def test_main_without_reactions_has_no_reaction_section(tmp_path: Path):
    log = tmp_path / "#afnet.2026-06-23.log"
    log.write_text("2026-06-23T10:00:00  <fc42> hello\n", encoding="utf-8")
    out = tmp_path / "report.md"
    _main([str(log), "--verse-dir", str(tmp_path), "--channel", "#afnet", "--out", str(out)])
    assert "## Explicit" not in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q -k reaction`
Expected: FAIL — `unrecognized arguments: --reactions`

- [ ] **Step 3: Write minimal implementation**

In `taste_report.py` `_main`, add the argument after the existing `--out` line:

```python
    ap.add_argument(
        "--reactions",
        default=None,
        help="optional reactions.jsonl; appends an explicit 👍/👎 section",
    )
```

Replace the final write block:

```python
    report = build_report(dated, store, rollout=args.rollout)
    out_text = render_report(report)
    if args.reactions:
        from .reactions import (
            build_reaction_report,
            parse_reaction_lines,
            render_reaction_section,
        )

        rlines = Path(args.reactions).read_text(encoding="utf-8", errors="replace").splitlines()
        rreport = build_reaction_report(
            parse_reaction_lines(rlines), rollout=args.rollout, channel=args.channel
        )
        out_text += "\n\n" + render_reaction_section(rreport)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(
        f"pre={report.pre.reactions}/{report.pre.fc42_msgs} "
        f"post={report.post.reactions}/{report.post.fc42_msgs} "
        f"wins={len(report.wins)} -> {args.out}"
    )
```

Note: `_main` currently carries `# pragma: no cover`. Remove that pragma from the `_main` signature line so the new `--reactions` branch is counted (the two tests above exercise both paths).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/verse/test_taste_report.py -q`
Expected: PASS (existing 17 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/verse/taste_report.py plugins/llm/tests/verse/test_taste_report.py
git commit -m "feat(verse): taste_report --reactions appends explicit signal section"
```

---

### Task 7: `AssistantResult.was_verse` field + set in `assistant_completion`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (`AssistantResult` at :537; `assistant_completion` at :3689, verse check :3810, returns :4114/:4281/:4301)
- Test: `plugins/llm/tests/test_service_completion.py`

**Red-team focus:** this is the one seam that crosses the heavy completion path. The field-level test below is deterministic; the *runtime* correctness (verse turns actually emit `was_verse=True`) is additionally proven by Task 9's consumption test and the post-merge prod smoke (Task 11). Flag it for the Codex review.

- [ ] **Step 1: Write the failing test**

```python
# append to test_service_completion.py
from llm.service import AssistantResult


def test_assistant_result_was_verse_defaults_false():
    assert AssistantResult(content="x").was_verse is False


def test_assistant_result_was_verse_settable():
    assert AssistantResult(content="x", was_verse=True).was_verse is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/test_service_completion.py -q -k was_verse`
Expected: FAIL — `TypeError: __new__() got an unexpected keyword argument 'was_verse'`

- [ ] **Step 3: Write minimal implementation**

In `service.py`, add the field to the `AssistantResult` NamedTuple (after `final_text_after_tools`):

```python
class AssistantResult(NamedTuple):
    """Result of an assistant tool-calling loop."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    grounding_used: bool = False
    error: str | None = None
    last_successful_tool: str | None = None
    final_text_after_tools: str = ""
    was_verse: bool = False
```

In `assistant_completion`, immediately after the function's opening (where `route_profile` is in scope), compute the flag once:

```python
        was_verse = route_profile == PROFILE_VERSE
```

Then add `was_verse=was_verse` to the **content-success** return(s) only — the primary sanitized-content return near line 4114, and the URL/teaser success returns near 4281 and 4301 if they carry assistant text. Leave the error/empty returns (3780, 4103, 4242, 4344, 4345, 4352) at the default `False`. Example for the primary return:

```python
        return AssistantResult(
            content=self.sanitize_output(content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            model=model,
            grounding_used=grounding_used,
            last_successful_tool=last_successful_tool,
            final_text_after_tools=final_text_after_tools,
            was_verse=was_verse,
        )
```

(Read the real return at :4114 first and add only the `was_verse=was_verse` kwarg, preserving its existing kwargs.)

- [ ] **Step 4: Run tests to verify field test passes and nothing regressed**

Run: `cd plugins/llm && uv run pytest tests/test_service_completion.py -q`
Expected: PASS (existing + 2 new). The default keeps all 9 construction sites valid.

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service_completion.py
git commit -m "feat(verse): tag AssistantResult.was_verse on verse completions"
```

---

### Task 8: Register `verseReactionCaptureEnabled` config (default True)

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (after the `verseStorybookEnabled` block, ~:432-437)

- [ ] **Step 1: Add the registration**

Mirror `verseStorybookEnabled` exactly, default `True`:

```python
conf.registerChannelValue(
    LLM,
    "verseReactionCaptureEnabled",
    registry.Boolean(
        True,
        _("""Capture inbound IRCv3 emoji reactions (+draft/react) to the bot's
        verse lines as an offline approval signal (reactions.jsonl). Recency-
        attributed; measurement only — no behaviour change. Kill-switch."""),
    ),
)
```

- [ ] **Step 2: Verify the plugin still loads and the key reads True by default**

Run: `cd plugins/llm && uv run python -c "import llm.config"`
Expected: no error.

Run: `cd plugins/llm && uv run pytest tests/test_config.py -q`
Expected: PASS (no regressions).

(The gate behaviour with this key True/False is unit-tested in Task 10 via `doTagmsg`; the registration itself is a verbatim mirror of a working pattern and is exercised by `make preflight` loading the plugin.)

- [ ] **Step 3: Commit**

```bash
git add plugins/llm/src/llm/config.py
git commit -m "feat(verse): add verseReactionCaptureEnabled flag (default on)"
```

---

### Task 9: Plugin state + send-hook (record last verse line)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`__init__` after :690; `_dispatch_assistant_reply` send at :2500)
- Test: `plugins/llm/tests/test_plugin_verse.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_plugin_verse.py
import time

from llm.service import AssistantResult


class TestVerseReactionSendHook:
    def test_verse_reply_records_last_bot_line(self, plugin_env):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        result = AssistantResult(content="A tale of Methane Max", was_verse=True)
        plugin._dispatch_assistant_reply(
            irc, msg, result, nick="fc42", channel="#test",
            response="A tale of Methane Max",
        )
        last = plugin._last_bot_line.get(("testnet", "#test"))
        assert last is not None
        assert last["text"].endswith("Methane Max")
        assert isinstance(last["ts"], float)

    def test_non_verse_reply_does_not_record(self, plugin_env):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        result = AssistantResult(content="just chatting", was_verse=False)
        plugin._dispatch_assistant_reply(
            irc, msg, result, nick="fc42", channel="#test", response="just chatting",
        )
        assert ("testnet", "#test") not in plugin._last_bot_line
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -q -k VerseReactionSendHook`
Expected: FAIL — `AttributeError: 'LLM' object has no attribute '_last_bot_line'`

- [ ] **Step 3: Write minimal implementation**

In `plugin.py` `__init__`, after the `self._irc_send_lock = threading.Lock()` line (~:690):

```python
        # Recency-attributed verse reaction signal (see verse/reactions.py).
        # Last verse line the bot said per (network, channel); read by doTagmsg.
        self._last_bot_line: dict[tuple[str, str], dict] = {}
        self._reaction_log_lock = threading.Lock()
```

In `_dispatch_assistant_reply`, at the send site (~:2500), insert between the `_send_long_reply` call and `return response, True`:

```python
        self._send_long_reply(irc, msg, display_response, prefixNick=False)
        if getattr(result, "was_verse", False):
            try:
                with self._irc_send_lock:
                    self._last_bot_line[(irc.network, channel)] = {
                        "text": display_response,
                        "ts": time.time(),
                    }
            except Exception:  # never let signal capture disturb the reply path
                self.log.exception("last_bot_line store failed")
        return response, True
```

Confirm `import time` exists at the top of `plugin.py` (it is used elsewhere); if missing, add it.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -q -k VerseReactionSendHook`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "feat(verse): record last verse line per channel for reaction attribution"
```

---

### Task 10: `doTagmsg` inbound handler + `_append_reaction_event`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (new methods on `LLM`; add `from llm.verse import reactions` import; confirm `Path`/`conf` imported)
- Test: `plugins/llm/tests/test_plugin_verse.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test_plugin_verse.py
import json

from .conftest import make_registry_side_effect


class TestDoTagmsgReactionCapture:
    @pytest.fixture
    def reaction_env(self, plugin_env, tmp_path, mocker):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        mocker.patch(
            "llm.plugin.conf.supybot.directories.data", return_value=str(tmp_path)
        )
        base = make_registry_side_effect()

        def _reg(key, *a):
            if key == "verseReactionCaptureEnabled":
                return True
            return base(key, *a)

        plugin.registryValue.side_effect = _reg
        return plugin, irc, msg, tmp_path

    def _path(self, tmp_path):
        return tmp_path / "verse" / "reactions.jsonl"

    def test_thumbs_up_on_recent_verse_line_logged(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        plugin._last_bot_line[("testnet", "#test")] = {
            "text": "Methane Max hacked the tannoy", "ts": time.time(),
        }
        msg.server_tags = {"+draft/react": "\U0001f44d", "+draft/reply": "abc"}
        msg.channel = "#test"
        msg.nick = "fc42"
        plugin.doTagmsg(irc, msg)
        lines = self._path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["sentiment"] == "approve"
        assert ev["reactor"] == "fc42"
        assert ev["was_verse"] is True

    def test_non_react_tagmsg_ignored(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        msg.server_tags = {"+typing": "active"}
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_capture_disabled_skips(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        base = make_registry_side_effect()
        plugin.registryValue.side_effect = (
            lambda key, *a: False if key == "verseReactionCaptureEnabled" else base(key, *a)
        )
        plugin._last_bot_line[("testnet", "#test")] = {"text": "x", "ts": time.time()}
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = "#test"
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_no_recent_verse_line_skips(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = "#test"
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -q -k DoTagmsgReactionCapture`
Expected: FAIL — `AttributeError: 'LLM' object has no attribute 'doTagmsg'`

- [ ] **Step 3: Write minimal implementation**

At the top of `plugin.py`, add to the verse imports: `from llm.verse import reactions`. Confirm `Path` (from `pathlib`) and `conf` (`supybot.conf`) are already imported (they are — see the verse-store dir usage at ~:5487).

Add these methods to the `LLM` class (near `doPrivmsg`, ~:1101):

```python
    def doTagmsg(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
        """Capture inbound IRCv3 emoji reactions (+draft/react) to verse lines.

        Recency-attributed, measurement only (no reply). Fully exception-isolated
        so a capture bug can never disturb the IRC event loop. See verse/reactions.py.
        """
        try:
            server_tags = getattr(msg, "server_tags", None) or {}
            react_emoji = server_tags.get("+draft/react")
            if not react_emoji:
                return
            channel = msg.channel or (msg.args[0] if msg.args else "")
            if not channel or not channel.startswith(("#", "&")):
                return
            if not self.registryValue("verseReactionCaptureEnabled", channel):
                return
            with self._irc_send_lock:
                last = self._last_bot_line.get((irc.network, channel))
            event = reactions.process_reaction(
                react_emoji=react_emoji,
                reactor=msg.nick,
                channel=channel,
                network=irc.network,
                target_msgid=server_tags.get("+draft/reply"),
                last_bot_line=last,
                now=time.time(),
                capture_enabled=True,
            )
            if event is not None:
                self._append_reaction_event(event)
        except Exception:
            self.log.exception("doTagmsg reaction capture failed")

    def _append_reaction_event(self, event: dict) -> None:
        """Append one reaction event to <data>/verse/reactions.jsonl (thread-safe)."""
        base = Path(conf.supybot.directories.data()) / "verse"
        path = base / "reactions.jsonl"
        with self._reaction_log_lock:
            base.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(reactions.event_to_jsonl(event) + "\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd plugins/llm && uv run pytest tests/test_plugin_verse.py -q -k DoTagmsgReactionCapture`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin_verse.py
git commit -m "feat(verse): doTagmsg inbound reaction capture -> reactions.jsonl"
```

---

### Task 11: Full verification + prod smoke prep

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite + lint + types**

Run: `make preflight`
Expected: green. If `make` targets differ, run `cd plugins/llm && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run ty check`.

- [ ] **Step 2: Confirm new-module coverage**

Run: `cd plugins/llm && uv run pytest tests/verse/test_reactions.py -q --cov=llm.verse.reactions --cov-report=term-missing`
Expected: `reactions.py` 100% (or document any intentional misses).

- [ ] **Step 3: Confirm no behaviour change when disabled / no reactions**

Verify by inspection + tests already written: `taste_report` without `--reactions` is unchanged (Task 6 second test); `doTagmsg` with `verseReactionCaptureEnabled=False` writes nothing (Task 10).

- [ ] **Step 4: Prod smoke (post-merge, after auto-deploy)**

After merge + deploy, in #afternet: have the bot produce a verse line, react to it 👍 from a reaction-capable client, then run the prod recipe and confirm a row lands:

```bash
docker exec vibebot sh -lc "tail -n 3 /config/data/verse/reactions.jsonl"
docker exec vibebot sh -lc "/app/.venv/bin/python -m llm.verse.taste_report \
  /config/logs/ChannelLogger/afternet/'#afternet'/'#afternet'.*.log \
  --verse-dir /config/data/verse --channel '#afternet' --rollout 2026-06-22 \
  --reactions /config/data/verse/reactions.jsonl --out /tmp/verse_landing_report.md"
```

---

## Self-Review

**1. Spec coverage:**
- Recency attribution, no echo-message → Tasks 2, 9, 10. ✓
- Measurement-only, no reply → Task 10 (no outbound). ✓
- `was_verse` plumb-through → Task 7 + Task 9. ✓
- `doTagmsg` capture → Task 10. ✓
- `reactions.jsonl` append-only → Task 10 `_append_reaction_event`. ✓
- `taste_report --reactions` section → Tasks 4-6. ✓
- `verseReactionCaptureEnabled` default True kill-switch → Task 8, gated in Task 10. ✓
- Exception-wrapped paths → Task 9 (send-hook try/except), Task 10 (doTagmsg try/except). ✓
- Verse-scoped by construction (only verse lines stored) → Task 9 (`if result.was_verse`). ✓
- Pure tested core + thin glue → Tasks 1-6 pure (100%), Tasks 7-10 glue (unit-tested via harness). ✓
- Honest caveats in report → Task 5 `_CAVEATS`. ✓

**2. Placeholder scan:** none — every code step has full code; the one integration-verified seam (Task 7 runtime correctness) is explicitly flagged for red-team + smoke, not left vague.

**3. Type consistency:** `process_reaction` kwargs match between Task 2 (def) and Task 10 (call). `_last_bot_line` shape `{"text", "ts"}` matches between Task 9 (write) and Task 2/10 (read). `ReactionBucket`/`ReactionReport` fields match Task 4 (def), Task 5 (render), Task 6 (build). `was_verse` field name consistent across Tasks 7/9.
