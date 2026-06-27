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


_CAVEATS = [
    "",
    "## How to read this",
    "",
    "- **Positive signal only.** This counts reactions fc42 *volunteered* "
    "(re-pastes + praise). Silence is not dislike; absence of a reaction is not a "
    "negative score.",
    "- **Denominator is activity, not verse turns.** Verse completions are logged as "
    '`command="ask"` and the events table is compaction-lossy, so there is no clean '
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


def _main(argv=None):
    import argparse
    import re
    from pathlib import Path

    from .store import VerseStore

    ap = argparse.ArgumentParser(description="Report fc42's verse landing-rate from logs")
    ap.add_argument("logs", nargs="+", help="ChannelLogger .log files (named ...YYYY-MM-DD.log)")
    ap.add_argument("--verse-dir", required=True, help="verse store base dir")
    ap.add_argument("--channel", default="#afternet")
    ap.add_argument("--rollout", default=DEFAULT_ROLLOUT, help="pre/post boundary date YYYY-MM-DD")
    ap.add_argument("--out", default="verse_landing_report.md")
    ap.add_argument(
        "--reactions",
        default=None,
        help="optional reactions.jsonl; appends an explicit 👍/👎 section",
    )
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


if __name__ == "__main__":  # pragma: no cover
    _main()
