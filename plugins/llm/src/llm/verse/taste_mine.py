"""Offline taste-miner: extract fc42's liked verse lines from ChannelLogger logs.

Read-only. Produces a candidate review file for human curation; never writes the
verse store or config. See docs/superpowers/specs/2026-06-21-fc42-taste-exemplars-design.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NamedTuple

_SEP = "  "  # ChannelLogger separates the timestamp from the body with two spaces


class Msg(NamedTuple):
    nick: str
    body: str


def _is_fc42(nick: str) -> bool:
    """Match fc42 and his connection variants (fc42_, fc42|away, Fc42)."""
    return nick.lower().startswith("fc42")


def iter_messages(lines: Iterable[str]) -> Iterator[Msg]:
    """Yield (nick, body) for privmsg `<nick> …` and action `* nick …` lines.

    Splits the timestamp off at the first double-space (robust to any ts format),
    then parses the body. Skips system (`*** …`), notice (`-x- …`), empty-body, and
    malformed lines. Caller opens files with errors='replace'.
    """
    for raw in lines:
        line = raw.rstrip("\r\n")
        ts, sep, rest = line.partition(_SEP)
        if not sep:
            continue
        if rest.startswith("<"):
            end = rest.find("> ")
            if end < 1:
                continue
            nick, body = rest[1:end], rest[end + 2 :]
        elif rest.startswith("* "):
            parts = rest[2:].split(" ", 1)
            if len(parts) != 2:
                continue
            nick, body = parts[0], parts[1]
        else:
            continue
        body = body.strip()
        if not body:
            continue
        yield Msg(nick, body)


_MIN_REPASTE_CHARS = 120
_URL_RE = re.compile(r"https?://")
#: Default bot nicks whose addressed lines are real bot triggers, not taste.
#: Override with extract_candidates(..., bot_nicks=...) / the --bot-nicks CLI
#: flag when the live supybot.nick config diverges from this.
_DEFAULT_BOT_NICKS: tuple[str, ...] = ("grok", "vibebot")


def _addressed_re(bot_nicks: Sequence[str]) -> re.Pattern[str]:
    """Compile the '^<botnick> …' addressed-line matcher for the given nicks."""
    return re.compile(r"^(?:" + "|".join(re.escape(n) for n in bot_nicks) + r")\b", re.IGNORECASE)


_ADDRESSED_RE = _addressed_re(_DEFAULT_BOT_NICKS)  # only real bot triggers


class Candidate(NamedTuple):
    text: str
    kind: str  # "repaste" | "praise"
    source_line: str
    needs_review: bool


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _strong_entity_match(text: str, ents: list[Any]) -> bool:
    """Auto-trustable: a multiword entity name, or a capitalized whole-word
    occurrence of the name in the ORIGINAL text."""
    for e in ents:
        name = e.name
        if " " in name:
            return True
        cap = name[:1].upper() + name[1:]
        if re.search(r"(?<!\w)" + re.escape(cap) + r"(?!\w)", text):
            return True
    return False


def classify_repaste(
    body: str,
    store: Any,
    *,
    min_chars: int = _MIN_REPASTE_CHARS,
    addressed_re: re.Pattern[str] = _ADDRESSED_RE,
):
    text = _norm_ws(body)
    if len(text) < min_chars:
        return None
    if _URL_RE.search(text) or addressed_re.match(text):
        return None
    ents = store.match_entities_in_text(text)
    if not ents:
        return None
    return Candidate(text, "repaste", body, not _strong_entity_match(text, ents))


_PRAISE_WORDS = (
    "good one",
    "amazing",
    "brilliant",
    "genius",
    "love it",
    "so good",
    "this is gold",
    "lmao that",
)
_PRAISE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _PRAISE_WORDS) + r")\b", re.IGNORECASE
)
_INLINE_RE = re.compile(r"when it said\s+(.*)$", re.IGNORECASE)
_LEADING_STOPWORDS = {"earlier", "that", "it", "when", "said"}  # NOT articles


def _strip_leading_stopwords(s: str) -> str:
    toks = s.split()
    i = 0
    while i < len(toks) and toks[i].lower().strip(",.!?") in _LEADING_STOPWORDS:
        i += 1
    return " ".join(toks[i:])


def classify_praise(body: str, store: Any, *, prev_line: str = ""):
    if not _PRAISE_RE.search(body):
        return None
    inline = _INLINE_RE.search(body)
    if inline:
        span = _strip_leading_stopwords(_norm_ws(inline.group(1)))
        if span and store.match_entities_in_text(span):
            return Candidate(span, "praise", body, True)
    prev = _norm_ws(prev_line)
    if prev:
        return Candidate(prev, "praise", body, True)
    return None


_DENIAL_RE = re.compile(
    r"\b(i can'?t|i cannot|i'?m sorry|as an ai|i won'?t|unable to|cannot help)\b",
    re.IGNORECASE,
)


def _dedup_key(text: str) -> str:
    return re.sub(r"[^\w ]+", "", text.lower()).strip()


def _nearest_source(
    msgs: list[Msg], i: int, *, addressed_re: re.Pattern[str] = _ADDRESSED_RE
) -> str:
    """Nearest prior non-fc42, non-URL, non-addressed line (spec attribution)."""
    for j in range(i - 1, -1, -1):
        m = msgs[j]
        if _is_fc42(m.nick):
            continue
        b = _norm_ws(m.body)
        if _URL_RE.search(b) or addressed_re.match(b):
            continue
        return m.body
    return ""


def extract_candidates(
    lines: Iterable[str],
    store: Any,
    *,
    min_repaste_chars: int = _MIN_REPASTE_CHARS,
    bot_nicks: Sequence[str] = _DEFAULT_BOT_NICKS,
) -> list[Candidate]:
    addressed_re = _addressed_re(bot_nicks)
    msgs = list(iter_messages(lines))
    out: list[Candidate] = []
    seen: set[str] = set()
    for i, m in enumerate(msgs):
        if not _is_fc42(m.nick):
            continue
        cand = classify_repaste(
            m.body, store, min_chars=min_repaste_chars, addressed_re=addressed_re
        ) or classify_praise(
            m.body, store, prev_line=_nearest_source(msgs, i, addressed_re=addressed_re)
        )
        if cand is None:
            continue
        key = _dedup_key(cand.text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cand)
    out.sort(key=lambda c: (c.needs_review, -len(c.text)))
    return out


def render_review(cands: list[Candidate]) -> str:
    trusted = [c.text for c in cands if not c.needs_review and not _DENIAL_RE.search(c.text)]
    lines = ["# fc42 taste-mine candidates", ""]
    for c in cands:
        flags = []
        if c.needs_review:
            flags.append("REVIEW")
        if _DENIAL_RE.search(c.text):
            flags.append("DENIAL?")
        tag = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"- [{c.kind}{tag}] {c.text}")
        lines.append(f"  src: {c.source_line}")
    lines += [
        "",
        "## Ready-to-paste (auto-trusted) JSON for verseStyleExemplars",
        "```json",
        json.dumps(trusted, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines)


def _main(argv=None):  # pragma: no cover - thin CLI wiring over tested core
    import argparse
    from pathlib import Path

    from .store import VerseStore

    ap = argparse.ArgumentParser(description="Mine fc42 taste exemplars from logs")
    ap.add_argument("logs", nargs="+", help="ChannelLogger .log files")
    ap.add_argument("--verse-dir", required=True, help="verse store base dir")
    ap.add_argument("--channel", default="#afternet")
    ap.add_argument("--out", default="taste_candidates.md")
    ap.add_argument(
        "--bot-nicks",
        default=",".join(_DEFAULT_BOT_NICKS),
        help="comma-separated bot nicks whose addressed lines are excluded "
        "(keep in sync with the live bot's nick config)",
    )
    args = ap.parse_args(argv)
    store = VerseStore(Path(args.verse_dir), args.channel)
    bot_nicks = [n.strip() for n in args.bot_nicks.split(",") if n.strip()]
    lines: list[str] = []
    for p in args.logs:
        lines += Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
    # real store.match_entities_in_text
    cands = extract_candidates(lines, store, bot_nicks=bot_nicks or _DEFAULT_BOT_NICKS)
    Path(args.out).write_text(render_review(cands), encoding="utf-8")
    print(f"{len(cands)} candidates -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    _main()
