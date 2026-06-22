"""Offline taste-miner: extract fc42's liked verse lines from ChannelLogger logs.

Read-only. Produces a candidate review file for human curation; never writes the
verse store or config. See docs/superpowers/specs/2026-06-21-fc42-taste-exemplars-design.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
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
_ADDRESSED_RE = re.compile(r"^(grok|vibebot)\b", re.IGNORECASE)  # only real bot triggers


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


def classify_repaste(body: str, store: Any, *, min_chars: int = _MIN_REPASTE_CHARS):
    text = _norm_ws(body)
    if len(text) < min_chars:
        return None
    if _URL_RE.search(text) or _ADDRESSED_RE.match(text):
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
