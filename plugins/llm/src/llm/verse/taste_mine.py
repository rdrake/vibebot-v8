"""Offline taste-miner: extract fc42's liked verse lines from ChannelLogger logs.

Read-only. Produces a candidate review file for human curation; never writes the
verse store or config. See docs/superpowers/specs/2026-06-21-fc42-taste-exemplars-design.md.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple

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
