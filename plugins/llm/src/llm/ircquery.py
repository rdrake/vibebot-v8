"""Correlate asynchronous LIST / NAMES replies with the caller that asked.

IRC answers ``LIST`` with a stream of 322 rows closed by 323, and ``NAMES``
with 353 rows closed by 366 — none of which carry a request id. Nothing in
stock Limnoria collects either (``Channel.nicks`` only reads the state of
channels the bot has joined), so this module owns the pending-query table:
the plugin's ``doNNN`` handlers feed rows in, and a blocked caller (a
threaded @command or an assistant tool handler) waits on the closing
numeric. Deliberately free of Limnoria imports so it is testable in
isolation, the same way ``statuspage`` is.

LIST is one query per network, whatever the caller wanted filtered: the
322 rows do not say which LIST they answer, so two patterns in flight would
interleave. The full list is fetched once, cached for ``cache_ttl``, and
filtered client-side by :func:`format_channels`.
"""

from __future__ import annotations

import fnmatch
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "ChannelRow",
    "IrcQueryRegistry",
    "NamesResult",
    "clean_text",
    "format_channels",
    "format_names",
    "match_channel",
]


@dataclass(frozen=True)
class ChannelRow:
    """One 322 row."""

    name: str
    users: int
    topic: str


@dataclass(frozen=True)
class NamesResult:
    """A closed NAMES query. ``nicks`` keep their @/+/% prefixes."""

    channel: str
    nicks: list[str]
    error: str | None = None


@dataclass
class _Pending:
    done: threading.Event = field(default_factory=threading.Event)
    rows: list = field(default_factory=list)
    error: str | None = None


def _lower(s: str) -> str:
    # RFC 1459 casemapping: {}| are the upper-case forms of []\ .
    return s.lower().translate({ord("["): "{", ord("]"): "}", ord("\\"): "|"})


class IrcQueryRegistry:
    """Pending LIST / NAMES queries, keyed by network (and channel for NAMES).

    Thread-safe: feeders run on the IRC driver thread, waiters on command or
    executor threads. A waiter that times out removes its own pending entry
    so a server that never closes the reply cannot wedge later queries.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        cache_ttl: float = 60.0,
    ) -> None:
        self._clock = clock
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._lists: dict[str, _Pending] = {}
        self._list_cache: dict[str, tuple[float, list[ChannelRow]]] = {}
        self._names: dict[tuple[str, str], _Pending] = {}
        self._errors: dict[tuple[str, str], str] = {}

    # ---------------------------------------------------------------- LIST

    def list_channels(
        self, network: str, send: Callable[[], None], *, timeout: float
    ) -> list[ChannelRow] | None:
        """Return every channel the server lists, or None on timeout/error.

        ``send`` is called exactly once per LIST actually issued; callers
        arriving while one is in flight attach to it instead.
        """
        with self._lock:
            cached = self._list_cache.get(network)
            if cached is not None and self._clock() - cached[0] < self._cache_ttl:
                return list(cached[1])
            pending = self._lists.get(network)
            owner = pending is None
            if owner:
                pending = self._lists[network] = _Pending()
        assert pending is not None
        if owner:
            send()
        if not pending.done.wait(timeout):
            with self._lock:
                if self._lists.get(network) is pending:
                    del self._lists[network]
            return None
        if pending.error is not None:
            return None
        return list(pending.rows)

    def on_list_row(self, network: str, name: str, users: int, topic: str) -> None:
        with self._lock:
            pending = self._lists.get(network)
            if pending is not None:
                pending.rows.append(ChannelRow(name=name, users=users, topic=topic))

    def on_list_end(self, network: str) -> None:
        with self._lock:
            pending = self._lists.pop(network, None)
            if pending is None:
                return
            self._list_cache[network] = (self._clock(), list(pending.rows))
        pending.done.set()

    # --------------------------------------------------------------- NAMES

    def names(
        self, network: str, channel: str, send: Callable[[], None], *, timeout: float
    ) -> NamesResult | None:
        """Return the visible members of ``channel``, or None on timeout."""
        key = (network, _lower(channel))
        with self._lock:
            pending = self._names.get(key)
            owner = pending is None
            if owner:
                pending = self._names[key] = _Pending()
        assert pending is not None
        if owner:
            send()
        if not pending.done.wait(timeout):
            with self._lock:
                if self._names.get(key) is pending:
                    del self._names[key]
            return None
        return NamesResult(channel=channel, nicks=list(pending.rows), error=pending.error)

    def on_names_row(self, network: str, channel: str, nicks: list[str]) -> None:
        with self._lock:
            pending = self._names.get((network, _lower(channel)))
            if pending is not None:
                pending.rows.extend(nicks)

    def on_names_end(self, network: str, channel: str) -> None:
        with self._lock:
            pending = self._names.pop((network, _lower(channel)), None)
        if pending is not None:
            pending.done.set()

    # -------------------------------------------------------------- errors

    def on_error(self, network: str, target: str, text: str) -> None:
        """Close whichever pending query ``target`` names (a channel or LIST)."""
        key = (network, _lower(target))
        with self._lock:
            pending = self._names.pop(key, None)
            if pending is None and target.upper() == "LIST":
                pending = self._lists.pop(network, None)
            if pending is None:
                # Every 401 on the network passes through here (the Network
                # plugin's whois misses included); only remember the ones
                # somebody was waiting on, or this dict grows forever.
                return
            self._errors[key] = text
        pending.error = text
        pending.done.set()

    def last_error(self, network: str, target: str) -> str | None:
        with self._lock:
            return self._errors.get((network, _lower(target)))


# ------------------------------------------------------------- formatting

_CONTROL_RE = re.compile(r"\x03(?:\d{1,2}(?:,\d{1,2})?)?|[\x00-\x1f\x7f]")


def clean_text(text: str, limit: int) -> str:
    """Strip IRC formatting and clip to ``limit`` characters with an ellipsis."""
    text = " ".join(_CONTROL_RE.sub("", text).split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def match_channel(name: str, pattern: str) -> bool:
    """Case-insensitive glob match; a pattern without wildcards is exact."""
    return fnmatch.fnmatchcase(_lower(name), _lower(pattern))


def format_channels(
    rows: list[ChannelRow],
    *,
    pattern: str | None = None,
    min_users: int = 0,
    limit: int = 10,
    topic_chars: int = 60,
) -> str:
    """One IRC line: the busiest matching channels with clipped topics.

    ``pattern`` is a case-insensitive glob; a pattern with no wildcard is an
    exact channel name and answers with that channel alone.
    """
    matched = [r for r in rows if r.users >= min_users]
    if pattern:
        matched = [r for r in matched if match_channel(r.name, pattern)]
        if not matched:
            return f"No channels match {pattern}."
        exact = not any(c in pattern for c in "*?[")
    else:
        exact = False
    matched.sort(key=lambda r: (-r.users, _lower(r.name)))
    shown = matched[:limit]

    def cell(r: ChannelRow) -> str:
        topic = clean_text(r.topic, topic_chars)
        return f"{r.name} ({r.users})" + (f" {topic}" if topic else "")

    body = " | ".join(cell(r) for r in shown)
    if exact:
        return body
    if len(shown) < len(matched):
        return f"{len(shown)} of {len(matched)} channels: {body}"
    noun = "channel" if len(matched) == 1 else "channels"
    return f"{len(matched)} {noun}: {body}"


def format_names(channel: str, nicks: list[str], *, max_chars: int = 400) -> str:
    """One IRC line: ``#chan (N): nick nick …``, clipped with a remainder."""
    if not nicks:
        return f"{channel}: no visible members (empty, secret, or private)."
    head = f"{channel} ({len(nicks)}): "
    shown: list[str] = []
    for i, nick in enumerate(nicks):
        remainder = f" +{len(nicks) - i} more"
        candidate = head + " ".join([*shown, nick])
        if len(candidate) + len(remainder) > max_chars:
            return candidate.removesuffix(" " + nick).rstrip() + remainder
        shown.append(nick)
    return head + " ".join(shown)
