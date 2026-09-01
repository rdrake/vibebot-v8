"""One IRCv3 +typing indicator per (network, target), shared by every holder.

Every path that wants the bot to look busy on a target — a command's planner
turn, a chat reply, the render refresher that covers a two-minute clip — takes
a *hold*. Holds are refcounted per ``(network, target)``: the first sends
``+typing=active``, the last release sends ``+typing=done``, and everything in
between is silent. A single daemon thread re-sends ``active`` for every held
target every ``interval`` seconds, because clients drop the state after about
six seconds without a refresh.

Named groups (``set_group``) let a poller hand over the whole set of targets it
wants held each pass and have the diff computed here, so the render refresher
does not keep its own copy of the state and cannot disagree with the direct
holders about who owns the final ``done``.

The network is part of the key because ``#chan`` on two networks is two rooms.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

Key = tuple[str, str]


class TypingHolds:
    def __init__(
        self,
        send: Callable[[Any, str, str], None],
        log: Any,
        interval: float = 4.0,
    ) -> None:
        self._send = send
        self._log = log
        self._interval = interval
        self._lock = threading.Lock()
        self._counts: dict[Key, int] = {}
        self._ircs: dict[Key, Any] = {}
        self._groups: dict[str, set[Key]] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- public -----------------------------------------------------------

    def hold(self, irc: Any, target: str) -> Callable[[], None]:
        """Take one hold on ``target``; returns an idempotent release."""
        key = (getattr(irc, "network", "") or "", target)
        self._acquire(key, irc)
        released = False
        lock = threading.Lock()

        def release() -> None:
            nonlocal released
            with lock:
                if released:
                    return
                released = True
            self._release(key)

        return release

    def set_group(self, name: str, wanted: dict[Key, Any]) -> None:
        """Make group ``name`` hold exactly ``wanted`` (key -> Irc to send on).

        Keys new to the group are acquired, keys gone from it are released,
        and keys that stay get their Irc replaced so keepalives go out on the
        connection the caller resolved *this* pass, not a zombie from an
        earlier one.
        """
        with self._lock:
            current = self._groups.get(name, set())
        for key in wanted.keys() - current:
            self._acquire(key, wanted[key])
        for key in current - wanted.keys():
            self._release(key)
        with self._lock:
            for key in wanted.keys() & current:
                if key in self._ircs:
                    self._ircs[key] = wanted[key]
            self._groups[name] = set(wanted)

    def holds(self, network: str, target: str) -> bool:
        with self._lock:
            return (network, target) in self._counts

    def active_targets(self) -> set[Key]:
        with self._lock:
            return set(self._counts)

    def stop(self) -> None:
        """Stop the keepalive thread. Sends nothing: callers stop at shutdown,
        when the send path is already closed, and clients expire the state."""
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # -- internals --------------------------------------------------------

    def _acquire(self, key: Key, irc: Any) -> None:
        with self._lock:
            self._ircs[key] = irc
            self._counts[key] = self._counts.get(key, 0) + 1
            first = self._counts[key] == 1
            self._ensure_thread()
        if first:
            self._safe_send(irc, key[1], "active")
            self._wake.set()

    def _release(self, key: Key) -> None:
        with self._lock:
            count = self._counts.get(key, 0)
            if count <= 1:
                self._counts.pop(key, None)
                irc = self._ircs.pop(key, None)
                last = count == 1
            else:
                self._counts[key] = count - 1
                irc = None
                last = False
        if last and irc is not None:
            self._safe_send(irc, key[1], "done")

    def _safe_send(self, irc: Any, target: str, state: str) -> None:
        try:
            self._send(irc, target, state)
        except Exception:
            self._log.exception("typing: %s send failed for %s", state, target)

    def _ensure_thread(self) -> None:
        """Start the keepalive thread on first use. Caller holds ``_lock``."""
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stop.is_set():
            return
        self._thread = threading.Thread(target=self._loop, name="typing-keepalive", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=1.0):
                continue
            while not self._stop.is_set():
                with self._lock:
                    if not self._ircs:
                        self._wake.clear()
                        break
                if self._stop.wait(timeout=self._interval):
                    return
                with self._lock:
                    snapshot = list(self._ircs.items())
                for (_network, target), irc in snapshot:
                    self._safe_send(irc, target, "active")
