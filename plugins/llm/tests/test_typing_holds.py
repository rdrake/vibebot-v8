"""TypingHolds: one +typing indicator per (network, target), refcounted."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest
from llm.typing_holds import TypingHolds


class _Recorder:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, irc, target: str, state: str) -> None:
        with self.lock:
            self.sent.append((irc.network, target, state))

    def states(self, network: str, target: str) -> list[str]:
        with self.lock:
            return [s for n, t, s in self.sent if n == network and t == target]


@pytest.fixture
def rec() -> _Recorder:
    return _Recorder()


@pytest.fixture
def holds(rec: _Recorder) -> TypingHolds:
    h = TypingHolds(rec, logging.getLogger("test"), interval=0.05)
    yield h
    h.stop()


def _irc(network: str = "afternet") -> SimpleNamespace:
    return SimpleNamespace(network=network)


class TestRefcount:
    def test_first_hold_sends_active_and_last_release_sends_done(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        assert rec.states("afternet", "#chan") == ["active"]
        release()
        assert rec.states("afternet", "#chan") == ["active", "done"]

    def test_second_hold_on_same_target_sends_nothing(self, holds, rec) -> None:
        r1 = holds.hold(_irc(), "#chan")
        r2 = holds.hold(_irc(), "#chan")
        assert rec.states("afternet", "#chan") == ["active"]
        r1()
        assert rec.states("afternet", "#chan") == ["active"], "done must wait for the last holder"
        r2()
        assert rec.states("afternet", "#chan") == ["active", "done"]

    def test_release_is_idempotent(self, holds, rec) -> None:
        r1 = holds.hold(_irc(), "#chan")
        r2 = holds.hold(_irc(), "#chan")
        r1()
        r1()
        r1()
        assert holds.holds("afternet", "#chan")
        r2()
        assert not holds.holds("afternet", "#chan")
        assert rec.states("afternet", "#chan").count("done") == 1

    def test_same_channel_on_two_networks_is_two_indicators(self, holds, rec) -> None:
        ra = holds.hold(_irc("afternet"), "#chan")
        rb = holds.hold(_irc("other"), "#chan")
        assert holds.active_targets() == {("afternet", "#chan"), ("other", "#chan")}
        ra()
        assert rec.states("afternet", "#chan") == ["active", "done"]
        assert rec.states("other", "#chan") == ["active"]
        rb()


class TestKeepalive:
    def test_active_is_resent_while_held(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        time.sleep(0.3)
        release()
        assert rec.states("afternet", "#chan").count("active") >= 2
        assert rec.states("afternet", "#chan")[-1] == "done"

    def test_nothing_is_sent_after_release(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        release()
        n = len(rec.states("afternet", "#chan"))
        time.sleep(0.15)
        assert len(rec.states("afternet", "#chan")) == n

    def test_send_failure_does_not_kill_the_loop(self, rec) -> None:
        calls = {"n": 0}

        def flaky(irc, target, state):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            rec(irc, target, state)

        h = TypingHolds(flaky, logging.getLogger("test"), interval=0.02)
        try:
            release = h.hold(_irc(), "#chan")
            time.sleep(0.15)
            release()
        finally:
            h.stop()
        assert rec.states("afternet", "#chan").count("active") >= 2


class TestGroups:
    def test_set_group_acquires_new_and_releases_missing(self, holds, rec) -> None:
        a, b = _irc(), _irc()
        holds.set_group("render", {("afternet", "#a"): a, ("afternet", "#b"): b})
        assert rec.states("afternet", "#a") == ["active"]
        assert rec.states("afternet", "#b") == ["active"]
        holds.set_group("render", {("afternet", "#b"): b})
        assert rec.states("afternet", "#a") == ["active", "done"]
        assert rec.states("afternet", "#b") == ["active"]
        holds.set_group("render", {})
        assert rec.states("afternet", "#b") == ["active", "done"]

    def test_group_and_direct_hold_share_one_indicator(self, holds, rec) -> None:
        release = holds.hold(_irc(), "#chan")
        holds.set_group("render", {("afternet", "#chan"): _irc()})
        release()
        assert rec.states("afternet", "#chan") == ["active"], "render still holds it"
        holds.set_group("render", {})
        assert rec.states("afternet", "#chan") == ["active", "done"]

    def test_set_group_refreshes_the_irc_used_for_keepalive(self, holds, rec) -> None:
        stale = SimpleNamespace(network="afternet", tag="stale")
        fresh = SimpleNamespace(network="afternet", tag="fresh")
        seen: list[str] = []
        holds._send = lambda irc, target, state: (
            seen.append(getattr(irc, "tag", "?")),
            rec(irc, target, state),
        )
        holds.set_group("render", {("afternet", "#chan"): stale})
        holds.set_group("render", {("afternet", "#chan"): fresh})
        time.sleep(0.15)
        holds.set_group("render", {})
        assert "fresh" in seen[1:], "keepalives after the second pass use the fresh Irc"


class TestStop:
    def test_stop_ends_the_thread_and_sends_nothing_more(self, rec) -> None:
        h = TypingHolds(rec, logging.getLogger("test"), interval=0.02)
        h.hold(_irc(), "#chan")
        h.stop()
        n = len(rec.sent)
        time.sleep(0.1)
        assert len(rec.sent) == n
        assert not any(t.name == "typing-keepalive" and t.is_alive() for t in threading.enumerate())

    def test_a_hold_taken_after_stop_does_not_restart_the_thread(self, rec) -> None:
        """A worker finishing after die() must not resurrect the keepalive."""
        h = TypingHolds(rec, logging.getLogger("test"), interval=0.02)
        h.stop()

        h.hold(_irc(), "#chan")

        assert not any(t.name == "typing-keepalive" and t.is_alive() for t in threading.enumerate())
