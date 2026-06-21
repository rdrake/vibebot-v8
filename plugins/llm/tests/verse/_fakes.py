"""Shared fakes for loom tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm.verse.compaction import VerseCallUsage as LoomCallUsage
from llm.verse.loom import LoomConfig, VerseSnapshot


def make_loom_config(**overrides: Any) -> LoomConfig:
    """Build a ``LoomConfig`` with the defaults common across the loom tests.

    The defaults match the dominant instantiation in ``test_loom.py``
    (``network="afnet"``, ``model="m"``, ``verse_cooldown_s=1200``,
    ``crosspoll_per_cycle_limit=1``). Each call site overrides only the
    fields it actually varies.
    """
    defaults: dict[str, Any] = {
        "network": "afnet",
        "loom_channel": "#forest",
        "bot_nicks": (),
        "model": "m",
        "cycle_interval_s": 300,
        "verse_cooldown_s": 1200,
        "beat_window_s": 90,
        "transcript_max_lines": 40,
        "transcript_max_chars": 8000,
        "auto_apply_threshold": 0.85,
        "crosspoll_per_cycle_limit": 1,
    }
    defaults.update(overrides)
    return LoomConfig(**defaults)


def make_snapshot(channel: str = "#x", **overrides: Any) -> VerseSnapshot:
    """Build a ``VerseSnapshot`` for *channel* with empty defaults.

    Defaults to ``summary="x"`` with no entities or events; callers pass
    ``summary``/``top_entities``/``recent_events`` only when they vary.
    """
    return VerseSnapshot(
        channel=channel,
        summary=overrides.get("summary", "x"),
        top_entities=overrides.get("top_entities", []),
        recent_events=overrides.get("recent_events", []),
    )


class FakeBridge:
    """Synchronous fake. Records every interaction for assertions.

    ``submit()`` runs ``fn()`` inline; ``schedule_after()`` records
    ``(delay, fn, name)`` into ``self.scheduled`` but does not fire —
    tests fire by calling ``bridge.scheduled[i][1]()`` explicitly.

    The optional keyword params let a single shared fake stand in for the
    many one-off inline fakes the loom tests used to define:

    - ``weight`` forces a constant ``candidate_weight`` (else the per-channel
      ``weights`` map is consulted).
    - ``crosspoll`` is returned from ``crosspoll_store()`` (default ``None``).
      Pass an object that *raises* to exercise the swallow paths.
    - ``crosspoll_raises`` makes ``crosspoll_store()`` itself raise that
      exception, simulating a bridge-construction failure.
    - ``allow_send`` / ``allow_receive`` control the corresponding verse
      permission gates (both default ``False``).
    - ``snapshot_summary`` is the summary used when ``snapshot()`` is asked
      for a channel not present in ``snapshots`` (a fresh empty snapshot is
      synthesised on demand).
    """

    def __init__(
        self,
        *,
        channels: list[str],
        weights: dict[str, int],
        store: Any,
        snapshots: dict[str, VerseSnapshot],
        post_returns: bool = True,
        weight: int | None = None,
        crosspoll: Any | None = None,
        crosspoll_raises: BaseException | None = None,
        allow_send: bool = False,
        allow_receive: bool = False,
        snapshot_summary: str = "x",
    ) -> None:
        self.channels = list(channels)
        self.weights = dict(weights)
        self.store = store
        self.snapshots = dict(snapshots)
        self.posts: list[str] = []
        self.scheduled: list[tuple[float, Callable[[], None], str]] = []
        self.usage_log: list[tuple[str, str, str, LoomCallUsage]] = []
        self.submitted_labels: list[str] = []
        self.t = 1000.0
        self.post_returns = post_returns
        self._weight = weight
        self._crosspoll = crosspoll
        self._crosspoll_raises = crosspoll_raises
        self._allow_send = allow_send
        self._allow_receive = allow_receive
        self._snapshot_summary = snapshot_summary

    def list_candidate_channels(self) -> list[str]:
        return list(self.channels)

    def candidate_weight(self, channel: str) -> int:
        if self._weight is not None:
            return self._weight
        return self.weights.get(channel, 0)

    def snapshot(self, channel: str) -> VerseSnapshot:
        if channel in self.snapshots:
            return self.snapshots[channel]
        return make_snapshot(channel, summary=self._snapshot_summary)

    def post_to_loom_channel(self, text: str) -> bool:
        self.posts.append(text)
        return self.post_returns

    def schedule_after(self, delay_s: float, fn: Callable[[], None], name: str) -> None:
        self.scheduled.append((delay_s, fn, name))

    def submit(self, label: str, fn: Callable[[], None]) -> None:
        self.submitted_labels.append(label)
        fn()

    def now(self) -> float:
        return self.t

    def store_for(self, channel: str) -> Any:
        return self.store

    def log_usage(self, *, channel: str, op: str, model: str, usage: LoomCallUsage) -> None:
        self.usage_log.append((channel, op, model, usage))

    def crosspoll_store(self) -> Any | None:
        if self._crosspoll_raises is not None:
            raise self._crosspoll_raises
        return self._crosspoll

    def verse_allow_send(self, channel: str) -> bool:
        return self._allow_send

    def verse_allow_receive(self, channel: str) -> bool:
        return self._allow_receive


class StubClient:
    """Deterministic ``LoomModelClient`` returning canned replies per op."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = dict(replies)
        self.calls: list[str] = []
        self.last_user_content: str = ""

    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, LoomCallUsage]:
        self.calls.append(op)
        self.last_user_content = messages[-1]["content"]
        return self.replies[op], LoomCallUsage(prompt_tokens=10, completion_tokens=5, cost=0.0001)
