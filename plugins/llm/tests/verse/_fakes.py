"""Shared fakes for loom tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm.verse.loom import LoomCallUsage, VerseSnapshot


class FakeBridge:
    """Synchronous fake. Records every interaction for assertions.

    ``submit()`` runs ``fn()`` inline; ``schedule_after()`` records
    ``(delay, fn, name)`` into ``self.scheduled`` but does not fire —
    tests fire by calling ``bridge.scheduled[i][1]()`` explicitly.
    """

    def __init__(
        self,
        *,
        channels: list[str],
        weights: dict[str, int],
        store: Any,
        snapshots: dict[str, VerseSnapshot],
        post_returns: bool = True,
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

    def list_candidate_channels(self) -> list[str]:
        return list(self.channels)

    def candidate_weight(self, channel: str) -> int:
        return self.weights.get(channel, 0)

    def snapshot(self, channel: str) -> VerseSnapshot:
        return self.snapshots[channel]

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
        return None

    def verse_allow_send(self, channel: str) -> bool:
        return False

    def verse_allow_receive(self, channel: str) -> bool:
        return False


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
