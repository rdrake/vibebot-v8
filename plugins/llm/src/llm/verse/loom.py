"""Forest-verse loom orchestrator: rotation, beats, digest, proposal apply."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoomConfig:
    """All registry-derived knobs the loom needs for one cycle."""

    network: str
    loom_channel: str
    bot_nicks: tuple[str, ...]
    """Empty tuple = capture all non-self lines (bot-heavy channel default)."""
    model: str
    cycle_interval_s: int
    verse_cooldown_s: int
    beat_window_s: int
    transcript_max_lines: int
    transcript_max_chars: int
    auto_apply_threshold: float
