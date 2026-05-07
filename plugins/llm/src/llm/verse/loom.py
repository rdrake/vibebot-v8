"""Forest-verse loom orchestrator: rotation, beats, digest, proposal apply."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


class VerseCandidate(NamedTuple):
    channel: str
    weight: int
    """2 * active_avatars + recent_events."""
    last_cycle_at: float | None


class VerseSnapshot(NamedTuple):
    channel: str
    summary: str
    top_entities: list[tuple[str, str]]
    """``(kind, name)`` pairs."""
    recent_events: list[str]
    """Newest-first."""


LOOM_STATIC_PREFIX = """\
You are the loom: a narrator that watches improv between several IRC bots
and proposes mutations to a shared fictional world. Your role is to
*propose*, not to declare canon. A reviewer either approves your proposals
or rejects them.

Each proposal MUST be valid JSON with these fields:
  op          — one of: add_event, set_attribute, add_relation, add_entity
  payload     — object whose required keys depend on op:
                  add_event:     summary (str), entity_ids (list[int])
                  set_attribute: entity_id (int), key (str), value (str)
                  add_relation:  from_id (int), to_id (int), kind (str), note (str?)
                  add_entity:    kind (str: avatar|npc|place|faction|item),
                                 name (str), summary (str?)
  confidence  — float between 0.0 and 1.0
  provenance  — short string identifying which transcript line(s) drove this
  rationale   — one sentence in your voice

Always emit the proposal list as a single JSON array, no prose around it.
"""


def build_verse_stable_block(snap: VerseSnapshot) -> str:
    """Per-cycle prompt block reused across seed/beat/digest calls."""
    parts = [
        f"# Focus verse: {snap.channel}",
        f"# Summary: {snap.summary}",
        "# Active entities:",
    ]
    for kind, name in snap.top_entities:
        parts.append(f"- {kind}: {name}")
    parts.append("# Recent events (newest first):")
    for ev in snap.recent_events:
        parts.append(f"- {ev}")
    return "\n".join(parts)


def build_seed_tail() -> str:
    return (
        "Emit a single line of dialogue or scene-setting that invites the "
        "other bots in this channel to riff on it. Stay in fiction. "
        "One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )


def build_beat_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "The other bots have replied:\n"
        f"{lines}\n\n"
        "Post a single follow-up that picks up a thread or pushes the scene. "
        "One line, ≤ 350 chars. Do NOT emit JSON for this call."
    )


def build_digest_tail(*, loom_transcript_so_far: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"{nick}: {text}" for nick, text in loom_transcript_so_far)
    return (
        "Full transcript:\n"
        f"{lines}\n\n"
        "Now emit a JSON array of proposals derived from this transcript. "
        "If nothing notable happened, emit []."
    )


def pick_focus_verse(
    candidates: list[VerseCandidate],
    *,
    now: float,
    cooldown_s: int,
    pointer: int,
) -> VerseCandidate | None:
    """Highest-weighted candidate outside cooldown; round-robin ties."""
    eligible = [
        c for c in candidates if c.last_cycle_at is None or (now - c.last_cycle_at) >= cooldown_s
    ]
    if not eligible:
        return None
    top_weight = max(c.weight for c in eligible)
    top = [c for c in eligible if c.weight == top_weight]
    return top[pointer % len(top)]


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
