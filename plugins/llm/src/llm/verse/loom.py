"""Forest-verse loom orchestrator: rotation, beats, digest, proposal apply."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, cast


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


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)

_VALID_OPS = ("add_event", "set_attribute", "add_relation", "add_entity")


def _is_strict_int(v: Any) -> bool:
    """Reject bool, accept int. (bool is a subclass of int in Python.)"""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_strict_int(x) for x in v)


_PAYLOAD_SCHEMA: dict[str, tuple[tuple[str, Callable[[Any], bool], str], ...]] = {
    "add_event": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
    "set_attribute": (
        ("entity_id", _is_strict_int, "int"),
        ("key", lambda v: isinstance(v, str), "str"),
        ("value", lambda v: isinstance(v, str), "str"),
    ),
    "add_relation": (
        ("from_id", _is_strict_int, "int"),
        ("to_id", _is_strict_int, "int"),
        ("kind", lambda v: isinstance(v, str), "str"),
    ),
    "add_entity": (
        ("kind", lambda v: isinstance(v, str), "str"),
        ("name", lambda v: isinstance(v, str), "str"),
    ),
}


class ParsedProposal(NamedTuple):
    op: str
    payload: dict[str, Any]
    confidence: float
    provenance: str
    rationale: str


def parse_digest(text: str) -> list[ParsedProposal]:
    """Parse a digest-call response into ParsedProposal instances.

    Strips an optional ``json`` code fence, parses JSON, validates each
    proposal's shape, and drops bad proposals with a warning. Returns
    ``[]`` on hard parse error.
    """
    cleaned = _FENCE_RE.sub("", text).strip()
    log = logging.getLogger("llm.verse.loom")
    try:
        raw = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("loom digest hard parse error: %s", exc)
        return []
    if not isinstance(raw, list):
        log.warning("loom digest top-level was %s, expected list", type(raw).__name__)
        return []

    out: list[ParsedProposal] = []
    for i, raw_item in enumerate(raw):
        if not isinstance(raw_item, dict):
            log.warning("loom proposal %d not a dict; dropped", i)
            continue
        item = cast("dict[str, Any]", raw_item)
        op = item.get("op")
        if op not in _VALID_OPS:
            log.warning("loom proposal %d bad op %r; dropped", i, op)
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            log.warning("loom proposal %d payload not dict; dropped", i)
            continue
        bad_field: str | None = None
        for key, predicate, label in _PAYLOAD_SCHEMA[op]:
            if key not in payload:
                bad_field = f"missing {key}"
                break
            if not predicate(payload[key]):
                bad_field = f"{key} not {label}"
                break
        if bad_field is not None:
            log.warning("loom proposal %d %s; dropped", i, bad_field)
            continue
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        out.append(
            ParsedProposal(
                op=op,
                payload=payload,
                confidence=conf,
                provenance=str(item.get("provenance", "")),
                rationale=str(item.get("rationale", "")),
            )
        )
    return out


def truncate_transcript(
    lines: list[tuple[str, str]],
    *,
    max_lines: int,
    max_chars: int,
) -> list[tuple[str, str]]:
    """Drop consecutive duplicates of the (nick, text) tuple, then cap.

    Caps to ``max_lines`` (most recent kept) and ``max_chars`` (most
    recent kept). Input is oldest-first.
    """
    deduped: list[tuple[str, str]] = []
    for nick, text in lines:
        if deduped and deduped[-1] == (nick, text):
            continue
        deduped.append((nick, text))
    deduped = deduped[-max_lines:]
    out: list[tuple[str, str]] = []
    total = 0
    for nick, text in reversed(deduped):
        if total + len(text) > max_chars:
            break
        out.append((nick, text))
        total += len(text)
    out.reverse()
    return out


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
