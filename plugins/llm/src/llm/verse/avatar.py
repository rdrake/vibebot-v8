"""Avatar shim: wraps @ask for opted-in users, exposes verb-whitelist tools."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from .store import Event, VerseStore
from .validation import validate_payload

_log = logging.getLogger(__name__)

#: Boundary between the cacheable stable prefix (identity + persona + durable
#: CANON roster) and the per-turn volatile scene block in the verse system
#: prompt. Everything before this marker is byte-stable across turns so the
#: LLM prefix cache can hit it; everything after is message-/scene-derived.
VERSE_SCENE_MARKER = "In play right now:"

_MAX_EXEMPLARS = 5
_MAX_EXEMPLAR_CHARS = 600
_STYLE_HEADER = (
    "The channel's sharpest critic singled these lines out as the good stuff — "
    "match this voice and energy; never copy them verbatim:"
)


def _render_style_exemplars(exemplars: Sequence[str]) -> list[str]:
    """Sanitize + cap curated exemplars into prompt lines. Returns [] when empty,
    so a default-empty key leaves the prompt byte-identical."""
    out: list[str] = []
    total = 0
    for ex in exemplars or ():
        s = " ".join(str(ex).split())  # collapse ALL whitespace incl \n\r\t and U+2028/9
        s = "".join(c for c in s if c.isprintable())  # drop zero-width/bidi/control chars
        if not s:
            continue
        if VERSE_SCENE_MARKER in s or s.startswith("Scene:") or s.startswith("- "):
            continue  # never let an exemplar forge prefix structure
        if len(s) > _MAX_EXEMPLAR_CHARS:
            continue  # skip a single oversized exemplar (keep the rest)
        if total + len(s) > _MAX_EXEMPLAR_CHARS:
            break
        out.append(f"- {s}")
        total += len(s)
        if len(out) >= _MAX_EXEMPLARS:
            break
    return [_STYLE_HEADER, *out] if out else []


@dataclass(frozen=True)
class VerseDispatchResult:
    """Structured result of a verse tool dispatch.

    The mutation tools (verse_act / verse_move) return ok=True with
    payload={'status': 'ok'}. The read tools (verse_look / verse_recall)
    additionally carry their data ('description' / 'events') so the model
    can use what it asked for. verse_record populates payload with
    tool-specific data on success or error with a model-facing string on
    failure.
    """

    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None


def make_verse_tool_specs(*, max_actors: int = 8, storybook: bool = False) -> list[dict]:
    """Return OpenAI/LiteLLM tool specs for the five verse tools.

    The tools are model-callable but only meaningful when the @ask path
    is verse-routed (see plugin._verse_route_for + C7d dispatch).

    ``max_actors`` controls the JSON-schema ``maxItems`` on
    ``verse_record.actors`` so the model is told the per-call cap up
    front (the dispatch branch also enforces it server-side).
    """
    specs: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "verse_act",
                "description": (
                    "Record an in-character action. The whitelist of verbs "
                    "with side effects is: move, flee, follow (relocate), "
                    "take, drop, give (item refs), and event-only verbs "
                    "whisper, speak, listen, examine, wait, signal, gesture, "
                    "search. Off-list verbs are recorded as events with no "
                    "world change."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verb": {"type": "string"},
                        "target": {"type": "string"},
                        "details": {"type": "string"},
                    },
                    "required": ["verb"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verse_move",
                "description": "Move the avatar to a named place.",
                "parameters": {
                    "type": "object",
                    "properties": {"place_name": {"type": "string"}},
                    "required": ["place_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verse_look",
                "description": (
                    "Describe an entity in the scene. With no target, "
                    "describes the avatar's current location."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verse_recall",
                "description": (
                    "Recall up to 5 recent events whose summaries match any token of the query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verse_record",
                "description": (
                    "Record a NEW in-world event involving one or more "
                    "named actors. Use whenever a member narrates a NEW "
                    "event happening right now that isn't strictly about "
                    'their own avatar (e.g. "stinky dan threw a guff '
                    'grenade at Andrew" — record actors=["stinky dan",'
                    '"Andrew"], the grenade stays in the summary as '
                    "prose). Do NOT use this tool for retellings, recall "
                    "queries, or to answer 'what happened at X' / 'tell "
                    "me about Y' / 'remember when Z' style questions — "
                    "use verse_recall to look up past summaries instead, "
                    "and put the retelling in your reply text. Names "
                    "that don't match an existing entity are "
                    "auto-created as kind=npc. Items, places, and "
                    "weapons are NOT actors — only put characters/"
                    "people in the actors list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": (
                                "What happened, in past tense, ≤200 chars. "
                                "The full prose narration including any "
                                "items, places, or weapons mentioned. e.g. "
                                "'stinky dan threw a guff grenade at Andrew'."
                            ),
                        },
                        "actors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": max_actors,
                            "description": (
                                "Names of CHARACTERS (people/npcs) central "
                                "to the event. Do NOT include items, "
                                "weapons, places, or abstractions."
                            ),
                        },
                    },
                    "required": ["summary"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verse_edit",
                "description": (
                    "Create or modify forest-verse canon (entities, "
                    "attributes, relations, events). Requires the caller to "
                    "hold llm.verse.edit; calls from anyone else are refused. "
                    "Constructive ops only — you cannot delete or retire "
                    "anything with this tool. ``op`` selects the kind of "
                    "change; ``payload`` carries its fields (add_entity: kind, "
                    "name, summary?; add_event: summary, entity_ids; "
                    "set_attribute: entity_id, key, value; add_relation: "
                    "from_id, to_id, kind, note?; update_entity: entity_id "
                    "plus name and/or summary). Reuse existing entity ids from "
                    "the roster; do not invent ids."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": [
                                "add_entity",
                                "add_event",
                                "set_attribute",
                                "add_relation",
                                "update_entity",
                            ],
                        },
                        "payload": {"type": "object"},
                    },
                    "required": ["op", "payload"],
                },
            },
        },
    ]
    if storybook:
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": "verse_storybook",
                    "description": (
                        "Create a short ILLUSTRATED story page (prose plus AI-drawn "
                        "pictures) and return a link to share in character. Reach for "
                        "this liberally — any time someone wants a story, tale, saga, "
                        "myth, legend, recap, or 'tell me about...', or whenever a "
                        "scene could be richer with pictures. When in doubt, use it: a "
                        "linked illustrated page beats a wall of inline text, and only "
                        "this tool can draw the pictures (it illustrates generously, "
                        "several per story)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "brief": {
                                "type": "string",
                                "description": "What the story should be about.",
                            },
                        },
                        "required": ["brief"],
                    },
                },
            }
        )
    return specs


class VerbEffect(Enum):
    EVENT_ONLY = "event_only"
    MOVE = "move"
    ITEM = "item"


VERB_TABLE: dict[str, VerbEffect] = {
    "whisper": VerbEffect.EVENT_ONLY,
    "speak": VerbEffect.EVENT_ONLY,
    "listen": VerbEffect.EVENT_ONLY,
    "examine": VerbEffect.EVENT_ONLY,
    "wait": VerbEffect.EVENT_ONLY,
    "signal": VerbEffect.EVENT_ONLY,
    "gesture": VerbEffect.EVENT_ONLY,
    "search": VerbEffect.EVENT_ONLY,
    "move": VerbEffect.MOVE,
    "flee": VerbEffect.MOVE,
    "follow": VerbEffect.MOVE,
    "take": VerbEffect.ITEM,
    "drop": VerbEffect.ITEM,
    "give": VerbEffect.ITEM,
}


class ActResult(NamedTuple):
    event_id: int
    scene_shift_text: str


def _third_person(verb: str) -> str:
    """Naive third-person-singular: 'move' -> 'moves', 'search' -> 'searches'."""
    return verb + ("es" if verb.endswith(("s", "sh", "ch", "x", "z")) else "s")


def verse_act(
    store: VerseStore,
    avatar_id: int,
    verb: str,
    target: str | None = None,
    details: str | None = None,
) -> ActResult:
    """Execute a verse action for an avatar and return the result.

    Raises ValueError if the avatar is retired or does not exist.
    ``details`` is accepted for forward compatibility but unused in v1.
    """
    _ = details  # unused in v1; forward-compat parameter

    # 1. Retired-avatar guard
    avatar = store.get_entity(avatar_id)
    if avatar is None or avatar.status == "retired":
        raise ValueError("avatar retired")

    # 2. Verb classification
    effect = VERB_TABLE.get(verb.lower())

    # 3. MOVE
    if effect is VerbEffect.MOVE:
        place = (
            store.find_entity_by_name(target or "", kind="place", active_only=True)
            if target
            else None
        )

        if place is None and target is not None:
            # Try resolving target as another avatar's name, then use their location
            other = store.find_entity_by_name(target, kind="avatar", active_only=True)
            if other is not None:
                loc = store.get_attribute(other.id, "location")
                if loc is not None:
                    place = store.find_entity_by_name(loc, kind="place", active_only=True)

        if place is not None:
            store.set_attribute(avatar_id, "location", place.name)
            event_id = store.add_event(
                summary=f"{avatar.name} {_third_person(verb)} to {place.name}",
                entity_ids=[avatar_id, place.id],
                source="avatar",
            )
            return ActResult(event_id, f"You {verb} to {place.name}.")
        else:
            event_id = store.add_event(
                summary=f"{avatar.name} tries to {verb} to {target}",
                entity_ids=[avatar_id],
                source="avatar",
            )
            return ActResult(event_id, "You can't find that place.")

    # 4. ITEM
    if effect is VerbEffect.ITEM:
        item = (
            store.find_entity_by_name(target or "", kind="item", active_only=True)
            if target
            else None
        )

        if item is not None:
            event_id = store.add_event(
                summary=f"{avatar.name} {_third_person(verb)} {item.name}",
                entity_ids=[avatar_id, item.id],
                source="avatar",
            )
            return ActResult(event_id, f"You {verb} the {item.name}.")
        else:
            event_id = store.add_event(
                summary=f"{avatar.name} tries to {verb} {target}",
                entity_ids=[avatar_id],
                source="avatar",
            )
            return ActResult(event_id, f"You can't find any {target}.")

    # 5. EVENT_ONLY
    if effect is VerbEffect.EVENT_ONLY:
        summary = f"{avatar.name} {_third_person(verb)}" + (f" {target}" if target else "")
        event_id = store.add_event(
            summary=summary,
            entity_ids=[avatar_id],
            source="avatar",
        )
        scene = f"You {verb}" + (f" {target}." if target else ".")
        return ActResult(event_id, scene)

    # 6. Off-list verb
    summary = f"{avatar.name} {_third_person(verb)}" + (f" {target}" if target else "")
    event_id = store.add_event(
        summary=summary,
        entity_ids=[avatar_id],
        source="avatar",
    )
    scene = f"You attempt to {verb}" + (f" {target}." if target else ".")
    return ActResult(event_id, scene)


# ---------------------------------------------------------------------------
# Verse navigation / info helpers
# ---------------------------------------------------------------------------


def verse_move(store: VerseStore, avatar_id: int, place_name: str) -> str:
    """Move the avatar to the given place by name.

    Returns the canonical place name on success.
    Raises ValueError("avatar retired") if the avatar is retired or missing,
    or ValueError("no such place") if no active place matches.
    """
    # Retired-avatar guard before any write (parity with verse_act, #26).
    avatar = store.get_entity(avatar_id)
    if avatar is None or avatar.status == "retired":
        raise ValueError("avatar retired")
    place = store.find_entity_by_name(place_name, kind="place", active_only=True)
    if place is None:
        raise ValueError("no such place")
    store.set_attribute(avatar_id, "location", place.name)
    return place.name


def verse_look(
    store: VerseStore,
    avatar_id: int,
    target: str | None = None,
) -> str | None:
    """Return summary text for the avatar's current location or a named entity.

    When target is None: return the avatar's current location's summary
    (None if avatar has no location set or the named place doesn't exist).
    When target is given: return that entity's summary (case-insensitive
    name match across all kinds), or None if not found.

    Both lookups are active-only (parity with verse_move/verse_act): a
    retired entity is not part of the scene, so verse_look must not
    describe it while the system prompt says the avatar is "nowhere".
    """
    if target is None:
        location = store.get_attribute(avatar_id, "location")
        if location is None:
            return None
        place = store.find_entity_by_name(location, kind="place", active_only=True)
        return place.summary if place is not None else None
    else:
        entity = store.find_entity_by_name(target, active_only=True)
        return entity.summary if entity is not None else None


def verse_recall(store: VerseStore, query: str) -> list[Event]:
    """Return up to 5 recent events whose summary contains any whitespace-split
    token of ``query`` (case-insensitive substring match). Newest-first."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []
    events = store.recent_events(limit=100, require_active_entity=True)
    filtered = [
        event for event in events if any(token in event.summary.lower() for token in tokens)
    ]
    return filtered[:5]


# ---------------------------------------------------------------------------
# OOC escape detector
# ---------------------------------------------------------------------------

OOC_PREFIX = "(("
OOC_SUFFIX = "))"
# Ergonomic single-message OOC marker: a leading ``//`` means "plain
# question, no in-character routing" — easier to type than wrapping a
# whole message in ((double parens)).
OOC_LINE_PREFIX = "//"


def is_ooc(message: str) -> bool:
    """True if ``message`` is an OOC (out-of-character) aside.

    Two equivalent forms are recognised, both tolerating surrounding
    whitespace:

    - ``((wrapped like this))`` — BOTH prefix and suffix present. An empty
      wrapper "(())" returns True (syntactically OOC, even if useless).
    - ``// leading-slash form`` — a leading ``//``. A bare ``//`` returns
      True, mirroring the empty-wrapper case.
    """
    s = message.strip()
    if s.startswith(OOC_LINE_PREFIX):
        return True
    return s.startswith(OOC_PREFIX) and s.endswith(OOC_SUFFIX)


def strip_ooc(message: str) -> str:
    """Return the inner text of an OOC aside, marker removed.

    Strips the ``((``/``))`` wrapper or the leading ``//`` (whichever
    applies) plus surrounding whitespace. A degenerate ``(())`` or bare
    ``//`` yields an empty string. A message that is not OOC-marked is
    returned stripped but otherwise unchanged, so the call is safe even
    when ``is_ooc`` was not checked first.
    """
    s = message.strip()
    if s.startswith(OOC_LINE_PREFIX):
        return s[len(OOC_LINE_PREFIX) :].strip()
    if not (s.startswith(OOC_PREFIX) and s.endswith(OOC_SUFFIX)):
        return s
    return s[len(OOC_PREFIX) : -len(OOC_SUFFIX)].strip()


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def build_verse_system_prompt(
    store: VerseStore,
    avatar_id: int,
    instruct_text: str,
    roster_max_chars: int = 4000,
    message_text: str = "",
    *,
    style_exemplars: Sequence[str] = (),
) -> str:
    """Build the verse-aware @ask system prompt, STABLE-FIRST for prefix caching.

    Cacheable prefix (changes only on explicit author/operator action):
    - "You are <avatar.name>."
    - "Persona: <instruct_text>" (or "Persona: no persona set.")
    - "Established characters in this world:" + the durable CANON roster
      (pinned OR author_locked), char-capped.
    Then VERSE_SCENE_MARKER, after which everything is per-turn / message-derived:
    scene/location, active-only recent events involving the avatar, co-located
    avatars, message-matched cast (not already in the roster), their 1-hop
    relations, and their recent events. Nothing time/heartbeat-derived appears
    before the marker, so the prefix stays byte-stable across turns.
    """
    avatar = store.get_entity(avatar_id)
    if avatar is None:
        raise ValueError("avatar not found")

    # ===== STABLE PREFIX (cacheable across turns) =====
    identity_line = f"You are {avatar.name}."
    persona_line = (
        f"Persona: {instruct_text}" if instruct_text.strip() else "Persona: no persona set."
    )

    canon = store.list_canon_entities()
    roster_lines: list[str] = []
    if canon:
        used = 0
        for e in canon:
            line = f"- {e.name}: {e.summary}" if e.summary else f"- {e.name}"
            if used + len(line) + 1 > roster_max_chars:
                roster_lines.append("- (roster truncated)")
                break
            roster_lines.append(line)
            used += len(line) + 1

    parts: list[str] = [identity_line, persona_line]
    if roster_lines:
        parts.append("Established characters in this world:")
        parts.extend(roster_lines)

    parts.extend(_render_style_exemplars(style_exemplars))  # static, cacheable

    # ===== VOLATILE SCENE BLOCK (per-turn; not in the cached prefix) =====
    parts.append(VERSE_SCENE_MARKER)

    location = store.get_attribute(avatar_id, "location")
    place = (
        store.find_entity_by_name(location, kind="place", active_only=True)
        if location is not None
        else None
    )
    parts.append(
        f"Scene: You are at {place.name}. {place.summary}"
        if place is not None
        else "Scene: You are nowhere in particular."
    )

    own = [
        ev
        for ev in store.recent_events(limit=50, require_active_entity=True)
        if avatar_id in ev.entity_ids
    ][:5]
    parts.append("Recent events involving you:")
    parts.extend([f"- {ev.summary}" for ev in own] or ["- (none yet)"])

    others = []
    if place is not None:
        for a in store.list_entities_by_kind("avatar", status="active"):
            if a.id != avatar_id and store.get_attribute(a.id, "location") == location:
                others.append(a)
    parts.append("Other avatars present here:")
    parts.extend(
        [f"- {a.name}: {a.summary}" if a.summary else f"- {a.name}" for a in others]
        or ["- (no other avatars present)"]
    )

    roster_ids = {e.id for e in canon}
    scene = [e for e in store.match_entities_in_text(message_text) if e.id != avatar_id]
    fresh = [e for e in scene if e.id not in roster_ids]
    if fresh:
        parts.append("Characters referenced in this scene:")
        parts.extend([f"- {e.name}: {e.summary}" if e.summary else f"- {e.name}" for e in fresh])

    rel_ids = list(roster_ids | {e.id for e in scene} | {avatar_id})
    rels = store.relations_for(rel_ids)
    if rels:
        parts.append("Known relationships:")
        parts.extend(
            [
                f"- {r.from_name} {r.kind.replace('_', ' ')} {r.to_name}"
                + (f" ({r.note})" if r.note else "")
                for r in rels
            ]
        )

    scene_events = store.events_for_entities([e.id for e in scene], limit=8)
    if scene_events:
        parts.append("Recent events involving them:")
        parts.extend([f"- {ev.summary}" for ev in scene_events])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# verse_edit: gated canon-editing tool
# ---------------------------------------------------------------------------

#: Constructive ops the verse_edit tool may apply. Destructive / lifecycle ops
#: (delete_event, delete_relation, set_status, set_pinned, edit_event) are
#: deliberately excluded — verse_edit grows canon, it does not tear it down.
_VERSE_EDIT_OPS = frozenset(
    {"add_entity", "add_event", "set_attribute", "add_relation", "update_entity"}
)


def dispatch_verse_edit(store, *, op, payload, authorized, account):
    """Execute a verse_edit tool call. Constructive ops only; gated.

    Returns a JSON-able dict: {status: ok|refused|error, ...}.

    ``authorized`` MUST reflect whether the user who triggered this verse
    turn holds the ``llm.verse.edit`` capability; the caller computes it.
    An unauthorized call is a no-op (nothing is written to the store).
    """
    if not authorized:
        return {
            "status": "refused",
            "detail": "not authorized to edit canon (needs llm.verse.edit)",
        }
    if op not in _VERSE_EDIT_OPS:
        return {"status": "error", "detail": f"op {op!r} not permitted via verse_edit"}
    reason = validate_payload(op, payload)
    if reason is not None:
        return {"status": "error", "detail": reason}
    try:
        new_id = store.apply_direct(
            op=op, payload=payload, source="llm", provenance=f"verse_edit:{account}"
        )
    except (LookupError, ValueError, PermissionError) as exc:
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "id": new_id}


# ---------------------------------------------------------------------------
# C7d: verse tool dispatch
# ---------------------------------------------------------------------------


#: ToolResult-like namedtuple for verse handlers returning to assistant loop.
#: Mirrors the shape expected by assistant_request's extra_handlers callables.
class _VerseToolResult(NamedTuple):
    content: str


def dispatch_verse_tool_call(
    store: VerseStore,
    avatar_id: int,
    name: str,
    args: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
) -> VerseDispatchResult:
    """Dispatch a single verse tool call, swallowing all exceptions.

    Applies the side effect (event write, attribute update, etc.) to
    ``store``.  If the tool raises for any reason — including retired avatar,
    OperationalError (DB deleted mid-session), or missing required args —
    logs at WARNING and returns without raising.  Call order is the
    caller's responsibility.

    This function is the core of C7d's failure-handling contract:
    - B2-level business failures (bad move target) are handled INSIDE
      verse_act and return normally (event row written).
    - Exception-level failures (retired avatar, DB gone) are caught here
      and logged at WARNING; no event row is written.

    Returns a :class:`VerseDispatchResult`. The mutation tools (verse_act /
    verse_move) return ``ok=True`` with ``payload={'status': 'ok'}``, and
    their failures keep the historical swallow-and-skip semantics (logged,
    ``ok=True``). The read tools now return their data to the model —
    verse_look carries ``description`` and verse_recall carries ``events``
    (≤5, summary + ts) — instead of computing it and dropping it.
    ``verse_record`` may return ``ok=False`` with a model-facing ``error``
    string.
    """
    log = logger or _log
    _ok = VerseDispatchResult(ok=True, payload={"status": "ok"})
    try:
        if name == "verse_act":
            verb = args.get("verb")
            if not verb:
                log.warning("verse_act missing 'verb' arg (avatar=%s)", avatar_id)
                return _ok
            verse_act(store, avatar_id, verb, args.get("target"), args.get("details"))
            return _ok
        elif name == "verse_move":
            place = args.get("place_name")
            if not place:
                log.warning("verse_move missing 'place_name' arg (avatar=%s)", avatar_id)
                return _ok
            verse_move(store, avatar_id, place)
            return _ok
        elif name == "verse_look":
            description = verse_look(store, avatar_id, args.get("target"))
            return VerseDispatchResult(
                ok=True, payload={"status": "ok", "description": description}
            )
        elif name == "verse_recall":
            q = args.get("query")
            if q is None:
                log.warning("verse_recall missing 'query' arg (avatar=%s)", avatar_id)
                return _ok
            events = verse_recall(store, q)
            return VerseDispatchResult(
                ok=True,
                payload={
                    "status": "ok",
                    "events": [{"summary": ev.summary, "ts": ev.ts} for ev in events],
                },
            )
        elif name == "verse_record":
            return _dispatch_verse_record(store, avatar_id, args, log=log)
        else:
            log.warning("unknown verse tool: %s (avatar=%s)", name, avatar_id)
            return _ok
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "verse tool dispatch failed: name=%s avatar=%s err=%s",
            name,
            avatar_id,
            exc,
        )
        return _ok


def _dispatch_verse_record(
    store: VerseStore,
    avatar_id: int,
    args: dict[str, Any],
    *,
    log: logging.Logger,
) -> VerseDispatchResult:
    """Validate verse_record args and call store.record_user_event.

    Returns ok=False with a model-facing error string for empty / too-long
    summaries or non-list actors. The actors list is filtered (drop
    non-strings and empty / whitespace-only entries) BEFORE slicing to
    ``_max_actors``, so a payload like ``["alice", 42, "bob"]`` with
    max=2 yields ``["alice", "bob"]`` rather than ``["alice"]``.
    """
    _ = log  # currently unused; kept for symmetry with other branches
    # Retired-avatar guard (mirrors verse_act): fail with a model-facing error
    # instead of letting record_user_event raise into the broad dispatch except,
    # which would silently swallow it. Also closes the TOCTOU window where the
    # avatar retires between dispatch and the write.
    avatar = store.get_entity(avatar_id)
    if avatar is None or avatar.status == "retired":
        return VerseDispatchResult(ok=False, error="avatar retired")
    summary = (args.get("summary") or "").strip()
    if not summary:
        return VerseDispatchResult(ok=False, error="summary required")
    if len(summary) > 200:
        return VerseDispatchResult(
            ok=False,
            error=f"summary too long: {len(summary)} chars (max 200)",
        )
    raw = args.get("actors") or []
    if not isinstance(raw, list):
        return VerseDispatchResult(ok=False, error="actors must be an array")
    max_actors = args.get("_max_actors", 8)
    cleaned = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    actors = cleaned[:max_actors]
    event_id = store.record_user_event(
        actor_id=avatar_id,
        summary=summary,
        actor_names=actors,
        now=time.time,
    )
    return VerseDispatchResult(ok=True, payload={"status": "ok", "event_id": event_id})


def make_verse_extra_handlers(
    store: VerseStore,
    avatar_id: int,
    logger: logging.Logger | None = None,
    *,
    max_actors: int = 8,
) -> dict[str, Callable[[dict[str, Any]], Any]]:
    """Return an extra_handlers dict for the five verse tools.

    Each handler calls ``dispatch_verse_tool_call`` (which swallows failures)
    then returns a JSON-encoded ToolResult-compatible object so the
    assistant_request loop can continue normally.  The return value uses
    ``_VerseToolResult`` which has a ``content`` attribute matching the
    ``ToolResult`` duck-type expected by the loop.

    ``max_actors`` is the per-call cap on ``verse_record.actors`` after
    filtering; it travels into the dispatch closure via ``_max_actors``
    in the args dict. The handler always overwrites the key — the args
    dict is model-controlled, so a tool call passing its own
    ``_max_actors`` must not be able to raise the server-side cap.
    (Tests that call ``dispatch_verse_tool_call`` directly may still pass
    ``_max_actors`` themselves.)
    """
    log = logger or _log
    _verse_names = {
        "verse_act",
        "verse_move",
        "verse_look",
        "verse_recall",
        "verse_record",
    }

    def _handler(name: str) -> Callable[[dict[str, Any]], _VerseToolResult]:
        def _call(args: dict[str, Any]) -> _VerseToolResult:
            args = dict(args)
            args["_max_actors"] = max_actors
            result = dispatch_verse_tool_call(store, avatar_id, name, args, logger=log)
            if result.ok:
                payload: dict[str, Any] = {"status": "ok", "tool": name}
                if result.payload:
                    payload.update(result.payload)
                return _VerseToolResult(content=json.dumps(payload))
            return _VerseToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "error": result.error or "unknown error",
                        "tool": name,
                    }
                )
            )

        _call.__name__ = f"_verse_handler_{name}"
        return _call

    return {name: _handler(name) for name in _verse_names}


def make_verse_denial_handlers(
    verse_tool_specs: list[dict],
) -> dict[str, Callable[[dict[str, Any]], _VerseToolResult]]:
    """Return handlers that reject every verse tool call.

    Used when verse tool *schemas* are advertised to the model on a
    verse-enabled channel for cache stability (so the channel's tool
    bytes don't vary with per-user opt-in state), but the caller hasn't
    actually joined the verse and any invocation must be rejected.

    Each handler returns ``{"error": ...}`` with the same channel-level
    onboarding hint so the model can self-correct and tell the user how
    to opt in.
    """
    message = (
        "You haven't joined the forest-verse on this channel. Tell the "
        "speaker to use @verse opt-in <persona> to participate before "
        "calling this tool again."
    )
    payload = json.dumps({"error": message})

    def _denied(_args: dict[str, Any]) -> _VerseToolResult:
        return _VerseToolResult(content=payload)

    names: list[str] = []
    for spec in verse_tool_specs:
        try:
            names.append(spec["function"]["name"])
        except (KeyError, TypeError):
            continue
    return dict.fromkeys(names, _denied)
