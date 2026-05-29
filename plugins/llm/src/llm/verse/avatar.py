"""Avatar shim: wraps @ask for opted-in users, exposes verb-whitelist tools."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from .store import Event, VerseStore

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerseDispatchResult:
    """Structured result of a verse tool dispatch.

    The four legacy tools (verse_act / verse_move / verse_look / verse_recall)
    return ok=True with payload={'status': 'ok'}, preserving the wrapper's
    historical observable JSON. New branches (verse_record) populate
    payload with tool-specific data on success or error with a model-
    facing string on failure.
    """

    ok: bool
    payload: dict[str, Any] | None = None
    error: str | None = None


def make_verse_tool_specs(*, max_actors: int = 8) -> list[dict]:
    """Return OpenAI/LiteLLM tool specs for the five verse tools.

    The tools are model-callable but only meaningful when the @ask path
    is verse-routed (see plugin._verse_route_for + C7d dispatch).

    ``max_actors`` controls the JSON-schema ``maxItems`` on
    ``verse_record.actors`` so the model is told the per-call cap up
    front (the dispatch branch also enforces it server-side).
    """
    return [
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
    ]


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
                summary=f"{avatar.name} {verb}s to {place.name}",
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
                summary=f"{avatar.name} {verb}s {item.name}",
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
        summary = f"{avatar.name} {verb}s" + (f" {target}" if target else "")
        event_id = store.add_event(
            summary=summary,
            entity_ids=[avatar_id],
            source="avatar",
        )
        scene = f"You {verb}" + (f" {target}." if target else ".")
        return ActResult(event_id, scene)

    # 6. Off-list verb
    summary = f"{avatar.name} {verb}s" + (f" {target}" if target else "")
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
    Raises ValueError("no such place") if no active place matches.
    """
    place = store.find_entity_by_name(place_name, kind="place")
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
    """
    if target is None:
        location = store.get_attribute(avatar_id, "location")
        if location is None:
            return None
        place = store.find_entity_by_name(location, kind="place")
        return place.summary if place is not None else None
    else:
        entity = store.find_entity_by_name(target)
        return entity.summary if entity is not None else None


def verse_recall(store: VerseStore, query: str) -> list[Event]:
    """Return up to 5 recent events whose summary contains any whitespace-split
    token of ``query`` (case-insensitive substring match). Newest-first."""
    tokens = [t for t in query.lower().split() if t]
    if not tokens:
        return []
    events = store.recent_events(limit=100)
    filtered = [
        event for event in events if any(token in event.summary.lower() for token in tokens)
    ]
    return filtered[:5]


# ---------------------------------------------------------------------------
# OOC escape detector
# ---------------------------------------------------------------------------

OOC_PREFIX = "(("
OOC_SUFFIX = "))"


def is_ooc(message: str) -> bool:
    """True if ``message`` is wrapped in OOC parentheses ((like this)).

    Whitespace around the wrapper is tolerated, but BOTH the prefix and
    suffix must be present. An empty wrapper "(())" returns True (it's
    syntactically OOC, even if useless).
    """
    s = message.strip()
    return s.startswith(OOC_PREFIX) and s.endswith(OOC_SUFFIX)


def strip_ooc(message: str) -> str:
    """Return the inner text of an OOC-wrapped message, parentheses removed.

    Strips the outer ``((``/``))`` wrapper and surrounding whitespace. A
    degenerate empty wrapper ``(())`` yields an empty string. A message
    that is not OOC-wrapped is returned stripped but otherwise unchanged,
    so the call is safe even when ``is_ooc`` was not checked first.
    """
    s = message.strip()
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
) -> str:
    """Build the system prompt for the verse-aware @ask flow.

    Composes (in order):
    - "You are <avatar.name>."
    - "Persona: <instruct_text>" — or "Persona: no persona set." if instruct_text is empty/whitespace.
    - "Scene: You are at <place name>. <place summary>" — derived from avatar's
      ``location`` attribute. If no location set or place not found,
      "Scene: You are nowhere in particular." is used.
    - "Recent events involving you:" followed by up to 5 bulleted lines.
      Each line is "- <event.summary>". If no events, "- (none yet)".
    - "Other avatars present here:" followed by bulleted lines for each ACTIVE
      avatar (kind='avatar', status='active') whose location matches this
      avatar's own location, EXCLUDING this avatar. Each line: "- <name>: <summary>"
      (or just "- <name>" if summary is empty). If none,
      "- (no other avatars present)".
    """
    avatar = store.get_entity(avatar_id)
    if avatar is None:
        raise ValueError("avatar not found")

    # --- Identity ---
    identity_line = f"You are {avatar.name}."

    # --- Persona ---
    if instruct_text.strip():
        persona_line = f"Persona: {instruct_text}"
    else:
        persona_line = "Persona: no persona set."

    # --- Scene ---
    location = store.get_attribute(avatar_id, "location")
    place = store.find_entity_by_name(location, kind="place") if location is not None else None

    if place is not None:
        scene_line = f"Scene: You are at {place.name}. {place.summary}"
    else:
        scene_line = "Scene: You are nowhere in particular."

    # --- Recent events involving this avatar ---
    all_events = store.recent_events(limit=50)
    avatar_events = [ev for ev in all_events if avatar_id in ev.entity_ids][:5]

    events_header = "Recent events involving you:"
    if avatar_events:
        event_bullets = "\n".join(f"- {ev.summary}" for ev in avatar_events)
    else:
        event_bullets = "- (none yet)"

    # --- Other avatars present at same location ---
    others_header = "Other avatars present here:"
    if location is not None:
        all_avatars = store.list_entities_by_kind("avatar", status="active")
        others = [
            a
            for a in all_avatars
            if a.id != avatar_id and store.get_attribute(a.id, "location") == location
        ]
    else:
        others = []

    if others:
        other_bullets = "\n".join(
            f"- {a.name}: {a.summary}" if a.summary else f"- {a.name}" for a in others
        )
    else:
        other_bullets = "- (no other avatars present)"

    # The verse_record / verse_act behavior rules and the length-cap
    # exception live in the verse framework (VERSE_SYSTEM_PROMPT in
    # prompts.py) so they get the framework footer's "rules above still
    # apply — personality changes voice, not structure" weight. The
    # personality overlay only carries scene context; per-call tool
    # argument shapes come from the tool schemas themselves.

    parts = [
        identity_line,
        persona_line,
        scene_line,
        events_header,
        event_bullets,
        others_header,
        other_bullets,
    ]
    return "\n".join(parts)


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

    Returns a :class:`VerseDispatchResult`. The four legacy tools always
    return ``ok=True`` with ``payload={'status': 'ok'}`` — preserving today's
    swallow-and-skip semantics so the wrapper's observable JSON does not
    change. Future branches (e.g. ``verse_record``) may return ``ok=False``
    with a model-facing ``error`` string.
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
            verse_look(store, avatar_id, args.get("target"))
            return _ok
        elif name == "verse_recall":
            q = args.get("query")
            if q is None:
                log.warning("verse_recall missing 'query' arg (avatar=%s)", avatar_id)
                return _ok
            verse_recall(store, q)
            return _ok
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
    in the args dict (callers that pass ``_max_actors`` explicitly win,
    so existing tests continue to work).
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
            args.setdefault("_max_actors", max_actors)
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
