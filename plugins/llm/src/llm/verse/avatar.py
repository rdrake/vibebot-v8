"""Avatar shim: wraps @ask for opted-in users, exposes verb-whitelist tools."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from .store import VerseStore


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
        place = store.find_entity_by_name(target or "", kind="place") if target else None

        if place is None and target is not None:
            # Try resolving target as another avatar's name, then use their location
            other = store.find_entity_by_name(target, kind="avatar")
            if other is not None:
                loc = store.get_attribute(other.id, "location")
                if loc is not None:
                    place = store.find_entity_by_name(loc, kind="place")

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
        item = store.find_entity_by_name(target or "", kind="item") if target else None

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
