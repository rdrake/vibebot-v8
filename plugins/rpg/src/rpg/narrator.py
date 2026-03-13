"""LLM narrator — flavor text with deterministic fallback."""

from __future__ import annotations

import logging

import litellm

_log = logging.getLogger("supybot.plugins.RPG.narrator")

# Max output lines per narration type
_MAX_ROOM_LINES = 4
_MAX_COMBAT_LINES = 3

_NARRATOR_SYSTEM = (
    "You are the narrator for a Linux-filesystem-themed IRC RPG. "
    "Describe game events in 1-3 short sentences. Plain text only — no markdown, "
    "no formatting, no emojis. Be atmospheric but concise. "
    "Players navigate with shell commands (cd, ls, rm, etc). "
    "The game world IS a Linux filesystem."
)


class Narrator:
    """Generates flavor text for game events via LLM, with deterministic fallback."""

    def __init__(self, *, model: str, api_key: str, timeout: int) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    def _call_llm(self, prompt: str, max_lines: int) -> str | None:
        """Call LLM and return text, or None on any failure."""
        if not self._model or not self._api_key:
            return None
        try:
            response = litellm.completion(
                model=self._model,
                api_key=self._api_key,
                messages=[
                    {"role": "system", "content": _NARRATOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                timeout=self._timeout,
                max_tokens=150,
            )
            text = response.choices[0].message.content or ""
            # Truncate to max lines
            lines = text.strip().split("\n")
            return "\n".join(lines[:max_lines])
        except Exception:
            _log.debug("Narrator LLM call failed, using fallback", exc_info=True)
            return None

    def narrate_room(
        self,
        *,
        room_path: str,
        description_hint: str,
        enemies: list[str],
        items: list[str],
        exits: list[str],
    ) -> str:
        """Generate room description."""
        prompt = (
            f"The player enters {room_path}. "
            f"Setting: {description_hint}. "
            f"Enemies present: {', '.join(enemies) if enemies else 'none'}. "
            f"Items on the ground: {', '.join(items) if items else 'none'}. "
            f"Exits: {', '.join(exits)}."
        )
        result = self._call_llm(prompt, _MAX_ROOM_LINES)
        if result is not None:
            return result
        return self._fallback_room(room_path, description_hint, enemies, items, exits)

    def narrate_combat(
        self,
        *,
        attacker: str,
        target: str,
        hit: bool,
        damage: int,
        enemy_killed: bool,
    ) -> str:
        """Generate combat narration."""
        if hit and enemy_killed:
            action = f"{attacker} strikes {target} for {damage} damage, destroying it!"
        elif hit:
            action = f"{attacker} hits {target} for {damage} damage."
        else:
            action = f"{attacker} swings at {target} but misses."

        prompt = f"Narrate this combat action in the Linux RPG: {action}"
        result = self._call_llm(prompt, _MAX_COMBAT_LINES)
        if result is not None:
            return result
        return self._fallback_combat(attacker, target, hit, damage, enemy_killed)

    @staticmethod
    def _fallback_room(
        room_path: str,
        description_hint: str,
        enemies: list[str],
        items: list[str],
        exits: list[str],
    ) -> str:
        """Deterministic room description."""
        parts = [f"{room_path} — {description_hint}"]
        if enemies:
            parts.append(f"Enemies: {', '.join(enemies)}")
        if items:
            parts.append(f"Items: {', '.join(items)}")
        parts.append(f"Exits: {', '.join(exits)}")
        return " | ".join(parts)

    @staticmethod
    def _fallback_combat(
        attacker: str,
        target: str,
        hit: bool,
        damage: int,
        enemy_killed: bool,
    ) -> str:
        """Deterministic combat narration."""
        if hit and enemy_killed:
            return f"{attacker} rm -f {target} [{damage} dmg] — process terminated."
        if hit:
            return f"{attacker} rm {target} [{damage} dmg] — still running."
        return f"{attacker} rm {target} — permission denied (miss)."
