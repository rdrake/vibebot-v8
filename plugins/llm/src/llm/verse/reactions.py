"""Offline verse reaction signal: capture + report fc42's 👍/👎 to verse lines.

Pure, side-effect-free core (classification, recency attribution, JSONL
(de)serialisation, and the offline report section). The live IRC glue (doTagmsg,
the send-hook) lives in plugin.py and delegates here. See
docs/superpowers/specs/2026-06-27-verse-reaction-signal-design.md.
"""

from __future__ import annotations

_THUMB_UP = "\U0001f44d"
_THUMB_DOWN = "\U0001f44e"
_SKIN_TONES = {"\U0001f3fb", "\U0001f3fc", "\U0001f3fd", "\U0001f3fe", "\U0001f3ff"}
_VARIATION_SELECTOR = "️"


def classify_emoji(emoji: str) -> str:
    """Map a reaction emoji to 'approve' | 'disapprove' | 'other'.

    Strips skin-tone modifiers and the U+FE0F variation selector so 👍🏽 / 👍️ match 👍.
    """
    core = "".join(c for c in (emoji or "") if c not in _SKIN_TONES and c != _VARIATION_SELECTOR)
    if core == _THUMB_UP:
        return "approve"
    if core == _THUMB_DOWN:
        return "disapprove"
    return "other"
