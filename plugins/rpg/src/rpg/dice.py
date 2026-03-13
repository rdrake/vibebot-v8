"""Lightweight dice roller — replaces d20 library (incompatible with Python 3.14).

Supports simple NdM+K expressions used by the combat system.
"""

from __future__ import annotations

import random
import re
from typing import NamedTuple

_DICE_RE = re.compile(r"^(\d+)d(\d+)([+-]\d+)?$")


class RollResult(NamedTuple):
    """Result of a dice roll."""

    total: int
    expression: str


def roll(expression: str) -> RollResult:
    """Roll dice using NdM or NdM+K notation.

    Examples:
        roll("1d20+3")  -> RollResult(total=<random 4-23>, expression="1d20+3")
        roll("2d6")     -> RollResult(total=<random 2-12>, expression="2d6")
    """
    expr = expression.replace(" ", "")
    match = _DICE_RE.match(expr)
    if match is None:
        msg = f"Invalid dice expression: {expression!r}"
        raise ValueError(msg)

    count = int(match.group(1))
    sides = int(match.group(2))
    modifier = int(match.group(3)) if match.group(3) else 0

    total = sum(random.randint(1, sides) for _ in range(count)) + modifier
    return RollResult(total=total, expression=expression)
