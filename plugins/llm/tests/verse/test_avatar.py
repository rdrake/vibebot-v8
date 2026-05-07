"""Tests for verb whitelist (VERB_TABLE) in avatar module."""

from __future__ import annotations

import pytest
from llm.verse.avatar import VERB_TABLE, VerbEffect


class TestVerbWhitelist:
    def test_event_only_verbs(self) -> None:
        event_only = [
            "whisper",
            "speak",
            "listen",
            "examine",
            "wait",
            "signal",
            "gesture",
            "search",
        ]
        for verb in event_only:
            assert VERB_TABLE[verb] is VerbEffect.EVENT_ONLY, f"{verb!r} should be EVENT_ONLY"

    def test_move_verbs(self) -> None:
        for verb in ("move", "flee", "follow"):
            assert VERB_TABLE[verb] is VerbEffect.MOVE, f"{verb!r} should be MOVE"

    def test_item_verbs(self) -> None:
        for verb in ("take", "drop", "give"):
            assert VERB_TABLE[verb] is VerbEffect.ITEM, f"{verb!r} should be ITEM"

    @pytest.mark.parametrize("verb", ["teleport", "attack", "cast", "eat"])
    def test_off_list_verbs_absent(self, verb: str) -> None:
        assert verb not in VERB_TABLE

    def test_table_size(self) -> None:
        assert len(VERB_TABLE) == 14
