"""Tests for the forest-verse loom orchestrator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def test_loomconfig_holds_all_settings() -> None:
    from llm.verse.loom import LoomConfig

    cfg = LoomConfig(
        network="afternet",
        loom_channel="#forest",
        bot_nicks=("botA", "botB"),
        model="gemini/gemini-flash-lite-latest",
        cycle_interval_s=300,
        verse_cooldown_s=1200,
        beat_window_s=90,
        transcript_max_lines=40,
        transcript_max_chars=8000,
        auto_apply_threshold=0.85,
    )
    assert cfg.loom_channel == "#forest"
    assert cfg.network == "afternet"
    assert cfg.bot_nicks == ("botA", "botB")
    with pytest.raises(FrozenInstanceError):
        cfg.cycle_interval_s = 1  # type: ignore[misc]
