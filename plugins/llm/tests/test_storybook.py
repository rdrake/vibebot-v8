"""Tests for the verse storybook tool."""

from __future__ import annotations


def test_storybook_config_defaults():
    import llm.config as cfg

    group = cfg.LLM
    assert group.verseStorybookEnabled is not None
    assert int(group.verseStorybookMaxImages()) == 3
    assert int(group.verseStorybookMaxPerTurn()) == 1
    assert int(group.verseStorybookCooldownSeconds()) == 300
    assert int(group.verseStorybookDailyImageCap()) == 30
    assert int(group.verseStorybookMaxChars()) == 6000
    assert int(group.verseStorybookImageTimeout()) == 45
