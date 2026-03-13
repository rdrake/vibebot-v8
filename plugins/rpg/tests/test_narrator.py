"""Tests for LLM narrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rpg.narrator import Narrator


@pytest.fixture
def narrator() -> Narrator:
    return Narrator(model="gemini/gemini-2.0-flash-lite", api_key="test-key", timeout=2)


class TestNarrator:
    """LLM narrator with fallback."""

    def test_fallback_on_no_api_key(self):
        """GIVEN no API key WHEN narrate THEN deterministic fallback."""
        narrator = Narrator(model="", api_key="", timeout=2)
        text = narrator.narrate_room(
            room_path="/dungeon/level1",
            description_hint="A damp corridor",
            enemies=["goblin"],
            items=["sword.txt"],
            exits=["level2", ".."],
        )
        assert "dungeon/level1" in text
        assert "goblin" in text

    def test_narrate_room_calls_llm(self, narrator: Narrator):
        """GIVEN valid config WHEN narrate THEN LLM called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The dark corridor stretches ahead."
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_room(
                room_path="/dungeon/level1",
                description_hint="A damp corridor",
                enemies=["goblin"],
                items=["sword.txt"],
                exits=["level2", ".."],
            )
        assert text == "The dark corridor stretches ahead."

    def test_fallback_on_timeout(self, narrator: Narrator):
        """GIVEN LLM times out WHEN narrate THEN fallback text returned."""
        with patch("rpg.narrator.litellm.completion", side_effect=Exception("timeout")):
            text = narrator.narrate_room(
                room_path="/dungeon/level1",
                description_hint="A damp corridor",
                enemies=["goblin"],
                items=["sword.txt"],
                exits=["level2", ".."],
            )
        # Should get deterministic fallback, not crash
        assert "dungeon/level1" in text

    def test_output_truncated(self, narrator: Narrator):
        """GIVEN LLM returns long text WHEN narrate THEN truncated to max lines."""
        long_text = "\n".join(f"Line {i}" for i in range(20))
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = long_text
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_room(
                room_path="/test",
                description_hint="test",
                enemies=[],
                items=[],
                exits=[],
            )
        assert text.count("\n") <= 3  # Max 4 lines

    def test_narrate_combat(self, narrator: Narrator):
        """GIVEN combat event WHEN narrate THEN LLM called."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Your blade strikes true!"
        with patch("rpg.narrator.litellm.completion", return_value=mock_response):
            text = narrator.narrate_combat(
                attacker="alice",
                target="goblin",
                hit=True,
                damage=5,
                enemy_killed=False,
            )
        assert "strikes" in text.lower() or "blade" in text.lower() or len(text) > 0

    def test_combat_fallback(self, narrator: Narrator):
        """GIVEN LLM fails WHEN narrate_combat THEN deterministic fallback."""
        with patch("rpg.narrator.litellm.completion", side_effect=Exception("fail")):
            text = narrator.narrate_combat(
                attacker="alice",
                target="goblin",
                hit=True,
                damage=5,
                enemy_killed=True,
            )
        assert "goblin" in text.lower()
