"""Tests for RPG plugin structure."""

from __future__ import annotations


def test_plugin_importable():
    """GIVEN the rpg package WHEN imported THEN it exposes Class and configure."""
    import rpg

    assert hasattr(rpg, "Class")
    assert hasattr(rpg, "configure")
    assert rpg.__version__ == "0.1.0"
