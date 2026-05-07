"""Pytest fixtures for verse tests — real SQLite, no mocks."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    """Per-test directory for verse SQLite files."""
    d = tmp_path / "verse"
    d.mkdir()
    return d
