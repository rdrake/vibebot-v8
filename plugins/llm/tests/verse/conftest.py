"""Pytest fixtures for verse tests — real SQLite, no mocks."""

from __future__ import annotations

import json as _json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    """Per-test directory for verse SQLite files."""
    d = tmp_path / "verse"
    d.mkdir()
    return d


def insert_event_at(
    store: Any,
    *,
    summary: str,
    entity_ids: Iterable[int],
    source: str,
    ts: float,
) -> int:
    """Test helper: insert an events row with a caller-specified ``ts``.

    Production code always stamps ``ts`` to ``time.time()``; this helper
    bypasses that for retention/compaction tests. Lives in conftest so
    no production class carries test-only methods.
    """
    with store.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, _json.dumps(list(entity_ids)), source),
        )
        return int(cur.lastrowid)
