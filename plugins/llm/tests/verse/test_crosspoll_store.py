from pathlib import Path

import pytest


@pytest.fixture
def crosspoll_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verse"
    d.mkdir()
    return d


class TestCrosspollStoreInit:
    def test_creates_db_file_on_first_use(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="hello", payload={})
        assert (crosspoll_dir / "_crosspoll.db").exists()

    def test_schema_version_recorded(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="hello", payload={})
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
        assert row[0] == 1
