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


class TestEnqueueAndClaim:
    def test_claim_returns_seed_to_other_channel_and_marks_consumed(
        self, crosspoll_dir: Path
    ) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        sid = store.enqueue_seed(source_channel="#a", summary="A whisper", payload={"n": 1})
        seed = store.claim_seed_for("#b", proposal_id="p-1")
        assert seed is not None
        assert seed.id == sid
        assert seed.source_channel == "#a"
        assert seed.summary == "A whisper"
        assert seed.payload == {"n": 1}
        # Second claim from same dest returns None — already consumed.
        assert store.claim_seed_for("#b", proposal_id="p-2") is None

    def test_source_cannot_claim_its_own_seed(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="x", payload={})
        assert store.claim_seed_for("#a", proposal_id="p") is None

    def test_claim_returns_oldest_first(self, crosspoll_dir: Path) -> None:
        import time as _t

        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="first", payload={})
        _t.sleep(0.001)
        store.enqueue_seed(source_channel="#a", summary="second", payload={})
        seed = store.claim_seed_for("#b", proposal_id="p-b1")
        assert seed is not None and seed.id == s1
        seed2 = store.claim_seed_for("#b", proposal_id="p-b2")
        assert seed2 is not None and seed2.summary == "second"

    def test_two_destinations_can_each_claim_same_seed(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="x", payload={})
        seed_b = store.claim_seed_for("#b", proposal_id="p-b")
        seed_c = store.claim_seed_for("#c", proposal_id="p-c")
        assert seed_b is not None and seed_c is not None
        assert seed_b.id == seed_c.id == s1
        # Each dest's second claim returns None (already consumed there).
        assert store.claim_seed_for("#b", proposal_id="p-b2") is None
        assert store.claim_seed_for("#c", proposal_id="p-c2") is None

    def test_concurrent_claims_one_winner(self, crosspoll_dir: Path) -> None:
        """Two threads try to claim the same seed for the same dest;
        exactly one wins, exactly one consumption row exists."""
        import threading

        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        s1 = store.enqueue_seed(source_channel="#a", summary="x", payload={})
        results: list = []
        barrier = threading.Barrier(2)

        def claim(pid: str) -> None:
            barrier.wait()
            results.append(store.claim_seed_for("#b", proposal_id=pid))

        t1 = threading.Thread(target=claim, args=("p-1",))
        t2 = threading.Thread(target=claim, args=("p-2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        won = [r for r in results if r is not None]
        lost = [r for r in results if r is None]
        assert len(won) == 1 and len(lost) == 1
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM crosspoll_consumptions WHERE seed_id=? AND dest_channel=?",
                (s1, "#b"),
            ).fetchone()[0]
        assert count == 1

    def test_pending_count_reflects_unconsumed(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="x", payload={})
        store.enqueue_seed(source_channel="#a", summary="y", payload={})
        assert store.pending_count_for("#b") == 2
        store.claim_seed_for("#b", proposal_id="p")
        assert store.pending_count_for("#b") == 1


class TestReleaseClaim:
    """Regression: when the receiver's local proposal insert fails after
    a successful claim, the consumption row must be released so the seed
    isn't lost forever.
    """

    def test_release_then_reclaim(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        sid = store.enqueue_seed(source_channel="#a", summary="x", payload={})
        first = store.claim_seed_for("#b", proposal_id="p1")
        assert first is not None and first.id == sid
        # Without release, a re-claim returns None (already consumed).
        assert store.claim_seed_for("#b", proposal_id="p2") is None
        # Release lets the next claim succeed for the same dest.
        assert store.release_claim(sid, "#b") is True
        second = store.claim_seed_for("#b", proposal_id="p3")
        assert second is not None and second.id == sid

    def test_release_missing_row_is_idempotent(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        # Nothing consumed yet — release reports False but does not raise.
        assert store.release_claim(seed_id=999, dest_channel="#b") is False


class TestNextUnconsumedFor:
    def test_returns_none_when_empty(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        assert store.next_unconsumed_for("#b") is None

    def test_returns_oldest_unconsumed_without_marking(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        sid = store.enqueue_seed(source_channel="#a", summary="x", payload={"k": "v"})
        peek1 = store.next_unconsumed_for("#b")
        peek2 = store.next_unconsumed_for("#b")
        assert peek1 is not None and peek1.id == sid
        # Diagnostic API does NOT mark consumed — second peek still sees it.
        assert peek2 is not None and peek2.id == sid
        assert peek1.payload == {"k": "v"}

    def test_excludes_self_emissions(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="x", payload={})
        assert store.next_unconsumed_for("#a") is None

    def test_skips_already_consumed(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        store.enqueue_seed(source_channel="#a", summary="first", payload={})
        sid2 = store.enqueue_seed(source_channel="#a", summary="second", payload={})
        store.claim_seed_for("#b", proposal_id="p-1")
        peek = store.next_unconsumed_for("#b")
        assert peek is not None and peek.id == sid2


class TestWriteTransactionRollback:
    def test_rolls_back_on_exception_inside_block(self, crosspoll_dir: Path) -> None:
        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        with pytest.raises(RuntimeError), store.write_transaction() as conn:
            conn.execute(
                "INSERT INTO crosspoll_seeds "
                "(source_channel, summary, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("#a", "x", "{}", 1.0),
            )
            raise RuntimeError("boom")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM crosspoll_seeds").fetchone()[0]
        assert count == 0


class TestCrosspollConcurrency:
    def test_concurrent_enqueue_serialises(self, crosspoll_dir: Path) -> None:
        import threading

        from llm.verse.crosspoll_store import CrosspollStore

        store = CrosspollStore(crosspoll_dir)
        n_writers = 50
        errors: list[BaseException] = []

        def writer(i: int) -> None:
            try:
                store.enqueue_seed(
                    source_channel=f"#chan-{i % 4}",
                    summary=f"line-{i}",
                    payload={"i": i},
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM crosspoll_seeds").fetchone()[0]
        assert count == n_writers
