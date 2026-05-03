# Memory Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Periodically consolidate per-user memories by merging duplicates, dropping stale/low-value entries, and resolving contradictions via the ask model.

**Architecture:** After every N new memory saves (tracked by a monotonic DB counter), a background LLM call reviews all of a user's memories and returns index-based edit operations (keep/drop/merge). Edits are validated strictly and applied atomically; invalid output causes no DB mutation.

**Tech Stack:** Python 3.12+, SQLite (WAL), LiteLLM, Limnoria scheduler

---

### Task 1: DB schema — cleanup state table and counter methods

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:18` (bump SCHEMA_VERSION 5 → 6)
- Modify: `plugins/llm/src/llm/persistence.py:270` (add migration block after version 5)
- Modify: `plugins/llm/src/llm/persistence.py:1320` (add new methods after `delete_all_memories`)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing tests**

```python
class TestMemoryCleanupState:
    """Test memory cleanup counter methods."""

    def test_increment_memory_saves(self, tmp_path: Path) -> None:
        """GIVEN no prior saves WHEN increment THEN returns 1."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.increment_memory_saves("user1") == 1

    def test_increment_memory_saves_accumulates(self, tmp_path: Path) -> None:
        """GIVEN two increments WHEN get THEN returns 2."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.increment_memory_saves("user1")
        assert db.increment_memory_saves("user1") == 2

    def test_increment_memory_saves_case_insensitive(self, tmp_path: Path) -> None:
        """GIVEN mixed case nick WHEN increment THEN stored lowercased."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.increment_memory_saves("Alice")
        assert db.increment_memory_saves("alice") == 2

    def test_reset_memory_saves(self, tmp_path: Path) -> None:
        """GIVEN incremented counter WHEN reset THEN returns to 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.increment_memory_saves("user1")
        db.increment_memory_saves("user1")
        db.reset_memory_saves("user1")
        assert db.increment_memory_saves("user1") == 1

    def test_get_memory_saves_default_zero(self, tmp_path: Path) -> None:
        """GIVEN no prior saves WHEN get THEN returns 0."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.get_memory_saves("user1") == 0
```

**Step 2: Run tests to verify they fail**

Run: `make test 2>&1 | grep -E "test_.*memory_saves"`
Expected: FAIL — methods don't exist yet

**Step 3: Write implementation**

In `persistence.py:18`, bump:
```python
SCHEMA_VERSION = 6
```

After the `if current_version < 5:` block (~line 270), add:
```python
            if current_version < 6:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memory_cleanup_state (
                        nick TEXT PRIMARY KEY,
                        saves_since_cleanup INTEGER NOT NULL DEFAULT 0
                    );
                """)
                conn.commit()
```

After `delete_all_memories` method (~line 1320), add:
```python
    def increment_memory_saves(self, nick: str) -> int:
        """Increment the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).

        Returns:
            The new counter value after incrementing.
        """
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO memory_cleanup_state (nick, saves_since_cleanup) "
                "VALUES (?, 1) "
                "ON CONFLICT(nick) DO UPDATE SET saves_since_cleanup = saves_since_cleanup + 1",
                (nick.lower(),),
            )
            conn.commit()
            row = conn.execute(
                "SELECT saves_since_cleanup FROM memory_cleanup_state WHERE nick = ?",
                (nick.lower(),),
            ).fetchone()
            return row[0] if row else 0
        finally:
            pass

    def reset_memory_saves(self, nick: str) -> None:
        """Reset the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).
        """
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE memory_cleanup_state SET saves_since_cleanup = 0 WHERE nick = ?",
                (nick.lower(),),
            )
            conn.commit()
        finally:
            pass

    def get_memory_saves(self, nick: str) -> int:
        """Get the current memory-saves-since-cleanup count for a user.

        Args:
            nick: IRC nick (matched case-insensitively).

        Returns:
            Current counter value, or 0 if no record exists.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT saves_since_cleanup FROM memory_cleanup_state WHERE nick = ?",
                (nick.lower(),),
            ).fetchone()
            return row[0] if row else 0
        finally:
            pass
```

**Step 4: Run tests to verify they pass**

Run: `make test 2>&1 | grep -E "test_.*memory_saves"`
Expected: all 5 PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add memory_cleanup_state table and counter methods"
```

---

### Task 2: Config — memoryCleanupInterval

**Files:**
- Modify: `plugins/llm/src/llm/config.py:271` (add config after memoryApiKey)

**Step 1: Write implementation**

After the `memoryApiKey` registration (~line 271), add:
```python
conf.registerGlobalValue(
    LLM,
    "memoryCleanupInterval",
    registry.NonNegativeInteger(
        3,
        _("""Number of new memory saves between automatic cleanup passes.
        Set to 0 to disable periodic cleanup."""),
    ),
)
```

**Step 2: Run preflight**

Run: `make lint && make typecheck`
Expected: clean (pre-existing errors only)

**Step 3: Commit**

```bash
git add plugins/llm/src/llm/config.py
git commit -m "feat: add memoryCleanupInterval config (default 3)"
```

---

### Task 3: Service — cleanup prompt and cleanup_memories method

**Files:**
- Modify: `plugins/llm/src/llm/service.py:92` (add cleanup prompt constant after extraction prompt)
- Modify: `plugins/llm/src/llm/service.py:132` (add CleanupResult NamedTuple after ExtractionResult)
- Modify: `plugins/llm/src/llm/service.py` (add cleanup_memories method after extract_memories)
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Write the failing tests**

```python
class TestMemoryCleanup:
    """Test memory cleanup LLM call and validation."""

    def test_cleanup_returns_valid_edits(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN memories with duplicates WHEN cleanup THEN returns keep/drop/merge."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = (
            '{"keep": [0, 3], "drop": [4], "merge": [[1, 2, "likes Python"]]}'
        )
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "moved to Vancouver", "#test", 500.0),
            MemoryRow(11, "user1", "likes Python programming", "#test", 400.0),
            MemoryRow(12, "user1", "enjoys writing Python", "#test", 300.0),
            MemoryRow(13, "user1", "works at Acme", "#test", 200.0),
            MemoryRow(14, "user1", "asked about weather", "#test", 100.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.keep == [0, 3]
        assert result.drop == [4]
        assert result.merge == [[1, 2, "likes Python"]]

    def test_cleanup_returns_empty_on_error(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN API error WHEN cleanup THEN returns empty result."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_litellm.completion.side_effect = Exception("API down")

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.keep == []
        assert result.drop == []
        assert result.merge == []
        assert result.error is not None

    def test_cleanup_rejects_invalid_json(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN garbage LLM output WHEN cleanup THEN returns error result."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = "not json at all"
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_duplicate_indices(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN index in both keep and drop WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = '{"keep": [0, 1], "drop": [1], "merge": []}'
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_out_of_range_index(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN out-of-range index WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = '{"keep": [0, 5], "drop": [], "merge": []}'
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_rejects_empty_merge_text(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN merge with empty text WHEN cleanup THEN returns error."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = '{"keep": [], "drop": [], "merge": [[0, 1, ""]]}'
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "fact a", "#test", 100.0),
            MemoryRow(11, "user1", "fact b", "#test", 200.0),
        ]
        result = service.cleanup_memories("user1", "#test", rows)
        assert result.error is not None

    def test_cleanup_prompt_includes_indexed_memories(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN memories WHEN cleanup called THEN prompt lists them with indices."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = '{"keep": [0, 1], "drop": [], "merge": []}'
        mock_litellm.completion.return_value = mock_response

        rows = [
            MemoryRow(10, "user1", "likes Python", "#test", 200.0),
            MemoryRow(11, "user1", "works at Acme", "#test", 100.0),
        ]
        service.cleanup_memories("user1", "#test", rows)

        call_args = mock_litellm.completion.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        prompt_text = " ".join(m["content"] for m in messages)
        assert "[0] likes Python" in prompt_text
        assert "[1] works at Acme" in prompt_text

    def test_cleanup_uses_ask_model(self, make_service, mocker: MockerFixture) -> None:
        """GIVEN cleanup call WHEN LLM invoked THEN uses askModel and askApiKey."""
        from llm.persistence import MemoryRow

        service, mock_plugin = make_service()
        mock_litellm = mocker.patch("llm.service.litellm")
        mock_response = mocker.MagicMock()
        mock_response.choices = [mocker.MagicMock()]
        mock_response.choices[0].message.content = '{"keep": [0], "drop": [], "merge": []}'
        mock_litellm.completion.return_value = mock_response

        rows = [MemoryRow(10, "user1", "fact", "#test", 100.0)]
        service.cleanup_memories("user1", "#test", rows)

        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"  # default askModel in test fixture
        assert call_kwargs["api_key"] == "test-api-key"  # default askApiKey in test fixture
```

**Step 2: Run tests to verify they fail**

Run: `make test 2>&1 | grep -E "test_cleanup"`
Expected: FAIL — `cleanup_memories` doesn't exist

**Step 3: Write implementation**

Add cleanup prompt after `_MEMORY_EXTRACTION_PROMPT` (~line 92 in service.py):
```python
_MEMORY_CLEANUP_PROMPT = (
    "You are a memory curator. Review these stored facts about an IRC user and "
    "return edit operations as JSON.\n\n"
    "Rules:\n"
    "- ONLY reference facts by their index numbers below\n"
    "- Do NOT invent new facts — merge text must combine existing information only\n"
    "- Facts are listed newest-first; when facts contradict, prefer the newer one "
    "(lower index)\n"
    "- Merge near-duplicates into one clear statement\n"
    "- Drop vague, trivial, or clearly transient/time-bound facts\n"
    "- Keep all genuinely useful long-term information\n\n"
    'Return JSON: {"keep": [...], "drop": [...], "merge": [[idx_a, idx_b, "text"], ...]}\n'
    "Every index must appear in exactly one category (keep, drop, or as a source in merge).\n"
)
```

Add `CleanupResult` after `ExtractionResult` (~line 134):
```python
class CleanupResult(NamedTuple):
    """Result of memory cleanup: index-based edit operations."""

    keep: list[int] = []
    drop: list[int] = []
    merge: list[list] = []
    error: str | None = None
```

Add `cleanup_memories` method after `extract_memories` on `LLMService`:
```python
    def cleanup_memories(
        self,
        nick: str,
        channel: str,
        memory_rows: list[MemoryRow],
    ) -> CleanupResult:
        """Review a user's memories and return index-based edit operations.

        Uses the ask model (more capable) to identify duplicates,
        contradictions, stale entries, and low-quality facts.

        Args:
            nick: The user's IRC nick.
            channel: Channel for config lookups.
            memory_rows: Current memories (newest-first from get_memories).

        Returns:
            CleanupResult with validated edit operations, or error on failure.
        """
        memory_section = "\n".join(f"[{i}] {r.fact}" for i, r in enumerate(memory_rows))

        messages = [
            {"role": "system", "content": _MEMORY_CLEANUP_PROMPT},
            {
                "role": "user",
                "content": f"Current memories for {nick}:\n{memory_section}",
            },
        ]

        try:
            model = self.plugin.registryValue("askModel", channel)
            api_key = self.plugin.registryValue("askApiKey")
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                timeout=30,
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
        except Exception as e:
            return CleanupResult(error=f"LLM call failed: {e}")

        # Validate structure
        if not isinstance(parsed, dict):
            return CleanupResult(error="Response is not a JSON object")

        keep = parsed.get("keep", [])
        drop = parsed.get("drop", [])
        merge = parsed.get("merge", [])

        if not isinstance(keep, list) or not isinstance(drop, list) or not isinstance(merge, list):
            return CleanupResult(error="keep/drop/merge must be arrays")

        num_memories = len(memory_rows)

        # Validate all indices are ints and in range
        all_indices: list[int] = []
        for idx in keep:
            if not isinstance(idx, int) or idx < 0 or idx >= num_memories:
                return CleanupResult(error=f"Invalid keep index: {idx}")
            all_indices.append(idx)

        for idx in drop:
            if not isinstance(idx, int) or idx < 0 or idx >= num_memories:
                return CleanupResult(error=f"Invalid drop index: {idx}")
            all_indices.append(idx)

        # Validate merge entries
        validated_merge: list[list] = []
        for entry in merge:
            if not isinstance(entry, list) or len(entry) != 3:
                return CleanupResult(error=f"Invalid merge entry: {entry}")
            idx_a, idx_b, text = entry
            if not isinstance(idx_a, int) or not isinstance(idx_b, int):
                return CleanupResult(error=f"Merge indices must be ints: {entry}")
            if idx_a < 0 or idx_a >= num_memories or idx_b < 0 or idx_b >= num_memories:
                return CleanupResult(error=f"Merge index out of range: {entry}")
            if not isinstance(text, str) or not text.strip():
                return CleanupResult(error=f"Merge text must be non-empty: {entry}")
            all_indices.append(idx_a)
            all_indices.append(idx_b)
            validated_merge.append([idx_a, idx_b, text.strip()])

        # Check for duplicate indices
        if len(all_indices) != len(set(all_indices)):
            return CleanupResult(error="Duplicate index across keep/drop/merge")

        return CleanupResult(keep=keep, drop=drop, merge=validated_merge)
```

Note: You'll need to add `MemoryRow` to the imports at the top of service.py (under `TYPE_CHECKING`):
```python
if TYPE_CHECKING:
    from llm.persistence import MemoryRow
```

**Step 4: Run tests to verify they pass**

Run: `make test 2>&1 | grep -E "test_cleanup"`
Expected: all 8 PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: clean

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: add cleanup_memories method with index-based edits and validation"
```

---

### Task 4: Plugin — wire up trigger and apply edits

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:344-358` (add in-flight cleanup guard set)
- Modify: `plugins/llm/src/llm/plugin.py:1555-1573` (add counter increment and trigger after extraction saves)
- Modify: `plugins/llm/src/llm/plugin.py:1480` (add `_schedule_memory_cleanup` method near `_get_user_memories`)
- Test: `plugins/llm/tests/test_integration.py`

**Step 1: Write the failing tests**

```python
class TestMemoryCleanup:
    """Test background memory cleanup trigger and application."""

    @pytest.fixture
    def plugin_with_real_db(
        self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path: Path
    ) -> tuple:
        """Create plugin with real database for cleanup testing."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        db_path = str(tmp_path / "test.db")
        registry = make_registry_side_effect({
            "databasePath": db_path,
            "memoryEnabled": True,
            "memoryCleanupInterval": 3,
        })
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()

        mock_irc.state.nickToAccount.return_value = "testuser"

        return plugin, mock_irc

    def test_cleanup_applies_drop(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN cleanup returns drop WHEN applied THEN memories are deleted."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "useful fact", "#test")
        plugin.db.save_memory("testuser", "stale fact", "#test")

        # Mock the service cleanup to return drop for index 1
        # Memories are newest-first: [0]="stale fact", [1]="useful fact"
        plugin.llm_service.cleanup_memories.return_value = CleanupResult(
            keep=[1], drop=[0], merge=[]
        )

        plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "useful fact"

    def test_cleanup_applies_merge(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN cleanup returns merge WHEN applied THEN memories are merged."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "likes Python programming", "#test")
        plugin.db.save_memory("testuser", "enjoys writing Python", "#test")

        plugin.llm_service.cleanup_memories.return_value = CleanupResult(
            keep=[], drop=[], merge=[[0, 1, "likes Python programming"]]
        )

        plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "likes Python programming"

    def test_cleanup_aborts_on_error(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN cleanup returns error WHEN applied THEN no DB changes."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")

        plugin.llm_service.cleanup_memories.return_value = CleanupResult(
            error="LLM returned garbage"
        )

        plugin._run_memory_cleanup("testuser", "#test")

        # Both memories should still exist
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 2

    def test_cleanup_aborts_on_snapshot_mismatch(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN memory count changes during cleanup WHEN applying THEN abort."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")

        # Simulate a new memory being added during the LLM call
        original_cleanup = plugin.llm_service.cleanup_memories

        def cleanup_with_side_effect(*args, **kwargs):
            plugin.db.save_memory("testuser", "new fact during cleanup", "#test")
            return CleanupResult(keep=[0], drop=[1], merge=[])

        plugin.llm_service.cleanup_memories.side_effect = cleanup_with_side_effect

        plugin._run_memory_cleanup("testuser", "#test")

        # All 3 memories should still exist (abort due to mismatch)
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 3

    def test_cleanup_skips_if_already_in_flight(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN nick already being cleaned WHEN scheduled THEN skip."""
        plugin, mock_irc = plugin_with_real_db

        plugin._cleanup_in_flight.add("testuser")

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")

        plugin._run_memory_cleanup("testuser", "#test")

        # No changes — cleanup was skipped
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 2

    def test_cleanup_resets_counter_on_success(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN successful cleanup WHEN done THEN saves counter is reset."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.increment_memory_saves("testuser")
        plugin.db.increment_memory_saves("testuser")
        plugin.db.increment_memory_saves("testuser")

        plugin.llm_service.cleanup_memories.return_value = CleanupResult(
            keep=[0], drop=[], merge=[]
        )

        plugin._run_memory_cleanup("testuser", "#test")

        assert plugin.db.get_memory_saves("testuser") == 0

    def test_cleanup_disabled_when_interval_zero(
        self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """GIVEN memoryCleanupInterval=0 WHEN saves happen THEN no cleanup scheduled."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        db_path = str(tmp_path / "test.db")
        registry = make_registry_side_effect({
            "databasePath": db_path,
            "memoryEnabled": True,
            "memoryCleanupInterval": 0,
        })
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()

        # Simulate saving 5 memories — no cleanup should trigger
        for i in range(5):
            plugin.db.save_memory("testuser", f"fact {i}", "#test")
            plugin.db.increment_memory_saves("testuser")

        cleanup_calls = [c for c in mock_add_event.call_args_list if "llm_cleanup_" in str(c)]
        assert len(cleanup_calls) == 0
```

**Step 2: Run tests to verify they fail**

Run: `make test 2>&1 | grep -E "TestMemoryCleanup"`
Expected: FAIL — methods don't exist yet

**Step 3: Write implementation**

In `plugin.py` `__init__` (~line 355, after `_spontaneous_events`), add:
```python
        # In-flight memory cleanup guard: set of nicks currently being cleaned
        self._cleanup_in_flight: set[str] = set()
```

After `_get_user_memories` (~line 1483), add:
```python
    def _schedule_memory_cleanup(self, nick: str, channel: str) -> None:
        """Schedule a background memory cleanup for a user."""
        if nick in self._cleanup_in_flight:
            return

        def _cleanup_bg() -> None:
            self._run_memory_cleanup(nick, channel)

        event_name = f"llm_cleanup_{uuid.uuid4().hex[:8]}"
        schedule.addEvent(_cleanup_bg, time.time() + 0.5, name=event_name)

    def _run_memory_cleanup(self, nick: str, channel: str) -> None:
        """Run memory cleanup for a user. Called from scheduled event."""
        if nick in self._cleanup_in_flight:
            return

        self._cleanup_in_flight.add(nick)
        try:
            # Snapshot current memories
            snapshot = self.db.get_memories(nick)
            if len(snapshot) < 2:
                return

            result = self.llm_service.cleanup_memories(nick, channel, snapshot)

            if result.error:
                log.warning("Memory cleanup failed for %s: %s", nick, result.error)
                return

            # Abort if memory count changed during LLM call (race protection)
            current = self.db.get_memories(nick)
            if len(current) != len(snapshot):
                log.info("Memory cleanup aborted for %s: count changed", nick)
                return

            # Apply drops
            for idx in result.drop:
                if 0 <= idx < len(snapshot):
                    self.db.delete_memory(nick, snapshot[idx].id)

            # Apply merges: delete sources, insert merged fact
            for entry in result.merge:
                idx_a, idx_b, merged_text = entry
                if 0 <= idx_a < len(snapshot) and 0 <= idx_b < len(snapshot):
                    # Preserve oldest source_channel and created_at
                    source_a = snapshot[idx_a]
                    source_b = snapshot[idx_b]
                    oldest = source_a if source_a.created_at <= source_b.created_at else source_b
                    self.db.delete_memory(nick, source_a.id)
                    self.db.delete_memory(nick, source_b.id)
                    self.db.save_memory(nick, merged_text, oldest.source_channel)

            # Success — reset counter
            self.db.reset_memory_saves(nick)
        except Exception:
            log.exception("Memory cleanup error for %s", nick)
        finally:
            self._cleanup_in_flight.discard(nick)
```

In `_extract_memories_bg` (~line 1568, after the `self.db.save_memory` loop), add the trigger:
```python
                            # Check if cleanup should run
                            interval = self.registryValue("memoryCleanupInterval")
                            if interval:
                                for fact in extraction.add:
                                    # ... (already saving above)
                                    pass
                                # After saving, check counter
                                new_count = self.db.increment_memory_saves(nick)
                                if new_count >= interval:
                                    self._schedule_memory_cleanup(nick, channel)
```

Actually, this needs to be integrated into the existing save loop. The modified `_extract_memories_bg` should look like:
```python
                    def _extract_memories_bg() -> None:
                        try:
                            extraction = self.llm_service.extract_memories(
                                nick, channel, text, raw_response, existing_facts
                            )
                            # Remove superseded/contradicted memories
                            for idx in extraction.remove:
                                if 0 <= idx < len(existing_rows):
                                    self.db.delete_memory(nick, existing_rows[idx].id)
                            # Add new facts
                            saved_count = 0
                            for fact in extraction.add:
                                if len(self.db.get_memories(nick)) >= max_memories:
                                    break
                                self.db.save_memory(nick, fact, channel)
                                saved_count += 1
                            # Trigger cleanup if enough new saves accumulated
                            if saved_count > 0:
                                interval = self.registryValue("memoryCleanupInterval")
                                if interval:
                                    new_count = self.db.increment_memory_saves(nick)
                                    if new_count >= interval:
                                        self._schedule_memory_cleanup(nick, channel)
                        except Exception:
                            log.exception("Memory extraction failed for %s", nick)
```

**Step 4: Run tests to verify they pass**

Run: `make test 2>&1 | grep -E "TestMemoryCleanup|test_cleanup"`
Expected: all PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: clean

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_integration.py
git commit -m "feat: wire up memory cleanup trigger and edit application"
```

---

### Task 5: Update conftest fixtures for new config key

**Files:**
- Modify: `plugins/llm/tests/conftest.py` (add `memoryCleanupInterval` to default registry)

**Step 1: Check conftest for registry defaults**

Find the `make_registry_side_effect` function and add `"memoryCleanupInterval": 3` to the defaults dict alongside the other memory config keys.

**Step 2: Run full preflight**

Run: `make preflight`
Expected: all tests pass, lint/typecheck clean

**Step 3: Commit**

```bash
git add plugins/llm/tests/conftest.py
git commit -m "chore: add memoryCleanupInterval to test fixture defaults"
```

---

### Task 6: Final integration test and squash commit

**Step 1: Run the full preflight**

Run: `make preflight`
Expected: 80%+ coverage, all tests pass, lint/typecheck clean

**Step 2: Verify the design doc is up to date**

Read `docs/plans/2026-03-12-memory-cleanup-design.md` and confirm it matches the implementation.

**Step 3: Final commit (if any remaining changes)**

```bash
git add -A
git commit -m "feat: periodic memory cleanup with index-based edits

Adds background memory cleanup that triggers every N new saves per user.
Uses the ask model to review memories and return keep/drop/merge operations
referencing input indices. Validates strictly and fails closed on bad output.
Includes race condition protection via in-flight guard and snapshot check."
```
