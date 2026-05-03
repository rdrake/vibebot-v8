# Memory Quality Improvement Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the memory system so it stops creating unnecessary, contradicting, and verbose entries by tightening the extraction prompt and making cleanup resilient.

**Architecture:** Two changes: (1) Replace the permissive extraction prompt with a strict one that limits output to 2 durable identity facts per exchange and instructs the LLM to consolidate related existing facts. (2) Simplify the cleanup contract — remove the `keep` field, default unmentioned indices to keep, and support multi-way merges so cleanup can consolidate N related facts into one.

**Tech Stack:** Python, LiteLLM, pytest

---

### Task 1: Tighten the extraction prompt

**Files:**
- Modify: `plugins/llm/src/llm/service.py:78-92`

**Step 1: Write the failing test**

Add a test that asserts the extraction prompt contains key strictness markers. Add to `plugins/llm/tests/test_service.py` in the `TestMemoryExtraction` class area (after line ~3226):

```python
def test_extract_memories_prompt_limits_facts(self, make_service, mocker: MockerFixture) -> None:
    """GIVEN extraction prompt WHEN checked THEN contains strictness markers."""
    from llm.service import _MEMORY_EXTRACTION_PROMPT

    assert "at most 2" in _MEMORY_EXTRACTION_PROMPT.lower()
    assert "DO NOT SAVE" in _MEMORY_EXTRACTION_PROMPT
    assert "CONSOLIDATION" in _MEMORY_EXTRACTION_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — current prompt doesn't contain these markers.

**Step 3: Replace the extraction prompt**

Replace `_MEMORY_EXTRACTION_PROMPT` in `plugins/llm/src/llm/service.py:78-92` with:

```python
_MEMORY_EXTRACTION_PROMPT = (
    "You are a fact extractor. Given a conversation between a user and an assistant, "
    "extract ONLY durable identity facts about the user — things that would still be "
    "true and useful in a month.\n\n"
    "SAVE: occupation, technical skills, OS/tool preferences, location, pets, hobbies, "
    "strong opinions they have stated directly.\n\n"
    "DO NOT SAVE:\n"
    "- Conversation topics or questions they asked (not facts about them)\n"
    "- Jokes, sarcasm, or hypotheticals taken literally\n"
    "- Transient activities (working on X right now, debugging Y)\n"
    "- One-time preferences or situational advice\n"
    "- Vague or trivial observations\n"
    "- Facts already known (listed below)\n\n"
    "CONSOLIDATION: If a new fact overlaps with existing facts, include ALL related "
    "existing indices in \"remove\" and provide ONE consolidated fact in \"add\".\n"
    "Example: if [3] \"uses Arch Linux\" and [5] \"uses Debian\" exist, and the user "
    "mentions Fedora, return: "
    '{\"add\": [\"uses Linux (Arch, Debian, Fedora)\"], \"remove\": [3, 5]}\n\n'
    "Return ONLY a JSON object with two keys:\n"
    '- \"add\": array of short factual strings (at most 2 per exchange)\n'
    '- \"remove\": array of 0-based indices of existing facts that are contradicted, '
    "corrected, or superseded\n\n"
    'If nothing worth saving: {\"add\": [], \"remove\": []}\n'
    "Prefer saving nothing over saving junk.\n"
)
```

**Step 4: Run test to verify it passes**

Run: `make test`
Expected: All tests pass. Existing extraction tests should still pass because they mock the LLM response — the prompt change doesn't affect them.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(memory): tighten extraction prompt to reduce junk facts"
```

---

### Task 2: Remove `keep` from `CleanupResult`

**Files:**
- Modify: `plugins/llm/src/llm/service.py:152-158`

**Step 1: Write the failing test**

Add a test in `plugins/llm/tests/test_service.py` after the `TestMemoryCleanup` class area:

```python
def test_cleanup_result_has_no_keep_field(self) -> None:
    """GIVEN CleanupResult WHEN inspected THEN has no keep field."""
    from llm.service import CleanupResult

    assert not hasattr(CleanupResult, "keep") or "keep" not in CleanupResult._fields
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — `CleanupResult` still has `keep`.

**Step 3: Remove `keep` from `CleanupResult`**

In `plugins/llm/src/llm/service.py:152-158`, change `CleanupResult` to:

```python
class CleanupResult(NamedTuple):
    """Result of memory cleanup: index-based edit operations."""

    drop: list[int] = []
    merge: list[list] = []
    error: str | None = None
```

**Step 4: Fix all references to `CleanupResult.keep`**

Update every test that constructs a `CleanupResult` with `keep=`:

In `plugins/llm/tests/test_service.py`:
- Line 3241: `'{"keep": [0, 3], "drop": [4], ...'` → `'{"drop": [4], ...'` (LLM response mock)
- Line 3252: `assert result.keep == [0, 3]` → remove this assertion
- Line 3300: `'{"keep": [0, 1], "drop": [1], ...'` → `'{"drop": [0, 1], ...'` (testing duplicate indices — both in drop)
- Line 3318: `'{"keep": [0, 5], "drop": [], ...'` → `'{"drop": [], "merge": [[0, 5, "merged"]]}'` (test out-of-range differently — put bad index in merge)
- Line 3356: `'{"keep": [], "drop": [0, 1], ...'` → stays valid, just remove `keep` key
- Line 3376: `'{"keep": [0, 1], "drop": [], ...'` → `'{"drop": [], "merge": []}'`
- Line 3399: `'{"keep": [0], "drop": [], ...'` → `'{"drop": [], "merge": []}'`
- Line 3419 area: same pattern

In `plugins/llm/tests/test_integration.py`:
- Line 718: `CleanupResult(keep=[1], drop=[0], merge=[])` → `CleanupResult(drop=[0], merge=[])`
- Line 737: `CleanupResult(keep=[], drop=[], merge=[[0, 1, ...]])` → `CleanupResult(drop=[], merge=[[0, 1, ...]])`
- Line 779: `CleanupResult(keep=[0], drop=[1], merge=[])` → `CleanupResult(drop=[1], merge=[])`
- Line 819: `CleanupResult(keep=[0, 1], drop=[], merge=[])` → `CleanupResult(drop=[], merge=[])`

**Step 5: Run tests to verify they pass**

Run: `make test`
Expected: All pass.

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py plugins/llm/tests/test_integration.py
git commit -m "refactor(memory): remove keep field from CleanupResult"
```

---

### Task 3: Update cleanup validation — default unmentioned indices to keep

**Files:**
- Modify: `plugins/llm/src/llm/service.py:2601-2696` (the `cleanup_memories` method)

**Step 1: Write the failing test**

Add a test that verifies unmentioned indices are implicitly kept (currently this would fail because the old code requires every index to appear):

```python
def test_cleanup_keeps_unmentioned_indices(self, make_service, mocker: MockerFixture) -> None:
    """GIVEN LLM omits some indices WHEN cleanup THEN unmentioned indices are kept."""
    from llm.persistence import MemoryRow

    service, mock_plugin = make_service()
    mock_litellm = mocker.patch("llm.service.litellm")
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    # Only mentions index 2 (drop) — indices 0, 1 should be implicitly kept
    mock_response.choices[0].message.content = '{"drop": [2], "merge": []}'
    mock_litellm.completion.return_value = mock_response

    rows = [
        MemoryRow(10, "user1", "fact a", "#test", 300.0),
        MemoryRow(11, "user1", "fact b", "#test", 200.0),
        MemoryRow(12, "user1", "trivial fact", "#test", 100.0),
    ]
    result = service.cleanup_memories("user1", "#test", rows)
    assert result.error is None
    assert result.drop == [2]
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — old code would reject this because not every index appears in keep/drop/merge.

**Step 3: Rewrite `cleanup_memories` validation**

Replace the validation section of `cleanup_memories` in `plugins/llm/src/llm/service.py` (from line ~2646 to ~2696). The new validation:

```python
        # Validate structure
        if not isinstance(parsed, dict):
            return CleanupResult(error="Response is not a JSON object")

        drop = parsed.get("drop", [])
        merge = parsed.get("merge", [])

        if not isinstance(drop, list) or not isinstance(merge, list):
            return CleanupResult(error="drop/merge must be arrays")

        num_memories = len(memory_rows)

        # Validate drop indices
        all_indices: list[int] = []
        for idx in drop:
            if not isinstance(idx, int) or idx < 0 or idx >= num_memories:
                return CleanupResult(error=f"Invalid drop index: {idx}")
            all_indices.append(idx)

        # Validate merge entries — each is [list_of_indices, merged_text]
        validated_merge: list[list] = []
        for entry in merge:
            if not isinstance(entry, list) or len(entry) != 2:
                return CleanupResult(error=f"Invalid merge entry: {entry}")
            indices, text = entry
            if not isinstance(indices, list) or len(indices) < 2:
                return CleanupResult(error=f"Merge needs at least 2 indices: {entry}")
            for idx in indices:
                if not isinstance(idx, int) or idx < 0 or idx >= num_memories:
                    return CleanupResult(error=f"Merge index out of range: {entry}")
                all_indices.append(idx)
            if not isinstance(text, str) or not text.strip():
                return CleanupResult(error=f"Merge text must be non-empty: {entry}")
            validated_merge.append([indices, text.strip()])

        # Check for duplicate indices across drop and merge
        if len(all_indices) != len(set(all_indices)):
            return CleanupResult(error="Duplicate index across drop/merge")

        # Ensure at least one memory survives
        surviving = num_memories - len(drop) - sum(len(e[0]) for e in validated_merge) + len(validated_merge)
        if surviving <= 0 and num_memories > 0:
            return CleanupResult(error="Cleanup would leave user with zero memories")

        return CleanupResult(drop=drop, merge=validated_merge)
```

**Step 4: Update the test for duplicate indices**

The old `test_cleanup_rejects_duplicate_indices` used `keep` — update it:

```python
def test_cleanup_rejects_duplicate_indices(self, make_service, mocker: MockerFixture) -> None:
    """GIVEN index in both drop and merge WHEN cleanup THEN returns error."""
    from llm.persistence import MemoryRow

    service, mock_plugin = make_service()
    mock_litellm = mocker.patch("llm.service.litellm")
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = '{"drop": [0], "merge": [[[0, 1], "merged"]]}'
    mock_litellm.completion.return_value = mock_response

    rows = [
        MemoryRow(10, "user1", "fact a", "#test", 100.0),
        MemoryRow(11, "user1", "fact b", "#test", 200.0),
    ]
    result = service.cleanup_memories("user1", "#test", rows)
    assert result.error is not None
```

**Step 5: Update `test_cleanup_rejects_out_of_range_index` for new format**

```python
def test_cleanup_rejects_out_of_range_index(self, make_service, mocker: MockerFixture) -> None:
    """GIVEN out-of-range index WHEN cleanup THEN returns error."""
    from llm.persistence import MemoryRow

    service, mock_plugin = make_service()
    mock_litellm = mocker.patch("llm.service.litellm")
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = '{"drop": [5], "merge": []}'
    mock_litellm.completion.return_value = mock_response

    rows = [
        MemoryRow(10, "user1", "fact a", "#test", 100.0),
        MemoryRow(11, "user1", "fact b", "#test", 200.0),
    ]
    result = service.cleanup_memories("user1", "#test", rows)
    assert result.error is not None
```

**Step 6: Update `test_cleanup_rejects_empty_merge_text` for new format**

```python
def test_cleanup_rejects_empty_merge_text(self, make_service, mocker: MockerFixture) -> None:
    """GIVEN merge with empty text WHEN cleanup THEN returns error."""
    from llm.persistence import MemoryRow

    service, mock_plugin = make_service()
    mock_litellm = mocker.patch("llm.service.litellm")
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = '{"drop": [], "merge": [[[0, 1], ""]]}'
    mock_litellm.completion.return_value = mock_response

    rows = [
        MemoryRow(10, "user1", "fact a", "#test", 100.0),
        MemoryRow(11, "user1", "fact b", "#test", 200.0),
    ]
    result = service.cleanup_memories("user1", "#test", rows)
    assert result.error is not None
```

**Step 7: Run tests to verify they pass**

Run: `make test`
Expected: All pass.

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(memory): resilient cleanup validation with multi-way merges"
```

---

### Task 4: Update cleanup prompt

**Files:**
- Modify: `plugins/llm/src/llm/service.py:94-107`

**Step 1: Write the failing test**

```python
def test_cleanup_prompt_uses_new_format(self) -> None:
    """GIVEN cleanup prompt WHEN checked THEN uses new merge format without keep."""
    from llm.service import _MEMORY_CLEANUP_PROMPT

    assert "keep" not in _MEMORY_CLEANUP_PROMPT.lower()
    assert "Be aggressive" in _MEMORY_CLEANUP_PROMPT
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — current prompt contains "keep".

**Step 3: Replace the cleanup prompt**

Replace `_MEMORY_CLEANUP_PROMPT` in `plugins/llm/src/llm/service.py:94-107` with:

```python
_MEMORY_CLEANUP_PROMPT = (
    "You are a memory curator. Review these stored facts about an IRC user and "
    "return edit operations as JSON.\n\n"
    "Rules:\n"
    "- ONLY reference facts by their index numbers below\n"
    "- Do NOT invent new facts — merge text must combine existing information only\n"
    "- Facts are listed newest-first; when facts contradict, prefer the newer one "
    "(lower index)\n"
    "- Merge related facts into single consolidated statements\n"
    "- Drop jokes, transient info, vague observations, or anything not a durable "
    "fact about the user\n"
    "- Be aggressive — fewer high-quality facts beat many low-quality ones\n\n"
    'Return JSON: {"drop": [...], "merge": [[[idx, idx, ...], "merged text"], ...]}\n'
    "Indices not mentioned in drop or merge are kept as-is.\n"
)
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: All pass.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(memory): update cleanup prompt for aggressive pruning"
```

---

### Task 5: Update `_run_memory_cleanup` in plugin.py for multi-way merges

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1536-1550`

**Step 1: Write the failing test**

Add a test for multi-way merge in `plugins/llm/tests/test_integration.py` in the `TestMemoryCleanup` class:

```python
def test_cleanup_applies_multiway_merge(
    self, plugin_with_real_db: tuple, mocker: MockerFixture
) -> None:
    """GIVEN cleanup returns 3-way merge WHEN applied THEN memories consolidated."""
    from llm.service import CleanupResult

    plugin, mock_irc = plugin_with_real_db

    plugin.db.save_memory("testuser", "uses Arch Linux", "#test")
    plugin.db.save_memory("testuser", "uses Debian", "#test")
    plugin.db.save_memory("testuser", "uses Fedora", "#test")

    # Memories newest-first: [0]=Fedora, [1]=Debian, [2]=Arch
    plugin.llm_service.cleanup_memories = mocker.MagicMock(
        return_value=CleanupResult(drop=[], merge=[[[0, 1, 2], "uses Linux (Arch, Debian, Fedora)"]])
    )

    plugin._run_memory_cleanup("testuser", "#test")

    rows = plugin.db.get_memories("testuser")
    assert len(rows) == 1
    assert rows[0].fact == "uses Linux (Arch, Debian, Fedora)"
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL — current merge loop expects `idx_a, idx_b, merged_text` (3 elements), not `[indices], text` (2 elements).

**Step 3: Update the merge loop in `_run_memory_cleanup`**

In `plugins/llm/src/llm/plugin.py`, replace lines 1541-1550 (the merge application section):

```python
            # Apply merges: delete sources, insert merged fact
            for entry in result.merge:
                indices, merged_text = entry
                sources = [snapshot[i] for i in indices if 0 <= i < len(snapshot)]
                if len(sources) < 2:
                    continue
                oldest = min(sources, key=lambda s: s.created_at)
                for source in sources:
                    self.db.delete_memory(nick, source.id)
                self.db.save_memory(nick, merged_text, oldest.source_channel)
```

**Step 4: Update existing `test_cleanup_applies_merge` for new format**

In `plugins/llm/tests/test_integration.py`, update the existing merge test (line ~727):

```python
def test_cleanup_applies_merge(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
    """GIVEN cleanup returns merge WHEN applied THEN memories are merged."""
    from llm.service import CleanupResult

    plugin, mock_irc = plugin_with_real_db

    plugin.db.save_memory("testuser", "likes Python programming", "#test")
    plugin.db.save_memory("testuser", "enjoys writing Python", "#test")

    plugin.llm_service.cleanup_memories = mocker.MagicMock(
        return_value=CleanupResult(drop=[], merge=[[[0, 1], "likes Python programming"]])
    )

    plugin._run_memory_cleanup("testuser", "#test")

    rows = plugin.db.get_memories("testuser")
    assert len(rows) == 1
    assert rows[0].fact == "likes Python programming"
```

**Step 5: Run tests to verify they pass**

Run: `make test`
Expected: All pass.

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_integration.py
git commit -m "feat(memory): support multi-way merges in cleanup"
```

---

### Task 6: Remove legacy array format from extraction

**Files:**
- Modify: `plugins/llm/src/llm/service.py:2584-2587`
- Modify: `plugins/llm/tests/test_service.py` (remove legacy test)

**Step 1: Remove the legacy array format handler**

In `plugins/llm/src/llm/service.py`, remove lines 2584-2587:

```python
            # Accept both {"add": [...], "remove": [...]} and legacy [...] format
            if isinstance(parsed, list):
                add = [f for f in parsed if isinstance(f, str)]
                return ExtractionResult(add=add)
```

**Step 2: Remove `test_extract_memories_legacy_array_format`**

Delete the test at `plugins/llm/tests/test_service.py:3162-3174`.

**Step 3: Run tests to verify they pass**

Run: `make test`
Expected: All pass.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "refactor(memory): remove legacy array format from extraction"
```

---

### Task 7: Run preflight and verify

**Step 1: Run full preflight**

Run: `make preflight`
Expected: All checks pass (format, lint, typecheck, tests).

**Step 2: Verify no regressions in test count**

Run: `make test 2>&1 | tail -5`
Expected: Test count should be close to the original (±2 from added/removed tests), all passing, coverage >= 80%.
