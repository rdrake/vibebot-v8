# Persistence Error-Handling Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `try/finally: pass` shells in `plugins/llm/src/llm/persistence.py` with real rollback handling on every transactional write, drop the meaningless wrappers from reads, and stop hiding the `lastrowid is None` invariant behind silent `or 0`.

**Architecture:** The file currently has 32 `try:` and 30 `finally:` keywords; only one site (`claim_due_pending_tasks`, line 955) actually performs `conn.rollback()`, and that site uses an explicit `BEGIN IMMEDIATE` transaction it must keep managing itself. Every other "transactional" write is wrapped in `try: ... finally: pass`, which is a no-op. This plan introduces one private helper `_write_txn` that wraps the standard implicit-transaction commit/rollback/raise pattern, applies it to every write that does not manage its own `BEGIN IMMEDIATE`, removes the noise wrapper from reads, and replaces `cursor.lastrowid or 0` with an explicit invariant.

**Tech Stack:** Python 3.12+, `sqlite3` stdlib, pytest, `make lint` / `make typecheck` / `make test` per AGENTS.md.

**Pre-flight expectations:** All tasks must end with the workspace tests green (`make test`) before commit. Tasks that change behavior add coverage; tasks that only remove no-ops keep coverage flat. Coverage floor is **93%** (per `pyproject.toml`), not 80%.

**Carve-out:** `claim_due_pending_tasks` (line 955) opens its own `BEGIN IMMEDIATE` and already has a real `conn.rollback()` on exception. **Do not convert this site to `_write_txn`.** `_write_txn` commits the implicit transaction; layering it over an explicit `BEGIN IMMEDIATE` is incorrect.

---

### Task 1: Add `_write_txn` context-manager helper

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (insert near `_connect`, ~line 150)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Add the helper imports**

At the top of `persistence.py`, ensure both `contextmanager` and `Iterator` are imported:

```python
from collections.abc import Iterator
from contextlib import contextmanager
```

(`Iterator` may already be imported via `typing`; prefer `collections.abc` for new code per modern-python skill.)

**Step 2: Write the failing test**

Append to `test_persistence.py`. The fixture `tmp_path` and the existing `LLMDatabase` constructor are already in the file's test patterns; mirror them:

```python
def test_write_txn_rolls_back_on_error(tmp_path):
    db = LLMDatabase(str(tmp_path / "t.db"))
    db.save_memory("alice", "fact-1", "#chan")

    # Trigger an IntegrityError mid-block by violating a UNIQUE/PK constraint.
    with pytest.raises(sqlite3.IntegrityError):
        with db._write_txn() as conn:
            conn.execute(
                "INSERT INTO memories (id, nick, fact, source_channel, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (1, "bob", "x", "#c", 0.0),
            )

    # Existing row still present; bob's row never committed.
    rows = db.get_memories("alice")
    assert len(rows) == 1
    assert db.get_memories("bob") == []


def test_write_txn_commits_on_success(tmp_path):
    db = LLMDatabase(str(tmp_path / "t.db"))
    with db._write_txn() as conn:
        conn.execute(
            "INSERT INTO memories (nick, fact, source_channel, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("alice", "fact-x", "#c", 0.0),
        )
    assert any(r.fact == "fact-x" for r in db.get_memories("alice"))
```

**Step 3: Run, confirm fail**

```bash
uv run pytest plugins/llm/tests/test_persistence.py::test_write_txn_rolls_back_on_error -v
```
Expected: FAIL — `_write_txn` does not exist.

**Step 4: Implement `_write_txn`**

Add to `LLMDatabase` (just below `_connect`, before `_migrate`):

```python
@contextmanager
def _write_txn(self) -> Iterator[sqlite3.Connection]:
    """Yield a connection, commit on clean exit, rollback+raise on exception.

    Use for any write that relies on sqlite3's implicit transaction. Callers
    that issue their own ``BEGIN IMMEDIATE`` (e.g. ``claim_due_pending_tasks``)
    must NOT use this helper -- they manage commit/rollback themselves.

    Reads should not use this helper; it adds a commit on every successful
    SELECT, which is harmless but obscures intent.
    """
    conn = self._connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

**Step 5: Run new tests, confirm pass**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -k write_txn -v
```
Expected: PASS for both new tests.

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(persistence): add _write_txn context manager for write rollback"
```

---

### Task 2: Adopt `_write_txn` in INSERT writes that return `lastrowid`

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` — `save_reminder` (~572-597), `save_scheduled_llm_task` (~736-760), `save_pending_task` (~891-953), `save_memory` (~1679-1699)

**Step 1: Replace each block per this pattern**

```python
# Before
conn = self._connect()
try:
    cursor = conn.execute("INSERT INTO ...", (...))
    conn.commit()
    return cursor.lastrowid or 0
finally:
    pass

# After
with self._write_txn() as conn:
    cursor = conn.execute("INSERT INTO ...", (...))
    assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
    return cursor.lastrowid
```

Apply at lines 595, 760, 951, 1697 — the four `cursor.lastrowid or 0` sites.

**Step 2: Run persistence tests**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```
Expected: PASS.

**Step 3: Run linters and types**

```bash
make lint && make typecheck
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): use _write_txn for INSERT writes; drop lastrowid sentinel"
```

---

### Task 3: Adopt `_write_txn` in remaining writes (UPDATE/DELETE/REPLACE)

**Files & return-type inventory:** apply per the table below. Each function currently sits in a `try/finally: pass` shell (or an unwrapped commit). Match the existing return type — do not coerce to `bool`.

| Method | Line | Returns |
|---|---|---|
| `delete_reminder` | 599 | `bool` (`rowcount > 0`) |
| `delete_expired_reminders` | 661 | `int` (`rowcount`) |
| `update_scheduled_llm_task_fire_at` | 762 | `None` |
| `delete_scheduled_llm_task` | 784 | `bool` |
| `release_pending_task` | 1029 | `bool` |
| `delete_pending_task` | 1069 | `bool` |
| `update_task_for_delivery` | 1089 | `bool` |
| `update_delivery_attempt` | 1118 | `bool` |
| `delete_expired_pending_tasks` | 1158 | `list[PendingTaskRow]` (SELECT then DELETE — keep both inside one `_write_txn`) |
| `migrate_nick` | 1246 | `int` |
| `log_usage` | 1320 | `None` |
| `delete_memory` | 1721 | `bool` |
| `update_memory` | 1744 | `bool` |
| `delete_all_memories` | 1768 | `int` |
| `increment_memory_saves` | 1788 | `int` |
| `reset_memory_saves` | 1812 | `None` |
| `save_instruction` | 1860 | `None` |
| `delete_instruction` | 1872 | `bool` |

**Carve-out:** `claim_due_pending_tasks` (955) is exempt — it manages its own `BEGIN IMMEDIATE`/`rollback`. `save_conversation` and `delete_conversation` should be checked: if either currently uses `try/finally: pass`, include them and add a row to this table.

**Step 1: Per-method transformation**

```python
# Before
conn = self._connect()
try:
    cursor = conn.execute("UPDATE/DELETE ...", (...))
    conn.commit()
    return cursor.rowcount > 0      # or rowcount, or None, or list/dict
finally:
    pass

# After (preserve the existing return expression)
with self._write_txn() as conn:
    cursor = conn.execute("UPDATE/DELETE ...", (...))
    return cursor.rowcount > 0
```

For methods returning `None`, omit the `return` (or keep an explicit `return` after the `with` block). For `delete_expired_pending_tasks`, keep both the SELECT and the DELETE inside one `with self._write_txn():` block so rollback covers the pair.

**Step 2: Verify existing tests pass**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```

**Step 3: Skip the brittle release_pending_task rollback test**

The original draft proposed monkeypatching `db._connect().execute` — that patches a returned object reference, which is fragile and produces false passes. **Do not add this test.** The `_write_txn` semantics are already proven by Task 1's tests; per-method rollback tests give little additional confidence.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): use _write_txn for UPDATE/DELETE/INSERT writes"
```

---

### Task 4: Make `migrate_conversations` atomic via `_write_txn`

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:1274-1318`

**Step 1: Wrap the DELETE+UPDATE pair**

```python
# Before
conn = self._connect()
try:
    conn.execute("DELETE FROM conversations WHERE ...", (...))
    cursor = conn.execute("UPDATE conversations SET nick = ? ...", (...))
    conn.commit()
    return cursor.rowcount
finally:
    pass

# After
with self._write_txn() as conn:
    conn.execute("DELETE FROM conversations WHERE ...", (...))
    cursor = conn.execute("UPDATE conversations SET nick = ? ...", (...))
    return cursor.rowcount
```

**Step 2: Add a code comment instead of a brittle monkeypatch test**

Above the `with` block, add:

```python
# DELETE and UPDATE share one transaction so a failure in either
# rolls back both rather than leaving conversations orphaned.
```

The original draft proposed a monkeypatch-based regression test; mid-transaction `execute` patching on `sqlite3.Connection` is unreliable. The atomicity is now visible from the code structure; skip the test.

**Step 3: Run tests**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): make migrate_conversations atomic via _write_txn"
```

---

### Task 5: Drop `try/finally: pass` from read-only methods

**Prerequisite:** Tasks 2–4 must complete first. After those, `grep -n 'finally:' persistence.py` returns only read methods (the writes have moved to `_write_txn`). If you run Task 5 before Task 2/3/4, you risk stripping wrappers from writes that haven't yet been converted.

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` — every read with the empty wrapper.

Likely sites (verify with `grep -n 'finally:' plugins/llm/src/llm/persistence.py` after Tasks 2-4):
- `load_conversations` (495)
- `load_pending_reminders` (619)
- `load_active_scheduled_llm_tasks` (793)
- `get_scheduled_llm_task` (804)
- `load_scheduled_llm_tasks_for_target` (814)
- `load_scheduled_llm_tasks_for` (832)
- `count_scheduled_llm_tasks_for` (856)
- `load_pending_tasks` (1215)
- `get_next_due_time` (1190)
- `get_usage_summary` (1374), `get_usage_by_nick` (1415), `get_usage_by_channel` (1427), `_get_usage_by_dimension` (1441), `get_usage_summary_for_channel` (1488), `get_usage_summary_for_nick` (1527), `get_channel_rank` (1568), `get_nick_rank` (1580), `_get_rank` (1595)
- `get_memories` (1701), `get_memory_saves` (1828), `get_instruction` (1851)

**Step 1: For each, simplify**

```python
# Before
conn = self._connect()
try:
    rows = conn.execute("SELECT ...", (...)).fetchall()
    return [Row(*r) for r in rows]
finally:
    pass

# After
conn = self._connect()
rows = conn.execute("SELECT ...", (...)).fetchall()
return [Row(*r) for r in rows]
```

**Step 2: Sweep stale "NickServ" terminology in docstrings**

While in the file, replace any "NickServ account" or "NickServ" mention in docstrings with "authenticated" or "account" per AGENTS.md / project convention. Likely sites: lines 818, 1251, 1290 — verify with:

```bash
grep -n -i "nickserv" plugins/llm/src/llm/persistence.py
```

**Step 3: Run full test suite**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```

**Step 4: Verify no `try: ... finally: pass` shells remain**

```bash
python -c "
import re, pathlib
src = pathlib.Path('plugins/llm/src/llm/persistence.py').read_text()
hits = re.findall(r'try:\s*\n(?:.*\n)*?\s*finally:\s*\n\s*pass\s*\n', src)
print(len(hits))
"
```
Expected: 0.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): drop try/finally:pass shells from reads; sweep nickserv docstrings"
```

---

### Task 6: Standardize `lastrowid` and `COUNT(*)` defensiveness

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:878` (the lone `int(row[0] if row else 0)` in `count_scheduled_llm_tasks_for`)

**Step 1: Simplify the COUNT site**

```python
# Before
return int(row[0] if row else 0)

# After  (a COUNT(*) query always returns a single row of one int)
return row[0]
```

**Step 2: Verify `lastrowid` sites already converted**

```bash
grep -n "lastrowid" plugins/llm/src/llm/persistence.py
```
Expected: only `assert cursor.lastrowid is not None` and `return cursor.lastrowid` lines from Task 2; no `or 0`.

**Step 3: Run tests**

```bash
make test
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): drop unreachable defensiveness around COUNT and lastrowid"
```

---

### Task 7: Final preflight and coverage check

**Step 1: Run full preflight**

```bash
make preflight
```
Expected: PASS, with coverage at or above the 93% floor enforced in `pyproject.toml`.

**Step 2: If coverage dropped below floor**

Add focused tests for any uncovered lines in the modified persistence functions. Do not lower the floor.
