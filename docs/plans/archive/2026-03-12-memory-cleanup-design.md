# Periodic Memory Cleanup Design

**Date:** 2026-03-12
**Status:** Approved (revised after Codex review)

## Problem

Memories accumulate quickly and degrade in quality over time:

- **Duplicates** — "likes Python" and "enjoys Python programming" coexist
- **Contradictions** — "lives in Toronto" and "lives in Vancouver" both persist
  when real-time extraction misses the conflict
- **Staleness** — "working on project X" from months ago is no longer relevant
- **Low quality** — vague or transient facts slip through extraction

## Design

### Trigger

Cleanup runs **per-user** after every N new memory saves (default: 3). A
monotonic counter `memory_saves_since_cleanup` is tracked per-nick in the
`memory_cleanup_state` table. After each save, increment the counter; when
it reaches the configured interval, schedule cleanup and reset the counter
only after successful completion.

This avoids the modulo drift problem where deletions during cleanup could
cause premature or missed triggers.

### LLM output format: index-based edits

Instead of returning a full replacement list (which risks data loss from
hallucination and loses metadata), the LLM returns **edit operations**
referencing input indices:

```json
{
  "keep": [0, 3, 5],
  "drop": [4],
  "merge": [[1, 2, "likes Python programming"]]
}
```

Rules:
- Every input index must appear in exactly one of `keep`, `drop`, or as a
  source in `merge`
- `merge` entries are `[idx_a, idx_b, "merged text"]` — the merged text
  replaces both source memories (oldest `created_at` and `source_channel`
  are preserved)
- Unaccounted indices are kept (fail-safe)

### Validation (fail closed)

Before applying any edits, validate:
1. Response parses as JSON with the expected schema
2. Every index in `drop` and `merge` sources is valid (0 <= idx < len)
3. No index appears in more than one category
4. `merge` texts are non-empty strings
5. The result would not leave the user with zero memories (unless they
   started with zero)

If validation fails, log the error and make **no DB changes**.

### Race condition protection

- **In-flight guard**: A `set` of nicks currently being cleaned tracks
  active cleanups. If a cleanup is already running for a nick, skip.
- **Snapshot check**: Before applying edits, re-fetch the current memory
  count. If it differs from the snapshot count (extraction added a memory
  during the LLM call), abort this cleanup. The counter will naturally
  trigger another cleanup soon.

### Metadata preservation

- `keep` — no DB changes, memories are untouched
- `drop` — delete by ID
- `merge` — delete source IDs, insert one new row preserving the earliest
  `created_at` and `source_channel` from the sources

### Cleanup prompt

```
You are a memory curator. Review these stored facts about an IRC user and
return edit operations as JSON.

Rules:
- ONLY reference facts by their index numbers below
- Do NOT invent new facts — merge text must combine existing information only
- Facts are listed newest-first; when facts contradict, prefer the newer one
  (lower index)
- Merge near-duplicates into one clear statement
- Drop vague, trivial, or clearly transient/time-bound facts
- Keep all genuinely useful long-term information

Return JSON: {"keep": [...], "drop": [...], "merge": [[idx_a, idx_b, "text"], ...]}
Every index must appear in exactly one category.

Current memories:
[0] moved to Vancouver last month
[1] likes Python programming
[2] enjoys writing Python code
[3] works at Acme Corp
[4] asked about the weather
[5] lives in Toronto
```

Expected: `{"keep": [0, 3], "drop": [4, 5], "merge": [[1, 2, "likes Python programming"]]}`

### Database changes

**New table** (`memory_cleanup_state`):
```sql
CREATE TABLE IF NOT EXISTS memory_cleanup_state (
    nick TEXT PRIMARY KEY,
    saves_since_cleanup INTEGER NOT NULL DEFAULT 0
);
```

**New methods on `LLMDatabase`**:
- `increment_memory_saves(nick) -> int` — increment and return new count
- `reset_memory_saves(nick)` — set counter to 0
- `get_memory_saves(nick) -> int` — read current counter

### Configuration

| Key | Type | Default | Scope |
|-----|------|---------|-------|
| `memoryCleanupInterval` | NonNegativeInteger | 3 | Global |

Set to 0 to disable periodic cleanup.

### Integration points

**In `_extract_memories_bg()`** — after saving new memories:
```python
new_count = self.db.increment_memory_saves(nick)
interval = self.registryValue("memoryCleanupInterval")
if interval and new_count >= interval:
    self._schedule_memory_cleanup(nick, channel)
```

**New method `_schedule_memory_cleanup()`** — schedules a background event
that calls `cleanup_memories()` on the service.

**New method on `LLMService`** — `cleanup_memories(nick, channel)`:
1. Check in-flight guard
2. Fetch all memories (snapshot)
3. If fewer than 2 memories, skip (nothing to clean)
4. Call ask model with cleanup prompt
5. Validate response
6. Re-check memory count matches snapshot
7. Apply edits (delete dropped, delete+insert merged)
8. Reset saves counter
9. Release in-flight guard

### Error handling

- LLM timeout/error: log, no DB changes, counter not reset (retry next cycle)
- Invalid JSON / failed validation: log, no DB changes, counter not reset
- DB lock contention: SQLite WAL + thread-local connections handle this

### Cost

Each cleanup is one LLM call to the ask model. With default interval of 3,
a very active user triggering 9 new memories gets 3 cleanup calls. The prompt
is small (memory list + instructions, typically under 1K tokens).
