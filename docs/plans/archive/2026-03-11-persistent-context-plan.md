# Persistent Conversation Context Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Persist per-user conversation context to SQLite so it survives bot restarts.

**Architecture:** Add a `conversations` table to the existing SQLite database. Inject an optional `LLMDatabase` reference into `ConversationContext`. Write-through on command interactions; skip persistence for passively observed IRC messages via a `persist` flag. Load all non-expired conversations at startup.

**Tech Stack:** Python 3.12+, SQLite (existing `LLMDatabase`), JSON serialization, pytest

**Design doc:** `docs/plans/2026-03-11-persistent-context-design.md`

---

### Task 1: Schema Migration — Add `conversations` Table

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:16` (SCHEMA_VERSION)
- Modify: `plugins/llm/src/llm/persistence.py:237-258` (migration block)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing test**

Add to `test_persistence.py` in the `TestDatabaseInit` class:

```python
def test_creates_conversations_table(self, tmp_path: Path) -> None:
    """GIVEN a new database WHEN initialized THEN conversations table exists."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        )
        assert cursor.fetchone() is not None
    finally:
        conn.close()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest plugins/llm/tests/test_persistence.py::TestDatabaseInit::test_creates_conversations_table -v`
Expected: FAIL — conversations table does not exist

**Step 3: Write minimal implementation**

In `persistence.py`, bump the version:

```python
SCHEMA_VERSION = 4
```

In `_migrate()`, after the `if current_version < 3:` block (after line 253 `conn.commit()`) and before the `PRAGMA user_version` stamp, add:

```python
if current_version < 4:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            nick TEXT NOT NULL,
            channel TEXT NOT NULL,
            messages TEXT NOT NULL,
            last_activity REAL NOT NULL,
            PRIMARY KEY (nick, channel)
        );
    """)
    conn.commit()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest plugins/llm/tests/test_persistence.py::TestDatabaseInit -v`
Expected: all PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: all checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add conversations table migration (schema v4)"
```

---

### Task 2: Persistence Methods — `save_conversation`, `delete_conversation`, `delete_all_conversations`, `load_conversations`

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (add 4 methods to `LLMDatabase`)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing tests**

Add a new test class to `test_persistence.py`:

```python
class TestConversationPersistence:
    """Test conversation persistence methods."""

    def test_save_and_load_conversation(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN load_conversations THEN it is returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        db.save_conversation("User1", "#Channel", messages, 1000.0)

        loaded = db.load_conversations()
        assert len(loaded) == 1
        nick, channel, msgs, last_activity = loaded[0]
        assert nick == "user1"
        assert channel == "#channel"
        assert msgs == messages
        assert last_activity == 1000.0

    def test_save_conversation_upserts(self, tmp_path: Path) -> None:
        """GIVEN an existing conversation WHEN saved again THEN it is replaced."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "first"}], 1000.0)
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "second"}], 2000.0)

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][2] == [{"role": "user", "content": "second"}]
        assert loaded[0][3] == 2000.0

    def test_save_lowercases_nick_and_channel(self, tmp_path: Path) -> None:
        """GIVEN mixed-case nick/channel WHEN saved THEN stored lowercased."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("UserName", "#MyChannel", [], 1000.0)

        loaded = db.load_conversations()
        assert loaded[0][0] == "username"
        assert loaded[0][1] == "#mychannel"

    def test_delete_conversation(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN deleted THEN load returns empty."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [{"role": "user", "content": "hi"}], 1000.0)
        db.delete_conversation("user1", "#chan")

        loaded = db.load_conversations()
        assert len(loaded) == 0

    def test_delete_conversation_lowercases(self, tmp_path: Path) -> None:
        """GIVEN a saved conversation WHEN deleted with different case THEN still deleted."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [], 1000.0)
        db.delete_conversation("User1", "#Chan")

        assert len(db.load_conversations()) == 0

    def test_delete_all_conversations(self, tmp_path: Path) -> None:
        """GIVEN multiple conversations WHEN delete_all THEN all are removed."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation("user1", "#chan", [], 1000.0)
        db.save_conversation("user2", "#chan", [], 1000.0)
        db.delete_all_conversations()

        assert len(db.load_conversations()) == 0

    def test_load_skips_corrupt_json(self, tmp_path: Path) -> None:
        """GIVEN a row with invalid JSON WHEN load THEN it is skipped."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # Insert valid row
        db.save_conversation("good", "#chan", [{"role": "user", "content": "ok"}], 1000.0)
        # Manually insert corrupt row
        conn = db._connect()
        conn.execute(
            "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
            "VALUES (?, ?, ?, ?)",
            ("bad", "#chan", "NOT VALID JSON{{{", 1000.0),
        )
        conn.commit()

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][0] == "good"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_persistence.py::TestConversationPersistence -v`
Expected: FAIL — methods do not exist

**Step 3: Write the implementation**

Add these methods to `LLMDatabase` (after `close`/`__del__`, before the reminders section):

```python
# ------------------------------------------------------------------
# Conversation persistence
# ------------------------------------------------------------------

def save_conversation(
    self,
    nick: str,
    channel: str,
    messages: list[dict[str, str]],
    last_activity: float,
) -> None:
    """Persist a conversation's messages to the database.

    Args:
        nick: User's IRC nick (lowercased before storage).
        channel: IRC channel (lowercased before storage).
        messages: List of message dicts (role + content).
        last_activity: Timestamp of last activity.
    """
    import json

    conn = self._connect()
    conn.execute(
        "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
        "VALUES (?, ?, ?, ?)",
        (nick.lower(), channel.lower(), json.dumps(messages), last_activity),
    )
    conn.commit()

def delete_conversation(self, nick: str, channel: str) -> None:
    """Delete a conversation from the database.

    Args:
        nick: User's IRC nick.
        channel: IRC channel.
    """
    conn = self._connect()
    conn.execute(
        "DELETE FROM conversations WHERE nick = ? AND channel = ?",
        (nick.lower(), channel.lower()),
    )
    conn.commit()

def delete_all_conversations(self) -> None:
    """Delete all conversations from the database."""
    conn = self._connect()
    conn.execute("DELETE FROM conversations")
    conn.commit()

def load_conversations(self) -> list[tuple[str, str, list[dict[str, str]], float]]:
    """Load all conversations from the database.

    Returns:
        List of (nick, channel, messages, last_activity) tuples.
        Rows with corrupt JSON are logged and skipped.
    """
    import json
    import logging

    log = logging.getLogger("supybot.plugins.LLM")
    conn = self._connect()
    rows = conn.execute(
        "SELECT nick, channel, messages, last_activity FROM conversations"
    ).fetchall()

    result: list[tuple[str, str, list[dict[str, str]], float]] = []
    for nick, channel, messages_json, last_activity in rows:
        try:
            messages = json.loads(messages_json)
        except (json.JSONDecodeError, TypeError):
            log.warning("Skipping corrupt conversation for %s/%s", nick, channel)
            conn.execute(
                "DELETE FROM conversations WHERE nick = ? AND channel = ?",
                (nick, channel),
            )
            conn.commit()
            continue
        result.append((nick, channel, messages, last_activity))
    return result
```

Move the `import json` to the top of `persistence.py` (alongside the other stdlib imports) instead of using inline imports.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_persistence.py::TestConversationPersistence -v`
Expected: all PASS

**Step 5: Run preflight**

Run: `make preflight`
Expected: all checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add conversation persistence methods to LLMDatabase"
```

---

### Task 3: Wire Persistence Into `ConversationContext`

**Files:**
- Modify: `plugins/llm/src/llm/context.py:48-72` (constructor + `add_message` + `clear` + `clear_all` + `_prune_expired`)
- Test: `plugins/llm/tests/test_context.py`

**Step 1: Write the failing tests**

Add a new test class to `test_context.py`:

```python
import json
import sqlite3
from pathlib import Path

from llm.persistence import LLMDatabase


class TestPersistentContext:
    """Test conversation context with SQLite persistence."""

    def _make_ctx(self, tmp_path: Path) -> tuple[ConversationContext, LLMDatabase]:
        db = LLMDatabase(str(tmp_path / "test.db"))
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=db)
        return ctx, db

    def test_add_message_persists_to_db(self, tmp_path: Path) -> None:
        """GIVEN context with db WHEN add_message THEN conversation is in SQLite."""
        ctx, db = self._make_ctx(tmp_path)
        ctx.add_message("user1", "#chan", "user", "Hello")

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][2] == [{"role": "user", "content": "Hello"}]

    def test_add_message_persist_false_skips_db(self, tmp_path: Path) -> None:
        """GIVEN context with db WHEN add_message(persist=False) THEN not in SQLite."""
        ctx, db = self._make_ctx(tmp_path)
        ctx.add_message("user1", "#chan", "user", "Hello", persist=False)

        loaded = db.load_conversations()
        assert len(loaded) == 0

        # But still in memory
        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1

    def test_clear_deletes_from_db(self, tmp_path: Path) -> None:
        """GIVEN persisted conversation WHEN clear THEN removed from SQLite."""
        ctx, db = self._make_ctx(tmp_path)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.clear("user1", "#chan")

        assert len(db.load_conversations()) == 0

    def test_clear_all_deletes_from_db(self, tmp_path: Path) -> None:
        """GIVEN persisted conversations WHEN clear_all THEN all removed from SQLite."""
        ctx, db = self._make_ctx(tmp_path)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.add_message("user2", "#chan", "user", "Hi")
        ctx.clear_all()

        assert len(db.load_conversations()) == 0

    def test_startup_loads_from_db(self, tmp_path: Path) -> None:
        """GIVEN conversations in db WHEN new ConversationContext THEN loaded into memory."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_conversation(
            "user1", "#chan",
            [{"role": "user", "content": "Hello"}],
            time.time(),
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    def test_startup_skips_expired(self, tmp_path: Path) -> None:
        """GIVEN expired conversation in db WHEN new ConversationContext THEN not loaded."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        # last_activity is 2 hours ago, timeout is 30 minutes
        old_time = time.time() - 7200
        db.save_conversation(
            "user1", "#chan",
            [{"role": "user", "content": "Hello"}],
            old_time,
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 0

    def test_without_db_works_unchanged(self, tmp_path: Path) -> None:
        """GIVEN context without db WHEN operations THEN works as before."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#chan", "user", "Hello")
        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1
        ctx.clear("user1", "#chan")
        assert ctx.get_messages("user1", "#chan") == []
```

Add `import time` to the top if not already there.

**Step 2: Run tests to verify they fail**

Run: `uv run pytest plugins/llm/tests/test_context.py::TestPersistentContext -v`
Expected: FAIL — `ConversationContext` does not accept `db` parameter

**Step 3: Write the implementation**

Modify `context.py`:

Add at the top, under existing imports:

```python
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persistence import LLMDatabase
```

Modify `ConversationContext.__init__` to accept and use `db`:

```python
def __init__(self, config: ContextConfig, *, db: LLMDatabase | None = None) -> None:
    self.config = config
    self._db = db
    self._lock = Lock()
    self._conversations: dict[tuple[str, str], Conversation] = {}
    self._channel_contexts: dict[str, Conversation] = {}
    self._last_prune: float = 0.0

    if self._db is not None:
        self._load_from_db()
```

Add `_load_from_db` method:

```python
def _load_from_db(self) -> None:
    """Load persisted conversations from the database at startup."""
    assert self._db is not None
    log = logging.getLogger("supybot.plugins.LLM")
    rows = self._db.load_conversations()
    timeout_seconds = self.config.timeout_minutes * 60
    now = time.time()
    loaded = 0
    for nick, channel, messages, last_activity in rows:
        if now - last_activity > timeout_seconds:
            self._db.delete_conversation(nick, channel)
            continue
        key = (nick, channel)  # already lowercased by load_conversations
        self._conversations[key] = Conversation(
            messages=messages, last_activity=last_activity
        )
        loaded += 1
    if loaded:
        log.info("Loaded %d conversation(s) from database", loaded)
```

Modify `add_message` signature — add `persist: bool = True` parameter:

```python
def add_message(
    self,
    nick: str,
    channel: str,
    role: str,
    content: str,
    *,
    config: ContextConfig | None = None,
    persist: bool = True,
) -> None:
```

At the end of `add_message`, after the existing trimming logic, add:

```python
    if persist and self._db is not None:
        self._db.save_conversation(
            nick, channel, conv.messages, conv.last_activity
        )
```

Modify `clear` — after `del self._conversations[key]` and before `return True`:

```python
    if self._db is not None:
        self._db.delete_conversation(nick, channel)
```

Modify `clear_all` — after `self._channel_contexts.clear()`:

```python
    if self._db is not None:
        self._db.delete_all_conversations()
```

Modify `_prune_expired` — change `cfg` to `self.config` for the expiry check, and add DB cleanup. Replace the existing method body:

```python
def _prune_expired(self, cfg: ContextConfig, *, force: bool = False) -> None:
    now = time.time()
    if not force and now - self._last_prune < _PRUNE_INTERVAL:
        return
    self._last_prune = now

    # Use instance default config for prune sweep to avoid
    # cross-channel config mismatch (see design doc).
    prune_cfg = self.config
    expired_keys = [
        key for key, conv in self._conversations.items()
        if self._is_expired(conv, prune_cfg)
    ]
    for key in expired_keys:
        del self._conversations[key]
        if self._db is not None:
            self._db.delete_conversation(key[0], key[1])

    expired_channels = [
        ch for ch, ctx in self._channel_contexts.items()
        if self._is_expired(ctx, prune_cfg)
    ]
    for ch in expired_channels:
        del self._channel_contexts[ch]
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest plugins/llm/tests/test_context.py -v`
Expected: all PASS (both old and new tests)

**Step 5: Run preflight**

Run: `make preflight`
Expected: all checks pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/context.py plugins/llm/tests/test_context.py
git commit -m "feat: add SQLite persistence to ConversationContext"
```

---

### Task 4: Plugin Wiring — Init Order + Pass DB + persist=False in doPrivmsg

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:314-321` (init order)
- Modify: `plugins/llm/src/llm/plugin.py:742-753` (`_init_context`)
- Modify: `plugins/llm/src/llm/plugin.py:649` (doPrivmsg persist=False)
- Test: `plugins/llm/tests/test_integration.py` (or existing test that covers startup)

**Step 1: Modify `plugin.py` init order**

In `__init__`, move the database initialization **before** `_init_context()`. Change lines 314-321 from:

```python
# Initialize conversation context
self._init_context()

# Initialize database for persistence
db_path = self.registryValue("databasePath")
if not db_path:
    db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
self.db = LLMDatabase(db_path)
```

To:

```python
# Initialize database for persistence (before context, which loads from DB)
db_path = self.registryValue("databasePath")
if not db_path:
    db_path = str(Path(conf.supybot.directories.data()) / "LLM.db")
self.db = LLMDatabase(db_path)

# Initialize conversation context (loads persisted conversations from DB)
self._init_context()
```

**Step 2: Modify `_init_context` to pass db**

Change line 753 from:

```python
self.context = ConversationContext(config)
```

To:

```python
self.context = ConversationContext(config, db=self.db)
```

**Step 3: Add persist=False to doPrivmsg**

Change line 649 from:

```python
self.context.add_message(nick, channel, Role.USER, message_text, config=ctx_cfg)
```

To:

```python
self.context.add_message(
    nick, channel, Role.USER, message_text, config=ctx_cfg, persist=False
)
```

**Step 4: Run preflight**

Run: `make preflight`
Expected: all checks pass. Existing tests still work because test fixtures mock `registryValue` and `ConversationContext` is created with `db=None` in test helpers.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "feat: wire persistent context into plugin startup and commands"
```

---

### Task 5: Final Verification + Push + Deploy

**Step 1: Full preflight**

Run: `make preflight`
Expected: all checks pass, coverage >= 80%

**Step 2: Push and wait for CI**

```bash
git push
make wait-ci
```

**Step 3: Wait for Docker build**

```bash
gh run list --workflow=docker.yml --limit 1 --json databaseId,status
gh run watch <run_id> --exit-status
```

**Step 4: Deploy**

```bash
ssh vibebot@rdrake.org "systemctl --user restart vibebot"
```

**Step 5: Verify**

Check the logs to confirm conversations loaded:

```bash
ssh vibebot@rdrake.org "tail -50 ~/vibebot-v8/logs/messages.log"
```
