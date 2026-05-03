# Long-Term Memory & Spontaneous Participation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give the bot permanent memory of user facts and the ability to occasionally participate in channel conversations unprompted.

**Architecture:** Two independent features sharing the same persistence layer. Memory extraction runs as a background side-call after command interactions. Spontaneous participation hooks into the existing `doPrivmsg` passive tracking with a random trigger and cooldown.

**Tech Stack:** SQLite (existing), LiteLLM (existing), Limnoria schedule API (existing)

---

### Task 1: Schema Migration v4->v5 — `memories` Table

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py`
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing tests**

Add `MemoryRow` NamedTuple and tests in `TestDatabaseInit`:

```python
class MemoryRow(NamedTuple):
    id: int
    nick: str
    fact: str
    source_channel: str
    created_at: float
```

```python
def test_creates_memories_table(self, tmp_path: Path) -> None:
    """GIVEN fresh database WHEN initialized THEN memories table exists."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "memories" in tables

def test_memories_table_has_nick_index(self, tmp_path: Path) -> None:
    """GIVEN fresh database WHEN initialized THEN idx_memories_nick exists."""
    db = LLMDatabase(str(tmp_path / "test.db"))
    conn = db._connect()
    indexes = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()}
    assert "idx_memories_nick" in indexes
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_persistence.py::TestDatabaseInit::test_creates_memories_table -v`
Expected: FAIL — "memories" not in tables

**Step 3: Implement schema migration**

In `persistence.py`:
- Add `MemoryRow` NamedTuple after existing NamedTuples
- Bump `SCHEMA_VERSION = 5`
- Add migration block in `_migrate()`:

```python
if current_version < 5:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nick TEXT NOT NULL,
            fact TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_nick ON memories(nick);
    """)
    conn.commit()
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_persistence.py::TestDatabaseInit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add memories table with schema migration v4->v5"
```

---

### Task 2: Memory CRUD Methods

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py`
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing tests**

Add `TestMemoryPersistence` class:

```python
class TestMemoryPersistence:
    """Test memory persistence methods."""

    def test_save_and_get_memory(self, tmp_path: Path) -> None:
        """GIVEN a saved memory WHEN get_memories THEN it is returned."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        row_id = db.save_memory("user1", "likes Python", "#test")
        memories = db.get_memories("user1")
        assert len(memories) == 1
        assert memories[0].id == row_id
        assert memories[0].nick == "user1"
        assert memories[0].fact == "likes Python"
        assert memories[0].source_channel == "#test"

    def test_save_memory_lowercases_nick(self, tmp_path: Path) -> None:
        """GIVEN mixed-case nick WHEN saved THEN stored lowercased."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_memory("Alice", "fact", "#test")
        memories = db.get_memories("alice")
        assert len(memories) == 1
        assert memories[0].nick == "alice"

    def test_get_memories_empty(self, tmp_path: Path) -> None:
        """GIVEN no memories WHEN get_memories THEN returns empty list."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.get_memories("unknown") == []

    def test_get_memories_ordered_by_created_at(self, tmp_path: Path) -> None:
        """GIVEN multiple memories WHEN get_memories THEN ordered by created_at."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_memory("user1", "first fact", "#test")
        db.save_memory("user1", "second fact", "#test")
        memories = db.get_memories("user1")
        assert len(memories) == 2
        assert memories[0].fact == "first fact"
        assert memories[1].fact == "second fact"

    def test_delete_memory_by_id(self, tmp_path: Path) -> None:
        """GIVEN two memories WHEN one deleted THEN only one remains."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        id1 = db.save_memory("user1", "fact one", "#test")
        db.save_memory("user1", "fact two", "#test")
        result = db.delete_memory("user1", id1)
        assert result is True
        memories = db.get_memories("user1")
        assert len(memories) == 1
        assert memories[0].fact == "fact two"

    def test_delete_memory_wrong_nick(self, tmp_path: Path) -> None:
        """GIVEN memory for alice WHEN bob tries to delete THEN fails."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        row_id = db.save_memory("alice", "secret", "#test")
        result = db.delete_memory("bob", row_id)
        assert result is False
        assert len(db.get_memories("alice")) == 1

    def test_delete_all_memories(self, tmp_path: Path) -> None:
        """GIVEN three memories WHEN delete_all THEN all removed."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_memory("user1", "a", "#test")
        db.save_memory("user1", "b", "#test")
        db.save_memory("user1", "c", "#test")
        count = db.delete_all_memories("user1")
        assert count == 3
        assert db.get_memories("user1") == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_persistence.py::TestMemoryPersistence -v`
Expected: FAIL — AttributeError: 'LLMDatabase' has no attribute 'save_memory'

**Step 3: Implement CRUD methods**

Add to `LLMDatabase`:

```python
def save_memory(self, nick: str, fact: str, source_channel: str) -> int:
    """Save a memory fact for a user. Returns the row ID."""
    conn = self._connect()
    cursor = conn.execute(
        "INSERT INTO memories (nick, fact, source_channel, created_at) VALUES (?, ?, ?, ?)",
        (nick.lower(), fact, source_channel.lower(), time.time()),
    )
    conn.commit()
    return cursor.lastrowid or 0

def get_memories(self, nick: str) -> list[MemoryRow]:
    """Get all memories for a user, ordered by creation time."""
    conn = self._connect()
    rows = conn.execute(
        "SELECT id, nick, fact, source_channel, created_at FROM memories "
        "WHERE nick = ? ORDER BY created_at",
        (nick.lower(),),
    ).fetchall()
    return [MemoryRow(*row) for row in rows]

def delete_memory(self, nick: str, memory_id: int) -> bool:
    """Delete a specific memory by ID and nick. Returns True if deleted."""
    conn = self._connect()
    cursor = conn.execute(
        "DELETE FROM memories WHERE id = ? AND nick = ?",
        (memory_id, nick.lower()),
    )
    conn.commit()
    return cursor.rowcount > 0

def delete_all_memories(self, nick: str) -> int:
    """Delete all memories for a user. Returns count deleted."""
    conn = self._connect()
    cursor = conn.execute(
        "DELETE FROM memories WHERE nick = ?",
        (nick.lower(),),
    )
    conn.commit()
    return cursor.rowcount
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_persistence.py::TestMemoryPersistence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat: add memory CRUD methods to LLMDatabase"
```

---

### Task 3: Memory Config and Conftest Update

**Files:**
- Modify: `plugins/llm/src/llm/config.py`
- Modify: `plugins/llm/tests/conftest.py`

**Step 1: Add config registrations**

In `config.py`, add after existing command config groups:

```python
# Memory extraction
conf.registerChannelValue(
    LLM,
    "memoryEnabled",
    registry.Boolean(True, _("""Enable automatic memory extraction from command interactions.""")),
)
conf.registerChannelValue(
    LLM,
    "memoryExtractionModel",
    ValidatedModelName(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for memory extraction (cheap flash-tier recommended)."""),
    ),
)
conf.registerGlobalValue(
    LLM,
    "memoryMaxPerUser",
    registry.PositiveInteger(50, _("""Maximum number of memories stored per user.""")),
)
```

**Step 2: Update conftest defaults**

In `make_registry_side_effect`, add to defaults dict:

```python
"memoryEnabled": True,
"memoryExtractionModel": "gemini/gemini-2.0-flash-lite",
"memoryMaxPerUser": 50,
```

**Step 3: Run preflight**

Run: `make preflight`
Expected: All tests pass, no regressions

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat: add memory extraction configuration"
```

---

### Task 4: Memory Extraction Service Method

**Files:**
- Modify: `plugins/llm/src/llm/service.py`
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Write the failing tests**

Add `TestMemoryExtraction` class in `test_service.py`:

```python
class TestMemoryExtraction:
    """Test memory fact extraction from conversations."""

    def test_extract_memories_returns_facts(self, make_service: Callable) -> None:
        """GIVEN conversation with facts WHEN extracted THEN returns fact list."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = '["likes Python", "lives in Toronto"]'
            mock_litellm.completion.return_value = mock_response
            facts = service.extract_memories(
                "user1", "#test", "I love Python and live in Toronto", "Cool!", []
            )
        assert facts == ["likes Python", "lives in Toronto"]

    def test_extract_memories_empty_on_no_facts(self, make_service: Callable) -> None:
        """GIVEN boring conversation WHEN extracted THEN returns empty list."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "[]"
            mock_litellm.completion.return_value = mock_response
            facts = service.extract_memories("user1", "#test", "hello", "hi", [])
        assert facts == []

    def test_extract_memories_empty_on_error(self, make_service: Callable) -> None:
        """GIVEN API error WHEN extracting THEN returns empty list."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = Exception("API down")
            facts = service.extract_memories("user1", "#test", "hi", "hello", [])
        assert facts == []

    def test_extract_memories_empty_on_invalid_json(self, make_service: Callable) -> None:
        """GIVEN non-JSON response WHEN extracting THEN returns empty list."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "not json at all"
            mock_litellm.completion.return_value = mock_response
            facts = service.extract_memories("user1", "#test", "hi", "hello", [])
        assert facts == []

    def test_extract_memories_includes_existing_in_prompt(self, make_service: Callable) -> None:
        """GIVEN existing memories WHEN extracting THEN included in prompt."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "[]"
            mock_litellm.completion.return_value = mock_response
            service.extract_memories(
                "user1", "#test", "hi", "hello", ["already knows Python"]
            )
            call_args = mock_litellm.completion.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            prompt_text = " ".join(m["content"] for m in messages)
            assert "already knows Python" in prompt_text
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_service.py::TestMemoryExtraction -v`
Expected: FAIL — AttributeError

**Step 3: Implement `extract_memories()`**

Add to `LLMService`:

```python
_MEMORY_EXTRACTION_PROMPT = (
    "You are a fact extractor. Given a conversation between a user and an assistant, "
    "extract any new factual information about the user worth remembering long-term. "
    "Examples: preferences, skills, location, job, interests, opinions.\n\n"
    "Return ONLY a JSON array of short factual strings. "
    "Return [] if there is nothing notable or new.\n\n"
    "Do NOT include:\n"
    "- Facts already known (listed below)\n"
    "- Transient information (what they're doing right now)\n"
    "- Questions they asked (only facts about them)\n"
)

def extract_memories(
    self,
    nick: str,
    channel: str,
    user_message: str,
    assistant_response: str,
    existing_memories: list[str],
) -> list[str]:
    """Extract memorable facts from a conversation exchange.

    Returns list of fact strings, or empty list on any error.
    """
    existing_section = ""
    if existing_memories:
        existing_section = "\n\nAlready known facts:\n" + "\n".join(
            f"- {m}" for m in existing_memories
        )

    messages = [
        {"role": "system", "content": self._MEMORY_EXTRACTION_PROMPT + existing_section},
        {"role": "user", "content": f"User ({nick}): {user_message}\nAssistant: {assistant_response}"},
    ]

    try:
        model = self.plugin.registryValue("memoryExtractionModel", channel)
        api_key = self.plugin.registryValue("askApiKey")
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            timeout=15,
        )
        content = response.choices[0].message.content.strip()
        facts = json.loads(content)
        if isinstance(facts, list) and all(isinstance(f, str) for f in facts):
            return facts
        return []
    except Exception:
        return []
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_service.py::TestMemoryExtraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: add memory extraction service method"
```

---

### Task 5: Memory Injection into System Prompt

**Files:**
- Modify: `plugins/llm/src/llm/service.py`
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Write the failing tests**

```python
class TestMemoryInjection:
    """Test memory injection into system prompts."""

    def test_completion_with_memories_injects_into_prompt(self, make_service: Callable) -> None:
        """GIVEN memories WHEN completion called THEN facts in system prompt."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "Hello!"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response._hidden_params = {"response_cost": 0.001}
            mock_litellm.completion.return_value = mock_response
            service.completion(
                "hi", command="ask", memories=["likes Python", "lives in Toronto"]
            )
            call_args = mock_litellm.completion.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            system_msg = next(m for m in messages if m["role"] == "system")
            assert "likes Python" in system_msg["content"]
            assert "lives in Toronto" in system_msg["content"]

    def test_completion_without_memories_no_section(self, make_service: Callable) -> None:
        """GIVEN no memories WHEN completion called THEN no memory section."""
        service, mock_plugin = make_service()
        with mock.patch("llm.service.litellm") as mock_litellm:
            mock_response = mock.MagicMock()
            mock_response.choices = [mock.MagicMock()]
            mock_response.choices[0].message.content = "Hello!"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response._hidden_params = {"response_cost": 0.001}
            mock_litellm.completion.return_value = mock_response
            service.completion("hi", command="ask")
            call_args = mock_litellm.completion.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            system_msg = next(m for m in messages if m["role"] == "system")
            assert "What you know about this user" not in system_msg["content"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_service.py::TestMemoryInjection -v`
Expected: FAIL — TypeError: completion() got unexpected keyword argument 'memories'

**Step 3: Implement memory injection**

- Add `memories: list[str] | None = None` parameter to `completion()` signature
- In `_build_system_prompt()` (or inline in completion after building system prompt), append memory section when non-empty:

```python
if memories:
    memory_section = "\n\nWhat you know about this user from past conversations:\n" + "\n".join(
        f"- {fact}" for fact in memories
    )
    system_prompt += memory_section
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_service.py::TestMemoryInjection -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat: inject user memories into system prompt"
```

---

### Task 6: Memory Extraction Hook and Retrieval in Plugin

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_integration.py` (or `test_plugin.py`)

**Step 1: Write the failing tests**

```python
class TestMemoryIntegration:
    """Test memory extraction and retrieval wiring."""

    def test_ask_passes_memories_to_completion(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN user has memories WHEN ask called THEN memories passed to completion."""
        plugin, mock_irc = plugin_with_real_db
        plugin.db.save_memory("testuser", "likes Python", "#test")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "hello")
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.time = time.time() + 100

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.ask(mock_irc, mock_msg, ["hello"])

        completion_call = plugin.llm_service.completion.call_args
        assert "likes Python" in str(completion_call)

    def test_ask_triggers_background_extraction(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN successful ask WHEN completed THEN memory extraction scheduled."""
        plugin, mock_irc = plugin_with_real_db
        mock_schedule = mocker.patch("llm.plugin.schedule.addEvent")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "hello")
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.time = time.time() + 100

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        from llm.service import CompletionResult
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hi there!", prompt_tokens=10, completion_tokens=5, cost=0.001, model="test"
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Check that a memory extraction event was scheduled
        memory_events = [
            call for call in mock_schedule.call_args_list
            if "llm_memory_" in str(call)
        ]
        assert len(memory_events) >= 1

    def test_ask_skips_extraction_when_memory_disabled(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN memoryEnabled=False WHEN ask THEN no extraction scheduled."""
        plugin, mock_irc = plugin_with_real_db
        # Override memoryEnabled to False
        original_side_effect = plugin.registryValue.side_effect
        def patched_registry(key, *args, **kwargs):
            if key == "memoryEnabled":
                return False
            return original_side_effect(key, *args, **kwargs)
        plugin.registryValue = mocker.MagicMock(side_effect=patched_registry)

        mock_schedule = mocker.patch("llm.plugin.schedule.addEvent")

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "hello")
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.time = time.time() + 100

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        from llm.service import CompletionResult
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hi!", prompt_tokens=10, completion_tokens=5, cost=0.001, model="test"
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        plugin.ask(mock_irc, mock_msg, ["hello"])

        memory_events = [
            call for call in mock_schedule.call_args_list
            if "llm_memory_" in str(call)
        ]
        assert len(memory_events) == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest plugins/llm/tests/test_integration.py::TestMemoryIntegration -v`
Expected: FAIL

**Step 3: Implement extraction hook and retrieval**

In `plugin.py`:

Add helper method:
```python
def _get_user_memories(self, nick: str) -> list[str]:
    """Get memory facts for a user as a list of strings."""
    if self.db is None:
        return []
    rows = self.db.get_memories(nick)
    return [row.fact for row in rows]
```

In `ask()`, `picard()`, and `code()` — before calling `completion()`:
```python
memory_facts = self._get_user_memories(nick)
# Pass to completion:
result = self.llm_service.completion(..., memories=memory_facts)
```

In `_store_context_and_log_usage()` — after logging, if successful and memoryEnabled.
Only fire for text-producing commands (ask, picard, code), not draw/animate.
Use `result.content` (raw LLM output), not the display-formatted `response` parameter.
Wrap the entire extraction block in try/except so failures never disrupt usage logging.

```python
_MEMORY_COMMANDS = {"ask", "picard", "code"}

# In _store_context_and_log_usage, after existing logging:
try:
    if (
        command in self._MEMORY_COMMANDS
        and result.error is None
        and self.registryValue("memoryEnabled", channel)
    ):
        existing = self._get_user_memories(nick)
        max_memories = self.registryValue("memoryMaxPerUser")
        if len(existing) < max_memories:
            raw_response = result.content  # Raw LLM output, not display-formatted
            def _extract_memories_bg():
                try:
                    facts = self.llm_service.extract_memories(
                        nick, channel, text, raw_response, existing
                    )
                    for fact in facts:
                        if len(self.db.get_memories(nick)) >= max_memories:
                            break
                        self.db.save_memory(nick, fact, channel)
                except Exception:
                    log.exception("Memory extraction failed for %s", nick)
            event_name = f"llm_memory_{uuid4().hex[:8]}"
            schedule.addEvent(_extract_memories_bg, time.time() + 0.1, name=event_name)
except Exception:
    log.exception("Memory extraction scheduling failed for %s", nick)
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_integration.py::TestMemoryIntegration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_integration.py
git commit -m "feat: wire memory extraction and retrieval into command flow"
```

---

### Task 7: `%memories` Command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing tests**

```python
class TestMemoriesCommand:
    """Test %memories command."""

    def test_memories_list_shows_facts(self, ...) -> None:
        """GIVEN user has memories WHEN %memories THEN lists them."""

    def test_memories_list_empty(self, ...) -> None:
        """GIVEN no memories WHEN %memories THEN shows no-memories message."""

    def test_memories_delete_removes_fact(self, ...) -> None:
        """GIVEN memory exists WHEN %memories delete <id> THEN removed."""

    def test_memories_delete_invalid_id(self, ...) -> None:
        """GIVEN bad ID WHEN %memories delete <id> THEN error message."""

    def test_memories_clear_deletes_all(self, ...) -> None:
        """GIVEN memories exist WHEN %memories clear THEN all removed."""
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `memories` command**

```python
def memories(self, irc, msg, args, text):
    """[delete <id> | clear]

    View or manage your stored memories. Use 'delete <id>' to remove
    a specific memory, or 'clear' to remove all.
    """
    nick = self._get_identity(irc, msg)

    if not text:
        # List memories
        rows = self.db.get_memories(nick)
        if not rows:
            irc.reply("I don't have any memories about you.", prefixNick=False)
            return
        lines = [f"[{r.id}] {r.fact}" for r in rows]
        irc.reply(" | ".join(lines), prefixNick=False)
        return

    parts = text.split(None, 1)
    subcommand = parts[0].lower()

    if subcommand == "clear":
        count = self.db.delete_all_memories(nick)
        irc.reply(f"Cleared {count} memories.", prefixNick=False)

    elif subcommand == "delete" and len(parts) == 2:
        try:
            memory_id = int(parts[1])
        except ValueError:
            irc.error("Usage: memories delete <id>")
            return
        if self.db.delete_memory(nick, memory_id):
            irc.reply("Memory deleted.", prefixNick=False)
        else:
            irc.error("Memory not found or doesn't belong to you.")

    else:
        irc.error("Usage: memories [delete <id> | clear]")

memories = wrap(memories, [optional("text")])
```

**Step 4: Run tests to verify they pass**

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat: add %memories command for user memory management"
```

---

### Task 8: Spontaneous Participation Config

**Files:**
- Modify: `plugins/llm/src/llm/config.py`
- Modify: `plugins/llm/src/llm/service.py` (update `_sanitize`)
- Modify: `plugins/llm/tests/conftest.py`

**Step 1: Add config registrations**

In `config.py`:

```python
# Spontaneous participation
conf.registerChannelValue(
    LLM,
    "spontaneousEnabled",
    registry.Boolean(False, _("""Enable spontaneous channel participation.""")),
)
conf.registerChannelValue(
    LLM,
    "spontaneousChance",
    registry.PositiveInteger(15, _("""Percent chance (1-100) of evaluating a spontaneous reply per message.""")),
)
conf.registerChannelValue(
    LLM,
    "spontaneousCooldown",
    registry.PositiveInteger(2, _("""Minimum minutes between spontaneous replies per channel.""")),
)
conf.registerChannelValue(
    LLM,
    "spontaneousModel",
    ValidatedModelName(
        "gemini/gemini-2.0-flash-lite",
        _("""Model for spontaneous participation (cheap flash-tier recommended)."""),
    ),
)
conf.registerGlobalValue(
    LLM,
    "spontaneousApiKey",
    registry.String("", _("""API key for spontaneous model. Falls back to askApiKey if empty."""), private=True),
)
conf.registerChannelValue(
    LLM,
    "spontaneousSystemPrompt",
    registry.String(
        "You are a regular in this IRC channel. You see the recent conversation "
        "and can jump in if you have something useful, funny, or relevant to add. "
        "Keep it brief — one or two sentences max. Match the tone of the channel. "
        "If the conversation is dead or you have nothing to add, respond with exactly PASS. "
        "You're a channel regular, not an assistant — be natural, have opinions, be yourself.",
        _("""System prompt for spontaneous channel participation."""),
    ),
)
```

**Step 2: Update `_sanitize` in `service.py`**

Add `"spontaneousApiKey"` to the key name loop in `_sanitize()`.

**Step 3: Update conftest defaults**

```python
"spontaneousEnabled": False,
"spontaneousChance": 15,
"spontaneousCooldown": 2,
"spontaneousModel": "gemini/gemini-2.0-flash-lite",
"spontaneousApiKey": "",
"spontaneousSystemPrompt": "You are a participant in an IRC channel...",
```

**Step 4: Run preflight**

Run: `make preflight`
Expected: All tests pass

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/src/llm/service.py plugins/llm/tests/conftest.py
git commit -m "feat: add spontaneous participation configuration"
```

---

### Task 9: Spontaneous Participation Logic

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Create: `plugins/llm/tests/test_spontaneous.py`

**Step 1: Write the failing tests**

New file `test_spontaneous.py`:

```python
class TestSpontaneousParticipation:
    """Test spontaneous channel participation."""

    def test_skips_when_disabled(self, ...) -> None:
        """GIVEN spontaneousEnabled=False WHEN message received THEN no evaluation."""

    def test_fires_on_chance_hit(self, ...) -> None:
        """GIVEN enabled and chance hit WHEN message THEN schedules evaluation."""

    def test_respects_cooldown(self, ...) -> None:
        """GIVEN recent spontaneous reply WHEN message THEN skips."""

    def test_sends_message_on_non_pass(self, ...) -> None:
        """GIVEN evaluation returns text WHEN callback fires THEN sends to channel."""

    def test_discards_pass_response(self, ...) -> None:
        """GIVEN evaluation returns PASS WHEN callback fires THEN no message."""

    def test_uses_ask_api_key_as_fallback(self, ...) -> None:
        """GIVEN empty spontaneousApiKey WHEN evaluating THEN uses askApiKey."""

    def test_uses_dedicated_api_key_when_set(self, ...) -> None:
        """GIVEN spontaneousApiKey set WHEN evaluating THEN uses it."""

    def test_logs_usage_on_spontaneous_reply(self, ...) -> None:
        """GIVEN spontaneous reply sent WHEN completed THEN usage logged."""
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement spontaneous logic**

Add `import random` to plugin.py imports.

In `plugin.py` `__init__`:
```python
self._spontaneous_cooldowns: dict[str, float] = {}
```

In `doPrivmsg`, after existing tracking block:
```python
# Spontaneous participation
if self.registryValue("spontaneousEnabled", channel):
    cooldown_minutes = self.registryValue("spontaneousCooldown", channel)
    last_spontaneous = self._spontaneous_cooldowns.get(channel, 0)
    if time.time() - last_spontaneous >= cooldown_minutes * 60:
        chance = self.registryValue("spontaneousChance", channel)
        if random.randint(1, 100) <= chance:
            self._spontaneous_cooldowns[channel] = time.time()
            self._schedule_spontaneous(irc, channel)
```

**Completion parameter changes in `service.py`:**

Add `api_key: str | None = None` and `model_override: str | None = None` parameters
to `completion()`. When provided, they bypass the normal `registryValue(f"{command}ApiKey")`
and `registryValue(f"{command}Model", channel)` lookups. This keeps the spontaneous API key
fallback logic cleanly in the caller (plugin.py), not buried in service.py.

```python
def completion(
    self,
    prompt: str,
    command: str = "ask",
    images: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    channel_history: list[dict[str, str]] | None = None,
    irc: Irc | None = None,
    msg: IrcMsg | None = None,
    system_prompt: str | None = None,
    memories: list[str] | None = None,
    api_key: str | None = None,        # NEW: override config lookup
    model_override: str | None = None,  # NEW: override config lookup
) -> CompletionResult:
```

In the body, change config lookups to respect overrides:
```python
# API key: use override if provided, else config lookup
effective_api_key = api_key or self.plugin.registryValue(f"{command}ApiKey")
if not effective_api_key:
    return CompletionResult(content="", error="API key not configured")

# Model: use override if provided, else config lookup
effective_model = model_override or self.plugin.registryValue(f"{command}Model", channel)
```

New method in plugin.py:
```python
def _schedule_spontaneous(self, irc, channel):
    """Schedule a spontaneous reply evaluation."""
    def _evaluate():
        try:
            channel_msgs = self.context.get_channel_messages(channel)
            if not channel_msgs:
                return

            api_key = self.registryValue("spontaneousApiKey")
            if not api_key:
                api_key = self.registryValue("askApiKey")
            if not api_key:
                return

            model = self.registryValue("spontaneousModel", channel)
            system_prompt = self.registryValue("spontaneousSystemPrompt", channel)

            prompt = "Respond to the conversation above, or say PASS."
            result = self.llm_service.completion(
                prompt,
                command="ask",  # Reuse ask for config fallbacks
                channel_history=channel_msgs,
                system_prompt=system_prompt,
                api_key=api_key,          # Explicit override
                model_override=model,     # Explicit override
            )

            if result.error or "PASS" in result.content.strip().upper():
                return

            response = self.llm_service.sanitize_output(result.content)
            irc.queueMsg(ircmsgs.privmsg(channel, response))

            self.db.log_usage(
                irc.nick, channel, "spontaneous", result.model,
                result.prompt_tokens, result.completion_tokens, result.cost,
                prompt="[spontaneous]", status="success",
            )
        except Exception:
            log.exception("Spontaneous evaluation failed for %s", channel)

    event_name = f"llm_spontaneous_{uuid4().hex[:8]}"
    schedule.addEvent(_evaluate, time.time() + 0.5, name=event_name)
```

**Step 4: Run tests to verify they pass**

Run: `pytest plugins/llm/tests/test_spontaneous.py -v`
Expected: PASS

**Step 5: Run full preflight**

Run: `make preflight`
Expected: All tests pass

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/src/llm/service.py plugins/llm/tests/test_spontaneous.py
git commit -m "feat: add spontaneous participation in channel conversations"
```

---

### Task 10: Cleanup and Edge Cases

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: existing test files

**Step 1: Write edge case tests**

```python
def test_spontaneous_no_send_if_channel_empty(self, ...) -> None:
    """GIVEN empty channel context WHEN evaluation fires THEN no message."""

def test_die_cleans_up_events(self, ...) -> None:
    """GIVEN pending memory/spontaneous events WHEN die THEN cleaned up."""

def test_import_random_is_present(self) -> None:
    """GIVEN plugin module WHEN imported THEN random is available."""
```

**Step 2: Implement cleanup**

In `die()`: Clean up spontaneous cooldowns dict. Memory extraction events are fire-and-forget with unique names — they'll expire naturally. No special cleanup needed.

Add `import random` to plugin.py imports.

**Step 3: Run preflight**

Run: `make preflight`
Expected: All 900+ tests pass

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/
git commit -m "fix: clean up spontaneous state on plugin unload"
```
