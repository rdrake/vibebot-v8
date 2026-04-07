# Command Surface Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate `%remindme`/`%reminders`/`%unremind` into `%remind`, fix `%usage` to use `wrap()`, replace `%picard` with `%instruct`, and update all terminology to volatile/non-volatile memory.

**Architecture:** The remind consolidation and usage fix follow the existing `%memories` pattern: `wrap(command, [optional("text")])` with internal `text.split()` dispatch. The `%instruct` command stores user instructions in a new `user_instructions` DB table and injects them into the `%ask` system prompt. `%picard` is removed entirely.

**Tech Stack:** Python, Limnoria, SQLite, pytest

---

### Task 1: Add `user_instructions` table to persistence layer

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py`
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing tests**

Add to `test_persistence.py`:

Note: `test_persistence.py` does NOT have a `db` fixture — each test creates `LLMDatabase` instances directly with `tmp_path`. Follow that pattern:

```python
class TestUserInstructions:
    """Tests for user_instructions table CRUD."""

    def test_get_instruction_returns_none_when_empty(self, tmp_path: Path) -> None:
        """GIVEN no instruction WHEN queried THEN returns None."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.get_instruction("testnick") is None

    def test_save_and_get_instruction(self, tmp_path: Path) -> None:
        """GIVEN saved instruction WHEN queried THEN returns text."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_instruction("testnick", "You are Captain Picard.")
        assert db.get_instruction("testnick") == "You are Captain Picard."

    def test_save_instruction_overwrites(self, tmp_path: Path) -> None:
        """GIVEN existing instruction WHEN saved again THEN overwrites."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_instruction("testnick", "old")
        db.save_instruction("testnick", "new")
        assert db.get_instruction("testnick") == "new"

    def test_delete_instruction(self, tmp_path: Path) -> None:
        """GIVEN existing instruction WHEN deleted THEN returns True and clears."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.save_instruction("testnick", "text")
        assert db.delete_instruction("testnick") is True
        assert db.get_instruction("testnick") is None

    def test_delete_instruction_missing(self, tmp_path: Path) -> None:
        """GIVEN no instruction WHEN deleted THEN returns False."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        assert db.delete_instruction("testnick") is False
```

**Step 2: Run tests to verify they fail**

```bash
make test
```

Expected: FAIL — `get_instruction`, `save_instruction`, `delete_instruction` not defined.

**Step 3: Implement the migration and CRUD methods**

In `persistence.py`:

1. Bump `SCHEMA_VERSION` from `6` to `7`.
2. Add migration block after the `current_version < 6` block:

```python
if current_version < 7:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_instructions (
            nick TEXT PRIMARY KEY,
            instruction TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
    """)
    conn.commit()
```

3. Add three methods to the `LLMDatabase` class:

```python
def get_instruction(self, nick: str) -> str | None:
    """Get the user's persistent instruction, or None if not set."""
    conn = self._connect()
    row = conn.execute(
        "SELECT instruction FROM user_instructions WHERE nick = ?",
        (nick,),
    ).fetchone()
    return row[0] if row else None

def save_instruction(self, nick: str, instruction: str) -> None:
    """Save or overwrite the user's persistent instruction."""
    conn = self._connect()
    conn.execute(
        "INSERT INTO user_instructions (nick, instruction, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(nick) DO UPDATE SET instruction = excluded.instruction, "
        "updated_at = excluded.updated_at",
        (nick, instruction, time.time()),
    )
    conn.commit()

def delete_instruction(self, nick: str) -> bool:
    """Delete the user's instruction. Returns True if one was deleted."""
    conn = self._connect()
    cursor = conn.execute(
        "DELETE FROM user_instructions WHERE nick = ?", (nick,),
    )
    conn.commit()
    return cursor.rowcount > 0
```

**Step 4: Run tests to verify they pass**

```bash
make test
```

Expected: PASS

**Step 5: Run lint/typecheck**

```bash
make lint && make typecheck
```

**Step 6: Commit**

```bash
git commit -m "feat(persistence): add user_instructions table (schema v7)"
```

---

### Task 2: Add `%instruct` command to plugin.py

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Write the failing tests**

Add a new `TestInstructCommand` class to `test_commands.py`:

```python
class TestInstructCommand:
    """Tests for the %instruct command."""

    def test_instruct_sets_instruction(self, plugin_env):
        """GIVEN text WHEN instruct called THEN saves to DB and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.instruct(mock_irc, mock_msg, [], "You are Captain Picard.")
        plugin.db.save_instruction.assert_called_once_with("testnick", "You are Captain Picard.")
        mock_irc.reply.assert_called_once()
        assert "set" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_no_args_shows_current(self, plugin_env):
        """GIVEN no text and existing instruction WHEN instruct called THEN shows it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Picard."
        plugin.instruct(mock_irc, mock_msg, [], None)
        mock_irc.reply.assert_called_once()
        assert "Picard" in mock_irc.reply.call_args.args[0]

    def test_instruct_no_args_no_instruction(self, plugin_env):
        """GIVEN no text and no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = None
        plugin.instruct(mock_irc, mock_msg, [], None)
        mock_irc.reply.assert_called_once()
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_removes(self, plugin_env):
        """GIVEN 'clear' WHEN instruct called THEN deletes and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = True
        plugin.instruct(mock_irc, mock_msg, [], "clear")
        plugin.db.delete_instruction.assert_called_once_with("testnick")
        mock_irc.reply.assert_called_once()
        assert "cleared" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_when_none_set(self, plugin_env):
        """GIVEN 'clear' with no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = False
        plugin.instruct(mock_irc, mock_msg, [], "clear")
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()
```

**Step 2: Run tests to verify they fail**

```bash
make test
```

**Step 3: Implement `%instruct` in plugin.py**

Add the command method after the `memories` command (around line 2160):

```python
def instruct(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    text: str | None,
) -> None:
    """[<instruction> | clear]

    Set persistent instructions that shape how %ask responds to you.
    Your instruction is prepended to the system prompt for every %ask call.

    Examples:
      %instruct You are Captain Picard. Respond in character.
      %instruct Respond only in haiku
      %instruct clear
      %instruct          (show current instruction)
    """
    nick = self._get_identity(irc, msg)

    if not text:
        current = self.db.get_instruction(nick)
        if current:
            irc.reply(f"Current instruction: {current}", prefixNick=False)
        else:
            irc.reply("No instruction set. Use %instruct <text> to set one.", prefixNick=False)
        return

    if text.strip().lower() == "clear":
        if self.db.delete_instruction(nick):
            irc.reply("Instruction cleared.", prefixNick=False)
        else:
            irc.reply("No instruction to clear.", prefixNick=False)
        return

    self.db.save_instruction(nick, text)
    irc.reply("Instruction set.", prefixNick=False)

instruct = wrap(instruct, [optional("text")])
```

**Step 4: Wire instruction into `%ask`**

In the `ask()` method (around line 1754, after `memories = self._get_user_memories(nick)`), add:

```python
user_instruction = self.db.get_instruction(nick)
```

Then pass it as `system_prompt` override when calling `self.llm_service.completion()`. The instruction should be prepended to the channel's `askSystemPrompt`:

```python
# Build system prompt with optional user instruction
ask_prompt = self.registryValue("askSystemPrompt", channel)
if user_instruction:
    effective_prompt = f"{user_instruction}\n\n{ask_prompt}"
else:
    effective_prompt = None  # let service.py use default

# Then pass system_prompt=effective_prompt to completion()
```

Update both the `images` and non-`images` branches of the `completion()` call to include `system_prompt=effective_prompt`.

**Step 5: Add a test for instruction injection into ask**

```python
def test_ask_prepends_user_instruction(self, plugin_env):
    """GIVEN user has instruction WHEN ask called THEN instruction prepended to system prompt."""
    plugin, mock_irc, mock_msg = plugin_env
    plugin.db.get_instruction.return_value = "You are Captain Picard."
    plugin.llm_service.detect_images.return_value = []
    plugin.llm_service.completion.return_value = CompletionResult(
        content="Make it so.",
        grounding_used=False,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="gpt-4",
    )
    plugin.ask(mock_irc, mock_msg, ["hello"])
    call_kwargs = plugin.llm_service.completion.call_args.kwargs
    assert "Picard" in call_kwargs["system_prompt"]

def test_ask_no_instruction_uses_default(self, plugin_env):
    """GIVEN no instruction WHEN ask called THEN no system_prompt override."""
    plugin, mock_irc, mock_msg = plugin_env
    plugin.db.get_instruction.return_value = None
    plugin.llm_service.detect_images.return_value = []
    plugin.llm_service.completion.return_value = CompletionResult(
        content="Hello!",
        grounding_used=False,
        prompt_tokens=10,
        completion_tokens=5,
        cost=0.001,
        model="gpt-4",
    )
    plugin.ask(mock_irc, mock_msg, ["hello"])
    call_kwargs = plugin.llm_service.completion.call_args.kwargs
    assert call_kwargs.get("system_prompt") is None
```

**Step 6: Run tests**

```bash
make test
```

**Step 7: Run lint/typecheck**

```bash
make lint && make typecheck
```

**Step 8: Commit**

```bash
git commit -m "feat: add %instruct command for user-settable system prompt instructions"
```

---

### Task 3: Remove `%picard` command

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/config.py`
- Modify: `plugins/llm/tests/test_commands.py`
- Modify: `plugins/llm/tests/conftest.py`
- Modify: `README.md` (remove picard from command tables/examples)
- Modify: `CLAUDE.md` (remove picard from IRC commands table)

**Step 1: Remove `picardSystemPrompt` from config.py**

Delete the `conf.registerChannelValue` block for `picardSystemPrompt` (lines 157-173 in config.py).

**Step 2: Remove `picardSystemPrompt` from conftest.py**

Remove the `"picardSystemPrompt"` entry from the `make_registry_side_effect` defaults dictionary (line 118 in conftest.py).

**Step 3: Remove the `picard()` method from plugin.py**

Delete the entire `picard()` method and its `wrap()` assignment (lines 1814-1899 in plugin.py).

**Step 4: Remove `"picard"` from `_MEMORY_COMMANDS`**

Update line 51 in plugin.py from:
```python
_MEMORY_COMMANDS = frozenset({"ask", "picard", "code"})
```
to:
```python
_MEMORY_COMMANDS = frozenset({"ask", "code"})
```

**Step 5: Remove `TestPicardCommand` from test_commands.py**

Delete the entire `TestPicardCommand` class (lines 343-441 in test_commands.py).

**Step 6: Remove `%picard` from HELP_HTML_TEMPLATE**

Delete the picard block from the HTML template (lines 139-143 in plugin.py).

**Step 7: Remove picard from README.md and CLAUDE.md**

Grep for `picard` in both files and remove from command tables, feature descriptions, and examples. In `CLAUDE.md`, remove the `%picard` row from the IRC Commands table.

**Step 8: Run preflight**

```bash
make preflight
```

Expected: PASS — no remaining references to picard in src/tests/docs (grep to verify: `grep -rn picard plugins/llm/src/ plugins/llm/tests/ README.md CLAUDE.md`).

**Step 9: Commit**

```bash
git commit -m "refactor: remove %picard command, replaced by %instruct"
```

---

### Task 4: Consolidate reminders into `%remind`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/tests/test_commands.py`
- Modify: `plugins/llm/tests/test_reminders.py`

**Step 1: Add `%remind` command to plugin.py**

Replace the three reminder commands (`remindme`, `reminders`, `unremind` — lines 2374-2495) with a single `remind` command:

```python
def remind(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    text: str | None,
) -> None:
    """[<reminder text> | list | del(ete) <id> [<id>...] | clear]

    Set and manage reminders using natural language.

    Examples:
      %remind in 30 minutes check the build
      %remind list
      %remind delete abc1
      %remind clear
    """
    nick = self._get_identity(irc, msg)

    if not text:
        # No args = list (same as %remind list)
        self._remind_list(irc, nick)
        return

    parts = text.split(None, 1)
    subcommand = parts[0].lower()

    if subcommand == "list":
        self._remind_list(irc, nick)

    elif subcommand in ("delete", "del") and len(parts) >= 2:
        raw_ids = text.split()[1:]
        deleted = 0
        for rid in raw_ids:
            target = self._find_user_reminder(nick, rid)
            if target:
                with contextlib.suppress(KeyError):
                    schedule.removeEvent(target)
                with self._reminders_lock:
                    self._reminders.pop(target, None)
                self.db.delete_reminder(target)
                deleted += 1
        if deleted == 0:
            irc.error(_("No matching reminders found."))
        elif deleted == 1:
            irc.reply(_("Reminder cancelled."), prefixNick=False)
        else:
            irc.reply(f"Cancelled {deleted} reminders.", prefixNick=False)

    elif subcommand == "clear":
        user_reminders = self._get_user_reminders(nick)
        if not user_reminders:
            irc.reply(_("No reminders to clear."), prefixNick=False)
            return
        for name, _ in user_reminders:
            with contextlib.suppress(KeyError):
                schedule.removeEvent(name)
            with self._reminders_lock:
                self._reminders.pop(name, None)
            self.db.delete_reminder(name)
        label = "reminder" if len(user_reminders) == 1 else "reminders"
        irc.reply(f"Cleared {len(user_reminders)} {label}.", prefixNick=False)

    else:
        # Everything else is treated as a natural language reminder
        self._remind_set(irc, msg, nick, text)

remind = wrap(remind, [optional("text")])
```

Extract the existing `remindme` body into `_remind_set` and the `reminders` body into `_remind_list` as private helpers:

```python
def _remind_list(self, irc: callbacks.Irc, nick: str) -> None:
    """List pending reminders for a user."""
    user_reminders = self._get_user_reminders(nick)
    if not user_reminders:
        irc.reply(_("You have no pending reminders."))
        return
    irc.reply(self._format_reminders(user_reminders))

def _remind_set(self, irc: callbacks.Irc, msg: IrcMsg, nick: str, text: str) -> None:
    """Parse and schedule a natural language reminder.

    This is the body extracted from the old remindme() command.
    The caller has already resolved nick via _get_identity().
    """
    channel = self._get_channel(msg)

    with self._trace_request("remind", nick, channel):
        with self._allow_concurrent():
            result = self.llm_service.parse_reminder(text, channel)

        if result.action == "clarify":
            irc.reply(result.confirmation)
            return

        if result.seconds is None:
            irc.reply(_("I couldn't determine when to remind you. Please try again."))
            return

        if result.seconds < 10:
            irc.error(_("Reminder must be at least 10 seconds from now."))
            return

        if result.seconds > 604800:  # 7 days
            irc.error(_("Reminder can't be more than 7 days out."))
            return

        reminder_message = result.message or text
        event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
        deliver = self._make_reminder_delivery_closure(
            nick, channel, reminder_message, event_name
        )

        try:
            schedule.addEvent(deliver, time.time() + result.seconds, name=event_name)
            with self._reminders_lock:
                self._reminders[event_name] = (nick, channel, reminder_message)

            self.db.save_reminder(
                event_name, nick, channel, reminder_message,
                time.time() + result.seconds,
            )

            reply = self.llm_service.sanitize_output(result.confirmation)
            if result.note:
                reply = f"{reply} ({self.llm_service.sanitize_output(result.note)})"
            irc.reply(reply)
        except Exception as e:
            self.log.error("Failed to schedule reminder: %s", e)
            irc.error(_("Failed to set reminder."))
```

**Step 2: Remove old commands**

Delete `remindme`, `reminders`, `unremind` methods and their `wrap()` assignments.

**Step 3: Update test_commands.py**

Rename test classes and update method calls:
- `TestRemindMeCommand` → `TestRemindCommand` — update all `plugin.remindme(...)` calls to `plugin.remind(...)` with adjusted args (the `text` arg is now a single string via `optional("text")`, not tokenized list).
- `TestRemindersCommand` → fold into `TestRemindCommand` — update `plugin.reminders(...)` to `plugin.remind(...)` with `text="list"`.
- `TestUnremindCommand` → fold into `TestRemindCommand` — update `plugin.unremind(...)` to `plugin.remind(...)` with `text="delete <id>"`.
- Add tests for `%remind clear`.

**Step 4: Update test_reminders.py**

- Update `TestReminderCommands` to check for `remind` instead of `remindme`/`reminders`/`unremind`.
- Update docstring references.
- Helper method tests (`_get_user_reminders`, `_format_reminders`, `_find_user_reminder`) stay unchanged.

**Step 5: Run preflight**

```bash
make preflight
```

Grep to verify no remaining references: `grep -rn 'remindme\|unremind' plugins/llm/src/ plugins/llm/tests/`

**Step 6: Commit**

```bash
git commit -m "refactor: consolidate remindme/reminders/unremind into %remind"
```

---

### Task 5: Fix `%usage` to use `wrap()`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`

**Step 1: Refactor `usage()` to use `wrap()`**

Replace the current `usage()` signature and `_extract_raw_arg` call with the standard pattern:

```python
def usage(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    args: list,
    text: str | None,
) -> None:
    """[<nick or #channel>]

    Show API usage statistics.

    No argument in a channel: shows channel stats and your personal stats.
    No argument via PM: shows global overview (admin only).
    <nick>: shows that user's stats (scoped to current channel if in one).
    <#channel>: shows that channel's stats.
    """
    target = text.strip() if text else None

    # Strip IRC status prefixes (@op, +voice, %halfop) from nick targets
    if target and not ircutils.isChannel(target):
        target = target.lstrip("@+%")
    if target and ircutils.isChannel(target):
        self._usage_for_channel(irc, msg, target)
    elif target:
        self._usage_for_nick(irc, msg, target)
    elif msg.channel:
        self._usage_channel(irc, msg)
    else:
        if not ircdb.checkCapability(msg.prefix, "admin"):
            irc.error(_("You need the 'admin' capability to view global usage stats."))
            return
        self._usage_global(irc, msg)

usage = wrap(usage, [optional("text")])
```

**Step 2: Remove `_extract_raw_arg` if no other callers**

Grep for `_extract_raw_arg` in plugin.py. If `usage` was the only caller, delete the method entirely (lines 1469-1493).

**Step 3: Update tests in `test_commands.py` — `TestUsageCommand`**

This is the most labor-intensive part. The existing tests mock `callbacks.addressed` (via `_extract_raw_arg`) to feed targets. After the refactoring, targets arrive as the `text` parameter.

1. **Remove the `_mock_addressed` autouse fixture** (line 862-865). It patched `callbacks.addressed` to return `None` — no longer needed since `_extract_raw_arg` is gone.

2. **Update all no-arg tests** — tests that call `plugin.usage(mock_irc, mock_msg, [])` stay the same, but add `None` as the `text` arg:
   - `plugin.usage(mock_irc, mock_msg, [], None)` (or simply `plugin.usage(mock_irc, mock_msg, [])` if wrap handles it)

3. **Update all target-nick tests** — ~6 tests mock `callbacks.addressed` to return strings like `"usage @Larry"`, `"usage Rubin[F]"`, `"usage othernick"`. Replace each with passing `text` directly:
   - Remove the `mocker.patch("llm.plugin.callbacks.addressed", ...)` line
   - Change `plugin.usage(mock_irc, mock_msg, [])` → `plugin.usage(mock_irc, mock_msg, [], "@Larry")`
   - The bracket-nick test (`Rubin[F]`) now works naturally: `plugin.usage(mock_irc, mock_msg, [], "Rubin[F]")`

4. **Update all target-channel tests** — ~2 tests mock `callbacks.addressed` to return `"usage #other"` etc. Same pattern:
   - Remove the `mocker.patch("llm.plugin.callbacks.addressed", ...)` line
   - Change to `plugin.usage(mock_irc, mock_msg, [], "#other")`
   - Keep `mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)` if needed for channel detection

There are approximately 12 tests to update. Each change is mechanical: remove `callbacks.addressed` mock, pass target as `text` arg.

**Step 4: Run preflight**

```bash
make preflight
```

**Step 5: Commit**

```bash
git commit -m "refactor: make %usage use wrap() with optional text, matching %memories pattern"
```

---

### Task 6: Update terminology to volatile/non-volatile memory

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (docstrings, help text)
- Modify: `plugins/llm/README.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Step 1: Update `%forget` docstring**

Change:
```python
"""[<channel>]

Clear your conversation context (memory) for the current or specified channel.
Use this to start fresh.
"""
```

To:
```python
"""[<channel>]

Clear your volatile memory (conversation context) for the current or specified channel.
Use this to start fresh. Volatile memory expires automatically after a timeout.
"""
```

**Step 2: Update `%memories` docstring**

Add volatile/non-volatile framing:
```python
"""[<nick> | del(ete) <id> [<id>...] | edit <id> <text> | clear | cleanup [nick]]

Manage your non-volatile memory (stored facts the bot remembers about you
across conversations). Use 'delete <id>' to remove, 'edit <id> <text>'
to update, 'clear' to remove all, or 'cleanup' to trigger a cleanup pass.
"""
```

**Step 3: Update HELP_HTML_TEMPLATE**

Update the Features section (around line 174-180) to use volatile/non-volatile terminology:
- "Conversation Context" → "Volatile Memory — recent exchanges for natural follow-up questions (cleared by `%forget`, expires after timeout)"
- "Long-term Memory" → "Non-volatile Memory — facts the bot remembers about you across conversations (managed by `%memories`)"

**Step 4: Update `%forget` help text in HTML**

Change the `%forget` description (line 157):
```
Clear your volatile memory (conversation context) for the current or specified channel.
```

**Step 5: Update `%memories` help text in HTML**

Change the `%memories` description (line 162):
```
Manage your non-volatile memory (stored facts about you that persist across conversations).
```

**Step 6: Update README.md and CLAUDE.md**

Replace "conversation context (memory)" / "long-term memory" with volatile/non-volatile terminology in command tables and feature descriptions.

**Step 7: Run preflight**

```bash
make preflight
```

**Step 8: Commit**

```bash
git commit -m "docs: adopt volatile/non-volatile memory terminology across all help surfaces"
```

---

### Task 7: Final verification

**Step 1: Search for orphaned references**

```bash
grep -rn 'remindme\|unremind\|picard\|_extract_raw_arg' plugins/llm/src/ plugins/llm/tests/ --include="*.py"
```

Expect only benign hits (e.g., "picard" in test data strings if kept as instruct examples). Fix any real orphans.

**Step 2: Run full preflight**

```bash
make preflight
```

Expected: format clean, lint clean, typecheck clean, all tests pass, coverage >= 80%.

**Step 3: Commit any fixups**

```bash
git commit -m "chore: clean up orphaned references from command surface overhaul"
```
