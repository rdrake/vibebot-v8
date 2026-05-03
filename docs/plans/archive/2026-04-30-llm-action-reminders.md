# LLM-Action Reminders Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow reminders to optionally re-invoke the LLM at fire time (e.g. "check every 2 hours and tell me when CVE-2026-31431 is fixed in Debian"), instead of only echoing static text.

**Architecture:** Extend the existing custom reminder system in the LLM plugin (which already uses `supybot.schedule.addEvent` as a timer and SQLite for persistence). Add `action_prompt` and `account` columns to the `reminders` table. When `action_prompt` is set, the delivery closure synthesizes a minimal `IrcMsg`, builds an `AssistantRequestContext` for the saved nick/channel/account, and dispatches through the existing `assistant_request` facade with the same callback set the live `@ask` path uses (search/fetch/code/draw/memory/reminder tools — the last capped to one nested schedule per fire). Existing reminders keep their echo behavior (default empty `action_prompt`). Rate-limiting reuses the existing `ask` bucket — scheduled actions count against the same daily budget as user-typed `@ask`. **Tool surface is the same; capabilities are NOT.** Scheduled actions run with `is_owner=False` and `capabilities=frozenset()` regardless of what the scheduling user had — running a tool 24 hours later under elevated owner/admin rights without their live consent is a footgun, and tier resolution at fire time is best-effort anyway.

**Tech Stack:** Python 3.12, Limnoria (Supybot), SQLite (project's own DB, not Limnoria's), pytest + pytest-mock, litellm.

**Reference files:**
- `plugins/llm/src/llm/plugin.py` — reminder scheduling and delivery (lines 954-1022, 2475-2680), rate limiting (1378-1542), `_build_request_context` (1227)
- `plugins/llm/src/llm/persistence.py` — schema migrations (146-302), reminder ops (390-490)
- `plugins/llm/src/llm/service.py` — `parse_reminder` (1963), `assistant_request` (1885), `AssistantRequestContext` (260)
- `plugins/llm/tests/test_reminders.py` — existing test patterns
- `plugins/llm/tests/conftest.py` — test fixtures (`make_registry_side_effect`, `plugin_init_patches`, `mock_irc`)

---

## Code-Review Adjustments

This plan was reviewed by `superpowers:code-reviewer` and Codex before any code was written. Adjustments made in response:

**Round 1 (superpowers:code-reviewer):**
- **Per-fire cap on nested reminder scheduling** (Task 4): a single LLM action may schedule at most ONE follow-up reminder. Without this cap, the meta loop's `metaMaxSteps=12` iterations could create 12 children each fire — exponential fan-out.
- **`_check_rate_limit_silent` helper** (new sub-step in Task 4) instead of inlining the over-limit check. Honors `enforceRateLimits` shadow mode and matches the live `@ask` accounting exactly.
- **Synthetic IrcMsg `args[0]`** uses the actual reply target (`channel` for channel reminders, `nick` for PMs) so downstream code reading `msg.args[0]` (e.g. `_begin_typing`) targets the right place.
- **`_find_user_reminder` 3-tuple unpack fix** explicitly listed in Task 3 — previously only flagged via the final grep sweep.
- **Bot-nick guard** at the top of `_deliver_action` for early-startup `world.ircs` entries.
- **`save_reminder(action_prompt=)` is keyword-only** to prevent silent regressions where future callers forget to pass it.

**Round 2 (Codex):**
- **Persisted `account` column** (Task 1) separately from `nick`. Today `reminders.nick` is the result of `_get_identity()`, which silently falls back to a bare nick when the user is unauthenticated. The plan no longer fabricates `account=nick` at fire time — instead, `account` is captured at schedule time via `_account_from_msg` and stored as NULL if the user wasn't authenticated. The synthetic IrcMsg's `account` server-tag is omitted in that case, and the rate-limit tier drops to `"unregistered"`.
- **No exception text in user-visible error.** The fallback message no longer includes `str(e)[:120]` — full traceback goes to the log, the user sees a generic "Reminder action '{message}' failed." Avoids leaking API keys, internal paths, or upstream error bodies into IRC.
- **Tool surface ≠ capability surface, explicitly.** The Architecture section now states that scheduled actions get the same *tool* surface as live `@ask` but always run with `is_owner=False, capabilities=frozenset()`. Running tools 24h later under elevated rights without live consent is a footgun; this is a deliberate scope-down.
- **In-memory tuple shape is now 5-tuple** (`nick, channel, message, action_prompt, account`) — Task 3 readers updated accordingly.
- **Syntax error fixed** in the Task 1 test snippet (`db_path = str(tmp_path / "test.db"))` had an extra paren).

## Migration & Rollout Notes

**Existing reminders DO NOT upgrade.** The new column gets `DEFAULT ''`, so every pre-existing row falls into the legacy echo path. Users who set "let me know when X" reminders before deployment will still get a plain echo at fire time and need to delete + re-set them to get LLM-action behavior. This is intentional — silently re-interpreting historical text would surprise users and could ramp up cost unexpectedly.

**Rollout:** No feature flag. The behavior is gated by the parser deciding `action_prompt` is non-empty, so users who don't phrase reminders as imperatives will keep getting plain reminders.

**Cost gate:** Reuse the existing `ask` rate-limit bucket. Scheduled actions hit the same per-tier daily budget the user already sees for `@ask`. Checked at fire time using the saved account; if over budget, fall back to delivering `message` as a plain reminder with an annotation. No new config. Tier resolution at fire time is best-effort — we cannot re-check Limnoria capabilities without a live `IrcMsg`, so we default to `"registered"` (the saved account exists, after all). Owners/admins do not get an exemption at fire time — fine, since the limits are generous and this is a scheduled action, not interactive use.

---

## Task 1: Schema migration — add `action_prompt` and `account` columns

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py` (bump `SCHEMA_VERSION`, add v9 migration, update `ReminderRow`, `save_reminder`, `load_pending_reminders`)
- Modify: `plugins/llm/tests/test_persistence.py` (add migration test)

**Why two columns?** Today `reminders.nick` stores the result of `_get_identity()`, which falls back to a bare nick when the user is not authenticated. At fire time we cannot distinguish "this string is an authenticated account" from "this string is a nick fallback." That matters because action reminders re-enter the assistant pipeline, which uses `account` for tier resolution, usage tracking, and capability checks. Persisting `account` (NULL when the scheduler was unauthenticated) preserves the original auth posture.

**Step 1: Write the failing test**

Add to `plugins/llm/tests/test_persistence.py`:

```python
def test_reminders_table_has_action_prompt_column(tmp_path):
    """GIVEN fresh DB WHEN migrated THEN reminders.action_prompt exists with empty default."""
    from llm.persistence import LLMDatabase

    db = LLMDatabase(str(tmp_path / "test.db"))
    db._migrate()

    conn = db._connect()
    cols = {row[1]: row for row in conn.execute("PRAGMA table_info(reminders)").fetchall()}
    assert "action_prompt" in cols
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    assert cols["action_prompt"][2] == "TEXT"
    assert cols["action_prompt"][3] == 1  # NOT NULL
    assert cols["action_prompt"][4] == "''"
    # account is nullable — preserves "user was unauthenticated" state
    assert "account" in cols
    assert cols["account"][2] == "TEXT"
    assert cols["account"][3] == 0  # nullable


def test_existing_reminder_gets_empty_action_prompt(tmp_path):
    """GIVEN reminder saved before migration WHEN row loaded THEN action_prompt is ''."""
    from llm.persistence import LLMDatabase

    db_path = str(tmp_path / "test.db")
    db = LLMDatabase(db_path)
    db._migrate()
    db.save_reminder("evt1", "alice", "#chan", "echo me", 9999999999.0)

    rows = db.load_pending_reminders()
    assert len(rows) == 1
    assert rows[0].action_prompt == ""
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && uv run pytest tests/test_persistence.py::test_reminders_table_has_action_prompt_column tests/test_persistence.py::test_existing_reminder_gets_empty_action_prompt -v`

Expected: FAIL — `action_prompt` doesn't exist.

**Step 3: Implement the migration**

Edit `plugins/llm/src/llm/persistence.py`:

1. Bump constant at top: `SCHEMA_VERSION = 9`
2. Add new field to `ReminderRow` NamedTuple (line ~24):

```python
class ReminderRow(NamedTuple):
    """A reminder loaded from the database."""
    id: int
    event_name: str
    nick: str
    channel: str
    message: str
    fire_at: float
    created_at: float
    action_prompt: str  # NEW: empty string = echo, non-empty = LLM-action
    account: str | None  # NEW: authenticated account or None (nick-only fallback)
```

3. Add v9 migration after the `current_version < 8` block (~line 297):

```python
if current_version < 9:
    conn.executescript("""
        ALTER TABLE reminders
            ADD COLUMN action_prompt TEXT NOT NULL DEFAULT '';
        ALTER TABLE reminders
            ADD COLUMN account TEXT;
    """)
    conn.commit()
```

4. Update `save_reminder` signature (~line 396) to accept and persist the new field:

```python
def save_reminder(
    self,
    event_name: str,
    nick: str,
    channel: str,
    message: str,
    fire_at: float,
    *,
    action_prompt: str = "",
    account: str | None = None,
) -> int:
    """Save a reminder to the database.

    Args:
        event_name: Unique identifier for the reminder event.
        nick: Display identity (account-resolved by _get_identity, may
            fall back to bare nick).
        channel: IRC channel (or bot nick for PM).
        message: Display text shown when the reminder fires (or, for
            action reminders, a short human-readable description).
        fire_at: Unix timestamp when the reminder should fire.
        action_prompt: When non-empty, the reminder fires by dispatching
            this prompt through the assistant instead of echoing
            ``message``.
        account: Authenticated account name at schedule time, or None
            if the user was unauthenticated. Used at fire time to
            decide tier and request_context.account semantics — do not
            confuse with ``nick``, which may itself be a nick fallback.
    """
    conn = self._connect()
    try:
        cursor = conn.execute(
            "INSERT INTO reminders "
            "(event_name, nick, channel, message, fire_at, created_at, "
            "action_prompt, account) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_name, nick, channel, message, fire_at, time.time(),
             action_prompt, account),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        pass
```

5. Update `load_pending_reminders` (~line 451) to select and return the new column:

```python
rows = conn.execute(
    "SELECT id, event_name, nick, channel, message, fire_at, created_at, "
    "action_prompt, account "
    "FROM reminders WHERE fire_at > ? ORDER BY fire_at",
    (cutoff,),
).fetchall()
return [ReminderRow(*row) for row in rows]
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_persistence.py -v`

Expected: All persistence tests pass, including the two new ones. Existing reminder tests must still pass — we haven't changed call sites yet, but the default-arg means callers don't need to change.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(llm): add reminders.action_prompt column for LLM-triggered reminders"
```

---

## Task 2: Extend `ReminderParseResult` and parser to detect action intent

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (`ReminderParseResult` ~294, `parse_reminder` ~1963)
- Modify: `plugins/llm/tests/test_reminders.py`

**Design note — when does the parser return an action_prompt?**

The classifier err's toward echo. The LLM should only emit `action_prompt` when the user phrases an imperative directed at the bot — i.e. they want a *task done* at fire time. Examples:
- `"in 30m check if the build is green"` → action (`action_prompt = "check if the build is green"`)
- `"every 2 hours check if CVE-2026-31431 is fixed in Debian"` → action (single fire still — recurrence is out of scope for this plan; see Task 7)
- `"in 5m remind me to check the build"` → echo (the user is asking *themselves* to check)
- `"tomorrow at 3pm call Bob"` → echo (the bot can't call Bob)

**Step 1: Write the failing test**

Add to `plugins/llm/tests/test_reminders.py` (in `TestReminderParseResult`):

```python
def test_schedule_result_with_action_prompt(self) -> None:
    """GIVEN action result WHEN creating result THEN action_prompt stored."""
    result = ReminderParseResult(
        action="schedule",
        seconds=7200,
        message="check Debian CVE status",
        action_prompt="Check status of CVE-2026-31431 in Debian Bookworm and Trixie",
        confirmation="OK, I'll check in 2 hours.",
    )
    assert result.action_prompt == "Check status of CVE-2026-31431 in Debian Bookworm and Trixie"


def test_default_action_prompt_is_empty(self) -> None:
    """GIVEN minimal args WHEN creating result THEN action_prompt defaults to ''."""
    result = ReminderParseResult(action="schedule", seconds=60, message="hi")
    assert result.action_prompt == ""
```

Then add a parser test (in `TestParseReminder` or equivalent — find the existing class for parse_reminder tests):

```python
def test_parse_reminder_returns_action_prompt_for_imperative(
    self, service: MagicMock, mocker: MockerFixture
) -> None:
    """GIVEN imperative phrasing WHEN parsing THEN action_prompt set."""
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = (
        '{"action": "schedule", "seconds": 7200, '
        '"message": "Check Debian CVE-2026-31431 status", '
        '"action_prompt": "Check the status of CVE-2026-31431 in Debian 12 and 13", '
        '"confirmation": "OK, I will check in 2 hours."}'
    )
    mocker.patch.object(service, "_completion_with_tool_fallback", return_value=mock_response)

    result = service.parse_reminder("in 2h check the status of CVE-2026-31431 in Debian 12 and 13")

    assert result.action == "schedule"
    assert result.seconds == 7200
    assert result.action_prompt.startswith("Check the status")


def test_parse_reminder_omits_action_prompt_for_echo(
    self, service: MagicMock, mocker: MockerFixture
) -> None:
    """GIVEN passive 'remind me' phrasing WHEN parsing THEN action_prompt empty."""
    mock_response = mocker.MagicMock()
    mock_response.choices = [mocker.MagicMock()]
    mock_response.choices[0].message.content = (
        '{"action": "schedule", "seconds": 1800, '
        '"message": "check the build", '
        '"confirmation": "Reminder set for 30 minutes."}'
    )
    mocker.patch.object(service, "_completion_with_tool_fallback", return_value=mock_response)

    result = service.parse_reminder("in 30 minutes remind me to check the build")
    assert result.action == "schedule"
    assert result.action_prompt == ""
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v -k "action_prompt or imperative or echo"`

Expected: FAIL — `action_prompt` field doesn't exist.

**Step 3: Implement the field and parser update**

Edit `plugins/llm/src/llm/service.py`:

1. Update `ReminderParseResult` (line ~294):

```python
class ReminderParseResult(NamedTuple):
    """Result of parsing a natural language reminder request."""

    action: str  # 'schedule' or 'clarify'
    seconds: int | None = None
    message: str | None = None
    action_prompt: str = ""  # NEW: non-empty triggers LLM call at fire time
    confirmation: str = ""
    note: str | None = None
```

2. Update the `parse_reminder` system prompt (~line 2001) — append to the existing rules:

```python
system_prompt = f"""You parse reminder requests. Return JSON only, no markdown fences.

Current time: {current_time}

Response format (choose one):
{{"action": "schedule", "seconds": <int>, "message": "<string>", "action_prompt": "<string or empty>", "confirmation": "<string>", "note": "<string or null>"}}
or
{{"action": "clarify", "confirmation": "<question to ask user>"}}

Rules:
- "seconds" = seconds from now until reminder fires (must be positive)
- For relative times ("in 30 minutes"), set note to null — timezone is irrelevant
- For absolute times ("at 3pm") without a timezone, assume UTC and set note suggesting they specify next time
- If request is too vague (missing time or message), use "clarify"
- Keep confirmation concise (under 100 chars)
- Extract just the reminder message, not the time part
- For relative times ("in 30 minutes"), calculate seconds directly
- For absolute times ("at 3pm"), calculate seconds until that time

action_prompt rules:
- Set "action_prompt" to a non-empty string ONLY when the user is asking
  the bot to PERFORM A TASK at fire time (look something up, check a
  status, fetch a URL, run a query, etc.).
- Set "action_prompt" to "" (empty) for passive "remind me to X"
  phrasings where the user themselves will act.
- When in doubt, prefer "" — false positives surprise users.
- action_prompt is fed directly to the same engine that handles `@ask`.
  Write it as a self-contained instruction the user could literally type
  as `@ask <action_prompt>` and get the result they want — no time
  qualifier ("in 2 hours"), no "remind me", just the task.
- "message" should still be a short human-readable description shown
  in `@remind list` (e.g., "check Debian CVE-2026-31431 status").

Examples:
- "in 30m check if the build is green" → action_prompt: "check if the build is green"
- "in 5m remind me to check the build" → action_prompt: ""
- "every 2 hours check CVE-2026-31431 in Debian" → action_prompt: "check CVE-2026-31431 status in Debian 12 and 13" (note: only fires once; tell user via note)
- "tomorrow at 3pm call Bob" → action_prompt: ""
"""
```

3. Update the JSON-decode branch (~line 2055) to extract and pass through `action_prompt`:

```python
return ReminderParseResult(
    action="schedule",
    seconds=seconds,
    message=data.get("message", text),
    action_prompt=(data.get("action_prompt") or "").strip(),
    confirmation=data.get("confirmation", f"Reminder set for {seconds}s from now."),
    note=data.get("note"),
)
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v`

Expected: All reminder parse tests pass, including the two new ones. Existing tests still pass because the new field defaults to `""`.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_reminders.py
git commit -m "feat(llm): teach reminder parser to recognize action-prompt intent"
```

---

## Task 3: Persist `action_prompt` from `_schedule_reminder`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_schedule_reminder` ~2546, `_make_reminder_delivery_closure` ~954, `_reminders` dict shape)
- Modify: `plugins/llm/tests/test_reminders.py`

**Step 1: Write the failing test**

Add to `plugins/llm/tests/test_reminders.py`:

```python
def test_schedule_reminder_persists_action_prompt(
    self, plugin: MagicMock, mocker: MockerFixture
) -> None:
    """GIVEN parser returns action_prompt WHEN _schedule_reminder called THEN DB row stores it."""
    from llm.service import ReminderParseResult

    mocker.patch.object(
        plugin.llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=7200,
            message="check CVE status",
            action_prompt="check CVE-2026-31431 status",
            confirmation="OK.",
        ),
    )
    save_spy = mocker.spy(plugin.db, "save_reminder")

    irc = mocker.MagicMock()
    msg = mocker.MagicMock()
    msg.args = ("#chan", "text")

    plugin._schedule_reminder(irc, msg, "alice", "in 2h check CVE")

    save_spy.assert_called_once()
    # save_reminder(event_name, nick, channel, message, fire_at, action_prompt=...)
    kwargs = save_spy.call_args.kwargs
    args = save_spy.call_args.args
    action_prompt = kwargs.get("action_prompt", args[5] if len(args) > 5 else "")
    assert action_prompt == "check CVE-2026-31431 status"
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py::TestReminderHelperMethods::test_schedule_reminder_persists_action_prompt -v`

Expected: FAIL — `save_reminder` is called without `action_prompt`.

**Step 3: Update `_schedule_reminder` to pass `action_prompt` and `account`**

Edit `plugins/llm/src/llm/plugin.py` `_schedule_reminder` (~line 2546). Capture the *real* authenticated account separately from `nick` (which is the display identity from `_get_identity` and may itself be a nick fallback):

```python
reminder_message = result.message or text
event_name = f"llm_remind_{uuid.uuid4().hex[:12]}"
# Capture authenticated account at schedule time. None when the user
# was unauthenticated — at fire time we'll treat such reminders as
# "unregistered" tier and refuse to fabricate an account in the
# request_context.
account = self._account_from_msg(irc, msg)

deliver = self._make_reminder_delivery_closure(
    nick, channel, reminder_message, event_name,
    action_prompt=result.action_prompt,
    account=account,
)

try:
    schedule.addEvent(deliver, time.time() + result.seconds, name=event_name)
    with self._reminders_lock:
        # Tuple shape extended: (nick, channel, message, action_prompt, account)
        self._reminders[event_name] = (
            nick, channel, reminder_message, result.action_prompt, account
        )

    self.db.save_reminder(
        event_name,
        nick,
        channel,
        reminder_message,
        time.time() + result.seconds,
        action_prompt=result.action_prompt,
        account=account,
    )
    ...
```

Also update `_make_reminder_delivery_closure` signature (~line 954) to accept the new kwargs (we'll wire actual behavior in Task 4):

```python
def _make_reminder_delivery_closure(
    self,
    nick: str,
    channel: str,
    message: str,
    event_name: str,
    *,
    action_prompt: str = "",
    account: str | None = None,
):
```

For now, keep the body unchanged — Task 4 will branch on `action_prompt`.

Update `_reload_reminders` (~line 991) to pass both fields through:

```python
deliver = self._make_reminder_delivery_closure(
    nick, channel, message, event_name,
    action_prompt=reminder.action_prompt,
    account=reminder.account,
)
...
with self._reminders_lock:
    self._reminders[event_name] = (
        nick, channel, message, reminder.action_prompt, reminder.account
    )
```

Also fix `_find_user_reminder` (~line 2522) — its `for name, (owner, _, _) in self._reminders.items()` will raise `ValueError: too many values to unpack` once the dict carries 5-tuples:

```python
def _find_user_reminder(self, nick: str, reminder_id: str) -> str | None:
    with self._reminders_lock:
        for name, data in self._reminders.items():
            owner = data[0]
            if name.endswith(f"_{reminder_id}") and owner.lower() == nick.lower():
                return name
        return None
```

Also update `_get_user_reminders` and `_format_reminders` (~line 2477-2510) — the tuple now has 5 elements. Update the unpacking:

```python
ReminderTuple = tuple[str, str, str, str, str | None]
# (nick, channel, message, action_prompt, account)


def _get_user_reminders(self, nick: str) -> list[tuple[str, ReminderTuple]]:
    """Get reminders belonging to a specific user."""
    with self._reminders_lock:
        return [
            (name, data)
            for name, data in self._reminders.items()
            if data[0].lower() == nick.lower()
        ]


def _format_reminders(self, reminders: list[tuple[str, ReminderTuple]]) -> str:
    if not reminders:
        return _("You have no pending reminders.")
    parts = []
    for name, data in reminders:
        message = data[2]
        action_prompt = data[3] if len(data) > 3 else ""
        preview = message if len(message) <= 50 else message[:47] + "..."
        reminder_id = name.split("_")[-1]
        # Mark action reminders so users can tell them apart in `@remind list`
        marker = " [auto]" if action_prompt else ""
        parts.append(f"#{reminder_id}: {preview}{marker}")
    return ", ".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v`

Expected: New test passes; all existing reminder tests still pass (some may need a small fixture update if they assert exact tuple shape — fix as needed).

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_reminders.py
git commit -m "feat(llm): persist action_prompt on reminder schedule"
```

---

## Task 4: Wire LLM-action delivery into the closure (with rate-limit + fallback)

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_make_reminder_delivery_closure` ~954)
- Modify: `plugins/llm/tests/test_reminders.py`

This is the heart of the feature. The delivery closure needs to:

1. If `action_prompt == ""` → existing echo behavior (unchanged).
2. Otherwise:
   a. Check the existing `ask` rate limit for the saved account at tier `"registered"`.
   b. If over limit → deliver fallback: `"{nick}: Reminder: {message} (action skipped — daily ask limit reached)"`.
   c. Else: synthesize a minimal `IrcMsg`, build an `AssistantRequestContext`, call `assistant_request` with the **full** callback set (search/fetch/code/draw/cleanup/list_reminders/set_reminder/delete_reminder), deliver the response.
   d. On any exception → deliver `"{nick}: Reminder action '{message}' failed: {short_error}. (Set this reminder again to retry.)"`

**Step 1: Write the failing tests**

Add to `plugins/llm/tests/test_reminders.py` in a new `TestReminderActionDelivery` class:

```python
class TestReminderActionDelivery:
    """Tests for LLM-action reminder delivery."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture) -> MagicMock:
        from llm.plugin import LLM
        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        return LLM(mock_irc)

    def test_action_delivery_invokes_assistant_with_full_callbacks(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN action_prompt set WHEN delivery fires THEN assistant_request called with full tool surface."""
        mock_world_irc = mocker.MagicMock()
        mocker.patch("llm.plugin.world.ircs", [mock_world_irc])

        assistant_spy = mocker.patch.object(
            plugin.llm_service,
            "assistant_request",
            return_value=mocker.MagicMock(content="CVE is fixed in Bookworm.", grounding_used=False),
        )
        # Disable rate limit
        mocker.patch.object(plugin, "_is_rate_limited", return_value=False)

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "check CVE status", "evt1",
            action_prompt="check CVE-2026-31431 in Debian"
        )
        deliver()

        assistant_spy.assert_called_once()
        # Verify reuse: same callback surface as @ask
        kwargs = assistant_spy.call_args.kwargs
        for fn in (
            "search_fn", "fetch_fn", "code_fn", "draw_fn", "cleanup_fn",
            "list_reminders_fn", "set_reminder_fn", "delete_reminder_fn",
        ):
            assert callable(kwargs.get(fn)), f"{fn} should be wired"
        assert mock_world_irc.queueMsg.called

    def test_action_delivery_uses_ask_rate_limit_bucket(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN delivery fires THEN _is_rate_limited called with command='ask'."""
        mock_world_irc = mocker.MagicMock()
        mocker.patch("llm.plugin.world.ircs", [mock_world_irc])
        mocker.patch.object(
            plugin.llm_service,
            "assistant_request",
            return_value=mocker.MagicMock(content="ok", grounding_used=False),
        )
        rl_spy = mocker.patch.object(plugin, "_is_rate_limited", return_value=False)

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "msg", "evt", action_prompt="do thing"
        )
        deliver()

        rl_spy.assert_called_once()
        assert rl_spy.call_args.args[0] == "ask" or rl_spy.call_args.kwargs.get("command") == "ask"

    def test_action_delivery_falls_back_on_rate_limit(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN rate limited WHEN delivery fires THEN echo fallback sent, no assistant call."""
        mock_world_irc = mocker.MagicMock()
        mocker.patch("llm.plugin.world.ircs", [mock_world_irc])

        assistant_spy = mocker.patch.object(plugin.llm_service, "assistant_request")
        mocker.patch.object(plugin, "_is_rate_limited", return_value=True)

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "check CVE status", "evt1",
            action_prompt="check CVE-2026-31431"
        )
        deliver()

        assistant_spy.assert_not_called()
        msg_text = mock_world_irc.queueMsg.call_args.args[0].args[1]
        assert "limit" in msg_text.lower()

    def test_action_delivery_falls_back_on_exception(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN assistant_request raises WHEN delivery fires THEN fallback echo sent."""
        mock_world_irc = mocker.MagicMock()
        mocker.patch("llm.plugin.world.ircs", [mock_world_irc])
        mocker.patch.object(plugin, "_is_rate_limited", return_value=False)
        mocker.patch.object(
            plugin.llm_service,
            "assistant_request",
            side_effect=RuntimeError("boom"),
        )

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "check CVE", "evt1", action_prompt="check CVE"
        )
        deliver()

        msg_text = mock_world_irc.queueMsg.call_args.args[0].args[1]
        assert "failed" in msg_text.lower()

    def test_echo_delivery_unchanged(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN action_prompt empty WHEN delivery fires THEN legacy echo behavior."""
        mock_world_irc = mocker.MagicMock()
        mocker.patch("llm.plugin.world.ircs", [mock_world_irc])
        assistant_spy = mocker.patch.object(plugin.llm_service, "assistant_request")

        deliver = plugin._make_reminder_delivery_closure(
            "alice", "#chan", "echo me", "evt1", action_prompt=""
        )
        deliver()

        assistant_spy.assert_not_called()
        msg_text = mock_world_irc.queueMsg.call_args.args[0].args[1]
        assert "echo me" in msg_text
```

**Step 2: Run tests to verify they fail**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py::TestReminderActionDelivery -v`

Expected: FAIL — closure ignores `action_prompt`.

**Step 3: Add a silent rate-limit helper**

`_check_rate_limit` (`plugin.py:1441`) is the single source of truth for the rate-limit semantics — it always records the timestamp, supports `enforceRateLimits` shadow mode, and emits the same telemetry the rest of the codebase relies on. We can't reuse it directly because it calls `irc.error()` on the over-limit path, which would deliver a raw error message to the channel instead of our customized fallback.

Add a sibling helper that mirrors the bookkeeping but skips `irc.error`:

```python
def _check_rate_limit_silent(
    self, command: str, account: str, now: float, *, tier: str
) -> bool:
    """Record a rate-limit hit and return True if over the limit.

    Mirrors :meth:`_check_rate_limit` minus the irc.error reply, for
    callers (scheduled-action delivery) that want to format their own
    fallback message. Honors ``enforceRateLimits`` shadow mode the
    same way: when shadow mode is on, always returns False but still
    records and logs.
    """
    over_limit = self._is_rate_limited(command, account, now, tier=tier)
    self._record_rate_limit_hit(command, account, now)

    if over_limit:
        enforce = self.registryValue("enforceRateLimits")
        max_count, window = self._get_tier_limits(command, tier)
        if enforce:
            self.log.info(
                "rate_limited (silent) command=%s account=%s tier=%s limit=%d window=%ss",
                command, account, tier, max_count, window,
            )
            return True
        # Shadow mode: log but don't enforce
        self.log.info(
            "rate_limit_shadow (silent) command=%s account=%s tier=%s limit=%d window=%ss",
            command, account, tier, max_count, window,
        )
        return False
    return False
```

Place it next to `_check_rate_limit` (~line 1441). Add a quick unit test asserting both the over-limit and shadow-mode branches.

**Step 4: Implement the action branch in the closure**

Edit `plugins/llm/src/llm/plugin.py` `_make_reminder_delivery_closure` (~line 954). Replace the body with two delivery paths:

```python
def _make_reminder_delivery_closure(
    self,
    nick: str,
    channel: str,
    message: str,
    event_name: str,
    *,
    action_prompt: str = "",
    account: str | None = None,
):
    """Create a reminder delivery closure with error handling."""
    lock = self._reminders_lock
    target = channel if ircutils.isChannel(channel) else nick

    def _send(active_irc, text: str) -> None:
        safe = self.llm_service.sanitize_output(text)
        active_irc.queueMsg(ircmsgs.privmsg(target, f"{nick}: {safe}"))

    def _cleanup() -> None:
        with lock:
            self._reminders.pop(event_name, None)
        self.db.delete_reminder(event_name)

    def _deliver_echo(active_irc) -> None:
        _send(active_irc, f"Reminder: {message}")

    def _deliver_action(active_irc) -> None:
        # Guard for early-startup ircs without a populated nick.
        if not getattr(active_irc, "nick", None):
            self.log.warning(
                "Reminder %s fired but irc has no nick yet; falling back to echo",
                event_name,
            )
            _deliver_echo(active_irc)
            return

        # Rate limit at fire time, reusing the existing `ask` bucket
        # via _check_rate_limit_silent (records, honors enforceRateLimits
        # shadow mode, no irc.error reply).
        now = time.time()
        # Tier reflects the auth state CAPTURED AT SCHEDULE TIME. If the
        # user had no authenticated account, treat the reminder as
        # unregistered tier — do NOT promote a bare nick to "registered"
        # by fabricating an account string.
        rl_account = account if account else nick
        rl_tier = "registered" if account else "unregistered"
        over_limit = self._check_rate_limit_silent("ask", rl_account, now, tier=rl_tier)
        if over_limit:
            _send(
                active_irc,
                f"Reminder: {message} (action skipped — daily ask limit reached)",
            )
            return

        # Synthesize a minimal IrcMsg so msg-bound callbacks
        # (_draw_for_assistant, _remind_set_for_assistant, etc.) work.
        # account-tag carries the saved authenticated account ONLY if
        # one was captured at schedule time — never fabricate one from
        # a nick fallback. prefix is a placeholder; args[0] = the actual
        # reply target so downstream code like _begin_typing(msg.args[0])
        # targets the right place.
        msg_target = channel if ircutils.isChannel(channel) else nick
        server_tags = {"account": account} if account else {}
        synthetic_msg = ircmsgs.IrcMsg(
            prefix=f"{nick}!~remind@scheduled",
            command="PRIVMSG",
            args=(msg_target, ""),
            server_tags=server_tags,
        )

        # Cap recursive scheduling: a single LLM action may schedule at
        # most ONE follow-up reminder (the "check every 2h until done"
        # pattern from the kickoff conversation). Without this cap, a
        # single fire's meta loop (up to metaMaxSteps=12 iterations)
        # could schedule 12 children; each fires later and could spawn
        # 12 more — exponential. Per-fire bookkeeping is local to the
        # closure since multiple reminders can fire concurrently.
        nested_set_count = [0]
        MAX_NESTED_SETS = 1

        def _capped_set_reminder(text: str) -> str:
            if nested_set_count[0] >= MAX_NESTED_SETS:
                return (
                    "Cannot schedule another reminder from inside an "
                    "automated reminder action (limit reached)."
                )
            nested_set_count[0] += 1
            return self._remind_set_for_assistant(active_irc, synthetic_msg, nick, text)

        from .service import AssistantRequestContext

        request_context = AssistantRequestContext(
            entry_route="remind_action",
            profile="chat",
            nick=nick,
            raw_nick=nick,
            account=account,  # NULL if user was unauthenticated at schedule time
            channel=channel,
            is_private=not ircutils.isChannel(channel),
            is_owner=False,         # deliberately scoped down — see Architecture
            capabilities=frozenset(),  # likewise
        )

        history, channel_history = self._gather_history(nick, channel)
        memories = self._get_user_memories(nick)
        user_instruction = self.db.get_instruction(nick)
        ask_prompt = self.registryValue("askSystemPrompt", channel)
        effective_prompt = (
            f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None
        )

        with self._allow_concurrent():
            # Mirror _ask_core's assistant_request call site (plugin.py:1918).
            # Full callback surface: scheduled actions are first-class @ask
            # invocations and get the same tool access. Recursion is bounded
            # by the shared ask rate-limit bucket.
            result = self.llm_service.assistant_request(
                action_prompt,
                request_context=request_context,
                db=self.db,
                context=self.context,
                bot_nick=active_irc.nick,
                history=history,
                channel_history=channel_history,
                memories=memories,
                system_prompt=effective_prompt,
                irc=active_irc,
                msg=synthetic_msg,
                search_fn=lambda q: self.llm_service.search_completion(q, channel=channel),
                fetch_fn=lambda u: self.llm_service.url_completion(u, channel=channel),
                code_fn=lambda p: self._code_for_assistant(p, channel),
                draw_fn=lambda p: self._draw_for_assistant(active_irc, synthetic_msg, p),
                cleanup_fn=lambda n: self._run_memory_cleanup(n, channel),
                list_reminders_fn=lambda: self._get_user_reminders(nick),
                set_reminder_fn=_capped_set_reminder,
                delete_reminder_fn=lambda r: self._remind_delete_for_assistant(nick, r),
            )

        response = (result.content or "").strip()
        if not response:
            _send(active_irc, f"Reminder: {message} (action returned empty response)")
            return

        prefix = f"Reminder ({message}): " if message else "Reminder: "
        _send(active_irc, f"{prefix}{response}")

    def _deliver() -> None:
        try:
            for active_irc in world.ircs:
                try:
                    if action_prompt:
                        _deliver_action(active_irc)
                    else:
                        _deliver_echo(active_irc)
                except Exception:
                    # Log full traceback for ops; show the user a
                    # generic message — exception text may carry API
                    # keys, internal paths, or upstream error bodies.
                    self.log.exception(
                        "Reminder action delivery failed for %s", event_name
                    )
                    try:
                        _send(
                            active_irc,
                            f"Reminder action '{message}' failed. "
                            "(Set this reminder again to retry.)",
                        )
                    except Exception:
                        self.log.exception("Failed to deliver fallback message")
                break
        finally:
            _cleanup()

    return _deliver
```

**Step 5: Add a recursion-cap test**

```python
def test_action_delivery_caps_nested_reminder_scheduling(
    self, plugin: MagicMock, mocker: MockerFixture
) -> None:
    """GIVEN action runs WHEN it tries to schedule >1 nested reminders THEN second is rejected."""
    mock_world_irc = mocker.MagicMock()
    mocker.patch("llm.plugin.world.ircs", [mock_world_irc])
    mocker.patch.object(plugin, "_check_rate_limit_silent", return_value=False)

    captured_set_fn = []

    def capture(*args, **kwargs):
        captured_set_fn.append(kwargs.get("set_reminder_fn"))
        return mocker.MagicMock(content="ok", grounding_used=False)

    mocker.patch.object(plugin.llm_service, "assistant_request", side_effect=capture)
    mocker.patch.object(
        plugin, "_remind_set_for_assistant", return_value="Reminder set."
    )

    deliver = plugin._make_reminder_delivery_closure(
        "alice", "#chan", "msg", "evt", action_prompt="do thing"
    )
    deliver()

    set_fn = captured_set_fn[0]
    assert set_fn("first") == "Reminder set."
    second = set_fn("second")
    assert "limit reached" in second.lower()
```

**Step 6: Run tests to verify they pass**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v`

Expected: All `TestReminderActionDelivery` tests pass; existing echo and persistence tests still pass.

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_reminders.py
git commit -m "feat(llm): dispatch LLM-action reminders through assistant at fire time"
```

---

## Task 5: Restore action reminders across bot restarts

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_reload_reminders` ~991) — already partially updated in Task 3, verify here
- Modify: `plugins/llm/tests/test_reminders.py`

**Step 1: Write the failing test**

Add to `test_reminders.py`:

```python
def test_reload_preserves_action_prompt(
    self, plugin: MagicMock, mocker: MockerFixture
) -> None:
    """GIVEN persisted action reminder WHEN reloaded THEN closure has action_prompt."""
    from llm.persistence import ReminderRow
    import time as _time

    mocker.patch.object(
        plugin.db,
        "load_pending_reminders",
        return_value=[
            ReminderRow(
                id=1,
                event_name="evt_persist",
                nick="alice",
                channel="#chan",
                message="check CVE",
                fire_at=_time.time() + 3600,
                created_at=_time.time(),
                action_prompt="check CVE-2026-31431 status",
                account="alice",
            )
        ],
    )
    closure_spy = mocker.spy(plugin, "_make_reminder_delivery_closure")
    irc = mocker.MagicMock()

    plugin._reload_reminders(irc)

    closure_spy.assert_called_once()
    assert closure_spy.call_args.kwargs["action_prompt"] == "check CVE-2026-31431 status"
```

**Step 2: Run test to verify it fails**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v -k reload_preserves`

Expected: PASS if Task 3 already wired `_reload_reminders` correctly; otherwise FAIL.

**Step 3: Implement (if needed)**

Verify the `_reload_reminders` change from Task 3 passes `action_prompt=reminder.action_prompt`. If not, fix it now.

**Step 4: Run all reminder tests**

Run: `cd plugins/llm && uv run pytest tests/test_reminders.py -v`

Expected: All pass.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_reminders.py
git commit -m "test(llm): verify action_prompt survives reminder reload"
```

---

## Task 6: Update `@remind list` formatting & docs

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — docstring of `remind` command (~line 2628), `_format_reminders` already updated in Task 3
- Modify: `plugins/llm/README.md` (if user-facing docs exist)
- Modify: `plugins/llm/tests/test_reminders.py`

**Step 1: Write the failing test**

```python
def test_format_reminders_marks_action_reminders(self, plugin: MagicMock) -> None:
    """GIVEN action reminder WHEN formatting THEN marker shown."""
    plugin._reminders = {
        "llm_remind_aaa1": ("alice", "#chan", "check CVE", "check CVE status", "alice"),
        "llm_remind_bbb2": ("alice", "#chan", "echo this", "", None),
    }
    formatted = plugin._format_reminders(plugin._get_user_reminders("alice"))
    assert "[auto]" in formatted
    # Echo reminder should NOT be marked
    parts = formatted.split(", ")
    auto_count = sum(1 for p in parts if "[auto]" in p)
    assert auto_count == 1
```

**Step 2: Implement (already partly done in Task 3)** — verify the marker logic.

**Step 3: Update the `remind` command docstring**

In `plugins/llm/src/llm/plugin.py`, edit the `remind` docstring (~line 2628) to add an example:

```python
"""[<reminder text> | list | del(ete) <id> [<id>...] | clear]

Set and manage reminders using natural language. If your reminder
asks the bot to *do* something (look something up, check a status),
it will run that as an LLM query at fire time. Otherwise it just
echoes your text. Reminders marked [auto] in `list` are LLM actions.

Examples:
  @remind in 30 minutes check the build
  @remind in 2 hours check status of CVE-2026-31431 in Debian
  @remind list
  @remind delete abc1
  @remind clear
"""
```

**Step 4: Run, verify, commit**

```bash
cd plugins/llm && uv run pytest tests/test_reminders.py -v
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_reminders.py
git commit -m "feat(llm): mark LLM-action reminders in @remind list"
```

---

## Task 7: Out of scope (explicitly NOT in this plan)

Document these in the plan only — do not implement:

1. **Recurrence** ("every 2 hours"). The current schema fires once. The parser will silently fold "every 2 hours" into a single 2-hour delay and add a note. A follow-up plan can add a `recurrence_seconds` column and reschedule on fire. The interaction sample shows the user wants this — flag it as the obvious next step but don't bundle it here; the recurrence design has its own UX/cost concerns (when to stop, how to cancel one of N future fires, etc.).
2. **A `_record_rate_limit_hit` for echo reminders.** Echo reminders are essentially free — no need to throttle.
3. **Streaming intermediate progress** during the meta loop back to the channel.
4. **Per-channel disable** of action reminders (could be config later).

End your work after Task 6. Verify the full test suite passes and run a manual smoke test if possible (set a short action reminder in dev, watch the closure fire).

---

## Final verification

```bash
cd plugins/llm && uv run pytest -q
```

Expected: full green. If anything outside `test_reminders.py` / `test_persistence.py` regressed, the most likely cause is the `_reminders` dict tuple shape change (3-tuple → 5-tuple). Grep for `self._reminders[` and ensure all readers handle the extra elements.

```bash
cd plugins/llm && rg "self\._reminders\[" src/
```

---

## Out-of-band sanity check

After deploying, ask the bot in a test channel:
- `@ask in 1 minute check the current time` — expect `[auto]` marker on `@remind list`, action runs at fire time.
- `@ask in 1 minute remind me to check the time` — expect plain echo, no marker.

If the parser misclassifies, iterate on the system prompt in `parse_reminder` (Task 2) — that's the single highest-impact lever for UX.
