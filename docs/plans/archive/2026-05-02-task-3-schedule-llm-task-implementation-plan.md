---
status: ready-for-review
date: 2026-05-02
phase: 2
task: 3 (schedule_llm_task native tool — Scheduler-as-agent)
design_plan: docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md
predecessor_implementation_plans:
  - docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md
  - (Task 5a — config consolidation, shipped as commits 6dc0f92..5a74bf9)
---

# Limnoria Tool Bridge — Phase 2 Task 3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship a native LLM tool `schedule_llm_task` that schedules a future
`@ask`-equivalent invocation, with companion `list_scheduled_llm_tasks` and
`cancel_scheduled_llm_task` tools. Recurring tasks (numeric cadence and
RRULE-driven) are supported and survive bot restart. Loop hazards are
contained by a depth cap of 1; runaway creation is bounded by a per-creator
budget. Tool descriptions distinguish this primitive from `set_reminder`
clearly enough that the LLM picks correctly.

**Architecture:** New SQLite table `scheduled_llm_tasks` (schema v13),
persisted by `LLMDatabase`. Scheduling glue lives next to the existing
reminder machinery in `service.py` and uses raw `supybot.schedule.addEvent`
+ DB-backed restore (the same pattern as reminders) — **not** the Scheduler
plugin's pickle. The row stores `network` because raw scheduler events do not
carry Scheduler-plugin metadata and recurring reschedules need the network
after the first fire. NL parsing reuses the existing
`LLMService.parse_reminder` by parsing a composed text
`"{when_natural} {prompt}"`; the parsed message/action text is ignored and
the tool's structured `prompt` is stored verbatim. Three native tools register
through the existing `assistant.py` plumbing (`ASSISTANT_TOOLS` +
`_TOOL_SPEC_OVERRIDES` + `AssistantToolExecutor._tool_*`). Depth cap is
enforced in the service method by reading a fresh
`msg.tags["llm_schedule_depth"]` set on the rehydrated `IrcMsg` inside our
fire closure (no Limnoria-side alias command needed because we own the
closure end-to-end). Per-creator per-channel budget lives behind a new
`bridgeScheduledTaskLimit: NonNegativeInteger` channel registry value
(default 5; `0` disables scheduling), enforced at create time.

**Fire-time dispatch path (load-bearing).** The fire closure does NOT call
`LLM.ask`. Directly calling the wrapped command would bypass normal Limnoria
command dispatch/threading and would enter the command wrapper rather than
the service facade we need. We mirror the existing reminder-fire pattern
(`plugin.py:1134-1259`): perform the same manual ask-rate-limit check,
synthesise an `AssistantRequestContext`, build a fresh `IrcMsg` from the
persisted wire string, set `msg.tags["llm_schedule_depth"]=1`, call
`LLMService.assistant_request(...)` directly, sanitize generated IRC output,
and log usage for the schedule owner. The direct service call bypasses the
wrap/command-dispatch layer, so these guardrails must live in the fire path.

**Tech stack:** Python 3.14, Limnoria (`supybot.schedule`,
`supybot.ircmsgs.IrcMsg`, `supybot.callbacks`, `supybot.conf`,
`supybot.registry`), pytest, sqlite3 + WAL. All build/test commands run via
`uv run`. AGENTS.md:22-24 says: after editing Python files, run `make lint`
and `make typecheck`; before declaring the task done, run `make preflight`.
These are agent obligations, not auto-hooks — the implementer must invoke
them explicitly. The Step 5 commit blocks below name the gates that need to
pass before each commit.

**Naming note:** the public tool names are `schedule_llm_task`,
`list_scheduled_llm_tasks`, `cancel_scheduled_llm_task` (plural list,
matching `list_memories` / `list_reminders`). The supporting Python names
are `_tool_schedule_llm_task`, `_tool_list_scheduled_llm_tasks`,
`_tool_cancel_scheduled_llm_task` on `AssistantToolExecutor`, and
`schedule_llm_task_fn` / `list_scheduled_llm_tasks_fn` /
`cancel_scheduled_llm_task_fn` on the executor's constructor (matches the
existing `set_reminder_fn` / `delete_reminder_fn` pattern). The new channel
registry key is `bridgeScheduledTaskLimit` — the registry key is what
operators type into IRC `config channel` commands and is load-bearing.

**Key design pivot from the Phase 2 plan:** the design plan said "Scheduler's
pickle persists periodic events for free" and proposed routing through
`schedule.addEvent` / `schedule.addPeriodicEvent` directly. Investigation
(see "Pre-plan investigation findings" below and Task 0) showed two
problems:

1. The LLM plugin's existing reminder system uses raw
   `supybot.schedule.addEvent` + DB-backed restore; it does NOT use the
   Scheduler **plugin**'s pickle persistence. To get persistence from
   Scheduler-plugin's pickle we would have to call its private `_add` /
   `_repeat` and accept its own command-string dispatch path.
2. `IrcMsg.__reduce__` (`supybot/ircmsgs.py:363-364`) rehydrates from the
   wire string only; `msg.tags` is dropped on pickle. Empirically verified
   2026-05-02: `m.tag('llm_schedule_depth', 1)` + `pickle.loads(pickle.dumps(m))`
   yields `m2.tags == {}`. The Phase 2 plan flagged this as a risk; it is
   real.

The pivot, applied throughout this plan: persist in our SQLite table,
schedule via raw `supybot.schedule.addEvent`, rebuild on startup the same
way reminders do (`service.py:1338-1340`), and set
`msg.tags["llm_schedule_depth"]=1` on the freshly-built `IrcMsg` *inside our
own fire closure*, before dispatch. No reliance on Scheduler-plugin pickle
behaviour. No `@askscheduled` Limnoria alias command needed (the design plan
proposed it as a fallback for tag-pickle failure; the pivot makes the
fallback unnecessary because the tag is set after rehydration, not before
pickling).

---

## Pre-plan investigation findings

These observations shaped the plan. They are reproduced here so the
implementer doesn't have to re-derive them. Cite them in commit messages
where relevant.

1. **`supybot.schedule` API (verified `.venv/lib/python3.14/site-packages/supybot/schedule.py`):**
   - `addEvent(f, t, name=None, args=[], kwargs={})` — returns `name`
     (auto-incrementing int when `name=None`). `f` is a no-arg callable;
     `t` is a unix timestamp.
   - `removeEvent(name)` — pops the named event; raises `KeyError` on miss.
   - `addPeriodicEvent(f, t, name=None, now=True, args=[], kwargs={}, count=None)`
     — uses `makePeriodicWrapper` to re-schedule itself. **`now=True` runs
     `f` immediately**; for our purposes always pass `now=False`.
   - These are module-level shortcuts wrapping the global
     `schedule.schedule` singleton.
2. **Periodic-event persistence (verified `.venv/.../supybot/plugins/Scheduler/plugin.py:79-119`):**
   - The Scheduler **plugin** persists `self.events` to `Scheduler.pickle`
     via `world.flushers`. `_restoreEvents` rebuilds both `'single'` and
     `'repeat'` events on init.
   - But the LLM plugin's existing reminder code uses
     `schedule.addEvent` directly (`service.py:1340, 3447, 3627`) and
     restores from our own DB on startup. We mirror that pattern.
3. **`msg.tags` pickle round-trip (verified empirically 2026-05-02):**
   - `IrcMsg.__reduce__` returns `(self.__class__, (str(self),))`
     (`ircmsgs.py:363-364`). After `pickle.loads(pickle.dumps(m))`,
     `m2.tags == {}`. Server-side IRCv3 tags (`m.server_tags`) DO survive
     because they are in the wire string.
   - Our pivot avoids the issue: we never pickle a tagged msg. The depth
     tag is set on the rehydrated `IrcMsg` inside our fire closure, just
     before dispatch.
4. **`parse_reminder` reuse (`service.py:2029-2237`):**
   - Returns `ReminderParseResult(action, seconds, message, confirmation,
     note, action_prompt, recurrence_seconds, recurrence_rrule, watch_mode)`.
     Validates RRULE via `dateutil.rrule.rrulestr`. `seconds` is the
     first-fire offset. `recurrence_seconds` and `recurrence_rrule` are
     mutually exclusive.
   - All three fields we need (`seconds`, `recurrence_seconds`,
     `recurrence_rrule`) come for free; we reuse `parse_reminder` as-is.
     `watch_mode` is also useful for scheduled LLM tasks: if the parser
     identifies a check-until/watch request, the fire path suppresses
     `[silent]`; otherwise `[silent]` is treated as ordinary model output.
5. **Persistence layer (`persistence.py`):**
   - `SCHEMA_VERSION = 12`. Migrations are version-gated
     (`if current_version < N`). Adding `if current_version < 13:` with the
     new table is a one-bumb migration.
   - Thread-local connections + WAL; we add helpers alongside
     `save_reminder` / `load_pending_reminders`.
6. **Native tool registration (`assistant.py`):**
   - Three integration points: append to `ASSISTANT_TOOLS` (schema), add an
     entry to `_TOOL_SPEC_OVERRIDES` (capability/visibility), implement
     `_tool_*` handler methods on `AssistantToolExecutor`. New `*_fn`
     callbacks pass through the executor's `__init__`. The `_reminder_fns`
     pattern in `plugin.py:3218-3264` is what we mirror.
7. **Identity drift (open question from the design plan):**
   - The persisted msg's prefix (`nick!user@host`) is captured at create
     time. If the user disconnects or changes nick, that prefix is stale.
     AfterNet's auth is account-tag-based via IRCv3 `account-tag`
     `server_tags`, which DO survive the wire round-trip. Capability
     resolution at fire time should still find the user via account-tag.
     We capture the account explicitly in our DB row as belt-and-suspenders.
     PR-review item if the implementer sees a clean way to refresh the
     hostmask at fire time without forging it; otherwise document and ship.

---

## Pre-flight (do first, do not skip)

### Task 0: Verify codebase facts before touching anything

**Step 0.1: Confirm the integration points the plan refers to are still where the design says.**

```bash
grep -n "SCHEMA_VERSION\|def save_reminder\|def load_pending_reminders\|def delete_reminder\b" \
    plugins/llm/src/llm/persistence.py
grep -n "ASSISTANT_TOOLS\|_TOOL_SPEC_OVERRIDES\|class AssistantToolExecutor\|def _tool_set_reminder" \
    plugins/llm/src/llm/assistant.py
grep -n "def parse_reminder\|def _schedule_reminder\|def _mechanical_reschedule\|schedule.addEvent" \
    plugins/llm/src/llm/service.py
grep -n "_reminder_fns\|bridgeAllowedPlugins\|bridgeAllowMutating\|bridgeEnabled" \
    plugins/llm/src/llm/plugin.py
grep -n "bridgeAllowedPlugins\|bridgeAllowMutating\|bridgeEnabled\|bridgeScheduledTaskLimit" \
    plugins/llm/src/llm/config.py
```

Expected (verified 2026-05-02):

- `persistence.py`: `SCHEMA_VERSION = 12` (line 18). `save_reminder` ~456,
  `load_pending_reminders` ~548, `delete_reminder` ~528.
- `assistant.py`: `ASSISTANT_TOOLS` list ~112, `_TOOL_SPEC_OVERRIDES` ~563,
  `class AssistantToolExecutor` ~623, `def _tool_set_reminder` ~913.
- `service.py`: `parse_reminder` ~2029, `_mechanical_reschedule` ~3369,
  `_schedule_reminder` ~3538, `schedule.addEvent(...)` calls at 1340,
  3447, 3627.
- `plugin.py`: `_reminder_fns` ~3218 (returns `set_reminder_fn` /
  `delete_reminder_fn` / `cancel_all_reminders_fn` /
  `list_reminders_fn`), `bridgeAllowedPlugins` ~1583, `bridgeAllowMutating`
  ~1583 (post-Task-1).
- `config.py`: `bridgeAllowedPlugins`, `bridgeAllowMutating`,
  `bridgeEnabled`, `bridgeDebugInChannel` already registered;
  `bridgeScheduledTaskLimit` should NOT match.

If line numbers have drifted (post-Task-1 commits landed first), update
references below before implementing — the *symbols* are the truth, not
the line numbers.

**Step 0.2: Confirm baseline tests are green.**

```bash
uv run pytest plugins/llm/tests -q
```

Expected: all green. Stop and report on any pre-existing failure — fixing
unrelated failures is not in scope.

**Step 0.3: Confirm none of the new symbols already exist.**

```bash
grep -rn "scheduled_llm_tasks\|schedule_llm_task\|bridgeScheduledTaskLimit\|llm_schedule_depth" \
    plugins/llm/ docs/
```

Expected: matches only inside this plan file
(`docs/plans/2026-05-02-task-3-schedule-llm-task-*.md`) and the kickoff
prompt. If a stray match appears elsewhere, an earlier draft has
landed — reconcile before continuing.

**Step 0.4: Re-verify the pickle behaviour referenced throughout this plan.**

```bash
uv run python - <<'PY'
import pickle
from supybot.ircmsgs import IrcMsg
m = IrcMsg(s=':rdrake!user@host PRIVMSG #test :@ask hi')
m.tag('llm_schedule_depth', 1)
m.tag('account', 'rdrake')
data = pickle.dumps(m)
m2 = pickle.loads(data)
assert m2.tags == {}, ('tags survived pickle?', m2.tags)
assert m2.nick == 'rdrake', m2.nick
assert m2.args == ('#test', '@ask hi'), m2.args
print('OK — tags lost (expected); prefix/args preserved.')
PY
```

If this fails (i.e. tags survive), the design pivot premise is wrong and
we revert to the design plan's `msg.tags`-based depth cap. Stop and
re-read this plan's Architecture section.

**Step 0.5: Commit nothing. Task 0 is read-only.**

---

## A — Foundation: persistence

### Task A1: Bump SCHEMA_VERSION; create `scheduled_llm_tasks` table

**Files:**

- Modify: `plugins/llm/src/llm/persistence.py` (bump `SCHEMA_VERSION`,
  append a `current_version < 13` block in `_migrate`).
- Modify: `plugins/llm/tests/test_persistence.py` (add a migration test).

**Background — schema:**

The new table tracks LLM-scheduled task metadata. One row per active
schedule. `event_name` mirrors the reminders shape (used as the scheduler
event name + a stable cancel handle). Columns:

| Column                 | Type    | Notes                                                                           |
| ---------------------- | ------- | ------------------------------------------------------------------------------- |
| `id`                   | INTEGER | PRIMARY KEY AUTOINCREMENT                                                       |
| `event_name`           | TEXT    | UNIQUE NOT NULL — the scheduler event name (e.g. `llm_task_<uuid12>`)           |
| `creator_nick`         | TEXT    | NOT NULL — IRC nick at create time                                              |
| `account`              | TEXT    | nullable — resolved account name at create time, used for owner attribution    |
| `channel`              | TEXT    | NOT NULL — channel the schedule was created in (also default delivery target)   |
| `network`              | TEXT    | NOT NULL — Limnoria network name used to resolve the IRC connection at fire time |
| `wire_msg`             | TEXT    | NOT NULL — `str(msg)` at create time, used to rehydrate the `IrcMsg` on fire    |
| `prompt`               | TEXT    | NOT NULL — bare instruction; passed through unchanged from the LLM's tool call  |
| `fire_at`              | REAL    | NOT NULL — unix timestamp of next fire                                          |
| `created_at`           | REAL    | NOT NULL                                                                        |
| `recurrence_seconds`   | INTEGER | nullable, mutually exclusive with `recurrence_rrule`                            |
| `recurrence_rrule`     | TEXT    | nullable, RFC 5545 RRULE body, no DTSTART                                       |
| `chain_position`       | INTEGER | NOT NULL DEFAULT 1 — runaway guard, mirrors reminders                           |
| `watch_mode`           | INTEGER | NOT NULL DEFAULT 0 — suppress `[silent]` only for watch-style scheduled tasks    |

Indexes: `idx_scheduled_llm_tasks_fire_at`, `idx_scheduled_llm_tasks_account`,
`idx_scheduled_llm_tasks_creator_nick`, `idx_scheduled_llm_tasks_owner_channel`.

The `wire_msg` column is the load-bearing piece for fire-time identity: at
fire time we do `msg = IrcMsg(s=row.wire_msg)`, then tag with
`llm_schedule_depth=1`, then dispatch. `prefix`, `nick`, `args`,
`server_tags` (including `account-tag`) all survive that round-trip.

**Step 1: Write the failing migration test.**

Append to `plugins/llm/tests/test_persistence.py`:

```python
def test_schema_v13_creates_scheduled_llm_tasks_table(tmp_path):
    """Task 3/A1: SCHEMA_VERSION bumps to 13; scheduled_llm_tasks table
    exists with the documented columns, index, and uniqueness constraint."""
    import sqlite3
    from llm.persistence import LLMDatabase, SCHEMA_VERSION

    assert SCHEMA_VERSION >= 13

    db_path = tmp_path / "llm.sqlite"
    LLMDatabase(str(db_path))  # runs migrations

    conn = sqlite3.connect(str(db_path))
    try:
        cols = {
            row[1]: row
            for row in conn.execute("PRAGMA table_info(scheduled_llm_tasks)")
        }
        for expected in (
            "id",
            "event_name",
            "creator_nick",
            "account",
            "channel",
            "network",
            "wire_msg",
            "prompt",
            "fire_at",
            "created_at",
            "recurrence_seconds",
            "recurrence_rrule",
            "chain_position",
            "watch_mode",
        ):
            assert expected in cols, f"missing column: {expected}"

        idx_names = {
            row[1]
            for row in conn.execute(
                "SELECT * FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'scheduled_llm_tasks'"
            )
        }
        assert "idx_scheduled_llm_tasks_fire_at" in idx_names
        assert "idx_scheduled_llm_tasks_account" in idx_names
        assert "idx_scheduled_llm_tasks_creator_nick" in idx_names
        assert "idx_scheduled_llm_tasks_owner_channel" in idx_names

        # event_name is UNIQUE
        conn.execute(
            "INSERT INTO scheduled_llm_tasks "
            "(event_name, creator_nick, channel, network, wire_msg, prompt, "
            "fire_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ev1", "nick1", "#a", "afternet", ":wire1", "p", 0.0, 0.0),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO scheduled_llm_tasks "
                "(event_name, creator_nick, channel, network, wire_msg, prompt, "
                "fire_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("ev1", "nick2", "#a", "afternet", ":wire2", "p", 0.0, 0.0),
            )
    finally:
        conn.close()


def test_schema_v13_idempotent(tmp_path):
    """Re-opening a v13 DB does not error or re-run the migration body."""
    from llm.persistence import LLMDatabase

    db_path = tmp_path / "llm.sqlite"
    LLMDatabase(str(db_path))
    LLMDatabase(str(db_path))  # second open must succeed unchanged
```

**Step 2: Run; verify it fails.**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v -k "schema_v13"
```

Expected: 2 FAIL — `assert SCHEMA_VERSION >= 13` fires first; the table-info
test fires after if you bump SCHEMA_VERSION manually but don't add the
migration body.

**Step 3: Add the migration in `persistence.py`.**

3a. Bump `SCHEMA_VERSION` to `13` (line 18).

3b. After the `if current_version < 12:` block (~line 357), insert:

```python
if current_version < 13:
    # Task 3 (Limnoria bridge Phase 2): native LLM tool
    # ``schedule_llm_task`` and friends. One row per active schedule.
    # Persists wire-format msg so the fire closure can rebuild a fresh
    # IrcMsg without relying on pickle (msg.tags would be lost).
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scheduled_llm_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT UNIQUE NOT NULL,
            creator_nick TEXT NOT NULL,
            account TEXT,
            channel TEXT NOT NULL,
            network TEXT NOT NULL,
            wire_msg TEXT NOT NULL,
            prompt TEXT NOT NULL,
            fire_at REAL NOT NULL,
            created_at REAL NOT NULL,
            recurrence_seconds INTEGER,
            recurrence_rrule TEXT,
            chain_position INTEGER NOT NULL DEFAULT 1,
            watch_mode INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_fire_at
            ON scheduled_llm_tasks(fire_at);
        CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_account
            ON scheduled_llm_tasks(account);
        CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_creator_nick
            ON scheduled_llm_tasks(creator_nick);
        CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_owner_channel
            ON scheduled_llm_tasks(account, creator_nick, channel);
    """)
    conn.commit()
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v -k "schema_v13"
```

Expected: 2 PASS.

**Step 5: Commit.**

Run `make lint` and `make typecheck` first (AGENTS.md:22). If either fails,
fix; do NOT bypass with `--no-verify`.

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(llm): add scheduled_llm_tasks table (schema v13)"
```

**Done when:** SCHEMA_VERSION is 13; the new table + indexes exist on a
fresh DB; the v13 migration is idempotent on re-open. No production code
references the table yet.

---

### Task A2: `LLMDatabase` helpers for the new table

**Files:**

- Modify: `plugins/llm/src/llm/persistence.py` (new methods + new
  `ScheduledLlmTaskRow` NamedTuple).
- Modify: `plugins/llm/tests/test_persistence.py` (one test per helper).

**Helpers to add (mirrors the reminder helpers):**

| Method                                              | Purpose                                                        |
| --------------------------------------------------- | -------------------------------------------------------------- |
| `save_scheduled_llm_task(...)`                      | Insert; raises on duplicate `event_name`. Returns row id.     |
| `update_scheduled_llm_task_fire_at(event_name, t)`  | Reschedule (recurring tasks update this on every fire).       |
| `delete_scheduled_llm_task(event_name) -> bool`     | Delete by event name.                                          |
| `load_active_scheduled_llm_tasks() -> list`         | All rows where `fire_at > now - 24h` (matches reminders' window). |
| `count_scheduled_llm_tasks_for(account, nick, channel) -> int` | For per-creator per-channel budget enforcement.              |

Plus a `ScheduledLlmTaskRow` NamedTuple mirroring `ReminderRow`.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_persistence.py` (one test per helper —
keep each test small; same shape as the reminder-helper tests):

```python
def test_save_scheduled_llm_task_inserts_row(tmp_path):
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    row_id = db.save_scheduled_llm_task(
        event_name="llm_task_abc",
        creator_nick="rdrake",
        account="rdrake_acct",
        channel="#test",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #test :@ask hi",
        prompt="check the build",
        fire_at=1_700_000_000.0,
        recurrence_seconds=None,
        recurrence_rrule=None,
        chain_position=1,
    )
    assert row_id > 0


def test_save_scheduled_llm_task_rejects_duplicate_event_name(tmp_path):
    import sqlite3
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="dup",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=0.0,
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.save_scheduled_llm_task(
            event_name="dup",
            creator_nick="n",
            account=None,
            channel="#x",
            network="afternet",
            wire_msg=":n!u@h PRIVMSG #x :@ask hi2",
            prompt="p2",
            fire_at=0.0,
        )


def test_save_scheduled_llm_task_rejects_both_recurrence_kinds(tmp_path):
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        db.save_scheduled_llm_task(
            event_name="ev",
            creator_nick="n",
            account=None,
            channel="#x",
            network="afternet",
            wire_msg=":n!u@h PRIVMSG #x :@ask hi",
            prompt="p",
            fire_at=0.0,
            recurrence_seconds=300,
            recurrence_rrule="FREQ=DAILY",
        )


def test_load_active_scheduled_llm_tasks_excludes_old(tmp_path):
    """Mirror reminders: anything past EXPIRY_THRESHOLD is excluded."""
    import time
    from llm.persistence import LLMDatabase, EXPIRY_THRESHOLD_SECONDS
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    now = time.time()
    db.save_scheduled_llm_task(
        event_name="future",
        creator_nick="n", account=None, channel="#x", network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=now + 60,
    )
    db.save_scheduled_llm_task(
        event_name="recent_overdue",
        creator_nick="n", account=None, channel="#x", network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=now - 60,
    )
    db.save_scheduled_llm_task(
        event_name="ancient",
        creator_nick="n", account=None, channel="#x", network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=now - EXPIRY_THRESHOLD_SECONDS - 60,
    )
    rows = db.load_active_scheduled_llm_tasks()
    names = {r.event_name for r in rows}
    assert "future" in names
    assert "recent_overdue" in names
    assert "ancient" not in names


def test_count_scheduled_llm_tasks_for_account_then_nick(tmp_path):
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    # Two rows for the same account; one for a different nick with no account.
    db.save_scheduled_llm_task(
        event_name="ev1", creator_nick="rdrake", account="rdrake_a",
        channel="#x", network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=1.0,
    )
    db.save_scheduled_llm_task(
        event_name="ev2", creator_nick="rdrake_alt", account="rdrake_a",
        channel="#x", network="afternet",
        wire_msg=":rdrake_alt!u@h PRIVMSG #x :@ask hi",
        prompt="p", fire_at=2.0,
    )
    db.save_scheduled_llm_task(
        event_name="ev3", creator_nick="anon", account=None,
        channel="#x", network="afternet",
        wire_msg=":anon!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=3.0,
    )
    db.save_scheduled_llm_task(
        event_name="ev4", creator_nick="rdrake", account="rdrake_a",
        channel="#other", network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #other :@ask hi", prompt="p",
        fire_at=4.0,
    )
    # When account is given, count by account regardless of nick, but only
    # in the requested channel. Matching is case-insensitive.
    assert db.count_scheduled_llm_tasks_for(
        account="RDRAKE_A", nick="anything", channel="#x"
    ) == 2
    assert db.count_scheduled_llm_tasks_for(
        account="rdrake_a", nick="anything", channel="#other"
    ) == 1
    # When account is None, count by nick.
    assert db.count_scheduled_llm_tasks_for(
        account=None, nick="anon", channel="#x"
    ) == 1
    assert db.count_scheduled_llm_tasks_for(
        account=None, nick="rdrake", channel="#x"
    ) == 0


def test_update_scheduled_llm_task_fire_at(tmp_path):
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev", creator_nick="n", account=None,
        channel="#x", network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=1.0,
    )
    db.update_scheduled_llm_task_fire_at("ev", 9.0, chain_position=2)
    rows = db.load_active_scheduled_llm_tasks()
    [row] = [r for r in rows if r.event_name == "ev"]
    assert row.fire_at == 9.0
    assert row.chain_position == 2


def test_delete_scheduled_llm_task_returns_bool(tmp_path):
    from llm.persistence import LLMDatabase
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev", creator_nick="n", account=None,
        channel="#x", network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi", prompt="p",
        fire_at=0.0,
    )
    assert db.delete_scheduled_llm_task("ev") is True
    assert db.delete_scheduled_llm_task("ev") is False
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v -k scheduled_llm_task
```

Expected: 7 FAIL with `AttributeError` on `save_scheduled_llm_task` (or
similar) since the methods don't exist.

**Step 3: Implement.**

Add a `ScheduledLlmTaskRow` NamedTuple under `ReminderRow` (around
`persistence.py:24-39`):

```python
class ScheduledLlmTaskRow(NamedTuple):
    """A scheduled LLM task loaded from the database."""
    id: int
    event_name: str
    creator_nick: str
    account: str | None
    channel: str
    network: str
    wire_msg: str
    prompt: str
    fire_at: float
    created_at: float
    recurrence_seconds: int | None
    recurrence_rrule: str | None
    chain_position: int
    watch_mode: bool

    def rehydrate_msg(self):
        """Build a fresh ``IrcMsg`` from the persisted wire string.

        Encapsulates the IrcMsg construction so callers don't reach into
        ``wire_msg`` directly; also the natural place to add any future
        post-rehydration tag plumbing if it becomes needed.
        """
        from supybot.ircmsgs import IrcMsg
        return IrcMsg(s=self.wire_msg)
```

Add the helper methods in a new section "Scheduled LLM task operations"
right after the reminder section (~`persistence.py:608`). Match the
reminder helpers' shape:

```python
# ------------------------------------------------------------------
# Scheduled LLM task operations
# ------------------------------------------------------------------

_SCHEDULED_LLM_TASK_COLUMNS = (
    "id, event_name, creator_nick, account, channel, network, wire_msg, prompt, "
    "fire_at, created_at, recurrence_seconds, "
    "recurrence_rrule, chain_position, watch_mode"
)

def save_scheduled_llm_task(
    self,
    event_name: str,
    creator_nick: str,
    account: str | None,
    channel: str,
    network: str,
    wire_msg: str,
    prompt: str,
    fire_at: float,
    *,
    recurrence_seconds: int | None = None,
    recurrence_rrule: str | None = None,
    chain_position: int = 1,
    watch_mode: bool = False,
) -> int:
    """Save a scheduled LLM task row.

    Raises:
        ValueError: if both recurrence kinds are non-null.
        sqlite3.IntegrityError: if event_name already exists.
    """
    if recurrence_seconds is not None and recurrence_rrule is not None:
        raise ValueError(
            "recurrence_seconds and recurrence_rrule are mutually exclusive"
        )
    now = time.time()
    conn = self._connect()
    cursor = conn.execute(
        "INSERT INTO scheduled_llm_tasks "
        "(event_name, creator_nick, account, channel, network, wire_msg, prompt, "
        "fire_at, created_at, recurrence_seconds, "
        "recurrence_rrule, chain_position, watch_mode) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_name, creator_nick, account, channel, network, wire_msg, prompt,
            fire_at, now, recurrence_seconds,
            recurrence_rrule, chain_position, int(watch_mode),
        ),
    )
    conn.commit()
    return cursor.lastrowid or 0


def update_scheduled_llm_task_fire_at(
    self,
    event_name: str,
    fire_at: float,
    *,
    chain_position: int | None = None,
) -> None:
    """Update fire_at (and optionally chain_position) for a row."""
    conn = self._connect()
    if chain_position is None:
        conn.execute(
            "UPDATE scheduled_llm_tasks SET fire_at = ? WHERE event_name = ?",
            (fire_at, event_name),
        )
    else:
        conn.execute(
            "UPDATE scheduled_llm_tasks SET fire_at = ?, chain_position = ? "
            "WHERE event_name = ?",
            (fire_at, chain_position, event_name),
        )
    conn.commit()


def delete_scheduled_llm_task(self, event_name: str) -> bool:
    conn = self._connect()
    cursor = conn.execute(
        "DELETE FROM scheduled_llm_tasks WHERE event_name = ?",
        (event_name,),
    )
    conn.commit()
    return cursor.rowcount > 0


def load_active_scheduled_llm_tasks(self) -> list[ScheduledLlmTaskRow]:
    """Load rows whose fire_at is within the last 24 hours or in the future.

    Used at plugin init for restore. For per-fire and per-cancel lookups,
    prefer ``get_scheduled_llm_task`` (indexed point lookup) and
    ``load_scheduled_llm_tasks_for`` (owner-filtered) so we don't scan
    the table when one row will do.
    """
    cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
    conn = self._connect()
    rows = conn.execute(
        f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
        "FROM scheduled_llm_tasks WHERE fire_at > ? ORDER BY fire_at",
        (cutoff,),
    ).fetchall()
    return [ScheduledLlmTaskRow(*r[:-1], watch_mode=bool(r[-1])) for r in rows]


def get_scheduled_llm_task(
    self, event_name: str
) -> ScheduledLlmTaskRow | None:
    """Indexed point-lookup by event_name. Hits the UNIQUE index."""
    conn = self._connect()
    row = conn.execute(
        f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
        "FROM scheduled_llm_tasks WHERE event_name = ?",
        (event_name,),
    ).fetchone()
    return ScheduledLlmTaskRow(*row[:-1], watch_mode=bool(row[-1])) if row else None


def load_scheduled_llm_tasks_for(
    self, *, account: str | None, nick: str
) -> list[ScheduledLlmTaskRow]:
    """Active rows owned by the caller. Case-insensitive Identity semantics."""
    cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
    conn = self._connect()
    if account is not None:
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks "
            "WHERE lower(account) = lower(?) AND fire_at > ? ORDER BY fire_at",
            (account, cutoff),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks "
            "WHERE account IS NULL AND lower(creator_nick) = lower(?) AND fire_at > ? "
            "ORDER BY fire_at",
            (nick, cutoff),
        ).fetchall()
    return [ScheduledLlmTaskRow(*r[:-1], watch_mode=bool(r[-1])) for r in rows]


def count_scheduled_llm_tasks_for(
    self, *, account: str | None, nick: str, channel: str
) -> int:
    """Count active rows owned by the caller in this channel.

    When ``account`` is non-None, count rows with that account regardless
    of nick. Otherwise count by raw nick. Comparisons are case-insensitive
    to match ``Identity.matches``.
    """
    cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
    conn = self._connect()
    if account is not None:
        row = conn.execute(
            "SELECT COUNT(*) FROM scheduled_llm_tasks "
            "WHERE lower(account) = lower(?) AND channel = ? AND fire_at > ?",
            (account, channel, cutoff),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM scheduled_llm_tasks "
            "WHERE account IS NULL AND lower(creator_nick) = lower(?) "
            "AND channel = ? AND fire_at > ?",
            (nick, channel, cutoff),
        ).fetchone()
    return int(row[0] if row else 0)
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v -k scheduled_llm_task
```

Expected: 7 PASS.

**Step 5: Commit.** `make lint && make typecheck` first.

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(llm): persistence helpers for scheduled_llm_tasks"
```

**Done when:** the helpers exist, the 7 tests pass, and no existing
persistence test regresses.

---

## B — Service layer: scheduling + listing + cancel + restore

### Task B1: `schedule_llm_task` service method (one-shot)

**Files:**

- Modify: `plugins/llm/src/llm/service.py` (new method on `LLMService`,
  positioned next to the reminder methods around line 3530+).
- Modify: `plugins/llm/tests/test_service.py` (or appropriate test file
  for `LLMService` — find with
  `grep -ln "class TestLLMService\|def test_parse_reminder" plugins/llm/tests/`).

**Background — fire-time closure shape:**

The fire closure (`_make_scheduled_llm_task_callback`) is the load-bearing
piece. At fire time it must:

1. Look up the row from the DB by `event_name` (it may have been cancelled
   between schedule and fire — log & skip if so).
2. Resolve a current `irc` via `world.getIrc(row.network) or world.ircs[0]`
   (mirrors `Scheduler/plugin.py:142-143` and the reminder fire path).
3. Rebuild a fresh `IrcMsg(s=row.wire_msg)` — preserves prefix, args,
   server_tags (account-tag), but starts with empty `tags`.
4. Set `msg.tags["llm_schedule_depth"] = 1` on the rebuilt msg. (This is
   the depth-cap mechanism. See Task D4.)
5. Check the schedule owner's ask rate-limit bucket, then dispatch through
   `LLMService.assistant_request(...)` directly with a synthetic
   `AssistantRequestContext`. This mirrors reminder-action fires and keeps
   the scheduler thread out of the command wrapper path.
6. For one-shot tasks: delete the DB row and call
   `schedule.removeEvent(event_name)` is a no-op because Scheduler
   auto-removes single events from its in-memory dict (look at
   `schedule.py:148-149`); just delete the DB row.
7. For recurring tasks: re-check the DB row after the LLM call (cancel wins
   over an in-flight fire), compute `next_fire`, call
   `db.update_scheduled_llm_task_fire_at(event_name, next_fire,
   chain_position=row.chain_position+1)`, then
   `schedule.addEvent(self._make_scheduled_llm_task_callback(event_name),
   next_fire, name=event_name)`.

**Why we bypass `LLM.ask`:**

Calling `irc.getCallback("LLM").ask(...)` directly is the wrong shape: it
would enter the wrapped command function from the scheduler thread, bypassing
normal Limnoria command dispatch/thread spawning while still using command
wrapper plumbing. The existing reminder fire path (`plugin.py:1134-1259`)
sidesteps this by doing the manual preflight pieces it needs, synthesising an
`AssistantRequestContext`, calling `LLMService.assistant_request` directly,
sanitizing output, and logging usage. We mirror that.

**Per-creator budget (Task D4 has the registry value):** at create time,
read `bridgeScheduledTaskLimit`. If it is `0`, scheduling is disabled for
that channel. Otherwise refuse if
`db.count_scheduled_llm_tasks_for(..., channel=channel)` >= that value.

**Step 1: Write the failing tests.**

Append to the LLMService test file. Current shared fixtures are
`make_service` and `test_db` (`plugins/llm/tests/conftest.py`); if no local
`llm_service` / `db` aliases exist, add these aliases near the new tests so
the snippets below are concrete:

```python
import time

from llm.service import ReminderParseResult


@pytest.fixture
def db(test_db):
    return test_db


@pytest.fixture
def llm_service(make_service, db):
    service, plugin = make_service()
    plugin.db = db
    return service
```

Use mocker to patch `supybot.schedule.addEvent` and `world.getIrc`. The plan
tests *plumbing* (the right functions are called with the right args); the
fire-time integration is exercised by the manual operational test in
Validation.

```python
def test_schedule_llm_task_creates_db_row_and_schedules_event(
    llm_service, db, mocker
):
    """B1: a one-shot schedule writes a DB row and registers the event
    with supybot.schedule.addEvent."""
    add_event = mocker.patch("llm.service.schedule.addEvent")

    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":rdrake!u@h PRIVMSG #test :@ask hi"
    msg.nick = "rdrake"
    msg.args = ("#test", "@ask hi")
    msg.tagged.return_value = None  # no schedule depth

    irc = mocker.MagicMock()
    irc.network = "afternet"

    # Mock parse_reminder to skip LLM call.
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="check build",
            confirmation="ok",
            note=None,
            action_prompt="check the build",
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    # No depth tag; budget is well under the limit (registry returns 5).
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#test",
        when_natural="in 60s", prompt="check the build",
    )

    assert result.status == "ok"
    assert result.event_name.startswith("llm_task_")
    rows = db.load_active_scheduled_llm_tasks()
    [row] = [r for r in rows if r.event_name == result.event_name]
    assert row.creator_nick == "rdrake"
    assert row.account == "rdrake_a"
    assert row.channel == "#test"
    assert row.network == "afternet"
    assert row.prompt == "check the build"
    assert row.recurrence_seconds is None
    assert row.recurrence_rrule is None

    add_event.assert_called_once()
    args = add_event.call_args
    callback = args[0][0]
    fire_at = args[0][1]
    name_kwarg = args[1].get("name") or args[0][2]
    assert callable(callback)
    assert name_kwarg == result.event_name
    # fire_at should be ~ now + 60.
    assert fire_at == pytest.approx(time.time() + 60, abs=2)


def test_schedule_llm_task_recurrence_seconds(llm_service, db, mocker):
    """B1: a numeric-cadence recurrence stores recurrence_seconds and
    schedules the FIRST fire at parser.seconds."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"

    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=300,
            message="ping me",
            confirmation="ok",
            note=None,
            action_prompt="ping me",
            recurrence_seconds=300,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account="acct", channel="#t",
        when_natural="every 5 minutes", prompt="ping me",
    )
    assert result.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == result.event_name]
    assert row.recurrence_seconds == 300
    assert row.recurrence_rrule is None
    assert add_event.call_count == 1


def test_schedule_llm_task_recurrence_rrule(llm_service, db, mocker):
    add_event = mocker.patch("llm.service.schedule.addEvent")
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="weekly digest",
            confirmation="ok",
            note=None,
            action_prompt="post the weekly summary",
            recurrence_seconds=None,
            recurrence_rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0",
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )
    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account="acct", channel="#t",
        when_natural="every Monday at 9am", prompt="post the weekly summary",
    )
    assert result.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == result.event_name]
    assert row.recurrence_rrule.startswith("FREQ=WEEKLY")


def test_schedule_llm_task_refuses_when_depth_tag_set(llm_service, mocker):
    """B1 + D4: a fired task can't recursively call schedule_llm_task."""
    msg = mocker.MagicMock()
    msg.tagged.side_effect = lambda key: 1 if key == "llm_schedule_depth" else None
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    irc = mocker.MagicMock(); irc.network = "afternet"
    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account=None, channel="#t",
        when_natural="in 1m", prompt="do something else",
    )
    assert result.status == "error"
    assert "depth" in result.message.lower() or "scheduled" in result.message.lower()


def test_schedule_llm_task_enforces_per_creator_limit(llm_service, db, mocker):
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"
    # Pre-populate the table so the count is at the limit.
    for i in range(5):
        db.save_scheduled_llm_task(
            event_name=f"existing_{i}", creator_nick="n", account="a",
            channel="#t", network="afternet",
            wire_msg=":n!u@h PRIVMSG #t :@ask hi", prompt="p",
            fire_at=time.time() + 60,
        )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    mocker.patch.object(
        llm_service, "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule", seconds=60, message="x", confirmation="ok",
            note=None, action_prompt="x",
            recurrence_seconds=None, recurrence_rrule=None, watch_mode=False,
        ),
    )

    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account="a", channel="#t",
        when_natural="in 1m", prompt="do x",
    )
    assert result.status == "error"
    assert "limit" in result.message.lower()


def test_schedule_llm_task_limit_zero_disables_scheduling(llm_service, mocker):
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        0 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account="a", channel="#t",
        when_natural="in 1m", prompt="do x",
    )

    assert result.status == "error"
    assert "disabled" in result.message.lower()


def test_schedule_llm_task_clarify_returns_clarify_envelope(llm_service, mocker):
    """When parse_reminder returns action='clarify', schedule_llm_task
    surfaces the parser's clarification text instead of scheduling."""
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":n!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"
    mocker.patch.object(
        llm_service, "parse_reminder",
        return_value=ReminderParseResult(
            action="clarify", confirmation="When should I run that?"
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="n", account="acct", channel="#t",
        when_natural="vague request", prompt="some action",
    )
    assert result.status == "clarify"
    assert "When should I run that?" in result.message
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_service.py -v -k schedule_llm_task
```

Expected: 7 FAIL with `AttributeError: 'LLMService' object has no attribute
'schedule_llm_task'`.

**Step 3: Implement `LLMService.schedule_llm_task` and the fire callback factory.**

Add the runtime imports needed by this section at the top of `service.py`:

```python
import sqlite3
import uuid

import supybot.schedule as schedule
```

Also import the new persistence row/types outside `TYPE_CHECKING`:

```python
from .persistence import LLMDatabase, ScheduledLlmTaskRow
```

Add a `ScheduleLlmTaskResult` NamedTuple under `ReminderParseResult` (around
service.py:308):

```python
class ScheduleLlmTaskResult(NamedTuple):
    """Outcome of a schedule_llm_task call."""
    status: str  # "ok", "clarify", "error"
    event_name: str = ""
    fire_at: float = 0.0
    message: str = ""  # confirmation (status=ok) or reason (clarify/error)
    note: str | None = None
```

In the `LLMService` class body, near the reminder methods, add:

```python
def schedule_llm_task(
    self,
    *,
    irc: Irc,
    msg: IrcMsg,
    creator_nick: str,
    account: str | None,
    channel: str,
    when_natural: str,
    prompt: str,
) -> ScheduleLlmTaskResult:
    """Schedule a future @ask invocation (Phase 2 Task 3).

    Uses ``parse_reminder`` for the natural-language → seconds / rrule
    shape by parsing ``f"{when_natural} {prompt}"``. The parsed
    message/action text is ignored; ``prompt`` is the LLM's already-bare
    instruction and is stored verbatim.

    Refuses (without scheduling) when:
    - The caller is already inside a fired schedule
      (``msg.tagged('llm_schedule_depth')`` is truthy) — depth cap of 1.
    - The caller is unidentified (defense in depth; the tool spec also
      requires an authenticated account).
    - The caller already has ``bridgeScheduledTaskLimit`` active tasks in
      this channel, or the limit is 0.
    - ``parse_reminder`` returns ``action='clarify'`` — surface the parser's
      question to the LLM via the ``clarify`` status.
    """
    db = getattr(self.plugin, "db", None)
    if db is None:
        return ScheduleLlmTaskResult(
            status="error", message="No database available."
        )

    # Depth cap (D4). Tags are set fresh on the rehydrated msg in our
    # fire callback (msg.tags is lost on pickle; the cap relies on the
    # closure setting the tag, not on persistence).
    if msg.tagged("llm_schedule_depth"):
        return ScheduleLlmTaskResult(
            status="error",
            message="Cannot schedule another task from inside a fired "
            "schedule (depth cap reached).",
        )

    if not account:
        return ScheduleLlmTaskResult(
            status="error",
            message="schedule_llm_task requires an authenticated account.",
        )

    # Per-creator budget.
    limit = int(self.plugin.registryValue("bridgeScheduledTaskLimit", channel) or 0)
    if limit == 0:
        return ScheduleLlmTaskResult(
            status="error",
            message="Scheduled LLM tasks are disabled in this channel.",
        )
    existing = db.count_scheduled_llm_tasks_for(
        account=account, nick=creator_nick, channel=channel
    )
    if existing >= limit:
        return ScheduleLlmTaskResult(
            status="error",
            message=f"Scheduled-task limit reached ({existing}/{limit}). "
            "Cancel one with cancel_scheduled_llm_task to free a slot.",
        )

    # parse_reminder expects both time and message. Parse the composed text,
    # then persist the tool's structured prompt verbatim.
    parsed = self.parse_reminder(f"{when_natural} {prompt}", channel=channel)
    if parsed.action != "schedule" or not parsed.seconds:
        return ScheduleLlmTaskResult(
            status="clarify",
            message=parsed.confirmation or "Could not parse that schedule.",
            note=parsed.note,
        )

    fire_at = time.time() + parsed.seconds
    event_name = f"llm_task_{uuid.uuid4().hex[:12]}"
    try:
        db.save_scheduled_llm_task(
            event_name=event_name,
            creator_nick=creator_nick,
            account=account,
            channel=channel,
            network=irc.network,
            wire_msg=str(msg),
            prompt=prompt,
            fire_at=fire_at,
            recurrence_seconds=parsed.recurrence_seconds,
            recurrence_rrule=parsed.recurrence_rrule,
            chain_position=1,
            watch_mode=parsed.watch_mode,
        )
    except sqlite3.IntegrityError:
        return ScheduleLlmTaskResult(
            status="error",
            message="event-name collision; please retry",
        )

    callback = self._make_scheduled_llm_task_callback(event_name)
    try:
        schedule.addEvent(callback, fire_at, name=event_name)
    except Exception:
        db.delete_scheduled_llm_task(event_name)
        self.log.exception("schedule_llm_task addEvent failed: %s", event_name)
        return ScheduleLlmTaskResult(
            status="error",
            message="Could not register the scheduled task.",
        )

    return ScheduleLlmTaskResult(
        status="ok",
        event_name=event_name,
        fire_at=fire_at,
        message=parsed.confirmation
        or f"Scheduled for {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(fire_at))}.",
        note=parsed.note,
    )


def _make_scheduled_llm_task_callback(self, event_name: str) -> Callable[[], None]:
    """Build the no-arg fire closure for ``schedule.addEvent``.

    Rebuilds a fresh ``IrcMsg`` from the persisted wire string, tags it
    with ``llm_schedule_depth=1``, and dispatches via
    ``assistant_request`` directly — NOT via the LLM plugin's wrapped
    ``ask`` command, which would bypass normal command dispatch/threading
    and enter command-wrapper plumbing from the scheduler thread (mirrors
    the existing reminder fire path at ``plugin.py:1134-1259``).
    """
    db = self.plugin.db

    def fire() -> None:
        row = db.get_scheduled_llm_task(event_name)
        if row is None:
            self.log.info("scheduled_llm_task fire: %s cancelled", event_name)
            return

        from supybot import world
        irc = world.getIrc(row.network) or (world.ircs[0] if world.ircs else None)
        if irc is None:
            self.log.warning(
                "scheduled_llm_task fire: %s no irc; skipping (no reschedule)",
                event_name,
            )
            return

        try:
            msg = row.rehydrate_msg()
            msg.tag("llm_schedule_depth", 1)
            self._dispatch_scheduled_task(irc, msg, row)
        except Exception:
            self.log.exception(
                "scheduled_llm_task fire failed: %s", event_name
            )

        self._maybe_reschedule_or_clean(row, db)

    return fire


def _dispatch_scheduled_task(
    self,
    irc: Irc,
    msg: IrcMsg,
    row: ScheduledLlmTaskRow,
) -> None:
    """Run the fired prompt through ``assistant_request`` directly.

    Mirrors ``plugin.py:1134-1259`` (the reminder fire path): do the
    manual preflight pieces the command wrapper would normally handle,
    synthesise an ``AssistantRequestContext``, call the service entry
    point, sanitize output, and log usage.
    """
    plugin = self.plugin
    now = time.time()
    rl_account = row.account if row.account else row.creator_nick
    rl_tier = "registered" if row.account else "unregistered"
    target = row.channel if ircutils.isChannel(row.channel) else row.creator_nick
    if plugin._check_rate_limit(
        None,
        "ask",
        rl_account,
        "",
        "",
        "",
        tier=rl_tier,
        silent=True,
        now=now,
    ):
        irc.queueMsg(
            ircmsgs.privmsg(
                target,
                f"{row.creator_nick}: Scheduled task skipped — daily ask limit reached.",
            )
        )
        return

    request_context = AssistantRequestContext(
        entry_route="scheduled_llm_task",
        profile="remind_action",
        nick=row.creator_nick,
        raw_nick=row.creator_nick,
        account=row.account,
        channel=row.channel,
        is_private=not ircutils.isChannel(row.channel),
        is_owner=False,
        capabilities=frozenset({"llm.ask", "llm.draw", "llm.code"}),
    )
    history, channel_history = plugin._gather_history(
        row.creator_nick, row.channel
    )
    memories = plugin._get_user_memories(row.creator_nick)
    user_instruction = plugin.db.get_instruction(row.creator_nick)
    ask_prompt = resolve_setting(
        plugin, "assistantSystemPrompt", row.channel,
        fallbacks=("askSystemPrompt",),
    )
    effective_prompt = (
        f"{user_instruction}\n\n{ask_prompt}" if user_instruction else None
    )
    # Local import avoids a service.py -> plugin.py import cycle at module load.
    from .plugin import Identity
    caller = Identity(raw_nick=row.creator_nick, account=row.account)

    result = self.assistant_request(
        prompt=row.prompt,
        request_context=request_context,
        db=plugin.db,
        context=plugin.context,
        bot_nick=irc.nick,
        history=history,
        channel_history=channel_history,
        irc=irc,
        msg=msg,
        memories=memories,
        system_prompt=effective_prompt,
        search_fn=lambda q: self.search_completion(q, channel=row.channel),
        fetch_fn=lambda u: self.url_completion(u, channel=row.channel),
        code_fn=lambda p: plugin._code_for_assistant(p, row.channel),
        draw_fn=lambda p, _i=irc, _m=msg: plugin._draw_for_assistant(_i, _m, p),
        cleanup_fn=lambda n: plugin._run_memory_cleanup(n, row.channel),
        # The depth tag on ``msg`` keeps schedule_llm_task itself off the
        # tool surface for this turn (B1 refuses on depth>=1). No need to
        # exclude it here.
        **plugin._reminder_fns(
            caller=caller, irc=irc, msg=msg,
            pass_irc_msg_to_callbacks=False,
        ),
    )

    response = (result.content or "").strip()
    try:
        plugin.db.log_usage(
            row.account or row.creator_nick,
            row.channel,
            "scheduled_llm_task",
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost,
            prompt=row.prompt,
            status=("silent" if row.watch_mode and response == "[silent]" else "success"),
            error_detail=(result.error or "")[:200],
        )
    except Exception:
        self.log.exception("scheduled_llm_task usage log failed: %s", row.event_name)

    if not response or (row.watch_mode and response == "[silent]"):
        return
    safe_response = self.sanitize_output(response)
    irc.queueMsg(ircmsgs.privmsg(target, safe_response))


def _maybe_reschedule_or_clean(
    self,
    row: ScheduledLlmTaskRow,
    db: LLMDatabase,
) -> None:
    """Reschedule recurring tasks; delete one-shots after fire."""
    if row.recurrence_seconds is None and row.recurrence_rrule is None:
        db.delete_scheduled_llm_task(row.event_name)
        return
    if db.get_scheduled_llm_task(row.event_name) is None:
        self.log.info(
            "scheduled_llm_task reschedule skipped: %s cancelled mid-fire",
            row.event_name,
        )
        return
    next_fire = self._compute_next_fire(row)
    if next_fire is None:
        # RRULE exhausted (e.g. UNTIL passed) — treat as one-shot done.
        db.delete_scheduled_llm_task(row.event_name)
        return
    db.update_scheduled_llm_task_fire_at(
        row.event_name, next_fire, chain_position=row.chain_position + 1
    )
    callback = self._make_scheduled_llm_task_callback(
        row.event_name
    )
    schedule.addEvent(callback, next_fire, name=row.event_name)


def _compute_next_fire(self, row: ScheduledLlmTaskRow) -> float | None:
    """Next fire time for a recurring task; ``None`` exhausts the schedule."""
    if row.recurrence_seconds:
        return time.time() + row.recurrence_seconds
    if row.recurrence_rrule:
        return self.plugin._next_rrule_fire(row.recurrence_rrule, time.time())
    return None
```

`LLM._next_rrule_fire` (`plugin.py:3350`) already encapsulates rrulestr +
exception handling; reuse it rather than re-implementing.

The network is persisted in the DB row and used by the callback after the row
lookup. Do not derive it from `wire_msg`; IRC wire strings do not contain the
Limnoria network name.

**Notes for the implementer:**

- `ScheduleLlmTaskResult` is an internal-to-Python type; the LLM-tool layer
  serialises this to JSON in Task C3.
- `uuid.uuid4().hex[:12]` matches the format used by the reminder/spontaneous
  event names in `service.py` (`f"llm_remind_{uuid.uuid4().hex[:12]}"`,
  `f"llm_spontaneous_{uuid.uuid4().hex[:8]}"`); use the same primitive
  here. No retry on `IntegrityError` — return an error envelope and let
  the LLM retry if it wants.
- The depth-cap test path uses `msg.tagged(...)` (the documented public
  API), not `msg.tags[...]` (deprecated per `ircmsgs.py:382-385`).

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_service.py -v -k schedule_llm_task
```

Expected: 7 PASS.

**Step 5: Commit.** `make lint && make typecheck` first.

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(llm): schedule_llm_task service method (one-shot + recurring)"
```

**Done when:** the service method passes its 7 plumbing tests, including
the depth-cap and budget refusals; no integration with the LLM tool layer
yet (that's Task C).

---

### Task B2: `list_scheduled_llm_tasks` + `cancel_scheduled_llm_task` service methods

**Files:**

- Modify: `plugins/llm/src/llm/service.py` (two new methods).
- Modify: the LLMService test file (3 tests).

**Step 1: Write failing tests.**

```python
def test_list_scheduled_llm_tasks_filters_by_owner(llm_service, db, mocker):
    """B2: list returns only the caller's active tasks. Match policy:
    account-when-account, nick-when-no-account (mirrors reminders)."""
    db.save_scheduled_llm_task(
        event_name="ev1", creator_nick="rdrake", account="rdrake_a",
        channel="#t", network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p", fire_at=time.time() + 60,
    )
    db.save_scheduled_llm_task(
        event_name="ev2", creator_nick="rdrake_alt", account="rdrake_a",
        channel="#t", network="afternet",
        wire_msg=":rdrake_alt!u@h PRIVMSG #t :@ask hi",
        prompt="p", fire_at=time.time() + 600,
    )
    db.save_scheduled_llm_task(
        event_name="other", creator_nick="other_user", account="other_a",
        channel="#t", network="afternet",
        wire_msg=":other!u@h PRIVMSG #t :@ask hi",
        prompt="p", fire_at=time.time() + 600,
    )

    rows = llm_service.list_scheduled_llm_tasks(
        creator_nick="rdrake", account="rdrake_a"
    )
    names = {r.event_name for r in rows}
    assert names == {"ev1", "ev2"}


def test_cancel_scheduled_llm_task_owner_scoped(llm_service, db, mocker):
    remove_event = mocker.patch("llm.service.schedule.removeEvent")
    db.save_scheduled_llm_task(
        event_name="mine", creator_nick="rdrake", account="rdrake_a",
        channel="#t", network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p", fire_at=time.time() + 60,
    )
    db.save_scheduled_llm_task(
        event_name="theirs", creator_nick="other", account="other_a",
        channel="#t", network="afternet",
        wire_msg=":other!u@h PRIVMSG #t :@ask hi",
        prompt="p", fire_at=time.time() + 60,
    )

    ok = llm_service.cancel_scheduled_llm_task(
        event_name="mine", creator_nick="rdrake", account="rdrake_a",
    )
    assert ok.status == "ok"
    assert db.delete_scheduled_llm_task("mine") is False  # already deleted
    remove_event.assert_called_once_with("mine")

    # Foreign cancel must refuse and not call removeEvent again.
    remove_event.reset_mock()
    foreign = llm_service.cancel_scheduled_llm_task(
        event_name="theirs", creator_nick="rdrake", account="rdrake_a",
    )
    assert foreign.status == "error"
    remove_event.assert_not_called()


def test_cancel_scheduled_llm_task_unknown_returns_error(llm_service):
    out = llm_service.cancel_scheduled_llm_task(
        event_name="does_not_exist", creator_nick="x", account=None,
    )
    assert out.status == "error"
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_service.py -v \
    -k "list_scheduled_llm_tasks or cancel_scheduled_llm_task"
```

**Step 3: Implement.**

In `LLMService`:

```python
def list_scheduled_llm_tasks(
    self, *, creator_nick: str, account: str | None
) -> list[ScheduledLlmTaskRow]:
    """Return active rows owned by the caller.

    Match policy is the standard account-when-known / nick-fallback applied
    by the indexed query in ``load_scheduled_llm_tasks_for``.
    """
    return self.plugin.db.load_scheduled_llm_tasks_for(
        account=account, nick=creator_nick
    )


def cancel_scheduled_llm_task(
    self,
    *,
    event_name: str,
    creator_nick: str,
    account: str | None,
) -> ScheduleLlmTaskResult:
    """Cancel a single task (owner-scoped).

    On success removes the schedule event AND deletes the DB row.
    Uses ``Identity.matches`` so the owner check is consistent with the
    reminder system's ``_get_user_reminders`` policy.
    """
    db = self.plugin.db
    row = db.get_scheduled_llm_task(event_name)
    if row is None:
        return ScheduleLlmTaskResult(
            status="error", message=f"No scheduled task with id {event_name}."
        )
    # Local import avoids a service.py -> plugin.py import cycle at module load.
    from .plugin import Identity
    caller = Identity(raw_nick=creator_nick, account=account)
    owner = Identity(raw_nick=row.creator_nick, account=row.account)
    if not owner.matches(caller):
        return ScheduleLlmTaskResult(
            status="error",
            message=f"Scheduled task {event_name} belongs to someone else.",
        )

    try:
        schedule.removeEvent(event_name)
    except KeyError:
        # Already fired or already cancelled in the scheduler — DB row is
        # the authoritative state, keep going and delete it.
        self.log.info(
            "cancel_scheduled_llm_task: %s not in scheduler (already fired?)",
            event_name,
        )
    db.delete_scheduled_llm_task(event_name)
    return ScheduleLlmTaskResult(
        status="ok", event_name=event_name,
        message=f"Cancelled scheduled task {event_name}.",
    )
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_service.py -v \
    -k "list_scheduled_llm_tasks or cancel_scheduled_llm_task"
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(llm): list + cancel scheduled_llm_task service methods"
```

**Done when:** the 3 tests pass and `cancel_scheduled_llm_task` is
owner-scoped (refuses cross-owner cancels with no side effects).

---

### Task B3: Restore active scheduled tasks on bot start

**Files:**

- Modify: `plugins/llm/src/llm/plugin.py` (call a new restore method from
  the existing post-load hook — find it with `grep -n "_reschedule_pending_reminders\|def __init__" plugins/llm/src/llm/plugin.py`).
- Modify: `plugins/llm/src/llm/service.py` (the new restore method).
- Modify: the LLMService test file (one test).

**Background:** the reminder system has `_reschedule_pending_reminders`
(service.py ~1309) called on plugin init that reads pending rows and
re-registers them with `schedule.addEvent`. We mirror this for
scheduled-LLM-tasks. Past-due rows are fired immediately (matches reminder
behaviour for "the bot was offline when this should have fired"). The
24-hour expiry threshold from `load_active_scheduled_llm_tasks` already
filters out very old rows.

**Step 1: Write failing test.**

```python
def test_restore_scheduled_llm_tasks_reregisters_events(
    llm_service, db, mocker
):
    add_event = mocker.patch("llm.service.schedule.addEvent")
    now = time.time()
    db.save_scheduled_llm_task(
        event_name="future_ev", creator_nick="n", account=None,
        channel="#t", network="afternet",
        wire_msg=":n!u@h PRIVMSG #t :@ask hi", prompt="p",
        fire_at=now + 600,
    )
    db.save_scheduled_llm_task(
        event_name="overdue_ev", creator_nick="n", account=None,
        channel="#t", network="afternet",
        wire_msg=":n!u@h PRIVMSG #t :@ask hi", prompt="p",
        fire_at=now - 60,
    )

    restored, skipped = llm_service.restore_scheduled_llm_tasks()
    assert restored == 2
    assert skipped == 0

    names = {
        (call.kwargs["name"] if "name" in call.kwargs else call.args[2])
        for call in add_event.call_args_list
    }
    assert names == {"future_ev", "overdue_ev"}
    # Overdue events fire ~immediately (the second arg is unix time;
    # confirm it's <= now plus a small skew).
    for call in add_event.call_args_list:
        name = call.kwargs["name"] if "name" in call.kwargs else call.args[2]
        fire_at = call.args[1]
        if name == "overdue_ev":
            assert fire_at <= now + 5
```

**Step 2: Run; verify fail.** (`AttributeError: ... no attribute 'restore_scheduled_llm_tasks'`)

**Step 3: Implement in `LLMService`.**

```python
def restore_scheduled_llm_tasks(self) -> tuple[int, int]:
    """Re-register every active scheduled task with the schedule module.

    Past-due rows fire immediately (next ``schedule.run`` tick). Mirrors
    ``_reschedule_pending_reminders``.

    Returns ``(restored, skipped)``.
    """
    db = self.plugin.db
    now = time.time()
    rows = db.load_active_scheduled_llm_tasks()
    restored = 0
    skipped = 0
    for row in rows:
        callback = self._make_scheduled_llm_task_callback(row.event_name)
        fire_at = max(row.fire_at, now + 1)  # past-due → fire ~immediately
        try:
            schedule.addEvent(callback, fire_at, name=row.event_name)
            restored += 1
        except AssertionError:
            skipped += 1
            self.log.warning(
                "restore_scheduled_llm_tasks: %s already scheduled; skip",
                row.event_name,
            )
    return restored, skipped
```

**Step 3b: Wire the call into plugin init.**

Find the existing init point (likely next to where reminders are
restored — `grep -n "_reschedule_pending_reminders" plugins/llm/src/llm/plugin.py`)
and add right after it:

```python
self.llm_service.restore_scheduled_llm_tasks()
```

The callback resolves the network from each persisted row. If the named
network is unavailable at fire time, it falls back to `world.ircs[0]` using
the same best-effort shape as Scheduler.

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_service.py -v -k restore_scheduled_llm_tasks
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py \
    plugins/llm/tests/test_service.py
git commit -m "feat(llm): restore scheduled_llm_tasks on plugin init"
```

**Done when:** plugin restart re-registers active rows with
`schedule.addEvent`; the restore test passes; manual smoke (covered in
Validation) confirms a recurring schedule survives restart.

---

## C — Native tool registration (assistant.py)

### Task C1: Tool schemas in `ASSISTANT_TOOLS`

**Files:**

- Modify: `plugins/llm/src/llm/assistant.py` (append three schemas to
  `ASSISTANT_TOOLS`).
- Modify: `plugins/llm/tests/test_assistant.py` (one assertion confirming
  the names register).

**Background — copy guidance distinguishing from set_reminder:**

The Phase 2 plan §"Tool-selection guidance vs. `set_reminder`" is the
canonical word here. Concrete tool descriptions:

- `schedule_llm_task`: "Schedule a future LLM task. At fire time the bot
  runs an `@ask` invocation **as you**, with full bridge access. Use this
  for tasks that need TOOLS at fire time (search, fetch, draw, code,
  Limnoria bridge calls). For plain text reminders, use `set_reminder`."
- `list_scheduled_llm_tasks`: "List **your** scheduled LLM tasks. Returns
  id, when, prompt, channel for each. Use before cancel."
- `cancel_scheduled_llm_task`: "Cancel one of **your** scheduled LLM
  tasks by id. Get ids via `list_scheduled_llm_tasks`."

The "as you" / "your" framing is intentional: it matches `set_reminder`'s
ownership semantics, which the LLM already understands.

**Step 1: Write the failing assertion.**

```python
def test_assistant_tools_includes_schedule_llm_task_family():
    from llm.assistant import ASSISTANT_TOOLS
    names = {t["function"]["name"] for t in ASSISTANT_TOOLS}
    assert "schedule_llm_task" in names
    assert "list_scheduled_llm_tasks" in names
    assert "cancel_scheduled_llm_task" in names

    by_name = {t["function"]["name"]: t for t in ASSISTANT_TOOLS}
    sch = by_name["schedule_llm_task"]
    # Description must call out the @ask-with-tools shape and contrast
    # with set_reminder, so the LLM picks correctly.
    desc = sch["function"]["description"].lower()
    assert "@ask" in desc or "ask " in desc
    assert "set_reminder" in desc
    assert "tool" in desc

    # when_natural and prompt are required.
    params = sch["function"]["parameters"]
    assert "when_natural" in params["properties"]
    assert "prompt" in params["properties"]
    assert set(params["required"]) >= {"when_natural", "prompt"}
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -v -k schedule_llm_task_family
```

**Step 3: Append the schemas.**

In `assistant.py:ASSISTANT_TOOLS` (after the existing
`cancel_all_reminders` entry around line 430):

```python
{
    "type": "function",
    "function": {
        "name": "schedule_llm_task",
        "description": (
            "Schedule a future LLM task. At fire time the bot runs an "
            "@ask invocation as you, with full tool access (search, fetch, "
            "draw, code, Limnoria bridge). Use this for tasks that need "
            "TOOLS at fire time, e.g. 'every Monday at 9am check my open "
            "PRs and tell me which are stale'. For plain text reminders "
            "with no action, use set_reminder instead. Recurring is "
            "supported (numeric and calendar cadences)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "when_natural": {
                    "type": "string",
                    "description": (
                        "Natural-language schedule, e.g. 'in 30 min', "
                        "'every Monday at 9am', 'every 5 minutes'."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": (
                        "The bare instruction the bot should run at fire "
                        "time. Write it like you would type after `@ask`. "
                        "No 'remind me to', no time qualifier."
                    ),
                },
            },
            "required": ["when_natural", "prompt"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "list_scheduled_llm_tasks",
        "description": (
            "List your scheduled LLM tasks. Returns id, when, channel, "
            "and prompt for each. Use before cancel_scheduled_llm_task."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "cancel_scheduled_llm_task",
        "description": (
            "Cancel one of your scheduled LLM tasks by id. Get ids "
            "from list_scheduled_llm_tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "The scheduled-task id (e.g. 'llm_task_abc123').",
                },
            },
            "required": ["id"],
        },
    },
},
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -v -k schedule_llm_task_family
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/tests/test_assistant.py
git commit -m "feat(llm): register schedule_llm_task tool schemas"
```

**Done when:** the three names appear in `ASSISTANT_TOOLS`; their
descriptions distinguish from `set_reminder`; the assertion passes. No
handlers or capabilities yet — that's C2 + C3.

---

### Task C2: ToolSpec overrides (capability + visibility profile)

**Files:**

- Modify: `plugins/llm/src/llm/assistant.py` (`_TOOL_SPEC_OVERRIDES` ~563).
- Modify: the `test_assistant.py` test from C1 (extend with capability +
  visibility assertions).

**Decision: tool visibility and capability:**

| Tool                          | capability        | require_account | visible_in                            |
| ----------------------------- | ----------------- | --------------- | ------------------------------------- |
| schedule_llm_task             | `llm.ask`         | True            | `{"chat", "remind_action"}`           |
| list_scheduled_llm_tasks      | `llm.ask`         | False           | `{"chat", "remind_action"}`           |
| cancel_scheduled_llm_task     | `llm.ask`         | False           | `{"chat", "remind_action"}`           |

Why `require_account=True` on schedule but not list/cancel: scheduling is
the load-bearing identity claim — the schedule fires "as you" and gets
your bridge tools at fire time. Anonymous owner attribution is brittle
and would let any user with the same nick on a future connect take over
your schedule. List + cancel can use raw-nick fallback (matches the
reminder-system pattern and the design plan's "v1: rely on default
behaviour" stance on identity drift).

`visible_in={"chat", "remind_action"}` mirrors `set_reminder` — the
remind-action profile lets a fired task chain into another scheduled task
*if* the depth-cap allows it (which it doesn't for v1, since fire-time
msgs are tagged depth=1; this is why depth cap matters).

**Step 1: Extend the C1 test (or write a new one alongside).**

```python
def test_schedule_llm_task_specs_overrides_applied():
    from llm.assistant import ASSISTANT_TOOL_REGISTRY

    sch = ASSISTANT_TOOL_REGISTRY["schedule_llm_task"]
    assert sch.capability == "llm.ask"
    assert sch.require_account is True
    assert sch.visible_in == frozenset({"chat", "remind_action"})

    lst = ASSISTANT_TOOL_REGISTRY["list_scheduled_llm_tasks"]
    assert lst.capability == "llm.ask"
    assert lst.require_account is False
    assert lst.visible_in == frozenset({"chat", "remind_action"})

    can = ASSISTANT_TOOL_REGISTRY["cancel_scheduled_llm_task"]
    assert can.capability == "llm.ask"
    assert can.require_account is False
    assert can.visible_in == frozenset({"chat", "remind_action"})
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -v \
    -k schedule_llm_task_specs_overrides_applied
```

**Step 3: Add overrides.**

In `_TOOL_SPEC_OVERRIDES` (after `generate_code`):

```python
"schedule_llm_task": {
    "require_account": True,
},
# list/cancel default capability=llm.ask, require_account=False,
# visible_in={"chat", "remind_action"} — explicit no-op overrides not needed.
```

(`require_account=True` is the only deviation from defaults. The defaults
already give us capability `llm.ask` and `visible_in={"chat",
"remind_action"}`.)

**Step 4: Run; verify pass.**

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/tests/test_assistant.py
git commit -m "feat(llm): tool-spec overrides for schedule_llm_task family"
```

**Done when:** the override test passes; `denial_reason` (assistant.py:546)
correctly refuses unidentified callers from `schedule_llm_task` with
`Tool schedule_llm_task requires an authenticated account.`

---

### Task C3: `AssistantToolExecutor` handler methods + new fn parameters

**Files:**

- Modify: `plugins/llm/src/llm/assistant.py` (`AssistantToolExecutor.__init__`
  and three new `_tool_*` methods).
- Modify: the `test_assistant.py` test file (4 tests).

**Background — the four-callback pattern:**

`AssistantToolExecutor` already takes `set_reminder_fn` /
`delete_reminder_fn` / `cancel_all_reminders_fn` / `list_reminders_fn`
keyword args. The handler methods (`_tool_set_reminder`, etc.) call those
callbacks. The plumbing is the same for our three tools; we add three
matching kwargs and three handlers. The callbacks themselves are
constructed in plugin.py's `_scheduled_llm_task_fns` helper (Task D2).

**Step 1: Write failing tests.**

```python
def test_executor_accepts_scheduled_task_fns(mocker):
    from llm.assistant import AssistantToolExecutor

    schedule_fn = mocker.MagicMock()
    list_fn = mocker.MagicMock()
    cancel_fn = mocker.MagicMock()

    ex = AssistantToolExecutor(
        db=mocker.MagicMock(), context=mocker.MagicMock(),
        nick="n", channel="#t",
        capabilities=frozenset({"llm.ask"}),
        account="acct",
        schedule_llm_task_fn=schedule_fn,
        list_scheduled_llm_tasks_fn=list_fn,
        cancel_scheduled_llm_task_fn=cancel_fn,
    )
    assert ex._schedule_llm_task_fn is schedule_fn
    assert ex._list_scheduled_llm_tasks_fn is list_fn
    assert ex._cancel_scheduled_llm_task_fn is cancel_fn


def test_tool_schedule_llm_task_calls_callback_and_returns_json(mocker):
    import json
    from llm.assistant import AssistantToolExecutor

    schedule_fn = mocker.MagicMock(return_value={
        "status": "ok",
        "event_name": "llm_task_abc",
        "fire_at": 1700000000.0,
        "message": "Scheduled.",
    })

    ex = AssistantToolExecutor(
        db=mocker.MagicMock(), context=mocker.MagicMock(),
        nick="n", channel="#t",
        capabilities=frozenset({"llm.ask"}), account="acct",
        schedule_llm_task_fn=schedule_fn,
    )

    out = ex.execute(
        "schedule_llm_task",
        {"when_natural": "in 60s", "prompt": "ping me"},
    )
    parsed = json.loads(out.content)
    assert parsed["status"] == "ok"
    assert parsed["event_name"] == "llm_task_abc"
    schedule_fn.assert_called_once_with(
        when_natural="in 60s", prompt="ping me"
    )


def test_tool_list_scheduled_llm_tasks_returns_summary(mocker):
    import json
    from llm.assistant import AssistantToolExecutor

    list_fn = mocker.MagicMock(return_value=[
        {"id": "ev1", "when": "2026-05-02T13:00:00Z", "channel": "#t",
         "prompt": "check build", "recurrence": None},
        {"id": "ev2", "when": "2026-05-09T13:00:00Z", "channel": "#t",
         "prompt": "weekly digest",
         "recurrence": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9"},
    ])
    ex = AssistantToolExecutor(
        db=mocker.MagicMock(), context=mocker.MagicMock(),
        nick="n", channel="#t",
        capabilities=frozenset({"llm.ask"}), account="acct",
        list_scheduled_llm_tasks_fn=list_fn,
    )
    out = ex.execute("list_scheduled_llm_tasks", {})
    parsed = json.loads(out.content)
    assert parsed["status"] == "ok"
    assert len(parsed["tasks"]) == 2


def test_tool_cancel_scheduled_llm_task_passes_id(mocker):
    import json
    from llm.assistant import AssistantToolExecutor

    cancel_fn = mocker.MagicMock(return_value={
        "status": "ok",
        "event_name": "llm_task_abc",
        "message": "Cancelled.",
    })
    ex = AssistantToolExecutor(
        db=mocker.MagicMock(), context=mocker.MagicMock(),
        nick="n", channel="#t",
        capabilities=frozenset({"llm.ask"}), account="acct",
        cancel_scheduled_llm_task_fn=cancel_fn,
    )
    out = ex.execute("cancel_scheduled_llm_task", {"id": "llm_task_abc"})
    parsed = json.loads(out.content)
    assert parsed["status"] == "ok"
    cancel_fn.assert_called_once_with(event_name="llm_task_abc")
```

**Step 2: Run; verify fail.**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -v \
    -k "scheduled_task_fns or schedule_llm_task or list_scheduled_llm_tasks or cancel_scheduled_llm_task"
```

**Step 3: Implement.**

Add three kwargs to `AssistantToolExecutor.__init__` (after the existing
`code_fn`):

```python
schedule_llm_task_fn: Callable[..., dict[str, Any]] | None = None,
list_scheduled_llm_tasks_fn: Callable[[], list[dict[str, Any]]] | None = None,
cancel_scheduled_llm_task_fn: Callable[..., dict[str, Any]] | None = None,
```

Store them as attributes:

```python
self._schedule_llm_task_fn = schedule_llm_task_fn
self._list_scheduled_llm_tasks_fn = list_scheduled_llm_tasks_fn
self._cancel_scheduled_llm_task_fn = cancel_scheduled_llm_task_fn
```

Add three handler methods on `AssistantToolExecutor`:

```python
def _tool_schedule_llm_task(self, args: dict[str, Any]) -> str:
    if self._schedule_llm_task_fn is None:
        return self._err("Scheduling is not configured on this bot.")
    when_natural = str(args.get("when_natural") or "").strip()
    prompt = str(args.get("prompt") or "").strip()
    if not when_natural:
        return self._err("when_natural is required.")
    if not prompt:
        return self._err("prompt is required.")
    result = self._schedule_llm_task_fn(
        when_natural=when_natural, prompt=prompt
    )
    # The callback returns a plain dict with status/event_name/fire_at/message.
    return json.dumps(result)


def _tool_list_scheduled_llm_tasks(self, args: dict[str, Any]) -> str:
    del args  # noqa: ARG002 (no parameters)
    if self._list_scheduled_llm_tasks_fn is None:
        return self._err("Scheduling is not configured on this bot.")
    tasks = self._list_scheduled_llm_tasks_fn()
    return json.dumps({"status": "ok", "tasks": tasks})


def _tool_cancel_scheduled_llm_task(self, args: dict[str, Any]) -> str:
    if self._cancel_scheduled_llm_task_fn is None:
        return self._err("Scheduling is not configured on this bot.")
    event_name = str(args.get("id") or "").strip()
    if not event_name:
        return self._err("id is required.")
    result = self._cancel_scheduled_llm_task_fn(event_name=event_name)
    return json.dumps(result)
```

The dispatch path (`AssistantToolExecutor.execute`) already routes by
`spec.handler_name = f"_tool_{name}"` — no execute() changes needed.

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_assistant.py -v \
    -k "scheduled_task_fns or schedule_llm_task or list_scheduled_llm_tasks or cancel_scheduled_llm_task"
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/tests/test_assistant.py
git commit -m "feat(llm): AssistantToolExecutor handlers for scheduled-task tools"
```

**Done when:** the four tests pass; the `_tool_*` methods are wired and
dispatch through the existing `execute()` path; missing-callback paths
return clear errors.

---

## D — Plugin wiring + per-creator budget + depth cap

### Task D1: Register `bridgeScheduledTaskLimit` channel value

**Files:**

- Modify: `plugins/llm/src/llm/config.py` (after the Phase 2 Task 1
  `bridgeAllowMutating` block).
- Modify: `plugins/llm/tests/test_config.py` (one default-value test).

**Step 1: Failing test.**

```python
def test_bridge_scheduled_task_limit_default_is_five():
    import supybot.conf as conf
    import llm.config  # noqa: F401 — import side effect registers
    assert conf.supybot.plugins.LLM.bridgeScheduledTaskLimit() == 5
```

**Step 2: Run; verify fail.**

**Step 3: Register.**

```python
conf.registerChannelValue(
    LLM,
    "bridgeScheduledTaskLimit",
    registry.NonNegativeInteger(
        5,
        _("""Maximum number of active LLM-scheduled tasks per creator
        in this channel. Enforced at create time by the schedule_llm_task
        tool. Set to 0 to disable scheduling entirely.

        Each fire still counts against the user's normal askRateLimit
        bucket — this value caps the *number* of pending schedules, not
        their cumulative cost. The bridge* prefix is intentional: this is
        Phase 2 bridge-adjacent scheduling that can run bridge tools at
        fire time, even though schedule_llm_task itself is a native tool."""),
    ),
)
```

**Step 4: Run; verify pass.**

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat(llm): register bridgeScheduledTaskLimit channel value"
```

**Done when:** the registry value is exposed and defaults to 5; B1's
budget-enforcement test (Task B1, `test_schedule_llm_task_enforces_per_creator_limit`)
remains green.

---

### Task D2: `_scheduled_llm_task_fns` helper in `plugin.py`

**Files:**

- Modify: `plugins/llm/src/llm/plugin.py` (new helper next to `_reminder_fns`).
- Modify: `plugins/llm/tests/test_plugin.py` (one test that exercises the
  helper's wiring — match the style of the existing `_reminder_fns`-related
  tests; find with `grep -ln "_reminder_fns\|set_reminder_fn" plugins/llm/tests/test_plugin.py`).

**Step 1: Failing test.** (Inline a self-contained test that calls
`_scheduled_llm_task_fns` and verifies the returned dict's three callables
delegate to `LLMService.schedule_llm_task` / `list_scheduled_llm_tasks` /
`cancel_scheduled_llm_task` with the right caller identity bound in.)

**Step 3: Implement.**

```python
def _scheduled_llm_task_fns(
    self,
    *,
    caller: Identity,
    irc: callbacks.Irc,
    msg: IrcMsg,
    channel: str,
) -> dict[str, Callable[..., object]]:
    """Build the three-callable dict for the scheduled-task tools."""

    def schedule_fn(*, when_natural: str, prompt: str) -> dict[str, object]:
        result = self.llm_service.schedule_llm_task(
            irc=irc, msg=msg,
            creator_nick=caller.raw_nick,
            account=caller.account,
            channel=channel,
            when_natural=when_natural,
            prompt=prompt,
        )
        return {
            "status": result.status,
            "event_name": result.event_name,
            "fire_at": result.fire_at,
            "message": result.message,
            "note": result.note,
        }

    def list_fn() -> list[dict[str, object]]:
        rows = self.llm_service.list_scheduled_llm_tasks(
            creator_nick=caller.raw_nick, account=caller.account
        )
        return [
            {
                "id": row.event_name,
                "when": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(row.fire_at)
                ),
                "channel": row.channel,
                "prompt": row.prompt[:80],
                "recurrence": (
                    f"every {row.recurrence_seconds}s"
                    if row.recurrence_seconds is not None
                    else row.recurrence_rrule
                ),
            }
            for row in rows
        ]

    def cancel_fn(*, event_name: str) -> dict[str, object]:
        result = self.llm_service.cancel_scheduled_llm_task(
            event_name=event_name,
            creator_nick=caller.raw_nick,
            account=caller.account,
        )
        return {
            "status": result.status,
            "event_name": result.event_name,
            "message": result.message,
        }

    return {
        "schedule_llm_task_fn": schedule_fn,
        "list_scheduled_llm_tasks_fn": list_fn,
        "cancel_scheduled_llm_task_fn": cancel_fn,
    }
```

**Step 4: Run; verify pass.**

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): _scheduled_llm_task_fns helper for tool wiring"
```

**Done when:** the helper exists and the closure correctly binds
`caller` / `irc` / `msg` / `channel` so each tool call respects the
calling user's identity.

---

### Task D3: Pass scheduled-task fns into `assistant_request` from all entry points

**Files:**

- Modify: `plugins/llm/src/llm/plugin.py` (the chat/code/g/draw
  `assistant_request` call sites at ~2530, 2658, 2767, 2865 — search for
  `_reminder_fns` to locate them).
- Modify: `plugins/llm/src/llm/service.py` (`assistant_request`
  signature — add three pass-through kwargs and forward them to
  `AssistantToolExecutor`). The scheduled fire path in Task B1 already uses
  `getattr(plugin, "_scheduled_llm_task_fns", None)` to pass these callbacks
  once D2 exists, because it lives in service.py rather than plugin.py.
- Modify: `plugins/llm/tests/test_service.py` and `tests/test_plugin.py`
  for any plumbing tests that need extending.

**Background:** the existing pattern is `**self._reminder_fns(...)`
splatted into the call. We do the same with `**self._scheduled_llm_task_fns(...)`.
Service.py's `assistant_request` already accepts a long kwargs list; we
add three more.

**Step 1: Failing tests.** (Extend an existing assistant-request integration
test if one exists, otherwise add a minimal one in test_service.py that
verifies the three new fns flow through to `AssistantToolExecutor`.)

**Step 3: Implement.**

3a. In `service.py:assistant_request` signature (~2499):

```python
schedule_llm_task_fn: Callable[..., dict[str, Any]] | None = None,
list_scheduled_llm_tasks_fn: Callable[[], list[dict[str, Any]]] | None = None,
cancel_scheduled_llm_task_fn: Callable[..., dict[str, Any]] | None = None,
```

3b. Forward in the executor construction (~service.py:2604):

```python
schedule_llm_task_fn=schedule_llm_task_fn,
list_scheduled_llm_tasks_fn=list_scheduled_llm_tasks_fn,
cancel_scheduled_llm_task_fn=cancel_scheduled_llm_task_fn,
```

3c. In each of the four plugin.py call sites, add the splat. For
`_ask_impl` at ~2530:

```python
result = self.llm_service.assistant_request(
    request_text,
    request_context=request_context,
    db=self.db,
    context=self.context,
    bot_nick=irc.nick,
    images=images,
    history=history,
    channel_history=channel_history,
    irc=irc,
    msg=msg,
    memories=memories,
    system_prompt=effective_prompt,
    search_fn=...,
    fetch_fn=...,
    code_fn=...,
    draw_fn=...,
    cleanup_fn=...,
    extra_tools=extra_tools,
    extra_handlers=bridge_handlers,
    **self._reminder_fns(caller=caller, irc=irc, msg=msg),
    **self._scheduled_llm_task_fns(  # NEW
        caller=caller, irc=irc, msg=msg, channel=channel
    ),
)
```

Repeat for the three other plugin.py chat-profile call sites (`code`, `g`,
and `draw`). The existing reminder-action fire path at `plugin.py:1218`
should also receive the scheduled-task fns next to its `_reminder_fns(...)`
splat so action reminders and scheduled-task fires expose the same tool
surface:

```python
**self._scheduled_llm_task_fns(
    caller=caller, irc=active_irc, msg=synthetic_msg, channel=channel
),
```

3d. Update `LLMService._dispatch_scheduled_task` (added in B1) to pass the
same callbacks to its direct `assistant_request` call:

```python
**plugin._scheduled_llm_task_fns(
    caller=caller, irc=irc, msg=msg, channel=row.channel
),
```

**Note on the remind_action call site:** the fire-time @ask path also
gets the scheduled-task tools, so a fired schedule could ask the LLM to
schedule another. The depth cap (D4, set on the rehydrated msg) is what
prevents the loop. Make sure the `caller` / `msg` passed at this site
carries the depth tag — it does, because the rehydration happens in the
fire callback before any LLM dispatch.

**Step 4: Run; verify pass.**

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/src/llm/service.py \
    plugins/llm/tests/test_service.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): wire scheduled-task fns into assistant_request"
```

**Done when:** all chat/code/g/draw entry points and the existing
reminder-action fire path pass the three new callables; existing tests stay
green; `assistant_request` forwards them to the executor.

---

### Task D4: Depth-cap enforcement (sanity test only — already implemented in B1)

The depth check lives in `LLMService.schedule_llm_task` (Task B1) and reads
`msg.tagged("llm_schedule_depth")`. The fire callback in B1 sets that tag
to 1 on the rehydrated msg. This task is one **integration** test that
proves the depth cap fires end-to-end.

**Files:**

- Modify: `plugins/llm/tests/test_service.py` (one integration-shaped test).

**Step 1: Test.**

```python
def test_fired_task_cannot_schedule_a_nested_task(llm_service, db, mocker):
    """End-to-end: schedule a task; trigger the fire callback; observe
    that within the fired @ask, schedule_llm_task refuses."""
    # Arrange: register a one-shot schedule.
    add_event = mocker.patch("llm.service.schedule.addEvent")
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":rdrake!u@h PRIVMSG #t :@ask hi"
    msg.tagged.return_value = None
    irc = mocker.MagicMock(); irc.network = "afternet"
    mocker.patch.object(
        llm_service, "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule", seconds=60, message="x", confirmation="ok",
            note=None, action_prompt="check the build",
            recurrence_seconds=None, recurrence_rrule=None, watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )
    res = llm_service.schedule_llm_task(
        irc=irc, msg=msg, creator_nick="rdrake", account="rdrake_a",
        channel="#t", when_natural="in 60s", prompt="do x",
    )
    assert res.status == "ok"

    # Capture the registered closure.
    fire_callable = add_event.call_args.args[0]

    captured: dict[str, object] = {}
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]
    llm_service.plugin._check_rate_limit.return_value = False
    llm_service.plugin._gather_history.return_value = ([], [])
    llm_service.plugin._get_user_memories.return_value = []
    mocker.patch.object(llm_service.plugin.db, "get_instruction", return_value="")
    llm_service.plugin._reminder_fns.return_value = {}
    llm_service.plugin._scheduled_llm_task_fns.return_value = {}

    def fake_assistant_request(*, msg, **_kwargs):
        captured["depth"] = msg.tagged("llm_schedule_depth")
        nested = llm_service.schedule_llm_task(
            irc=irc, msg=msg,
            creator_nick="rdrake", account="rdrake_a",
            channel="#t", when_natural="in 60s do y", prompt="do y",
        )
        captured["nested_status"] = nested.status
        captured["nested_message"] = nested.message
        return mocker.MagicMock(content="", model="m", prompt_tokens=0,
                                completion_tokens=0, cost=0.0,
                                grounding_used=False)

    mocker.patch.object(
        llm_service, "assistant_request", side_effect=fake_assistant_request
    )

    # Act: fire the schedule.
    fire_callable()

    # Assert.
    assert captured["depth"] == 1
    assert captured["nested_status"] == "error"
    assert "depth" in captured["nested_message"].lower() \
        or "scheduled" in captured["nested_message"].lower()
```

**Steps 2–4:** run, expect pass (the unit-level B1 test already covers the
gate logic; this exercises the closure path end-to-end). Adjust mock
patching as needed to match your test framework setup.

**Step 5: Commit.**

```bash
git add plugins/llm/tests/test_service.py
git commit -m "test(llm): end-to-end depth-cap on fired schedule_llm_task"
```

**Done when:** the integration test passes, proving the depth tag flows
from the fire closure through the fired @ask to a recursive
`schedule_llm_task` and that the nested call refuses.

---

## E — Documentation

### Task E1: Operator docs — describe the new tools and budget

**Files:**

- Modify: `docs/guide/operator/tuning-monitoring.md` (or whichever file
  Phase 1/Phase 2-Task-1 used; verify with
  `grep -ln 'bridgeAllowMutating' docs/guide/operator/`).

**Step 1: Add a "Scheduled LLM tasks" section after the
`bridgeAllowMutating` section.** Topics:

- What `schedule_llm_task` does and how it differs from `set_reminder`
  ("set_reminder = fixed text at fire time; schedule_llm_task = full @ask
  invocation with tools at fire time"). One example each.
- The per-channel budget knob:
  ```
  config channel #yourchan plugins.LLM.bridgeScheduledTaskLimit 10
  ```
- Identity drift caveat — schedules are bound to the creator's IRC
  identity at create time; if the user disconnects the schedule still
  fires (the bot replays the persisted msg) but tool-time capabilities
  fall back to whatever the bot can resolve from the persisted prefix
  and account-tag at fire time. Account-identified users are stable
  across disconnects.
- Pointer to `MUTATING_COMMANDS` (Phase 2 Task 1) for the bridge
  read/write distinction.
- How to inspect via debug: enable `bridgeDebugInChannel` to see fire-time
  bridge calls; use `@vibebot list scheduled tasks` for these DB-backed
  native schedules. `@scheduler list` will NOT show them because raw
  `supybot.schedule.addEvent` events do not populate the Scheduler plugin's
  `self.events` dict.

**Step 2: Skim build (if `mkdocs serve` is set up) or eyeball the rendered Markdown.**

**Step 3: Commit.**

```bash
git add docs/guide/operator/tuning-monitoring.md
git commit -m "docs(llm): document schedule_llm_task and per-creator budget"
```

### Task E2: AGENTS.md mention

**File:** `AGENTS.md`.

Update the existing LLM-plugin entry to mention Phase 2 Task 3:

```markdown
- `plugins/llm/src/llm/{service,plugin,assistant,persistence}.py` —
  Phase 2 Task 3 native scheduling (`schedule_llm_task` + companions);
  see docs/plans/2026-05-02-task-3-schedule-llm-task-implementation-plan.md.
```

```bash
git add AGENTS.md
git commit -m "docs: link Phase 2 Task 3 plan in AGENTS.md"
```

**Done when:** operator docs explain the new tools and budget; AGENTS.md
points at this plan.

---

## Validation

### Automated

```bash
# New service tests (depth cap, budget, parse, list, cancel, restore, fire-loop).
uv run pytest plugins/llm/tests/test_service.py -v \
    -k "schedule_llm_task or list_scheduled_llm_tasks or cancel_scheduled_llm_task or restore_scheduled_llm_tasks"

# Persistence tests for v13 + helpers.
uv run pytest plugins/llm/tests/test_persistence.py -v -k "schema_v13 or scheduled_llm_task"

# Assistant tool registration + executor wiring.
uv run pytest plugins/llm/tests/test_assistant.py -v \
    -k "schedule_llm_task or list_scheduled_llm_tasks or cancel_scheduled_llm_task"

# Config — new registry value default.
uv run pytest plugins/llm/tests/test_config.py -v -k bridge_scheduled_task_limit

# Full LLM suite — must remain green.
uv run pytest plugins/llm/tests -q

# Repository-wide gates from AGENTS.md.
make lint
make typecheck
make preflight
```

All commands green before declaring Task 3 done.

### Operational verification on the running bot

The standard CI → Docker build → systemctl restart cycle is pre-authorized
for this repo (auto-memory `feedback_restart_authorization`). Run after the
PR merges to `main` and the Docker build completes.

1. **Wait for both CI and the Docker image build to finish.** They are
   separate workflows; restarting after only CI runs the *previous*
   image (auto-memory `feedback_wait_for_docker`).

2. **SSH and restart:**
   ```bash
   ssh -i ~/.ssh/id_rsa vibebot@rdrake.org "systemctl --user restart vibebot"
   ```
   On `Permission denied (publickey)`: ask the user to run
   `security unlock-keychain` locally (auto-memory `feedback_ssh_keychain_unlock`).

3. **Smoke — one-shot @ask scheduling, in `#test`:**
   - `@vibebot in 60 seconds tell me one fun fact about postgres`
     - Expected: bot reacts with the clock emoji (set_reminder ack
       pattern); 60s later, the LLM speaks one fun fact in `#test`.
   - During the wait, run `@vibebot list scheduled tasks` — the task should
     appear via the `list_scheduled_llm_tasks` tool. Do not use
     `@scheduler list` for this smoke: these are raw DB-backed schedule
     events, not Scheduler-plugin events.

4. **Smoke — recurring tasks (numeric cadence):**
   - `@vibebot every 2 minutes search for the latest mastodon release and post it once`
     - Expected: scheduled; first fire ~2 min later; the second fire
       lands ~2 min after the first. Cancel before this gets out of hand:
       `@vibebot cancel my mastodon release task` → bot uses
       `list_scheduled_llm_tasks` then `cancel_scheduled_llm_task`.

5. **Smoke — recurring tasks (RRULE):**
   - `@vibebot every Monday at 9 UTC post the weekly digest`
     - Expected: scheduled; the parser confirmation reflects the next
       Monday 9 UTC. (You don't need to wait until Monday for soak —
       confirm the row, then cancel.)

6. **Restart-survival:**
   - Schedule a task `@vibebot in 5 minutes ping me`.
   - Restart the bot via `systemctl --user restart vibebot`.
   - Wait ≤5 min and observe the ping fire. If the bot was offline
     past the fire_at, the task should fire ~immediately on restart
     (matches the reminder pattern — past-due → near-immediate fire).

7. **Depth cap:**
   - `@vibebot in 60s schedule a task that schedules another task`
   - Expected at fire time: the LLM tries to call
     `schedule_llm_task` from inside the fired @ask, gets the depth-cap
     refusal envelope, and either says "I can't schedule from inside a
     fire" or apologises. Inspect logs:
     `journalctl --user -u vibebot -e | grep llm_schedule_depth`.

8. **Per-creator budget:**
   - Schedule 5 tasks in a row (`bridgeScheduledTaskLimit` defaults to 5).
   - The 6th attempt should refuse with the budget message; the LLM
     surfaces it.

9. **Tool-selection sanity (informal):** in 3 separate `#test` exchanges,
   ask things that *should* go to `set_reminder` (no tool needed at fire
   time) and things that *should* go to `schedule_llm_task` (tool needed).
   Verify the LLM picks correctly. If it picks wrong twice in a row,
   tighten the descriptions in C1 before declaring done.

**Done when:** all automated tests pass, the 9 smoke scenarios behave as
documented, and the bot's logs show no unhandled exceptions during the
fire-time path.

---

## Open questions for code review

These flags are restated here so the reviewer doesn't have to scroll.

1. **Reply-target override.** Phase 2 plan §"Open question #1" defers
   cross-target scheduling. v1 always replies in the channel where the
   schedule was created (the `wire_msg`'s `args[0]`). When the feature
   is wanted, add a `reply_target` column + tool parameter then —
   not now (avoids unused parameters in the v1 surface).

2. **Identity drift on disconnect.** `schedule_llm_task` requires an
   authenticated account, and the persisted `wire_msg` plus explicit
   `account` column carry that account into fire-time capability/tool
   attribution. The raw prefix may still be stale after disconnect/reconnect,
   but account-tag-based identity is the supported v1 path; unidentified
   callers are refused at create time.

3. **Network capture at restart.** Resolved in this plan: v13 stores a
   `network` column and the fire callback resolves `world.getIrc(row.network)`
   first, falling back to `world.ircs[0]` only if the named network is
   unavailable.

4. **`schedule_llm_task` description tuning.** C1's wording was chosen to
   match the design plan's "actions need tools at fire time" framing. If
   the LLM picks the wrong tool in soak (e.g. answers "in 5 min draw a
   cat" with `set_reminder` instead of `schedule_llm_task`), tighten the
   description and re-soak. Bikeshed at code review.

5. **Migration of existing reminders to `schedule_llm_task`.** Out of
   scope (Phase 2 plan §"What this plan does not commit to" — both
   systems coexist). No migration code; `set_reminder` and
   `schedule_llm_task` are independent persistence layers. PR-review
   item if anyone proposes folding them; lean: defer.

---

## Execution order summary

| Order | Task | Output |
| --- | --- | --- |
| 0 | Pre-flight (verify line numbers, baseline tests, no name collision, pickle behaviour) | (no commit) |
| 1 | A1: SCHEMA_VERSION = 13 + scheduled_llm_tasks table | commit |
| 2 | A2: persistence helpers (save / update / delete / load / count) | commit |
| 3 | B1: `schedule_llm_task` service method + fire callback | commit |
| 4 | B2: `list_scheduled_llm_tasks` + `cancel_scheduled_llm_task` service methods | commit |
| 5 | B3: `restore_scheduled_llm_tasks` on plugin init | commit |
| 6 | C1: tool schemas in `ASSISTANT_TOOLS` | commit |
| 7 | C2: ToolSpec overrides | commit |
| 8 | C3: `AssistantToolExecutor` handlers + new fn params | commit |
| 9 | D1: `bridgeScheduledTaskLimit` registry value | commit |
| 10 | D2: `_scheduled_llm_task_fns` helper in plugin.py | commit |
| 11 | D3: pass through to `assistant_request` from all entry points | commit |
| 12 | D4: integration test for depth cap | commit |
| 13 | E1: operator docs | commit |
| 14 | E2: AGENTS.md mention | commit |
| 15 | Validation: full automated suite + 9 smoke scenarios | (no commit) |

Each task is independently verifiable: tests pass after the commit, and
reverting one commit cleanly leaves the codebase in a working state.
A1 → B3 are the load-bearing scheduler glue; C1 → C3 are the LLM-tool
surface; D1 → D4 are the per-channel knobs and the loop guard;
E1 → E2 are docs.
