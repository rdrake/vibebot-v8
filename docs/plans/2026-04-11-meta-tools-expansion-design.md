# Meta Tools Expansion Design

Expand the meta command with usage, memory cleanup, and reminder tools.

## Problem

Phase 1 of the meta command covers instructions, memories, and context.
Users still need explicit commands for usage stats, memory cleanup, and
reminders — all natural candidates for natural language interaction.

## New Tools

### Usage (read-only, simple DB queries)

| Tool | Parameters | Maps to |
|------|-----------|---------|
| `get_usage` | — | `db.get_usage_summary_for_nick(nick, since=month_start)` |
| `get_channel_usage` | — | `db.get_usage_summary_for_channel(channel, since=month_start)` |

Both return `UsageSummary` (requests, prompt tokens, completion tokens,
cost) for the current calendar month. The LLM formats it naturally.

### Memory Cleanup (triggers existing LLM-based cleanup)

| Tool | Parameters | Maps to |
|------|-----------|---------|
| `cleanup_memories` | — | `plugin._run_memory_cleanup(nick, channel)` |

Triggers the dedup/merge/prune pass. Returns a summary string.
Requires at least 2 memories. Makes its own LLM call using the
cleanup model.

### Reminders (complex lifecycle — scheduler + dict + DB)

| Tool | Parameters | Maps to |
|------|-----------|---------|
| `list_reminders` | — | `plugin._get_user_reminders(nick)` |
| `set_reminder` | `text: str` | `plugin._remind_set_for_meta(irc, msg, nick, text)` |
| `delete_reminder` | `id: str` | `plugin._remind_delete_for_meta(nick, rid)` |

`set_reminder` takes natural language text (e.g., "check the build in
30 minutes"). The existing `parse_reminder()` service method handles
time extraction.

`delete_reminder` takes the short hex ID (last 12 chars of event_name),
matching the existing `remind del` interface.

## Architecture: Callable Pattern

Reminders and memory cleanup require plugin-level coordination (Limnoria
scheduler, in-memory dicts, database). Rather than giving the executor
direct access to the plugin, we pass pre-bound callables:

```python
MetaToolExecutor(
    db=self.db,
    context=self.context,
    nick=nick,
    channel=channel,
    cleanup_fn=lambda: self._run_memory_cleanup(nick, channel),
    list_reminders_fn=lambda: self._get_user_reminders(nick),
    set_reminder_fn=lambda text: self._remind_set_for_meta(irc, msg, nick, text),
    delete_reminder_fn=lambda rid: self._remind_delete_for_meta(nick, rid),
)
```

The executor calls these opaquely and returns the result string to the
LLM. All scheduler/dict/DB coordination stays in the plugin.

## New Plugin Methods

### `_remind_set_for_meta(irc, msg, nick, text) -> str`

Same logic as `_remind_set()` but returns a result string instead of
calling `irc.reply()`. Reuses `llm_service.parse_reminder()` for time
parsing, `schedule.addEvent()` for scheduling, and `db.save_reminder()`
for persistence.

Returns:
- Success: `"Reminder set: <message> (in <duration>)"`
- Parse error: `"Could not parse reminder: <reason>"`
- Validation error: `"Reminder must be between 10 seconds and 7 days"`

### `_remind_delete_for_meta(nick, rid) -> str`

Wraps the find → removeEvent → dict pop → db.delete dance.

Returns:
- Success: `"Deleted reminder <id>"`
- Not found: `"Reminder <id> not found"`

## Tool Count

Phase 1: 9 tools (instructions, memories, context)
Phase 2: 6 new tools (usage, cleanup, reminders)
Total: 15 tools

## Example Interactions

```
<user> @meta how much have I used this month?
<bot>  You've made 47 requests this month, costing $0.12.

<user> @meta remind me to deploy in 2 hours
<bot>  Reminder set: deploy (in 2 hours).

<user> @meta clean up my memories
<bot>  Cleanup complete. Before: 12 | dropped 3, merged 4 → 2 | after: 7

<user> @meta show my reminders
<bot>  You have 2 reminders: 1) abc123: deploy (in 1h42m) | 2) def456: check CI (tomorrow)

<user> @meta cancel the deploy reminder
<bot>  Deleted reminder abc123.
```

## Security

- Usage tools are read-only, scoped to the calling user/channel
- Cleanup triggers the existing cleanup path with all its validations
- Reminder tools use the same constraints as the explicit commands
  (10 second min, 7 day max, scoped to calling user)
- All callables are pre-bound to the user's nick — no cross-user access

## Testing

- Usage tools: mock `db.get_usage_summary_for_nick()` and
  `db.get_usage_summary_for_channel()`
- Cleanup: mock the callable, verify it's called
- Reminder list/set/delete: mock callables, verify arguments and
  return values
- Integration: real DB + mocked LLM for set_reminder flow
