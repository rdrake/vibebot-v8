# Persistent Conversation Context

## Problem

Conversation context lives entirely in memory. When the bot restarts (deploys, crashes, server maintenance), all user conversations are lost. Users must re-establish context from scratch.

## Design

Persist per-user conversation context to the existing SQLite database. Channel context (shared, 10-message max) is not persisted — it refills naturally as people talk.

### Schema

Bump `SCHEMA_VERSION` from 3 to 4. Migration creates one table:

```sql
CREATE TABLE conversations (
    nick TEXT NOT NULL,
    channel TEXT NOT NULL,
    messages TEXT NOT NULL,
    last_activity REAL NOT NULL,
    PRIMARY KEY (nick, channel)
);
```

`nick` and `channel` are stored lowercased, matching `_get_key()` normalization.

`messages` stores the full message list as a JSON array of `{"role": str, "content": str}` dicts — the same format `ConversationContext` already uses internally.

### Persistence Methods (`LLMDatabase`)

Four new methods:

- **`save_conversation(nick, channel, messages, last_activity)`** — `INSERT OR REPLACE` with JSON-serialized messages. Nick/channel are lowercased before storage.
- **`delete_conversation(nick, channel)`** — `DELETE` by primary key. Called on `clear`.
- **`delete_all_conversations()`** — `DELETE FROM conversations`. Called on `clear_all`.
- **`load_conversations()`** — `SELECT *`, deserialize JSON per row with try/except (log and skip corrupt rows), return list of `(nick, channel, messages_list, last_activity)` tuples. Called once at startup.

### Integration (`ConversationContext`)

`ConversationContext.__init__` gains an optional `db: LLMDatabase | None = None` parameter, stored as `self._db`:

- **Startup**: when `db` is provided, call `db.load_conversations()` and populate `self._conversations`. Set each `Conversation.last_activity` from the stored DB value (not `time.time()`). Filter out already-expired rows using the instance default config.
- **`add_message`**: after updating in-memory state (existing logic unchanged), call `self._db.save_conversation()` if db is set.
- **`clear`**: after deleting from memory, call `self._db.delete_conversation()`.
- **`clear_all`**: after clearing memory dicts, call `self._db.delete_all_conversations()`.
- **`_prune_expired`**: after removing expired keys from memory, also delete them from DB via `self._db.delete_conversation()` for each pruned key.

When `db` is `None` (tests, default), all persistence calls are skipped. Existing tests pass without modification.

### Persistence Scope

Only command interactions (`%ask`, `%picard`, `%code`) trigger persistence — not passively observed channel messages. The `doPrivmsg` handler calls `add_message` for every IRC message seen, which would cause excessive writes. To solve this, `add_message` gains an optional `persist: bool = True` parameter. The `doPrivmsg` call site passes `persist=False`; command handlers use the default `True`.

This means two UPSERTs per `%ask` interaction (user + assistant message). This is acceptable — the volume is low and SQLite WAL handles it fine.

### Plugin Wiring

Two changes in `plugin.py`:

1. **Init order**: move `self.db = LLMDatabase(db_path)` before `_init_context()` so the db exists when `ConversationContext` is constructed.
2. **Pass db**: `_init_context()` passes `self.db` to `ConversationContext`.

### Cross-Channel Prune Safety

`_prune_expired` currently receives a single `cfg` from whatever channel triggered the call and applies it to all conversations. This is an existing in-memory bug that becomes worse with persistence since DB deletes are permanent. For now, use the instance default config (`self.config`) for the prune sweep — matching what `get_stats` already does. This does not make the existing behavior worse and avoids a larger refactor.

## What Changes

| File | Change |
|------|--------|
| `persistence.py` | Schema v3->v4 migration, 4 new methods |
| `context.py` | Accept optional `db`, persist on mutating operations, `persist` flag on `add_message` |
| `plugin.py` | Init order swap, pass `self.db` to `ConversationContext`, `persist=False` in `doPrivmsg` |

## What Doesn't Change

- Channel context remains in-memory only
- All existing `ConversationContext` tests pass (db defaults to None)
- Timeout-based expiry logic unchanged
- No new dependencies
