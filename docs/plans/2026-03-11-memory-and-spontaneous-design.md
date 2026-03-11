# Long-Term Memory & Spontaneous Participation

## Problem

The bot has no long-term memory. It forgets everything about a user when conversation context expires (30 minutes) or resets. Users must re-establish preferences and facts every session.

Separately, the bot only speaks when directly commanded. It passively tracks channel messages but never participates on its own, making it feel like a tool rather than a channel member.

## Feature 1: Long-Term Memory

### What It Does

The bot automatically extracts and remembers facts about users from command interactions (ask, picard, code). Memories persist permanently in SQLite and are injected into future conversations.

### Privacy Boundary

Only messages from explicit bot commands are eligible for memory extraction. Passively observed channel messages (via `doPrivmsg` tracking) are never mined for facts. If a user invokes `%ask`, that conversation is fair game. If they're just chatting in channel, it's not.

### Schema

Bump `SCHEMA_VERSION` from 4 to 5. Migration creates one table:

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nick TEXT NOT NULL,
    fact TEXT NOT NULL,
    source_channel TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX idx_memories_nick ON memories(nick);
```

Nick is stored lowercased, matching existing normalization.

### Extraction Flow

After every command interaction that produces a successful LLM response:

1. Fire a separate cheap call to a flash-tier model.
2. Send the conversation (user messages + assistant response) with a purpose-built extraction prompt: "Extract new facts about this user worth remembering long-term. Return a JSON array of short factual strings, or an empty array if nothing notable."
3. Insert any extracted facts into the `memories` table.
4. This call runs asynchronously (background thread) so it doesn't delay the user's response.

### Retrieval

When building the system prompt for any command, query all memories for the requesting user and append them as a "What you know about this user" section in the system prompt. Simple `SELECT fact FROM memories WHERE nick = ?`.

### User Control

- `%memories` — Lists all facts the bot has stored about you.
- `%memories delete <id>` — Delete a specific memory by ID.
- `%memories clear` — Delete all your memories.

### Deduplication

The extraction prompt instructs the model to avoid facts the bot already knows. Pass existing memories into the extraction call so it can skip duplicates. Over time, if duplicates slip through, users can clean up via `%memories`.

### Persistence Methods (LLMDatabase)

- **`save_memory(nick, fact, source_channel)`** — INSERT with lowercased nick, `time.time()` for created_at.
- **`get_memories(nick)`** — SELECT all facts for nick (lowercased), return list of `(id, fact, created_at)`.
- **`delete_memory(nick, memory_id)`** — DELETE by id AND nick (prevents deleting other users' memories).
- **`delete_all_memories(nick)`** — DELETE all for nick.

## Feature 2: Spontaneous Participation

### What It Does

In explicitly enabled channels, the bot occasionally evaluates whether it has something worth contributing to the conversation. It uses a cheap model with a system prompt tuned for casual, brief participation.

### Trigger

Inside `doPrivmsg`, after existing passive tracking logic, a random roll decides whether to evaluate a reply:

- Only fires when `spontaneousEnabled=True` for the channel (default: False).
- Probability controlled by `spontaneousChance` (1-100, default: 3, meaning ~3% per message).
- Per-channel cooldown prevents firing more than once per `spontaneousCooldown` minutes (default: 5).

### Evaluation Flow

When triggered:

1. Grab recent channel context (already tracked by existing `doPrivmsg` logic).
2. Call `service.completion()` (reusing the existing ask path) with:
   - `spontaneousModel` (channel-specific)
   - `spontaneousApiKey` (falls back to `askApiKey` if empty)
   - `spontaneousSystemPrompt` (tuned for casual participation)
3. If the model responds with "PASS", discard. Otherwise, send the reply to the channel.

The system prompt heavily encourages PASS — most evaluations should result in silence.

### Config

| Setting | Scope | Default | Purpose |
|---------|-------|---------|---------|
| `spontaneousEnabled` | Channel | False | Master switch |
| `spontaneousChance` | Channel | 3 | % chance per message |
| `spontaneousCooldown` | Channel | 5 | Minimum minutes between replies |
| `spontaneousModel` | Channel | `gemini/gemini-2.5-flash-lite-preview` | Cheap model |
| `spontaneousApiKey` | Global | `""` (falls back to askApiKey) | API key override |
| `spontaneousSystemPrompt` | Channel | (see below) | Personality for interjections |

### System Prompt (Default)

Tuned for brevity and restraint. Something like:

> You are a participant in an IRC channel. You see the recent conversation and may reply if you have something genuinely useful, funny, or relevant to add. Keep it brief — one or two sentences max. Match the tone of the channel. If you don't have anything worth saying, respond with exactly PASS. Most of the time you should PASS. You're a channel regular, not an assistant.

### Threading

The evaluation runs in a background thread (or via `schedule.addEvent`) so it doesn't block `doPrivmsg`. A slight delay before sending also makes the reply feel more natural.

### Rate Limiting

A simple in-memory dict of `{channel: last_spontaneous_time}`. Check before making the API call, not after. This means the random roll can "waste" a hit if cooldown is active, but that's fine — it's cheaper than the API call.

## What Changes

| File | Change |
|------|--------|
| `persistence.py` | Schema v4->v5, `memories` table, 4 CRUD methods |
| `service.py` | `extract_memories()` method for async fact extraction |
| `config.py` | Memory config (model for extraction) + spontaneous config (6 settings) |
| `plugin.py` | `%memories` command, extraction hook after commands, spontaneous branch in `doPrivmsg` |
| `context.py` | No changes |

## What Doesn't Change

- Conversation context system unchanged
- Existing commands unchanged
- Passive tracking behavior unchanged (still opt-in per channel)
- No new dependencies

## Future: Connecting the Features

Once both features are stable, a follow-up wires memories into spontaneous replies: pull memories for users visible in the channel context and include them in the spontaneous system prompt. This is a small retrieval addition, no architectural changes needed.
