# Memory & Instructions

VibeBot has two kinds of memory: **volatile context** (short-term conversation history) and **non-volatile memory** (long-term stored facts). You also have **custom instructions** to personalize how the bot responds.

!!! tip "Natural language"
    You can manage all of these by mentioning the bot or sending a PM. For example: "VibeBot, what do you remember about me?" or "VibeBot, forget our conversation." Use the commands when you want direct control.

## Volatile context

Every time you use `@ask` or `@code`, the bot remembers the conversation. This lets you ask follow-up questions without repeating yourself.

- Context is per user, per channel -- your conversation in `#general` is separate from `#dev`.
- Context expires automatically after a few minutes of inactivity (default: 5 minutes).
- Other users' messages in the channel can also be included in context, so the bot can follow group conversations.

### Clearing context

Use `@forget` to wipe your conversation history and start fresh:

```
@forget
@forget #otherchannel
```

## Non-volatile memory

The bot automatically extracts facts from your conversations and stores them long-term. These facts are recalled in future conversations to give more relevant answers -- for example, if you mention your preferred programming language, the bot will remember that.

### Viewing your memories

```
@memories
```

This lists all stored facts with their IDs:

```
[1] Prefers Python for scripting | [2] Works on network infrastructure | [3] Located in Ontario
```

### Managing memories

You can manage memories with commands or natural language:

```
VibeBot, delete the memory about Python
VibeBot, update memory 2 to say cloud infrastructure
VibeBot, clean up my memories
```

Or use commands directly:

**Delete one or more memories:**

```
@memories delete 3
@memories delete 1 2 3
```

**Edit a memory:**

```
@memories edit 2 Works on cloud infrastructure
```

**Delete all memories:**

```
@memories clear
```

**Deduplicate and clean up:**

```
@memories cleanup
```

This runs an AI pass over your memories to merge duplicates and remove outdated facts.

## Custom instructions

Set persistent instructions that change how `@ask` responds to you. Your instruction is prepended to the system prompt for every `@ask` call.

You can set instructions with natural language or with the command:

```
VibeBot, from now on respond in French
VibeBot, what's my current instruction?
```

**Set an instruction:**

```
@instruct Always respond in French
@instruct You are Captain Picard. Respond in character.
@instruct Explain things like I'm a senior developer, skip the basics
```

**View your current instruction:**

```
@instruct
```

**Remove your instruction:**

```
@instruct clear
```

Instructions only affect `@ask` -- they don't change `@code` or `@draw` behavior.
