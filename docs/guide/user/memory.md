# Memory and instructions

VibeBot has two kinds of memory: **conversation context** (short-term) and **long-term memories** (stored facts). You can also set **custom instructions** that persist across conversations.

!!! tip "Natural language"
    You can manage all three by talking to the bot: "VibeBot, what do you remember about me?" or "VibeBot, forget our conversation." Use the commands when you want direct control.

## Conversation context

Every `@ask` or `@code` adds to a short-term conversation history, so you can ask follow-up questions without repeating yourself.

- Context is per user, per channel: your conversation in `#general` stays separate from `#dev`.
- Context expires after a few minutes of inactivity (default: 5 minutes).
- The bot can also weave in recent channel messages, so it can follow group conversations.

Clear your context and start fresh with:

```
@forget
@forget #otherchannel
```

## Saved memories

The bot picks up durable facts from your conversations, such as your preferred programming language, and recalls them later to give more relevant answers.

A fact isn't saved the first time it comes up. It becomes a candidate, and the bot promotes it to a saved memory when it recurs in a later conversation. Candidates that never recur expire after about two weeks. This keeps one-off remarks out of your memory list.

### Viewing your memories

```
@memories
```

This lists your stored facts with their IDs:

```
[1] Prefers Python for scripting | [2] Works on network infrastructure | [3] Located in Ontario
```

### Managing memories

Natural language works:

```
VibeBot, delete the memory about Python
VibeBot, update memory 2 to say cloud infrastructure
VibeBot, clean up my memories
```

Commands work too:

```
@memories delete 3          # delete one (del also works)
@memories delete 1 2 3      # delete several
@memories edit 2 Works on cloud infrastructure
@memories clear             # delete all
@memories cleanup           # AI pass: merge duplicates, drop stale facts
```

## Custom instructions

Set a persistent instruction that shapes how `@ask` responds to you. The bot prepends your instruction to the system prompt on every `@ask`.

```
@instruct Always respond in French
@instruct You are Captain Picard. Respond in character.
@instruct           # show your current instruction
@instruct clear     # remove it
```

Natural language works too: "VibeBot, from now on treat me as a senior developer."

Instructions only affect `@ask`; they don't change `@code` or `@draw`.

## Avatar persona

In verse-enabled channels, your avatar has its own voice, separate from `@instruct`:

```
@avatar A moss-covered tree spirit who speaks in riddles.
@avatar           # show your current persona
@avatar clear     # remove it
```

The persona shapes how the verse portrays your avatar. It never affects `@ask`. See [The verse](ai-commands.md#the-verse) for the other verse commands.
