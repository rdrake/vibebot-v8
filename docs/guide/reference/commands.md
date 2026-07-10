# Command reference

Quick reference for every VibeBot command. For walkthroughs and
examples, see the [user guide](../user/getting-started.md).

## Natural language

Most bot features work through plain language. Mention the bot by name
in a channel, or send it a PM:

```
VibeBot, what's the weather like on Mars?
VibeBot, remind me in 2 hours to deploy
VibeBot, what do you remember about me?
VibeBot, how much have I used this month?
VibeBot, set my instruction to respond in haiku
```

The bot uses tools internally to handle your request: managing
memories, setting reminders, checking usage, and more. Natural language
works especially well for combining actions in one message.

Use the commands below when you want direct, predictable behaviour.

## Core AI commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@ask` | `<question>` | Ask the AI a question. Supports follow-ups and image URLs. Requires `llm.ask`. |
| `@code` | `<request>` | Generate code, served as an HTTP link with syntax highlighting. Requires `llm.code`. |
| `@draw` | `<prompt>` | Generate an image from text. Requires `llm.draw` and an authenticated account. |
| `@story` | `<brief>` | Generate an illustrated page (tale or explainer) and post a link when ready. Same gate as `@draw`. |

### ask

Ask the AI a question. Supports conversation context for follow-up
questions and vision for image URLs.

```
@ask What is the capital of France?
@ask Describe this: https://example.com/image.jpg
@ask And what about Germany?
```

See [AI commands](../user/ai-commands.md) for full details.

### code

Generate code from your request. The bot saves the code to an HTTP link
with syntax highlighting and keeps context, so you can iterate.

```
@code Python function to calculate fibonacci numbers
@code Now add memoization to that
```

### draw

Generate an image from a text description. Requires an authenticated
account.

```
@draw A sunset over mountains in watercolor style
@draw A cyberpunk cityscape at night
```

### story

Generate an illustrated page from your brief and post a link when the
page is ready. The bot picks one of two modes from your wording: an
in-character illustrated tale, or a concept explainer with labelled
diagrams. No verse mode required. A per-account cooldown applies,
shared with the verse storybook tool.

```
@story an illustrated tale of the crew winning the pub quiz
@story explain how photosynthesis works, with diagrams
```

## Memory commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@forget` | `[<channel>]` | Clear your volatile conversation context. |
| `@memories` | `[<nick> \| del <id> \| edit <id> <text> \| clear \| cleanup]` | Manage stored facts. Viewing another user's memories is owner-only. |
| `@instruct` | `[<instruction> \| clear]` | Set persistent instructions for `@ask`. Empty shows the current one. |

```
@memories
@memories del 3
@memories edit 5 corrected fact
@memories clear
@instruct Respond only in haiku
@instruct clear
```

See [memory and instructions](../user/memory.md) for full details.

## Reminders

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@remind` | `[<text> \| list \| del <id> \| clear \| admin <list\|del\|clear> <nick> [<id>...]]` | Natural-language reminders. The `admin` subcommands are owner-only. |

Reminders that ask the bot to *do* something (look up, check, fetch,
summarize) run as an LLM query at fire time and appear as `[auto]` in
`list`. Recurring action work, such as "every weekday at 9 a.m. check
the build", becomes a scheduled task; ask the bot in plain language to
set, list, or cancel scheduled tasks.

```
@remind in 30 minutes check the build
@remind in 2 hours check status of CVE-2026-31431 in Debian
@remind list
@remind del abc1
@remind admin list someone        # owner only
```

See [reminders and usage](../user/reminders-usage.md) for caveats:
reminders count against the `@ask` rate limit, run with no elevated
capabilities at fire time, and recurring chains cap at 50 fires.

## Accounting

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@usage` | `[<nick> \| #channel]` | Show API usage statistics. The global overview by PM is admin-only. |

```
@usage
@usage someone
@usage #channel
```

## Verse commands (user)

All verse commands require the channel to have `verseEnabled` set and
the caller to hold the `llm.verse` capability, except `@avatar`.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@verseopt` | `<in \| out>` | Opt your avatar in or out of this channel's verse. |
| `@verse` | | Show your current scene in one line. |
| `@look` | `[<target>]` | Describe your scene or a named entity. |
| `@who` | | List active avatars and their locations. |
| `@avatar` | `[<persona> \| clear]` | Set the persona that shapes your verse avatar. Independent of `@instruct`. |

## Verse commands (editor)

These require the `llm.verse.edit` capability, granted globally.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@canon` | `<lock \| unlock \| forget> <name>` | Mark a character as durable canon, or release it. |
| `@versedit` | `[#channel] <verb> <args>` | Edit the verse universe directly. |

`@versedit` verbs: `add`, `pin`, `unpin`, `set`, `name`, `desc`,
`retire`, `restore`, `relate`, `unrelate`, `event`, `editevent`,
`delevent`, and `show`. Entity kinds are `avatar`, `npc`, `place`,
`faction`, and `item`. Refer to entities by `#id` or name.

```
@versedit add npc Headmaster Pringle :: stern keeper of the academy
@versedit pin #12
@versedit set #12 mood=grumpy
@versedit show #12
```

## Verse commands (GM)

These require the `llm.verse.gm` capability.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@versedump` | `[#channel] [--format=json]` | Dump the full verse state as JSON. |
| `@versepurge` | `[#channel] [<token>]` | Irreversibly wipe a channel's verse. Two-step token confirmation. |
| `@versecompact` | `<channel>` | Run retention compaction for the channel now. |

See [verse operations](../operator/forest-verse.md) for what compaction
does and how to operate the verse.

## Features at a glance

- **Natural language interaction**: mention the bot by name or send a
  PM to ask questions, manage memories, and set reminders without
  commands.
- **Volatile memory**: the bot remembers your recent conversation for
  follow-up questions. Context is per-user and per-channel, and expires
  after inactivity.
- **Non-volatile memory**: durable facts about you that persist across
  conversations.
- **Vision**: include image URLs in `@ask` messages and the bot
  describes or reasons about them.
- **Syntax-highlighted code**: `@code` responses arrive as HTTP links,
  keeping IRC clean.
- **[The verse](../operator/forest-verse.md)**: a per-channel world
  model with user-driven roleplay, avatars, and a persistent entity
  graph. Opt in with `@verseopt in`.
- **Multi-provider AI**: LiteLLM routes to OpenAI, Anthropic, Google
  Gemini, xAI, and Vertex AI models behind one interface.
