# Getting started

Chat with AI models, generate code, create images, and set reminders, all from IRC.

## Talking to the bot

Mention the bot by name in a channel, or send it a private message. The bot notices its name even in the middle of a sentence:

```
<you>     VibeBot, what's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), Denali (North America),
          Kilimanjaro (Africa), Elbrus (Europe), Vinson (Antarctica),
          Puncak Jaya (Oceania).

<you>     ok but which of those can you climb without a guide, VibeBot?
<VibeBot> Kilimanjaro is the most accessible: no technical climbing, only
          a multi-day trek...
```

Natural language covers most features: questions, reminders, memory management, and scheduled tasks.

```
<you>     VibeBot, remind me in 2 hours to check the build
<VibeBot> Reminder set: check the build (in 2 hours).

<you>     VibeBot, what's my usage this month?
<VibeBot> You've made 47 requests this month, costing $0.12.
```

## Using commands

Every command starts with `@`. Commands give you direct, predictable behaviour:

```
<you>     @ask What's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), ...
```

The bot remembers your recent conversation, so follow-ups work naturally:

```
<you>     @ask Which of those can a beginner climb?
<VibeBot> Kilimanjaro is the most accessible...
```

Context is per user and per channel. It expires after a few minutes of inactivity, or you can clear it with `@forget`.

Both paths reach the same features. Natural language suits conversational requests and combined actions ("VibeBot, delete memory 3 and remind me tomorrow to add it back"). Commands suit precise operations (`@memories delete 3`).

## Accounts and permissions

Some commands, such as `@draw` and `@story`, need you to be authenticated with your network account. The bot tells you when a command needs authentication.

!!! note
    AfterNet uses network account authentication, not a NickServ service. Log in to your account through your usual method (for example, SASL) before using account-gated commands.

Your bot operator controls access to each command through Limnoria's capability system. If the bot says you lack permission, ask your operator for access. See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for background.

## Rate limits

Commands carry rate limits to prevent misuse. Your limits depend on your account status: unregistered, registered, or trusted. If you reach a limit, wait a moment and try again. Bot admins and owners are exempt.

## The verse

Channels with the verse enabled host a persistent, collaborative fiction: a shared world where you play as an avatar. Join with `@verseopt in`, give your avatar a voice with `@avatar <persona>`, and check your surroundings with `@verse`, `@look`, and `@who`.

Mention anyone or anything in the world and the bot answers with a tale. Use `@rp <text>` to act as your avatar for one turn, or `@rp on` to stay in character until `@rp off`.

To speak out of character in a verse channel, start your message with `//` or wrap it in `((...))`.

See [AI commands](ai-commands.md#the-verse) for the verse command list.

## All commands

| Command | Description | Details |
|---------|-------------|---------|
| `@ask` | Ask the AI a question (supports vision and follow-ups) | [AI commands](ai-commands.md#ask) |
| `@code` | Generate code, delivered as a highlighted web page | [AI commands](ai-commands.md#code) |
| `@draw` | Generate an image from a text description | [AI commands](ai-commands.md#draw) |
| `@story` | Generate an illustrated story or explainer page | [AI commands](ai-commands.md#story) |
| `@forget` | Clear your conversation context | [Memory](memory.md#conversation-context) |
| `@memories` | Manage stored facts about you | [Memory](memory.md#saved-memories) |
| `@instruct` | Set persistent instructions for `@ask` | [Memory](memory.md#custom-instructions) |
| `@avatar` | Set your verse avatar's persona | [Memory](memory.md#avatar-persona) |
| `@remind` | Set reminders with natural language | [Reminders](reminders-usage.md#remind) |
| `@usage` | View API usage statistics | [AI commands](ai-commands.md#usage) |
| `@verseopt` | Opt your avatar in or out of the verse | [AI commands](ai-commands.md#the-verse) |
| `@rp` | Speak or act as your verse avatar | [AI commands](ai-commands.md#the-verse) |

## Getting help

- `@help <command>` shows a command's built-in help.
- This guide lives at the address the bot shares when you ask it for help.
