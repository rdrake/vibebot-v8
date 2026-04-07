# Getting Started

VibeBot is an AI-powered IRC bot that lets you chat with large language models, generate code, create images, and more -- all from IRC.

## Your first conversation

All commands use the `@` prefix. Try asking a question:

```
<you>     @ask What's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), Denali (North America),
          Kilimanjaro (Africa), Elbrus (Europe), Vinson (Antarctica),
          Puncak Jaya (Oceania).
```

The bot remembers what you talked about, so follow-ups work naturally:

```
<you>     @ask Which of those can a beginner climb?
<VibeBot> Kilimanjaro is the most accessible -- no technical climbing, just
          a multi-day trek. Elbrus is also popular with guided groups...
```

Context is per user and per channel. It expires after a few minutes of inactivity, or you can clear it with `@forget`.

## Generating code

Use `@code` instead of `@ask` when you want code. The output is served as a syntax-highlighted HTTP link to keep IRC clean:

```
<you>     @code Python script to rename files by date
<VibeBot> https://bot.example.com/llm/abc123.html
<you>     @code Add a --dry-run flag
<VibeBot> https://bot.example.com/llm/def456.html
```

## Creating images

Generate images with `@draw` (requires an authenticated account):

```
<you>     @draw A fox reading a book in a cozy library
<VibeBot> https://bot.example.com/llm/img789.png
```

## Making it yours

Set persistent instructions to change how the bot talks to you:

```
@instruct Explain things like I'm a senior developer, skip the basics
@instruct You are Captain Picard. Respond in character.
```

The bot also picks up facts from your conversations automatically -- things like your preferred language or what you're working on. View and manage them with `@memories`.

See [Memory & Instructions](memory.md) for details.

## Account requirements

Some commands require you to be authenticated. For example, `@draw` requires a registered account. If you're not authenticated, the bot will let you know.

## Rate limits

Commands are rate-limited to prevent abuse. Your limits depend on your account status (unregistered, registered, or trusted). If you hit a limit, wait a moment and try again. Bot admins and owners are exempt.

## Permissions

Your bot operator controls who can use which commands via Limnoria's capability system. If a command tells you that you lack permission, ask your channel operator to grant you access.

See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for details.

## All commands

| Command | Description | Details |
|---------|-------------|---------|
| `@ask` | Ask the AI a question (supports vision and follow-ups) | [AI Commands](ai-commands.md#ask) |
| `@code` | Generate code with syntax-highlighted HTTP link | [AI Commands](ai-commands.md#code) |
| `@draw` | Generate an image from a text description | [AI Commands](ai-commands.md#draw) |
| `@forget` | Clear your conversation context | [Memory](memory.md#volatile-context) |
| `@memories` | Manage stored facts about you | [Memory](memory.md#non-volatile-memory) |
| `@instruct` | Set persistent instructions for `@ask` | [Memory](memory.md#custom-instructions) |
| `@remind` | Set reminders with natural language | [Reminders & Usage](reminders-usage.md#remind) |
| `@usage` | View API usage statistics | [Reminders & Usage](reminders-usage.md#usage) |
