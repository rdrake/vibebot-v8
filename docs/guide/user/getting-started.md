# Getting Started

VibeBot is an AI-powered IRC bot that lets you chat with large language models, generate code, create images, and more -- all from IRC. It supports multiple AI providers (OpenAI, Anthropic, Google Gemini) through a unified interface.

## Command prefix

All commands use the `@` prefix:

```
@ask What's the weather like on Mars?
@code Python script to parse CSV files
@draw A cat wearing a top hat
```

## Account requirements

Some commands require you to be identified with NickServ. For example, `@draw` requires a registered account. If you're not identified, the bot will let you know.

## Permissions

Your bot operator controls who can use which commands via Limnoria's capability system. If a command tells you that you lack permission, ask your channel operator to grant you access.

See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for details on how capabilities work.

## Rate limits

Commands are rate-limited to prevent abuse. Your limits depend on your account status (unregistered, registered, or trusted). If you hit a limit, wait a moment and try again. Bot admins and owners are exempt.

## Commands

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
