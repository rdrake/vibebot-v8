# Getting Started

Chat with large language models, generate code, create images, and more -- all from IRC.

## Talking to the bot

The easiest way to interact with the bot is to mention it by name in a channel or send it a private message:

```
<you>     VibeBot, what's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), Denali (North America),
          Kilimanjaro (Africa), Elbrus (Europe), Vinson (Antarctica),
          Puncak Jaya (Oceania).
```

The bot understands natural language for most things -- questions, reminders, memory management, scheduled tasks, and more:

```
<you>     VibeBot, remind me in 2 hours to check the build
<VibeBot> Reminder set: check the build (in 2 hours).

<you>     VibeBot, every weekday at 9 a.m. summarize the overnight CVE feed
<VibeBot> ⏰

<you>     VibeBot, what's my usage this month?
<VibeBot> You've made 47 requests this month, costing $0.12.
```

Anything that needs *tools at fire time* (search, fetch, code, image) becomes a recurring scheduled task. Plain echoes work the same way -- just ask. See [Reminders & Usage](reminders-usage.md) for the full picture.

You can also send a PM for the same behavior.

## Using commands

All commands use the `@` prefix. Commands are useful when you want predictable, direct behavior:

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

Use `@code` instead of `@ask` when you want code. The bot returns a syntax-highlighted HTTP link to keep IRC clean:

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
```

Natural language works too:

```
<you>     VibeBot, from now on explain things like I'm a senior developer
<VibeBot> Instruction set.
```

The bot also picks up facts from your conversations automatically -- such as your preferred language or what you're working on. View and manage them with `@memories` or just ask:

```
<you>     VibeBot, what do you remember about me?
<VibeBot> I have 3 memories: [1] Prefers Python | [2] Works on infrastructure | ...
```

See [Memory & Instructions](memory.md) for details.

## Account requirements

Some commands require an authenticated account. For example, `@draw` requires registration. The bot tells you if you need to log in first.

## Rate limits

Commands are rate-limited to prevent misuse. Your limits depend on your account status (unregistered, registered, or trusted). If you reach a limit, wait a moment and try again. Bot admins and owners are exempt.

## Permissions

Your bot operator controls who can use each command through Limnoria's capability system. If a command tells you that you lack permission, ask your channel operator to grant you access.

See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for details.

## Commands and natural language

You can interact with the bot two ways:

- **Natural language** -- Mention the bot or PM it. Good for conversational requests and combining multiple actions. ("VibeBot, delete memory 3 and remind me tomorrow to add it back")
- **Commands** -- Use `@` prefix commands for direct, predictable behavior. Good for specific operations. (`@memories delete 3`)

Both approaches have access to the same capabilities. Use whichever feels more natural for the task.

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
