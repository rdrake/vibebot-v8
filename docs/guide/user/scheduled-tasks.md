# Scheduled tasks

The bot can run a future AI task on your behalf. At run time it runs a fresh `@ask`-style query as you, with full tool access: search, fetch, draw, code, and the Limnoria bridge. Use a scheduled task for jobs that need *tools* at run time, not a string echoed back.

There's no `@schedule` command. You create scheduled tasks conversationally: ask the bot, and its scheduler tool does the rest.

## Task or reminder

A [reminder](reminders-usage.md) fits when:

- You already know the exact text you want delivered later.
- You want a poke or a deadline, not a synthesis.

A scheduled task fits when:

- The bot needs to look something up at run time.
- The output should reflect the world as of the run, not as of when you scheduled it.
- You want the bot to do work, not echo a string.

Good candidates:

- `every Monday at 9am check my open PRs and tell me which are stale`
- `at 6pm summarize the day's #news traffic and post a one-paragraph digest`
- `every 2 hours check the build status and message me if anything's red`

## Scheduling in natural language

Schedule a task the same way you'd ask the bot anything else. Mention it by name, or send a PM:

```
VibeBot, every weekday at 9am check the changelog of <repo> and tell me what shipped
VibeBot, in 30 minutes check the weather and tell me whether to bring a jacket
```

The bot confirms in plain English, describing the schedule rather than the underlying tool call:

```
<VibeBot> I'll check the changelog every weekday at 9am and tell you what shipped.
```

Scheduling needs an authenticated network account.

## Listing and cancelling

Ask the bot:

```
VibeBot, what tasks do I have scheduled?
VibeBot, cancel the changelog one
VibeBot, cancel everything
```

Tasks and reminders share one list and one cancel path, so you can refer to a task by topic ("the changelog one") rather than by ID.

Bot owners can inspect and cancel another user's tasks and reminders with `@remind admin list <nick>` and `@remind admin del <nick> <id>`.

## Where the result lands

By default, the result lands where you scheduled it: a task scheduled in `#dev` posts back to `#dev`; a task scheduled by PM posts back by PM.

You can ask for a different target. Cross-channel delivery needs both you and the bot present in the destination channel, with the bridge enabled there. You can't redirect output to another user's private messages, only your own.

```
VibeBot, every morning at 8am summarize yesterday's #ops traffic and post it in #dev
VibeBot, every Monday at 9am check my PRs and message me directly
```

## What a task can do at run time

A scheduled task runs through the same chat profile as `@ask`, with the same tools:

- **Search** the web, when a search provider is configured.
- **Fetch** URLs.
- **Draw and code** with the image and code tools.
- **Limnoria bridge:** call read-only commands from allowlisted plugins (Time, Math, Seen, Web, Note, Karma, QuoteGrabs, RSS, DDG, and so on).
- **Memory and reminders:** save facts or set follow-up reminders on your behalf.

Each run counts against your normal `@ask` rate limit. The number of active schedules is capped separately per user.

## Recurring schedules

Both numeric and calendar recurrences work:

- `every 5 minutes`
- `every 2 hours`
- `every weekday at 9am`
- `every first of the month at noon`

A recurring task keeps running until you cancel it.

## When things go wrong

- **API key missing or invalid:** the task logs an error and skips that run; recurring tasks try again next time.
- **Tool error:** a failed search or fetch can still produce a partial reply explaining what went wrong.
- **Rate limit hit:** the run is dropped; recurring tasks try again next time.
- **Cancelled mid-run:** the current run completes; the next one doesn't fire.

The bot doesn't message you when a run errors out. If a recurring task goes quiet, ask the bot to list your tasks.

## Privacy

Scheduled tasks run as you. Their output respects your channel context and your `@instruct` text. A task can't read messages from a channel you're not in. A task that posts to a channel speaks over the bot's connection, so others see the bot speaking, not you.

Owners can inspect any user's tasks through `@remind admin list <nick>`. Treat scheduled tasks as you would any other automated traffic on a shared bot: useful, observable, and not strictly private.
