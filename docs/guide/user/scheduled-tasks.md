# Scheduled tasks

The bot can run a future LLM task on your behalf. At run time it runs an `@ask` invocation as you, with full tool access: search, fetch, draw, code, and the Limnoria bridge. Use this for jobs that need *tools* at run time, not just a string echoed back.

A scheduled task differs from a reminder. A reminder sends a literal text message that you wrote earlier. A scheduled task triggers a fresh LLM run that decides what to say from the live world: today's date, today's news, the current state of a feed, the current contents of a channel.

## When to schedule a task instead of a reminder

A reminder is the right pick when:

- The exact text you want delivered later is the text you already know now.
- You want a poke or a deadline, not a synthesis.
- Plain text is enough.

A scheduled task is the right pick when:

- The bot needs to look something up at run time.
- The output should reflect the world as of the run, not as of when you scheduled the task.
- You want the bot to do work, not just echo a string.

Examples that fit a task rather than a reminder:

- `every Monday at 9am check my open PRs and tell me which are stale`
- `at 6pm summarize the day's #news traffic and post a one-paragraph digest`
- `every 2 hours check the build status and message me if anything's red`
- `at 4pm draw a picture of whatever's on the front page of Hacker News`

## Asking the bot in natural language

Schedule a task the same way you'd ask the bot for anything else. Mention the bot by name in a channel, or send it a private message:

```
VibeBot, every weekday at 9am check the changelog of <repo> and tell me what shipped
VibeBot, in 30 minutes check the weather and remind me whether to bring a jacket
VibeBot, every 2 minutes check <feed-url> for new posts and message me when one shows up
```

The bot confirms in plain English. It describes the schedule, not the underlying tool call:

```
<VibeBot> I'll check the changelog every weekday at 9am and tell you what shipped.
```

The bot does not show a task ID, the underlying `@ask` syntax, or the natural-language schedule string it parsed. If you need to list or cancel something, ask in plain English again.

## Listing and canceling

Ask the bot:

```
VibeBot, what tasks do I have scheduled?
VibeBot, cancel the changelog one
VibeBot, cancel everything
```

The bot will list active tasks and reminders together, since they share the same list and cancel surface. You can refer to a task by topic ("the changelog one") rather than by ID.

Operators with `llm.admin` can also use the explicit `@remind admin list <nick>` and `@remind admin del <nick> <id>` forms to inspect and cancel another user's tasks and reminders.

## Where the result lands

By default, the result lands in the same place you scheduled it. A task scheduled in `#dev` posts back to `#dev`. A task scheduled in a private message posts back to that private message.

You can ask for a different target. Cross-channel delivery requires you and the bot to both be present in the destination channel, with the bridge enabled there. Cross-user private messages are not allowed; you can only redirect to your own private message.

```
VibeBot, every morning at 8am summarize yesterday's #ops traffic and post it in #dev
VibeBot, every Monday at 9am check my PRs and message me directly
```

## What the task can do when it runs

A scheduled task runs through the same chat profile as `@ask`, with the same tool surface:

- **Search:** the bot can run web searches when configured with a search provider.
- **Fetch:** the bot can fetch URLs through the URL fetch tool.
- **Draw and code:** `generate_image` and `generate_code` are available.
- **Limnoria bridge:** any read-only command from a loaded, allowlisted plugin (Time, Math, Seen, Web, Note, Karma, QuoteGrabs, RSS, DDG, and so on) is callable.
- **Memory and reminders:** the task can set follow-up reminders or remember facts on your behalf.

Each run counts against your normal `@ask` rate limit bucket. The schedule itself is rate-limited separately by `bridgeScheduledTaskLimit` (default 5 active tasks per channel per user).

## Recurring schedules

The bot accepts both numeric and calendar recurrences:

- `every 5 minutes`
- `every 2 hours`
- `every weekday at 9am`
- `every Monday at 9am`
- `every first of the month at noon`

A recurring task keeps running until you cancel it. There's no hard cap on the total number of runs, only the per-channel cap on active schedules.

## When things go wrong

A few things can go wrong at run time:

- **API key missing or invalid:** the task logs an error and skips that run. Recurring tasks try again on the next cadence.
- **Tool error:** if a tool call inside the task encounters an error (search timeout, fetch problem), the task can still produce a partial reply explaining what went wrong.
- **Rate limit hit:** if your `@ask` bucket is exhausted, the run is dropped. Recurring tasks try again on the next cadence.
- **Canceled mid-run:** a cancel during an in-flight run lets the current run complete. What stops is the next cadence.

The bot does not message you when a run errors out. If a recurring task goes quiet, ask the bot to list your tasks and check whether it's still active.

## Privacy

Scheduled tasks run as you. Their output respects your channel context and your `@instruct` text. A task scheduled in `#dev` cannot read messages from a channel you're not in. A task that posts to a channel does so over the bot's connection, not yours, so other users see the bot speaking and not you.

Operators can inspect any user's tasks through `@remind admin list <nick>`. Treat scheduled tasks the same way you'd treat any other automated traffic on a shared bot: useful, observable, and not strictly private.
