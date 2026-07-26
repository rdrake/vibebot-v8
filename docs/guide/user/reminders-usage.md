# Reminders

Set reminders in natural language. When the time comes, the bot either echoes your text back or, if you asked it to *do* something, runs the task at fire time.

!!! tip "Natural language"
    Where an operator has enabled it, you can manage reminders by talking to the bot: "VibeBot, remind me in 2 hours to check the build" or "VibeBot, cancel the build reminder." The channel key is `supybot.plugins.LLM.pendingTasksEnabled`, and it's off by default. `@remind` always works.

## `remind`

**Usage:** `@remind <natural language time and message>`

```
@remind in 30 minutes check the build
@remind at 5pm review the pull request
@remind tomorrow at 9am standup meeting
```

The bot parses times such as "in 30 minutes", "at 5 p.m.", and "tomorrow at 9 a.m.".

## Action reminders

If your reminder asks the bot to *perform a task* (look something up, check a status, fetch a URL, summarize something), the bot runs the task as a fresh AI query when the timer fires. The query has the bot's full tool surface: web search, URL fetch, code, drawing, memory, and nested reminders.

```
@remind in 2 hours check the status of CVE-2026-31431 in Debian 12 and 13
@remind in 30 minutes check if https://example.com/build is green
@remind tomorrow at 9 a.m. summarize the top 3 HN headlines about Rust
```

Action reminders appear marked **`[auto]`** in `@remind list`, so you can tell them apart from plain echo reminders.

"Remind me to" phrasing, where *you* do the thing, stays a plain echo:

```
@remind in 5 minutes remind me to check the build   # echo only
@remind tomorrow at 3 p.m. call Bob                 # echo only
```

When in doubt, the bot prefers echo. If it misclassifies, rephrase.

## Recurring reminders and scheduled tasks

For repeating work, ask in plain English:

```
VibeBot, every 2 hours check the status of #1234 and ping me if it changed
VibeBot, every weekday at 9 a.m. summarize the overnight CVE feed
```

Anything that needs tools at fire time becomes a [scheduled task](scheduled-tasks.md). Plain echo reminders also recur: `@remind every Friday at 5 p.m. switch laundry over` schedules a chain of one-shot reminders that re-arm themselves, up to 50 times, before you have to set them again.

Scheduled tasks and reminders share one list and one cancel path. Ask:

```
VibeBot, what do I have scheduled?
VibeBot, cancel the laundry reminder
VibeBot, cancel everything
```

Things to know:

- **Action reminders count against your `@ask` rate limit.** If you're over the limit when one fires, the bot delivers your original text as a plain reminder with a note, and makes no API call.
- **No elevated capabilities at fire time.** Even if an owner or admin scheduled the action, it runs without those rights.
- **One nested reminder per fire.** An action reminder can schedule at most one follow-up during its run, which prevents fan-out.
- **Recurring chains cap at 50 fires.** After that, re-arm the reminder.
- **Scheduled tasks need an authenticated account.** Log in to your network account first.
- **Conversational scheduling is per channel.** The plain-English forms in this section need `pendingTasksEnabled` on that channel; `@remind` does not.

## Listing and cancelling

```
@remind list                # your reminders, with IDs ([auto] marks action reminders)
@remind delete abc1         # cancel one (del also works)
@remind delete abc1 def2    # cancel several
@remind clear               # cancel all your plain reminders
```

`@remind clear` only clears plain reminders. To cancel scheduled tasks too, ask the bot: "VibeBot, cancel everything."

## Delivery

The bot delivers a reminder to the channel where you set it, or by private message if you set it by PM. If the bot was offline when a reminder came due, it delivers the reminder shortly after it comes back.

## Owner admin

Bot owners can list, delete, and clear other users' reminders and scheduled tasks:

```
@remind admin list someone
@remind admin del someone abc1
@remind admin clear someone
```

Everyone else receives a permission error.
