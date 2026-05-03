# Reminders & Usage

!!! tip "Natural language"
    You can manage reminders and check usage by mentioning the bot or sending a PM. For example: "VibeBot, remind me in 2 hours to check the build" or "VibeBot, how much have I used this month?" Use the commands when you want direct control.

## remind

Set reminders using natural language. When the time comes, the bot either echoes your text back or — if you asked it to *do* something — runs that as an LLM query at fire time.

**Usage:** `@remind <natural language time and message>`

### Examples

Via natural language:

```
<you>     VibeBot, remind me in 30 minutes to check the build
<VibeBot> Reminder set: check the build (in 30 minutes).

<you>     VibeBot, show my reminders
<VibeBot> You have 2 reminders: 1) abc123: check the build (in 28m) ...

<you>     VibeBot, cancel the build reminder
<VibeBot> Deleted reminder abc123.
```

Or with commands:

```
@remind in 30 minutes check the build
@remind at 5pm review the pull request
@remind tomorrow at 9am standup meeting
@remind in 2 hours take a break
```

The bot parses times like "in 30 minutes", "at 5 p.m.", "tomorrow at 9 a.m.", and many other natural language formats.

### Action reminders (LLM at fire time)

If your reminder asks the bot to *perform a task* — look something up, check a status, fetch a URL, summarize something — it runs the task as an LLM query when the timer fires, with the bot's full tool surface (web search, URL fetch, code, drawing, memory, nested reminders).

```
@remind in 2 hours check the status of CVE-2026-31431 in Debian 12 and 13
@remind in 30 minutes check if https://example.com/build is green
@remind tomorrow at 9 a.m. summarize the top 3 HN headlines about Rust
```

Action reminders are marked **`[auto]`** in `@remind list` so you can tell them apart from passive echo reminders.

**What stays passive:** "remind me to …" phrasing where *you* are doing the thing.

```
@remind in 5 minutes remind me to check the build   # echo only
@remind tomorrow at 3 p.m. call Bob                  # echo only
```

When in doubt, the bot prefers echo. If it misclassifies, just rephrase.

### Recurring reminders and scheduled tasks

For repeating work, ask in plain English:

```
VibeBot, every 2 hours check the status of #1234 and ping me if it changed
VibeBot, every weekday at 9 a.m. summarize the overnight CVE feed
```

Anything that needs *tools at fire time* (search, fetch, code, draw) becomes a **scheduled task**. Plain echo reminders also recur — `@remind every Friday at 5 p.m. switch laundry over` schedules a chain of one-shot reminders that re-arm themselves up to 50 times before you have to set them again.

Scheduled tasks and reminders share one list and one cancel path. Just ask:

```
VibeBot, what do I have scheduled?
VibeBot, cancel the laundry reminder
VibeBot, cancel everything
```

`@remind list`, `@remind delete <id>`, and `@remind clear` cover plain reminders; the natural-language path covers both.

**Caveats:**

- **Counts against your `@ask` rate limit.** If you're over the limit when an action or scheduled task fires, the bot delivers the original text as a plain reminder with a note (no API call is made).
- **No elevated capabilities at fire time.** Even if you scheduled the action as an owner or admin, it runs without those rights — owner-only tools are unavailable.
- **One nested reminder per fire.** An action reminder schedules at most one follow-up reminder during its run, to prevent fan-out. Scheduled tasks cannot schedule further scheduled tasks (depth cap of 1).
- **Recurring chains cap at 50 fires.** After that you have to re-arm.
- **Existing reminders don't upgrade.** Reminders set before action behavior shipped stay as plain echoes; delete and re-set them to get action behavior.
- **Scheduled tasks need an authenticated account.** The natural-language scheduler refuses anonymous callers — log in to your network account first.

### Listing reminders

```
@remind list
```

Shows your active reminders with their IDs. LLM-action reminders are marked `[auto]`. To see scheduled tasks alongside reminders, ask the bot in plain language ("what do I have scheduled?").

### Canceling reminders

**Cancel specific reminders:**

```
@remind delete abc1
@remind delete abc1 def2
```

**Cancel all your reminders:**

```
@remind clear
```

`@remind clear` only clears plain reminders. To cancel scheduled tasks too, ask the bot ("VibeBot, cancel everything").

### Owner-only admin

Bot owners can list, delete, and clear reminders and scheduled tasks belonging to other users:

```
@remind admin list someone
@remind admin del someone abc1
@remind admin clear someone
```

Non-owners receive a permission error.

### Delivery

Reminders are delivered to the channel where you set them, or via private message (PM) if you set the reminder via PM. For PM delivery, make sure you can receive PMs on the network.

---

## usage

View API usage statistics to see how much you and your channel have been using the bot.

**Usage:** `@usage [nick | #channel]`

### Natural language

```
<you>     VibeBot, how much have I used this month?
<VibeBot> You've made 47 requests this month, costing $0.12.

<you>     VibeBot, what's the channel usage?
<VibeBot> #dev has made 312 requests this month, costing $1.45.
```

### Your stats in a channel

Run `@usage` in a channel to see both your personal stats and the channel's stats for the current month:

```
@usage
```

### Another user's stats

```
@usage someone
```

### A channel's stats

```
@usage #somechannel
```
