# Reminders & Usage

!!! tip "Natural language"
    You can manage reminders and check usage by mentioning the bot or sending a PM. For example: "VibeBot, remind me in 2 hours to check the build" or "VibeBot, how much have I used this month?" Use the commands when you want direct control.

## remind

Set reminders using natural language. When the time comes, the bot delivers the reminder via PM.

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

### Listing reminders

```
@remind list
```

Shows your active reminders with their IDs and scheduled times.

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

### Delivery

Reminders are delivered via private message (PM) from the bot. Make sure you can receive PMs on the network.

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
