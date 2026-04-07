# Reminders & Usage

## remind

Set reminders using natural language. When the time comes, the bot delivers the reminder via PM.

**Usage:** `@remind <natural language time and message>`

### Examples

```
@remind in 30 minutes check the build
@remind at 5pm review the pull request
@remind tomorrow at 9am standup meeting
@remind in 2 hours take a break
```

The bot parses times like "in 30 minutes", "at 5pm", "tomorrow at 9am", and many other natural language formats.

### Listing reminders

```
@remind list
```

Shows your active reminders with their IDs and scheduled times.

### Cancelling reminders

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
