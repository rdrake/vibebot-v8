# Service status

The bot watches a live status page — Claude's by default — and can answer
questions about it in conversation.

## Asking

Ask in plain language. There's no command.

```
<you>     VibeBot, is Claude down?
<VibeBot> Yeah, they're investigating elevated error rates on Opus 4.5.
          The API's showing degraded; everything else looks fine.
```

The bot reads the live page rather than answering from memory. When the page
can't be reached, it says so and reports the last reading it has, rather than
guessing.

## Announcements

Where an operator has enabled it for the channel, the bot announces newly
opened incidents on its own:

```
<VibeBot> Claude status: Elevated error rates on the API (investigating) — https://status.claude.com
```

Only newly opened incidents are announced. Resolutions, status updates within
an incident, component-only changes, and scheduled maintenance are not.
