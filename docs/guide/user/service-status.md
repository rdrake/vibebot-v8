# Service status

The bot watches a live status page — Claude's by default — and can answer
questions about it in conversation. Which page it watches, and whether a
channel gets announcements, are operator settings: see
[Status page](../operator/tuning-monitoring.md#status-page). With no status
page configured the bot has no status awareness at all and falls back to
whatever the chat model already believes, which is usually months out of date.

## Asking

Ask in plain language. There's no command.

```
<you>     VibeBot, is Claude down?
<VibeBot> Yeah, they're investigating elevated error rates on Opus 4.5.
          The API's showing degraded; everything else looks fine.
```

The bot reads the live page rather than answering from memory. It re-reads
every two minutes, and re-reads immediately if its last reading is more than
four minutes old, so an answer can trail the page by a few minutes. When the
page can't be reached, it says so and reports the last reading it has, rather
than guessing; before the first successful read it says only that it hasn't
read the page yet.

A scheduled check works too. `@remind in 2 hours check whether Claude is still
having problems` runs the same lookup at fire time — see
[Action reminders](reminders.md#action-reminders). Where an operator has enabled
conversational scheduling and you're logged in, "VibeBot, every weekday at 9
a.m. tell me if Claude is having problems" makes it recurring — see
[Recurring reminders and scheduled tasks](reminders.md#recurring-reminders-and-scheduled-tasks).

Roleplay replies can't reach the status page. Ask outside the story.

## Past incidents

Questions about the past pull a second reading — the recent incident list,
not just what's broken now:

```
<you>     VibeBot, has Claude been flaky lately?
<VibeBot> Three times this week: elevated API errors on Tuesday (about 40
          minutes), a Sonnet latency spike Thursday that ran two hours, and
          degraded search yesterday, cleared in 25 minutes.
```

Five incidents at most, newest first, each with its name, impact level, how
long ago it started, and how long it lasted. Update text and per-incident links
are left out, so the bot can't quote you a postmortem or link a single
incident — only the status page as a whole. An incident still open has no
duration to report.

That history is fetched only when the question calls for it, and cached for an
hour afterwards, so it can lag the live page by that much. If the fetch fails
the bot falls back to the last history it read — past that hour, with nothing
marking it stale — and says nothing about the past if it has none. Current
status still answers.

## Announcements

Where an operator has enabled it for the channel, the bot announces newly
opened incidents on its own:

```
<VibeBot> Claude status: Elevated error rates on the API (investigating) — https://status.claude.com
```

That template is the floor, not the usual output. Up to six times an hour the
bot restates the same facts in the channel's own voice instead; a rewrite is
thrown away and the template sent if it drops the service name or carries a
link to anywhere but the configured status page.

Only newly opened incidents are announced. Resolutions, status updates within
an incident, component-only changes, and scheduled maintenance are not.
Incidents already open when the bot starts count as already announced, so a
restart mid-outage doesn't replay one into the channel — and that incident
stays silent for the rest of its life. Ask the bot if you need its current
state.
