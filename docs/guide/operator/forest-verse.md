# Forest-Verse

The forest-verse is a per-channel, structured world model — a SQLite entity
graph of avatars, places, and events — populated by user-driven `@ask`
roleplay through the avatar shim.

This page is the operator reference. For architecture and design rationale, see
`docs/plans/2026-05-07-forest-verse-design.md` in the repository.

## Enabling the verse for a channel

The verse is off by default. Enable it per channel with the `verseEnabled` registry key:

```
@config channel #yourchan plugins.LLM.verseEnabled True
@flush
```

Users then opt in individually with `@verseopt in`. The channel switch and user opt-in are independent: flipping `verseEnabled False` suspends the verse for everyone without discarding their avatars.

## Capabilities

Two capabilities gate verse access:

| Capability | Who needs it | What it unlocks |
|------------|-------------|-----------------|
| `llm.verse` | Regular users | `@verseopt in/out`, `@verse`, `@look`, `@who`, `@instruct` double-write |
| `llm.verse.gm` | Trusted operators | `@versedump`, `@versepurge`, `@versecompact` |

Grant capabilities the normal Limnoria way:

```
@capabilities add someone llm.verse
@capabilities add trusted-op llm.verse.gm
```

Without `llm.verse`, a user can still send `@ask` in a verse channel — their
messages simply do not route through the verse path.

## User commands

### `@verseopt in` / `@verseopt out`

`@verseopt in` opts the calling user into the verse for the current channel.
The bot runs a short starter scene, creates an avatar entity, and acknowledges
the new entry. Requires `llm.verse`.

`@verseopt out` retires the user's avatar in this channel. The avatar entity is
soft-deleted; event history is preserved. The user can opt back in later with a
fresh avatar.

### `@verse`

Emit a one-line scene summary for the current channel's verse — where the
action is, who is present, and the rough mood. Requires `llm.verse`.

### `@look [target]`

Describe a specific entity (avatar, place, or object) in the verse. Without a
target, describes the current location. Requires `llm.verse`.

```
@look
@look the tavern
@look rdrake
```

### `@who`

List active avatars in the current channel's verse. Requires `llm.verse`.

### `@instruct`

In a verse-enabled channel, `@instruct` double-writes: the instruction is saved
to the standard user instruction store *and* written to `avatar.summary` for
the user's verse avatar. This gives their character a persistent persona the
verse narrator uses when describing them.

```
@instruct You are a gruff cartographer who speaks in short declarative sentences.
```

Clearing the instruction with `@instruct clear` also clears the avatar summary.
Outside verse-enabled channels, `@instruct` behaves as normal.

### OOC escape

Wrap a message in double parentheses to bypass the verse path entirely:

```
((just testing the bot, ignore this))
```

The bot processes the message through the normal `@ask` path. Nothing is
recorded to the verse. Useful for meta questions or bot debugging without
breaking scene.

## Retention compaction

Once a day at `verseCompactionDailyAt` (global, default `"03:00"`
local time), the plugin walks every channel where `verseEnabled=True`
and replaces the **oldest 200** events past `verseEventRetentionDays`
(per-channel, default `30`) with a single lore-digest event. The
summary is produced by the same cheap model the loom uses
(`loomModel`), tagged `loom:compact` in `@usage`.

`verseCompactionMinKeepEvents` (global, default `20`) sets a floor:
verses with fewer than that many total events are skipped. This keeps
small verses from thrashing.

### Drain rate and backlog math

A single compaction pass touches at most **200 events** — a safety
cap so one model call cannot blow past the cheap model's context
window. Practical implications:

- A backlog of 10,000 events past the retention window converges in
  about **50 daily runs** (~50 days).
- A verse that produces **more than 200 events/day past its retention
  window** will not converge under the daily cap; the events table
  grows unboundedly. If you see this, lower
  `verseEventRetentionDays`, or run `@versecompact #channel`
  repeatedly to drain a backlog manually (each invocation processes
  another 200-event batch).
- Realistic verses (avatar-driven) produce on the order of 1-10
  events/day, so the cap rarely matters.

Failures are logged at WARNING and never block the timer; the next
day's run will retry.

## Owner commands

All commands require `llm.verse.gm`.

### `@versedump #chan`

Dump the full verse state for the specified channel as JSON. Output is sent to
the caller as a notice (may be long). Useful for debugging entity state or
backing up before a purge.

```
@versedump #afternet
```

YAML output is not supported in PR 1.

### `@versepurge #chan` (two-step confirmation)

Permanently delete all verse state for the specified channel: entities,
attributes, relations, events, and avatar links. **This is irreversible.**

Because the command is destructive, it uses a token-confirmation flow:

1. Run `@versepurge #chan` — the bot responds with a one-time token and a
   60-second expiry.
2. Confirm with `@versepurge #chan <token>` within 60 seconds.

If the token expires or you supply the wrong token, the purge is cancelled.
The token is single-use; a new `@versepurge` call generates a new token.

```
@versepurge #afternet
# bot: "Purge token: abc123 (expires in 60s). Confirm: @versepurge #afternet abc123"
@versepurge #afternet abc123
# bot: "Verse state for #afternet purged."
```

### `@versecompact #channel`

Manually run retention compaction for `#channel`. Useful for testing or
forcing a digest before the daily timer fires. Requires capability
`llm.verse.gm`. Reports the outcome (`compacted`, `skipped_no_events`,
`skipped_below_floor`, `skipped_disabled`).

## Loom orchestrator

The loom is a separate orchestrator that runs cheap-model cycles inside one
configured "venue" channel and digests the resulting improv into proposed
mutations against per-channel verses. By default the loom is **disabled**:
no scheduler event, no model calls, zero cost.

### Enabling

Set both `supybot.plugins.LLM.loomNetwork` and `supybot.plugins.LLM.loomChannel`.
The loom resolves the venue Irc via `world.getIrc(network)`; if either
setting is empty, or the network isn't connected, the loom stays inert.

```
config supybot.plugins.LLM.loomNetwork afternet
config supybot.plugins.LLM.loomChannel #forest
```

Verses opt in via the per-channel `verseEnabled` flag. The loom only
considers verses whose channel is *also joined on the loom network*.

### Source filter

`loomBotNicks` is a comma-separated allowlist. Empty means capture every
non-self line in the venue (the original design intent, suitable for the
bot-heavy channel the loom was built for). Set it to a strict list when
the venue mixes humans and bots:

```
config supybot.plugins.LLM.loomBotNicks botA,botB,botC
```

### Cycle anatomy

A cycle is `seed → 90 s listen → beat → 90 s listen → digest`. Three
cheap-model calls per non-idle cycle. Idle cycles short-circuit to one
call (seed); a cycle whose listen windows produce no transcript skips
both the beat and the digest.

### Proposal moderation

```
@verseproposals [#chan] [pending|approved|rejected] [<limit>]
@verseapprove <id> [#chan]
@versereject <id> [#chan]
```

Default channel = current; default status = `pending`; default limit = 3
(footer says `… more pending` when there are more). Pass an explicit
limit up to 50 to scroll the queue. Auto-applied proposals carry
`status='approved' reviewer='loom'` and appear under
`@verseproposals #chan approved`. `<id>` accepts unique-prefix matches.
Both `@verseapprove` and `@versereject` require `llm.verse.gm`.

### Cost transparency

Each loom call is logged in `@usage` tagged `loom:seed`, `loom:beat`, or
`loom:digest`. Until the Gemini cache plumbing lands in `service.py`,
projections assume zero cache hits.

### Tuning

| Knob                       | Bump up when                                       |
|----------------------------|----------------------------------------------------|
| `loomCycleInterval`        | The venue is overstimulated; cycles too frequent.  |
| `loomVerseCooldown`        | One verse dominates; force rotation.               |
| `loomBeatWindow`           | The bot reply cadence is slow; transcripts empty.  |
| `loomTranscriptMaxLines`   | Transcript truncation drops salient lines.         |
| `verseAutoApplyThreshold`  | Auto-apply approves too aggressively (raise it).   |

## Cross-pollination

Two verses can exchange seeds — short rumours that flow from one
channel's loom digest to another's pending-proposals queue, where the
receiving operator decides whether to canonise them. **Both ends must
opt in:** the source needs `verseCrosspollAllowSend=True`; the receiver
needs `verseCrosspollAllowReceive=True`. Defaults are `False` everywhere.

`verseCrosspollPerCycleLimit` (global, default `1`) caps how many seeds
a source verse's digest may emit per loom cycle. Seeds in excess are
dropped with a warning.

Receivers pull at most one seed per loom cycle, oldest first. A seed
becomes a pending `add_event` proposal in the receiver's verse; approve
or reject it with `@verseapprove` / `@versereject` as usual. Approved
seeds materialise as events with `source='crosspoll'`.

A verse cannot consume its own emissions.

## Registry keys

| Key | Scope | Type | Default | Purpose |
|-----|-------|------|---------|---------|
| `verseEnabled` | per-channel | bool | `False` | Master switch — enables the verse path and verse commands for the channel |
| `verseEventRetentionDays` | per-channel | int | `30` | Events older than this are eligible for retention compaction (oldest 200 collapse into a single lore-digest event per daily run) |
| `verseCrosspollAllowSend` | per-channel | bool | `False` | Allow this verse's loom digest to emit crosspoll seeds to other verses |
| `verseCrosspollAllowReceive` | per-channel | bool | `False` | Allow this verse to receive crosspoll seeds as pending `add_event` proposals |
| `verseCrosspollPerCycleLimit` | global | int | `1` | Maximum crosspoll seeds a source verse may emit per loom cycle |
| `verseCompactionDailyAt` | global | str | `"03:00"` | Local-time `HH:MM` for the daily retention-compaction sweep |
| `verseCompactionMinKeepEvents` | global | int | `20` | Floor on total events; verses below this count are skipped by compaction |
| `verseAutoApplyThreshold` | global | float | `0.85` | Minimum confidence at which loom proposals auto-apply without operator review (`add_entity` always queues) |
| `loomNetwork` | global | str | `""` | Network where the loom orchestrator runs. Empty = disabled |
| `loomChannel` | global | str | `""` | Channel where the loom orchestrator runs. Empty = disabled |
| `loomModel` | global | str | `gemini/gemini-flash-lite-latest` | Cheap model used by the loom for seed/beat/digest calls |
| `loomCycleInterval` | global | int | `5` | Loom timer cadence in minutes |
| `loomVerseCooldown` | global | int | `20` | Minimum gap in minutes between consecutive loom cycles for the same verse |
| `loomBeatWindow` | global | int | `90` | Listen window in seconds after each loom beat is posted |
| `loomTranscriptMaxLines` | global | int | `40` | Per-window cap on loom transcript lines (most recent kept) |
| `loomTranscriptMaxChars` | global | int | `8000` | Per-window cap on loom transcript characters (most recent kept) |
| `loomBotNicks` | global | str | `""` | Comma-separated allowlist of nicks captured into the loom transcript. Empty = capture all non-self lines |

Set from IRC:

```
@config channel #yourchan plugins.LLM.verseEnabled True
@config channel #yourchan plugins.LLM.verseEventRetentionDays 14
```

## What is NOT shipped yet

The following features are planned but not yet implemented:

- **Gemini cache plumbing** — `cached_tokens` accounting in `@usage`.
- **Web view at `/verse/<channel>`** — read-only HTML inspector.

## Migration / data note

There is no migration from the old forest-mode or spontaneous participation
features. Those registry keys and code paths have been removed. Users who were
in forest mode must opt in to the verse fresh with `@verseopt in`. No old state
carries over.
