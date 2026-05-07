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
| `llm.verse.gm` | Trusted operators | `@versedump`, `@versepurge` |

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

## Owner commands

Both commands require `llm.verse.gm`.

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

## Registry keys

| Key | Scope | Type | Default | Purpose |
|-----|-------|------|---------|---------|
| `verseEnabled` | per-channel | bool | `False` | Master switch — enables the verse path and verse commands for the channel |
| `verseEventRetentionDays` | per-channel | int | `30` | Reserved for retention compaction (PR 2). Currently unused — set it now if you want a value in place before compaction lands |

Set from IRC:

```
@config channel #yourchan plugins.LLM.verseEnabled True
@config channel #yourchan plugins.LLM.verseEventRetentionDays 14
```

## What is NOT in PR 1

The following features are planned but not yet implemented:

- **Loom orchestrator** — automated event scheduling and narrative arcs.
- **Proposal queue** — multi-user approval flow for world-state changes.
- **Retention compaction** — `verseEventRetentionDays`-driven pruning of old events.
- **Cross-channel pollination** — shared entities across channels.

These land in PR 2 and PR 3.

## Migration / data note

There is no migration from the old forest-mode or spontaneous participation
features. Those registry keys and code paths have been removed. Users who were
in forest mode must opt in to the verse fresh with `@verseopt in`. No old state
carries over.
