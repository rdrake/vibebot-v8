# The verse

The verse is a per-channel, persistent world model: a SQLite graph of avatars, places, factions, items, and events that grows out of roleplay in the channel. Opted-in users act through avatars; the bot narrates scenes, records events, and carries canon forward between sessions.

This page is the operator reference: enabling the verse, controlling when it fires, curating canon, and running the maintenance surface.

## Enabling the verse

The verse is off by default. Enable it per channel:

```
@config channel #yourchan plugins.LLM.verseEnabled True
```

Users then opt in individually with `@verseopt in`. The channel switch and the user opt-in are independent: flipping `verseEnabled` to `False` suspends the verse for everyone without discarding avatars or history.

### Capabilities

Three capabilities gate verse access:

| Capability | Who needs it | What it unlocks |
|------------|--------------|-----------------|
| `llm.verse` | Regular users | `@verseopt`, `@verse`, `@look`, `@who`, and verse-routed messages |
| `llm.verse.edit` | Canon editors | `@versedit`, `@canon`, and the model's `verse_edit` tool |
| `llm.verse.gm` | Game moderators (GMs) | `@versedump`, `@versepurge`, `@versecompact` |

Grant them the normal Limnoria way:

```
@capabilities add someone llm.verse
@capabilities add editor llm.verse.edit
```

The `llm.verse.edit` capability is checked globally against the caller's account. A channel-scoped grant fails closed: it denies rather than escalates. Grant trusted accounts globally.

Without `llm.verse`, a user can still talk to the bot in a verse channel; their messages never route through the verse path.

## When a message becomes a verse turn

A message from an opted-in avatar holder routes into the verse only when a verse signal is present. Two signals count:

- **Entity reference.** The message names a known active entity other than the speaker's own avatar. Matching is whole-word and alias-aware.
- **Trigger keyword.** The message matches `verseTriggerRegex`, a case-insensitive pattern with default `\bverse\b`. Set a channel-specific phrase to give users a deliberate entry cue. An empty value disables the keyword signal; a malformed pattern is ignored.

Messages without a signal fall through to normal chat. This keeps ordinary conversation from being narrated at.

### Going out of character

Users bypass the verse for a single message by wrapping it in double parentheses or starting it with `//`:

```
((just testing the bot, ignore this))
// what's the weather tomorrow?
```

The message takes the normal chat path and nothing is recorded to the verse. On verse-enabled channels the marker is stripped before the chat model sees the text.

## User commands

| Command | Effect |
|---------|--------|
| `@verseopt in` / `@verseopt out` | Join the verse with a fresh avatar, or retire it. History is preserved; opting back in creates a new avatar |
| `@verse` | One-line scene summary: where the action is and who is present |
| `@look [target]` | Describe the current location, or a named entity |
| `@who` | Roster of active avatars and their locations |
| `@avatar [persona\|clear]` | Set the persona that shapes your avatar. Independent of `@instruct`; affects only the verse |

`@avatar` stores the persona in the registry and mirrors it onto the avatar's summary when one is active, so `@look` reflects the change immediately.

## The store

Each channel's verse lives in its own SQLite database under `data/verse/`. The schema has nine tables; the ones an operator meets are:

- **entities**: avatars, non-player characters (NPCs), places, factions, and items, each with a kind, name, summary, and status.
- **attributes**: key-value details attached to entities.
- **relations**: directed, typed links between entities, with optional notes.
- **events**: the timeline. Each event links the entities involved, both as a JSON list and through the `event_actor` join table. Any tool that mutates events must handle both linkages.
- **entity_alias**: alternate names that entity matching resolves.
- **avatar_link**: maps IRC accounts and nicks to avatar entities.

### Canon, pinning, and aging

Entities the model creates during play are tagged auto-created. The daily sweep retires auto-created NPCs that go unmentioned for `verseAutoEntityRetireDays` days (default 14, `0` disables). Retirement is a soft status flip and is reversible.

Two marks exempt an entity from aging and pull it into every verse prompt:

- **Pinned** (`@versedit pin`): part of the always-on canon roster.
- **Author-locked** (`@canon lock`): durable canon claimed by an editor.

The canon roster block in the verse system prompt is capped at `verseRosterMaxChars` characters (default 4000). Pinning is what makes a character appear in every turn; unpinned entities surface contextually through events and relations that mention them.

### Member-driven worldbuilding

Opted-in members can narrate events involving entities other than their own avatar:

> vibebot, stinky dan threw a guff grenade at Andrew

The assistant calls the `verse_record` tool with `actors=["stinky dan", "Andrew"]`. Names that match existing entities link to them (avatar before NPC before item before place, case-insensitive, retired entities skipped). Unmatched names become auto-created NPCs. Items and props stay as prose inside the recorded summary; they are never auto-created as actors.

`verseAutoEntityMaxNamesPerCall` (default 8) caps the actors array. Raise it for verses with large casts; values past 16 invite entity flooding.

## Editing canon

### `@versedit`

`@versedit` hand-edits a channel's verse as operator canon. Writes are immediate and audit-logged with an operator source tag. It requires `llm.verse.edit`.

By default the edit applies to the channel where you run the command. A leading `#channel` token overrides that, which also lets you batch edits from a private message without flooding the channel:

```
/query yourbot
@versedit #afternet add npc Gurning Gary :: Supply teacher with a twitchy eye
@versedit #afternet pin Gurning Gary
```

A `<ref>` is either `#<id>` or an entity name. Summaries follow a `::` separator.

| Verb | Syntax | Effect |
|------|--------|--------|
| `add` | `add <kind> <name> [:: summary]` | Create an entity. Kinds: `avatar`, `npc`, `place`, `faction`, `item`. Rejects duplicate active names |
| `show` | `show <ref>` | Inspect id, kind, status, summary, and attributes |
| `desc` | `desc <ref> :: <summary>` | Replace the entity summary |
| `name` | `name <ref> <new-name>` | Rename. Rejects duplicate active names |
| `set` | `set <ref> <key> <value>` | Set an attribute |
| `pin` / `unpin` | `pin <ref>` | Add to or remove from the pinned roster |
| `retire` / `restore` | `retire <ref>` | Soft-delete or restore an entity |
| `relate` | `relate <ref> <kind> <ref> [:: note]` | Add a directed relation; replies with its id |
| `unrelate` | `unrelate <relation-id>` | Delete a relation |
| `event` | `event <summary> [@id,id,…]` | Add a timeline event, optionally linking entity ids |
| `editevent` | `editevent <event-id> :: <summary>` | Edit an event summary |
| `delevent` | `delevent <event-id>` | Delete an event |

### `@canon`

`@canon lock <name>` marks a character as author-locked canon; `@canon unlock <name>` releases it; `@canon forget <name>` clears the mark and lets normal aging apply. Requires `llm.verse.edit`.

### The model's `verse_edit` tool

Editors also get canon edits in-band: when a user holding `llm.verse.edit` triggers a verse turn, the model can call `verse_edit`. The tool is constructive-only: it can add entities, events, relations, and attributes, and update summaries, but it cannot delete, retire, pin, or rewrite history. Destructive verbs stay with `@versedit`. Authorization is computed per triggering user; for anyone else the tool refuses as a no-op.

## Style exemplars

`verseStyleExemplars` holds a short list of curated lines that show the model what good verse output looks like in this channel. The injector enforces a hard budget: at most 5 lines and 600 characters in total, with sanitization that drops lines mimicking prompt structure. Curate 3 or 4 short lines.

Populate the list offline with the taste miner, which scans channel logs for lines the resident author re-pasted or praised:

```bash
python -m llm.verse.taste_mine <logfiles...> --verse-dir data/verse --channel "#afternet" --out taste_candidates.md
```

Review the candidates by hand, then set the survivors as a JSON list on the channel key.

## Measurement

Two signals tell you whether verse output lands:

- **Reactions.** With `verseReactionCaptureEnabled` (default on), the bot records IRCv3 emoji reactions to its verse lines into `data/verse/reactions.jsonl`. Capture is measurement-only; the bot never replies to a reaction.
- **Landing-rate report.** The offline `taste_report` CLI computes how often verse output gets engagement, with channel log volume as the denominator, and folds in the reaction signal:

```bash
python -m llm.verse.taste_report <logfiles...> --verse-dir data/verse --channel "#afternet"
```

## Storybook

With `verseStorybookEnabled` on, verse turns gain a `verse_storybook` tool that renders an illustrated page and posts a link. An explicit request ("make a storybook of this", "illustrate that fight") forces the tool call, so explicit asks fire reliably.

The `@story <brief>` command generates the same illustrated page on request, in two prompt-inferred modes: a story mode for in-character tales and an explainer mode for accurate illustrated explanations. `@story` requires the `llm.draw` capability and an authenticated account, and it works outside verse mode.

Cost controls: `verseStorybookMaxImages` (default 5), `verseStorybookMaxPerTurn` (1), `verseStorybookCooldownSeconds` (300, per account, shared between the tool and `@story`), `verseStorybookDailyImageCap` (30), `verseStorybookMaxChars` (6000), and `verseStorybookImageTimeout` (45 seconds). Illustrations render concurrently; the page is published through the bot's HTTP output.

## GM operations

All three commands require `llm.verse.gm`.

### `@versedump [#channel] [--format=json]`

Dump the full verse state: entities with attributes, relations, the 200 most recent events, and avatar links. The dump publishes to the bot's HTTP pastebin and the reply carries only the URL; if the HTTP server is not configured, the JSON falls back inline. Useful for debugging and for a backup before a purge.

### `@versepurge [#channel] [token]`

Permanently delete all verse state for a channel: entities, attributes, relations, events, and avatar links. **This is irreversible**, so it uses two-step confirmation:

1. `@versepurge #chan` issues a one-time 6-character token valid for 60 seconds.
2. `@versepurge #chan <token>` performs the purge.

An expired or wrong token cancels the purge. Tokens are single-use and compared in constant time.

### `@versecompact <channel>`

Run retention compaction immediately instead of waiting for the daily timer. Reports the outcome; see the reference below.

## Compaction

Once a day at `verseCompactionDailyAt` (global, default `03:00` local time), the plugin walks every verse-enabled channel and replaces the oldest events past `verseEventRetentionDays` (channel, default 30) with a single lore-digest event written by `verseCompactionModel`. The digest is stamped at compaction time and lists up to 32 involved entities.

`verseCompactionMinKeepEvents` (global, default 20) sets a floor: verses with fewer total events are left alone, which keeps small verses from thrashing.

### Drain rate

A single pass compacts at most 200 events, a safety cap that keeps one model call inside the cheap model's context window. Consequences:

- A backlog of 10,000 over-retention events converges in about 50 daily runs.
- A verse producing more than 200 over-retention events per day never converges under the daily cap. Lower `verseEventRetentionDays`, or run `@versecompact` repeatedly; each run drains another 200-event batch.
- Realistic avatar-driven verses produce 1 to 10 events per day, so the cap rarely matters.

Failures log at WARNING and never block the timer; the next day's run retries.

### Outcome reference

| State | Meaning |
|-------|---------|
| `compacted` | Old events were summarized into one digest event |
| `skipped_disabled` | `verseEventRetentionDays` is 0 or lower; retention is off |
| `skipped_below_floor` | Total events are under `verseCompactionMinKeepEvents` |
| `skipped_no_events` | Nothing is older than the retention cutoff |

The compaction log line also reports aging: `aged N entities (kept M)` counts auto-created NPCs retired by the sweep and those scanned but kept.

## Quality guards

Verse output runs through guards that keep long-form roleplay from degrading:

- **Denial retry.** Some non-reasoning models refuse fictional premises. A detected refusal triggers one retry with a corrective nudge, and past refusals are stripped from the history each turn so they cannot become self-reinforcing.
- **Degradation detection.** Run-on or repetitive output (too few unique words, endless sentences) triggers one retry.
- **History window.** Verse history is trimmed to the 10 most recent messages, which limits self-imitation drift.

These run automatically; no configuration is required.

## Registry key reference

See the verse and storybook tables in [Configuration](configuration.md) for every key, scope, and default in one place.
