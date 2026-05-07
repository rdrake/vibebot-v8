# Forest-verse design

**Status:** Approved (brainstorm 2026-05-07)
**Author:** Richard Drake (with claude)
**Replaces:** Forest mode (`forestNicks`) and spontaneous mode (`spontaneousEnabled` and friends).

## Summary

Collapse two underused features — Forest mode and Spontaneous mode — into a
single coherent system: a **per-channel forest-verse**, populated by
user-controlled **avatars**, mutated by a separate **loom channel** in which
vibebot directs other AfterNET bots into chaotic improv that gets digested into
canonical world state.

```
                    ┌──────────────────┐
  Forest user  ───► │ Channel: #afnet  │ ◄── Verse-aware @ask
  (avatar)         │  (verse)         │     replies in-character
                    └────────┬─────────┘
                             │ persistent state
                    ┌────────▼─────────┐
                    │  Verse store     │  (SQLite entity graph
                    │  per channel     │   per channel)
                    └────────▲─────────┘
                             │ mutations
              ┌──────────────┴────────────────┐
              │  Loom orchestrator             │
              │  - rotates focus verse         │
              │  - runs multi-turn cycles      │
              │  - cheap model picks the       │
              │    plot prompt + digests       │
              │    other bots' replies         │
              └──────────────┬─────────────────┘
                             │ posts/reads via bridge
                    ┌────────▼─────────┐
                    │  Loom channel    │  e.g. #botspam
                    │  (other bots     │  (other bots reply
                    │  reply chaos)    │   with their nonsense)
                    └──────────────────┘
```

## Goals

- Make Forest mode interesting: a living shared fiction per channel rather
  than a long-reply toggle.
- Repurpose spontaneous mode as the engine that mutates that fiction, run in
  a separate loom channel populated by other bots — cheap model, real chaos.
- Keep cost predictable: at most ~3 cheap-model calls per loom cycle, with a
  cacheable prompt prefix that hits Gemini's prompt cache on calls 2 and 3.
- Preserve what was good about old Forest: bypassed line cap, `@instruct`-as-
  persona, opt-in scoping per channel.

## Non-goals

- Combat, inventory, XP, or any RPG mechanics — see existing `plugins/rpg`
  for that. The verse is social/narrative, not mechanical.
- Federated multi-bot protocols. The loom channel uses *whatever bots
  happen to be there*; we do not coordinate with their operators.
- A global cross-channel verse. (Considered and rejected: tonal collisions
  and privacy. Cross-pollination is a controlled escape valve instead.)

## Architecture

Three components, all under `plugins/llm/src/llm/verse/`:

1. **Verse store (`store.py`)** — SQLite entity graph per channel.
2. **Avatar shim (`avatar.py`)** — wraps `@ask` for opted-in users so replies
   are rendered through their avatar; exposes verse tools to the model.
3. **Loom orchestrator (`loom.py`)** — timer-driven; runs multi-turn cycles
   in the configured loom channel and digests the transcript into mutations.

### Verse data model

One SQLite file per channel: `data/verse/<sanitized-channel>.db`.

| Table | Columns | Notes |
|---|---|---|
| `entities` | `id`, `kind`, `name`, `summary`, `status`, `created_at`, `updated_at` | `kind` ∈ `avatar`, `npc`, `place`, `faction`, `item`. `status` ∈ `active`, `retired`. |
| `attributes` | `entity_id`, `key`, `value` | KV bag per entity. Lets the model add fields without migrations. |
| `relations` | `from_id`, `to_id`, `kind`, `note` | Directed edges: `lives_in`, `allied_with`, `hates`, `carries`, etc. |
| `events` | `id`, `ts`, `summary`, `entity_ids` (JSON), `source` | Append-only chronicle. `source` ∈ `avatar`, `loom`, `crosspoll`. |
| `avatar_link` | `entity_id`, `nick`, `account` | Maps verse avatar → IRC user. Unique per channel. |

Why both an `attributes` table and a JSON column on `events`: attributes stay
queryable for tool calls (`SELECT * FROM attributes WHERE key='location' AND
value='woods'`); events are inert chronicle records, JSON is fine.

**Avatar lifecycle**

- `@verseopt in` → entity row created with `kind='avatar'`, name from nick,
  summary derived from the user's `@instruct` text.
- `@avatar persona <text>` → updates summary and the user's `@instruct`.
- `@verseopt out` → soft-delete (status flag). Events keep referencing the
  entity by id; the avatar simply isn't "present" any more.

**Retention**

Events older than `verseEventRetentionDays` (default 30) are summarized by
the loom into a single "lore digest" event with `source='loom'`, originals
deleted. Keeps the table bounded, preserves continuity.

**Cross-pollination**

Each loom cycle has a `verseCrosspollChance` (default 25%) chance of
producing a one-line "seed" that gets inserted into a *different* random
verse's `events` with `source='crosspoll'` and no entity links. The
receiving verse treats it as inspirational background, not canonical state.

### Forest user `@ask` flow

When an opted-in user runs `@ask` in a verse-enabled channel:

```
@ask  ──►  forest path?  ──Yes──►  load avatar + scene context
                                          │
                                          ▼
                              build verse system prompt (cacheable):
                              - "You are <avatar.name>, persona: <@instruct>"
                              - "Scene: <avatar's current location summary>"
                              - "Recent events involving you (last 5): ..."
                              - "Other avatars present: ..."
                                          │
                                          ▼
                              expose verse tools alongside chat tools:
                              - verse_look(target?)        → describe entity
                              - verse_recall(query)        → RAG over events
                              - verse_act(verb, target?)   → records an event,
                                                              returns scene shift
                              - verse_move(place_name)     → updates avatar location
                                          │
                                          ▼
                              run completion with assistantModel
                              (the "advanced" model) — line cap bypassed
                                          │
                                          ▼
                              if model called verse_act/move,
                              mutations applied AFTER reply rendered
```

`verse_act` is the canonical "player did a thing" path. The model picks a
verb (`whispers to`, `searches`, `flees`), an optional target, and free-text
details. The shim writes an `events` row, optionally adjusts attributes
(verbs in a small known set auto-update location), returns "what happens
next" for the model to narrate. This keeps user actions on rails — they get
plausible-character actions, not free-form world rewrites. Full god-mode
mutations require capability `llm.verse.gm`.

**Out-of-character escape:** prefixing a message with `((...))` skips the
verse path entirely — runs as plain `@ask`. Tabletop convention; keeps the
bot usable for non-fiction questions without forcing opt-out.

**Persona:** the user's `@instruct` text is the avatar's `summary` and the
persona overlay. Channel-level `assistantSystemPrompt` is bypassed (same as
old Forest).

**Line cap bypass:** preserved from old Forest. Long replies still go
through the HTML link path via `longReplyLineThreshold`.

**Commands** (capability `llm.verse`):

- `@verseopt in|out` — self-managed opt-in
- `@verse` — current scene (1-line)
- `@look [target]` — describe an entity
- `@who` — avatars present in the channel's verse
- `@avatar persona <text>` — alias for `@instruct` that also updates the avatar summary

**Owner commands:**

- `@versedump #chan` — JSON dump for inspection
- `@versepurge #chan` — wipe (confirmation gated)

### Loom orchestrator

A single `asyncio` task in the plugin spins on a timer
(`loomCycleInterval`, default 5 min). When it fires:

1. **Pick focus verse.** Round-robin through enabled verses, weighted by
   `(active_avatars * 2 + recent_events)`. Skip verses whose last cycle was
   within `loomVerseCooldown` (default 20 min). Chatty verses get more
   attention; quiet ones still get turns.
2. **Build seed prompt.** Cheap model receives focus verse summary (top
   entities, last 10 events, current avatar locations) plus the last ~20
   lines of loom-channel traffic. Instruction: "Post a short opening line
   that invites the bots in this channel to riff. ≤ 1 line."
3. **Emit beat 1.** Vibebot posts that line in the loom channel.
4. **Listen window 1.** For `loomBeatWindow` seconds (default 90), record
   every other-bot reply into a transcript buffer. Vibebot ignores its own
   lines and human lines.
5. **Emit beat 2.** Cheap model reads transcript so far + verse state, posts
   a follow-up reacting to one bot's nonsense.
6. **Listen window 2.** Another window of replies (default 60s).
7. **Digest.** Cheap model receives the full transcript + verse summary,
   returns a structured JSON list of mutations:

   ```json
   {
     "mutations": [
       {"op": "add_event",      "summary": "...", "entity_ids": [3, 7]},
       {"op": "set_attribute",  "entity_id": 3, "key": "mood",     "value": "paranoid"},
       {"op": "add_relation",   "from_id": 3, "to_id": 7, "kind": "suspects"},
       {"op": "add_entity",     "kind": "npc", "name": "...", "summary": "..."}
     ],
     "crosspoll_seed": "optional one-liner to send to a neighbor verse, or null"
   }
   ```

8. **Apply.** Mutations validated against the verse store schema; bad
   entries dropped with a log line. `crosspoll_seed`, if present, is added
   to a random other verse's `events` (gated by `verseCrosspollChance`).

**Failure handling:** any step's exception aborts the cycle, logs at
WARNING, advances rotation pointer so the next cycle picks a different
verse. No partial mutations.

### Prompt caching strategy (loom)

The cheap loom model is **Gemini Flash Lite** (default
`gemini/gemini-flash-lite-latest`; operators should pin a specific version
for stable cache hits). Each cycle makes three calls (seed, beat 2,
digest), and we want calls 2 and 3 to hit Gemini's prompt cache.

Prompt structure for every loom call, in order:

1. **Static prefix (highly cacheable):** role description, mutation schema,
   format instructions, `PASS` contract. ~600 tokens, identical across all
   cycles.
2. **Verse-stable block (cacheable within cycle):** focus verse summary,
   active entities, last N events. ~400 tokens, identical across the three
   calls of one cycle.
3. **Volatile tail (uncached):** the loom-channel transcript so far, the
   per-call instruction ("emit seed" / "emit beat 2" / "digest").
   ~200–600 tokens, different on every call.

Calls 2 and 3 within a cycle should report `cached_tokens` covering the
first two blocks. Existing `service.py` already logs `cached_tokens` per
call, so we can verify with `@usage`.

If we end up swapping providers, the same layout still helps: Anthropic's
explicit `cache_control` markers go after blocks 1 and 2; xAI's automatic
cache picks up the static prefix the same way.

## Configuration

**Removed registry keys** (with a one-release deprecation warning if
present, then deletion):

- `spontaneousEnabled`, `spontaneousChance`, `spontaneousCooldown`,
  `spontaneousSystemPrompt`, `spontaneousModel` (if any)
- `forestNicks` (auto-migrated: every nick becomes a verse opt-in row on
  first plugin load)

**New registry keys:**

| Key | Scope | Default | Purpose |
|---|---|---|---|
| `verseEnabled` | per-channel bool | `False` | Turn verse on for the channel |
| `verseEventRetentionDays` | per-channel int | `30` | Event log compaction window |
| `verseCrosspollChance` | global int 0–100 | `25` | % chance a digest crosspollinates a neighbor verse |
| `loomChannel` | global string | `""` | Channel name where the loom runs (e.g. `#botspam`). Empty = loom disabled. |
| `loomModel` | global string | `gemini/gemini-flash-lite-latest` | Cheap model for orchestrator |
| `loomCycleInterval` | global int (minutes) | `5` | How often the timer fires |
| `loomVerseCooldown` | global int (minutes) | `20` | Minimum gap between cycles for the same verse |
| `loomBeatWindow` | global int (seconds) | `90` | Listen window after each beat |

**Capabilities:**

- `llm.verse` — required for `@verseopt in`, all `@verse*` commands, and the
  verse-aware `@ask` flow.
- `llm.verse.gm` — required for arbitrary mutations beyond the `verse_act`
  verb whitelist.

## Code layout

```
plugins/llm/src/llm/
  verse/
    __init__.py
    store.py          # SQLite entity graph
    avatar.py         # forest @ask shim, verse tools
    loom.py           # orchestrator, cycle, digest
    schema.sql        # tables
    tests/
      test_store.py
      test_avatar.py
      test_loom.py
```

## Tests

Existing `plugins/llm/tests/` conventions apply. Real SQLite (the
`feedback_wait_for_docker.md` rule about not mocking infrastructure also
applies to data-layer tests — use real DBs in tmp dirs, not mocks).

- `test_spontaneous.py` is renamed and rewritten as `test_loom.py`.
- New `test_verse_store.py`: schema migration, CRUD, retention compaction,
  soft-delete semantics.
- New `test_avatar.py`: opt-in flow, verse tool calls, OOC escape, persona
  derivation, line-cap bypass, capability gates.
- `test_loom.py`: rotation weighting, cooldown enforcement, beat/listen
  cycle (mock the IRC bridge but run real cheap-model digestion against a
  recorded transcript fixture), digest JSON validation, crosspoll
  application.

## Rollout

Three PRs, each independently reviewable.

1. **PR 1 — Verse store + avatar shim.** Schema, CRUD, `@verseopt`,
   `@verse`, `@look`, `@who`, `@avatar`, verse-aware `@ask` rendering, kill
   old `forestNicks` (auto-migrate). No loom — verses sit there, mutated
   only by `verse_act`. Ship; let people opt in and try avatars.
2. **PR 2 — Loom orchestrator.** `loom.py`, cycle timer, beat/digest
   pipeline, `loomChannel` registry. Kill old spontaneous mode keys.
   Operator sets `loomChannel` to test in a bot-heavy channel.
3. **PR 3 — Crosspollination + retention compaction.** Smaller polish PR
   once cycles are demonstrably working.

## Docs

- `docs/guide/operator/forest-mode.md` and
  `docs/guide/operator/spontaneous.md` collapse into a new
  `docs/guide/operator/forest-verse.md`.
- `CHANGELOG.md` entry called out as breaking (registry keys removed).
- `docs/guide/reference/commands.md` updated with the `@verse*` family.

## Open follow-ups (not blocking)

- Web view: `web/` already exists. A read-only verse dashboard (entities,
  recent events, current scene per channel) would be a nice second-screen
  for forest users. Out of scope for the three rollout PRs.
- Avatar art: `generate_image` tool already in the chat surface; a
  one-shot `@avatar portrait` command could render the avatar. Trivial
  later add.
- Operator dashboard for loom-cycle inspection: useful when tuning
  `loomBeatWindow` etc. Defer until we have real cycles to look at.
