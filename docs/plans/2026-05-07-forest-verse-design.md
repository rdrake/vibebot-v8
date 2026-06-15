# Forest-verse design

**Status:** Approved (brainstorm 2026-05-07, revised 2026-05-07 post-review)
**Author:** Richard Drake (with claude)
**Replaces:** Forest mode (`forestNicks`), Spontaneous mode (`spontaneousEnabled`
and friends), and **the entire `plugins/rpg/` plugin**.
**Data migration:** none. All three replaced features have their state
discarded on rollout.

## Revisions

- **2026-05-07 v1** — initial design.
- **2026-05-07 v2** — applied feedback from code-reviewer + Codex adversarial
  reviews. Strategic objections (loom-channel social risk, no adoption
  hypothesis) withdrawn given Forest is the eponymous user and owns the loom
  venue. Technical blockers addressed inline. Scope expanded to subsume
  `plugins/rpg/`.
- **2026-05-07 v3** — drop all data migration. Clean break for rpg,
  forestNicks, and spontaneous state. Operator just deletes the old plugin
  and old registry keys; users `@verseopt in` fresh.

## Summary

Collapse three features into one:

- Old **Forest mode** (`forestNicks`, long-reply opt-in, personal `@instruct`
  overlay).
- Old **Spontaneous mode** (`spontaneousEnabled`, dice-roll channel chatter).
- The whole **`plugins/rpg/`** plugin (per-channel characters, room graph,
  navigation, narrator).

Replacement: a **per-channel forest-verse** — a SQLite entity graph of
characters, places, and events — populated by user-controlled **avatars**
and mutated by a **loom orchestrator** that uses a cheap Gemini Flash Lite
model to direct other AfterNET bots in a separate channel into improv that
gets digested into *proposed* world mutations. Proposals above a confidence
threshold auto-apply; the rest sit in a moderation queue.

```
                    ┌──────────────────┐
  Forest user  ───► │ Channel: #afnet  │ ◄── Verse-aware @ask
  (avatar)         │  (verse)         │     replies in-character
                    └────────┬─────────┘
                             │ persistent state
                    ┌────────▼─────────┐
                    │  Verse store     │  (SQLite entity graph
                    │  per channel     │   per channel; thread-local
                    │  + proposals     │   conn + WAL + write lock)
                    └────────▲─────────┘
                             │ proposals → moderation → mutations
              ┌──────────────┴────────────────┐
              │  Loom orchestrator             │
              │  - rotates focus verse         │
              │  - runs multi-turn cycles      │
              │  - cheap model proposes,       │
              │    confidence-gated commit     │
              └──────────────┬─────────────────┘
                             │ posts/reads via bridge
                    ┌────────▼─────────┐
                    │  Loom channel    │  Forest's bot-heavy channel
                    │  (other bots     │  (vibebot is welcome there;
                    │  reply chaos)    │   venue owner is the feature
                    └──────────────────┘   namesake)
```

## Goals

- Make Forest mode interesting: a living shared fiction per channel rather
  than a long-reply toggle.
- Repurpose spontaneous mode as the engine that mutates that fiction, run in
  Forest's bot-heavy channel — cheap model, real chaos, owner-consented venue.
- **Subsume `plugins/rpg/`.** Verse becomes the canonical "structured world"
  primitive in this codebase. Combat/XP/inventory are explicitly dropped (not
  reimplemented in v1). Old rpg state is **discarded, not migrated**.
- Keep cost predictable and transparent.
- Preserve the spirit of old Forest: bypassed line cap, `@instruct`-as-
  persona, opt-in scoping per channel. (Old `forestNicks` rosters are
  discarded; users opt in fresh.)

## Non-goals (v1)

- Data migration of any kind. Old rpg characters, old forestNicks rosters,
  old spontaneous chat history — all discarded on rollout.
- Combat, inventory, XP, currency. The verse is social/narrative. If there's
  later demand, these can return as `verse_act` verb extensions, not as a
  parallel plugin.
- Federated multi-bot protocols. The loom uses *whatever bots happen to be in
  Forest's channel*; we don't coordinate with their operators (we don't need
  to — Forest is the channel owner).
- A global cross-channel verse. Cross-pollination is **opt-in per channel**
  and defaults off (see §"Cross-pollination").
- Stock-Limnoria primitive duplication. Verse uses
  `supybot.schedule.addEvent` for the loom timer, mirroring the rest of the
  plugin. No fresh asyncio tasks.

## Architecture

Three components, all under `plugins/llm/src/llm/verse/`:

1. **Verse store (`store.py`)** — SQLite entity graph per channel, plus a
   proposals table for unapplied loom output. Thread-local connection +
   WAL, mirroring the existing pattern in
   `plugins/llm/src/llm/persistence.py`.
2. **Avatar shim (`avatar.py`)** — wraps `@ask` for opted-in users, exposes
   a fixed verb-whitelist tool surface.
3. **Loom orchestrator (`loom.py`)** — `schedule.addEvent`-driven; runs
   multi-turn cycles in the loom channel; emits *proposals*, not direct
   mutations.

### Verse data model

One SQLite file per channel: `data/verse/<key>.db` where `<key>` is
`<lowercased-channel>` with non-`[a-z0-9_-]` replaced by `_`, suffixed with
the first 8 hex chars of a SHA-256 of the *original* channel name to
disambiguate case-insensitive collisions. Example: `#Foo` →
`_foo_a3b1c9d4.db`, `#foo` → `_foo_2c26b46b.db`.

| Table | Columns | Notes |
|---|---|---|
| `schema_version` | `version`, `applied_at` | Single row. Applied migrations table follows. |
| `entities` | `id`, `kind`, `name`, `summary`, `status`, `created_at`, `updated_at` | `kind` ∈ `avatar`, `npc`, `place`, `faction`, `item`. `status` ∈ `active`, `retired`. |
| `attributes` | `entity_id`, `key`, `value` | KV bag per entity. Lets the model add fields without migrations. |
| `relations` | `from_id`, `to_id`, `kind`, `note` | Directed edges: `lives_in`, `allied_with`, `hates`, `carries`, etc. |
| `events` | `id`, `ts`, `summary`, `entity_ids` (JSON), `source` | Append-only chronicle. `source` ∈ `avatar`, `loom`, `crosspoll`. |
| `avatar_link` | `entity_id`, `nick`, `account` | Maps verse avatar → IRC user. Unique per channel. |
| `proposals` | `id`, `created_at`, `cycle_id`, `op`, `payload` (JSON), `confidence` (real 0–1), `provenance` (text), `status`, `reviewer`, `reviewed_at` | Loom output; only commits when status flips to `approved`. |

**Concurrency.** Single SQLite file per channel. Connection is **thread-local**,
opened lazily, with `PRAGMA journal_mode=WAL` and `PRAGMA
foreign_keys=ON`. All writes go through a `with store.write_transaction(channel):`
context manager that takes a per-channel `threading.Lock`, mirroring the
existing pattern in `plugins/llm/src/llm/persistence.py`. Reads are
lock-free under WAL. Limnoria runs commands on its scheduler thread; the
loom runs on `supybot.schedule.addEvent` callbacks (also on that thread);
no asyncio.

**Avatar lifecycle**

- `@verseopt in` → entity row created with `kind='avatar'`, name from nick,
  summary derived from the user's `@instruct` text. Plus an opt-in starter
  scene (one `place` entity created if no places exist, avatar moved there,
  the bot replies with a one-paragraph scene description and the verb
  whitelist as a hint).
- `@instruct <text>` → updates the user's `@instruct` AND the avatar's
  summary atomically. **`@instruct` is the single writer** for avatar
  persona; `@avatar persona` is removed.
- `@verseopt out` → soft-delete (status flag). Events keep referencing the
  entity by id; the avatar simply isn't "present" any more.

**Retention.** Events older than `verseEventRetentionDays` (default 30) are
summarized into a single "lore digest" event with `source='loom'`,
originals deleted. Compaction runs as its own scheduled job
(`schedule.addEvent`, daily) so it works even when `loomChannel=""`.

**Cross-pollination** — see §"Cross-pollination".

### Forest user `@ask` flow

When an opted-in user runs `@ask` in a verse-enabled channel:

```
@ask  ──►  forest path?  ──Yes──►  load avatar + scene context
                                          │
                                          ▼
                              build verse system prompt:
                              - "You are <avatar.name>, persona: <@instruct>"
                              - "Scene: <avatar's current location summary>"
                              - "Recent events involving you (last 5): ..."
                              - "Other avatars present: ..."
                                          │
                                          ▼
                              expose verse tools alongside chat tools:
                              - verse_look(target?)              → describe entity
                              - verse_recall(query)              → RAG over events
                              - verse_act(verb, target?, details?) → records an event;
                                                                    side-effects per
                                                                    whitelist below
                              - verse_move(place_name)           → updates avatar location
                                          │
                                          ▼
                              run completion with assistantModel
                              (the "advanced" model) — line cap bypassed
                                          │
                                          ▼
                              if model called verse_act/move,
                              mutations applied AFTER reply rendered
                              (failed mutations logged + reply still sent)
```

**Verb whitelist for `verse_act`.** Exactly these verbs have side-effects
beyond writing an event row:

| Verb | Side effect |
|---|---|
| `whisper`, `speak`, `listen`, `examine`, `wait`, `signal`, `gesture` | Event only. No state change. |
| `move`, `flee`, `follow` | Updates avatar's `location` attribute to target if target is a `place` or another avatar's current `place`. |
| `take`, `drop`, `give` | Records an event linking avatar to the named item. **Does not** create new `item` entities — only references existing ones. (No inventory ledger in v1.) |
| `search` | Event only. Re-renders the scene on the next reply (`verse_look` is more efficient if that's all you want). |

Any verb the model invents that's *not* in this list is recorded as an
event with no side-effects. Free-form mutations beyond this surface require
capability `llm.verse.gm`.

**OOC escape.** Prefixing a message with `((...))` skips the verse path —
runs as plain `@ask` with the channel's normal persona overlay. OOC applies
*only* to the verse-aware `@ask` path; commands (anything starting with
`@`) are dispatched normally regardless.

**Capability fallthrough.** When `verseEnabled=True` but the calling user
lacks `llm.verse`, `@ask` falls through to the regular chat path
gracefully. No error message, no warning. Users without verse capability
see the channel as a normal chat channel.

**Persona.** The user's `@instruct` text is the avatar's `summary` and the
persona overlay. Channel-level `assistantSystemPrompt` is bypassed.

**Line cap bypass.** Preserved from old Forest in spirit. Long replies
still go through the HTML link path via `longReplyLineThreshold`.

**Commands** (capability `llm.verse`):

- `@verseopt in|out` — self-managed opt-in
- `@verse` — current scene (1-line)
- `@look [target]` — describe an entity
- `@who` — avatars present in the channel's verse

**Owner commands** (capability `llm.verse.gm`):

- `@versedump #chan [--format=json|yaml]` — full state dump
- `@versepurge #chan` — wipe; two-step confirmation: command issues a
  six-character token; second invocation `@versepurge #chan <token>` within
  60s commits.
- `@verseproposals #chan [--status=pending|approved|rejected]` — list
  proposals.
- `@verseapprove <id>` / `@versereject <id>` — moderation actions for
  pending proposals.

### Loom orchestrator

Driven by `supybot.schedule.addEvent` (not asyncio), at `loomCycleInterval`
(default 5 min). When it fires:

1. **Pick focus verse.** Among verses whose last cycle was outside
   `loomVerseCooldown` (default 20 min), weighted by
   `(active_avatars * 2 + recent_events)`. Round-robin is the tiebreaker.
2. **Idle short-circuit.** If no eligible verse, advance pointer and exit.
   No model calls.
3. **Build seed prompt** (cheap model). Inputs:
   - Static prefix (~600 tokens, identical across all cycles): role,
     proposal-schema, instruction format.
   - Verse-stable block (~400 tokens, identical across the cycle's three
     calls): focus verse summary, top entities, last 10 events
     **excluding** events with `source='crosspoll'`.
   - Volatile tail: last ~20 lines of loom-channel traffic, "emit seed"
     instruction.
   - Output: a single line ≤ 1 IRC line that invites the bots in that
     channel to riff.
4. **Emit beat 1.** Vibebot posts that line in the loom channel.
5. **Listen window 1.** For `loomBeatWindow` seconds (default 90), record
   replies from other participants. Vibebot always ignores its own lines.
   Human lines are filtered out **only** when `loomBotNicks` is set to a bot
   allowlist; when `loomBotNicks` is empty the loom records EVERY other
   participant (humans included), so the loom channel must be bot-only. The
   bot logs a WARN at wiring time when `loomChannel` is set with an empty
   `loomBotNicks` to flag this.
   Truncate to `loomTranscriptMaxLines` (40) and `loomTranscriptMaxChars`
   (8000); per-source-nick dedupe (drop consecutive identical lines).
   **If transcript is empty after the window**, skip beats 2 and digest;
   log `loom_idle`; advance rotation.
6. **Emit beat 2.** Cheap model reads transcript-so-far + verse summary,
   posts a follow-up.
7. **Listen window 2.** Another window, same caps and dedupe.
8. **Digest** (cheap model). Inputs: full transcript + verse summary +
   proposal-schema. Output: a JSON list of **proposals** — not direct
   mutations. Each proposal has:

   ```json
   {
     "id": "<uuid>",
     "op": "add_event" | "set_attribute" | "add_relation" | "add_entity",
     "payload": { ... op-specific fields ... },
     "confidence": 0.0,
     "provenance": "transcript-line-<n>: \"<other-bot> said: ...\"",
     "rationale": "one short sentence"
   }
   ```

9. **Apply / queue.** For each proposal:
   - Validate against schema. Bad → drop, log warning.
   - If `confidence >= verseAutoApplyThreshold` (default 0.85) AND
     `op != add_entity`: insert into `events` / `attributes` / `relations`
     directly with `source='loom'`.
   - Otherwise: insert into `proposals` table with `status='pending'`. The
     bot owner sees these via `@verseproposals`.
   - `add_entity` always goes through the proposal queue regardless of
     confidence. (Creating new canon people/places is high-leverage and
     deserves human review.)

**Cost ceiling per cycle:** at most 3 cheap-model calls (seed, beat 2,
digest). With idle short-circuit, silent cycles drop to 1 call (seed). With
loom disabled (`loomChannel=""`), zero calls — the timer doesn't run.

**Failure handling.** Any step's exception aborts the cycle, logs at
WARNING, advances rotation pointer. No partial mutations (proposals are
inserted atomically per cycle).

### Cross-pollination

**Default off.** Two per-channel settings:

| Setting | Default | Purpose |
|---|---|---|
| `verseCrosspollAllowSend` | `False` | Loom may emit a `crosspoll_seed` from this verse's digest. |
| `verseCrosspollAllowReceive` | `False` | This verse may receive `crosspoll_seed` events from other verses. |

When both ends are `True` and a digest produces a seed:
- The seed is itself a proposal (goes through the `proposals` table with
  `op='add_event'`, `source='crosspoll'`).
- The seed is **excluded from the focus-verse seed prompt's recent-events
  window** (see step 3) so verses can't recursively riff on each other.
- A separate `verseCrosspollPerCycleLimit` (default 1) caps fan-out per
  cycle.

This prevents the "rejected global verse" anti-pattern from sneaking back
in: nothing crosses without both venue operators consenting.

### Prompt structure for caching (loom)

Default loom model: `gemini/gemini-flash-lite-latest`. Operators should
pin a specific revision for stable cache behavior.

Each loom call uses three prompt blocks in order:

1. **Static prefix** (~600 tokens, identical across all cycles): role
   description, proposal schema, format instructions.
2. **Verse-stable block** (~400 tokens, identical across the cycle's three
   calls): focus verse summary, active entities, recent events.
3. **Volatile tail** (~200–600 tokens, different per call): transcript +
   per-call instruction.

**Honest cost case.** As of this design's writing, `service.py` has cache
plumbing only for xAI (`x-grok-conv-id`) and observation for Anthropic
(`cache_read_input_tokens`); no Gemini cache wiring exists. We design the
prompts to be cache-friendly *if and when* Gemini caching is plumbed
through LiteLLM, but the **cost projections in this doc assume zero cache
hits**. Plumbing Gemini caching is a follow-up (see §"Open follow-ups").

`@usage` accounting tags loom calls as `loom:seed`, `loom:beat`,
`loom:digest` so per-cycle cost is auditable.

## Subsuming `plugins/rpg/`

`plugins/rpg/` is **deleted outright** — no data migration, no compatibility
shim. Existing rpg characters, rooms, items, XP, gold, combat logs are all
discarded. Any rpg user who wants to continue starts over by running
`@verseopt in` in a verse-enabled channel.

The conceptual lineage survives — verse `entities` of kind `avatar` and
`place` cover the same ground rpg's `Character` and `Room` covered, but
without combat, dice, or stats. If anyone asks for those mechanics back
later, they'd return as `verse_act` verb extensions, not as a separate
plugin.

The implementation borrows rpg's persistence patterns (thread-local
connection, WAL, schema-version table) but *not* its data.

## Configuration

**Removed registry keys** (deleted on rollout, no deprecation window):

- `forestNicks` — superseded by avatar opt-in.
- `spontaneousEnabled`, `spontaneousChance`, `spontaneousCooldown`,
  `spontaneousSystemPrompt` — superseded by loom.
- All `plugins.RPG.*` keys — plugin removed.

**New registry keys:**

| Key | Scope | Default | Purpose |
|---|---|---|---|
| `verseEnabled` | per-channel bool | `False` | Turn verse on for the channel |
| `verseEventRetentionDays` | per-channel int | `30` | Event log compaction window |
| `verseAutoApplyThreshold` | global float 0–1 | `0.85` | Min confidence for auto-applied loom proposals |
| `verseCrosspollAllowSend` | per-channel bool | `False` | Verse may emit crosspoll seeds |
| `verseCrosspollAllowReceive` | per-channel bool | `False` | Verse accepts crosspoll seeds |
| `verseCrosspollPerCycleLimit` | global int | `1` | Max crosspoll seeds emitted per cycle |
| `loomChannel` | global string | `""` | Channel where the loom runs. Empty = loom disabled (timer not scheduled). |
| `loomModel` | global string | `gemini/gemini-flash-lite-latest` | Cheap model for orchestrator |
| `loomCycleInterval` | global int (minutes) | `5` | Timer cadence |
| `loomVerseCooldown` | global int (minutes) | `20` | Minimum gap between cycles for the same verse |
| `loomBeatWindow` | global int (seconds) | `90` | Listen window after each beat |
| `loomTranscriptMaxLines` | global int | `40` | Per-window transcript truncation cap |
| `loomTranscriptMaxChars` | global int | `8000` | Per-window transcript truncation cap |

**Capabilities:**

- `llm.verse` — required for `@verseopt in`, all `@verse*` commands, the
  verse-aware `@ask` flow, and `@look`/`@who`.
- `llm.verse.gm` — required for `@verseapprove`/`@versereject`,
  `@versedump`, `@versepurge`, and any free-form mutation beyond the
  `verse_act` whitelist.

## Code layout

```
plugins/llm/src/llm/
  verse/
    __init__.py
    store.py          # SQLite entity graph + proposals; thread-local conn + WAL
    avatar.py         # forest @ask shim, verse tools, verb whitelist
    loom.py           # orchestrator (schedule.addEvent), cycle, digest, proposal apply
    schema.sql        # tables + schema_version
    tests/
      test_store.py
      test_avatar.py
      test_loom.py
```

`plugins/rpg/` is **deleted** in the same PR that introduces the verse
store (PR 1).

## Tests

Existing `plugins/llm/tests/` conventions apply. Real SQLite (the
`feedback_wait_for_docker.md` rule about not mocking infrastructure also
applies to data-layer tests — use real DBs in tmp dirs, not mocks).

- `test_spontaneous.py` is renamed and rewritten as `test_loom.py`. Uses
  recorded transcript fixtures (vcr-style) for the loom model — **CI does
  not need provider keys**. Live model calls are gated behind a
  `VIBEBOT_TEST_LIVE=1` env flag and excluded from default `make test`.
- `test_verse_store.py`: schema migration, CRUD, retention compaction,
  soft-delete semantics, WAL + thread-local connection behavior under
  concurrent writes (real threads, real `threading.Lock`).
- `test_avatar.py`: opt-in flow + starter scene, verse tool calls, OOC
  escape, persona derivation through `@instruct`, line-cap bypass,
  capability fallthrough behavior, verb whitelist (each whitelisted verb's
  side effect; one off-list verb's no-side-effect behavior).
- `test_loom.py`: rotation weighting, cooldown enforcement, idle
  short-circuit, beat/listen cycle (mock the IRC bridge but run real
  digestion against fixtures), proposal validation, auto-apply threshold,
  crosspoll opt-in gating, exclusion of crosspoll events from seed prompt.

The rpg plugin's tests in `plugins/rpg/tests/` are deleted along with the
plugin.

## Rollout

Three PRs.

1. **PR 1 — Verse store + avatar shim + rpg removal.** Schema, CRUD,
   `@verseopt`, `@verse`, `@look`, `@who`, verse-aware `@ask` rendering
   with the verb whitelist + capability fallthrough + OOC escape, opt-in
   starter scene. **Deletes `plugins/rpg/` and removes `forestNicks` /
   spontaneous registry keys** in the same commit. No loom yet — verses
   sit there, mutated only by `verse_act`. Ship; let people opt in.
   CHANGELOG entry calls out the breaking removals (rpg plugin gone, old
   forest/spontaneous state discarded; users opt in fresh).
2. **PR 2 — Loom orchestrator + proposal queue.** `loom.py`,
   `schedule.addEvent` timer, beat/digest pipeline, proposal table,
   auto-apply threshold, `@verseproposals`/`@verseapprove`/`@versereject`,
   `loomChannel` registry. Operator points `loomChannel` at Forest's bot
   venue.
3. **PR 3 — Crosspollination + retention compaction.** Polish PR once
   cycles are demonstrably working. Both crosspoll flags default off.

## Docs

- `docs/guide/operator/forest-mode.md` and
  `docs/guide/operator/spontaneous.md` collapse into a new
  `docs/guide/operator/forest-verse.md`.
- The rpg plugin's docs are removed.
- `CHANGELOG.md`: breaking entries for rpg removal, `forestNicks`
  removal, spontaneous registry removal — all called out as no-migration.
- `docs/guide/reference/commands.md`: rpg commands removed; `@verse*`
  family added.

## Open follow-ups (not blocking PRs 1–3)

- **Gemini cache plumbing in `service.py`.** LiteLLM has hooks for
  `cached_content` / explicit-cache calls; once we have one real cache hit
  through the loom path, log it via `cached_tokens` and update the cost
  projection. Until then, treat caching as nominal-only.
- **Web view at `/verse/<channel>`.** Read-only dashboard: entities,
  recent events, current scene, pending proposals. The `web/` dir already
  exists; this is a thin extension.
- **Avatar art via `generate_image`.** `@avatar portrait` command. Trivial
  later add.
- **Loom-cycle inspection dashboard.** Useful when tuning beat windows.
  Defer until cycles are demonstrably running.
- **Combat / inventory return as verse mechanics.** Only if anyone asks
  after rpg removal lands. Would extend `verse_act` verbs and add an
  optional `inventory` attribute convention. Not v1.
- **Embedding-based `verse_recall`.** PR 1 ships substring matching.
