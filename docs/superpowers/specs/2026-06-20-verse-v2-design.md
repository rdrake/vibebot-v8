# Verse Retention Fix — Design (v1, minimal in-place)

Status: Approved scope — ready for implementation planning
Date: 2026-06-20
Author: rdrake (with Claude)
Supersedes: the over-scoped "new plugin + platform" draft, cut after a 12-lens +
Codex red-team found ~half of it was speculative infrastructure with no
user-visible value (see §10). This is the re-scoped, verified design.

## 1. Goal

Fix the one thing that actually limits verse today: **the model forgets canon.**
Deliver it as a **minimal, in-place change** to the existing `llm` verse path,
behind the existing per-channel flag, with registry-flip rollback — days of work,
against real #afternet data, no new plugin and no data migration.

### Success criteria
1. The model reliably carries forward the author's named roster and the
   scene-relevant cast/relations — **measured on the model's REPLY**, not on
   context overlap (see §8). The true-forget rate (a roster member that should be
   in scene but is absent from the injected context) does not increase.
2. fc42 makes canon stick **by talking**, with no operator step — gated on the
   capability he actually holds (`llm.verse.edit`).
3. Stories keep working (existing single-pass storybook), and an illustrated turn
   still records canon.
4. chat/code/draw are untouched.

## 2. Why (verified findings)

Deep analysis + a code-grounded red-team confirmed the root cause and several
mis-groundings the original draft had baked in:

- **Forgetting is the root cause.** Durable canon the model sees per turn is only
  a ~600-char char-capped pinned roster + ~5 avatar events; **relations are
  persisted but never read into any prompt**; unpinned/aged-out entities are
  invisible. This is fc42's two-month complaint; his value proposition is canon
  retention of a named roster, and he benchmarks verse ~5:1 against raw grok.
- **Verified against code:**
  - `llm.verse.gm` is the *operator* capability (dump/pin/merge/compact,
    `plugin.py:5905`); fc42 holds `llm.verse.edit`. `verse_record`
    (`store.py:599`) carries no caller authorization, unlike `verse_edit`
    (`plugin.py:2876`).
  - The storybook **daily image cap is an unimplemented `TODO`** (`plugin.py:2823`);
    only a per-*account* cooldown + per-turn cap bound spend.
  - Verse logic is `route_profile == PROFILE_VERSE` branches *inside* the shared
    completion method (`service.py:3810/3854/4041/4324`) — not a cleanly
    separable module.
  - `frequency_penalty` is set with `drop_params=True` (`service.py:3906-3909`);
    xAI silently drops it, so on grok **only `temperature` survives** — the cheap
    anti-run-on lever is a placebo and retry-on-degrade is the real defense.
  - There is **no alias/nickname concept** in the store (`find_entity_by_name`
    matches canonical name only) — so "the stinky lads" / "Tobes" would not match.

## 3. Scope

### In v1 (this spec)
1. **Retrieval into the existing prompt builder** — read the author-locked roster
   *in full* + scene-matched cast + 1-hop relations + active-only recent events
   into `build_verse_system_prompt`, under a token budget (§7).
2. **`author_locked` + promotion-on-reinforcement** — lock canon on
   reinforcement/correction of **human-offered** names, never on first mention and
   never on model-invented names (§6).
3. **Alias matching** — an `entity_alias` table, queried in retrieval, seeded from
   observed nicknames (§7).
4. **`event_actor` join table** — the one structural schema fix worth doing now:
   replaces the `entity_ids` JSON blob, enables SQL-side active-only filtering,
   kills the dead-lore Python scan (§7, §9).
5. **Cache byte-freeze layout** — freeze the verse *system* message; move per-turn
   scene context to a user-role message after it (§5).
6. **Surgical generation fixes in place** — kill the silent
   `verseModel→reasoning-model` fallback; raise/remove `verseRosterMaxChars`;
   keep `drop_params` (§9).
7. **Keep the single-pass storybook**, ensure illustrated turns still record
   canon, and state the real cost guards honestly (§9).

### Explicit non-goals for v1 (deferred — see §11)
New `Verse` plugin; gen-core extraction; two-stage story fan-out; the full
normalized schema (`entity_state`, `event_archive`, merge apparatus,
proposal reaper); one-time data port; formal shadow/A-B harness; portraits;
model-tool-surface trim (retiring move/look has UX fallout — Codex P3).

## 4. Architecture

No new plugin. The retention fix lives where the data already is:
- `verse/store.py` — add `author_locked`, `entity_alias`, `event_actor`; add
  retrieval queries (scene-match incl. aliases, 1-hop relations, SQL active-only
  events) and the promotion logic.
- `verse/avatar.py::build_verse_system_prompt` — assemble the frozen system block
  (roster only) and the separate per-turn user-role scene block.
- `service.py` — the surgical generation fixes; thread the caller's capability
  into the record path; ensure illustrated turns record canon.
- Gate everything behind the existing per-channel verse flag.

This keeps the verse package's clean internal seam without paying the
cross-plugin/reload/config-migration cost a second plugin would add.

## 5. Cache-aware prompt layout

The repo uses xAI automatic **prefix** caching (breaks at the first differing
byte), so a "small volatile block" buried in the system message buys nothing. Fix:

- **Frozen verse system message** = framework + channel `assistantSystemPrompt`
  overlay + deterministically-ordered (`kind, name COLLATE NOCASE`) author-locked
  roster **only**. Render only fields that change on explicit author action —
  **never `last_seen_ts`, heartbeats, day-granular date, or build-SHA** in this
  block, or every turn busts the prefix.
- **Per-turn scene context** (scene-matched cast, relations, recent events) → a
  separate **user-role** message after the frozen prefix, mirroring the existing
  `_build_topic_message` / `_build_speaker_message` pattern.
- **Verify provider prefix-cache behavior before relying on it for cost** — this
  is a v1 prerequisite, not a post-hoc check.

## 6. author_locked + promotion-on-reinforcement

- **Invariant:** `author_locked` = pinned (always injected) + aging-exempt +
  loom-protected. One flag, three protections. (Affirmed by the red-team — only
  the trigger needed fixing.)
- **Trigger (the fix):** do NOT auto-lock on first mention. Promote an entity to
  `author_locked` when a **human author** (`llm.verse.edit`) **reinforces** it —
  re-referenced across N turns, explicitly corrected, or explicitly affirmed
  ("remember X"). Mirrors the codebase's `memory_candidates → memories`
  two-stage promotion.
- **Human-offered only:** promotion candidacy is scoped to actor names that appear
  in the **triggering user's message** (incl. alias match), never to names the
  model invented in its own narration. First-mention/model NPCs stay ordinary
  `auto_created` — retrievable in-scene, still reapable, preserving the settled
  `npc + auto_created` aging/reactivate boundary.
- **Capability:** thread the triggering user's `msg.prefix` into the record path
  (it currently is not) and gate promotion on `llm.verse.edit`. A cutover
  preflight asserts fc42's hostmask-authed cap resolves; a test exercises his real
  hostmask-authed account.
- **`@canon`** (human command, invisible to Grok) remains the explicit override
  (lock/unlock/forget/correct). Reversible, so a stray promotion is cheap.

## 7. Retrieval (the fix), schema, and budget

Per turn, fill a bounded budget in priority order — a SELECT, not a platform:
1. **Author-locked roster, in full** (from the frozen system block).
2. **Scene-referenced entities** — case-insensitive match of the incoming message
   + recent scene lines against entity **name and `entity_alias`**.
3. **One-hop relations** for everyone in (1)+(2), with the related entity's
   one-line summary (relations read into the prompt for the first time).
4. **Recent events** for the in-scene cast via `event_actor`, **active-only in
   SQL** (no Python full-scan), newest first.

No similarity search / embeddings — plain SQL over names, the alias table, the
relation graph, and recency.

**Schema changes (minimal):**
- `entity.author_locked` (bool) and a human-reinforcement counter (promotion).
- `entity_alias(entity_id, alias COLLATE NOCASE)`, `PRIMARY KEY(entity_id, alias)`.
- `event_actor(event_id, entity_id)`, `PRIMARY KEY(event_id, entity_id)`,
  both FKs `ON DELETE CASCADE`, indexed both ways; backfilled from `entity_ids`
  with an **element-wise** tolerant decoder (keep valid ids, drop only bad
  elements — never all-or-nothing).
- **Per-channel SQLite DB stays** (no `world_id`); migration is idempotent and
  in-place (additive columns/tables, backfill), not a port to a new store.

## 8. Measurement (honest, lightweight)

The original A/B metric ("did injected canon include the entities the turn
referenced?") was circular and scored the wrong population. Replace with:
- **Grade recall on the REPLY:** did the output correctly carry forward
  names/relations/traits for in-scene entities, **including roster members not
  named in the input** (fc42's actual complaint)?
- **True-forget counter:** roster member that should be in scene but is absent
  from the injected context — must not increase.
- **Gate on fc42's live benchmark:** lightweight injected-vs-referenced +
  recall-on-reply logging in the existing store; his documented ~5:1 head-to-head
  on real scenes is the real cutover signal. No separate shadow/A-B subsystem.

## 9. Generation fixes & stories (in place)

- **Kill the silent `verseModel→assistantModel` fallback.** Set an explicit
  known-good **non-reasoning** default; on a missing/invalid key **hard-fail the
  verse turn** with a clear operator message — never fall to a reasoning model.
  Startup validation alerts if the verse key resolves to a reasoning model on a
  verse-enabled channel.
- **Sampling reality:** state that on grok only `temperature` survives
  (`frequency_penalty` is dropped); retry-on-degrade is the only structural
  defense — budget its extra-generation cost; one-time loud log when a verse
  sampling param is dropped for the active provider. Keep `drop_params` (the
  shipped xAI fix — do not regress).
- **Roster cap:** raise/remove `verseRosterMaxChars` so a ~15-NPC locked roster is
  never truncated.
- **The "re-seed rejected reply before retry" item is NOT assumed a bug** — the
  re-seed gives the nudge a referent and the bad text never reaches the channel.
  Characterize current behavior in a test and A/B whether dropping it helps on
  grok **before** changing it.
- **Stories:** keep the single-pass storybook. **Ensure an illustrated turn still
  records canon** — today `verse_storybook` success can short-circuit before
  `verse_record` (Codex P1); the storybook path must record a canon event for the
  story. Cost guards are stated honestly: per-account cooldown + per-turn cap; the
  daily cap is a TODO — either add a tiny date-keyed per-account counter or state
  there is no daily ceiling so the decision is eyes-open. Cap planned images at
  the per-turn cap.

## 10. Why this scope (red-team consensus)

9+ of 13 lenses independently flagged scope inflation: the retention fix is a
SELECT into the existing prompt builder, and the draft wrapped it in five-plus
speculative workstreams (new plugin, gen-core extraction, 9-table schema, story
fan-out, shadow/A-B, one-time port) with cross-plugin/reload/data-divergence/
refactor risk and no new user value. Cutting to in-place removes that risk, ships
faster on real data, and keeps rollback to a registry flip. Affirmed-and-kept:
the forgetting diagnosis, plain-SQL retrieval (no vectors), the `event_actor`
join, the durable/transient distinction (as a principle), the `author_locked`
invariant, trigger+generation staying in `llm`, and killing the silent model
fallback.

## 11. Deferred to later phases (schema-compatible)
- **New `Verse` plugin** — defer to the phase C config-collapse work it forces.
- **Gen-core extraction** — only consumer is verse (zero reuse before C); highest
  blast radius; do surgical in-place fixes now, extract in C.
- **Two-stage story fan-out** — keep single-pass until eval shows degradation at
  the cap; the fan-out also breaks the global LLM concurrency cap (Codex P1).
- **Full normalized schema** (`entity_state`, `event_archive`, merge apparatus,
  proposal reaper) — B2/B3, with the loom split and inspection suite they serve.
- **One-time data port** — unnecessary while retrieving in place.
- **Formal shadow/A-B harness** — overkill for one channel with a vocal author.
- **Portraits** — require a reference-image-capable provider (the prod image path
  has none); their own spec. For v1, lean on a canonical `appearance` trait + a
  channel style anchor injected into every image prompt (text consistency only).
- **Model-tool-surface trim** (retiring move/look) — UX fallout (Codex P3); keep
  human `@look`/movement as compatibility for now.

## 12. Settled / do NOT re-flag
Loom kept (game-state/lore split is B2); denial-regex false-positives on in-world
prose = accepted tradeoff; reactivate-by-name scoping to `npc + auto_created` is
load-bearing; provider content boundaries are policy, not bugs.

## 13. Testing
- **Retrieval:** locked roster always present (in full, untruncated); scene match
  via name **and alias**; 1-hop relations; active-only events filtered **in SQL**.
- **Promotion:** human-offered name reinforced N times → locked; model-invented
  name never promoted; first mention stays reapable `auto_created`.
- **Capability:** auto-lock fires for fc42's real hostmask-authed `llm.verse.edit`
  account; an unauthorized caller cannot promote.
- **Schema:** `event_actor` PK + cascade; element-wise tolerant blob backfill
  (counts reconcile, partial decode keeps valid ids); additive in-place migration
  is idempotent/re-runnable.
- **Cache:** the verse system message is byte-identical across turns when no
  author lock changed (no timestamps/heartbeats/date leak into it).
- **Generation:** missing/invalid verse model → hard-fail (not reasoning-model);
  illustrated turn records a canon event; characterize the re-seed behavior before
  any change.

## 14. Open risks
- Provider prefix-cache semantics are assumed — **verify before relying on the
  layout for cost.**
- `frequency_penalty` is a placebo on grok — retry-on-degrade is the only defense;
  watch its cost.
- Alias coverage depends on seeding from observed nicknames — incomplete aliases
  mean some phrasings still miss.
- Promotion threshold (N) needs tuning — too high and canon is slow to stick, too
  low and it over-locks.
