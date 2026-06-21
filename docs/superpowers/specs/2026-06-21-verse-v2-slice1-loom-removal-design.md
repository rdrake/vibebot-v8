# Verse v2 — Slice 1: Remove the Loom (+ Crosspoll), Decouple Compaction, and Purge Loom Data — Design

Status: Approved scope — revised after spec red-team (task w386fk5cd)
Date: 2026-06-21
Author: rdrake (with Claude)

First slice of the v2 "architectural leap", decomposed as: **this slice → gen-core extraction → verse-as-its-own-plugin**. Supersedes the earlier "normalized schema" Slice-1 framing (its reaper/event_archive/merge pieces existed to manage *loom exhaust* — see §2).

> **Revision note (post red-team):** a 30-agent red-team of the first draft found a blocker and 21 other confirmed gaps. The big one: **compaction is a kept subsystem entangled with the loom** — it borrows the loom's LLM client, its model key, and stamps its lore-digests `source='loom'`. There are **10 real-lore compaction digests** in prod `source='loom'` that a naïve purge would have destroyed. This revision decouples compaction and protects the digests.

## 1. Goal

Remove the **loom** (the #idlerpg game-state digester) and its **crosspoll** appendage, **decouple the kept compaction subsystem** from the loom, and purge the loom's #idlerpg exhaust from canon — leaving fc42's authored lore (including compaction digests) untouched. The loom adds no value, is a lot of code (~2,800 lines incl. tests, **14 config keys**, 120 plugin.py refs), and live data shows it has been drowning real canon ~50:1.

### Success criteria
1. `loom.py` + the crosspoll subsystem are deleted; **`validate_payload` AND the verse-completion client are relocated** (not deleted); the loom-moderation commands and `_PluginLoomBridge` are gone; the bot starts and runs with no loom/crosspoll config or wiring.
2. **Compaction still works** — the daily timer (armed at plugin.py:831) and `@versecompact` run against a relocated client + a new `verseCompactionModel` key.
3. chat/code/draw and **verse retrieval for real lore are behavior-preserving** — the ~47 authored events, the **10 compaction digests**, and the pinned roster remain.
4. #afternet canon purged of loom/crosspoll exhaust: ~2,406 #idlerpg events + ~18 orphaned auto-NPCs deleted; authored events + digests + roster intact.
5. **No schema migration.** Rollback = code revert + DB backup restore.

## 2. Background — why this supersedes the schema slice

The decomposition originally put a "normalized schema" slice first (event_archive + proposal reaper + merge_entity). The design red-team (`project_verse_v2_redteam_2026_06_21`, task w8l4mupxl) + the decision to drop the loom dissolved it: event_archive carried two HIGH data-loss bugs and had no retrieval payoff (**cut**); the reaper only mattered because the loom ballooned `proposals` (**cut**); merge is dedup, but the purge is deletion (**deferred**, hardening list parked).

## 3. Evidence (live prod #afternet, 2026-06-21)

`_afternet_2de47b99.db`: **avatar 24, loom 2416, operator 23** (loom = 98%).

- **Most loom events are #idlerpg game junk** (median summary 94 chars) — *"blaat defeats jspiros in combat, gaining a substantial temporal reduction"*.
- **The 47 non-loom events are fc42's canon** — *"The Cathedral Siege…"*, *"The Canonical Roster Lock: fc42 fixes the year groups"*.
- **⚠️ `source='loom'` is OVERLOADED:** compaction stamps its lore-digests `source='loom'` (store.py:999), and **10 such digests exist in prod** (700–835 chars), e.g. *"Chronicler fc42 recounts the anarchic reign of the Stinky Lads—Poo Pete, Assripping Alex, Stinky Sebastian…"*. Compaction **deletes the originals** (store.py:963), so a digest is the **only** surviving record of that older lore. The purge MUST preserve these (§6).
- **Mixed-source roster entities are safe:** Farty Freddie has avatar:4, loom:7, operator:2 — deleting only his #idlerpg loom events strips junk; his real lore survives and he is pinned.
- **Orphan set:** `auto_created`, NOT pinned/author_locked, referenced **only** by loom/crosspoll #idlerpg events = ~18 (blaat, jspiros, "The Boiler Room", a junk "stinky dan" dup).

## 4. Scope

### In scope
- **Part A:** remove loom + crosspoll code/config/tests + the loom-moderation commands; **decouple compaction** (relocate its LLM client, add `verseCompactionModel`, re-source digests to `'llm'`); relocate `validate_payload`; no schema migration.
- **Part B:** a one-time, tested purge of #idlerpg loom/crosspoll exhaust from prod #afternet — **after re-stamping the 10 compaction digests** so they survive.

### Non-goals
- gen-core extraction (next slice); verse-as-its-own-plugin (later slice).
- The deferred merge_entity / reaper / event_archive (parked in `project_verse_v2_redteam_2026_06_21`).
- fc42 reaction-mining (`project_fc42_reaction_mining_idea` — separate brainstorm).
- Removing compaction (kept; it is general retention, not #idlerpg-specific).
- Scrubbing loom-authored relations between two *surviving* entities (no `source` column; see §6.3).

## 5. Part A — code & config removal + compaction decoupling

### 5.1 Files deleted
- Source: `verse/loom.py` (930) — **only after the relocations in §5.2** — + `verse/crosspoll_store.py` + `verse/crosspoll_schema.sql`.
- Tests: `tests/verse/test_loom.py` (1864), `tests/verse/test_loom_integration.py`, `tests/verse/test_crosspoll_store.py`, `tests/test_plugin_loom.py`. (`test_loom_validate_payload.py` → relocated, §5.2.)

### 5.2 Relocations (must NOT be deleted with loom.py)
- **validate_payload + its schema table.** `validate_payload` dereferences the module-global `_PAYLOAD_SCHEMA` (loom.py:123) and the predicates it binds (`_is_strict_int`, `_is_int_list`). Move **all four** into a new `verse/validation.py` (with the `typing.Any` / `collections.abc.Callable` imports the annotations need). Do NOT move `_VALID_OPS` (used only by the deleted `parse_digest`). Update `avatar.py:13` import; move `test_loom_validate_payload.py` → `tests/verse/test_validation.py`.
- **The verse-completion LLM client (compaction's runtime dependency).** `compact_verse` needs a client; the only impl is `LiteLLMLoomClient` (loom.py:448), its return type `LoomCallUsage` (loom.py:436), and the `LoomModelClient` Protocol (loom.py:442). **Move all three into `verse/compaction.py`** (its sole surviving consumer) and rename neutrally → `LiteLLMVerseClient` / `VerseCallUsage` / `VerseModelClient`. Repoint the production importers: plugin.py:5587 + 5604 (daily pass) and plugin.py:6641 + 6664 (`@versecompact`); and `tests/verse/test_compaction.py:23` (`_FakeClient`'s return type) → import from `compaction`.

### 5.3 plugin.py wiring removed (KEEP `@versedit` and `@versecompact`)
- **Loom lifecycle:** `_wire_loom_if_enabled`, `_on_loom_config_change` (and its registry change-hook loop), the loom transcript-capture branch (~1241–1253), the `llm_loom_after_chime` schedule events, all `from .verse.loom import …` sites except the relocated client.
- **Crosspoll:** `_get_or_create_crosspoll_store` + `_crosspoll_store`/`_crosspoll_store_lock` fields + the `event_source = "crosspoll" if … else "loom"` branch (~6557).
- **Loom-moderation commands (now dead — the loom was the only producer of `status='pending'` proposals):** delete `verseproposals`/`verseapprove`/`versereject` (plugin.py ~6477/6528/6575), their `wrap` registrations + COMMAND_REGISTRY entries (~377–410), the helpers `_proposal_snippet`/`_proposal_target_store`/`_load_proposal`, the `_VERSEPROPOSALS_MAX_LIMIT` constant, and the dispatch test.
- **`_PluginLoomBridge`:** delete the class (plugin.py:6710–6807) and its `_loom_bridge`/`_loom`/`_loom_*_cache` init fields (744–748) — it is the only non-loom-file `VerseSnapshot` importer (6747) and crosspoll-store accessor.
- **KEEP** `@versedit` + its `store.apply_direct(...)` (~6173–6279) and `@versecompact` (repointed per §5.2/§5.5). The implementer must distinguish `apply_direct` (keep) from the loom's `apply_or_queue`/proposal path (delete) at every site.

### 5.4 store.py
- Delete the loom-only proposal helpers `add_proposal`, `apply_and_record_proposal`, and `apply_proposal_and_mark` — the last only **after** the §5.3 moderation commands (its sole non-loom callers) are deleted. **Keep `apply_direct` and the `proposals` table** (the `@versedit` audit trail).
- Delete `bump_last_seen_ts` (store.py:566) — loom-only (callers loom.py:365/378); becomes dead code.
- Remove the crosspoll-store coupling (8 refs).
- **Re-source compaction digests:** change `replace_events_with_lore_digest` to stamp `source='llm'` instead of `'loom'` (store.py:999) so future digests are unambiguous. (`'llm'` is already a valid `events.source` CHECK value — no migration.)
- Reword the surviving `apply_direct` docstring (store.py:~1495) to drop its reference to the now-deleted `apply_and_record_proposal`.

### 5.5 config.py
- Remove **14 loom-family keys**: the 10 `loom*` keys, the 3 crosspoll keys (`verseCrosspollAllowSend`, `verseCrosspollAllowReceive`, `verseCrosspollPerCycleLimit`), and `verseAutoApplyThreshold` (config.py:392 — loom-only; sole consumer was `_wire_loom_if_enabled`, so a grep-for-`crosspoll` pass misses it).
- **Add `verseCompactionModel`** (default `"gemini/gemini-flash-lite-latest"`, the old `loomModel` default) and repoint plugin.py:5602 + 6662 from `registryValue("loomModel")` → `registryValue("verseCompactionModel")`.
- **Startup safety:** stale `loom*` values remaining in prod `bot.conf` are harmless — Limnoria ignores registry entries with no registered definition on load; confirm with a clean start after deploy (no fatal on unknown keys).

### 5.6 Tests de-loomed (not deleted)
- `tests/verse/test_verse_aging.py`: replace the 5 `apply_or_queue` seedings with direct store inserts.
- `tests/verse/test_compaction.py`: repoint the `LoomCallUsage` import → `VerseCallUsage` from `compaction` (§5.2); compaction tests still need the usage type.
- `tests/verse/_fakes.py`: repoint `LoomCallUsage`/`LoomConfig`/`VerseSnapshot` (drop loom-config/snapshot fakes; keep the client-usage fake against the new name).
- `tests/test_plugin_verse.py`: drop the 4 `LiteLLMLoomClient` patch sites (or repoint to the relocated client).

### 5.7 No schema migration
Leave `'loom'`/`'crosspoll'` in the `events.source` CHECK and `'crosspoll_seed'` in `proposals.op` CHECK as harmless dead enum values. Digests move to `'llm'` (already allowed). `verseCompactionModel` is config, not schema. `SCHEMA_VERSION` stays 3.

### 5.8 Coverage
The gate is thin (~93.9% vs 93%). Deleting the loom-only-but-kept dead helpers (`bump_last_seen_ts`, the moderation helpers) is required so no now-uncovered kept code erodes the buffer; verify `make test` still clears 93% after the de-loom, adding focused tests for any kept helper that loses its only (loom-test) coverage.

## 6. Part B — one-time data purge

### 6.1 Operation
A tested function `purge_loom_data(store)` run once against prod #afternet after a WAL-safe backup, in **ONE `write_transaction` using only `_*_inline` writers / direct SQL** (the store's `self._lock` is non-reentrant; public methods commit mid-transaction):

0. **Protect the digests first.** Inventory `source='loom'` events that are **compaction lore-digests** (long chronicle summaries referencing the canon roster — ~10 on prod, all >300 chars vs the <150-char #idlerpg combat lines) and **re-stamp them `source='llm'`** (`UPDATE events SET source='llm' WHERE id IN (<reviewed digest ids>)`). **Review the id list before running** — do not re-stamp on length alone.
1. **Compute the orphan set** (now that digests are `'llm'`): entities where `auto_created='1'` AND id NOT IN (pinned ∪ author_locked) AND every `event_actor` link is to a `source IN ('loom','crosspoll')` event.
2. `DELETE FROM events WHERE source IN ('loom','crosspoll')` — cascades `event_actor` (FK `ON DELETE CASCADE`, schema.sql:83-84). The legacy `entity_ids` JSON lives on the deleted row, so both linkage sources go together.
3. `DELETE FROM entities WHERE id IN (<orphan set>)` — cascades `relations`, `entity_alias`, `event_actor`, `attributes` (all `ON DELETE CASCADE`).
4. Return `(events_deleted, entities_deleted, digests_restamped)` for the operator log.

### 6.2 Why a one-time op, not a migration
Destructive, channel-specific, must never auto-run on other DBs or re-run. Invoked explicitly once against prod #afternet after a backup — mirroring the June dedup, but as tested code.

### 6.3 Residuals (accepted)
- Loom-authored relations between two *surviving* entities can't be detected (`relations` has no `source`); live data shows surviving-entity relations look authored — minimal residual, left.
- Auto-created entities with zero event links are out of scope; aging handles them.

## 7. Test plan
- **Purge unit test** (fixture DB: mixed loom/avatar/operator events + a `source='loom'` **compaction digest** + a pinned roster + orphan auto-NPCs): asserts #idlerpg loom/crosspoll events gone; **the digest is re-stamped `'llm'` and survives**; authored events + roster + their non-loom events intact; orphans deleted; a mixed-source roster entity keeps its non-loom events.
- **Relocation:** validate-payload tests pass from `verse/validation.py` (incl. `_PAYLOAD_SCHEMA`); `@versedit` validation works; **compaction runs against the relocated client + `verseCompactionModel`**, and new digests are stamped `'llm'`.
- **De-loomed** aging/compaction/plugin tests pass; the deleted moderation commands have no dangling dispatch test.
- **Regression:** `build_verse_system_prompt` for an avatar unchanged except junk removed; `make test` clears the 93% gate; `make lint && make typecheck` clean; chat/code/draw untouched (`test_assistant`, `test_service_core`); clean bot start with stale `loom*` keys still in a test `bot.conf`.

## 8. Rollout (operator)
1. Merge Part A → deploy; confirm clean start (no fatal on stale `loom*` registry keys) and that the compaction timer arms.
2. WAL-safe backup of prod `_afternet_2de47b99.db` (+`-wal`/`-shm`).
3. **Review the ~10 digest ids**, then run `purge_loom_data` once against prod #afternet; log `(events_deleted, entities_deleted, digests_restamped)`.
4. Spot-check a verse turn for fc42: roster + authored events + the chronicle digests present, no #idlerpg combat.

Rollback: revert the PR + restore the DB backup.

## 9. Risks
- **`source='loom'` overload (compaction digests)** — mitigated by the §6.0 re-stamp + reviewed id list + the go-forward `'llm'` source.
- **Compaction's loom coupling** (client/model) — mitigated by the §5.2 relocation + `verseCompactionModel`; without it the daily pass and `@versecompact` break.
- **Keep/delete boundary** (`apply_proposal_and_mark`, the moderation trio) — resolved by deleting the commands and the helper together.
- **Thin coverage margin** — mitigated by deleting dead helpers + targeted tests (§5.8).
- **Destructive purge** — WAL-safe backup + tested single-transaction function + one-shot invocation.

## 10. What this enables
A smaller gen-core extraction and verse-plugin slice (fewer config keys, less code). The deferred merge_entity hardening list, reaper, and event_archive remain parked for if/when a concrete non-loom need appears.
