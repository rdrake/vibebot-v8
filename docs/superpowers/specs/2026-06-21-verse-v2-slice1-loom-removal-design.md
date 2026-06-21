# Verse v2 — Slice 1: Remove the Loom (+ Crosspoll) and Purge Its Data — Design

Status: Approved scope — ready for implementation planning
Date: 2026-06-21
Author: rdrake (with Claude)

First slice of the v2 "architectural leap", decomposed as: **this slice → gen-core extraction → verse-as-its-own-plugin**. Supersedes the earlier "normalized schema" Slice-1 framing (reaper + merge + event_archive), which was gutted once we decided to drop the loom: those pieces existed to manage *loom exhaust* (see §2).

## 1. Goal

Remove the **loom** (the #idlerpg game-state digester) and its **crosspoll** appendage (cross-channel canon seeding) — code, config, and the canon data they produced — leaving fc42's authored lore untouched. The loom adds no value, is a lot of code (~2,800 lines incl. tests, 10 config keys, 120 plugin.py refs), and live data shows it has been drowning real canon ~50:1.

### Success criteria
1. `loom.py` + the crosspoll subsystem are deleted; `validate_payload` is preserved for the `@versedit`/verse_edit path; the bot starts with no loom/crosspoll config or wiring.
2. chat/code/draw and **verse retrieval for real lore are behavior-preserving** — the 47 non-loom events + the pinned roster remain; `build_verse_system_prompt` for fc42's avatar is unchanged except that #idlerpg junk is gone.
3. #afternet canon is purged of loom/crosspoll exhaust: ~2,413 loom/crosspoll events + ~18 orphaned auto-NPCs deleted; ~47 authored events + the roster intact.
4. **No schema migration.** Rollback = code revert + DB backup restore.

## 2. Background — why this supersedes the schema slice

The decomposition originally put a "normalized schema" slice first (event_archive + proposal reaper + merge_entity). A 30-agent red-team (memory: `project_verse_v2_redteam_2026_06_21`, task w8l4mupxl) plus the decision to drop the loom dissolved it:
- **event_archive** — its two worst (HIGH) findings were data-loss (archiving the very events reactivate-by-name resurrects; diverging from the `entity_ids` JSON read path). Its only real consumer was the lore/game-state split, which the loom drop eliminates. **Cut.**
- **proposal reaper** — `proposals` only ballooned because of the loom. With the loom gone, the only writer is `@versedit`'s `apply_direct`. **Cut.**
- **merge_entity** — the purge is *deletion, not dedup*, so it isn't needed. A correct merge is also non-trivial (9 hardening constraints); its hardening list is parked in memory for a future, properly-spec'd merge. **Deferred.**

## 3. Evidence (live prod #afternet, 2026-06-21)

`_afternet_2de47b99.db` source breakdown: **avatar 24, loom 2413, operator 23** (loom = 98%).

- **Loom events are #idlerpg game junk** — e.g. *"blaat defeats jspiros in combat, gaining a substantial temporal reduction"*, *"choirboy installs flux capacitors to their sound card"*, *"chosen by the gods to escape Pyongyang"*.
- **The 47 non-loom events are fc42's canon** — e.g. *"The Cathedral Siege: fermented eggs lobbed through clerestory windows"*, *"The Canonical Roster Lock: fc42 fixes the year groups"*, *"The Original Stink: Assgas Archie's legendary assembly-hall arse-atomiser"*.
- **Mixed-source roster entities are safe:** Farty Freddie has avatar:4, loom:7, operator:2. Deleting only his *loom* events strips the junk; his real lore survives and the entity is pinned (never deleted).
- **Orphan set:** entities that are `auto_created`, NOT pinned/author_locked, and referenced **only** by loom/crosspoll events = **18** (blaat, jspiros, "The Boiler Room", a junk "stinky dan" duplicate, etc.).

## 4. Scope

### In scope
- **Part A:** remove loom + crosspoll code, config, and tests; relocate the one shared symbol; no schema migration.
- **Part B:** a one-time, tested data purge of loom/crosspoll exhaust from prod #afternet.

### Non-goals
- gen-core extraction (next slice) and verse-as-its-own-plugin (later slice).
- The deferred merge_entity / reaper / event_archive (parked with their hardening list in `project_verse_v2_redteam_2026_06_21`).
- fc42 reaction-mining (`project_fc42_reaction_mining_idea` — separate future brainstorm).
- Retroactively scrubbing loom-authored relations between two *surviving* entities (see §6 residuals).

## 5. Part A — code & config removal (a PR, behavior-preserving for real lore)

### 5.1 Files deleted
- Source: `verse/loom.py` (930), `verse/crosspoll_store.py`, `verse/crosspoll_schema.sql`.
- Tests: `tests/verse/test_loom.py` (1864), `tests/verse/test_loom_integration.py`, `tests/test_plugin_loom.py`. (`test_loom_validate_payload.py` → relocated, see 5.2.)

### 5.2 Shared-symbol relocation (the one thing that must NOT be deleted)
`validate_payload` is imported by `avatar.py:13` and used at `avatar.py:607` to validate `@versedit`/verse_edit payloads. Move `validate_payload` (and the private helpers it needs — `_is_strict_int`, `_is_int_list`) into a new neutral module `verse/validation.py`; update the `avatar.py` import; move `test_loom_validate_payload.py` → `tests/verse/test_validation.py`.

### 5.3 plugin.py wiring removed (120 refs — KEEP the `@versedit` path)
- Loom lifecycle: `_wire_loom_if_enabled`, `_on_loom_config_change`, the loom transcript-capture branch (~1241–1253), the `schedule.removeEvent("llm_loom_after_chime")` events, all `from .verse.loom import …` sites.
- Crosspoll: `_get_or_create_crosspoll_store` + the `_crosspoll_store`/`_crosspoll_store_lock` fields, the `event_source = "crosspoll" if … else "loom"` branch (~6557).
- **KEEP** the `@versedit` handler and its `store.apply_direct(...)` calls (~6173–6279) — that is the operator audit path, NOT the loom. The implementer must distinguish `apply_direct` (keep) from the loom's `apply_or_queue`/digest path (delete) at every site.

### 5.4 store.py
- Remove the loom-only proposal helpers **after verifying each has no non-loom caller**: `add_proposal`, `apply_and_record_proposal`, `apply_proposal_and_mark` (+ any loom-only reseed/proposal-resolve helpers). **Keep `apply_direct` and the `proposals` table** (the `@versedit` audit trail).
- Remove the crosspoll-store coupling (8 refs).

### 5.5 config.py
- Remove the 10 `loom*` keys and the 4 `crosspoll*` keys.

### 5.6 Tests de-loomed (not deleted)
- `tests/verse/test_verse_aging.py`: replace the 5 `apply_or_queue`-based seedings with direct store inserts (e.g. `store.add_event(...)`).
- `tests/verse/test_compaction.py`, `tests/verse/_fakes.py`: drop `LoomCallUsage`/`LoomConfig`/`VerseSnapshot` usage.
- `tests/test_plugin_verse.py`: drop the 4 `LiteLLMLoomClient` patch sites.

### 5.7 No schema migration
Leave `'loom'`/`'crosspoll'` in the `events.source` CHECK and `'crosspoll_seed'` in the `proposals.op` CHECK as harmless, unused enum values. `SCHEMA_VERSION` stays 3. This keeps Part A pure code-deletion (no migration to write, test, or roll back).

## 6. Part B — one-time data purge

### 6.1 Operation
A single tested function `purge_loom_data(store) -> (events_deleted, entities_deleted)` that runs in **ONE `write_transaction`, using only `_*_inline` writers / direct SQL — never public store methods** (the store's `self._lock` is non-reentrant and public methods commit mid-transaction; red-team finding):

1. **Compute the orphan set first** (before any delete): entities where `auto_created='1'` AND id NOT IN (pinned ∪ author_locked) AND every `event_actor` link is to a `source IN ('loom','crosspoll')` event — i.e. `loom_actor_ids − nonloom_actor_ids` intersected with the auto-created, non-canon set.
2. `DELETE FROM events WHERE source IN ('loom','crosspoll')` — cascades `event_actor` (FK `ON DELETE CASCADE`). The legacy `entity_ids` JSON lives on the deleted row, so **both** entity-linkage sources go together (red-team dual-linkage note: `events.entity_ids` JSON vs the `event_actor` join).
3. `DELETE FROM entities WHERE id IN (<orphan set>)` — cascades `relations`, `entity_alias`, `event_actor`, `attributes`.
4. Return counts for the operator log.

### 6.2 Why a one-time op, not a migration
The purge is destructive, channel-specific, and must never auto-run on other channels' DBs or re-run. It is invoked explicitly once against prod #afternet after a backup — mirroring the June 2026 manual dedup, but as tested code instead of ad-hoc SQL.

### 6.3 Residuals (accepted)
- **Loom-authored relations between two *surviving* entities** can't be detected (the `relations` table carries no `source`). Live data shows surviving-entity relations look authored (Mad Miss Muffet, Y12 Posh Lads, Stinky Girls); residual is minimal and left.
- Auto-created entities with **zero** event links are out of scope (not loom-attributable via `event_actor`); aging continues to handle them.

## 7. Test plan
- **Purge unit test** against a fixture DB seeded with mixed loom/avatar/operator events, a pinned roster, and orphan auto-NPCs: asserts loom/crosspoll events gone; real events + roster + their non-loom events intact; the 18-style orphans deleted; a mixed-source roster entity keeps its non-loom events and survives.
- **Relocation:** the validate-payload tests pass from `verse/validation.py`; `@versedit`/verse_edit validation still works via the new import.
- **De-loomed** aging/compaction/plugin tests pass.
- **Regression:** `build_verse_system_prompt` for an avatar is unchanged except junk removed; full `make test` (93% coverage gate); `make lint && make typecheck` clean; chat/code/draw untouched (`test_assistant`, `test_service_core`).

## 8. Rollout (operator)
1. Merge Part A (code removal) → deploy.
2. WAL-safe backup of prod `_afternet_2de47b99.db` (+`-wal`/`-shm`).
3. Run `purge_loom_data` once against prod #afternet; log the counts.
4. Spot-check a verse turn for fc42: roster + authored events present, no #idlerpg combat.

Rollback: revert the PR + restore the DB backup.

## 9. Risks
- **Distinguishing loom-only store methods from shared/`@versedit` ones** — mitigated by per-caller grep + the existing test suite; `apply_direct` must survive.
- **Test entanglement with loom types** — de-loom work, already sized (§5.6).
- **Destructive purge** — mitigated by the WAL-safe backup, a tested single-transaction function, and one-shot invocation.

## 10. What this enables
A smaller gen-core extraction (less completion-adjacent code) and a smaller verse-plugin slice (fewer config keys to migrate). The deferred merge_entity hardening list, reaper, and event_archive remain parked for if/when a concrete non-loom need appears.
