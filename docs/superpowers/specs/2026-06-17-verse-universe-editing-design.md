# Verse Universe Editing — Design (v2, post-red-team)

**Date:** 2026-06-17
**Status:** Approved design, pre-implementation
**Author:** rdrake + Claude
**Supersedes:** v1 of this file (red-teamed; v1 misdiagnosed the problem — see "What changed from v1").

## Problem

The forest-verse subsystem stores a per-channel universe (entities, places,
factions, items, attributes, relations, events) in a SQLite store
(`plugins/llm/src/llm/verse/store.py`). Today it can only be mutated by the
loom (propose → `@verseapprove`, or auto-apply above `verseAutoApplyThreshold`)
and by avatar/engine lifecycle paths. There is **no operator-direct editing**
and **no user-driven direct editing**.

**The deeper problem (found by red-team):** even if you *could* author canon,
**nothing reads NPC canon into the model's context on a verse turn.**
`build_verse_system_prompt` (`avatar.py:428-520`) injects only (a) up to ~5 of
the speaking avatar's own recent events and (b) co-located *avatars*. It never
enumerates `npc`/`place`/`faction` entities. So adding 15 NPC rows to SQLite is
invisible to the model — the motivating "remember the 15 stinky lads" goal is
**not** solved by an editing surface alone. There is no "pinned canon" path
anywhere in `verse/`; `versecompact` only folds old events into a digest event
that still reaches the prompt through the same ~5-event window.

## Goal

1. **Make authored canon actually reach the model** every verse turn (the
   consumption layer) — this is what solves roster memory.
2. Let **authorized users** manipulate a channel's universe both **manually**
   (operator commands) and **automatically** (an LLM tool), through one
   validated mutation core, with consistent provenance and audit.

## Decisions (locked with user)

- **Scope:** per-channel store (current model). No shared cross-channel world.
- **Shape:** single combined spec (not phased), but the consumption layer is
  Component 1 because it is load-bearing for the stated goal.
- **Access control:** a **single new capability `llm.verse.edit`** is THE gate.
  "Certain users" = accounts holding it. No separate allowlist. It gates BOTH
  the operator commands AND the LLM tool (keyed on the *triggering* user).
- **Minimal knobs:** because editing is restricted to trusted capability
  holders, we **drop** the v1 `verseEditEnabled`, `verseEditMode`, and
  per-account rate-limit knobs. Authorized edits apply directly; the loom keeps
  proposing exactly as today.
- **Manual operator edits:** apply immediately, `source='operator'`.
- **LLM edits:** apply immediately when the triggering user holds
  `llm.verse.edit`; otherwise the tool is not offered / the call is refused.
  `source='llm'`. Constructive ops only.
- **Architecture:** thin command/tool layer over one validated mutation core
  (the existing `_apply_op_inline`, extended).

## What changed from v1 (red-team deltas)

The v1 spec made three false claims and missed the consumption layer:

- ❌ "No schema change needed" — **false.** `proposals.op` and `events.source`
  CHECK constraints reject the new ops/sources, and `_migrate` has no
  version-stepping. → real migration added (Component 5).
- ❌ "All callers funnel through one core" — **partly false.** `apply_proposal`
  (`store.py:1098`) has its own op dispatch separate from `_apply_op_inline`.
  → both dispatchers extended, or the divergent one collapsed (Component 2).
- ❌ "Reuse loom `_validate`" — **fiction.** No such function; `parse_digest`
  is array-shaped and covers none of the new ops. → extract real
  `validate_payload(op, payload)` (Component 2).
- ❌ Editing surface solves roster memory — **no.** → Component 1 added.

Severity-ranked findings and their resolutions are tracked in the
"Red-team resolution table" at the end.

## Component 1 — Consumption layer: pinned roster reaches the prompt

This is the part that actually makes the lads stick.

- Add a boolean entity marker **`pinned`** (an `attributes` row, key `pinned`,
  value `"1"`; *not* a schema column — attributes are free-form, and `pinned`
  is added to `_RESERVED_ATTRIBUTE_KEYS` so only the engine/operator path sets
  it, never `set_attribute` from loom/LLM).
- Extend `build_verse_system_prompt` (`avatar.py:428`) with a bounded
  **"Established characters in this world:"** block listing active `pinned`
  entities (any kind), `name — summary`, capped at `verseRosterMaxChars`
  (one new knob, the only one we add for consumption; default ~600 chars) so a
  large roster can't blow the context budget. Deterministic order (kind
  precedence, then name) so the block is cache-stable.
- The same block feeds the loom snapshot (`plugin.py:6364`) so the loom also
  sees pinned canon, not just `top_entities[:5]`.

Net effect: `@versedit add npc "Assgas Archie" … && @versedit pin "Assgas
Archie"` (or one `@versedit add --pin`) puts Archie in *every* verse turn's
context until unpinned. 15 pinned lads → all 15 enumerated each turn (within
the char cap).

## Component 2 — Mutation core (one validated primitive)

`VerseStore._apply_op_inline(conn, *, op, payload, source)` (`store.py:946`) is
the in-transaction primitive. We:

1. **Add new ops:** `update_entity` (name/summary only — rejects a `kind`
   field), `set_status` (validates `status ∈ {active,retired}`), `edit_event`,
   `delete_event`, `delete_relation`, and `set_pinned` (engine/operator path
   for the Component 1 marker).
2. **Compute `privileged` *inside* the core from a validated `source` enum** —
   never accept a caller-passed bool. `source` must be one of
   `{operator, loom, llm, crosspoll, avatar}` (validated against a frozenset;
   raise on anything else). `privileged = (source == 'operator')`.
   - Privileged: may run destructive ops (`delete_event`, `delete_relation`,
     `set_status → retired`), target retired entities (to restore), set the
     `location` reserved key. May **not** raw-set `status`/`kind`/`pinned`/
     bookkeeping keys (those have managed ops).
   - Constructive (`loom`/`llm`/`crosspoll`/`avatar`): today's guards exactly —
     no reserved keys, no retired targets, **constructive ops only**
     (`add_entity`, `add_event`, `set_attribute` non-reserved, `add_relation`,
     `update_entity`). Destructive ops `raise PermissionError`.
3. **Collapse the second dispatcher:** make `apply_proposal` (`store.py:1098`)
   call `_apply_op_inline` instead of re-dispatching, so there is exactly one
   op→SQL mapping.
4. **Extract `validate_payload(op, payload) -> str|None`** from the
   `loom.py:189-199` predicate block; extend `_PAYLOAD_SCHEMA` (`loom.py:123`)
   with entries for every new op. Both the loom and the `verse_edit` tool call
   it **before** `_apply_op_inline`. Type/shape validation (`_is_strict_int`
   etc.) runs server-side, never trusting LLM/operator JSON.
5. **Idempotency:** `delete_event`/`delete_relation`/`edit_event`/`set_status`
   check `rowcount`/existence and `raise LookupError` on a phantom id (mirror
   `update_proposal_status`, `store.py:1145`).

## Component 3 — Operator commands: `@versedit`

Per-verb Limnoria subcommands using real `wrap()` converters (NOT a freeform
`::` parser — v1's grammar didn't fit `wrap()` and had id-vs-name/`::`/`@`
footguns). Each is `wrap`'d with `[("checkCapability","llm.verse.edit"),
…slots…, optional("channel")]` so the capability is evaluated **against the
target channel** (fixes the cross-channel scoping bug; `versedump`'s in-body
global check is the wrong pattern to copy):

```
@versedit add    <kind> <name> [summary]      add entity (npc|place|faction|item|avatar)
@versedit pin    <ref>                          mark pinned (Component 1)
@versedit unpin  <ref>
@versedit set    <ref> <key> <value>            attribute (operator may set `location`)
@versedit name   <ref> <new-name>               rename
@versedit desc   <ref> <summary>                re-summarize
@versedit retire <ref>                          soft-delete
@versedit restore<ref>
@versedit relate <ref> <kind> <ref> [note]      add relation
@versedit unrelate <relation-id>
@versedit event  <summary> [ids]                add canon event
@versedit editevent <event-id> <summary>
@versedit delevent  <event-id>
@versedit show   [ref]                           inspect (delegates to @look)
```

- **`<ref>` disambiguation rule (explicit):** a token of the form `#<int>`
  (e.g. `#42`) is always an entity id; anything else is a name. This dodges the
  "an NPC named `7`" collision. Names resolve **inside the write transaction**
  via `_find_active_entity_by_name_inline` (`store.py:248`) so resolve+apply is
  atomic (fixes the TOCTOU).
- **Name-uniqueness guard:** `add`/`name`/`restore` reject creating a *second
  active entity with an existing active name* (raise with the colliding id),
  since `find_active_entity_by_name` is `LIMIT 1` and silent collisions brick
  name refs.
- Direct apply, `source='operator'`, one `write_transaction`, one-line
  confirmation with the affected id. `LookupError`/`ValueError`/`PermissionError`
  → friendly IRC replies, never tracebacks.

## Component 4 — Automatic: `verse_edit` tool (gated per triggering user)

A model-invoked tool on verse routes, **constructive ops only**, `source='llm'`.

- **Gate:** the tool only mutates when the **triggering user holds
  `llm.verse.edit`**. The requesting account is already threaded into the tool
  path (`service.py` account resolver at :224, `account` field at :425/:449/
  :1875). At dispatch: resolve the triggering user's identity → check
  `llm.verse.edit` → if absent, the tool call is refused with a benign result
  ("not authorized to edit canon") and **no mutation occurs**. Unauthorized
  users' messages can never change canon — this closes the prompt-injection
  hole without a rate-limit.
- **Apply:** authorized → apply immediately via the core through a purpose-built
  `apply_direct(op, payload, *, source)` that writes the row(s) **and** an
  audit `proposals` row with `status='approved'`, without the
  `cycle_id`/`confidence`/`reviewer` ceremony `apply_and_record_proposal`
  demands (v1 would have polluted the proposals table with synthetic-approved
  loom-shaped rows). `apply_direct` records real provenance
  (`provenance='verse_edit'`, the triggering account).
- Payload validated by `validate_payload` (Component 2) before apply.
- The loom is unchanged: still proposes; `verseAutoApplyThreshold` still governs
  its auto-apply.

## Component 5 — Schema migration

The new ops/sources violate two CHECK constraints, and `_migrate`
(`store.py:159`) currently has no upgrade path. Build minimal versioned
migration:

- Bump `SCHEMA_VERSION` 1 → 2 (`store.py:94`).
- `_migrate` reads `schema_version.version`; if `< 2`, run an upgrade step that
  **rebuilds** `proposals` and `events` with widened CHECKs
  (`proposals.op` adds the 6 new ops; `events.source` adds `'operator'`,`'llm'`)
  via the SQLite 12-step table-rebuild (create new, copy, drop, rename, inside
  one transaction), then stamps version 2. Idempotent and re-runnable.
- `schema.sql` (the fresh-install DDL) updated to the v2 constraints so new
  stores are born correct.

## Soft-delete coherence (explicit rules)

- **Retire is shallow by design** for non-avatars: status flips; historical
  events keep their `entity_ids` references (intentional — canon history is
  immutable). Prompt/snapshot builders must filter to `status='active'` when
  *listing* entities (Component 1 lists only active pinned; presence query at
  `avatar.py:488` already filters active).
- **Retiring a `kind='avatar'` entity** must atomically clear its `avatar_link`
  (reuse `unlink_avatar`'s paired status+link logic, `store.py:551`) — a bare
  `set_status` would leave a dangling link that bricks the user
  (`record_user_event` raises on the next action) and silently un-retires on
  next opt-in. The `set_status → retired` op special-cases `kind='avatar'`.
- **Restore** re-checks active-name collision before reactivating.

## Provenance, audit, error handling

- Every mutation writes a validated `source`; operator/llm/loom/crosspoll
  distinguishable in `events.source` and the `proposals` audit rows.
- Operator command edits also write an `approved` audit `proposals` row (via
  the same `apply_direct`, `source='operator'`) so manual canon edits are
  auditable (v1 left them unaudited; `events` has no `updated_at`).
- Soft-delete only for entities (retire); only leaf event/relation rows are
  truly deleted, and those raise `LookupError` on a missing id.
- All mutations single-transaction; an error leaves no partial write.

## Testing

Store tests are pure-SQLite (no IRC). Add:

1. **Migration:** open a v1 DB (legacy CHECKs), run `_migrate`, assert version 2
   and that new ops/sources now insert; assert idempotent re-run.
2. **Core ops:** each new op applies; `apply_proposal` and `_apply_op_inline`
   produce identical results (single-dispatcher guarantee).
3. **Privilege tier:** `_apply_op_inline(op='delete_event', source='llm')`
   raises `PermissionError`; `source='operator'` succeeds; invalid `source`
   raises; caller cannot pass `privileged`.
4. **Gating:** `verse_edit` from an account without `llm.verse.edit` mutates
   nothing and returns the benign refusal; with it, applies. `@versedit #B`
   checks the cap against #B, not the origin channel.
5. **Command parser:** `#<int>` vs name disambiguation, in-txn resolve, ambiguous
   /colliding-name rejection, channel defaulting.
6. **Soft-delete:** retiring a linked avatar clears the link and does not brick
   `record_user_event`; restore re-collision-checks.
7. **Consumption:** a pinned entity appears in `build_verse_system_prompt`
   output; the roster block respects `verseRosterMaxChars`; unpinned drops out.
8. **Idempotency/provenance:** delete-of-phantom raises; rows carry expected
   `source`.

## Out of scope (YAGNI)

- Cross-channel / shared global universe.
- Hard `DELETE` of entities (retire is soft-delete; only leaf rows truly delete).
- Web/GUI editor; bulk import/export beyond `@versedump`.
- Per-channel enable flag, edit-mode flag, rate-limiting (cut — the capability
  gate makes them redundant).

## Affected files (anticipated)

- `verse/store.py` — extend `_apply_op_inline` (new ops, validated-source
  privilege), collapse `apply_proposal` dispatch, `apply_direct`, soft-delete
  avatar handling, name-collision guards, `SCHEMA_VERSION`+`_migrate` upgrade.
- `verse/schema.sql` — widen `proposals.op` / `events.source` CHECKs.
- `verse/avatar.py` — pinned-roster block in `build_verse_system_prompt`.
- `verse/loom.py` — extract `validate_payload`, extend `_PAYLOAD_SCHEMA`.
- `plugin.py` — `@versedit` subcommands; add `llm.verse.edit` to the default
  capability set (`:91`); pinned roster into the loom snapshot (`:6364`).
- `service.py` — advertise/dispatch `verse_edit`; per-triggering-user gate.
- `config.py` — `verseRosterMaxChars` (the one new knob).
- `tests/` — per the Testing section.

## Red-team resolution table

| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| A1 | 🔴 | NPC canon never reaches prompt | Component 1 (pinned roster block) |
| A2 | 🔴 | "No schema change" false (CHECKs, no migrate) | Component 5 |
| D2 | 🟠 | Two op dispatchers | Component 2.3 (collapse) |
| S1/S2 | 🟠 | Tool payload unvalidated; gate must be in core | Component 2.2/2.4 |
| A3 | 🟠 | loom `_validate` fiction | Component 2.4 (`validate_payload`) |
| D5 | 🟠 | Retire avatar dangles `avatar_link` | Soft-delete rules |
| D6/A2b | 🟠 | No name uniqueness → ref brick | Component 3 guards + `#id` rule |
| S5 | 🟡 | direct-mode injection sink | Per-user `llm.verse.edit` gate (Comp 4) |
| A4 | 🟡 | `apply_and_record_proposal` impedance | `apply_direct` (Comp 4) |
| S6 | 🟡 | cross-channel cap scoping | `wrap` checkCapability vs target channel (Comp 3) |
| S7/D8/D9 | 🟡 | enum/kind/idempotency/audit guards | Component 2.1/2.5 + audit rows |
| D7 | 🟡 | resolve→apply TOCTOU | in-txn name resolution (Comp 3) |
| D3 | 🟡 | edit/delete vs compaction race | documented last-writer semantics; deletes idempotent |
