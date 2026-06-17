# Verse Universe Editing — Design

**Date:** 2026-06-17
**Status:** Approved design, pre-implementation
**Author:** rdrake + Claude

## Problem

The forest-verse subsystem stores a per-channel "universe" (entities, places,
factions, items, attributes, relations, events) in a SQLite store
(`plugins/llm/src/llm/verse/store.py`). Today that universe can only be mutated
by:

1. The **loom**, which proposes batched mutations that an operator approves
   (`@verseapprove`) or that auto-apply above `verseAutoApplyThreshold`.
2. Avatar opt-in / engine lifecycle paths.

There is **no way for an operator to directly create or edit canon** (add a
character, rename a place, fix an event, retire an NPC), and **no way for the
in-scene LLM to mutate canon directly** — it can only emit loom proposals after
the fact. The operator's only durable lever is the channel
`assistantSystemPrompt` overlay, which is the wrong home for structured world
state.

## Goal

Let operators and the LLM manipulate a channel's verse universe — characters,
places, events, attributes, relations — both **manually** (operator commands)
and **automatically** (an LLM tool), through the store's existing single
mutation primitive, with consistent provenance and audit.

## Decisions (locked with user)

- **Scope:** per-channel store (current model). No shared cross-channel world.
- **Operator edits:** apply immediately (direct), recorded with
  `source='operator'`.
- **Automatic edits:** configurable per channel — direct-apply in trusted
  channels, propose-and-approve elsewhere.
- **Gating:** operator commands require the existing `llm.verse.gm` capability
  (same as `@versedump`).
- **Architecture:** Approach 1 — thin command/tool layer over the existing
  unified mutation core.

## Architecture

### The mutation core already exists

`VerseStore._apply_op_inline(conn, *, op, payload, source)`
(`store.py:946`) is the single in-transaction primitive. Every existing apply
path routes through it:

- `apply_proposal_and_mark` (human `@verseapprove`)
- `apply_and_record_proposal` (loom auto-apply, crosspoll receive)
- `apply_proposal` (connection-less convenience wrapper)

Today it handles ops: `add_event`, `set_attribute`, `add_relation`,
`add_entity`. It enforces two guards for proposal-sourced edits:

- `_RESERVED_ATTRIBUTE_KEYS = {last_seen_ts, auto_created, status, kind,
  location}` cannot be set via `set_attribute` (engine/lifecycle-only).
- `set_attribute` / `add_relation` reject **retired** entity targets.

**Approach 1 reuses this primitive unchanged in spirit:** we extend its op
vocabulary and add a privilege tier keyed on `source`. All three callers
(operator commands, `verse_edit` tool, loom) continue to funnel through it.

### New ops added to `_apply_op_inline`

| op | payload | effect |
|----|---------|--------|
| `update_entity` | `entity_id`, `name?`, `summary?` | rename / re-summarize |
| `set_status` | `entity_id`, `status` (`active`\|`retired`) | retire / restore (soft-delete) |
| `edit_event` | `event_id`, `summary` | fix a canon event |
| `delete_event` | `event_id` | remove an event (leaf row) |
| `delete_relation` | `relation_id` | remove a relation (leaf row) |

Existing ops (`add_event`, `set_attribute`, `add_relation`, `add_entity`)
are unchanged.

### Privilege tier (keyed on `source`)

The primitive gains a notion of **privileged** vs **constructive** sources.

- **Privileged** (`source='operator'`): may run *destructive* ops
  (`delete_event`, `delete_relation`, `set_status → retired`), may target a
  **retired** entity (e.g. to restore it), and may set the **`location`**
  reserved key (relocation). It may **not** raw-set `status`/`kind`/
  `last_seen_ts`/`auto_created` as attributes — `status` is managed via the
  `set_status` op; `kind` is immutable after creation; the other two are
  engine bookkeeping. This keeps lifecycle invariants intact even for
  operators.
- **Constructive** (`source ∈ {'loom','llm'}`): today's behavior, exactly.
  Constructive ops only (`add_entity`, `add_event`, `set_attribute` on
  non-reserved keys, `add_relation`, `update_entity` summary/name). **No**
  `set_status`, `delete_*`, retired targets, or reserved keys.

Implementation: `_apply_op_inline` gains a `privileged: bool` parameter
(default `False`). The existing guards run unless `privileged`; the new
destructive ops raise `PermissionError` unless `privileged`. Callers pass
`privileged=(source == 'operator')`.

## Component 1 — Operator command surface: `@versedit`

A single capability-gated dispatcher command (`llm.verse.gm`), direct-apply,
`source='operator'`, `privileged=True`, channel-aware (defaults to the current
channel; accepts an explicit `#channel`).

```
@versedit add    <kind> <name> :: <summary>     new entity (npc|place|faction|item|avatar)
@versedit set    <ref> <key> <value>            attribute (operator may set `location`)
@versedit name   <ref> <new name>               rename
@versedit desc   <ref> :: <summary>             re-summarize
@versedit retire <ref>                          soft-delete (set_status retired)
@versedit restore<ref>                          un-retire (set_status active)
@versedit relate <ref> <kind> <ref> [:: note]   add relation
@versedit unrelate <relation_id>                delete relation
@versedit event  <summary> [@ <ref>,<ref>,...]  add canon event
@versedit editevent <event_id> :: <summary>     edit event
@versedit delevent  <event_id>                  delete event
@versedit show   [<ref>]                         inspect (delegates to @look)
```

- **`<ref>`** = entity id (int) or name. Names resolve via
  `find_active_entity_by_name` (case-insensitive, active-only). Ambiguous or
  missing names return a clear error listing candidates; a bare integer is
  treated as an id.
- **`::`** separates a free-text summary tail from positional args so summaries
  may contain spaces. `@ id,id` attaches event actors.
- Each subcommand maps to exactly one core op (or a `@look` read for `show`),
  applied in one `write_transaction`, and replies with a one-line confirmation
  including the affected id.
- Errors (`LookupError`, `ValueError`, `PermissionError`) become friendly IRC
  replies, never tracebacks.

Single dispatcher (vs. N top-level `verseAdd`/`verseSet`/… commands) chosen so
gating, channel-resolution, ref-parsing, and tests live in one place, and the
verb set maps 1:1 to the tool in Component 2.

## Component 2 — Automatic surface: `verse_edit` tool

A model-invoked tool exposed **only on verse routes**, **constructive ops
only**, `source='llm'`.

### Schema (sketch)

```jsonc
{
  "name": "verse_edit",
  "description": "Create or modify forest-verse canon (entities, attributes, relations, events).",
  "parameters": {
    "op": "add_entity | update_entity | set_attribute | add_relation | add_event",
    "payload": { /* op-specific, mirrors _apply_op_inline */ }
  }
}
```

(One tool, `op` + `payload`, mirroring the loom proposal shape so validation is
shared via the existing `_validate`/coercion helpers in `loom.py`.)

### Per-channel mode

New registry keys (`config.py`):

- **`verseEditEnabled`** (channel bool, default **False**) — whether
  `verse_edit` is advertised in the channel's tool set at all.
- **`verseEditMode`** (channel string, default **`propose`**) — `propose` |
  `direct`:
  - `propose` → tool call enqueues a proposal via `add_proposal`; surfaces in
    `@verseproposals` for `@verseapprove`. (Today's effective behavior, now
    reachable live.)
  - `direct` → tool call applies immediately via the core **and** records an
    `approved` proposal row for audit via `apply_and_record_proposal`. No human
    step.

The operator enables `verse_edit` and sets `direct` only on trusted channels
(e.g. the forest channel); everywhere else it stays off / `propose`.

The loom is unchanged: it keeps proposing, and `verseAutoApplyThreshold`
governs its auto-apply as before.

## Data flow

```
operator  @versedit … ──► dispatcher (cap check, ref resolve, channel resolve)
                              └─► store: write_transaction → _apply_op_inline(privileged=True, source='operator')

LLM       verse_edit tool ─► mode = verseEditMode(channel)
                              ├─ propose: store.add_proposal(...)              → @verseproposals
                              └─ direct:  store.apply_and_record_proposal(source='llm')  → _apply_op_inline(privileged=False)

loom      (unchanged) ─────► store.apply_and_record_proposal / add_proposal     → _apply_op_inline(privileged=False, source='loom')
```

## Error handling

- Capability failure → standard Limnoria capability error.
- Unknown verb / malformed args → usage string for that verb.
- Unresolvable/ambiguous `<ref>` → error naming the problem and candidates.
- Core raises `LookupError` (no such id / retired target when not privileged),
  `ValueError` (bad op/status/missing key), `PermissionError` (constructive
  source attempting a privileged op) → mapped to friendly IRC replies.
- All mutations are single-transaction; a raised error leaves no partial write.

## Testing

Unit tests (mirror `tests/` patterns; store tests are pure-SQLite, no IRC):

1. **Core ops:** each new op applies correctly (`update_entity`, `set_status`,
   `edit_event`, `delete_event`, `delete_relation`).
2. **Privilege tier:**
   - operator/privileged may set `location`, retire+restore, delete events;
   - constructive (`llm`/`loom`) attempting `set_status`/`delete_*`/reserved
     key/retired target **raises** (assert the exception type).
3. **Command parser:** id-vs-name resolution, `::` summary splitting, `@`
   actor-id parsing, ambiguous-name error, channel defaulting.
4. **Tool dispatch:** `propose` mode creates a pending proposal and applies
   nothing; `direct` mode applies and records an `approved` proposal;
   `verseEditEnabled=False` hides the tool.
5. **Provenance:** rows created by each path carry the expected `source`.

## Out of scope (YAGNI)

- Cross-channel / shared global universe.
- Hard `DELETE` of entities (retire is soft-delete; only leaf event/relation
  rows are truly deleted).
- Web/GUI editor, bulk import/export beyond existing `@versedump`.
- Seeding the canonical "15 stinky lads" roster — trivial once `@versedit add`
  exists, but a separate operator action, not part of this build.

## Affected files (anticipated)

- `plugins/llm/src/llm/verse/store.py` — extend `_apply_op_inline` (new ops +
  `privileged` param); small helpers for event/relation edit/delete.
- `plugins/llm/src/llm/plugin.py` — `@versedit` dispatcher command + registration.
- `plugins/llm/src/llm/config.py` — `verseEditEnabled`, `verseEditMode` keys.
- `plugins/llm/src/llm/service.py` — advertise/dispatch `verse_edit` tool on
  verse routes; mode branching.
- `plugins/llm/src/llm/verse/loom.py` — reuse proposal validation for the tool
  payload (no behavior change to the loom itself).
- `tests/` — new store/command/tool tests.
