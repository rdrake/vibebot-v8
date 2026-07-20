# Verse Canon Layer — decoupling retrieval from roleplay mode

**Date:** 2026-07-20
**Status:** design approved; Slice 1 active

## Problem

Mentioning a verse trigger phrase (prod `#afternet`: `verseTriggerRegex = \bstinky lads\b`)
or any known canon entity flips the *entire* turn into verse **roleplay mode**:

| Dial | Normal chat | Verse turn (today) |
|------|-------------|--------------------|
| System prompt | assistant overlay | `build_verse_system_prompt` — "You *are* <avatar>…" |
| Model | grok-fast (non-reasoning) | `verseModel` = grok-4.3 (reasoning) |
| Tools | ~22 chat tools | 15 verse tools incl. `verse_storybook` ("reach liberally") |
| Profile | chat | `PROFILE_VERSE` |

Consequences observed in prod:

- **Surprise stories.** A bare mention lands in an in-character persona holding a
  storybook tool told to fire liberally → `vibebot story stinky lads …` produced a
  full illustrated story when the user only referenced the phrase.
- **Slowness.** Every mention pays the grok-4.3 reasoning tax (6–13 s/step, logged)
  plus story-gen (~10 s) plus 3× image-gen (~12 s each).
- **All-or-nothing.** There is no way to get canon *facts* into a normal answer
  without also assuming the avatar persona and the narrative toolset.

## Insight

Three concerns are welded to one boolean (`_verse_triggered`): **reading** canon,
**writing** canon, and **being in-character**. They should be separable:

- **Canon layer (read + write)** — a lore database that enriches *any* output and
  grows from conversation, independent of mode.
- **Roleplay mode** — an explicit hat: you *become* your avatar, with narrative
  tools, on the big model. Reads/writes the same canon layer; does not own it.

`build_story_world_context` + `scene_context` (shipped 2026-07-20) already give the
storybook the read side. This design generalises that: **read = deterministic
injection (cheap, no round-trip), write = model-judged tool, roleplay = opt-in.**

## Design (target state)

### 1. Canon layer — read (deterministic injection)
On a verse-enabled channel, when a message references canon (entity match or the
channel's retrieval regex), staple a **facts-only** lore block into the *same*
completion — no persona line, no model swap, no tool swap. Chosen over a
`lookup_lore` tool because a tool adds a round-trip (opposite of the slowness goal)
and Grok under/over-calls a large tool surface (per prior notes).

### 2. Canon layer — write (model-judged tool)
`record_canon(fact)` available on the normal path so canon accrues from ordinary
conversation, not only roleplay turns. (Generalises today's roleplay-only
`verse_record`.) Deferred to a later slice.

### 3. Roleplay mode — explicit door
Keep today's roleplay internals unchanged; it just stops auto-arming on a bare
mention. Entered explicitly (Slice 1: `@rp <text>` command — `@verse` was already
taken by the scene-readout command; single-fire, dodges the
double-dispatch/relay problem; richer doors — sticky toggle — possible later).

## Slices

- **Slice 1 (SHIPPED).** Read-injection on the chat path + demote the auto-trigger +
  minimal `@rp` roleplay door. Fixes the surprise-story and the grok-4.3 tax
  without touching tuned roleplay internals.
- **Slice 2 (SHIPPED).** Chat-path canon WRITE: `verse_record` made live on the
  normal chat path for opted-in avatars, gated behind `verseChatRecordEnabled`
  (default OFF), plus a terse recording nudge in the canon block. Realises the
  "record_canon" intent by generalising the existing `verse_record` tool rather
  than adding a new one (keeps the tool surface small). Ships DORMANT — rdrake
  flips the flag per channel when ready; canon-pollution risk stays behind the
  opt-in. Roleplay-only tools (act/move) remain denied on the chat path.
- **Slice 3.** Richer roleplay door (sticky toggle / auto-expiry) if wanted.

## Slice 1 — detail

**New:** `build_verse_context_block(store, avatar_id, message_text, *, roster_max_chars)`
in `verse/avatar.py` — the retrieval half of `build_verse_system_prompt` (roster +
message-matched cast + 1-hop relations + recent events) **without** the identity
line, persona line, or `VERSE_SCENE_MARKER` roleplay framing. Returns `""` when
nothing matches.

**Dispatch (`_dispatch_addressed_async` / `_verse_route_for`):**
- Roleplay route is taken only for the explicit `@verse` command (a new
  `verse_command` entry route), **not** for `_verse_triggered`.
- Otherwise the turn stays on the chat path. If `_verse_triggered` (now a
  *retrieval* signal), compute the canon block and pass it into `_ask_impl` as
  additive context (rides alongside `memories`/`channel_history`, not as a
  persona-bearing `system_prompt_override`).

**`@rp` command:** thin wrapper that runs `_dispatch_with_verse_routing` with
`force_roleplay=True` (reusing `_verse_route_for`'s builder), gated like verse
today (`llm.verse`), and sharing `@ask`'s rate-limit config (no new `rp*` keys).

**Unchanged:** `build_verse_system_prompt`, denial/degradation guards, prefix
caching, storybook (`@story` + its `world_context`/`scene_context`), `verse_record`
on roleplay turns.

## Risks

- **Stranding roleplay.** Demoting the trigger without a door makes roleplay
  unreachable → the `@verse` door ships *in the same slice*.
- **Behaviour change for fc42/Forest.** Ambient "just talk and your avatar answers"
  becomes "`@verse …` to act in-character." Called out; sticky toggle (Slice 3) can
  restore the ambient feel if missed.
- **Injection cost.** The facts block rides every canon-referencing chat turn;
  reuse the existing `roster_max_chars` cap.

## Testing

- `build_verse_context_block`: includes matched cast + relations, excludes the
  identity/persona lines, `""` on no match, respects the char cap.
- Dispatch: `_verse_triggered` message → chat path + injected block, **not** the
  verse route/model/tools; `@verse` → verse route.
- Regression: bare mention no longer advertises the storybook force path.
