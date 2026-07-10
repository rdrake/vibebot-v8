# Limnoria tool bridge — Phase 2 initial plan

**Date:** 2026-05-02
**Predecessor:** `docs/plans/2026-05-02-limnoria-tool-bridge-plan.md` (Phase 1 design),
`docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md` (Phase 1 impl).
**Companion:** `docs/plans/2026-05-02-settings-config-simplification-findings.md`
(authoritative for Task 5 — the config consolidation work).
**Status:** initial plan — implementation plan to follow once scope is locked.

## Goal

Two parallel work streams, both motivated by the post-Phase-1 reality
that the LLM plugin is one assistant + a few native capabilities + the
Limnoria bridge — not the command-era collection of independent
features it grew from:

1. **Bridge expansion (Tasks 1–4).** Let the LLM safely use a wider
   set of stock Limnoria plugins, and let users compose simple agentic
   workflows by scheduling LLM tasks for future execution.
2. **Config consolidation (Task 5).** Replace the command-era model /
   API-key / system-prompt registry surface with capability-based
   names that match how the assistant actually works today. Designed
   in the companion findings doc; this plan ratifies it as Phase 2
   scope and sets sequencing.

No memory subsystem changes in Phase 2. Codex investigation found
stock plugins are not a clean replacement; revisit after Phase 2
ships and the bridge has more soak time.

## Out of scope

- Replacing the LLM-native memory subsystem with Factoids or any stock
  plugin. The system does extraction, dedupe/cleanup, and per-turn
  prompt injection — only the storage layer maps to Factoids cleanly,
  and Factoids is channel-scoped while our memories are per-nick.
  Defer indefinitely; re-evaluate after the bridge soak.
  (Note: Task 5's config consolidation does fold
  `memoryExtractionModel` / `memoryCleanupModel` / `memoryApiKey`
  into the `assistant*` settings — that's a registry rename, not a
  subsystem change.)
- Replacing the LLM-native NL reminder system (`set_reminder`,
  `delete_reminder`, `cancel_all_reminders`). Phase 2 adds
  `schedule_llm_task` *alongside* it, not as a replacement. The LLM
  scheduler does NL parsing, watch mode, and structured-recurrence
  rescheduling that Scheduler does not.
- Retiring `generate_image` or the LLM's `fetch_url`. Codex confirmed
  these are not pure pass-throughs — `generate_image` calls our own
  pipeline; `fetch_url` LLM-summarizes; `Web.fetch` is denied as an
  SSRF risk. They stay as native tools.
- Per-command tools vs. generic dispatcher. Decision deferred until
  more soak data shows whether the LLM picks correctly with the
  generic form.

## Tasks

### Task 1 — Per-command mutation classification (prerequisite)

**Why first:** the bridge today gates only at plugin granularity. Once
we expose `Later`, `Note`, or any plugin with both reads and writes,
allowlisting the plugin is too coarse — we need to gate writes
separately. This is a hard prerequisite for Task 2 (which expands the
default allowlist to include such plugins). The dependency chain is
1 → 2 → 4. Task 3 does *not* depend on Task 1 in the current design
(see Task 3 design below).

**Design:**

- Add `MUTATING_COMMANDS: frozenset[tuple[str, str]]` to
  `limnoria_bridge.py` (mirrors `DENY_COMMANDS` shape — lowercase
  `(canonical_plugin, leaf)` tuples). Manual classification, not
  heuristic — the stock plugin set is finite and stable.
- Add channel registry `bridgeAllowMutating: Boolean` (default
  `False`). When `False`:
  - `enumerate_commands` skips entries whose `(canonical, leaf)` is in
    `MUTATING_COMMANDS`. The LLM never sees them in the tool's
    available-commands description.
  - `dispatch` rejects mutating commands with
    `{"error": "denied: write commands disabled"}`. Defense in depth in
    case the LLM tries one anyway from training memory.
- When `True`, mutating commands are subject only to the existing
  `DENY_*` lists and Limnoria capability checks.

**Initial classification (representative, not exhaustive — full list
in implementation plan):**

| Plugin     | Mutating leaves                        | Read-only leaves                       |
| ---------- | -------------------------------------- | -------------------------------------- |
| Later      | `tell`, `remove`, `undo`               | `notes` (lists pending)                |
| Note       | `send`, `unsend`, `setnotify`          | `note`, `next`, `unread`, `search`     |
| Karma      | `add`, `inc`, `dec`, `clear`           | `karma`, `most`                        |
| QuoteGrabs | `grab`, `ungrab`                       | `random`, `say`, `list`, `search`      |
| Todo       | `add`, `remove`                        | `todo`                                 |
| RSS        | `add`, `remove`, `announce add/remove` | `rss`, `info`, `announce list`         |
| Quote      | `add`, `remove`, `change`              | `quote`, `random`                      |
| Factoids   | `learn`, `forget`, `change`, `lock`    | `whatis`, `random`, `info`             |

The implementation plan will source the canonical list by walking each
allowlisted plugin's `__init__.py` / `plugin.py` and reading the
docstrings — we want this checked in, not derived at runtime.

**Open questions:**

1. Should the available-commands description include a footer like
   "(write commands hidden — set bridgeAllowMutating True to expose)"
   when `bridgeAllowMutating` is False and the channel allowlist
   contains a plugin with both kinds? Could help the LLM generate
   sensible refusals. Lean: yes.
2. Per-command override: a `bridgeAllowedMutatingCommands` list for
   "expose `Karma.add` but no other writes." Lean: defer to v2 of
   this task — wait for a real ask.
3. What about commands that are *config* mutating (e.g. `RSS announce
   add`) vs. *user-data* mutating (`Note send`)? Both are writes;
   treat both the same in v1. Operators who want fine-grained control
   already have the per-channel allowlist + Limnoria capability
   system.

**Verification:**

- Unit test: `enumerate_commands` returns no mutating entries when
  `bridgeAllowMutating=False`, returns them when True.
- Unit test: `dispatch` rejects a mutating command with the right
  error envelope when disabled.
- Integration test: with `Later` allowlisted and mutating disabled,
  the bridge tool description lists `Later.notes` but not
  `Later.tell`.
- Manual: in `#test`, `bridgeAllowMutating=False`, ask the bot to
  leave a note for someone via the bridge → bridge call attempt is
  refused with the gate error. Flip the toggle, repeat, observe the
  same call succeed via `Later.tell`. (This verifies the bridge gate,
  not the LLM's tool-selection — `set_reminder` and other LLM-native
  tools are out of scope for this gate either way.)

**Done when:** the gate is wired into both `enumerate_commands` and
`dispatch`, the manual test above produces the expected refuse/accept
pair, and the canonical mutating-command list is committed alongside
the code.

### Task 2 — Default read-only allowlist

**Why:** Phase 1 ships `bridgeAllowedPlugins=[]` because there was no
mutation gate. With Task 1 in place, defaulting to a curated read-only
set is safe and gives operators an immediate payoff without forcing
per-channel config.

**Design:**

- Change `bridgeAllowedPlugins` default from `[]` to a curated list:

  ```
  ["Misc", "Time", "Math", "Utilities", "Seen", "Web",
   "Later", "Note", "Karma", "QuoteGrabs", "RSS", "DDG"]
  ```

  Rationale: each is either pure-read or has reads gated behind Task 1.
  `Web` is safe because `Web.fetch` is in `DENY_COMMANDS`; only `title`
  / `headers` / `urlquote` etc. are exposed.

- Bump `bridgeEnabled` to remain `False` by default — operators must
  still opt the channel in. The change is "when you opt in, you get a
  reasonable starter set," not "every channel gets the bridge."

- Update the `bridgeAllowedPlugins` registry help text to describe what
  ships by default and how to override.

- Update `AGENTS.md` and the existing `docs/...limnoria-tool-bridge`
  docs to reflect the default.

**Open questions:**

1. Operators who set the value to `[]` explicitly in Phase 1 keep
   their explicit empty (Limnoria's registry distinguishes set-to-empty
   from default). Verify this is true for `SpaceSeparatedListOfStrings`
   before shipping.
2. Should `DDG` ship enabled? It hits an external service, but it's
   read-only and matches the LLM's existing `search_fn` capability.
   Lean: yes — gives the LLM a stock alternative to ground answers.
3. `Anonymous`, `Praise`, `Lart`, `Dunno`: all mutating-only or
   action-emitting. Skip.

**Sub-task — RSS surfacing (was Task 4):** `RSS` is included in the
default list above. Mutating commands (`add`, `remove`, `announce
add/remove`) are gated by Task 1; read-only `rss <feed>` and `rss
info` work out of the box. `feedparser` is already shipped (`f1efb92`).
Manual smoke: `@load RSS`, then `@ask what's the latest from
hackernews?` → bot uses `RSS.rss` against a configured feed (or
refuses if no feed registered).

**Verification:**

- Unit test: with the bridge enabled and no per-channel allowlist
  override, `enumerate_commands` yields commands from each default
  plugin (filtered through Task 1).
- Manual: fresh channel, `bridgeEnabled True`, no other config →
  `@ask seen rdrake?` exercises `Seen.last`; `@ask convert 5 km to
  miles` exercises `Math.convert`.

**Done when:** the default value of `bridgeAllowedPlugins` reflects
the curated list, existing operator overrides are unaffected (verified
on the dev bot with a pinned config), and AGENTS.md / bridge docs
describe the new default.

### Task 3 — `schedule_llm_task` tool (Scheduler-as-agent)

**Why:** users want to compose agentic workflows ("every Monday at 9
check my open PRs and tell me which are stale"). The LLM scheduler
delivers fixed text; we need a primitive that schedules a future
*LLM invocation* with bridge tools available.

**Design — the persistence shortcut:** Limnoria's Scheduler persists
events as `{command: str, msg: IrcMsg, time, ...}` pickled to disk.
At restart it re-registers periodic events with the persisted `msg`,
firing the persisted command string as if the original sender had
typed it again. **Identity comes from the persisted `msg` for free.**
That gives us a clean shape: the scheduled command is just `@ask
<prompt>`, sent as if the original creator typed it.

**`schedule_llm_task` is a native LLM tool, not a bridge dispatch.**
It calls Scheduler's Python API directly (e.g. `schedule.addEvent` /
`schedule.addPeriodicEvent` — verify exact entry points at impl
time, since the public IRC commands `Scheduler add` / `Scheduler
repeat` are user-facing wrappers around private methods). Going
through the bridge would self-block once Task 1 lands — `Scheduler
add` is a mutating command — so the bridge dispatch path is the
wrong route for this tool. Native tool, native call. This is why
Task 3 does not list Task 1 as a hard prerequisite.

**Tool-selection guidance vs. `set_reminder`:** users can ask for
similar-sounding things from either tool ("remind me to check the
build" vs. "every Monday check my open PRs"). The split:
`set_reminder` delivers a fixed string at fire time; `schedule_llm_task`
fires an `@ask` that re-enters the LLM. The tool descriptions must
make the distinction explicit so the LLM picks the right one. Rough
rule: if the user is asking for an *action* at fire time (which
implies tool use), `schedule_llm_task`; if they want fixed text or
a notification, `set_reminder`. Both stay shipped.

**Tool surface:**

```python
{
    "name": "schedule_llm_task",
    "description": (
        "Schedule a future LLM task. At fire time the bot runs an "
        "@ask invocation as you, with full bridge access. Use this "
        "for periodic agentic work; for plain text reminders, use "
        "set_reminder."
    ),
    "parameters": {
        "when_natural": "string — e.g. 'in 30 min', 'every Monday 9am'",
        "prompt":       "string — the @ask text to run at fire time",
        "reply_target": "string | null — channel or nick (default: "
                        "the channel this was scheduled in)",
    },
}
```

Companion tools:

- `list_scheduled_llm_tasks()` → IDs, when, prompt, target, owner
- `cancel_scheduled_llm_task(id)` → owner-scoped cancel

**Internal flow at create:**

1. NL-parse `when_natural` via the existing reminder parser (reuse,
   do not fork). One-shot → datetime; recurring → interval seconds.
2. Build the command string: `@ask <prompt>` (or `@askscheduled
   <prompt>` if we need a separate command for depth-cap purposes —
   see open question below).
3. Call Scheduler's Python API directly (`schedule.addEvent` for
   one-shot, the equivalent periodic entry point for recurring —
   exact names verified at impl time). Pass the *creator's* `msg`
   so Scheduler persists it for fire-time identity.
4. Write a row to a small VibeBot table mapping
   `(scheduled_id, creator_nick, account, prompt, when, target)` so
   `list_scheduled_llm_tasks` can show owner-attributed entries
   (Scheduler's own list isn't owner-scoped).

**Internal flow at fire (free from Scheduler):**

1. Scheduler's stored closure runs `_callCommand` with the original
   creator's `msg` and the `@ask <prompt>` tokens.
2. The @ask path runs as normal — capability checks pass (real `msg`),
   bridge tools are available, the LLM does its agentic work.
3. Reply target: `@ask` replies to `msg.channel`. If the user wanted
   a different target, we'd need to override that — see open
   question.

**Loop hazard mitigation:**

- Define a depth cap. A scheduled `@ask` whose LLM body schedules
  another `@ask` is a loop. Cap depth at 1 (a scheduled task can
  schedule nothing further).
- Scope: the cap applies only to `schedule_llm_task` recursion. A
  fired schedule can still call `set_reminder`, the bridge tool, and
  any other LLM-native tool — those are not loops, just normal tool
  use. Make this explicit in tool description so the LLM doesn't
  over-restrict itself.
- Implementation candidate: use `msg.tags` to mark the synthesized
  fire-time `msg` with `llm_schedule_depth=1`. The tool refuses
  `schedule_llm_task` when `msg.tags.get("llm_schedule_depth", 0) >=
  1`. **Risk:** `msg.tags` may not survive pickle round-trip; verify.
- Fallback: a separate command `@askscheduled` that mirrors `@ask` but
  enters with depth=1 already set. Trivial alias; no tag dependency.
  Lean toward this if tags don't pickle cleanly.

**Per-creator budget:**

- Add `bridgeScheduledTaskLimit: Integer` (channel-int, default 5).
  Enforced at create-time: if the creator already has N active
  scheduled-llm-tasks (queried from our tracking table), refuse with
  `{"error": "scheduled task limit reached"}`.
- Existing `askRateLimit*` quotas already cover the *fire-time* @ask
  invocation — verified during Codex pass. Scheduled tasks count
  against the user's normal @ask budget when they fire. No change
  needed there.

**Capability-gating at fire time (decision, not open):** the fired
@ask runs through the @ask wrapper which calls
`checkCommandCapability(msg, ...)`. If the creator has lost the `ask`
capability (anti-flood ban, capability removed by an admin, etc.),
the dispatch raises and Scheduler's caller logs and skips. **Decision
for v1:** rely on this default behavior — log the failure and let the
schedule keep firing. Adding self-cancel-on-revoke is a real behavior
change, not a default; defer until we see it actually happen in the
wild.

**Open questions:**

1. **Reply target override.** Default behavior (reply to the channel
   the schedule was created in) is fine for v1. Cross-target
   scheduling ("schedule in #foo, deliver to me in DM") needs to
   override `msg.args[0]` since `irc.reply` infers the target from
   it. Concrete mechanism: at fire time, wrap Scheduler's stored
   `msg` with a copy whose `args[0]` is the requested target before
   passing it into the dispatch path. This also requires a permission
   check at create time — does the creator have the right to send
   messages to the target channel/nick? Defer the wrapper code to
   the impl plan, but commit to the wrap-not-mutate approach so the
   stored `msg` stays intact for the next fire.
2. **`msg.tags` pickling.** Need to confirm tags survive
   Scheduler's pickle. If not, the `@askscheduled` alias path is
   simpler.
3. **Recurring event storage.** Scheduler's pickle persists periodic
   events (verified at `Scheduler/plugin.py:117`); Codex's earlier
   claim that periodic events were in-memory was wrong. So recurring
   `schedule_llm_task` survives bot restart for free.
4. **Listing/cancel UX.** Should the LLM offer these as tools, or
   should they be human bot commands (`@scheduled list` etc.)? Both
   are useful. Lean: tools first (the LLM can self-serve), human
   commands later.
5. **Identity drift.** If the creator's nick changes (or they
   disconnect/reconnect), the persisted `msg.nick` is stale. The
   account-tag persistence might fix this — need to verify what's in
   `msg.server_tags` after pickle reload.

**Verification:**

- Unit test: schedule a one-shot task, fast-forward (mock time),
  observe the @ask path enters with creator identity.
- Unit test: depth cap rejects nested `schedule_llm_task` from a
  fired task.
- Unit test: per-creator limit enforced at create.
- Integration test: schedule "in 60s ping me", observe reply in the
  origin channel after the delay.
- Manual: schedule a recurring task, restart the bot, confirm the
  task fires after restart with the right identity.

**Done when:** `schedule_llm_task` + companion list/cancel tools are
shipped, the depth cap and per-creator budget hold under unit tests,
the manual recurring-restart test passes, and the tool descriptions
make the `set_reminder` / `schedule_llm_task` distinction clear
enough that the LLM picks correctly in 3+ informal smoke tests.

### Task 5 — Config consolidation (capability-based settings)

**Why:** the registry surface still names settings after the old
user-facing commands (`askModel`, `drawModel`, `metaModel`,
`grokModel`, etc.) even though every one of those flows now runs
through the same assistant loop with tool access. Operators can't
predict what `metaModel` does without reading the code. The findings
doc proposes a capability-based surface (`assistantModel`,
`imageModel`, `codeModel`, `searchModel` + matching keys) and a
single-release compatibility window.

**Source of truth:** the design lives in
`docs/plans/2026-05-02-settings-config-simplification-findings.md`.
It is complete enough to drive an implementation plan as-is. This
section is for sequencing + risk callouts only — do not duplicate
the design here.

**Scope summary (from the findings doc):**

- New settings: `assistantModel`, `assistantApiKey`,
  `assistantSystemPrompt`, `imageModel`, `imageApiKey`. Keep
  `codeModel` / `codeApiKey` / `codeSystemPrompt` as-is. Keep
  `searchModel` / `searchApiKey` as-is.
- Map old names to new ones via a one-release compatibility shim
  (read-old-if-new-is-empty), then delete the old registrations the
  release after.
- Remove or deprecate `%g` and its three `grok*` settings.
- Update fixtures, docs, and error messages to stop teaching
  command-era names.

**Sequencing relative to bridge tasks:**

- Independent of Tasks 1–4. No code overlap; touches `config.py`
  and the lookup sites in `service.py` / `plugin.py`. Can ship in
  parallel with bridge work, or before, or after.
- Recommendation: ship Task 5 *before* Task 3 (`schedule_llm_task`).
  Task 3 introduces a new model-using code path (the fired @ask);
  doing the rename first means Task 3 reads `assistantModel` from
  day one rather than `askModel` → `assistantModel` later.
- No dependency on Task 1's mutation gate or Task 2's allowlist
  default.

**Risk callouts not covered in the findings doc:**

1. **Per-channel overrides.** Several of the renamed settings are
   `registerChannelValue`. Existing per-channel overrides (e.g. a
   channel that pinned `askModel` to a non-default model) must
   continue to work through the compatibility window. Verify the
   compat-shim resolves channel-scoped reads correctly, not just
   global.
2. **Test fixtures.** Several plugin tests construct conf with
   command-era names. The compat shim will keep them passing in the
   short term; the cleanup step needs to update them in the same PR
   that removes the old registrations, or the next release will
   break tests.
3. **Bridge debug footer.** The footer added to `@ask` replies in
   `b5bc85e` doesn't depend on any of these settings, so no churn.
4. **`metaModel` chain.** The findings doc notes the assistant loop
   currently looks up `metaModel` then falls back to `askModel`.
   Removing this two-step lookup is part of the consolidation —
   confirm during impl that no other code path relies on the
   distinction.

**Recommendation for the impl plan (not in the findings doc):** when
the compat shim falls back to an old key, log a deprecation warning
once per process per key. Helps operators discover what they need to
rename without spamming.

**Verification:**

- Unit test: with only old settings configured, the resolver returns
  values mapped to the new names.
- Unit test: with both old and new configured, the new wins.
- Unit test: with only new configured, old reads return empty/default
  cleanly (no exception).
- Integration test: a fresh deploy with a freshly written
  `assistantModel` runs a chat turn end-to-end without any old key
  set.

**Done when:** new settings + compat shim ship in 5a; every internal
lookup uses the new names; docs and fixtures use the new names; the
old registrations are removed in 5b a release later.

## Sequencing

Two parallel tracks. Tracks are independent; the only coupling is
that Track A's Task 3 *prefers* (does not require) Track B's Task 5a
to land first, to avoid renaming a model lookup that was just written.

**Track A — bridge expansion:**

1. **Task 1** (mutation gate) — must ship first within Track A.
   Hard prerequisite for Task 2 (which expands the default allowlist
   to include plugins with writes). Single PR.
2. **Task 2** (default allowlist, including the RSS sub-task) —
   ships after Task 1 lands. Single PR.
3. **Task 3** (`schedule_llm_task`) — independent of Task 1 (it
   calls Scheduler natively, not via the bridge). Prefers Task 5a
   so the new code reads `assistantModel` directly. Larger PR; can
   develop in parallel with Task 2.

**Track B — config consolidation:**

4. **Task 5** — independent of all bridge tasks. Two PRs: (a)
   introduce new settings + compat shim; (b) remove old registrations
   one release later.

Recommended overall order: 1 → 2 → 5a → 3 → 5b.

## Cross-cutting risks

- **Pre-existing operators.** Anyone who set `bridgeAllowedPlugins`
  explicitly in Phase 1 keeps their config. Need to verify Limnoria's
  default-vs-explicit semantics for `SpaceSeparatedListOfStrings`
  before merging Task 2.
- **Soak data may flip the per-command-tools decision.** If we see
  the LLM picking the wrong command/args frequently with the generic
  dispatcher, we may want hand-written per-command tools for the
  most-used 3–5 commands. That decision is still deferred — Phase 2
  doesn't block on it.
- **Mutating gate adds a state space.** `bridgeAllowMutating` × per-
  channel allowlist × DENY lists × capability checks. Implementation
  must keep the truth table simple and tested.
- **Fire-time state drift (Task 3).** Scheduler persists `msg` at
  create-time but fires it potentially much later, when the bot's
  state may differ — the user disconnected, the channel was parted,
  the bot reconnected to a different network. **Decision for v1:**
  best-effort dispatch. If `irc.getCallback("LLM")` or the target
  channel is unavailable, log and skip; do not retry, do not
  auto-cancel. Periodic events keep firing on their schedule and
  may succeed on a later attempt. Revisit if this produces noisy
  logs in practice.

## What this plan does not commit to

- Final mutating-command list (impl plan will enumerate).
- Concrete tool/command names (`schedule_llm_task` vs. `schedule_ask`
  vs. ...) — bikeshed at impl time.
- Migration of existing `set_reminder` rows to anything else (no
  migration; both systems coexist).
- Per-creator scheduled-task quota number (default 5 is a placeholder).
- The compatibility-shim flavor for Task 5 (registry alias vs. lookup
  helper). The findings doc proposes "read-old-if-new-is-empty" — fine
  as a constraint, but the implementation choice is left to the impl
  plan.

## Next steps

- User review of this plan.
- If approved, write implementation plans in this order:
  1. Task 1 (mutation gate) — the bridge-track prerequisite.
  2. Task 5a (config consolidation, introduce-new-settings phase) —
     can be drafted in parallel with Task 1 since they don't overlap.
  3. Task 2 (default allowlist + RSS sub-task).
  4. Task 3 (`schedule_llm_task`) — after Task 5a is merged.
  5. Task 5b (remove old registrations) — a release cycle after 5a.
