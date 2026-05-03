# Unified Assistant Plan Review

## Design Doc Review

**Overall**: This is a well-structured design that correctly identifies the
real problem — the artificial split between "chat" and "management" paths.
The route-profile model and `ToolSpec` registry are sound architectural
choices.

### Strengths

- **Route profiles** cleanly preserve behavioral differences without
  duplicating routing logic
- **ToolSpec with server-side policy** is the right abstraction — it
  prevents the chat route from becoming a permissions bypass
- **Direct mode + planner mode coexistence** avoids wasting LLM calls on
  deterministic commands
- **Grounding as a leaf tool** resolves the current tool/grounding conflict
  cleanly
- **Two layers of enforcement** (route-level preflight + tool-level checks)
  is defense in depth done right
- **Phased migration** keeps the bot usable throughout the refactor

### Issues to Address

1. **NickServ references** (lines 99, 174, 195 of the design doc): AfterNet
   has no NickServ. These should say "authenticated" or "identified"
   instead. Same in the plan doc references to "NickServ auth."

2. **Command prefix**: Both docs use `%` throughout, but the bot uses `@`.
   Cosmetic but should be accurate.

3. **Extra latency from grounding-as-leaf-tool**: Currently grounding is
   baked into the Gemini call — one round trip. The leaf-tool approach adds
   a round trip: planner decides to call `search_web` → internal Gemini
   grounding call → results come back → planner synthesizes a final answer.
   The design should acknowledge this tradeoff and confirm it's acceptable.

4. **NOT_META_SENTINEL replacement is unspecified**: The current meta path
   uses `NOT_META` to signal "this isn't a config request, fall through to
   ask." When `invalidCommand` routes through the unified facade with the
   `chat` profile, the sentinel concept disappears — but the design should
   state this explicitly, including what replaces the fallthrough behavior.

5. **System prompt design is missing**: The current meta system prompt tells
   the LLM *when* to respond `NOT_META` and scopes it to config-like
   requests. The unified assistant needs a different prompt strategy for the
   `chat` profile. This is important enough to warrant a section in the
   design doc.

6. **Conversation context across routes**: Currently `meta_completion` and
   `completion` handle context storage differently. The design should
   specify how the unified path stores context — does a mention-triggered
   multi-tool response go into conversation history the same way `@ask`
   does?

7. **`doPrivmsg` complexity risk**: This method already handles context
   tracking, spontaneous replies, and memory extraction. Adding mention
   detection plus dedupe on top makes it a candidate for extraction into
   smaller pieces. The design mentions dedupe but should also call out the
   refactoring need.

8. **Missing config toggle for mention routing**: If Phase 3 causes
   excessive bot responses (false-positive nick matches, unexpected
   triggering), there's no way to turn it off without reverting code. A
   `mentionEnabled` config toggle is cheap insurance.

### Open Questions — Recommendations

1. **Should `chat` expose `generate_image` immediately?** No. Defer until
   tool-level cost accounting is confirmed working. Reply with guidance to
   use `@draw` in the interim.

2. **Bulk destructive confirmation?** Only for bulk operations
   (`clear_memories`, `clear_instruction`). Single-item deletes are fine
   without confirmation — the user already named what they want deleted.

3. **Per-tool audit logging vs. database table?** Structured logs are
   sufficient for Phase 7. A database table is premature until you know
   what queries you actually need.

## Implementation Plan Review

### Structural Feedback

- **Task 1 is pure rename churn** and it's the riskiest place to start.
  Renaming `MetaToolExecutor` and `META_TOOLS` touches every test file
  before any value is delivered. Introduce `ToolSpec` first, then let
  naming evolve as you backfill specs. The rename can happen organically
  in Phase 6 when `meta` is demoted.

- **Tasks 3 and 4 are tightly coupled** — the facade (`assistant_request()`)
  and its first caller (`@ask` / `invalidCommand`) should be built together.
  An unused facade is untestable; a caller without a facade is premature.
  Consider merging these.

- **Task 6 (grounding tools) may block earlier phases**: If `@ask` currently
  depends on grounding, converting it to the unified facade in Task 4
  before grounding tools exist in Task 6 means grounding temporarily breaks
  for ask-via-facade. The plan should clarify: does the old grounding path
  keep working in `completion()` until Task 6 lands? If so, say that
  explicitly.

- **No rollback plan for Task 5 (mention routing)**: This is the
  highest-risk phase for user-visible regressions. Add a config toggle
  (`mentionEnabled`) that can disable the new path without redeploying.

- **Task 9 references doc paths** (`docs/guide/reference/commands.md`,
  `docs/guide/user/ai-commands.md`) — these should be verified to exist
  before starting that task.

### Suggested Revised Order

1. **ToolSpec registry with policy metadata** (current Task 2) —
   foundational, low risk
2. **Unified facade + first caller conversion** (merge Tasks 3–4) — proves
   the facade works
3. **Mention/PM routing with config toggle** (Task 5) — high visibility,
   needs escape hatch
4. **Grounding tools** (Task 6) — removes the tool/grounding conflict
5. **`@code` / `@draw` wrapper conversion** (Task 7) — straightforward once
   facade exists
6. **State command wrappers** (Task 8) — low urgency, already working
7. **`meta` demotion + naming cleanup** (merge Tasks 1 and 9) — rename
   happens here, not at the start
8. **Observability** (Task 10)

## Bottom Line

The architecture is sound. The main feedback is (a) don't lead with rename
churn, (b) specify the missing pieces (system prompt, context handling,
NOT_META replacement), and (c) add a config toggle for mention routing as a
safety valve.
