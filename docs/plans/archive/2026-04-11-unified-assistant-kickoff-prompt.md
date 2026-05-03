# Unified Assistant Kickoff Prompt

```text
Implement the unified assistant refactor for VibeBot v8, starting with the first value-bearing milestone only.

Read and follow:
- AGENTS.md
- docs/plans/2026-04-11-unified-assistant-design.md
- docs/plans/2026-04-11-unified-assistant-plan.md
- docs/plans/2026-04-11-unified-assistant-review.md

Scope for this pass:
1. Implement Phase 1 / Task 1 from the revised plan:
   - introduce server-side tool policy metadata (`ToolSpec` or equivalent)
   - convert the existing assistant/meta tools to a registry-driven shape
   - enforce policy in executor dispatch, not just in prompts
2. Implement Phase 2 / Task 2 from the revised plan:
   - add request context
   - add a shared `assistant_request()` facade
   - convert `@ask` and `invalidCommand` to use the facade
   - preserve current no-double-preflight behavior
   - preserve current ask grounding behavior via the temporary bridge until grounding leaf tools exist
3. Do not implement mention routing yet.
4. Do not demote or rename `meta` yet unless a minimal compatibility alias/helper is needed internally.
5. Do not do rename churn up front. Favor additive refactoring that keeps the current code working.
6. Keep existing explicit command behavior stable unless the plan explicitly changes it.

Important design constraints:
- Route-level preflight stays in `plugin.py`.
- Tool-level policy enforcement must exist even if route-level preflight already passed.
- Unknown addressed text should become a normal `chat` request through the shared facade; there is no new `NOT_META` sentinel path.
- Keep current context storage behavior for `@ask`.
- Do not break current grounding support for ask during this pass.
- Do not remove the dedicated summarization path yet unless it becomes trivial as part of this work; summary cleanup is planned later when large tool outputs flow through the planner’s final turn.

Implementation expectations:
- Build incrementally with tests.
- Add or update tests for the new registry/policy behavior and for the new shared facade path.
- Avoid reverting unrelated worktree changes.
- Prefer minimal, reviewable commits in code shape even if you are not creating git commits.
- If you hit an ambiguity, follow the design doc first, then the review doc.

Verification:
- Run focused tests for touched areas first.
- Then run `make lint`
- Then run `make typecheck`
- If the changes are sufficiently contained, stop there and report what remains for the next phase.
- If broader checks are cheap enough, run `make preflight`.

Deliverable:
- Implement the code for this milestone.
- Summarize what changed, what tests passed, what remains for the next phase, and any design adjustments needed.
```
