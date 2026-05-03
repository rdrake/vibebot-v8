---
purpose: Codex review prompt for the reminder simplification plan
target-plan: 2026-04-30-reminder-simplification-plan.md
date: 2026-04-30
---

# Codex review prompt

Paste the codeblock below into a Codex session in this repo (`/Users/rdrake/workspace/afternet/vibebot-v8`). Codex should have read access to the repo. The prompt asks for an independent, second-opinion review of the plan after it has already been revised once based on a Claude code-reviewer subagent pass.

```
You are reviewing an implementation plan saved at
`docs/plans/2026-04-30-reminder-simplification-plan.md` in the
vibebot-v8 repo. Your job is an independent second-opinion review —
the plan has already been revised once based on a Claude code-review
agent pass (see the "Revision note" at the top of the plan). Don't
just confirm prior findings; look for what *that* review missed.

## Background

A 21-commit burst (commits cf8ad30..74772e6, all touching
`plugins/llm/`) added significant complexity to the LLM plugin's
reminder system: action_prompt column for LLM-action reminders, a
new `remind_action` profile, chain-tracking columns + caps, watch
mode via parser-embedded parentheticals, IRCv3 reaction acks paired
with a `[silent]` model-compliance contract, an Identity dataclass
splitting nick from account, set_reminder visibility expanded to
chat/code/draw/remind_action, and a `cancel_all_reminders` bulk tool.

The plan proposes simplifying this in two sequenced PRs:
- PR A: nine mechanical refactors (no behavior change). Replace the
  in-memory 8-tuple with the existing `ReminderRow` NamedTuple,
  extract `_ack` and `_reminder_fns` helpers, narrow set_reminder
  visibility back to chat-only, merge `_check_rate_limit_silent`
  into a `silent=` param, drop the unused `chain_id` column, finish
  the Identity migration, drop commit-narrating comments. ~44 test
  fixture sites need rewriting (Task A1b).
- PR B: structural redesign. Promote recurrence (numeric seconds OR
  RFC 5545 RRULE — two-column shape, exactly one populated) and
  watch_mode from parser parentheticals to first-class columns.
  Recurring reminders reschedule mechanically (no LLM tool call) at
  fire time. Move post-tool silence from a `[silent]` chat-prompt
  contract into reply-path suppression. Drop the 30-day chain TTL.

## What to check

Read the plan top to bottom. Then read the actual current state of:
- `plugins/llm/src/llm/plugin.py`
- `plugins/llm/src/llm/persistence.py`
- `plugins/llm/src/llm/assistant.py`
- `plugins/llm/src/llm/service.py`
- `plugins/llm/tests/conftest.py`
- `plugins/llm/tests/test_reminders.py`
- `AGENTS.md`

Then assess:

1. **Correctness of cited file:line references.** The plan cites
   many specific lines. Spot-check 5-10 of the load-bearing ones
   and flag any that drifted by more than ~10 lines or describe a
   pattern that's no longer there.

2. **Sequencing and migration safety.** PR A Task A6 drops
   `chain_id` (v11). PR B Task B1 bundles a v12 migration that drops
   `chain_started_at` and adds `recurrence_seconds`,
   `recurrence_rrule`, `watch_mode`. Are the SQL operations safe in
   that order on a populated production DB? Is there a risk of
   in-flight transaction or constraint trouble?

3. **B0.5 in-flight migration strategy.** The plan recommends
   option (b) — graceful degradation, keep `_set_reminder_capped`
   for one release. Is this the right call? Are there edge cases
   where a pre-v12 reminder with a parenthetical-encoded recurrence
   would now fail to reschedule (e.g. because the legacy path
   reads from `action_prompt` but the v12 schema migration is
   independent of action_prompt content)? Or worse — is there a
   case where a reminder ends up *both* mechanically rescheduled AND
   LLM-tool-rescheduled (double-fire)?

4. **B4 mechanical reschedule with RRULE.** The plan uses
   `dateutil.rrule.rrulestr` for calendar cadences. Is `dateutil`
   actually a transitive dep? Run `uv tree | head -50` to verify.
   If not, the plan must add it as a direct dep before B4 — flag
   this. Also: `rrulestr(...)` parsing has subtle DST/timezone
   behavior; the plan computes `next_fire = rule.after(...)` from
   `datetime.fromtimestamp(now)` (naive local time). Is that going
   to produce wrong fires across DST transitions?

5. **A2 `_ack` helper.** The plan's table at Task A2 Step 1 lists
   five sites with their emoji, fallback, and prefixNick values.
   Verify by reading `plugin.py:3094-3242`. Are all five truly
   covered? Did the plan miss a sixth?

6. **A5 `_check_rate_limit` merge.** The plan's merged signature
   accepts `irc: callbacks.Irc | None` and uses an `assert
   irc is not None` in the non-silent path. Is the assert safe (i.e.
   is there any path where a caller could pass `silent=False` and
   `irc=None`)? Should this be a real runtime check rather than an
   assert that vanishes under `python -O`?

7. **PR B Task B5 reply-path suppression.** The plan says: "After
   the assistant turn returns, if the meta loop's last successful
   tool call was a reminder mutation AND the assistant text is
   empty/whitespace/an acknowledgment phrase: skip irc.reply." How
   robust is "acknowledgment phrase" detection? Could legitimate
   short replies ("Done!", "Okay") get suppressed when the user
   actually wants to see them? Is there a cleaner signal (e.g.
   "the meta loop ended on a reminder tool call AND the model
   produced no text after that tool call" — strict empty)?

8. **What's missing entirely.** What did *both* prior reviews
   (Claude's plus the original plan-writer's) miss? Examples to
   prompt your thinking — but don't limit yourself:
   - Is there a concurrency hazard around `chain_position`
     increments under the new mechanical path vs the legacy
     LLM-tool-call path during the B0.5 transition?
   - Does `cancel_all_reminders` interact correctly with mechanical
     reschedule mid-fire?
   - Are there observability/logging hooks that should be added
     (e.g. structured log when a mechanical reschedule fires) that
     the plan omits?
   - Does the plan need a rollback story? If B4 ships and an RRULE
     parse fails for a row created by an older parser version, is
     there a graceful degradation path?

9. **Granularity sanity check.** The plan's hard constraint says
   each task is 2-5 minutes. A1a alone has 8 modify sites. Is the
   granularity achievable, or does the plan over-promise the
   bite-size?

## Output

Produce a single review with three buckets, prioritized:

- **Blockers** — would cause the plan to fail or ship broken
  behavior. file:line + concrete suggested fix.
- **Significant gaps** — needs a new task or expanded step.
  description + suggested addition.
- **Nits** — line drift, wording, missing minor steps.

If the plan is solid and ready to execute as-is, say so explicitly.
Don't manufacture findings. Be concrete and cite file:line.

Cap the review at ~800 words.
```

## Where this prompt expects to land

Paste the codeblock above into a Codex CLI session run from the repo
root. Codex's output is the review. Save the review (manually) to
`docs/plans/2026-04-30-reminder-simplification-codex-review.md`
once received, so future readers can trace the revision lineage.
