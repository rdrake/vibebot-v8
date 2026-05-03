---
purpose: Kickoff prompt for a fresh Claude Code session to execute the reminder simplification plan
target-plan: 2026-04-30-reminder-simplification-plan.md
date: 2026-04-30
---

# Kickoff prompt — execute the reminder simplification plan

Open a **new Claude Code session** in the repo root
(`/Users/rdrake/workspace/afternet/vibebot-v8`) and paste the
codeblock below as the first message. The plan was written for
`superpowers:subagent-driven-development` — Claude will dispatch a
fresh subagent per task, review between tasks, and stop at PR
boundaries.

```
Execute the implementation plan at
`docs/plans/2026-04-30-reminder-simplification-plan.md`.

REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
Dispatch a fresh subagent per task, review the diff between tasks,
and STOP at the end of PR A (after Task A8 / "PR A finalize") for
my review before starting PR B.

Context:
- The plan simplifies the LLM plugin's reminder system after a
  21-commit burst (commits cf8ad30..74772e6).
- It has been through two review rounds: a Claude code-reviewer
  subagent pass and a Codex second-opinion pass. The "Revision note"
  at the top of the plan summarizes both.
- PR A is mechanical refactors (no behavior change). PR B is
  structural and includes schema migrations, RRULE recurrence, and
  reply-path suppression. Do PR A first.

Hard rules:
1. Each task is a separate commit. Do NOT squash, do NOT amend.
2. Run `uv run pytest plugins/llm/ -q` and `uv run ruff check
   plugins/llm/` before every commit. They must pass.
   (Exception: A1a, where tests are expected to fail until A1b
   lands. Follow the plan's instructions there.)
3. Do NOT skip pre-commit hooks. If a hook fails, investigate and
   fix the underlying issue, then create a new commit.
4. Do NOT push to origin. Do NOT open a PR. Stop at the end of PR A
   and wait for me.
5. If a task's actual file:line state diverges from what the plan
   says by more than ~10 lines or the described pattern is no
   longer there: stop, summarize the divergence, and ask before
   proceeding.
6. The plan's hard-constraints section lists a migration safety
   procedure for any task that does a `DROP COLUMN` or
   `ADD COLUMN` (A6, B0.6, B1). Read it before A6.
7. AfterNet has no NickServ — never use that term in new strings,
   prompts, or comments. Say "identified" instead.
8. Pushing to main is otherwise fine in this repo, but PR A and
   PR B specifically should land via PR for review.

Start with Task A0 (snapshot baseline test count). After each task,
report:
- Files changed (summary, not full diff)
- Test count delta from A0 baseline
- Whether `ruff` and `ty` are clean
- The commit SHA you created

Then dispatch the next task's subagent. Keep going through A8.
After A8 + PR A finalize: STOP and summarize the 9 commits.
```

## What to expect from the executing session

- It will create one commit per task (A0 has no commit; A1a, A1b,
  A2, A3, A4, A5, A6, A7, A8 — nine commits).
- Test count should match the A0 baseline at every commit boundary
  except between A1a and A1b (A1a leaves tests failing on purpose;
  A1b restores green).
- It will stop after A8 / PR A finalize and wait for your review.
- For PR B, paste a similar kickoff prompt referencing PR B tasks
  (B0 through B6) once PR A is reviewed and merged.
