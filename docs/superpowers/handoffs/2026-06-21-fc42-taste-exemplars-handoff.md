# Handoff: implement fc42 taste-tuned verse exemplars

Paste the block below into a fresh Claude Code session in this repo to execute the
(already brainstormed, planned, and twice-red-teamed) feature.

---

You are implementing a completed, adversarially-red-teamed feature for the VibeBot v8
Limnoria IRC plugin. The design and a task-by-task TDD plan are already written and
committed to `main`. Your job is to EXECUTE the plan — do not re-design it.

**Repo:** `/Users/rdrake/workspace/afternet/vibebot-v8` (branch `main`).
**Plan:** `docs/superpowers/plans/2026-06-21-fc42-taste-exemplars.md`
**Spec:** `docs/superpowers/specs/2026-06-21-fc42-taste-exemplars-design.md`

**What it builds:** an offline channel-log miner that surfaces fc42's "liked" verse
lines (verbatim re-pastes + explicit praise) into a human-curated review file; a
per-channel `registry.Json` key `verseStyleExemplars`; and sanitized + capped
injection of those exemplars into `build_verse_system_prompt`. Verse-only. With the
key at its default `[]`, the verse prompt is byte-identical to today — it ships INERT.

**How to execute:**
1. Invoke `superpowers:subagent-driven-development` and follow it: read the plan,
   extract all 8 tasks with full text, dispatch a fresh implementer subagent per task,
   and run the two-stage review (spec-compliance, then code-quality) after each.
2. Do NOT implement on `main` directly. Create a feature branch (or git worktree) off
   `main` first; commit per task. When all 8 tasks pass and the final whole-implementation
   review is clean, use `superpowers:finishing-a-development-branch` to merge to `main`
   and push (CI → Docker → auto-deploy; it deploys inert because the key defaults empty).

**Hard constraints (these are folded red-team learnings — do NOT re-introduce):**
- `make lint` (ruff rules E, F, N) + `make typecheck` (`ty`) run after EVERY Edit and
  gate each task. In tests: fake helper classes must use `self` (N805); all imports at
  module top (E402); no unused imports (F401).
- The new prompt param is `style_exemplars: Sequence[str] = ()` — `Sequence`, NOT
  `list`, or `ty` rejects the `()` default. Add `Sequence` to avatar.py's
  `from collections.abc import …` line.
- `registry.Json([], "help")` REQUIRES the help arg. The round-trip test uses
  `restored.set(str(v))`, NOT `set(serialize())`.
- The miner's entity gate REUSES `store.match_entities_in_text(text)` (truthiness) —
  there is NO separate roster accessor and NO second matcher.
- The "default-empty key ⇒ verse prompt byte-identical" property is the safety
  guarantee; the plan's byte-identical test must pass.
- Run tests with `uv run pytest …`; full gate is `make test` (~2300 tests, coverage ≥ 93%).

**Definition of done:** all 8 tasks green; `make test && make lint && make typecheck`
clean; final code review; branch finished (merged + pushed). Then STOP. Do NOT run the
miner against prod logs or set `verseStyleExemplars` — that is a separate operator step
(rdrake runs `uv run python -m llm.verse.taste_mine <logs…> --verse-dir <dir> --channel
'#afternet'`, reviews `taste_candidates.md`, drops any `DENIAL?`-flagged line, then sets
the key in `bot.conf` while the bot is stopped), per the plan's Rollout section.

---
