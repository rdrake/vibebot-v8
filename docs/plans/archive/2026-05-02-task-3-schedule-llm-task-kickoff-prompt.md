# Kickoff prompt — Phase 2 Task 3 (`schedule_llm_task`)

Paste the block below into a new Claude Code session in this repo.

---

I'm starting Phase 2 Task 3 of the Limnoria tool bridge in vibebot-v8: a new native LLM tool `schedule_llm_task` that schedules a future `@ask` invocation via Limnoria's Scheduler, with companion `list_scheduled_llm_tasks` and `cancel_scheduled_llm_task` tools.

DESIGN PLAN (read the Task 3 section in full before doing anything else):
  docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md  (§"Task 3 — `schedule_llm_task` tool (Scheduler-as-agent)")

NO IMPLEMENTATION PLAN EXISTS YET. Your first job is to write one in the same shape as the predecessors:
  docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md  (the canonical exemplar — read its structure)

REQUIRED SUB-SKILLS:
  superpowers:writing-plans  → use to write the implementation plan first
  superpowers:executing-plans → use to execute it once it's reviewed

PREDECESSOR WORK (already shipped — assume the codebase reflects this):
  - Task 1 (mutation gate): MUTATING_COMMANDS classification + bridgeAllowMutating channel value + dispatch defense-in-depth. See docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md and the 8 commits ca61702..642c059.
  - Task 2 (curated default allowlist): DEFAULT_ALLOWED_PLUGINS frozenset + empty-allowlist fallback in _build_bridge_tool. See commits 57dc552 and a74d883. Note: registry default stays as []; the curated set kicks in via the bridge code, not via a registry-default change. This was a forced design pivot because Limnoria persists default values to disk and on-disk wins on reload.
  - Task 5a (config consolidation, introduce-new phase): assistantModel / assistantApiKey / assistantSystemPrompt / imageModel / imageApiKey + a resolve_setting compat shim in config.py that prefers the new keys and falls back to old ones with a one-time deprecation warning per fallback used. 18 service.py + 3 plugin.py lookup sites migrated. See commits 6dc0f92..5a74bf9. Task 3's @ask path goes through resolve_setting, so the fired @ask will pick up assistantModel naturally.

Hard rules for this session:

1. **Write the implementation plan first.** Do not start coding until the plan is reviewed. Use the structure from docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md: pre-flight (Task 0), then numbered task sections (A1, B1, B2, ...) each with Files / Step 1 (failing test) / Step 2 (verify red) / Step 3 (implement) / Step 4 (verify green) / Step 5 (commit, with the exact commit message inline). Plan length should be in the same ballpark — ~1500-1800 lines is fine if it earns it.

2. **Critical pre-plan investigations** (do these BEFORE writing the plan; they shape it):
   a. Confirm Scheduler's actual public Python API for one-shot and recurring events. Read `.venv/lib/python3.14/site-packages/supybot/plugins/Scheduler/plugin.py` AND `.venv/lib/python3.14/site-packages/supybot/schedule.py`. The Phase 2 plan §Task 3 says `schedule.addEvent` / `schedule.addPeriodicEvent` "verify exact entry points at impl time" — verify them now and capture the signatures in the plan. Note that Scheduler's `add` / `repeat` IRC commands are wrappers around private internals; the plan needs to call those internals correctly.
   b. Verify `msg.tags` survives Scheduler's pickle round-trip (Phase 2 plan §Task 3 §"Loop hazard mitigation" flagged this as a risk). Write a small standalone test that pickles a fake `IrcMsg` with tags, unpickles it, and checks the tags survive. If they don't, fall back to the `@askscheduled` alias path the plan describes.
   c. Verify Scheduler persists periodic events to disk across restarts (Phase 2 plan claims this is true at `Scheduler/plugin.py:117`; confirm).
   d. Find the existing reminder NL parser (likely `parse_reminder` in service.py) and confirm it can be reused as-is for `when_natural` parsing — the plan says "reuse, do not fork."
   e. Check what column names the existing `pending_tasks` / reminders persistence layer uses so the new `(scheduled_id, creator_nick, account, prompt, when, target)` table doesn't collide and can reuse the same DB connection / migration mechanism. See `plugins/llm/src/llm/persistence.py`.
   f. Read `plugins/llm/src/llm/assistant.py` to understand how native tools are registered (look for the `extra_tools` / `extra_handlers` plumbing — Phase 1 says we explicitly avoided expanding this for the bridge, but Task 3 IS a native tool and should use this path, NOT the bridge dispatch). Identify the right place to register `schedule_llm_task` so it ships in the chat profile alongside `set_reminder`.

3. **Scope discipline — what is and is not Task 3:**
   - In scope:
     - `schedule_llm_task` (one-shot + recurring; recurring uses Scheduler's persistent periodic events).
     - `list_scheduled_llm_tasks` (owner-attributed via the new tracking table).
     - `cancel_scheduled_llm_task` (owner-scoped).
     - Depth cap: a fired schedule cannot itself call `schedule_llm_task` (depth >= 1 → refuse with envelope error). Implementation candidate: `msg.tags["llm_schedule_depth"] = 1`; fall back to `@askscheduled` alias if tags don't pickle.
     - Per-creator budget: new `bridgeScheduledTaskLimit` channel int, default 5. Refuse at create time when exceeded.
     - Tracking table for owner attribution (Scheduler's own list isn't owner-scoped).
     - Tool-description guidance distinguishing `schedule_llm_task` from `set_reminder`. Both stay shipped.
   - Out of scope:
     - Reply-target override (Phase 2 plan §"Open question #1" — defer; v1 replies to the channel where the schedule was created).
     - Self-cancel-on-capability-revoke (Phase 2 plan decision: log+skip, do not auto-cancel).
     - Cross-channel scheduling permissions.
     - Migration of existing `set_reminder` rows to anything else (no migration; both systems coexist).

4. **Capture open questions in the plan, do not silently resolve them.** The Phase 2 plan §Task 3 lists five (reply-target override, msg.tags pickling, recurring storage, listing/cancel UX, identity drift). Your impl plan should restate any that aren't fully resolved by your investigations and propose a concrete answer for each — but flag the proposed answer as "PR-review item" if it's a judgement call.

5. **TDD strictly.** Same rhythm as Tasks 1, 2, 5a: failing test → verify red for the right reason → implement → verify green → commit per task with the inline commit message.

6. **AGENTS.md gates are agent-run.** After every Python edit run `make lint` and `make typecheck` (AGENTS.md:22). Run `make preflight` once at the end (AGENTS.md:23). Never bypass with `--no-verify`.

7. **Pre-existing project memory you should respect:**
   - Pushing directly to main is fine (no PR required).
   - The CI workflow and the Docker-build workflow are separate; wait for both before any operational verification on the running bot.
   - SSH access: `ssh -i ~/.ssh/id_rsa vibebot@rdrake.org`.
   - `systemctl --user restart vibebot` is pre-authorized over SSH.
   - On `Permission denied (publickey)`, ask the user to run `security unlock-keychain` locally — do not try workarounds.
   - Reading production config files via SSH requires explicit per-action authorization; don't do it without asking.

8. **If you get stuck or surprised, stop and ask.** Particularly likely surprise points:
   - Scheduler's pickle file format may not round-trip `msg.tags` (Phase 2 plan flagged this).
   - The existing `parse_reminder` may return a shape that doesn't quite match what `schedule_llm_task` needs (it returns `recurrence_seconds` and `recurrence_rrule` fields — confirm both can be passed to Scheduler's API).
   - The persistence layer may not have a migration mechanism that's friendly to adding a new table mid-release.
   - The chat profile's tool list may need to be filtered per-channel (e.g. `bridgeEnabled` controls bridge tools; should `schedule_llm_task` be similarly gated, or always-on?). Decide and document.

9. **Auto mode is on for me; pushing to main is fine for this repo.** After plan review and execution, push directly. Wait for Docker before operational verification.

Begin with the pre-plan investigations (item #2 above). Report findings before writing the implementation plan. Do not start writing the plan until the investigations are documented and any blockers are surfaced.
