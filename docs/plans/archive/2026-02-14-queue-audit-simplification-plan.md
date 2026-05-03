# LLM Queue and Audit Simplification Plan

**Date:** 2026-02-14
**Status:** Draft for Review
**Owner:** LLM plugin maintainers

## Goal

Further simplify the LLM plugin while improving reliability and observability by:

1. Hardening async request handling (especially video generation) against bot restarts.
2. Reducing scheduler/polling complexity.
3. Making usage/audit data complete and queryable across immediate + deferred execution paths.
4. Leveraging Limnoria-native capabilities where they help, without creating cross-plugin coupling that increases operational risk.

## Executive Summary

This is a fun IRC bot, not mission-critical infrastructure. The plan favors low operational complexity over perfect lifecycle/event modeling throughout. Phases 1–2 are committed scope; Phases 3–4 are future-work markers to revisit only when operational pain justifies them.

Recent work has already removed meaningful complexity:

- Shared preflight and identity handling in `plugin.py`.
- Manual moderation + scoped rate limits.
- Durable `pending_tasks` queue in SQLite with retries/backoff.
- Usage logging helper in `persistence.py` (`log_usage`), though immediate and deferred paths do not yet pass the same argument set (deferred delivery omits `prompt`, `status`, and `error_detail`, and currently creates separate rows rather than updating a single request lifecycle row).

The highest remaining complexity is concentrated in deferred task execution and auditing:

1. **Animate restart gap**: provider `request_id` is only persisted on timeout, not immediately after submit.
2. **Queue processing model**: periodic full scan every 30 seconds is simple but wasteful and less deterministic.
3. **Delivery semantics**: results can be deleted before delivery is confirmed, and a batch-delivery failure can silently lose multiple already-deleted results (see hotspot #3 below).
4. **Audit semantics**: immediate and deferred paths do not emit a single coherent lifecycle record. The deferred path logs usage only for completed tasks, so terminal deferred outcomes are not captured in usage with the same lifecycle semantics.

This plan keeps the internal SQLite queue as the primary durability mechanism and uses a hybrid scheduler model: next-due wakeups for normal operation plus a low-frequency periodic safety poll. Limnoria `Later`/`Note` patterns remain optional references for fallback delivery behavior.

## Current State Review

### What is already strong

1. **Preflight consistency**
   - Shared flow in `plugins/llm/src/llm/plugin.py` via `_run_preflight`.
   - Reduced auth/flag/rate-limit duplication across commands.

2. **Durable persistence foundation**
   - SQLite-backed `pending_tasks` table and claim/release semantics in `plugins/llm/src/llm/persistence.py`.
   - Retry/backoff logic in `plugins/llm/src/llm/service.py`.

3. **Moderation simplification landed**
   - Manual `%flag/%unflag/%flagged`, no runtime auto-flag side effects.
   - Optional monitor-first rate limiting for expensive commands.

4. **Test coverage in critical areas**
   - Focused tests for pending queue behavior and rate limits are currently green.

5. **Request tracing infrastructure**
   - `tracing.py` generates 8-char hex request IDs via `ContextVar`, propagated through logging filters.
   - Server response headers (`x-request-id`, `cf-ray`) extracted for upstream correlation.
   - This infrastructure is ready to wire into the usage table (see Proposed Data Model Changes).

### Current complexity and risk hotspots

1. **Animate request durability gap**
   - `request_id` is produced at submit time, but queue persistence for animate currently happens during timeout handling (`_stash_timeout` in `service.py`).
   - If bot exits after submit and before timeout path persists, the provider job may complete but local state is lost.

2. **Periodic polling driver**
   - `llm_pending_tasks` periodic event wakes every 30s regardless of queue state (`plugin.py` `addPeriodicEvent`).
   - This keeps logic simple, but introduces unnecessary wakeups and less precise next-attempt timing.

3. **Delivery-before-delete ordering (with batch cascade risk)**
   - Pending tasks are deleted in `check_pending_tasks` when service considers them done/terminal.
   - Plugin delivery via `_deliver_pending_result` then calls `queueMsg`; if that call fails, the result is lost with no recovery path.
   - **Batch cascade**: `_check_pending_tasks` iterates results in a try/except. If `queueMsg` raises for one result, the exception stops the loop and all remaining results in the batch — already deleted from DB — are silently lost. This is the most dangerous variant because it can lose multiple tasks at once.

4. **Split audit model (deferred terminal outcomes not usage-visible)**
   - Immediate command paths log usage with full context (`prompt=`, `status=`, `error_detail=`).
   - Deferred completion path (`plugin.py` `_deliver_pending_result`) logs usage with cost/tokens only, and **only for completed tasks** (`status == "completed"`). Deferred failures/expirations are not represented as terminal usage outcomes.
   - A single request can also produce multiple usage rows today, which complicates lifecycle reporting until request-linked updates are introduced.

5. **Terminal error classification gap in `_retry_video`**
   - `_retry_video()` calls `requests` (not `litellm`), so HTTP errors are `requests.HTTPError`.
   - `_is_terminal_error()` only checks `litellm.*` exception types.
   - A 404 from xAI (e.g., provider-side expired/deleted video) is classified as transient and retried indefinitely until local expiry. This wastes API calls and delays user notification.

6. **Documentation drift**
   - Root `README.md` (line 142) mentions "Automatic flagging after repeated content-safety refusals" which was never implemented. `plugins/llm/README.md` is already correct ("No automatic flagging side effects"). Only the root README needs updating.

## Limnoria Capability Assessment

Limnoria offers `supybot.schedule` (already used; non-persistent), the Scheduler plugin (persistent but designed for IRC commands, not job orchestration), and Later/Note plugins (useful as delivery-pattern references). Integrating queue ownership into any of these would add coupling and dual state surfaces.

**Decision:** Keep the internal SQLite queue as source of truth for provider jobs. Use `supybot.schedule` for in-process wakeups rebuilt from DB on startup, plus a permanent low-frequency safety poll. Optionally borrow Later/Note style fallback delivery semantics without depending on their storage.

## Target Simplified Architecture

### Request lifecycle (single path)

1. User request accepted.
2. Lifecycle row created with stable request ID.
3. Immediate success:
   - Reply sent.
   - Lifecycle closed as delivered.
4. Deferred path:
   - Job persisted (with full recoverable payload).
   - Scheduler sets next due wakeup.
   - Worker claims and executes due jobs.
   - Delivery is attempted.
   - On success, mark delivered and finalize.
   - On failure, release with retry/backoff and keep durable state.

### Core principles

1. **Durable first**: persist before any operation that cannot be recomputed.
2. **Single source of truth**: SQLite for job state and lifecycle/audit state.
3. **Deterministic ownership**: claim/lease model remains in DB.
4. **Best-effort durable delivery**: never delete completed output before delivery acknowledgment; if bounded retries are exhausted, mark terminal delivery failure and keep metadata for operator retry.
5. **Pragmatic audit shape**: in Phases 1-2, keep append-only usage rows and avoid event-log complexity; evaluate request-linked update-in-place semantics in Phase 3 only if operational pain justifies it.

## Proposed Data Model Changes

### 1) Usage lifecycle enrichment (Phase 3 future work)

The existing `usage` table has a `status` column (added in schema v2) with values: `success`, `error`, `content_blocked`, `flagged_blocked`, `auth_failure`, `rate_limited`. For this pass, keep `status` as the canonical status field and add only minimal, additive fields that improve traceability.

Extend `usage` schema with lifecycle fields:

- `request_id TEXT NOT NULL DEFAULT ''` — wired from the existing `tracing.py` `ContextVar`, making log correlation trivial.
- `attempt INTEGER NOT NULL DEFAULT 0`
- `delivered_at REAL` (intentionally nullable — `NULL` means "not yet delivered", distinguishing from epoch-zero sentinels; other columns use `NOT NULL DEFAULT` because their defaults are meaningful values)

Deferred execution bridge (Phase 1b additive field on `pending_tasks`):

- `origin_request_id TEXT NOT NULL DEFAULT ''` — trace request ID captured at initial command handling time and carried with the deferred row. This preserves a stable join key even when deferred processing runs outside the original `ContextVar` scope.

Migration notes:
- This becomes schema **v4** (v2 added `prompt`/`status`/`error_detail`; v3 is reserved for pending-task delivery fields in Phase 1b). Follow the existing `_migrate_v1_to_v2` pattern in `persistence.py`.
- No `status` rename in this plan.
- Existing rows retain current `status` values; keep `flagged_blocked` distinct.
- Existing queries continue to reference `status`.
- Additive columns with defaults means old code ignores new columns on rollback — no destructive migration needed.
- If richer lifecycle taxonomy is needed later, add `lifecycle_status` additively and dual-write before any cutover.

Indexes:

- `(request_id)`
- `(timestamp, status)`
- `(nick, timestamp)`

### 2) Pending task output acknowledgment (minimal option, Phase 1b)

Add to `pending_tasks`:

- `result_payload TEXT NOT NULL DEFAULT ''` — stores a bounded delivery payload envelope:
  - `{"kind":"url","value":"..."}` for code/draw/animate.
  - `{"kind":"text","value":"..."}` for ask when bounded inline text is sufficient.
  - If ask output exceeds the inline cap, persist it as an artifact and store it as `kind=url`.
- `delivery_state TEXT NOT NULL DEFAULT 'pending'` — `pending|ready|retrying|delivery_failed`. The `pending` value means "provider still working"; `ready` means "result available, awaiting delivery." This single column replaces a separate `result_ready` boolean to avoid dual-state inconsistency.
- `last_delivery_error TEXT NOT NULL DEFAULT ''`
- `delivery_attempt_count INTEGER NOT NULL DEFAULT 0`
- `origin_request_id TEXT NOT NULL DEFAULT ''`

Behavior:

- Worker sets `delivery_state='ready'` and stores payload when provider result is complete.
- Delivery component attempts send.
- On successful send, delete row.
- On failure, keep row, set `delivery_state='retrying'`, persist `last_delivery_error`, and schedule delivery retry.
- After maximum delivery attempts, set `delivery_state='delivery_failed'` and retain row for operator-visible/manual retry paths.
- Expiry queries must only apply to provider-retry rows (`delivery_state='pending'`), so ready/retrying/delivery_failed rows are never expired away.
- Provider claim query must select only `delivery_state='pending'`.
- Delivery claim query must select only `delivery_state IN ('ready', 'retrying') AND delivery_attempt_count < 10`.
- `delivery_failed` rows are excluded from automatic claim/retry until an operator explicitly changes state.

Alternative (cleaner, more normalized):

- New `pending_deliveries` table keyed by `task_id`.

This plan recommends the minimal option first to limit schema/logic spread.

### 3) Animate submit durability fix

On successful provider submit (`request_id` returned), immediately persist job metadata before entering polling loop.

To avoid foreground/background duplicate processing:

- Persist with `next_attempt_at = submitted_at + animateTimeout` (not immediate), so background work remains dormant while foreground polling is still active.
- Foreground path must best-effort delete the persisted row on successful completion and on terminal foreground failure.
- Timeout path must not insert a second row; it only updates/releases the already persisted row (typically by setting `next_attempt_at=now`).

Result: restart-safe from the moment provider accepts the job. If the bot crashes mid-poll (after provider completion but before the result is returned to the caller), the task is already in the DB and will be retried on restart — no additional durability gap exists.

## Queue Engine Simplification

### Current

- Periodic 30s polling event scans for due work and expirations.

### Proposed

1. Use **next-due event as the primary driver**:
   - Compute earliest due `next_attempt_at` at enqueue/release/delete.
   - Schedule one wakeup at that timestamp.
   - On wakeup, process up to claim limit and reschedule next wakeup.
   - Keep a periodic safety poll (e.g., every 5 minutes) permanently.

2. Startup behavior:
   - On plugin init, query earliest due item and schedule wakeup.
   - No active event if queue empty.

3. Enqueue into empty queue:
   - `save_pending_task` (or its caller) must schedule a wakeup when inserting into an empty queue, since no existing event will fire. The helper that computes next-due should be called after every enqueue.

4. Reschedule-on-earlier-enqueue:
   - If a new task has `next_attempt_at` earlier than the currently scheduled wakeup, cancel the existing event via `schedule.removeEvent` and schedule a new one at the earlier time.

5. Stale wakeup idempotency:
   - The wakeup handler must tolerate firing when no work is due (task was deleted, expired, or already processed). This is inherent in the claim-based model: `claim_due_pending_tasks` simply returns an empty list.

6. Expiry handling:
   - Expired rows processed during each wakeup cycle.
   - Safety poll (every 5 minutes) remains enabled as a guard rail to catch edge cases where a wakeup was missed. This bounds expiry notification latency to ~5 minutes (up from ~30 seconds today) — acceptable for this use case.

7. Past-due enqueue handling:
   - `_stash_timeout` sets `next_attempt_at` to `submitted_at` (in the past) so stashed tasks are immediately eligible. The next-due scheduling helper must handle past timestamps by scheduling an immediate (or near-immediate) wakeup, not just computing `min(future timestamps)`.

Benefits:

- Fewer unnecessary scheduler events.
- More predictable retry timing.
- Low-overhead safety recovery path without complicated scheduler ownership logic.

## Delivery Reliability Simplification

### Pending task state model

A single pending task progresses through two independent retry phases:

```
pending → [provider_retry] → ready → [delivery_retry] → delivered
              ↓       ↓                           ↓
       failed_terminal expired              delivery_failed
              ↓       ↓                           ↓
          (deleted) (deleted)               (row retained)
```

- **Provider retry** uses the existing exponential backoff (`PENDING_INITIAL_BACKOFF_SECONDS * 2^attempt`).
- **Delivery retry** uses a separate, shorter exponential backoff (see below).
- The two loops are independent: a task transitions from provider retry to delivery retry when `delivery_state` moves from `pending` to `ready`.
- **Expiry boundary**: `expires_at` applies only to the provider retry phase. Once a task reaches `delivery_state='ready'`, it is no longer subject to expiry — delivery is bounded by `delivery_attempt_count` exhaustion, not time. This follows core principle #4: never discard a completed result.

### 1) Make delivery explicit and retryable

- If send fails (network/channel unavailable/exception), do not discard result.
- Persist failure reason in `last_delivery_error`.
- Retry delivery with bounded exponential backoff, separate from provider retry backoff:
  - Formula: `15 * 2^attempt` capped at 120 seconds (matching the provider retry pattern).
  - Sequence: 15s, 30s, 60s, 120s, 120s, 120s, ...
  - Maximum delivery attempts: 10 (after which mark `delivery_failed`, stop auto-retries, and require operator/manual retry if desired).
- Delivery attempt count is tracked separately from provider attempt count (add `delivery_attempt_count INTEGER NOT NULL DEFAULT 0` to `pending_tasks`).

### 2) Channel availability handling

- Current behavior defers when channel unavailable; retain this.
- Add optional fallback mode (Phase 4, not required for initial work):
  - `deliveryFallbackToNotice` (bool, default false)
  - `deliveryFallbackToPM` (bool, default false)

### 3) Optional "deliver when seen" mode

For PM targets or unavailable channels, optional future mode:

- Queue as pending delivery.
- Attempt when user is seen in `doPrivmsg`/`doJoin`.
- Pattern borrowed from Later/Note plugin behavior.

Not required for initial simplification pass.

## Auditing and Usage Tracking Enhancements

### 1) Near-term vs. future audit contract

Phase 1-2:

- Add structured operator logs for deferred terminal outcomes (`failed_terminal`, `expired`, `delivery_failed`) with task ID and request metadata.
- Avoid adding extra usage rows in Phase 1 for deferred terminal outcomes to prevent amplifying duplicate-row lifecycle noise before request-linked updates exist.

Phase 3 (future work):

- Extend `log_usage()` inputs so immediate and deferred paths share the same argument set (`prompt=`, `status=`, `error_detail=` plus `request_id`/attempt metadata).
- Add `update_usage_delivery(request_id, delivered_at)` to stamp delivery time on the existing request-linked usage row.

This keeps the API to two methods (`log_usage` + `update_usage_delivery`) rather than introducing a three-method lifecycle API.

### 2) Add operator-visible audit commands (lightweight)

Proposed admin commands:

- `%llmqueue [all|due|failed|delivery] [limit]`
- `%llmrequest <request_id>`
- `%llmretry <task_id>` — retriable only for `delivery_failed` rows (manual re-delivery). Provider `failed_terminal`/`expired` outcomes are informational and not auto-retriable in this pass.
- `%llmdrop <task_id> <reason>`

These reduce ad hoc DB inspection and speed incident response.

### 3) Usage reporting compatibility

Keep `%usage` output cost-oriented by default, but add optional status views:

- `%usage statuses`
- `%usage failures`

This avoids disrupting existing operator expectations while exposing richer audits.

## Detailed Implementation Plan

### Phase 1: Correctness first (low risk)

Split into two PRs to limit blast radius:

#### Phase 1a: Animate durability + bug fixes (minimal risk)

1. Persist animate job immediately after submit (before entering polling loop), but set initial `next_attempt_at` to `submitted_at + animateTimeout` so the background worker does not race the foreground poll. On foreground success/terminal failure, delete that row; on timeout, update/release that same row for immediate retry instead of inserting a duplicate. `_stash_timeout` for animate becomes an update path, not an insert path.
2. Fix `_is_terminal_error` to handle `requests.HTTPError` from `_retry_video` with explicit status-code mapping:
   - Terminal: `400`, `401`, `403`, `404`, `410`
   - Transient/retryable: `408`, `409`, `425`, `429`, and all `5xx`
3. Add structured deferred-outcome logs (`failed_terminal`/`expired`) for operator visibility without writing additional usage rows yet.
4. Update root `README.md` to remove stale auto-flag reference (line 142).

**Files likely touched**

- `plugins/llm/src/llm/service.py`
- `plugins/llm/src/llm/plugin.py`
- `README.md`
- `plugins/llm/tests/test_service.py` (terminal error classification, animate persistence)
- `plugins/llm/tests/test_plugin.py` (deferred-outcome logging)

**Tests**

- Restart-window test for animate submit persistence (simulate submit, verify DB row exists before timeout/polling begins).
- `_retry_video` with 404 response classified as terminal; 429 remains transient.
- Deferred failure/expiry emits structured operator log entries.

#### Phase 1b: Delivery acknowledgment semantics (control flow rework)

1. Add `result_payload`, `delivery_state`, `last_delivery_error`, `delivery_attempt_count`, and `origin_request_id` columns to `pending_tasks` (schema v3 migration).
2. Split `check_pending_tasks()` into provider processing and delivery phases with explicit query filters:
   - provider phase claims only `delivery_state='pending'`
   - delivery phase claims only `delivery_state IN ('ready','retrying') AND delivery_attempt_count < 10`
   - expiry deletion applies only to `delivery_state='pending'`
3. Ensure pending result rows are not deleted until delivery success.
4. Add delivery retry loop with bounded exponential backoff: `15 * 2^attempt` capped at 120s, maximum 10 delivery attempts (so: 15, 30, 60, 120, 120, ...).
5. Isolate per-result delivery failures so one `queueMsg` exception does not cascade to the rest of the batch.

**Known Phase 1b limitation**: delivery retry timing is still driven by the 30s periodic poll, so a 15s delivery retry may be delayed up to 30s. Phase 2's event-driven wakeups resolve this. This is acceptable since delivery delays of this magnitude are invisible to IRC users.

**Files likely touched**

- `plugins/llm/src/llm/service.py`
- `plugins/llm/src/llm/plugin.py`
- `plugins/llm/src/llm/persistence.py`

**Tests**

- Delivery failure retention test (`queueMsg` throws, verify result row retained for retry).
- Batch cascade test (one delivery failure does not lose remaining results).
- Delivery retry exhaustion test (10 failures → `delivery_failed` status, row retained).
- Ready/retrying tasks are never removed by expiry sweeps.
- `delivery_failed` tasks are not auto-claimed for retry.

### Phase 2: Event-driven queue wakeups

1. Add next-due scheduling helper as the primary queue wakeup mechanism.
2. Rebuild scheduled wakeup from DB on startup.
3. Handle enqueue-into-empty-queue, reschedule-on-earlier-enqueue, and past-due `next_attempt_at` edge cases.
4. Keep a 5-minute safety poll permanently (bounds expiry notification latency).

**Files likely touched**

- `plugins/llm/src/llm/plugin.py`
- `plugins/llm/src/llm/service.py`
- `plugins/llm/src/llm/persistence.py`
- `plugins/llm/tests/test_plugin.py`
- `plugins/llm/tests/test_service.py`

### Phase 3: Audit enrichment (future work — only if operators report real pain)

1. Additive schema changes only: `request_id`, `attempt`, `delivered_at` on `usage` table (schema v4).
2. Keep `status` as canonical; do not rename in this pass.
3. Wire `request_id` from `tracing.py` ContextVar into `log_usage` calls.
4. Add `update_usage_delivery(request_id, delivered_at)` helper.
5. Convert queue completion path to include request-linked updates.
6. Re-evaluate richer lifecycle/event modeling only if operational pain justifies it.

**Files likely touched**

- `plugins/llm/src/llm/persistence.py`
- `plugins/llm/src/llm/plugin.py`
- `plugins/llm/src/llm/service.py`
- `plugins/llm/tests/test_persistence.py`
- `plugins/llm/tests/test_commands.py`
- `plugins/llm/tests/test_integration.py`

### Phase 4: Operator tooling (future work)

1. Add queue/audit admin commands.
2. Add concise status summaries.
3. Extend README with operational playbook.

**Files likely touched**

- `plugins/llm/src/llm/plugin.py`
- `plugins/llm/tests/test_commands.py`
- `README.md`
- `plugins/llm/README.md`

## Testing Strategy

### Unit tests

1. Persist-before-poll animate behavior.
2. `_retry_video` terminal/transient classification for `requests.HTTPError` (404 terminal, 429 transient).
3. Deferred failure/expiry structured operator logging.
4. Delivery retry semantics and non-destructive failure handling.
5. Batch delivery isolation (one failure does not cascade).
6. Next-due scheduler recomputation (including past-due timestamps).
7. Lifecycle status transitions.

### Integration tests

1. Simulated restart between animate submit and completion retrieval.
2. Channel unavailable then available delivery scenario.
3. Mixed command load with queued + immediate completions.
4. Audit query for single request across all transitions.

### Reliability tests

1. Forced exceptions during `queueMsg` — single and batch scenarios.
2. Forced DB busy/lock contention on claim/release.
3. Duplicate wakeup invocations to verify claim lease safety.

## Rollout Plan

Stages map 1:1 to implementation phases. Stages A–B are committed scope; C–D are future work.

### Stage A: Correctness fixes (Phase 1a + 1b)

1. Deploy animate durability fix and error classification fix (Phase 1a).
2. Deploy delivery acknowledgment semantics (Phase 1b).
3. Monitor queue depth and stale delivery rows (`delivery_state IN ('ready','retrying')`).

### Stage B: Event-driven wakeups (Phase 2)

1. Enable next-due scheduling.
2. Keep 5-minute safety poll permanently.

### Stage C: Audit enrichment (Phase 3, future work)

1. Deploy additive request-tracing fields with no user-visible behavior change.
2. Validate that deferred completions can be joined by `request_id` in logs/usage.

### Stage D: Operator tooling (Phase 4, future work)

1. Ship queue/audit commands.
2. Add on-call runbook examples.

### Rollback

Schema changes are additive only (new columns with defaults), so rollback is a no-op at the schema level. However, rollback across Phase 1b/2 is behavior-sensitive: old code may not honor delivery states and can mishandle retained delivery rows. Practical rollback guidance:

1. Prefer forward-fix over rollback once Phase 1b is live.
2. If rollback is required, accept best-effort behavior for in-flight deferred rows or run a one-time operator SQL cleanup/requeue procedure first.
3. No destructive schema migration is required at any stage.

## Metrics and Alerts (manual debugging reference)

Since the bot has no metrics export infrastructure, these are **ad hoc SQL queries** for manual debugging against the usage and pending_tasks tables — not automated deliverables. If metrics export is added later, they map directly to counters/gauges.

Useful queries (annotated with the earliest phase that enables them):

1. `SELECT task_type, delivery_state, count(*) FROM pending_tasks GROUP BY task_type, delivery_state` — pending by type and delivery state. *(Phase 1b+)*
2. `SELECT min(submitted_at) FROM pending_tasks` — oldest pending age. *(Phase 1a+, uses existing columns)*
3. `SELECT count(*), last_delivery_error FROM pending_tasks WHERE delivery_state='delivery_failed' GROUP BY last_delivery_error` — terminal delivery failures and top reasons. *(Phase 1b+)*
4. `SELECT status, count(*) FROM usage WHERE timestamp > ? GROUP BY status` — request status distribution. *(existing, schema v2)*
5. `SELECT avg(delivered_at - timestamp) FROM usage WHERE delivered_at IS NOT NULL AND command='animate'` — animate submit-to-deliver latency. *(Phase 3+)*

Alert suggestions (implemented as periodic log warnings):

1. Pending oldest age > configured expiry threshold / 2.
2. Delivery failures sustained above baseline.
3. Queue depth monotonic increase over N intervals.

## Data Retention

Adding `result_payload` and request-linked usage fields will grow the DB over time. Consider:

1. **Pending tasks**: rows are deleted on successful delivery, so growth is bounded by failure/expiry rates. The safety poll should log a warning if stale delivery rows (`delivery_state IN ('ready', 'retrying')`) exceed a configurable threshold (e.g., 50).
2. **Usage table**: no automatic retention policy today. A future pass could add a `%llmpurge` admin command or a configurable `usageRetentionDays` setting to prune rows older than N days. Not required for this plan but worth noting.

## Risks and Mitigations

1. **Schema expansion risk**
   - Mitigation: additive columns only; defaults preserve old readers. Rollback requires no schema changes.

2. **Behavioral regressions in queue semantics**
   - Mitigation: keep claim/release contract unchanged; add migration invariants tests.

3. **More payload storage volume**
   - Mitigation: `result_payload` is bounded and stores references when possible (especially for large outputs). Index for query patterns.

4. **Operational complexity from new commands**
   - Mitigation: admin-only commands, tight output limits, clear docs.

5. **Audit scope creep**
   - Mitigation: defer event-log/lifecycle redesign until operational pain is proven; keep Phase 3 explicitly future work.

6. **Batch delivery cascade (existing bug)**
   - A single `queueMsg` exception in `_deliver_pending_result` stops iteration and loses all remaining already-deleted results.
   - Mitigation: Phase 1b isolates per-result delivery with individual try/except.

## Out of Scope (this pass)

1. Replacing SQLite with external queue/broker.
2. Full integration with Limnoria Scheduler plugin as queue backend.
3. Cross-network distributed worker model.
4. End-user self-service status dashboard beyond IRC commands.
5. Metrics export infrastructure (Prometheus, StatsD).
6. Automatic usage data retention/purging.

## Acceptance Criteria

1. Animate requests are restart-safe from submit acknowledgment onward.
2. Completed provider results are retained across transient delivery failures and either delivered or marked `delivery_failed` after bounded retries.
3. Queue wakeups are primarily driven by next due item, with a low-frequency safety poll retained.
4. Deferred outcomes are operator-queryable with durable states (`delivery_state` plus error context); request-ID-linked usage joins are a Phase 3 enhancement, not required for Phases 1-2 acceptance.
5. Documentation reflects actual runtime moderation and queue behavior.

## Decisions for This Pass

1. Delivery remains bounded-retry best effort; no mandatory fallback to PM/notice by default.
2. Use minimal `pending_tasks` extension first; no separate `pending_deliveries` table yet.
3. Keep `%usage` cost-first behavior; status views are optional and lightweight.
4. Retain a 5-minute safety polling event permanently.
5. Keep `flagged_blocked` as a distinct status value.
6. Keep append-only usage rows in Phases 1-2; evaluate single-row request-linked updates in Phase 3.
7. Do not persist an `accepted` status row by default (avoid cost/reporting complications).
8. Include request IDs in operator-facing failure paths where practical.
9. If Phase 3 request-linked updates are implemented, aggregate retry cost/tokens on the same request row when possible.
10. Split Phase 1 into two PRs (1a: durability/bug fixes, 1b: delivery control flow rework) to limit blast radius.
11. For animate immediate persistence, schedule first background eligibility at `submitted_at + animateTimeout` to avoid foreground/background races.
12. Persist `origin_request_id` on deferred rows so deferred outcomes can be deterministically joined to request-level audits later.
13. `%llmretry` is delivery-only (`delivery_failed`) in this pass; provider terminal/expired outcomes are not retried automatically.

## Suggested First PR (Phase 1a)

1. Animate submit durability fix (persist immediately after provider returns `request_id`).
2. Fix `_is_terminal_error` to classify `_retry_video` `requests.HTTPError` statuses with an explicit terminal/transient mapping (404 terminal, 429 transient).
3. Add structured deferred-outcome logs (failed/expired) for operator visibility without extra usage-row writes.
4. Root `README.md` auto-flag reference correction.
5. Focused tests: restart window, terminal error classification, deferred-outcome logging.

This gives immediate reliability gains with minimal architectural churn. The delivery acknowledgment rework (Phase 1b) follows as a separate PR since it involves schema migration and control flow changes.
