# Reactive Loom — Design

**Date:** 2026-06-03
**Status:** Approved, pending implementation plan
**Scope:** `plugins/llm/src/llm/verse/loom.py`, loom wiring in `plugins/llm/src/llm/plugin.py`, two config help strings in `plugins/llm/src/llm/config.py`, and the loom test suite.

## Problem

The loom currently behaves like a host: a periodic timer fires, the loom **seeds** the
channel with an opening line, the other AfterNet bots react to *that line*, and the loom
digests the result into world-mutation proposals. But the other bots on the channel do
not work that way — they throw out messages on their own, unprompted. So the loom's
seed is noise the other bots are reacting *to us* about, and the digest captures their
reactions to our interval messages rather than their genuine spontaneous output.

## Goal

Flip the loom from proactive to reactive. Idle in the channel until another user or bot
speaks. When one does — and only once per interval at most — chime in ourselves,
reacting to *them*. Record the triggering line as the genuine first input, not a
reaction to anything we said.

## Decisions (locked)

- **Cycle shape:** single chime-in. React once to the triggering line, listen for one
  beat window, then digest. Drops the second post that exists today.
- **Trigger model:** event-driven cooldown. Retire the periodic cycle timer; the first
  eligible inbound line after `cycle_interval_s` has elapsed since our last chime-in
  opens a cycle. The clock is "time since we last chimed in."
- **Cold start:** pure reactive. The loom never speaks unprompted. If nobody talks, it
  stays silent. The other bots' self-triggering output is the input source.

## Architecture

### Entry point: `observe_transcript` (driver thread)

`observe_transcript(nick, text)` becomes the trigger. It is invoked from `doPrivmsg` on
the **IRC driver thread** (the main IRC loop), so the path must stay cheap — see the
driver-thread audit (typing-lag root cause was blocking this thread). The trigger path
does only: acquire lock, compare a timestamp, append to a list.

New `Loom` instance state: `_last_chime_at: float | None` (initialized `None` =
eligible immediately; a reload re-arms the loom for the next line, which is acceptable
and desirable as a liveness signal).

Under `self._lock`, for each line that survived the source filter:

1. **Active cycle exists** → `cycle.append_transcript(nick, text)`; return. (unchanged)
2. **Not due** — `_last_chime_at is not None and (now - _last_chime_at) < cycle_interval_s`
   → ignore the line (there is no cycle to record into); return.
3. **Due** → form a cycle synchronously and cheaply:
   - stash `prev_last_chime = self._last_chime_at`
   - create `LoomCycle(cycle_id, channel="", started_at=now, verse_stable_block="",
     transcript=[(nick, text)])` — channel/snapshot deferred to the worker
   - `self._active = cycle`; `self._last_chime_at = now`
   - outside the lock: `bridge.submit("loom:open", lambda: self._open_and_chime(cycle, prev_last_chime))`

Lines arriving while the worker spins up see an active cycle (step 1) and append to the
forming cycle, so nothing is lost in the gap between forming and opening.

### Worker phase 1: open + chime-in (`loom:open`)

Runs on the LLM executor thread. Heavy DB work lives here, off the driver thread.

1. **Pick focus verse** — `list_candidate_channels` / `candidate_weight` /
   `pick_focus_verse` with `verse_cooldown_s` and the round-robin pointer. Unchanged
   logic, moved here from the old `tick()`.
   - If `None` (no eligible verse) → roll back: under lock set `self._active = None` and
     restore `self._last_chime_at = prev_last_chime` so the next inbound line retries.
     Lines that appended to the aborted forming cycle are dropped (rare; acceptable —
     matches today's aborted-seed behavior).
2. **Snapshot** the chosen verse → set `cycle.channel` and `cycle.verse_stable_block`
   under lock.
3. `_maybe_consume_one_seed_for(cycle.channel)` (crosspoll receive — unchanged).
4. **Chime-in LLM call** — new `build_chimein_tail(loom_transcript_so_far=...)` framed
   as reacting to what the others *just said* (not "they replied to us"). One line,
   ≤ 350 chars, stay in fiction, no JSON. Post via `post_to_loom_channel`.
   - On post failure (network down) → roll back the cycle as the seed phase does today:
     `_active = None`, pop the verse cooldown entry (`_last_cycle_by_channel`), and
     restore `_last_chime_at = prev_last_chime` so a later line can retry rather than
     waiting a full silent interval after a failed post.
5. `schedule_after(beat_window_s, self.after_chime, "llm_loom_after_chime")`.

### Worker phase 2: digest (`loom:digest`)

`after_chime` → `submit("loom:digest", ...)`. The digest phase is **identical** to
today's `_digest_phase`: truncate transcript → `build_digest_tail` → `parse_digest` →
`apply_or_queue` per proposal → `_active = None` in `finally`. The only difference is
upstream: `transcript[0]` is now the bot's spontaneous line.

## Removed / renamed

- **Removed:** `Loom.tick()`, `_schedule_loom_tick`, `_loom_tick`, the `llm_loom_cycle`
  periodic registration; `_seed_phase` + `build_seed_tail`; `_beat_phase` +
  `build_beat_tail`; `after_beat1`; `after_beat2`.
- **Kept:** `pick_focus_verse` (called from the worker now), `truncate_transcript`,
  `apply_or_queue`, `parse_digest`, crosspoll send/receive, the multi-verse selection
  machinery, `LoomConfig` fields.
- **Renamed event:** `llm_loom_after_beat1` / `llm_loom_after_beat2` → single
  `llm_loom_after_chime`. Teardown updated in both `plugin.py` spots (the disable branch
  around line 838 and the re-wire branch around line 5010).

## Config

- `loomCycleInterval` — semantics shift from "timer cadence" to "minimum gap in minutes
  between loom chime-ins." Key unchanged; help string updated.
- `loomCaptureTranscript=False` — now makes the loom fully inert, because the loom only
  ever acts on captured lines (there is no seed path left). Help string updated to say
  so. `False` is no longer "seed-only mode" — that mode is gone.
- `loomVerseCooldown`, `loomBeatWindow`, transcript caps, crosspoll knobs — unchanged.

## Concurrency notes

- The trigger path holds `_lock` only for the cheap form-or-append decision. The worker
  reacquires `_lock` only to mutate cycle fields and `_active`.
- Self-lines do not re-trigger: the bot does not receive its own PRIVMSG echo, and a
  cycle is active throughout chime-in + beat window regardless.
- No two cycles overlap: `_active` gates forming (step 1), and a forming cycle is set
  before the worker is submitted, so a second due-line cannot form a parallel cycle.

## Testing (TDD)

Rework `test_loom.py` / `test_loom_integration.py`. Seed/beat/tick tests are replaced by:

- **Cooldown gate:** first line opens a cycle; a second line within `cycle_interval_s`
  is ignored; a line after the interval opens a new cycle.
- **Forming-cycle race:** lines arriving between form and open are captured in the
  transcript handed to the chime-in/digest calls.
- **No-eligible-verse rollback:** worker finds no verse → `_active` cleared and
  `_last_chime_at` restored so the next line retries.
- **Single chime-in + digest:** exactly one post before the beat window, then digest;
  no second beat.
- **Chime-in prompt framing:** `build_chimein_tail` references the others' lines as
  spontaneous, and `transcript[0]` is the triggering line.
- **Driver-thread cheapness:** the trigger path does not call `snapshot` /
  `list_candidate_channels` (asserted via the fake bridge call log).

## Out of scope

- Multi-verse selection, crosspoll, compaction, and aging logic are untouched.
- No changes to the digest proposal schema or `apply_or_queue`.
