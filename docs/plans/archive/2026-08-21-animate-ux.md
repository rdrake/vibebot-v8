# @animate progress and delivery UX

**Status:** Designed, not started (2026-08-21)
**Author:** Richard Drake (with claude; red-teamed by codex)
**Affects:** `@animate` / `@video` and the `generate_video` chat tool, shipped
2026-08-21 (`7fdbd8a..b261e8e`)
**Priority:** Medium. Nothing is broken; a clip renders and arrives. The gap is
that a two-minute silence looks identical to a failure, and a delivered link
carries no context of its own.

## Summary

A clip takes about 135 seconds for the default seven seconds of video, and a
clip queued behind another has been observed waiting a further 119 seconds
before its render even starts. Across that whole window the bot says nothing
after its first acknowledgment, and the acknowledgment is written by the
model, so a real submission and an invented one read the same. On 2026-08-21
that stopped being hypothetical: two requests were answered with a job marker
copied out of history, and a third with a fabricated `oaiusercontent.com`
link. No clip was queued for any of them.

The tool-call side of that is fixed (`b261e8e`). What is left is the part a
user can see: give the wait a signal the model cannot forge, and give the
delivered link enough context to stand on its own.

Two changes, and one thing deliberately left alone.

## The acknowledgment stays as it is

Stated first because it is the change not being made. The model keeps writing
the acknowledgment in its own voice, and no ETA or job id is added to it.

The concern the acknowledgment seemed to answer — "is this real, or is the
bot improvising again?" — is answered better by the typing indicator below.
That is driven by the plugin, not by the model, so it cannot be faked by a
turn that never called the tool. Wording cannot make that guarantee, and a job
id in the channel is the same opaque bookkeeping marker that
`_strip_job_markers` now exists to keep out of history.

## Typing held for the render

`_begin_typing` already does the hard part: it sends `+typing=active`
immediately and re-sends it every four seconds from a daemon thread, because
clients expire the state after roughly six seconds. Today it is scoped to the
planner turn, so it stops about four seconds in — long before the clip lands.

### Derived from the database, not tracked in memory

The obvious design — a refcount per target, incremented at submission and
decremented on delivery — is unsound here, and the red-team section below says
why in detail. The short version: delivery is deliberately at-least-once, so a
decrement can run twice for one job and steal another job's hold.

So track nothing. `pending_tasks` already knows which clips are in flight and
it is the durable source of truth. One plugin-owned daemon thread wakes every
four seconds and asks it:

```sql
SELECT DISTINCT reply_target FROM pending_tasks
WHERE task_type = 'animate'
  AND delivery_state = 'pending'
  AND submitted_at > :now - 360
```

`delivery_state` is an existing indexed column (`persistence.py:342`). A row
is `pending` while the box renders and moves to `ready` once the clip is in
hand, so this set is exactly "clips still rendering" — typing stops when the
video exists, not when the message is delivered.

**Add a read-only `active_animate_targets(now, max_age)` to persistence** that
does this filtering in SQL and returns the targets. Do NOT reach for
`claim_due_pending_tasks`, despite its inviting `delivery_state_filter`
argument: it opens `BEGIN IMMEDIATE` and leases the rows it returns
(`persistence.py:1102`), so calling it from the refresher would steal work
from the poller every four seconds. `load_pending_tasks` is read-only but
loads whole rows and is documented as a debugging helper; a purpose-built
reader is cheaper and says what it is for.

Each pass sends `+typing=active` to every target in the set. A target present
in the previous pass and absent from this one gets one `+typing=done`.

This deletes problems rather than solving them:

- **At-least-once delivery** cannot double-release, because nothing releases.
- **Task identity** is not needed; there is no lease to match to a job.
- **`delivery_failed`**, the fourth end state after ten failed delivery
  attempts, is a `WHERE` clause rather than a fourth code path.
- **Restart and `@reload`** recover on the next tick, because the state is
  rebuilt from the database rather than carried in memory.
- **The cap** is the `submitted_at` predicate. A job may stay pending for the
  configured 1800s (`animateExpiry`), but nobody should watch the bot type for
  half an hour; typing gives up after six minutes and the clip still arrives.

### Threading and IRC identity

**Do not capture `irc`.** A zombie connection makes `queueMsg` return `False`
rather than raise, so a captured object goes silently no-op. Resolve the
connection per pass from `world.ircs`, the way the delivery path already does
(`plugin.py:2213`), and key the in-memory "previously active" set by
`(irc.network, reply_target)` — `reply_target` alone would merge unrelated
targets across networks.

**One thread, not one per submission.** `TAGMSG` is normal priority while
`PRIVMSG` is low (`irclib.py:217`), so typing refreshes jump the queue ahead
of ordinary replies, against a configured budget of one queued message per
second (`bot.conf:1239`). A single loop that batches all active targets bounds
that; a thread per submission does not.

**The thread idles instead of polling forever.** It blocks on a
`threading.Event` that submission sets, polls every four seconds while the set
is non-empty, and goes back to blocking when it empties. Set the event once at
startup so a restart mid-render resumes. Guard the "previously active" set
with its own lock, following the convention every other mutable map in the
plugin follows (`plugin.py:818-842`).

**Shutdown.** `die()` must stop the thread. It cannot send `+typing=done` —
`_safe_queue` drops sends once shutdown has begun (`plugin.py:3773`) — and it
does not need to: clients expire the state in about six seconds.

### The done/active flicker

The planner turn's own stopper sends `+typing=done` when the turn ends, a few
seconds *after* the render refresher has started on the same target. Left
alone that shows as a done-then-active flicker.

Give `_begin_typing` an optional `suppress_done_if` predicate, defaulting to
`None` so every other caller is untouched. The animate paths pass one that
returns True when the target is in the refresher's active set. That is a
single lock-guarded read, not a rework of the closure.

## Delivery carries its context

Today an animate delivery is the bare URL:

    <vibebot> https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4

Change it to match what `@draw`'s deferred path already sends:

    <vibebot> rdrake: your video is ready! "a corgi riding a unicorn" → https://…mp4

**Budget the line backward from the URL.** `prompt_preview` is truncated to
100 *characters* (`service.py:3250`), which can be far more than 100 bytes,
and the animate delivery path does no length fitting at all —
`_collapse_for_irc` only flattens newlines. Limnoria truncates the outbound
message at the wire limit, and because the URL comes last, a long prompt would
cut off the one part that matters. Reserve the nick, the boilerplate, and the
URL against the same budget the rest of the plugin uses
(`conf.supybot.reply.mores.length`, channel-scoped), truncate the prompt to
what is left with `truncate_to_word_boundary`, and drop the quoted prompt
entirely if it cannot fit. The URL is never sacrificed.

**Prompt echo is display, not injection.** `sanitize_output` already runs on
`prompt_preview` (`plugin.py:2151`) and `safeArgument` blocks CR/LF/NUL at the
send boundary, but `sanitize_output` deliberately preserves IRC formatting
codes. Echoing a prompt back can therefore reproduce bold and colour. Strip
formatting codes from the echoed prompt specifically; a delivery line is not
the place to let a requester colour the bot's output.

**This reverses a deliberate decision in `84dbb67`,** and the reasoning there
was sound: the IRCv3 `+draft/reply` tag already shows the request above the
message, so repeating it restates what the client is displaying. That holds on
clients that render replies and fails on the ones that do not, where a bare
URL arrives two minutes later attached to nothing.

The reply tag stays. This is additive: threading for clients that support it,
readable text for those that do not. It also makes animate consistent with
draw and code, which both spell out nick and prompt on deferred delivery.

## Testing

- The refresher's target set comes from the database: seed `pending_tasks`
  rows directly and assert the set, including that `ready`, `delivery_failed`,
  and rows older than six minutes are excluded.
- A restart mid-render resumes typing from the database on the next tick.
- Two overlapping submissions to one target: the set holds one entry, and the
  first clip landing does not stop typing while the second still renders.
- A target dropping out of the set sends exactly one `+typing=done`.
- The refresher resolves `irc` per pass and skips a target whose connection is
  gone, without raising.
- `die()` stops the thread.
- The planner stopper does not send `done` while the refresher holds the
  target.
- The delivered line carries nick, prompt, and URL, still carries the reply
  tag, and keeps the URL intact when the prompt is long enough to overflow the
  line — including a multi-byte prompt whose character count fits but whose
  byte count does not.

## Red-team findings

Codex reviewed the first draft of this design against the code. What it found,
and what happened to each:

| Finding | Severity | Resolution |
| --- | --- | --- |
| Delivery is at-least-once; a refcount can double-decrement or steal another job's hold | Blocking | Design changed: state derived from `pending_tasks`, nothing to decrement |
| No stable task identity (`_stash_timeout` returns a bool; expiry results drop `task_id`) | Blocking | Moot under derived state |
| Unsynchronized dict mutated from poller, timer, and command threads | Blocking | Kept: dedicated lock on the "previously active" set |
| New delivery prefix can truncate the URL past the wire limit | Blocking | Kept: budget backward from the URL |
| `delivery_failed` is a fourth end state the design ignored | Major | Moot: a `WHERE` clause |
| `die()` / `@reload` leave keepalive threads running | Major | Kept: `die()` stops the thread |
| `reply_target` alone merges targets across networks | Major | Kept: key by `(network, target)` |
| Captured `irc` silently does nothing on a zombie connection | Major | Kept: resolve from `world.ircs` per pass |
| Thread per submission floods the one-message-per-second budget | Major | Kept: one batched loop |
| Six-minute cap contradicts the 1800s job lifetime | Major | Kept: stated explicitly; typing gives up, the clip still arrives |
| Echoed prompt can carry IRC formatting codes | Minor | Kept: strip formatting from the echo |

Codex's own recommendation was a locked lease manager with idempotent per-task
leases and task IDs threaded through submission and every result. That solves
the release-signal problem; deriving the state from the database removes it,
for less code.

## Not in scope

- **Queue-depth messaging** ("you are second in line"). Still the open item
  from the shipping notes: it needs `GET /v1/videos` and a decision about how
  often to say it. The typing indicator covers the "is it alive" half of that
  question without any of the machinery.
- **ETA in the acknowledgment.** Deliberately declined; see above.
- **The 10 MB upload ceiling.** A clip over the limit silently falls back from
  `paste.boxlabs.uk` to local `httpUrlBase` storage. Observed again on
  2026-08-21 (`irc.rdrake.org/llm/vid_70202102d320ab88.mp4`). Real, unrelated
  to this work, and tracked in the shipping notes.
