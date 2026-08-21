# @animate progress and delivery UX

**Status:** Designed, not started (2026-08-21)
**Author:** Richard Drake (with claude)
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

Start a second keepalive at submission, in `_animate_for_assistant`, which is
the one place holding both `irc` and `msg` at the moment a job id comes back.
Stop it from the pending-task delivery path, on all three end states:
`completed`, `failed_terminal`, and `expired`. A render that fails must not
leave the channel watching the bot type.

**Refcounted per `reply_target`.** Two people in one channel can both have a
clip in flight; the second delivery must not cancel the first's indicator.
Increment on submission, decrement on delivery, send `+typing=done` when the
count reaches zero.

**Capped at six minutes.** Covers the worst observed case (about 119 seconds
queued plus about 135 rendering) with room to spare. Past the cap the
keepalive gives up on its own, so a job the poller never resolves cannot leave
the bot typing indefinitely.

**Restart drops it, deliberately.** The keepalive is a daemon thread with no
persistence. A container restart mid-render loses it, clients expire the tag
within six seconds, and the poller still delivers the clip — the existing
recovery path is untouched. Persisting a typing indicator across restarts
would be more machinery than the signal is worth.

### The done/active flicker

The planner turn's own stopper sends `+typing=done` when the turn ends, which
is a few seconds *after* the render keepalive has started for the same target.
Left alone that produces a visible done-then-active flicker.

The turn's stopper must skip its `done` when a render keepalive holds the same
target. The refcount above is the natural place to check: non-zero means
somebody else is still typing, so leave the state alone.

## Delivery carries its context

Today an animate delivery is the bare URL:

    <vibebot> https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4

Change it to match what `@draw`'s deferred path already sends:

    <vibebot> rdrake: your video is ready! "a corgi riding a unicorn" → https://…mp4

**This reverses a deliberate decision in `84dbb67`,** and the reasoning there
was sound: the IRCv3 `+draft/reply` tag already shows the request above the
message, so repeating it restates what the client is displaying. That holds on
clients that render replies and fails on the ones that do not, where a naked
URL arrives two minutes later attached to nothing.

The reply tag stays. This is additive: threading for clients that support it,
readable text for those that do not. It also makes animate consistent with
draw and code, which both spell out nick and prompt on deferred delivery —
animate was the only one that did not.

## Testing

- Typing starts at submission and survives past the planner turn.
- All three end states stop it; `failed_terminal` and `expired`
  especially, since those are the paths where a forgotten indicator would
  outlive the job.
- Two overlapping submissions to one target: the first delivery does not stop
  the second's indicator.
- The cap fires when no delivery ever arrives.
- The planner stopper does not send `done` while a render keepalive is active.
- The delivered line carries nick, truncated prompt, and URL, and still
  carries the reply tag.

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
