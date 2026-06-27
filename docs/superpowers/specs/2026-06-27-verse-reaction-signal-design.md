# Design — Verse Reaction Signal (recency-attributed, measurement-only)

**Status:** DRAFT for review (2026-06-27). Slice 2-adjacent: a cleaner approval signal for the
verse landing-rate instrument. Supersedes nothing; complements
[`2026-06-27-verse-landing-instrument-design.md`](2026-06-27-verse-landing-instrument-design.md).

## Why

The shipped landing-rate instrument (`taste_report`) mines fc42's re-pastes + praise from
ChannelLogger text logs. It is **positive-only** (silence ≠ dislike) and **indirect** (a heuristic
over chat text). IRC reactions give a cleaner signal: when fc42 reacts 👍/👎 to a bot line, that is
an **explicit, bidirectional, zero-token** judgement of a specific response.

Feasibility was confirmed against the live stack: the bot already *sends* IRCv3 `+draft/react`
reactions (`service.py:send_reaction`, `plugin.py:_react`), so AfterNet negotiates `message-tags`
and carries reactions on the wire, and fc42's client can emit them. The only missing half is
**inbound capture**.

## Goal / Non-goals

**Goal:** capture inbound 👍/👎 reactions to the bot's verse lines, attribute each to the verse turn
it reacts to, persist it offline, and surface approve/disapprove counts (pre/post the 2026-06-22
exemplar rollout) in the landing report. Measurement only.

**Non-goals (explicit YAGNI):**
- No exact-msgid anchoring via the `echo-message` capability (a bot-wide change; see "Approach").
- No bot behavior change: the bot does **not** acknowledge, reply to, or adapt to reactions.
- No configurable emoji maps / reactor allowlists / tunable window in v1 (module constants).
- No SQLite or verse-schema change; no backfill (no historical reactions exist — signal accrues
  forward, like the instrument itself).

## Approach — recency attribution, no new capability

A reaction arrives as a `TAGMSG` *from the reactor* carrying `+draft/react=<emoji>` and
`+draft/reply=<the bot line's msgid>`. The bot cannot map that msgid back to its own line, because
it never learns the msgid of its own outgoing messages (that would require negotiating
`echo-message` — a bot-wide change where the bot starts receiving copies of its own messages and
every handler must ignore self-echoes). Rejected as disproportionate.

Instead we attribute **by recency**: the bot remembers the last line it said in each channel **when
that line was a verse turn**, and a reaction shortly after is attributed to it. fc42 reacts to the
most recent line ~always, so accuracy is high in practice. This also makes capture **inherently
verse-scoped**: in non-verse channels there is no remembered line, so nothing is recorded — no
separate "is verse enabled here" predicate needed.

## Architecture & data flow

```
verse reply sent ─▶ record last_bot_line[(network,channel)] = {text, ts}   (only for verse turns)
reactor sends TAGMSG +draft/react ─▶ doTagmsg:
    gate: verseReactionCaptureEnabled(channel)?  (default True; kill-switch)
    read last_bot_line under _irc_send_lock; within recency window?  ──no──▶ drop (not verse)
    classify emoji ─▶ approve | disapprove | other
    append one JSON line to reactions.jsonl
(offline) taste_report --reactions reactions.jsonl ─▶ "Explicit 👍/👎" section, pre/post + monthly
```

## Component 1 — outbound tagging (small plumb-through)

- Add `was_verse: bool` to `AssistantResult` (`service.py`); set `True` when
  `route_profile == PROFILE_VERSE` (near `service.py:3810`), default `False`.
- Carry it to the send site: `_dispatch_assistant_reply` → `_send_long_reply`
  (`plugin.py:~2438`/`~2500`).
- New `self._last_bot_line: dict[tuple[str, str], dict]` in `LLM.__init__`, keyed by
  `(irc.network, channel)`. **Only verse turns are stored** (`{"text": line, "ts": time.time()}` —
  wall-clock, so the recency interval, `recency_s`, and the JSONL ISO timestamp all share one clock).
  Written right after `_safe_reply` returns, where the worker thread already holds
  `_irc_send_lock` (`plugin.py:690`, `:2294`/`:2333`). Write is wrapped so any error is logged and
  swallowed — it must never disturb the reply path.

Non-verse replies, scheduled tasks, pending-task delivery, storybook/@story are left untouched
(they never set `last_bot_line`), so reactions to them are never mis-counted as verse.

## Component 2 — inbound capture (new `doTagmsg`)

New `doTagmsg(self, irc, msg)` on the `LLM(callbacks.Plugin)` class (Limnoria auto-dispatches
`do<COMMAND>` methods). The whole body is wrapped in try/except-log-swallow.

1. **Cheap early-return:** if `msg.server_tags` has no `+draft/react`, return immediately
   (typing-indicator TAGMSGs and any other client tags cost one dict-get).
2. **Gate:** `channel = msg.args[0]`; if not a channel or `not verseReactionCaptureEnabled(channel)`,
   return.
3. **Read tags:** `emoji = server_tags["+draft/react"]`; `target_msgid = server_tags.get("+draft/reply")`
   (stored raw for audit / future exact-anchoring); `reactor = msg.nick`.
4. **Attribute by recency:** under `_irc_send_lock`, read `last_bot_line[(network, channel)]`. If
   absent or `now - ts > RECENCY_WINDOW_S` (300), **drop** — the reaction is not attributable to a
   verse turn.
5. **Classify:** normalize skin-tone / variation selectors, then 👍→`approve`, 👎→`disapprove`,
   else `other` (recorded, not scored).
6. **Persist:** append one JSON line to `<verse-dir>/reactions.jsonl` under a dedicated
   `_reaction_log_lock` (not the send lock, to avoid coupling a file write to the send path).

The classification + attribution logic lives in a **pure, fully-tested helper**; `doTagmsg` is thin
IRC glue.

### Persistence format — `reactions.jsonl`

Append-only JSONL at `<verse-dir>/reactions.jsonl` (e.g. `/config/data/verse/reactions.jsonl`).
One object per reaction event:

```json
{"ts": "2026-06-27T12:06:58Z", "network": "afternet", "channel": "#afternet",
 "reactor": "fc42", "emoji": "👍", "sentiment": "approve", "was_verse": true,
 "target_msgid": "abc123", "recency_s": 7.4, "verse_excerpt": "Diarrhoea Dan, the Year Eight…"}
```

Chosen over a SQLite table because it needs **no schema migration**, is append-safe for a single
writer, is human-inspectable, and matches the offline-reporter pattern (`taste_report` already reads
files). `was_verse` is always `true` by construction (we only record matched verse reactions);
the field is kept explicit for forward-compatibility. `verse_excerpt` is capped (~120 chars).

## Component 3 — offline reporter (extend `taste_report`)

Add an optional `--reactions <file>` argument to `taste_report`. When present, parse the JSONL and
append an **"Explicit 👍/👎 reactions"** section to the same report, reusing `per_100` / `per_day` /
`_month` / `BucketStats`:

- **Headline pre/post table:** window | reactions | approve | disapprove | net | distinct reactors.
- **Monthly trend** of the same.
- **Recent examples** (latest-first, capped): `date [sentiment by reactor] excerpt`.
- **Caveats block** (honest by construction): recency may mis-attribute a reaction to a non-bot
  message that lands within the window; reactions require a reaction-capable client (absence ≠
  dislike); both polarities are captured now; thin buckets flagged `thin sample`.

When `--reactions` is omitted, `taste_report` output is **byte-for-byte unchanged** — the existing
17 tests are untouched.

## Config & safety

One new registry key: **`supybot.plugins.LLM.verseReactionCaptureEnabled`** — per-channel
`ChannelValue` Boolean, **default `True`**.

Rationale for default-on (operator preference, 2026-06-27): the handler is measurement-only, the
new paths are exception-wrapped, and capture is inherently verse-scoped (Component 1 stores only
verse lines), so a blanket default neither collects bot-wide nor risks verse/chat. The flag remains
a **per-channel kill-switch** if reaction capture ever needs disabling without disabling verse.

The trade-off accepted: the feature reaches prod **enabled** on the next auto-deploy with no
dark-validation window. Acceptable because the worst realistic failure is log noise or dropped
events (the reporter tolerates malformed lines), never a disturbed reply path.

## Threading & concurrency

- `last_bot_line` is **written** on a worker thread that already holds `_irc_send_lock`; it is
  **read** in `doTagmsg` on the main IRC thread, which must acquire `_irc_send_lock` for the read
  (per the outbound-path audit).
- `reactions.jsonl` appends happen on the main IRC thread under a dedicated `_reaction_log_lock`.
- Both new code paths are wrapped in try/except → log → swallow.

## Testing strategy

Mirror `taste_report`/`taste_mine`: a pure, fully-tested core + thin `pragma: no cover` IRC glue.

- **Pure core (100% line+branch target):**
  - emoji → sentiment classification, incl. skin-tone / variation-selector normalization and the
    `other` bucket;
  - recency attribution (`last_bot_line + reaction_event + now → record | drop`), incl. window
    boundary, missing line, stale line;
  - reactions-JSONL parse + monthly/pre-post bucketing + net/per-reactor aggregation in the
    reporter; malformed-line tolerance.
- **Thin glue (lightly covered):** `doTagmsg` wiring, the send-hook store, `was_verse` plumb-through.

## Integration points to confirm during planning

(approximate locations from exploration — verify exact lines when implementing)
- `AssistantResult` definition + `route_profile == PROFILE_VERSE` set point (`service.py:~3810`).
- `_dispatch_assistant_reply` / `_send_long_reply` send site (`plugin.py:~2438`/`~2500`); confirm
  `irc.network` + channel are both in scope there.
- Confirm Limnoria surfaces inbound **client** tags (`+draft/react`, `+draft/reply`) in
  `msg.server_tags` keyed with the leading `+` (consistent with how `send_reaction` builds them
  outbound).
- Registry-key registration site for `verseReactionCaptureEnabled` (`config.py`).
- Verse store base dir for `reactions.jsonl` placement (`/config/data/verse`).
