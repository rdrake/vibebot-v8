# NickInMiddle plugin

A small `inFilter` plugin that rewrites incoming channel messages that
contain the bot's nick in the *middle* of the text, moving the nick to
the front. Limnoria's normal addressing logic then recognises the
message as addressed, with no core changes.

## Why it exists

On AfterNet, and in most casual IRC channels, people naturally write:

```
can you, vibebot, tell me the weather
```

The LLM plugin matches the bot's nick at the start or the end of a
message, including inside a `/me` action. NickInMiddle handles the
in-the-middle case, so the bot still responds naturally.

Limnoria's own `supybot.reply.whenAddressedBy.nick` and `.nick.atEnd`
settings do not decide this: the LLM plugin's `inFilter` tags every
non-command line `addressed=''` to keep plain English out of the command
tokenizer, so its own matcher is the one that runs.

## What it does

For every channel `PRIVMSG`, NickInMiddle:

1. Skips the message unless `enabled` is `True` for that channel and
   network.
2. Skips PMs and CTCP or ACTION messages.
3. Looks for the bot's nick, or any nick configured under
   `supybot.reply.whenAddressedBy.nicks`, appearing strictly between
   two separators (space, tab, comma, colon, semicolon). A trailing
   `?`, `.`, or `!` comes off before the comparison, so `vibebot?`
   counts, and matching uses RFC 1459 case folding, so `vibe{bot}`
   matches `Vibe[bot]`.
4. Rewrites the `PRIVMSG` with the bot's own nick at the front — a
   configured alias is replaced by the real nick, not carried through —
   then lets Limnoria dispatch the message normally.

Example rewrite, with bot nick `vibebot`:

| Before | After |
|--------|-------|
| `can you, vibebot, tell me the weather` | `vibebot can you, tell me the weather` |

The separator before the nick stays where it was; the one after it goes
with the nick.

## Loading and configuration

```
@load NickInMiddle
```

`supybot.plugins.NickInMiddle.enabled` defaults to `True`, so loading
the plugin is all it takes. It is a channel value, scoped per network
and per channel, so turn it off where you don't want it:

```
@config channel #noisy plugins.NickInMiddle.enabled False
```
