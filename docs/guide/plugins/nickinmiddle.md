# NickInMiddle plugin

A small `inFilter` plugin that rewrites incoming channel messages that
contain the bot's nick in the *middle* of the text, moving the nick to
the front. Limnoria's normal addressing logic then recognizes the
message as addressed, with no core changes.

## Why it exists

On AfterNet, and in most casual IRC channels, people naturally write:

```
can you, vibebot, tell me the weather
```

Stock Limnoria only matches the bot's nick at the start or end of a
message (through `supybot.reply.whenAddressedBy.nick` and
`.nick.atEnd`). NickInMiddle handles the in-the-middle case, so the bot
still responds naturally.

## What it does

For every channel `PRIVMSG`, NickInMiddle:

1. Skips the message unless `enabled` is `True` for that channel and
   network.
2. Skips PMs and CTCP or ACTION messages.
3. Looks for the bot's nick, or any nick configured under
   `supybot.reply.whenAddressedBy.nicks`, appearing strictly between
   two word-boundary separators (space, comma, colon, semicolon).
4. Rewrites the `PRIVMSG` so the nick sits at the front, then lets
   Limnoria dispatch the message normally.

Example rewrite, with bot nick `vibebot`:

| Before | After |
|--------|-------|
| `can you, vibebot, tell me the weather` | `vibebot can you tell me the weather` |

## Loading and configuration

```
@load NickInMiddle
@config plugins.NickInMiddle.enabled True
```

The plugin is channel-aware: enable or disable it independently per
network and per channel through `supybot.plugins.NickInMiddle.enabled`.
