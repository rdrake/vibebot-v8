# Channels

Channel operations live in two plugins: `Admin` (join/part) and `Channel` (mode-setting and per-channel admin).

## Joining and parting

```
@join <#channel> [<key>]
@part <#channel> [<reason>]
@channels                              # list channels the bot is in
@nicks <#channel>                      # users in a channel (current channel if omitted)
```

Persistence: `@join` adds the channel to `supybot.networks.<net>.channels` so the bot rejoins on restart. To join transiently, you'd have to remove the channel from that registry afterwards.

To stop the bot from following invites:

```
@config supybot.alwaysJoinOnInvite False
```

To make the bot invite-only for itself:

```
@config supybot.networks.<net>.channels.key.<#chan> <key>
```

## Granting modes (requires `#chan,op`)

```
@op [<#chan>] [<nick>...]              # +o; defaults to caller in current channel
@deop [<#chan>] [<nick>...]            # -o
@halfop [<#chan>] [<nick>...]
@dehalfop [<#chan>] [<nick>...]
@voice [<#chan>] [<nick>...]
@devoice [<#chan>] [<nick>...]
@mode <#chan> <modes> [<args>]         # raw mode string, e.g. +mi
```

Without arguments, `op`/`voice`/`halfop` operate on the caller in the current channel — provided they have the matching `#chan,op` / `#chan,halfop` / `#chan,voice` capability.

## Removing users (requires `#chan,op`)

```
@kick [<#chan>] <nick> [<reason>]
@ban add [<#chan>] <hostmask|nick> [<expires-seconds>]
@ban remove [<#chan>] <hostmask>
@ban list [<#chan>]
@kban [<#chan>] <nick> [<expires>] [<reason>]   # kick + ban
@unban [<#chan>] [<hostmask>]
```

Expirations are in seconds; `0` means permanent. The bot will auto-unban when the timer fires (provided it's still in the channel).

## Channel-scoped ignores

Different from a ban — the bot just stops responding to that user in that channel.

```
@channel ignore add [<#chan>] <hostmask> [<expires>]
@channel ignore remove [<#chan>] <hostmask>
@channel ignore list [<#chan>]
```

## Channel-scoped configuration

Anything marked `#` in `@config list` can be overridden per-channel. Common ones:

```
@config channel supybot.reply.whenAddressedBy.chars !
@config channel supybot.reply.withNickPrefix False
@config channel supybot.plugins.Web.titleSnarfer False
```

See `references/configuration.md` for full syntax.

## Channel-scoped capabilities

To allow/forbid commands or whole plugins per channel — see `references/capabilities.md`. Quick recap:

```
@channel capability add <#chan> <user> op       # promote user in this channel
@channel capability set -Games                  # nobody can run Games here
@channel capability unset -voice                # let anyone self-voice
```

## Auto-mode on join

Load `AutoMode` to automatically op/voice users by capability when they join:

```
@load AutoMode
@channel capability add <#chan> alice op        # alice gets +o on join
@channel capability set voice                   # everyone gets +v on join
```

Knobs in `supybot.plugins.AutoMode`:

- `enable` — master switch (default `True`).
- `op`, `halfop`, `voice` — which modes the plugin will grant (booleans).
- `fallthrough` — if `True`, a denied higher mode falls back to the next one (e.g. denied op falls back to voice).

## Topics

The `Channel` plugin can manage topics:

```
@topic                                  # show current
@topic <new topic>
@topic add <segment>                    # append a separated segment
@topic remove <n>                       # remove the n-th segment
@topic replace <n> <new>
@topic separator <string>               # default " || "
```

## Quitting and disconnecting

```
@disconnect [<reason>]                  # disconnect from current network
@reconnect [<network>]
@quit [<reason>]                        # owner-only: stops the bot entirely
```
