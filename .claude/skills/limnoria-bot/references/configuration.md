# Configuration

Limnoria's behaviour is driven by a hierarchical registry rooted at `supybot.*`. The running bot writes its registry back to `botname.conf` periodically, so the file on disk and the running state can briefly diverge. **Always prefer IRC `@config` commands over editing the file.**

## Type symbols in `@config list`

When you list a group, each child is decorated:

- `@` — child is itself a group (you can `@config list` it).
- `:` — value is per-network.
- `#` — value is per-channel. Often combined as `#:` (per-channel-and-per-network).
- (none) — global only.

## Common top-level groups

| Group | Purpose |
|-------|---------|
| `supybot.directories` | where logs, plugins, data, conf live |
| `supybot.networks` | one subgroup per known network (servers, SASL, channels, etc.) |
| `supybot.log` | log verbosity, format, file vs stdio |
| `supybot.plugins.<Name>` | each loaded plugin's settings |
| `supybot.replies` | standard reply strings (e.g. `success`, `error`) |
| `supybot.reply` | how the bot formats output (line lengths, addressing, mores) |
| `supybot.commands` | command parsing, nesting brackets, `allowShell` |
| `supybot.abuse` | flood protection (e.g. `supybot.abuse.flood.command.punishment`) |
| `supybot.capabilities` | default capabilities every user gets |
| `supybot.databases` | which DB plugins use, on-disk layout |

## Reading

```
@config supybot.nick
@config help supybot.snarfThrottle    # description + current value
@config list supybot                  # children of root
@config list supybot.reply
@config search whenAddressed          # find keys by substring
@config default supybot.nick          # show the built-in default
```

## Sharing sanitized config

Use `Config.export` when you need to share config for support without exposing private registry values:

```
@config export /tmp/bot-public.conf
```

The export contains public variables only; passwords and other private values are omitted or hidden. Still review the file before posting it, because channel names, network names, plugin choices, and paths may be sensitive.

## Setting

```
@config supybot.reply.whenAddressedBy.chars @$
@config supybot.snarfThrottle 30
```

The Config plugin validates the new value against the option's type before writing.

## Network- and channel-specific overrides

Many keys (those marked `:` or `#` in `@config list`) can be scoped:

```
# in the channel you want to change
@config channel supybot.reply.whenAddressedBy.chars !

# explicit form for a channel you're not currently in
@config channel <network> <#channel> <key> <value>

# for every channel on a network
@config network supybot.reply.whenAddressedBy.chars !
```

Precedence: **channel > network > global**.

Reset a scoped override:

```
@config reset channel <key>
@config reset network <key>
```

Restore a key to its built-in default:

```
@config setdefault <key>
@config setdefault capabilities       # restore default capability list
```

## Editing the file by hand

Only do this if the bot is offline or `@config` cannot reach it.

1. Either stop the bot, or set `@config supybot.flush False` on the running bot to suspend periodic writes.
2. Edit `botname.conf` (and the `conf/` sidecar files: `users.conf`, `channels.conf`, `networks.conf`, `ignores.conf`, `userdata.conf`).
3. Bring the bot back / `@config reload` to re-read from disk. On POSIX, `kill -HUP <pid>` does the same as `config reload` if the Config plugin is loaded.
4. Re-enable `supybot.flush` if you set it to `False`.

## Cosmetic settings worth knowing

```
supybot.nick                              # bot nick
supybot.user                              # IRC ident/realname
supybot.reply.whenAddressedBy.chars       # additional command prefixes
supybot.reply.whenAddressedBy.strings     # full-word prefixes
supybot.reply.mores.length                # max length per line before paginating
supybot.reply.mores.maximum               # max queued "more" pages per user
supybot.reply.withNickPrefix              # prepend "<nick>: " to replies
supybot.commands.nested                   # enable [nested] commands
supybot.commands.nested.brackets          # change the bracket pair
supybot.commands.allowShell               # lets owner load shell-exec plugins
```
