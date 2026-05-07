# Capabilities

Limnoria's access-control system attaches **capability strings** to bot users. The capability name usually mirrors a command name, so `rot13` is the capability for the `rot13` command.

## Forms a capability can take

For a command `Filter.rot13`:

| Form | Matches |
|------|---------|
| `rot13` | every plugin's `rot13` command |
| `Filter.rot13` | only the `Filter` plugin's `rot13` |
| `Filter` | every command in the `Filter` plugin |

Compound names like `user hostmask add` accept any of: `add`, `User`, `User.hostmask`, `User.hostmask.add`.

## Anticapabilities

Prefix a capability with `-` to **deny** it. Anticapabilities override capabilities.

- `rot13` → can run `rot13`
- `-rot13` → cannot run `rot13`
- `-Filter` → cannot run any `Filter` command

Most commands implicitly grant their capability to everyone, so adding the `-cap` form is how you restrict a command.

## Channel capabilities

Prefix with `#chan,`:

- `#chat,op` → op privileges in `#chat`
- `#chat,-Games` → block the `Games` plugin in `#chat`

When `echo` runs in `#chat`, the bot checks (in order): `-echo`, `echo`, `#chat,-echo`, `#chat,echo`, then the four equivalents for `Utilities` and `Utilities.echo`. First definite hit wins.

## Special built-in capabilities

| Capability | What it grants |
|------------|----------------|
| `owner` | everything, including loading plugins, connecting networks, editing global config, running shell-exec commands |
| `admin` | bot administration short of plugin loading: change nick, manage ignores, join/part channels, manage networks |
| `#chan,op` | all `Channel` commands for that channel: op, kban, channel ignore, channel config; implies all other `#chan,*` capabilities |
| `#chan,halfop` | a subset of channel-op privileges |
| `#chan,voice` | use of `voice` (self-voice); AutoMode auto-voices on join |
| `trusted` | run commands that may be slow or resource-heavy (e.g. `Math.icalc`) without being denied |

`admin` does **not** imply `#chan,op` for any channel — channel admin is granted per channel.

## Managing user capabilities

User scope (requires `admin`; admins can only grant capabilities they themselves hold):

```
@admin capability add <user> <capability>
@admin capability remove <user> <capability>
```

Channel scope (requires `#chan,op` for the channel):

```
@channel capability add <#chan> <user> <cap>      # in any channel
@channel capability remove <#chan> <user> <cap>
@channel capability add <user> <cap>              # in current channel
```

The `channel capability add` form is equivalent to `admin capability add <user> #chan,<cap>`, but uses the channel-op grant instead of global admin.

Inspect:

```
@user capabilities <user>
```

Worked examples:

```
@admin capability add alice admin                 # promote alice
@channel capability add #chat alice op            # give alice op in #chat
@admin capability add bob -Games                  # bob can never run Games commands
@channel capability add #chat bob -Games          # bob can't run Games in #chat
```

## Default capabilities (everyone)

Defaults apply to **every** caller, identified or not. Managed by an owner:

```
@defaultcapability add -<cap>            # disable a command/plugin globally
@defaultcapability remove -<cap>
@config setdefault capabilities          # reset to built-in defaults
```

Examples:

```
@defaultcapability add -user.register    # disable self-registration
@defaultcapability add -Games            # nobody can run Games unless explicitly granted
```

## Channel default capabilities

Apply to everyone in a specific channel:

```
@channel capability set <cap>            # current channel
@channel capability unset <cap>
@channel capability set <#chan> <cap>
```

Examples:

```
@channel capability unset -voice         # remove default anti-voice
@channel capability set voice            # everyone can self-voice
@channel capability set -Games           # nobody can use Games here
```

## Pattern: lock down noisy or risky commands

`trusted` is checked by certain commands (e.g. `Math.icalc`) **before** they run an expensive operation. Just granting the capability is enough — there is no separate `-Math.icalc` to add:

```
@admin capability add alice trusted      # alice can now use icalc, etc.
```

To deny commands or whole plugins for everyone unless explicitly granted:

```
# kill a plugin everywhere
@defaultcapability add -SomePlugin

# disable a single command everywhere
@defaultcapability add -SomePlugin.thecommand

# whitelist: deny by default, then grant per-user
@defaultcapability add -SomePlugin.thecommand
@admin capability add alice SomePlugin.thecommand
```

## Pattern: minimal-trust setup

For maximum safety, after configuring everything you need:

1. `@config supybot.commands.allowShell False` — block `Unix`, `@call`, etc.
2. Remove the `owner` capability from your account (or delete it). Privileged commands then become unavailable to everyone, including network operators who could spoof your hostmask. Recovery requires editing config files on disk.
3. Channel ops can still manage their channel via `#chan,op`. Global config is read-only from IRC.

## Pattern: hide discovery surface

The bot exposes its config and command list by default for ease of use. To make it boring to probe:

```
@defaultcapability add -config              # nobody can read config from IRC
@defaultcapability add -misc.list           # hide plugin/command listing
@defaultcapability add -misc.apropos
@defaultcapability add -plugin              # hide which plugins are loaded
@defaultcapability add -misc.version        # hide version string
```

Also consider:

- `supybot.reply.error.noCapability` — replace specific "you need cap X" errors with a generic one.
- `supybot.reply.error.detailed` — strip exception detail from error replies.
- Clear version strings out of `supybot.user`, `supybot.plugins.Owner.quitMsg`, `supybot.plugins.Channel.partMsg`.

Sensitive config values (passwords, tokens) are always hidden by default.
