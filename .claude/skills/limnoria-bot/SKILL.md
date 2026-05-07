---
name: limnoria-bot
description: Use when operating a Limnoria IRC bot: installing or starting it, loading plugins, changing supybot.* registry config, managing bot users/NickAuth/SASL/hostmasks/capabilities, channel ops, Aka aliases, MessageParser triggers, botname.conf edits, or troubleshooting user recognition and capability errors.
---

# Limnoria Bot Operations

Limnoria is a Python IRC bot (fork of Supybot). Most operation happens **over IRC** by sending privileged commands to the bot after identifying as an `owner` user. The config file (`botname.conf`) is normally written by the bot itself — prefer IRC commands over hand-editing.

## Mental model

- **Plugins** group commands. The bot's name is the default address prefix; configurable extra prefixes (e.g. `@`) are set in `supybot.reply.whenAddressedBy.chars`.
- **Config registry** is a hierarchical key-value tree rooted at `supybot.*`. Some keys are global, some per-network, some per-channel. Managed via the `Config` plugin.
- **Capabilities** are string flags ("op", "admin", "owner", "-rot13", "#chan,op", …) attached to user accounts. They gate every command. Anticapabilities (leading `-`) override capabilities.
- **User accounts** are the bot's own user database, separate from IRC nicks/services. You log in with `identify <user> <password>`. Optional auto-login via NickAuth (services account) or hostmask.

## Command address forms

The bot recognises commands prefixed by its nick (`mybot: list`) **or** by any character in `supybot.reply.whenAddressedBy.chars` (often `@`, e.g. `@list`). Choose whichever the operator's running config uses; examples below use `@`.

## Workflow selection

Pick the reference for the task at hand. Read only what is relevant — these references are independently scoped.

| Task | Reference |
|------|-----------|
| Read or change a `supybot.*` setting (globally, per-network, per-channel); edit `botname.conf` | [references/configuration.md](references/configuration.md) |
| Load, unload, reload, or list plugins; install third-party plugins | [references/plugins.md](references/plugins.md) |
| Add bot users, identify, set up auto-login (NickAuth/hostmask), SASL/CertFP for the bot itself | [references/users.md](references/users.md) |
| Grant/revoke `owner`, `admin`, `#chan,op`, `voice`, `trusted`; default and channel-default caps | [references/capabilities.md](references/capabilities.md) |
| Make the bot join/part, op/voice/kick/ban users, manage channel modes | [references/channels.md](references/channels.md) |
| Create aliases (Aka), regex triggers (MessageParser), or use nested commands | [references/automation.md](references/automation.md) |
| First-time install, `supybot-wizard`, starting/stopping the bot, log files | [references/install_and_runtime.md](references/install_and_runtime.md) |
| Harden SSL/TLS, owner access, shell-exec risk, error disclosure, and public command surface | [references/security.md](references/security.md) |
| Back up or restore config, user DB, channel DB, plugin data, and sanitized config exports | [references/backup_and_recovery.md](references/backup_and_recovery.md) |
| "Bot doesn't recognise me", missing `owner`, brackets-in-nick, common FAQ | [references/troubleshooting.md](references/troubleshooting.md) |
| Validate whether this skill guides agents correctly under realistic operator tasks | [references/pressure_scenarios.md](references/pressure_scenarios.md) |

## Hard rules

- **Never run privileged commands without first identifying.** Send `identify <user> <password>` in a private message (`/query`), never in a channel.
- **Never assign wide hostmasks** like `nick!user@*` to bot users — anyone matching the pattern can use that account's capabilities.
- **Hand-editing `botname.conf` requires stopping the bot or setting `supybot.flush` to `False` first**, otherwise the running bot will overwrite your changes on its next periodic flush. Reload with `@config reload` (or `SIGHUP` on POSIX).
- **`supybot.commands.allowShell` defaults to `True`** to make `PluginDownloader` work. If the bot does not need to install plugins from IRC, set it to `False` to deny shell-execution plugins (`Unix`, `@call`, etc.) to anyone — including a compromised owner account or a malicious network operator.
- **The first command to discover anything is `@help <name>` and `@list [<plugin>]`.** When you don't know the exact syntax of a command, ask the bot — `@help` returns the canonical signature.

## Quick reference

```
@list                             # show loaded plugins
@list <Plugin>                    # show commands in a plugin
@help <command>                   # canonical signature for a command
@apropos <substring>              # find commands by substring

identify <user> <password>        # log in (PM only)
@user list                        # list bot user accounts
@whoami                           # which bot account am I logged into?

@config <key>                     # read a setting
@config <key> <value>             # set a setting
@config help <key>                # description + current value
@config list <group>              # list children of a config group
@config search <substring>        # find keys by substring
@config channel [<net>] [<chan>] <key> [<value>]
@config network <key> [<value>]
@config reload                    # re-read botname.conf from disk

@load <Plugin>                    # load a plugin
@unload <Plugin>                  # unload a plugin
@reload <Plugin>                  # reload a plugin's code

@join <#channel>                  # join channel
@part <#channel> [reason]         # leave channel
@op <#channel> [<nick>]           # give op
@voice <#channel> [<nick>]        # give voice
@kban <#channel> <nick> [reason]  # kick + ban
```
