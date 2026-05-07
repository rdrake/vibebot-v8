# Users and authentication

Limnoria has its own user database, stored in `users.conf`. A bot user is **distinct** from an IRC nick: you log in to the bot to gain capabilities, regardless of what nick you're using on the network.

## The first user

`supybot-wizard` typically creates an initial owner user. If it didn't, stop the bot and run:

```
supybot-adduser botname.conf
```

Answer the prompts and grant the `owner` capability. Then start the bot.

## Adding more users

Two paths:

```
# self-service in PM (must be enabled — see below)
@register <name> <password>

# offline owner path
supybot-adduser botname.conf
```

There is no core `user add` IRC command. In normal operation, users register themselves, then an admin/owner grants capabilities as needed. `user.register` is enabled by default. To disallow self-registration:

```
@defaultcapability add -user.register
```

## Logging in

Always send the password in a private message:

```
/query mybot
identify <name> <password>
```

Confirm with `@whoami`.

Log out with `@unidentify`. Change password with `@user set password [<name>] <old-password> <new-password>` (PM only; owner may omit/correct the old password requirement when changing another user's password).

## Automatic login (optional)

Two mechanisms; both let you skip `identify` after the bot restarts.

### NickAuth (recommended where available)

Ties a bot account to a network services (NickServ) account. Requires the network to support `extended-join` + `WHOX` for fully automatic activation; otherwise the bot will check on next interaction.

```
@load NickAuth
@nickauth nick add <bot-user> <services-account>
@nickauth nick add <network> <bot-user> <services-account>   # explicit network
@nickauth nick list
@nickauth nick remove <bot-user> <services-account>
```

### Hostmask login

Works on any network. Match a specific `nick!user@host` pattern:

```
@hostmask add                              # add your current hostmask to your account
@hostmask add <name> <hostmask> [password]
@hostmask list
@hostmask remove <name> <hostmask>
```

**Security:** keep the hostmask narrow. `nick!user@your.cloak` for a unique cloak/vhost is fine. `*!*@*.dial-up.example` is not — anyone reconnecting from that ISP can hijack the account.

## Bot's own services identification

Separate concern: this is how the **bot** identifies to NickServ on the network it joins. Configured per-network. In recommended-first order:

### SASL PLAIN

```
@config networks.<network>.sasl.username <account>
@config networks.<network>.sasl.password <password>
```

### CertFP / SASL EXTERNAL

Generate a TLS client cert, register it with NickServ (`/msg NickServ cert add`), and point the bot at it:

```
@config networks.<network>.certfile /path/to/client.pem
```

Once SASL EXTERNAL works, `sasl.username` and `sasl.password` can be removed.

### Server password (`PASS`)

For ZNC, Soju, or networks that take `services_account:password` in `PASS`:

```
@config networks.<network>.password <password>
```

### Services plugin (fallback)

If SASL is unavailable, load `Services` and configure NickServ commands:

```
@load Services
@config supybot.plugins.Services.NickServ NickServ
@config networks.<network>.plugins.Services.NickServ.password <password>
```

This identifies the bot **after** connecting, so other users see it briefly unauthenticated.

## Inspecting

```
@user list                                 # all bot users
@user list <hostmask>                      # users matching a hostmask
@user capabilities <name>                  # capabilities of a user
@user hostmask list <name>                 # hostmasks bound to a user
```

## Removing or renaming

```
@user unregister <name>                    # delete bot account
@user changename <old> <new>
```

## Ignoring users on IRC

Separate from the user database: silence noisy users.

```
@admin ignore add <hostmask> [<expires>]
@admin ignore remove <hostmask>
@admin ignore list
```

Per-channel ignores live in the `Channel` plugin: `@channel ignore add ...`.
