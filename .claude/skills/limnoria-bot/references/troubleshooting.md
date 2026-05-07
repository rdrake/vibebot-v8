# Troubleshooting

## "I don't recognize that user" / missing `owner` capability

You haven't given the bot a way to recognise you. Three fixes, in increasing convenience:

1. Identify each session: `/query mybot` → `identify <user> <password>`.
2. Bind a hostmask: `@hostmask add` (in PM) — see `references/users.md`. Use a narrow pattern.
3. Bind a services account: load `NickAuth`, then `@nickauth nick add <bot-user> <services-account>`.

Verify with `@whoami` after the next interaction.

If you've lost the owner password entirely:

1. Stop the bot (`@quit` if you can still reach it; otherwise `SIGTERM` the process).
2. Run `supybot-adduser <path/to/botname.conf>` and create a new owner account (or any name; you can rename later). The script writes directly into `conf/users.conf`.
3. Start the bot. Identify as the new account. Use `@admin capability add` to copy capabilities across, then `@user unregister <old-name>` if you no longer need the old account.

## "Spurious or missing right brackets" / "not a valid command"

Bracket characters in nicks or args are being parsed as nested commands. Two fixes:

```
# quote the offending arg
@some "weird[nick]" arg

# or change the bracket pair
@config supybot.commands.nested.brackets <>

# or disable nesting entirely
@config supybot.commands.nested False
```

## Bot doesn't respond to commands at all

In order:

1. Is the bot in the channel? `@channels` (in PM).
2. Does the message use a recognised prefix? Check `@config supybot.reply.whenAddressedBy.chars` and `...strings`. Default is **the bot's own nick**: `mybot: list`.
3. Is the bot ignoring you? `@admin ignore list`.
4. Is the command capability denied? `@config supybot.capabilities` and `@user capabilities <you>`.
5. Are you flooded out? Check `supybot.abuse.flood.command.punishment`.

## "Error: ..." with no detail

Server-side detail is in the log. Find the directory: `@config supybot.directories.log`. Tail `messages.log`. Crank verbosity if needed:

```
@config supybot.log.level DEBUG
```

(Remember to set it back — DEBUG is noisy.)

## Hand edits to `botname.conf` got overwritten

The running bot flushes its registry periodically. Either stop the bot before editing, or:

```
@config supybot.flush False
# edit files
@config reload
@config supybot.flush True
```

## Plugin won't load

```
@load Foo
# Error: Couldn't load plugin Foo: ...
```

Causes:

- Wrong directory: `@config supybot.directories.plugins` doesn't include the path.
- Import error inside the plugin: tail `messages.log` for the traceback.
- Name collision: a stale copy in another plugins dir is shadowing the new one. Check `@list --unloaded`.

## Connecting to multiple networks

```
@connect ircnet2 irc.example.org 6697
@config networks.ircnet2.ssl True
@config networks.ircnet2.channels #foo,#bar
```

Each subsequent IRC command can be qualified with the network: `@network ircnet2 ...` or used directly when sending the command from that network.

## SASL fails silently

The only reliable check is to whois the bot post-connect:

```
/whois mybot
# look for "is logged in as" or "Account: ..."
```

If absent:

- Network may not support SASL on that port — try TLS port + `supybot.networks.<n>.ssl True`.
- Wrong account name in `sasl.username` — case matters on some networks.
- For SASL EXTERNAL: cert isn't registered, or `certfile` path isn't readable by the bot user, or the file lacks both cert + private key.

Ask the network's operators to confirm SASL is enabled; turning on `supybot.log.level DEBUG` shows the CAP negotiation.

## Bot loops snarfing URLs with another bot

Two bots fetching each other's snarfs forever. Mitigations:

```
@config supybot.snarfThrottle 30
# or disable snarfing in that channel
@config channel supybot.plugins.Web.titleSnarfer False
```

## "More" pagination eating output

Output longer than `supybot.reply.mores.length` is queued; users fetch it with `@more`. Knobs:

```
supybot.reply.mores.length            # bytes per chunk (default 460)
supybot.reply.mores.maximum           # queue depth per user
supybot.reply.mores.instant           # send first N chunks at once
```

## Bot won't quit / stuck on shutdown

`@quit` is the clean path. If it hangs, `SIGTERM` the process. As a last resort `SIGKILL`, but you may lose unflushed config — on next start the bot reads what was last written, not what was in memory.

## Reset a single setting to default

```
@config setdefault supybot.<key>
```

Reset every channel- or network-scoped override of a key:

```
@config reset channel supybot.<key>
@config reset network supybot.<key>
```

Reset the entire default capability list:

```
@config setdefault capabilities
```

## Diagnosing capability denials

```
@user capabilities <you>                   # what you have
@config supybot.capabilities               # what's granted globally by default
@channel capability list <#chan>           # what the channel grants/denies
```

Remember: `-cap` overrides `cap`, channel overrides global, more-specific overrides less-specific. The first explicit `-cap` in the chain wins.
