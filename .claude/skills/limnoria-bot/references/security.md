# Security hardening

Security in Limnoria is mostly about reducing who can issue privileged commands, what privileged commands can do, and how much trust the bot places in IRC network transport.

## Owner and network-operator risk

An IRC network operator can potentially spoof hostmasks or observe plaintext traffic. Do not rely on hostmask-only owner access on untrusted networks.

Hardening sequence:

```
@config supybot.commands.allowShell False
@config supybot.reply.error.detailed False
@config supybot.reply.error.noCapability False
```

After setup, consider removing the `owner` capability or deleting the owner account entirely. Recovery then requires shell access and editing config files, but IRC-originated global admin becomes unavailable.

## SSL/TLS verification

TLS without certificate verification does not prevent active man-in-the-middle attacks.

```
@config supybot.networks.<network>.ssl True
@config supybot.protocols.ssl.verifyCertificates true
```

If the network uses a private CA:

```
@config networks.<network>.ssl.authorityCertificate /path/to/ca.crt
```

If the network publishes fingerprints instead:

```
@config supybot.networks.<network>.ssl.serverFingerprints <fingerprint1> <fingerprint2>
```

Only pin fingerprints or CA files obtained from the network through a trusted channel.

## Public surface reduction

For quiet production bots, consider hiding discovery and version details from unauthenticated users:

```
@defaultcapability add -config
@defaultcapability add -misc.list
@defaultcapability add -misc.apropos
@defaultcapability add -plugin
@defaultcapability add -status.commands
@defaultcapability add -misc.version
```

Sensitive config values are hidden by default, but public config, loaded plugin names, paths, versions, and capability errors can still reveal useful information.

## Risky plugins

Do not load shell/eval-capable plugins (`Debug`, `Unix`, owner `@call` access through shell-capable command handling) on production bots unless the host and network are trusted. Downloaded plugins and manually installed plugins run as the bot's OS user.

## Flood and abuse controls

Useful knobs:

```
@config supybot.reply.mores.length 460
@config supybot.reply.mores.maximum 3
@config supybot.abuse.flood.command.punishment 300
@defaultcapability add -scheduler
```

Tune these for channel size and bot role; do not hide diagnostics permanently if you need to ask upstream for help.
