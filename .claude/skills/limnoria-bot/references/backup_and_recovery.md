# Backup and recovery

Back up before upgrades, manual config edits, plugin installs, or owner-account recovery.

## What to back up

Minimum:

```
botname.conf
conf/users.conf
conf/channels.conf
conf/networks.conf
conf/ignores.conf
conf/userdata.conf
data/
plugins/
```

Optional:

```
logs/
backup/
```

`users.conf` and `networks.conf` can contain password hashes, capabilities, hostmasks, SASL passwords, server passwords, and service credentials. Store backups with the same care as the live config.

## Live backup checklist

Prefer stopping the bot. If you cannot:

```
@flush
@upkeep
```

Then copy `botname.conf`, `conf/`, `data/`, and custom `plugins/` atomically enough for your deployment. For SQLite-backed plugin data, stop the bot if consistency matters.

## Before manual edits

```
@config supybot.flush False
# copy files
# edit files
@config reload
@config supybot.flush True
```

Stopping the bot is cleaner. If reload fails, restore the copied files and restart.

## Sanitized support bundle

For sharing config publicly:

```
@config export /tmp/bot-public.conf
```

Review the export before posting it. It should omit private registry values, but topology, plugin names, channels, and paths may still be sensitive.

## Owner password recovery

If the owner password is lost:

1. Stop the bot.
2. Back up `botname.conf` and `conf/users.conf`.
3. Use `supybot-adduser` against a throwaway config to generate a known-good owner entry or password hash.
4. Carefully update `conf/users.conf`.
5. Start the bot and verify with `identify` in PM plus `@whoami`.

Do not paste real owner passwords into channels, logs, issue trackers, or chat transcripts.
