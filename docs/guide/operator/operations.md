# Operations and auto-updates

This page covers day-to-day operation of a deployed bot: updates, restarts, logs, configuration changes, and housekeeping.

## How updates reach production

Two independent paths keep the running bot current. Both end in the same place: a restart that pulls the newest image.

**Path 1: CI push deploy.** A push to `main` runs CI. When CI passes, a second workflow builds the Docker image, pushes it to `ghcr.io/rdrake/vibebot-v8:latest`, and then connects to the production host over SSH with a restricted deploy key. The remote key runs one forced command: `systemctl --user restart vibebot`. The service unit pulls the latest image on start, so the restart deploys the new build.

**Path 2: updater timer.** `vibebot-updater.timer` fires every 15 minutes. Its service compares the local image digest against the registry and restarts `vibebot` only when the digest changed. This path catches anything the push deploy missed, such as a host that was offline during the push.

```
git push to main → CI → Docker build → GHCR
                              ↓                ↓
                   SSH restart (immediate)   updater timer (≤15 min)
                              ↓                ↓
                        systemctl --user restart vibebot
                              ↓
                     ExecStartPre pulls :latest
```

The source code never needs to live on the server. The Docker image has everything; the server holds only configuration and data.

## Files involved

| File | Purpose |
|------|---------|
| `vibebot.service` | Runs the Docker container; pulls the latest image on start |
| `vibebot-updater.service` | Compares image digests; restarts the bot when they differ |
| `vibebot-updater.timer` | Triggers the update check every 15 minutes |

Install all three with `make install-deploy`. See [Installation](installation.md) for the full setup walkthrough.

## Quick reference

| Task | Command |
|------|---------|
| Check status | `systemctl --user status vibebot` |
| View logs | `journalctl --user -u vibebot -f` |
| View recent logs | `journalctl --user -u vibebot -n 100` |
| Manual restart | `systemctl --user restart vibebot` |
| Force an update check | `systemctl --user start vibebot-updater` |
| Check timer status | `systemctl --user status vibebot-updater.timer` |
| Stop the bot | `systemctl --user stop vibebot` |
| Start the bot | `systemctl --user start vibebot` |

## Editing bot.conf safely

Limnoria flushes its in-memory registry to `bot.conf` on shutdown. If you edit the file while the bot is running, the shutdown flush overwrites your changes.

Always follow this order:

```bash
systemctl --user stop vibebot
# edit ~/.config/vibebot/bot.conf
systemctl --user start vibebot
```

For single settings, prefer the `@config` command from IRC instead; it changes the live registry and survives restarts. Reserve direct file edits for bulk changes and values that are awkward to type in IRC.

## Log locations

| Log | Command |
|-----|---------|
| Service logs | `journalctl --user -u vibebot` |
| Update logs | `journalctl --user -u vibebot-updater` |
| Bot message logs | `~/.local/share/vibebot/logs/messages.log` |

Container output is also available through `docker logs vibebot`.

## Housekeeping

The repository ships two maintenance scripts:

| Script | Purpose |
|--------|---------|
| `scripts/prune_backups.py` | Trim old configuration and data backups |
| `scripts/rotate_logs.py` | Rotate and compress bot log files |

Run them from cron or a systemd timer on the host if your deployment accumulates large logs or backups.

## Troubleshooting

### Service fails to start

Check that Docker runs, then read the service log:

```bash
docker info
journalctl --user -u vibebot -n 50 --no-pager
docker images | grep vibebot
```

### Timer is not running

```bash
systemctl --user list-timers
systemctl --user enable --now vibebot-updater.timer
```

### Updates are not applying

Force a check and read the updater log:

```bash
systemctl --user start vibebot-updater
journalctl --user -u vibebot-updater -n 20 --no-pager
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

### Service stops after logout

Enable lingering for your user:

```bash
loginctl enable-linger $USER
```

### Configuration changes are not applied

Restart the service after editing configuration, and remember the stop-edit-start order for `bot.conf`:

```bash
systemctl --user restart vibebot
```

## Uninstalling

Remove the service and timer:

```bash
make uninstall-timer
make uninstall-service
```

Configuration in `~/.config/vibebot/` and data in `~/.local/share/vibebot/` are preserved. Delete them manually if you no longer need them.
