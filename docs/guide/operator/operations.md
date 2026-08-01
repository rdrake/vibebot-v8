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

## Rolling back a deploy

There is no automatic rollback. A revert push through CI takes 10-20 minutes end to end, and `docker.yml` only builds after `CI` succeeds across a three-version Python matrix (3.12, 3.13, 3.14) — one flaky test blocks the image from ever publishing. When production needs to go back to a known-good build now, pin the previous image directly rather than waiting on a revert.

Every image build is tagged with the commit SHA (`type=sha` in `docker.yml`, alongside `:latest`), so a previous build stays pullable by digest-adjacent tag even after `:latest` moves past it.

1. **Find the previous image tag.** Either read it off the host before the bad deploy:

   ```bash
   docker inspect vibebot --format '{{.Config.Image}}'
   ```

   or from the GitHub Actions run summary for the last known-good `Build and Push Docker Image` workflow run.

2. **Stop the updater first.** `vibebot-updater.service` hardcodes `IMAGE=ghcr.io/rdrake/vibebot-v8:latest` (not the pin) and runs every 15 minutes; left running, it restarts `vibebot` back onto `:latest` the next time it fires, undoing the rollback.

   ```bash
   systemctl --user stop vibebot-updater.timer
   ```

3. **Pin the image with a systemd drop-in.** Overriding `Environment=IMAGE=...` for the `vibebot.service` unit points `ExecStartPre`'s `docker pull ${IMAGE}` and the `docker run` at the pinned tag instead of `:latest`.

   ```bash
   mkdir -p ~/.config/systemd/user/vibebot.service.d
   printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot-v8:sha-<PREV>\n' \
     > ~/.config/systemd/user/vibebot.service.d/override.conf
   systemctl --user daemon-reload && systemctl --user restart vibebot
   ```

   Replace `<PREV>` with the short SHA found in step 1.

4. **Roll forward again once the fix ships.** Remove the drop-in and restart the updater timer:

   ```bash
   rm ~/.config/systemd/user/vibebot.service.d/override.conf
   systemctl --user daemon-reload && systemctl --user restart vibebot
   systemctl --user start vibebot-updater.timer
   ```

## Credentials in the environment

API keys live in environment variables passed to the container via `--env-file` (see [Configuration → API keys](configuration.md#api-keys)). That is simpler than the old registry-based keys, but it changes where the credential is visible on the host — worth knowing plainly rather than discovering it during an incident:

- `docker inspect vibebot --format '{{.Config.Env}}'` prints every variable, values included, to anyone in the `docker` group — and that output gets pasted into tickets more often than people expect.
- `/proc/<pid>/environ` for the container's process exposes the same values to anything with host-level access to that PID.
- Every child process the container spawns inherits the full environment, key values included, whether or not that process needs them.

None of this is unique to this design — registry-stored keys were also readable by anyone with `@config` access at owner level — but the exposure surface moves from "Limnoria owner capability" to "docker group and host process access." Scope access to the host and the `docker` group accordingly.

## Uninstalling

Remove the service and timer:

```bash
make uninstall-timer
make uninstall-service
```

Configuration in `~/.config/vibebot/` and data in `~/.local/share/vibebot/` are preserved. Delete them manually if you no longer need them.
