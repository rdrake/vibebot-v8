# VibeBot operations guide

This guide covers deploying and operating VibeBot by using Docker and systemd.

## How the auto-update system works

```
git push to main → GitHub Actions builds image → GHCR
                                                   ↓
                        ← vibebot-updater.timer (15 min) ←
                        ↓
              pulls latest → compares → restarts if changed
```

The source code does not need to live on the server. The Docker image has
everything. You only need configuration files and data directories.

## Files involved

| File | Purpose |
|------|---------|
| `vibebot.service` | Runs the Docker container |
| `vibebot-updater.service` | Checks for updates, restarts if changed |
| `vibebot-updater.timer` | Triggers update check every 15 minutes |

## Installation

### Prerequisites

Install Docker and verify it runs:

```bash
docker --version
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

### Clone the repository

Clone the repository to get the service files:

```bash
cd ~/workspace/afternet
git clone https://github.com/rdrake/vibebot-v8.git
cd vibebot-v8
```

### Install service and timer

```bash
make install-deploy
```

This creates:

- `~/.config/systemd/user/vibebot.service`
- `~/.config/systemd/user/vibebot-updater.service`
- `~/.config/systemd/user/vibebot-updater.timer`
- `~/.config/vibebot/` (configuration directory)
- `~/.local/share/vibebot/{conf,data,logs}` (data directories)

### Configure

Copy your existing bot.conf to the conf directory (must be writable for backups):

```bash
cp /path/to/your/bot.conf ~/.local/share/vibebot/conf/bot.conf
```

Edit the environment file with your API keys:

```bash
nano ~/.config/vibebot/env
```

### Enable and start

```bash
systemctl --user enable vibebot
systemctl --user start vibebot

# Keep service running after logout
loginctl enable-linger $USER
```

### Verify

```bash
systemctl --user status vibebot
systemctl --user status vibebot-updater.timer
journalctl --user -u vibebot -f
```

## Quick reference

| Task | Command |
|------|---------|
| Check status | `systemctl --user status vibebot` |
| View logs | `journalctl --user -u vibebot -f` |
| View recent logs | `journalctl --user -u vibebot -n 100` |
| Manual restart | `systemctl --user restart vibebot` |
| Force update check | `systemctl --user start vibebot-updater` |
| Check timer status | `systemctl --user status vibebot-updater.timer` |
| Stop bot | `systemctl --user stop vibebot` |
| Start bot | `systemctl --user start vibebot` |

## Log locations

| Log | Command |
|-----|---------|
| Service logs | `journalctl --user -u vibebot` |
| Update logs | `journalctl --user -u vibebot-updater` |
| Bot message logs | `cat ~/.local/share/vibebot/logs/messages.log` |

## Troubleshooting

### Service fails to start

Check Docker is running:

```bash
docker info
```

Check the service logs:

```bash
journalctl --user -u vibebot -n 50 --no-pager
```

Verify the image exists locally:

```bash
docker images | grep vibebot
```

### Timer is not running

Check timer status:

```bash
systemctl --user list-timers
```

Re-enable the timer:

```bash
systemctl --user enable --now vibebot-updater.timer
```

### Updates are not applying

Force an update check:

```bash
systemctl --user start vibebot-updater
journalctl --user -u vibebot-updater -n 20 --no-pager
```

Verify the latest image is available:

```bash
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

### Service stops after logout

Enable lingering for your user:

```bash
loginctl enable-linger $USER
```

### Configuration changes not applied

Restart the service after editing configuration:

```bash
systemctl --user restart vibebot
```

## Uninstalling

Remove the service and timer:

```bash
make uninstall-timer
make uninstall-service
```

Configuration files in `~/.config/vibebot/` and data in `~/.local/share/vibebot/`
are preserved. Delete them manually if needed.
