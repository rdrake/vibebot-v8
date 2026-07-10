# Installation and deployment

## Prerequisites

You need one of:

- **Docker** (recommended): any recent version with `docker pull` support.
- **Python 3.12 or newer** with [uv](https://docs.astral.sh/uv/): for manual and development installs.

VibeBot is a [Limnoria](https://docs.limnoria.net/) plugin. If you have never used Limnoria before, follow the [Limnoria getting started guide](https://docs.limnoria.net/use/getting_started.html) to create your initial `bot.conf`. You need a working `bot.conf` before proceeding.

## Docker deployment (recommended)

Pull the latest image:

```bash
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

Run the container, mounting your configuration and data directories:

```bash
docker run --rm --name vibebot \
    --user $(id -u):$(id -g) \
    -v /path/to/config:/config \
    -v /var/www/llm:/var/www/llm \
    -w /config \
    ghcr.io/rdrake/vibebot-v8:latest
```

The `/config` volume holds your `bot.conf` and its `conf/`, `data/`, and `logs/` directories. The `/var/www/llm` mount is optional: add it only when an external web server serves the `@code` and `@draw` output files.

## systemd service

The repository includes a systemd user service for production deployments. Install it with:

```bash
make install-service
```

This copies the unit file and creates the standard directory layout:

| Path | Purpose |
|------|---------|
| `~/.config/systemd/user/vibebot.service` | systemd unit |
| `~/.config/vibebot/` | `bot.conf` and `env` file |
| `~/.local/share/vibebot/{conf,data,logs}` | Persistent bot data |

The unit runs the Docker image with `~/.config/vibebot` mounted at `/config`, reads API keys from the `env` file, and pulls the latest image on every start.

After installing, complete the setup:

1. Copy your `bot.conf` to `~/.config/vibebot/bot.conf`.
2. Add your API keys to `~/.config/vibebot/env`.
3. Enable and start the service:

```bash
systemctl --user enable vibebot
systemctl --user start vibebot
loginctl enable-linger $USER   # keeps the service running after logout
```

### Auto-updates

Install the update timer to pull new Docker images automatically (checks every 15 minutes):

```bash
make install-timer
```

Or install both the service and the timer at once:

```bash
make install-deploy
```

See [Operations](operations.md) for how the two update paths work together.

## Manual and development setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rdrake/vibebot-v8.git
cd vibebot-v8
make install
```

Start the bot:

```bash
make run
```

This runs `uv run limnoria bot.conf` from the repository root. You need a `bot.conf` in the working directory.

For development, also install the pre-commit hooks:

```bash
make install-hooks
```

Useful development targets:

| Target | What it does |
|--------|--------------|
| `make test` | Run the fast test suite with coverage (93% floor) |
| `make test-all` | Run every test, including slow ones |
| `make lint` | Run `ruff check` |
| `make typecheck` | Run `ty check` on both plugins |
| `make check` | Lint, format check, typecheck, syntax check, and tests |
| `make preflight` | Format, then run the full `check` gate |
| `make docs-serve` | Preview this documentation site locally |

## Verifying the installation

For systemd deployments, check the service status:

```bash
systemctl --user status vibebot
```

For Docker, check the running container:

```bash
docker ps | grep vibebot
```

Once the bot connects to IRC, verify the plugin loaded:

```
@list LLM
```

This lists the available commands (`ask`, `code`, `draw`, and so on). If the plugin is not loaded, load it from IRC:

```
@load LLM
```
