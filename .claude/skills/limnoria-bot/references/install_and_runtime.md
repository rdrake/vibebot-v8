# Install and runtime

## First-time install (POSIX)

Pick one:

- Distro package: `apt install limnoria` / `dnf install limnoria` / `pacman -S limnoria-git` / `emerge net-irc/limnoria`. Distro packages may lag upstream.
- `pipx install limnoria`. Recommended for getting current Limnoria on any distro that ships pipx.
- Manual: `python -m venv ~/limnoria && ~/limnoria/bin/pip install limnoria`.

Python ≥ 3.9 required. Optional extras: `chardet`, `feedparser`.

## Initial configuration

Run the wizard. It produces a working `botname.conf` and prompts for nick, server, owner account, plugins to enable, etc.

```
mkdir -p ~/bot && cd ~/bot
supybot-wizard
```

The wizard creates four directories alongside `botname.conf`:

- `conf/`  — sidecar config (users.conf, channels.conf, networks.conf, ignores.conf, userdata.conf)
- `data/`  — plugin databases
- `logs/`  — log files (`messages.log` is the IRC log; `<channel>.log` per channel)
- `plugins/` — drop-in for custom plugins

To create the owner account outside the wizard:

```
supybot-adduser botname.conf
```

## Running

Foreground:

```
supybot botname.conf
```

Useful flags:

- `--debug` — extra verbose logging
- `--profile` — run under cProfile
- `-n <nick>` — override `supybot.nick` for this run
- `--allow-root` — required if running as root (don't)

Daemonising:

- systemd system service is the recommended Linux setup when available.
- Distro packages typically ship `/etc/init.d/limnoria` or a system unit; `/etc/default/limnoria` selects which conf file to launch.

Minimal systemd service:

```
[Unit]
Description=Limnoria bot
After=network.target

[Service]
Type=simple
User=bot
WorkingDirectory=/home/bot/botname
ExecStart=/usr/bin/supybot /home/bot/botname/botname.conf
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
SyslogIdentifier=limnoria-botname

[Install]
WantedBy=multi-user.target
```

Use the actual `supybot` path from the install method. After editing the unit: `systemctl daemon-reload`, `systemctl enable --now botname.service`, inspect with `systemctl status botname.service` and `journalctl -fu botname.service`.

## Stopping and reloading

From IRC (owner only):

```
@quit [<reason>]                   # full shutdown
@disconnect [<reason>]             # leave current network only
@reconnect [<network>]
@upkeep                            # flush config + DB to disk now
```

POSIX signals:

- `SIGTERM` / `SIGINT` — clean shutdown.
- `SIGHUP` — equivalent to `@config reload`, if the `Config` plugin is loaded.

## Logs

```
supybot.log.level                   # DEBUG, INFO, WARNING, ERROR (default INFO)
supybot.log.stdout                  # also write to stdout (useful under systemd)
supybot.log.timestampFormat
supybot.log.plugins.individualLogFiles
supybot.log.format
supybot.directories.log             # default ./logs
```

Per-channel chat logs come from the `ChannelLogger` plugin:

```
@load ChannelLogger
@config supybot.plugins.ChannelLogger.enable True
@config channel supybot.plugins.ChannelLogger.enable True
```

Under systemd, prefer `supybot.log.stdout True` and read process logs with `journalctl -fu botname.service`; keep channel logs separate if operators need searchable chat history.

## File layout (typical)

```
botname.conf            # primary registry (auto-written)
conf/
  users.conf            # bot user accounts (passwords, capabilities, hostmasks)
  channels.conf         # per-channel settings
  networks.conf         # per-network settings
  ignores.conf          # ignore list
  userdata.conf         # arbitrary per-user state
data/
  <plugin>.db / .json   # plugin databases
logs/
  messages.log
  <network>/<#channel>.log
plugins/                # custom plugin drop-ins (added to supybot.directories.plugins)
```

## Upgrading

Same channel as install:

- `pipx upgrade limnoria`
- distro: package manager
- venv: `pip install -U limnoria`

Then `@quit` and restart the process. `@reload <Plugin>` is for re-importing a single plugin's code (e.g. after editing it under `plugins/`); it does **not** upgrade Limnoria itself.

## Connecting to additional networks

```
@connect <network> <server> [<port>] [<password>]
@disconnect [<reason>]
```

Subsequent config for the new network lives under `supybot.networks.<name>.*`.
