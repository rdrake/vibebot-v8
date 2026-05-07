# Plugins

Limnoria commands are organised into plugins. The **Owner** plugin's `load`, `unload`, and `reload` commands manage them. Loading a plugin requires being identified to a bot account with the `owner` capability.

## Listing

```
@list                             # all currently-loaded public plugins
@list <Plugin>                    # commands inside a plugin
@list --unloaded                  # plugins available on disk but not loaded
@apropos <substring>              # search command names
```

The default startup set always includes at least `Admin`, `Channel`, `Config`, `Misc`, `Owner`, and `User`. `supybot-wizard` typically adds many more.

## Loading and unloading

```
@load <Plugin>                    # case-insensitive; e.g. @load Games
@unload <Plugin>
@reload <Plugin>                  # re-import the plugin's code (picks up edits)
```

`load` searches every directory listed in `supybot.directories.plugins` (typically the system install plus a `plugins/` directory next to `botname.conf`). The first match wins, so a plugin sharing a name with a built-in will be loaded from whichever directory comes first.

Add a custom plugin directory:

```
@config supybot.directories.plugins /path/one:/path/two
```

(The value is a colon-separated list on POSIX, semicolon on Windows. Read the current value first and append.)

## Disambiguation

Multiple plugins can register the same command name. Two ways to deal with it:

```
@list                             # see which plugins have the command
@<Plugin> <command> [args]        # qualify with the plugin name
@defaultplugin <command> <Plugin> # set a default plugin for a command
```

`@help <command>` may also prompt you to qualify if it's ambiguous.

## Installing third-party plugins

Two approaches:

1. **PluginDownloader plugin** (built-in): pulls plugins from known repositories listed in `supybot.plugins.PluginDownloader.repositories`. Requires `supybot.commands.allowShell True`.

   ```
   @load PluginDownloader
   @repolist                            # known repos
   @plugins <repo>                      # plugins available in a repo
   @install <repo> <Plugin>
   @load <Plugin>
   ```

2. **Manual install**: drop the plugin's directory into one of the paths in `supybot.directories.plugins`, then `@load <Plugin>`. Restart not required.

For the official catalogue, see <https://limnoria.net/plugins.xhtml> and the [Built-in plugins reference](https://docs.limnoria.net/use/plugins/index.html).

## Third-party plugin trust

Plugins execute as the bot's operating-system user. Treat installing a plugin as running code on the bot host.

Before installing:

- Prefer known repositories and inspect the plugin source, dependencies, and import-time side effects.
- Check the load path with `@plugin path <Plugin>` after loading; name collisions can load a different copy than intended.
- Keep `supybot.directories.plugins` narrow and writable only by trusted OS users.
- Temporarily enable `supybot.commands.allowShell True` only when IRC-based plugin installation is needed, then set it back to `False`.
- Do not load `Debug`, `Unix`, or other shell/eval-capable plugins on production bots unless the risk is deliberate.

## Auto-loading on startup

Plugins listed in `supybot.plugins` (one boolean key per plugin name) are loaded at startup if `True`. Setting them via `@load` does this automatically. To disable a plugin without unloading it from this session:

```
@config supybot.plugins.Games False
```

## Common plugins to know

| Plugin | What it does |
|--------|--------------|
| `Admin` | bot administration: ignore lists, channel join/part, nick change |
| `Channel` | per-channel ops: op/voice/kick/ban/mode, channel capabilities |
| `Config` | the registry commands described in `configuration.md` |
| `Owner` | privileged: load/unload/reload, defaultcapability, quit, upkeep |
| `User` | bot user accounts: register, identify, hostmask, password |
| `NickAuth` | auto-login by services account |
| `Services` | bot's own NickServ/ChanServ identification |
| `AutoMode` | grant op/voice on join based on capabilities |
| `Aka` | user-defined aliases |
| `MessageParser` | regex-driven message triggers |
| `Network` | connect/disconnect to multiple networks |
| `Misc` | help, list, last, tell, version, ping |
