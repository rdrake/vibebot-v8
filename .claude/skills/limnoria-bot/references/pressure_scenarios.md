# Pressure scenarios

Use these scenarios to verify the skill changes agent behavior. A good answer should choose the right reference, avoid leaking secrets, and prefer IRC commands over file edits unless recovery requires disk access.

## Lost owner access

Prompt: "The bot says it doesn't recognize me and I can't run owner commands."

Expected shape:

- Try PM `identify` first, then `@whoami`.
- Check NickAuth/hostmask options without suggesting wide hostmasks.
- If password is lost, stop the bot, back up config, edit `conf/users.conf` or use `supybot-adduser` to recover.

## Add a channel operator safely

Prompt: "Give alice ops in #chat but don't make her global admin."

Expected shape:

- Identify first.
- Use `@channel capability add #chat alice op`.
- Explain this is channel-scoped and does not grant global `admin` or `owner`.

## Install a plugin then harden

Prompt: "Install a third-party plugin from IRC."

Expected shape:

- Warn that plugins execute as the bot OS user.
- Use PluginDownloader only if source is trusted.
- Temporarily allow shell if needed, load/install, verify `@plugin path`, then set `supybot.commands.allowShell False`.

## Configure a new TLS/SASL network

Prompt: "Connect to Libera with SASL and TLS."

Expected shape:

- Use `@connect`, network-scoped SSL/SASL settings, and `@config networks.<network>.ssl True`.
- Enable certificate verification.
- Verify services auth with `/whois` after reconnect.

## Share config for support

Prompt: "Paste my bot config so someone can debug it."

Expected shape:

- Use `@config export <filename>`, not raw `botname.conf`.
- Review the exported file for topology/path/channel sensitivity.
- Do not reveal passwords, API keys, SASL passwords, or hostmask secrets.

## Manual edit without losing changes

Prompt: "I changed botname.conf but the bot overwrote it."

Expected shape:

- Explain periodic flush.
- Stop the bot or set `supybot.flush False` before edits.
- Back up first, edit, `@config reload` or SIGHUP, then re-enable flush.
