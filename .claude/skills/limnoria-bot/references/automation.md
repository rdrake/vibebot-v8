# Aliases, regex triggers, and command parsing

Two ways to add bot behaviour without writing Python: the `Aka` plugin (named aliases) and the `MessageParser` plugin (regex triggers). Both build on Limnoria's command parser.

## Command parser features

### Quoting

Multi-word arguments must be double-quoted. Embedded `"`, `[`, `]` need escaping inside quotes:

```
@echo "hello world"
@echo "\"quoted\""
@len "waiter, there's a \" in my soup!"
@echo ""                          # empty argument
```

### Nested commands

Square brackets evaluate a command and substitute its output (with surrounding spaces):

```
@caps [filter reverse hello world]    # → DLROW OLLEH
@echo [coin] [coin]                   # → heads tails
@reply [coin] [coin]                  # prefixes with caller's nick
```

`echo` vs `reply`:

- `reply` prepends `<caller>: `; `echo` does not.
- `echo` interpolates `$nick`, `$version`, `$now`, etc.; `reply` does not.

To join two outputs without a space, use `Format.concat`:

```
@echo Random: [format concat [dice 1d100] %]
```

If a nick contains brackets and trips the parser:

```
@config supybot.commands.nested.brackets <>      # change the pair
@config supybot.commands.nested False            # or disable nesting
```

## Aka — named aliases

Load it once: `@load Aka`.

```
@aka add [--channel <#chan>] <name> "<command body>"
@aka set [--channel <#chan>] <name> "<new body>"             # overwrite
@aka remove [--channel <#chan>] <name>
@aka show [--channel <#chan>] <name>
@aka list [--channel <#chan>] [--keys] [--unlocked|--locked]
@aka lock [--channel <#chan>] <name>                         # only owner can edit/remove
@aka unlock [--channel <#chan>] <name>
```

Body placeholders:

- `$1`, `$2`, … — required positional arguments (the alias errors if omitted).
- `@1`, `@2`, … — optional positional arguments (empty string if omitted).
- `$*` — every remaining argument joined by spaces (includes the optional ones).

`$nick`, `$channel`, `$now`, `$version` etc. are **not** Aka placeholders. They are standard substitutions resolved by output commands like `echo`, `reply`, and the bot's quit message — so you get them by routing the alias body through one of those.

Examples:

```
@aka add rules "echo Channel rules: be excellent."
@aka add trout "reply action slaps $1 with a large trout"
```

**Quote nested commands inside the body** so they evaluate at call-time, not at `aka add` time:

```
# correct — randomises each call
@aka add randpercent "squish [dice 1d100]%"

# wrong — runs dice once, then bakes the result into the alias
@aka add randpercent  squish [dice 1d100]%
```

Two levels of quoting are common when the inner command itself takes a quoted arg — escape the inner quotes:

```
@aka add greetme "reply [sample 1 \"hi\" \"hey\"]"
```

Akas are global by default. Pass `--channel <#chan>` on `add`, `set`, `remove`, `show`, `list`, `lock`, or `unlock` to scope to a single channel. Channel-scoped akas shadow same-named globals when invoked in that channel.

For optional-argument logic (do something different when an arg is missing), combine `@1` with the `Conditional` plugin:

```
@load Conditional
@aka add trout "reply action slaps [cif [ceq \"@1\" \"\"] \"echo $nick\" \"echo @1\"] with a large trout"
```

### Replacing built-in commands

You can shadow a built-in command with an Aka:

```
@aka add ping "echo Pong! [echo $now]"
@aka lock ping
```

Remove the alias to restore the original.

### Migrating from old Alias plugin

```
@load Aka
@load Alias
@aka importaliasdatabase
@unload Alias
```

## MessageParser — regex triggers

Run a command whenever a channel message matches a regex. Load: `@load MessageParser`.

```
@messageparser add "<regex>" "<command>"
@messageparser list
@messageparser show --id <n>
@messageparser remove --id <n>
@messageparser remove "<regex>"
@messageparser lock <regex>           # protect from edit
```

Case-insensitive: prefix the regex with `(?i)`:

```
@messageparser add "(?i)^test$" "echo Test failed, try again later"
```

Capture groups become `$1`, `$2`, …:

```
@messageparser add "define (.+)" "dict $1"
```

### Use case: respond to relay-bot messages

If a relay bot delivers messages as `<RelayBot> <originalnick> @cmd ...` and the bot's normal prefix wouldn't match:

```
@messageparser add "^<RelayBot> <[^>]+> @(.+)$" "$1"
```

Tighten the regex to the relay format you actually receive. Relayed commands run as the relay bot's Limnoria identity, so fine-grained access control for the original sender is not preserved; do not use this pattern for privileged command access.

### Limitations

- Triggers run as the user who posted the message, with their capabilities.
- They don't fire on the bot's own messages.
- Recursion is bounded; deeply nested chains may be cut off.
- Per-channel scoping is on by default (each channel has its own trigger set).

## When to choose which

| Need | Tool |
|------|------|
| Run a fixed command behind a short name | Aka |
| Take parameters in command syntax (`@trout someone`) | Aka |
| React to free-form chat without a command prefix | MessageParser |
| Extract values out of a sentence | MessageParser |
| Anything requiring real Python logic | write a plugin |
