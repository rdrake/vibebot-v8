# Command UX Overhaul Design

## Problem

The bot's command surface has inconsistencies that confuse users:

1. **Reminders use three separate commands** (`%remindme`, `%reminders`, `%unremind`) while `%memories` groups subcommands under one roof.
2. **`%usage` bypasses Limnoria's `wrap()` system** with custom IRC parsing, making it an implementation outlier.
3. **`%picard` is a novelty command occupying top-level namespace** — it's really `%ask` with a custom system prompt, but has no `picardModel` config and shares the `llm.ask` capability.
4. **"Memory" means two things** — `%forget` clears conversation context while `%memories` manages stored facts. No terminology distinguishes them.
5. **Help surfaces are out of sync** — `getPluginHelp()` lists four commands, but the bot has ten. The HTML help page is also stale.

## Decisions

### Consolidate reminders under `%remind`

Replace three commands with one using the `%memories` pattern:

| Before | After |
|--------|-------|
| `%remindme <text>` | `%remind <text>` |
| `%reminders` | `%remind list` |
| `%unremind <id>` | `%remind delete <id>` or `%remind del <id>` |
| (none) | `%remind clear` |

Parsing: `wrap(remind, [optional("text")])` with internal `text.split()` dispatch, identical to how `%memories` works.

### Fix `%usage` internals

Replace custom `_extract_raw_arg()` parsing with `wrap(usage, [optional("text")])` + internal split, matching `%memories` and `%remind`. The bracket-nick edge case (`Rubin[F]`) works because `optional("text")` captures the full remaining string without tokenizer interference.

User-facing behavior stays the same:
- `%usage` — personal stats (channel) or global overview (PM, admin)
- `%usage <nick>` — that user's stats
- `%usage #channel` — that channel's stats

### Replace `%picard` with `%instruct`

Add a user-settable instruction system:

- `%instruct <text>` — save persistent instruction to DB
- `%instruct` (no args) — show current instruction
- `%instruct clear` — remove instruction

Scope: per-user, global (same instruction everywhere). Stored in the `memories` SQLite database in a new `user_instructions` table.

Behavior: when a user has an active instruction, it is prepended to the system prompt for `%ask` calls. The channel's `askSystemPrompt` config still applies — the instruction augments it.

`%picard` is removed. The curated Picard system prompt is documented in help as an example of `%instruct` usage. `picardSystemPrompt` config is removed.

### Volatile / non-volatile terminology

Commands keep their existing names (`%forget`, `%memories`). All help text, docstrings, and documentation updated to use consistent framing:

- **Volatile memory** = conversation context (cleared by `%forget`, expires after timeout)
- **Non-volatile memory** = stored facts (managed by `%memories`, persists indefinitely)

### Generate help from source

Both `getPluginHelp()` and the HTML help page derive their content from the actual registered commands and their docstrings. No more hardcoded command lists. Drift becomes impossible.

Implementation: a module-level or class-level registry of user-facing commands with their category (generation, memory, utility). `getPluginHelp()` iterates this to build the summary string. The HTML template uses it to build the full documentation page.

### RPG stays in main

No changes. Already merged, gated behind `enabled: False` per channel.

## Final command surface

### Generation
| Command | Description |
|---------|-------------|
| `%ask <question>` | Ask with context, vision, and optional instructions |
| `%code <request>` | Generate code with HTTP link output |
| `%draw <prompt>` | Generate image (account required) |

### Memory
| Command | Description |
|---------|-------------|
| `%forget [channel]` | Clear volatile memory (conversation context) |
| `%memories [subcommand]` | Manage non-volatile memory (stored facts) |
| `%instruct [text]` | Set persistent instructions for ask |

### Utility
| Command | Description |
|---------|-------------|
| `%remind <text>` | Set, list, delete, or clear reminders |
| `%usage [nick\|#channel]` | View API usage statistics |
