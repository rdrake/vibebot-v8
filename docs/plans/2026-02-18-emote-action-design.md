# Emote/Action Response Design

**Date:** 2026-02-18
**Status:** Approved

## Summary

Allow the bot to respond with IRC actions (`/me`) in addition to regular replies. The LLM decides when an emote feels more natural than a direct answer.

## Changes

### 1. Sanitization: Remove `/` from default command prefixes

**File:** `config.py`

Change `commandPrefixes` default from `[".", "/"]` to `["."]`.

`/` in a PRIVMSG is literal text with no protocol-level risk. The `.` prefix is the real concern (Limnoria command prefix). Operators who want the old behavior can add `/` back via config.

### 2. Action detection and sending

**File:** `plugin.py` (ask command only)

After getting the completion result, check if the response starts with `/me `:

- If yes: strip the `/me ` prefix, send via `ircmsgs.action(target, action_text)`
- If no: existing `irc.reply()` path (unchanged)

Details:
- Only the `ask` command gets action support. `code`, `draw`, `animate` don't make sense as actions.
- Grounding icon is prepended to the action text if grounding was used.
- `invalidCommand` delegates to `ask`, so addressed messages get action support for free.
- Context stores actions as `* BotNick action_text` so follow-ups understand the bot emoted.

### 3. System prompt nudge

**File:** `service.py` (`_build_system_prompt`)

Append one sentence to the code-level system prompt builder:

> "You may occasionally respond with /me for actions when it feels natural (e.g., /me shrugs)."

This applies to all commands (shared builder) but only `ask` has detection logic. If `code` model emits `/me`, it appears as literal text (harmless).

### 4. Testing

- Response starting with `/me ` triggers `ircmsgs.action()` instead of `irc.reply()`
- Normal response still uses `irc.reply()`
- Context stores `* BotNick action_text` for actions
- `/me` passes through `sanitize_output` with new default prefixes
