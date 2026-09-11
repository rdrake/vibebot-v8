# Plugin INFO log lines do not reach `docker logs`

**Status:** Noted, not started (2026-09-11)
**Author:** Richard Drake (with claude)
**Affects:** `LLM.log.info(...)` in `plugin.py` (for example the
`irc_lookup: kind=… target=…` line added in `51a2e4e`) and the
`LLM.bridge` logger's `bridge call:` / `bridge result:` lines in
`limnoria_bridge.py`.
**Priority:** Low. Nothing is wrong with the bot; a diagnostic that was
meant to be a one-line grep is not there when you grep for it.

## Summary

Prod has `supybot.log.level: DEBUG` and `supybot.log.stdout.level: INFO`,
and root-logger INFO lines do appear in `docker logs vibebot` (the
`secret redaction: 4 handler(s) filtered …` line at startup is one). But
INFO lines from plugin loggers never do. On 2026-09-11 a whole afternoon
of `irc_lookup` and bridge dispatches produced zero `irc_lookup:` or
`bridge call:` lines, while the WARNING-level `completion_timing` lines
from the same code paths were all present.

Everything the plugin logs at WARNING is visible; everything it logs at
INFO is not. So something between `supybot.log.getPluginLogger` and the
stdout handler is filtering plugin loggers to WARNING.

## Where to look

- `supybot.log.plugins.*` keys in `/home/vibebot/.config/vibebot/bot.conf`
  (`individualLogfiles` is `False`; check for a per-plugin level key or a
  `supybot.log.plugins.LLM.level`).
- `plugins/llm/src/llm/apikeys.py` `SecretFilter` is installed on every
  handler — confirm it filters by content, not by level.
- `plugins/llm/src/llm/tracing.py` `TraceFilter` — same question.
- Limnoria's `log.py`: `PluginLogger` / `getPluginLogger` and whether the
  plugin logger's effective level inherits from `supybot.log.level`.

## Until it is fixed

"Did the model call the tool?" is answered by the `completion_timing`
lines: `tool_calls=1` on `assistant_step_1` followed by an
`assistant_step_2` line about a second later (the WHOIS/LIST/NAMES round
trip). It does not say *which* tool, which is what the missing INFO line
was for.
