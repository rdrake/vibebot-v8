# Picard Command Design

**Date:** 2026-02-24

## Summary

Add `%picard [topic]` command that shares random Captain Picard facts, optionally steered by a topic. Uses the ask infrastructure (model, API key, rate limits, conversation context) with a Picard-themed system prompt.

## Behavior

- `%picard` — random Picard fact, drawing inspiration from conversation context
- `%picard tea` — Picard fact related to tea
- Shares conversation context with `%ask` so Picard can reference prior exchanges
- Usage logged as "picard" command for tracking

## Approach: System prompt override on `completion()`

Add an optional `system_prompt` parameter to `LLMService.completion()`. When provided, skip the `{command}SystemPrompt` registry lookup. The picard command calls `completion(prompt, command="ask", system_prompt=picard_prompt)`, reusing ask's model, API key, and rate limits.

## Changes

| File | Change |
|------|--------|
| `config.py` | Add `picardSystemPrompt` channel value with default Picard personality |
| `service.py` | Add optional `system_prompt` param to `completion()` — use instead of registry lookup when set |
| `plugin.py` | Add `picard` command (~40 lines) — ask preflight, picard system prompt, completion with override, store context, reply |
| tests | Test picard command and system_prompt override |

## System prompt default

"You are Captain Jean-Luc Picard of the USS Enterprise. Share an interesting, surprising, or amusing fact — it can be about you, Starfleet, the Enterprise crew, or the Star Trek universe. Draw inspiration from the ongoing conversation when relevant. Stay in character. Be concise (1-3 sentences for IRC). If given a topic, relate your fact to it."

## What picard shares with ask

- Model and API key (via `command="ask"`)
- Rate limits (preflight uses `"ask"`)
- Conversation context (reads/writes same history)

## What picard has independently

- System prompt (picardSystemPrompt in config, customizable per channel)
- Usage tracking (logged as "picard" for visibility)
