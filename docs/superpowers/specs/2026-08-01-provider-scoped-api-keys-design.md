# Provider-scoped API keys — design sketch

**Status:** SKETCH. Not an approved spec. To be fleshed out in a fresh session.
**Date:** 2026-08-01

## Problem

API keys are configured per *role* and per *channel*, so the same credential is
duplicated across many registry entries and drifts out of sync. Today `bot.conf`
holds twelve `apiKey` entries covering four distinct key values:

| Entry | Key | Provider |
|---|---|---|
| `assistantApiKey` (global) | KEY-A | gemini |
| `searchApiKey` | KEY-A | gemini |
| `assistantApiKey.#afternet` / `.#CyberCafe` / `.#HexDroid` / `.#tv` | KEY-B | xai |
| `assistantApiKey.\:afternet.#afternet` / `.#cybercafe` / `.#hexdroid` | KEY-B | xai |
| `assistantApiKey.\:afternet.#tv` | KEY-C | xai |
| `codeApiKey` | KEY-D | xai |
| `imageApiKey` | KEY-B | xai |

The three xAI keys (B, C, D) are **unintentionally** different — a symptom of the
management problem, not a design choice. `#tv` resolves to KEY-C through its
network-scoped entry while its plain-channel entry says KEY-B.

The original reason keys were scoped per channel — a mix of free and paid Gemini
keys — **no longer applies**. There is one Gemini key in use, it is the paid one
(project `417371945641` / `gemini-api-paid-469818`), and the `vibebot-free` key
is not referenced anywhere in `bot.conf`. Grounded search is free on the paid key
(~336 requests/month against a 5,000/month allowance), so there is no remaining
reason to reintroduce a second Gemini key.

## Goal

A key is a property of **which provider you are paying**, not of which channel or
which role. One key per provider, configured once, outside `bot.conf`.

## Approach

Delete the API-key registry entries entirely and rely on LiteLLM's native
provider environment variables, set in the container env file:

- `GEMINI_API_KEY` — the paid Gemini key (current KEY-A)
- `XAI_API_KEY` — **KEY-B**, the key powering `#afternet`, confirmed canonical
- `OPENAI_API_KEY` — if/when the luna migration proceeds

LiteLLM selects the correct variable from the model's provider prefix, so nothing
in the plugin needs to choose between them. Changing a key becomes an env edit
plus a restart, which matches the existing deploy flow, and removes the most
sensitive values from the file Limnoria rewrites on shutdown.

Net effect: **twelve key entries → zero**, and per-channel key drift becomes
structurally impossible.

## Required changes

1. **Env file** (`/home/vibebot/.config/vibebot/env`) — populate real values.
   Current contents are placeholders (`GEMINI_API_KEY` 25 chars vs a real 39;
   `OPENAI_API_KEY` 20; `ANTHROPIC_API_KEY` 23) and **`XAI_API_KEY` does not
   exist at all**. Cutting over before populating these is a full outage.

2. **`service.py` — key resolution.** ~14 `registryValue(...ApiKey...)` call
   sites. At least four (`2638`, `3029`, `3788`, `4091`) guard with
   `if not api_key:` and raise a configuration error. LiteLLM only consults the
   environment when `api_key` is `None`, so these must pass `None` through rather
   than bail.

3. **`service.py` — log redaction. Security-critical.**
   `_configured_api_keys()` (line 1193) walks the registry — global plus every
   channel override — and `_sanitize()` (line 1222) replaces those literal values
   with `[REDACTED]` in anything logged or returned to IRC. Its docstring records
   that regex matching was deliberately rejected as less reliable. Once keys live
   in the environment this set returns **empty** and redaction silently becomes a
   no-op, so an upstream error echoing a key could reach a channel verbatim. It
   must read the provider env vars instead.

4. **`config.py`** — remove or deprecate `assistantApiKey`, `codeApiKey`,
   `imageApiKey`, `searchApiKey`. Decide whether to keep them as an optional
   override that wins over env, or drop them outright.

5. **`bot.conf`** — delete the twelve entries, with the bot stopped.

## Migration order

Non-negotiable, since step 2 removes the only working credential path:

1. Populate the env file with real keys and restart; the registry values still
   win, so this is a no-op change that proves the file loads.
2. Land the code changes (`None` passthrough + redaction from env).
3. Verify a live request per provider still authenticates.
4. Stop the bot, delete the registry entries, restart, verify again.

## Out of scope

- **Named model sets** (`fun` / `professional`). Deferred deliberately. Once keys
  are provider-scoped, making a channel "fun" is `assistantModel.#chan` plus
  `verseModel.#chan` for the one channel using verse — possibly not painful
  enough across six channels to justify another naming layer.
- If it is revisited: **`Profile` is already taken.** `profile.py` defines
  `chat` / `code` / `draw` / `verse` / `remind_action` — the *mode* axis, mapping
  mode to which registry key holds the model. A channel-personality axis is
  orthogonal and needs a different name (`modelSet`, `tier`) or two different
  things called "profile" collide in the same dispatch path.

## Open questions for the fresh session

1. Keep the registry keys as an optional override that beats env, or remove them?
   Override-wins preserves an escape hatch but reintroduces the drift this fixes.
2. Should `searchApiKey`'s fallback chain (`searchApiKey or assistantApiKey` in
   `_grounded_completion`) collapse entirely, given both now resolve to the same
   provider var?
3. Does anything outside the LLM plugin read these registry entries?
4. Test coverage: which existing tests stub `registryValue` for API keys and will
   need updating?
