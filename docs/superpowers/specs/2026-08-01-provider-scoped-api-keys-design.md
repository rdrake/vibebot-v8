# Provider-scoped API keys — design

**Status:** approved
**Date:** 2026-08-01

## Problem

API keys are configured per *role* and per *channel*, so one credential is
duplicated across many registry entries and drifts out of sync. Prod `bot.conf`
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
keys — **no longer applies**. There is one Gemini key in use, the paid one
(project `417371945641` / `gemini-api-paid-469818`), and the `vibebot-free` key
is not referenced anywhere in `bot.conf`. Grounded search is free on the paid key
(~336 requests/month against a 5,000/month allowance), so there is no remaining
reason to reintroduce a second Gemini key.

## Goal

A key is a property of **which provider you are paying**, not of which channel or
which role. One key per provider, set once in the container environment, and the
code that existed only to reconcile per-channel and per-role keys is deleted
along with the entries.

## Approach

Every model identifier in use already carries its provider as a prefix
(`xai/grok-4.3`, `gemini/gemini-3-flash-preview`, `openai/gpt-5.2`). That prefix
is the only input needed to choose a key, so key selection collapses to a pure
function of the model:

```python
_api_key_for("xai/grok-4.3")  ->  os.environ["XAI_API_KEY"]
```

Resolution is explicit in the plugin rather than left to LiteLLM's implicit env
lookup. Explicit resolution keeps the existing "API key not configured" guards
meaningful (a missing key is a clear configuration error at the top of the call,
not an `AuthenticationError` from deep inside a provider request), gives log
redaction a concrete value to scrub, and is unit-testable without a provider.

The four registry settings are deleted outright — not retained as an override.
Retaining them would preserve every mechanism this change exists to remove.

## Components

### New module: `plugins/llm/src/llm/apikeys.py`

`service.py` is already oversized, and key resolution has no dependency on the
service, the plugin, or Limnoria. It goes in its own module, testable as a pure
function.

```python
PROVIDER_ENV_VARS = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

def provider_of(model: str) -> str: ...
def api_key_for(model: str) -> str | None: ...
def known_api_key_values() -> set[str]: ...
```

- `provider_of` returns the lowercased prefix before `/`, or `""` when the model
  carries no prefix.
- `api_key_for` returns the non-empty environment value for that provider, else
  `None`. An unprefixed or unknown-provider model yields `None`, which the
  existing guards report as a configuration error.
- `known_api_key_values` feeds log redaction: every non-empty value of the four
  mapped variables, plus any other environment variable whose name ends in
  `_API_KEY`, filtered to values of **at least 8 characters**. The length floor
  matters — a junk one-character value would otherwise turn redaction into
  find-and-replace on a common letter across every log line.

Environment is read at call time, not cached at import. Reads are cheap, and a
cache would make the value that redaction scrubs diverge from the value actually
sent if the process environment were ever changed.

### `service.py`

1. Add a thin `_api_key_for(self, model)` delegating to the module, so call sites
   stay one line and tests can patch one seam.
2. Roughly fourteen `registryValue(...ApiKey...)` sites become
   `self._api_key_for(model)`. Several need the model fetched one line earlier
   than the key currently is; ordering is the only structural change.
3. The seven guards that report a missing key —
   `2638`, `2718`, `3029`, `3602`, `3788`, `4091`, `5117` — keep their current
   error strings and now test the resolved environment value.
4. `_configured_api_keys()` (`1193`) and its `_API_KEY_NAMES` tuple are replaced
   by `known_api_key_values()`. `_sanitize()` (`1222`) is otherwise unchanged.
   **This is the security-critical edit:** the current implementation redacts by
   walking registry values, so moving keys to the environment without this change
   would leave `_sanitize` a silent no-op and let a provider error echo a key
   into a channel.
5. The `searchApiKey or assistantApiKey` chain (`3182`) collapses to a single
   resolve on the search model. Both branches resolved to the same provider.

### Deletions

These exist only to reconcile per-role and per-channel keys:

| What | Where |
|---|---|
| Four `registerChannelValue` key blocks | `config.py:131-171` |
| `configure()` wizard key line, README key line | `config.py:120`, `README.md:34` |
| `api_key_setting` field, `"assistantApiKey"` on all five profiles | `profile.py:60,80,93,106,118,132,157` |
| `api_key_name = "codeApiKey" / "imageApiKey" / else` ladders | `service.py:2631-2636`, `3020-3027` |
| `api_key: str \| None = None` parameters and their `or` lines | `service.py:2969/3028`, `3500/3544`, `4242/4325` |
| Key stubs in the shared fixture | `tests/conftest.py:403,406,417,474` |

The three `api_key` parameters are dead in production: no caller in `src/llm`
passes one, only tests do. `model_override` alongside them **is** live (verse and
storybook use it) and stays.

`plugin.py:6348` resolves from the compaction model rather than
`assistantApiKey`. That also fixes a latent mismatch: verse compaction pairs
`registryValue("assistantApiKey")` with a hardcoded `gemini/` model, so any xAI
channel override would hand an xAI key to Gemini.

## Error handling

A missing provider variable produces the same user-visible message it does today
("API key not configured"), from the same seven guard sites, before any network
call. A model with no provider prefix resolves to `None` and takes that same
path. Nothing new can raise: `api_key_for` performs dictionary and environment
lookups only.

## Testing

New `tests/test_apikeys.py` covers the pure function: each mapped prefix, mixed
case, an unprefixed model, an unknown provider, an empty environment variable, a
variable set to whitespace, the 8-character redaction floor, and `*_API_KEY`
discovery.

`service.py` tests get a redaction case proving an environment-only key is
scrubbed from a simulated provider error, and the existing guard tests are
repointed from a registry stub to `monkeypatch.delenv`. Tests currently passing
an explicit `api_key=` are updated to set the environment instead.

## Migration

The order is not optional — the code change removes the only working credential
path from `bot.conf`.

1. Populate `/home/vibebot/.config/vibebot/env` with real values: `GEMINI_API_KEY`
   (KEY-A) and `XAI_API_KEY` (**KEY-B**, the key powering `#afternet`, confirmed
   canonical). Today's contents are placeholders — `GEMINI_API_KEY` is 25
   characters against a real 39, `OPENAI_API_KEY` 20, `ANTHROPIC_API_KEY` 23 —
   and `XAI_API_KEY` is absent entirely. Give the OpenAI and Anthropic variables
   real values or delete them; a wrong-length placeholder is worse than an absent
   variable. Restart. Registry values still win at this point, so this proves the
   env file loads and changes nothing.
2. Land the code. Auto-deploy on Docker green.
3. Verify one live request per provider: `@ask` on a grok channel, `@ask` on the
   global Gemini path, `@draw`.
4. Stop the bot, delete the twelve entries from `bot.conf`, start, verify again.
   Editing `bot.conf` while the bot runs is pointless — Limnoria rewrites the
   file from its in-memory registry on shutdown.
5. Revoke KEY-C and KEY-D at xAI once nothing references them.

Rollback differs by stage. Before step 2, revert the env file. After step 2 the
registry settings no longer exist, so a restored `bot.conf` entry would be
ignored: rollback is reverting the deploy, and after step 4 it is reverting the
deploy *and* restoring the entries, both with the bot stopped.

## Out of scope

- **Named model sets** (`fun` / `professional`). Once keys are provider-scoped,
  making a channel "fun" is `assistantModel.#chan` plus `verseModel.#chan` for
  the one channel using verse — likely not painful enough across six channels to
  justify another naming layer.
- If it is revisited: **`Profile` is already taken.** `profile.py` defines
  `chat` / `code` / `draw` / `verse` / `remind_action` — the *mode* axis, mapping
  mode to which registry key holds the model. A channel-personality axis is
  orthogonal and needs a different name (`modelSet`, `tier`), or two different
  things called "profile" collide in the same dispatch path.
- Provider variables beyond the four mapped. Adding one is a single dictionary
  entry.
