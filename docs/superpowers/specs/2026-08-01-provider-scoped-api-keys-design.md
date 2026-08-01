# Provider-scoped API keys — design

**Status:** approved, red-teamed
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

Because keys and models are configured independently, they have drifted apart.
Three live cross-provider mismatches exist today, all verified in code:

1. `service.py:2631-2637` — the pending-task retry path stores `task_type="code"`
   with `model=assistantModel` (the code profile's `model_setting` is
   `assistantModel`, `profile.py:105`), then pairs it with **`codeApiKey`**. In
   prod that is KEY-D (xai) sent with whatever `assistantModel` resolves to.
2. `service.py:3179-3184` — grounded search resolves its model as
   `searchModel or assistantModel` and its key as `searchApiKey or
   assistantApiKey`, independently. `searchModel` defaults to `""` and
   `searchApiKey` has no channel override, so on grok channels a Gemini key
   (KEY-A) is sent with an xAI model.
3. `plugin.py:6347-6348` — verse compaction pairs `registryValue("assistantApiKey")`
   with a hardcoded `gemini/` model.

## Goal

A key is a property of **which provider you are paying**, not of which channel or
which role. One key per provider, set once in the container environment, resolved
from the model actually being called — which makes the three mismatches above
structurally impossible. The code that existed only to reconcile per-channel and
per-role keys is deleted along with the entries.

## Approach

Key selection becomes a pure function of the model:

```python
api_key_for("xai/grok-4.3")  ->  os.environ["XAI_API_KEY"]
```

The provider comes from `litellm.get_llm_provider(model)[1]`, **not** from
splitting on `/`. Unprefixed model names are legal — `config.py:40` validates
models through litellm, which resolves `gpt-4` and `dall-e-3` to openai (verified;
note it *raises* on bare `claude-3-opus`, so the wrapper must catch) — and the
test suite uses unprefixed names in roughly 100 places. A
`split("/")` implementation would return `None` for all of them and report
"API key not configured" for a configuration that is actually valid.

Resolution is explicit in the plugin rather than left to LiteLLM's implicit env
lookup. Explicit resolution keeps the missing-key guards meaningful (a clear
configuration error at the top of the call, not an `AuthenticationError` from
inside a provider request), gives log redaction a concrete value to scrub, and is
unit-testable without a provider.

The four registry settings are deleted outright, not retained as an override.
Retaining them would preserve every mechanism this change exists to remove.

## Components

### New module: `plugins/llm/src/llm/apikeys.py`

`service.py` is already oversized, and key resolution depends on nothing in the
service, the plugin, or Limnoria. It goes in its own module, testable as a pure
function.

```python
PROVIDER_ENV_VARS = {
    "xai": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
SECRET_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_CREDENTIALS")
MIN_REDACTABLE_LEN = 16

def provider_of(model: str) -> str: ...
def env_var_for(model: str) -> str | None: ...
def api_key_for(model: str) -> str | None: ...
def known_secret_values() -> set[str]: ...
```

- `provider_of` wraps `litellm.get_llm_provider(model)[1]`, returning `""` when
  litellm cannot classify the model. It must not raise: litellm raises
  `BadRequestError` on unrecognized names, and key resolution is called on paths
  that have their own error handling.
- `env_var_for` maps the provider to its variable name, or `None` if unmapped. It
  exists so the guards can say *which* variable they wanted.
- `api_key_for` returns the non-empty environment value, else `None`.
- `known_secret_values` feeds redaction: every non-empty environment value whose
  variable name ends in one of `SECRET_SUFFIXES`, filtered to values of at least
  `MIN_REDACTABLE_LEN` characters. The suffix set is wider than `_API_KEY` alone
  because adjacent credentials (`GOOGLE_APPLICATION_CREDENTIALS`,
  `VERTEX_CREDENTIALS`, `HF_TOKEN`, `ANTHROPIC_AUTH_TOKEN`) would otherwise sit
  outside redaction. The length floor stops a short junk value turning redaction
  into find-and-replace on a common word; 16 is comfortably below a real key
  (Gemini 39, xAI ~84) and above the placeholders currently in the env file.

Environment is read at call time, not cached at import, so the value redaction
scrubs cannot diverge from the value actually sent.

At startup the plugin logs the **count and variable names** in the redaction set,
never values, so an operator can confirm redaction is live.

### Unmapped providers

An unmapped provider resolves to `None` and the guards report it by name:
`"Error: no API key configured for provider 'vertex_ai'"`. Naming the provider
and, where known, the expected variable prevents the misdiagnosis where a
correctly-spelled model looks like a missing key.

`vertex_ai` is deliberately unsupported (see Out of scope), and the `imageModel`
default moves off it — see Deletions.

### Redaction: `SecretFilter`

`_sanitize` (`service.py:1222`) is applied by hand at ~25 call sites and does not
cover the path that leaks most. Supybot's `Logger.exception`
(`supybot/log.py:76-88`) writes the raw traceback *and* calls
`utils.python.collect_extra_debug_data()`, which `repr()`s every local in every
frame plus every attribute of `self`. There are ~12 `log.exception` sites in
`service.py` and ~30 in `plugin.py`; `verse/compaction.py` and `executor.py` have
no `_sanitize` coverage at all, and `compaction.py:276` stores the key on `self`
where the attribute walk finds it.

Add a `logging.Filter` installed beside the existing `TraceFilter`
(`service.py:1164`, `plugin.py:730`) that scrubs `record.getMessage()` and
`record.exc_text`, sourcing values from `known_secret_values()`. This closes the
exception path, covers the modules `_sanitize` never reached, and removes the
need to preserve the "never bind the key to a local" pattern that
`service.py:2718/3600/5115` maintain by hand — a pattern the refactor would
otherwise destroy.

`_sanitize` is **kept**, sourcing from `known_secret_values()` instead of the
registry, because two paths are not logging:

- `service.py:5171-5239` feeds `self._sanitize(str(e))[:200]` back into
  `_rewrite_prompt` as **prompt text**. litellm places Gemini keys in the request
  URL query string, so an unsanitized provider error here could be laundered
  through the model into channel output.
- `AssistantResult.error` (sanitized at `5071/5078`) is persisted by
  `db.log_usage(error_detail=...)`.

### `service.py`

1. Add `_api_key_for(self, model)` delegating to the module, so call sites stay
   one line and tests have one seam to patch.
2. Fourteen `registryValue(...ApiKey...)` sites in `service.py` plus one in
   `plugin.py:6348` become model-anchored resolves: `2637`, `2718`, `3182`,
   `3184`, `3602`, `3672`, `3787`, `4090`, `4175`, `4325`, `5117`, `6124`,
   `6195`, and the two inside `_configured_api_keys` (`1203`, `1212`). Several
   need the model fetched one line earlier than the key currently is.
3. **Eight** guards report a missing key and keep doing so, with the message
   extended to name the provider: `2638`, `2718`, `3029`, `3602`, `3788`,
   `4091`, `4328`, `5117`. The spec's earlier draft listed seven and omitted
   `4328` — the guard inside `assistant_completion`, covering chat, verse, code,
   draw and `@rp`, i.e. most traffic.
4. Four sites pass a key to litellm with no guard: `3672`, `4175`, `6124`,
   `6195`. Today a missing key sends `""`; after the change they send `None`,
   which is what triggers litellm's implicit env lookup. That lookup would find
   the same value `api_key_for` just failed to find, so behaviour is unchanged —
   but the equivalence is stated here deliberately rather than left implied.
   `3672` and `4175` are already covered by guards on their callers (`3602`,
   `2718`/`5117`); `6124`/`6195` are best-effort memory paths that stay
   unguarded.
5. `_configured_api_keys()` (`1193`) and `_API_KEY_NAMES` (`1186`) are replaced
   by `known_secret_values()`.
6. The `searchApiKey or assistantApiKey` chain (`3182-3184`) collapses to a
   single resolve on the search model. The justification is **not** that both
   branches resolved to the same provider — they demonstrably did not, which is
   mismatch 2 above. It is that the key is now anchored to the model being
   called.
7. `plugin.py:6348` resolves from the compaction model. Better still, resolve
   inside `LiteLLMVerseClient.call()` from its own `model` argument
   (`verse/compaction.py:147`), since binding the key at construction
   (`compaction.py:273-286`) is model-independent by design and would re-diverge
   if compaction ever called a second model. That also removes the one surviving
   key-injection parameter.

### Deletions

These exist only to reconcile per-role and per-channel keys:

| What | Where |
|---|---|
| Four `registerChannelValue` key blocks | `config.py:131-171` |
| `configure()` wizard key line | `config.py:120` |
| `api_key_setting` field, `"assistantApiKey"` on all five profiles | `profile.py:60,80,93,106,118,132,157` |
| `api_key_name` ladder (whole block) | `service.py:2631-2636` |
| `api_key_name` lines **only** (`3021`, `3025`) | `service.py:3020-3026` |
| `api_key: str \| None = None` parameters and their `or` lines | `service.py:2969/3028`, `3500/3544`, `4242/4325` |
| `channel` parameter of `_generate_image_once` | `service.py:4155-4175` |

`service.py:3020-3026` must **not** be deleted as a block: the same `if/else`
selects `codeModel`/`codeSystemPrompt` versus `assistantModel`/
`assistantSystemPrompt`. Only the two `api_key_name` lines come out.

The three `api_key` parameters are dead in `src/` **and** in tests: no caller in
`plugins/llm` passes one, the `**` spreads at those call sites expand
`_pending_task_fns` (callables only), and there is no `functools.partial` or
dynamic dispatch reaching them. `model_override` alongside them **is** live
(verse, storybook) and stays. `_generate_image_once`'s `channel` parameter exists
solely for the per-channel `imageApiKey` lookup (`service.py:4158` docstring) and
becomes dead; it also currently passes a raw `msg.args[0]`, which in a PM is a
nick, as a registry scope — unlike every other site, which normalises through
`_channel_target`.

Configuration and documentation that becomes wrong:

| What | Where |
|---|---|
| `imageModel` default `vertex_ai/imagen-4.0-generate-001` → `gemini/imagen-4.0-fast-generate-001` | `config.py:228-233` (assert at `tests/test_config.py:203`) |
| Env-var guidance that contradicts this design, incl. `VERTEX_*` | `.env.example` (copied to the live env path by `Makefile:244-258`) |
| API-key table and `@config` example | `docs/guide/operator/configuration.md:25-28,33` |
| Key references, incl. the troubleshooting step "`@config plugins.LLM.assistantApiKey` shows a masked value" | `docs/guide/operator/tuning-monitoring.md:22,25,166` |
| Key reference | `docs/guide/operator/memory-promotion.md:88` |
| Key configuration example | `README.md:34` |
| Rollback procedure (absent today) | `docs/guide/operator/operations.md` |

`tuning-monitoring.md:166` matters disproportionately: it is the diagnostic an
operator would reach for *during* this migration, and it stops working.

Locale files contain no `ApiKey` msgids, but deleting `_()` docstrings will churn
`messages.pot` on next extraction.

## Error handling

A missing or unmapped provider variable produces a configuration error from one
of the eight guards, before any network call, naming the provider. Nothing new
can raise: `api_key_for` performs a litellm lookup (wrapped) plus dictionary and
environment reads.

Pending-task rows persist `task.model`; a row with an empty model now fails
terminally with a configuration error instead of attempting the call. Bounded and
self-clearing.

## Testing

The suite is 2549 tests, currently passing, with `fail_under = 93`
(`pyproject.toml:75`) against a current 94% — about one point of headroom.
`apikeys.py` and the filter must land near 100%.

**Fixture work comes first, because it gates everything else:**

1. `conftest.py:28` `TEST_MODEL = "gpt-4"` and `:405` `imageModel: "dall-e-3"`
   resolve through litellm to openai — correct, but only if the environment has a
   value. Add an **autouse** fixture that deletes every variable matching
   `SECRET_SUFFIXES` and sets the four provider variables to fake values.
2. That fixture is load-bearing for isolation, not just convenience:
   `litellm/__init__.py:27` calls `load_dotenv()` at import, and `.env` is
   gitignored, so importing `llm.service` injects a developer's **real** keys
   into `os.environ` before any test runs. Without the scrubber, tests pass
   locally for the wrong reason and a redaction test cannot assert an exact set.
3. Remove the four key stubs at `conftest.py:403,406,417,474`.

**Named migrations** (113 references, 13 files, ~53 test functions):

| Test | Work |
|---|---|
| `test_stress.py:264` `test_completion_api_key_isolation` | Redesign: 20 concurrent requests across 2-3 provider prefixes, asserting each call received its own provider's key. Preserves the cross-request bleed property in terms the new model supports. |
| `test_service_core.py:218-300` | Five `_sanitize` tests, one of which calls `conf.…get("assistantApiKey").get("#forest").setValue(...)` → `NonExistentRegistryEntry`. Repoint at env values. |
| `test_assistant.py:4378` | Asserts `"assistantApiKey" not in registry_calls`. Dies outright. |
| `test_assistant.py:4399,4451,4515,4596,4658` | Five `Profile(...)` constructions → `TypeError` on `api_key_setting`. Four are overlay/model tests, pure collateral. |
| `test_profile.py:62-65,127,182-183` | Two parametrized methods × 5 profiles = 10 cases, plus the `EXPECTED_API_KEY` table. |
| `test_config.py:137,159,196-209` | Wizard string, `codeApiKey._private`, defaults (incl. the `imageModel` assert), `_private` flags. |
| `test_plugin_verse.py:2806,2855` | Bare-LLM stubs branching on `key == "assistantApiKey"`. |
| `test_reminders.py:721`, `test_service_memory.py:105,725,757,806`, `test_service_completion.py:117` | Assert the exact `api_key` kwarg forwarded to litellm. |

The 10 existing `api_key=` call sites in tests all target
`_xai_responses_call` (`3286`) or `_completion_with_tool_fallback` (`1870`),
where the parameter is required and stays. No test passes `api_key=` to
`completion`/`assistant_request`/`assistant_completion`.

**New** `tests/test_apikeys.py`: each mapped provider, unprefixed names, an
unknown provider, a model litellm rejects, empty and whitespace values, the
length floor, each secret suffix, and the redaction filter scrubbing both a
message and an `exc_text` traceback.

## Migration

Verified against Limnoria's registry implementation:

- Unknown keys in `bot.conf` are read into a flat cache and never validated
  (`supybot/registry.py:78-113`), so a `bot.conf` holding entries for
  now-unregistered settings loads without error. The repo's own local `bot.conf`
  proves it — it still carries `askApiKey` and `drawApiKey`, renamed away long
  ago. **The deploy and the `bot.conf` cleanup can therefore be separate steps.**
- `registry.close` writes only the *registered* tree (`registry.py:130-177`), and
  `world.upkeep` flushes on a timer — `supybot.flush` defaults `True`
  (`conf.py:795`), `supybot.upkeepInterval` defaults 3600 (`conf.py:789`). So
  **within about an hour of the deploy, `bot.conf` rewrites itself without the
  twelve entries, with no operator action.** The cleanup step is largely
  automatic, and the rollback material disappears on a timer.

**Pre-flight, on the host, read-only:** confirm `supybot.flush` and
`supybot.upkeepInterval` are not overridden in prod `bot.conf` (if `flush` is
false, the flush finding inverts and the manual cleanup becomes mandatory);
record the current `searchModel` and `imageModel` values; record the deployed
`sha-` image tag from the running container.

1. **Back up before anything.** Copy `bot.conf` off the host, and record KEY-A
   through KEY-D in a password manager. Limnoria's rolling
   `bot.conf.backup.<timestamp>` files roll forward and will all be
   post-deletion within a couple of flush cycles.
2. **Populate the env file.** `EnvironmentFile=-` (`vibebot.service:12`) and
   `--env-file` (`:19`) read the same file with **different parsers**, and only
   `--env-file` reaches Python. `docker run --env-file` is the strict one: no
   quotes (they become part of the value), no trailing comments, no spaces around
   `=` (a hard error), LF endings. A malformed file makes `docker run` exit, and
   `Restart=always` + no `HEALTHCHECK` turns that into a silent crashloop — this
   step, not the code change, is the highest outage risk in the plan. Also avoid
   a stray `IMAGE=` line: `Environment=IMAGE=` precedes `EnvironmentFile=` in the
   unit, so the file would override the deployed image.
   Set `GEMINI_API_KEY` (KEY-A) and `XAI_API_KEY` (**KEY-B**, confirmed
   canonical). Give `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` real values or delete
   them; a wrong-length placeholder is worse than an absent variable.
   Restart, then **verify from inside the container**, not by reading the file:
   `docker exec vibebot python3 -c "import os;k=os.environ.get('XAI_API_KEY','');print(len(k),k[:4],k[-4:])"`.
   Length plus prefix plus suffix catches every mangling mode above.
3. **Land the code**, docs in the same PR. Auto-deploy on Docker green.
4. **Verify within the first hour**: `@ask` on a grok channel, `@ask` on the
   global Gemini path, `@draw`, and one grounded-search request. After the first
   upkeep flush the entries are gone and only the image-pin rollback remains.
5. **Sweep `bot.conf` by prefix, not by the known twelve.** Delete every
   `supybot.plugins.LLM.*ApiKey*` line with the bot stopped, including the stale
   `askApiKey`/`drawApiKey` entries. Expect the flush to have done most of it.
6. **Revoke** KEY-C and KEY-D, plus whatever credentials the stale
   `askApiKey`/`drawApiKey` entries hold.
7. **Decide on `supybot.commands.allowShell`** (currently `True`,
   `bot.conf:153`). `Debug`'s `environ` command replies `repr(os.environ)` **to
   the channel**, and `Debug` is one `@load` away for an owner. Today the
   worst-case owner fat-finger is one key delivered to a PM by `@config`
   (`private=True` routes to PM, it does not censor); afterwards it is all four
   keys in a channel at once. Setting `allowShell: False` closes it. The LLM
   bridge cannot reach any of this — `Owner`, `Config`, `Admin`, `User`,
   `Channel` and `LLM` are all in `DENY_PLUGINS` (`limnoria_bridge.py:26-38`).

**Rollback.** Before step 3, revert the env file. After step 3 the registry
settings no longer exist, so restoring a `bot.conf` entry does nothing. A revert
push is 10-20 minutes and can be blocked entirely by one flaky test, since
`docker.yml` gates on CI success across a three-version matrix with
`--cov-fail-under=93`. The fast path is an image pin, and it must be written into
`docs/guide/operator/operations.md` before step 3:

```
systemctl --user stop vibebot-updater.timer
mkdir -p ~/.config/systemd/user/vibebot.service.d
printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot-v8:sha-<PREV>\n' \
  > ~/.config/systemd/user/vibebot.service.d/override.conf
systemctl --user daemon-reload && systemctl --user restart vibebot
```

Every build is tagged `type=sha` (`docker.yml:41-46`), so the last good image is
addressable. Stopping the updater timer first is required: `vibebot-updater.service`
hardcodes `:latest` and would otherwise bounce the pinned container every 15
minutes.

## New exposure accepted

Environment variables are visible where `bot.conf` was not: `docker inspect`
returns them under `.Config.Env` (readable by anyone in the `docker` group, and
routinely pasted into tickets), `/proc/<pid>/environ` on the host, and child
processes inherit them — `plugin.py:1428` shells out for a git SHA. `journalctl`
is unaffected, since `ExecStart` passes a file path rather than `-e` pairs. This
is accepted: it trades a broad-but-shallow exposure for the elimination of key
drift, and the redaction filter closes the path that actually reaches users.

## Out of scope

- **`vertex_ai` and ADC.** It authenticates by service account, not an API key,
  so "one variable per provider" does not model it. Nothing in the deployment
  uses it. Re-adding it means one dictionary entry plus a keyless-provider flag
  so the guards do not fire on a provider that correctly has no key.
- **Named model sets** (`fun` / `professional`). Once keys are provider-scoped,
  making a channel "fun" is `assistantModel.#chan` plus `verseModel.#chan` for
  the one channel using verse. If revisited: **`Profile` is already taken.**
  `profile.py` defines the *mode* axis (`chat`/`code`/`draw`/`verse`/
  `remind_action`), mapping mode to which registry key holds the model. A
  channel-personality axis is orthogonal and needs a different name (`modelSet`,
  `tier`).
- **`assistantModel`'s `-latest` alias default.** `gemini/gemini-flash-latest`
  silently re-points across model versions. A real problem, unrelated to keys.
