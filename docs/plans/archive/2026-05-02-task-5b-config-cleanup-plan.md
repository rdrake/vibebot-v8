---
status: ready-for-review
date: 2026-05-02
phase: 2
task: 5b (config consolidation — remove command-era registrations)
predecessor: T5a (commits 6dc0f92..5a74bf9)
design_plan: docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md
findings_doc: docs/plans/2026-05-02-settings-config-simplification-findings.md
---

# Phase 2 Task 5b — Remove command-era registry keys

## Goal

Delete the deprecated command-era registry keys that T5a's `resolve_setting`
shim has been forwarding from. After this task ships:

- The plugin registers only the capability-based keys (`assistantModel`,
  `assistantApiKey`, `assistantSystemPrompt`, `imageModel`, `imageApiKey`,
  plus existing `codeModel`, `codeApiKey`, `codeSystemPrompt`, `searchModel`,
  `searchApiKey`).
- `resolve_setting` is either removed (callers go back to `registryValue`)
  or kept as a no-op pass-through with the fallback list dropped.
- The `%g` / grok command and its three `grok*` settings are removed.
- All fixtures, tests, docs, and error messages name the new keys.

## Operator migration prerequisite

T5a has been deployed since commit `5a74bf9`. T5b is destructive — once the
old registrations are gone, any value an operator still has under
`askModel` / `askApiKey` / `drawModel` / etc. is silently dropped on next
plugin load. **Before this task merges, the production operator must mirror
their old keys onto the new names.** The recommended sequence:

1. From IRC, for each old key with a non-empty value, copy it to the new
   key (e.g. `@config plugins.LLM.assistantModel <value>`,
   `@config plugins.LLM.assistantApiKey <value>`, ...).
2. Confirm the deprecation warnings stop appearing in journalctl after the
   next bot restart (one warning per old key per process; absence of the
   warning over a full uptime confirms the new key is winning).
3. Only then merge T5b.

A short migration helper is in scope for this task (see Step 1 below) so the
operator does not have to grep their config by hand.

## Scope

### In scope

- `plugins/llm/src/llm/config.py` — remove the registrations listed below,
  drop the docstrings that explain "falls back to ..." for the surviving
  keys.
- `plugins/llm/src/llm/service.py` — strip `fallbacks=(...)` arguments from
  every `resolve_setting` call, then either:
  - convert each call to a plain `registryValue(...)` (preferred — fewer
    indirections), or
  - leave `resolve_setting` as a one-liner wrapper for future use.
  Pick (a). The shim was a transition aid, not a permanent abstraction.
- `plugins/llm/src/llm/plugin.py` — same treatment; remove the `%g`
  command (`grok` wrapper) and the lookup of `grok*` settings.
- `plugins/llm/src/llm/config.py` — delete the `resolve_setting` helper and
  the deprecation-warning bookkeeping (`_resolve_setting_warned`,
  `_resolve_setting_lock`).
- Tests under `plugins/llm/tests/` — every reference to a deleted key
  becomes the new key. Tests asserting deprecation-warning behaviour are
  deleted.
- Docs (`docs/guide/operator/configuration.md`,
  `docs/guide/operator/tuning-monitoring.md`, `README.md`) — replace
  command-era key names with capability-based names. Remove any "old name
  X maps to new name Y" tables once the old name is gone from code.
- `AGENTS.md` — update the T5a compat-shim line to note T5b removed it.

### Removed registrations (config.py)

API keys: `askApiKey`, `drawApiKey`, `memoryApiKey`, `spontaneousApiKey`,
`grokApiKey`.

System prompts: `askSystemPrompt`, `grokSystemPrompt`.

Models: `askModel`, `drawModel`, `metaModel`, `memoryExtractionModel`,
`memoryCleanupModel`, `spontaneousModel`, `grokModel`.

`metaApiKey` was already a fallback target; remove it too.

### Kept as-is

- `codeModel`, `codeApiKey`, `codeSystemPrompt` — code generation has its
  own model surface (per findings doc §Code).
- `searchModel`, `searchApiKey` — provider grounding requirements (per
  findings doc §Search). Existing fallback to `assistantModel` /
  `assistantApiKey` stays; that's a runtime fallback, not a registry one.
- `spontaneousSystemPrompt` — behaviour mode prompt, not a model lookup
  (per findings doc §System Prompt Settings).
- All non-model/key settings (`memoryEnabled`, `spontaneousChance`,
  context settings, draw timing, etc.) untouched.

### Out of scope

- Any new feature work. Pure cleanup.
- Migrating reminder-fire path or memory-extraction path to a different
  model strategy. T5a already pointed all those callsites at
  `assistantModel` via `resolve_setting`; T5b just removes the fallback
  arm.
- Deeper provider-aware fallback rework. The findings doc's "fallback
  policy" already shipped as documentation in T5a; T5b doesn't change
  runtime behaviour beyond the rename.

## Implementation steps

### Step 1 — Migration helper command

Add a one-shot operator command `@migrateLlmConfig` (or a Makefile target,
or a doc snippet — pick the cheapest form) that prints, for each
non-empty old key, the matching new key and the value the operator should
copy. Example output:

```
askModel = gemini/gemini-flash-latest  → set assistantModel
askApiKey = sk-xxx (redacted)          → set assistantApiKey
drawModel = vertex_ai/imagen-4.0-...   → set imageModel
```

This runs before the destructive merge so the operator can mirror values
without grepping the config dump. Decision: ship as a small admin command
in `plugin.py` rather than a Makefile target — it needs registry access
that's awkward outside the bot process. Gate behind `owner` capability.

**Done when:** the command exists, lists every non-empty old key with the
matching new key, redacts API key values, and is covered by a unit test.

### Step 2 — Drop fallbacks from `resolve_setting` callers

For each `resolve_setting(self.plugin, "X", channel, fallbacks=(...))`
in `service.py` and `plugin.py`, replace with
`self.plugin.registryValue("X", channel)` (or `plugin.registryValue` for
the bare-plugin form).

Affected callsites (from grep): `plugin.py:868,877,1193,2512`,
`service.py:1495,1895,1898,1950,1953,2118,2123,2188,2306,2311,2411,2417,2494,2614,2620,2906,2909,3603,3609,3668,3674,3954` (verify offsets at impl time, file may have shifted).

The "search/fetch falls back to assistant" pairs in service.py
(1895/1898, 1950/1953, etc.) keep their runtime `or resolve_setting(...)`
fallback to `assistantModel`/`assistantApiKey` — that's a search-to-
assistant fallback, not an old-key fallback. Convert to a plain
`registryValue` for the assistant arm and delete only the old-key shim.

**Done when:** no `resolve_setting(...)` call passes a `fallbacks=` argument
that names a removed key; `make lint` and `make typecheck` pass.

### Step 3 — Delete the `resolve_setting` helper

After Step 2, every remaining call would be a one-liner forwarder. Delete
the helper and the bookkeeping globals (`_resolve_setting_warned`,
`_resolve_setting_lock`). Update the import block in `service.py` and
`plugin.py` accordingly.

**Done when:** `resolve_setting` is gone from `config.py`; no remaining
imports reference it; `make lint` clean.

### Step 4 — Remove old registrations

Delete the registry blocks for the keys listed in *Removed registrations*
above. Be careful around the comment headers (`API Keys`, `System
Prompts`, `Model Configuration`, `Memory Extraction`, `Spontaneous
Participation`) — keep the section comments but trim them to reflect
what remains.

**Done when:** `config.py` no longer registers any key on the removed
list; the file still parses; the test under `tests/test_config.py` that
enumerates registered keys passes (update its expected list).

### Step 5 — Remove the `%g` / grok command

In `plugin.py`, delete the `grok` command wrapper (the `def grok(...)`
method registered by `@wrap`), the help string, and the example lines
(`%g what's the deal with airline food` etc.). Update any code that
references `_run_preflight(... command="g")` if that name is in use.

**Done when:** no `grok` symbol in `plugin.py`; the help test passes
without it; smoke test `@list LLM` no longer shows `g` / `grok`.

### Step 6 — Migrate tests and fixtures

Update `tests/conftest.py` defaults (the dict that pre-populates plugin
registry values for tests) to use the new keys. Update each test file
listed in the grep (test_assistant, test_config, test_etiquette,
test_provider_edge_cases, test_reminders, test_service, test_spontaneous,
test_stress) to reference the new keys.

The compat-shim deprecation tests in `test_config.py` are deleted.

**Done when:** `make test` is green; coverage is at or above the prior
floor.

### Step 7 — Migrate docs

`docs/guide/operator/configuration.md` and
`docs/guide/operator/tuning-monitoring.md` reference the old keys in
example commands and tuning advice. Replace the names. The migration map
("askModel → assistantModel") stays as a one-time historical note in the
T5a section if any remains, otherwise gets deleted.

`README.md` — the install/setup snippet may reference `askApiKey`. Replace
with `assistantApiKey`.

`AGENTS.md` — change the T5a line ("T5a compat shim that prefers ...") to
note T5b deleted the shim and only the new keys remain.

**Done when:** `make docs` builds clean; no remaining mentions of the
deleted keys outside of changelogs / git history.

### Step 8 — Final sweep + preflight

`grep -nrE 'askModel|askApiKey|...' plugins/llm/ docs/guide/ README.md
AGENTS.md` should produce zero hits. Run `make preflight`. Smoke-test on
the dev bot: a fresh channel with only `assistantModel` and
`assistantApiKey` set runs an `@ask` end to end.

**Done when:** preflight passes; smoke test succeeds; commit pushed.

## Verification

- Unit test: with only new settings configured, every code path that
  previously fell back via the shim (assistant chat, memory extraction,
  reminder parsing, image-prompt rewrite, spontaneous, search) reads
  the new key directly with no warnings logged.
- Unit test: deleting the registrations does not break plugin load
  (`@load LLM` succeeds with a stale conf file that still has the old
  entries — they're ignored by the registry, not raised as errors).
- Integration test: an operator who *only* set the old keys and skipped
  the migration sees their commands fall back to the registry default
  (which for `assistantModel` is empty, so `@ask` returns the
  "Assistant model not configured" error). This is the expected
  destructive behaviour and is documented in the changelog.
- Smoke test (dev bot): `@ask`, `@code`, `@draw`, memory extraction
  trigger, reminder fire, scheduled-task fire — all succeed with new
  keys only.

## Risk callouts

1. **Operator config drop.** As above — destructive. The Step 1 migration
   helper + the warning that's been live since T5a are the mitigation.
2. **Hidden references in third-party plugins.** Anything inheriting from
   the LLM plugin or reading its registry values directly will break. A
   single-operator deployment makes this near-zero risk; for a wider
   release we'd add a one-version-shim fallback. Skip for v1.
3. **Test fixture churn.** 8 test files touch old keys. The conftest
   default dict is the natural single point of truth — update it once,
   and most tests pick up the new defaults without per-file edits. Audit
   per-file overrides anyway.
4. **`%g` removal is user-visible.** Any operator who taught their
   channel `%g` will see "no such command" until they relearn `%ask`.
   Mention in the commit message and in the operator docs.

## Sequencing

This plan has no further dependencies after T5a. T5b can ship in a single
PR (steps 1–8 above) or be split as 1+2+3 → 4+5+6+7+8 if the diff is too
large to review at once. Lean toward single PR unless the diff exceeds
~600 lines after fixture updates.

## Done when

- Every old registration listed in *Removed registrations* is gone.
- `resolve_setting` is gone.
- `%g` / `grok` command and grok* settings are gone.
- All tests, docs, and fixtures use the capability-based names.
- `make preflight` passes.
- A smoke test on the dev bot exercises chat, code, draw, memory, and
  scheduled-task fire paths end to end with only the new keys configured.
