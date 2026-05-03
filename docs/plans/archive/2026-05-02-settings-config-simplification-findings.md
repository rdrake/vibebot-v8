---
status: findings
date: 2026-05-02
topic: LLM settings config simplification
---

# Settings Config Simplification Findings

## Summary

The current LLM config still reflects the older command-oriented architecture:
`ask`, `code`, `draw`, `meta`, `grok`, `search`, `memory`, and `spontaneous`
each have separate model and/or API key settings.

That no longer matches the direction of the plugin. The architecture is moving
toward one assistant that routes requests through a small set of native tools
and a lean Limnoria bridge. The config should describe capabilities, not legacy
entry commands.

Recommended target model surface:

- `assistantModel`
- `imageModel`
- `codeModel`
- `searchModel`

Recommended target API key surface:

- `assistantApiKey`
- `imageApiKey`
- `codeApiKey`
- `searchApiKey`

This keeps the operational model simple while preserving the workloads that
still have genuinely different provider requirements.

## Current State

Current model settings include:

- `askModel`
- `codeModel`
- `drawModel`
- `searchModel`
- `grokModel`
- `metaModel`
- `memoryExtractionModel`
- `memoryCleanupModel`
- `spontaneousModel`

Current API key settings include:

- `askApiKey`
- `codeApiKey`
- `drawApiKey`
- `searchApiKey`
- `grokApiKey`
- `metaApiKey`
- `memoryApiKey`
- `spontaneousApiKey`

The command names are now misleading. `ask` is no longer just a plain ask
completion; it uses the shared assistant loop with tool access. `metaModel`
already represents the real assistant loop. Memory extraction, memory cleanup,
reminder parsing, image prompt rewriting, and spontaneous participation are
background assistant tasks rather than separate user-facing products.

## Proposed Config Surface

### Assistant

`assistantModel` and `assistantApiKey` should become the default text/tool
model configuration.

Use for:

- Chat and normal assistant requests.
- Tool-calling/planning loop.
- Limnoria bridge selection and result synthesis.
- Reminder parsing.
- Reminder fire-time actions.
- Memory extraction.
- Memory cleanup.
- Spontaneous participation.
- Image prompt rewrite helper.
- Vision in chat, if supported by the selected model.

This replaces:

- `askModel`
- `askApiKey`
- `metaModel`
- `metaApiKey`
- `memoryExtractionModel`
- `memoryCleanupModel`
- `memoryApiKey`
- `spontaneousModel`
- `spontaneousApiKey`

### Image

`imageModel` and `imageApiKey` should configure native image generation.

This replaces:

- `drawModel`
- `drawApiKey`

Image generation should not silently fall back to `assistantModel`, because
image providers use different APIs and model families. Falling back from
`imageApiKey` to `assistantApiKey` is only safe when the configured image model
uses the same provider account, so explicit configuration is cleaner.

### Code

`codeModel` and `codeApiKey` should remain separate.

Reason: code generation has different cost and quality requirements from chat.
It is also an inner tool call that produces a saved artifact, not just final
assistant prose.

`codeSystemPrompt` can remain as the inner code-generation prompt. It should
not be confused with the assistant planner prompt for the `code` profile.

### Search

`searchModel` and `searchApiKey` should remain separate.

Reason: search/fetch uses provider-specific grounding features. Keeping this
separate lets operators choose a model known to support Google Search or URL
Context without constraining the main assistant model.

Search may fall back to `assistantApiKey` when the provider account is shared,
but the model should stay explicit. If `searchModel` is empty, a documented
fallback to `assistantModel` is acceptable only when the assistant model supports
the grounding tools being requested.

## System Prompt Settings

Recommended prompt surface:

- `assistantSystemPrompt`
- `codeSystemPrompt`
- `spontaneousSystemPrompt`, only if spontaneous participation remains a
  distinct behavior mode.

`assistantSystemPrompt` replaces `askSystemPrompt`.

`codeSystemPrompt` remains useful because the inner code tool has a different
output contract from normal assistant chat.

`spontaneousSystemPrompt` is not primarily a model-routing setting. It is a
behavior/personality setting for unsolicited channel participation. It can stay
if spontaneous participation stays.

`grokSystemPrompt` should go away with the provider-specific Grok command, or
become a generic per-route override only if there is still a strong operational
reason to support one.

## Grok Command

The `%g` command cuts against the simplification goal because it creates a
provider-specific first-class route:

- `grokModel`
- `grokApiKey`
- `grokSystemPrompt`

Recommendation: remove it or deprecate it as a compatibility alias. If operators
want Grok, they can set `assistantModel` to an xAI model for a channel. Keeping a
provider-specific command means the config will keep drifting back toward
command/provider special cases.

## Fallback Policy

Recommended fallback rules:

- `assistantModel` and `assistantApiKey` are required for assistant behavior.
- `imageModel` is required for image generation.
- `imageApiKey` is required unless explicitly documented to fall back to
  `assistantApiKey` for same-provider deployments.
- `codeModel` is required for code generation.
- `codeApiKey` may fall back to `assistantApiKey`.
- `searchModel` may fall back to `assistantModel` only if the selected model
  supports grounding.
- `searchApiKey` may fall back to `assistantApiKey`.

Avoid deep fallback chains. They make operator behavior hard to predict and can
send requests to an unintended provider.

## Migration Map

Suggested compatibility mapping:

| Old setting | New setting |
| --- | --- |
| `askModel` | `assistantModel` |
| `askApiKey` | `assistantApiKey` |
| `askSystemPrompt` | `assistantSystemPrompt` |
| `metaModel` | `assistantModel` |
| `metaApiKey` | `assistantApiKey` |
| `drawModel` | `imageModel` |
| `drawApiKey` | `imageApiKey` |
| `codeModel` | `codeModel` |
| `codeApiKey` | `codeApiKey` |
| `searchModel` | `searchModel` |
| `searchApiKey` | `searchApiKey` |
| `memoryExtractionModel` | `assistantModel` |
| `memoryCleanupModel` | `assistantModel` |
| `memoryApiKey` | `assistantApiKey` |
| `spontaneousModel` | `assistantModel` |
| `spontaneousApiKey` | `assistantApiKey` |
| `grokModel` | remove or map manually to `assistantModel` |
| `grokApiKey` | remove or map manually to `assistantApiKey` |
| `grokSystemPrompt` | remove |

## Things Easy To Miss

- Chat vision currently rides on `askModel`. After renaming, document that
  `assistantModel` must support vision if image URLs in chat should work.
- The assistant loop currently uses `metaModel` before falling back to
  `askModel`. Removing `metaModel` should simplify that to one lookup.
- Reminder parsing currently uses `askModel` and `askApiKey`. It should become
  an assistant workload.
- Image prompt rewriting currently uses `askModel` and `askApiKey`. It should
  become an assistant workload, not image workload.
- Memory cleanup currently has a separate model from memory extraction. That is
  operationally flexible but probably not worth the config surface.
- Spontaneous participation has separate model/key settings but is just another
  assistant evaluation with a different system prompt.
- `codeSystemPrompt` is still used by the inner code-generation tool. Do not
  remove it while the code tool remains.
- Existing tests and docs still use command-era wording. Any implementation
  should update fixtures, docs, and error messages together.

## Recommendation

Make the next cleanup a single config migration, not a feature change:

1. Introduce the four new model settings and four new API key settings.
2. Read old settings as compatibility fallbacks for one release cycle.
3. Update service/plugin lookups to use capability names.
4. Update docs and tests to stop teaching command-era config.
5. Remove or formally deprecate `%g` and its settings.

This keeps the bridge lean and aligns the operator-facing surface with the
actual architecture: one assistant, a few native AI capabilities, and Limnoria
for reusable bot functionality.
