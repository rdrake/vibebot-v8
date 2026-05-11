# Profile Abstraction Design

**Date:** 2026-05-11
**Author:** Richard Drake (with Claude Code)
**Status:** Approved for plan
**Precedent:** `docs/plans/2026-05-11-prompts-consolidation.md`

## Problem

The LLM plugin has 5 chat-loop modes (chat, code, draw, verse, remind_action) each parameterized along multiple axes: model registry key, API key registry key, framework system prompt, channel-overridable personality overlay key, tool visibility set, max output token cap, and a "force initial search" opt-in. These per-mode facts are currently scattered:

- Model and API key registry lookups happen in different files and pass through `model_override`/`api_key` parameters (service.py:3136-3137; callers in plugin.py:2547 etc.).
- Framework prompts are looked up from `PROMPTS[route_profile]` (service.py:3163).
- Tool visibility is encoded on each `ToolSpec.visible_in: frozenset[str]` (assistant.py).
- `max_output_tokens` lives in an inline dict (service.py:3196).
- `force_initial_search` lives in a `route_profile in {…}` check (service.py:3232).
- Overlay registry key is hardcoded per caller (plugin.py:1442 reads `assistantSystemPrompt`, plugin.py:2547 reads `codeSystemPrompt`, plugin.py:3385 reads `assistantSystemPrompt` for verse, etc.).

Adding a 6th mode means touching all these locations. The implicit pairing "this caller hardcodes a registry key that must match that route_profile" is invisible to readers.

## Goal

Introduce a `Profile` frozen dataclass and a `PROFILES: dict[str, Profile]` registry as the single named bundle for the per-mode facts. Mirror the discipline of the prompts consolidation refactor:

- Behavior-preserving at every step.
- Phased commits, each green on `main` for auto-deploy.
- Byte-identity gate before declaring complete.
- New file (`profile.py`) for the registry, parallel to `prompts.py`.

## Non-Goals

- **Overlay composition logic** — the `str.replace`/`format` substitution, the "rules above still apply" footer, the position of overlay in the prompt — stays in service.py untouched. Profile only carries the *registry key name*, not the composition.
- **PROFILE_VERSE special-case footer** — `if route_profile == PROFILE_VERSE` branch at service.py:3169 stays. That's overlay-machinery.
- **Provider adapter extraction** — xAI lane routing in service.py is a separate refactor.
- **Splitting plugin.py per command** — separate refactor.
- **search and image one-shot completions** — they aren't chat-loop profiles and have no `PROFILE_*` constant. PROFILES covers only the 5 chat-loop modes.
- **Flipping tool→profile authority** — `ToolSpec.visible_in` remains the source of truth for tool visibility. Profile delegates tool resolution to the existing `get_tools_for_profile(profile.id)` helper rather than carrying duplicate data.

## Design

### Module layout

New file `plugins/llm/src/llm/profile.py`. Owns:

- The 5 PROFILE_* string constants (moved from assistant.py:27-31): `PROFILE_CHAT`, `PROFILE_CODE`, `PROFILE_DRAW`, `PROFILE_VERSE`, `PROFILE_REMIND_ACTION`.
- The `Profile` dataclass.
- The `PROFILES: dict[str, Profile]` registry.

### Dependency direction

```
prompts.py        (no internal deps)
   ↑
profile.py        (imports prompts only, to validate prompt_id keys exist)
   ↑
assistant.py      (imports PROFILE_* constants from profile)
   ↑
service.py        (imports Profile + PROFILES from profile; ToolSpec stuff from assistant)
   ↑
plugin.py         (imports Profile + PROFILES for caller-side lookups)
```

`profile.py` does **not** import `assistant.py`. This keeps the cycle clear and means Profile is data-only: consumers call `get_tools_for_profile(profile.id, ...)` from the call site rather than `profile.tool_names()`.

### Profile dataclass

```python
@dataclass(frozen=True)
class Profile:
    """Bundle of per-mode configuration for an assistant chat-loop request.

    One Profile per route_profile string. Collapses the route-keyed lookups
    currently scattered across service.assistant_completion into a single
    named record. Tools are *not* a field — they're a many-to-many relation
    owned by assistant.ToolSpec.visible_in; consumers should call
    get_tools_for_profile(profile.id, ...) directly.
    """

    id: str
    model_setting: str
    api_key_setting: str
    prompt_id: str
    overlay_setting: str | None
    max_output_tokens: int | None
    force_search_on_explicit: bool
```

Field semantics:

- `id` — route identifier; must equal the dict key in PROFILES and a key in `prompts.PROMPTS`.
- `model_setting` — Limnoria registry key for the model name (e.g., `"assistantModel"`).
- `api_key_setting` — Limnoria registry key for the API key (e.g., `"assistantApiKey"`).
- `prompt_id` — key into `prompts.PROMPTS` for the framework system prompt.
- `overlay_setting` — Limnoria registry key for the channel-overridable personality overlay (e.g., `"assistantSystemPrompt"`). None means this profile takes no overlay. All 5 current profiles take one; None is reserved for future one-shot profiles.
- `max_output_tokens` — `max_tokens` cap for the LiteLLM completion. None == unbounded.
- `force_search_on_explicit` — if True and an explicit search trigger matches the user prompt, force `tool_choice={search_web}` on step 0.

### PROFILES registry

```python
PROFILES: dict[str, Profile] = {
    PROFILE_CHAT: Profile(
        id=PROFILE_CHAT,
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="chat",
        overlay_setting="assistantSystemPrompt",
        max_output_tokens=2000,
        force_search_on_explicit=True,
    ),
    PROFILE_CODE: Profile(
        id=PROFILE_CODE,
        model_setting="codeModel",
        api_key_setting="codeApiKey",
        prompt_id="code",
        overlay_setting="codeSystemPrompt",
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_DRAW: Profile(
        id=PROFILE_DRAW,
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="draw",
        overlay_setting="assistantSystemPrompt",
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_VERSE: Profile(
        id=PROFILE_VERSE,
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="verse",
        overlay_setting="assistantSystemPrompt",
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_REMIND_ACTION: Profile(
        id=PROFILE_REMIND_ACTION,
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="remind_action",
        overlay_setting="assistantSystemPrompt",
        max_output_tokens=400,
        force_search_on_explicit=True,
    ),
}
```

Sources for these mappings (verified during exploration, pinned with tests):

- `PROFILE_CHAT` model/key — service.py:3136-3137; overlay — plugin.py:1442, service.py:4734.
- `PROFILE_CODE` model/key — service.py:2628 area (codeApiKey check) and plugin.py:2547 overlay.
- `PROFILE_DRAW`, `PROFILE_VERSE`, `PROFILE_REMIND_ACTION` — model/key default to `assistantModel`/`assistantApiKey` because callers don't pass overrides (and `service.assistant_completion` falls back to those when overrides are None).
- `max_output_tokens` — service.py:3196 dict.
- `force_search_on_explicit` — service.py:3232 set.

A verification step at the start of Commit 2 will re-confirm draw/verse/remind_action's model/key reads by reading each caller in plugin.py before committing. If any caller reads a non-assistant model/key for these profiles, the registry entry will be corrected before the migration commit.

## Migration sites

### service.py:assistant_completion (~3120–3260)

Before:
```python
target = self._channel_target(channel)
model = model_override or self.plugin.registryValue("assistantModel", target)
effective_api_key = api_key or self.plugin.registryValue("assistantApiKey", target)
...
framework = PROMPTS.get(route_profile, PROMPTS["chat"]).format(bot_nick=bot_nick)
...
profile_max_output = {PROFILE_CHAT: 2000, PROFILE_REMIND_ACTION: 400}.get(route_profile)
if profile_max_output is not None:
    optional_kwargs["max_tokens"] = profile_max_output
...
profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)
...
force_initial_search = (
    route_profile in {PROFILE_CHAT, PROFILE_REMIND_ACTION}
    and search_fn is not None
    and _has_tool(profile_tools, "search_web")
    and EXPLICIT_SEARCH_RE.search(prompt) is not None
)
```

After:
```python
profile = PROFILES[route_profile]
target = self._channel_target(channel)
model = model_override or self.plugin.registryValue(profile.model_setting, target)
effective_api_key = api_key or self.plugin.registryValue(profile.api_key_setting, target)
...
framework = PROMPTS[profile.prompt_id].format(bot_nick=bot_nick)
...
if profile.max_output_tokens is not None:
    optional_kwargs["max_tokens"] = profile.max_output_tokens
...
profile_tools = get_tools_for_profile(profile.id, exclude=exclude_tools)
...
force_initial_search = (
    profile.force_search_on_explicit
    and search_fn is not None
    and _has_tool(profile_tools, "search_web")
    and EXPLICIT_SEARCH_RE.search(prompt) is not None
)
```

The PROFILE_VERSE overlay-footer branch (service.py:3169-3180) stays unchanged.

### plugin.py overlay reads

Three caller sites + one in service.py:4734:

- `plugin.py:1442` (ask path): `registryValue("assistantSystemPrompt", channel)` → `registryValue(PROFILES[PROFILE_CHAT].overlay_setting, channel)`
- `plugin.py:2547` (code path): `registryValue("codeSystemPrompt", channel)` → `registryValue(PROFILES[PROFILE_CODE].overlay_setting, channel)`
- `plugin.py:3385` (verse path): `registryValue("assistantSystemPrompt", channel)` → `registryValue(PROFILES[PROFILE_VERSE].overlay_setting, channel)`
- `service.py:4734` (scheduled task fire): `registryValue("assistantSystemPrompt", row.channel)` → `registryValue(PROFILES[PROFILE_CHAT].overlay_setting, row.channel)`

Behavior preserved (same string read, same channel target). The pairing "this caller belongs to that profile" becomes data-driven.

### Out of migration scope

- Loom paths reading `assistantApiKey` directly (plugin.py:4983, 5080, 5870) — not profile dispatch, direct-model usage. Untouched.
- One-shot `search_completion` (service.py:2271) — not a chat-loop profile. Untouched.
- `image_completion` paths reading `imageModel`/`imageApiKey` — not a chat-loop profile. Untouched.

## Testing strategy

New file: `plugins/llm/tests/test_profile.py`. Pattern mirrors `test_prompts.py` — invariants over the registry.

Test groups:

- **TestProfilesRegistry** — registry has exactly the 5 expected keys; `Profile.id` matches dict key for every entry.
- **TestProfileResolution** — `prompt_id` is a real key in `prompts.PROMPTS`; `model_setting` / `api_key_setting` / `overlay_setting` are members of pinned literal sets from config.py.
- **TestProfileToolsAlignment** — `get_tools_for_profile(profile.id)` returns a non-empty list for every entry.
- **TestBehaviorPreservation** — frozen literal mappings (max_tokens, force_search membership, prompt_id per profile) pinned to pre-refactor values. Any future change to PROFILES must explicitly update these tests.

Existing tests:

- `plugins/llm/tests/test_assistant.py` may import `PROFILE_CHAT` etc. from `llm.assistant`. During the transition (commits 1–3), assistant.py keeps PROFILE_* as re-exports. Commit 4 deletes the re-exports; any test still importing from assistant.py at that point gets switched to `llm.profile`.
- `plugins/llm/tests/test_service.py` exercises `route_profile=` paths and should pass unchanged since Profile keys on the same strings.

Coverage: profile.py is ~50 lines of pure data + frozen dataclass. The invariant tests touch every field of every entry, so coverage is 100%. Migrated lines in service.py/plugin.py retain their existing test coverage paths since behavior is unchanged.

## Sequencing

Four commits, each green on `main` for auto-deploy.

### Commit 1: Add profile.py + tests, no consumers

- Create `plugins/llm/src/llm/profile.py` as the *new definition site* for the PROFILE_* string constants, the Profile dataclass, and the PROFILES registry.
- Update `assistant.py` so the existing `PROFILE_CHAT = "chat"` etc. lines become `from .profile import PROFILE_CHAT, PROFILE_CODE, ...` (re-imports, not duplicate string literals). This keeps every existing import path through `llm.assistant` working without two divergent sources of truth for the literal values.
- Create `plugins/llm/tests/test_profile.py` with invariant tests.
- CI green, no behavior change.
- Commit message: `refactor(llm): add profile.py as single source of truth (no consumers yet)`

### Commit 2: Migrate service.py:assistant_completion to PROFILES (atomic with test updates)

- Verification step first: re-read draw/verse/remind_action caller sites in plugin.py to confirm model/key registry reads. Correct PROFILES if any divergence found.
- `assistant_completion` reads `profile = PROFILES[route_profile]` and pulls model_setting / api_key_setting / prompt_id / max_output_tokens / force_search_on_explicit from it.
- The 5 scattered lookups inside the function collapse to one named record.
- PROFILE_VERSE footer at service.py:3169 stays untouched.
- Test updates that asserted literal `route_profile in {CHAT, REMIND_ACTION}` shape get rewritten to assert via Profile (kept atomic with service.py change so CI stays green).
- Commit message: `refactor(llm): migrate service.assistant_completion to PROFILES registry`

### Commit 3: Migrate plugin.py + service.py:4734 caller sites

- 4 call sites (plugin.py:1442, 2547, 3385; service.py:4734) read overlay registry key via `PROFILES[<profile_const>].overlay_setting` instead of hardcoded string.
- Same behavior, data-driven pairing.
- Commit message: `refactor(llm): migrate plugin.py overlay reads to PROFILES.overlay_setting`

### Commit 4: Cleanup — delete re-exports, byte-identity gate

- Remove the `from .profile import PROFILE_*` re-export lines from assistant.py. Any remaining consumer that imports `PROFILE_CHAT` etc. from `llm.assistant` gets switched to import from `llm.profile`.
- Run `make preflight`.
- Run the byte-identity script (see below).
- Commit message: `refactor(llm): drop assistant.py PROFILE_* re-export shim`

## Byte-identity gate

A standalone Python script (one-off harness, not checked in — runs locally only) captures, for each route_profile, every observable selected by Profile dispatch:

- Resolved framework prompt text after `.format(bot_nick="testbot")`
- Resolved `max_tokens` value
- Resolved `force_initial_search` predicate against a fixed sample prompt
- Resolved tool name set from `get_tools_for_profile(profile_id)`
- Overlay registry key string

The script outputs a single sha256 digest. The digest produced on `main` *before* the refactor must equal the digest produced after Commit 4. Any divergence means the refactor changed observable behavior — block the merge.

This mirrors the prompts plan's "sha256-identical to pre-refactor source" gate.

## Workflow notes

- Push directly to `main`, no PRs. Auto-deploy on green CI.
- `make lint && make typecheck` runs on every Edit via hook.
- 93% coverage floor is hard; new tests are designed to keep coverage at or above current.
- Memory prompts and framework prompts stay in `prompts.py`; nothing in this refactor relocates them.
- If any technical call is uncertain during implementation, invoke `codex:codex-rescue` mid-flight rather than guessing.

## Adversarial review plan

Before executing, run two reviewers in parallel:

1. `codex:codex-rescue` — reviews the implementation plan for correctness, hidden assumptions, and missed call sites.
2. `superpowers` code-reviewer subagent — reviews the plan for byte-identity-gate validity, test invariant completeness, and discipline matching the prompts precedent.

Both must clear before execution. On the prompts plan they caught a broken hash check, an unused-import lint break, and missing test invariants — apply the same bar here.
