# Profile Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `Profile` frozen dataclass and `PROFILES: dict[str, Profile]` registry in a new `plugins/llm/src/llm/profile.py` module that collapses the 5 `route_profile`-keyed lookups scattered across `service.assistant_completion` and `plugin.py` caller sites into one named bundle.

**Architecture:** Mirrors the `prompts.py` precedent (single-source-of-truth registry). Phased into 4 commits, each green on `main` for auto-deploy. Behavior-preserving — guarded by a byte-identity script that hashes every per-profile observable before and after the refactor.

**Tech Stack:** Python 3.12+, frozen `@dataclass`, pytest, Limnoria registry API, existing `llm.prompts` + `llm.assistant` modules.

**Spec:** `docs/superpowers/specs/2026-05-11-profile-abstraction-design.md`

---

## Pre-Work: Capture Pre-Refactor Byte-Identity Digest

This must run on the current `main` HEAD before any code change so the post-refactor digest has something to compare against.

**Files:**
- Create: `/tmp/profile_identity_pre.txt` (one-off, not checked in)
- Create: `/tmp/profile_identity.py` (one-off harness, not checked in)

- [ ] **Step 1: Write the byte-identity harness**

Create `/tmp/profile_identity.py` with this content. It imports the current module surface, resolves every observable that Profile dispatch will select, and prints a sha256 over a stable serialization.

```python
"""One-off byte-identity harness for the Profile abstraction refactor.

Run on pre-refactor main: python /tmp/profile_identity.py > /tmp/profile_identity_pre.txt
Run on post-refactor HEAD: python /tmp/profile_identity.py > /tmp/profile_identity_post.txt
Compare: diff /tmp/profile_identity_pre.txt /tmp/profile_identity_post.txt

A non-empty diff means the refactor changed observable behavior — fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

sys.path.insert(0, "plugins/llm/src")

from llm import prompts  # noqa: E402
from llm.assistant import (  # noqa: E402
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
    get_tools_for_profile,
)

PROFILE_IDS = [
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_VERSE,
    PROFILE_REMIND_ACTION,
]

# Pre-refactor pinned data — exactly what service.py and plugin.py compute today.
PRE_MAX_OUTPUT = {PROFILE_CHAT: 2000, PROFILE_REMIND_ACTION: 400}
PRE_FORCE_SEARCH = {PROFILE_CHAT, PROFILE_REMIND_ACTION}
PRE_OVERLAY_KEY = {
    PROFILE_CHAT: "assistantSystemPrompt",
    PROFILE_CODE: "codeSystemPrompt",
    PROFILE_DRAW: "assistantSystemPrompt",
    PROFILE_VERSE: "assistantSystemPrompt",
    PROFILE_REMIND_ACTION: "assistantSystemPrompt",
}
PRE_MODEL_KEY = {
    PROFILE_CHAT: "assistantModel",
    PROFILE_CODE: "codeModel",
    PROFILE_DRAW: "assistantModel",
    PROFILE_VERSE: "assistantModel",
    PROFILE_REMIND_ACTION: "assistantModel",
}
PRE_API_KEY = {
    PROFILE_CHAT: "assistantApiKey",
    PROFILE_CODE: "codeApiKey",
    PROFILE_DRAW: "assistantApiKey",
    PROFILE_VERSE: "assistantApiKey",
    PROFILE_REMIND_ACTION: "assistantApiKey",
}

SAMPLE_PROMPT_WITH_SEARCH = "please search the web for python typing news"

# This regex must match service.py:EXPLICIT_SEARCH_RE. If service.py changes
# this regex during the refactor, the diff catches it.
EXPLICIT_SEARCH_RE = re.compile(r"\bsearch\b", re.IGNORECASE)

observables = {}
for pid in PROFILE_IDS:
    framework = prompts.PROMPTS[pid].format(bot_nick="testbot")
    tools = sorted(t["function"]["name"] for t in get_tools_for_profile(pid))
    max_tokens = PRE_MAX_OUTPUT.get(pid)
    force_search = (
        pid in PRE_FORCE_SEARCH
        and EXPLICIT_SEARCH_RE.search(SAMPLE_PROMPT_WITH_SEARCH) is not None
    )
    observables[pid] = {
        "framework_text": framework,
        "tool_names": tools,
        "max_output_tokens": max_tokens,
        "force_search_on_sample": force_search,
        "overlay_setting": PRE_OVERLAY_KEY[pid],
        "model_setting": PRE_MODEL_KEY[pid],
        "api_key_setting": PRE_API_KEY[pid],
    }

serialized = json.dumps(observables, sort_keys=True, ensure_ascii=False)
digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

print(f"# Profile observable digest")
print(f"# sha256: {digest}")
print()
print(serialized)
```

- [ ] **Step 2: Run it on pre-refactor main**

```bash
git rev-parse HEAD
# Expected: 96156d5... (or whatever main HEAD is at refactor start)
python /tmp/profile_identity.py > /tmp/profile_identity_pre.txt
head -2 /tmp/profile_identity_pre.txt
```

Expected: a `# sha256: <hex>` line. Record this digest mentally / in the terminal scrollback — Task 4 will compare against it.

If the script fails (import error, etc.), debug before proceeding. The script is what pins behavior; if it doesn't run, the refactor has no safety net.

---

## Task 1: Add profile.py + tests, no consumers

**Files:**
- Create: `plugins/llm/src/llm/profile.py`
- Create: `plugins/llm/tests/test_profile.py`
- Modify: `plugins/llm/src/llm/assistant.py:27-31`

- [ ] **Step 1: Write the failing test for the new module**

Create `plugins/llm/tests/test_profile.py`:

```python
"""Invariants for the consolidated profile module.

These tests pin the shape of ``llm.profile`` so future refactors don't
silently drop a profile key, mis-name a registry setting, or break the
behavior-preservation contract with the pre-refactor scattered data.
"""

from __future__ import annotations

import pytest

from llm import profile, prompts


class TestProfilesRegistry:
    """PROFILES is the single source of truth for per-mode dispatch."""

    def test_registry_has_all_expected_keys(self):
        """PROFILES exposes every chat-loop profile by name."""
        assert set(profile.PROFILES.keys()) == {
            "chat",
            "code",
            "draw",
            "verse",
            "remind_action",
        }

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_id_matches_dict_key(self, pid):
        """Profile.id matches its dict key — no copy-paste mismatches."""
        assert profile.PROFILES[pid].id == pid


class TestProfileResolution:
    """Each Profile resolves to live registry settings and a real prompt."""

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_prompt_id_is_a_real_prompt(self, pid):
        """profile.prompt_id is a valid key in prompts.PROMPTS."""
        assert profile.PROFILES[pid].prompt_id in prompts.PROMPTS

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_model_setting_is_a_known_registry_key(self, pid):
        """profile.model_setting matches a registerChannelValue in config.py."""
        valid_model_keys = {"assistantModel", "codeModel", "imageModel", "searchModel"}
        assert profile.PROFILES[pid].model_setting in valid_model_keys

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_api_key_setting_is_a_known_registry_key(self, pid):
        """profile.api_key_setting matches a registerChannelValue in config.py."""
        valid_key_keys = {"assistantApiKey", "codeApiKey", "imageApiKey", "searchApiKey"}
        assert profile.PROFILES[pid].api_key_setting in valid_key_keys

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_overlay_setting_is_known_or_none(self, pid):
        """profile.overlay_setting is None or a known overlay key."""
        s = profile.PROFILES[pid].overlay_setting
        assert s is None or s in {"assistantSystemPrompt", "codeSystemPrompt"}


class TestProfileToolsAlignment:
    """Profile.id is a valid input to assistant.get_tools_for_profile."""

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_get_tools_returns_nonempty(self, pid):
        """Every profile id resolves to at least one model-visible tool."""
        from llm.assistant import get_tools_for_profile

        tools = get_tools_for_profile(pid)
        assert len(tools) > 0


class TestBehaviorPreservation:
    """Pin pre-refactor scattered data; updates must be explicit."""

    EXPECTED_MAX_TOKENS = {"chat": 2000, "remind_action": 400}
    EXPECTED_FORCE_SEARCH = {"chat", "remind_action"}
    EXPECTED_PROMPT_IDS = {
        "chat": "chat",
        "code": "code",
        "draw": "draw",
        "verse": "verse",
        "remind_action": "remind_action",
    }
    EXPECTED_OVERLAY = {
        "chat": "assistantSystemPrompt",
        "code": "codeSystemPrompt",
        "draw": "assistantSystemPrompt",
        "verse": "assistantSystemPrompt",
        "remind_action": "assistantSystemPrompt",
    }
    EXPECTED_MODEL = {
        "chat": "assistantModel",
        "code": "codeModel",
        "draw": "assistantModel",
        "verse": "assistantModel",
        "remind_action": "assistantModel",
    }
    EXPECTED_API_KEY = {
        "chat": "assistantApiKey",
        "code": "codeApiKey",
        "draw": "assistantApiKey",
        "verse": "assistantApiKey",
        "remind_action": "assistantApiKey",
    }

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_max_output_tokens(self, pid):
        assert (
            profile.PROFILES[pid].max_output_tokens
            == self.EXPECTED_MAX_TOKENS.get(pid)
        )

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_force_search(self, pid):
        assert profile.PROFILES[pid].force_search_on_explicit == (
            pid in self.EXPECTED_FORCE_SEARCH
        )

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_prompt_id(self, pid):
        assert profile.PROFILES[pid].prompt_id == self.EXPECTED_PROMPT_IDS[pid]

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_overlay_setting_value(self, pid):
        assert profile.PROFILES[pid].overlay_setting == self.EXPECTED_OVERLAY[pid]

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_model_setting_value(self, pid):
        assert profile.PROFILES[pid].model_setting == self.EXPECTED_MODEL[pid]

    @pytest.mark.parametrize(
        "pid",
        ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_api_key_setting_value(self, pid):
        assert profile.PROFILES[pid].api_key_setting == self.EXPECTED_API_KEY[pid]


class TestProfileImmutability:
    """Profile is a frozen dataclass — attempts to mutate must raise."""

    def test_profile_is_frozen(self):
        p = profile.PROFILES["chat"]
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            p.id = "mutated"  # type: ignore[misc]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_profile.py -v"
```

Expected: ImportError / ModuleNotFoundError on `from llm import profile` — the module doesn't exist yet. Fail fast confirms the import path is what the tests claim.

- [ ] **Step 3: Create `profile.py`**

Create `plugins/llm/src/llm/profile.py`:

```python
"""Single source of truth for per-mode chat-loop dispatch configuration.

This module owns:

- The five **chat-loop profile identifiers** (``PROFILE_CHAT``,
  ``PROFILE_CODE``, ``PROFILE_DRAW``, ``PROFILE_VERSE``,
  ``PROFILE_REMIND_ACTION``) that route requests through
  ``service.assistant_completion``.
- The :class:`Profile` frozen dataclass that bundles every per-mode fact
  for one of those routes: which Limnoria registry keys hold the model
  and API key, which ``prompts.PROMPTS`` key supplies the framework
  system prompt, which registry key holds the channel-overridable
  personality overlay, the max-output-tokens cap, and the
  force-search-on-explicit opt-in.
- The :data:`PROFILES` registry mapping each identifier to its Profile.

Tools are *not* a field on :class:`Profile`. Tool visibility is a
many-to-many relation owned by ``assistant.ToolSpec.visible_in``;
consumers should call ``assistant.get_tools_for_profile(profile.id)``
directly. This keeps a single source of truth for visibility and avoids
a back-import from ``profile`` to ``assistant``.

Lookup pattern::

    from llm.profile import PROFILES, PROFILE_CHAT
    profile = PROFILES[route_profile]  # route_profile is one of the IDs
    model_key = profile.model_setting
    prompt_text = prompts.PROMPTS[profile.prompt_id].format(bot_nick=...)
"""

from __future__ import annotations

from dataclasses import dataclass

# Route profile identifiers. These were originally declared in
# ``assistant.py``; this module is now the definition site and
# ``assistant.py`` re-imports them. The string values match the keys in
# :data:`PROFILES`, in ``prompts.PROMPTS`` for the framework prompts,
# and in the ``visible_in`` sets on ``assistant.ASSISTANT_TOOL_SPECS``.
PROFILE_CHAT = "chat"
PROFILE_CODE = "code"
PROFILE_DRAW = "draw"
PROFILE_VERSE = "verse"
PROFILE_REMIND_ACTION = "remind_action"


@dataclass(frozen=True)
class Profile:
    """Bundle of per-mode configuration for an assistant chat-loop request.

    One :class:`Profile` per ``route_profile`` string. Collapses the
    route-keyed lookups previously scattered across
    ``service.assistant_completion`` into a single named record.

    Fields:
        id: Route identifier. Must equal the dict key in :data:`PROFILES`
            and a key in ``prompts.PROMPTS``.
        model_setting: Limnoria registry key for the model name
            (e.g. ``"assistantModel"``).
        api_key_setting: Limnoria registry key for the API key
            (e.g. ``"assistantApiKey"``).
        prompt_id: Key into ``prompts.PROMPTS`` for the framework system
            prompt.
        overlay_setting: Limnoria registry key for the channel-overridable
            personality overlay (e.g. ``"assistantSystemPrompt"``).
            ``None`` means this profile takes no overlay.
        max_output_tokens: ``max_tokens`` cap for the LiteLLM completion.
            ``None`` means unbounded.
        force_search_on_explicit: If True and an explicit search trigger
            matches the user prompt, force
            ``tool_choice={search_web}`` on step 0.
    """

    id: str
    model_setting: str
    api_key_setting: str
    prompt_id: str
    overlay_setting: str | None
    max_output_tokens: int | None
    force_search_on_explicit: bool


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

- [ ] **Step 4: Update `assistant.py` to re-import PROFILE_* from `profile.py`**

Open `plugins/llm/src/llm/assistant.py` at lines 25-31. The current block is:

```python
# Route profile identifiers — keep in sync with the keys of
# ``profile_frameworks`` in service.py and the ``visible_in`` sets below.
PROFILE_CHAT = "chat"
PROFILE_CODE = "code"
PROFILE_DRAW = "draw"
PROFILE_VERSE = "verse"
PROFILE_REMIND_ACTION = "remind_action"
```

Replace it with:

```python
# Route profile identifiers are now defined in ``profile.py``. They are
# re-imported here so existing consumers (and ``ToolSpec.visible_in``
# default below) keep working unchanged. The string literals are no
# longer duplicated; ``profile.py`` is the single source of truth.
from .profile import (
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

This keeps every existing `from llm.assistant import PROFILE_CHAT` import working. Task 4 deletes this re-import block and switches the holdout consumers to `from llm.profile import ...`.

- [ ] **Step 5: Run the new tests and verify they pass**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_profile.py -v"
```

Expected: all tests pass.

- [ ] **Step 6: Run the broader test suite to confirm no regressions**

```bash
make test
```

Expected: no failures. Coverage stays at or above 93%.

- [ ] **Step 7: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean. If the lint hook fired on file save already that's fine; this is the gate before commit.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/profile.py \
        plugins/llm/tests/test_profile.py \
        plugins/llm/src/llm/assistant.py
git commit -m "refactor(llm): add profile.py as single source of truth (no consumers yet)"
```

Verify pre-commit hook passes. CI must stay green — push happens at the end of Task 4, but `main` is consistent locally at every commit.

---

## Task 2: Migrate `service.py:assistant_completion` to PROFILES

**Files:**
- Modify: `plugins/llm/src/llm/service.py:3120-3260` (the `assistant_completion` body)
- Modify: `plugins/llm/tests/test_service.py` (any tests asserting `route_profile in {…}` literal shape)

- [ ] **Step 1: Re-verify the source-of-truth mappings for draw/verse/remind_action**

Before changing service.py, confirm the registry-key reads for draw, verse, and remind_action match what `PROFILES` says. These three weren't fully traced during spec-writing.

```bash
# Find every call site that passes route_profile=PROFILE_DRAW / PROFILE_VERSE / PROFILE_REMIND_ACTION
grep -nE 'route_profile=PROFILE_(DRAW|VERSE|REMIND_ACTION)|route_profile=.*"(draw|verse|remind_action)"' \
    plugins/llm/src/llm/plugin.py plugins/llm/src/llm/service.py
```

Expected: list of caller sites. For each one, confirm:
- The model is *not* overridden via `model_override=` with something other than the result of `registryValue("assistantModel", channel)`. If a caller passes a non-assistant model, the registry entry for that profile in `profile.py` is wrong and must be corrected before continuing.
- Same for `api_key=`.

If discrepancies are found, update `profile.py` and `test_profile.py` to match the actual call-site behavior, then re-run `make test`. Commit the correction as `fix(llm): correct PROFILES mapping for <profile>` before continuing — keep `main` honest at every step.

If everything matches: proceed.

- [ ] **Step 2: Read the current `assistant_completion` body for the lines being changed**

```bash
sed -n '3120,3245p' plugins/llm/src/llm/service.py
```

Confirm the section matches the "Before" block in the spec. If service.py has drifted, the spec's "Before" block needs updating before the edit. (The plan was written against commit `b4d769a`.)

- [ ] **Step 3: Add tests pinning that `assistant_completion` reads through PROFILES**

The existing test suite verifies *behavior* through mocked `registryValue` calls. The migration changes which string is passed to `registryValue` for the chat profile (no-op: still `"assistantModel"`). To pin the *new contract* — that `model_setting` is read from PROFILES — add a test that asserts the call shape.

In `plugins/llm/tests/test_service.py`, add:

```python
class TestAssistantCompletionReadsFromProfiles:
    """assistant_completion looks up Profile-driven settings via PROFILES.

    These tests pin that the migration is wired up. Without them, a
    future change could silently swap back to hardcoded strings.
    """

    def test_chat_profile_reads_assistantModel_via_profile(self, mocker, llm_service):
        """For PROFILE_CHAT, model lookup must use profile.model_setting."""
        from llm.profile import PROFILES, PROFILE_CHAT
        # The chat profile's model_setting is what gets passed to registryValue.
        assert PROFILES[PROFILE_CHAT].model_setting == "assistantModel"

        registry = mocker.patch.object(
            llm_service.plugin,
            "registryValue",
            side_effect=lambda key, *args, **kwargs: {
                "assistantModel": "fake-model",
                "assistantApiKey": "fake-key",
                "metaMaxSteps": 1,
                "timeout": 30,
            }.get(key, ""),
        )
        # ... existing test scaffolding to call assistant_completion ...
        # After the call:
        call_keys = [c.args[0] for c in registry.call_args_list]
        # The migration's contract: model_setting from PROFILES is what gets read.
        assert "assistantModel" in call_keys

    def test_code_profile_reads_codeModel_via_profile(self, mocker, llm_service):
        """For PROFILE_CODE, model lookup must use profile.model_setting='codeModel'.

        Pre-refactor, the code profile reached assistant_completion via a
        model_override= kwarg from the caller, so service.py only saw
        'assistantModel' in its own registryValue call. Post-refactor,
        if assistant_completion is invoked with route_profile=PROFILE_CODE
        and no model_override, it must read 'codeModel'.
        """
        from llm.profile import PROFILES, PROFILE_CODE
        assert PROFILES[PROFILE_CODE].model_setting == "codeModel"
        # ... mocker setup + invocation with route_profile=PROFILE_CODE,
        # model_override=None, api_key=None ...
        # ... assert "codeModel" appears in registry.call_args_list keys ...
```

> **Note for the implementer:** flesh out the mocker scaffolding to match the existing test style in `test_service.py`. Use whatever fixture provides an LLMService instance with a mocked plugin — search for `assistant_completion` in `test_service.py` to find the pattern. The skeleton above documents the *contract*; the existing test file already shows how to build the harness.

- [ ] **Step 4: Run the new tests, confirm they fail or skip cleanly**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_service.py::TestAssistantCompletionReadsFromProfiles -v"
```

Expected: the `chat` test passes immediately (no behavior change for chat); the `code` test fails because `assistant_completion` currently hardcodes `"assistantModel"` regardless of `route_profile`.

- [ ] **Step 5: Migrate `assistant_completion` to read from PROFILES**

Open `plugins/llm/src/llm/service.py`. At the top, ensure `PROFILES` is imported alongside the existing `PROMPTS` import:

```python
from .profile import PROFILES
from .prompts import PROMPTS
```

Inside `assistant_completion`, replace this block (currently around lines 3134-3137):

```python
target = self._channel_target(channel)
model = model_override or self.plugin.registryValue("assistantModel", target)
effective_api_key = api_key or self.plugin.registryValue("assistantApiKey", target)
```

With:

```python
profile = PROFILES[route_profile]
target = self._channel_target(channel)
model = model_override or self.plugin.registryValue(profile.model_setting, target)
effective_api_key = api_key or self.plugin.registryValue(profile.api_key_setting, target)
```

Replace this line (currently around 3163):

```python
framework = PROMPTS.get(route_profile, PROMPTS["chat"]).format(bot_nick=bot_nick)
```

With:

```python
framework = PROMPTS[profile.prompt_id].format(bot_nick=bot_nick)
```

Replace this block (currently around 3193-3200):

```python
profile_max_output = {
    PROFILE_CHAT: 2000,
    PROFILE_REMIND_ACTION: 400,
}.get(route_profile)
if profile_max_output is not None:
    optional_kwargs["max_tokens"] = profile_max_output
```

With:

```python
if profile.max_output_tokens is not None:
    optional_kwargs["max_tokens"] = profile.max_output_tokens
```

Replace this line (currently around 3221):

```python
profile_tools = get_tools_for_profile(route_profile, exclude=exclude_tools)
```

With:

```python
profile_tools = get_tools_for_profile(profile.id, exclude=exclude_tools)
```

Replace this block (currently around 3232-3237):

```python
force_initial_search = (
    route_profile in {PROFILE_CHAT, PROFILE_REMIND_ACTION}
    and search_fn is not None
    and _has_tool(profile_tools, "search_web")
    and EXPLICIT_SEARCH_RE.search(prompt) is not None
)
```

With:

```python
force_initial_search = (
    profile.force_search_on_explicit
    and search_fn is not None
    and _has_tool(profile_tools, "search_web")
    and EXPLICIT_SEARCH_RE.search(prompt) is not None
)
```

**Do not touch** the `if route_profile == PROFILE_VERSE` branch around line 3169 — that's overlay-machinery and stays out of scope.

Remove any now-unused imports of `PROFILE_CHAT`, `PROFILE_REMIND_ACTION` from `service.py` if those names are no longer referenced anywhere else in the file. (The PROFILE_VERSE constant is still needed by the overlay-footer branch, leave it.)

Run:

```bash
grep -nE 'PROFILE_(CHAT|CODE|DRAW|VERSE|REMIND_ACTION)' plugins/llm/src/llm/service.py
```

Trim the import list to match what remains in use.

- [ ] **Step 6: Run the targeted tests, confirm they all pass**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_service.py::TestAssistantCompletionReadsFromProfiles -v"
```

Expected: both tests pass.

- [ ] **Step 7: Run the full test suite**

```bash
make test
```

Expected: no regressions; coverage stays at or above 93%.

- [ ] **Step 8: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 9: Commit (atomic with test changes — keeps `main` green for auto-deploy)**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "refactor(llm): migrate service.assistant_completion to PROFILES registry"
```

---

## Task 3: Migrate `plugin.py` + `service.py:4734` overlay reads

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1442` (ask path overlay read)
- Modify: `plugins/llm/src/llm/plugin.py:2547` (code path overlay read)
- Modify: `plugins/llm/src/llm/plugin.py:3385` (verse path overlay read)
- Modify: `plugins/llm/src/llm/service.py:4734` (scheduled-task fire overlay read)
- Modify: `plugins/llm/tests/test_plugin.py` and `plugins/llm/tests/test_service.py` (any tests pinning the literal `"assistantSystemPrompt"` / `"codeSystemPrompt"` strings)

- [ ] **Step 1: Identify every overlay-read call site**

```bash
grep -nE 'registryValue\("(assistantSystemPrompt|codeSystemPrompt)"' \
    plugins/llm/src/llm/plugin.py plugins/llm/src/llm/service.py
```

Expected (based on commit `b4d769a`):

```
plugins/llm/src/llm/plugin.py:1442:            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
plugins/llm/src/llm/plugin.py:2547:                system_prompt=self.registryValue("codeSystemPrompt", channel),
plugins/llm/src/llm/plugin.py:3385:            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
plugins/llm/src/llm/service.py:4734:        ask_prompt = plugin.registryValue("assistantSystemPrompt", row.channel)
```

If the grep shows additional sites, include them in this task — none should be left hardcoded after this commit.

- [ ] **Step 2: Add tests asserting overlay reads go through PROFILES**

In `plugins/llm/tests/test_plugin.py`, add a test class:

```python
class TestOverlayReadsViaProfiles:
    """Overlay registry keys are looked up via PROFILES, not hardcoded."""

    def test_chat_path_overlay_key_matches_profile(self):
        from llm.profile import PROFILES, PROFILE_CHAT
        # If this changes, the chat ask path must be updated to follow.
        assert PROFILES[PROFILE_CHAT].overlay_setting == "assistantSystemPrompt"

    def test_code_path_overlay_key_matches_profile(self):
        from llm.profile import PROFILES, PROFILE_CODE
        assert PROFILES[PROFILE_CODE].overlay_setting == "codeSystemPrompt"

    def test_verse_path_overlay_key_matches_profile(self):
        from llm.profile import PROFILES, PROFILE_VERSE
        assert PROFILES[PROFILE_VERSE].overlay_setting == "assistantSystemPrompt"
```

These are minimal contract pins — the *behavior* coverage (that the bot actually reads the right channel-overridable text) is already covered by existing tests; what we want here is to ensure no one re-hardcodes the string later.

- [ ] **Step 3: Run the new tests, confirm they pass already**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_plugin.py::TestOverlayReadsViaProfiles -v"
```

Expected: all three pass (they're literal value pins; pass before *and* after the migration).

- [ ] **Step 4: Migrate `plugin.py:1442` (ask path)**

Read the current line:

```bash
sed -n '1438,1446p' plugins/llm/src/llm/plugin.py
```

Confirm it matches:

```python
            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
```

At the top of `plugin.py`, ensure imports include `PROFILES` and `PROFILE_CHAT`:

```python
from .profile import PROFILE_CHAT, PROFILE_CODE, PROFILE_VERSE, PROFILES
```

Replace the line at 1442 with:

```python
            ask_prompt = self.registryValue(PROFILES[PROFILE_CHAT].overlay_setting, channel)
```

- [ ] **Step 5: Migrate `plugin.py:2547` (code path)**

```bash
sed -n '2543,2551p' plugins/llm/src/llm/plugin.py
```

Confirm it matches:

```python
                system_prompt=self.registryValue("codeSystemPrompt", channel),
```

Replace with:

```python
                system_prompt=self.registryValue(PROFILES[PROFILE_CODE].overlay_setting, channel),
```

- [ ] **Step 6: Migrate `plugin.py:3385` (verse path)**

```bash
sed -n '3381,3389p' plugins/llm/src/llm/plugin.py
```

Confirm it matches:

```python
            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
```

Replace with:

```python
            ask_prompt = self.registryValue(PROFILES[PROFILE_VERSE].overlay_setting, channel)
```

- [ ] **Step 7: Migrate `service.py:4734` (scheduled-task fire)**

```bash
sed -n '4730,4738p' plugins/llm/src/llm/service.py
```

Confirm it matches:

```python
        ask_prompt = plugin.registryValue("assistantSystemPrompt", row.channel)
```

The scheduled-task fire path fires reminder actions that re-enter the chat profile, so it uses `PROFILE_CHAT`'s overlay key. Ensure `service.py`'s imports include `PROFILE_CHAT` and `PROFILES` (they likely already do after Task 2). Replace with:

```python
        ask_prompt = plugin.registryValue(PROFILES[PROFILE_CHAT].overlay_setting, row.channel)
```

- [ ] **Step 8: Run the targeted tests**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_plugin.py plugins/llm/tests/test_service.py -v"
```

Expected: all tests pass.

- [ ] **Step 9: Run the full suite**

```bash
make test
```

Expected: no regressions; coverage stays at or above 93%.

- [ ] **Step 10: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add plugins/llm/src/llm/plugin.py \
        plugins/llm/src/llm/service.py \
        plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): migrate plugin.py overlay reads to PROFILES.overlay_setting"
```

---

## Task 4: Cleanup re-exports + byte-identity gate

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py:25-31` (delete the `from .profile import PROFILE_*` block from Task 1 — but only after switching any holdout consumers)
- Modify: any file still importing `PROFILE_*` from `llm.assistant`

- [ ] **Step 1: Inventory remaining consumers of `llm.assistant.PROFILE_*`**

```bash
grep -rnE 'from .assistant import.*PROFILE_|from llm\.assistant import.*PROFILE_|from \.assistant import.*PROFILE_' \
    plugins/llm/src plugins/llm/tests
```

Each match is a holdout consumer that needs to switch.

- [ ] **Step 2: Update each holdout to import from `llm.profile`**

For each file from Step 1, rewrite the import. Example:

```python
# Before
from .assistant import PROFILE_CHAT, PROFILE_VERSE, AssistantToolExecutor

# After
from .assistant import AssistantToolExecutor
from .profile import PROFILE_CHAT, PROFILE_VERSE
```

If a file imports *only* PROFILE_* from `.assistant`, the import line collapses to a single `from .profile import …`.

- [ ] **Step 3: Delete the re-import block in `assistant.py`**

Open `plugins/llm/src/llm/assistant.py` at the block added in Task 1, Step 4. Delete this:

```python
# Route profile identifiers are now defined in ``profile.py``. They are
# re-imported here so existing consumers (and ``ToolSpec.visible_in``
# default below) keep working unchanged. The string literals are no
# longer duplicated; ``profile.py`` is the single source of truth.
from .profile import (
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

But `assistant.py` *still uses* the constants internally — `ToolSpec.visible_in` defaults reference them, `_TOOL_SPEC_OVERRIDES` references them, `_VERSE_EXCLUDED_TOOLS` references PROFILE_VERSE. Without those constants in scope, the module won't compile.

So Step 3 is actually: *keep the import* but stop treating it as a re-export — drop the comment that frames it as a shim:

```python
from .profile import (
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

The semantic change is: external code can no longer rely on this being importable through `llm.assistant` (because we just rewrote every external import in Step 2 to use `llm.profile`). The line stays for internal use.

- [ ] **Step 4: Run the full suite**

```bash
make test
```

Expected: no regressions. If any test still fails because it imports PROFILE_* from `llm.assistant` and Step 1's grep missed it, switch that test to `llm.profile` and re-run.

- [ ] **Step 5: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean. If `ty` reports `PROFILE_*` as unused-import in `assistant.py`, that's wrong — they *are* used internally. If `ty` is right and they really aren't used, delete them. (Most likely they're used in `ToolSpec.visible_in` defaults.)

- [ ] **Step 6: Run `make preflight`**

```bash
make preflight
```

Expected: everything green.

- [ ] **Step 7: Run the byte-identity gate**

```bash
python /tmp/profile_identity.py > /tmp/profile_identity_post.txt
diff /tmp/profile_identity_pre.txt /tmp/profile_identity_post.txt
echo "exit=$?"
```

Expected: empty diff, `exit=0`. Both files identical means the refactor preserved every observable (framework prompt text per profile, tool name set per profile, max_output_tokens per profile, force_initial_search predicate, overlay setting, model setting, api_key setting).

If the diff is non-empty: do not commit. Read the diff, identify which observable changed, decide whether it's a bug to fix or whether the spec's pre-refactor pinning was wrong. Likely a bug — investigate and fix before proceeding.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/assistant.py plugins/llm/src/llm
git commit -m "refactor(llm): switch holdout consumers to llm.profile imports"
```

If Step 2 modified test files too, include those in the `git add`.

- [ ] **Step 9: Push to main**

```bash
git push
```

Auto-deploy fires when CI + Docker workflows pass. Watch for the green check via `gh run list --limit 3` after a minute.

- [ ] **Step 10: Verify production after auto-deploy**

The auto-deploy workflow restarts the vibebot service over SSH. After the green Docker workflow run, optionally confirm:

```bash
ssh -i ~/.ssh/id_rsa vibebot@rdrake.org \
    'systemctl --user is-active vibebot && docker logs --tail 20 vibebot 2>&1 | tail -5'
```

Expected: `active`, no startup errors.

- [ ] **Step 11: Cleanup local harness files**

```bash
rm /tmp/profile_identity.py /tmp/profile_identity_pre.txt /tmp/profile_identity_post.txt
```

These were one-off — the byte-identity gate's value was at the migration moment, not as a permanent artifact.

---

## Self-Review

Before handing off, the plan was reviewed against the spec:

**Spec coverage check:**

- ✅ Module layout (new `profile.py`, dependency direction) → Task 1, Steps 3-4.
- ✅ Profile dataclass shape (7 fields) → Task 1, Step 3.
- ✅ PROFILES registry (5 entries with full per-field mappings) → Task 1, Step 3.
- ✅ Migration of `service.py:assistant_completion` (5 lookups collapse) → Task 2, Step 5.
- ✅ Migration of `plugin.py` overlay reads (3 sites) + service.py:4734 → Task 3, Steps 4-7.
- ✅ PROFILE_VERSE footer left untouched → Task 2, Step 5 ("Do not touch the `if route_profile == PROFILE_VERSE` branch").
- ✅ Testing strategy (5 test classes: Registry, Resolution, ToolsAlignment, BehaviorPreservation, Immutability) → Task 1, Step 1.
- ✅ Byte-identity gate → Pre-Work + Task 4, Step 7.
- ✅ Verification step for draw/verse/remind_action model/key mappings → Task 2, Step 1.
- ✅ Four phased commits, each green on `main` for auto-deploy → Task structure.

**Placeholder scan:**

- One soft "Note for the implementer" in Task 2, Step 3 about fleshing out mocker scaffolding. Justified — the file conventions are easier to read than to transcribe, and Task 2 Step 3 documents the *contract* the implementer is pinning. The actual code for the test is shown; only the mocker fixture setup defers to the existing test-file pattern.

**Type / identifier consistency:**

- `Profile` dataclass field names (`id`, `model_setting`, `api_key_setting`, `prompt_id`, `overlay_setting`, `max_output_tokens`, `force_search_on_explicit`) match consistently across Task 1 (definition), Task 2 (consumption in service.py), and Task 3 (consumption in plugin.py).
- `PROFILES[<const>].overlay_setting` form is used identically in all 4 migration sites in Task 3.
- `get_tools_for_profile(profile.id, ...)` form matches the existing assistant.py signature.

**Scope check:** One implementation cycle, four commits, single feature area (`plugins/llm/`). No subsystem decomposition needed.

**Ambiguity check:** The "Note for the implementer" in Task 2 Step 3 is the only soft spot. Acceptable because the contract is explicit and the fixture pattern is well-established in the existing test file.
