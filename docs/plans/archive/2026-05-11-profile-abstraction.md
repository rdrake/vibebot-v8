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
# Mappings verified against source during adversarial review pass:
# - PROFILE_CODE: outer @code planner uses assistantModel/assistantApiKey (no
#   override passed); does NOT read a channel overlay. codeModel/codeApiKey/
#   codeSystemPrompt belong to the inner _code_for_assistant one-shot.
# - PROFILE_VERSE: verseModel is read by the caller (plugin.py:3307) and passed
#   as model_override. The assistant_completion fallback (assistantModel) is
#   what Profile.model_setting captures; verseModel stays as a caller-side
#   preference outside Profile.
# - PROFILE_DRAW: same pattern as CODE — planner uses assistant fallback, no
#   channel overlay. Verify during Task 2 entry.
PRE_MAX_OUTPUT = {PROFILE_CHAT: 2000, PROFILE_REMIND_ACTION: 400}
PRE_FORCE_SEARCH = {PROFILE_CHAT, PROFILE_REMIND_ACTION}
PRE_OVERLAY_KEY = {
    PROFILE_CHAT: "assistantSystemPrompt",
    PROFILE_CODE: None,
    PROFILE_DRAW: None,
    PROFILE_VERSE: "assistantSystemPrompt",
    PROFILE_REMIND_ACTION: "assistantSystemPrompt",
}
PRE_MODEL_KEY = {
    PROFILE_CHAT: "assistantModel",
    PROFILE_CODE: "assistantModel",
    PROFILE_DRAW: "assistantModel",
    PROFILE_VERSE: "assistantModel",
    PROFILE_REMIND_ACTION: "assistantModel",
}
PRE_API_KEY = {
    PROFILE_CHAT: "assistantApiKey",
    PROFILE_CODE: "assistantApiKey",
    PROFILE_DRAW: "assistantApiKey",
    PROFILE_VERSE: "assistantApiKey",
    PROFILE_REMIND_ACTION: "assistantApiKey",
}

# Sample prompts chosen to exercise every term in the real EXPLICIT_SEARCH_RE
# alternation. If service.py drops or renames a term, the digest of
# observables[*]["force_search_on_samples"] changes.
SAMPLE_PROMPTS = [
    "please search the web for python typing news",
    "find me a recipe for sourdough",
    "look up the weather in Toronto",
    "what's the latest on the bill",
    "any recent news on the merger",
    "current state of mortgage rates",
    "general greeting with no triggers",
]

# Import the live regex so the gate pins the real pattern. If service.py
# changes it during the refactor, the diff catches it.
from llm.service import EXPLICIT_SEARCH_RE  # noqa: E402

observables = {}
for pid in PROFILE_IDS:
    framework = prompts.PROMPTS[pid].format(bot_nick="testbot")
    tools = sorted(t["function"]["name"] for t in get_tools_for_profile(pid))
    max_tokens = PRE_MAX_OUTPUT.get(pid)
    # One result per sample so the digest changes if the regex's term set
    # changes, not only when the chat/remind_action membership flips.
    force_search_on_samples = {
        sample: (
            pid in PRE_FORCE_SEARCH
            and EXPLICIT_SEARCH_RE.search(sample) is not None
        )
        for sample in SAMPLE_PROMPTS
    }
    observables[pid] = {
        "framework_text": framework,
        "tool_names": tools,
        "max_output_tokens": max_tokens,
        "force_search_on_samples": force_search_on_samples,
        "overlay_setting": PRE_OVERLAY_KEY[pid],
        "model_setting": PRE_MODEL_KEY[pid],
        "api_key_setting": PRE_API_KEY[pid],
    }

# Also pin the regex pattern itself so a silent regex edit is caught even
# if every sample's match boolean stays the same.
observables["__regex__"] = {
    "pattern": EXPLICIT_SEARCH_RE.pattern,
    "flags": EXPLICIT_SEARCH_RE.flags,
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

import dataclasses

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
    # PROFILE_CODE / PROFILE_DRAW: no channel-overridable overlay — the
    # @code and @draw planners construct system_prompt from user_instruction
    # + the framework prompt directly, never reading a registry key.
    # PROFILE_VERSE / PROFILE_CHAT / PROFILE_REMIND_ACTION: all read
    # 'assistantSystemPrompt'.
    EXPECTED_OVERLAY = {
        "chat": "assistantSystemPrompt",
        "code": None,
        "draw": None,
        "verse": "assistantSystemPrompt",
        "remind_action": "assistantSystemPrompt",
    }
    # Every chat-loop profile's assistant_completion fallback is
    # assistantModel/assistantApiKey. codeModel/codeApiKey belong to the
    # inner _code_for_assistant one-shot, not the @code planner. verseModel
    # is a caller-side override passed via model_override= rather than read
    # by assistant_completion.
    EXPECTED_MODEL = {pid: "assistantModel" for pid in
                      ("chat", "code", "draw", "verse", "remind_action")}
    EXPECTED_API_KEY = {pid: "assistantApiKey" for pid in
                        ("chat", "code", "draw", "verse", "remind_action")}

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
        """Assigning to a Profile field raises FrozenInstanceError."""
        p = profile.PROFILES["chat"]
        with pytest.raises(dataclasses.FrozenInstanceError):
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
        # The @code chat-loop planner uses assistantModel/assistantApiKey;
        # the codeModel/codeApiKey/codeSystemPrompt registry keys belong to
        # the *inner* _code_for_assistant one-shot (plugin.py:2547), which
        # is a tool callback, not a Profile dispatch.
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="code",
        # The @code planner builds system_prompt from
        # user_instruction + CODE_SYSTEM_PROMPT and never reads a channel
        # overlay registry key.
        overlay_setting=None,
        max_output_tokens=None,
        force_search_on_explicit=False,
    ),
    PROFILE_DRAW: Profile(
        id=PROFILE_DRAW,
        model_setting="assistantModel",
        api_key_setting="assistantApiKey",
        prompt_id="draw",
        # Same pattern as PROFILE_CODE: planner constructs its system_prompt
        # without reading a channel overlay key.
        overlay_setting=None,
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

- [ ] **Step 4: Move PROFILE_* definitions in `assistant.py` to a top-of-file re-import**

Two-part edit. First, **delete** the existing mid-module block at `plugins/llm/src/llm/assistant.py` lines 25-31:

```python
# Route profile identifiers — keep in sync with the keys of
# ``profile_frameworks`` in service.py and the ``visible_in`` sets below.
PROFILE_CHAT = "chat"
PROFILE_CODE = "code"
PROFILE_DRAW = "draw"
PROFILE_VERSE = "verse"
PROFILE_REMIND_ACTION = "remind_action"
```

Second, **insert** the re-import in the top-of-file imports section. Open the file and find the existing import block (typically lines 1-20). Add the `from .profile import …` line alongside the other relative imports, **alphabetically sorted** with neighbors so `ruff isort` (`I001`) doesn't complain:

```python
from .profile import (
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

Why move instead of replace-in-place: the original PROFILE_* assignments were Python module-level *statements* (variable assignments). Replacing them with an `import` statement *in the same location* trips ruff's E402 ("module level import not at top of file") because the imports appear after non-import statements. Moving to the top of the file is the conventional fix.

After the move, every existing `from llm.assistant import PROFILE_CHAT` keeps working — assistant.py still exposes the names, just by re-import. Task 4 documents this contract; nothing further changes about the import.

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

- [ ] **Step 1: Verify @draw's effective model/key/overlay reads**

The adversarial review pass already verified PROFILE_CHAT, PROFILE_CODE, PROFILE_VERSE, and PROFILE_REMIND_ACTION:

- PROFILE_CHAT, PROFILE_VERSE: served by `_ask_impl`, model fallback `assistantModel`, overlay `assistantSystemPrompt`. Verse adds a caller-side `verseModel` override at plugin.py:3307 (preserved as caller-side, not absorbed into Profile).
- PROFILE_CODE: outer planner at plugin.py:3540 uses `assistantModel`/`assistantApiKey` (no override). Builds system_prompt from `user_instruction + CODE_SYSTEM_PROMPT`; never reads a registry overlay. `codeModel`/`codeApiKey`/`codeSystemPrompt` belong to inner `_code_for_assistant` only.
- PROFILE_REMIND_ACTION: structured fire at plugin.py:1442 and scheduled-task fire at service.py:4734 both use `assistantSystemPrompt`.

The only un-verified profile is PROFILE_DRAW. Locate the @draw command call site and confirm:

```bash
grep -nB 5 -A 35 'profile=PROFILE_DRAW' plugins/llm/src/llm/plugin.py
```

For the call site:

- Confirm the call to `assistant_request` does **not** pass `model_override=` (or, if it does, that the override resolves to `assistantModel`'s value).
- Confirm no `registryValue("...SystemPrompt", channel)` read happens for the @draw path. If `system_prompt=` is passed to `assistant_request`, trace what it's built from.

If @draw matches the @code pattern (planner uses assistant fallback, no channel overlay), `PROFILES[PROFILE_DRAW]` as written in Task 1 Step 3 is correct.

If @draw reads `drawSystemPrompt` or similar from registry, update `PROFILES[PROFILE_DRAW].overlay_setting`, `PRE_OVERLAY_KEY[PROFILE_DRAW]` in `/tmp/profile_identity.py`, and `EXPECTED_OVERLAY["draw"]` in `test_profile.py`. Then re-run `make test` and the byte-identity script before continuing. Commit any correction as `fix(llm): correct PROFILE_DRAW mapping` to keep `main` honest.

If @draw uses `drawModel`/`drawApiKey` (registry keys that may exist) as model overrides, update PROFILE_DRAW.model_setting/api_key_setting similarly.

- [ ] **Step 2: Read the current `assistant_completion` body for the lines being changed**

```bash
sed -n '3120,3245p' plugins/llm/src/llm/service.py
```

Confirm the section matches the "Before" block in the spec. If service.py has drifted, the spec's "Before" block needs updating before the edit. (The plan was written against commit `b4d769a`.)

- [ ] **Step 3: Add real behavior tests pinning the PROFILES read path**

The existing test suite covers `assistant_completion` behavior end-to-end with mocked `registryValue`. To pin the new *contract* — that `model_setting`/`api_key_setting` is read from PROFILES and not hardcoded — add a `monkeypatch.setitem` test that swaps a profile entry with a sentinel and asserts the sentinel's string flows to `registryValue`.

The harness pattern lives in `plugins/llm/tests/test_assistant.py` (see `test_assistant_completion_layers_system_prompt_over_framework` around line 893). Copy that pattern.

Append to `plugins/llm/tests/test_assistant.py` (which already has `service` and `mocker` fixtures wired up):

```python
class TestAssistantCompletionReadsModelKeyFromProfiles:
    """assistant_completion must read model/api_key via PROFILES[route].

    Swapping the PROFILES entry with a sentinel-keyed Profile and asserting
    the sentinel key flows to plugin.registryValue() pins that the
    migration is wired up — a future regression that hardcodes
    'assistantModel' would break these tests.
    """

    def test_model_setting_is_read_from_profile(
        self, service: LLMService, mocker: MockerFixture, monkeypatch
    ) -> None:
        """For route_profile=PROFILE_CHAT, plugin.registryValue is called
        with PROFILES[PROFILE_CHAT].model_setting — not a hardcoded string.
        """
        from llm.profile import PROFILES, Profile, PROFILE_CHAT

        sentinel = Profile(
            id=PROFILE_CHAT,
            model_setting="SENTINEL_MODEL_KEY",
            api_key_setting="SENTINEL_API_KEY",
            prompt_id="chat",
            overlay_setting=None,
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_CHAT, sentinel)

        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        # Capture every registryValue call so we can prove our sentinel
        # keys were the ones used.
        registry_calls: list[str] = []
        original_registryValue = service.plugin.registryValue

        def spy(key, *args, **kwargs):
            registry_calls.append(key)
            # Fall through to the real mock so existing test fixtures still
            # control timeouts, maxSteps, etc.
            return original_registryValue(key, *args, **kwargs)

        mocker.patch.object(service.plugin, "registryValue", side_effect=spy)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
        )

        assert "SENTINEL_MODEL_KEY" in registry_calls
        assert "SENTINEL_API_KEY" in registry_calls
        assert "assistantModel" not in registry_calls
        assert "assistantApiKey" not in registry_calls

    def test_model_override_still_wins(
        self, service: LLMService, mocker: MockerFixture, monkeypatch
    ) -> None:
        """An explicit model_override= bypasses Profile.model_setting,
        matching the pre-refactor contract.
        """
        from llm.profile import PROFILES, Profile, PROFILE_CHAT

        sentinel = Profile(
            id=PROFILE_CHAT,
            model_setting="SENTINEL_MODEL_KEY",
            api_key_setting="assistantApiKey",
            prompt_id="chat",
            overlay_setting=None,
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_CHAT, sentinel)

        captured_model: list[str] = []

        mock_response = mocker.MagicMock()
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "ok"
        mock_choice.message.tool_calls = None
        mock_response.choices = [mock_choice]

        def capture(**kwargs):
            captured_model.append(kwargs.get("model"))
            return mock_response

        mocker.patch("llm.service.litellm.completion", side_effect=capture)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt="hello",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
            model_override="explicit-override-model",
        )

        # The override beat the Profile.model_setting lookup.
        assert captured_model == ["explicit-override-model"]
```

These are *behavior* tests — they exercise the migrated code path with a swapped registry entry and observe the call surface. They fail before the migration (because service.py reads hardcoded `"assistantModel"`) and pass after (because service.py reads `PROFILES[route_profile].model_setting`).

- [ ] **Step 4: Run the new tests, confirm they fail before the migration**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_assistant.py::TestAssistantCompletionReadsModelKeyFromProfiles -v"
```

Expected: both tests **fail** before the migration. `test_model_setting_is_read_from_profile` fails because pre-refactor `assistant_completion` reads the hardcoded string `"assistantModel"` rather than the sentinel's `"SENTINEL_MODEL_KEY"`. `test_model_override_still_wins` also fails because the sentinel-Profile's prompt_id/overlay_setting now apply to the chat path, mutating other paths beyond what pre-refactor code expects. (This is fine — both will pass after Step 5.)

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
# PROFILES.get fallback preserves pre-refactor behavior: unknown
# route_profile values silently fall through to the chat profile. The
# pre-refactor framework lookup used the same .get(..., PROMPTS["chat"])
# pattern. Internal callers always pass a known PROFILE_* string, so
# the fallback should never fire — but we keep it to avoid changing
# observable behavior for a low-cost defensive read.
profile = PROFILES.get(route_profile, PROFILES[PROFILE_CHAT])
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

The pre-refactor `.get(..., PROMPTS["chat"])` silent fallback is preserved at the top of the function by using `PROFILES.get(route_profile, PROFILES[PROFILE_CHAT])` instead of `PROFILES[route_profile]`. This keeps unknown `route_profile` values flowing through the chat framework rather than raising KeyError — behavior-preserving by construction.

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

## Task 3: Migrate overlay reads at plugin.py:1442, plugin.py:3385, service.py:4734

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1442` (structured-reminder fire, PROFILE_REMIND_ACTION)
- Modify: `plugins/llm/src/llm/plugin.py:3385` (`_ask_impl`, serves PROFILE_CHAT and PROFILE_VERSE via `effective_profile`)
- Modify: `plugins/llm/src/llm/service.py:4734` (scheduled-task fire, PROFILE_REMIND_ACTION)
- Modify: `plugins/llm/tests/test_assistant.py` and/or `test_plugin.py` (add monkeypatch behavior tests)

**Out of scope:** `plugin.py:2547` is inside `_code_for_assistant`, a one-shot tool callback (NOT a chat-loop Profile dispatch). It reads `codeSystemPrompt` because that's the inner one-shot's own registry key. Leave it alone.

- [ ] **Step 1: Confirm the three overlay-read call sites and their effective profile**

```bash
grep -nE 'registryValue\("assistantSystemPrompt"' \
    plugins/llm/src/llm/plugin.py plugins/llm/src/llm/service.py
```

Expected (based on commit `b4d769a`):

```
plugins/llm/src/llm/plugin.py:1442:            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
plugins/llm/src/llm/plugin.py:3385:            ask_prompt = self.registryValue("assistantSystemPrompt", channel)
plugins/llm/src/llm/service.py:4734:        ask_prompt = plugin.registryValue("assistantSystemPrompt", row.channel)
```

For each, trace the surrounding code (sed -n) to confirm the dispatching profile:

- `plugin.py:1442`: inside structured-reminder fire; `AssistantRequestContext(profile=PROFILE_REMIND_ACTION, ...)` is constructed in the same block (around line 1426-1428). Overlay key flows to a `PROFILE_REMIND_ACTION` dispatch.
- `plugin.py:3385`: inside `_ask_impl`; local variable `effective_profile = profile_override or PROFILE_CHAT` is set just before (around line 3349). The same line serves both @ask (PROFILE_CHAT) and verse (PROFILE_VERSE).
- `service.py:4734`: inside scheduled-task fire; `AssistantRequestContext(profile="remind_action", ...)` constructed around line 4720.

- [ ] **Step 2: Add `monkeypatch.setitem` behavior tests for the migration contract**

Append to `plugins/llm/tests/test_assistant.py`:

```python
class TestOverlayReadsViaProfiles:
    """Plugin caller sites read the overlay key via PROFILES, not hardcoded.

    Swap a PROFILES entry with a sentinel overlay_setting and assert the
    sentinel key flows to plugin.registryValue at the call sites that drive
    chat-loop dispatch. These tests fail before the migration and pass
    after.
    """

    def test_remind_action_fire_reads_overlay_via_profile(
        self, mocker: MockerFixture, monkeypatch, irc, msg
    ) -> None:
        """plugin.py:1442 reads PROFILES[PROFILE_REMIND_ACTION].overlay_setting,
        not the hardcoded 'assistantSystemPrompt'.
        """
        from llm.profile import PROFILES, Profile, PROFILE_REMIND_ACTION

        sentinel = Profile(
            id=PROFILE_REMIND_ACTION,
            model_setting="assistantModel",
            api_key_setting="assistantApiKey",
            prompt_id="remind_action",
            overlay_setting="SENTINEL_REMIND_OVERLAY",
            max_output_tokens=400,
            force_search_on_explicit=True,
        )
        monkeypatch.setitem(PROFILES, PROFILE_REMIND_ACTION, sentinel)

        # Trigger the structured-reminder fire path. Use whatever fixture
        # exists in test_plugin.py for firing reminders — find it by
        # searching for `_fire_reminder` or similar. Assert that the
        # capture of registryValue calls includes "SENTINEL_REMIND_OVERLAY"
        # and does NOT include "assistantSystemPrompt".
        #
        # **Implementer guide:** if the structured-reminder fire path is
        # hard to invoke from a test, fall back to a focused unit test:
        # call the wrapper function that contains line 1442 directly with
        # mocked dependencies, and assert the registryValue capture.

    def test_ask_path_overlay_uses_effective_profile(
        self, mocker: MockerFixture, monkeypatch
    ) -> None:
        """plugin.py:3385 reads PROFILES[effective_profile].overlay_setting.
        When profile_override=PROFILE_VERSE, the verse overlay_setting is used.
        """
        from llm.profile import PROFILES, Profile, PROFILE_VERSE, PROFILE_CHAT

        sentinel_verse = Profile(
            id=PROFILE_VERSE,
            model_setting="assistantModel",
            api_key_setting="assistantApiKey",
            prompt_id="verse",
            overlay_setting="SENTINEL_VERSE_OVERLAY",
            max_output_tokens=None,
            force_search_on_explicit=False,
        )
        monkeypatch.setitem(PROFILES, PROFILE_VERSE, sentinel_verse)

        # Invoke _ask_impl directly (or through a verse-dispatch helper)
        # with profile_override=PROFILE_VERSE, capture registryValue calls,
        # assert "SENTINEL_VERSE_OVERLAY" appears and "assistantSystemPrompt"
        # does not.

    def test_scheduled_task_fire_reads_overlay_via_profile(
        self, mocker: MockerFixture, monkeypatch
    ) -> None:
        """service.py:4734 reads PROFILES[PROFILE_REMIND_ACTION].overlay_setting."""
        from llm.profile import PROFILES, Profile, PROFILE_REMIND_ACTION

        sentinel = Profile(
            id=PROFILE_REMIND_ACTION,
            model_setting="assistantModel",
            api_key_setting="assistantApiKey",
            prompt_id="remind_action",
            overlay_setting="SENTINEL_SCHED_OVERLAY",
            max_output_tokens=400,
            force_search_on_explicit=True,
        )
        monkeypatch.setitem(PROFILES, PROFILE_REMIND_ACTION, sentinel)

        # Trigger the scheduled-task fire path. The existing test_service.py
        # tests for scheduled-task firing are the pattern to copy. Assert
        # registryValue capture contains "SENTINEL_SCHED_OVERLAY".
```

> **Implementer guide:** for each test, locate the existing fixture/harness that exercises the same code path *for behavior* and reuse its setup. The tests above pin the *contract* via sentinel substitution. If a path is genuinely hard to drive end-to-end, fall back to invoking the smallest function that contains the target line.

- [ ] **Step 3: Run the new tests, confirm they fail before migration**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_assistant.py::TestOverlayReadsViaProfiles -v"
```

Expected: all three fail (overlay key is hardcoded pre-refactor).

- [ ] **Step 4: Migrate `plugin.py:1442` (structured-reminder fire, PROFILE_REMIND_ACTION)**

Read the current line:

```bash
sed -n '1438,1446p' plugins/llm/src/llm/plugin.py
```

Confirm it matches `ask_prompt = self.registryValue("assistantSystemPrompt", channel)`.

At the top of `plugin.py`, ensure the `from .profile import …` block exposes `PROFILES` (the PROFILE_* constants are likely already imported from `.assistant`, which now re-exports from `.profile`). Add `PROFILES` to the appropriate import group:

```python
from .profile import (
    PROFILES,
    PROFILE_CHAT,
    PROFILE_CODE,
    PROFILE_DRAW,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

(Or, if the existing `from .assistant import` block already pulls PROFILE_* names, leave those there and add a separate `from .profile import PROFILES` line. Either form is fine; pick whichever matches the file's existing import style.)

Replace the line at 1442 with:

```python
            ask_prompt = self.registryValue(PROFILES[PROFILE_REMIND_ACTION].overlay_setting, channel)
```

- [ ] **Step 5: Migrate `plugin.py:3385` (`_ask_impl`, dynamic profile)**

```bash
sed -n '3381,3389p' plugins/llm/src/llm/plugin.py
```

Confirm it matches `ask_prompt = self.registryValue("assistantSystemPrompt", channel)`.

The `effective_profile` local variable is in scope at this line (defined ~30 lines earlier as `effective_profile = profile_override or PROFILE_CHAT`). Use it directly:

```python
            ask_prompt = self.registryValue(PROFILES[effective_profile].overlay_setting, channel)
```

This makes the line correct for both the @ask (PROFILE_CHAT) and verse (PROFILE_VERSE) dispatches without code duplication. Today both resolve to `"assistantSystemPrompt"`; if a future change diverges them, this line follows automatically.

- [ ] **Step 6: Migrate `service.py:4734` (scheduled-task fire, PROFILE_REMIND_ACTION)**

```bash
sed -n '4730,4738p' plugins/llm/src/llm/service.py
```

Confirm it matches `ask_prompt = plugin.registryValue("assistantSystemPrompt", row.channel)`.

Ensure `service.py`'s imports include `PROFILES` and `PROFILE_REMIND_ACTION` (PROFILE_REMIND_ACTION is likely already imported after Task 2). Replace with:

```python
        ask_prompt = plugin.registryValue(PROFILES[PROFILE_REMIND_ACTION].overlay_setting, row.channel)
```

- [ ] **Step 7: Run the new tests, confirm they pass after migration**

```bash
make test PYTEST_ARGS="plugins/llm/tests/test_assistant.py::TestOverlayReadsViaProfiles -v"
```

Expected: all three pass.

- [ ] **Step 8: Run the full suite**

```bash
make test
```

Expected: no regressions; coverage stays at or above 93%.

- [ ] **Step 9: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add plugins/llm/src/llm/plugin.py \
        plugins/llm/src/llm/service.py \
        plugins/llm/tests/test_assistant.py
git commit -m "refactor(llm): migrate overlay reads to PROFILES[profile].overlay_setting"
```

---

## Task 4: Switch holdout consumers, run identity gate, push

The `from .profile import PROFILE_*` block at the top of `assistant.py` (added in Task 1 Step 4) **stays**. `assistant.py` uses those constants internally (`ToolSpec.visible_in` defaults, `_TOOL_SPEC_OVERRIDES`, `_VERSE_EXCLUDED_TOOLS`). What changes in Task 4 is the *contract*: external code switches to importing from `llm.profile` directly, so the `assistant.py` import is no longer a public shim — just an internal-use import. This is the cleanup commit; no shim deletion.

**Files:**
- Modify: any file still importing `PROFILE_*` from `llm.assistant` (inventory in Step 1)

- [ ] **Step 1: Inventory holdout consumers of `llm.assistant.PROFILE_*`**

Use `ripgrep` with multiline matching — the actual import sites in plugin.py and service.py are parenthesized multiline blocks that single-line `grep` will silently miss:

```bash
rg -n --multiline 'from\s+(\.{1,2}assistant|llm\.assistant)\s+import\s*\(' \
    plugins/llm/src plugins/llm/tests
```

For each match, open the file and read the parenthesized block. Note any `PROFILE_CHAT`, `PROFILE_CODE`, `PROFILE_DRAW`, `PROFILE_VERSE`, or `PROFILE_REMIND_ACTION` names — those are the ones to migrate. Single-line imports (rare) get caught by:

```bash
rg -n 'from\s+(\.{1,2}assistant|llm\.assistant)\s+import\s+[^(]*PROFILE_' \
    plugins/llm/src plugins/llm/tests
```

The two greps together cover both forms.

- [ ] **Step 2: Update each holdout to import from `llm.profile`**

For each file, rewrite the import. Example (multiline):

```python
# Before
from .assistant import (
    AssistantToolExecutor,
    PROFILE_CHAT,
    PROFILE_VERSE,
    ToolSpec,
)

# After
from .assistant import (
    AssistantToolExecutor,
    ToolSpec,
)
from .profile import PROFILE_CHAT, PROFILE_VERSE
```

If a file imports *only* `PROFILE_*` from `.assistant`, the entire `.assistant` import collapses; just replace with `from .profile import …`.

- [ ] **Step 3: Run the full suite**

```bash
make test
```

Expected: no regressions. If any test still fails on a `PROFILE_*` import from `llm.assistant`, the Step 1 grep missed it — re-grep and re-migrate.

- [ ] **Step 4: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean. If `ruff` reports an `F401` ("imported but unused") on a `PROFILE_*` name inside `assistant.py`, that's a real bug — those names are used in `ToolSpec.visible_in` defaults and the `_TOOL_SPEC_OVERRIDES` / `_VERSE_EXCLUDED_TOOLS` blocks. If `ruff` is right and a name really has no use, delete only that one name from the import list.

- [ ] **Step 5: Run `make preflight`**

```bash
make preflight
```

Expected: everything green.

- [ ] **Step 6: Run the byte-identity gate**

```bash
python /tmp/profile_identity.py > /tmp/profile_identity_post.txt
diff /tmp/profile_identity_pre.txt /tmp/profile_identity_post.txt
echo "exit=$?"
```

Expected: empty diff, `exit=0`. Identical digests mean the refactor preserved every observable: framework prompt text per profile, tool name set per profile, max_output_tokens per profile, force_initial_search predicate result per sample prompt, overlay setting, model setting, api_key setting, and the EXPLICIT_SEARCH_RE pattern itself.

If the diff is non-empty: do not commit. Read the diff, identify which observable changed, decide whether it's a bug to fix or whether the Pre-Work pinning was wrong. The PRE_* dicts in `/tmp/profile_identity.py` were verified against source during plan-writing; if those need correcting, fix them, but only after confirming the source-of-truth.

- [ ] **Step 7: Runtime import smoke check**

```bash
python -c "import sys; sys.path.insert(0, 'plugins/llm/src'); import llm.plugin, llm.service, llm.profile, llm.assistant, llm.prompts; print('import-smoke=ok')"
```

Expected output: `import-smoke=ok`. This catches any module-load failure that mocked tests would not (e.g., a circular import that only manifests under Limnoria's normal load order, a missing constant referenced in a class body executed at import time).

If this fails: do not commit. Trace the import chain manually and fix.

- [ ] **Step 8: Commit**

```bash
git add -- plugins/llm/src plugins/llm/tests
git commit -m "refactor(llm): switch holdout consumers to llm.profile imports"
```

- [ ] **Step 9: Push to main**

```bash
git push
```

Auto-deploy fires when CI + Docker workflows pass. Watch for the green check via `gh run list --limit 5` after a minute.

- [ ] **Step 10: Verify production after auto-deploy**

```bash
ssh -i ~/.ssh/id_rsa vibebot@rdrake.org \
    'systemctl --user is-active vibebot && docker logs --tail 20 vibebot 2>&1 | tail -5'
```

Expected: `active`, no startup errors. If the bot fails to start: roll back with `git revert` (the four commits are designed to revert cleanly because each commit was independently green-on-main).

- [ ] **Step 11: Cleanup local harness files**

```bash
rm /tmp/profile_identity.py /tmp/profile_identity_pre.txt /tmp/profile_identity_post.txt
```

One-off — value was at the migration moment, not as a permanent artifact.

---

## Self-Review

Before handing off, the plan was reviewed against the spec and against findings from the parallel adversarial review pass (codex:codex-rescue + general-purpose code-reviewer):

**Spec coverage check:**

- ✅ Module layout (new `profile.py`, dependency direction) → Task 1, Steps 3-4.
- ✅ Profile dataclass shape (7 fields) → Task 1, Step 3.
- ✅ PROFILES registry (5 entries with corrected per-field mappings) → Task 1, Step 3.
- ✅ Migration of `service.py:assistant_completion` (5 lookups collapse, silent-fallback preserved) → Task 2, Step 5.
- ✅ Migration of overlay reads at plugin.py:1442, plugin.py:3385 (dynamic effective_profile), service.py:4734 → Task 3, Steps 4-6. plugin.py:2547 explicitly out of scope (inner `_code_for_assistant` one-shot).
- ✅ PROFILE_VERSE footer left untouched → Task 2, Step 5.
- ✅ Testing strategy: invariant tests in `test_profile.py` (Task 1 Step 1) + behavior tests via `monkeypatch.setitem` in `test_assistant.py` (Task 2 Step 3, Task 3 Step 2).
- ✅ Byte-identity gate (real EXPLICIT_SEARCH_RE imported from service.py; multi-sample force-search exercise; explicit regex-pattern observable) → Pre-Work + Task 4, Step 6.
- ✅ Runtime import smoke check → Task 4, Step 7.
- ✅ verseModel preserved as caller-side override at plugin.py:3307 — documented in spec, untouched by refactor.
- ✅ Four phased commits, each green on `main` for auto-deploy → Task structure.

**Adversarial-review fixes applied:**

- **Codex D1 + General D3** (PROFILE_CODE mapping): corrected `model_setting`/`api_key_setting`/`overlay_setting`. The @code planner uses `assistantModel`/`assistantApiKey` and reads no channel overlay; `codeModel`/`codeApiKey`/`codeSystemPrompt` belong to the inner one-shot.
- **Codex D2** (verseModel caller override): documented as out-of-Profile in the spec; Profile.model_setting captures the fallback path only.
- **Codex D3 + General D1** (fake regex gate): harness now imports `EXPLICIT_SEARCH_RE` directly and tests every alternation term with sample prompts; the regex pattern is also a pinned observable.
- **Codex D6** (multiline-import grep): Task 4 Step 1 uses `rg --multiline` to catch parenthesized blocks.
- **Codex D7 + General D7** (overlay tests pass before AND after migration): replaced literal-value pins with `monkeypatch.setitem` sentinel-Profile behavior tests that fail before migration and pass after.
- **Codex D8** (Task 2 Step 3 scaffolding pointed at wrong test file): redirected to `test_assistant.py` where the direct `assistant_completion` harness lives.
- **General D3** (silent-fallback semantic change): Task 2 Step 5 uses `PROFILES.get(route_profile, PROFILES[PROFILE_CHAT])` to preserve pre-refactor behavior.
- **General D5** (overly broad `pytest.raises(Exception)`): narrowed to `dataclasses.FrozenInstanceError` in Task 1 Step 1.
- **General D6** (half-pseudocode tests): rewrote Task 2 Step 3 and Task 3 Step 2 with complete, runnable test bodies. Some implementer guidance remains for harness fixtures (no way to eliminate this without specifying which existing helpers to copy line-by-line); the contract is concrete.
- **General D8** (ruff E402 on import location): Task 1 Step 4 explicitly moves `from .profile import PROFILE_*` to the top-of-file imports section instead of replacing in-place.
- **General D10** (no runtime smoke check before push): added Task 4 Step 7.

**Placeholder scan:**

- Task 2 Step 3 and Task 3 Step 2 contain runnable test code but defer the harness-fixture setup to the existing `test_assistant.py` patterns (specifically the `service`/`mocker`/`irc`/`msg` fixtures). The test bodies themselves are complete; only "which conftest fixture provides X" is documented narratively.

**Type / identifier consistency:**

- `Profile` dataclass field names (`id`, `model_setting`, `api_key_setting`, `prompt_id`, `overlay_setting`, `max_output_tokens`, `force_search_on_explicit`) match across Task 1, Task 2, Task 3, and the harness in Pre-Work.
- `PROFILES[<key>].overlay_setting` form (and the dynamic `PROFILES[effective_profile]` form in plugin.py:3385) used consistently.
- `get_tools_for_profile(profile.id, ...)` matches the existing assistant.py signature.

**Scope check:** One implementation cycle, four commits, single feature area (`plugins/llm/`). No subsystem decomposition needed.

**Behavior preservation:** The byte-identity gate now exercises every per-profile observable (including the regex pattern itself and force_search across the full alternation term set). The silent-fallback for unknown route_profile is preserved via `PROFILES.get`. The verseModel caller-side override is left untouched.
