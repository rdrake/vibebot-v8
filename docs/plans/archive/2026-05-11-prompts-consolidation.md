# Prompts Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all 7 framework/internal system prompts into a single `prompts.py` module so there is one obvious place to find, edit, and reason about any prompt.

**Architecture:** Today, 5 framework prompts live in `assistant.py:66–227` and 2 memory prompts live in `service.py:159–203`. Two callers (`service.py:3218`, `plugin.py:3555`) import them directly. We create `plugins/llm/src/llm/prompts.py` as the single owner, expose a `PROMPTS: dict[str, str]` registry keyed by profile name (plus the 2 memory prompts under their own keys), keep the shared `IRC_OUTPUT_FORMAT` building block public so future prompts can compose it, and migrate every consumer one file at a time. Prompt **content is unchanged** — this is a pure code-move refactor; behavior must be character-for-character identical.

**Out of scope:**
- `assistantSystemPrompt` / `codeSystemPrompt` registry values in `config.py` — these are channel-overridable personality overlays, not framework prompts; they're already in one obvious place and they have a different role.
- The `Profile` abstraction (next refactor step).
- Loading prompts from external markdown/YAML files (premature).
- Editing any prompt text.

**Tech Stack:** Python 3.12+, pytest, ruff, ty. Tests live under `plugins/llm/tests/`.

---

## File Structure

| File | Disposition | Role after refactor |
|------|-------------|---------------------|
| `plugins/llm/src/llm/prompts.py` | **NEW** | Single owner of all 7 framework + internal prompts and the shared `IRC_OUTPUT_FORMAT` block. Exposes `PROMPTS` dict registry. |
| `plugins/llm/src/llm/assistant.py` | Modify | Drops all `*_SYSTEM_PROMPT` constants and the shared `_IRC_OUTPUT_FORMAT` / `_MARKDOWN_BANNED_TOKENS` blocks. Keeps `PROFILE_*` identifiers and tool-spec code. Shrinks from 1,218 → ~1,030 lines. |
| `plugins/llm/src/llm/service.py` | Modify | Drops `_MEMORY_EXTRACTION_PROMPT` and `_MEMORY_CLEANUP_PROMPT`. The `profile_frameworks` dict in `assistant_request` collapses to a `PROMPTS` lookup. |
| `plugins/llm/src/llm/plugin.py` | Modify | One import line changes (`assistant.CODE_SYSTEM_PROMPT` → `prompts.CODE_SYSTEM_PROMPT`). |
| `plugins/llm/tests/test_prompts.py` | **NEW** | Invariant tests for the new module: registry has the right keys, format placeholders are present, building blocks are shared correctly. |
| `plugins/llm/tests/test_assistant.py` | Modify | Update imports (`from llm.assistant import …` → `from llm.prompts import …`) and move the existing prompt invariant assertions over to `test_prompts.py` (they currently sit at lines 2352–2423). |
| `plugins/llm/tests/test_service.py` | Modify | Update `from llm.service import _MEMORY_EXTRACTION_PROMPT` etc. to `from llm.prompts import MEMORY_EXTRACTION_PROMPT` (dropping the underscore prefix, since the constants are no longer module-private helpers). |
| `plugins/llm/tests/test_commands.py` | No change | Only references `CODE_SYSTEM_PROMPT` inside a docstring — purely cosmetic. |

**Naming decision:** The two memory prompts move from `_MEMORY_EXTRACTION_PROMPT` / `_MEMORY_CLEANUP_PROMPT` (leading-underscore = module-private to `service.py`) to `MEMORY_EXTRACTION_PROMPT` / `MEMORY_CLEANUP_PROMPT` in `prompts.py`. Their job in the new module is to *be* the export; the underscore loses meaning.

---

## Task 1: Create `prompts.py` with all 7 prompts and the `PROMPTS` registry

**Files:**
- Create: `plugins/llm/src/llm/prompts.py`
- Create: `plugins/llm/tests/test_prompts.py`

The new module owns the literal text of every framework/internal system prompt plus the shared `IRC_OUTPUT_FORMAT` building block. Source text is copied verbatim from `assistant.py:39–227` and `service.py:159–203`. Do not paraphrase, do not "improve", do not reformat whitespace — character-identical copies only. We will diff before/after in Task 6.

- [ ] **Step 1: Write the failing test for the new module**

Create `plugins/llm/tests/test_prompts.py` with the following content:

```python
"""Invariants for the consolidated prompts module.

These tests pin the shape of ``llm.prompts`` so future refactors don't
silently drop a profile key or break the shared IRC output rules.
"""

from __future__ import annotations

import pytest

from llm import prompts


class TestPromptsRegistry:
    """The PROMPTS dict is the single source of truth for prompt lookup."""

    def test_registry_has_all_expected_keys(self):
        """PROMPTS exposes every framework and internal prompt by name."""
        assert set(prompts.PROMPTS.keys()) == {
            "chat",
            "code",
            "draw",
            "verse",
            "remind_action",
            "memory_extraction",
            "memory_cleanup",
        }

    @pytest.mark.parametrize(
        "name",
        ["chat", "code", "draw", "verse", "remind_action",
         "memory_extraction", "memory_cleanup"],
    )
    def test_every_prompt_is_nonempty(self, name):
        """Each registered prompt is a non-empty string."""
        text = prompts.PROMPTS[name]
        assert isinstance(text, str)
        assert text.strip(), f"prompt {name!r} is empty"


class TestProfilePromptInvariants:
    """Profile-facing prompts share the {bot_nick} placeholder contract."""

    @pytest.mark.parametrize(
        "name", ["chat", "code", "draw", "verse", "remind_action"],
    )
    def test_profile_prompts_contain_bot_nick_placeholder(self, name):
        """Profile prompts are formatted with .format(bot_nick=...)."""
        assert "{bot_nick}" in prompts.PROMPTS[name]

    @pytest.mark.parametrize(
        "name", ["memory_extraction", "memory_cleanup"],
    )
    def test_memory_prompts_have_no_placeholders(self, name):
        """Memory prompts are used as raw constants — no format() call."""
        assert "{" not in prompts.PROMPTS[name]


class TestIrcOutputFormatSharing:
    """IRC_OUTPUT_FORMAT is shared by Q&A modes but NOT by verse."""

    def test_irc_output_format_is_exposed(self):
        """The shared building block is importable for future composition."""
        assert isinstance(prompts.IRC_OUTPUT_FORMAT, str)
        assert "OUTPUT FORMAT" in prompts.IRC_OUTPUT_FORMAT

    @pytest.mark.parametrize("name", ["chat", "code", "draw", "remind_action"])
    def test_qa_prompts_embed_irc_output_format(self, name):
        """Chat-style prompts embed the shared format block verbatim."""
        assert prompts.IRC_OUTPUT_FORMAT in prompts.PROMPTS[name]

    def test_verse_does_not_embed_irc_output_format(self):
        """Verse owns its own length/format rules (no 3-line cap)."""
        assert prompts.IRC_OUTPUT_FORMAT not in prompts.PROMPTS["verse"]
        # But verse must still ban markdown in its own words — assert
        # explicitly so deleting IRC_OUTPUT_FORMAT wholesale doesn't
        # silently strip verse's markdown ban as a side effect.
        assert "Plain text only" in prompts.PROMPTS["verse"]
        assert "**bold**" in prompts.PROMPTS["verse"]


class TestVersePromptInvariants:
    """Behavior-critical invariants for the verse framework prompt.

    Migrated from test_assistant.py — the verse prompt is the most
    bug-prone of the lot, so these assertions stay close to the source.
    """

    def test_verse_in_world_roleplay_framing(self):
        text = prompts.PROMPTS["verse"]
        assert "in-world roleplay" in text
        assert "Stay in character" in text
        assert "Adopt user-offered details" in text

    def test_verse_omits_three_line_cap(self):
        """Verse deliberately drops the chat 3-line length cap."""
        assert "Length cap: 3 lines" not in prompts.PROMPTS["verse"]

    def test_verse_record_hard_rule_present(self):
        text = prompts.PROMPTS["verse"]
        assert "HARD RULE" in text
        assert "verse_record" in text
        assert "user describes" in text
        assert "narrate" in text

    def test_verse_exposes_avatar_tools(self):
        text = prompts.PROMPTS["verse"]
        assert "verse_act" in text
        assert "verse_recall" in text

    def test_verse_recall_vs_narrate_distinction(self):
        assert "RECALL" in prompts.PROMPTS["verse"]

    def test_verse_single_message_discipline(self):
        assert "single message" in prompts.PROMPTS["verse"]


class TestProfileSpecificContent:
    """Per-profile content invariants migrated from test_assistant.py.

    Owning these here means Task 5's deletion of the equivalent block
    in test_assistant.py cannot accidentally drop a load-bearing check.
    Without these, a future edit could re-mix verse rules into the chat
    prompt or remove the tool-name anchor from code/draw and only show
    up as a behavior bug in production.
    """

    def test_chat_omits_internal_meta_token(self):
        """CHAT_SYSTEM_PROMPT does not contain the NOT_META control word."""
        assert "NOT_META" not in prompts.PROMPTS["chat"]

    def test_chat_does_not_carry_verse_rules(self):
        """Chat and verse stay structurally separate — chat must not
        mention verse-mode mechanics or it'll start cross-routing tools."""
        assert "VERSE MODE" not in prompts.PROMPTS["chat"]
        assert "verse_record" not in prompts.PROMPTS["chat"]

    def test_code_prompt_mentions_generate_code_tool(self):
        """The code prompt anchors the planner on the right tool name."""
        assert "generate_code" in prompts.PROMPTS["code"]

    def test_draw_prompt_mentions_generate_image_tool(self):
        """The draw prompt anchors the planner on the right tool name."""
        assert "generate_image" in prompts.PROMPTS["draw"]

    def test_remind_action_does_not_mention_set_reminder(self):
        """Remind-action runs INSIDE a fired reminder — telling the
        model about set_reminder there causes scheduling loops."""
        assert "set_reminder" not in prompts.PROMPTS["remind_action"]

    def test_remind_action_documents_mechanical_recurrence(self):
        """The fire-time prompt must say the scheduler handles recurrence."""
        assert "Recurrence is handled mechanically" in prompts.PROMPTS["remind_action"]
```

- [ ] **Step 2: Run the test to confirm it fails**

Run from repo root:

```bash
cd plugins/llm && uv run pytest tests/test_prompts.py -v
```

Expected: `ModuleNotFoundError: No module named 'llm.prompts'` (collection-time failure). That's the failing-first signal.

- [ ] **Step 3: Create `plugins/llm/src/llm/prompts.py` with verbatim prompt copies**

The content below is **exactly** the strings already living in `assistant.py` and `service.py`. Do not edit a single character of prompt text. Copy `_MARKDOWN_BANNED_TOKENS` from `assistant.py:39–46`, `_IRC_OUTPUT_FORMAT` from `assistant.py:47–63`, `CHAT_SYSTEM_PROMPT` from `assistant.py:66–123`, `VERSE_SYSTEM_PROMPT` from `assistant.py:140–184`, `CODE_SYSTEM_PROMPT` from `assistant.py:186–197`, `DRAW_SYSTEM_PROMPT` from `assistant.py:199–207`, `REMIND_ACTION_SYSTEM_PROMPT` from `assistant.py:210–227`, `_MEMORY_EXTRACTION_PROMPT` from `service.py:159–186` (renamed to `MEMORY_EXTRACTION_PROMPT`), `_MEMORY_CLEANUP_PROMPT` from `service.py:188–203` (renamed to `MEMORY_CLEANUP_PROMPT`).

Note: `_IRC_OUTPUT_FORMAT` becomes the public `IRC_OUTPUT_FORMAT` (drop the underscore — it is exported). `_MARKDOWN_BANNED_TOKENS` stays underscore-private (still a building block internal to this module).

Skeleton (fill in prompt bodies by copying from current locations):

```python
"""Single source of truth for all framework and internal LLM prompts.

This module owns:

- Five **framework prompts** — the structural system prompts that wrap every
  LLM call routed through ``service.assistant_request``: chat, code, draw,
  verse, and remind_action. Each is formatted with ``.format(bot_nick=...)``.
- Two **internal prompts** — used by the memory pipeline in
  ``service.extract_memories`` and ``service.cleanup_memories``. These are
  used as raw constants (no placeholders).
- One shared building block — ``IRC_OUTPUT_FORMAT`` — embedded by every
  Q&A-style framework prompt (chat, code, draw, remind_action). Verse owns
  its own format rules because it deliberately drops the 3-line length cap.

Channel-overridable personality overlays (``assistantSystemPrompt``,
``codeSystemPrompt``) live in ``config.py`` and intentionally stay there —
they're operator-tunable settings, not framework prompts.

Lookup via ``PROMPTS[name]``. Keys match ``assistant.PROFILE_*`` identifiers
for profile prompts, plus ``"memory_extraction"`` and ``"memory_cleanup"``
for the internal pair.
"""

from __future__ import annotations


# Shared IRC output rules. Embedded near the top of every Q&A framework
# prompt because models (notably Grok) ignore these constraints when
# they're buried in a long rule list. Keep this block concrete and
# example-driven — abstract instructions like "be concise" do not work;
# an explicit list of forbidden tokens does.
_MARKDOWN_BANNED_TOKENS = (
    "    **bold** or __bold__\n"
    "    *italics* or _italics_\n"
    "    `inline code` or ``` fenced code blocks ```\n"
    "    # headings (of any depth)\n"
    "    [label](url) — write the bare URL instead\n"
    "    | tables |, ASCII art, or box-drawing\n"
)
IRC_OUTPUT_FORMAT = (
    "OUTPUT FORMAT — this is IRC, NOT a chat UI. Read this carefully:\n"
    "- Lead with the answer. Skip preambles like 'Sure!', 'Great question', "
    "'Of course', 'Here's what I found', or restating the question.\n"
    "- Length cap: 3 lines. One line is ideal. Only exceed the cap when the "
    "user explicitly asks for detail, a list, or a step-by-step.\n"
    "- Plain text only — IRC clients DO NOT render markdown. Do NOT emit any "
    "of these tokens, in any form:\n"
    + _MARKDOWN_BANNED_TOKENS
    + "    - bullet lists, * bullet lists, 1. numbered lists\n"
    "- URLs: write them bare. No brackets, no surrounding link text.\n"
    "- Code or commands in a reply: emit the bare content on its own line "
    "with NO backticks and NO fences.\n"
    "- No emoji-spam. At most one emoji, only if it genuinely adds meaning.\n"
    "If the answer would naturally want a list, render it as a single line "
    "with comma separation instead.\n"
)


CHAT_SYSTEM_PROMPT = (
    # === paste verbatim from assistant.py:66–123 ===
    # (uses IRC_OUTPUT_FORMAT — change the local name reference accordingly)
)


# Verse mode is interactive in-world roleplay, not Q&A. It needs a different
# output discipline (long-form scenes, not 3-line replies) and a different
# tool stance (verse_record is mandatory canon-logging, not optional). A
# dedicated framework lets us drop the chat-mode "Answer directly" framing
# and the 3-line cap that empirically suppress verse-mode storytelling, and
# lets the framework footer's "rules above still apply" weight enforce the
# verse_record HARD RULE rather than relying on the personality overlay
# (which the same footer explicitly demotes to "voice, not structure"). The
# caching cost is minor: one cache miss per channel-session when the path
# first switches; subsequent verse turns share a verse-mode prefix and
# cache among themselves. We tried a shared-framework approach with a
# conditional VERSE MODE block, but the model kept respecting the chat
# defaults (sentence-per-item replies, tool_calls=0) — the structural
# rules at the top of the framework dominate any override added later.
VERSE_SYSTEM_PROMPT = (
    # === paste verbatim from assistant.py:140–184 ===
)


CODE_SYSTEM_PROMPT = (
    # === paste verbatim from assistant.py:186–197 ===
    # (uses IRC_OUTPUT_FORMAT)
)


DRAW_SYSTEM_PROMPT = (
    # === paste verbatim from assistant.py:199–207 ===
    # (uses IRC_OUTPUT_FORMAT)
)


REMIND_ACTION_SYSTEM_PROMPT = (
    # === paste verbatim from assistant.py:210–227 ===
    # (uses IRC_OUTPUT_FORMAT)
)


# Two-stage memory: extracted facts enter ``memory_candidates`` with a
# mention count. They are only promoted to durable ``memories`` once the
# extractor reinforces them across multiple exchanges. The prompt asks the
# LLM to pick between adding a brand-new candidate and reinforcing an
# existing one to keep paraphrases from spawning duplicates.
MEMORY_EXTRACTION_PROMPT = (
    # === paste verbatim from service.py:159–186 ===
)


MEMORY_CLEANUP_PROMPT = (
    # === paste verbatim from service.py:188–203 ===
)


PROMPTS: dict[str, str] = {
    "chat": CHAT_SYSTEM_PROMPT,
    "code": CODE_SYSTEM_PROMPT,
    "draw": DRAW_SYSTEM_PROMPT,
    "verse": VERSE_SYSTEM_PROMPT,
    "remind_action": REMIND_ACTION_SYSTEM_PROMPT,
    "memory_extraction": MEMORY_EXTRACTION_PROMPT,
    "memory_cleanup": MEMORY_CLEANUP_PROMPT,
}
```

**Critical:** while pasting, replace any reference to `_IRC_OUTPUT_FORMAT` inside the prompt-body assignments with `IRC_OUTPUT_FORMAT` (because we dropped the underscore). The text concatenation pattern is `... + IRC_OUTPUT_FORMAT + "\nTool & behavior rules:\n" ...`. Do not change anything else.

**Mechanical extraction recipe** (use this to avoid hand-typos):

```bash
# From repo root — extracts the prompt blocks as bytes, no manual retyping:
git show HEAD:plugins/llm/src/llm/assistant.py | sed -n '34,227p' > /tmp/prompts_assistant.txt
git show HEAD:plugins/llm/src/llm/service.py   | sed -n '152,203p' > /tmp/prompts_service.txt
# Now paste those blocks into prompts.py, then apply two renames:
#   _IRC_OUTPUT_FORMAT      → IRC_OUTPUT_FORMAT
#   _MEMORY_EXTRACTION_PROMPT → MEMORY_EXTRACTION_PROMPT
#   _MEMORY_CLEANUP_PROMPT    → MEMORY_CLEANUP_PROMPT
```

After paste, verify with `diff <(git show HEAD:plugins/llm/src/llm/assistant.py | sed -n '34,227p') <(sed -n '<start>,<end>p' plugins/llm/src/llm/prompts.py)` — the only diffs should be on the four rename lines.

- [ ] **Step 4: Run the new module's tests — must pass**

```bash
cd plugins/llm && uv run pytest tests/test_prompts.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run the full suite — must still pass (nothing else depends on `prompts.py` yet)**

```bash
make test
```

Expected: same pass count as before this task (the existing tests still import from `assistant.py` and `service.py`, which we have not touched yet).

- [ ] **Step 6: Lint and typecheck the new file**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add plugins/llm/src/llm/prompts.py plugins/llm/tests/test_prompts.py
git commit -m "refactor(llm): add prompts.py as single source of truth (no consumers yet)"
```

---

## Task 2: Re-export prompt constants from `assistant.py` for backwards compat

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py` (replace lines 39–227 — the prompt block — with re-exports from `prompts.py`)

Why a shim? We want every consumer-migration commit (Tasks 3–5) to be tiny and individually verifiable with `make test`. The shim keeps the old import paths alive during migration. Task 6 deletes it once all consumers are off it.

- [ ] **Step 1: Replace the prompt block in `assistant.py` with re-exports**

In `plugins/llm/src/llm/assistant.py`, **delete** the entire block from line 34 ("# Shared IRC output rules…") through line 227 (closing paren of `REMIND_ACTION_SYSTEM_PROMPT`). In its place, insert:

```python
# Prompt constants moved to ``prompts.py``. Re-exported here so existing
# imports (`from llm.assistant import CHAT_SYSTEM_PROMPT`) keep working
# during the consolidation migration. Remove this block once every
# consumer has been switched to import from ``llm.prompts`` directly.
from .prompts import (  # noqa: F401,E402
    CHAT_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    DRAW_SYSTEM_PROMPT,
    REMIND_ACTION_SYSTEM_PROMPT,
    VERSE_SYSTEM_PROMPT,
)
```

Keep the `PROFILE_CHAT = "chat"` etc. block (lines 25–31) and everything after line 227 (the tool specs) untouched.

- [ ] **Step 2: Run the full suite**

```bash
make test
```

Expected: same pass count as Task 1. Existing tests still import the constants by name from `llm.assistant`; the re-export keeps that import live, and the constants themselves are character-identical (sourced from `prompts.py`, which we pasted verbatim).

- [ ] **Step 3: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean. (Ruff may complain about the import position — that's why we added `# noqa: E402`. Confirm no real errors.)

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/src/llm/assistant.py
git commit -m "refactor(llm): re-export prompt constants from prompts.py via assistant shim"
```

---

## Task 3: Migrate `service.py` to consume `prompts.PROMPTS`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` lines 152–203 (delete the `_MEMORY_*_PROMPT` constants)
- Modify: `plugins/llm/src/llm/service.py` lines 3178–3225 (collapse `profile_frameworks` into a `PROMPTS` lookup)
- Modify: `plugins/llm/src/llm/service.py` lines 4355, 4423 (callers of memory prompts)

- [ ] **Step 1: Delete the two memory-prompt constants from `service.py`**

In `plugins/llm/src/llm/service.py`, **delete** lines 152–203 entirely (the block starting with the `_MEMORY_EXTRACTION_PROMPT` comment through the closing paren of `_MEMORY_CLEANUP_PROMPT`). Do not delete the `_PYGMENTS_CSS` line (150) or the `_EXTRACTION_SCHEMA` block (206 onward) — those stay.

- [ ] **Step 2: Update the two memory-prompt call sites in `service.py`**

At `service.py:4355` (currently `{"role": "system", "content": _MEMORY_EXTRACTION_PROMPT}`), change to:

```python
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
```

At `service.py:4423` (currently `{"role": "system", "content": _MEMORY_CLEANUP_PROMPT}`), change to:

```python
            {"role": "system", "content": MEMORY_CLEANUP_PROMPT},
```

Add the import near the top of `service.py` (where other intra-package imports live — search for `from .` to find the import block):

```python
from .prompts import MEMORY_CLEANUP_PROMPT, MEMORY_EXTRACTION_PROMPT, PROMPTS
```

- [ ] **Step 3: Collapse `profile_frameworks` into a `PROMPTS` lookup**

At `service.py:3178–3186`, the local import currently reads:

```python
        from .assistant import (
            CHAT_SYSTEM_PROMPT,
            CODE_SYSTEM_PROMPT,
            DRAW_SYSTEM_PROMPT,
            REMIND_ACTION_SYSTEM_PROMPT,
            VERSE_SYSTEM_PROMPT,
            AssistantToolExecutor,
            get_tools_for_profile,
        )
```

Change it to drop the prompt imports (we already imported `PROMPTS` at module top):

```python
        from .assistant import (
            AssistantToolExecutor,
            get_tools_for_profile,
        )
```

**Also trim the module-top `PROFILE_*` imports at `service.py:34–40`.** After this task, only `PROFILE_CHAT`, `PROFILE_REMIND_ACTION`, and `PROFILE_VERSE` are referenced in `service.py` (verify with `grep -n PROFILE_ plugins/llm/src/llm/service.py`). `PROFILE_CODE` and `PROFILE_DRAW` were only used by the `profile_frameworks` dict you're deleting; leaving them imported will trigger ruff `F401` (`make lint` will break this commit). Edit the import block:

```python
from .assistant import (
    PROFILE_CHAT,
    PROFILE_CODE,           # <-- delete this line
    PROFILE_DRAW,           # <-- delete this line
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

becomes:

```python
from .assistant import (
    PROFILE_CHAT,
    PROFILE_REMIND_ACTION,
    PROFILE_VERSE,
)
```

At `service.py:3218–3227`, replace the `profile_frameworks` dict and the `framework =` line:

```python
            profile_frameworks = {
                PROFILE_CHAT: CHAT_SYSTEM_PROMPT,
                PROFILE_CODE: CODE_SYSTEM_PROMPT,
                PROFILE_DRAW: DRAW_SYSTEM_PROMPT,
                PROFILE_REMIND_ACTION: REMIND_ACTION_SYSTEM_PROMPT,
                PROFILE_VERSE: VERSE_SYSTEM_PROMPT,
            }
            framework = profile_frameworks.get(route_profile, CHAT_SYSTEM_PROMPT).format(
                bot_nick=bot_nick
            )
```

becomes:

```python
            framework = PROMPTS.get(route_profile, PROMPTS["chat"]).format(
                bot_nick=bot_nick
            )
```

The structural comment above (lines 3206–3217 — "Structural framework (IRC output rules…)") stays — it's still accurate and still load-bearing.

- [ ] **Step 4: Run the full suite**

```bash
make test
```

Expected: pass count unchanged **except** at three import sites in `test_service.py` (lines 4055, 4221, 4531) which now fail with `ImportError` for the deleted `_MEMORY_*_PROMPT` constants. That is the signal that Task 5 is now necessary. Note this is three sites across two test functions, not "two tests".

If any *other* test fails, stop — the prompt content must have drifted. Revert and investigate.

- [ ] **Step 5: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean (apart from the test failures above, which are import-time, not lint).

- [ ] **Step 6: DO NOT COMMIT YET — proceed straight to Task 5**

This repo auto-deploys when CI goes green on `main` (see memory: "Auto-deploy on Docker green"). A red intermediate commit on `main` either blocks deploy or — if force-merged — ships a broken build to production. Keep the working-tree changes from this task in place and execute Task 5 in the same session, then commit Tasks 3 + 5 together as a single atomic commit at Task 5 Step 6.

If you must commit separately (e.g. you're working on a feature branch and will squash before merging), the message for this intermediate commit would be `refactor(llm): migrate service.py to prompts.PROMPTS registry (WIP)` — but the default path is one combined commit.

---

## Task 4: Migrate `plugin.py:3555` to import `CODE_SYSTEM_PROMPT` from `prompts.py`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:3551–3559`

There is exactly one direct prompt import in `plugin.py` (the code-command path that layers user instructions onto the planner facade).

- [ ] **Step 1: Change the import line**

At `plugin.py:3555` (currently `from .assistant import CODE_SYSTEM_PROMPT`), change to:

```python
                from .prompts import CODE_SYSTEM_PROMPT
```

The surrounding comment at lines 3551–3554 ("Layer user instruction onto CODE_SYSTEM_PROMPT (the facade prompt that tells the planner to call generate_code) — not the registry codeSystemPrompt, which is the inner-call prompt used by _code_for_assistant.") stays — it's still accurate.

- [ ] **Step 2: Run the suite**

```bash
make test
```

Expected: same failures as end of Task 3 (the three `test_service.py` tests still fail on import). No new failures.

- [ ] **Step 3: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "refactor(llm): import CODE_SYSTEM_PROMPT from prompts.py in plugin.py"
```

---

## Task 5: Migrate tests to import from `prompts.py`

**Files:**
- Modify: `plugins/llm/tests/test_assistant.py:11–15` (imports) and `:2352–2423` (prompt invariant tests)
- Modify: `plugins/llm/tests/test_service.py:4055, 4057–4058, 4221, 4245, 4531, 4533–4534`

- [ ] **Step 1: Update `test_assistant.py` imports**

At `test_assistant.py:11–15`, the import block currently reads:

```python
from llm.assistant import (
    CHAT_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    DRAW_SYSTEM_PROMPT,
    REMIND_ACTION_SYSTEM_PROMPT,
    VERSE_SYSTEM_PROMPT,
    # ... possibly more symbols on later lines
)
```

Split: keep non-prompt imports from `llm.assistant`, add a new block above (or below) for prompts:

```python
from llm.assistant import (
    # ... whatever non-prompt symbols are imported here, untouched
)
from llm.prompts import (
    CHAT_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    DRAW_SYSTEM_PROMPT,
    REMIND_ACTION_SYSTEM_PROMPT,
    VERSE_SYSTEM_PROMPT,
)
```

Open the file first to confirm exactly which other symbols are imported alongside the prompt constants before editing.

- [ ] **Step 2: Delete the now-duplicate prompt-invariant tests in `test_assistant.py:2352–2423`**

Task 1's `test_prompts.py` was deliberately written to cover **every** assertion in the existing `test_assistant.py:2352–2423` block — `{bot_nick}` placeholders, `NOT_META`, `generate_code` / `generate_image` tool anchors, the chat-vs-verse structural separation, the `set_reminder` exclusion and "Recurrence is handled mechanically" line in remind-action, and the full set of verse invariants. The lines at `test_assistant.py:2352–2423` are now strict duplicates.

Delete that whole block. Read the file first to confirm the exact boundary: those tests may live inside a class with other still-relevant tests, in which case delete only the prompt-invariant methods, not the whole class.

**Safety check before deleting:** `diff` the assertions you're removing against `test_prompts.py` to be sure none slipped through:

```bash
grep -E "assert .*PROMPT" plugins/llm/tests/test_assistant.py | sort -u
grep -E "assert .*prompts\\.PROMPTS\\[" plugins/llm/tests/test_prompts.py | sort -u
```

Every meaningful assertion from the first list should appear (possibly reworded for `PROMPTS["name"]`) in the second list. If anything is unique to `test_assistant.py`, copy it into `test_prompts.py` before deleting.

- [ ] **Step 3: Update `test_service.py` memory-prompt imports and assertions**

Find each of these spots and update:

- `test_service.py:4055`: `from llm.service import _MEMORY_EXTRACTION_PROMPT` → `from llm.prompts import MEMORY_EXTRACTION_PROMPT`
- `test_service.py:4057`: `assert "at most 2" in _MEMORY_EXTRACTION_PROMPT.lower()` → `assert "at most 2" in MEMORY_EXTRACTION_PROMPT.lower()`
- `test_service.py:4058`: `assert "DO NOT SAVE" in _MEMORY_EXTRACTION_PROMPT` → `assert "DO NOT SAVE" in MEMORY_EXTRACTION_PROMPT`
- `test_service.py:4221`: `from llm.service import _MEMORY_EXTRACTION_PROMPT` → `from llm.prompts import MEMORY_EXTRACTION_PROMPT`
- `test_service.py:4245`: `assert first_system["content"] == _MEMORY_EXTRACTION_PROMPT` → `assert first_system["content"] == MEMORY_EXTRACTION_PROMPT`
- `test_service.py:4531`: `from llm.service import _MEMORY_CLEANUP_PROMPT` → `from llm.prompts import MEMORY_CLEANUP_PROMPT`
- `test_service.py:4533`: `assert "keep" not in _MEMORY_CLEANUP_PROMPT.lower()` → `assert "keep" not in MEMORY_CLEANUP_PROMPT.lower()`
- `test_service.py:4534`: `assert "Be aggressive" in _MEMORY_CLEANUP_PROMPT` → `assert "Be aggressive" in MEMORY_CLEANUP_PROMPT`

The docstring references at `test_service.py:134, 5615, 5636, 5660, 5681` ("CHAT_SYSTEM_PROMPT", "CODE_SYSTEM_PROMPT", etc.) are inside docstrings only — leave them as-is, they're descriptive prose. Same for `test_commands.py:560`.

- [ ] **Step 4: Run the suite — all tests must pass**

```bash
make test
```

Expected: pass count is back to baseline (the failures from Task 3 are fixed; `test_prompts.py` adds new passing tests).

- [ ] **Step 5: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 6: Commit Tasks 3 + 5 together as a single atomic commit**

Per the Task 3 Step 6 note, Tasks 3 and 5 are committed together to keep `main` green for auto-deploy.

```bash
git add plugins/llm/src/llm/service.py \
        plugins/llm/tests/test_assistant.py \
        plugins/llm/tests/test_service.py
git commit -m "refactor(llm): migrate service.py and tests to prompts.PROMPTS registry"
```

Verify CI passes (or `make preflight` locally) before pushing.

---

## Task 6: Delete the `assistant.py` re-export shim

**Files:**
- Modify: `plugins/llm/src/llm/assistant.py` (remove the re-export block added in Task 2)
- Modify: `plugins/llm/src/llm/verse/avatar.py:432` (stale comment reference to `CHAT_SYSTEM_PROMPT in assistant.py`)
- Modify: `plugins/llm/tests/verse/test_avatar.py:414` (stale docstring reference to `CHAT_SYSTEM_PROMPT's VERSE MODE block (see test_assistant.py)`)

- [ ] **Step 1: Verify no remaining consumer imports prompts from `llm.assistant`**

Use ripgrep with multi-line matching — the existing imports are parenthesized across multiple lines, so a plain `grep` will miss them:

```bash
rg --multiline --multiline-dotall -n \
  "from \.?assistant import[^)]*?(CHAT|CODE|DRAW|VERSE|REMIND_ACTION)_SYSTEM_PROMPT" \
  plugins/llm/src/ plugins/llm/tests/

rg --multiline --multiline-dotall -n \
  "from (\.|llm\.)assistant import[^)]*?(CHAT|CODE|DRAW|VERSE|REMIND_ACTION)_SYSTEM_PROMPT" \
  plugins/llm/
```

Expected: zero matches. If anything turns up, fix it before continuing — that consumer was missed in Tasks 3–5.

Also confirm no consumer still imports the `_MEMORY_*` constants from `service.py`:

```bash
rg -n "from (\.|llm\.)service import[^)]*_MEMORY_" plugins/llm/
```

Expected: zero matches.

- [ ] **Step 2: Delete the re-export block from `assistant.py`**

Remove the entire `from .prompts import (...)` block (the one added in Task 2 with the `# Prompt constants moved to prompts.py…` comment).

- [ ] **Step 3: Update the two stale references to the prompts' old home**

Open `plugins/llm/src/llm/verse/avatar.py` and find the comment at line 432 (currently: `# CHAT_SYSTEM_PROMPT in assistant.py) so they get the framework`). Update the file reference:

```python
                    # CHAT_SYSTEM_PROMPT in prompts.py) so they get the framework
```

(Read the surrounding line first to keep the rest of the sentence intact — the diff is just `assistant.py` → `prompts.py`.)

Then open `plugins/llm/tests/verse/test_avatar.py` and find the docstring at line 414 (currently: `CHAT_SYSTEM_PROMPT's VERSE MODE block (see test_assistant.py),`). Update both the prompt-name framing and the file pointer — the chat prompt no longer has a "VERSE MODE block" (verse is its own framework now), so the sentence needs a small rewrite, not a one-token swap. Replace with:

```
The chat framework (CHAT_SYSTEM_PROMPT in prompts.py) deliberately
omits verse-mode rules — see test_prompts.py for the separation
invariant — so verse channels must use a separate verse-mode call.
```

(Adjust line breaks to match the surrounding docstring style.)

- [ ] **Step 4: Run the suite**

```bash
make test
```

Expected: all green.

- [ ] **Step 5: Lint and typecheck**

```bash
make lint && make typecheck
```

Expected: clean.

- [ ] **Step 6: Final content-identity verification — the whole point of the refactor is "no behavior change"**

Python's builtin `hash()` is randomized per-interpreter when `PYTHONHASHSEED=random` (the default), so it can't be used for cross-run comparison. Use `hashlib.sha256` (deterministic) and compare against the pre-refactor source.

Capture pre-refactor digests by reading the prompt text from the parent commit of Task 1:

```bash
# From repo root. Find the pre-refactor SHA — the commit just before
# Task 1's "add prompts.py" commit:
PRE_SHA=$(git log --grep="add prompts.py as single source of truth" --pretty=%H -n1)^

# Render each pre-refactor prompt and digest it. Open the parent-commit
# files in an interactive Python session so you don't have to copy
# string concatenations by hand:
git show $PRE_SHA:plugins/llm/src/llm/assistant.py > /tmp/pre_assistant.py
git show $PRE_SHA:plugins/llm/src/llm/service.py   > /tmp/pre_service.py

uv run python <<'PY'
import hashlib, importlib.util, sys

def load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

pre_a = load("/tmp/pre_assistant.py", "pre_assistant")
pre_s = load("/tmp/pre_service.py",   "pre_service")
from llm import prompts as post

pairs = [
    ("chat",              pre_a.CHAT_SYSTEM_PROMPT,           post.PROMPTS["chat"]),
    ("code",              pre_a.CODE_SYSTEM_PROMPT,           post.PROMPTS["code"]),
    ("draw",              pre_a.DRAW_SYSTEM_PROMPT,           post.PROMPTS["draw"]),
    ("verse",             pre_a.VERSE_SYSTEM_PROMPT,          post.PROMPTS["verse"]),
    ("remind_action",     pre_a.REMIND_ACTION_SYSTEM_PROMPT,  post.PROMPTS["remind_action"]),
    ("memory_extraction", pre_s._MEMORY_EXTRACTION_PROMPT,    post.PROMPTS["memory_extraction"]),
    ("memory_cleanup",    pre_s._MEMORY_CLEANUP_PROMPT,       post.PROMPTS["memory_cleanup"]),
]

failures = []
for name, before, after in pairs:
    h_before = hashlib.sha256(before.encode()).hexdigest()[:12]
    h_after  = hashlib.sha256(after.encode()).hexdigest()[:12]
    ok = "OK " if before == after else "DIFF"
    print(f"{ok} {name:>18}: {h_before} -> {h_after}  ({len(before)} -> {len(after)} chars)")
    if before != after:
        failures.append(name)

if failures:
    raise SystemExit(f"Prompts drifted: {failures}")
PY
```

Expected: seven `OK` lines, all `h_before == h_after`. Any `DIFF` means a copy-paste lost or altered a character — locate the byte difference with `python -c "import difflib; ..."` against the pre-refactor source, then restore.

- [ ] **Step 7: Run preflight**

```bash
make preflight
```

Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add plugins/llm/src/llm/assistant.py \
        plugins/llm/src/llm/verse/avatar.py \
        plugins/llm/tests/verse/test_avatar.py
git commit -m "refactor(llm): drop assistant.py prompt re-export shim; update stale refs"
```

---

## Self-Review

**Spec coverage:**
- 5 framework prompts moved ✓ (Task 1)
- 2 memory prompts moved ✓ (Task 1, Task 3)
- Shared `IRC_OUTPUT_FORMAT` building block moved and exposed ✓ (Task 1)
- All consumers migrated: `service.py` (Task 3), `plugin.py` (Task 4), tests (Task 5)
- Backwards-compat shim removed (Task 6)
- Behavior unchanged: enforced by character-identical copy + hash check (Task 6 Step 6) + every existing test still passing (Tasks 3–6)
- Out of scope items are documented and untouched: `config.py` registry overlays not modified; no Profile abstraction introduced; no external prompt files

**Placeholder scan:** No "TBD", "implement later", or vague steps. Each code change shows the before/after literally. The one "# === paste verbatim from … ===" markers in Task 1 are explicit pointers to the source lines being copied — they're instructions to copy specific known content, not unspecified blanks.

**Type consistency:** `PROMPTS: dict[str, str]` is used consistently. Keys (`"chat"`, `"code"`, `"draw"`, `"verse"`, `"remind_action"`, `"memory_extraction"`, `"memory_cleanup"`) are the same in every task. Renamed constants (`_MEMORY_EXTRACTION_PROMPT` → `MEMORY_EXTRACTION_PROMPT`) are renamed identically wherever they appear (Task 1 declaration, Task 3 call sites, Task 5 test assertions).

**Risk notes:**

- Tasks 3 and 5 commit together (Task 5 Step 6) because this repo auto-deploys on green CI on `main`. A red intermediate commit would either block deploy or ship a broken build. Don't split them.
- Behavior preservation is enforced by `hashlib.sha256` comparison against the pre-refactor source in Task 6 Step 6, NOT by Python's builtin `hash()` (which is salted per interpreter via `PYTHONHASHSEED` and useless for cross-run comparison).
- Every assertion in `test_assistant.py:2352–2423` is duplicated up-front in `test_prompts.py` (Task 1 Step 1) — so Task 5's deletion of those tests cannot accidentally weaken the invariant set. The grep cross-check at Task 5 Step 2 catches anything missed.
- After Task 3 collapses `profile_frameworks`, `PROFILE_CODE` and `PROFILE_DRAW` become unused imports in `service.py`. Task 3 Step 3 trims them so ruff doesn't break the commit.
- Task 6 Step 1's grep uses ripgrep `--multiline` because the existing imports are parenthesized across multiple lines — a plain `grep` would silently miss them.
