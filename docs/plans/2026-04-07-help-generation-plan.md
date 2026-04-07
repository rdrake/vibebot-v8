# Help Generation System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate `getPluginHelp()` and the HTML help page from actual command metadata so they can never drift out of sync.

**Architecture:** A module-level command registry (list of dataclass entries) defines each user-facing command's name, args, description, examples, and category. `getPluginHelp()` iterates the registry to build the IRC summary. The HTML help page is generated from the same registry. The hardcoded `HELP_HTML_TEMPLATE` command blocks are replaced by dynamic generation; only the CSS/layout shell remains static.

**Tech Stack:** Python, Limnoria, pytest

**Prerequisite:** This plan should be executed AFTER the command surface plan (2026-04-07-command-surface-plan.md), since it needs to know the final command set.

**Before starting:** Verify the prerequisite was completed — grep for `def instruct` (should exist) and `def picard` (should not) in `plugin.py`.

---

### Task 1: Define the command registry

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

Add to `test_plugin.py`:

```python
class TestCommandRegistry:
    """Tests for the command metadata registry."""

    def test_registry_contains_all_commands(self):
        """GIVEN command registry WHEN checked THEN contains all user-facing commands."""
        from llm.plugin import COMMAND_REGISTRY
        names = {cmd.name for cmd in COMMAND_REGISTRY}
        expected = {"ask", "code", "draw", "forget", "memories", "instruct", "remind", "usage"}
        assert names == expected

    def test_registry_entries_have_required_fields(self):
        """GIVEN command registry WHEN checked THEN all entries have name, args, description."""
        from llm.plugin import COMMAND_REGISTRY
        for cmd in COMMAND_REGISTRY:
            assert cmd.name, "name is required"
            assert cmd.description, "description is required"
            assert cmd.category in ("generation", "memory", "utility")

    def test_registry_entries_have_examples(self):
        """GIVEN command registry WHEN checked THEN all entries have at least one example."""
        from llm.plugin import COMMAND_REGISTRY
        for cmd in COMMAND_REGISTRY:
            assert cmd.examples, f"{cmd.name} needs at least one example"
```

**Step 2: Run tests to verify they fail**

```bash
make test
```

**Step 3: Implement the registry**

Add near the top of `plugin.py` (after the imports, before `HELP_HTML_TEMPLATE`):

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CommandInfo:
    """Metadata for a user-facing command, used to generate help."""

    name: str
    args: str
    description: str
    examples: tuple[str, ...]
    category: str  # "generation", "memory", "utility"


COMMAND_REGISTRY: tuple[CommandInfo, ...] = (
    CommandInfo(
        name="ask",
        args="<question>",
        description=(
            "Ask the AI a question. Supports conversation context "
            "(follow-up questions) and vision (include image URLs)."
        ),
        examples=(
            "%ask What is the capital of France?",
            "%ask Describe this: https://example.com/image.jpg",
            "%ask And what about Germany?  (follow-up using context)",
        ),
        category="generation",
    ),
    CommandInfo(
        name="code",
        args="<request>",
        description=(
            "Generate code based on your request. "
            "Code is saved to an HTTP link with syntax highlighting."
        ),
        examples=(
            "%code Python function to calculate fibonacci numbers",
            "%code Now add memoization to that",
        ),
        category="generation",
    ),
    CommandInfo(
        name="draw",
        args="<prompt>",
        description="Generate an image from a text description.",
        examples=(
            "%draw A sunset over mountains in watercolor style",
            "%draw A cyberpunk cityscape at night",
        ),
        category="generation",
    ),
    CommandInfo(
        name="forget",
        args="[channel]",
        description=(
            "Clear your volatile memory (conversation context) "
            "for the current or specified channel."
        ),
        examples=("%forget", "%forget #channel"),
        category="memory",
    ),
    CommandInfo(
        name="memories",
        args="[del <id> | edit <id> <text> | clear | cleanup]",
        description=(
            "Manage your non-volatile memory (stored facts the bot "
            "remembers about you across conversations)."
        ),
        examples=(
            "%memories",
            "%memories delete 3",
            "%memories edit 5 corrected fact",
            "%memories clear",
        ),
        category="memory",
    ),
    CommandInfo(
        name="instruct",
        args="[<instruction> | clear]",
        description=(
            "Set persistent instructions that shape how %ask responds to you. "
            "Your instruction is prepended to the system prompt."
        ),
        examples=(
            "%instruct You are Captain Picard. Respond in character.",
            "%instruct Respond only in haiku",
            "%instruct clear",
            "%instruct",
        ),
        category="memory",
    ),
    CommandInfo(
        name="remind",
        args="[<text> | list | del <id> | clear]",
        description="Set and manage reminders using natural language.",
        examples=(
            "%remind in 30 minutes check the build",
            "%remind list",
            "%remind delete abc1",
            "%remind clear",
        ),
        category="utility",
    ),
    CommandInfo(
        name="usage",
        args="[nick | #channel]",
        description="Show API usage statistics.",
        examples=("%usage", "%usage someone", "%usage #channel"),
        category="utility",
    ),
)
```

**Step 4: Run tests to verify they pass**

```bash
make test
```

**Step 5: Run lint/typecheck**

```bash
make lint && make typecheck
```

**Step 6: Commit**

```bash
git commit -m "feat: add command metadata registry for help generation"
```

---

### Task 2: Generate `getPluginHelp()` from registry

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

```python
def test_get_plugin_help_lists_all_commands(self, mock_irc, mocker):
    """GIVEN plugin WHEN getPluginHelp called THEN lists all registered commands."""
    from llm.plugin import COMMAND_REGISTRY, LLM
    # ... plugin setup ...
    help_text = plugin.getPluginHelp()
    for cmd in COMMAND_REGISTRY:
        assert cmd.name in help_text, f"{cmd.name} missing from help"
```

**Step 2: Run to verify it fails** (currently only lists ask, code, draw, forget)

**Step 3: Rewrite `getPluginHelp()`**

Replace the hardcoded string (lines 1053-1066) with:

```python
def getPluginHelp(self) -> str:  # noqa: N802
    """Return plugin help with dynamic documentation URL."""
    url = self._get_help_url()
    names = ", ".join(cmd.name for cmd in COMMAND_REGISTRY)
    return (
        _(
            "AI-powered commands using LiteLLM. "
            "Commands: %s. "
            "Full documentation: %s"
        )
        % (names, url)
    )
```

**Step 4: Run tests**

```bash
make test
```

**Step 5: Commit**

```bash
git commit -m "feat: generate getPluginHelp() from command registry"
```

---

### Task 3: Generate HTML help page from registry

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

Note: The current codebase exports `HELP_HTML_TEMPLATE` (string) and `_HELP_HTML_BYTES` (private, bytes). After this change, keep `HELP_HTML_TEMPLATE` as the public export name (now dynamically generated) so existing test imports continue to work. `_HELP_HTML_BYTES` stays as `HELP_HTML_TEMPLATE.encode("utf-8")`.

```python
def test_html_help_lists_all_commands():
    """GIVEN HELP_HTML_TEMPLATE WHEN checked THEN contains all registered commands."""
    from llm.plugin import COMMAND_REGISTRY, HELP_HTML_TEMPLATE
    for cmd in COMMAND_REGISTRY:
        assert f"%{cmd.name}" in HELP_HTML_TEMPLATE, f"%{cmd.name} missing from HTML help"

def test_html_help_groups_by_category():
    """GIVEN HTML help WHEN parsed THEN has generation, memory, utility sections."""
    from llm.plugin import HELP_HTML_TEMPLATE
    assert "Generation" in HELP_HTML_TEMPLATE
    assert "Memory" in HELP_HTML_TEMPLATE
    assert "Utility" in HELP_HTML_TEMPLATE
```

**Step 2: Run to verify it fails**

**Step 3: Replace hardcoded HTML command blocks with generation**

Keep the CSS/layout shell as a constant (`_HELP_HTML_HEAD` and `_HELP_HTML_FOOT`). Generate the command blocks dynamically.

Add a module-level function:

**IMPORTANT:** The `args` field contains angle brackets (e.g., `<question>`) which must be HTML-escaped or they become actual HTML tags. Use `html.escape()` from the standard library.

```python
import html as _html

_CATEGORY_LABELS = {"generation": "Generation", "memory": "Memory", "utility": "Utility"}

def _build_help_html() -> str:
    """Build the command sections of the HTML help page from the registry."""
    sections: list[str] = []
    for category in ("generation", "memory", "utility"):
        cmds = [c for c in COMMAND_REGISTRY if c.category == category]
        if not cmds:
            continue
        sections.append(f'<h2>{_CATEGORY_LABELS[category]}</h2>')
        for cmd in cmds:
            escaped_args = _html.escape(cmd.args)
            sections.append(
                f'<h3><code class="command">%{cmd.name}</code> '
                f'<span class="param">{escaped_args}</span></h3>'
            )
            sections.append(f"<p>{_html.escape(cmd.description)}</p>")
            example_lines = "\n".join(
                f'<span class="example">{_html.escape(ex)}</span>'
                for ex in cmd.examples
            )
            sections.append(f"<pre><code>{example_lines}</code></pre>")
    return "\n".join(sections)
```

Then build the full HTML from head + generated sections + features section + foot. Replace `HELP_HTML_TEMPLATE` and `HELP_HTML_BYTES` with the generated result.

**Step 4: Keep features and configuration sections**

Move the existing Features and Configuration HTML sections into the footer constant so they're still present but no longer mixed with command blocks.

**Step 5: Update existing tests that import `HELP_HTML_TEMPLATE`**

Two test files reference the old template:

1. `test_html_output.py` — `TestHelpPageContent` class (4 tests) imports `HELP_HTML_TEMPLATE` and asserts specific strings. Update assertions to match the new generated content:
   - `"%ask" in HELP_HTML_TEMPLATE` — still true, keep
   - `"Conversation Context" in HELP_HTML_TEMPLATE` — update to match new terminology ("Volatile Memory")
   - Other structural assertions (DOCTYPE, html tags, viewport) — still true, keep

2. `test_plugin.py` — `TestHTTPCallbackServeHelpPage` class (2 tests at lines 293 and 333) imports `HELP_HTML_TEMPLATE` and checks `HELP_HTML_TEMPLATE.encode("utf-8")`. These still work as long as the public export name stays `HELP_HTML_TEMPLATE`.

**Step 6: Run tests**

```bash
make test
```

**Step 7: Run preflight**

```bash
make preflight
```

**Step 8: Commit**

```bash
git commit -m "feat: generate HTML help page from command registry"
```

---

### Task 4: Add CI test to prevent drift

**Files:**
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the test**

Limnoria identifies commands via `isCommandMethod()` on the `Commands` base class. It checks:
1. `name == canonicalName(name)` (lowercase, alphanumeric only) — this filters out `getPluginHelp`, `invalidCommand`, `inFilter`, `outFilter`, `doPrivmsg`, etc.
2. The method's code object has args `['self', 'irc', 'msg', 'args']`

We replicate this logic to find all commands and assert they're in the registry:

```python
import inspect
from supybot.callbacks import canonicalName

def test_all_wrapped_commands_in_registry():
    """GIVEN plugin class WHEN checking command methods THEN all are in registry.

    This test prevents adding a new command to plugin.py without updating
    the command registry. It uses the same introspection as Limnoria's
    isCommandMethod() to find all commands.
    """
    from llm.plugin import COMMAND_REGISTRY, LLM
    registry_names = {cmd.name for cmd in COMMAND_REGISTRY}
    command_args = ["self", "irc", "msg", "args"]

    for name in dir(LLM):
        if name.startswith("_"):
            continue
        if name != canonicalName(name):
            continue  # filters getPluginHelp, invalidCommand, inFilter, etc.
        obj = getattr(LLM, name, None)
        if not inspect.isfunction(obj):
            continue
        if inspect.getargs(obj.__code__)[0] == command_args:
            assert name in registry_names, (
                f"Command '{name}' is registered with Limnoria but missing from "
                f"COMMAND_REGISTRY. Add it to keep help in sync."
            )
```

**Step 2: Run tests**

```bash
make test
```

**Step 3: Commit**

```bash
git commit -m "test: add drift-prevention test for command registry completeness"
```

---

### Task 5: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `plugins/llm/README.md`

**Step 1: Update command tables**

In all three files, update the IRC commands table to reflect the final command surface:

| Command | Description |
|---------|-------------|
| `%ask <question>` | Ask with context, vision, and optional instructions |
| `%code <request>` | Generate code with HTTP link output |
| `%draw <prompt>` | Generate image (account required) |
| `%forget [channel]` | Clear volatile memory (conversation context) |
| `%memories [subcommand]` | Manage non-volatile memory (stored facts) |
| `%instruct [text \| clear]` | Set persistent instructions for ask |
| `%remind [text \| list \| del \| clear]` | Set and manage reminders |
| `%usage [nick \| #channel]` | View API usage statistics |

Remove `%picard`, `%remindme`, `%reminders`, `%unremind` from all references.

**Step 2: Update feature descriptions**

Replace "conversation context (memory)" with "volatile memory" and "long-term memory" with "non-volatile memory" throughout.

**Step 3: Run preflight**

```bash
make preflight
```

**Step 4: Commit**

```bash
git commit -m "docs: update command tables and terminology for command surface overhaul"
```

---

### Task 6: Final verification

**Step 1: Search for orphaned references**

```bash
grep -rn 'remindme\|unremind\|picard\|_extract_raw_arg\|HELP_HTML_TEMPLATE' plugins/llm/src/ plugins/llm/tests/ --include="*.py"
```

Fix any real orphans.

**Step 2: Run full preflight**

```bash
make preflight
```

Expected: format clean, lint clean, typecheck clean, all tests pass, coverage >= 80%.

**Step 3: Final commit**

```bash
git commit -m "chore: final cleanup for command UX overhaul"
```
