---
status: ready-to-execute
date: 2026-05-02
phase: 2
task: 1 (per-command mutation classification gate)
design_plan: docs/plans/2026-05-02-limnoria-bridge-phase-2-plan.md
predecessor_design: docs/plans/2026-05-02-limnoria-tool-bridge-plan.md
predecessor_implementation: docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md
---

# Limnoria Tool Bridge — Phase 2 Task 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a per-command mutation gate to the Limnoria bridge so the LLM only sees mutating commands when the operator has explicitly opted in via a new `bridgeAllowMutating` channel registry value. Bridge enumeration hides mutating commands by default; bridge dispatch rejects them with an error envelope as defense in depth. Ship the canonical `MUTATING_COMMANDS` set committed alongside the gate so the classification is auditable, not derived at runtime.

**Architecture:** One PR. New module-level constant `MUTATING_COMMANDS` in `plugins/llm/src/llm/limnoria_bridge.py` (mirrors `DENY_COMMANDS` shape). New channel registry value `bridgeAllowMutating` in `plugins/llm/src/llm/config.py`. Two enforcement points wired off the same registry value: (a) `enumerate_commands` filters mutating leaves before the tool description is rendered; (b) `dispatch` rejects mutating leaves with `{"error": "denied: write commands disabled"}` even if a hallucinated call slips through. One footer line on the bridge tool description when the gate is closed and the channel allowlist contains a plugin with both kinds of commands. No new abstractions; no changes to `assistant_completion`'s `extra_tools` / `extra_handlers` plumbing.

**Tech stack:** Python 3.14, Limnoria (`supybot.callbacks`, `supybot.conf`, `supybot.registry`), pytest. All build/test commands run via `uv run`. AGENTS.md:22-24 says: after editing Python files, run `make lint` and `make typecheck`; before considering the work complete, run `make preflight`. These are agent obligations, not auto-hooks — the implementer must invoke them explicitly. The Step 5 commit blocks below name the gates that need to pass before each commit.

**Naming note:** the gate parameter is `allow_mutating` in function signatures and `bridgeAllowMutating` as the channel registry key. Both names are load-bearing — the registry key is what operators type into IRC `config channel` commands.

---

## Pre-flight (do first, do not skip)

### Task 0: Verify codebase facts before touching anything

**Step 0.1: Confirm the integration points the plan refers to are still where the design says.**

Run these and read the matched lines:

```bash
grep -n "DENY_PLUGINS\|DENY_COMMANDS\|class BufferingIrcProxy\|def enumerate_commands\|def dispatch" plugins/llm/src/llm/limnoria_bridge.py
grep -n "_build_bridge_tool\|bridgeAllowedPlugins\|bridgeEnabled" plugins/llm/src/llm/plugin.py
grep -n "bridgeEnabled\|bridgeAllowedPlugins\|bridgeDebugInChannel" plugins/llm/src/llm/config.py
```

Expected (verified 2026-05-02):
- `limnoria_bridge.py`: `DENY_PLUGINS` ~line 25, `DENY_COMMANDS` ~line 41, `BufferingIrcProxy` ~line 57, `enumerate_commands` ~line 95, `dispatch` ~line 139.
- `plugin.py`: `_build_bridge_tool` ~line 1569, `registryValue("bridgeAllowedPlugins", channel)` call ~line 1583.
- `config.py`: `bridgeEnabled` ~line 854, `bridgeAllowedPlugins` ~line 865, `bridgeDebugInChannel` ~line 878.

If line numbers have drifted (another commit landed first), update the references below before implementing — the *symbols* are the truth, not the line numbers.

**Step 0.2: Confirm baseline tests are green.**

```bash
uv run pytest plugins/llm/tests -q
```

Expected: all green. If anything fails, stop and report — fixing pre-existing failures is not in scope.

**Step 0.3: Confirm no `bridgeAllowMutating` symbol already exists (it must not).**

```bash
grep -rn "bridgeAllowMutating\|MUTATING_COMMANDS\|allow_mutating" plugins/llm/
```

Expected: zero matches. If matches exist, an earlier draft has landed and this plan needs reconciliation before continuing.

**Step 0.4: Commit nothing. This task is read-only.**

---

## A — Foundation: the `MUTATING_COMMANDS` constant

### Task A1: Land the canonical `MUTATING_COMMANDS` frozenset

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py` (add constant near `DENY_COMMANDS` at line 41-54)
- Modify: `plugins/llm/tests/test_limnoria_bridge.py` (add membership assertions to the existing module-level test at line 8)

**Background — classification methodology:**

Each entry below is sourced by reading the canonical command method's signature in the stock plugin. A method is a user-facing command if it takes `(self, irc, msg, args, ...)` — that's the convention `wrap()` enforces. We classify by side-effect on persistent state or third-party identity, not by network access:

- **Mutating:** writes to a plugin-owned database, sends a message *to a different user* (PM/notice), reads-and-marks-as-side-effect (e.g. `setRead`), or alters operator-visible config.
- **Read-only:** computes a reply from existing state, including pure transforms, look-ups, and outbound HTTP that doesn't change persistent state. (`Web.title` hits the network but doesn't mutate anything; it stays read-only.)

The set is keyed `(canonical_plugin_name, leaf_command)` — both lowercase — to match the existing `DENY_COMMANDS` shape (`limnoria_bridge.py:41-54`) and so the lookup in `enumerate_commands`/`dispatch` can reuse the canonical-name pattern at `limnoria_bridge.py:122` and `:177`.

**Canonical entries — Phase 2 Task 2 default allowlist set:**

| Plugin | Leaf | Source citation | Why mutating |
| --- | --- | --- | --- |
| Misc | `tell` | `.venv/lib/python3.14/site-packages/supybot/plugins/Misc/plugin.py:629` | Sends a PM/notice on behalf of the user via `_tell` (`Misc/plugin.py:603-626`). |
| Misc | `noticetell` | `Misc/plugin.py:639` | Same as `tell`, but as a NOTICE (`_tell` with `notice=True`). |
| Later | `tell` | `.venv/lib/python3.14/site-packages/supybot/plugins/Later/plugin.py:163` | Stores a pending offline message in the Later DB. |
| Later | `remove` | `Later/plugin.py:216` | Deletes a pending offline message. |
| Later | `undo` | `Later/plugin.py:230` | Removes the last `tell` the user queued. |
| Note | `send` | `.venv/lib/python3.14/site-packages/supybot/plugins/Note/plugin.py:183` | Inserts a row into the Note DB. |
| Note | `reply` | `Note/plugin.py:199` | Inserts a row into the Note DB (reply chain). |
| Note | `unsend` | `Note/plugin.py:223` | Deletes a row from the Note DB. |
| Karma | `clear` | `.venv/lib/python3.14/site-packages/supybot/plugins/Karma/plugin.py:398` | Resets karma rows; gated by `op` capability already, but still write. |
| Karma | `dump` | `Karma/plugin.py:409` | Writes karma DB to a file in the bot's data dir; owner-only via capability, still write. |
| Karma | `load` | `Karma/plugin.py:422` | Loads karma DB from a file; mutates DB; owner-only via capability, still write. |
| QuoteGrabs | `grab` | `.venv/lib/python3.14/site-packages/supybot/plugins/QuoteGrabs/plugin.py:271` | Inserts a quote-grab row. |
| QuoteGrabs | `ungrab` | `QuoteGrabs/plugin.py:304` | Deletes a quote-grab row. |
| RSS | `add` | `.venv/lib/python3.14/site-packages/supybot/plugins/RSS/plugin.py:564` | Registers a feed (config + dynamic command). |
| RSS | `remove` | `RSS/plugin.py:577` | Unregisters a feed. |
| RSS | `rss` | `RSS/plugin.py:698` | Reads feed content, but `update_feed_if_needed` (`RSS/plugin.py:396`) can call `update_feed` → `announce_feed` (`RSS/plugin.py:434`), which queues IRC PRIVMSG/NOTICE to every channel subscribed to the feed (`RSS/plugin.py:553-557`). LLM-triggered `rss <feed>` could push entries to third-party channels. |
| RSS | `info` | `RSS/plugin.py:741` | Same `update_feed_if_needed` side effect as `rss`. |
| DDG | *(none)* | `.venv/lib/python3.14/site-packages/supybot/plugins/DDG/plugin.py:149` | `search` is the only command; pure HTTP read. |

**Read-only leaves in the default-allowlist plugins (do NOT enter `MUTATING_COMMANDS`):**

| Plugin | Read-only leaves | Source |
| --- | --- | --- |
| Misc | `list`, `apropos`, `help`, `version`, `source`, `last`, `ping`, `completenick` | `Misc/plugin.py:200, 270, 294, 349, 391, 465, 649, 657` |
| Time | `seconds`, `at`, `until`, `ctime`, `time`, `elapsed`, `tztime`, `ddate` | `.venv/lib/python3.14/site-packages/supybot/plugins/Time/plugin.py:84, 122, 145, 164, 175, 197, 207, 232` — all pure compute |
| Math | `base`, `calc`, `icalc`, `rpn`, `convert`, `units` | `.venv/lib/python3.14/site-packages/supybot/plugins/Math/plugin.py:52, 129, 156, 193, 242, 285` — all pure compute |
| Utilities | `ignore`, `success`, `last`, `echo`, `shuffle`, `sort`, `sample`, `countargs` | `.venv/lib/python3.14/site-packages/supybot/plugins/Utilities/plugin.py:47, 58, 71, 85, 98, 108, 119, 132` |
| Seen | `seen`, `any`, `last`, `user`, `since` | `.venv/lib/python3.14/site-packages/supybot/plugins/Seen/plugin.py:239, 254, 303, 338, 352` |
| Web | `headers`, `location`, `doctype`, `size`, `title`, `urlquote`, `urlunquote` | `.venv/lib/python3.14/site-packages/supybot/plugins/Web/plugin.py:315, 336, 352, 374, 406, 427, 435` |
| Later | `notes` | `Later/plugin.py:192` — lists pending notes the *caller* set, no side effect |
| Note | `note`, `next`, `search`, `list` | `Note/plugin.py:255, 373, 285, 332` — `note`/`next` mark notes read via `db.setRead`, but the side effect is a read-receipt scoped to the caller's own notes (`Note/plugin.py:265-267` enforces `note.frm/note.to == user.id`); benign, so classified read-only |
| Karma | `karma`, `most` | `Karma/plugin.py:317, 379` |
| QuoteGrabs | `quote`, `list`, `random`, `say`, `get`, `search` | `QuoteGrabs/plugin.py:322, 336, 359, 377, 391, 405` |
| RSS | *(none in this column — see "Ambiguous classifications" #3 below for `rss`/`info` and dynamic feed-name leaves)* | — |
| DDG | `search` | `DDG/plugin.py:149` |

**Forward-look entries — Phase 2+ candidate plugins (Quote, Todo, Factoids, Scheduler):**

These plugins are **not** in Phase 2 Task 2's default allowlist but are likely candidates for later. We classify them now so when they get allowlisted, the gate already covers them:

| Plugin | Leaf | Source citation | Why mutating |
| --- | --- | --- | --- |
| Quote | `add` | `.venv/lib/python3.14/site-packages/supybot/plugins/__init__.py:376` (inherited from `ChannelIdDatabasePlugin`) | Inserts row. |
| Quote | `remove` | `supybot/plugins/__init__.py:391` | Deletes row. |
| Quote | `change` | `supybot/plugins/__init__.py:490` | Edits row via sed-style replacer. |
| Quote | `replace` | `.venv/lib/python3.14/site-packages/supybot/plugins/Quote/plugin.py:54` | Overwrites quote text by id. |
| Todo | `add` | `.venv/lib/python3.14/site-packages/supybot/plugins/Todo/plugin.py:190` | Inserts todo row. |
| Todo | `remove` | `Todo/plugin.py:207` | Deletes todo row. |
| Todo | `setpriority` | `Todo/plugin.py:269` | Mutates todo row priority. |
| Todo | `change` | `Todo/plugin.py:283` | Edits todo text. |
| Factoids | `learn` | `.venv/lib/python3.14/site-packages/supybot/plugins/Factoids/plugin.py:281` | Creates factoid. |
| Factoids | `alias` | `Factoids/plugin.py:477` | Creates a key alias. |
| Factoids | `lock` | `Factoids/plugin.py:591` | Sets locked flag. |
| Factoids | `unlock` | `Factoids/plugin.py:609` | Clears locked flag. |
| Factoids | `forget` | `Factoids/plugin.py:648` | Deletes factoid. |
| Factoids | `change` | `Factoids/plugin.py:772` | Edits factoid via sed-style replacer. |
| Factoids | `whatis` | `Factoids/plugin.py:447` | Reads factoid value but `_replyFactoids` calls `_updateRank` (`Factoids/plugin.py:372-383, 397, 420`), which `UPDATE`s `relations.usage_count` whenever `keepRankInfo` is True. `keepRankInfo` defaults to `True` (`Factoids/config.py:85-87`), so by default `whatis` writes the DB on every call. |
| Scheduler | `add` | `.venv/lib/python3.14/site-packages/supybot/plugins/Scheduler/plugin.py:184` | Schedules a one-shot command. |
| Scheduler | `remind` | `Scheduler/plugin.py:199` | Schedules a one-shot reminder. |
| Scheduler | `remove` | `Scheduler/plugin.py:212` | Cancels a scheduled event. |
| Scheduler | `repeat` | `Scheduler/plugin.py:247` | Schedules a periodic command. |

Read-only leaves in the forward-look plugins:

| Plugin | Read-only leaves | Source |
| --- | --- | --- |
| Quote | `search`, `get`, `stats`, `random` | `supybot/plugins/__init__.py:412, 476, 508`; `Quote/plugin.py:41` |
| Todo | `todo`, `search` | `Todo/plugin.py:138, 232` |
| Factoids | `rank`, `random`, `info`, `search` | `Factoids/plugin.py:546, 706, 733, 801` (`whatis` excluded — see mutating table above) |
| Scheduler | `list` | `Scheduler/plugin.py:273` |

**Ambiguous classifications:**

1. **`Note.note` / `Note.next` — classified read-only.** Both call `db.setRead` as a side effect (`Note/plugin.py:271, 394`), but the side effect is a read-receipt scoped to the caller's own notes — the plugin enforces `note.frm/note.to == user.id` at `Note/plugin.py:265-267`, so the blast radius is "the calling user's own notes." Hiding these behind the gate would make `Note` effectively unusable for reading when the gate is closed (the LLM could `search` and `list` to get IDs, but couldn't read note bodies). Read-only classification matches user intuition and the gate stays meaningful — `send`/`reply`/`unsend` remain hidden.

2. **`Karma.dump` / `Karma.load` — owner-only writes.**
   Both already require `owner` capability (`Karma/plugin.py:419, 432`), so they would be filtered by `enumerate_commands`'s capability check (`limnoria_bridge.py:124-126`) for non-owner users anyway. Including them in `MUTATING_COMMANDS` is belt-and-suspenders: it protects an owner who runs the bridge from accidentally letting the LLM dump or restore the karma DB. The safe call — keep them classified mutating.

3. **RSS subcommand group `announce` — out of scope for Task 1 (verified multi-word).**
   The RSS plugin uses a nested `class announce(callbacks.Commands)` (`RSS/plugin.py:601`) for `add` / `remove` / `list` / `channels`. Supybot's `callbacks.Commands.listCommands` (`callbacks.py:1554-1560`) prefixes nested-Commands leaves with the nested-class name and a space, so `cb.listCommands()` on an `RSS` instance returns the announce sub-leaves as multi-word strings: `"announce add"`, `"announce remove"`, `"announce list"`, `"announce channels"` — **not** a single `"announce"` leaf. RSS itself delegates `listCommands` to the parent (`RSS/plugin.py:317-318`); the prefixing happens in the parent's loop.

   The bridge's enumerate/dispatch path is not designed for multi-word leaves: `enumerate_commands` would yield `BridgeCommand(command="announce add")`, and `dispatch` would call `cb.isCommandMethod("announce add")` / `cb.getCommandMethod(["announce add"])`, neither of which reliably resolves nested-Commands leaves under the current bridge implementation (also note `canonicalName("announce add") != "announce add"`, breaking any direct lookup against `MUTATING_COMMANDS`).

   **Decision for v1:** do **not** add any `("rss", "announce …")` tuple to `MUTATING_COMMANDS`. The classification can't bite — `enumerate_commands` would never construct the matching key, and `dispatch` couldn't be reached for these leaves anyway. Multi-word/nested-Commands leaves are out of scope for Task 1 and tracked separately; the operator-doc known-limitation note in E1 is *not* about this gate but about the underlying bridge gap. Document the gap there; don't pretend the gate covers it.

   **Verified** (read 2026-05-02 from `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1554-1560` and `supybot/plugins/RSS/plugin.py:317-318, 601`): nested-Commands leaves are returned as space-prefixed multi-word strings. Resolved — no implementer follow-up required for this point.

4. **`Misc.tell` / `Misc.noticetell` vs `Later.tell`.**
   Both plugins have a `tell` leaf. They are different methods on different plugins, and the (canonical, leaf) tuple distinguishes them: `("misc", "tell")` and `("later", "tell")`. No collision risk; the lookup is plugin-scoped.

**`DENY_COMMANDS` extension for `Utilities.let` (B2 — pre-existing bypass, not a mutation):**

`Utilities.let` (`.venv/lib/python3.14/site-packages/supybot/plugins/Utilities/plugin.py:156-178`) is structurally identical to the already-denied `Utilities.apply` (`Utilities/plugin.py:150-155`): it tokenizes arbitrary user-supplied text and redispatches it through `self.Proxy(irc, fake_msg, tokens)`, effectively letting the caller invoke *any* command on *any* plugin under their identity. That sidesteps every `DENY_PLUGINS` / `DENY_COMMANDS` / `MUTATING_COMMANDS` filter that the bridge applies to first-level dispatch.

This is an arbitrary-command-redispatch bypass, not a mutation per se — so the right home is `DENY_COMMANDS` (hard deny, no gate to flip), not `MUTATING_COMMANDS`. Phase 1 missed it because `apply` was the obvious case; `let` has the same shape.

**Action in Task A1, Step 3a (below):** when adding `MUTATING_COMMANDS`, also add a single new entry `("utilities", "let")` to `DENY_COMMANDS` (`limnoria_bridge.py:41-54`). The new test assertion in the existing `test_module_exposes_deny_lists_and_dataclass` test (already shown in Step 1) verifies it. No separate registry value, no separate test function.

**Step 1: Write the failing tests.**

Leave the existing `test_module_exposes_deny_lists_and_dataclass` body (`plugins/llm/tests/test_limnoria_bridge.py:8-29`) **unchanged** apart from one new assertion (the `("utilities", "let") in lb.DENY_COMMANDS` line — see "DENY_COMMANDS extension for `Utilities.let`" below). Then **add** three new tests *below* it that exercise the new `MUTATING_COMMANDS` constant. The shape of the existing test is reproduced here for context:

```python
def test_module_exposes_deny_lists_and_dataclass():
    from llm import limnoria_bridge as lb

    assert isinstance(lb.DENY_PLUGINS, frozenset)
    assert "LLM" in lb.DENY_PLUGINS
    assert "Owner" in lb.DENY_PLUGINS
    assert "Admin" in lb.DENY_PLUGINS
    assert "Config" in lb.DENY_PLUGINS
    assert "Channel" in lb.DENY_PLUGINS
    assert "User" in lb.DENY_PLUGINS

    assert isinstance(lb.DENY_COMMANDS, frozenset)
    assert ("misc", "more") in lb.DENY_COMMANDS
    assert ("misc", "clearmores") in lb.DENY_COMMANDS
    assert ("web", "fetch") in lb.DENY_COMMANDS
    assert ("utilities", "apply") in lb.DENY_COMMANDS
    assert ("utilities", "let") in lb.DENY_COMMANDS  # NEW (B2 — see DENY_COMMANDS extension below)

    cmd = lb.BridgeCommand(
        plugin="Misc", command="ping", arg_syntax="", description="takes no arguments"
    )
    assert cmd.plugin == "Misc"
    assert cmd.command == "ping"


def test_mutating_commands_covers_default_allowlist_writes():
    """Every mutating command in the Phase 2 Task 2 default allowlist must be
    in MUTATING_COMMANDS. Reads must NOT be in it. Tuples are
    (canonical_plugin_lowercase, leaf_lowercase) — same shape as DENY_COMMANDS."""
    from llm import limnoria_bridge as lb

    assert isinstance(lb.MUTATING_COMMANDS, frozenset)

    expected_mutating = {
        ("misc", "tell"),
        ("misc", "noticetell"),
        ("later", "tell"),
        ("later", "remove"),
        ("later", "undo"),
        ("note", "send"),
        ("note", "reply"),
        ("note", "unsend"),
        ("karma", "clear"),
        ("karma", "dump"),
        ("karma", "load"),
        ("quotegrabs", "grab"),
        ("quotegrabs", "ungrab"),
        ("rss", "add"),
        ("rss", "remove"),
        ("rss", "rss"),    # update_feed_if_needed → announce_feed → IRC writes
        ("rss", "info"),   # same update_feed_if_needed side effect
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    # Reads in the same plugins must NOT be classified mutating.
    expected_read_only = {
        ("misc", "ping"),
        ("misc", "last"),
        ("misc", "version"),
        ("time", "time"),
        ("math", "calc"),
        ("utilities", "echo"),
        ("seen", "seen"),
        ("seen", "last"),
        ("web", "title"),
        ("later", "notes"),
        ("note", "search"),
        ("note", "list"),
        ("note", "note"),
        ("note", "next"),
        ("karma", "karma"),
        ("karma", "most"),
        ("quotegrabs", "quote"),
        ("quotegrabs", "random"),
        ("ddg", "search"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_mutating_commands_covers_forward_look_writes():
    """Quote/Todo/Factoids/Scheduler are not yet in the default allowlist but
    we classify them now so the gate is correct when they're added later."""
    from llm import limnoria_bridge as lb

    expected_mutating = {
        ("quote", "add"),
        ("quote", "remove"),
        ("quote", "change"),
        ("quote", "replace"),
        ("todo", "add"),
        ("todo", "remove"),
        ("todo", "setpriority"),
        ("todo", "change"),
        ("factoids", "learn"),
        ("factoids", "alias"),
        ("factoids", "lock"),
        ("factoids", "unlock"),
        ("factoids", "forget"),
        ("factoids", "change"),
        ("factoids", "whatis"),  # _updateRank writes when keepRankInfo=True (default)
        ("scheduler", "add"),
        ("scheduler", "remind"),
        ("scheduler", "remove"),
        ("scheduler", "repeat"),
    }
    assert expected_mutating <= lb.MUTATING_COMMANDS

    expected_read_only = {
        ("quote", "search"),
        ("quote", "get"),
        ("quote", "stats"),
        ("quote", "random"),
        ("todo", "todo"),
        ("todo", "search"),
        ("factoids", "random"),
        ("factoids", "info"),
        ("factoids", "rank"),
        ("factoids", "search"),
        ("scheduler", "list"),
    }
    assert expected_read_only.isdisjoint(lb.MUTATING_COMMANDS)


def test_mutating_commands_lowercase_invariant():
    """Match the DENY_COMMANDS shape — both elements lowercase."""
    from llm import limnoria_bridge as lb

    for plugin, leaf in lb.MUTATING_COMMANDS:
        assert plugin == plugin.lower(), plugin
        assert leaf == leaf.lower(), leaf
```

**Step 2: Run the tests; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py::test_mutating_commands_covers_default_allowlist_writes -v
uv run pytest plugins/llm/tests/test_limnoria_bridge.py::test_mutating_commands_covers_forward_look_writes -v
uv run pytest plugins/llm/tests/test_limnoria_bridge.py::test_mutating_commands_lowercase_invariant -v
```

Expected: 3 FAIL with `AttributeError: module 'llm.limnoria_bridge' has no attribute 'MUTATING_COMMANDS'`.

**Step 3: Update `limnoria_bridge.py` — extend `DENY_COMMANDS` and add `MUTATING_COMMANDS`.**

**Step 3a:** add `("utilities", "let")` to the existing `DENY_COMMANDS` frozenset (`limnoria_bridge.py:41-54`), next to `("utilities", "apply")`. Keep the surrounding comment style. This is the B2 fix (see "DENY_COMMANDS extension for `Utilities.let`" above).

**Step 3b:** insert `MUTATING_COMMANDS` immediately after the (now-extended) `DENY_COMMANDS` block. Match the formatting and comment style of `DENY_COMMANDS`:

```python
# (canonical_plugin_name, leaf_command) tuples for commands that modify
# persistent state, send IRC traffic to a different user, or read-with-side-
# effect (e.g. marking notes as read). Both elements lowercase — matched
# against ``cb.canonicalName()`` and the leaf name from ``cb.listCommands()``.
#
# Gated by the ``bridgeAllowMutating`` channel registry value: when False
# (the default), ``enumerate_commands`` skips these and ``dispatch`` rejects
# them defense-in-depth.
#
# Sourcing: each entry is keyed to a method in a stock Limnoria plugin
# under .venv/lib/python3.14/site-packages/supybot/plugins/<Plugin>/plugin.py
# — see docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md
# for line-level citations.
MUTATING_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        # Misc — sends a private message to a third user.
        ("misc", "tell"),
        ("misc", "noticetell"),
        # Later — offline-tell DB.
        ("later", "tell"),
        ("later", "remove"),
        ("later", "undo"),
        # Note — registered-user notes DB. ``note``/``next`` call
        # ``db.setRead`` but the side effect is a read-receipt scoped to
        # the caller's own notes (the plugin enforces
        # ``note.frm/note.to == user.id``); classified read-only.
        ("note", "send"),
        ("note", "reply"),
        ("note", "unsend"),
        # Karma — clear/dump/load all touch persistent state.
        ("karma", "clear"),
        ("karma", "dump"),
        ("karma", "load"),
        # QuoteGrabs — grab/ungrab insert/delete rows.
        ("quotegrabs", "grab"),
        ("quotegrabs", "ungrab"),
        # RSS — add/remove register/unregister feeds. ``rss`` and
        # ``info`` are reads on their face but ``update_feed_if_needed``
        # (RSS/plugin.py:396) can call ``announce_feed`` (line 434),
        # which queues PRIVMSG/NOTICE to every channel subscribed to the
        # feed (line 553-557). Classified mutating to keep LLM-triggered
        # reads from pushing entries into third-party channels.
        # NB: nested ``announce add/remove/list/channels`` leaves are
        # NOT classified here — see "Ambiguous classifications" #3 in
        # the plan; multi-word leaves are out of scope for Task 1.
        ("rss", "add"),
        ("rss", "remove"),
        ("rss", "rss"),
        ("rss", "info"),
        # Forward-look: not in Phase 2 Task 2's default allowlist but
        # classified now so the gate is correct when they're added.
        # Quote — ChannelIdDatabasePlugin write commands plus the
        # plugin-local ``replace`` override.
        ("quote", "add"),
        ("quote", "remove"),
        ("quote", "change"),
        ("quote", "replace"),
        # Todo — user-scoped todo DB writes.
        ("todo", "add"),
        ("todo", "remove"),
        ("todo", "setpriority"),
        ("todo", "change"),
        # Factoids — channel-scoped fact DB writes. ``whatis`` looks
        # like a read but ``_replyFactoids`` calls ``_updateRank``
        # (Factoids/plugin.py:372-383, 397, 420) which UPDATEs
        # ``relations.usage_count`` whenever ``keepRankInfo`` is True.
        # ``keepRankInfo`` defaults True (Factoids/config.py:85-87) so
        # by default ``whatis`` writes the DB on every call.
        ("factoids", "learn"),
        ("factoids", "alias"),
        ("factoids", "lock"),
        ("factoids", "unlock"),
        ("factoids", "forget"),
        ("factoids", "change"),
        ("factoids", "whatis"),
        # Scheduler — every leaf except ``list`` is a write.
        ("scheduler", "add"),
        ("scheduler", "remind"),
        ("scheduler", "remove"),
        ("scheduler", "repeat"),
    }
)
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k "mutating_commands or exposes_deny_lists"
```

Expected: 4 PASS (existing module test + 3 new mutating tests).

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): add MUTATING_COMMANDS classification to bridge"
```

**Done when:** `MUTATING_COMMANDS` is committed with the canonical entries above, lowercase invariant holds, and the three new tests pass alongside the existing module-level test. The constant is referenced by no production code yet — the gate wiring lives in B1/B2.

---

## B — The gate: registry value + enforcement points

### Task B1: Register `bridgeAllowMutating` channel value

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (insert between `bridgeAllowedPlugins` at line 865 and `bridgeDebugInChannel` at line 878)
- Modify: `plugins/llm/tests/test_config.py` (add a default-value test next to the `bridge_registry_values_registered_with_safe_defaults` test introduced in Phase 1 — same pattern)

**Background:** `bridgeAllowedPlugins` is a `SpaceSeparatedListOfStrings` with default `[]`. `bridgeEnabled` and `bridgeDebugInChannel` are `Boolean` channel values with default `False`. The new gate is a `Boolean(False)` to match the existing safe-default pattern (see `config.py:854-888`).

**Step 1: Write the failing test.**

Append to `plugins/llm/tests/test_config.py`. Match the style of the existing `test_bridge_registry_values_registered_with_safe_defaults` test (introduced by Phase 1 — find it with `grep -n "bridge_registry_values_registered" plugins/llm/tests/test_config.py`; if it lives inside a `TestConfigValues` class, place this new test next to it):

```python
def test_bridge_allow_mutating_registered_with_safe_default():
    """B1: bridgeAllowMutating defaults to False (gate closed).

    Note on the assertion shape: ``bridgeAllowMutating`` is a
    ``registerChannelValue`` (per-channel), but calling it as
    ``conf.supybot.plugins.LLM.bridgeAllowMutating()`` with no channel
    argument returns the channel-independent default. That is exactly
    what we want to verify here — that the registered default is
    ``False``, not the value for any particular channel. Same pattern as
    the Phase 1 ``test_bridge_registry_values_registered_with_safe_defaults``
    test; per-channel behaviour is exercised separately in C1's tests.
    """
    import supybot.conf as conf
    import llm.config  # noqa: F401 — import side effect registers the value

    assert conf.supybot.plugins.LLM.bridgeAllowMutating() is False
```

**Step 2: Run; verify it fails.**

```bash
uv run pytest plugins/llm/tests/test_config.py -v -k bridge_allow_mutating
```

Expected: FAIL — registry value not defined.

**Step 3: Register the value in `config.py`.**

Insert between the existing `bridgeAllowedPlugins` block and `bridgeDebugInChannel` block (after `config.py:876`, before `config.py:878`):

```python
conf.registerChannelValue(
    LLM,
    "bridgeAllowMutating",
    registry.Boolean(
        False,
        _("""When True, the Limnoria bridge exposes commands that modify
        persistent state (sending notes, registering feeds, mutating karma,
        etc.). When False (the default), only read-only commands are exposed
        — write commands are hidden from the LLM's tool description and any
        attempt to dispatch one returns an error envelope.

        Per-command classification lives in MUTATING_COMMANDS in
        plugins/llm/src/llm/limnoria_bridge.py."""),
    ),
)
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_config.py -v -k bridge
```

Expected: PASS for both the Phase 1 bridge defaults test and the new `bridge_allow_mutating` test.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "feat(llm): register bridgeAllowMutating channel value (default False)"
```

**Done when:** the channel value is registered, defaults to `False`, and the new test passes. The value is not yet read by any production code — that's B2.

---

### Task B2: Enforce the gate in `enumerate_commands`

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py` (`enumerate_commands` at lines 95-136)
- Modify: `plugins/llm/tests/test_limnoria_bridge.py` (extend the existing enumerate-tests block at lines 167-289)

**Background:** the gate is implemented as a parameter on `enumerate_commands`, **not** by reading the registry inside the bridge module. Reasons:
1. The bridge module is testable in isolation today (it imports `supybot.callbacks`, not `supybot.conf`); pulling registry reads in would force tests to bootstrap a config.
2. The caller `_build_bridge_tool` (`plugin.py:1569`) already reads `registryValue("bridgeEnabled", channel)` and `registryValue("bridgeAllowedPlugins", channel)` at lines 1581-1583 — adding `registryValue("bridgeAllowMutating", channel)` next to them keeps all registry reads in one place.

The new parameter is keyword-only `allow_mutating: bool` with default `False`. When False, leaves whose `(canonical, leaf)` tuple is in `MUTATING_COMMANDS` are skipped — same shape as the existing `DENY_COMMANDS` filter at `limnoria_bridge.py:122-123`.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_limnoria_bridge.py`. Reuse the existing `_stub_callback` / `_fake_irc_with_callbacks` / `_fake_msg` helpers at lines 132-164:

```python
def test_enumerate_skips_mutating_commands_when_gate_closed(mocker):
    """With allow_mutating=False (the default), MUTATING_COMMANDS leaves
    are filtered out even if their plugin is allowlisted."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes", "remove", "undo"],
        docstrings={
            "tell": "<nick> <text>",
            "notes": "takes no arguments",
            "remove": "<id>",
            "undo": "takes no arguments",
        },
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(
        lb.enumerate_commands(irc, msg, frozenset({"Later"}), allow_mutating=False)
    )

    leaves = {c.command for c in result}
    assert leaves == {"notes"}  # tell, remove, undo all in MUTATING_COMMANDS


def test_enumerate_yields_mutating_commands_when_gate_open(mocker):
    """With allow_mutating=True, MUTATING_COMMANDS is not consulted —
    the existing capability + DENY filters still apply, but writes pass."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes", "remove", "undo"],
        docstrings={
            "tell": "<nick> <text>",
            "notes": "takes no arguments",
            "remove": "<id>",
            "undo": "takes no arguments",
        },
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(
        lb.enumerate_commands(irc, msg, frozenset({"Later"}), allow_mutating=True)
    )

    leaves = {c.command for c in result}
    assert leaves == {"tell", "notes", "remove", "undo"}


def test_enumerate_default_keyword_is_gate_closed(mocker):
    """Calling enumerate_commands without allow_mutating= defaults to the
    closed gate — backwards-compat safety: an old caller that forgets to
    pass the kwarg still gets safe behavior."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Later",
        canonical="later",
        commands=["tell", "notes"],
        docstrings={"tell": "<nick> <text>", "notes": "x"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(lb.enumerate_commands(irc, msg, frozenset({"Later"})))
    leaves = {c.command for c in result}
    assert leaves == {"notes"}


def test_enumerate_gate_does_not_affect_pure_read_only_plugins(mocker):
    """A plugin with no entries in MUTATING_COMMANDS (e.g. Time) yields
    the same set whether the gate is open or closed."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Time",
        canonical="time",
        commands=["time", "at", "until"],
        docstrings={"time": "x", "at": "y", "until": "z"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    closed = {
        c.command
        for c in lb.enumerate_commands(
            irc, msg, frozenset({"Time"}), allow_mutating=False
        )
    }
    open_ = {
        c.command
        for c in lb.enumerate_commands(
            irc, msg, frozenset({"Time"}), allow_mutating=True
        )
    }
    assert closed == open_ == {"time", "at", "until"}


def test_enumerate_gate_preserves_deny_commands_filtering(mocker):
    """DENY_COMMANDS still bites even when allow_mutating=True. Web.fetch
    is denied unconditionally (SSRF), independent of the mutation gate."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(
        mocker,
        "Web",
        canonical="web",
        commands=["fetch", "title"],
        docstrings={"fetch": "<url>", "title": "<url>"},
    )
    irc = _fake_irc_with_callbacks(mocker, [cb])
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)

    result = list(
        lb.enumerate_commands(irc, msg, frozenset({"Web"}), allow_mutating=True)
    )
    leaves = {c.command for c in result}
    assert leaves == {"title"}  # fetch is in DENY_COMMANDS
```

**Step 2: Run; verify all five fail.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v \
    -k "skips_mutating_commands or yields_mutating_commands or default_keyword or read_only_plugins or preserves_deny_commands"
```

Expected: 5 FAIL with `TypeError: enumerate_commands() got an unexpected keyword argument 'allow_mutating'` (or 0 fails for the test that doesn't pass the kwarg — that one will fail on the assertion because the gate doesn't exist yet).

**Step 3: Add the parameter to `enumerate_commands` and the filter.**

Modify `limnoria_bridge.py:95-136`. The function signature gains a keyword-only `allow_mutating` parameter with default `False`; the loop body adds one filter line that mirrors the existing `DENY_COMMANDS` filter at line 122:

```python
def enumerate_commands(
    irc: Any,
    msg: Any,
    allowed_plugins: frozenset[str],
    *,
    allow_mutating: bool = False,
) -> Iterator[BridgeCommand]:
    """Yield every loaded command the LLM is allowed to call.

    A command is yielded when ALL of:
    - Its plugin is in ``allowed_plugins`` (operator allowlist).
    - Its plugin is NOT in ``DENY_PLUGINS`` (hard deny).
    - Its (canonical_plugin, leaf) tuple is NOT in ``DENY_COMMANDS``.
    - When ``allow_mutating`` is False (the default), its
      (canonical_plugin, leaf) tuple is NOT in ``MUTATING_COMMANDS``.
    - ``checkCommandCapability(msg, cb, leaf)`` returns falsy
      (i.e. allowed for the calling user).

    The capability check uses the string form of the leaf name to
    mirror ``_callCommand``'s pattern at supybot/callbacks.py:1591;
    list form ``[leaf]`` triggers an AssertionError because the leaf
    is not the plugin's canonical name.
    """
    for cb in irc.callbacks:
        plugin_name = cb.name()
        if plugin_name in DENY_PLUGINS:
            continue
        if plugin_name not in allowed_plugins:
            continue
        canonical = cb.canonicalName()
        for leaf in cb.listCommands():
            if (canonical, leaf) in DENY_COMMANDS:
                continue
            if not allow_mutating and (canonical, leaf) in MUTATING_COMMANDS:
                continue
            denial = callbacks.checkCommandCapability(msg, cb, leaf)
            if denial:
                continue
            method = cb.getCommandMethod([leaf])
            doc_lines = (method.__doc__ or "").strip().splitlines()
            arg_syntax = doc_lines[0].strip() if doc_lines else ""
            description = " ".join(line.strip() for line in doc_lines[1:]).strip()
            yield BridgeCommand(
                plugin=plugin_name,
                command=leaf,
                arg_syntax=arg_syntax,
                description=description,
            )
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k enumerate
```

Expected: all enumerate tests pass — both the existing ones (which call `enumerate_commands(irc, msg, allowed)` with no kwarg, exercising the default-False path) and the five new ones.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): gate mutating commands in bridge enumerate_commands"
```

**Done when:** the existing tests still pass without modification (default-False kwarg is backwards compatible), the five new tests pass, and `enumerate_commands` skips `MUTATING_COMMANDS` entries when `allow_mutating=False`.

---

### Task B3: Defense-in-depth gate in `dispatch`

**Files:**
- Modify: `plugins/llm/src/llm/limnoria_bridge.py` (`dispatch` at lines 139-203)
- Modify: `plugins/llm/tests/test_limnoria_bridge.py` (extend the dispatch tests block)

**Background:** even though `enumerate_commands` hides mutating commands from the LLM's tool description, the LLM may hallucinate a call from training memory ("I know `Note send` exists, let me try it"). `dispatch` must reject mutating commands when the gate is closed and return the same JSON-envelope shape as the existing deny paths (`limnoria_bridge.py:174-186`). Error message: `"denied: write commands disabled"` — distinct from the existing `"denied: <plugin>.<cmd>"` so log scraping can tell the gate apart from `DENY_PLUGINS` / `DENY_COMMANDS` rejections.

**Ordering goal — what the dispatch path is and is not trying to hide:**

The mutation-gate check is placed *after* `cb.isCommandMethod(command)` and *before* `checkCommandCapability`. This intentionally hides **capability status** but does **not** hide **existence**:

- Unknown leaves return `"unknown command: <plugin>.<cmd>"` — existence leakage is acceptable and arguably necessary (the LLM and the user need clear errors when they invoke a name that doesn't exist).
- Known mutating leaves return `"denied: write commands disabled"` regardless of whether the calling user *would* have had the capability to invoke them. This prevents probing ("does this user have `op`?") via the bridge.

If at code review someone proposes also hiding existence (returning `"unknown command"` for everything in `MUTATING_COMMANDS` when the gate is closed), reject it: it makes the LLM's debug-loop worse and doesn't add real defence — the LLM can already enumerate via the (publicly-described) plugin allowlist.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_limnoria_bridge.py`:

```python
def test_dispatch_rejects_mutating_when_gate_closed(mocker):
    """With allow_mutating=False (default), dispatching a MUTATING_COMMANDS
    leaf returns {"error": "denied: write commands disabled"}."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Later", canonical="later", commands=["tell"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(
        irc, msg, plugin="Later", command="tell", arg_string="alice hi"
    )
    assert out == {"error": "denied: write commands disabled"}
    cb._callCommand.assert_not_called()


def test_dispatch_allows_mutating_when_gate_open(mocker):
    """With allow_mutating=True, dispatch goes through to _callCommand
    and returns the captured reply envelope."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Later", canonical="later", commands=["tell"])

    def _fake_call(_command, proxy, _msg, _tokens):
        proxy.reply("ok, I'll tell alice next time I see her")

    cb._callCommand.side_effect = _fake_call
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=["alice", "hi"])

    out = lb.dispatch(
        irc,
        msg,
        plugin="Later",
        command="tell",
        arg_string="alice hi",
        allow_mutating=True,
    )
    assert out == {"status": "ok", "reply": "ok, I'll tell alice next time I see her"}


def test_dispatch_default_keyword_is_gate_closed(mocker):
    """Same backwards-compat safety as enumerate: a caller that forgets
    the kwarg defaults to safe behavior."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Note", canonical="note", commands=["send"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    out = lb.dispatch(
        irc, msg, plugin="Note", command="send", arg_string="bob hi"
    )
    assert out == {"error": "denied: write commands disabled"}


def test_dispatch_gate_does_not_affect_read_commands(mocker):
    """A non-MUTATING leaf dispatches normally regardless of allow_mutating."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Misc", canonical="misc", commands=["ping"])

    def _fake_call(_command, proxy, _msg, _tokens):
        proxy.reply("pong")

    cb._callCommand.side_effect = _fake_call
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    irc.network = "testnet"
    msg = _fake_msg(mocker)
    mocker.patch.object(lb.callbacks, "checkCommandCapability", return_value=False)
    mocker.patch.object(lb.callbacks, "tokenize", return_value=[])

    closed = lb.dispatch(
        irc, msg, plugin="Misc", command="ping", arg_string=""
    )
    open_ = lb.dispatch(
        irc,
        msg,
        plugin="Misc",
        command="ping",
        arg_string="",
        allow_mutating=True,
    )
    assert closed == {"status": "ok", "reply": "pong"}
    assert open_ == {"status": "ok", "reply": "pong"}


def test_dispatch_gate_check_runs_after_command_existence_check(mocker):
    """An unknown command must still surface as 'unknown command', not
    'denied: write commands disabled' — order matters for clear errors."""
    from llm import limnoria_bridge as lb

    # canonical=note, but the leaf doesn't exist on the plugin.
    cb = _stub_callback(mocker, "Note", canonical="note", commands=["search"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)

    # 'send' IS in MUTATING_COMMANDS, but it's not a valid command on this
    # particular cb (isCommandMethod returns False). Existence wins.
    out = lb.dispatch(
        irc, msg, plugin="Note", command="send", arg_string="bob hi"
    )
    assert out == {"error": "unknown command: Note.send"}


def test_dispatch_gate_check_runs_before_capability_check(mocker):
    """A capability-blocked mutating command must surface as 'denied: write
    commands disabled' (the gate), not 'not permitted' — we don't want to
    leak which mutating commands the user would otherwise be allowed to run."""
    from llm import limnoria_bridge as lb

    cb = _stub_callback(mocker, "Later", canonical="later", commands=["tell"])
    irc = mocker.MagicMock()
    irc.getCallback.return_value = cb
    msg = _fake_msg(mocker)
    # Capability check would block — but the gate fires first.
    cap = mocker.patch.object(
        lb.callbacks, "checkCommandCapability", return_value="anti.cap"
    )

    out = lb.dispatch(
        irc, msg, plugin="Later", command="tell", arg_string="alice hi"
    )
    assert out == {"error": "denied: write commands disabled"}
    cap.assert_not_called()
```

**Step 2: Run; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k "dispatch_rejects_mutating or dispatch_allows_mutating or dispatch_default_keyword or dispatch_gate"
```

Expected: 6 FAIL — `TypeError: dispatch() got an unexpected keyword argument 'allow_mutating'` for the gate-open tests, and assertion failures for the closed-gate tests.

**Step 3: Add the parameter and the gate check to `dispatch`.**

Modify `limnoria_bridge.py:139-203`:

```python
def dispatch(
    irc: Any,
    msg: Any,
    *,
    plugin: str,
    command: str,
    arg_string: str,
    allow_mutating: bool = False,
) -> dict[str, Any]:
    """Run ``plugin.command arg_string`` through Limnoria's command path.

    Layered checks before dispatch:
    1. Plugin must resolve via ``irc.getCallback(plugin)``.
    2. Plugin must not be in ``DENY_PLUGINS``.
    3. (canonical_plugin, command) must not be in ``DENY_COMMANDS``.
    4. ``cb.isCommandMethod(command)`` must be True.
    5. When ``allow_mutating`` is False (the default), (canonical_plugin,
       command) must not be in ``MUTATING_COMMANDS``. Defense in depth on
       top of ``enumerate_commands``'s filter — even if the LLM hallucinates
       a write command, the dispatch path still rejects.
    6. ``checkCommandCapability(msg, cb, command)`` must be falsy.

    On success, returns ``{"status": "ok", "reply": "<captured text>"}``.
    On any check failure or uncaught exception, returns
    ``{"error": "<reason>"}``. The shape matches ``AssistantToolExecutor._ok``
    / ``_err`` (see assistant.py:676-683) so the assistant loop's
    ``last_successful_tool`` guard at service.py:2705-2710 fires correctly.
    """
    _log.info(
        "bridge call: %s.%s args=%r nick=%s channel=%s allow_mutating=%s",
        plugin,
        command,
        arg_string,
        getattr(msg, "nick", "?"),
        getattr(msg, "channel", "?"),
        allow_mutating,
    )
    cb = irc.getCallback(plugin)
    if cb is None:
        _log.info("bridge result: %s.%s -> error: unknown plugin", plugin, command)
        return {"error": f"unknown plugin: {plugin}"}
    if cb.name() in DENY_PLUGINS:
        _log.info("bridge result: %s.%s -> error: denied (plugin)", plugin, command)
        return {"error": f"denied: {plugin}.{command}"}
    if (cb.canonicalName(), command) in DENY_COMMANDS:
        _log.info("bridge result: %s.%s -> error: denied (command)", plugin, command)
        return {"error": f"denied: {plugin}.{command}"}
    if not cb.isCommandMethod(command):
        _log.info("bridge result: %s.%s -> error: unknown command", plugin, command)
        return {"error": f"unknown command: {plugin}.{command}"}
    if not allow_mutating and (cb.canonicalName(), command) in MUTATING_COMMANDS:
        _log.info(
            "bridge result: %s.%s -> error: denied (mutation gate closed)",
            plugin,
            command,
        )
        return {"error": "denied: write commands disabled"}
    denial = callbacks.checkCommandCapability(msg, cb, command)
    if denial:
        _log.info("bridge result: %s.%s -> error: not permitted", plugin, command)
        return {"error": f"not permitted: {plugin}.{command}"}

    proxy = BufferingIrcProxy(irc, msg)
    try:
        # tokenize() raises SyntaxError on malformed bracket/pipe/quote
        # syntax (callbacks.py:431) — keep it inside the try so the
        # error-envelope contract holds for malformed args too.
        tokens = callbacks.tokenize(arg_string, channel=msg.channel, network=irc.network)
        # Positional args; keyword `args=tokens` would land in **kwargs and
        # break wrap()-based commands. See callbacks.py:1213.
        cb._callCommand([command], proxy, msg, tokens)
    except Exception as exc:  # noqa: BLE001 — translating to JSON envelope
        _log.info("bridge result: %s.%s -> exception: %s", plugin, command, exc)
        return {"error": str(exc) or exc.__class__.__name__}
    reply = "\n".join(proxy.buffer)
    _log.debug("bridge result: %s.%s -> ok reply=%r", plugin, command, reply)
    _log.info("bridge result: %s.%s -> ok (%d chars)", plugin, command, len(reply))
    return {"status": "ok", "reply": reply}
```

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v -k dispatch
```

Expected: every dispatch test passes — the 11 existing dispatch tests (which don't pass `allow_mutating=`, exercising the default-False path; importantly, the existing `test_dispatch_captures_reply` and similar test against `Misc.ping`, which is read-only, so they are unaffected) plus the 6 new gate tests.

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/limnoria_bridge.py plugins/llm/tests/test_limnoria_bridge.py
git commit -m "feat(llm): defense-in-depth mutation gate in bridge dispatch"
```

**Done when:** all 11 existing dispatch tests still pass without modification, the 6 new gate tests pass, and the gate fires in the documented order (existence-check before mutation gate before capability check).

---

## C — Wire the gate into `_build_bridge_tool`

### Task C1: Read `bridgeAllowMutating` and pass it through to enumerate / dispatch

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (`_build_bridge_tool` at lines 1569-1646)
- Modify: `plugins/llm/tests/test_plugin.py` (extend the existing `_build_bridge_tool` tests; find them with `grep -n "build_bridge_tool" plugins/llm/tests/test_plugin.py`)

**Background:** `_build_bridge_tool` reads `bridgeEnabled` (`plugin.py:1581`) and `bridgeAllowedPlugins` (`plugin.py:1583`) today. Add `bridgeAllowMutating` next to those reads, and forward it to both `enumerate_commands` (line 1587) and the inner `handler` closure (line 1631-1641) so `dispatch` enforces the same gate.

**B5 — update existing `TestBuildBridgeTool` lambdas before adding the new tests.**

Once C1 lands, `_build_bridge_tool` will call `self.registryValue("bridgeAllowMutating", channel)` on every code path. The four pre-existing tests in `plugins/llm/tests/test_plugin.py` that exercise `_build_bridge_tool` —

- `test_build_bridge_tool_returns_none_when_disabled`
- `test_build_bridge_tool_returns_none_when_allowlist_empty`
- `test_build_bridge_tool_returns_schema_and_handler_when_commands_present`
- `test_build_bridge_tool_handler_returns_tool_result_with_json`

— each set `plugin.registryValue.side_effect` to a lambda that returns `None` for any unhandled key. After C1, the new `bridgeAllowMutating` read will fall through to `None`, and `bool(None)` is `False`, so the tests would *appear* to pass without ever verifying the gate value flowed correctly. **Before writing the new tests below**, find each of the four lambdas (`grep -n "registryValue.side_effect" plugins/llm/tests/test_plugin.py`) and add an explicit `False if k == "bridgeAllowMutating" else` clause **before** the trailing `None`. Same for any future-added registry keys (`bridgeDebugInChannel` is already handled in those tests; if it isn't, add it too — the side-effect order should be `bridgeEnabled` → `bridgeAllowedPlugins` → `bridgeAllowMutating` → `bridgeDebugInChannel` → `None`). This makes the existing tests verify by intent, not by coincidence.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_plugin.py`. Match the registry-side-effect lambda pattern in the existing `test_build_bridge_tool_returns_schema_and_handler_when_commands_present` test:

```python
def test_build_bridge_tool_passes_allow_mutating_false_by_default(plugin_env, mocker):
    """When bridgeAllowMutating is False (default), enumerate_commands is
    called with allow_mutating=False."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Misc"] if k == "bridgeAllowedPlugins" else
        False if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    enum_mock = mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Misc", command="ping", arg_syntax="", description=""
            )
        ],
    )

    plugin._build_bridge_tool(irc, msg, "#test")

    args, kwargs = enum_mock.call_args
    assert kwargs["allow_mutating"] is False


def test_build_bridge_tool_passes_allow_mutating_true_when_gate_open(
    plugin_env, mocker
):
    """When bridgeAllowMutating is True, enumerate_commands receives
    allow_mutating=True and the bridge dispatch handler does too."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Later"] if k == "bridgeAllowedPlugins" else
        True if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    enum_mock = mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Later", command="tell", arg_syntax="<nick> <text>",
                description="",
            )
        ],
    )
    dispatch_mock = mocker.patch(
        "llm.limnoria_bridge.dispatch",
        return_value={"status": "ok", "reply": "ok"},
    )

    _, handlers = plugin._build_bridge_tool(irc, msg, "#test")

    enum_kwargs = enum_mock.call_args.kwargs
    assert enum_kwargs["allow_mutating"] is True

    handlers["run_limnoria_command"](
        {"plugin": "Later", "command": "tell", "args": "alice hi"}
    )
    dispatch_kwargs = dispatch_mock.call_args.kwargs
    assert dispatch_kwargs["allow_mutating"] is True


def test_build_bridge_tool_dispatch_handler_uses_closed_gate_by_default(
    plugin_env, mocker
):
    """The handler closure captures the gate value at build time and uses it
    for every dispatch call within that turn."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Misc"] if k == "bridgeAllowedPlugins" else
        False if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Misc", command="ping", arg_syntax="", description=""
            )
        ],
    )
    dispatch_mock = mocker.patch(
        "llm.limnoria_bridge.dispatch",
        return_value={"status": "ok", "reply": "pong"},
    )

    _, handlers = plugin._build_bridge_tool(irc, msg, "#test")
    handlers["run_limnoria_command"](
        {"plugin": "Misc", "command": "ping", "args": ""}
    )

    assert dispatch_mock.call_args.kwargs["allow_mutating"] is False


def test_build_bridge_tool_behavior_later_notes_visible_tell_hidden_when_gate_closed(
    plugin_env, mocker
):
    """Behavior-level (not plumbing): with Later allowlisted and the gate
    closed, the rendered tool description must list Later.notes (read) and
    NOT list Later.tell (mutating). Same setup with gate open must list both.

    This exercises the real ``enumerate_commands`` (no mock) against a stub
    Limnoria callback; if a future refactor decouples the gate from the
    enumeration path, this test catches the regression that the
    plumbing-level tests above would miss.
    """
    plugin, irc, msg = plugin_env

    # Stub callback shaped to look like the Later plugin: 'tell' (mutating)
    # and 'notes' (read-only). Reuse the helpers in test_limnoria_bridge.py
    # if they're importable; otherwise inline the minimal MagicMock shape.
    later = mocker.MagicMock()
    later.name.return_value = "Later"
    later.canonicalName.return_value = "later"
    later.listCommands.return_value = ["tell", "notes"]
    later.getCommandMethod.side_effect = lambda cmd: mocker.MagicMock(
        __doc__={"tell": "<nick> <text>\n\nQueue offline message.",
                 "notes": "takes no arguments\n\nList queued notes."}[cmd[0]]
    )
    irc.callbacks = [later]
    mocker.patch(
        "llm.limnoria_bridge.callbacks.checkCommandCapability",
        return_value=False,
    )

    # Gate closed: only 'notes' should appear in the description table.
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Later"] if k == "bridgeAllowedPlugins" else
        False if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    schema_closed, _ = plugin._build_bridge_tool(irc, msg, "#test")
    desc_closed = schema_closed["function"]["description"]
    assert "later.notes" in desc_closed.lower()
    assert "later.tell" not in desc_closed.lower()

    # Gate open: both should appear.
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Later"] if k == "bridgeAllowedPlugins" else
        True if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    schema_open, _ = plugin._build_bridge_tool(irc, msg, "#test")
    desc_open = schema_open["function"]["description"]
    assert "later.notes" in desc_open.lower()
    assert "later.tell" in desc_open.lower()
```

> **Note on table format:** the assertions above assume the bridge-tool description renders the leaves as `<plugin>.<leaf>` pairs (case-insensitive contains-check). If the existing Phase 1 description format differs (e.g. just bare leaf names, or a JSON list), adapt the substrings to match what `_build_bridge_tool` actually emits today — read the description rendering at `plugin.py:1591-1604` once and pick the right substring before running the test. The point is to assert on the rendered text, not on the mock-call kwargs.

**Step 2: Run; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k "build_bridge_tool_passes_allow_mutating or build_bridge_tool_dispatch_handler_uses"
```

Expected: 3 FAIL — the production code doesn't read `bridgeAllowMutating` and doesn't pass `allow_mutating=` to `enumerate_commands` or `dispatch`.

**Step 3: Modify `_build_bridge_tool`.**

In `plugins/llm/src/llm/plugin.py`, modify the helper between lines 1569 and 1646. Two surgical edits:

3a. Read the registry value once, at the same point we read the others (after the existing `allowed = frozenset(...)` line at `plugin.py:1583`):

```python
allow_mutating = bool(self.registryValue("bridgeAllowMutating", channel))
```

3b. Forward it to `enumerate_commands` (modify line 1587):

```python
commands = list(
    limnoria_bridge.enumerate_commands(
        irc, msg, allowed, allow_mutating=allow_mutating
    )
)
```

3c. Forward it from the handler closure to `dispatch` (modify the call at lines 1635-1641):

```python
def handler(arguments):
    plugin_name = str(arguments.get("plugin", ""))
    command_name = str(arguments.get("command", ""))
    arg_string = str(arguments.get("args", ""))
    envelope = limnoria_bridge.dispatch(
        irc,
        msg,
        plugin=plugin_name,
        command=command_name,
        arg_string=arg_string,
        allow_mutating=allow_mutating,
    )
    if trace is not None:
        # ... existing trace-append logic unchanged ...
```

(The closure captures `allow_mutating` from the enclosing scope, exactly like it captures `irc` and `msg`.)

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge
uv run pytest plugins/llm/tests -q  # nothing else regressed
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): wire bridgeAllowMutating through _build_bridge_tool"
```

**Done when:** the helper reads `bridgeAllowMutating` once per turn, both `enumerate_commands` and the dispatch handler closure receive the captured value, and every other `_build_bridge_tool` test (registry-disabled, allowlist-empty, no-commands) still passes unchanged.

---

### Task C2: Tool-description footer when gate is closed and a both-kinds plugin is allowlisted

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (extend the `table = "\n".join(...)` block at lines 1591-1596 plus the description string at lines 1601-1604)
- Modify: `plugins/llm/tests/test_plugin.py` (extend the existing `_build_bridge_tool` tests)

**Background:** the design plan asks for a footer like *"(write commands hidden — set bridgeAllowMutating True to expose)"* when `bridgeAllowMutating` is False **and** the channel allowlist contains a plugin that has both mutating and read-only commands. Why both-kinds: if the operator allowlists a pure-read plugin like `Time`, surfacing the footer adds no signal — there are no hidden writes. The footer earns its tokens only when the LLM might otherwise be confused about why `Note.send` isn't available even though it just used `Note.search`.

The "both-kinds" check is: does any allowlisted plugin appear in `MUTATING_COMMANDS`? Computed by intersecting the plugin's canonical name against the canonical names of `MUTATING_COMMANDS` entries.

**Step 1: Write failing tests.**

Append to `plugins/llm/tests/test_plugin.py`:

```python
def test_build_bridge_tool_appends_footer_when_gate_closed_and_both_kinds_present(
    plugin_env, mocker
):
    """Allowlist contains 'Later' (has both writes and reads), gate closed
    → tool description ends with the footer."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Later"] if k == "bridgeAllowedPlugins" else
        False if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Later", command="notes", arg_syntax="", description=""
            )
        ],
    )

    schema, _ = plugin._build_bridge_tool(irc, msg, "#test")
    desc = schema["function"]["description"]
    assert "write commands hidden" in desc
    assert "bridgeAllowMutating" in desc


def test_build_bridge_tool_omits_footer_when_gate_open(plugin_env, mocker):
    """Gate open → no footer (writes are exposed; nothing to flag)."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Later"] if k == "bridgeAllowedPlugins" else
        True if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Later", command="tell", arg_syntax="<nick> <text>",
                description="",
            )
        ],
    )

    schema, _ = plugin._build_bridge_tool(irc, msg, "#test")
    desc = schema["function"]["description"]
    assert "write commands hidden" not in desc


def test_build_bridge_tool_omits_footer_when_only_pure_read_plugins_allowed(
    plugin_env, mocker
):
    """Allowlist is Time + Math (both pure-read) and gate is closed —
    nothing was hidden, so the footer would be misleading."""
    plugin, irc, msg = plugin_env
    plugin.registryValue.side_effect = lambda k, ch=None: (
        True if k == "bridgeEnabled" else
        ["Time", "Math"] if k == "bridgeAllowedPlugins" else
        False if k == "bridgeAllowMutating" else
        False if k == "bridgeDebugInChannel" else
        None
    )
    mocker.patch(
        "llm.limnoria_bridge.enumerate_commands",
        return_value=[
            mocker.MagicMock(
                plugin="Time", command="time", arg_syntax="", description=""
            ),
            mocker.MagicMock(
                plugin="Math", command="calc", arg_syntax="<expr>", description=""
            ),
        ],
    )

    schema, _ = plugin._build_bridge_tool(irc, msg, "#test")
    desc = schema["function"]["description"]
    assert "write commands hidden" not in desc
```

**Step 2: Run; verify they fail.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k "footer"
```

Expected: 3 FAIL — first two by assertion (footer logic doesn't exist), third either passes by accident or fails on the registry mock; either way the implementation isn't there.

**Step 3: Add the footer logic to `_build_bridge_tool`.**

After the existing `table = "\n".join(...)` block at `plugin.py:1591-1596`, compute whether to append the footer, and wire it into the description string at `plugin.py:1601-1604`. The check:

```python
# Footer: if the gate is closed AND any allowlisted plugin has at least
# one mutating leaf, hint that more commands exist behind the gate. Skips
# the hint for pure-read allowlists (Time, Math, etc.) where no writes
# would be hidden.
mutating_plugins = {plugin for (plugin, _leaf) in limnoria_bridge.MUTATING_COMMANDS}
allowed_canonical = {p.lower() for p in allowed}
hidden_writes_present = (
    not allow_mutating and bool(allowed_canonical & mutating_plugins)
)
footer = (
    "\n\n(write commands hidden — set bridgeAllowMutating True to expose)"
    if hidden_writes_present
    else ""
)
```

Then change the schema description at lines 1601-1604 from:

```python
"description": (
    "Run a Limnoria plugin command on the user's behalf. "
    "Available commands:\n" + table
),
```

to:

```python
"description": (
    "Run a Limnoria plugin command on the user's behalf. "
    "Available commands:\n" + table + footer
),
```

**Note on canonical-name matching:** `MUTATING_COMMANDS` entries are canonical (lowercase), but `bridgeAllowedPlugins` is operator-typed and may be CamelCase (`"Later"`, `"Note"`). The `.lower()` on the allowlist names matches the canonical-name shape. This mirrors the existing canonical-vs-camel handling at `limnoria_bridge.py:120-122`.

**Step 4: Run; verify pass.**

```bash
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge
uv run pytest plugins/llm/tests -q
```

**Step 5: Commit.**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): footer when bridge mutation gate hides writes"
```

**Done when:** the footer appears in the tool description when (gate closed AND a both-kinds plugin is allowlisted) and is omitted otherwise; tests cover all three branches; no regression in the existing `_build_bridge_tool` tests.

---

## D — Migration / backwards compat

### Task D1: Verify Phase 1 operators are unaffected

**Files:** none modified — this task is verification only.

**Background:** Phase 1 ships with `bridgeAllowedPlugins=[]` by default and the recommended starter set in the docstring is `Misc Time Math Utilities Seen` — every one of which is *pure-read*. An operator who set `bridgeAllowedPlugins` explicitly during Phase 1 either:
- (a) Picked from the recommended pure-read starter set → no command becomes invisible after Task 1 lands. The footer does not fire (no both-kinds plugin allowlisted). Behavior unchanged.
- (b) Picked plugins beyond the starter set, e.g. `Later`, `Note`, `Karma` → some commands become invisible after Task 1 lands (the writes). Operator must set `bridgeAllowMutating True` to restore the prior behavior.

Case (b) is a behavior change and the only meaningful migration concern. The change is intentional (the whole point of Task 1 is to make writes opt-in), but operators need to know.

**Test-mock compatibility check:** after C1 lands, `_build_bridge_tool` reads four registry values per turn, in this order:

1. `bridgeEnabled`
2. `bridgeAllowedPlugins`
3. `bridgeAllowMutating` (new in Task 1)
4. `bridgeDebugInChannel` (added in Phase 1)

The existing `TestBuildBridgeTool` tests (and `conftest.py`'s `make_registry_side_effect()` helper) all use **keyword-dispatching lambdas** — `lambda k, ch=None: True if k == "..." else ...`. Keyword-dispatch tolerates new keys gracefully (provided the B5 fix above adds an explicit clause for `bridgeAllowMutating`). **Do not introduce positional-sequence mocks** anywhere in `test_plugin.py` (e.g. `side_effect=[True, [...], True, False]`); they would silently break the moment a fifth registry key is added in Phase 2 Task 3+. Verify in passing while editing C1's tests: every `side_effect` you touch should be a lambda or `make_registry_side_effect()` call, never a list.

**Step 1: Confirm the recommended starter set is pure-read.**

Read `config.py:870-874`:

```text
Recommended starter set: Misc Time Math Utilities Seen.
```

Cross-check against the read-only-leaves table in Task A1 — every one of those plugins is in the pure-read column. Confirmed: case (a) operators see no behavior change.

**Step 2: Smoke-check the both-kinds detection on a Phase-1-style explicit allowlist.**

Run a one-off Python REPL check (no commit; this is just verification) — replace the `cd` with the project root and use `uv run python` so the venv is active:

```bash
uv run python -c "
from llm import limnoria_bridge as lb
allow = {'Later', 'Note', 'Karma'}
canonical_mutating = {p for (p, _) in lb.MUTATING_COMMANDS}
print('would hide writes from:', {a.lower() for a in allow} & canonical_mutating)
"
```

Expected output: `would hide writes from: {'later', 'note', 'karma'}` — i.e. an operator who allowlisted any of these in Phase 1 will see hidden commands on first restart after Task 1 lands.

**Step 3: Document the behavior change for case-(b) operators.**

This step belongs in Task E1 (operator docs); the verification step is just to confirm we have the right set.

**Done when:** the implementer confirms via the REPL one-liner that `Later`, `Note`, `Karma`, `QuoteGrabs`, `RSS` are the both-kinds plugins from the Phase 2 Task 2 default allowlist (i.e. the plugins where this gate matters operationally), AND confirms the Phase 1 starter set in the `bridgeAllowedPlugins` registry help text is purely read-only.

---

## E — Documentation

### Task E1: Operator docs — describe the new gate

**Files:**
- Modify: `docs/guide/operator/tuning-monitoring.md` (verified 2026-05-02: Phase 1 landed `bridgeEnabled` / `bridgeAllowedPlugins` documentation here; `grep -ln 'bridgeEnabled' docs/guide/operator/` returns this single file).

**Step 1: Add a section describing `bridgeAllowMutating`.**

Topics to cover (one short paragraph each — match Phase 1's brevity, not this plan's verbosity):

- What the gate does: hides commands that modify persistent state (sending notes, registering RSS feeds, mutating karma, etc.) from the LLM's tool description by default. Defense-in-depth — even if the LLM hallucinates a write call, dispatch refuses.
- How to enable per channel:
  ```
  config channel #yourchan plugins.LLM.bridgeAllowMutating True
  ```
- Behavior change for Phase 1 operators: an operator who allowlisted `Later`, `Note`, `Karma`, `QuoteGrabs`, or `RSS` in Phase 1 will see write commands disappear from the bridge after upgrading. Setting `bridgeAllowMutating True` per channel restores the prior behavior. Pure-read allowlists (Misc, Time, Math, Utilities, Seen, Web, DDG) are unaffected.
- Pointer to source-of-truth `MUTATING_COMMANDS` in `plugins/llm/src/llm/limnoria_bridge.py` for the canonical list.
- Known limitation (not gate-related): the bridge does not currently surface RSS's nested `announce` sub-leaves (`announce add`, `announce remove`, `announce list`, `announce channels`) cleanly. Supybot returns those as multi-word leaf names (`"announce add"`, etc.) and the bridge's enumerate/dispatch path treats leaves as single-token canonical names. Whether the gate is open or closed, the LLM cannot meaningfully use these sub-leaves until the bridge's nested-Commands handling is extended (separate work, post-Task-1). Operators wanting to manage announcements should continue using the native `@rss announce …` IRC command.

**Step 2: Build the docs locally if there's a build step (`mkdocs serve` or similar) and skim. Otherwise inspect the rendered Markdown.**

**Step 3: Commit.**

```bash
git add docs/guide/operator/<file>.md
git commit -m "docs(llm): document bridgeAllowMutating gate"
```

### Task E2: AGENTS.md mention

**Files:** `AGENTS.md`.

The Phase 1 entry already exists (`AGENTS.md:96`). Update it to mention Phase 2 Task 1:

```markdown
- `plugins/llm/src/llm/limnoria_bridge.py` - Limnoria → LLM tool bridge (Phase 1; Phase 2 mutation gate; see docs/plans/2026-05-02-limnoria-tool-bridge-plan.md and docs/plans/2026-05-02-limnoria-bridge-task-1-implementation-plan.md)
```

(One-line edit; no separate test or commit step beyond the `git add AGENTS.md && git commit -m "docs: ..."`.)

**Done when:** operator docs describe `bridgeAllowMutating`, the migration note for case-(b) operators is captured, and AGENTS.md points at this plan.

---

## Validation

### Automated

```bash
# Bridge module unit tests — must include the new MUTATING_COMMANDS,
# enumerate gate, and dispatch gate tests.
uv run pytest plugins/llm/tests/test_limnoria_bridge.py -v

# Plugin tests — must include the new _build_bridge_tool gate forwarding
# and footer tests.
uv run pytest plugins/llm/tests/test_plugin.py -v -k bridge

# Config tests — must include the new bridgeAllowMutating default test.
uv run pytest plugins/llm/tests/test_config.py -v -k bridge

# Full LLM suite — must remain green.
uv run pytest plugins/llm/tests -q

# Repository-wide gates from AGENTS.md before declaring done.
make lint
make typecheck
make preflight
```

All commands must be green before declaring Task 1 done. Per `AGENTS.md:23`, `make preflight` is the canonical sign-off; the narrower `uv run pytest` invocations are for task-local TDD.

**Lint/typecheck reminder:** AGENTS.md:22 obliges agents to run `make lint` and `make typecheck` after editing Python files. These are *not* automatic pre-commit hooks — invoke them explicitly after every Step 3 (production-code) change before staging. If either fails, fix the issue and re-stage; do NOT bypass with `--no-verify` (project rule + general agent rule on destructive shortcuts). `make preflight` (AGENTS.md:23) is the canonical sign-off run once at the end of the task.

### Operational verification on the running bot

The standard CI → Docker build → systemctl restart cycle is pre-authorized for this repo (see auto-memory `feedback_restart_authorization`). Run after the PR merges to `main` and the Docker build completes:

1. **Wait for both CI and the Docker image build to finish.** The two are separate workflows; restarting after only CI passes runs the *previous* image (memory `feedback_wait_for_docker`).

2. **SSH into the bot and restart:**
   ```bash
   ssh -i ~/.ssh/id_rsa vibebot@rdrake.org "systemctl --user restart vibebot"
   ```
   (If SSH fails with "Permission denied (publickey)", run `security unlock-keychain` locally — see auto-memory `feedback_ssh_keychain_unlock`.)

3. **In `#test` on AfterNet, gate-closed smoke (default behavior):**
   - `@config channel #test plugins.LLM.bridgeEnabled True`
   - `@config channel #test plugins.LLM.bridgeAllowedPlugins Misc Time Later`
   - `@vibebot leave a note for alice that I'll be back tomorrow`
     - Expected: the LLM should **not** be able to use `Later.tell`. The reply should refuse, fall back to native `set_reminder`, or apologize. The bridge tool description should not list `Later.tell`. (Confirm by setting `bridgeDebugInChannel True` and seeing no `later.tell` line in the footer.)
   - `@vibebot list any pending notes I left`
     - Expected: the LLM may use `Later.notes` (read-only) and report results.

4. **Gate-open smoke (opt-in):**
   - `@config channel #test plugins.LLM.bridgeAllowMutating True`
   - `@vibebot leave a note for alice that I'll be back tomorrow`
     - Expected: the LLM uses `Later.tell`. Bridge debug footer (if enabled) shows `later.tell ok`.
   - `@vibebot what notes have I queued?`
     - Expected: `Later.notes` returns the queued tell.

5. **Footer surfacing:** with the gate closed and `Later` allowlisted, ask the LLM "what tools do you have for managing offline notes?" The LLM's response should reflect the footer ("write commands hidden — set bridgeAllowMutating True to expose") in some natural form.

6. **Defense-in-depth check:** with `Later` allowlisted and gate closed, send a message that the LLM might interpret as a tell-request. Inspect the bot logs (`journalctl --user -u vibebot -e`) for `bridge result: Later.tell -> error: denied (mutation gate closed)` — that line proves the dispatch-time gate fired even though enumerate hid it. (If the LLM never tries it, this is a no-op; the test is proving the path exists, not that it gets hit on every turn.)

**Done when:** all automated tests pass, both gate-closed and gate-open smoke tests behave as documented, and the bot logs show the dispatch-time gate firing at least once when the LLM hallucinates a denied write.

---

## Open questions for code review

These are flags raised earlier in the plan; restated here so the code reviewer doesn't have to scroll back:

1. **`RSS.announce` as a nested-subcommand-group leaf (Task A1, "Ambiguous classifications" #3).** **Resolved 2026-05-02 (review pass).** Verified by reading `supybot/callbacks.py:1554-1560` and `supybot/plugins/RSS/plugin.py:317-318, 601`: nested-Commands leaves come back from `listCommands` as multi-word strings (`"announce add"`, etc.), not as a single `"announce"` leaf, and the bridge's existing dispatch path can't resolve multi-word leaves anyway. No `("rss", "announce …")` tuple is added to `MUTATING_COMMANDS`. The gap is documented as a known bridge limitation in E1 and is out of scope for Task 1 — see the rewritten Ambiguous classifications #3 and E1's known-limitation bullet.

2. **Footer wording (Task C2).** Default: *"(write commands hidden — set bridgeAllowMutating True to expose)"*. Could be terser. Bikeshed at code review.

3. **Per-command override.** Phase 2 plan defers a `bridgeAllowedMutatingCommands` list; Task 1 does not implement it. Confirmed scope. No action required.

4. **Logging line for the gate-fired path** (`limnoria_bridge.py` new line in B3): `bridge result: %s.%s -> error: denied (mutation gate closed)`. Distinct phrasing from `denied (plugin)` / `denied (command)` for log-grep clarity. Confirm the line shape matches the project's existing log conventions; if there's a different idiom, swap it in.

5. **Future work — dynamic classifier (deferred).** Hand-curated `MUTATING_COMMANDS` is correct for v1 but goes stale as Limnoria evolves. A follow-up plan should derive classification from a heuristic (scan `wrap()` capability tags + AST-walk method bodies for write-pattern calls like `db.add`/`db.set`/`db.clear`) plus a small `MUTATING_COMMANDS_OVERRIDE` map for cases the heuristic gets wrong. Track separately as Task 1.5.

---

## Execution order summary

| Order | Task | Output |
| --- | --- | --- |
| 0 | Pre-flight (verify line numbers, baseline tests, no name collision) | (no commit) |
| 1 | A1: `MUTATING_COMMANDS` constant + classification tests | commit |
| 2 | B1: `bridgeAllowMutating` registry value | commit |
| 3 | B2: `enumerate_commands` gate | commit |
| 4 | B3: `dispatch` defense-in-depth gate | commit |
| 5 | C1: `_build_bridge_tool` reads gate, forwards through enumerate + dispatch | commit |
| 6 | C2: tool-description footer | commit |
| 7 | D1: migration verification (no commit) | (no commit) |
| 8 | E1: operator docs | commit |
| 9 | E2: AGENTS.md update | commit |
| 10 | Validation: full automated suite + operational smoke | (no commit) |

Each task is independently verifiable: tests pass after the commit, and reverting one commit cleanly leaves the codebase in a working state. Tasks A1 → B3 are the load-bearing classification + gate work; C1/C2 wire it into the request path; D1/E1/E2 ensure operators understand the change.
