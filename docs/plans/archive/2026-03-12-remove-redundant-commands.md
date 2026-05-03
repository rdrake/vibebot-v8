# Remove Redundant Commands Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `%flag`/`%unflag`/`%flagged`, `%llmkeys`, and `%animate`/`%video` commands along with all supporting code, config, tests, and documentation.

**Architecture:** Three independent command groups removed from plugin, service, persistence, config, and test layers. Flag check removed from shared preflight. Animate removal is the largest — touches config, service (video generation + retry), plugin (command + HTTP callback), and a dedicated test file.

**Tech Stack:** Python, Limnoria, pytest

**Rationale:**
- **Flag commands:** Limnoria's built-in `%admin ignore add` already blocks users from the entire bot. Custom flagging is redundant since per-plugin granularity and audit trails aren't needed.
- **llmkeys:** Only reads key status. Any admin who can run `%llmkeys` can also run `%config` to see the same info. Redundant.
- **animate/video:** Feature being dropped by operator decision.

---

### Task 1: Remove flag/unflag/flagged from plugin.py

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`

**Step 1: Remove `_check_flagged` method**

Delete the `_check_flagged()` method (around line 1421-1430) and remove its call from `_run_preflight()` (around line 1193). The preflight method calls `self._check_flagged(irc, msg, account)` — remove that call and the early return that follows it.

**Step 2: Remove flag/unflag/flagged command methods**

Delete the three command methods and their `wrap()` assignments:
- `flag()` method + `flag = wrap(flag, ["admin", "nick", "text"])` (around lines 2096-2126)
- `unflag()` method + `unflag = wrap(unflag, ["admin", "nick"])` (around lines 2128-2154)
- `flagged()` method + `flagged = wrap(flagged, ["admin"])` (around lines 2156-2177)

**Step 3: Run lint/typecheck**

```bash
make lint && make typecheck
```

Expected: PASS (no references to removed code remain in plugin.py)

---

### Task 2: Remove flagging from persistence.py

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py`

**Step 1: Remove FlaggedUserRow**

Delete `class FlaggedUserRow(NamedTuple)` (around line 89).

**Step 2: Remove flagged_users table creation**

In `_migrate()` (not `_create_tables()` — persistence uses a migration system), delete the `CREATE TABLE IF NOT EXISTS flagged_users` block (around line 237). Also remove the `"flagged_blocked"` status value from any `log_usage()` docstrings (around line 903).

**Step 3: Remove flag methods**

Delete these methods:
- `flag_user()` (around lines 1237-1289)
- `unflag_user()` (around lines 1290-1310)
- `is_user_flagged()` (around lines 1312-1330)
- `get_flagged_users()` (around lines 1331-1346)

**Step 4: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 3: Remove llmkeys from plugin.py and safe_key_display from service.py

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`
- Modify: `plugins/llm/src/llm/service.py`

**Step 1: Remove llmkeys command**

In plugin.py, delete the `llmkeys()` method and its `llmkeys = wrap(llmkeys, ["admin"])` (around lines 2059-2094).

**Step 2: Remove llmkeys from HTML help template**

In plugin.py's `HELP_HTML_TEMPLATE`, delete the `%llmkeys` command block (the `<h3>` through `</pre>` added in the earlier docs commit).

**Step 3: Remove safe_key_display from service.py**

Delete the `safe_key_display()` method (around lines 588-600). Grep the codebase first to confirm no other callers remain.

**Step 4: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 4: Remove animate/video from config.py

**Files:**
- Modify: `plugins/llm/src/llm/config.py`

**Step 1: Remove animate config registrations**

Delete all animate-related config blocks:
- `animateApiKey` (around line 245-246)
- `animateModel` (around line 251-254)
- `animateTimeout` (around line 260-265)
- `animateExpiry` (around line 382-387)
- `animateRateLimitCount` (around line 743-750)
- `animateRateLimitWindow` (around line 753-756)
- `animateTrustedRateLimitCount` (around line 762-772)
- `animateTrustedRateLimitWindow` (around line 775)
- `animateUnregRateLimitCount` (around line 781-791)
- `animateUnregRateLimitWindow` (around line 792-795)

**Step 2: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 5: Remove animate/video from plugin.py

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py`

**Step 1: Remove animate HTML help block**

In `HELP_HTML_TEMPLATE`, delete the `%animate` command block (the `<h3>` through `</pre>`).

**Step 2: Remove animate from help text**

Update the `getPluginHelp()` method (around line 1021) to remove "animate (video)" from the commands list.

**Step 3: Remove animate command method and video alias**

Delete the `animate()` method, `animate = wrap(...)`, and `video = animate` (around lines 1933-1985).

**Step 4: Remove animate from HTTP callback**

In the HTTP callback handler, remove the branch handling completed animate tasks (around line 597-598).

**Step 5: Remove animate references from docstrings/comments**

Search for remaining "animate" references in parameter docs (around lines 1158, 1223, 1410) and update them to say "ask, code, draw" instead of "ask, code, draw, animate".

**Step 6: Remove VideoResult import and type annotation**

Remove `VideoResult` from the imports at the top of plugin.py (around line 39). Also update the `_store_context_and_log_usage` type annotation (around line 1511) to remove `VideoResult` from the union type `CompletionResult | ImageResult | VideoResult` → `CompletionResult | ImageResult`. Update the accompanying docstring (around lines 1517, 1522) to remove "animate" references.

**Step 7: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 6: Remove animate/video from service.py

**Files:**
- Modify: `plugins/llm/src/llm/service.py`

**Step 1: Remove VideoResult class**

Delete `class VideoResult(NamedTuple)` (around line 128-130).

**Step 2: Remove video cost mapping**

Delete the entire `VIDEO_COST_PER_VIDEO` dict (around lines 44-46), not just the entry — it has no remaining callers after the video methods are removed.

**Step 3: Remove video generation methods**

Delete these methods:
- `_extract_video_url()` (around lines 2225-2243)
- `video_generation()` (around lines 2245-2476)
- `_save_video_bytes()` (around lines 2672-2705)
- `_download_and_save_video()` (around lines 2773-2806)
- `_retry_video()` (around lines 1226-1320)

**Step 4: Remove animate references from retry handling**

In the task retry method (around line 1394-1406), remove the animate branch.

**Step 5: Remove animateApiKey from `_sanitize()` error scrubbing**

In `_sanitize()` (around line 232), remove `"animateApiKey"` from the list of registry keys iterated for API key scrubbing. This is a correctness issue — after removing the config registration, `self.plugin.registryValue("animateApiKey")` would crash at runtime.

**Step 6: Remove dead `requests.HTTPError` branch from `_is_terminal_error`**

In `_is_terminal_error()` (around lines 1076-1079), the `requests.HTTPError` branch exists solely for `_retry_video`. After removing `_retry_video`, this branch is dead code. Remove it along with the `import requests as _requests` if no other callers remain.

**Step 7: Clean up remaining animate references in comments/docstrings**

Search for remaining "animate" in service.py and update:
- `PendingTaskResult.task_type` comment (line 143): change "ask, code, draw, animate" to "ask, code, draw"
- Retry method docstring (around line 980): remove animate reference

**Step 8: Remove VideoResult from service.py exports**

Check `__init__.py` or the imports at top of plugin.py — remove VideoResult from the export list.

**Step 9: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 7: Remove animate references from persistence.py

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py`

**Step 1: Update comments mentioning animate**

Update docstrings/comments that list "ask|code|draw|animate" to "ask|code|draw" (around lines 69, 513, 821).

**Step 2: Run lint/typecheck**

```bash
make lint && make typecheck
```

---

### Task 8: Update test files — remove flagging, llmkeys, and animate tests

**Files:**
- Delete: `plugins/llm/tests/test_animate.py`
- Modify: `plugins/llm/tests/test_commands.py`
- Modify: `plugins/llm/tests/test_plugin.py`
- Modify: `plugins/llm/tests/test_persistence.py`
- Modify: `plugins/llm/tests/test_integration.py`
- Modify: `plugins/llm/tests/test_service.py`
- Modify: `plugins/llm/tests/conftest.py`

**Step 1: Delete test_animate.py**

```bash
rm plugins/llm/tests/test_animate.py
```

**Step 2: Update conftest.py**

Remove all animate config entries from `make_registry_side_effect` defaults:
- `animateApiKey`, `animateModel`, `animateTimeout`, `animateExpiry`
- `animateRateLimitCount`, `animateRateLimitWindow`
- `animateTrustedRateLimitCount`, `animateTrustedRateLimitWindow`
- `animateUnregRateLimitCount`, `animateUnregRateLimitWindow`

**Step 3: Update test_commands.py**

- Remove `is_user_flagged` mock from fixtures (around line 84-85)
- Remove llmkeys tests (around lines 898-942)
- Remove flag/unflag/flagged tests (around lines 2009-2145)
- Remove animate rate limiting test (around lines 2229-2249)
- Update llmkeys test comment about "4 keys" to "3 keys" — actually, delete the whole llmkeys test block
- Remove `"animate="` assertion from any remaining test

**Step 4: Update test_plugin.py**

- Remove `is_user_flagged` mock from fixtures (around line 2226)
- Remove `test_preflight_blocks_flagged_user` test (around lines 2254-2266)
- Remove animate task persistence test (around lines 1574-1585)
- Remove animate rate limit config from fixtures (around lines 2121-2122)
- Remove animate rate limit independence test (around lines 2152-2156)

**Step 5: Update test_persistence.py**

- Remove `FlaggedUserRow` import (line 12)
- Remove `test_creates_flagged_users_table` (around lines 98-109)
- Remove flagged_users schema verification (around line 245-250)
- Remove all flagging tests: `test_flag_user_*`, `test_is_user_flagged_*`, `test_get_flagged_users_*` (around lines 1457-1550)
- Update animate task_type test data (around lines 964-981, 1239-1244) — change `task_type="animate"` to `"draw"` to preserve legacy-queue test coverage
- Keep the `flagged_users` CREATE TABLE in migration fixture SQL (around lines 1223-1229) — it tests backward compatibility with old DBs that have this table

**Step 6: Update test_integration.py**

- Remove flagging integration test block (around lines 319-340)

**Step 7: Update test_service.py**

- Remove `safe_key_display` tests (around lines 92-112)
- Remove `animateApiKey` from mock config fixtures (around lines 122, 138, 160, 176)

**Step 8: Run all tests**

```bash
make test
```

Expected: All tests pass, coverage >= 80%

---

### Task 9: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `plugins/llm/README.md`
- Modify: `plugins/llm/locales/messages.pot` and all `.po` files

**Step 1: Update README.md**

- Remove `%animate` / `%video` from user commands table
- Remove `%flag`, `%unflag`, `%flagged` from admin commands table
- Remove `%llmkeys` from admin commands table
- Remove `%animate` / `%video` from protection matrix
- Remove animate rate-limit config from rate-limit config block
- Remove animate from IRC staging smoke checklist
- Remove `%flag`/`%unflag`/`%flagged` from staging smoke checklist (around lines 214-216)
- Remove NickServ gating reference to animate in abuse controls description
- Update "Abuse controls" feature bullet (no more manual moderation commands)
- Update Troubleshooting section: replace `%llmkeys` reference (around line 297) with `%config` alternative

**Step 2: Update CLAUDE.md**

- Remove `%animate` / `%video`, `%flag`, `%unflag`, `%flagged`, `%llmkeys` from IRC commands table
- Remove `safe_key_display` reference from Security Patterns section (around line 166)
- Remove animate from "Adding a New Command" section config references if present

**Step 3: Update plugins/llm/README.md**

- Remove llmkeys command reference (line 42) and `safe_key_display` API reference (line 58)
- Remove `%animate`/`%video` from protection matrix (line 83)
- Remove `%flag`/`%unflag`/`%flagged` moderation references (lines 87-89)
- Remove animate rate limit config (lines 90-93)
- Remove staging smoke test references (lines 97-100)

**Step 4: Update i18n locale files**

Remove translated strings for removed commands from `plugins/llm/locales/messages.pot` and all `.po` files (`de.po`, `fi.po`, `fr.po`, `it.po`, `ru.po`). Search for "llmkeys", "flag", "animate", "video" strings and remove the corresponding `msgid`/`msgstr` pairs.

**Step 5: Run lint (for any markdown issues)**

```bash
make lint
```

---

### Task 10: Final verification

**Step 1: Search for orphaned references**

```bash
grep -rn "animate\|video\|flag_user\|flagged\|llmkeys\|safe_key_display" plugins/llm/src/ plugins/llm/tests/ --include="*.py"
```

Expect only benign hits (e.g., "flag" in unrelated words). Fix any real orphans.

**Step 2: Run full preflight**

```bash
make preflight
```

Expected: format clean, lint clean, typecheck clean, all tests pass, coverage >= 80%

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove flag/unflag/flagged, llmkeys, and animate/video commands

Drop redundant commands: flagging (use Limnoria ignore instead),
llmkeys (use %config), and animate/video (feature removed).
Remove all supporting code, config, tests, and documentation."
```
