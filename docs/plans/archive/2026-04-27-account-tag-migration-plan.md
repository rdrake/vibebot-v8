---
status: revised-after-second-review
date: 2026-04-27
overview: 2026-04-27-account-tag-migration-overview.md
---

# Account-tag Identity Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate identity resolution from `irc.state.nickToAccount(nick)` to a two-layer resolver that prefers `msg.server_tags['account']` (IRCv3 `account-tag`), capture the requesting account onto pending task rows, then drop the slow `MODE +b` and (conditionally) auto-`WHO` queries that Limnoria fires on every channel join.

**Architecture:** A new `_account_from_msg(irc, msg)` helper centralizes account resolution: layer 1 reads `msg.server_tags['account']`, layer 2 falls back to Limnoria's session cache via `irc.state.nickToAccount(nick)`. Five sender-side call sites in `plugin.py` switch to it. The `pending_tasks` table gains a nullable `account` column captured at submission so delivery-time logging doesn't need a late lookup. A monkey-patch on `supybot.irclib.Irc.doJoin` strips `MODE +b` unconditionally and strips auto-`WHO` when both `account-tag` and `extended-join` are ACK'd. A one-line casemap fix in `_maybe_migrate_nick` rides along.

**Tech Stack:** Python 3.14, Limnoria (Supybot), SQLite (`PRAGMA user_version` migrations), pytest with `pytest-mock`.

**Source of truth:** `docs/plans/2026-04-27-account-tag-migration-overview.md` (high-level rationale, scope, hard constraints).

**Hard constraints:**
- Resolver is **two layers only** — no `ircdb` hostmask layer (would silently promote unidentified users to `registered` tier).
- Auto-`WHO` drop **must** check both `'account-tag'` AND `'extended-join'` in `capabilities_ack`. `account-tag` alone does not ride on JOINs.
- Don't parse hostnames for account names. AfterNet's `<account>.users.afternet.org` cloak is context only.
- Owner/admin/trusted gating uses `ircdb.checkCapability(prefix, …)` and is untouched.
- AfterNet has no NickServ branding — error strings say "identified", not "identified with NickServ".

---

## Task 0: Move `plugin_env` fixture into `conftest.py`

**Why:** `plugin_env` currently lives at `plugins/llm/tests/test_commands.py:33-123` and is only visible to that one module. New tests in `test_plugin.py` (Tasks 1-7) need the same fixture. Promoting it to `conftest.py` makes it shared without duplicating the LLM-instantiation boilerplate.

**Files:**
- Modify: `plugins/llm/tests/test_commands.py:33-123` (delete the local copy)
- Modify: `plugins/llm/tests/conftest.py` (add the fixture verbatim, with imports)

**Step 1: Read the existing fixture**

Open `plugins/llm/tests/test_commands.py:33-123` and copy the entire `plugin_env` fixture, including its inner `_assistant_request_bridge` closure. Note the imports it depends on (`LLM`, `make_registry_side_effect`, `threading`, `time`).

**Step 2: Move fixture to `conftest.py`**

Insert the fixture into `plugins/llm/tests/conftest.py` between `mock_irc` (line 90-104) and `make_registry_side_effect` (line 107). Add the imports at the top:

```python
import threading
import time

from llm.plugin import LLM
```

Then paste the fixture body unchanged. The signature should be:

```python
@pytest.fixture
def plugin_env(mocker: MockerFixture):
    """Create an LLM plugin instance wired to mocked dependencies.

    Returns (plugin, mock_irc, mock_msg) ready for command invocation.
    """
    # ... body identical to the version moved from test_commands.py:33-123
```

**Step 3: Delete the local copy from `test_commands.py`**

Remove lines 33-123 from `test_commands.py` (the entire `plugin_env` fixture and its closure). The header comment block at line 30-32 can stay or go.

**Step 4: Run the full suite**

```bash
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS. The fixture move is purely a relocation — every test in `test_commands.py` that depended on it picks it up from `conftest.py` automatically.

**Step 5: Commit**

```bash
git add plugins/llm/tests/test_commands.py plugins/llm/tests/conftest.py
git commit -m "test(llm): move plugin_env fixture into shared conftest"
```

---

## Pre-flight

Before starting Task 1, verify the working tree is clean and tests are green from a known baseline.

```bash
git status
cd /Users/rdrake/workspace/afternet/vibebot-v8
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: all tests pass. If anything is red on `main`, stop and fix it first — this plan assumes a green baseline.

Also confirm nothing reads channel ban-list state (so dropping `MODE +b` is safe):

```bash
grep -rn '\.bans\b' plugins/
```

Expected: no matches (already verified during planning).

---

## Task 1: Add `_account_from_msg` resolver with TDD

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (add new method near `_resolve_nick_to_identity` at line ~1063)
- Test: `plugins/llm/tests/test_plugin.py` (add new test class)

**Step 1: Read the existing helper neighborhood**

Open `plugins/llm/src/llm/plugin.py:1063-1110` to see `_resolve_nick_to_identity` and `_maybe_migrate_nick`. The new helper goes immediately above `_resolve_nick_to_identity`.

**Step 2: Write the failing tests**

`plugin_env` is now a shared fixture (Task 0) returning `(plugin, mock_irc, mock_msg)`. Append a new class to `plugins/llm/tests/test_plugin.py`:

```python
class TestAccountFromMsg:
    """Two-layer account resolver: server_tags then state cache."""

    def test_layer1_account_tag_wins(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "tag_acct"

    def test_layer2_state_cache_when_no_tag(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_returns_none_when_unknown(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_keyerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = KeyError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_state_cache_attributeerror_returns_none(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.side_effect = AttributeError
        mock_msg.server_tags = {}

        assert plugin._account_from_msg(mock_irc, mock_msg) is None

    def test_empty_string_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        # account-tag value of "" or "*" means "logged out" per IRCv3 — treat as no tag.
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": ""}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"

    def test_star_tag_falls_through(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "cached_acct"
        mock_msg.server_tags = {"account": "*"}

        assert plugin._account_from_msg(mock_irc, mock_msg) == "cached_acct"
```

Note: every test sets `mock_msg.server_tags` explicitly. The `plugin_env` fixture's default `mock_msg` is a bare `MagicMock`, so `msg.server_tags.get("account")` would otherwise return a truthy MagicMock — set to `{}` or `{"account": "..."}` for deterministic behavior.

**Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestAccountFromMsg -v
```

Expected: 7 FAIL with `AttributeError: 'LLM' object has no attribute '_account_from_msg'`

**Step 4: Implement `_account_from_server_tags` and `_account_from_msg`**

Insert at `plugins/llm/src/llm/plugin.py` immediately above `_resolve_nick_to_identity` (currently at line 1063). Two functions: one static helper that handles the IRCv3 layer-1 read (reusable from `service.py` stash sites without an `irc` reference), and the full two-layer method.

```python
@staticmethod
def _account_from_server_tags(msg: IrcMsg) -> str | None:
    """Layer 1 of the account resolver — IRCv3 ``account-tag`` only.

    Returns the tag value when present and not the IRCv3 logout sentinel
    (``"*"`` or empty string), otherwise None. No ``irc`` reference needed,
    so this is callable from places (like the service-layer stash path)
    that don't have one in scope.
    """
    if not msg.server_tags:
        return None
    tag = msg.server_tags.get("account")
    if tag and tag != "*":
        return tag
    return None

def _account_from_msg(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
    """Resolve the requesting user's account name from an incoming message.

    Two layers, in order:
    1. ``msg.server_tags['account']`` via :meth:`_account_from_server_tags`
       — the IRCv3 ``account-tag`` capability. Rides on every
       PRIVMSG/NOTICE/TAGMSG from an identified user, so it's valid even
       for users idling in-channel since before bot start.
    2. ``irc.state.nickToAccount(nick)`` — Limnoria's session cache.
       Populated by account-tag ingest, account-notify, extended-join,
       and WHO replies.

    Returns ``None`` when the user is not identified or unknown.

    Note: this resolver does NOT consult ``ircdb`` hostmask matching. That
    path would silently promote unidentified users to the ``registered``
    tier; owner/admin/trusted gating uses ``ircdb.checkCapability(prefix, …)``
    separately and is unaffected.
    """
    tag_account = self._account_from_server_tags(msg)
    if tag_account:
        return tag_account
    nick = ircutils.nickFromHostmask(msg.prefix)
    try:
        return irc.state.nickToAccount(nick)
    except (KeyError, AttributeError):
        return None
```

**Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestAccountFromMsg -v
```

Expected: 7 PASS.

**Step 6: Run full plugin test suite (regression check)**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py -q
```

Expected: PASS (resolver is additive, no callers yet).

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): add _account_from_msg two-layer resolver (account-tag → state cache)"
```

---

## Task 2: Default `mock_msg.server_tags` to `{}` in `plugin_env`

**Files:**
- Modify: `plugins/llm/tests/conftest.py` (the `plugin_env` fixture moved in Task 0)

**Why:** Existing tests build `mock_msg` as a bare `MagicMock`, so after the resolver migration `msg.server_tags.get("account")` returns a truthy MagicMock by default — every preflight call would resolve to a fake account. Setting `mock_msg.server_tags = {}` in the fixture itself makes per-test overrides additive (`mock_msg.server_tags = {"account": "X"}`) and stops the fixture leak before it spreads.

**Step 1: Locate the `mock_msg` block**

In `conftest.py`, find the `plugin_env` fixture's `mock_msg` setup (moved from `test_commands.py:49-54`):

```python
mock_msg = mocker.MagicMock()
mock_msg.prefix = "testnick!user@host"
mock_msg.args = ("#test", "test message")
mock_msg.time = time.time() + 100  # future time -- not ZNC playback
mock_msg.channel = "#test"
mock_msg.nick = "testnick"
```

**Step 2: Add explicit `server_tags = {}`**

Append:

```python
mock_msg.server_tags = {}  # default: no IRCv3 account-tag
```

**Step 3: Run the suite**

```bash
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS. No callers depend on the implicit MagicMock behavior of `server_tags`.

**Step 4: Commit**

```bash
git add plugins/llm/tests/conftest.py
git commit -m "test(llm): default mock_msg.server_tags to empty dict in plugin_env"
```

---

## Task 3: Migrate `_require_account` to `_account_from_msg`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1205-1219`
- Test: `plugins/llm/tests/test_plugin.py`
- Modify: `plugins/llm/tests/test_plugin.py:1992-2027` (existing `_require_account` tests need to drop the "NickServ" assertion)

**Step 1: Pre-audit — find tests that assert on the old error string**

```bash
grep -n "NickServ" plugins/llm/tests/test_plugin.py plugins/llm/tests/test_commands.py
```

Expected hits include `test_plugin.py:2027` (`assert "NickServ" in mock_irc.error.call_args[0][0]`). These will need updating in Step 5 (after the implementation flips the error string).

**Step 2: Write the failing test**

Append to `test_plugin.py`:

```python
class TestRequireAccountUsesResolver:
    """_require_account must read account-tag via _account_from_msg."""

    def test_returns_tag_when_present(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._require_account(mock_irc, mock_msg) == "tag_acct"
        mock_irc.error.assert_not_called()

    def test_returns_none_and_errors_when_unidentified(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._require_account(mock_irc, mock_msg) is None
        mock_irc.error.assert_called_once()
        err_text = mock_irc.error.call_args[0][0]
        assert "NickServ" not in err_text
        assert "identified" in err_text.lower()
```

**Step 3: Run to confirm new test fails**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestRequireAccountUsesResolver -v
```

Expected: `test_returns_tag_when_present` FAILS (current code only checks `nickToAccount`).

The current string at `plugin.py:1217` is:

```python
irc.error(_("You must be identified with NickServ to use this command."))
```

Per AfterNet conventions, drop the "NickServ" wording.

**Step 4: Update `_require_account`**

Replace `plugin.py:1205-1219` with:

```python
def _require_account(self, irc: callbacks.Irc, msg: IrcMsg) -> str | None:
    """Require account identification. Returns account name or None.

    Uses the IRCv3 account-tag-aware resolver. When the user is not
    identified, sends an error reply and returns None. Callers should
    ``return`` immediately when None is returned.
    """
    account = self._account_from_msg(irc, msg)
    if not account:
        irc.error(_("You must be identified to use this command."))
        return None
    return account
```

**Step 5: Update the pre-existing `_require_account` test**

`test_plugin.py:2027` currently asserts `"NickServ" in mock_irc.error.call_args[0][0]`. Change the assertion to match the new wording:

```python
err_text = mock_irc.error.call_args[0][0]
assert "NickServ" not in err_text
assert "identified" in err_text.lower()
```

**Step 6: Run new tests + full suite**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestRequireAccountUsesResolver -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS. If a test fails because it set `mock_irc.state.nickToAccount.return_value = "X"` AND a bare-MagicMock `mock_msg`, that's the fixture-leak risk Task 2 already prevented — but if a test bypasses `plugin_env` (`test_plugin.py` has its own `plugin` fixture at line 2175 for `TestRunPreflight`), it will need its own `mock_msg.server_tags = {}` set. Audit:

```bash
grep -n "MagicMock\|mock_msg\s*=" plugins/llm/tests/test_plugin.py | head
```

For each `mock_msg = mocker.MagicMock()` site, ensure `server_tags` is explicit or migrate the test to use `plugin_env`.

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): _require_account reads account-tag via resolver"
```

---

## Task 4: Migrate `_resolve_tier` to `_account_from_msg`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1465-1474`
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Write the failing test**

The shared `plugin_env` fixture mocks `ircdb.checkCapability` to return False for owner/admin/trusted (so users default to registered/unregistered).

```python
class TestResolveTierUsesResolver:
    def test_registered_tier_via_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "registered"

    def test_unregistered_when_no_tag_no_cache(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mock_msg.server_tags = {}

        assert plugin._resolve_tier(mock_irc, mock_msg) == "unregistered"
```

**Step 2: Run, confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestResolveTierUsesResolver -v
```

Expected: `test_registered_tier_via_account_tag` FAILS (current code uses `nickToAccount` only, returns "unregistered").

**Step 3: Update `_resolve_tier`**

Replace `plugin.py:1469-1474` (just the bottom half — keep the owner/admin/trusted checks):

```python
nick = ircutils.nickFromHostmask(prefix)
try:
    account = irc.state.nickToAccount(nick)
except (KeyError, AttributeError):
    account = None
return "registered" if account else "unregistered"
```

with:

```python
account = self._account_from_msg(irc, msg)
return "registered" if account else "unregistered"
```

The resulting method body becomes (full context for clarity):

```python
def _resolve_tier(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
    prefix = msg.prefix
    if ircdb.checkCapability(prefix, "owner"):
        return "owner"
    if ircdb.checkCapability(prefix, "admin"):
        return "admin"
    if ircdb.checkCapability(prefix, "trusted"):
        return "trusted"
    account = self._account_from_msg(irc, msg)
    return "registered" if account else "unregistered"
```

**Step 4: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestResolveTierUsesResolver -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS, no regressions.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): _resolve_tier reads account-tag via resolver"
```

---

## Task 5: Migrate `_run_preflight` (both branches) to `_account_from_msg`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1253-1277`
- Modify: `plugins/llm/tests/test_plugin.py:2202-2227` (existing `TestRunPreflight` tests need `mock_msg.server_tags` set)
- Test: `plugins/llm/tests/test_plugin.py` (new tests)

**Why both branches:** the `require_account=True` branch already calls `_require_account` (now resolver-aware). The `else` branch at 1271-1277 still does a raw `nickToAccount` call and a separate `_get_identity(irc, msg)` call — collapse both to the resolver.

**Step 1: Update existing TestRunPreflight tests to set `server_tags`**

`TestRunPreflight` at `test_plugin.py:2172-2227` builds `mock_msg = mocker.MagicMock()` directly without going through `plugin_env`. After the migration, `_account_from_msg` will read `msg.server_tags.get("account")` — on a bare MagicMock that returns a truthy MagicMock, breaking both existing tests for the wrong reason.

Edit both tests in that class to add `mock_msg.server_tags = {}` immediately after `mock_msg.args = (...)`:

```python
mock_msg = mocker.MagicMock()
mock_msg.prefix = "alice!user@host"
mock_msg.args = ("#test", "hello")
mock_msg.server_tags = {}  # explicit: no IRCv3 account-tag
```

Apply this in both `test_preflight_passes_for_ask` (line 2202) and `test_preflight_blocks_unidentified_for_draw` (line 2215).

**Step 2: Write the new failing test**

Append to `test_plugin.py`:

```python
class TestPreflightOptionalAccountTag:
    """When require_account=False, account-tag should still populate the account."""

    def test_optional_path_picks_up_account_tag(self, plugin_env):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None  # cache empty
        mock_msg.server_tags = {"account": "tag_acct"}

        result = plugin._run_preflight(
            mock_irc, mock_msg, text="hi", command="ask", require_account=False
        )
        assert result.account == "tag_acct"
```

Verify `PreflightResult` import in the file with `grep -n "PreflightResult" plugins/llm/tests/test_plugin.py`. If not imported at top, the test still works because the assertion accesses `result.account` (a NamedTuple attribute).

**Step 3: Confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestPreflightOptionalAccountTag -v
```

Expected: FAIL — `result.account` is `None` because the else branch only consults `nickToAccount` and the cache is empty.

**Step 4: Update both branches**

Replace `plugin.py:1253-1277`:

```python
# --- account resolution ---
if require_account:
    account = self._require_account(irc, msg)
    if account is None:
        nick = ircutils.nickFromHostmask(msg.prefix)
        self.db.log_usage(
            nick,
            channel,
            command,
            "",
            0,
            0,
            0.0,
            prompt=text,
            status="auth_failure",
        )
        return PreflightResult(blocked=True, nick=nick, channel=channel, account=None)
    nick = self._resolve_nick_to_identity(irc, ircutils.nickFromHostmask(msg.prefix))
else:
    raw_nick = ircutils.nickFromHostmask(msg.prefix)
    try:
        account = irc.state.nickToAccount(raw_nick)
    except (KeyError, AttributeError):
        account = None
    nick = self._get_identity(irc, msg)
```

with:

```python
# --- account resolution ---
if require_account:
    account = self._require_account(irc, msg)
    if account is None:
        nick = ircutils.nickFromHostmask(msg.prefix)
        self.db.log_usage(
            nick,
            channel,
            command,
            "",
            0,
            0,
            0.0,
            prompt=text,
            status="auth_failure",
        )
        return PreflightResult(blocked=True, nick=nick, channel=channel, account=None)
    # On success _require_account returned the account; trigger nick→account migration.
    raw_nick = ircutils.nickFromHostmask(msg.prefix)
    self._maybe_migrate_nick(raw_nick, account)
    nick = account
else:
    account = self._account_from_msg(irc, msg)
    if account:
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        self._maybe_migrate_nick(raw_nick, account)
        nick = account
    else:
        nick = ircutils.nickFromHostmask(msg.prefix)
```

Note: this preserves the existing behavior of `_resolve_nick_to_identity` (account fallback to nick + migration trigger) without going through it — we already have the account in hand.

**Step 5: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestPreflightOptionalAccountTag -v
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestRunPreflight -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS.

**Step 6: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "refactor(llm): _run_preflight reads account-tag via resolver in both branches"
```

---

## Task 6: Update `_get_identity` / `_resolve_nick_to_identity` chain

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1063-1110, 1183-1203`

**Why:** After Task 5, `_get_identity(irc, msg)` is no longer called from `_run_preflight`. But it's still called from `_check_rate_limit` paths and other helpers. Audit and update:

```bash
grep -n "_get_identity\|_resolve_nick_to_identity" plugins/llm/src/llm/plugin.py
```

For each `_get_identity(irc, msg)` call site that has `msg` in scope, change the implementation to use the resolver. Keep `_resolve_nick_to_identity(irc, nick)` as a nick-only fallback for the two call sites without `msg` (line 620 task delivery, line 2342 `%usage <nick>`) — these stay best-effort as documented in the overview.

**Step 1: Audit call sites**

```bash
grep -n "_get_identity\|_resolve_nick_to_identity" plugins/llm/src/llm/plugin.py
```

Catalog each into "has `msg`" vs "nick-only".

**Step 2: Update `_get_identity` to use the resolver (and fix docstring)**

Replace `plugin.py:1183-1203` entirely. The old docstring referenced "delegates to :meth:`_resolve_nick_to_identity`" which is no longer true:

```python
def _get_identity(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
    """Resolve a message sender to a stable identity (account or nick).

    Reads the IRCv3 account-tag (or layer-2 session cache) via
    :meth:`_account_from_msg`. Triggers a one-time DB migration of
    nick→account usage rows on first successful resolution per session.
    Falls back to the bare nick when no account can be resolved.
    """
    nick = ircutils.nickFromHostmask(msg.prefix)
    account = self._account_from_msg(irc, msg)
    if account:
        self._maybe_migrate_nick(nick, account)
        return account
    return nick
```

Leave `_resolve_nick_to_identity(irc, nick)` (line 1063-1089) untouched — it's the nick-only path retained for the two `msg`-less callers (delivery-time `log_usage` at line 620, `%usage <nick>` at line 2342).

**Step 3: Run full plugin tests**

```bash
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS. If a test breaks because it now resolves an identity it didn't before, update the test — that's the intended behavior change.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/plugin.py
git commit -m "refactor(llm): _get_identity reads account-tag via resolver"
```

---

## Task 7: Fix the casemap nit in `_maybe_migrate_nick`

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py:1102-1107`
- Test: `plugins/llm/tests/test_plugin.py`

**Why:** `.lower()` is wrong for RFC1459 — `{}|^` are the lowercase of `[]\\~`. Happens to work on AfterNet (no users with those characters in their nicks) but the migration touches identity code, so fix it now.

**Step 1: Write the failing test**

```python
class TestMaybeMigrateNickCasemap:
    def test_rfc1459_brackets_treated_as_same(self, plugin_env, mocker):
        plugin = plugin_env["plugin"]
        plugin.db.migrate_nick = mocker.MagicMock(return_value=0)
        # In RFC1459, "[" lowers to "{". toLower("Foo[") == "foo{".
        plugin._maybe_migrate_nick("Foo[", "foo{")
        plugin.db.migrate_nick.assert_not_called()

    def test_distinct_account_still_migrates(self, plugin_env, mocker):
        plugin = plugin_env["plugin"]
        plugin.db.migrate_nick = mocker.MagicMock(return_value=1)
        plugin._maybe_migrate_nick("Foo", "BarAccount")
        plugin.db.migrate_nick.assert_called_once_with("Foo", "BarAccount")
```

**Step 2: Confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestMaybeMigrateNickCasemap -v
```

Expected: `test_rfc1459_brackets_treated_as_same` FAILS — current `.lower()` says `"foo["` ≠ `"foo{"`.

**Step 3: Fix it**

Replace `plugin.py:1102-1107`:

```python
if old_nick.lower() == account.lower():
    return
key = old_nick.lower()
if key in self._migrated_nicks:
    return
```

with:

```python
if ircutils.toLower(old_nick) == ircutils.toLower(account):
    return
key = ircutils.toLower(old_nick)
if key in self._migrated_nicks:
    return
```

`ircutils` is already imported at the top of `plugin.py` (used by `nickFromHostmask`). Verify with `grep -n "^import\|^from supybot" plugins/llm/src/llm/plugin.py | head`.

**Step 4: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestMaybeMigrateNickCasemap -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "fix(llm): use ircutils.toLower for RFC1459-correct nick comparison"
```

---

## Task 8: Add `account` column to `pending_tasks` (schema migration)

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:18, 145-296` (bump SCHEMA_VERSION, add migration block)
- Modify: `plugins/llm/src/llm/persistence.py:65-87` (extend `PendingTaskRow` NamedTuple)
- Modify: `plugins/llm/src/llm/persistence.py:487-492` (extend `_PENDING_TASK_COLUMNS`)
- Modify: `plugins/llm/src/llm/persistence.py:494-551` (extend `save_pending_task` signature & INSERT)
- Test: `plugins/llm/tests/test_persistence.py`

**Step 1: Write the failing schema test**

Append to `test_persistence.py`:

```python
class TestPendingTasksAccountColumn:
    def test_save_with_account(self, test_db):
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            request_data="{}",
            submitted_at=1000.0,
            expires_at=2000.0,
            next_attempt_at=1000.0,
            account="alice_acct",
        )
        assert task_id > 0
        rows = test_db.load_pending_tasks()
        assert len(rows) == 1
        assert rows[0].account == "alice_acct"

    def test_save_with_null_account(self, test_db):
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            request_data="{}",
            submitted_at=1000.0,
            expires_at=2000.0,
            next_attempt_at=1000.0,
            account=None,
        )
        assert task_id > 0
        rows = test_db.load_pending_tasks()
        assert rows[0].account is None
```

**Step 2: Confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_persistence.py::TestPendingTasksAccountColumn -v
```

Expected: FAIL — `save_pending_task() got an unexpected keyword argument 'account'`.

**Step 3: Bump the schema version**

`persistence.py:18`:

```python
SCHEMA_VERSION = 7
```

→

```python
SCHEMA_VERSION = 8
```

**Step 4: Add the v8 migration block**

In `_migrate()`, after the `if current_version < 7:` block (around line 289) and before the `PRAGMA user_version` stamp (line 293), insert:

```python
if current_version < 8:
    conn.executescript("""
        ALTER TABLE pending_tasks
            ADD COLUMN account TEXT;
    """)
    conn.commit()
```

Note: nullable, no `NOT NULL`, no `DEFAULT`. NULL means "user wasn't identified at submission time" — delivery falls back to `nick`.

**Step 5: Extend `PendingTaskRow`**

`persistence.py:65-87` — add `account: str | None` as the last field:

```python
class PendingTaskRow(NamedTuple):
    """A pending task loaded from the database."""

    id: int
    task_type: str
    nick: str
    reply_target: str
    is_channel: int
    prompt_preview: str
    model: str
    request_data: str
    submitted_at: float
    expires_at: float
    attempt_count: int
    next_attempt_at: float
    claimed_until: float
    last_error: str
    delivery_state: str
    result_payload: str
    last_delivery_error: str
    delivery_attempt_count: int
    origin_request_id: str
    account: str | None
```

**Step 6: Extend `_PENDING_TASK_COLUMNS`**

`persistence.py:487-492`:

```python
_PENDING_TASK_COLUMNS = (
    "id, task_type, nick, reply_target, is_channel, prompt_preview, model, "
    "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
    "claimed_until, last_error, delivery_state, result_payload, "
    "last_delivery_error, delivery_attempt_count, origin_request_id, account"
)
```

(`account` appended last to keep positional order matching the NamedTuple.)

**Step 7: Extend `save_pending_task`**

`persistence.py:494-551`:

```python
def save_pending_task(
    self,
    task_type: str,
    nick: str,
    reply_target: str,
    is_channel: bool,
    prompt_preview: str,
    model: str,
    request_data: str,
    submitted_at: float,
    expires_at: float,
    next_attempt_at: float,
    origin_request_id: str = "",
    account: str | None = None,
) -> int:
    """Save a pending task to the database. (… keep existing docstring; add:)

    Args:
        …
        account: Resolved account name at submission time, or None if the
            requester was not identified. Delivery-time logging reads this
            directly instead of doing a late nick→account lookup.
    """
    conn = self._connect()
    try:
        cursor = conn.execute(
            "INSERT INTO pending_tasks "
            "(task_type, nick, reply_target, is_channel, prompt_preview, model, "
            "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
            "claimed_until, last_error, origin_request_id, account) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, '', ?, ?)",
            (
                task_type,
                nick,
                reply_target,
                1 if is_channel else 0,
                prompt_preview,
                model,
                request_data,
                submitted_at,
                expires_at,
                next_attempt_at,
                origin_request_id,
                account,
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        pass
```

**Step 8: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_persistence.py -v
```

Expected: PASS, including the new ones. Existing `save_pending_task` callers in tests don't pass `account=` — they should still work because of the default.

```bash
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS overall.

**Step 9: Commit**

```bash
git add plugins/llm/src/llm/persistence.py plugins/llm/tests/test_persistence.py
git commit -m "feat(llm): add nullable account column to pending_tasks (schema v8)"
```

---

## Task 9: Capture account at task submission

**Files:**
- Modify: `plugins/llm/src/llm/service.py:1135-1203` (`_stash_timeout` signature & body)
- Modify: `plugins/llm/src/llm/service.py:1685-1704, 2580-2602` (call sites — pass account through)
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Find existing patterns**

```bash
grep -n "_stash_timeout\|save_pending_task\|make_service" plugins/llm/tests/test_service.py | head -20
```

The existing test pattern uses the `make_service` factory fixture from `conftest.py:213`. It returns `(service, mock_plugin)`.

**Step 2: Write the failing test**

Append to `test_service.py` (use the existing fixture style — `make_service` returning `(service, mock_plugin)`):

```python
class TestStashTimeoutCapturesAccount:
    def test_passes_account_to_save_pending_task(
        self, make_service, mocker: MockerFixture
    ):
        service, mock_plugin = make_service()
        save = mocker.MagicMock(return_value=42)
        mock_plugin.db = mocker.MagicMock(save_pending_task=save)
        mock_plugin.registryValue = mocker.MagicMock(return_value=300)

        service._stash_timeout(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt="hi",
            model="gpt-4",
            request_data={"messages": []},
            submitted_at=1000.0,
            account="alice_acct",
        )
        save.assert_called_once()
        kwargs = save.call_args.kwargs
        assert kwargs["account"] == "alice_acct"

    def test_account_defaults_to_none(self, make_service, mocker: MockerFixture):
        service, mock_plugin = make_service()
        save = mocker.MagicMock(return_value=42)
        mock_plugin.db = mocker.MagicMock(save_pending_task=save)
        mock_plugin.registryValue = mocker.MagicMock(return_value=300)

        service._stash_timeout(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt="hi",
            model="gpt-4",
            request_data={"messages": []},
            submitted_at=1000.0,
        )
        kwargs = save.call_args.kwargs
        assert kwargs["account"] is None
```

If `make_service` returns `(service, mock_plugin)` differently (named tuple, etc.), adapt the unpacking. Confirm by reading `conftest.py:213-340`.

**Step 3: Confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_service.py::TestStashTimeoutCapturesAccount -v
```

Expected: FAIL — `_stash_timeout()` got an unexpected keyword argument `account`.

**Step 4: Update `_stash_timeout`**

`plugins/llm/src/llm/service.py:1135-1203` — add `account: str | None = None` to the signature, and pass it through to `save_pending_task`:

```python
def _stash_timeout(
    self,
    task_type: str,
    nick: str,
    reply_target: str,
    is_channel: bool,
    prompt: str,
    model: str,
    request_data: dict,
    submitted_at: float,
    account: str | None = None,
) -> bool:
    """Stash a timed-out request for background retry.

    …existing docstring…

    Args:
        …
        account: Resolved account name at submission, or None if the
            requester was not identified. Persisted to pending_tasks.account
            so delivery logging doesn't need a late nick→account lookup.
    """
    expiry = self.plugin.registryValue(f"{task_type}Expiry")
    if not expiry:
        return False

    db = getattr(self.plugin, "db", None)
    if db is None:
        self.log.warning("No database available for pending task stashing")
        return False

    prompt_preview = prompt[:100]
    expires_at = submitted_at + expiry
    data_json = json.dumps(request_data)

    task_id = db.save_pending_task(
        task_type=task_type,
        nick=nick,
        reply_target=reply_target,
        is_channel=is_channel,
        prompt_preview=prompt_preview,
        model=model,
        request_data=data_json,
        submitted_at=submitted_at,
        expires_at=expires_at,
        next_attempt_at=submitted_at,
        origin_request_id=request_id.get(),
        account=account,
    )
    self.log.info(
        "Stashed timed-out %s request as pending_task id=%d (expires in %ds)",
        task_type,
        task_id,
        expiry,
    )

    schedule_wakeup = getattr(self.plugin, "_schedule_queue_wakeup", None)
    if schedule_wakeup is not None:
        schedule_wakeup(at_time=submitted_at)

    return True
```

**Step 5: Update both call sites to pass `account`**

Both call sites have `msg` in scope under an existing `if msg:` guard. Use the new `LLM._account_from_server_tags(msg)` static helper (added in Task 1) — it reads layer 1 (account-tag) only, no `irc` reference needed. This is a tactical exception to "go through `_account_from_msg`" because the service has no `irc` handle; the helper is the shared seam that keeps the IRCv3 sentinel logic (`""`, `"*"` → None) DRY.

At `service.py:1685-1704`, replace:

```python
nick = ""
reply_target = ""
is_channel = False
if msg:
    nick = msg.nick or ""
    reply_target = msg.args[0] if msg.args else ""
    is_channel = bool(reply_target) and ircutils.isChannel(reply_target)
stashed = self._stash_timeout(
    task_type=command,
    nick=nick,
    reply_target=reply_target,
    is_channel=is_channel,
    prompt=prompt,
    model=model,
    request_data={"messages": messages},
    submitted_at=time.time(),
)
```

with:

```python
nick = ""
reply_target = ""
is_channel = False
account: str | None = None
if msg:
    nick = msg.nick or ""
    reply_target = msg.args[0] if msg.args else ""
    is_channel = bool(reply_target) and ircutils.isChannel(reply_target)
    # Best-effort account capture via IRCv3 account-tag.
    # No irc handle here, so layer-2 (state cache) is skipped; NULL is OK
    # because delivery-time logging falls back to nick.
    from llm.plugin import LLM
    account = LLM._account_from_server_tags(msg)
stashed = self._stash_timeout(
    task_type=command,
    nick=nick,
    reply_target=reply_target,
    is_channel=is_channel,
    prompt=prompt,
    model=model,
    request_data={"messages": messages},
    submitted_at=time.time(),
    account=account,
)
```

Apply the same edit at `service.py:2580-2602` (the draw timeout site). The pattern is identical.

If the lazy `from llm.plugin import LLM` causes a circular-import warning at test time, hoist it to a module-level import in `service.py` (most LLM-package modules already cross-import safely).

**Step 6: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_service.py::TestStashTimeoutCapturesAccount -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS.

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "feat(llm): capture account-tag onto pending_tasks at stash time"
```

---

## Task 10: Plumb `account` onto `PendingTaskResult` and use it at delivery

**Files:**
- Modify: `plugins/llm/src/llm/service.py:231-247` (add `account` field to `PendingTaskResult`)
- Modify: `plugins/llm/src/llm/service.py:1417` (expired-sweep ctor — populate from row)
- Modify: `plugins/llm/src/llm/service.py:1538` (delivery-phase ctor — populate from task row)
- Modify: `plugins/llm/src/llm/plugin.py:617-630` (use `r.account` at log-usage)
- Test: `plugins/llm/tests/test_service.py` and `plugins/llm/tests/test_plugin.py`

**Why this task is bigger than it looked:** the delivery-time snippet at `plugin.py:620` does `identity = self._resolve_nick_to_identity(irc_conn, nick)` — a live `nickToAccount` lookup that fails for users who logged out between submission and delivery. The captured `pending_tasks.account` (Task 8) is the source of truth, but it has to flow through `PendingTaskResult` to reach this site.

**Step 1: Audit all `PendingTaskResult` construction sites**

```bash
grep -n "PendingTaskResult(" plugins/llm/src/llm/service.py
```

Expect ~8 hits. The two that matter (carry a `pending_tasks` row) are:
- `service.py:1417` — expired-sweep loop, `for row in expired_rows:` — has access to `row.account`.
- `service.py:1538` — delivery-phase loop, `task = ...` — has access to `task.account`.

The other six (lines 1270/1283/1308/1334/1346/1360/1371) are construction sites for in-flight failures where there is no DB row yet (errors during fetch, parse, etc.). They should default `account=None` — falling back to nick at delivery-time is fine.

**Step 2: Write the failing test**

Append to `test_service.py`:

```python
class TestPendingTaskResultCarriesAccount:
    def test_account_field_default_is_none(self):
        from llm.service import PendingTaskResult
        r = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
        )
        assert r.account is None

    def test_account_field_round_trips(self):
        from llm.service import PendingTaskResult
        r = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            account="alice_acct",
        )
        assert r.account == "alice_acct"
```

**Step 3: Confirm failure**

```bash
.venv/bin/pytest plugins/llm/tests/test_service.py::TestPendingTaskResultCarriesAccount -v
```

Expected: FAIL — `PendingTaskResult.__new__()` got an unexpected keyword argument `account`.

**Step 4: Add the field**

`service.py:231-247` — append `account` after `delivery_attempt_count`:

```python
class PendingTaskResult(NamedTuple):
    """Result from checking a single pending task."""

    status: str
    task_type: str
    nick: str
    reply_target: str
    is_channel: bool
    prompt_preview: str
    model: str
    content: str = ""
    reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    task_id: int | None = None
    delivery_attempt_count: int = 0
    account: str | None = None  # captured at submission via account-tag
```

**Step 5: Populate at the two row-aware ctor sites**

`service.py:1417` (expired-sweep): add `account=row.account,` to the kwargs.

`service.py:1538` (delivery-phase): add `account=task.account,` to the kwargs (the loop variable is named `task`, not `row` — verify by reading lines 1500-1560).

Other ctors (1270/1283/1308/1334/1346/1360/1371): leave alone — they default to `account=None`.

**Step 6: Write the delivery-side test**

Append to `test_plugin.py`:

```python
class TestDeliveryLogsAccountWhenPresent:
    def test_log_usage_uses_captured_account(self, plugin_env, mocker: MockerFixture):
        plugin, _, _ = plugin_env
        from llm.service import PendingTaskResult
        result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            cost=0.01,
            prompt_tokens=10,
            completion_tokens=5,
            account="alice_acct",
        )
        # Drive the deliver-result helper directly. The snippet at plugin.py:617-630
        # is inside _deliver_pending_result (or similarly named) — verify by grep.
        # If the snippet is still inline in _check_pending_tasks, factor it out
        # into a small helper named _log_pending_delivery_usage(result, nick, target)
        # in this task and call that helper from both the test and the loop.
        ...
        plugin.db.log_usage.assert_called_with(
            "alice_acct", "#chan", "ask", "gpt-4", 10, 5, 0.01
        )

    def test_log_usage_falls_back_to_resolver_when_account_null(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, _, _ = plugin_env
        from llm.service import PendingTaskResult
        result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            cost=0.01,
            account=None,
        )
        # Mock _resolve_nick_to_identity to return the legacy fallback.
        mocker.patch.object(plugin, "_resolve_nick_to_identity", return_value="alice")
        ...
        plugin.db.log_usage.assert_called_with(
            "alice", "#chan", "ask", "gpt-4", mocker.ANY, mocker.ANY, mocker.ANY
        )
```

**Step 7: Refactor the snippet into a testable helper**

The current snippet at `plugin.py:617-630` is inside the `_deliver_pending_result` (or `_check_pending_tasks` deliver loop — confirm via `grep -n "def _deliver\|def _check_pending\|world.ircs" plugins/llm/src/llm/plugin.py`). Extract into:

```python
def _log_pending_delivery_usage(
    self, result: "PendingTaskResult", nick: str, target: str
) -> None:
    """Log usage for a delivered pending task.

    Prefers the account captured at submission time; falls back to live
    resolution by nick when the captured account is NULL (e.g., user was
    unidentified at request time).
    """
    if result.cost <= 0 and result.prompt_tokens <= 0:
        return
    for irc_conn in world.ircs:
        identity = result.account or self._resolve_nick_to_identity(irc_conn, nick)
        self.db.log_usage(
            identity,
            target,
            result.task_type,
            result.model,
            result.prompt_tokens,
            result.completion_tokens,
            result.cost,
        )
        break
```

Replace the inline snippet at `plugin.py:617-630` with `self._log_pending_delivery_usage(r, nick, target)` (gated by the existing `if r.status == "completed" and delivered:` check).

**Step 8: Run tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_service.py::TestPendingTaskResultCarriesAccount -v
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestDeliveryLogsAccountWhenPresent -v
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS.

**Step 9: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/src/llm/plugin.py plugins/llm/tests/test_service.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): plumb captured account through PendingTaskResult to delivery logging"
```

---

## Task 11: Add `skipAutoWhoOnJoin` config flag

**Files:**
- Modify: `plugins/llm/src/llm/config.py` (append at end of file, after the last `registerGlobalValue`)
- Modify: `plugins/llm/tests/conftest.py` (`make_registry_side_effect` defaults — Task 12 will read this in tests)

**Step 1: Append the config registration**

```python
conf.registerGlobalValue(
    LLM,
    "skipAutoWhoOnJoin",
    registry.Boolean(
        True,
        _(
            """If True (default), suppress Limnoria's automatic WHO query on channel join
            when both 'account-tag' and 'extended-join' IRCv3 capabilities are ACK'd.
            Set False to restore the legacy WHO query (emergency disable for servers
            where account-tag/extended-join misbehave). The MODE +b ban-list query is
            always suppressed regardless of this flag — nothing reads ban state."""
        ),
    ),
)
```

**Step 2: Add the default to `make_registry_side_effect`**

In `conftest.py`, the `defaults` dict in `make_registry_side_effect` (line 119+) needs a default value for `skipAutoWhoOnJoin`. Otherwise Task 12's `_patch_irc_dojoin` reads `plugin.registryValue("skipAutoWhoOnJoin")` and the fixture's `defaults.get(key, "")` returns the empty string (falsy!) — defeating the patch silently.

Append to the defaults dict:

```python
"skipAutoWhoOnJoin": True,
```

**Step 3: Verify Limnoria loads the config without error**

```bash
.venv/bin/pytest plugins/llm/tests/test_config.py -q
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat(llm): add skipAutoWhoOnJoin config flag (default True)"
```

---

## Task 12: Patch `Irc.doJoin` and fix the `_pending_channels` startup-notification path

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (new `_will_skip_auto_who` helper, new `_patch_irc_dojoin` patcher, `__init__` invocation, and the plugin's own `doJoin` at line 788)
- Test: `plugins/llm/tests/test_plugin.py`

**Step 1: Read the upstream method and the plugin's own `doJoin`/`do315`**

Upstream `irclib.py:2456-2463`:

```python
def doJoin(self, msg):
    if msg.nick == self.nick:
        channel = msg.args[0]
        self.queueMsg(ircmsgs.who(channel, args=('%tuhnairf,1',))) # Ends with 315.
        self.queueMsg(ircmsgs.mode(channel)) # Ends with 329.
        for channel in msg.args[0].split(','):
            self.queueMsg(ircmsgs.mode(channel, '+b'))
        self.startedSync[channel] = time.time()
```

Plugin's own `doJoin` at `plugin.py:788-796`:

```python
def doJoin(self, irc, msg):
    if ircutils.strEqual(irc.nick, msg.nick):
        channel = msg.args[0]
        self._pending_channels.add(channel)
```

Plugin's `do315` at `plugin.py:798-810` discards from `_pending_channels` and fires the startup notification when the set is empty.

**The bug we're avoiding:** if WHO is skipped, `do315` (end-of-WHO numeric) never fires, so `_pending_channels` never empties via that path, so the startup notification never sends. Fortunately `do376` (end-of-MOTD) at `plugin.py:812-829` schedules a 2-second delayed callback that *also* fires the notification when `_pending_channels` is empty. So if the plugin's `doJoin` doesn't add to `_pending_channels` when WHO is being skipped, the do376 fallback will fire correctly.

**Goals:**
- Always skip `MODE +b` (unconditional — nothing reads bans).
- Skip the `WHO` only when both `'account-tag'` AND `'extended-join'` are in `capabilities_ack` AND the `skipAutoWhoOnJoin` config is True.
- Keep the channel `MODE` query (Limnoria reads channel-mode state in many places).
- When WHO is skipped, the plugin's own `doJoin` must NOT add the channel to `_pending_channels` (because `do315` won't fire). The do376 2-second fallback at `plugin.py:828` becomes the trigger for the startup notification.
- Single source of truth for the "should we skip?" decision: a `_will_skip_auto_who(irc)` helper used by both the patched `Irc.doJoin` and the plugin's `doJoin`.

**Step 2: Write the failing tests**

Append to `test_plugin.py`. The `plugin_env` fixture instantiates `LLM(mock_irc)`, which calls `_patch_irc_dojoin` from `__init__` — so the patch is already installed by the time the fixture returns.

```python
class TestPatchedDoJoin:
    """The plugin patches supybot.irclib.Irc.doJoin to skip slow auto-queries."""

    def _self_join(self, mocker: MockerFixture, channel="#test", nick="testbot"):
        msg = mocker.MagicMock()
        msg.nick = nick
        msg.args = (channel,)
        return msg

    def test_mode_b_never_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        for call in mock_irc.queueMsg.call_args_list:
            sent = call.args[0]
            if getattr(sent, "command", "") == "MODE" and "+b" in getattr(sent, "args", ()):
                pytest.fail(f"MODE +b should never be queued: {sent}")

    def test_who_skipped_when_both_caps_and_flag_enabled(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" not in commands

    def test_who_kept_when_account_tag_missing(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_extended_join_missing(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_flag_disabled(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        # Override the registry default for this test.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "skipAutoWhoOnJoin" else ""
        )
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_channel_mode_always_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc
        Irc.doJoin(mock_irc, msg)

        mode_calls = [
            c.args[0] for c in mock_irc.queueMsg.call_args_list
            if getattr(c.args[0], "command", "") == "MODE"
        ]
        # Plain MODE <channel> has args=(channel,) — length 1.
        assert any(len(getattr(m, "args", ())) == 1 for m in mode_calls)


class TestPluginDoJoinPendingChannels:
    """Plugin's own doJoin must not add to _pending_channels when WHO is skipped."""

    def test_pending_added_when_who_will_fire(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = set()  # no caps → WHO fires
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" in plugin._pending_channels

    def test_pending_NOT_added_when_who_will_be_skipped(
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" not in plugin._pending_channels, (
            "When WHO is skipped, do315 won't fire — the bot must not add to "
            "_pending_channels or startup notification will never send."
        )
```

**Step 3: Confirm tests fail**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestPatchedDoJoin -v
```

Expected: most FAIL because `doJoin` is unpatched and queues `WHO` + `MODE` + `MODE +b` unconditionally.

**Step 4: Implement the helper, the patcher, and update plugin's doJoin**

Add to `plugins/llm/src/llm/plugin.py` near the top of the LLM class (after `__init__`-adjacent helpers), a `_will_skip_auto_who` method on the LLM class plus a module-level patcher. Then update the plugin's `doJoin` at line 788 to use the same helper.

```python
def _will_skip_auto_who(self, irc: callbacks.Irc) -> bool:
    """Return True iff the auto-WHO on channel join should be suppressed.

    Gate: both 'account-tag' AND 'extended-join' IRCv3 caps must be ACK'd
    (account-tag rides on PRIVMSG-class messages; extended-join rides on
    JOIN itself — together they obviate the auto-WHO scan), AND the
    operator-controlled ``skipAutoWhoOnJoin`` config must be True.
    """
    caps = getattr(getattr(irc, "state", None), "capabilities_ack", set()) or set()
    if "account-tag" not in caps or "extended-join" not in caps:
        return False
    return bool(self.registryValue("skipAutoWhoOnJoin"))
```

Module-level patcher (place above the `LLM` class definition):

```python
def _patch_irc_dojoin(plugin: "LLM") -> None:
    """Replace supybot.irclib.Irc.doJoin to skip slow auto-queries on JOIN.

    Always: skip MODE +b (ban-list) — nothing reads ban state.
    Conditional: skip auto-WHO when ``plugin._will_skip_auto_who(irc)``.
    The plain MODE <channel> query (channel modes; ends with 329) is kept
    because Limnoria reads channel-mode state in many places.

    Re-patches on every call (e.g., plugin reload) so the closure always
    references the current LLM instance. Cheap; runs once per __init__.
    """
    from supybot import irclib, ircmsgs

    def doJoin(self, msg):
        if msg.nick != self.nick:
            return
        channel = msg.args[0]
        skip_who = plugin._will_skip_auto_who(self)
        if not skip_who:
            self.queueMsg(ircmsgs.who(channel, args=("%tuhnairf,1",)))
            # Track start of WHO sync so do315 can compute elapsed time.
            self.startedSync[channel] = time.time()
        self.queueMsg(ircmsgs.mode(channel))  # plain channel modes; ends with 329
        # Always skip MODE +b — nothing in the codebase reads ban-list state.
        # If WHO is skipped, do NOT touch startedSync — do315 will never arrive
        # and the dict would leak across rejoins.

    irclib.Irc.doJoin = doJoin
```

Update plugin's own `doJoin` at `plugin.py:788-796`:

```python
def doJoin(self, irc: callbacks.Irc, msg: IrcMsg) -> None:  # noqa: N802
    """Track channels the bot is joining for startup notification.

    When the bot joins a channel, we add it to _pending_channels.
    The channel is removed when we receive do315 (end of WHO).

    If the auto-WHO on join is being suppressed (account-tag + extended-join
    + skipAutoWhoOnJoin), do315 will never fire — so we must NOT add to
    _pending_channels here. The do376 2-second fallback (line 828) is then
    responsible for firing the startup notification.
    """
    if not ircutils.strEqual(irc.nick, msg.nick):
        return
    if self._will_skip_auto_who(irc):
        return
    channel = msg.args[0]
    self._pending_channels.add(channel)
```

In `LLM.__init__` (after `self.db = LLMDatabase(...)`, around `plugin.py:317`), call:

```python
_patch_irc_dojoin(self)
```

**Step 5: Run the new tests**

```bash
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestPatchedDoJoin -v
.venv/bin/pytest plugins/llm/tests/test_plugin.py::TestPluginDoJoinPendingChannels -v
```

Expected: PASS.

**Step 6: Run full test suite — watch for unrelated regressions**

```bash
.venv/bin/pytest plugins/llm/tests/ -q
```

Expected: PASS. The patch is global, so once any `LLM` is constructed in the test session, all other `Irc` instances see the patched behavior. If a test elsewhere asserted on legacy `WHO`/`MODE +b` queueing, update it.

Note: the patch closes over `plugin`. Subsequent test fixture sessions that construct a new `LLM` will replace the closure — fine. But test ordering matters if a test asserts on the patched behavior using a stale closure from a previous fixture. If you see flaky failures, check whether `plugin_env` is shared across test classes (it should not be — `@pytest.fixture` defaults to function scope, so each test gets a fresh `LLM`).

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_plugin.py
git commit -m "feat(llm): patch doJoin to drop MODE +b and conditionally drop auto-WHO; preserve startup notification path"
```

---

## Task 13: End-to-end manual verification

Bot is deployed via Docker from ghcr.io. After all tests are green and committed:

**Step 1: Wait for CI and Docker build**

After pushing to main, both the CI workflow AND the Docker build workflow must complete before restarting the service. They run separately.

```bash
gh run list --branch main --limit 5
```

Wait for both to show `completed/success`.

**Step 2: Restart and tail logs**

```bash
ssh -i ~/.ssh/id_rsa vibebot@rdrake.org
systemctl --user restart vibebot
journalctl --user -u vibebot -f
```

**Step 3: Verify in-channel**

In an AfterNet channel where the bot is present:

1. From an identified user, run `@ask hi`. Expected: succeeds, identity logged as account.
2. From an unidentified user, run `@ask hi`. Expected: error message saying "You must be identified to use this command." (no "NickServ").
3. Check `@usage` — current month's usage should reflect the account, not the bare nick.

**Step 4: Confirm the JOIN speedup**

Note: the upstream `Join to <chan> on <network> synced in N.NN seconds` log line at `irclib.py:2471` is emitted by `do315`. **When WHO is skipped, that log line will not appear at all** — that's the expected steady-state on AfterNet (which ACKs both caps).

Instead, time the JOIN end-to-end:

```bash
journalctl --user -u vibebot --since "5 minutes ago" | grep -E "JOIN|MODE|WHO|startup"
```

You should see:
- `JOIN #channel` (incoming) and the bot's own `JOIN` going out
- `MODE #channel` (incoming 329 reply for channel modes)
- **NO** `WHO #channel` outbound and **NO** `MODE #channel +b` outbound (verify with `tcpdump` or Limnoria's debug log)
- The startup-notification line should appear within ~2 seconds of MOTD-end (driven by do376's delayed callback) when all configured channels have joined

To confirm `_pending_channels` empties correctly, watch for the bot's startup-notification PRIVMSG to its operator channel/PM after a restart.

**Step 5: Roll back plan (if needed)**

If account-tag breaks gating or auto-WHO drop causes issues:

- For auto-WHO: set `supybot.plugins.LLM.skipAutoWhoOnJoin = False` via `@config` and reload the plugin. WHO is restored; MODE +b stays dropped (which is fine).
- For deeper issues: revert the merge commit on `main`, push, wait for the Docker rebuild, restart.

---

## Out of scope (per overview)

- Owner/admin/trusted gating (separate ircdb path, untouched).
- Adding ircdb hostmask matching to the resolver (rejected by review — silent privilege escalation risk).
- WHOIS-on-demand for idle users (account-tag obsoletes the need).
- Hostname parsing for account names (not portable).
- JOIN-handling code paths beyond the doJoin patch.

## Notes for the executor

- DRY: the resolver lives once. Don't inline `msg.server_tags['account']` reads at any other call site — go through `_account_from_msg`.
- YAGNI: don't add a `force_refresh` arg, don't add caching, don't extend the resolver to three layers later. If a future need appears, add it in a separate PR.
- TDD: every task in this plan starts with a failing test. Don't skip Step 2.
- Frequent commits: each task ends with a commit. Don't bundle.
- Use `superpowers:verification-before-completion` before claiming the plan is done — no "should work", run the tests.
