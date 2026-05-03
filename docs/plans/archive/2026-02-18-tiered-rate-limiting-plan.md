# Tiered Per-Command Rate Limiting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-user-tier rate limits to all commands, where limits scale by Limnoria capability (owner/admin exempt, trusted gets relaxed limits, registered gets standard, unregistered gets strictest) and vary per command sorted by cost.

**Architecture:** Extend the existing `_is_rate_limited` / `_check_rate_limit` pipeline in `plugin.py` to resolve a user's tier from Limnoria capabilities (`ircdb.checkCapability`), then look up tier-specific config values. Add new config keys for each command+tier combination. Apply rate limiting to all commands (ask, code, draw, animate) instead of just draw/animate.

**Tech Stack:** Python 3.14, Limnoria (supybot.ircdb for capability checks), pytest + pytest-mock for testing.

---

## Design Options

Three options were considered. **Option A is recommended.** All options share the same tier model and config structure — they differ only in where the rate limit check lives.

### Tier Model (all options)

| Tier | How Detected | Behavior |
|------|-------------|----------|
| **owner/admin** | `ircdb.checkCapability(prefix, 'owner')` or `'admin'` | **Exempt** — no rate limits |
| **trusted** | `ircdb.checkCapability(prefix, 'trusted')` | Relaxed limits (higher count or wider window) |
| **registered** | Has NickServ account (`account is not None`) | Standard limits (current defaults) |
| **unregistered** | No NickServ account | Strictest limits (or blocked for expensive commands) |

Note: Limnoria's `owner` implies `admin` implies `trusted`, but we check explicitly from most to least privileged.

### Config Structure (all options)

Per-command config keys follow the pattern `{command}RateLimit{Tier}{Count|Window}`. Commands sorted by cost:

| Command | Cost | Config Keys (count / window per tier) |
|---------|------|--------------------------------------|
| `ask` | $ | `askRateLimitCount/Window` (registered), `askTrustedRateLimitCount/Window` (trusted), `askUnregRateLimitCount/Window` (unregistered) |
| `code` | $$ | Same pattern with `code` prefix |
| `draw` | $$$ | Already has `drawRateLimitCount/Window`; add trusted + unreg variants |
| `animate` | $$$$ | Already has `animateRateLimitCount/Window`; add trusted + unreg variants |

Setting any count to `0` disables rate limiting for that command+tier (useful for not limiting `ask` for trusted users). Owner/admin are always exempt — no config needed.

**Default values (suggested):**

| Command | Unreg Count/Window | Registered Count/Window | Trusted Count/Window |
|---------|-------------------|------------------------|---------------------|
| `ask` | 5 / 60s | 15 / 60s | 0 (disabled) |
| `code` | 3 / 60s | 10 / 60s | 0 (disabled) |
| `draw` | 0 / 60s (disabled\*) | 3 / 60s (existing) | 10 / 60s |
| `animate` | 0 / 600s (disabled\*) | 2 / 600s (existing) | 5 / 600s |

\* `draw` and `animate` already require NickServ via `require_account=True`, so unregistered users are blocked before rate limiting. `count=0` means "no rate limit", not "blocked". If you later relax the NickServ gate and still want to block unregistered users, add an explicit auth check rather than relying on rate-limit config.

---

### Option A: Extend existing `_check_rate_limit` pipeline (Recommended)

**Where:** Keep rate limiting inside `plugin.py`'s `_run_preflight` → `_check_rate_limit` → `_is_rate_limited` chain.

**How:** Add a `_resolve_tier(irc, msg)` method that returns `"owner"`, `"admin"`, `"trusted"`, `"registered"`, or `"unregistered"`. Modify `_is_rate_limited` to accept a tier parameter and look up `{command}{Tier}RateLimitCount` / `{command}{Tier}RateLimitWindow`. Remove `apply_rate_limit` from `_run_preflight` and always run tier-aware checks for ask/code/draw/animate (with owner/admin exemption).

**Pros:**
- Minimal structural change — extends what already works
- All rate limit logic stays in one place
- Easy to test (existing test patterns work)
- No new dependencies or architectural concepts

**Cons:**
- More config keys in `config.py` (20 new keys total: 12 for ask+code tiers, 8 for draw/animate trusted+unreg tiers)
- Rate limit logic coupled to plugin class

**Files touched:**
- `plugins/llm/src/llm/config.py` — Add ~20 new registry values
- `plugins/llm/src/llm/plugin.py` — Add `_resolve_tier()`, modify `_is_rate_limited()` and `_check_rate_limit()`, remove `apply_rate_limit` and run tier-aware checks for all commands
- `plugins/llm/tests/test_commands.py` — Add tier-based tests, update existing tests
- `plugins/llm/tests/conftest.py` — Add new config keys to `make_registry_side_effect` defaults

---

### Option B: `pre_command_callbacks` middleware

**Where:** Register a class-level callback via Limnoria's `callbacks.Commands.pre_command_callbacks`.

**How:** Create a `_pre_command_callback` method, register it in `__init__`, unregister in `die()`. The callback resolves the tier, checks rate limits, and returns `True` to block.

**Pros:**
- Fires before `wrap()` argument parsing — blocks earliest possible
- Clean separation from command methods
- Follows Limnoria's official extension pattern (same as ProgVal's RateLimit plugin)

**Cons:**
- Class-level list (shared across all plugin instances) — must be careful with registration/cleanup
- Callback receives `(plugin, command, irc, msg, *args)` — command is a list of strings, not the simple command name
- Less control over error messaging (no `irc.error` in the callback context without careful handling)
- Harder to test — requires mocking the callback registration
- Must handle the `wrap` tokenizer running after the callback, so rate limit hits on invalid syntax waste a rate limit slot

---

### Option C: Decorator-based rate limiting

**Where:** A `@rate_limited("draw")` decorator applied to each command method.

**How:** Create a decorator that wraps command methods, checks rate limits before calling the real method.

**Pros:**
- Declarative — visible on each command method
- Could be reused across plugins

**Cons:**
- Interacts poorly with Limnoria's `wrap()` decorator (which replaces the method at class level)
- Decorator must be applied before `wrap()` in the right order
- Harder to access plugin state (registryValue, ircdb) from decorator scope
- Over-engineering for 4 commands

---

## Implementation Plan (Option A)

### Task 1: Add `_resolve_tier` method

**Files:**
- Create: *(none)*
- Modify: `plugins/llm/src/llm/plugin.py`
- Test: `plugins/llm/tests/test_commands.py`

**Step 1: Write the failing test for `_resolve_tier`**

Add to `plugins/llm/tests/test_commands.py`:

```python
class TestResolveTier:
    """Tests for _resolve_tier user classification."""

    def test_owner_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with owner capability WHEN _resolve_tier THEN returns 'owner'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", side_effect=lambda prefix, cap: cap == "owner")
        assert plugin._resolve_tier(mock_irc, mock_msg) == "owner"

    def test_admin_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with admin (not owner) WHEN _resolve_tier THEN returns 'admin'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap in ("admin", "trusted"),
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == "admin"

    def test_trusted_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN user with trusted (not admin) WHEN _resolve_tier THEN returns 'trusted'."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted",
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == "trusted"

    def test_registered_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN identified user without trusted WHEN _resolve_tier THEN returns 'registered'."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "some_account"
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        assert plugin._resolve_tier(mock_irc, mock_msg) == "registered"

    def test_unregistered_tier(self, plugin_env, mocker: MockerFixture):
        """GIVEN unidentified user WHEN _resolve_tier THEN returns 'unregistered'."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        assert plugin._resolve_tier(mock_irc, mock_msg) == "unregistered"
```

**Step 2: Run test to verify it fails**

Run: `make test`
Expected: FAIL with `AttributeError: 'LLM' object has no attribute '_resolve_tier'`

**Step 3: Implement `_resolve_tier` in `plugin.py`**

Add after `_check_flagged` method (around line 1257):

```python
def _resolve_tier(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
    """Classify a user into a rate-limit tier based on Limnoria capabilities.

    Checks capabilities from most to least privileged.

    Args:
        irc: IRC connection (for account lookup).
        msg: IRC message (uses msg.prefix for capability check).

    Returns:
        One of: "owner", "admin", "trusted", "registered", "unregistered".
    """
    prefix = msg.prefix
    if ircdb.checkCapability(prefix, "owner"):
        return "owner"
    if ircdb.checkCapability(prefix, "admin"):
        return "admin"
    if ircdb.checkCapability(prefix, "trusted"):
        return "trusted"
    nick = ircutils.nickFromHostmask(prefix)
    try:
        account = irc.state.nickToAccount(nick)
    except (KeyError, AttributeError):
        account = None
    return "registered" if account else "unregistered"
```

**Step 4: Run test to verify it passes**

Run: `make test`
Expected: PASS

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat: add _resolve_tier method for user classification"
```

---

### Task 2: Add tier-aware config keys

**Files:**
- Modify: `plugins/llm/src/llm/config.py`
- Modify: `plugins/llm/tests/conftest.py`

**Step 1: Add config keys to `config.py`**

Replace the existing "Rate Limiting" section (lines 443-497) with expanded config:

```python
# ============================================================================
# Rate Limiting (per-command, per-tier)
# ============================================================================

conf.registerGlobalValue(
    LLM,
    "enforceRateLimits",
    registry.Boolean(
        False,
        _("""Enable per-user rate limiting for commands.
        When False, limits are tracked and logged but not enforced (monitor mode).
        Set to True to actively block requests that exceed the limit."""),
    ),
)

# --- ask (cheapest) ---
conf.registerGlobalValue(
    LLM,
    "askRateLimitCount",
    registry.NonNegativeInteger(
        15,
        _("""Max ask requests per registered user within askRateLimitWindow seconds.
        Set to 0 to disable rate limiting for this tier."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askTrustedRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max ask requests per trusted user within askTrustedRateLimitWindow seconds.
        Set to 0 to disable (trusted users unlimited for ask)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askUnregRateLimitCount",
    registry.NonNegativeInteger(
        5,
        _("""Max ask requests per unregistered user within askUnregRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "askUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting ask requests (unregistered tier)."""),
    ),
)

# --- code ---
conf.registerGlobalValue(
    LLM,
    "codeRateLimitCount",
    registry.NonNegativeInteger(
        10,
        _("""Max code requests per registered user within codeRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeTrustedRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max code requests per trusted user within codeTrustedRateLimitWindow seconds.
        Set to 0 to disable (trusted users unlimited for code)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeUnregRateLimitCount",
    registry.NonNegativeInteger(
        3,
        _("""Max code requests per unregistered user within codeUnregRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "codeUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting code requests (unregistered tier)."""),
    ),
)

# --- draw (expensive) ---
conf.registerGlobalValue(
    LLM,
    "drawRateLimitCount",
    registry.NonNegativeInteger(
        3,
        _("""Max draw requests per registered user within drawRateLimitWindow seconds."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting draw requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawTrustedRateLimitCount",
    registry.NonNegativeInteger(
        10,
        _("""Max draw requests per trusted user within drawTrustedRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawTrustedRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting draw requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawUnregRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max draw requests per unregistered user within drawUnregRateLimitWindow seconds.
        Set to 0 to disable. Note: draw already requires NickServ, so unreg users
        are blocked before this check."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "drawUnregRateLimitWindow",
    registry.PositiveInteger(
        60,
        _("""Time window in seconds for counting draw requests (unregistered tier)."""),
    ),
)

# --- animate (most expensive) ---
conf.registerGlobalValue(
    LLM,
    "animateRateLimitCount",
    registry.NonNegativeInteger(
        2,
        _("""Max animate requests per registered user within animateRateLimitWindow seconds."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateRateLimitWindow",
    registry.PositiveInteger(
        600,
        _("""Time window in seconds for counting animate requests (registered tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateTrustedRateLimitCount",
    registry.NonNegativeInteger(
        5,
        _("""Max animate requests per trusted user within animateTrustedRateLimitWindow seconds.
        Set to 0 to disable."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateTrustedRateLimitWindow",
    registry.PositiveInteger(
        600,
        _("""Time window in seconds for counting animate requests (trusted tier)."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateUnregRateLimitCount",
    registry.NonNegativeInteger(
        0,
        _("""Max animate requests per unregistered user within animateUnregRateLimitWindow seconds.
        Set to 0 to disable. Note: animate already requires NickServ, so unreg users
        are blocked before this check."""),
    ),
)

conf.registerGlobalValue(
    LLM,
    "animateUnregRateLimitWindow",
    registry.PositiveInteger(
        600,
        _("""Time window in seconds for counting animate requests (unregistered tier)."""),
    ),
)
```

**Step 2: Update conftest.py defaults**

In `plugins/llm/tests/conftest.py`, update the `defaults` dict inside `make_registry_side_effect` to include all new keys:

```python
# Rate limiting
"enforceRateLimits": False,
# ask
"askRateLimitCount": 15,
"askRateLimitWindow": 60,
"askTrustedRateLimitCount": 0,
"askTrustedRateLimitWindow": 60,
"askUnregRateLimitCount": 5,
"askUnregRateLimitWindow": 60,
# code
"codeRateLimitCount": 10,
"codeRateLimitWindow": 60,
"codeTrustedRateLimitCount": 0,
"codeTrustedRateLimitWindow": 60,
"codeUnregRateLimitCount": 3,
"codeUnregRateLimitWindow": 60,
# draw
"drawRateLimitCount": 3,
"drawRateLimitWindow": 60,
"drawTrustedRateLimitCount": 10,
"drawTrustedRateLimitWindow": 60,
"drawUnregRateLimitCount": 0,
"drawUnregRateLimitWindow": 60,
# animate
"animateRateLimitCount": 2,
"animateRateLimitWindow": 600,
"animateTrustedRateLimitCount": 5,
"animateTrustedRateLimitWindow": 600,
"animateUnregRateLimitCount": 0,
"animateUnregRateLimitWindow": 600,
```

**Step 3: Run tests to verify nothing breaks**

Run: `make test`
Expected: All existing tests PASS (config changes are additive).

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/conftest.py
git commit -m "feat: add per-command per-tier rate limit config keys"
```

---

### Task 3: Modify rate limit pipeline to be tier-aware

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` (lines 1014-1216)
- Test: `plugins/llm/tests/test_commands.py`

This is the core change. The `_run_preflight` → `_check_rate_limit` → `_is_rate_limited` chain needs to:
1. Resolve the user's tier.
2. Short-circuit for owner/admin (exempt).
3. Map tier to the correct config key prefix.
4. Respect count=0 as "disabled for this tier".

**Step 1: Write failing tests for tier-aware rate limiting**

Add to `plugins/llm/tests/test_commands.py` inside `TestRateLimitIntegration`:

```python
def test_owner_exempt_from_rate_limits(self, plugin_env, mocker: MockerFixture):
    """GIVEN owner user over limit WHEN draw called THEN not blocked."""
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "owner_account"
    plugin.llm_service.image_generation.return_value = ImageResult(
        content="http://img.example/gen.png",
        prompt_tokens=5,
        completion_tokens=0,
        cost=0.02,
        model="dall-e-3",
    )

    plugin.registryValue = mocker.MagicMock(
        side_effect=make_registry_side_effect({
            "enforceRateLimits": True,
            "drawRateLimitCount": 1,
            "drawRateLimitWindow": 60,
        })
    )
    # Fill bucket way past limit
    now = time.time()
    for _ in range(10):
        plugin._record_rate_limit_hit("draw", "owner_account", now - 1)

    # Mock: user is owner
    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: True,  # owner has all caps
    )
    plugin.draw(mock_irc, mock_msg, ["test prompt"])

    mock_irc.error.assert_not_called()
    plugin.llm_service.image_generation.assert_called_once()

def test_trusted_gets_relaxed_limits(self, plugin_env, mocker: MockerFixture):
    """GIVEN trusted user within trusted limit but over registered limit WHEN draw called THEN allowed."""
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "trusted_account"
    plugin.llm_service.image_generation.return_value = ImageResult(
        content="http://img.example/gen.png",
        prompt_tokens=5,
        completion_tokens=0,
        cost=0.02,
        model="dall-e-3",
    )

    plugin.registryValue = mocker.MagicMock(
        side_effect=make_registry_side_effect({
            "enforceRateLimits": True,
            "drawRateLimitCount": 2,       # registered: 2
            "drawRateLimitWindow": 60,
            "drawTrustedRateLimitCount": 10,  # trusted: 10
            "drawTrustedRateLimitWindow": 60,
        })
    )
    # 5 hits — over registered limit (2), under trusted limit (10)
    now = time.time()
    for _ in range(5):
        plugin._record_rate_limit_hit("draw", "trusted_account", now - 1)

    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: cap == "trusted",
    )
    plugin.draw(mock_irc, mock_msg, ["test prompt"])

    mock_irc.error.assert_not_called()
    plugin.llm_service.image_generation.assert_called_once()

def test_ask_rate_limited_for_unregistered(self, plugin_env, mocker: MockerFixture):
    """GIVEN unregistered user over unreg limit WHEN ask called THEN blocked."""
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = None

    plugin.registryValue = mocker.MagicMock(
        side_effect=make_registry_side_effect({
            "enforceRateLimits": True,
            "askUnregRateLimitCount": 2,
            "askUnregRateLimitWindow": 60,
        })
    )
    now = time.time()
    # Use hostmask as bucket key for unregistered users
    nick = "testnick"
    for _ in range(3):
        plugin._record_rate_limit_hit("ask", nick, now - 1)

    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
    plugin.ask(mock_irc, mock_msg, ["hello"])

    mock_irc.error.assert_called_once()
    assert "Rate limit" in mock_irc.error.call_args[0][0]

def test_zero_count_disables_rate_limit(self, plugin_env, mocker: MockerFixture):
    """GIVEN trusted tier with count=0 WHEN ask called many times THEN never blocked."""
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "trusted_account"
    plugin.llm_service.detect_images.return_value = []
    plugin.llm_service.completion.return_value = CompletionResult(
        content="hello",
        prompt_tokens=5,
        completion_tokens=10,
        cost=0.001,
        model="gpt-4",
    )

    plugin.registryValue = mocker.MagicMock(
        side_effect=make_registry_side_effect({
            "enforceRateLimits": True,
            "askTrustedRateLimitCount": 0,  # 0 = disabled
            "askTrustedRateLimitWindow": 60,
        })
    )
    now = time.time()
    for _ in range(100):
        plugin._record_rate_limit_hit("ask", "trusted_account", now - 1)

    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: cap == "trusted",
    )
    plugin.ask(mock_irc, mock_msg, ["hello"])

    mock_irc.error.assert_not_called()
    mock_irc.reply.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `make test`
Expected: Multiple FAILs — owner not exempt, trusted not using relaxed limits, ask not rate limited yet, etc.

**Step 3: Implement tier-aware rate limiting**

Modify `plugins/llm/src/llm/plugin.py`:

**3a. Refine `_resolve_tier` to accept `irc` and `msg`:**

```python
def _resolve_tier(self, irc: callbacks.Irc, msg: IrcMsg) -> str:
    """Classify a user into a rate-limit tier based on Limnoria capabilities.

    Checks capabilities from most to least privileged.

    Args:
        irc: IRC connection (for account lookup).
        msg: IRC message (uses msg.prefix for capability check).

    Returns:
        One of: "owner", "admin", "trusted", "registered", "unregistered".
    """
    prefix = msg.prefix
    if ircdb.checkCapability(prefix, "owner"):
        return "owner"
    if ircdb.checkCapability(prefix, "admin"):
        return "admin"
    if ircdb.checkCapability(prefix, "trusted"):
        return "trusted"
    nick = ircutils.nickFromHostmask(prefix)
    try:
        account = irc.state.nickToAccount(nick)
    except (KeyError, AttributeError):
        account = None
    return "registered" if account else "unregistered"
```

**3b. Modify `_is_rate_limited` to accept tier-specific config keys:**

```python
def _is_rate_limited(self, command: str, identity: str, now: float, *, tier: str) -> bool:
    """Check if a user exceeds the per-command rate limit for their tier.

    Args:
        command: Command name (ask, code, draw, animate).
        identity: NickServ account or nick (bucket key).
        now: Current time (seconds since epoch).
        tier: User tier for config lookup.

    Returns:
        True if the user has exceeded the rate limit.
    """
    max_count, window = self._get_tier_limits(command, tier)

    # count=0 means rate limiting is disabled for this tier
    if max_count == 0:
        return False

    cutoff = now - window
    key = f"{command}:{identity}"
    bucket = self._rate_buckets.get(key)
    if bucket is None:
        return False

    # Evict expired entries
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()

    if not bucket:
        self._rate_buckets.pop(key, None)
        return False

    return len(bucket) >= max_count
```

**3c. Add `_get_tier_limits` helper:**

```python
_TIER_CONFIG_PREFIX = {
    "trusted": "Trusted",
    "unregistered": "Unreg",
    "registered": "",  # base config (no prefix)
}

def _get_tier_limits(self, command: str, tier: str) -> tuple[int, int]:
    """Look up rate limit count and window for a command+tier.

    Args:
        command: Command name (ask, code, draw, animate).
        tier: User tier (trusted, registered, unregistered).

    Returns:
        (max_count, window_seconds). max_count=0 means disabled.
    """
    prefix = self._TIER_CONFIG_PREFIX.get(tier, "")
    count_key = f"{command}{prefix}RateLimitCount"
    window_key = f"{command}{prefix}RateLimitWindow"
    return self.registryValue(count_key), self.registryValue(window_key)
```

**3d. Modify `_check_rate_limit` to accept tier:**

```python
def _check_rate_limit(
    self,
    irc: callbacks.Irc,
    command: str,
    identity: str,
    nick: str,
    channel: str,
    text: str,
    *,
    tier: str,
) -> bool:
    """Check rate limit for a user's tier and send error if exceeded.

    Owner/admin tiers should be filtered out before calling this method.

    Args:
        irc: IRC connection.
        command: Command name.
        identity: NickServ account or nick (bucket key).
        nick: Display nick for logging.
        channel: Channel name.
        text: Prompt text for logging.
        tier: User tier (trusted, registered, unregistered).

    Returns:
        True if the request should be blocked.
    """
    now = time.time()
    over_limit = self._is_rate_limited(command, identity, now, tier=tier)

    # Always record the hit
    self._record_rate_limit_hit(command, identity, now)

    if over_limit:
        enforce = self.registryValue("enforceRateLimits")
        max_count, window = self._get_tier_limits(command, tier)
        key = f"{command}:{identity}"
        count = len(self._rate_buckets.get(key, ()))
        if enforce:
            self.log.info(
                "rate_limited command=%s identity=%s tier=%s count=%d limit=%d window=%ss",
                command, identity, tier, count, max_count, window,
            )
            irc.error(
                _("Rate limit exceeded for %s. Please wait before trying again.") % command
            )
            self.db.log_usage(
                nick, channel, command, "", 0, 0, 0.0,
                prompt=text, status="rate_limited",
            )
            return True
        self.log.info(
            "rate_limit_shadow command=%s identity=%s tier=%s count=%d limit=%d window=%ss",
            command, identity, tier, count, max_count, window,
        )
    return False
```

**3e. Modify `_run_preflight` to resolve tier and apply rate limits to all commands:**

Key changes to `_run_preflight`:
- Remove the `apply_rate_limit` parameter.
- Always resolve tier.
- Short-circuit for owner/admin.
- Pass tier to `_check_rate_limit`.
- Use `nick` (for unregistered) or `account` (for registered+) as the bucket identity.

```python
def _run_preflight(
    self,
    irc: callbacks.Irc,
    msg: IrcMsg,
    text: str,
    command: str,
    *,
    require_account: bool,
) -> PreflightResult:
    """Shared preflight check for all commands.

    Runs: account resolution → flagged check → tier-based rate limit.
    """
    channel = self._get_channel(msg)

    # --- account resolution ---
    if require_account:
        account = self._require_account(irc, msg)
        if account is None:
            nick = ircutils.nickFromHostmask(msg.prefix)
            self.db.log_usage(nick, channel, command, "", 0, 0, 0.0, prompt=text, status="auth_failure")
            return PreflightResult(blocked=True, nick=nick, channel=channel, account=None)
        nick = self._resolve_nick_to_identity(irc, ircutils.nickFromHostmask(msg.prefix))
    else:
        raw_nick = ircutils.nickFromHostmask(msg.prefix)
        try:
            account = irc.state.nickToAccount(raw_nick)
        except (KeyError, AttributeError):
            account = None
        nick = self._get_identity(irc, msg)

    # --- flagged check ---
    if self._check_flagged(irc, msg, account):
        self.db.log_usage(nick, channel, command, "", 0, 0, 0.0, prompt=text, status="flagged_blocked")
        return PreflightResult(blocked=True, nick=nick, channel=channel, account=account)

    # --- tier-based rate limit ---
    tier = self._resolve_tier(irc, msg)
    if tier not in ("owner", "admin"):
        identity = account or nick  # account for registered+, nick for unreg
        if self._check_rate_limit(irc, command, identity, nick, channel, text, tier=tier):
            return PreflightResult(blocked=True, nick=nick, channel=channel, account=account)

    return PreflightResult(blocked=False, nick=nick, channel=channel, account=account)
```

**3f. Update all command call sites to remove `apply_rate_limit`:**

In `ask` method (~line 1399):
```python
pf = self._run_preflight(irc, msg, text, "ask", require_account=False)
```

In `code` method (~line 1505):
```python
pf = self._run_preflight(irc, msg, text, "code", require_account=False)
```

In `draw` method (~line 1589):
```python
pf = self._run_preflight(irc, msg, text, "draw", require_account=True)
```

In `animate` method (~line 1650):
```python
pf = self._run_preflight(irc, msg, text, "animate", require_account=True)
```

**Step 4: Run tests to verify they pass**

Run: `make test`
Expected: All new tier tests PASS, all existing tests PASS.

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "feat: tier-aware rate limiting for all commands"
```

---

### Task 4: Update existing tests and fixtures

**Files:**
- Modify: `plugins/llm/tests/test_commands.py` (existing `TestRateLimitIntegration` tests)
- Modify: `plugins/llm/tests/test_plugin.py` (`TestRateLimiter` + `TestRunPreflight` signatures/fixtures)
- Modify: `plugins/llm/tests/conftest.py` (new tiered rate-limit defaults)
- Search all tests for `ircdb.checkCapability` mocks that currently use `return_value=True`

**Step 1: Update tests for signature changes**

Run:
```bash
rg -n "apply_rate_limit|_run_preflight\\(|_is_rate_limited\\(|_check_rate_limit\\(" plugins/llm/tests/
```

Update:
- `_run_preflight(..., apply_rate_limit=...)` calls: remove the parameter.
- `_is_rate_limited(...)` calls: pass `tier=...`.
- `_check_rate_limit(...)` calls: pass `tier=...`.

**Step 2: Normalize capability mocks in command tests**

Any test that currently uses `mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)` can accidentally classify users as owner/admin under tier resolution. Use a side effect that allows only plugin command capabilities:

```python
# Allow the wrap() capability check but report user as "registered" tier
mocker.patch(
    "llm.plugin.ircdb.checkCapability",
    side_effect=lambda prefix, cap: cap.startswith("llm."),  # only plugin caps, not owner/admin/trusted
)
```

Apply this consistently across command-invocation tests (not just `TestRateLimitIntegration`).

**Step 3: Update `TestRateLimiter` and `TestRunPreflight` in `test_plugin.py`**

- Expand fixture defaults to include ask/code + trusted/unreg keys (not only draw/animate keys).
- Update `_is_rate_limited` assertions to pass `tier="registered"` (or specific tier under test).
- Update `_check_rate_limit` calls to pass `tier="registered"` (or specific tier under test).
- Remove `apply_rate_limit` from `_run_preflight` calls.

**Step 4: Update ask/code integration test names and expectations**

These tests previously verified ask/code were not rate-limited at all. Rename and keep assertions focused on "within configured limit succeeds":

- `test_ask_not_rate_limited` → `test_ask_within_limit_succeeds`
- `test_code_not_rate_limited` → `test_code_within_limit_succeeds`

**Step 5: Run full test suite**

Run: `make preflight`
Expected: All tests PASS, lint clean, types clean.

**Step 6: Commit**

```bash
git add plugins/llm/tests/
git commit -m "test: update rate limit tests for tier-aware system"
```

---

### Task 5: Final validation and cleanup

**Files:**
- Verify: All files modified above

**Step 1: Run full preflight**

Run: `make preflight`
Expected: format + lint + typecheck + test all PASS.

**Step 2: Verify test coverage**

Run: `make test`
Check that coverage is still ≥80%.

**Step 3: Review config documentation**

Spot-check that `config.py` docstrings are clear about:
- What each tier means
- That count=0 disables rate limiting for that tier
- That owner/admin are always exempt (no config needed)

**Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: cleanup and verify tiered rate limiting"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `plugins/llm/src/llm/config.py` | Add 20 new config keys (ask/code registered+trusted+unreg, draw/animate trusted+unreg) |
| `plugins/llm/src/llm/plugin.py` | Add `_resolve_tier()`, `_get_tier_limits()`, modify `_is_rate_limited()`, `_check_rate_limit()`, `_run_preflight()`. Remove `apply_rate_limit` parameter. |
| `plugins/llm/tests/test_commands.py` | Add `TestResolveTier` class (5 tests), add tier-aware integration tests, and normalize capability mocks |
| `plugins/llm/tests/conftest.py` | Add 20 new config key defaults to `make_registry_side_effect` |
| `plugins/llm/tests/test_plugin.py` | Update `TestRateLimiter`/`TestRunPreflight` fixtures and method calls for new signatures/tier args |
| Other test files | Update `apply_rate_limit` references and any `ircdb.checkCapability` mocks that accidentally imply owner/admin |

**Total new config keys:** 20
Breakdown: ask/code add 12 keys (registered + trusted + unreg counts/windows), draw/animate add 8 keys (trusted + unreg counts/windows).

**Backwards compatibility:** The `drawRateLimitCount` / `drawRateLimitWindow` and `animateRateLimitCount` / `animateRateLimitWindow` config keys keep their existing names and defaults, so existing bot configs don't break. The registered tier (most users) sees the same limits as before.
