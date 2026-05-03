# Cross-File DRY Extractions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the highest-mass copy-paste duplications surfaced by the code review: grounded-completion twins in `service.py`, ask/code/draw response-dispatch in `plugin.py`, the seven-copy usage SELECT in `persistence.py`, and the channel-target normalization in `service.py`.

**Architecture:** Each extraction introduces a small private helper, replaces every existing copy with a call to the helper, and locks the new behavior down with at least one test that would have caught the divergence. No public API changes. Plan B can be merged independently of Plan A and Plan C; tasks within Plan B are ordered so each commit is independently shippable.

**Tech Stack:** Python 3.12+, Limnoria callbacks, LiteLLM, pytest. Lint with `make lint`, types with `make typecheck`, tests with `make test`.

**Pre-flight:** Each task ends with `make lint && make typecheck && make test` green before commit. Coverage stays at or above the 93% floor.

**Cross-plan dependency note:** Task 3 (`_dispatch_assistant_reply`) shares its callback layer with Plan C Task 1 (`ToolCallbackResult`). Land Plan C Task 1 first if both plans are running concurrently; otherwise, Plan B Task 3 may need to be re-touched after Plan C Task 1.

---

### Task 1: Extract `_channel_target` helper in `service.py`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` (add helper near top of class; replace 8 sites)
- Test: `plugins/llm/tests/test_service.py`

**Inventory (verified):** 8 sites — `service.py:1923, 1992, 2335, 2518, 2621, 2822, 3812, 3867`. Two forms exist in the wild:

- Without truthiness guard: `channel if channel.startswith(("#", "&")) else None` — at lines 1923, 1992, 2822
- With truthiness guard: `channel if channel and channel.startswith(("#", "&")) else None` — at lines 2335, 2518, 2621, 3812, 3867

**Behavior change to flag:** The unguarded form raises `AttributeError` on a `None` `channel` argument; the helper returns `None`. After this task, those three sites will silently coerce `None` instead of raising. Audit the callers — if any of them currently rely on the implicit `AttributeError` to surface a bug, fix the caller separately.

**Step 1: Locate the existing service-test fixture and use it correctly**

Read the top of `plugins/llm/tests/test_service.py` and `plugins/llm/tests/conftest.py` to find the established fixture (likely a pytest fixture, e.g. `service` or `make_service`). Use it as a fixture parameter — do **not** call it as a global function:

```python
def test_channel_target_passes_through_channel_names(service):
    assert service._channel_target("#general") == "#general"
    assert service._channel_target("&local") == "&local"

def test_channel_target_returns_none_for_nicks_and_falsy(service):
    assert service._channel_target("alice") is None
    assert service._channel_target("") is None
    assert service._channel_target(None) is None
```

If the existing fixture name differs, mirror what the surrounding tests use.

**Step 2: Run, confirm fail**

```bash
uv run pytest plugins/llm/tests/test_service.py -k channel_target -v
```
Expected: FAIL — helper does not exist.

**Step 3: Implement helper**

In the `LLMService` class:

```python
@staticmethod
def _channel_target(channel: str | None) -> str | None:
    """Return ``channel`` if it is an IRC channel name, else ``None``.

    Use for registry-value lookups that accept a per-channel scope: a nick
    or empty value collapses to the global scope (``None``).
    """
    if not channel:
        return None
    return channel if channel.startswith(("#", "&")) else None
```

**Step 4: Run new tests, confirm pass**

```bash
uv run pytest plugins/llm/tests/test_service.py -k channel_target -v
```

**Step 5: Replace the 8 call sites**

For each site, swap the inline form:

```python
# Before (either form)
target = channel if channel.startswith(("#", "&")) else None
target = channel if channel and channel.startswith(("#", "&")) else None

# After
target = self._channel_target(channel)
```

**Step 6: Verify no remaining inline forms**

```bash
grep -n 'channel.startswith(("#"' plugins/llm/src/llm/service.py
```
Expected: 0 hits.

**Step 7: Run full service tests**

```bash
uv run pytest plugins/llm/tests/test_service.py -v
```

**Step 8: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "refactor(service): extract _channel_target helper; replace 8 inline sites"
```

---

### Task 2: Extract `_grounded_completion` helper from `search_completion`/`url_completion`

**Files:**
- Modify: `plugins/llm/src/llm/service.py` — `search_completion` (~1911), `url_completion` (~1975)
- Test: `plugins/llm/tests/test_service.py`

**Step 1: Add a focused test before refactoring**

The original draft suggested a key-substring filter that is too loose. Use an explicit kwarg comparison instead:

```python
def test_search_and_url_completion_use_same_provider_kwargs_base(service, monkeypatch):
    captured: list[dict] = []

    def fake_call(*, model, messages, api_key, timeout, optional_kwargs):
        captured.append(optional_kwargs)
        return _fake_response_ok()

    monkeypatch.setattr(service, "_completion_with_tool_fallback", fake_call)
    # Force a non-xAI path; pick a model that triggers _resolve_grounding_kwargs.
    monkeypatch.setattr(service, "_is_xai_model", lambda _m: False)

    service.search_completion("ping", channel="#c")
    service.url_completion("https://example.com", channel="#c")

    # Both calls must produce kwargs with the same set of keys.
    # The values may differ for the grounding tool itself, but the base
    # provider kwargs (any key not specific to the grounding kind) must match.
    assert set(captured[0].keys()) == set(captured[1].keys())
```

Use the existing `_fake_response_ok` test helper if it exists, or build a minimal stand-in.

**Step 2: Run, confirm fail or skip**

```bash
uv run pytest plugins/llm/tests/test_service.py -k grounded -v
```

**Step 3: Implement `_grounded_completion`**

Add to `LLMService` (near `search_completion`):

```python
def _grounded_completion(
    self,
    user_content: str,
    *,
    kind: str,                 # "search" or "url"
    channel: str,
    log_label: str,            # "search_completion" or "url_completion"
    error_message: str,        # "Search failed." or "URL fetch failed."
) -> "ToolResult":
    from .assistant import ToolResult

    try:
        target = self._channel_target(channel)
        model = (
            self.plugin.registryValue("searchModel", target)
            or self.plugin.registryValue("assistantModel", target)
        )
        api_key = (
            self.plugin.registryValue("searchApiKey", target)
            or self.plugin.registryValue("assistantApiKey", target)
        )
        timeout = self.plugin.registryValue("timeout")

        if self._is_xai_model(model):
            return self._xai_responses_call(
                user_content, model=model, api_key=api_key,
                timeout=timeout, kind=kind,
            )

        messages: list[dict[str, object]] = [{"role": "user", "content": user_content}]
        optional_kwargs = self._get_provider_kwargs(model)
        optional_kwargs.update(self._resolve_grounding_kwargs(model, kind))

        self.log.info("%s start model=%s content_len=%d", log_label, model, len(user_content))
        response = self._completion_with_tool_fallback(
            model=model, messages=messages, api_key=api_key,
            timeout=timeout, optional_kwargs=optional_kwargs,
        )
        content = response.choices[0].message.content
        grounding_used = self._check_grounding_used(response)
        prompt_tokens, completion_tokens, cost = self._extract_usage(response, model)
        self.log.info(
            "%s ok model=%s grounding_used=%s content_len=%d "
            "prompt_tokens=%d completion_tokens=%d",
            log_label, model, grounding_used, len(content or ""),
            prompt_tokens, completion_tokens,
        )
        return ToolResult(
            content=content, grounding_used=grounding_used,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost,
        )
    except Exception as e:
        self.log.exception("%s failed: %s", log_label, self._sanitize(str(e)))
        return ToolResult(content=json.dumps({"error": error_message}))
```

**Step 4: Replace `search_completion`**

```python
def search_completion(self, query: str, *, channel: str) -> "ToolResult":
    return self._grounded_completion(
        query, kind="search", channel=channel,
        log_label="search_completion", error_message="Search failed.",
    )
```

**Step 5: Replace `url_completion` (preserve URL validation)**

```python
def url_completion(self, url: str, *, channel: str) -> "ToolResult":
    from .assistant import ToolResult
    if not validate_external_url(url):
        return ToolResult(
            content='{"error": "URL is not allowed (invalid scheme or private address)."}'
        )
    return self._grounded_completion(
        f"Summarize the content at this URL: {url}",
        kind="url", channel=channel,
        log_label="url_completion", error_message="URL fetch failed.",
    )
```

**Step 6: Run service tests**

```bash
uv run pytest plugins/llm/tests/test_service.py -v
```

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/service.py plugins/llm/tests/test_service.py
git commit -m "refactor(service): extract _grounded_completion to merge search/url paths"
```

---

### Task 3: Extract `_dispatch_assistant_reply` helper for ask/code/draw

**Prerequisite:** Plan C Task 1 (`ToolCallbackResult`) should land first. If it has not, the suppression branch below still uses the existing `result.last_successful_tool` shape — no rework needed unless that field's source changes.

**Files:**
- Modify: `plugins/llm/src/llm/plugin.py` — `ask` (~2524-2568), `code` (~2652-2670), `draw` (~2741-2759)
- Test: `plugins/llm/tests/test_commands.py`

**Critical behaviors that the helper MUST preserve:**

1. The reminder-mutation suppression in `ask` runs **before** the empty-response error branch — the original ordering treats a successful reminder mutation with empty follow-up text as success (silent), not as an error.
2. After dispatching an action, the original code rebinds the local `response` variable to `f"* {irc.nick} {action_text}"` so the action gets stored in conversation context (the caller's `_store_context_and_log_usage`). The helper must return the rebound string so the caller can pass it through.
3. The bridge-debug footer (`bridge_debug and bridge_trace`) is appended to `result.content` in `ask` *before* dispatch. Keep that pre-processing in the caller; the helper receives the final `response` text, not the raw `result.content`.

**Step 1: Read each command body in full**

Reference the actual code at `plugin.py:2524-2568` (ask), `~2652-2670` (code), `~2741-2759` (draw). Note specifically: the ask suppression (`_REMINDER_MUTATION_TOOLS`) is structurally an `if/elif/else` with the empty-error branch as `elif` and the dispatch as `else`.

**Step 2: Add tests covering the three preserved behaviors**

```python
@pytest.mark.parametrize("command", ["ask", "code", "draw"])
def test_grounding_icon_prefixed_consistently(command, run_command):
    out = run_command(command, "hello", grounding_used=True, content="world")
    assert any(r.startswith(GROUNDING_ICON) for r in out.replies)


def test_ask_suppresses_empty_followup_after_reminder_mutation(run_command):
    out = run_command(
        "ask", "set a reminder", last_successful_tool="set_reminder",
        final_text_after_tools="", content="",
    )
    # No "empty response" error; emoji ack already happened from the tool path.
    assert not any("empty response" in r.lower() for r in out.errors)


def test_action_response_stored_with_action_prefix(run_command, captured_context):
    out = run_command("ask", "/me waves", content="*waves at you*")
    # The stored context uses the "* botnick action_text" form, not the raw text.
    stored = captured_context.last_assistant_message()
    assert stored.startswith("* ")
```

If the helpers `run_command`/`captured_context` don't exist in this exact form, mirror what `test_commands.py` already uses.

**Step 3: Run, confirm at least one fails**

```bash
uv run pytest plugins/llm/tests/test_commands.py -k "grounding_icon or suppresses_empty or action_response_stored" -v
```

**Step 4: Implement helper with the correct ordering and rebinding**

Add to the `Plugin` class (annotate `response` as the post-footer text, returned for the caller to log):

```python
def _dispatch_assistant_reply(
    self,
    irc, msg, result, *,
    nick: str,
    channel: str,
    response: str,
    suppress_reminder_mutations: bool = False,
) -> str:
    """Send the reply for an assistant result.

    Returns the (possibly rebound) ``response`` string so the caller can
    persist it via ``_store_context_and_log_usage``. The rebinding occurs
    when the assistant emits an action: context stores ``"* botnick action"``
    rather than the raw text so follow-ups understand the bot emoted.

    Reminder-mutation suppression (``ask`` only) is checked BEFORE the empty-
    response error branch; that is the existing behavior and must be preserved.
    """
    if suppress_reminder_mutations and (
        result.last_successful_tool in _REMINDER_MUTATION_TOOLS
        and not result.final_text_after_tools.strip()
    ):
        self.log.info(
            "suppressing empty post-reminder-mutation reply tool=%s %s/%s",
            result.last_successful_tool, channel, nick,
        )
        return response

    if not response or not response.strip():
        irc.error(_("The model returned an empty response. Please try again."))
        return response

    action_text = self._extract_action(irc, response)
    if action_text:
        if result.grounding_used:
            action_text = f"{GROUNDING_ICON} {action_text}"
        self.log.info("sending action to %s/%s", channel, nick)
        target = channel if ircutils.isChannel(channel) else nick
        irc.queueMsg(ircmsgs.action(target, action_text))
        return f"* {irc.nick} {action_text}"

    display_response = (
        f"{GROUNDING_ICON} {response}" if result.grounding_used else response
    )
    self.log.info("replying to %s/%s", channel, nick)
    self._send_long_reply(irc, msg, display_response, prefixNick=False)
    return response
```

**Step 5: Replace each command body**

```python
# ask
response = result.content
if bridge_debug and bridge_trace:
    footer = self._format_bridge_debug_footer(bridge_trace)
    if footer:
        response = f"{response}\n{footer}" if response else footer
response = self._dispatch_assistant_reply(
    irc, msg, result,
    nick=nick, channel=channel,
    response=response,
    suppress_reminder_mutations=True,
)
self._store_context_and_log_usage(nick, channel, "ask", text, response, result, irc, msg)

# code: same shape, suppress_reminder_mutations=False (the default)
# draw: same shape, suppress_reminder_mutations=False
```

The empty-response error path now lives in the helper; the `irc.error("error: ... empty response ...")` *and* a normal `return` from the command happen via `irc.error`'s side effect plus the helper's return. **Verify** that `irc.error` does not raise — if it does, restructure so the command stops cleanly. Otherwise the caller will reach `_store_context_and_log_usage` after an error, which currently doesn't happen.

**Action item before merging this task:** confirm that the original `ask` command bails out (does not log usage) when the empty-response branch fires. Looking at the current code: the `elif not response or not response.strip(): irc.error(...); return` does return early. The helper change must preserve that early-return. Two options:

(a) Have the helper raise a sentinel exception for the empty-response branch and let the caller catch it.
(b) Return a tuple `(response, should_log)` from the helper.

Pick (b) — it is explicit and avoids exception-as-control-flow:

```python
def _dispatch_assistant_reply(...) -> tuple[str, bool]:
    """Returns (response_for_context, should_log_and_store)."""
    ...
    if not response or not response.strip():
        irc.error(_("The model returned an empty response. Please try again."))
        return response, False
    ...
    return response, True  # action or normal reply path
```

Each command becomes:

```python
response, should_log = self._dispatch_assistant_reply(...)
if should_log:
    self._store_context_and_log_usage(nick, channel, "ask", text, response, result, irc, msg)
```

**Step 6: Run full command tests**

```bash
uv run pytest plugins/llm/tests/test_commands.py plugins/llm/tests/test_plugin.py -v
```

**Step 7: Commit**

```bash
git add plugins/llm/src/llm/plugin.py plugins/llm/tests/test_commands.py
git commit -m "refactor(plugin): extract _dispatch_assistant_reply for ask/code/draw"
```

---

### Task 4: Skip `_layer_instruction` extraction

**Decision (revised after review):** Do not extract `_layer_instruction`.

The current `ask` command passes `None` for `effective_prompt` when there is no user instruction, allowing the service layer to apply its own default. Replacing with `self._layer_instruction(user_instruction, ask_prompt)` would always pass the registry-resolved `assistantSystemPrompt` explicitly, changing the layering semantics for the no-instruction path.

The duplication is two short lines in two functions. The risk of subtle behavior change exceeds the DRY payoff. Drop this task.

---

### Task 5: Extract `_query_usage_summary` helper

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:1374-1565`
- Test: `plugins/llm/tests/test_persistence.py`

**Verified column name:** the table column is `timestamp`, not `created_at`. Existing `WHERE` predicates also use `nick = ?` (case-sensitive), not `lower(nick) = lower(?)`. **Preserve those exact predicates.**

**Verified shape of `get_usage_summary_for_nick`:** it accepts `(nick, since=None, channel=None)` and applies `nick = ?`, optionally `channel = ?`, optionally `timestamp >= ?`. The helper must support this 3-condition shape.

**Step 1: Add the helper using the actual column name**

```python
_USAGE_SELECT = (
    "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
    "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) FROM usage"
)

def _query_usage_summary(
    self, conditions: list[str], params: list[object]
) -> UsageSummary:
    """Run the usage aggregation SELECT with optional AND-joined conditions.

    Conditions are conjunctive only. COUNT(*) + COALESCE guarantees a row,
    so no None-row guard is needed.
    """
    where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
    row = self._connect().execute(
        f"{self._USAGE_SELECT}{where}", tuple(params),
    ).fetchone()
    return UsageSummary(
        total_requests=row[0],
        total_prompt_tokens=row[1],
        total_completion_tokens=row[2],
        total_cost=row[3],
    )
```

**Step 2: Replace the three function bodies**

Match the existing predicates exactly — `nick = ?`, not `lower(nick) = lower(?)`:

```python
def get_usage_summary(self, since: float | None = None) -> UsageSummary:
    conds: list[str] = []
    params: list[object] = []
    if since is not None:
        conds.append("timestamp >= ?")
        params.append(since)
    return self._query_usage_summary(conds, params)


def get_usage_summary_for_channel(
    self, channel: str, since: float | None = None
) -> UsageSummary:
    conds: list[str] = ["channel = ?"]
    params: list[object] = [channel]
    if since is not None:
        conds.append("timestamp >= ?")
        params.append(since)
    return self._query_usage_summary(conds, params)


def get_usage_summary_for_nick(
    self, nick: str, since: float | None = None, channel: str | None = None
) -> UsageSummary:
    conds: list[str] = ["nick = ?"]
    params: list[object] = [nick]
    if channel is not None:
        conds.append("channel = ?")
        params.append(channel)
    if since is not None:
        conds.append("timestamp >= ?")
        params.append(since)
    return self._query_usage_summary(conds, params)
```

**Step 3: Run persistence tests**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```
Expected: PASS — same SQL, same outputs.

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): extract _query_usage_summary helper"
```

---

### Task 6: Extract `_owner_where(account, nick)` for scheduled-task queries

**Files:**
- Modify: `plugins/llm/src/llm/persistence.py:832-878`

**Verification step:** Before adopting, read the existing SQL in `load_scheduled_llm_tasks_for` and `count_scheduled_llm_tasks_for`. Confirm whether the existing predicates use `lower(account) = lower(?)` (case-insensitive) or `account = ?` (exact). The helper below assumes case-insensitive — if the existing SQL is exact, change `lower(account) = lower(?)` to `account = ?` to preserve behavior.

**Step 1: Add helper**

```python
@staticmethod
def _owner_where(*, account: str | None, nick: str) -> tuple[str, list[object]]:
    """Return SQL fragment + params matching scheduled-task ownership.

    If ``account`` is set, match by account; otherwise match by creator_nick
    among rows with NULL account. Adjust the case-sensitivity of the
    comparison to match what the existing queries use.
    """
    if account is not None:
        return "lower(account) = lower(?)", [account]
    return "account IS NULL AND lower(creator_nick) = lower(?)", [nick]
```

**Step 2: Replace the branches in both functions**

```python
where, params = self._owner_where(account=account, nick=nick)
rows = conn.execute(
    f"... WHERE {where} AND ... fire_at > ?",
    (*params, cutoff),
).fetchall()
```

**Step 3: Run tests**

```bash
uv run pytest plugins/llm/tests/test_persistence.py -v
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/persistence.py
git commit -m "refactor(persistence): extract _owner_where for scheduled-task queries"
```

---

### Task 7: Extract `_get_channel_state` helper in `service.py`

**Files:**
- Modify: `plugins/llm/src/llm/service.py:630-638` and `:739-747`

**Step 1: Add helper**

```python
@staticmethod
def _get_channel_state(irc, channel: str):
    """Return ChannelState or None if irc has no state for channel."""
    state = getattr(irc, "state", None)
    if not state:
        return None
    return getattr(state, "channels", {}).get(channel)
```

**Step 2: Replace both opening blocks**

```python
ch_state = self._get_channel_state(irc, channel)
if ch_state is None:
    return None
```

**Step 3: Run tests**

```bash
uv run pytest plugins/llm/tests/test_service.py -v
```

**Step 4: Commit**

```bash
git add plugins/llm/src/llm/service.py
git commit -m "refactor(service): extract _get_channel_state helper"
```

---

### Task 8: Extract rate-limit registration helper in `config.py`

**Files:**
- Modify: `plugins/llm/src/llm/config.py:569-742`

**Verified per-tier defaults (DO NOT CHANGE these values):**

| Command | Tier | Count default | Window default |
|---|---|---|---|
| ask | registered | 15 | 60 |
| ask | trusted | 15 | 60 |
| ask | unreg | 15 | 60 |
| code | registered | 10 | 60 |
| code | trusted | 0 | 60 |
| code | unreg | 2 | 60 |
| draw | registered | 2 | **300** |
| draw | trusted | 5 | 60 |
| draw | unreg | 0 | 60 |

**Note:** `drawRateLimitWindow` (registered) is **300**, not 60. The helper must require the window be passed explicitly per tier — no default of 60 — to prevent silent regression.

**Step 1: Add helper that takes per-tier counts and windows**

```python
def _register_rate_limit_block(
    command: str,
    *,
    counts: tuple[int, int, int],   # (registered, trusted, unreg)
    windows: tuple[int, int, int],  # (registered, trusted, unreg)
) -> None:
    tiers = (
        ("",        "registered tier"),
        ("Trusted", "trusted tier"),
        ("Unreg",   "unregistered tier"),
    )
    for (tier, label), count, window in zip(tiers, counts, windows, strict=True):
        suffix = tier or "" if tier else ""  # blank for registered
        conf.registerGlobalValue(
            LLM, f"{command}{tier}RateLimitCount",
            registry.NonNegativeInteger(
                count,
                _(
                    f"Max {command} requests per {label} within "
                    f"{command}{tier}RateLimitWindow seconds. "
                    "Set to 0 to disable rate limiting for this tier."
                ),
            ),
        )
        conf.registerGlobalValue(
            LLM, f"{command}{tier}RateLimitWindow",
            registry.PositiveInteger(
                window,
                _(f"Time window in seconds for counting {command} requests ({label})."),
            ),
        )
```

**Step 2: Replace 18 `registerGlobalValue` calls with 3 helper invocations**

```python
_register_rate_limit_block(
    "ask",
    counts=(15, 15, 15),
    windows=(60, 60, 60),
)
_register_rate_limit_block(
    "code",
    counts=(10, 0, 2),
    windows=(60, 60, 60),
)
_register_rate_limit_block(
    "draw",
    counts=(2, 5, 0),
    windows=(300, 60, 60),
)
```

**Step 3: Boot-test the registry and check exact values**

```bash
uv run pytest plugins/llm/tests/test_config.py -v
```

If `test_config.py` doesn't already cover the rate-limit defaults, add one parametrized test that pins each (key → expected default) pair so a future helper change can't drift values:

```python
@pytest.mark.parametrize("key,expected", [
    ("askRateLimitCount", 15),  ("askRateLimitWindow", 60),
    ("askTrustedRateLimitCount", 15), ("askTrustedRateLimitWindow", 60),
    ("askUnregRateLimitCount", 15), ("askUnregRateLimitWindow", 60),
    ("codeRateLimitCount", 10), ("codeRateLimitWindow", 60),
    ("codeTrustedRateLimitCount", 0), ("codeTrustedRateLimitWindow", 60),
    ("codeUnregRateLimitCount", 2), ("codeUnregRateLimitWindow", 60),
    ("drawRateLimitCount", 2), ("drawRateLimitWindow", 300),
    ("drawTrustedRateLimitCount", 5), ("drawTrustedRateLimitWindow", 60),
    ("drawUnregRateLimitCount", 0), ("drawUnregRateLimitWindow", 60),
])
def test_rate_limit_defaults(plugin, key, expected):
    assert plugin.registryValue(key) == expected
```

**Step 4: Run tests**

```bash
make test
```

**Step 5: Commit**

```bash
git add plugins/llm/src/llm/config.py plugins/llm/tests/test_config.py
git commit -m "refactor(config): extract _register_rate_limit_block helper; pin defaults"
```

---

### Task 9: Reminder finder/filter consolidation — DEFERRED

**Decision (after review):** Defer this task.

The current target-side finder uses `nick == target OR account == target` (raw equality on either field). The proposed `Identity.matches` predicate has different semantics — when `Identity` is constructed with `account=None`, a target lookup by account name would not match. This is a privilege/isolation surface for `@reminders admin` operations and a regression risk.

A correct unification requires a new `Identity.matches_either_field(target_string)` helper that explicitly searches both `nick` and `account` fields. That is a separate design task. **Skip** this consolidation in Plan B; revisit after Plans A and C are merged.

---

### Task 10: Final preflight

```bash
make preflight
```
Expected: PASS, coverage ≥ 93%.

If a task introduced an uncovered branch, add a focused test before merging.
