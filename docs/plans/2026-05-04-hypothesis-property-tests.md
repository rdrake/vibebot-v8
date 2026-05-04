# Hypothesis Property-Test Adoption Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce Hypothesis to the LLM plugin test suite and lock down the four highest-leverage invariants identified in the audit: `ConversationContext` state, `LLMDatabase` pending-task lifecycle, conversation JSON round-trip, and `Identity.matches`. Each task replaces or augments existing example-based tests with one property test that subsumes them.

**Audit reference:** [`docs/reviews/2026-05-04-hypothesis-audit.md`](../reviews/2026-05-04-hypothesis-audit.md). Candidates 5-11 in that document are tracked as **Future Work** at the bottom of this plan.

**Plan reviews integrated:** This plan reflects feedback from [`docs/reviews/2026-05-04-hypothesis-property-tests-plan-review.md`](../reviews/2026-05-04-hypothesis-property-tests-plan-review.md) (Codex) and a follow-up code-reviewer pass.

**Architecture:** Hypothesis is added to the workspace `[dependency-groups].dev` in the **root** `pyproject.toml` (the plugin's own `pyproject.toml` has only runtime dependencies; `make test` runs through the root `uv` environment per `Makefile`). Property tests live alongside existing tests in `plugins/llm/tests/`, named `test_*_properties.py` so they are easy to find. Each task is independently shippable: land Task 0 once and the rest can land in any order.

**Tech Stack:** Python 3.12+, Hypothesis 6.x, pytest, pytest-mock. Lint with `make lint`, types with `make typecheck`, tests with `make test`. Coverage floor is **93%** -- property tests must not regress it.

**Pre-flight:** Each task ends with `make lint && make typecheck && make test` green before commit. Property tests use `@settings(deadline=None)` only when SQLite I/O makes per-example timing unpredictable; otherwise leave the default deadline in place so slow regressions surface.

**Cross-plan dependency note:** No coupling to the existing `2026-05-03-*` plans. Land in any order.

---

### Task 0: Add Hypothesis as a workspace dev dependency

**Files:**
- Modify: `pyproject.toml` (repo root, `[dependency-groups].dev` array at lines ~85-94)
- Modify: `uv.lock` (regenerated)

**Step 1: Add the dependency**

Add `"hypothesis>=6.100"` to the `[dependency-groups].dev` array in the **repo-root** `pyproject.toml` (existing entries: `mkdocs-material`, `prek`, `pytest`, `pytest-cov`, `pytest-mock`, `ruff`, `ty`). Match the existing single-line, alphabetized formatting. Do **not** edit `plugins/llm/pyproject.toml` -- it has only runtime `[project].dependencies` and `make test` runs from the workspace root.

**Step 2: Lock and verify**

```bash
uv lock
uv sync --all-groups
uv run python -c "import hypothesis; print(hypothesis.__version__)"
```

**Step 3: Verify**

```bash
make lint
make typecheck
make test
```

**Commit:** `chore: add hypothesis as workspace dev dependency`

---

### Task 1: Property test for `ConversationContext` state machine

**Files:**
- Add: `plugins/llm/tests/test_context_properties.py`
- Modify: `plugins/llm/tests/test_context.py` (remove subsumed cases)

**Audit reference:** Candidate #1 in `docs/reviews/2026-05-04-hypothesis-audit.md`.

**Step 1: Write the state machine**

Create `plugins/llm/tests/test_context_properties.py` with a `ConversationContextMachine(RuleBasedStateMachine)`:

- **Fixed config:** `ContextConfig(max_messages=4, timeout_minutes=30, enabled=True, channel_max_messages=3)`. Both caps are kept small so trim invariants are exercised. **`enabled=True` is fixed; do not parameterize it** -- when `enabled=False`, every `get_messages` returns `[]`, collapsing the case-insensitivity and isolation invariants to vacuous truths.
- **Pools:** `nicks = sampled_from(["alice","Alice","BOB","charlie","dave"])`, `channels = sampled_from(["#a","#b","#priv1"])`, `roles = sampled_from(["user","assistant"])`, `contents = text(max_size=200)`.
- **Two shadow models** (personal and channel context have different keying):
  - Personal: `dict[(nick.lower(), channel.lower())] -> list[(role, content)]`. Personal context lowercases both nick and channel (`context.py:117`, `context.py:127`).
  - Channel: `dict[channel.lower()] -> list[(nick, role, content)]`. Channel context keys only by lowercased channel (`context.py:299`).
- **Rules:** `add_message(nick, channel, role, content)`, `add_channel_message(channel, nick, role, content)`, `clear(nick, channel)`, `clear_channel(channel)`, `clear_all()`, `migrate_user(old_nick, new_nick)`.
- **Invariants** (`@invariant()`):
  - **Personal trim:** for every `(n, c)` ever touched, `len(get_messages(n, c)) <= cfg.max_messages`.
  - **Channel trim:** for every channel ever touched, `len(get_channel_messages(c)) <= cfg.channel_max_messages`. (Use `cfg.channel_max_messages`, not `max_messages` -- the trim at `context.py:307` uses the channel cap.)
  - **Case-insensitive personal lookup:** for any `(n, c)` ever touched, `get_messages(n.upper(), c.upper()) == get_messages(n.lower(), c.lower())`.
  - **Isolation:** for any two distinct `(n.lower(), c.lower())` keys, the returned message lists do not share content (compare against the personal shadow model).
  - **Deep-copy of returned messages:** call `get_messages(n, c)`, mutate a returned dict's *value* (e.g. `result[0]["role"] = "MUTATED"`), re-fetch, and assert the role is unchanged. **Mutate dict values, not the outer list** -- appending to the returned list cannot affect internal state regardless of implementation, so the meaningful mutation is on the dicts inside.
- **Rule postcondition** (assert immediately inside the rule, not as a global invariant): after `clear(n, c)`, `get_messages(n, c) == []`. (A subsequent `add_message(n, c, ...)` is allowed to repopulate the conversation; the assertion only holds right after the clear.)

**Step 2: Verify the new test catches a real bug**

Temporarily mutate `add_message` in `context.py:226` to skip the trim and re-run; the personal-trim invariant must fail. Revert.

**Step 3: Remove subsumed example tests**

In `plugins/llm/tests/test_context.py`, delete or shrink to one canonical example each:

- `test_context_per_user_isolation` (line 34)
- `test_context_per_channel_isolation` (line 50)
- `test_context_case_insensitive` (line 66)
- `test_context_max_messages_limit` (line 77)

Keep `test_context_add_and_get_messages` (line 19) as the executable spec.

**Intentionally retained** (the state machine cannot replace these):
- `test_context_time_expiry` (line 92), `test_get_messages_max_age_drops_stale` (line 116) — exercise `_is_expired` / `_is_stale` paths that depend on backdated `last_activity`; the state machine runs in real wall-clock time and never reaches expiry.
- Any `test_context_thread_safe`-style test — `RuleBasedStateMachine` runs single-threaded by design, so the Lock at `context.py:208` still needs the existing concurrent test.

**Step 4: Verify**

```bash
make lint
make typecheck
make test
```

Coverage on `context.py` must not drop.

**Commit:** `test(llm): replace ConversationContext example tests with state machine`

---

### Task 2: Property test for pending-task lifecycle

**Files:**
- Add: `plugins/llm/tests/test_persistence_pending_task_properties.py`
- Modify: `plugins/llm/tests/test_persistence.py` (mark subsumed cases for follow-up; do not delete in this task)

**Audit reference:** Candidate #2 in `docs/reviews/2026-05-04-hypothesis-audit.md`.

**Step 1: Write the state machine**

Create `PendingTaskLifecycleMachine(RuleBasedStateMachine)` against a fresh `LLMDatabase`. **Pytest fixtures cannot be injected into a `RuleBasedStateMachine`**, so allocate the temp directory in `__init__` via `tempfile.mkdtemp()` and clean it in `teardown()` with `shutil.rmtree(..., ignore_errors=True)`. Also call `self.db.close()` in `teardown()` to release the WAL connection before the directory is removed.

**Time:** the implementation calls `time.time()` inside `get_next_due_time()` and `delete_expired_pending_tasks()` (`persistence.py:1166-1186`, `persistence.py:1138-1164`); inside `claim_due_pending_tasks` the caller passes `now`. Patch `time.time` in `llm.persistence` to a settable container so each rule sets it deterministically:

```python
from unittest.mock import patch
def __init__(self):
    super().__init__()
    self._now = 1_000_000.0
    self._patcher = patch("llm.persistence.time.time", side_effect=lambda: self._now)
    self._patcher.start()
def teardown(self):
    self._patcher.stop()
    self.db.close()
    shutil.rmtree(self._dir, ignore_errors=True)
```

Rules set `self._now = self._t0 + offset` (`offset = integers(0, 3600)`) before invoking the DB call.

**Rules:**
- `save(task_type, expires_at_offset, next_attempt_at_offset)`
- `claim(now_offset, limit, lease_seconds)` — fixes `delivery_state_filter=None` and `max_delivery_attempts=None`. **The state machine intentionally omits these dimensions**; filtered-claim coverage remains in `TestDeliveryStatePersistence` (`tests/test_persistence.py:969`) and is the reason Step 3 keeps those tests.
- `release(task_id, next_attempt_offset, increment_attempt)`
- `update_for_delivery(task_id, delivery_state, payload)`
- `update_delivery_attempt(task_id, delivery_state, error, count, next_attempt_offset)`
- `delete_expired(now_offset)`

`task_id` is drawn from a Hypothesis `Bundle` populated by `save`; rules guard against the empty-bundle case.

**Invariants:**
- **Claim mutual exclusion:** two consecutive `claim_due_pending_tasks(now, ...)` calls with the same `now` return disjoint ID sets.
- **Claim respects `next_attempt_at`:** any returned row had `next_attempt_at <= now AND claimed_until <= now` at claim time (`persistence.py:1001-1003`).
- **Claim sets the lease deterministically:** every claimed row has `claimed_until == now + lease_seconds` after the call (`persistence.py:1012-1018`).
- **`release(..., increment_attempt=False)`:** `attempt_count` field on the reloaded `PendingTaskRow` is unchanged from before release.
- **`release(..., increment_attempt=True)`:** `attempt_count` increased by exactly 1.
- **`delete_expired_pending_tasks(now)`:** returned IDs equal the IDs that disappear from `load_pending_tasks()`; only rows with `delivery_state='pending' AND expires_at <= now` are removed (`persistence.py:1138-1164`).
- **`load_pending_tasks(task_type=t)`:** is a subset of `load_pending_tasks()` filtered by `task_type == t`.

**`get_next_due_time` — separate `@given` test, not a state-machine invariant.** Because patching `time.time` inside a `RuleBasedStateMachine` is fragile (every rule must set `self._now` before any DB call that reads time, or the invariant will see a stale clock), move this property to a standalone test where `mocker` from `pytest-mock` is in scope:

```python
def test_get_next_due_time_matches_oracle(tmp_path, mocker):
    db = LLMDatabase(str(tmp_path / "p.db"))
    mocker.patch("llm.persistence.time.time", return_value=1_000_000.0)
    # ... insert rows via @given strategy, assert MIN(next_attempt_at) for
    # actionable rows == db.get_next_due_time(); else None.
```

Drop it from the state machine entirely.

**Settings:** use `@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])`. SQLite + `BEGIN IMMEDIATE` per rule is genuinely slow.

**Step 2: Verify the test catches a real bug**

Temporarily change `claim_due_pending_tasks` at `persistence.py:1003` to drop the `claimed_until <= ?` predicate; the mutual-exclusion invariant must fail. Revert.

**Step 3: Mark subsumed example tests**

In `plugins/llm/tests/test_persistence.py`, add a module-level comment near the top of `TestPendingTasks` (line 761):

```python
# NOTE: lifecycle invariants (claim mutual-exclusion, attempt_count delta,
# lease-deadline correctness) are now covered by
# test_persistence_pending_task_properties.py. The cases below are kept as
# executable specifications. TestDeliveryStatePersistence (line 969) covers
# delivery_state_filter / max_delivery_attempts paths that the state machine
# intentionally does not parameterize -- do not delete those without first
# extending the property test.
```

Do not delete cases in this task -- a follow-up audit pass will prune the redundant ones once the property test has stabilized in CI for a release cycle.

**Step 4: Verify**

```bash
make lint
make typecheck
make test
```

**Commit:** `test(llm): add property test for pending-task lifecycle`

---

### Task 3: Property test for conversation JSON round-trip

**Files:**
- Add: `plugins/llm/tests/test_persistence_conversation_properties.py`

**Audit reference:** Candidate #3 in `docs/reviews/2026-05-04-hypothesis-audit.md`.

**Step 1: Write the round-trip properties**

Create `plugins/llm/tests/test_persistence_conversation_properties.py` with **three** `@given` tests against a per-test `LLMDatabase(tmp_path)`:

```python
@given(
    nick=text(alphabet=characters(min_codepoint=0x21, max_codepoint=0x7E,
                                   blacklist_characters=" "), min_size=1, max_size=15),
    channel=sampled_from(["#a", "#b", "#priv1"]),
    messages=lists(
        fixed_dictionaries({
            "role": sampled_from(["user", "assistant", "system"]),
            # Lone surrogates are safe here: persistence.py:489 calls
            # json.dumps with default ensure_ascii=True, which escapes them
            # as \uXXXX. If that ever changes, switch to
            # text(alphabet=characters(blacklist_categories=["Cs"]), ...).
            "content": text(max_size=500),
        }),
        max_size=20,
    ),
    last_activity=floats(min_value=0, max_value=2_000_000_000,
                         allow_nan=False, allow_infinity=False),
)
def test_save_load_round_trip(tmp_path, nick, channel, messages, last_activity):
    db = LLMDatabase(str(tmp_path / "c.db"))
    db.save_conversation(nick, channel, messages, last_activity)
    rows = db.load_conversations()
    matching = [(n, c, m, t) for (n, c, m, t) in rows
                if (n, c) == (nick.lower(), channel.lower())]
    assert len(matching) == 1
    assert matching[0][2] == messages
    assert matching[0][3] == pytest.approx(last_activity)
    db.close()
```

Second property: `save → save → load` returns exactly one row at the canonicalized key (no duplicates).

Third property: `save("Alice", "#X", ...)` then `delete_conversation("ALICE", "#x")` then `load_conversations()` returns no row at the canonicalized key.

**Step 2: Verify the test catches a real bug**

Strip `.lower()` from one of the lookup paths (e.g. `delete_conversation` at `persistence.py:501-503`); the third property must fail. Revert.

**Step 3: Verify**

```bash
make lint
make typecheck
make test
```

**Commit:** `test(llm): add JSON round-trip properties for conversation persistence`

---

### Task 4: Property test for `Identity.matches`

**Files:**
- Add: `plugins/llm/tests/test_identity_properties.py`

**Audit reference:** Candidate #4 in `docs/reviews/2026-05-04-hypothesis-audit.md`.

**Note on casemapping:** `Identity.matches` uses `ircutils.toLower` (`plugin.py:120-121`), which implements RFC 1459 casemapping where `[]\\^` are the upper-case partners of `{}|~`. Python's `str.lower()` does **not** bridge these characters (`'['.lower() == '['`), so a strategy that constructs case-equivalent pairs via `s.upper()` / `s.lower()` cannot exercise the IRC-specific behavior. Build pairs with `ircutils.toLower` directly.

**Step 1: Write the equivalence properties**

Create `plugins/llm/tests/test_identity_properties.py`:

```python
from string import ascii_letters, digits
from supybot import ircutils
from hypothesis import given
from hypothesis.strategies import builds, none, one_of, text

from llm.plugin import Identity

# Alphabet includes the IRC special-case characters []{}\\^| so the
# RFC 1459 casemapping in ircutils.toLower is exercised, not just str.lower.
nicks = text(alphabet=ascii_letters + digits + "[]{}\\^_-|", min_size=1, max_size=15)
accounts = one_of(none(), nicks)


@given(raw=nicks, account=accounts)
def test_matches_is_reflexive(raw, account):
    ident = Identity(raw_nick=raw, account=account)
    assert ident.matches(ident)


@given(raw_a=nicks, raw_b=nicks, acct_a=accounts, acct_b=accounts)
def test_matches_is_symmetric(raw_a, raw_b, acct_a, acct_b):
    a = Identity(raw_nick=raw_a, account=acct_a)
    b = Identity(raw_nick=raw_b, account=acct_b)
    assert a.matches(b) == b.matches(a)


@given(pair=builds(lambda r: (r, ircutils.toLower(r)), nicks),
       account=accounts)
def test_matches_uses_irc_casemapping_on_raw_nick(pair, account):
    """[, ], \\, ^ should be case-equivalent to {, }, |, ~ via toLower."""
    raw_upper, raw_lower = pair
    a = Identity(raw_nick=raw_upper, account=None)
    b = Identity(raw_nick=raw_lower, account=None)
    assert a.matches(b)


@given(pair=builds(lambda r: (r, ircutils.toLower(r)), nicks))
def test_matches_uses_irc_casemapping_on_account(pair):
    acct_upper, acct_lower = pair
    a = Identity(raw_nick="x", account=acct_upper)
    b = Identity(raw_nick="y", account=acct_lower)
    # When both have an account, raw_nick is irrelevant.
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_account_overrides_raw_nick_in_matches(raw, acct):
    """When both have the same account, mismatched raw_nicks still match."""
    a = Identity(raw_nick=raw + "_distinct_a", account=acct)
    b = Identity(raw_nick=raw + "_distinct_b", account=acct)
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_key_equals_account_when_present(raw, acct):
    """Two identities with the same account share a storage key
    regardless of raw nick."""
    assert Identity(raw_nick=raw, account=acct).key == Identity(
        raw_nick=raw + "_other", account=acct
    ).key


@given(raw=nicks)
def test_key_falls_back_to_raw_nick_when_unidentified(raw):
    assert Identity(raw_nick=raw, account=None).key == raw
```

**Why these are load-bearing:**
- `test_matches_uses_irc_casemapping_on_raw_nick` would catch a regression that swaps `ircutils.toLower` for `str.lower` (which would produce `'[' != '{'`).
- `test_account_overrides_raw_nick_in_matches` and `test_key_equals_account_when_present` are cross-object properties; they cannot be satisfied by re-stating `account or raw_nick`, unlike the previous draft's tautological `key == (account if account else raw)`.

**Empty-string accounts:** `accounts = one_of(none(), nicks)` and `nicks` has `min_size=1`, so `account` is never the empty string. If empty-string accounts are ever produced by `Identity` callers, add `accounts = one_of(none(), just(""), nicks)` and document the expected behavior of `matches` in that case (currently `account=""` is falsy and behaves like `None` in the `if self.account and other.account` guard at `plugin.py:119`).

**Step 2: Verify**

```bash
make lint
make typecheck
make test
```

**Commit:** `test(llm): add Identity.matches equivalence properties`

---

## Future work

The remaining candidates from the audit are tracked here for prioritization in a future cycle. Full descriptions, suggested properties, and custom strategies are in [`docs/reviews/2026-05-04-hypothesis-audit.md`](../reviews/2026-05-04-hypothesis-audit.md).

| # | Subsystem | File | Priority | Payoff |
|---|---|---|---|---|
| 5 | `validate_external_url` | `service.py:361-398` | MEDIUM-HIGH | Replace ~12 enumerated SSRF tests; catch IPv6 / embedded-auth oddities. **Note:** existing tests use fixed IPv4 literals only; IPv6 (`::1`, `fc00::/7`) and embedded-auth (`http://user@10.0.0.1/`) paths are not covered until this lands. |
| 6 | `sanitize_output` | `service.py:496-535` | MEDIUM | Replace ~10 enumerated tests with idempotence + no-prefix invariants |
| 7 | `_strip_markdown_fences` | `service.py:3278-3300` | MEDIUM | Replace 11 cases with strip/round-trip property |
| 8 | Usage ranking & aggregation | `persistence.py:1378-1592` | MEDIUM | Monotone-rank property; catches tie-break and zero-cost short-circuit edges |
| 9 | `_compute_backoff` | `service.py:1421-1433` | LOW | Trivial; fold into a "pure helpers" property file |
| 10 | Reminder & scheduled-task DB CRUD | `persistence.py:545-880` | MEDIUM | Lock down case-insensitive owner matching |
| 11 | `_next_rrule_fire` | `plugin.py:3349-3367` | MEDIUM | DST / leap-second edges; needs runtime check before adoption |

Explicitly skipped (see audit "Deprioritized / skip"): `limnoria_bridge.py`, `assistant.py` tool dispatchers, `parse_reminder` post-LLM coercion, trivial transformations in `service.py:3618-3725`, HTML rendering tests.
