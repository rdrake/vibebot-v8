---
status: review
date: 2026-05-02
reviewed_plan: docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md
---

# Limnoria Tool Bridge Implementation Plan Review

## Findings

### High: Capability checks still do not mirror Limnoria's full command gate

The implementation plan pre-checks only the leaf command:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:515`
- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:770`

Limnoria's `_callCommand()` checks more than that. It checks the leaf command
first, then checks the plugin and every command prefix:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1590`
- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1596`

That means a channel anti-capability such as `-misc` or `-misc.ping` can still
be missed during enumeration. During dispatch, `_callCommand()` will catch it
and call `irc.errorNoCapability(...)`, but the bridge then returns
`{"status": "ok", "reply": "<capability error>"}` because no exception was
raised. `service.py` treats JSON dicts without `"error"` as successful tools:

- `plugins/llm/src/llm/service.py:2705`

Recommendation: add a helper that mirrors `_callCommand()` exactly: check
`command_path[-1]`, then `[canonical]`, `[canonical, ...]` prefixes. Use it in
both `enumerate_commands()` and `dispatch()` before calling `_callCommand()`,
and add tests for plugin-level and full-command anti-capability denial.

### High: `BufferingIrcProxy.error()` drops `Raise=True` semantics

The planned proxy override appends the error text and returns:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:282`

Base `ReplyIrcProxy.error()` raises `callbacks.Error` when `Raise=True`:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:675`

Commands commonly use `irc.error(..., Raise=True)` / rich error helpers to stop
execution. The planned override would let command code continue after an error,
which can produce false success or unintended mutation.

Recommendation: preserve the base control-flow contract. At minimum, add a
test that `proxy.error("nope", Raise=True)` raises `callbacks.Error`, and make
the override raise instead of silently returning.

### Medium: Tokenization errors are outside the dispatch error envelope

`callbacks.tokenize(...)` can raise `SyntaxError` for malformed nested command,
pipe, bracket, or quote syntax:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:420`
- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:431`

The plan calls `tokenize()` before entering the `try` block:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:775`

So malformed `args` can escape the bridge handler and bubble into the assistant
loop instead of returning `{"error": "..."}` as promised.

Recommendation: include tokenization inside the dispatch error handling, or
catch `SyntaxError` explicitly and return an error envelope. Add a regression
test with a malformed bracket/pipe argument string.

### Medium: Nested command paths are treated as single leaf names

The plan says `cb.listCommands()` returns leaf names and then dispatches with
`[command]`:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:310`
- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:779`

Limnoria can return nested commands as space-joined command paths:

- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1554`
- `.venv/lib/python3.14/site-packages/supybot/callbacks.py:1560`

For those commands, `getCommandMethod([leaf])`, `isCommandMethod(command)`, and
`_callCommand([command], ...)` will not match Limnoria's real command path.

Recommendation: either explicitly skip command names containing spaces in
Phase 1, or normalize them with `command_path = command.split()` and use that
path consistently for help lookup, capability checks, deny-list matching, and
dispatch.

### Medium: The registry default test does not verify registry registration

The proposed B1 test calls `plugin.registryValue` after replacing it with
`make_registry_side_effect()`:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:815`

That fixture returns values from a local dict, not from Limnoria's real registry:

- `plugins/llm/tests/conftest.py:276`
- `plugins/llm/tests/conftest.py:363`

If the fixture is updated, the test can pass without the registry values being
registered. If it is not updated, `bridgeEnabled` returns `""`, so `is False`
still fails after adding the real config.

Recommendation: follow the existing `TestConfigValues` style and assert against
`conf.supybot.plugins.LLM.bridgeEnabled()` and
`conf.supybot.plugins.LLM.bridgeAllowedPlugins()` after importing `llm.config`.
Update `make_registry_side_effect()` separately only for plugin tests that need
the new keys.

### Low: Some test instructions are placeholders, not executable steps

B2 and C2 include `raise NotImplementedError(...)` test bodies and tell the
implementer to wire them to the existing harness:

- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:942`
- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:952`
- `docs/plans/2026-05-02-limnoria-tool-bridge-implementation-plan.md:1258`

That is useful guidance, but it conflicts with the plan's `ready-to-execute`
status. The existing assistant completion tests already live in
`plugins/llm/tests/test_assistant.py`, and the real `ask` command tests live in
`plugins/llm/tests/test_commands.py`.

Recommendation: replace placeholders with concrete tests in those existing
files, or mark those tasks as requiring implementation judgment instead of
copy-paste execution.

### Low: Validation commands diverge from repository guidance

The plan uses raw `uv run pytest` throughout, while `AGENTS.md` says to prefer
`make` targets and to run `make preflight` before considering work complete.

Recommendation: keep narrow `uv run pytest ...` commands for task-local TDD if
needed, but make final validation include `make lint`, `make typecheck`, and
`make preflight` unless there is a stated reason to use a narrower set.

## Overall Assessment

The architecture is much closer to the repo's actual assistant/tooling path
than the first draft. The remaining blockers are in behavioral fidelity to
Limnoria: capability checks must match `_callCommand()`, and the buffering
proxy must preserve error control flow. Address those before implementation.
