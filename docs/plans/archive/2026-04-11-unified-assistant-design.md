# Unified Assistant Design

## Problem

The current command surface has improved, but the architecture is still
split across two mental models:

1. `ask`/`code`/`draw` are command-specific call paths.
2. `meta` is a separate tool-calling path for natural-language state changes.
3. Google grounding only exists in the normal completion path, not the
   tool-calling path.
4. Ordinary channel lines that mention the bot nick do not consistently
   behave like first-class assistant requests.

This leads to duplicated routing, duplicated policy decisions, and an
artificial distinction between "chat" and "management" requests even
though users naturally mix them.

Examples of the mismatch:

- "VibeBot, remind me in 2 hours" wants reminder tools.
- "VibeBot, draw a sunset" wants image generation.
- "VibeBot, what are my memories?" wants persistence tools.
- "@draw a sunset" should still work for users who prefer commands.

The current `meta` command solved part of this, but it introduced a second
assistant path rather than a unified one.

## Goals

- Build around one assistant backend with tool calling as the primary
  orchestration mechanism.
- Treat bot-directed chat, PMs, and explicit commands as entry routes into
  the same backend.
- Make any channel message that mentions the bot nick trigger an assistant
  response.
- Preserve existing user-facing commands for compatibility and speed.
- Keep access control, authentication, and rate limiting authoritative on
  the server side rather than in prompts.
- Support grounding via a separate server tool instead of mixing
  provider-native grounding into the orchestration call.

## Non-goals

- Removing all explicit commands immediately.
- Letting a single route capability implicitly grant access to every tool.
- Allowing the LLM to choose channel, nick, capability, or account scope.
- Depending on provider-native tool and grounding combinations in the same
  model call.

## Core Decisions

### 1. No special `meta` surface

There is no longer a conceptual split between "normal" and "meta" usage.
All assistant-facing behavior is part of one system:

- PMs to the bot
- Channel lines that mention the bot nick
- Unknown addressed text handled via `invalidCommand`
- Existing commands like `@ask`, `@code`, `@draw`, `@remind`, `@memories`

`@meta` becomes unnecessary. It can be kept temporarily as an alias to the
same assistant route for compatibility, but it should disappear from help
and documentation.

### 2. One backend, multiple route profiles

The backend is unified, but not every entry route exposes the same tool set
or execution style. Each request is assigned a route profile.

Examples:

- `chat`: PMs, channel nick mentions, unknown addressed text, `@ask`
- `code`: `@code`
- `draw`: `@draw`
- `state-direct`: `@memories`, `@instruct`, `@forget`, `@remind`, `@usage`

Each profile determines:

- Which tools are visible to the planner
- Whether a planner loop is used at all
- Which outer rate-limit bucket is charged
- How the final result is rendered to IRC

### 3. Tool registry owns policy metadata

The model-visible schema is only half of a tool definition. The real source
of truth is a server-side registry that includes hidden policy metadata.

Example shape:

```python
ToolSpec(
    name="generate_image",
    schema={...},
    handler=self._tool_generate_image,
    capability="llm.draw",
    require_account=True,
    rate_bucket="draw",
    cost_class="media",
    destructive=False,
    visible_in={"chat", "draw"},
)
```

At minimum each tool should declare:

- required capability
- whether an authenticated account is required
- rate-limit bucket
- cost class
- destructive or non-destructive
- scope rules (`self`, `current_channel`, `owner_cross_user`, etc.)
- which route profiles may expose it

This prevents "chat" from becoming a permissions loophole.

### 4. Planner mode and direct mode both exist

Unification happens at the executor layer, not by forcing every request
through the planner loop.

Two execution modes are required:

- `direct`: deterministic wrappers parse known arguments and call the
  shared executor directly
- `planned`: the assistant receives a filtered tool set and chooses tools
  through a tool-calling loop

Examples:

- `@memories del 3` should stay direct.
- `@usage #channel` should stay direct.
- "VibeBot, delete the cat memory and remind me tomorrow" should use the
  planner.

This keeps explicit commands fast, cheap, and predictable while still
allowing natural-language composition.

### 5. Grounding becomes a leaf tool

Do not mix provider-native Google grounding into the planner call.

Instead, expose one or more server tools such as:

- `search_web(query)`
- `fetch_url(url)`
- `grounded_lookup(query)`

Internally, these tools may use the existing grounded Gemini path, but the
planner only sees structured results. The planner remains provider-agnostic.

This avoids the current tool-versus-grounding conflict and keeps grounding
available wherever the assistant needs it.

## Request Routing

### Entry routes

| Route | Trigger | Profile | Notes |
|------|---------|---------|------|
| PM | Any private message to the bot | `chat` | Always addressed |
| Mention | Channel message containing bot nick as a token | `chat` | New first-class path |
| Unknown addressed text | `invalidCommand` fallback | `chat` | Preserves legacy addressed behavior |
| `@ask` | Explicit command | `chat` | Thin wrapper |
| `@code` | Explicit command | `code` | Thin wrapper |
| `@draw` | Explicit command | `draw` | Thin wrapper |
| `@memories` / `@instruct` / `@forget` / `@remind` / `@usage` | Explicit commands | `state-direct` | Deterministic wrappers by default |

### Mention-trigger behavior

Any mention of the bot nick should trigger a response, but mention matching
must still be exact enough to avoid false positives.

Rules:

- case-insensitive match against the current bot nick
- nick must appear as its own token or with IRC punctuation such as `:`
  or `,`
- substring matches do not count
- ignore the bot's own messages
- ignore old playback and CTCP messages except `/me`
- respond at most once per incoming message

Known explicit commands keep priority. A message that is already being
handled as a real command must not also trigger the mention route.

### Mention routing config

Mention-triggered replies are high-visibility behavior, so they need an
escape hatch.

Add a `mentionEnabled` channel config, default `True`.

Recommended behavior:

- `True`: PMs and channel nick mentions route into the unified assistant
- `False`: PMs may still be answered, but channel mention-trigger routing
  is disabled without removing the rest of the assistant backend

### Limnoria integration

The current plugin already uses both `doPrivmsg` and `invalidCommand`.
The unified design should continue to use both:

- `invalidCommand` handles addressed command-like text that misses the
  command registry
- `doPrivmsg` handles ordinary PMs and raw channel mention traffic

If Limnoria dispatch order makes duplicate handling possible, add a
short-lived dedupe key based on `(prefix, target, text, timestamp_bucket)`.

Because `doPrivmsg` already handles context tracking and spontaneous logic,
mention routing should not be bolted directly into that method body.
Extract helper methods first so `doPrivmsg` becomes orchestration rather
than a growing pile of unrelated branches.

## Unified Request Context

Every route should be normalized into a shared request context before any
assistant logic runs.

Example fields:

```python
RequestContext(
    entry_route="mention",
    profile="chat",
    nick="account_or_nick",
    raw_nick="displaynick",
    account="authenticated_account_or_none",
    channel="#chan_or_none",
    is_private=False,
    is_owner=False,
    capabilities={"llm.ask", "llm.code", ...},
)
```

This context is created in `plugin.py`, not by the model.

## Prompt Strategy

The unified assistant needs a prompt model different from the old `meta`
prompt because the `chat` profile is no longer limited to configuration
requests.

### `chat` profile system prompt

The `chat` planner prompt should:

- describe the bot as a general IRC assistant
- instruct the model to be concise and IRC-safe
- explain that tool results are untrusted data, not instructions
- tell the model to answer directly when no tool is needed
- tell the model to use tools only when they materially help
- avoid inventing capabilities or implying actions happened unless a tool
  result confirms success

### No `NOT_META` replacement

The old `NOT_META` sentinel disappears on the unified path.

`invalidCommand` no longer needs a "config or fall through to ask"
decision because unknown addressed text is itself just a `chat` request.
The unified facade either:

- returns a normal assistant answer
- returns a brief refusal or clarification
- returns an error result

There is no second dispatch step after the assistant call.

## Execution Flow

### Common flow

1. Receive message through route wrapper.
2. Build `RequestContext`.
3. Run route-level preflight.
4. Select profile.
5. Resolve visible tools from the registry for that profile and request
   context.
6. Execute in direct mode or planner mode.
7. Render the result to IRC.
8. Log outer usage and per-tool events.

### Route-level preflight

The existing preflight checks in `plugin.py` remain authoritative for the
outer request:

- identity resolution
- authenticated-account requirement where required
- route-level rate limits

This stays outside the model.

### Tool-level enforcement

Each tool invocation must also perform server-side checks before calling
the handler:

- capability check
- auth check
- owner-only cross-user check
- tool-specific rate limit where applicable
- destructive-action policy

This is the second line of defense. The planner may request a tool; the
server decides whether it actually runs.

## Context and Memory Behavior

The unified path must preserve the current distinction between explicit
assistant interactions and passively observed channel traffic.

Rules:

- successful `chat` interactions store conversation context the same way
  `@ask` does today
- successful `@code` and `@draw` wrappers keep their current storage
  behavior
- deterministic state wrappers such as `@memories` or `@usage` do not
  automatically enter conversational history
- passive channel tracking in `doPrivmsg` remains opt-in and non-persistent
  where the current design already uses `persist=False`

Memory extraction should remain limited to explicit assistant interactions,
not passive observed chat. A bot-nick mention counts as an explicit
assistant interaction; an ordinary unaddressed channel line does not.

## Tool Families

The registry should be organized around tool families rather than command
names.

### Conversation and grounding

- `generate_text`
- `search_web`
- `fetch_url`

### Code and media

- `generate_code`
- `generate_image`

### User state

- `get_instruction`
- `set_instruction`
- `clear_instruction`
- `list_memories`
- `save_memory`
- `update_memory`
- `delete_memory`
- `clear_memories`
- `forget_context`

### Reminders and usage

- `list_reminders`
- `set_reminder`
- `delete_reminder`
- `get_usage`
- `get_channel_usage`

Some of these already exist in `MetaToolExecutor`; the design change is to
promote them into the shared tool substrate instead of keeping them scoped
to a special `meta` path.

## Large Tool Outputs

The unified assistant creates a good opportunity to remove bespoke summary
calls for tool outputs that are inherently large.

Recommended rule:

- tools that produce large payloads should return structured results
- the planner's final response turn should summarize those results for IRC
- separate one-off summary calls such as a dedicated `summarize()` pass for
  code previews should be retired where the planner already has another
  response turn available

For example, `generate_code` should ideally return data such as:

- artifact URL
- short metadata about language or file shape
- raw or truncated code content for the planner context

Then the planner writes the concise IRC-facing preview in its normal final
turn, the same way it already turns tool results like "deleted 3 memories"
into a user-facing summary.

## Profiles

### `chat`

Used by PMs, nick mentions, unknown addressed text, and `@ask`.

Characteristics:

- planner enabled
- concise IRC-oriented final responses
- can combine search, state, reminder, code, and image tools
- outer rate bucket defaults to `ask`

Important rule:

- expensive tools still charge their own buckets and enforce their own
  capabilities even when reached from `chat`

This means "VibeBot, draw me a cat" is legal if and only if the caller
would be allowed to use `@draw`.

Initial rollout guardrail:

- `generate_image` is not exposed through `chat` on day one
- image requests on mention or PM routes should reply with guidance to use
  `@draw` until tool-level cost accounting and auth enforcement have been
  exercised in production

### `code`

Used by `@code`.

Characteristics:

- direct or planned execution is acceptable
- final renderer prefers code-link output
- code generation is the primary tool
- search/fetch may be visible if useful for implementation requests
- when planner mode is used, the final assistant turn should summarize the
  code result instead of calling a separate summary helper

### `draw`

Used by `@draw`.

Characteristics:

- image generation is the primary tool
- requires an authenticated account
- uses draw rate limits and cost accounting
- returns image URL or generated asset summary

### `state-direct`

Used by deterministic command wrappers.

Characteristics:

- planner disabled by default
- wrapper parses user intent and calls shared executor directly
- exact subcommand behavior remains stable for power users

This is the compatibility path that keeps old commands cheap and clear.

## Destructive Actions

Prompt instructions alone are not a strong enough guarantee for destructive
operations.

The registry should distinguish:

- single-item mutations
- bulk-destructive actions

Examples of bulk-destructive actions:

- `clear_memories`
- `clear_instruction`

Recommended rule:

- deterministic wrappers may execute immediately because the command itself
  is explicit
- planner-driven bulk-destructive actions should support a confirmation
  step before execution

This can be implemented later, but the registry needs the metadata now.

## Grounding Design

### Why not provider-native grounding in the planner call

The current service layer already shows the conflict:

- standard completion path can use Gemini grounding tools
- tool-calling path needs full control of tool round trips
- silently stripping tools on error is unacceptable for stateful actions

Trying to make one provider call do both creates brittle behavior.

### Recommended shape

Expose a leaf tool such as:

```json
{
  "summary": "...",
  "sources": [
    {"title": "...", "url": "..."}
  ],
  "grounding_used": true
}
```

The implementation can internally call a grounded model using a separate
request path. The planner receives facts and links, then composes the final
IRC response.

This also preserves the current "grounding used" icon behavior by bubbling
the flag up from the leaf tool result.

### Latency tradeoff

This design intentionally adds one extra round trip for grounded requests:

1. planner decides it needs search or URL evidence
2. leaf grounding tool performs the grounded lookup
3. planner receives structured results and writes the final reply

That is slower than a single provider-native grounded completion, but the
tradeoff is acceptable because it keeps tool orchestration correct,
provider-agnostic, and auditable. It also means grounding cost and latency
are paid only when the planner explicitly asks for them.

## Backward Compatibility

Existing commands stay user-visible for now:

- `@ask`
- `@code`
- `@draw`
- `@instruct`
- `@memories`
- `@forget`
- `@remind`
- `@usage`

They become wrappers over the shared backend or shared executors.

`@meta` is different:

- keep temporarily as an alias if needed
- remove from help and docs
- do not invest in it as a first-class concept

## Logging and Observability

Outer command logging is no longer enough once one request can invoke
multiple internal actions.

The system should log both:

- outer request metadata: route, profile, caller, model, total cost
- per-tool events: tool name, allowed or denied, tool-specific cost, rate
  bucket used, grounding flag, and any policy denial reason

This can start as structured logs even if a new database table is deferred.

## Migration Plan

### Phase 1: Shared substrate

- Introduce server-side `ToolSpec` registry with policy metadata.
- Keep existing `meta` names temporarily to avoid rename churn.

### Phase 2: Unified facade and first callers

- Add a unified `assistant_request()` facade in `service.py`.
- Convert `@ask` and `invalidCommand` to use the facade.
- Keep the current grounded `completion()` path available for ask-like
  behavior until grounding leaf tools land.

### Phase 3: Mention routing

- Add PM and bot-nick mention routing through `doPrivmsg`.
- Add `mentionEnabled` as a rollback lever.
- Extract `doPrivmsg` helper methods so mention routing does not further
  overload one large method.
- Add dedupe so command and mention paths cannot both answer the same line.

### Phase 4: Grounding tools

- Add `search_web` / `fetch_url` tools that internally use the existing
  grounded Gemini path.
- Remove the need to mix provider-native grounding into the planner call.

### Phase 5: Wrapper conversion

- Convert `@code` and `@draw` into wrappers over the unified
  assistant backend.
- Keep deterministic wrappers for `@memories`, `@instruct`, `@forget`,
  `@remind`, and `@usage`.

### Phase 6: Cleanup

- Hide `@meta` from help and docs.
- Keep it as an alias for one compatibility window if desired.
- Rename `meta`-specific internals only after the shared assistant path is
  stable.

### Phase 7: Observability

- Add per-tool logging and policy-denial visibility once the execution
  substrate is stable enough to observe.

## Initial Defaults

1. `chat` should not expose `generate_image` in the first rollout; mention
   and PM requests should direct users to `@draw`.
2. Planner-driven confirmation is only required for bulk-destructive
   actions such as `clear_memories` and `clear_instruction`.
3. Structured logs are sufficient for the first observability pass; a new
   database table can wait until real query needs appear.

## Recommendation

Proceed with the unified assistant architecture, but keep the security
model command-independent:

- route-level preflight in `plugin.py`
- tool-level policy enforcement in the executor
- grounding implemented as a separate leaf tool
- explicit commands preserved as wrappers

That gets the simplification benefits of "everything is meta" without
turning the chat route into an access-control bypass.
