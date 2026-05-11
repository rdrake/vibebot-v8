# Routing/intent layer — design

**Date:** 2026-05-11
**Status:** Design approved, pending implementation plan. Revised 2026-05-11 after codex + code-reviewer pass.
**Context:** Sub-project E of the vibebot Go rewrite. The rewrite is greenfield, driven by four goals: escape Limnoria/Supybot quirks, lock down cache discipline with Gemini explicit `CachedContent`, slim the tool surface, and consolidate on Go. Surface is reimagined, not ported.

## Scope

This spec covers the routing/intent layer only — the component that converts an inbound IRC event into a decision about whether and how the bot responds, and the executor that carries that decision out.

Out of scope (separate sub-project specs):

- **A** IRC core (ergochat/irc-go wrapper, CAP, SASL, multi-network)
- **B** LLM core + cache (openai-go vs. Gemini compat, `CachedContent` lifecycle)
- **C** Tool surface (curated tool implementations, dispatcher, rate buckets)
- **D** Persona/overlay system (channel `assistantSystemPrompt`, scene/loom overlays)
- **F** Persistence (memories, instructions, usage, conversation history)
- **G** Deploy/ops (single-binary build, config, logging, observability)

The routing layer consumes interfaces from A/B/C/D/F. Their full surface is spec'd in their own docs, but constraints E imposes on them are pinned here in "Cache prefix composition".

## Drivers

- Reimagined surface: hybrid model — `@cmd` for admin ops, fully natural language for chat. No `@ask` required to speak to the bot.
- Per-channel engagement: address-only by default (nick mention / reply-to-bot / DM). Channels can opt into ambient chatter.
- Cache discipline as first-class concern: both the cache *key* (`CacheScope`) and the cache *bytes* (`CachedPrefix`) are deterministic functions of `(network, channel, profile, overlay_hash)` — never user nick, never message text. Spec'd together in "Cache prefix composition".
- Slimmer tool surface: per-profile static tool allowlist. Today's 21 tools become 5–10 per profile.

## Architecture

The layer is a single Go package `router` sitting between IRC and the LLM executor.

```
   IRC events                             IRC out
      │                                      ▲
      ▼                                      │
┌──────────────────┐    RouteDecision    ┌────────────────┐
│   router.Route   │ ──────────────────▶ │  exec.Run      │
│   (pure func)    │                     │  (stateful)    │
└──────────────────┘                     └────────────────┘
       │                                      │
       │ reads                                │ calls
       ▼                                      ▼
  ChannelState                            LLM core (B)
  ProfileRegistry                         Tool dispatch (C)
  BotState                                Persona overlay (D)
                                          Persistence (F)
                                          IRC out (A)
```

`router.Route` is a pure function: same inputs always produce the same `RouteDecision`. No I/O, no clocks, no randomness. Clock and last-ambient-timestamp are threaded in via `BotState`.

`exec.Run` is the only place that performs I/O. It owns the multi-step tool-call loop, typing notifications, response chunking, persistence writes, and error recovery.

`@cmd` admin commands bypass routing entirely — they go from IRC core to an admin dispatcher (separate spec, not E).

## Components

### Inputs

```go
type IRCEvent struct {
    Network    string
    Channel    string             // "" for DM
    Nick       string
    Account    string             // SASL-authenticated account, "" if unauth
    Text       string
    IsAction   bool
    Tags       map[string]string  // IRCv3 message tags
    MessageID  string             // for reply-to
    ReceivedAt time.Time
}

type ChannelState struct {
    Profile         string         // see Profile registry below
    Overlay         string         // channel assistantSystemPrompt; router never edits
    AmbientEnabled  bool
    AmbientCooldown time.Duration
    SceneActive     *SceneRef      // nil when no scene
    LoomActive      *LoomRef       // nil when no loom
}

type BotState struct {
    SelfNick       string
    LastAmbientAt  map[ChannelKey]time.Time  // snapshot taken at dispatch time; see "Ambient claim"
    Now            time.Time                 // dispatcher passes time.Now() at call site
    RecentSentIDs  []string                  // ring buffer of last 50 bot-sent message IDs for reply detection
}
```

### Output

```go
type Action int
const (
    Ignore Action = iota
    RespondChat
    RespondScene
    RespondLoom
)

type CacheScope struct {
    Network     string
    Channel     string
    Profile     string
    OverlayHash string  // sha256 of overlay text from D, truncated to 16 bytes hex
}

type RouteDecision struct {
    Action     Action
    Profile    string
    Tools      []string      // tool name allowlist (canonical order: alphabetical)
    CacheScope CacheScope
    Model      string        // pinned model ID, e.g. "gemini-2.5-flash"
    Prompt     PromptSpec    // overlay-by-ref, history tail spec, user text
    Delivery   DeliverySpec  // best-effort typing, chunk size, reply-to id if CAP negotiated, max iters
}
```

`CacheScope` is the cache *key*. The cache *bytes* are spec'd in "Cache prefix composition" — both must agree byte-for-byte for two turns with the same scope to hit Gemini's cache.

`Tools` is a list of names in alphabetical order. The router never imports tool implementations — it names them.

### Executor

```go
type Executor struct {
    llm     llmcore.Client       // B; EnsureCache + Complete, contract in "Cache prefix composition"
    tools   tooling.Dispatcher   // C
    overlay overlay.Resolver     // D; Get(scope) MUST be a pure function
    store   persist.Store        // F
    irc     ircout.Sender        // A
    log     *slog.Logger
}

func (e *Executor) Run(ctx context.Context, d RouteDecision) error
```

Concurrency model: **one `Executor.Run` in flight per `(Network, Channel)` at any time.** The dispatcher acquires a per-channel turn lock before calling `Run`. DM channels use the user's nick as the channel key. Cross-channel turns run in parallel. This is what makes the atomic ambient claim correct (see "Ambient claim" in Data flow).

### Profile registry

#### Profile naming — deliberate rename from v8

The Python codebase pins `PROFILE_CHAT`, `PROFILE_CODE`, `PROFILE_DRAW`, `PROFILE_VERSE`, `PROFILE_REMIND_ACTION` (see `docs/superpowers/specs/2026-05-11-profile-abstraction-design.md`). The Go rewrite reimagines the surface and renames profiles accordingly:

| v8 profile | v9 profile | Disposition |
|---|---|---|
| `PROFILE_CHAT` | `chat` | renamed |
| `PROFILE_VERSE` | `scene` | renamed; overlay-layering semantics preserved |
| `PROFILE_CODE` | folded into `admin` | code-generation tool restricted to admin contexts |
| `PROFILE_DRAW` | folded into the tool | `generate_image` is a tool available in `chat`/`scene`/`loom`/`admin`, not a routing mode |
| `PROFILE_REMIND_ACTION` | deleted | reminder/scheduler features cut entirely per production data |
| — | `quiet` | new: minimal-tool default for new channels |
| — | `loom` | new: multi-bot weave |
| — | `admin` | new: owner DMs / explicit admin channels |

#### v9 profile table

Profiles are declarative configs registered at startup. The `Tools` column is the **complete** allowlist for each profile — not a delta from another profile.

| Profile | Tools (exact set) | Model | Ambient | MaxIters | Notes |
|---------|---|---|---|---|---|
| `quiet` | `clear_instruction`, `get_instruction`, `list_memories`, `save_memory`, `set_instruction` | gemini-2.5-flash | off | 2 | Address-only. Minimal. Default for new channels. |
| `chat` | `clear_instruction`, `delete_memory`, `fetch_url`, `generate_image`, `get_instruction`, `list_memories`, `save_memory`, `search_web`, `set_instruction`, `update_memory` | gemini-2.5-flash | optional | 2 | General chatter. |
| `scene` | `fetch_url`, `generate_image`, `get_instruction`, `list_memories`, `save_memory`, `search_web` | gemini-2.5-pro or xai-grok-4 | off | 3 | Verse-style narrative. Overlay layers scene context on top of channel `assistantSystemPrompt`. Destructive memory/instruction tools (`delete_memory`, `update_memory`, `set_instruction`, `clear_instruction`) excluded — they break immersion. |
| `loom` | `fetch_url`, `generate_image`, `get_instruction`, `list_memories`, `loom_propose`, `loom_seed`, `loom_yield`, `save_memory`, `search_web` | gemini-2.5-pro | off | 3 | Multi-bot weave. `loom_*` tool names are placeholders; pinned in D. |
| `admin` | chat set + `generate_code` + admin tools (pinned in C) | gemini-2.5-pro | off | 5 | Owner DMs, explicit admin channels. |

Adding a new profile is editing the registry, not the router. Channels declare their profile in config as a single string (config schema in G).

Reminder/scheduler tools (`set_reminder`, `cancel_pending_task`, `cancel_all_pending_tasks`, `schedule_llm_task`, `list_pending_tasks`) are absent from every profile — confirmed dead per production data (2 firings in 90 days).

#### Unknown profile fallback

If a channel's configured profile string is not in the registry, the dispatcher logs at WARN and treats the channel as `quiet`. The fallback is intentional: a misconfigured channel becomes safely minimal rather than crashing or escalating to a higher-tool profile.

## Cache prefix composition

The most load-bearing part of this spec. The cache *key* (`CacheScope`) and the cache *bytes* (`CachedPrefix`) must agree byte-for-byte for two turns with the same scope to hit Gemini's `CachedContent`.

### Prefix bytes (cached, stable per `CacheScope`)

In this exact order, separated by the fixed delimiter `\n\n---\n\n`:

1. **Framework system prompt** — a fixed string per `Profile`, registered alongside the profile config. No interpolation that depends on channel state. Lives next to the registry.
2. **Overlay text** — returned by `overlay.Resolver.Get(CacheScope)`. D MUST return a string that depends only on `(Network, Channel, Profile)`. No user nick, no timestamp, no per-account data. User-specific data goes in the uncached tail.
3. **Tool schemas** — for every tool in `RouteDecision.Tools` (already in alphabetical order), the full JSON schema. Canonical serialization: within each schema, JSON keys sorted alphabetically; UTF-8; no insignificant whitespace. C MUST return byte-identical schemas across calls; B MUST serialize canonically.
4. **Static channel context** — a fixed-key-order block: `"Network: {n}\nChannel: {c}\nProfile: {p}\n"`. No timestamp, no participant list, no message count.

E builds the four blocks into a `CachedPrefix` struct and passes it to B's `EnsureCache(scope, prefix) (cacheName string, err error)`. B is responsible for canonical serialization, `CachedContent.Create` if no cache exists for scope, TTL refresh if it does, and returning the cache name.

### Uncached tail (per-turn, per-user)

Everything below the cached prefix:

1. Recent conversation history (last N turns from `store`, N from `PromptSpec`)
2. User-specific memories injected for this turn (from F; scoped to the channel, optionally filtered by `evt.Nick`)
3. The user's current message
4. (During tool-call loop iterations) tool result messages appended after each round

Within one `Run`, the tail grows but the prefix never changes. The `cacheName` returned by `EnsureCache` is reused for every iteration in the loop.

### Constraints this section places on other sub-projects

**On B (LLM core + cache):**
- `Complete(messages, tools, model, cacheName)` MUST send the cached prefix bytes as-is when the provider supports explicit cache, and append the uncached tail. If the provider is xAI or the cache lookup misses, the prefix is sent inline and not cached — E does not need to know which mode B is in.
- `Complete` MUST canonicalize `tools` to byte-identical bytes against the cached prefix (sorted by name, sorted JSON keys, fixed whitespace).
- `Complete` MUST present uniform tool-call semantics across providers (parallel calls, `tool_choice`). If Gemini's OpenAI-compat endpoint lacks a feature, B degrades gracefully — E's loop is provider-agnostic.
- B MUST report `cached_tokens` in its response so E's logging can verify cache hits.

**On D (overlay):**
- `overlay.Resolver.Get(scope CacheScope) (text string, err error)` MUST be a pure function of `scope`. Same scope, same overlay text, byte-identical, across goroutines.
- D MAY NOT inject user-specific text into the overlay return value. User layering happens in the uncached tail.

**On C (tools):**
- For every tool name listed in `RouteDecision.Tools`, the JSON schema returned by C MUST be byte-identical across calls (no random ordering, no field omitted/added based on caller).

## Data flow

A single chat turn:

1. **Inbound** — IRC core decodes `PRIVMSG`, builds `IRCEvent`. Dispatcher checks for `@cmd` prefix; `@cmd` goes to admin path (out of scope). Otherwise:
   - Dispatcher acquires the per-channel **turn lock**. Subsequent IRC events for the same channel queue until the current `Run` returns.
   - Dispatcher takes a snapshot of `LastAmbientAt[channel]` and `time.Now()`, builds `BotState`, and calls `router.Route(evt, channelState, botState)`.
   - **Ambient claim:** if `Route` returns an ambient `RespondChat` (i.e. `Action == RespondChat` and event was not addressed), dispatcher atomically commits `LastAmbientAt[channel] = botState.Now` *before* calling `Executor.Run`. The combination of turn lock + atomic commit ensures two concurrent ambient triggers cannot both pass cooldown.

2. **Decision** — `Route` is pure. It first checks "addressed", then the ambient throttle if not addressed.

   ### Addressed semantics

   `addressed(evt, selfNick, recentSentIDs) = true` iff ANY of:
   - `evt.Channel == ""` (direct message — auth state irrelevant; capability gating happens later via profile)
   - `evt.Tags["+draft/reply"]` references a message-id present in `BotState.RecentSentIDs`
   - The first whitespace-delimited token of `evt.Text`, lowercased and stripped of trailing punctuation in `[:,;.!?]`, equals `strings.ToLower(selfNick)`

   Mid-message mention (`"hey vibebot what's up"`) does NOT count as addressed. That's a separate feature (today's NickInMiddle plugin) and is out of scope for v1.

   ### Decision tree

   ```
   addressed?
     yes → if ChannelState.LoomActive != nil → RespondLoom
           else if ChannelState.SceneActive != nil → RespondScene
           else                                   → RespondChat
     no  → ambient enabled?
             yes → ReceivedAt - LastAmbientAt[channel] ≥ AmbientCooldown?
                     yes → RespondChat   (ambient never triggers Scene/Loom)
                     no  → Ignore
             no  → Ignore
   ```

   If `Action != Ignore`, `Route` resolves `Profile` from `ChannelState.Profile` (fallback to `quiet` if unknown — logs WARN at the dispatch site, not in `Route`), `Tools` from `ProfileRegistry[Profile].Tools`, builds `CacheScope = {Network, Channel, Profile, OverlayHash}`, references `Prompt` (overlay-by-ref, history tail spec, user text), and sets `Delivery` (best-effort typing, chunk at 380 chars, reply-to event message id if `+draft/reply` CAP negotiated, max iters from profile).

3. **Execution** — `Executor.Run(ctx, decision)`:
   1. Resolve overlay via `overlay.Resolver.Get(decision.CacheScope)`
   2. Build `CachedPrefix` from (framework system prompt for profile, overlay, canonicalized tool schemas, static channel context) per "Cache prefix composition"
   3. Call `llm.EnsureCache(decision.CacheScope, cachedPrefix)` — returns a `CachedContent` name on Gemini, empty string on xAI
   4. Hydrate uncached tail: history from `store`, user-specific memories, current user message
   5. Send `TAGMSG +typing=active` (best-effort; ignored if CAP not negotiated or server doesn't relay)
   6. Loop: call `llm.Complete(messages, tools, model, cacheName)` until completion has no `tool_calls` or `MaxIters` reached. See "Multi-step tool-call loop" for invariants.
   7. Send response to IRC in chunks. If `+draft/reply` CAP is negotiated, tag the first chunk with `+draft/reply=<evt.MessageID>`; otherwise prefix the response with `<evt.Nick>: ` as a fallback.
   8. Send `TAGMSG +typing=done` (in `defer`, fires on every exit path including panic)
   9. Persist conversation update + usage row (including `cached_tokens` from B's response for cache-hit verification)

## Multi-step tool-call loop

```go
for i := 1; i <= MaxIters; i++ {
    completion := llm.Complete(messages, tools, model, cacheName)
    if len(completion.ToolCalls) == 0 {
        deliver(completion.Text)
        return nil
    }
    for _, tc := range completion.ToolCalls {
        result := tools.Dispatch(ctx, tc)
        messages = append(messages, toolResultMsg(tc, result))
    }
}
deliver(fallbackMessage)
return nil
```

`messages` is the **uncached tail only**. The cached prefix is referenced by `cacheName`, not duplicated in `messages`. Tool result messages append to the tail across iterations.

Invariants:

- **Cached prefix is identical across all iterations.** Guaranteed by construction: `cacheName` is fixed for the whole `Run`, and the prefix bytes were canonicalized once before the loop started.
- **`MaxIters` comes from the profile.** Not a global constant.
- **Tool dispatch has a per-call timeout** (default 30s, override per tool spec in C).
- **One typing notification spans the entire loop.** Sent once before the loop, cleared once after via `defer`.
- **`Complete` MUST canonicalize tool schemas** to match the cached prefix bytes (B's responsibility — see "Cache prefix composition").
- **Provider abstraction.** B's `Complete` presents uniform semantics across providers; E's loop never branches on provider identity.

## Error handling

| Error | Source | Recovery |
|---|---|---|
| `ErrLLMTransient` | network blip, 5xx, rate-limit | retry with backoff (3 tries, 1s/3s/9s), then brief apology to user |
| `ErrLLMFatal` | 4xx, auth, model-not-found | log + "I'm broken, ping the owner" message, drop turn |
| `ErrToolDenied` | profile doesn't allow tool, rate bucket exhausted | inject deny message into `messages`, loop continues so LLM can recover |
| `ErrToolFailed` | handler returned exception | inject error result, let LLM apologize or retry |
| `ErrIRCSend` | bridge to network broken | log; no user-visible recovery — resilient reconnect lives in A |
| `ErrCacheStale` | Gemini `CachedContent` expired mid-turn | rebuild cache once inline via `EnsureCache`, retry; if it fails again, fall through to non-cached path |
| `ErrBudgetExceeded` | per-turn token/cost cap hit | deliver partial response, log overrun, no retry |

The executor wraps every step in a `defer` that ensures `+typing=done` fires even on panic. No stuck "bot is typing" indicators.

## Testing strategy

Four tiers:

1. **Router tests (table-driven, hermetic).** `(IRCEvent, ChannelState, BotState) → expected RouteDecision` cases. No mocks. Coverage target ≥95%. Pins engagement rule, addressed semantics, ambient throttle, profile resolution, cache scope construction, unknown-profile fallback.

2. **Executor tests (fakes for LLM + IRC + overlay + tool dispatcher).** In-memory LLM fake returns scripted completions including scripted tool calls. In-memory IRC sink captures sends. Fake overlay resolver returns canned strings. Asserts loop behavior, typing-notification ordering, chunking, error recovery, `defer`-based typing cleanup on panic, ambient-claim race with concurrent dispatch. Coverage target ≥80%.

3. **Local-IRCd integration (Ergo + fake LLM).** Real `ergochat/irc-go` client against a local Ergo IRCd in CI. Verifies CAP negotiation, SASL, `TAGMSG` framing, `+draft/reply` tagging, and the fallback path when CAP isn't negotiated. LLM remains a fake. One happy path + one tool-call path.

4. **Live-provider smoke (gated, manual, not in CI).** Real Gemini + real xAI keys, real `CachedContent` lifecycle. Verifies:
   - `EnsureCache` creates a cache the first call
   - Second `Complete` for the same scope reports `cached_tokens > 0`
   - Third `Complete` after overlay change reports `cached_tokens == 0`
   - xAI calls return `cached_tokens == 0` always (uncached path)

   Run manually pre-release. Results recorded in a tracking doc. This is the *only* way to verify cache discipline end-to-end — request/response record-replay cannot exercise Gemini-side cache state.

## Risks and open questions

- **Overlay determinism enforcement.** The hard purity constraint on D is spec'd here, but enforcement lives in D's test suite. D MUST include a property test asserting `Get(scope)` returns byte-identical text across N calls and N goroutines for the same scope.
- **xAI cache compatibility.** Gemini explicit `CachedContent` is the primary path. xAI has no equivalent; turns through xAI-routed profiles (e.g. `scene` may route to xai-grok-4) pay full prefix tokens every call. Decision: accepted; route most channels through Gemini.
- **`openai-go` + Gemini OpenAI-compat tool-call gaps.** Known gaps: parallel tool calls, `tool_choice=required/none`. B's `Complete` MUST abstract these. If a gap can't be hidden, the leak surfaces in B's spec, not E's.
- **IRCv3 `+typing` and `+draft/reply` are best-effort.** AfterNet may not relay `TAGMSG`; `+draft/reply` requires CAP and falls back to `nick: ` prefix. Documented in `Delivery`, not a hard requirement. Test tier 3 verifies fallback.
- **Loom tool names are placeholders.** `loom_propose`, `loom_yield`, `loom_seed` listed pending D. If D renames them, the profile registry is the single edit site.
- **Ambient bursts.** Per-channel turn lock + atomic `LastAmbientAt` commit handles documented races. If a future channel sees ambient bursts that overwhelm the lock, revisit with a per-channel rate limiter in F.
- **Memory privacy.** User-specific memories live in the uncached tail; the cached prefix carries no user data. Channel-shared memories merge into the tail per-turn. Confirm tail composition with F.

## Out of v1

- Streaming partial deliveries to IRC. v1 delivers final response only.
- Conversation summarization between turns. v1 uses raw tail with a fixed history cap.
- Dynamic tool selection. v1 is static per-profile; deferred unless data justifies it.
- Multi-bot loom coordination. The protocol lives in D; v1 routing hands off to a `RespondLoom` action that D's executor handles.
- Mid-message mention detection. Tracked in a future "natural addressed detection" iteration.
