# Routing/intent layer — design

**Date:** 2026-05-11
**Status:** Design approved, pending implementation plan
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

The routing layer consumes interfaces from A/B/C/D/F. Those interfaces are assumed in this spec and will be pinned down in their own specs.

## Drivers

- Reimagined surface: hybrid model — `@cmd` for admin ops, fully natural language for chat. No `@ask` required to speak to the bot.
- Per-channel engagement: address-only by default (nick mention / reply-to-bot / DM). Channels can opt into ambient chatter.
- Cache discipline as first-class concern: the cached prefix must be a deterministic function of `(network, channel, profile, overlay_hash)` — never user nick, never message text.
- Slimmer tool surface: per-profile static tool allowlist. Today's 21 tools become ~5–10 per profile.

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

`@cmd` admin commands bypass routing entirely — they go from IRC core to an admin dispatcher.

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
    Profile        string         // "quiet", "chat", "scene", "loom", "admin"
    Overlay        string         // channel assistantSystemPrompt; router never edits
    AmbientEnabled bool
    AmbientCooldown time.Duration
    SceneActive    *SceneRef      // nil when no scene
    LoomActive     *LoomRef       // nil when no loom
}

type BotState struct {
    SelfNick      string
    LastAmbientAt map[ChannelKey]time.Time  // populated by the dispatcher, read by Route
    Now           time.Time                 // dispatcher passes time.Now() at call site
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
    OverlayHash string  // sha256 of overlay text, truncated
}

type RouteDecision struct {
    Action     Action
    Profile    string
    Tools      []string      // tool name allowlist
    CacheScope CacheScope
    Model      string        // "gemini-2.5-flash" etc.
    Prompt     PromptSpec    // system ref + history tail spec + user text
    Delivery   DeliverySpec  // typing on/off, chunk size, reply-to id, max iters
}
```

`CacheScope` deliberately excludes nick and message text. That is what makes the Gemini `CachedContent` key stable per channel.

`Tools` is a list of names. The router never imports tool implementations — it just names them.

### Executor

```go
type Executor struct {
    llm     llmcore.Client       // B
    tools   tooling.Dispatcher   // C
    overlay overlay.Resolver     // D
    store   persist.Store        // F
    irc     ircout.Sender        // A
    log     *slog.Logger
}

func (e *Executor) Run(ctx context.Context, d RouteDecision) error
```

### Profile registry

Profiles are declarative configs registered at startup:

| Profile | Tools | Model | Ambient | MaxIters | Notes |
|---------|-------|-------|---------|----------|-------|
| `quiet` | `save_memory`, `list_memories`, `get_instruction`, `set_instruction`, `clear_instruction` | gemini-flash | off | 2 | Address-only. Minimal. Default for new channels. |
| `chat` | quiet + `search_web`, `fetch_url`, `delete_memory`, `update_memory`, `generate_image` | gemini-flash | optional | 2 | General chatter. |
| `scene` | chat minus destructive memory ops, plus `generate_image` | gemini-pro or xai-grok | off | 3 | Verse-style narrative. Overlay layers scene context on top of channel `assistantSystemPrompt`. |
| `loom` | scene + `loom_*` tools (defined in D) + `generate_image` | gemini-pro | off | 3 | Multi-bot weave, Forest's channel. |
| `admin` | everything in `chat` + `generate_image`, `generate_code`, admin tools | gemini-pro | off | 5 | Owner DMs, explicit admin channels. |

Adding a new profile is editing the registry, not the router. Channels declare their profile in config as a single string.

Reminder/scheduler tools (`set_reminder`, `cancel_pending_task`, `cancel_all_pending_tasks`, `schedule_llm_task`, `list_pending_tasks`) are intentionally absent from every profile — confirmed dead per production data (2 firings in 90 days).

## Data flow

A single chat turn:

1. **Inbound** — IRC core decodes `PRIVMSG`, builds `IRCEvent`. Dispatcher checks for `@cmd` prefix. `@cmd` goes to admin path. Otherwise, `router.Route` is called with the event, channel state, and current bot state.

2. **Decision** — `Route` runs a deterministic decision tree:

   ```
   addressed (nick mention / reply / DM)?
     yes → if ChannelState.LoomActive != nil → RespondLoom
           else if ChannelState.SceneActive != nil → RespondScene
           else                                   → RespondChat
     no  → ambient enabled?
             yes → ReceivedAt - LastAmbientAt[channel] ≥ AmbientCooldown?
                     yes → RespondChat   (ambient never triggers Scene/Loom)
                     no  → Ignore
             no  → Ignore
   ```

   If `Action != Ignore`, `Route` resolves `Profile` from `ChannelState.Profile`, `Tools` from `ProfileRegistry[Profile].Tools`, builds `CacheScope = {Network, Channel, Profile, OverlayHash}`, references `Prompt` (overlay by ref, history tail spec, user text), and sets `Delivery` (typing on, chunk at 380 chars, reply-to event message id, max iters from profile).

3. **Execution** — `Executor.Run(ctx, decision)`:
   1. Resolve overlay text from the resolver
   2. Hydrate conversation history from `store`
   3. Pre-warm Gemini cache via `llm.EnsureCache(CacheScope, prefix)` — returns a `CachedContent` name on Gemini, no-op on xAI
   4. Send `TAGMSG +typing=active`
   5. Call `llm.Complete(messages, tools, model, cacheName)` — streams
   6. If response has `tool_calls`: dispatch each via `tools.Dispatcher.Run`, append results, loop back to step 5 (bounded by `MaxIters`)
   7. Send response to IRC in chunks
   8. Send `TAGMSG +typing=done` (in `defer`, fires on every exit path)
   9. Persist conversation update + usage row

## Multi-step tool-call loop

```
for i := 1; i <= MaxIters; i++ {
    completion := llm.Complete(messages, tools, cache_name)
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

Invariants:

- The cached prefix is identical across all iterations within a single `Run`. Only the tail (`messages` appended with tool results) grows.
- `MaxIters` comes from the profile, not from a global constant.
- Tool dispatch has a per-call timeout (default 30s, override per tool spec).
- One typing notification spans the entire loop.

## Error handling

| Error | Source | Recovery |
|---|---|---|
| `ErrLLMTransient` | network blip, 5xx, rate-limit | retry with backoff (3 tries, 1s/3s/9s), then brief apology to user |
| `ErrLLMFatal` | 4xx, auth, model-not-found | log + "I'm broken, ping the owner" message, drop turn |
| `ErrToolDenied` | profile doesn't allow tool, rate bucket exhausted | inject deny message into `messages`, loop continues so LLM can recover |
| `ErrToolFailed` | handler returned exception | inject error result, let LLM apologize or retry |
| `ErrIRCSend` | bridge to network broken | log, no user-visible recovery; resilient reconnect lives in A |
| `ErrCacheStale` | Gemini `CachedContent` expired mid-turn | rebuild cache once inline, retry; if it fails again, fall through to non-cached path |
| `ErrBudgetExceeded` | per-turn token/cost cap hit | deliver partial response, log overrun, no retry |

The executor wraps every step in a `defer` that ensures `+typing=done` fires even on panic. No stuck "bot is typing" indicators.

## Testing strategy

Three tiers:

1. **Router tests (table-driven, hermetic).** `(IRCEvent, ChannelState, BotState) → expected RouteDecision` cases. No mocks. Coverage target ≥95%. Pins engagement rule, ambient throttle, profile resolution, cache scope construction.

2. **Executor tests (fakes for LLM + IRC).** In-memory LLM fake returns scripted completions including scripted tool calls. In-memory IRC sink captures sends. Asserts loop behavior, typing notifications, chunking, error recovery. Coverage target ≥80%.

3. **End-to-end smoke (local Ergo IRCd + recorded LLM).** One happy path + one tool-call path. Real `openai-go` client; LLM calls go through a record/replay fixture for determinism. SASL auth, channel join, send-and-receive.

## Risks and open questions

- **Cache scope granularity vs. memory privacy.** Today, channel cache prefix includes everyone's memories merged in. New design implies same — memories scoped per-channel, not per-user, otherwise cache fragments per nick. Confirm with persistence spec (F).
- **Ambient throttle clock.** Threaded through `BotState.Now` for purity, but the dispatcher needs to populate it correctly. Spec it in the IRC core hand-off (A).
- **xAI cache compatibility.** Gemini-specific `CachedContent` is the primary path. xAI fallback is no-cache or implicit. Decision: accept that xAI profile turns are uncached, document it, route most channels through Gemini. Settle in B.
- **Loom tooling.** `loom_*` tools listed but not specified — they live in D. Names are placeholders here.
- **Reply targeting.** `Delivery.ReplyToID` requires IRCv3 `+draft/reply` tag support. Confirm in A spec.

## Out of v1

- Streaming partial deliveries to IRC. v1 delivers final response only.
- Conversation summarization between turns. v1 uses raw tail with a fixed history cap.
- Dynamic tool selection. v1 is static per-profile; deferred to a future iteration if data justifies it.
- Multi-bot loom coordination. The protocol lives in D; v1 routing simply hands off to a `RespondLoom` action that D's executor handles.
