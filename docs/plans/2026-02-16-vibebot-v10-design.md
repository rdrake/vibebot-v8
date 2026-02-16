# VibeBot v10 Design

**Date:** 2026-02-16
**Status:** Draft
**Goal:** Complete rewrite of VibeBot as a multi-plugin Limnoria architecture with dramatically reduced complexity.

## Motivation

VibeBot v8 works well but has accumulated significant complexity — ~7,150 LOC across 6 source files, with the background retry queue alone accounting for ~1,500 lines. The custom two-phase state machine with lease-based locking, exponential backoff, and event-driven scheduling is the single largest maintenance burden. LLM latency has improved enough that most of this infrastructure is no longer necessary.

v10 is a ground-up rewrite that keeps the battle-tested Limnoria IRC framework, carries forward all security patterns, and restructures everything else around three principles:

1. **Plugins are thin.** All business logic lives in a shared library. Plugin command handlers are 15-25 lines.
2. **Redis replaces custom state management.** Conversation context, rate limiting, and expiry are handled by Redis natively — no threading locks, no cleanup threads, no manual expiry.
3. **Drop complexity that doesn't earn its keep.** No retry queue, no auto-rewrite loop, no reminders, no video generation. These can be added later as independent plugins if needed.

## Scope

### In scope (day one)

- `ai` command group: ask (default), code, draw subcommands
- Conversation context with personal + channel history (Redis-backed)
- Vision support (image URL detection in ask)
- Gemini Google Search grounding
- Code syntax highlighting with hosted HTML pages
- Image generation with hosted output
- Per-user rate limiting (Redis-backed)
- Admin flagging (flag/unflag/flagged)
- Usage tracking and cost reporting (SQLite)
- HTTP file serving via Limnoria's built-in server
- All v8 security patterns
- i18n support
- Direct addressing (vibebot: hello → ask)
- ZNC playback filtering

### Out of scope (add later as separate plugins)

- Background retry queue
- Auto-rewrite on draw safety blocks
- Reminders (`%remindme`)
- Video generation (`%animate`)
- IRCv3 typing indicators

## Project Structure

```
vibebot-v10/
├── lib/vibebot/                # Shared library (installed as a package)
│   ├── __init__.py
│   ├── llm.py                  # LiteLLM wrapper: completion, image gen, summarization
│   ├── redis.py                # Redis client: context store, rate limiter
│   ├── security.py             # Output sanitization, SSRF checks, key display
│   ├── types.py                # Pydantic models (CompletionResult, ImageResult, etc.)
│   └── tracing.py              # Request ID generation, header extraction
│
├── plugins/
│   ├── AI/                     # Core command group
│   │   ├── __init__.py
│   │   ├── plugin.py           # ai ask, ai code, ai draw, ai forget + default routing
│   │   ├── config.py           # Per-subcommand: model, apiKey, systemPrompt
│   │   ├── locales/
│   │   └── test.py
│   │
│   ├── AIAdmin/                # Admin & abuse prevention
│   │   ├── __init__.py
│   │   ├── plugin.py           # usage, flag, unflag, flagged, aikeys
│   │   ├── config.py           # Rate limit settings, flag config
│   │   ├── locales/
│   │   └── test.py
│   │
│   └── AIFiles/                # HTTP file serving
│       ├── __init__.py
│       ├── plugin.py           # HTTP callback, file cleanup
│       ├── config.py           # httpRoot, cleanup settings
│       └── test.py
│
├── tests/                      # Integration tests across plugins
├── pyproject.toml              # Workspace: lib + 3 plugins
├── Makefile
├── Dockerfile
└── docker-compose.yml          # Bot + Redis
```

## Shared Library Design

### `llm.py` — LiteLLM Wrapper (~200 LOC)

Three entry points cover all LLM operations:

```python
async def complete(
    prompt: str,
    model: str,
    api_key: str,
    system_prompt: str = "",
    history: list[Message] = [],
    images: list[str] = [],
) -> CompletionResult:

async def generate_image(
    prompt: str,
    model: str,
    api_key: str,
) -> ImageResult:

async def summarize(
    text: str,
    model: str,
    api_key: str,
    max_words: int = 50,
) -> str:
```

Handles internally: provider-specific kwargs (Gemini safety settings, grounding tools), vision content array building, tool fallback for Gemini, cost extraction, error classification.

Does not handle: IRC protocol, retries, file I/O, context lookup.

### `redis.py` — State Store (~100 LOC)

Two interfaces backed by Redis:

```python
class ContextStore:
    async def add_message(self, nick: str, channel: str, role: str, content: str) -> None
    async def get_history(self, nick: str, channel: str) -> list[Message]
    async def add_channel_message(self, nick: str, channel: str, content: str) -> None
    async def get_channel_history(self, channel: str) -> list[Message]
    async def clear(self, nick: str, channel: str | None = None) -> bool

class RateLimiter:
    async def check(self, account: str, command: str, limit: int, window: int) -> RateLimitResult
    # Uses ZRANGEBYSCORE + ZCARD atomically via Lua script
```

Redis key patterns:
- `vb:ctx:{nick}:{channel}` — personal context (list + LTRIM + EXPIRE)
- `vb:chctx:{channel}` — channel context (same pattern)
- `vb:rl:{command}:{account}` — rate limits (sorted set with timestamp scores)

### `security.py` — Safety (~100 LOC)

Carries forward all v8 security patterns:
- `sanitize_output(text, prefixes)` — IRC command injection prevention
- `sanitize_error(message, api_keys)` — scrub keys from error messages
- `validate_image_url(url)` — SSRF protection (block private IPs, validate scheme)
- `safe_key_display(key)` — show first 3 chars only

### `types.py` — Shared Models (~50 LOC)

Pydantic models replacing NamedTuples:

```python
class CompletionResult(BaseModel):
    content: str
    grounding_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

class ImageResult(BaseModel):
    image_data: bytes
    format: str = "png"
    cost: float = 0.0

class RateLimitResult(BaseModel):
    allowed: bool
    remaining: int = 0
    retry_after: float = 0.0
```

### `tracing.py` — Diagnostics (~40 LOC)

Request ID via `contextvars`, server header extraction from LiteLLM responses.

## Plugin Designs

### AI Plugin — Core Commands (~400 LOC)

Owns the `ai` command group with nested subcommands:

```python
class AI(callbacks.Plugin):
    threaded = True

    class ai(callbacks.Commands):
        class ask(callbacks.Commands):  # default subcommand
            @wrap(["text"])
            def __call__(self, irc, msg, args, prompt): ...

        class code(callbacks.Commands):
            @wrap(["text"])
            def __call__(self, irc, msg, args, prompt): ...

        class draw(callbacks.Commands):
            @wrap(["text"])
            def __call__(self, irc, msg, args, prompt): ...

        class forget(callbacks.Commands):
            @wrap([optional("channel")])
            def __call__(self, irc, msg, args, channel): ...
```

Command routing:
- `@ai what is rust?` → ask (default when no subcommand matches)
- `@ai draw a cat` → draw
- `@ai code fibonacci` → code
- `vibebot: hello` → ask via `invalidCommand()`

Preflight checks via `pre_command_callbacks`:

```python
def __init__(self, irc):
    super().__init__(irc)
    self.pre_command_callbacks.append(self._preflight)

def _preflight(self, plugin, command, irc, msg, *args, **kwargs):
    admin = irc.getCallback("AIAdmin")
    if admin and admin.is_flagged(msg):
        return True  # block command
    return False
```

### AIAdmin Plugin — Abuse Prevention & Stats (~300 LOC)

Public API consumed by AI plugin via `irc.getCallback("AIAdmin")`:
- `is_flagged(msg) -> bool`
- `check_rate_limit(msg, command) -> RateLimitResult`
- `log_usage(nick, channel, command, model, cost, status, ...)`

IRC commands: `usage`, `flag`, `unflag`, `flagged`, `aikeys`.

SQLite schema (2 tables, down from 4):

```sql
CREATE TABLE usage (
    id INTEGER PRIMARY KEY,
    timestamp TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    account TEXT NOT NULL,
    channel TEXT,
    command TEXT NOT NULL,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    status TEXT DEFAULT 'success'
);

CREATE TABLE flagged_users (
    account TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    flagged_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    flagged_by TEXT NOT NULL
);
```

### AIFiles Plugin — HTTP File Serving (~150 LOC)

Public API consumed by AI plugin via `irc.getCallback("AIFiles")`:
- `save_html(content, filename) -> str` (returns URL)
- `save_image(data, fmt, filename) -> str` (returns URL)

Registers Limnoria HTTP callback at `/ai/`. Runs hourly cleanup via `schedule.addPeriodicEvent()`.

### Inter-Plugin Communication

```
AI ──→ AIAdmin.is_flagged()         # preflight
AI ──→ AIAdmin.check_rate_limit()   # before draw
AI ──→ AIAdmin.log_usage()          # after every call
AI ──→ AIFiles.save_html()          # code output
AI ──→ AIFiles.save_image()         # draw output
```

Graceful degradation: if AIAdmin isn't loaded, skip flagging/rate checks/logging. If AIFiles isn't loaded, reply with raw text instead of URL.

## Configuration

```
supybot.plugins.AI.
    askModel                    # e.g., "gemini/gemini-2.0-flash"
    askApiKey                   # (private)
    askSystemPrompt
    codeModel
    codeApiKey                  # (private)
    codeSystemPrompt
    drawModel                   # e.g., "vertex_ai/imagen-4.0-generate-001"
    drawApiKey                  # (private)
    contextMaxMessages          # default: 20
    contextTTL                  # default: 300 (seconds)
    contextTrackChannel         # default: False

supybot.plugins.AIAdmin.
    drawRateLimit               # default: 3
    drawRateWindow              # default: 60 (seconds)
    enforceRateLimits           # default: True
    dbPath                      # default: data/ai-usage.db

supybot.plugins.AIFiles.
    httpRoot                    # optional external URL override
    maxFileAge                  # default: 30 (days)
    maxFileCount                # default: 1000
```

No `ValidatedModelName` — LiteLLM gives clear errors on typos. No shadow mode — enforce or don't. Channel-specific overrides via Limnoria's `registerChannelValue()` for model/prompt/context settings.

## Deployment

```yaml
# docker-compose.yml
services:
  bot:
    image: ghcr.io/rdrake/vibebot-v10:latest
    volumes:
      - ./bot.conf:/app/bot.conf
      - ./data:/app/data
    environment:
      - VIBEBOT_REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data
    command: redis-server --save 60 1

volumes:
  redis-data:
```

Redis connection via `VIBEBOT_REDIS_URL` environment variable.

## Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "limnoria>=2023.1.20",
    "litellm>=1.55",
    "redis[hiredis]>=5.0",
    "pydantic>=2.0",
    "nh3",
    "pygments",
]
```

## Testing Strategy

**Unit tests — shared library** (no IRC mocking):
- `test_llm.py` — mock LiteLLM, verify provider kwargs, error classification, cost extraction
- `test_redis.py` — fakeredis, verify TTL behavior, rate limit sliding window
- `test_security.py` — pure functions: sanitization, SSRF validation, key display
- `test_types.py` — Pydantic model validation

**Plugin tests** (Limnoria test harness):
- `AI/test.py` — command routing, default fallback, context integration, preflight blocking
- `AIAdmin/test.py` — usage logging, flag/unflag, rate limit enforcement, stats display
- `AIFiles/test.py` — file save/serve, cleanup, path traversal prevention

**Integration tests:**
- Full flows: ask → log usage → check stats
- Flagged user blocked across all commands
- Rate limit → rejection → stats reflect it
- Code generation → file saved → HTTP accessible

## Estimated Size

| Metric | v8 | v10 | Change |
|--------|-----|------|--------|
| Source LOC | ~7,150 | ~1,800 | -75% |
| Source files | 6 | ~10 | More files, each smaller |
| Largest file | 2,893 | ~400 | -86% |
| SQLite tables | 4 | 2 | Dropped pending_tasks, reminders |
| Threading locks | 3 | 0 | Redis handles concurrency |
| Custom retry logic | ~1,500 | 0 | Dropped |
| Lines per new command | ~80 | ~20 | Thin plugin pattern |

## Future Plugins (out of scope)

These can be added as independent plugins without modifying any existing code:

- **AIReminders** — `ai remind` subcommand, NL parsing, persistent delivery
- **AIVideo** — `ai animate` subcommand, async polling, background delivery
- **AIMetrics** — Prometheus endpoint for usage/cost/latency metrics
