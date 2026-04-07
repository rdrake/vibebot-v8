# Rate Limiting & Security

## Rate limiting

### Overview

Rate limiting controls how many requests each user can make per time window. Limits are checked per user and per command.

The `enforceRateLimits` setting controls whether limits are enforced or just monitored:

| Value | Behavior |
|-------|----------|
| `True` (default) | Requests exceeding the limit are blocked |
| `False` | Limits are tracked and logged but not enforced (monitor/shadow mode) |

### Tiers

Users fall into one of four tiers, checked from most to least privileged:

| Tier | How to qualify | Rate limit behavior |
|------|---------------|-------------------|
| **Owner/Admin** | Limnoria owner or admin capability | Always exempt |
| **Trusted** | Has the `trusted` capability | Uses `Trusted` limit settings |
| **Registered** | Identified with NickServ | Uses standard limit settings |
| **Unregistered** | Not identified | Uses `Unreg` limit settings |

To grant a user the `trusted` capability, see the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html).

### Per-command settings

Each command has three sets of rate limit settings (one per non-exempt tier). The pattern is:

- `{cmd}RateLimitCount` / `{cmd}RateLimitWindow` -- registered users
- `{cmd}TrustedRateLimitCount` / `{cmd}TrustedRateLimitWindow` -- trusted users
- `{cmd}UnregRateLimitCount` / `{cmd}UnregRateLimitWindow` -- unregistered users

Setting any count to `0` disables rate limiting for that command and tier (unlimited).

### Default limits

| Command | Tier | Count | Window | Effective rate |
|---------|------|-------|--------|---------------|
| `ask` | Registered | 15 | 60s | 15 per minute |
| `ask` | Trusted | 15 | 60s | 15 per minute |
| `ask` | Unregistered | 15 | 60s | 15 per minute |
| `code` | Registered | 10 | 60s | 10 per minute |
| `code` | Trusted | 0 | 60s | Unlimited |
| `code` | Unregistered | 2 | 60s | 2 per minute |
| `draw` | Registered | 2 | 300s | 2 per 5 minutes |
| `draw` | Trusted | 5 | 60s | 5 per minute |
| `draw` | Unregistered | 0 | 60s | Blocked (also requires NickServ) |

### Complete rate limit settings

| Setting | Default |
|---------|---------|
| `askRateLimitCount` | `15` |
| `askRateLimitWindow` | `60` |
| `askTrustedRateLimitCount` | `15` |
| `askTrustedRateLimitWindow` | `60` |
| `askUnregRateLimitCount` | `15` |
| `askUnregRateLimitWindow` | `60` |
| `codeRateLimitCount` | `10` |
| `codeRateLimitWindow` | `60` |
| `codeTrustedRateLimitCount` | `0` |
| `codeTrustedRateLimitWindow` | `60` |
| `codeUnregRateLimitCount` | `2` |
| `codeUnregRateLimitWindow` | `60` |
| `drawRateLimitCount` | `2` |
| `drawRateLimitWindow` | `300` |
| `drawTrustedRateLimitCount` | `5` |
| `drawTrustedRateLimitWindow` | `60` |
| `drawUnregRateLimitCount` | `0` |
| `drawUnregRateLimitWindow` | `60` |

## Security

### Capability-based access control

Each command requires a Limnoria capability:

| Command | Required capability |
|---------|-------------------|
| `@ask` | `llm.ask` |
| `@code` | `llm.code` |
| `@draw` | `llm.draw` |

By default, all users have these capabilities. To restrict a command, use Limnoria's capability system to remove it from specific users or channels. See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for details.

### NickServ gating

The `@draw` command requires users to be identified with NickServ before use. This is enforced regardless of capability settings and provides accountability for image generation.

### URL validation

When users include image URLs in `@ask` prompts (for vision), the bot validates them:

- **Scheme check:** Only `http://` and `https://` URLs are accepted. Schemes like `javascript:`, `data:`, and `file:` are blocked.
- **Path traversal:** URLs containing `..` in the path are rejected.
- **SSRF protection:** URLs resolving to private, loopback, link-local, or reserved IP addresses are blocked. The check fails closed -- if DNS resolution fails, the URL is rejected.
- **Extension check:** Only recognized image extensions are accepted (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`).

### Output sanitization

The bot sanitizes all LLM output before sending it to IRC:

- **IRC command injection:** Lines starting with configured command prefixes (default: `.`) are prefixed with a space to prevent them from being interpreted as bot commands. Configure additional prefixes with `commandPrefixes`.
- **API key scrubbing:** Error messages are scrubbed to remove API keys before they are displayed to users.
- **HTML sanitization:** Code output rendered as HTML is sanitized with `nh3` (allowlist-based) to prevent XSS.

### Prompt injection defenses

The bot uses several techniques to mitigate prompt injection:

- **Structured system prompts:** System prompts use clear `INSTRUCTIONS` vs `CONTEXT` sections with an anti-injection preamble that instructs the model to ignore any instructions embedded in user-provided context.
- **Channel topics as untrusted data:** Channel topics are included as user messages rather than system messages, since they can be set by any channel member.
- **Input length limits:** User prompts are capped at `maxPromptLength` characters (default: 10,000).
