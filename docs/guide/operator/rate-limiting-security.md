# Rate limiting and security

## Rate limiting

Rate limits are checked per user and per command. The `enforceRateLimits` setting (global) controls whether limits block requests or only observe them:

| Value | Behaviour |
|-------|-----------|
| `True` (default) | Requests over the limit are blocked |
| `False` | Limits are tracked and logged but not enforced (observe-only mode) |

### Tiers

Users fall into one of four tiers, checked from most to least privileged:

| Tier | How to qualify | Behaviour |
|------|----------------|-----------|
| Owner or admin | Limnoria `owner` or `admin` capability | Always exempt |
| Trusted | Has the `trusted` capability | Uses the `Trusted` settings |
| Registered | Authenticated with the network | Uses the base settings |
| Unregistered | Not identified | Uses the `Unreg` settings |

To grant the `trusted` capability, see the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html).

### Setting pattern

The rate-limited commands are `ask`, `code`, `draw`, and `story`. Each has three setting pairs, one per non-exempt tier:

- `{cmd}RateLimitCount` and `{cmd}RateLimitWindow`: registered users.
- `{cmd}TrustedRateLimitCount` and `{cmd}TrustedRateLimitWindow`: trusted users.
- `{cmd}UnregRateLimitCount` and `{cmd}UnregRateLimitWindow`: unregistered users.

A count of `0` blocks nothing; it disables the limit for that command and tier. All rate-limit settings are global.

### Default limits

| Command | Registered | Trusted | Unregistered |
|---------|------------|---------|--------------|
| `ask` | 15 per 60 s | 15 per 60 s | 15 per 60 s |
| `code` | 10 per 60 s | Unlimited | 2 per 60 s |
| `draw` | 2 per 300 s | 5 per 60 s | Blocked |
| `story` | 2 per 300 s | 5 per 60 s | Blocked |

`draw` and `story` also require an authenticated account regardless of tier, so image generation is always attributable.

## Security

### Capability-based access control

Each command surface requires a Limnoria capability:

| Capability | Gates |
|------------|-------|
| `llm.ask` | `@ask` and general assistant tool access |
| `llm.code` | `@code` and the code-generation tool |
| `llm.draw` | `@draw`, `@story`, and the image-generation tool |
| `llm.verse` | Verse participation: `@verseopt`, `@verse`, `@look`, `@who` |
| `llm.verse.edit` | Canon editing: `@versedit`, `@canon`, the `verse_edit` tool |
| `llm.verse.gm` | Game-moderator (GM) operations: `@versedump`, `@versepurge`, `@versecompact` |

By default all users hold the `llm.ask`, `llm.code`, and `llm.draw` capabilities. Restrict a command by removing its capability from specific users or channels; grant the verse capabilities explicitly. See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html).

### URL validation

When users include image URLs in `@ask` prompts for vision, the bot validates them:

- **Scheme check.** Only `http://` and `https://` URLs are accepted. Schemes such as `javascript:`, `data:`, and `file:` are blocked.
- **Path traversal.** URLs containing `..` in the path are rejected.
- **Request-forgery protection.** URLs resolving to private, loopback, link-local, or reserved IP addresses are blocked, which prevents server-side request forgery (SSRF) against the bot's network. The check fails closed: if DNS resolution fails, the URL is rejected.
- **Extension check.** Only recognized image extensions are accepted: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`.

### Output sanitization

The bot sanitizes all LLM output before sending it to IRC:

- **IRC command injection.** Lines starting with a configured command prefix (default `.`) get a leading space so other bots cannot be driven by model output. Configure extra prefixes with `commandPrefixes`.
- **API key scrubbing.** Error messages are scrubbed of API keys before display.
- **HTML sanitization.** Code output rendered as HTML passes through `nh3`, an allowlist-based sanitizer, to prevent script injection.

### Prompt injection defences

- **Structured system prompts.** Prompts separate instructions from context, with a preamble telling the model to ignore instructions embedded in user-provided content.
- **Channel topics as untrusted data.** Topics arrive as user messages, not system messages, because any channel member can set them.
- **Input length limit.** User prompts are capped at `maxPromptLength` characters (default 10,000).
- **Tool-loop cap.** `metaMaxSteps` (default 12) bounds tool-call round trips per turn, so a prompt cannot drive an unbounded tool loop.

### Verse write safety

The model's in-band canon tool, `verse_edit`, is constructive-only: it can add but never delete, retire, or rewrite. Destructive operations live behind operator commands with the `llm.verse.edit` and `llm.verse.gm` capabilities, and `@versepurge` adds a two-step token confirmation. See [The verse](forest-verse.md).
