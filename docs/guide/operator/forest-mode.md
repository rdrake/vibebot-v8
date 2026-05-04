# Forest Mode

Forest mode is a per-user opt-in that lets selected nicks bypass the
default 3-line reply cap on `@ask` so the bot can return long-form
prose, storytelling, or rants. It also bypasses the channel's
`assistantSystemPrompt` for those nicks and uses their personal
`@instruct` text as the sole personality overlay.

Plain text and markdown rules still apply, the tool surface stays
the same, and rate limits still apply.

## When to use it

The default `@ask` reply is capped at three lines because IRC
channels are shared spaces and most asks are factual. Forest mode is
the exception path for users who actually want monologues — channel
regulars who treat the bot as a creative collaborator, the storytime
crowd, the "tell me about X for ten paragraphs" crowd.

It is per-channel and per-nick. A nick opted in for `#afternet` is
not opted in for `#dev`.

## Configuration

| Setting | Type | Default | Scope |
|---------|------|---------|-------|
| `forestNicks` | space-separated list | empty | per-channel |

Match is case-insensitive against the account-resolved identity. If
a user reliably identifies, use their account name; otherwise the
bare nick is fine.

### Opt a user in

In a private message to the bot:

```
@config channel #afternet plugins.LLM.forestNicks fc42
@flush
```

### Opt several users in

```
@config channel #afternet plugins.LLM.forestNicks fc42 alice bob
```

### Opt a user out

Remove their nick from the list, or clear it entirely:

```
@config channel #afternet plugins.LLM.forestNicks ""
@flush
```

`@flush` is optional; Limnoria writes the registry to disk on a
graceful shutdown anyway. Run it if you want the change to survive an
ungraceful restart immediately.

## What changes for a forest user

When a forest-listed nick runs `@ask` in the configured channel:

1. The route profile flips from `chat` to `forest`. The structural
   framework swaps `CHAT_SYSTEM_PROMPT` (with its 3-line cap) for
   `FOREST_SYSTEM_PROMPT` (no cap, plain text still required).
2. The personality overlay is the user's own `@instruct` text. The
   channel-level `assistantSystemPrompt` is bypassed for them. If
   they have not set an `@instruct`, the request runs with no
   personality overlay at all.
3. Long replies still go through the same display path: when a reply
   is longer than `longReplyLineThreshold`, it is saved as an HTML
   page and the URL is appended in the footer (or a teaser is sent in
   place of the body, depending on `longReplyLinkMode`).

## What stays the same

- All chat-profile tools remain available: `search_web`, `fetch_url`,
  `generate_image`, `generate_code`, `set_reminder`, scheduled tasks,
  memory tools.
- The capability gates are unchanged. `llm.ask` is still required to
  run `@ask` at all; `llm.draw` is still required for inline image
  generation; `llm.code` for inline code generation.
- Rate limits are unchanged. Forest mode is permissive on length, not
  on volume.
- IRC injection defenses, HTML sanitization, and command-prefix
  protection are unchanged.

## Letting forest users set their own persona

The `@instruct` command lets each user set a persistent personality
that is layered on top of the chat framework. For forest users, the
`@instruct` text becomes the *only* personality overlay (the channel
persona is bypassed for them). They can set it themselves:

```
@instruct You are a salty raconteur. Tell long stories on demand.
```

To clear:

```
@instruct clear
```

To inspect:

```
@instruct
```

If you want to lock down the persona — for example, to keep a
forest user on a specific voice you've tuned — use the bot owner's
`@instruct nick=...` form (capability-gated) to set it on their
behalf.

## How it interacts with other settings

| Setting | Interaction |
|---------|-------------|
| `assistantSystemPrompt` | Bypassed for forest nicks. Non-forest nicks still receive it as the channel persona overlay. |
| `assistantModel` | Same model for forest and chat. If you want forest users on a longer-context model, set the channel-level `assistantModel` to one that supports it; the model is shared. |
| `longReplyLineThreshold` | Still controls when long replies are saved to HTML and surfaced via a link. Forest users will hit the threshold often, so the HTML link path becomes the common case for them. |
| `longReplyLinkMode` | `footer` (default) sends the full body inline plus a footer link; `teaser` replaces the body with a one-liner plus link. Both work in forest mode. |
| Rate limits (`askRateLimitCount` etc.) | Unchanged. Forest users count against the same buckets. |

## Operational notes

- `@config` writes are kept in memory until shutdown or `@flush`. If
  the bot crashes between an opt-in and the next clean shutdown, the
  setting is lost. Run `@flush` after opt-in changes you care about.
- Long replies cost more tokens. Forest mode multiplies the cost for
  opted-in users. Track per-user usage with `@usage` (the running
  totals are unchanged in shape, just larger).
- Token-cap errors are more likely in forest mode because long-form
  replies push closer to the model's context window. Consider raising
  `assistantModel` to a longer-context tier if you see truncations.

## Reverting

Removing a nick from `forestNicks` returns them to the default chat
profile on the next request. No per-message cache needs clearing.
