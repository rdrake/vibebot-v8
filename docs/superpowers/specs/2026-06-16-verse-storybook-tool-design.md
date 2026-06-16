# Verse Storybook Tool — Design

**Date:** 2026-06-16
**Status:** Approved (design); pending implementation plan

## Summary

Give **verse mode** a new, model-invoked tool that turns a rich roleplay
scenario into a **multi-modal illustrated story**: the bot writes a short tale
in its verse voice, decides which moment(s) deserve illustration, draws them,
and bundles prose + images into a single **storybook-themed HTML page**. The
bot posts an in-character line plus the page URL in channel.

This composes three pieces that already exist:
- the **verse voice** (verseModel + channel overlay + verse tool mechanism),
- the **image pipeline** (`image_generation()` with its safety-rewrite loop and
  cost tracking),
- the **storybook HTML page** (`save_markdown_to_http`, the storybook theme, and
  the newly-added useful `<title>`).

It is **not** a loom feature — no loom cycles, no loom channel, no background
posting. It fires only when the verse model chooses to call the tool during
normal play.

## Goals / Non-goals

**Goals**
- One new verse tool (`verse_storybook`) the model calls at its discretion.
- Story seed = the current verse conversation; works **either** standalone
  (invented cast) or grounded in the channel's verse cast — the model's choice.
- The model decides how many illustrations the story needs (0..N) and **where**
  they sit in the narrative.
- Output is a single storybook HTML page (prose + inline illustrations).
- Reuse existing voice/image/page machinery; keep new surface small.

**Non-goals**
- No loom integration or auto-posting.
- No explicit `@story` user command in v1 (tool-only trigger). Can be added later.
- No persistence of generated stories into the verse `events` canon (v1 is
  ephemeral output, like `@draw`).

## Trigger

Tool-only. `verse_storybook` is added to the verse tool specs alongside the
existing `verse_act / verse_move / verse_look / verse_recall / verse_record`
(`plugins/llm/src/llm/verse/avatar.py:34`, `make_verse_tool_specs`). The tool is
exposed only when `verseStorybookEnabled` is true for the channel. Verse tools
are already passed through `extra_tools` → `assistant_completion`
(`plugin.py:3426/3433/3607`, `service.py:3525-3527/3565`) and dispatched via the
verse handler map (`plugin.py:3568 _build_verse_handlers_for_route`,
`3574 make_verse_denial_handlers`). The new tool plugs into those same seams.

## Tool interface

The model calls `verse_storybook` with a brief, e.g.:
```
{ "brief": "Spin the last few turns into a short tale", "hint": "optional theme/style" }
```
The brief is intentionally thin — the real content comes from the verse
conversation already in context.

## Handler pipeline

A thin verse handler delegates to a new service method,
`LLMService.generate_storybook(brief, *, channel, conversation) -> StorybookResult`,
so the plugin stays thin and the logic is unit-testable in isolation.

1. **Write the story (one verseModel call, structured output).**
   System prompt = the short storybook framing layered on the channel's verse
   overlay (per the "verse must inherit channel overlay" rule). The model returns:
   ```jsonc
   {
     "title": "…",                          // page <title> + # heading
     "story_markdown": "…[[illustration:1]]…[[illustration:2]]…",
     "illustrations": [
       { "id": 1, "caption": "…", "image_prompt": "…" },
       { "id": 2, "caption": "…", "image_prompt": "…" }
     ]
   }
   ```
   Parsing/validation mirrors the loom proposal JSON handling. The verse denial
   guard / history de-poisoning already in place applies to this call.

2. **Enforce the image cap.** Honor at most `verseStorybookMaxImages` (default 3)
   illustrations. If the model requested more, keep the first N **and `log()` how
   many were dropped** (no silent truncation). Strip markers whose illustration
   was dropped.

3. **Draw each kept illustration.** For each, call `image_generation(image_prompt
   + storybook art-style prefix, channel=…)` — the existing pipeline (safety
   rewrite, cost tracking, `imageModel`). Bounded concurrency (≤ cap) via a small
   thread pool is acceptable; sequential is acceptable for v1. A failed or
   safety-blocked image **drops its marker and continues** — the story still ships.

4. **Assemble the page.** Build markdown:
   `# {title}` + `story_markdown` with each surviving `[[illustration:N]]` marker
   replaced by `![{caption}](…image url…)` immediately followed by an emphasised
   caption line. Orphan markers (no illustration) and orphan illustrations (no
   marker) are handled deterministically: unknown markers removed; illustrations
   with no marker are dropped (logged). Image URLs come from the existing image
   host (`save_image_to_http` / `_download_and_save_image`).

5. **Save + return.** `save_markdown_to_http(markdown, title=title)` → storybook
   URL. Return `StorybookResult(url, title, image_count, dropped)` to the handler.
   The tool result hands the URL + title back to the verse model, which announces
   it in-character; the in-channel line therefore reads naturally and a
   URL-title-echo bot surfaces the story's name (reusing the title work shipped
   2026-06-16).

## Page rendering changes

The storybook HTML theme currently has no `img` styling. Add an `img` rule
(max-width 100%, centered, rounded, gilt border + soft shadow) and a caption
style, so illustrations read as framed plates in the book. Model-supplied
captions flow through the existing `_sanitize_html` body sanitizer; the title is
already HTML-escaped.

## Configuration

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `verseStorybookEnabled` | bool (per-channel) | false | Exposes the tool. |
| `verseStorybookMaxImages` | int | 3 | Hard cap on illustrations per story. |
| `verseStorybookCooldownSeconds` | int | 300 | Per-user/channel rate limit. |
| `verseStorybookMaxChars` | int | 6000 | Story length cap. |

Models reuse existing keys: `verseModel` for prose, `imageModel` for art.

## Guardrails

- **Account required**, same capability gate as `@draw` (it spends image money).
- **Cooldown** per user+channel (`verseStorybookCooldownSeconds`); on cooldown the
  tool returns a polite refusal the model can voice.
- **Image cap** with drop logging (above).
- **Length cap** on the story.
- **Config toggle** off by default; tool absent when disabled.

## Error handling

| Failure | Behaviour |
|---------|-----------|
| Story completion error / denial | Tool returns error; verse denial guard applies; no page. |
| Malformed JSON from model | One retry, then graceful tool error. |
| An image fails / is blocked | Drop that marker, keep going; story ships with fewer plates. |
| All images fail | Ship a prose-only storybook page (still valid). |
| `save_markdown_to_http` fails | Tool returns error (the page is the deliverable). |
| Cap exceeded | Keep first N, log dropped count. |

## Components & isolation

- `verse/avatar.py` — add `verse_storybook` spec to `make_verse_tool_specs`
  (gated by config at the call site). *What:* declares the tool. *Depends on:* nothing new.
- `plugin.py` — register the handler in `_build_verse_handlers_for_route`; gate on
  `verseStorybookEnabled`; enforce cooldown/account; call the service method; map
  result → `ToolResult`. *Thin glue.*
- `service.py` — `generate_storybook(...)`: prompt build, structured parse, image
  fan-out, marker embedding, page save. *The testable core.* Add storybook prompt
  constant (short). *Depends on:* `image_generation`, `save_markdown_to_http`.
- `service.py` (storybook HTML template) — add `img`/caption CSS.
- `config.py` — four new registry keys.

## Testing

- Structured-story **parse/validation**: valid, malformed (retry), missing fields.
- **Marker embedding**: correct placement; orphan markers; orphan illustrations;
  duplicate ids.
- **Cap enforcement** + drop logging.
- **Image-failure resilience**: one image fails → marker dropped, page still saved;
  all fail → prose-only page.
- **Cooldown** gating and **config toggle** (tool absent when disabled).
- **Page assembly**: markdown contains `# title`, image tags, captions; title
  passed to `save_markdown_to_http`.
- Mock `image_generation` and the completion call; assert no real network.

## Open questions (deferred, not blocking)

- Should standout stories optionally be recorded into verse `events` canon later?
  (v1: no.)
- Explicit `@story` command for users who want to force one? (v1: no, tool-only.)
