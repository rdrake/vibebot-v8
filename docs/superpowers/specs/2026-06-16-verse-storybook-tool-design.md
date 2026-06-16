# Verse Storybook Tool — Design (v2, post red-team)

**Date:** 2026-06-16
**Status:** Revised after adversarial review; one open decision (execution model) before planning.

## Summary

Give **verse mode** a model-invoked tool that turns a rich roleplay scenario into
a **multi-modal illustrated story**: the bot writes a short tale in its verse
voice, decides which moment(s) deserve illustration, draws them, and bundles
prose + images into a single **storybook-themed HTML page**, then posts the link
in channel.

It is **not** a loom feature. Tool-only trigger; the model decides when a scenario
has earned a story.

## ⚠️ What the red-team changed

Three adversarial passes (security, cost/concurrency, correctness) showed the
"this just composes three existing things" framing was wrong on multiple seams.
The must-fixes below are now part of the design, not afterthoughts:

- The HTML sanitizer **strips every `<img>`** today → images would never render.
- `image_generation()` returns **no URL field** (URL is buried in a translated
  prose string) → nothing clean to embed.
- Verse tool handlers run **inline on the driver thread holding a permit** →
  story + 3 images stalls the bot for minutes and drains the 16-permit pool.
- The verse overlay **forbids markdown/bracket output** ("plain text only, bare
  URLs") → directly contradicts "emit markdown-in-JSON."
- "Same gate as `@draw`" is **false**: the verse route gates on `llm.verse`, not
  `llm.draw`, and runs `require_account=False`.

## Trigger & gating

Tool-only: `verse_storybook` added to `make_verse_tool_specs`
(`verse/avatar.py:34`), exposed only when `verseStorybookEnabled` is true for the
channel. Because the verse entry path is **not** `@draw`'s gate, the handler must
enforce, before doing any work:

1. **Account required** — `self._require_account(irc, msg)`; bail if None.
2. **Capability** — explicit `ircdb.checkCapability(msg.prefix, "llm.draw")`
   (image spend), in addition to the route's `llm.verse`.
3. **Per-account rate limit** — reuse the existing `_rate_buckets` + `_rate_buckets_lock`
   machinery (`plugin.py:2791`), registered as a rate-limit block keyed **per
   account** (not per channel — channel keying is rotatable). Check-and-**reserve**
   atomically under the lock *before* generation, recording the timestamp at the
   **start** so an in-flight multi-minute job can't be re-triggered.
4. **Per-completion cap** — a counter in the tool executor bounds `verse_storybook`
   to **1 invocation per completion** (config `verseStorybookMaxPerTurn`, default 1);
   2nd+ call returns a tool error. `metaMaxSteps=12` is far too loose otherwise.
5. **Aggregate daily image ceiling** — before drawing, check cumulative image cost
   for the account today against `verseStorybookDailyImageCap` (default 30 images)
   using the existing `db.log_usage` records; refuse over budget. Cooldown caps
   *rate*; this caps *total*.

## Execution model — DECIDED: fire-and-return

The single biggest red-team finding: the work must **not block the driver
thread/permit**. **Chosen: fire-and-return.** The tool handler validates gates,
then **submits** the whole `generate_storybook` job to `self._llm_executor` and
returns immediately with an in-character "the tale is being illustrated — I'll
post it shortly." The worker draws sequentially under its single permit and posts
  the URL when done (mirrors the `@draw` timeout-recovery/pending path,
  `service.py:3953`). No driver-thread stall, no pool drain. Must detect worker
  context: `_llm_executor.submit` raises `RecursiveSubmitError` from a worker
  thread (`executor.py:102`), so when already on a worker, run inline-without-submit
  under a hard wall-clock budget.

Either way: **sequential image draws are required** — no side thread pool (it would
bypass `maxConcurrentLLMCalls` or self-deadlock against the held permit).

## Story generation (dedicated prompt, not the verse overlay)

The verse overlay mandates "plain text only / bare URLs," which forbids the
structured markdown output we need. So the story call uses a **dedicated storybook
system prompt** that inherits the **persona/voice** but explicitly re-enables
markdown and structured output — it does **not** go through the standard verse
overlay-layering path (`service.py:3431`). The energy/length character still comes
from the persona, not the no-markdown clauses.

Structured output requested:
```jsonc
{
  "title": "…",
  "story_markdown": "…[[illustration:1]]…[[illustration:2]]…",
  "illustrations": [ { "id": 1, "caption": "…", "image_prompt": "…" } ]   // 0..N
}
```

**Robust parsing** (the verse model is non-reasoning and unreliable at JSON):
- Prefer a provider **JSON/structured-output mode** if `verseModel` supports it;
  otherwise fall back to prompt discipline.
- Extract with **brace-matching** (first `{` … matching last `}`), not the loom's
  whole-message fence regex — tolerate leading/trailing in-character prose.
- **Partial tolerance** like loom: if `title`/`story_markdown` parse but an
  illustration entry is malformed, drop that illustration and ship.
- **≥2 retries** with a JSON-specific nudge ("emit ONLY the JSON object").

**Denial guard:** run denial detection (`_is_verse_denial`) only on the **extracted
`story_markdown`**, never on the raw JSON, and tolerate legitimate story text like
"The Day That Never Happened" (don't burn the retry budget on it). Don't apply
denial-stripping to this safety-relevant call.

## Illustration generation (no prompt laundering)

For each kept illustration (sequential), call `image_generation(image_prompt + a
storybook art-style prefix, channel=…)` **with `drawAutoRewriteMax=0`** — model-
derived prompts must not be auto-re-engineered past the provider safety filter
(that turns the tool into a prompt-laundering amplifier). Run the `image_prompt`
through the existing `validate_prompt`/moderation before generation. A failed or
blocked image **drops its marker and continues**.

**URL plumbing (BLOCKER fix):** add a `url: str | None` field to `ImageResult`,
populated on success from the same save the pipeline already does — do **not**
regex-scrape the translated `content` string.

## Page assembly

1. **Sanitize the model's `story_markdown` first:** strip any user-echoed raw
   `![...](...)` image syntax and any literal `[[illustration:N]]` tokens, so only
   **server-controlled** markers placed by our own structured field are honored.
2. **Truncate** `story_markdown` to `verseStorybookMaxChars` before embedding.
3. **Single-regex marker substitution:** one `re.sub(r"\[\[illustration:(\d+)\]\]", …)`
   pass with a callback that looks each id up in a dict (avoids the `1` vs `11`
   substring-collision of per-id `str.replace`). Duplicate-id rule: **first marker
   wins**, later duplicates stripped. Orphan illustrations (no marker) are dropped
   and **logged**.
4. Replace each surviving marker with markdown `![caption](relative-image-path)` +
   an emphasised caption line.
5. `save_markdown_to_http(markdown, title=title or "An Untitled Tale")` → storybook
   URL. Default title prevents an empty `# ` heading.

## Sanitizer & rendering changes (BLOCKER fix + XSS containment)

`_sanitize_html` (`service.py:686`) currently has no `img`. Add — **as a called-out
security decision**:
- `img` to the `tags` set; `{"src", "alt", "title"}` to `attributes["img"]`.
- Keep `url_schemes` locked to `{http, https}` for `img` (no `data:`/`mailto:`),
  and **constrain `img src` to the bot's own image host/path** (same `http_root`),
  so this can't become an SSRF/tracking-pixel vector on the *existing* answer/code
  pastebin pages. Embedding images by **relative path** (same directory as the page)
  satisfies the host-scoping and survives `localhost`/relative `httpUrlBase`.
- Add `img`/caption CSS to the storybook theme (framed plate, centered, gilt
  border, soft shadow).
- **Regression test:** a saved storybook page contains a real `<img src=…>`, and
  `javascript:`/`data:` src and `onerror` are stripped.

## SSRF hardening

`_download_and_save_image` only fetches provider URLs (model supplies prompts, not
URLs) and blocks redirects (`service.py:4475`), but `validate_external_url` does no
DNS resolution (documented, `service.py:476`). Since storybook fires up to 3
downloads per call from a lower-trust caller, **resolve the hostname and re-check
the resolved IP** against the private/reserved set before fetch, and cap total
downloads per call.

## Configuration

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `verseStorybookEnabled` | bool (per-channel) | false | Exposes the tool. |
| `verseStorybookMaxImages` | int | 3 | Hard cap on illustrations per story. |
| `verseStorybookMaxPerTurn` | int | 1 | Max `verse_storybook` calls per completion. |
| `verseStorybookCooldownSeconds` | int | 300 | Per-account rate limit (reserve at start). |
| `verseStorybookDailyImageCap` | int | 30 | Per-account daily image-spend ceiling. |
| `verseStorybookMaxChars` | int | 6000 | Story length cap (pre-embed). |
| `verseStorybookImageTimeout` | int | 45 | Per-image timeout for this tool (< draw's 120). |

Models reuse `verseModel` (prose) and `imageModel` (art).

## Error handling

| Failure | Behaviour |
|---------|-----------|
| Account/capability/cooldown/daily-cap fail | Tool returns a polite refusal the model can voice; no work. |
| 2nd `verse_storybook` call in one completion | Tool error (per-turn cap). |
| Story completion error / denial | Tool error; denial detection on extracted prose only. |
| Malformed JSON | Brace-extract + ≥2 retries; partial-tolerant; then graceful error. |
| An image fails/blocked | Drop that marker; story ships with fewer plates. |
| All images fail | Prose-only storybook page (valid). |
| `save_markdown_to_http` fails | Tool error (the page is the deliverable). |
| Cap exceeded | Keep first N, **log** dropped count. |

## Components & isolation

- `service.py` — `ImageResult.url` field; `generate_storybook(...)` core (prompt
  build, robust parse, sequential image draws, sanitize+embed, save); storybook
  system-prompt constant; `_sanitize_html` img allowlist; DNS re-check.
- `verse/avatar.py` — `verse_storybook` tool spec (config-gated).
- `plugin.py` — handler in `_build_verse_handlers_for_route`: account+capability+
  rate-limit+per-turn+daily-cap gates, then execution-model dispatch (A or B);
  worker-context detection.
- executor / assistant — per-completion invocation counter.
- `config.py` — seven new keys; register the rate-limit block.

## Testing

- Gates: no account, missing `llm.draw`, cooldown active, per-turn 2nd call, daily
  cap exceeded → all refuse without spend (mock image gen asserts not called).
- Parse: valid; prose-wrapped JSON; fenced JSON; malformed field (partial); total
  garbage → graceful.
- Marker embedding: placement; `1` vs `11`; duplicate ids; orphan markers/illos;
  marker inside a code block; caption echoing the marker token; user-injected
  `![](url)` stripped.
- Sanitizer: storybook page has real `<img src>`; `javascript:`/`data:`/`onerror`
  stripped; existing answer/code pages still have no `<img>` unless server-authored.
- Image failure: one fails → marker dropped, page saved; all fail → prose-only.
- Execution model (B): handler returns immediately; worker posts URL; worker-context
  inline path doesn't call `submit` (no `RecursiveSubmitError`).
- Overlay: story call uses the dedicated prompt, not the verse plain-text overlay.

## Deferred (not v1)

- Recording standout stories into verse `events` canon.
- Explicit `@story` user command.
- Parallel image draws (sequential required for v1).
