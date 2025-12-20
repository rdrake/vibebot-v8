# Limnoria LLM Plugin Code Review (updated requirements)

## Updated understanding
- Conversation context should be scoped per channel (not global across all channels) and shared across bot commands in that channel (e.g., ask → draw).
- Generated HTML pages are publicly served and must be sanitized.
- Vision should accept CDN/signed image URLs (query strings, fragments).
- Prefer Limnoria utilities where available (URL snarfer, HTML handling).

## Findings

### High
- **Stored XSS via public HTML pages**: `save_code_to_http` renders Markdown to HTML without sanitization, so a user can prompt the model to emit raw HTML/JS and get a publicly served page. This is a classic stored‑XSS vector. `plugins/llm/src/llm/service.py:725`.

### Medium
- **Context not shared with `draw`**: `draw` bypasses conversation context entirely, which conflicts with the updated requirement to share context across commands within the same channel. `plugins/llm/src/llm/plugin.py:338`.
- **`codeThreshold` config unused**: the `code` command always writes to HTTP if it can, even for short outputs. This ignores the config and can unintentionally expose short answers publicly. `plugins/llm/src/llm/config.py:134`, `plugins/llm/src/llm/plugin.py:323`.
- **CDN/signed image URLs are dropped or truncated**: `detect_images` only matches URLs ending in the extension and `validate_image_url` requires `.endswith(ext)`, so common URLs like `https://cdn.example.com/img.jpg?sig=...` won’t pass. `plugins/llm/src/llm/service.py:58`, `plugins/llm/src/llm/service.py:377`.

### Low
- **Path traversal handling in HTTP callback is string‑based**: `doGet` only checks for `".."` and `path.startswith("/")`, but does not normalize or realpath the target path. Encoded traversal or symlink escapes are not explicitly blocked. `plugins/llm/src/llm/plugin.py:48`.

## Limnoria utilities to investigate
I could not inspect Limnoria’s Python modules locally (the `supybot` package is not installed in this environment), so I couldn’t confirm exact utility names. In Limnoria core, look for utilities in `supybot.utils.web` (or nearby) that:
- Extract URLs from message text (URL snarfer used by the Web plugin) — this should help preserve query strings and strip punctuation.
- Provide HTML escaping/sanitization helpers (if any). If no sanitizer exists, you’ll likely need to escape raw HTML or integrate a sanitizer (e.g., `bleach`) before serving public HTML.

## Recommendations
- **Sanitize HTML output**: ensure Markdown output is sanitized or HTML‑escaped before storing to disk. If Limnoria has a sanitizer, use it; otherwise, add a dedicated sanitizer or switch to rendering plain text inside `<pre>` with proper escaping.
- **Share context across commands per channel**: include prior conversation history in `draw` prompts (e.g., synthesize a combined prompt from recent context), or add a config toggle to include/exclude context in image generation.
- **Use Limnoria URL snarfer + robust validation**: parse URLs using the snarfer, then validate by checking the parsed URL’s path extension (not string `.endswith`) so query strings/fragments are allowed.
- **Respect `codeThreshold`**: only save to HTTP when line count exceeds the configured threshold, otherwise reply directly in IRC.
- **Harden HTTP file serving**: resolve to real paths and ensure the target is under the configured root (defense‑in‑depth against traversal and symlink escapes).

## Noted positives
- API keys are handled carefully and sanitized in logs.
- Thread‑safe conversation context storage is cleanly implemented.
- Error handling in the service layer is consistent and user‑friendly.
