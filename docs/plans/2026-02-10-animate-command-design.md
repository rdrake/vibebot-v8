# Animate Command Design

## Overview

New `animate` command (alias `video`) that generates short videos from text
prompts using xAI's `grok-imagine-video` model. Follows the `draw` command
pattern: accept a prompt, call the API, save the result, return a URL.

## API Integration

The xAI video API uses a two-step async flow (unlike image generation):

1. **Submit**: `POST https://api.x.ai/v1/videos/generations`
   - Body: `{"model": "grok-imagine-video", "prompt": "..."}`
   - Returns: `{"request_id": "..."}`
2. **Poll**: `GET https://api.x.ai/v1/videos/{request_id}`
   - Status values: `pending`, `done`, `expired`
   - When `done`: response includes a temporary video URL
3. **Download**: Fetch the temporary video URL, save locally, return our URL.

Authentication uses `Authorization: Bearer <api_key>` header.
Implementation uses `urllib.request` (no new dependencies).

Polling sleeps 3 seconds between checks, sends IRC typing indicators on each
poll, and times out after `animateTimeout` seconds (default 300).

## Auth: NickServ Identification

Unlike ask/code/draw (which use Limnoria capability checks), animate requires
NickServ identification via IRCv3 `account-notify`. The bot checks
`irc.state.nickToAccount()` and rejects unidentified users.

## Configuration

| Key | Type | Default | Scope |
|-----|------|---------|-------|
| `animateApiKey` | String (private) | `""` | Global |
| `animateModel` | String | `"grok-imagine-video"` | Channel |
| `animateTimeout` | NonNegativeInteger | `300` | Global |

No system prompt (video generation is prompt-only).
No auto-rewrite (keep it simple for v1).

## Cost Tracking

Fallback per-video cost dict: `{"grok-imagine-video": 0.10}`.

## Files Changed

- `config.py` -- three new config entries
- `service.py` -- `VideoResult`, `video_generation()`, `_save_video_bytes()`,
  `_download_and_save_video()`, cleanup and sanitize updates
- `plugin.py` -- `animate` command, `video` alias, `llmkeys` and help updates
- `conftest.py` -- animate keys in base test fixture
- `test_service.py` -- sanitize fixture updates
- `test_commands.py` -- llmkeys assertion updates
- `test_animate.py` -- new test file
