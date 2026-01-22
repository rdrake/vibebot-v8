# Code Review: vibebot-v8 (line-by-line)

## Scope
- plugins/llm/src/llm/plugin.py
- plugins/llm/src/llm/service.py
- plugins/llm/src/llm/context.py
- plugins/llm/src/llm/config.py
- README.md

## Findings (ordered by severity)

### High
- Channel-scoped context controls are not enforced. `contextEnabled`, `contextMaxMessages`, and `contextTimeoutMinutes` are registered as channel values but `ConversationContext` is built once from global defaults and then used for all channels; `ask`/`code` also store and read history unconditionally. This means disabling context for a channel does not prevent storage or sending of that channel's history to the LLM, which is a privacy/security regression relative to config intent. `plugins/llm/src/llm/plugin.py:350` `plugins/llm/src/llm/plugin.py:456` `plugins/llm/src/llm/plugin.py:488` `plugins/llm/src/llm/context.py:118` `plugins/llm/src/llm/context.py:148` `plugins/llm/src/llm/config.py:138`

### Medium
- Image URL detection and validation reject common CDN/signed URLs with query strings or fragments, so vision support silently drops a large class of valid images. `plugins/llm/src/llm/service.py:70` `plugins/llm/src/llm/service.py:391`
- `%code` previews can re-introduce IRC command injection risk because summaries are sent without passing through `_sanitize_output` (only the completion output is sanitized). A summary that starts with a command prefix can still trigger other bots/clients. `plugins/llm/src/llm/service.py:95` `plugins/llm/src/llm/service.py:582` `plugins/llm/src/llm/plugin.py:535`
- Docs advertise `codeThreshold`, but there is no config value and `%code` always writes to HTTP, including short outputs or error messages. This is a behavior mismatch and can expose responses publicly when users expect inline replies. `README.md:133` `plugins/llm/src/llm/plugin.py:535` `plugins/llm/src/llm/config.py:108`
- `contextTrackAllMessages` claims to track all channel messages for richer context, but implementation only adds messages to per-user context and never to shared channel context, so other users do not see the "tracked" history. `plugins/llm/src/llm/plugin.py:311` `plugins/llm/src/llm/plugin.py:347` `plugins/llm/src/llm/config.py:165` `plugins/llm/src/llm/service.py:961`

### Low
- `validate_image_url` blocks any URL containing `..` even when the normalized path is valid, which can reject legitimate URLs that use parent segments. `plugins/llm/src/llm/service.py:409`
- The comment about keeping message pairs during trimming is not implemented; the trim is by raw message count, which can split exchanges mid-pair. This is minor but can slightly reduce context coherence. `plugins/llm/src/llm/context.py:132`

## Questions / assumptions
- Should `contextEnabled` be enforced per channel on every command (ask/code/draw), or is the intention to only gate tracking of non-command messages? The current behavior does not match the config description. `plugins/llm/src/llm/config.py:138`
- Is `contextTrackAllMessages` meant to feed the shared channel context (so that other users benefit), or only the requesting user's personal context? The current implementation only does the latter. `plugins/llm/src/llm/plugin.py:347`

## Test gaps
- No tests cover channel-specific `contextEnabled` behavior or confirm that context storage is skipped when disabled. `plugins/llm/tests/test_context.py:1`
- No tests for image URLs with query strings/fragments. `plugins/llm/tests/test_service.py:1`
- No tests for `%code` preview sanitization or for the documented `codeThreshold` behavior. `plugins/llm/tests/test_service.py:1` `plugins/llm/tests/test_plugin.py:1`

## Change summary (if addressed)
- Enforce per-channel context settings at read/write points and/or refactor `ConversationContext` to accept per-channel limits.
- Accept image URLs with query strings by parsing the path extension.
- Sanitize summaries/previews the same way as completion output.
- Either implement `codeThreshold` or update the docs.
