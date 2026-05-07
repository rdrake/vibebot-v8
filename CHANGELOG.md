# Changelog

All notable changes to VibeBot v8 are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project's
own conventional-commit history (`type(scope): summary`).

## Unreleased

### Breaking

- Removed `plugins/rpg/` and all its registry keys. Existing rpg state is
  **discarded, not migrated**. See `docs/guide/operator/forest-verse.md`.
- Removed Forest mode (`plugins.LLM.forestNicks`). Existing rosters are
  discarded; users opt in fresh via `@verseopt in` in a channel where
  `verseEnabled=True`.
- Removed Spontaneous mode (`plugins.LLM.spontaneousEnabled` and friends).
  Replacement is the upcoming loom orchestrator (PR 2 of the forest-verse
  rollout); per-channel chatty-bot behaviour is no longer available in the
  interim.

### Added

- Forest-verse: per-channel SQLite entity graph + avatar shim. New commands
  `@verseopt`, `@verse`, `@look`, `@who`, plus owner commands `@versedump`,
  `@versepurge`. New capabilities `llm.verse` and `llm.verse.gm`.

### Bug Fixes

- Route nick-addressed text through assistant, not Limnoria dispatch (`llm`)
- Make tests deterministic with sync LLMExecutor stub (`llm`)
- Add _spontaneous_events_lock for worker-thread safety (`llm`)
- Add _rate_buckets_lock for concurrent rate-limit access (`llm`)
- Treat longReplyLineThreshold as a hard cap in footer mode (`replies`)
- Also read Responses-shape cached_tokens in completion_timing (`llm`)
- Use prompt_cache_key on Responses API and log Responses cached_tokens (`llm`)
- Gate pastebin on rendered wire-line count, threshold→3 (`replies`)
- Gate pastebin on logical lines only, not wrap chunks (`replies`)
- Raise longReplyLineThreshold default 1→4 (`replies`)
- Default >1-line replies to teaser+URL pastebin (`replies`)
- Bump APIError log truncation 150→1000 chars (`llm`)
- Per-chunk multiline-concat for wrap-continuations (`llm`)
- Drop multiline-concat from distinct logical lines (`llm`)
- Batch multi-line replies as one PRIVMSG to avoid Excess Flood (`llm`)
- Layer system_prompt over framework instead of replacing (`llm`)
- Tighten scheduled_llm_task cap from 50 to 5 fires (`llm`)
- Drop blank lines from multi-line IRC replies (`plugin`)
- Cap recurring scheduled_llm_task chain at 50 fires (`llm`)
- Standardize memories command on irc.error for failures (`plugin`)
- Upgrade severity, add tracebacks, sanitize error fields (`observability`)
- XAI search via Responses API web_search, not deprecated Live Search (`llm`)
- XAI live_search needs `sources` — bare type returns 422 (`llm`)
- Force assistant to invoke search_web on explicit search asks (`llm`)
- XAI search uses `live_search` tool type, not `web_search` (`llm`)
- Migrate xAI search grounding from deprecated Live Search → Agent Tools (`llm`)
- Provider-aware grounding for search/url completion (`llm`)
- Tighten schedule_llm_task descriptions to suppress tool-name leakage (`llm`)
- Collapse newlines on direct-queue PRIVMSG/ACTION paths (`llm`)
- Count blank-spaced reply chunks for linking (`llm`)
- Preserve newlines in sanitized replies (`llm`)
- Request experimental caps on CAP NEW (draft/multiline) (`llm`)
- Default grokModel to xai/grok-4.3-latest (`llm`)
- Expose generate_image in chat profile (`llm`)
- @code pastebin + multiline reply helper (`llm`)
- React + stay [silent] on chat-driven reminder ops (`llm`)
- Give action reminders a remind_action profile that includes draw + tighten parser confirmation (`llm`)
- Broaden reminder parser action_prompt detection to cover bot tool surface (`llm`)
- Address tasks 1-4 review findings (B1 parser prompt, B2 exception handler scope) (`llm`)
- Address subagent code review findings (H1, H2, M1-M6, L1) (`llm`)
- Use ircutils.toLower for RFC1459-correct nick comparison (`llm`)
- Keep IRCv3 typing indicator alive across long LLM calls
- Pass history, channel history, and memories through assistant_request
- Tell assistant profile prompts to emit plain text for IRC
- Preserve newlines in LLM output bound for HTML rendering
- Collapse real newlines in sanitize_output for single-line IRC
- Strip repr-style quoting from LLM output in sanitize_output
- Pass image URLs through to meta_completion for vision support
- Strip trailing punctuation before nick matching in NickInMiddle
- Detect image format from magic bytes instead of assuming PNG
- Strip literal \n from LLM output in sanitize_output
- Consolidate usage logging — leaf tools no longer log independently
- Resolve ACTION target correctly for PM context
- Use resolved channel for ACTION targets in PM context
- Collapse newlines in meta responses for IRC (`meta`)
- Use minor-only Python pin in .python-version
- Bump uv in Dockerfile from 0.9.18 to 0.11.1
- Catch openai.APIError instead of litellm.APIError in error handler
- Remove navigation.instant.progress to prevent layout shift on refresh
- Memory cleanup counter math and single-index merge application
- Allow single-index merge ops in memory cleanup
- Use json_object response format for memory cleanup
- Add tenacity dependency required by litellm num_retries
- Add retries to memory cleanup and surface error type
- Suppress timezone note for relative-time reminders
- Deliver PM reminders to user nick instead of bot nick
- Sanitize control chars and unbalanced brackets in inFilter (`ask`)
- Eliminate redundant DB reads and dead fields (`memory`)
- Streamline messages across all commands (`ux`)
- Remove unnecessary image processing message (`ask`)
- Use reply instead of error for usage messages (`memory`)
- Use object format for cleanup merge schema (`memory`)
- Cleanup event tracking, zero-memories guard, clarify config desc
- Track cleanup events for die() cancellation, add interval=0 test
- Detect '* BotNick ...' as IRC action in addition to '/me ...'
- Use display nick instead of account name in channel context
- Send spontaneous /me responses as proper IRC actions
- Cancel pending spontaneous events on plugin unload and simplify tests
- Tighten PASS filter and remove double-sanitization in spontaneous
- Clean up spontaneous state on plugin unload
- Suppress ResourceWarning from thread-local SQLite connections in tests
- Remove code-level suppression of consecutive /me actions
- Rewrite picard system prompt to actually be Picard
- Suppress consecutive /me actions in ask command
- Discourage consecutive /me actions in system prompt
- Add plain text formatting instructions to picard prompt
- Remove /me action handling from picard command
- Make picard system prompt more fun, suppress /me actions
- Strengthen /me action nudge in system prompt
- Harden delivery retries with attempt tracking and exhaustion cap
- Harden animate durability and error classification
- Close leaked sqlite3 connection in rate-limit integration test
- Protect extract_server_headers dict comprehension from non-iterable sources
- Handle empty LLM responses gracefully instead of cryptic Limnoria error
- Log pending video poll failures at info level, not debug
- Handle real xAI video API response format
- Switch video API from urllib to requests to bypass Cloudflare
- Add User-Agent header to video API requests to avoid Cloudflare 1010
- Replace gh run watch with polling loop in wait-ci
- Scope time mock to llm.service to fix CI on Python 3.12 (`tests`)
- Broaden SyntaxWarning filter to fix CI on Python 3.14.3
- Add SyntaxWarning filter for PEP 765, bump Python and deps
- Harden reliability and improve test coverage
- Lazily migrate old nick-based usage rows to NickServ account
- Bypass Limnoria tokenizer for usage command arguments
- Handle nicks with brackets in usage command
- Strip IRC status prefixes (@+%) from usage nick lookup
- Fallback per-image cost for xAI models when LiteLLM returns zero
- Always log draw usage even when image API returns no cost/token data
- Remove conversation history context from draw prompts
- Remove irrelevant channel context from draw command prompts
- P0/P1 code review fixes — sanitization, safety, thread safety, and DRY
- Thread safety, stale IRC ref, and DRY refactor (phase 2)
- Address code review findings (phase 1 quick wins)
- Deduplicate model validation warnings
- Make wait-ci resolve run ID automatically
- Retry completion without tools on INVALID_ARGUMENT from Gemini
- Simplify draw rewrite prompt to be faithful rather than funny
- Truncate rewritten prompt to 200 chars in draw reply
- Use %s instead of %d in log format strings
- Detect moderation blocks from BadRequestError in draw rewrite
- Harden llm edge cases and add syntax compatibility check
- Release MetaSynchronized RLock during blocking API calls
- InvalidCommand now correctly delegates to wrapped ask method
- Address code review findings across plugin and service layers
- Update litellm to 1.81.6+ for gpt-image-1.5 support
- Use ircdb to find bot owner for startup notification (#21)
- Mount entire config directory to allow bot.conf.bak writes
- Mount conf/data/logs from ~/.config/vibebot
- Correct working directory for user database
- Use calendar-based scheduling for update timer
- Handle None values in channel role checks and uptime type validation
- Mount /var/www for HTTP file output
- Add /var/www/llm volume mount for HTTP file output
- Mount config directory instead of single file
- Put bot.conf in writable conf directory
- Mount bot.conf in writable conf directory
- Add :latest tag to Docker builds and ignore log files
- Preserve existing env file during install
- Remove docker.service dependency from user service
- Handle None content in channel history formatting
- Protect LaTeX delimiters from markdown escaping
- Instruct LLM to use $$ delimiters for math equations
- Disable thinking for gemini-3 models to prevent hangs
- Set litellm.request_timeout at import time
- Set litellm.request_timeout to work around bug #14635
- Only pass optional kwargs when set
- Disable grounding tools for Gemini 3 models
- Revert to original completion code, add debug logging
- Add debug logging and handle None LLM responses
- Add debug logging and handle None LLM responses
- Make forget command channel argument optional
- Only pass reasoning_effort to Claude models
- Check grounding metadata value not just key existence
- Repair 7 failing tests in integration and markdown edge cases
- Correct grounding metadata key for LiteLLM responses
- Address code review findings for security and correctness
- Strengthen anti-injection warning against identity hijacking
- Add anti-injection instruction to system prompt
- Expand topic injection patterns to catch direct AI instructions
- Sanitize LLM output to prevent IRC command injection
- Remove context from draw command
- Exclude channel topic from prompt to prevent injection
- Handle KeyError when nick not in channel state
- Include exception type in error logs for debugging
- Reframe topic as vibe to prevent prompt injection
- Frame context naturally to reduce prompt injection
- Simplify system prompt to reduce prompt injection risk
- Add missing bleach dependency
- Prevent prompt injection via channel topics
- Handle HTTP connection errors and improve test coverage
- Upgrade openai to 2.13.0 to fix grader_inputs import bug
- Remove appuser from Dockerfile, use --user at runtime instead
- Run Docker container as current user to fix volume permissions
- Add llm workspace member as root dependency for Docker builds

### Build

- Add feedparser for Limnoria's stock RSS plugin
- Add RPG plugin to Docker build

### CI

- Ignore local uv workspace members (`dependabot`)
- Drop arm64 docker build, group dependabot updates
- Split lint from matrix tests, add docs path filter
- Bump actions/upload-pages-artifact from 3 to 5 (#51)
- Bump actions/deploy-pages from 4 to 5 (#47)
- Add GitHub Pages deployment workflow

### Chores

- Add deploy target — push, wait, restart (`make`)
- Add push-and-wait target (`make`)
- Add hypothesis as workspace dev dependency
- Diagnose xAI live_search — log + recognize citations (`llm`)
- Upgrade litellm to 1.83.14 and refresh transitive deps (`deps`)
- Drop commit-narrating comments from reminder code (`llm`)
- Demote reaction diagnostics to INFO (`llm`)
- Add diagnostic logging for reaction sends (`llm`)
- Bump metaMaxSteps default 7 -> 12 (`llm`)
- Remove abandoned v10 rewrite plans
- Upgrade Docker GitHub Actions to Node.js 24 versions
- Tighten rate limit defaults
- Remove startup notification diagnostics
- Use stderr for startup diagnostics (bypass Limnoria logger)
- Add temporary startup notification diagnostics
- Drop pre-commit in favor of prek
- Bump local Python 3.14.2 → 3.14.3
- Optimize Claude Code experience with hooks, quality gates, and streamlined config
- Add .worktrees/ to .gitignore
- Bump actions/checkout from 4 to 6 (#1) (`deps`)
- Add release changelog configuration (#19)
- Remove temporary debug logging from service.py
- Add docs/reviews/ to gitignore
- Upgrade dependencies to latest versions
- Consolidate CI to use Makefile and add PR docker validation
- Consolidate tooling and add GitHub Actions CI
- Remove Dutch locale (not supported by Limnoria)

### Dependencies

- Add python-dateutil as direct runtime dep (`llm`)
- Bump ruff from 0.15.11 to 0.15.12 in the dev-tools group (#57)
- Bump ty from 0.0.29 to 0.0.33 (#56)
- Bump prek from 0.3.8 to 0.3.11 (#55)
- Bump ruff from 0.15.9 to 0.15.11 (#52)
- Bump pytest from 9.0.2 to 9.0.3 (#50)
- Bump ty from 0.0.26 to 0.0.28
- Bump ruff from 0.15.8 to 0.15.9
- Bump ty from 0.0.15 to 0.0.16
- Bump ruff from 0.15.0 to 0.15.1
- Bump ruff from 0.14.13 to 0.14.14 (#16)
- Bump ty from 0.0.12 to 0.0.14 (#17)

### Documentation

- Document maxConcurrentLLMCalls and watch-reschedule change (`llm`)
- Revise async LLM impl plan after codex review pass (`plans`)
- Revise async LLM impl plan after third review pass (`plans`)
- Revise async LLM impl plan after second review pass (`plans`)
- Async LLM concurrency implementation plan (`plans`)
- Async LLM concurrency design (`plans`)
- Add scheduled tasks, spontaneous, memory promotion, bridge tools (`guide`)
- Mark hypothesis future-work candidates 5-11 as landed (`plans`)
- Add 2026-05-04 hypothesis property-test plan (`plans`)
- Add 2026-05-03 defensive-cleanup, DRY, persistence plans (`plans`)
- Refresh user/operator guide for current interface; vale clean
- Archive completed plans; fix two stale plan paths in limnoria_bridge
- Phase 2 Task 5b implementation plan (config cleanup)
- Link Phase 2 Task 3 plan in AGENTS.md
- Document schedule_llm_task and per-creator budget (`llm`)
- Add task 3 implementation review plan
- Document capability-based settings + migration; expand conftest defaults (`llm`)
- Document curated default plugin set for the bridge (`llm`)
- Note Phase 2 mutation gate in limnoria_bridge AGENTS entry
- Document bridgeAllowMutating gate (`llm`)
- Add Limnoria bridge to AGENTS.md important-files catalog
- Document Limnoria tool bridge configuration (`llm`)
- Track AGENTS.md as canonical agent instructions
- Document LLM-action reminders and [auto] marker (`reminders`)
- Add searchModel/searchApiKey config and code search capability
- Document natural language interaction via unified assistant
- Fix Vale warnings in Getting Started
- Rewrite Getting Started with examples and walkthroughs
- Expand homepage with features and example exchange
- Configure MkDocs Material theme
- Replace NickServ references with authenticated
- Fix Vale errors in user and operator guides
- Update CLAUDE.md for MkDocs migration
- Switch GitHub Pages to MkDocs, remove old help page
- Add command reference page
- Add operator guide pages
- Add user guide pages
- Scaffold MkDocs Material site
- Add GitHub Pages help deployment plan
- Add GitHub Pages help page design
- Update command tables and terminology for command surface overhaul
- Adopt volatile/non-volatile memory terminology across all help surfaces
- Fix review issues in implementation plans
- Add command UX overhaul design and implementation plans
- Add missing commands and features to README, CLAUDE.md, and HTML help
- Add implementation plan for memory and spontaneous features
- Design for long-term memory and spontaneous participation
- Add emote/action implementation plan
- Add emote/action response design
- Add VibeBot v10 implementation plan
- Add VibeBot v10 design document
- Add abuse mitigation implementation plan
- Add abuse mitigation design (auth gating, usage auditing, flagging)
- Add implementation plan for tracing server headers and log severity
- Add design for tracing server headers and log severity
- Add persistence layer implementation plan
- Update CONTRIBUTING.md and README.md for accuracy (#18)
- Add operations guide and install-deploy target
- Add production debugging commands to CLAUDE.md
- Add comprehensive CLAUDE.md for AI assistant guidance

### Features

- Surface executor running/queued/max in %usage output (`llm`)
- Bound command path with LLMExecutor.permit() (`llm`)
- Add _safe_queue for thread-safe worker IRC sends (`llm`)
- Wire LLMExecutor into plugin lifecycle with drain on die (`llm`)
- Register maxConcurrentLLMCalls (default 16) and wire test default (`llm`)
- Add LLMExecutor for bounded concurrent LLM I/O (`llm`)
- Set x-grok-conv-id per channel for xAI prompt cache (`llm`)
- Add search_bridge_commands tool + nudge for Limnoria meta-questions (`llm`)
- Add per-user forest mode for long-form @ask replies (`llm`)
- Default long replies to inline-with-footer pastebin link (`llm`)
- Stage facts as candidates before promoting (#59) (`llm/memory`)
- Log validation rejections from _remind_set (`plugin`)
- Add _write_txn context manager for write rollback (`persistence`)
- Unify pending-task tools; both Gemini grounding tools every call (`llm`)
- Schedule_llm_task reply_target + auto-cancel on revoke (`llm`)
- Owner-only `@remind admin` command (`llm`)
- T5b — remove command-era registry keys, %g, resolve_setting shim (`llm`)
- @migrateconfig owner command (T5b prep) (`llm`)
- Wire scheduled-task fns into assistant_request (`llm`)
- _scheduled_llm_task_fns helper for tool wiring (`llm`)
- Register bridgeScheduledTaskLimit channel value (`llm`)
- AssistantToolExecutor handlers for scheduled-task tools (`llm`)
- Tool-spec overrides for schedule_llm_task family (`llm`)
- Register schedule_llm_task tool schemas (`llm`)
- Restore scheduled_llm_tasks on plugin init (`llm`)
- List + cancel scheduled_llm_task service methods (`llm`)
- Schedule_llm_task service method (one-shot + recurring) (`llm`)
- Persistence helpers for scheduled_llm_tasks (`llm`)
- Add scheduled_llm_tasks table (schema v13) (`llm`)
- Route assistant + image lookups through resolve_setting compat shim (`llm`)
- Register assistant/image capability-based settings (`llm`)
- Add resolve_setting compat shim for capability-based config (`llm`)
- Default bridge to curated read-safe plugin set when allowlist empty (`llm`)
- Footer when bridge mutation gate hides writes (`llm`)
- Wire bridgeAllowMutating through _build_bridge_tool (`llm`)
- Defense-in-depth mutation gate in bridge dispatch (`llm`)
- Gate mutating commands in bridge enumerate_commands (`llm`)
- Register bridgeAllowMutating channel value (default False) (`llm`)
- Add MUTATING_COMMANDS classification to bridge (`llm`)
- Add bridge debug logging and optional in-channel footer (`llm`)
- Inject Limnoria bridge tool into @ask chat profile (`llm`)
- Add LLM._build_bridge_tool helper (`llm`)
- Thread extra_tools and extra_handlers through assistant loop (`llm`)
- Register bridgeEnabled and bridgeAllowedPlugins (`llm`)
- Bridge dispatch with JSON envelope and positional tokens (`llm`)
- Bridge enumerate_commands with deny lists and capability gate (`llm`)
- Add BufferingIrcProxy for bridge reply capture (`llm`)
- Scaffold Limnoria tool bridge module (`llm`)
- Link long chat replies (`llm`)
- Make all API keys per-channel overridable (`llm`)
- Give @g full chat-profile tool access (`llm`)
- Add @g escape-hatch command for direct Grok passthrough (`llm`)
- Mechanical reschedule + strict routing gate (B3.5+B4) (`llm`)
- Pipe recurrence_seconds/rrule and watch_mode through schedule path (`llm`)
- Parser returns recurrence_seconds/rrule and watch_mode as structured fields (`llm`)
- Schema v12 — drop 30-day TTL, add structured recurrence/watch (`llm`)
- Cap recurring reminder chains, react to remind ops, log action-fire usage (`llm`)
- Expose set_reminder in draw and code profiles (`llm`)
- Point chat prompt at set_reminder for delayed/repeating tasks (`llm`)
- Let recurring reminders reschedule themselves at fire time (`llm`)
- Mark LLM-action reminders in @remind list (`llm`)
- Dispatch LLM-action reminders through assistant at fire time (`llm`)
- Persist action_prompt on reminder schedule (`llm`)
- Teach reminder parser to recognize action-prompt intent (`llm`)
- Add reminders.action_prompt column for LLM-triggered reminders (`llm`)
- Return last assistant text when meta loop hits step cap (`llm`)
- Patch doJoin to drop MODE +b and conditionally drop auto-WHO; preserve startup notification path (`llm`)
- Add skipAutoWhoOnJoin config flag (default True) (`llm`)
- Plumb captured account through PendingTaskResult to delivery logging (`llm`)
- Capture account-tag onto pending_tasks at stash time (`llm`)
- Add nullable account column to pending_tasks (schema v8) (`llm`)
- Add _account_from_msg two-layer resolver (account-tag → state cache) (`llm`)
- Pass recent context to draw, gated by drawContextMaxAgeSeconds
- Add per-tool allow/deny logging to MetaToolExecutor
- Convert PNG images to JPEG when saving to disk
- Simplify invalidCommand to route through chat profile
- Convert @draw to thin wrapper over assistant facade
- Convert @code to thin wrapper over assistant facade
- Update plugin callers to handle MetaResult from assistant facade
- Rewrite assistant_request as real planner facade
- Meta_completion accepts system_prompt and accumulates leaf tool costs
- Expand MetaToolExecutor with new callables, structured returns, and tool handlers
- Add search_completion and url_completion service methods
- Add search_web, fetch_url, generate_code tool specs
- Add per-profile system prompts for chat, code, draw
- Add validate_external_url for fetch_url security
- Add foundation types for grounding leaf tools
- Add NickInMiddle plugin for mid-message bot addressing
- Route draw requests through meta in invalidCommand
- Unified assistant facade with ToolSpec access control
- Extend owner access control to instruction tools (`meta`)
- Add owner access control for memory tools (`meta`)
- Expand meta tools from 9 to 15 (`meta`)
- Add meta command for natural language configuration
- Add helpUrl config, replace dynamic URL computation
- Add build script for GitHub Pages help page
- Generate HTML help page from command registry
- Generate getPluginHelp() from command registry
- Add command metadata registry for help generation
- Add %instruct command for user-settable system prompt instructions
- Add user_instructions table (schema v7) (`persistence`)
- Extract memories from spontaneous responses (`memory`)
- Support deleting multiple memories at once (`memory`)
- Add high quality and 2k resolution for xAI image models (`draw`)
- Set 9:16 aspect ratio for xAI image models and update deps (`draw`)
- Enforce max 8 words per fact in extraction and cleanup (`memory`)
- Make cleanup manual-only with summary output (`memory`)
- Add cleanup subcommand to trigger manual memory cleanup (`memory`)
- Owner can view other users' memories, add del shorthand (`memory`)
- Trigger cleanup every N commands instead of N saves (`memory`)
- Enforce JSON schema on extraction and cleanup LLM calls (`memory`)
- Improve extraction quality and cleanup resilience (`memory`)
- Wire IRC commands to engine, combat, and narrator (`rpg`)
- Add LLM narrator with deterministic fallback (`rpg`)
- Add d20-based combat system with XP, loot, death (`rpg`)
- Add game engine — movement, inspection, rest (`rpg`)
- Add SQLite persistence for characters, inventory, world state (`rpg`)
- Add world map with room graph and path resolution (`rpg`)
- Add plugin skeleton and workspace wiring (`rpg`)
- Wire memory cleanup trigger and apply edits in plugin layer
- Add cleanup_memories method for periodic memory curation
- Add memory_cleanup_state table and counter methods
- Add memoryCleanupInterval config for periodic memory cleanup
- Reverse memory display order, add edit command, auto-remove contradictions
- Add dedicated memoryApiKey config with askApiKey fallback
- Implement spontaneous participation logic
- Add spontaneous participation configuration
- Add %memories command for user memory management
- Wire memory extraction and retrieval into command flow
- Inject user memories into system prompt
- Add memory extraction service method
- Add memory extraction configuration
- Add memory CRUD methods to LLMDatabase
- Add memories table with schema migration v4->v5
- Wire persistent context into plugin startup and commands
- Wire SQLite persistence into ConversationContext
- Add conversation persistence methods to LLMDatabase
- Add conversations table migration (schema v4)
- Show conversation context info in %usage output
- Add %picard command for random Picard facts
- Add picardSystemPrompt config value
- Add system_prompt override to completion()
- Add tiered per-command rate limiting
- Detect /me in ask responses and send as IRC action
- Add /me action nudge to system prompt
- Remove / from default commandPrefixes
- Event-driven queue wakeups with 5-minute safety poll (Phase 2)
- Add delivery acknowledgment semantics (Phase 1b)
- Replace auto-flag with per-user rate limiting
- Add auto-flag logic, owner alerts, and admin flag commands
- Add flag check gate and log all command outcomes
- Require NickServ identification for draw command
- Add flagThreshold and flagWindow config values
- Add flagged user methods and extend log_usage for audit tracking
- Add usage audit columns and flagged_users table
- Log server headers from successful completions at DEBUG level
- Log server headers from LiteLLM errors at DEBUG level
- Wire logLevel config to plugin and service loggers
- Add extract_server_headers for tracing server identity
- Add ValidatedLogLevel config type and logLevel setting
- Increase video duration to 10s and reduce poll interval to 60s
- Persist pending video requests across restarts
- Increase animate timeout to 600s, add background video recovery
- Add animate (video) command using xAI grok-imagine-video
- Switch from nick-based to account-based identity
- Add optional nick/channel argument to usage command
- Download provider-hosted images and serve locally
- Register xAI image model pricing with LiteLLM
- Dual-mode %usage command — channel stats for all, global stats for admins
- Add caller context and conversation history to draw command
- Per-channel context config, command tests, and fixture consolidation (phase 3)
- Prefix nick on draw/code replies and validate model names
- Add per-request trace IDs for log correlation
- Log reply timing for ask, code, and draw commands
- Make draw rewrite prompts fun and IRC-humored
- Add drawTimeout config (default 120s)
- Auto-rewrite draw prompts on content safety failure
- Inject build info (version + git SHA) into LLM context prompt
- Add %usage admin command for API cost reporting
- Log API usage to database after ask/code/draw commands
- Persist reminders to SQLite on create/fire/cancel
- Wire database into plugin lifecycle with reminder reload
- Extract usage data from LiteLLM responses into result objects
- Add databasePath config; fix persistence quality issues
- Add SQLite persistence module with schema for reminders and usage
- Default to ask command when bot is addressed without a command
- Add startup notification PM to bot owner (#20)
- Include bot help URL in LLM context
- Add bot uptime to LLM context for troubleshooting
- Include bot owner and channel roles in LLM context
- LLM-powered natural language reminders
- Add %remind, %reminders, and %unremind commands
- Add both gemini-3-flash and gemini-3-flash-preview to tools list
- Enable Google Search grounding for all gemini-3-flash models
- Enable Google Search grounding for gemini-3-flash-preview
- Add KaTeX math rendering to code output HTML
- Add reasoning_effort="high" to LLM completion calls
- Add grounding tool icon indicator for Google search usage
- Add AI-generated summaries for @code command
- Add web help documentation at HTTP root
- Add shared channel context for group conversations
- Harden channel topic against prompt injection
- Restore channel topic with prompt injection mitigation
- Add draw context, symlink security, and code preview
- Separate instructional and informational context in system prompt
- Add auto-update timer and coverage enforcement
- Add pre-commit hooks and contributing guide
- Add gitleaks pre-commit hook to prevent secret commits
- Improve image error messages, syntax highlighting, and add Dutch locale
- Add deep-clean target to remove venv and uv cache

### Performance

- Cap output tokens on chat / remind_action profiles (`llm`)
- Short-circuit imagine step_2 + place memories after channel history (`llm`)
- Move per-user memories out of system prompt to keep cache prefix stable (`llm`)
- Drop Bot uptime from context message to stop cache invalidation (`llm`)
- Speed up test suite from 4min to 6s

### Refactor

- Submit safety-poll via LLMExecutor with in-flight guard (`llm`)
- Submit scheduled LLM task fires via LLMExecutor (`llm`)
- Submit watch-mode reminder fires via LLMExecutor (`llm`)
- Submit memory extraction via LLMExecutor (`llm`)
- Submit spontaneous replies via LLMExecutor (`llm`)
- Replace profile-name string literals with PROFILE_* constants (`llm`)
- Dedupe markdown ban list, use ircutils.strEqual in forest check (`llm`)
- Drop msg.get('nick', '') guard; key is producer-guaranteed (`context`)
- Drop redundant defensive guards on registry value and hidden_params (`service`)
- Structured ToolCallbackResult; drop string-sniffing (`assistant`)
- Extract _register_rate_limit_block helper; pin defaults (`config`)
- Extract _get_channel_state helper (`service`)
- Extract _owner_where for scheduled-task queries (`persistence`)
- Extract _query_usage_summary helper (`persistence`)
- Extract _dispatch_assistant_reply for ask/code/draw (`plugin`)
- Extract _grounded_completion to merge search/url paths (`service`)
- Extract _channel_target helper; replace 8 inline sites (`service`)
- Drop unreachable defensiveness around COUNT and lastrowid (`persistence`)
- Drop try/finally:pass shells from reads; sweep nickserv docstrings (`persistence`)
- Make migrate_conversations atomic via _write_txn (`persistence`)
- Use _write_txn for UPDATE/DELETE/INSERT writes (`persistence`)
- Use _write_txn for INSERT writes; drop lastrowid sentinel (`persistence`)
- Remove legacy reminder reschedule path (#58) (`llm`)
- Suppress post-reminder-mutation reply via structured signal (`llm`)
- Finish Identity migration; delete _get_identity wrapper (`llm`)
- Drop unused chain_id column from reminders (`llm`)
- Merge _check_rate_limit_silent into a silent= param (`llm`)
- Scope set_reminder back to chat profile (`llm`)
- Extract _reminder_fns helper for tool-callback dict (`llm`)
- Extract _ack helper for react-with-text-fallback (`llm`)
- Use ReminderRow for in-memory reminder store (src) (`llm`)
- Split nick from account with Identity, fix reminder lookup, migrate conversations on identification (`llm`)
- Break service→plugin import cycle and dedupe stash-context extraction (`llm`)
- _get_identity reads account-tag via resolver (`llm`)
- _run_preflight reads account-tag via resolver in both branches (`llm`)
- _resolve_tier reads account-tag via resolver (`llm`)
- _require_account reads account-tag via resolver (`llm`)
- Dedupe message/memory assembly in assistant_completion
- Extract _is_stale helper and centralize 0-as-disabled in _gather_history
- Dedupe history fetch into _gather_history helper
- Rename meta → assistant in code
- Remove @meta command and NOT_META sentinel path
- Remove dynamic help page serving, now on GitHub Pages
- Make %usage use wrap() with optional text, matching %memories pattern
- Consolidate remindme/reminders/unremind into %remind
- Remove %picard command, replaced by %instruct
- Simplify extraction, wire up auto-cleanup, and improve UX (`memory`)
- Use RETURNING clause in increment_memory_saves for atomicity
- Remove flag/unflag/flagged, llmkeys, and animate/video commands
- Deduplicate context storage and usage logging
- Migrate from unittest.mock to pytest-mock (`tests`)
- Remove duplicate tests and use shared fixtures in remaining files (`tests`)
- Migrate remaining service-layer tests to shared fixtures (`tests`)
- Migrate test_service.py to shared fixtures and parametrize (`tests`)
- Expand conftest.py shared infrastructure and add pytest-mock (`tests`)
- Consolidate account resolution into _resolve_nick_to_identity
- Codebase improvements from comprehensive review
- Pythonic cleanup with security hardening
- Apply pythonic patterns for maintainability
- Unify context storage classes and fix shallow copy bug
- Move context to user messages to mitigate prompt injection
- Simplify error handling and use AtomicFile for writes
- Remove rate limiting (use Limnoria's built-in instead)

### Reverts

- Drop xai_conv_id diagnostic log
- Restore chunks-based gate on pastebin trigger
- Remove thinking disable to test if system prompt was the issue

### Tests

- Add _spontaneous_events_lock to TestDoPrivmsg fixture (`llm`)
- Stress + reload regression tests for LLM executor (`llm`)
- Add _next_rrule_fire property tests (`llm`)
- Add reminder & scheduled-task CRUD property tests (`llm`)
- Add _compute_backoff property tests in pure-helpers module (`llm`)
- Replace TestCodeFenceEdgeCases with property tests (`llm`)
- Prune example tests subsumed by Hypothesis properties (`llm`)
- Add property tests for usage ranking and aggregation (`llm`)
- Replace TestSanitizeOutput prefix examples with property tests (`llm`)
- Replace TestValidateExternalUrl with property tests (`llm`)
- Add Identity.matches equivalence properties (`llm`)
- Add JSON round-trip properties for conversation persistence (`llm`)
- Add property test for pending-task lifecycle (`llm`)
- Replace ConversationContext example tests with state machine (`llm`)
- Raise coverage floor to 93% and fill remaining gaps
- End-to-end depth-cap on fired schedule_llm_task (`llm`)
- Rewrite ~44 _reminders fixture sites for ReminderRow (`llm`)
- Verify action_prompt survives reminder reload (`llm`)
- Default server_tags to empty dict in test_integration mock_msg sites (`llm`)
- Default mock_msg.server_tags to empty dict in plugin_env (`llm`)
- Move plugin_env fixture into shared conftest (`llm`)
- Update image URL assertions to match debug remote-URL behavior
- Cover pending task delivery errors and spontaneous edge cases
- Cover HTTP callback errors, build info fallback, and delivery branches
- Cover message building, memory extraction fallback, and cleanup validation
- Cover HTTP file management and cleanup edge cases
- Cover pending task retry, stashing, and delivery edge cases
- Cover Gemini tool fallback and usage extraction errors
- Cover context building, uptime, and validation error paths
- Cover context update_config, repr, and db prune paths
- Cover persistence task filtering, usage edges, and zero-cost rank
- Fix unclosed database warnings with proper cleanup
- Add build script output verification
- Add drift-prevention test for command registry completeness
- Add plugin layer tests for coverage (`rpg`)
- Add integration tests for full game flow (`rpg`)
- Add memoryCleanupInterval default to conftest fixtures
- Add coverage for memoryApiKey non-fallback path in cleanup
- Add integration test for auto-flag abuse flow
- Add logLevel to shared test fixtures
- Add integration and concurrency tests for persistence
- Add comprehensive tests for coverage improvement

### Config

- Reduce context timeout default from 30 to 5 minutes

### Ops

- Add prefix_hash to completion_timing for cache-stability check (`llm`)
- Pre-format completion_timing log strings (fix lost args) (`llm`)
- Emit completion_timing logs at WARNING level (`llm`)
- Structured completion_timing logs around every LiteLLM call (`llm`)
- Prune dangling docker images after pull on every start (`systemd`)

### Polish

- Harden rate limiter, improve logging, expand test coverage

### Security

- Prevent API keys from appearing in exception traces

### Tune

- Drop JPEG quality from 90 to 85 for smaller file size
