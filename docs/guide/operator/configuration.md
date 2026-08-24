# Configuration

VibeBot uses Limnoria's registry for all configuration. See the [Limnoria configuration documentation](https://docs.limnoria.net/use/configuration.html) for how the registry works in general.

Set values from IRC with:

```
@config plugins.LLM.<settingName> <value>
```

View a current value by omitting the value. Channel-overridable settings also accept a per-channel form:

```
@config channel #channel plugins.LLM.<settingName> <value>
```

**Scope** in the tables below means: *channel* settings accept per-channel overrides; *global* settings apply everywhere.

!!! warning "A new default in a release does not change a running bot"

    Limnoria writes the whole registry to `bot.conf` when the bot shuts down, so every setting the bot has ever loaded exists there as an explicit line. That line wins over the default shipped in a later release.

    So when an upgrade changes a **Default** in the tables below, the new value applies to fresh installs only. An existing deployment keeps whatever was persisted, silently and with no error — the feature simply behaves as it did before.

    To adopt a new default, set it explicitly with `@config` (which writes through the registry), or stop the bot, edit `bot.conf`, and start it again. Editing `bot.conf` while the bot runs does not work: the next shutdown flush overwrites it.

    Verify by reading back the value with `@config plugins.LLM.<settingName>` — a green deploy and a quiet log do not prove the new value took effect.

## API keys

API keys are **not** part of the Limnoria registry. They come from environment variables, one per provider, read at the point each provider call is made and selected by the provider of the model being called — not by which command or channel triggered the call. The old `assistantApiKey` / `codeApiKey` / `imageApiKey` / `searchApiKey` registry settings were removed, not deprecated: `@config` on them now errors.

| Variable | Provider |
|----------|----------|
| `XAI_API_KEY` | xAI (Grok) |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |

Any other provider LiteLLM recognises — `vertex_ai`, `openrouter`, `azure`, `bedrock`, and so on — resolves to no key from this plugin, so LiteLLM falls back to that provider's own native credentials: Application Default Credentials, IAM, or its own environment variables. The default `imageModel` is `gemini/imagen-4.0-fast-generate-001`, so a fresh install needs only `GEMINI_API_KEY` for chat, `@code`, `@draw`, and search.

### Opting in to Vertex AI

Vertex AI is not the default, but it remains a supported `imageModel` (and general model) choice for anyone who wants it — for example to bill image generation to a separate GCP project. Credentials (ADC, IAM) are enough for authentication, but LiteLLM's Vertex Imagen path separately requires the project and region, and it fails closed with `vertex_project and vertex_location are required for Vertex AI` if they are missing:

| Variable | Required | Notes |
|----------|----------|-------|
| `VERTEXAI_PROJECT` | Yes | Your GCP project ID |
| `VERTEXAI_LOCATION` | Yes | Region, e.g. `us-central1`. `VERTEX_LOCATION` also works as a fallback name |

Vertex AI takes no bearer key from this plugin — no `VERTEX_AI_API_KEY` exists; it authenticates through ADC and IAM only.

### Drawing against a self-hosted endpoint

`imageApiBase` points `@draw` at any OpenAI-shaped `/v1/images/generations` endpoint — including the same box `@animate` uses, which serves images as well as video. Set it and three things change: the request carries an `api_base`, it authenticates with `ANIMATE_API_KEY` instead of the provider key `imageModel`'s name would imply, and `imageSteps`/`imageSize` ride along in the request body. Leave it empty and nothing changes; the hosted provider path is untouched.

```
@config channel #chan supybot.plugins.LLM.imageApiBase http://host:port/v1
@config channel #chan supybot.plugins.LLM.imageModel openai//path/to/model
@config channel #chan supybot.plugins.LLM.imageSteps 8
@config channel #chan supybot.plugins.LLM.imageSize 1024x576
```

The `openai/` prefix tells LiteLLM which wire format to speak; everything after it is the model name your server expects, which is why a filesystem path with slashes in it is fine.

Two things to weigh before switching a busy channel over:

- **Steps are the whole latency budget.** Measured against the reference box at 1024x576 on an idle server: 8 steps ≈ 32 seconds, 25 steps ≈ 94 seconds. Compare a hosted provider answering in a few seconds. Raise `drawTimeout` (default 120) to cover whatever you pick, or draws will fall into the stash-and-deliver-later path.
- **A busy box is much slower than an idle one.** That server runs jobs concurrently rather than queueing, so a draw submitted while clips are rendering can take well over twice as long, and every other job on the box slows down too.

Spend is recorded as `$0.00`, which is accurate for your own hardware — and unlike an unpriced hosted model, it does not log the "no price in `IMAGE_COST_PER_IMAGE`" warning, so that warning still means what it says.

### Falling back when a provider refuses

The same endpoint can sit behind a hosted provider instead of replacing it. Set `imageFallbackApiBase` and a refusal on content grounds is redrawn against your own box rather than reaching the user as an error:

```
@config channel #chan supybot.plugins.LLM.imageFallbackApiBase http://host:port/v1
@config channel #chan supybot.plugins.LLM.imageFallbackModel openai//path/to/model
@config channel #chan supybot.plugins.LLM.imageFallbackSteps 8
@config channel #chan supybot.plugins.LLM.imageFallbackSize 1024x576
```

The fallback is sent the **original prompt**, never a safety rewrite. A rewrite exists to talk a filter around, and an endpoint with no filter has nothing to be talked around — so the picture that arrives is the one that was asked for, and it carries no 🔁 marker because nothing was reworded.

This interacts with `drawAutoRewriteMax` (default `1`), and the two orderings buy different things:

| `drawAutoRewriteMax` | What a refused prompt does |
|---|---|
| `1` (default) | Rewrites once and redraws on the provider. If that is refused too, the fallback draws the original. Fast when it works, but the user gets a softened version of their request, and the rewrite costs a second billed call |
| `0` | Skips rewriting and goes straight to the fallback. Slower per refusal (a self-hosted draw is tens of seconds), free, and faithful to what was asked |

Refusals are still recorded either way. A recovered draw carries its `blocked_attempts` through to the usage table, so the refusal rate stays measurable, and the fallback logs `image_fallback_served` at WARNING so you can count how often it earns its keep.

If the fallback also fails — refused again, misconfigured, or the box is down — the user gets the original refusal message. Which of two backends disappointed them is not something they asked about.

### Video generation

`@animate` talks to a self-hosted vLLM video server rather than a LiteLLM provider, so it needs both halves of its own credential and stays off until it has them:

| Setting | Where | Notes |
|---------|-------|-------|
| `ANIMATE_API_KEY` | Environment | Bearer token for the video server. Covered by log redaction like every other `*_API_KEY` |
| `animateApiUrl` | Registry | Base URL, e.g. `http://videohost:14205` — no trailing `/v1`. Empty (the default) disables `@animate` and hides the `generate_video` tool |

With either half missing the command returns a configuration error and the tool never reaches the model, so it cannot promise a clip the bot has no way to render.

`@animate` never sends the words the user typed. A planner turn rewrites the ask into a script the video server can render, so the usage row records both: `prompt` holds the request and `rendered_prompt` holds what was submitted. Every other command leaves `rendered_prompt` empty. Rows written before this landed have it empty too — the rendered wording was only ever held in `pending_tasks`, which is cleared as each clip is delivered.

Copy `.env.example` to `~/.config/vibebot/env` and set the variables there; the systemd unit passes that file to the container with `--env-file` (see [Installation](installation.md)). A model whose provider has no key configured fails with an error naming the missing variable, for example `no API key configured for provider 'xai' (set XAI_API_KEY)`.

## Model selection

Models follow [LiteLLM's provider/model format](https://docs.litellm.ai/docs/providers). Model names are validated against LiteLLM's known providers; the bot rejects unrecognised names and suggests alternatives.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `assistantModel` | channel | `gemini/gemini-flash-latest` | All assistant text and tool work: chat, planner, memory, reminders, scheduled tasks. Needs vision support if users paste image URLs |
| `codeModel` | channel | `gemini/gemini-flash-latest` | The code-generation one-shot behind `@code` and the `generate_code` tool. The `@code` planner loop itself runs on `assistantModel` |
| `imageModel` | channel | `gemini/imagen-4.0-fast-generate-001` | Model for `@draw` |
| `imageApiBase` | channel | empty | Draw against an OpenAI-shaped endpoint of your own instead of the provider `imageModel` names — see [Drawing against a self-hosted endpoint](#drawing-against-a-self-hosted-endpoint) |
| `imageSteps` | channel | `0` | Denoising steps for a self-hosted endpoint. `0` lets the server choose. Ignored unless `imageApiBase` is set |
| `imageSize` | channel | empty | Output geometry as `WxH` for a self-hosted endpoint. Ignored unless `imageApiBase` is set |
| `imageFallbackApiBase` | channel | empty | Endpoint to redraw against when the primary provider refuses a prompt — see [Falling back when a provider refuses](#falling-back-when-a-provider-refuses) |
| `imageFallbackModel` | channel | empty | Model for `imageFallbackApiBase`. A plain string, not a validated model name |
| `imageFallbackSteps` | channel | `0` | Denoising steps for the fallback endpoint |
| `imageFallbackSize` | channel | empty | Output geometry as `WxH` for the fallback endpoint |
| `searchModel` | channel | empty | Model for web search and URL fetch. Falls back to `assistantModel` |
| `verseModel` | channel | empty | Model for verse-mode replies. Falls back to `assistantModel`. Set this when the assistant model is a terse reasoning model that writes poor prose |
| `verseCompactionModel` | global | `gemini/gemini-flash-lite-latest` | Cheap model for the daily verse compaction job. Unlike the channel model keys this one is a plain string, so a misspelled model name is accepted at `@config` time and only fails when the nightly job runs |

Channel overrides let different channels run different models:

```
@config channel #dev plugins.LLM.assistantModel anthropic/claude-sonnet-4-20250514
@config channel #casual plugins.LLM.assistantModel gemini/gemini-flash-latest
```

## System prompts

System prompts define the bot's personality and constraints. Both are channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantSystemPrompt` | Personality and constraints for all assistant work. Verse mode layers scene context on top of this prompt rather than replacing it |
| `codeSystemPrompt` | Instructions for `@code` output format |

Both defaults carry constraints worth keeping: `assistantSystemPrompt` holds replies to one to three lines of plain text with no markdown, and `codeSystemPrompt` keeps the code body in the tool's URL rather than in the channel. Read the shipped text before you replace it — `@config default plugins.LLM.assistantSystemPrompt` — because a replacement drops those constraints with it.

Example: give a channel a specialised personality:

```
@config channel #python plugins.LLM.assistantSystemPrompt You are a Python expert. Answer with Python examples. Be concise.
```

## Verse

The verse is the per-channel world model. See [The verse](verse.md) for concepts and operation; this table is the key reference.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseEnabled` | channel | `False` | Main switch. When off, verse commands are unavailable and no events are recorded |
| `verseTriggerRegex` | channel | `\bverse\b` | Case-insensitive pattern that marks a message as a canon reference, alongside entity mentions. A match grounds the reply in canon and, for an avatar holder, turns it into a prose tale. It does not enter roleplay mode; that needs `@rp`. Empty disables the keyword signal |
| `verseRoleplayStickyTtlSeconds` | channel | `900` | Sliding inactivity timeout for sticky `@rp on` roleplay. Each in-character turn refreshes it. `0` never expires |
| `verseChatRecordEnabled` | channel | `False` | Let an opted-in avatar's ordinary chat turn call `verse_record`. Off keeps canon growing only during roleplay |
| `verseStoryAmbientMaxImages` | channel | `1` | Vestigial. Since ambient canon mentions became inline prose, only an explicit illustrate cue reaches the storybook, and that path reads `verseStorybookMaxImages` instead. `0` does not give text only — the read falls back to `1` |
| `verseEventRetentionDays` | channel | `30` | Events older than this are eligible for compaction. `0` disables compaction for the channel |
| `verseAutoEntityRetireDays` | channel | `14` | Days without a mention before auto-created non-player characters (NPCs) retire. `0` disables the sweep. Pinned entities are exempt |
| `verseAutoEntityMaxNamesPerCall` | channel | `8` | Cap on the `actors` array in one `verse_record` call |
| `verseRosterMaxChars` | channel | `4000` | Character cap on the canon-roster block in the verse system prompt |
| `verseStyleExemplars` | channel | `[]` | Curated style exemplars injected into the verse prompt. Populate offline with the taste miner |
| `verseReactionCaptureEnabled` | channel | `True` | Capture IRCv3 emoji reactions to verse lines as an offline approval signal |
| `verseCompactionDailyAt` | global | `03:00` | Local `HH:MM` time for the daily compaction sweep. Empty falls back to `03:00`. A malformed value does not — it defers the next run by an hour, each time, and logs a warning |
| `verseCompactionMinKeepEvents` | global | `20` | Verses with fewer total events are skipped by compaction |

### Storybook

These caps cover both illustrated-page paths: the `verse_storybook` tool on verse turns, and the standalone `@story` command. `@story` needs no verse — it is gated like `@draw`, on `llm.draw` plus an authenticated account — so these limits still bind in a channel where `verseEnabled` is `False`. The `verse` prefix is historical; renaming the keys would orphan existing `bot.conf` values.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseStorybookEnabled` | channel | `False` | Expose the `verse_storybook` tool on verse turns. It does **not** gate `@story` |
| `verseStorybookMaxImages` | channel | `5` | Cap on illustrations per storybook |
| `verseStorybookMaxPerTurn` | channel | `1` | Cap on storybooks per verse turn. Tool-only; `@story` does not read it |
| `verseStorybookCooldownSeconds` | channel | `300` | Per-account cooldown between storybooks |
| `verseStorybookDailyImageCap` | channel | `30` | Registered but **not yet enforced** — there is no per-account daily image count to check it against |
| `verseStorybookMaxChars` | channel | `6000` | Character cap on the storybook body |
| `verseStorybookImageTimeout` | channel | `45` | Seconds to wait for each illustration |

Turning `verseStorybookEnabled` off does not stop image costs on its own, because `@story` still runs. What bounds them today is `verseStorybookMaxImages` per page and the per-account cooldown between pages, not the daily cap.

## Memory

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `memoryEnabled` | channel | `True` | Enable automatic memory extraction |
| `memoryMaxPerUser` | global | `50` | Cap on durable memories per user |
| `memoryCleanupInterval` | global | `3` | Extraction passes between cleanup runs. `0` disables |
| `memoryPromotionThreshold` | global | `2` | Mentions a candidate fact needs before promotion. `1` restores single-stage saving |
| `memoryCandidateTTLDays` | global | `14` | Days an unreinforced candidate survives. `0` disables pruning |

See [Memory promotion](memory-promotion.md) for how the two-stage pipeline works.

## Conversation context

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `contextEnabled` | channel | `True` | Enable conversation context |
| `contextMaxMessages` | channel | `20` | Messages kept in per-user history |
| `contextTimeoutMinutes` | channel | `5` | Clear context after this much inactivity |
| `contextTrackAllMessages` | channel | `False` | Track every channel message, not just bot interactions. Off by default for privacy |
| `channelContextMaxMessages` | channel | `10` | Messages in the shared channel context |

## Rate limiting

Rate limits follow the pattern `{command}{tier}RateLimitCount` and `{command}{tier}RateLimitWindow` for the commands `ask`, `code`, `draw`, `story`, and `animate`. All rate-limit settings are global. See [Rate limiting and security](rate-limiting-security.md) for tiers, defaults, and the observe-only switch `enforceRateLimits`.

## Web and HTTP output

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `httpRoot` | global | empty | Filesystem path for output files. Empty keeps them under Limnoria's `data/web/llm/` and serves them from its built-in HTTP server. Setting it **unhooks that server** — an external server such as nginx must serve the path, and `httpUrlBase` must be set to match, or every generated link 404s |
| `httpUrlBase` | global | empty | Public base URL for output files. Empty uses Limnoria's `publicUrl` plus `/llm`; if `publicUrl` is also unset the bot falls back to `http://localhost:<http port>/llm`, which nobody off the host can open |
| `imageUploadUrl` | global | empty | External image host for generated images. Empty stores them locally |
| `helpUrl` | global | `https://rdrake.github.io/vibebot-v8/` | URL shown in help output |
| `longReplyTeaserMaxChars` | channel | `220` | Characters in the one-line teaser for long replies saved as HTML |
| `fileCleanupAge` | global | `720` | Delete output files older than this many hours (30 days) |
| `fileCleanupMax` | global | `1000` | Cap on files kept in the output directory |

### Offloading images to an external host

Set `imageUploadUrl` to an uploader that takes a multipart `POST` on the `images[]` field and answers with JSON:

```json
{"results": [{"success": true, "filePath": "/img/img_abc123.png"}]}
```

Generated images then live on that host instead of under `httpRoot`, and storybook pages embed the absolute URL. Pages themselves (answers, code, stories) still come from your own HTTP server, so this offloads bandwidth rather than replacing it.

Every failure falls back to local storage: endpoint unreachable, upload rejected, image over 10 MB, or a reply naming a URL that is not an image on the configured host. An outage costs a slower draw, nothing more.

[paste.boxlabs.uk/img/](https://paste.boxlabs.uk/img/) is Eck's uploader, the host this contract follows, and it needs no key. Ask the host's owner before pointing the bot at somebody else's uploader.

## Limnoria tool bridge

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `bridgeEnabled` | channel | `False` | Expose loaded Limnoria plugin commands to the LLM as a tool |
| `bridgeAllowedPlugins` | channel | empty | Plugin allowlist. Empty uses the curated read-safe default set |
| `bridgeAllowMutating` | channel | `False` | Expose state-changing commands through the bridge |
| `bridgeScheduledTaskLimit` | channel | `5` | Active scheduled LLM tasks per creator per channel. `0` disables scheduling |
| `bridgeDebugInChannel` | channel | `False` | Append a bridge-call debug footer to replies |

See [Tuning and monitoring](tuning-monitoring.md) for the bridge's security model and the scheduled-task tool.

## Status pages

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `statusPageUrls` | global | `Claude=https://status.claude.com GitHub=https://www.githubstatus.com OpenAI=https://status.openai.com` | Space-separated status pages to poll and announce — both Atlassian Statuspage and incident.io pages work — each written as `Name=url` or as a bare url, which takes its host as its name. Names are 1-32 chars of `[A-Za-z0-9._-]`, case-insensitively unique across this key and `statusQueryablePages`, and are what the model uses to ask about one service. Unusable entries are dropped with a warning and the rest survive; at most 5 are polled |
| `statusQueryablePages` | global | empty | Same `Name=url` grammar as `statusPageUrls`, for pages the bot can be **asked** about but never polls or announces — the long tail like Cloudflare or AWS, where polling forever to answer an occasional question is the wrong trade. Fetched lazily on request and cached for 5 minutes, with a failure backoff. An entry naming the same page as a `statusPageUrls` entry is dropped, whatever its name — that page is already reachable, polled, as the fresher entry. At most 20 |
| `statusAnnounce` | channel | `False` | Announce incidents from every configured `statusPageUrls` page in this channel as they open and again as they resolve. `statusQueryablePages` pages never announce, no matter how this is set |

`check_service_status` is on the model's tool surface whenever either key holds at least one page. An empty `statusPageUrls` no longer removes the tool by itself — only polling and announcing stop; a non-empty `statusQueryablePages` keeps the tool available for on-request lookups.

See [Tuning and monitoring](tuning-monitoring.md#status-pages) for the polling rotation, staleness, and the RSS cutover order.

## Reminders and scheduled tasks

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `pendingTasksEnabled` | channel | `False` | Advertise `set_reminder` and `schedule_llm_task` to the model, so natural-language scheduling works. Off by default: the schemas and their rules cost roughly 1,100 prompt tokens on every completion in the channel |

The `@remind` command and the firing of reminders and tasks already created work regardless. This key only controls whether the model can *create* reminders and scheduled tasks from plain language. Listing and cancelling are not on the chat tool surface at any setting: users reach both their reminders and their scheduled tasks with `@remind list`, `del` and `clear`.

## Timeouts and retries

When an API call times out, the bot retries in the background for a bounded window.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `timeout` | global | `30` | Timeout for LLM API calls, in seconds |
| `drawTimeout` | global | `120` | Timeout for image generation. `0` uses the global timeout |
| `askExpiry` | global | `60` | Seconds to keep retrying a timed-out `@ask`. `0` disables |
| `codeExpiry` | global | `60` | Seconds to keep retrying a timed-out `@code`. `0` disables |
| `drawExpiry` | global | `60` | Seconds to keep retrying a timed-out `@draw`. `0` disables |

## Draw behaviour

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `drawAutoRewriteMax` | channel | `1` | Automatic prompt rewordings when a safety filter blocks a request. `0` disables. Each rewording costs a second billed image call on top of the refused one, and every recovery observed in prod landed on the first attempt, so raising this mostly buys repeat refusals of a prompt that was never going to pass |
| `drawContextMaxAgeSeconds` | channel | `60` | Pass conversation context to draw requests only when the last activity is this recent. `0` always starts fresh |

## Animate behaviour

`@animate` renders on a self-hosted video box (see [Video generation](#video-generation) for the credential). Generation is asynchronous: the command submits a job and returns, and the pending-task poller publishes the clip when it lands, so a bot restart mid-render does not lose the video.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `animateSteps` | channel | `25` | Denoising steps, and the dominant cost knob — latency is roughly linear in it. Measured on the reference box at 1280x704: a four-second clip took 68s at 25 steps and 171s at 50; at 25 steps a seven-second clip took 135s |
| `animateSize` | channel | `1280x704` | Output resolution. Must be a geometry the loaded model supports |
| `animateDuration` | channel | `7` | Clip length in seconds. The whole clip is exclusive GPU time, so raising it slows every queued request behind it. Also watch the file size: a 7s clip measured 8.87 MB, leaving about 1 MB under the uploader's 10 MB ceiling, and a clip over it falls back to local storage instead of `imageUploadUrl` |
| `animateAudio` | channel | `True` | Generate a soundtrack with the video. `False` requests silent video |
| `animateModel` | channel | empty | Model to request. Empty sends no model field, letting a single-model box pick its own — usually right |
| `animateFlowShift` | global | `12` | Flow-matching shift for the video sampler |
| `animateAudioFlowShift` | global | `3` | Flow-matching shift for the audio track. Ignored when `animateAudio` is off |
| `animateTimeout` | global | `60` | Timeout for individual HTTP calls to the video server. **Not** the generation budget — that is `animateExpiry` |
| `animateExpiry` | global | `1800` | Seconds to keep polling a submitted job before reporting it expired. Generous on purpose: a queued job can wait a long time before it starts |

Clips are published to `imageUploadUrl` when one is configured — the reference host accepts MP4 on the same `images[]` field and files it as `vid_*.mp4` — and fall back to local storage otherwise, including when a clip exceeds the 10 MB upload ceiling.

## Other settings

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `maxPromptLength` | global | `10000` | Longest accepted user prompt, in characters |
| `commandPrefixes` | global | `. @` | Space-separated prefixes sanitised in output to prevent IRC command injection. The default covers both the conventional `.` and VibeBot's own `@` |
| `databasePath` | global | empty | Path to the SQLite database. Empty uses `data/LLM.db` |
| `logLevel` | global | `WARNING` | Plugin log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `metaMaxSteps` | global | `12` | Cap on tool-call round trips per assistant turn. Stops runaway tool loops |
| `maxConcurrentLLMCalls` | global | `16` | Cap on simultaneous outbound LLM calls across all surfaces |
| `skipAutoWhoOnJoin` | global | `True` | Skip Limnoria's automatic `WHO` on join when the `account-tag` and `extended-join` capabilities are active |
| `enforceRateLimits` | global | `True` | Enforce rate limits. `False` tracks and logs without blocking |
