# Memory promotion

The bot's memory system has two stages. New facts arrive as candidates and become durable memories only after the extractor sees them mentioned again. This filters out one-off statements, model hallucinations, and noisy small talk that would otherwise pile up in the memory store.

Operators control how many reinforcements a candidate needs before promotion, how long unreinforced candidates survive, and the per-user ceiling on durable memories.

## Why two stages

Single-stage extraction promotes every extracted fact directly to memory. That makes the bot remember "I am tired today" and "my actual job is mechanical engineering" with equal weight. Two-stage extraction holds the first kind in a candidate table where it expires unreferenced, while the second kind gets reinforced on the next mention and promoted.

The threshold is the main knob between a noisy memory store and one that reflects what users keep saying about themselves. Set it low for trusted users on private channels, higher for large public channels where unmoderated extraction builds up noise.

## Configuration

| Setting | Type | Default | Scope |
|---------|------|---------|-------|
| `memoryEnabled` | boolean | `True` | channel |
| `memoryPromotionThreshold` | positive integer | `2` | global |
| `memoryCandidateTTLDays` | non-negative integer | `14` | global |
| `memoryMaxPerUser` | positive integer | `50` | global |
| `memoryCleanupInterval` | non-negative integer | `3` | global |

### Threshold

`memoryPromotionThreshold` sets how many mentions a candidate needs before promotion. The extractor inserts new facts at one mention. A second mention reinforces the row; with the default threshold of `2`, that promotes the candidate to a durable memory.

```
@config plugins.LLM.memoryPromotionThreshold 3
```

A threshold of `1` collapses the system to single-stage and saves every extracted fact immediately. Use this when migrating from older deployments or when the candidate stage feels too cautious for a small group.

### Candidate TTL

`memoryCandidateTTLDays` controls how long unreinforced candidates sit in the candidate table before the cleanup pass deletes them. TTL is short for time to live.

```
@config plugins.LLM.memoryCandidateTTLDays 30
```

A TTL of `0` disables pruning, so candidates accumulate forever. That makes sense only if you also raise the threshold and accept slow growth in the candidate table.

### Per-user ceiling

`memoryMaxPerUser` caps durable memories per user. When promotion would push a user past the ceiling, the oldest memory drops to make room.

```
@config plugins.LLM.memoryMaxPerUser 100
```

The cap applies only to the durable side. Candidates do not count against it.

### Cleanup cadence

`memoryCleanupInterval` triggers a cleanup pass every N successful extractions. Cleanup applies the candidate TTL and trims any user over `memoryMaxPerUser`. With the default of `3`, every third extraction that saves at least one row also runs cleanup.

```
@config plugins.LLM.memoryCleanupInterval 10
```

Set it to `0` to disable periodic cleanup; the candidate table then grows until something else triggers a manual pass.

## What users see

Users interact only with durable memories. The candidate stage is invisible to them.

- `@memories` lists their durable memories.
- `@memories del <id>` deletes one.
- `@memories edit <id> <text>` rewrites one.
- `@memories clear` deletes every durable memory for the calling user.
- `@memories cleanup` triggers a one-shot cleanup pass.

Candidates have no user-facing surface. Operators with database access can inspect the `memory_candidates` table directly if a fact looks stuck.

## What flows through the extractor

The bot runs memory extraction after these events:

- Successful `@ask` responses, on the user who asked.
- Tool-driven memory writes from the assistant's `remember` tool.

The candidate path applies only to automatic extraction. Anything the assistant chooses to remember through its own tool skips the candidate stage, so the model can record explicit user preferences in one shot.

## Interaction with other settings

| Setting | Interaction |
|---------|-------------|
| `assistantModel` | Memory extraction calls this model, so it needs that model provider's environment variable set (see [Configuration](configuration.md#api-keys)). Without it, extraction fails silently and no candidates accumulate |
| `memoryEnabled` | Channel-scoped. `False` blocks both candidate insertion and durable promotion for that channel's traffic |

## Operational notes

- Candidate rows carry `first_seen` and `last_seen` timestamps. A user who repeats a fact keeps moving `last_seen` forward, so the TTL fires only on truly stale candidates.
- Reinforcement updates the candidate text in place when the extractor returns a slight rephrase. The mention counter still increments, so a fact that reappears in different wording still promotes once it crosses the threshold.
- Lowering `memoryPromotionThreshold` does not retroactively promote old candidates. Promotion happens on the next reinforcement event.
- Raising `memoryMaxPerUser` does not restore memories that earlier hit the cap. Once a memory drops, the only path back is for the user to state the fact again.
