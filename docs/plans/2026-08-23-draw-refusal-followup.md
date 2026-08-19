# Draw refusal rate: measure the fixes, chase the embellishment lead

**Status:** Steps 1 and 2 answered early on 2026-08-19. Step 3 unblocked by the
refused-attempt rows shipped the same day; step 4 still waiting on an invoice
(2026-08-16)
**Author:** Richard Drake (with claude)
**Affects:** `image_generation` / `_is_content_safety_error` / `_draw_for_assistant`
(`9e67ba7`, `3399331`, `44e517e`, `98228d0`, all shipped 2026-08-15/16)
**Priority:** Medium. Every refusal is a billed call and a user who asked for a
picture and got an apology, but nothing is broken — this is measurement plus one
untested hypothesis.

## Why this is waiting

Three fixes shipped in two days against a measured baseline of **56% of draws
refused** (18 calls, 10 refused, over six hours on 2026-08-15). Two of them are
verified; the third cannot be verified without traffic that has not happened yet.

| Fix | Commit | State |
| --- | --- | --- |
| Partial-success draws deliver the image instead of narrating the failure | `9e67ba7` | Shipped, unit-tested. Needs a live partial failure to confirm |
| xAI moderation refusals arm the auto-rewrite loop (`drawAutoRewriteMax`, capped at 1) | `3399331` | **Verified live 2026-08-19.** 6 refusals, 6 recoveries, 0 reached a user |
| Image spend reaches the usage table, under the image model | `44e517e`, `98228d0` | Verified live 2026-08-16 10:48Z |

The rewrite loop is the one that matters. Before `3399331`, `_is_content_safety_error`
matched four substrings, none of which appear in xAI's
`imagine:content-moderated` error, so `drawAutoRewriteMax: 3` sat configured in
prod and had **never run once**. The 56% was the raw provider refusal rate with
no mitigation applied at all. Whether arming it actually helps is unmeasured.

## Measured 2026-08-19: the loop works

Window 2026-08-16 10:06Z to 2026-08-19 21:07Z, 3.5 days, two sources agreeing.

From `/config/logs/messages.log`, 22 draw requests: **6 refused on the first call
(27%), 6 rewrites fired, 6 delivered an image.** No `blocked after N rewrite
attempts`, no `No image generated`, no `LLM API error (image generation)` — no
refusal reached a user. From the usage table, 20 `draw:image` rows, all
`status='success'`, five of them at ~$0.0403 instead of $0.0200, which is the
double-billed recovery signature.

Against the 56% baseline that is the whole mitigation working, at about $0.02 per
recovered draw on a quarter of draws. What it does not settle is *why* it works:
two of the six recoveries came back **longer** than the prompt they replaced
(267→268 and 392→394 characters), so some share of the recovery may be xAI's
filter re-rolling differently rather than the rewrite defusing anything. The
rewriter now carries an explicit fidelity instruction and warns when a rewrite
grows (`prompt_rewrite_fidelity`), which is the signal to watch next.

## The open lead: the refusals may be self-inflicted

The user types a short, benign prompt. **The chat model expands it into the tool
argument, and the expansion is what reaches the content filter.** Confirmed on
request `03c929f3` (2026-08-16 10:48Z): `draw thatcher poll tax`, 22 characters,
became

> Margaret Thatcher as a grotesque tax collector demon, poll tax riot scene,
> 1980s London burning, cartoonishly evil, exaggerated features, dark satirical
> style

158 characters, sent verbatim — the stored prompt length and the logged
`prompt_chars` agree exactly, so nothing modified it in between.

This is not the safety rewriter. It is grok writing its own `generate_image`
argument during `assistant_step_1`, at no extra call or latency; the rewriter is
a separate completion labelled `op=prompt_rewrite` and only runs after a refusal.
Nothing in the code asks for the embellishment either: the tool description is a
bare "Generate an image from a text description" and `DRAW_SYSTEM_PROMPT` only
says to use the tool. The suspicion is that the channel's `assistantSystemPrompt`
overlay — deliberately crude and dark — is steering it.

If that holds, the fix is a line of draw guidance asking the model to carry the
user's subject through with minimal embellishment, which beats any retry loop
because it prevents the refusal instead of paying for a second one. **Untested.
Do not act on it without the query in step 3.**

## What to run

`command='draw:image'` rows have only existed since `98228d0` (2026-08-16), so
these need roughly a week of draws behind them. Unlike `docker logs`, the usage
table survives container recreates, and each row stores the exact prompt that
went to the provider.

Prod SQLite is read-only-safe at
`/home/vibebot/.config/vibebot/data/LLM.db`; there is no `sqlite3` binary on the
host, so drive it from `python3` with `?mode=ro`.

1. **Refusal rate now.** Compare against the 56% baseline.

   ```sql
   SELECT status, COUNT(*), ROUND(SUM(cost), 3)
   FROM usage WHERE command = 'draw:image' GROUP BY status;
   ```

2. **Is the loop firing?** Answered 2026-08-19: yes, 6 for 6.

   Do not grep for `attempting auto-rewrite` or `Rewrite attempt`. Both are
   `log.info` and no longer reach the log file in prod; the last occurrence is
   2026-02-06, on the previous image model. Grepping them returns zero and reads
   as "the loop never fired", which is a log-level artifact, not a regression.
   Count the `WARNING` timing lines instead, correlated by request id — a
   recovery is `result=error`, then `op=prompt_rewrite`, then a clean
   `op=image_generation`.

   ```
   docker exec vibebot grep -E "op=(image_generation|prompt_rewrite)" \
     /config/logs/messages.log
   ```

3. **The embellishment lead.** If refused prompts skew longer and more lurid
   than successful ones, the fix is prompt guidance, not retries.

   This was unanswerable until 2026-08-19. A recovered draw wrote one
   `status='success'` row and the refused first attempt vanished with its
   prompt, so `content_blocked` had never once been written to this table.
   `_log_image_usage` now writes one row per provider call, so a refusal the
   rewrite recovered from leaves its own row carrying the exact prompt the
   filter rejected. **Rows written before 2026-08-19 cannot answer this** — the
   comparison needs refusals recorded after that date.

   ```sql
   SELECT status, COUNT(*), ROUND(AVG(LENGTH(prompt)), 1)
   FROM usage WHERE command = 'draw:image' GROUP BY status;
   ```

   Then read the actual text of both groups — the averages only say where to
   look. Both columns are truncated to 200 characters, so treat the average
   length as a floor, not a measurement.

4. **Is the spend real?** Should be non-trivial. If an xAI invoice has landed,
   cross-check it, which also settles whether $0.02 per image is the right price
   (LiteLLM does not know this model, so `IMAGE_COST_PER_IMAGE` is a hardcoded
   guess that has never been checked against a bill).

   ```sql
   SELECT COUNT(*), ROUND(SUM(cost), 2) FROM usage WHERE command = 'draw:image';
   ```

## Gotchas

- Log timestamps are UTC; the IRC transcript is local (EDT, UTC−4).
- `docker logs` only reaches back to the last container recreate, and the
  auto-deploy updater recreates it on every push to `main`.
- Grouping `completion_timing op=image_generation` lines by request id matters:
  back-to-back calls under one id are **parallel tool calls in a single
  message**, executed sequentially by the loop, not separate turns.
- A `command='draw'` usage row is text cost only. The image bill is
  `command='draw:image'`. Summing both is correct; summing `draw` alone is not.
- Since 2026-08-19 one draw can write more than one `draw:image` row: one per
  provider call, with the refusals as `status='content_blocked'`. Counting rows
  now counts calls, not pictures. `SUM(cost)` is still the true spend — the
  refusal rows take their own share out of the delivered row rather than adding
  to it.

## Related

- [`docs/plans/archive`](archive) for the shipped work
- `IMAGE_COST_PER_IMAGE` in `plugins/llm/src/llm/service.py` — the only price
  source; neither litellm 1.93.0 nor 1.97.0 carries a grok-imagine entry
