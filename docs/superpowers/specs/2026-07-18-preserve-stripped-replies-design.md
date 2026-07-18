# Preserve Stripped Replies for Retry Detection

## Problem

The assistant completion path removes persisted near-duplicate assistant replies before building the model prompt. It then derives `prior_replies` from the filtered history. When persisted history already contains a duplicate cluster, both copies are correctly excluded from the prompt but are also absent from the retry guard's comparison set. A newly generated copy can therefore be delivered without the intended retry.

## Design

Keep prompt de-poisoning and retry detection as separate data-flow steps:

1. Remove degraded replies, and verse denials when applicable, because those replies must not anchor either the prompt or repetition detection.
2. Capture assistant reply text from that cleaned history for the in-loop repetition comparison.
3. Remove near-duplicate clusters from only the history passed to the model.
4. Preserve existing route behavior: chat considers personal and channel history, while verse considers only personal history and retains its tighter history window.

This avoids changing `_strip_repeated_replies` or its public behavior. It also avoids allowing degraded replies or verse denials to cause a repetition retry.

## Error Handling and Compatibility

No new failure mode or configuration is introduced. `None` and empty histories retain their existing behavior. The retry budget and best-effort response behavior remain unchanged.

## Testing

Add a regression test with two persisted near-duplicate assistant replies. The test will verify that:

- neither persisted duplicate is included in the model prompt;
- a newly generated near-duplicate triggers the repetition nudge and one retry;
- the fresh retry response is returned.

Run the targeted assistant test first, then the repository lint, typecheck, and preflight checks required for Python changes.
