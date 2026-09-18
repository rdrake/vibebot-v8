# Meme template inference on a resolver miss

**Status:** Noted, not started (2026-09-18)
**Author:** Richard Drake (with claude)
**Affects:** `plugins/llm/src/llm/meme.py` (`MemeCatalog.resolve`),
`plugin.py` `_build_meme_tool` / `_make_meme`, `prompts.py` `MEME_GUIDANCE`.
**Priority:** Low. Only worth building if the topic tags in
`meme_topics.py` keep missing what people ask for.

## Summary

Rubin could not find gym-related templates. memegen serves 209 templates,
files most of them under the quote, and only 51 carry keywords (65 in
all); nothing in the catalog contained "gym", and memegen's own
`?filter=gym` returned nothing either. The 2026-09-18 fix is a
checked-in keyword column: `meme_topics.py` tags every template with the
characters and franchises people name it by and the situations it is used
for, and the catalog folds those in as keywords. `@meme list gym` and
`@meme gym | ...` now reach `bd`; `list` and the "did you mean" list also
search the example captions.

That table is hand-written and finite. If it keeps missing, the next
step is inference: when `resolve` returns nothing, hand a model the
catalog (id, name, example captions, tags) and the query, ask for one id
or "none", and validate the answer against the catalog before using it.

## Design constraints

- Deterministic validation stays. The model's answer is only ever an id
  already in the catalog; anything else is a miss with suggestions, the
  same as today. `MEME_GUIDANCE`'s HARD RULE (transcribe the name the user
  said, never pick one) was needed because grok invented template names;
  an inference step must not reopen that — it runs inside the tool, on a
  miss, never in the chat model's own turn.
- One call per miss, small model, short output. The catalog with tags is
  about 15 KB of prompt; a cached system prompt keeps that cheap.
- Surface it explicitly. `@meme find <topic>` (or a `--find` flag) rather
  than silently picking on every miss, so a typo in a known name still
  gets the suggestion list and not a guess.
- Measure before and after: count `No meme template called` replies in
  the channel logs for a month on the tags alone.

## Not doing

- Self-hosting a memegen fork for more templates: each new template is
  an image plus text-box coordinates; memegen's own list is the ceiling
  unless someone wants to maintain that.
- Switching to imgflip's API: about 100 templates without a paid plan,
  and it needs an account.
